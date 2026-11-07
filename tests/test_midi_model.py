"""Checks the your-style model's tokens, numpy inference and generation (no GPU needed).
If PyTorch is installed, also checks that a trained, exported model gives the same answers
in numpy as in PyTorch."""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import midi_model as mm  # noqa: E402

PASSED = 0


def check(name, cond):
    global PASSED
    if not cond:
        raise AssertionError("FAILED: " + name)
    PASSED += 1
    print("PASS ", name)


def random_model(path, layers=2, dim=32, heads=4, context=256, seed=0):
    rng = np.random.default_rng(seed)
    V = len(mm.VOCAB)
    w = {"tok": rng.normal(0, .3, (V, dim)), "pos": rng.normal(0, .1, (context, dim)),
         "lnf.g": 1 + rng.normal(0, .1, dim), "lnf.b": rng.normal(0, .1, dim)}
    for i in range(layers):
        for ln in ("ln1", "ln2"):
            w[f"b{i}.{ln}.g"], w[f"b{i}.{ln}.b"] = 1 + rng.normal(0, .1, dim), rng.normal(0, .1, dim)
        for name, (a, b) in {"qkv": (dim, 3 * dim), "proj": (dim, dim), "fc": (dim, 4 * dim),
                             "out": (4 * dim, dim)}.items():
            w[f"b{i}.{name}.w"], w[f"b{i}.{name}.b"] = rng.normal(0, .2, (a, b)), rng.normal(0, .1, b)
    cfg = {"vocab": V, "dim": dim, "layers": layers, "heads": heads, "context": context,
           "pitch_range": {"bass": [30, 60]}}
    np.savez(path, config=json.dumps(cfg), **w)
    return mm.Model(path)


def naive_logits(m, toks):
    """Full causal attention over the whole sequence at once - no cache."""
    w, c = m.w, m.cfg
    H, D = c["heads"], c["dim"]
    hd, T = D // H, len(toks)
    x = w["tok"][toks] + w["pos"][:T]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(c["layers"]):
        p = f"b{i}."
        h = mm._ln(x, w[p + "ln1.g"], w[p + "ln1.b"])
        q, k, v = np.split(h @ w[p + "qkv.w"] + w[p + "qkv.b"], 3, axis=1)
        q, k, v = (t.reshape(T, H, hd).transpose(1, 0, 2) for t in (q, k, v))
        a = q @ k.transpose(0, 2, 1) / math.sqrt(hd) + mask
        a = np.exp(a - a.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        y = (a @ v).transpose(1, 0, 2).reshape(T, D)
        x = x + y @ w[p + "proj.w"] + w[p + "proj.b"]
        h = mm._ln(x, w[p + "ln2.g"], w[p + "ln2.b"])
        x = x + mm._gelu(h @ w[p + "fc.w"] + w[p + "fc.b"]) @ w[p + "out.w"] + w[p + "out.b"]
    return mm._ln(x, w["lnf.g"], w["lnf.b"]) @ w["tok"].T


def main():
    V = mm.VOCAB
    # tokens
    clip = {"role": "bass", "key": "D minor", "length_beats": 8, "project": "p",
            "notes": [[38, 0.5, 0.25, 100], [41, 1.5, 0.5, 90], [36, 4.0, 1.0, 120],
                      [38, 7.75, 0.25, 64]]}
    wins = mm.clip_windows(clip)
    check("an 8-beat clip is one 2-bar phrase", len(wins) == 1 and wins[0][1] == 2)
    toks = wins[0][0]
    check("header: role, mode, length",
          [V.itos[t] for t in toks[:4]] == ["BOS", "ROLE_bass", "MODE_minor", "LEN_2"])
    notes = mm.decode(toks)
    check("D minor is moved to C minor (down 2)", [n[0] for n in notes] == [36, 39, 34, 36])
    check("timing and lengths survive", [n[1] for n in notes] == [0.5, 1.5, 4.0, 7.75]
          and [n[2] for n in notes] == [0.25, 0.5, 1.0, 0.25])
    check("velocity is kept roughly", all(abs(a[3] - b[3]) <= 8 for a, b in zip(notes, clip["notes"])))
    check("drums and unknown keys are skipped",
          not mm.clip_windows(dict(clip, role="drums")) and not mm.clip_windows(dict(clip, key=None)))
    long = dict(clip, length_beats=64, notes=[[50, b, .25, 100] for b in range(64)])
    w = mm.clip_windows(long)
    check("a 16-bar clip becomes overlapping 8-bar phrases", [b for _, b in w] == [8, 8, 8])

    with tempfile.TemporaryDirectory() as d:
        js = Path(d) / "ds.jsonl"
        rows = [dict(clip, project=f"proj{i}", id=str(i)) for i in range(30)]
        rows.append(dict(clip, project="other", key="F major", role="lead", id="x"))
        js.write_text("\n".join(json.dumps(r) for r in rows))
        tr, va, st = mm.build_dataset(js)
        check("identical phrases are only kept once", len(tr) + len(va) == 2)
        check("pitch ranges are measured per role", "bass" in st["pitch_range"])

        # inference
        m = random_model(Path(d) / "m.npz")
        seq = toks[:40]
        check("cached step-by-step inference equals full attention",
              np.allclose(m.logits(seq), naive_logits(m, seq), atol=1e-4))
        ok = True
        for seed in range(6):
            t = m.generate("bass", "minor", bars=4, seed=seed, temperature=1.2)
            kinds = [V.kind(x) for x in t]
            ok &= kinds.count("BAR") == 4 and t[-1] == V.EOS
            body = [x for x in t[4:] if V.kind(x) in ("POS", "PITCH", "DUR", "VEL")]
            ok &= [V.kind(x) for x in body] == ["POS", "PITCH", "DUR", "VEL"] * (len(body) // 4)
            for n in mm.decode(t):
                ok &= n[0] % 12 in mm.SCALES["minor"] and 30 <= n[0] <= 60 and n[1] < 16
        check("generated parts are well formed, in key, in range, 4 bars", ok)
        out, info = mm.make_part(m, "bass", "F minor", bars=12, seed=3, octave=-1)
        ok = all(0 <= n["start"] < 48 and n["start"] + n["duration"] <= 48 + 1e-6 for n in out)
        pcs = {(n["pitch"] - 5) % 12 for n in out}
        check("make_part moves to the song's key and fits 12 bars", ok and pcs <= set(mm.SCALES["minor"])
              and info["key"] == "F minor" and info["bars"] == 12)
        same = mm.make_part(m, "bass", "F minor", bars=12, seed=3, octave=-1)[0]
        check("same seed, same part", same == out)
        try:
            mm.Model(Path(d) / "missing.npz")
            ok = False
        except mm.ModelError as e:
            ok = "train_midi_model.py" in str(e)
        check("a missing model explains how to train one", ok)

        try:
            import torch  # noqa: F401
        except Exception:
            print("(PyTorch not installed - skipping the PyTorch/numpy comparison)")
        else:
            import train_midi_model as tm
            cfg = {"name": "t", "vocab": len(V), "dim": 32, "layers": 2, "heads": 4,
                   "context": 128, "dropout": 0.0, "pitch_range": {}}
            torch.manual_seed(0)
            net = tm.build_net(torch, cfg).eval()
            for p in net.parameters():
                p.data.normal_(0, 0.2)
            path = Path(d) / "t.npz"
            tm.export(net, cfg, path)
            with torch.no_grad():
                ref = net(torch.tensor([seq])).numpy()[0]
            got = mm.Model(path).logits(seq)
            check("exported model gives the same answers in numpy as in PyTorch",
                  np.abs(got - ref).max() < 0.05 * np.abs(ref).max())
    print(f"\nAll {PASSED} checks passed.")


if __name__ == "__main__":
    main()
