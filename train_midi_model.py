"""
Train Ghost's your-style MIDI model on your PC's GPU.

It learns from midi_dataset.jsonl (the parts in your own projects, built by midi_dataset.py)
plus music_dataset.jsonl if you've made one (phrases read from tracks you own, built by
music_dataset.py; tagged so Ghost still asks the model for *your* style), and saves MusicAssistant/model/ghost_midi.npz, which Ghost's generate_part tool uses.

One-time setup (PyTorch with CUDA; pick the line for your setup at https://pytorch.org):
    py -3.13 -m pip install torch --index-url https://download.pytorch.org/whl/cu128

Train (a few minutes on a modern GPU):
    py -3.13 train_midi_model.py
    py -3.13 train_midi_model.py --gpu 1            # use the second GPU
    py -3.13 train_midi_model.py --tempo 118-135    # only parts from projects at those tempos
    py -3.13 train_midi_model.py --stats            # just show what it would learn from

No restart is needed afterwards: Ghost picks up a new model the next time it's used.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midi_model as mm  # noqa: E402
from indexer import app_dir  # noqa: E402


def build_net(torch, cfg):
    nn, F = torch.nn, torch.nn.functional

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            D = cfg["dim"]
            self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
            self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
            self.fc, self.out = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
            self.drop = nn.Dropout(cfg["dropout"])

        def forward(self, x):
            B, T, D = x.shape
            H = cfg["heads"]
            q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
            q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=cfg["dropout"] if self.training else 0)
            x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))
            h = F.gelu(self.fc(self.ln2(x)), approximate="tanh")
            return x + self.drop(self.out(h))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            D = cfg["dim"]
            self.tok = nn.Embedding(cfg["vocab"], D)
            self.pos = nn.Embedding(cfg["context"], D)
            self.blocks = nn.ModuleList(Block() for _ in range(cfg["layers"]))
            self.lnf = nn.LayerNorm(D)
            self.drop = nn.Dropout(cfg["dropout"])
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        def forward(self, idx):
            T = idx.shape[1]
            x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
            for b in self.blocks:
                x = b(x)
            return self.lnf(x) @ self.tok.weight.T   # output tied to the token embedding

    return Net()


def export(net, cfg, path: Path):
    """Save weights in the layout midi_model.Model reads (x @ W, so Linear weights are .T)."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()}
    out = {"tok": sd["tok.weight"], "pos": sd["pos.weight"],
           "lnf.g": sd["lnf.weight"], "lnf.b": sd["lnf.bias"]}
    for i in range(cfg["layers"]):
        p = f"blocks.{i}."
        for ln in ("ln1", "ln2"):
            out[f"b{i}.{ln}.g"], out[f"b{i}.{ln}.b"] = sd[p + ln + ".weight"], sd[p + ln + ".bias"]
        for lin in ("qkv", "proj", "fc", "out"):
            out[f"b{i}.{lin}.w"] = sd[p + lin + ".weight"].T.copy()
            out[f"b{i}.{lin}.b"] = sd[p + lin + ".bias"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, config=json.dumps(cfg), **{k: v.astype(np.float16 if v.ndim > 1 else np.float32)
                                             for k, v in out.items()})
    tmp.replace(path)


def batches(data, size, rng):
    order = list(range(len(data)))
    rng.shuffle(order)
    # group similar lengths so little of each batch is padding
    chunks = [order[i:i + size * 50] for i in range(0, len(order), size * 50)]
    out = []
    for ch in chunks:
        ch.sort(key=lambda i: len(data[i]))
        out += [ch[i:i + size] for i in range(0, len(ch), size)]
    rng.shuffle(out)
    return out


def to_tensor(torch, data, idx, device):
    T = max(len(data[i]) for i in idx)
    x = torch.zeros((len(idx), T), dtype=torch.long)
    for r, i in enumerate(idx):
        x[r, :len(data[i])] = torch.tensor(data[i])
    return x.to(device, non_blocking=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(app_dir() / "midi_dataset.jsonl"))
    ap.add_argument("--music", default=str(app_dir() / "music_dataset.jsonl"),
                    help="phrases read from your music collection (music_dataset.py); used if it exists")
    ap.add_argument("--no-music", action="store_true", help="learn from your own projects only")
    ap.add_argument("--mine-weight", type=int, default=2,
                    help="how many times each of your own phrases counts against one from the collection")
    ap.add_argument("--out", default=str(app_dir() / "model" / "ghost_midi.npz"))
    ap.add_argument("--tempo", help="only projects in this BPM range, e.g. 118-135")
    ap.add_argument("--gpu", type=int, default=0, help="which GPU (0, 1, ...)")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10,
                    help="stop after this many epochs without a better validation score")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stats", action="store_true", help="show the training data and stop")
    a = ap.parse_args()

    tempo = tuple(float(x) for x in a.tempo.split("-")) if a.tempo else None
    files = [Path(a.data)]
    if not a.no_music and Path(a.music).exists():
        files.append(Path(a.music))
    train, val, stats = mm.build_dataset(files, tempo=tempo)
    train = [t for t in train if len(t) <= a.context]
    val = [t for t in val if len(t) <= a.context]
    print(f"{stats['used_clips']} of {stats['clips']} clips usable -> {len(train)} training and "
          f"{len(val)} validation phrases ({stats['train_tokens']:,} tokens)")
    print("by role:", ", ".join(f"{r} {n}" for r, n in sorted(stats["by_role"].items())))
    print("by source:", ", ".join(f"{'your projects' if s == 'mine' else 'your music collection'} {n}"
                                  for s, n in sorted(stats["by_source"].items())))
    if a.stats:
        return
    if not train or not val:
        sys.exit("not enough data to train on - run midi_dataset.py over more projects first")

    try:
        import torch
    except ImportError:
        sys.exit("PyTorch isn't installed - see the setup line at the top of this file")
    if stats["by_source"].get("music"):        # keep your own parts the main voice
        train = train + [t for t in train if mm.is_source(t, "mine")] * (max(1, a.mine_weight) - 1)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    if torch.cuda.is_available():
        device = f"cuda:{a.gpu}"
        print("training on", torch.cuda.get_device_name(a.gpu))
    else:
        device = "cpu"
        print("no CUDA GPU found - training on the CPU (slow; check the PyTorch install)")

    cfg = {"name": "ghost_midi", "vocab": len(mm.VOCAB), "dim": a.dim, "layers": a.layers,
           "heads": a.heads, "context": a.context, "dropout": a.dropout,
           "pitch_range": stats["pitch_range"], "by_role": stats["by_role"],
           "by_source": stats["by_source"],
           "tempo": a.tempo, "trained": time.strftime("%Y-%m-%d %H:%M")}
    net = build_net(torch, cfg).to(device)
    print(f"model: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train) / a.batch)
    total = steps_per_epoch * a.epochs
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    use_amp = device.startswith("cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    F = torch.nn.functional

    def loss_of(x):
        with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
            logits = net(x[:, :-1])
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                               ignore_index=mm.VOCAB.PAD)

    best, bad, out = float("inf"), 0, Path(a.out)
    for epoch in range(1, a.epochs + 1):
        net.train()
        t0, tl, n = time.time(), 0.0, 0
        for idx in batches(train, a.batch, rng):
            x = to_tensor(torch, train, idx, device)
            loss = loss_of(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl, n = tl + loss.item(), n + 1
        net.eval()
        with torch.no_grad():
            vl = [loss_of(to_tensor(torch, val, idx, device)).item()
                  for idx in batches(val, a.batch, random.Random(0))]
        vloss = sum(vl) / len(vl)
        mark = ""
        if vloss < best - 1e-4:
            best, bad, mark = vloss, 0, "  saved"
            cfg["val_loss"], cfg["epoch"] = round(vloss, 4), epoch
            export(net, cfg, out)
        else:
            bad += 1
        print(f"epoch {epoch:3d}  train {tl / n:.3f}  validation {vloss:.3f}  "
              f"({time.time() - t0:.0f}s){mark}", flush=True)
        if bad >= a.patience:
            print("validation stopped improving - done")
            break

    print(f"\nbest model saved to {out} (validation {best:.3f})")
    model = mm.Model(out)
    avoid = mm.Fingerprints(Path(a.music)) if len(files) > 1 else None
    for role in ("bass", "arp", "lead"):
        notes, info = mm.make_part(model, role, "D minor", bars=2, seed=1, avoid=avoid)
        print(f"sample {role}: {len(notes)} notes, first: "
              + " ".join(f"{n['pitch']}@{n['start']}" for n in notes[:8]))


if __name__ == "__main__":
    main()
