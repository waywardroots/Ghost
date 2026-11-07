"""
Your-style MIDI model (Phase 3, step 3).

A small transformer that learns from the parts you wrote in your own projects
(midi_dataset.jsonl) and writes new ones: basslines, arps, leads, chords and pads,
1-8 bars, in any key.

This file has no PyTorch in it: it holds the note <-> token translation shared with
train_midi_model.py, and runs a trained model with numpy alone, so Ghost's connector
doesn't need PyTorch to use it. Training happens in train_midi_model.py on your GPU.

How a part becomes tokens (16th-note grid, 4/4):
    BOS SRC_mine ROLE_bass MODE_minor LEN_4  BAR POS_0 PITCH_38 DUR_2 VEL_6  POS_2 ...  BAR ...  EOS
Every part is moved to C (C minor or C major) first, so the model learns your shapes
and intervals rather than which key a project happened to be in; on the way out the
notes are moved to the song's key.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

import style

ROLES = ("bass", "arp", "lead", "chords", "pad")
SOURCES = ("mine", "music")   # your own projects / phrases read from tracks you own
MODES = ("minor", "major")
MAX_BARS = 8
STEPS = 16          # 16th notes per bar
MAX_DUR = 32        # longest note, in 16ths (2 bars)
VEL_BINS = 8
SCALES = {"minor": (0, 2, 3, 5, 7, 8, 10), "major": (0, 2, 4, 5, 7, 9, 11)}


class ModelError(Exception):
    pass


# ---------------------------------------------------------------------------- vocabulary
class Vocab:
    def __init__(self):
        self.itos = ["PAD", "BOS", "EOS", "BAR"]
        self.itos += [f"SRC_{s}" for s in SOURCES]
        self.itos += [f"ROLE_{r}" for r in ROLES]
        self.itos += [f"MODE_{m}" for m in MODES]
        self.itos += [f"LEN_{b}" for b in range(1, MAX_BARS + 1)]
        self.itos += [f"POS_{p}" for p in range(STEPS)]
        self.itos += [f"PITCH_{p}" for p in range(128)]
        self.itos += [f"DUR_{d}" for d in range(1, MAX_DUR + 1)]
        self.itos += [f"VEL_{v}" for v in range(VEL_BINS)]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.PAD, self.BOS, self.EOS, self.BAR = 0, 1, 2, 3
        first = lambda prefix: self.stoi[prefix]
        self.src0 = first("SRC_mine")
        self.role0, self.mode0, self.len0 = first("ROLE_bass"), first("MODE_minor"), first("LEN_1")
        self.pos0, self.pitch0 = first("POS_0"), first("PITCH_0")
        self.dur0, self.vel0 = first("DUR_1"), first("VEL_0")

    def __len__(self):
        return len(self.itos)

    def kind(self, tok: int) -> str:
        return self.itos[tok].split("_")[0]


VOCAB = Vocab()


def _to_c(pc: int) -> int:
    """Semitones that move tonic pc to C, the smaller way (-6..+5)."""
    s = (-pc) % 12
    return s - 12 if s > 5 else s


def vel_bin(v) -> int:
    return max(0, min(VEL_BINS - 1, int(v) * VEL_BINS // 128))


def bin_vel(b) -> int:
    return int(b * 128 / VEL_BINS + 64 / VEL_BINS)


def encode_window(notes, role: str, mode: str, bars: int, source: str = "mine") -> list[int]:
    """notes: [pitch (already in C), step 0.., dur steps, velocity] inside the window."""
    V = VOCAB
    toks = [V.BOS, V.src0 + SOURCES.index(source), V.role0 + ROLES.index(role),
            V.mode0 + MODES.index(mode), V.len0 + bars - 1]
    by_bar = [[] for _ in range(bars)]
    for p, s, d, v in notes:
        by_bar[s // STEPS].append((s % STEPS, p, d, v))
    for bar in by_bar:
        toks.append(V.BAR)
        last = None
        for pos, p, d, v in sorted(bar):
            if (pos, p) == last:              # the same note twice on one step
                continue
            last = (pos, p)
            toks += [V.pos0 + pos, V.pitch0 + p, V.dur0 + d - 1, V.vel0 + vel_bin(v)]
    toks.append(V.EOS)
    return toks


def decode(toks) -> list[list]:
    """Tokens -> [pitch, start_beats, duration_beats, velocity] (pitch still in C)."""
    V, out, bar, pos, cur = VOCAB, [], -1, 0, None
    for t in toks:
        k = V.kind(t)
        if t == V.BAR:
            bar += 1
        elif k == "POS":
            pos = t - V.pos0
        elif k == "PITCH":
            cur = [t - V.pitch0, (max(bar, 0) * STEPS + pos) / 4.0]
        elif k == "DUR" and cur is not None and len(cur) == 2:
            cur.append((t - V.dur0 + 1) / 4.0)
        elif k == "VEL" and cur is not None and len(cur) == 3:
            cur.append(bin_vel(t - V.vel0))
            out.append(cur)
            cur = None
        elif t == V.EOS:
            break
    return out


# ---------------------------------------------------------------------------- dataset
def _source(c) -> str:
    return "music" if c.get("source") == "music" else "mine"


def clip_windows(c: dict) -> list[tuple[list[int], int]]:
    """One stored clip -> token windows of up to 8 bars (hop 4 bars). Returns [] for clips
    that can't be used (drums/fx, unknown key, empty)."""
    if c.get("role") not in ROLES or not c.get("notes"):
        return []
    try:
        key = style.parse_key(c.get("key"))
    except style.StyleError:
        key = None
    if not key:
        return []
    shift = _to_c(key[0])
    length = int(round(float(c.get("length_beats") or 0) * 4))
    notes = []
    for p, s, d, v in c["notes"]:
        p = int(p) + shift
        st = int(round(float(s) * 4))
        if not 0 <= p < 128 or st < 0 or (length and st >= length):
            continue
        notes.append([p, st, max(1, min(MAX_DUR, int(round(float(d) * 4)))), int(v)])
    if len(notes) < 2:
        return []
    length = max(length, max(n[1] for n in notes) + 1)
    out, hop, win = [], STEPS * 4, STEPS * MAX_BARS
    for w0 in range(0, length, hop):
        bars = min(MAX_BARS, math.ceil((length - w0) / STEPS))
        if w0 and bars < 4:   # the tail of a long clip is already inside the previous window
            break
        end = w0 + bars * STEPS
        part = [[p, s - w0, min(d, end - s), v] for p, s, d, v in notes if w0 <= s < end]
        if len(part) >= 2:
            out.append((encode_window(part, c["role"], key[1], bars, _source(c)), bars))
        if end >= length:
            break
    return out


def build_dataset(files, val_share: float = 0.1, tempo: tuple | None = None):
    """-> (train, val, stats). files: midi_dataset.jsonl and optionally music_dataset.jsonl.
    Windows are de-duplicated. The validation set is whole projects of yours the model never
    trains on, so its score shows how well it writes *your* parts; phrases from your music
    collection are all used for training."""
    if isinstance(files, (str, Path)):
        files = [files]
    train, val, seen = [], [], set()
    stats = {"clips": 0, "used_clips": 0, "by_role": {}, "by_source": {}, "pitch_range": {}}
    pitches = {r: [] for r in ROLES}
    for jsonl in files:
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                stats["clips"] += 1
                if tempo and c.get("tempo") and not tempo[0] <= float(c["tempo"]) <= tempo[1]:
                    continue
                wins = clip_windows(c)
                if not wins:
                    continue
                stats["used_clips"] += 1
                src = _source(c)
                h = int(hashlib.md5(c["project"].lower().encode()).hexdigest(), 16) % 1000
                dest = val if src == "mine" and h < val_share * 1000 else train
                for toks, _ in wins:
                    t = tuple(toks)
                    if t in seen:
                        continue
                    seen.add(t)
                    dest.append(toks)
                    stats["by_role"][c["role"]] = stats["by_role"].get(c["role"], 0) + 1
                    stats["by_source"][src] = stats["by_source"].get(src, 0) + 1
                    if src == "mine":
                        pitches[c["role"]] += [x - VOCAB.pitch0 for x in toks
                                               if VOCAB.kind(x) == "PITCH"]
    for r, ps in pitches.items():
        if ps:
            lo, hi = np.percentile(ps, [1, 99])
            stats["pitch_range"][r] = [int(lo), int(hi)]
    stats["train_windows"], stats["val_windows"] = len(train), len(val)
    stats["train_tokens"] = sum(map(len, train))
    return train, val, stats


def is_source(toks, source: str) -> bool:
    return len(toks) > 1 and toks[1] == VOCAB.src0 + SOURCES.index(source)


# ---------------------------------------------------------------------------- inference
def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


class Model:
    """The trained transformer, run with numpy (weights exported by train_midi_model.py)."""

    def __init__(self, path: Path):
        path = Path(path)
        if not path.exists():
            raise ModelError(f"no trained model at {path} - train one first with "
                             "train_midi_model.py (see the README)")
        z = np.load(path, allow_pickle=False)
        self.cfg = json.loads(str(z["config"]))
        self.w = {k: z[k].astype(np.float32) for k in z.files if k != "config"}
        if self.cfg.get("vocab") != len(VOCAB):
            raise ModelError("this model was trained with a different Ghost version - retrain it")
        self.path, self.mtime = path, path.stat().st_mtime

    def _step(self, tok: int, pos: int, cache: list):
        """Feed one token; returns logits for the next one. cache holds keys/values."""
        w, c = self.w, self.cfg
        H, D = c["heads"], c["dim"]
        hd = D // H
        x = w["tok"][tok] + w["pos"][pos]
        for i in range(c["layers"]):
            p = f"b{i}."
            h = _ln(x, w[p + "ln1.g"], w[p + "ln1.b"])
            qkv = h @ w[p + "qkv.w"] + w[p + "qkv.b"]
            q, k, v = np.split(qkv, 3)
            if len(cache) <= i:
                cache.append(([], []))
            cache[i][0].append(k.reshape(H, hd))
            cache[i][1].append(v.reshape(H, hd))
            K = np.stack(cache[i][0], 1)              # H, T, hd
            Vv = np.stack(cache[i][1], 1)
            att = np.einsum("hd,htd->ht", q.reshape(H, hd), K) / math.sqrt(hd)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y = np.einsum("ht,htd->hd", att, Vv).reshape(D)
            x = x + y @ w[p + "proj.w"] + w[p + "proj.b"]
            h = _ln(x, w[p + "ln2.g"], w[p + "ln2.b"])
            x = x + _gelu(h @ w[p + "fc.w"] + w[p + "fc.b"]) @ w[p + "out.w"] + w[p + "out.b"]
        x = _ln(x, w["lnf.g"], w["lnf.b"])
        return x @ w["tok"].T

    def logits(self, toks: list[int]) -> np.ndarray:
        """Logits after every position (used by the tests to compare with PyTorch)."""
        cache, out = [], []
        for i, t in enumerate(toks):
            out.append(self._step(t, i, cache))
        return np.stack(out)

    def generate(self, role: str, mode: str = "minor", bars: int = 4, temperature: float = 0.9,
                 top_p: float = 0.92, seed: int | None = None, in_key: bool = True,
                 pitch_range: tuple | None = None, max_notes_per_bar: int = 32) -> list[int]:
        """Sample one part. Every step is constrained to a valid next token (a note is always
        POS PITCH DUR VEL, positions only move forward inside a bar, exactly `bars` bars)."""
        V = VOCAB
        if role not in ROLES:
            raise ModelError(f"role must be one of {', '.join(ROLES)}")
        bars = max(1, min(MAX_BARS, int(bars)))
        rng = np.random.default_rng(seed)
        toks = [V.BOS, V.src0 + SOURCES.index("mine"), V.role0 + ROLES.index(role),
                V.mode0 + MODES.index(mode), V.len0 + bars - 1]
        cache = []
        for i, t in enumerate(toks[:-1]):
            self._step(t, i, cache)
        lo, hi = pitch_range or self.cfg.get("pitch_range", {}).get(role, (0, 127))
        scale = set(SCALES[mode])
        pitch_ok = np.zeros(len(V), bool)
        for p in range(int(lo), int(hi) + 1):
            if not in_key or p % 12 in scale:
                pitch_ok[V.pitch0 + p] = True
        bar, last_pos, in_bar, want = 0, -1, 0, "BAR"
        last_pitch = same_step_pitch = None
        limit = self.cfg["context"]
        while len(toks) < limit:
            lg = self._step(toks[-1], len(toks) - 1, cache)
            allowed = np.zeros(len(V), bool)
            if want == "BAR":                     # right after the header
                allowed[V.BAR] = True
            elif want == "NEXT":                  # a new note, the next bar, or the end
                if in_bar < max_notes_per_bar:
                    first = max(last_pos, 0)
                    if last_pos >= 0 and not pitch_ok[V.pitch0 + last_pitch + 1:].any():
                        first = last_pos + 1          # no higher note left for this step
                    allowed[V.pos0 + first:V.pos0 + STEPS] = True
                allowed[V.BAR] = bar < bars
                allowed[V.EOS] = bar >= bars
            elif want == "PITCH":
                allowed = pitch_ok.copy()
                if same_step_pitch is not None:   # chords go upwards, no doubled notes
                    allowed[:V.pitch0 + same_step_pitch + 1] = False

            elif want == "DUR":
                allowed[V.dur0:V.dur0 + MAX_DUR] = True
            elif want == "VEL":
                allowed[V.vel0:V.vel0 + VEL_BINS] = True
            t = _sample(lg, allowed, temperature, top_p, rng)
            toks.append(t)
            k = V.kind(t)
            if t == V.EOS:
                break
            if t == V.BAR:
                bar, last_pos, in_bar, want = bar + 1, -1, 0, "NEXT"
            elif k == "POS":
                same_step_pitch = last_pitch if t - V.pos0 == last_pos else None
                last_pos, in_bar, want = t - V.pos0, in_bar + 1, "PITCH"
            elif k == "PITCH":
                last_pitch, want = t - V.pitch0, "DUR"
            elif k == "DUR":
                want = "VEL"
            elif k == "VEL":
                want = "NEXT"
        return toks


def _sample(logits, allowed, temperature, top_p, rng) -> int:
    lg = np.where(allowed, logits.astype(np.float64), -np.inf)
    if temperature <= 0:
        return int(np.argmax(lg))
    lg = lg / temperature
    p = np.exp(lg - lg[allowed].max())
    p /= p.sum()
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    keep = order[:max(1, int(np.searchsorted(cum, top_p) + 1))]
    q = p[keep] / p[keep].sum()
    return int(rng.choice(keep, p=q))


# ---------------------------------------------------------------------------- for Ghost
class Fingerprints:
    """Every phrase read from the user's music collection (music_dataset.jsonl), so a new part
    can be checked against them: the model must not hand back someone else's hook."""

    def __init__(self, path: Path):
        self.path, self._mtime, self.sets = Path(path), None, []

    def load(self):
        if not self.path.exists():
            self.sets = []
            return self
        m = self.path.stat().st_mtime
        if m != self._mtime:
            sets = []
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = json.loads(line)
                        for toks, _ in clip_windows(c):
                            sets.append(({(round(n[1] * 4), n[0]) for n in decode(toks)},
                                         c.get("project", "")))
            self.sets, self._mtime = sets, m
        return self

    def closest(self, notes, bars: int) -> tuple[float, str]:
        """Share of the new part's notes (in C, [pitch, start, ...]) found at the same place in
        the most similar collection phrase, comparing every 4-bar offset. 1.0 = a copy."""
        g = {(round(n[1] * 4), n[0]) for n in notes}
        if not g:
            return 0.0, ""
        span, best, where = bars * STEPS, 0.0, ""
        for f, src in self.load().sets:
            for off in range(0, MAX_BARS * STEPS - span + 1, 4 * STEPS):
                hit = sum(1 for s, p in g if (s + off, p) in f)
                if hit / len(g) > best:
                    best, where = hit / len(g), src
        return best, where


def make_part(model: Model, role: str, key: str = "D minor", bars: int = 4, octave: int = 0,
              temperature: float = 0.9, seed: int | None = None, in_key: bool = True,
              tries: int = 6, avoid: Fingerprints | None = None,
              max_overlap: float = 0.6) -> tuple[list[dict], dict]:
    """A new part in the song's key, as Live notes. Re-samples a take that comes out nearly
    empty (fewer than one note per bar) or that matches a phrase from the user's music
    collection too closely (more than max_overlap of its notes in the same places)."""
    k = style.parse_key(key) or (2, "minor")
    seed = random.randrange(1 << 30) if seed is None else int(seed)
    unit_bars = min(bars, MAX_BARS)
    best = None
    for n in range(max(1, tries)):
        toks = model.generate(role, k[1], unit_bars, temperature=temperature,
                              seed=seed + n, in_key=in_key)
        notes = decode(toks)
        overlap, src = avoid.closest(notes, unit_bars) if avoid else (0.0, "")
        good = len(notes) >= unit_bars and overlap <= max_overlap
        rank = (good, overlap <= max_overlap, len(notes) >= unit_bars, -overlap)
        if best is None or rank > best[0]:
            best = (rank, notes, seed + n, overlap, src)
        if good:
            break
    _, notes, used_seed, overlap, src = best
    unit = unit_bars * 4.0
    shift = -_to_c(k[0]) + 12 * int(octave)
    out, total = [], bars * 4.0
    reps = math.ceil(total / unit)
    for r in range(reps):                         # longer than 8 bars: repeat the 8
        for p, s, d, v in notes:
            t = s + r * unit
            if t < total:
                out.append({"pitch": int(max(0, min(127, p + shift))), "start": round(t, 4),
                            "duration": round(min(d, total - t), 4), "velocity": int(v)})
    out.sort(key=lambda n: (n["start"], n["pitch"]))
    info = {"role": role, "key": style.key_name(k), "bars": bars, "seed": used_seed,
            "temperature": temperature, "note_count": len(out),
            "model": model.cfg.get("name", model.path.name)}
    if avoid is not None:
        info["closest_to_collection"] = round(overlap, 2)
        if overlap > max_overlap:
            info["warning"] = (f"{overlap:.0%} of the notes match a phrase in "
                               f"{Path(src).name} - try another seed or a higher temperature")
    return out, info
