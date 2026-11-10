"""
Song library: how each track in the music you own is built - arrangement, drums, notes,
sound and space - bar by bar, to build new songs from (not an average of them).

For every track in the folders you give it (e.g. F:\\Music\\Melodic Techno Playlists and
F:\\Music\\Old Techno) this splits the stems with Demucs on your GPU, finds the bar grid from
the kick, and measures how loud each element is in every bar:

    kick   - drums below 120 Hz          bass   - the bass stem
    clap   - drums 1-4 kHz (claps/snares) body   - other stem 150 Hz-1 kHz (pads, chords)
    hats   - drums above 6 kHz           lead   - other stem 1-8 kHz (leads, arps, stabs)
    vocals - the vocal stem

Saved to MusicAssistant/song_library.jsonl. Then --blueprints finds the real tracks whose
layouts are most typical (k-medoids over the bar-by-bar on/off maps), so a new song can follow
an actual arrangement bar for bar instead of an average of hundreds.

    py -3.12 song_library.py "F:\\Music\\Melodic Techno Playlists" "F:\\Music\\Old Techno"
    py -3.12 song_library.py --blueprints            # the typical layouts, per folder
Resumes where it stopped; tracks already mapped are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", message=".*attention on Current AMD GPU is still experimental.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_dataset as md  # noqa: E402
from indexer import app_dir  # noqa: E402

ELEMENTS = ("kick", "clap", "hats", "bass", "body", "lead", "vocals")
BANDS = {  # element: (stem, low Hz, high Hz)
    "kick": ("drums", 30, 120), "clap": ("drums", 1000, 4000), "hats": ("drums", 6000, 16000),
    "bass": ("bass", 20, 400), "body": ("other", 150, 1000), "lead": ("other", 1000, 8000),
    "vocals": ("vocals", 100, 8000),
}
BLOCK = 4          # bars per block when comparing layouts
MAX_BLOCKS = 64    # compare up to 256 bars
ON_DB = 12         # an element is "on" in a bar within this many dB of its usual loud level


def track_map(stems: dict, path: str, tag_bpm=None, sr: int = md.SR) -> dict | None:
    """Stems -> per-bar level (dB, rounded) of every element, bars counted from the first
    downbeat."""
    import librosa
    beats, bpm = md.beat_grid(stems["drums"], sr, tag_bpm)
    if beats is None:
        return None
    phase = md.downbeat_phase(beats, stems, sr)
    bar_starts = beats[phase::4]
    if len(bar_starts) < 9:
        return None
    hop = 512
    edges = librosa.time_to_frames(bar_starts, sr=sr, hop_length=hop)
    levels = {}
    spectra = {}
    for el, (stem, lo, hi) in BANDS.items():
        y = stems.get(stem)
        if y is None:
            continue
        if stem not in spectra:
            spectra[stem] = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
        S = spectra[stem]
        f = librosa.fft_frequencies(sr=sr, n_fft=2048)
        band = S[(f >= lo) & (f < hi)].sum(0)
        row = []
        for a, b in zip(edges[:-1], edges[1:]):
            seg = band[a:max(b, a + 1)]
            row.append(int(round(10 * np.log10(seg.mean() + 1e-12))) if len(seg) else -120)
        levels[el] = row
    beats = beats[phase:]                          # step 0 = the first downbeat
    n_bars = len(edges) - 1
    row = {"path": path, "bpm": round(bpm, 2), "bars": n_bars,
           "first_downbeat_s": round(float(bar_starts[0]), 3), "levels": levels}
    row["drums"] = drum_patterns(spectra["drums"], beats, n_bars, sr, hop)
    notes = {}
    for stem, role in (("bass", "bass"), ("other", "lead")):
        y = stems.get(stem)
        if y is not None and np.sqrt(np.mean(y ** 2)) >= 1e-4:
            notes[role] = [n for n in md.transcribe(y, beats, role, sr) if n[1] < n_bars * 16]
    row["notes"] = notes
    row["key"] = md.settle_mode(md.detect_key(stems, sr), notes.get("bass", []))
    row["sound"] = sound_profile(stems, spectra, edges, sr, hop)
    row["fx"] = echo_profile(stems, beats, sr, hop)
    return row


def drum_patterns(S, beats, n_bars, sr, hop) -> dict:
    """Per drum element, one 16-character string per bar: 'X' accent, 'x' hit, '.' rest."""
    import librosa
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    period = float(np.median(np.diff(beats)))
    steps = beats[0] + np.arange(n_bars * 16) * period / 4
    fr = librosa.time_to_frames(steps, sr=sr, hop_length=hop)
    out, kick_hit = {}, None
    for el in ("kick", "clap", "hats"):
        _, lo, hi = BANDS[el]
        band = np.sqrt(S[(f >= lo) & (f < hi)].sum(0))   # magnitude: quiet bleed stays small
        flux = np.maximum(0, np.diff(band, prepend=band[:1]))
        win = max(1, int(round(period / 4 * sr / hop * 0.35)))
        strength = np.array([flux[max(0, a - 1):a + win].max() if a < len(flux) else 0.0 for a in fr])
        if el == "clap" and kick_hit is not None and kick_hit.any():
            # the kick's click lands in the clap band too: take its usual size off on kick
            # steps, so only claps/snares that stand out from it count there
            click = np.median(strength[kick_hit])
            strength = np.where(kick_hit, np.maximum(0, strength - 1.2 * click), strength)
        ref = np.percentile(strength[strength > 0], 90) if (strength > 0).any() else 1.0
        hit, accent = strength > 0.35 * ref, strength > 0.75 * ref
        if el == "kick":
            kick_hit = hit
        chars = np.where(accent, "X", np.where(hit, "x", "."))
        out[el] = ["".join(chars[b * 16:(b + 1) * 16]) for b in range(n_bars)]
    return out


def sound_profile(stems, spectra, edges, sr, hop, section: int = 16) -> dict:
    """Per stem and 16-bar section: brightness (spectral centroid, Hz), filter movement
    (how much the brightness moves inside the section) and level (dB)."""
    import librosa
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    out = {}
    for stem in ("bass", "other", "vocals"):
        S = spectra.get(stem)
        if S is None:
            continue
        cen = (f[:, None] * S).sum(0) / (S.sum(0) + 1e-12)
        lev = 10 * np.log10(S.sum(0) + 1e-12)
        rows = []
        for i in range(0, len(edges) - 1, section):
            a, b = edges[i], edges[min(i + section, len(edges) - 1)]
            c, l = cen[a:b], lev[a:b]
            loud = l > (l.max() - 30) if len(l) else l
            if not len(c) or not loud.any():
                rows.append(None)
                continue
            rows.append({"bright_hz": int(np.median(c[loud])),
                         "movement": round(float(np.std(np.log2(c[loud] + 1))), 2),
                         "level_db": int(round(float(np.median(l[loud]))))})
        out[stem] = rows
    return out


def echo_profile(stems, beats, sr, hop) -> dict:
    """How much of each synth/vocal stem is tail and space between hits (0 = dry and
    gated, toward 1 = washed in reverb/delay). Delay timings can't be told apart from the
    rhythm of the part itself in a mixed track, so they aren't guessed."""
    import librosa
    out = {}
    for stem in ("other", "vocals"):
        y = stems.get(stem)
        if y is None or np.sqrt(np.mean(y ** 2)) < 1e-4:
            continue
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        q = np.percentile(rms, [20, 95])
        out[stem] = {"space": round(float(q[0] / (q[1] + 1e-12)), 2)}
    return out


# ---------------------------------------------------------------------------- blueprints
def activity(row: dict) -> dict:
    """Per element, which bars it plays in: within ON_DB of its usual loud level (95th
    percentile), and the element must be a real part of the track at all."""
    out = {}
    refs = {el: np.percentile(v, 95) for el, v in row["levels"].items() if v}
    loudest = max(refs.values()) if refs else 0
    for el, v in row["levels"].items():
        v = np.array(v, float)
        if not len(v) or el not in refs or refs[el] < loudest - 40:  # absent (or only bleed)
            out[el] = np.zeros(len(v), bool)
        else:
            out[el] = v >= refs[el] - ON_DB
    return out


def blocks(act: dict, n_bars: int) -> np.ndarray:
    """elements x MAX_BLOCKS matrix: 1 = on for most of that 4-bar block, -1 = past the end."""
    m = -np.ones((len(ELEMENTS), MAX_BLOCKS))
    for i, el in enumerate(ELEMENTS):
        a = act.get(el)
        for b in range(min(MAX_BLOCKS, n_bars // BLOCK)):
            m[i, b] = 0.0 if a is None else float(a[b * BLOCK:(b + 1) * BLOCK].mean() >= 0.5)
    return m


def distance(x: np.ndarray, y: np.ndarray) -> float:
    both = (x >= 0) & (y >= 0)
    only = (x >= 0) ^ (y >= 0)                          # one track has ended: length mismatch
    return float(np.abs(x - y)[both].sum() + 0.5 * only.sum())


def kmedoids(D: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> tuple[list, np.ndarray]:
    n = len(D)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    med = [int(np.argmin(D.sum(1)))]
    while len(med) < k:                                 # spread the starting medoids out
        d = D[:, med].min(1)
        med.append(int(rng.choice(n, p=d / d.sum())) if d.sum() else int(rng.integers(n)))
    for _ in range(iters):
        lab = np.argmin(D[:, med], 1)
        new = []
        for c in range(k):
            idx = np.where(lab == c)[0]
            new.append(int(idx[np.argmin(D[np.ix_(idx, idx)].sum(1))]) if len(idx) else med[c])
        if new == med:
            break
        med = new
    return med, np.argmin(D[:, med], 1)


def ranges(on: np.ndarray) -> str:
    """bool per bar -> '1-16, 33-96' (bar numbers from 1, end inclusive)."""
    out, start = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 2:                          # ignore single-bar blips
                out.append(f"{start + 1}-{i}")
            start = None
    return ", ".join(out) or "-"


def blueprints(rows: list[dict], k: int = 3) -> list[dict]:
    acts = [activity(r) for r in rows]
    M = [blocks(a, r["bars"]) for a, r in zip(acts, rows)]
    n = len(M)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = distance(M[i], M[j])
    med, lab = kmedoids(D, k)
    out = []
    for c, m in enumerate(med):
        r, a = rows[m], acts[m]
        out.append({"track": Path(r["path"]).stem, "path": r["path"], "bpm": r["bpm"],
                    "bars": r["bars"], "tracks_like_it": int((lab == c).sum()),
                    "elements": {el: ranges(a[el]) for el in ELEMENTS if el in a}})
    return sorted(out, key=lambda b: -b["tracks_like_it"])


def summary(rows: list[dict]) -> dict:
    """Rules that hold across the folder (shares of tracks), to check a blueprint against."""
    acts = [activity(r) for r in rows]
    first = {el: [] for el in ELEMENTS}
    kick_gaps = []
    for a in acts:
        for el in ELEMENTS:
            on = np.where(a.get(el, np.zeros(0, bool)))[0]
            if len(on):
                first[el].append(int(on[0]) + 1)
        k = a.get("kick", np.zeros(0, bool))
        gaps, run = [], 0
        for v in list(k) + [True]:
            if not v:
                run += 1
            elif run:
                gaps.append(run)
                run = 0
        kick_gaps.append(gaps)
    med = lambda v: int(np.median(v)) if v else None
    return {"tracks": len(rows), "bars_median": med([r["bars"] for r in rows]),
            "element_enters_at_bar": {el: med(v) for el, v in first.items()},
            "share_with_element": {el: round(len(v) / max(1, len(rows)), 2) for el, v in first.items()},
            "kick_breaks_per_track": med([len([g for g in gs if g >= 4]) for gs in kick_gaps]),
            "kick_break_bars": med([g for gs in kick_gaps for g in gs if g >= 4])}


def load(path: Path, folder: str | None = None) -> list[dict]:
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if not folder or folder.lower() in r["path"].lower():
                        rows.append(r)
    return rows


# ---------------------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--out", default=str(app_dir() / "song_library.jsonl"))
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--blueprints", action="store_true", help="show the typical layouts")
    ap.add_argument("--k", type=int, default=3, help="how many typical layouts per folder")
    a = ap.parse_args()
    out = Path(a.out)

    if a.blueprints or not a.folders:
        rows = load(out)
        res = {"all": {"summary": summary(rows), "blueprints": blueprints(rows, a.k)}} if rows else {}
        for top in ("Melodic Techno", "Old Techno"):
            sub = [r for r in rows if top.lower() in r["path"].lower()]
            if sub:
                res[top] = {"summary": summary(sub), "blueprints": blueprints(sub, a.k)}
        (out.parent / "arrangement_blueprints.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1) if res else f"no maps yet in {out}")
        return

    done_file = out.with_suffix(".done.json")
    done = json.loads(done_file.read_text()) if done_file.exists() else {}
    files = sorted(p for f in a.folders for p in Path(f).rglob("*") if p.suffix.lower() in md.AUDIO)
    todo = [p for p in files if done.get(str(p)) != p.stat().st_mtime]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(files)} tracks found, {len(todo)} to map")
    sep = md.Separator(a.gpu)
    from concurrent.futures import ProcessPoolExecutor
    t_start, finished, pending = time.time(), 0, {}
    with open(out, "a", encoding="utf-8") as f, ProcessPoolExecutor(max_workers=a.workers) as pool:

        def collect(block):
            nonlocal finished
            for fut in [x for x in pending if block or x.done()]:
                p = pending.pop(fut)
                finished += 1
                try:
                    row = fut.result()
                except Exception as e:
                    row = None
                    print(f"[{finished}/{len(todo)}] {p.name}: skipped ({type(e).__name__}: {e})")
                if row:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                done[str(p)] = p.stat().st_mtime
                done_file.write_text(json.dumps(done))
                left = (time.time() - t_start) / finished * (len(todo) - finished) / 60
                if row:
                    print(f"[{finished}/{len(todo)}] {p.name}: {row['bars']} bars at {row['bpm']} BPM,"
                          f" ~{left:.0f} min left", flush=True)

        for p in todo:
            try:
                tags = md.read_tags(str(p))
                stems = sep(str(p), keep=("drums", "bass", "other", "vocals"))
            except Exception as e:
                finished += 1
                print(f"[{finished}/{len(todo)}] {p.name}: skipped ({type(e).__name__}: {e})")
                done[str(p)] = p.stat().st_mtime
                continue
            while len(pending) >= a.workers:
                collect(False)
                if len(pending) >= a.workers:
                    time.sleep(0.2)
            pending[pool.submit(track_map, stems, str(p), tags.get("bpm"))] = p
            collect(False)
        while pending:
            collect(True)
        done_file.write_text(json.dumps(done))
    print(f"\nsaved to {out}. Next:  py -3.12 song_library.py --blueprints")


if __name__ == "__main__":
    main()
