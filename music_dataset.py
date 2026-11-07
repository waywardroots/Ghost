"""
Learn note patterns from the finished tracks you own (Phase 3, extra training data).

For every track in a folder (e.g. F:\\Music\\Melodic Techno Playlists) this:
  1. splits it into stems with Demucs (drums / bass / other) on your GPU,
  2. finds the tempo and the bar grid from the drums,
  3. reads the bassline from the bass stem and the main lead/arp line from the other stem,
     note by note on the 16th-note grid,
  4. saves 8-bar phrases to MusicAssistant/music_dataset.jsonl in the same format as
     midi_dataset.jsonl, marked source "music", with the track's key.

train_midi_model.py picks that file up automatically and tells the model which phrases are
yours and which came from these tracks, so Ghost can still ask it for *your* style.

Setup (once; PyTorch first, see train_midi_model.py):
    py -3.13 -m pip install demucs mutagen
Run (stops and resumes safely - tracks already done are skipped):
    py -3.13 music_dataset.py "F:\\Music\\Melodic Techno Playlists"
    py -3.13 music_dataset.py "F:\\Music\\Melodic Techno Playlists" --limit 5   # try a few first
    py -3.13 music_dataset.py --stats

Notes read from audio are rougher than MIDI: basslines come out well; for leads and arps
only the main line is captured; pads and chords are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indexer import app_dir, key_from_profile  # noqa: E402

AUDIO = (".mp3", ".wav", ".flac", ".aif", ".aiff", ".m4a", ".ogg")
SR = 22050          # analysis rate
HOP = 256
PHRASE_BARS = 8
RANGES = {"bass": (28, 64), "lead": (55, 96)}     # MIDI pitch, C3 = 60


# ---------------------------------------------------------------------------- grid
def beat_grid(drums, sr=SR, tag_bpm=None):
    """A constant-tempo beat grid (beat times in seconds) fitted to the drums. Electronic
    tracks keep one tempo, so a straight line through the detected beats is more accurate
    than the beats themselves."""
    import librosa
    # the kick marks the beat; hats and claps (often on the off-beats) would pull the grid
    # half a beat off, so listen to the low end only
    env = librosa.onset.onset_strength(y=drums, sr=sr, hop_length=HOP, fmax=160, n_mels=24)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP,
                                           start_bpm=tag_bpm or 125, tightness=400)
    bt = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
    if len(bt) < 16:
        return None, None
    period = float(np.median(np.diff(bt)))
    bpm = 60.0 / period
    while bpm < 100:
        bpm *= 2
    while bpm > 160:
        bpm /= 2
    if tag_bpm and abs(tag_bpm - bpm) / bpm < 0.03:
        bpm = float(tag_bpm)                     # DJ software's BPM is usually exact
    locked = bool(tag_bpm) and bpm == float(tag_bpm)
    period = 60.0 / bpm
    # a straight line through the beats (only its start if the tag fixed the tempo),
    # refitted without the outliers
    # number each beat by counting beat gaps (1 per gap, 2 if a beat was missed), so a
    # slightly-off first guess at the tempo can't push later beats onto the wrong number
    idx = np.r_[0, np.cumsum(np.maximum(1, np.round(np.diff(bt) / period)))]
    keep = np.ones(len(bt), bool)
    t0 = float(np.median(bt - idx * period))
    for _ in range(3):
        if keep.sum() < 8:
            break
        if locked:
            t0 = float(np.median(bt[keep] - idx[keep] * period))
        else:
            a = np.vstack([idx[keep], np.ones(keep.sum())]).T
            (slope, t0), *_ = np.linalg.lstsq(a, bt[keep], rcond=None)
            if abs(slope - period) / period < 0.02:
                period = float(slope)
        resid = np.abs(bt - (t0 + idx * period))
        keep = resid < max(0.02, 3 * np.median(resid))
    t0 = t0 % period
    dur = len(drums) / sr
    n = int((dur - t0) / period)
    return t0 + np.arange(n) * period, 60.0 / period


def downbeat_phase(beats, stems, sr=SR) -> int:
    """Which of every 4 beats is the 1. Sections change on bar starts: compare the bar before
    each beat with the bar after it (notes, harmony and loudness of the bass, music and
    drums) and pick the phase where the changes land."""
    import librosa
    frames = librosa.time_to_frames(beats, sr=sr, hop_length=HOP)
    feats = []
    for name in ("bass", "other", "drums"):
        y = stems.get(name)
        if y is None or not np.any(y):
            continue
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
        mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=24))
        f = np.vstack([mel / 10.0, librosa.feature.chroma_stft(S=S, sr=sr) * 2]) \
            if name != "drums" else mel / 10.0
        feats.append(librosa.util.sync(f, frames[frames < f.shape[1]], aggregate=np.mean,
                                      pad=False))   # column k = beat k to beat k+1
    n = min(x.shape[1] for x in feats)
    F = np.vstack([x[:, :n] for x in feats])        # one column per beat
    nov = np.zeros(n)
    for k in range(4, n - 4):
        nov[k] = np.linalg.norm(F[:, k:k + 4].mean(1) - F[:, k - 4:k].mean(1))
    nov = np.maximum(0, nov - np.convolve(nov, np.ones(9) / 9, "same"))   # peaks only
    return int(np.argmax([nov[p::4].sum() for p in range(4)]))


# ---------------------------------------------------------------------------- notes
def _salience(y, sr, lo, hi):
    """Pitch salience per frame for MIDI pitches lo..hi: CQT with harmonics summed onto the
    fundamental, so a bright saw still reads as its root note."""
    import librosa
    fmin = librosa.midi_to_hz(lo)
    n = hi - lo + 1 + 28
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP, fmin=fmin, n_bins=n, bins_per_octave=12))
    sal = np.zeros((hi - lo + 1, C.shape[1]))
    for off, wgt in ((0, 1.0), (12, 0.6), (19, 0.45), (24, 0.35), (28, 0.25)):
        sal += wgt * C[off:off + hi - lo + 1]
    return sal, C


def transcribe(y, beats, role, sr=SR) -> list[list]:
    """Monophonic notes on the 16th grid: [pitch, step, length_steps, velocity]."""
    import librosa
    lo, hi = RANGES[role]
    sal, C = _salience(y, sr, lo, hi)
    rms = librosa.feature.rms(y=y, frame_length=HOP * 4, hop_length=HOP)[0]
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    nfr = min(sal.shape[1], len(rms), len(env))
    period = float(np.median(np.diff(beats)))
    steps = beats[0] + np.arange(len(beats) * 4) * period / 4
    tf = lambda t: int(round(t * sr / HOP))
    pitch, energy, conf, onset = [], [], [], []
    for t in steps:
        a, b = tf(t + period / 4 * 0.2), tf(t + period / 4 * 0.85)
        a, b = min(a, nfr - 1), min(max(b, a + 1), nfr)
        s = sal[:, a:b].mean(1)
        k = int(np.argmax(s))
        pitch.append(lo + k)
        conf.append(s[k] / (np.median(s) + 1e-9))
        energy.append(rms[a:b].mean())
        oa, ob = tf(t - period / 4 * 0.15), tf(t + period / 4 * 0.35)
        onset.append(env[max(0, oa):max(ob, oa + 1)].max() if oa < nfr else 0.0)
    energy, conf, onset = np.array(energy), np.array(conf), np.array(onset)
    db = 20 * np.log10(energy + 1e-9)
    ref = np.percentile(db, 90)
    active = (db > ref - 18) & (conf > 2.5)
    on_ref = np.percentile(onset[active], 90) if active.any() else 1.0
    hit = onset > 0.35 * on_ref
    notes, cur = [], None
    for i in range(len(steps)):
        if not active[i]:
            cur = None
            continue
        if cur is None or pitch[i] != cur[0] or hit[i]:
            vel = int(np.clip(100 + 2 * (db[i] - ref), 45, 127))
            cur = [int(pitch[i]), i, 1, vel]
            notes.append(cur)
        else:
            cur[2] += 1
    return notes


def detect_key(stems, sr=SR):
    import librosa
    y = sum(stems[n] for n in ("bass", "other") if stems.get(n) is not None)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP * 8)
    return key_from_profile(chroma.mean(axis=1))[0]


def phrases(notes, phase, role, n_beats):
    """8-bar phrases from bar starts. Repeats of the same loop inside a track are kept once."""
    bar0 = phase * 4                               # in steps
    span = PHRASE_BARS * 16
    out, seen = [], []
    for w0 in range(bar0, n_beats * 4 - span + 1, span):
        part = [[p, s - w0, min(d, w0 + span - s), v] for p, s, d, v in notes if w0 <= s < w0 + span]
        bars_with = {s // 16 for _, s, _, _ in part}
        if len(part) < 4 or len(bars_with) < PHRASE_BARS // 2:
            continue
        sig = {(s, p) for p, s, _, _ in part}
        if any(len(sig & o) / len(sig | o) > 0.8 for o in seen):
            continue
        seen.append(sig)
        r = role
        if role == "lead":
            per_bar = len(part) / len(bars_with)
            r = "arp" if per_bar >= 6 and np.median([d for _, _, d, _ in part]) <= 2 else "lead"
        out.append((w0, r, part))
    return out


def analyze_stems(stems, path: str, tag_bpm=None, tag_key=None, sr=SR) -> list[dict]:
    """Stems (mono arrays at sr) -> dataset rows."""
    beats, bpm = beat_grid(stems["drums"], sr, tag_bpm)
    if beats is None:
        return []
    phase = downbeat_phase(beats, stems, sr)
    key = tag_key or detect_key(stems, sr)
    rows = []
    for stem, role in (("bass", "bass"), ("other", "lead")):
        y = stems.get(stem)
        if y is None or np.sqrt(np.mean(y ** 2)) < 1e-4:
            continue
        notes = transcribe(y, beats, role, sr)
        for w0, r, part in phrases(notes, phase, role, len(beats)):
            bar = (w0 - phase * 4) // 16 + 1
            rid = hashlib.md5(f"{path}|{stem}|{w0}".encode()).hexdigest()[:16]
            rows.append({"project": path, "track": f"{stem} stem", "clip": f"bars {bar}-{bar + PHRASE_BARS - 1}",
                         "role": r, "instrument": "audio", "tempo": round(bpm, 2),
                         "length_beats": PHRASE_BARS * 4.0, "key": key, "source": "music", "id": rid,
                         "notes": [[p, s / 4, d / 4, v] for p, s, d, v in part]})
    return rows


# ---------------------------------------------------------------------------- stems
class Separator:
    def __init__(self, gpu: int = 0, fallback: bool = False):
        self.model = None
        if fallback:
            return
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
        except ImportError:
            sys.exit("Demucs isn't installed - run:  py -3.13 -m pip install demucs mutagen\n"
                     "(or add --no-demucs for a rougher split without it)")
        self.torch, self.apply = torch, apply_model
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("no CUDA GPU found - Demucs will be slow on the CPU")
        self.model = get_model("htdemucs").to(self.device).eval()

    def __call__(self, path: str) -> dict:
        import librosa
        if self.model is None:                     # rough split: low end vs the rest
            y, _ = librosa.load(path, sr=SR, mono=True)
            h, p = librosa.effects.hpss(y)
            from scipy.signal import butter, sosfilt
            lo = sosfilt(butter(4, 180, "low", fs=SR, output="sos"), h)
            hi = sosfilt(butter(4, 250, "high", fs=SR, output="sos"), h)
            return {"drums": p, "bass": lo, "other": hi}
        torch = self.torch
        y, _ = librosa.load(path, sr=44100, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        wav = torch.tensor(y, dtype=torch.float32)
        ref = wav.mean(0)
        mu, sd = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            src = self.apply(self.model, ((wav - mu) / sd)[None], device=self.device,
                             split=True, overlap=0.25, progress=False)[0]
        src = (src * sd + mu).mean(1).cpu().numpy()   # mono stems
        return {name: librosa.resample(src[i], orig_sr=44100, target_sr=SR)
                for i, name in enumerate(self.model.sources) if name in ("drums", "bass", "other")}


def read_tags(path):
    try:
        import reference_scan
        return reference_scan.read_tags(path)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--out", default=str(app_dir() / "music_dataset.jsonl"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only this many new tracks")
    ap.add_argument("--no-demucs", action="store_true", help="rough split without Demucs")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    done_file = out.with_suffix(".done.json")
    done = json.loads(done_file.read_text()) if done_file.exists() else {}

    if a.stats or not a.folders:
        rows = [json.loads(l) for l in open(out, encoding="utf-8")] if out.exists() else []
        roles = {}
        for r in rows:
            roles[r["role"]] = roles.get(r["role"], 0) + 1
        print(f"{len(done)} tracks read, {len(rows)} phrases: "
              + ", ".join(f"{k} {v}" for k, v in sorted(roles.items())))
        if not a.folders:
            print("give a folder to read, e.g.  music_dataset.py \"F:\\Music\\Melodic Techno Playlists\"")
        return

    files = sorted(p for f in a.folders for p in Path(f).rglob("*") if p.suffix.lower() in AUDIO)
    todo = [p for p in files if done.get(str(p)) != p.stat().st_mtime]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(files)} tracks found, {len(todo)} to read")
    sep = Separator(a.gpu, fallback=a.no_demucs)
    t_start = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, p in enumerate(todo, 1):
            t0 = time.time()
            try:
                tags = read_tags(str(p))
                rows = analyze_stems(sep(str(p)), str(p), tags.get("bpm"), tags.get("key"))
            except Exception as e:  # one bad file shouldn't stop the run
                print(f"[{i}/{len(todo)}] {p.name}: skipped ({type(e).__name__}: {e})")
                rows = []
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            done[str(p)] = p.stat().st_mtime
            done_file.write_text(json.dumps(done))
            left = (time.time() - t_start) / i * (len(todo) - i) / 60
            kinds = ", ".join(f"{k} {sum(r['role'] == k for r in rows)}"
                              for k in ("bass", "lead", "arp") if any(r["role"] == k for r in rows))
            print(f"[{i}/{len(todo)}] {p.name}: {len(rows)} phrases ({kinds or 'none'}) "
                  f"{time.time() - t0:.0f}s, ~{left:.0f} min left", flush=True)
    print(f"\nsaved to {out}. Now retrain:  py -3.13 train_midi_model.py")


if __name__ == "__main__":
    main()
