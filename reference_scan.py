"""
Learn from the finished music you own (Phase 4 groundwork): tempo, key, arrangement and mix
of every track in your music folders, saved to the library database so Ghost can use them as
references when it arranges and mixes.

For each track:
  - artist, title, genre (its folder), whether it's yours (James and the Cat / My Tracks)
  - tempo and key (from the tags your DJ software wrote, otherwise measured)
  - arrangement: loudness per bar, intro length, breakdowns, drops, total bars
  - mix on the loudest 32 bars: balance per band, brightness, loudness, stereo width

    py -3.13 reference_scan.py                       # F:\\Music
    py -3.13 reference_scan.py --music "D:\\DJ Music" --workers 6
    py -3.13 reference_scan.py --profile             # what your references have in common

Your music is only read, never changed. Later scans only touch new or changed files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from indexer import app_dir, key_from_profile

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}
SKIP_DIRS = {"_serato_", "engine library", "engine library backup", "pioneerdj", "video projects",
             "skill sheets", "recorded", "backup"}
YOURS_WORDS = ("james and the cat", "james and the cats", "my tracks to play")
SR = 32000
BANDS = [("sub", 20, 60), ("bass", 60, 250), ("low_mid", 250, 800), ("mid", 800, 2500),
         ("high_mid", 2500, 6000), ("air", 6000, 16000)]
CAMELOT = {"1A": "G# minor", "2A": "D# minor", "3A": "A# minor", "4A": "F minor",
           "5A": "C minor", "6A": "G minor", "7A": "D minor", "8A": "A minor", "9A": "E minor",
           "10A": "B minor", "11A": "F# minor", "12A": "C# minor", "1B": "B major",
           "2B": "F# major", "3B": "C# major", "4B": "G# major", "5B": "D# major",
           "6B": "A# major", "7B": "F major", "8B": "C major", "9B": "G major", "10B": "D major",
           "11B": "A major", "12B": "E major"}
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS reference_tracks(
  path TEXT PRIMARY KEY, size INTEGER, mtime REAL, artist TEXT, title TEXT, genre TEXT,
  yours INTEGER, seconds REAL, bpm REAL, bpm_source TEXT, musical_key TEXT, key_source TEXT,
  bars INTEGER, intro_bars INTEGER, sections TEXT, energy_per_bar TEXT,
  loudness_db REAL, peak_db REAL, crest_db REAL, bands TEXT, brightness_hz REAL,
  stereo_width REAL, low_end_width REAL, error TEXT, indexed_at REAL);
"""


# ---------------------------------------------------------------- tags
def normalize_key(text) -> str | None:
    """'8A', 'Am', 'A minor', 'Bbm', 'F#' -> 'A minor' style, or None."""
    if not text:
        return None
    t = str(text).strip()
    if t.upper() in CAMELOT:
        return CAMELOT[t.upper()]
    m = re.match(r"^([A-Ga-g])([#b]?)\s*(m|min|minor|maj|major)?$", t.replace("♯", "#").replace("♭", "b"))
    if not m:
        return None
    note = m.group(1).upper() + (m.group(2) or "")
    note = FLATS.get(note, note)
    if note in ("E#", "B#", "Cb", "Fb"):
        return None
    mode = (m.group(3) or "").lower()
    return f"{note} {'minor' if mode in ('m', 'min', 'minor') else 'major'}"


def read_tags(path: str) -> dict:
    out = {}
    try:
        import mutagen
    except ImportError:
        return out
    try:
        f = mutagen.File(path)
    except Exception:
        return out
    if f is None or not getattr(f, "tags", None):
        return out
    tags = f.tags

    def first(*names):
        for n in names:
            try:
                v = tags.get(n)
            except Exception:
                v = None
            if v is None:
                continue
            v = getattr(v, "text", v)
            if isinstance(v, (list, tuple)):
                v = v[0] if v else None
            if v not in (None, ""):
                return str(v)
        return None

    out["artist"] = first("TPE1", "artist", "ARTIST", "\xa9ART")
    out["title"] = first("TIT2", "title", "TITLE", "\xa9nam")
    bpm = first("TBPM", "bpm", "BPM", "tmpo")
    try:
        bpm = float(bpm) if bpm else None
    except ValueError:
        bpm = None
    out["bpm"] = bpm if bpm and 60 <= bpm <= 200 else None
    out["key"] = normalize_key(first("TKEY", "initialkey", "INITIALKEY", "KEY"))
    return out


def title_from_name(stem: str) -> tuple[str | None, str]:
    s = re.sub(r"^\d{5,}_", "", stem).replace("_", " ").strip()  # Beatport '17654266_Karma_(...)'
    if " - " in s:
        a, t = s.split(" - ", 1)
        return a.strip(), t.strip()
    return None, s


def is_yours(path: Path, artist: str | None) -> bool:
    text = f"{path} {artist or ''}".lower()
    return any(w in text for w in YOURS_WORDS)


# ---------------------------------------------------------------- audio
def _db(x: float) -> float:
    import math
    return round(20 * math.log10(max(x, 1e-9)), 1)


def _fold_bpm(bpm: float) -> float:
    while bpm < 85:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return round(bpm, 2)


def sections_from_energy(full_db, low_db) -> tuple[int, list[dict]]:
    """Split a track into sections from per-bar loudness (full band and below 150 Hz).
    'drop' = kick and full energy, 'groove' = kick but quieter, 'break' = no kick."""
    import numpy as np
    full = np.asarray(full_db, float)
    low = np.asarray(low_db, float)
    if full.size == 0:
        return 0, []
    loud = float(np.percentile(full, 90))
    kick_ref = float(np.percentile(low, 90))
    kinds = []
    for f, l in zip(full, low):
        if l < kick_ref - 9:
            kinds.append("break")
        elif f >= loud - 3.5:
            kinds.append("drop")
        else:
            kinds.append("groove")
    # smooth single-bar blips (fills, crashes) into their neighbours
    for i in range(1, len(kinds) - 1):
        if kinds[i - 1] == kinds[i + 1] != kinds[i]:
            kinds[i] = kinds[i - 1]
    runs = []
    for i, k in enumerate(kinds):
        if runs and runs[-1]["kind"] == k:
            runs[-1]["bars"] += 1
        else:
            runs.append({"kind": k, "start_bar": i + 1, "bars": 1})
    # fold runs shorter than 4 bars into the previous section
    merged = []
    for r in runs:
        if merged and r["bars"] < 4:
            merged[-1]["bars"] += r["bars"]
        else:
            merged.append(dict(r))
    first_drop = next((r["start_bar"] for r in merged if r["kind"] == "drop"), None)
    intro = (first_drop - 1) if first_drop else len(kinds)
    return intro, merged


def analyze_track(job) -> dict:
    path, tag_bpm, tag_key = job
    out = {"path": path}
    try:
        import numpy as np
        import librosa
        y, sr = librosa.load(path, sr=SR, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        mono = y.mean(axis=0)
        secs = mono.size / sr
        out["seconds"] = round(secs, 1)
        if secs < 30:
            out["error"] = "shorter than 30 s"
            return out
        # tempo and beat grid
        onset = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=512)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=512)
        est = _fold_bpm(float(np.atleast_1d(tempo)[0])) if np.size(tempo) else 0.0
        if tag_bpm:
            out["bpm"], out["bpm_source"] = round(tag_bpm, 2), "tag"
        else:
            out["bpm"], out["bpm_source"] = est, "measured"
        bpm = out["bpm"] or est or 125.0
        first_beat = float(librosa.frames_to_time(beats[0], sr=sr, hop_length=512)) if len(beats) else 0.0
        bar_s = 4 * 60.0 / bpm
        n_bars = int((secs - first_beat) // bar_s)
        out["bars"] = n_bars
        # loudness per bar, full band and below 150 Hz (where the kick and bass live)
        from scipy.signal import butter, sosfilt
        low = sosfilt(butter(4, 150, btype="low", fs=sr, output="sos"), mono)
        full_db, low_db = [], []
        for b in range(n_bars):
            a = int((first_beat + b * bar_s) * sr)
            z = int((first_beat + (b + 1) * bar_s) * sr)
            full_db.append(_db(float(np.sqrt(np.mean(mono[a:z] ** 2)))))
            low_db.append(_db(float(np.sqrt(np.mean(low[a:z] ** 2)))))
        out["energy_per_bar"] = full_db
        out["intro_bars"], out["sections"] = sections_from_energy(full_db, low_db)
        # mix: the loudest 32 bars
        if n_bars >= 8:
            win = min(32, n_bars)
            sums = np.convolve(np.asarray(full_db), np.ones(win), "valid")
            b0 = int(np.argmax(sums))
            a = int((first_beat + b0 * bar_s) * sr)
            z = int((first_beat + (b0 + win) * bar_s) * sr)
        else:
            a, z = 0, mono.size
        seg, segm = y[:, a:z], mono[a:z]
        rms = float(np.sqrt(np.mean(segm ** 2)))
        peak = float(np.max(np.abs(seg)))
        out["loudness_db"], out["peak_db"] = _db(rms), _db(peak)
        out["crest_db"] = round(out["peak_db"] - out["loudness_db"], 1)
        S = np.abs(librosa.stft(segm, n_fft=4096, hop_length=2048)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        total = float(S.sum()) or 1.0
        out["bands"] = {n: round(100 * float(S[(freqs >= lo) & (freqs < hi)].sum()) / total, 1)
                        for n, lo, hi in BANDS}
        out["brightness_hz"] = round(float(np.median(
            librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr, n_fft=4096))))
        mid, side = (seg[0] + seg[1]) / 2, (seg[0] - seg[1]) / 2
        out["stereo_width"] = round(float(np.sqrt(np.mean(side ** 2)) / max(np.sqrt(np.mean(mid ** 2)), 1e-9)), 2)
        lo_sos = butter(4, 120, btype="low", fs=sr, output="sos")
        lm, ls = sosfilt(lo_sos, mid), sosfilt(lo_sos, side)
        out["low_end_width"] = round(float(np.sqrt(np.mean(ls ** 2)) / max(np.sqrt(np.mean(lm ** 2)), 1e-9)), 2)
        # key
        if tag_key:
            out["musical_key"], out["key_source"] = tag_key, "tag"
        else:
            ds = librosa.resample(segm, orig_sr=sr, target_sr=11025)
            chroma = librosa.feature.chroma_cqt(y=ds, sr=11025)
            k, _ = key_from_profile(chroma.mean(axis=1))
            out["musical_key"], out["key_source"] = k, "measured"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


# ---------------------------------------------------------------- scan
def open_db(path: str):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def walk(roots):
    for root in roots:
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]
            for f in files:
                p = Path(dirpath) / f
                if p.suffix.lower() in AUDIO_EXTS:
                    yield p


def genre_of(p: Path, roots) -> str:
    for r in roots:
        try:
            rel = p.relative_to(r)
        except ValueError:
            continue
        return rel.parts[0] if len(rel.parts) > 1 else Path(r).name
    return p.parent.name


def scan(con, roots, workers: int, full: bool):
    roots = [Path(r) for r in roots]
    known = {r["path"]: (r["size"], r["mtime"]) for r in
             con.execute("SELECT path, size, mtime FROM reference_tracks")}
    todo, seen = [], set()
    for p in walk(roots):
        try:
            st = p.stat()
        except OSError:
            continue
        seen.add(str(p))
        if not full and known.get(str(p)) == (st.st_size, st.st_mtime):
            continue
        todo.append((p, st))
    print(f"[music] {len(seen)} tracks found, {len(todo)} new or changed")
    rows = {}
    for p, st in todo:
        tags = read_tags(str(p))
        artist, title = tags.get("artist"), tags.get("title")
        if not title:
            a2, title = title_from_name(p.stem)
            artist = artist or a2
        rows[str(p)] = dict(path=str(p), size=st.st_size, mtime=st.st_mtime, artist=artist,
                            title=title, genre=genre_of(p, roots), yours=int(is_yours(p, artist)),
                            seconds=None, bpm=tags.get("bpm"), bpm_source=None,
                            musical_key=tags.get("key"), key_source=None, bars=None,
                            intro_bars=None, sections=None, energy_per_bar=None,
                            loudness_db=None, peak_db=None, crest_db=None, bands=None,
                            brightness_hz=None, stereo_width=None, low_end_width=None,
                            error=None, indexed_at=time.time())

    def save(batch):
        con.executemany(
            "INSERT OR REPLACE INTO reference_tracks VALUES(:path,:size,:mtime,:artist,:title,"
            ":genre,:yours,:seconds,:bpm,:bpm_source,:musical_key,:key_source,:bars,:intro_bars,"
            ":sections,:energy_per_bar,:loudness_db,:peak_db,:crest_db,:bands,:brightness_hz,"
            ":stereo_width,:low_end_width,:error,:indexed_at)", batch)
        con.commit()

    batch, done, errors = [], 0, 0
    jobs = [(k, v["bpm"], v["musical_key"]) for k, v in rows.items()]
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(analyze_track, j) for j in jobs]
            for fut in as_completed(futs):
                res = fut.result()
                row = rows[res["path"]]
                for k, v in res.items():
                    if k in ("sections", "energy_per_bar", "bands") and v is not None:
                        v = json.dumps(v)
                    if v is not None:
                        row[k] = v
                errors += bool(res.get("error"))
                batch.append(row)
                done += 1
                if len(batch) >= 25:
                    save(batch)
                    batch = []
                print(f"\r[music] analyzed {done}/{len(jobs)}", end="", flush=True)
        if batch:
            save(batch)
        print(f"\n[music] done ({errors} track(s) could not be read)")
    gone = [p for p in known if p not in seen and any(p.startswith(str(r)) for r in roots)]
    if gone:
        con.executemany("DELETE FROM reference_tracks WHERE path=?", [(g,) for g in gone])
        con.commit()


# ---------------------------------------------------------------- profile
def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def profile(con, genre: str | None = None, yours: bool | None = None) -> dict:
    q = "SELECT * FROM reference_tracks WHERE error IS NULL"
    args = []
    if genre:
        q += " AND lower(genre) LIKE ?"
        args.append(f"%{genre.lower()}%")
    if yours is not None:
        q += " AND yours=?"
        args.append(int(yours))
    rows = con.execute(q, args).fetchall()
    if not rows:
        return {"tracks": 0}
    keys = defaultdict(int)
    for r in rows:
        if r["musical_key"]:
            keys[r["musical_key"]] += 1
    breaks, first_break, n_breaks = [], [], []
    for r in rows:
        secs = json.loads(r["sections"] or "[]")
        bs = [s for s in secs if s["kind"] == "break" and s["start_bar"] > (r["intro_bars"] or 0)]
        n_breaks.append(len(bs))
        if bs:
            first_break.append(bs[0]["start_bar"])
            breaks += [s["bars"] for s in bs]
    bands = defaultdict(list)
    for r in rows:
        for k, v in json.loads(r["bands"] or "{}").items():
            bands[k].append(v)
    return {
        "tracks": len(rows), "genre": genre, "yours_only": yours,
        "bpm": _med(r["bpm"] for r in rows),
        "minutes": _med((r["seconds"] or 0) / 60 for r in rows),
        "bars": _med(r["bars"] for r in rows),
        "top_keys": sorted(keys.items(), key=lambda kv: -kv[1])[:6],
        "arrangement": {"intro_bars": _med(r["intro_bars"] for r in rows),
                        "breakdowns_per_track": _med(n_breaks),
                        "first_breakdown_bar": _med(first_break),
                        "breakdown_bars": _med(breaks)},
        "mix": {"loudness_db": _med(r["loudness_db"] for r in rows),
                "crest_db": _med(r["crest_db"] for r in rows),
                "bands_percent": {k: _med(v) for k, v in bands.items()},
                "brightness_hz": _med(r["brightness_hz"] for r in rows),
                "stereo_width": _med(r["stereo_width"] for r in rows),
                "low_end_width": _med(r["low_end_width"] for r in rows)},
        "note": "medians across the tracks; loudness is RMS of the loudest 32 bars",
    }


def print_profile(con):
    genres = [r[0] for r in con.execute(
        "SELECT genre, COUNT(*) c FROM reference_tracks WHERE error IS NULL GROUP BY genre "
        "ORDER BY c DESC LIMIT 8")]
    print("================ YOUR REFERENCE MUSIC ================")
    for label, p in [("All tracks", profile(con)), ("Your own tracks", profile(con, yours=True))] + \
            [(g, profile(con, genre=g)) for g in genres]:
        if not p["tracks"]:
            continue
        a, m = p["arrangement"], p["mix"]
        print(f"\n{label} ({p['tracks']} tracks): {p['bpm']} BPM, {p['minutes']} min, "
              f"{p['bars']} bars")
        print(f"  keys: " + ", ".join(f"{k} ({n})" for k, n in p["top_keys"]))
        print(f"  intro {a['intro_bars']} bars, {a['breakdowns_per_track']} breakdown(s), first at "
              f"bar {a['first_breakdown_bar']}, {a['breakdown_bars']} bars long")
        print(f"  mix: {m['loudness_db']} dB RMS, crest {m['crest_db']} dB, bands {m['bands_percent']}, "
              f"width {m['stereo_width']}, low-end width {m['low_end_width']}")
    print("=======================================================")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--music", nargs="+", default=["F:\\Music"], metavar="DIR")
    ap.add_argument("--db", default=str(app_dir() / "music_library.db"))
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--full", action="store_true", help="re-analyze everything")
    ap.add_argument("--profile", action="store_true", help="only print the summary")
    a = ap.parse_args(argv)
    con = open_db(a.db)
    if not a.profile:
        roots = [r for r in a.music if Path(r).exists()]
        if not roots:
            sys.exit(f"None of these folders exist: {a.music}")
        scan(con, roots, a.workers, a.full)
    print_profile(con)


if __name__ == "__main__":
    main()
