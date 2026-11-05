#!/usr/bin/env python3
"""
Music Assistant - Phase 1: Library Indexer (Windows)

Scans your samples, plugins and Ableton Live projects and builds:
  - music_library.db   (SQLite catalog the assistant will query later)
  - style_profile.json (what your own music looks like: tempos, keys,
                        favourite plugins/devices/samples per track type)

Commands:
  python indexer.py scan    [--samples DIR ...] [--projects DIR ...] [--plugins DIR ...]
  python indexer.py search  [words ...] [--category C] [--key "A minor"] [--bpm 140] [--loops|--oneshots]
  python indexer.py profile
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sqlite3
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"}
PLUGIN_EXTS = {".vst3": "VST3", ".dll": "VST2", ".clap": "CLAP"}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E",
         "E#": "F", "B#": "C"}
# Krumhansl-Kessler key profiles (index 0 = tonic)
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# First matching rule wins. The file name is checked before its folders.
CATEGORY_RULES = [
    ("808", ["808"]),
    ("kick", ["kick", "kik", "kck", "bd"]),
    ("snare", ["snare", "snr", "sd", "rim", "rimshot"]),
    ("clap", ["clap", "clp", "snap"]),
    ("hihat", ["hihat", "hat", "hats", "hh", "ohh", "chh", "openhat", "closedhat"]),
    ("cymbal", ["cymbal", "crash", "ride", "splash"]),
    ("perc", ["perc", "percussion", "shaker", "tom", "toms", "conga", "bongo", "tamb",
              "tambourine", "cowbell"]),
    ("drums", ["drum", "drums", "drumloop", "breakbeat", "break", "breaks", "toploop", "kit",
               "beat", "beats"]),
    ("bass", ["bass", "sub", "reese"]),
    ("vocal", ["vocal", "vocals", "vox", "voice", "acapella", "adlib", "chant"]),
    ("fx", ["fx", "sfx", "riser", "impact", "sweep", "downlifter", "uplifter", "whoosh",
            "noise", "transition", "foley"]),
    ("pad", ["pad", "pads", "atmos", "atmosphere", "ambient", "texture", "drone"]),
    ("keys", ["piano", "keys", "rhodes", "epiano", "organ"]),
    ("guitar", ["guitar", "gtr"]),
    ("strings", ["string", "strings", "violin", "cello", "orchestral"]),
    ("brass", ["brass", "horn", "horns", "trumpet", "sax"]),
    ("lead", ["lead", "leads", "pluck", "plucks", "arp", "bell", "bells"]),
    ("chords", ["chord", "chords", "stab", "stabs"]),
    ("melody", ["melody", "melodies", "melodic", "loop", "loops"]),
    ("synth", ["synth", "synths"]),
]
DRUM_CATEGORIES = {"kick", "snare", "clap", "hihat", "cymbal", "perc", "drums"}

BPM_RE = re.compile(r"(?i)(?<![\d.])(\d{2,3}(?:\.\d+)?)\s*[-_ ]?bpm|bpm\s*[-_ ]?(\d{2,3}(?:\.\d+)?)(?![\d.])")
KEY_RE = re.compile(
    r"(?:^|[\s_\-\(\[,])([A-G])([#b\u266f\u266d]?)[\s_\-]?((?i:major|minor|maj|min)|m)?"
    r"(?=$|[\s_\-\)\]\.,])"
)


def app_dir() -> Path:
    d = Path.home() / "MusicAssistant"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Name parsing helpers
# --------------------------------------------------------------------------
def categorize(path: Path) -> str | None:
    parts = [path.stem] + list(reversed(path.parent.parts))[:4]
    for part in parts:
        tokens = re.findall(r"[a-z]+|\d+", part.lower())
        for cat, keywords in CATEGORY_RULES:
            for kw in keywords:
                for t in tokens:
                    if t == kw or (len(kw) >= 4 and t.startswith(kw)):
                        return cat
    return None


def bpm_from_name(name: str) -> float | None:
    m = BPM_RE.search(name)
    if m:
        v = float(m.group(1) or m.group(2))
        if 50 <= v <= 220:
            return v
    return None


def _norm_key(letter: str, acc: str, mode: str | None) -> str:
    acc = acc.replace("\u266f", "#").replace("\u266d", "b")
    note = letter + acc
    note = FLATS.get(note, note)
    is_minor = bool(mode) and (mode == "m" or mode.lower() in ("min", "minor"))
    return f"{note} {'minor' if is_minor else 'major'}"


def key_from_name(name: str) -> str | None:
    for m in KEY_RE.finditer(name):
        letter, acc, mode = m.group(1), m.group(2), m.group(3)
        if acc or mode:
            return _norm_key(letter, acc, mode)
    return None


def normalize_key_query(text: str) -> str | None:
    t = text.strip()
    m = re.fullmatch(r"([A-Ga-g])([#b]?)\s*(.*)", t)
    if not m:
        return None
    letter, acc, rest = m.group(1).upper(), m.group(2), m.group(3).strip()
    mode = None
    if rest:
        r = rest.lower()
        if r in ("m", "min", "minor"):
            mode = "minor"
        elif r in ("maj", "major"):
            mode = "major"
        else:
            return None
    return _norm_key(letter, acc, mode)


def key_from_profile(profile) -> tuple[str | None, float | None]:
    import numpy as np
    prof = np.asarray(profile, dtype=float)
    if prof.sum() <= 0 or prof.std() == 0:
        return None, None
    best_r, best_key = -2.0, None
    for i in range(12):
        for mode, p in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = float(np.corrcoef(prof, np.roll(p, i))[0, 1])
            if r > best_r:
                best_r, best_key = r, f"{NOTE_NAMES[i]} {mode}"
    return best_key, round(best_r, 3)


# --------------------------------------------------------------------------
# Audio analysis (runs in worker processes)
# --------------------------------------------------------------------------
def estimate_bpm(y, sr, duration: float, name: str):
    import numpy as np
    import librosa
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    est = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0

    # Loops are usually a whole number of beats long; use that to lock the tempo.
    cands = [60.0 * b / duration for b in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)]
    cands = [c for c in cands if 60 <= c <= 200]

    # A bare number in the file name ("..._128_...") counts if the loop length agrees.
    for tok in re.findall(r"(?<![\d.])(\d{2,3})(?![\d.])", name):
        v = float(tok)
        if 60 <= v <= 200 and any(abs(v - c) / c < 0.015 for c in cands):
            return v, "filename"

    if est > 0 and cands:
        def err(c):
            return min(abs(c - e) / c for e in (est, est * 2, est / 2))
        # ties (e.g. 64 vs 128) go to the tempo nearer 120, avoiding half/double-time
        best = min(cands, key=lambda c: (round(err(c), 3), abs(c - 120)))
        if err(best) < 0.04:
            return round(best, 1), "loop length"
    if est <= 0:
        return None, None
    while est < 70:
        est *= 2
    while est > 180:
        est /= 2
    return round(est, 1), "detected"


def analyze_audio(job: tuple[str, float, str | None]) -> dict:
    path_str, max_seconds, category = job
    out = {"path": path_str}
    try:
        import numpy as np
        import librosa
        try:
            duration = float(librosa.get_duration(path=path_str))
        except TypeError:  # older librosa
            duration = float(librosa.get_duration(filename=path_str))
        y, sr = librosa.load(path_str, sr=22050, mono=True, duration=max_seconds)
        out["duration"] = round(duration, 3)
        if y.size < sr * 0.05 or float(np.max(np.abs(y))) < 1e-4:
            out["is_loop"] = 0
            return out

        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        is_loop = duration >= 1.5 and len(onsets) >= 3
        out["is_loop"] = int(is_loop)

        name = Path(path_str).stem
        bpm, bpm_src = bpm_from_name(name), "filename"
        if bpm is None and is_loop:
            bpm, bpm_src = estimate_bpm(y, sr, duration, name)
        out["bpm"] = bpm
        out["bpm_source"] = bpm_src if bpm else None

        key = key_from_name(name)
        if key:
            out.update(musical_key=key, key_source="filename", key_confidence=1.0)
        elif category not in DRUM_CATEGORIES:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            k, conf = key_from_profile(chroma.mean(axis=1))
            out["key_confidence"] = conf
            if k and conf is not None and conf >= 0.5:
                out.update(musical_key=k, key_source="detected")
    except Exception as e:  # keep going on bad files
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS samples(
  path TEXT PRIMARY KEY, name TEXT, folder TEXT, ext TEXT, size INTEGER, mtime REAL,
  category TEXT, duration REAL, is_loop INTEGER, bpm REAL, bpm_source TEXT,
  musical_key TEXT, key_source TEXT, key_confidence REAL, error TEXT, indexed_at REAL);
CREATE TABLE IF NOT EXISTS plugins(
  path TEXT PRIMARY KEY, name TEXT, format TEXT, vendor TEXT, indexed_at REAL);
CREATE TABLE IF NOT EXISTS projects(
  path TEXT PRIMARY KEY, name TEXT, size INTEGER, mtime REAL, live_version TEXT, tempo REAL,
  midi_tracks INTEGER, audio_tracks INTEGER, return_tracks INTEGER, group_tracks INTEGER,
  note_count INTEGER, musical_key TEXT, key_confidence REAL, error TEXT, indexed_at REAL);
CREATE TABLE IF NOT EXISTS project_tracks(
  project TEXT, track TEXT, track_type TEXT, is_drum INTEGER, note_count INTEGER);
CREATE TABLE IF NOT EXISTS project_plugins(project TEXT, plugin TEXT, track TEXT, track_type TEXT);
CREATE TABLE IF NOT EXISTS project_devices(project TEXT, device TEXT, track TEXT, track_type TEXT);
CREATE TABLE IF NOT EXISTS project_samples(project TEXT, sample_path TEXT, track TEXT, track_type TEXT);
CREATE INDEX IF NOT EXISTS ix_samples_cat ON samples(category);
CREATE INDEX IF NOT EXISTS ix_samples_key ON samples(musical_key);
CREATE INDEX IF NOT EXISTS ix_samples_bpm ON samples(bpm);
CREATE INDEX IF NOT EXISTS ix_pp ON project_plugins(project);
CREATE INDEX IF NOT EXISTS ix_pd ON project_devices(project);
CREATE INDEX IF NOT EXISTS ix_ps ON project_samples(project);
CREATE INDEX IF NOT EXISTS ix_pt ON project_tracks(project);
"""


def open_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


# --------------------------------------------------------------------------
# Scanning: samples
# --------------------------------------------------------------------------
def default_sample_dirs() -> list[Path]:
    docs = Path.home() / "Documents"
    return [p for p in (docs / "Ableton" / "User Library", docs / "Splice" / "Samples",
                        docs / "Splice") if p.exists()][:2]


def scan_samples(con, roots: list[Path], workers: int, max_seconds: float,
                 analyze: bool, full: bool):
    print(f"\n[samples] scanning {len(roots)} folder(s)...")
    known = {r["path"]: (r["size"], r["mtime"]) for r in
             con.execute("SELECT path, size, mtime FROM samples")}
    seen, todo = set(), []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                p = Path(dirpath) / f
                if p.suffix.lower() not in AUDIO_EXTS:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                key = str(p)
                seen.add(key)
                if not full and known.get(key) == (st.st_size, st.st_mtime):
                    continue
                todo.append((p, st))

    print(f"[samples] {len(seen)} audio files found, {len(todo)} new or changed")
    now = time.time()
    base_rows = {}
    for p, st in todo:
        name = p.stem
        bpm = bpm_from_name(name)
        k = key_from_name(name)
        base_rows[str(p)] = dict(
            path=str(p), name=p.name, folder=str(p.parent), ext=p.suffix.lower(),
            size=st.st_size, mtime=st.st_mtime, category=categorize(p), duration=None,
            is_loop=None, bpm=bpm, bpm_source="filename" if bpm else None,
            musical_key=k, key_source="filename" if k else None,
            key_confidence=1.0 if k else None, error=None, indexed_at=now)

    def save(rows):
        con.executemany(
            "INSERT OR REPLACE INTO samples VALUES(:path,:name,:folder,:ext,:size,:mtime,"
            ":category,:duration,:is_loop,:bpm,:bpm_source,:musical_key,:key_source,"
            ":key_confidence,:error,:indexed_at)", rows)
        con.commit()

    if analyze and todo:
        jobs = [(k, max_seconds, v["category"]) for k, v in base_rows.items()]
        batch, done, errors = [], 0, 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(analyze_audio, j) for j in jobs]
            for fut in as_completed(futures):
                res = fut.result()
                row = base_rows[res["path"]]
                for k, v in res.items():
                    if v is not None:
                        row[k] = v
                if res.get("error"):
                    errors += 1
                batch.append(row)
                done += 1
                if len(batch) >= 200:
                    save(batch)
                    batch = []
                if done % 25 == 0 or done == len(jobs):
                    print(f"\r[samples] analyzed {done}/{len(jobs)}", end="", flush=True)
        if batch:
            save(batch)
        print(f"\n[samples] done ({errors} file(s) could not be read)")
    elif todo:
        save(list(base_rows.values()))
        print("[samples] saved (audio analysis skipped)")

    # forget files that were deleted from the scanned folders
    root_strs = [str(r) for r in roots]
    gone = [p for p in known if p not in seen and any(p.startswith(r) for r in root_strs)]
    if gone:
        con.executemany("DELETE FROM samples WHERE path=?", [(g,) for g in gone])
        con.commit()
        print(f"[samples] removed {len(gone)} missing file(s) from the catalog")


# --------------------------------------------------------------------------
# Scanning: plugins
# --------------------------------------------------------------------------
def default_plugin_dirs() -> list[Path]:
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    common = os.environ.get("CommonProgramFiles", os.path.join(pf, "Common Files"))
    dirs = [Path(common) / "VST3", Path(common) / "CLAP", Path(common) / "VST2",
            Path(pf) / "VSTPlugins", Path(pf) / "Steinberg" / "VSTPlugins",
            Path(pf86) / "VSTPlugins", Path(pf86) / "Steinberg" / "VSTPlugins"]
    out, seen = [], set()
    for d in dirs:
        if d.exists() and str(d).lower() not in seen:
            seen.add(str(d).lower())
            out.append(d)
    return out


def scan_plugins(con, roots: list[Path]):
    print(f"\n[plugins] scanning {len(roots)} folder(s)...")
    now, rows = time.time(), []

    def add(p: Path, fmt: str, root: Path):
        rel = p.relative_to(root).parts
        vendor = rel[0] if len(rel) > 1 else None
        rows.append((str(p), p.stem, fmt, vendor, now))

    for root in roots:
        for dirpath, dirnames, files in os.walk(root):
            for d in list(dirnames):
                if d.lower().endswith(".vst3"):  # VST3 bundles are folders
                    add(Path(dirpath) / d, "VST3", root)
                    dirnames.remove(d)
            for f in files:
                fmt = PLUGIN_EXTS.get(Path(f).suffix.lower())
                if fmt:
                    add(Path(dirpath) / f, fmt, root)
    con.execute("DELETE FROM plugins")
    con.executemany("INSERT OR REPLACE INTO plugins VALUES(?,?,?,?,?)", rows)
    con.commit()
    print(f"[plugins] {len(rows)} plugins found")


# --------------------------------------------------------------------------
# Scanning: Ableton projects (.als = gzipped XML)
# --------------------------------------------------------------------------
TRACK_TYPES = {"MidiTrack": "midi", "AudioTrack": "audio", "ReturnTrack": "return",
               "GroupTrack": "group"}
DRUM_WORDS = ("drum", "kick", "snare", "hat", "perc", "clap", "808 kit")


def _val(el, path):
    if el is None:
        return None
    f = el.find(path)
    return f.get("Value") if f is not None else None


def _plugin_name(pd):
    for info, field in (("VstPluginInfo", "PlugName"), ("Vst3PluginInfo", "Name"),
                        ("AuPluginInfo", "Name")):
        el = pd.find(f".//{info}/{field}")
        if el is not None and el.get("Value"):
            return el.get("Value")
    return None


def _devices(container):
    out = []
    for devs in container.iter("Devices"):
        for d in devs:
            if d.tag != "PluginDevice":
                out.append(d.tag)
    return out


def _sample_paths(container):
    out = []
    for sref in container.iter("SampleRef"):
        fr = sref.find("FileRef")
        if fr is None:
            continue
        p = _val(fr, "Path") or _val(fr, "RelativePath")
        if not p:  # Live 9 and older
            dirs = [e.get("Dir") for e in fr.iter("RelativePathElement") if e.get("Dir")]
            nm = _val(fr, "Name")
            if nm:
                p = "/".join(dirs + [nm])
        if p:
            out.append(p)
    return out


def _notes(container):
    hist, count = [0.0] * 12, 0
    for kt in container.iter("KeyTrack"):
        mk = _val(kt, "MidiKey")
        if mk is None:
            continue
        pitch = int(float(mk))
        for ev in kt.iter("MidiNoteEvent"):
            if (ev.get("IsEnabled") or "true").lower() == "false":
                continue
            try:
                dur = float(ev.get("Duration") or 0.25)
            except ValueError:
                dur = 0.25
            hist[pitch % 12] += dur
            count += 1
    return hist, count


def parse_als(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        root = ET.parse(f).getroot()
    ls = root.find("LiveSet")
    if ls is None:
        raise ValueError("no LiveSet in file")
    info = {"live_version": root.get("Creator"), "tempo": None, "tracks": []}
    for tag in ("MainTrack", "MasterTrack"):
        v = _val(ls.find(tag), ".//Tempo/Manual")
        if v:
            info["tempo"] = round(float(v), 2)
            break

    containers = []
    tracks_el = ls.find("Tracks")
    if tracks_el is not None:
        for tr in tracks_el:
            containers.append((tr, TRACK_TYPES.get(tr.tag, tr.tag.lower())))
    for tag in ("MainTrack", "MasterTrack"):
        mt = ls.find(tag)
        if mt is not None:
            containers.append((mt, "main"))
            break

    for tr, ttype in containers:
        name = (_val(tr, "Name/EffectiveName") or _val(tr, "Name/UserName")
                or ("Main" if ttype == "main" else ttype))
        devices = _devices(tr) if ttype != "group" else []
        plugins = [n for n in (_plugin_name(pd) for pd in tr.iter("PluginDevice")) if n] \
            if ttype != "group" else []
        hist, count = _notes(tr) if ttype == "midi" else ([0.0] * 12, 0)
        is_drum = "DrumGroupDevice" in devices or any(w in name.lower() for w in DRUM_WORDS)
        info["tracks"].append(dict(
            name=name, type=ttype, devices=devices, plugins=plugins,
            samples=_sample_paths(tr) if ttype != "group" else [],
            hist=hist, note_count=count, is_drum=is_drum))
    return info


def default_project_dirs() -> list[Path]:
    docs = Path.home() / "Documents"
    return [p for p in (docs / "Ableton" / "User Library",) if p.exists()]


def scan_projects(con, roots: list[Path], full: bool):
    print(f"\n[projects] scanning {len(roots)} folder(s)...")
    known = {r["path"]: (r["size"], r["mtime"]) for r in
             con.execute("SELECT path, size, mtime FROM projects")}
    files = []
    for root in roots:
        for dirpath, dirnames, fnames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() != "backup"]
            files += [Path(dirpath) / f for f in fnames if f.lower().endswith(".als")]
    print(f"[projects] {len(files)} Live sets found")

    parsed = errors = 0
    for i, p in enumerate(files, 1):
        st = p.stat()
        key = str(p)
        if not full and known.get(key) == (st.st_size, st.st_mtime):
            continue
        for t in ("project_tracks", "project_plugins", "project_devices", "project_samples"):
            con.execute(f"DELETE FROM {t} WHERE project=?", (key,))
        row = dict(path=key, name=p.stem, size=st.st_size, mtime=st.st_mtime,
                   live_version=None, tempo=None, midi_tracks=0, audio_tracks=0,
                   return_tracks=0, group_tracks=0, note_count=0, musical_key=None,
                   key_confidence=None, error=None, indexed_at=time.time())
        try:
            info = parse_als(p)
            row.update(live_version=info["live_version"], tempo=info["tempo"])
            hist, total_notes = [0.0] * 12, 0
            for t in info["tracks"]:
                col = f"{t['type']}_tracks"
                if col in row:
                    row[col] += 1
                total_notes += t["note_count"]
                if not t["is_drum"]:
                    hist = [a + b for a, b in zip(hist, t["hist"])]
                con.execute("INSERT INTO project_tracks VALUES(?,?,?,?,?)",
                            (key, t["name"], t["type"], int(t["is_drum"]), t["note_count"]))
                con.executemany("INSERT INTO project_plugins VALUES(?,?,?,?)",
                                [(key, n, t["name"], t["type"]) for n in t["plugins"]])
                con.executemany("INSERT INTO project_devices VALUES(?,?,?,?)",
                                [(key, d, t["name"], t["type"]) for d in t["devices"]])
                con.executemany("INSERT INTO project_samples VALUES(?,?,?,?)",
                                [(key, s, t["name"], t["type"]) for s in t["samples"]])
            row["note_count"] = total_notes
            if sum(hist) > 0:
                row["musical_key"], row["key_confidence"] = key_from_profile(hist)
            parsed += 1
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            errors += 1
        con.execute(
            "INSERT OR REPLACE INTO projects VALUES(:path,:name,:size,:mtime,:live_version,"
            ":tempo,:midi_tracks,:audio_tracks,:return_tracks,:group_tracks,:note_count,"
            ":musical_key,:key_confidence,:error,:indexed_at)", row)
        if i % 20 == 0:
            con.commit()
            print(f"\r[projects] {i}/{len(files)}", end="", flush=True)
    con.commit()
    print(f"\r[projects] {parsed} parsed, {errors} could not be read" + " " * 10)


# --------------------------------------------------------------------------
# Style profile
# --------------------------------------------------------------------------
def build_profile(con) -> dict:
    q = lambda sql, *a: con.execute(sql, a).fetchall()
    prof: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M")}

    prof["library"] = {
        "samples": q("SELECT COUNT(*) FROM samples")[0][0],
        "loops": q("SELECT COUNT(*) FROM samples WHERE is_loop=1")[0][0],
        "one_shots": q("SELECT COUNT(*) FROM samples WHERE is_loop=0")[0][0],
        "by_category": {r[0] or "uncategorized": r[1] for r in q(
            "SELECT category, COUNT(*) c FROM samples GROUP BY category ORDER BY c DESC")},
        "plugins_by_format": {r[0]: r[1] for r in q(
            "SELECT format, COUNT(*) FROM plugins GROUP BY format")},
    }

    projects = q("SELECT * FROM projects WHERE error IS NULL")
    n = len(projects)
    tempos = [p["tempo"] for p in projects if p["tempo"]]
    keys = Counter(p["musical_key"] for p in projects
                   if p["musical_key"] and (p["key_confidence"] or 0) >= 0.6)
    prof["projects"] = {
        "count": n,
        "tempo_median": round(statistics.median(tempos), 1) if tempos else None,
        "tempo_range": [min(tempos), max(tempos)] if tempos else None,
        "top_tempos": Counter(round(t) for t in tempos).most_common(10),
        "top_keys": keys.most_common(8),
        "minor_share": round(sum(c for k, c in keys.items() if k.endswith("minor"))
                             / max(1, sum(keys.values())), 2),
        "avg_tracks": {c: round(sum(p[c] for p in projects) / n, 1) for c in
                       ("midi_tracks", "audio_tracks", "return_tracks", "group_tracks")}
        if n else {},
    }

    def top_by_projects(table, col, where="", limit=25):
        return [(r[0], r[1]) for r in q(
            f"SELECT {col}, COUNT(DISTINCT project) c FROM {table} {where} "
            f"GROUP BY {col} ORDER BY c DESC LIMIT {limit}")]

    prof["plugins_most_used"] = top_by_projects("project_plugins", "plugin")
    prof["devices_most_used"] = top_by_projects("project_devices", "device")
    prof["plugins_by_track_type"] = {
        t: top_by_projects("project_plugins", "plugin", f"WHERE track_type='{t}'", 10)
        for t in ("midi", "audio", "return", "main")}

    used = q("SELECT sample_path, COUNT(DISTINCT project) c FROM project_samples "
             "GROUP BY sample_path ORDER BY c DESC LIMIT 25")
    prof["samples_most_used"] = [(Path(r[0].replace("\\", "/")).name, r[1]) for r in used]

    cat_counts = Counter()
    for r in q("SELECT sample_path FROM project_samples"):
        cat_counts[categorize(Path(r[0].replace("\\", "/"))) or "uncategorized"] += 1
    prof["sample_categories_used"] = cat_counts.most_common(15)
    return prof


def print_profile(prof: dict):
    lib, pj = prof["library"], prof["projects"]
    print("\n================ YOUR MUSIC PROFILE ================")
    print(f"Library: {lib['samples']} samples ({lib['loops']} loops, "
          f"{lib['one_shots']} one-shots), plugins: {lib['plugins_by_format']}")
    print(f"Projects analyzed: {pj['count']}")
    if pj["tempo_median"]:
        print(f"Tempo: median {pj['tempo_median']} BPM, range {pj['tempo_range']}")
        print("Most used tempos: " + ", ".join(f"{t} ({c})" for t, c in pj["top_tempos"][:6]))
    if pj["top_keys"]:
        print("Most used keys: " + ", ".join(f"{k} ({c})" for k, c in pj["top_keys"][:6])
              + f"  | minor share {int(pj['minor_share'] * 100)}%")
    if prof["plugins_most_used"]:
        print("Top plugins: " + ", ".join(f"{p} ({c})" for p, c in prof["plugins_most_used"][:10]))
    if prof["devices_most_used"]:
        print("Top Live devices: " + ", ".join(f"{d} ({c})" for d, c in prof["devices_most_used"][:10]))
    if prof["samples_most_used"]:
        print("Go-to samples: " + ", ".join(f"{s} ({c})" for s, c in prof["samples_most_used"][:8]))
    print("====================================================")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_scan(a):
    db = Path(a.db)
    con = open_db(db)
    t0 = time.time()
    if not a.skip_samples:
        roots = [Path(p) for p in a.samples] if a.samples else default_sample_dirs()
        roots = [r for r in roots if r.exists()]
        if roots:
            scan_samples(con, roots, a.workers, a.max_seconds, not a.no_audio_analysis, a.full)
        else:
            print("[samples] no sample folders found - pass them with --samples")
    if not a.skip_plugins:
        roots = [Path(p) for p in a.plugins] if a.plugins else default_plugin_dirs()
        scan_plugins(con, [r for r in roots if r.exists()])
    if not a.skip_projects:
        roots = [Path(p) for p in a.projects] if a.projects else default_project_dirs()
        roots = [r for r in roots if r.exists()]
        if roots:
            scan_projects(con, roots, a.full)
        else:
            print("[projects] no project folders found - pass them with --projects")
    prof = build_profile(con)
    out = db.with_name("style_profile.json")
    out.write_text(json.dumps(prof, indent=2), encoding="utf-8")
    print_profile(prof)
    print(f"\nCatalog: {db}\nProfile: {out}\nFinished in {time.time() - t0:.0f}s")


def cmd_search(a):
    con = open_db(Path(a.db))
    sql, args = "SELECT * FROM samples WHERE 1=1", []
    for w in a.words:
        sql += " AND lower(path) LIKE ?"
        args.append(f"%{w.lower()}%")
    if a.category:
        sql += " AND category=?"
        args.append(a.category.lower())
    if a.key:
        k = normalize_key_query(a.key)
        if not k:
            sys.exit(f"Could not understand key '{a.key}'. Try 'Am', 'F# minor' or 'C major'.")
        sql += " AND musical_key=?"
        args.append(k)
    if a.bpm:
        sql += " AND bpm BETWEEN ? AND ?"
        args += [a.bpm - a.bpm_tol, a.bpm + a.bpm_tol]
    if a.loops:
        sql += " AND is_loop=1"
    if a.oneshots:
        sql += " AND is_loop=0"
    sql += " ORDER BY name LIMIT ?"
    args.append(a.limit)
    rows = con.execute(sql, args).fetchall()
    for r in rows:
        bits = [r["category"] or "-",
                f"{r['bpm']:g} BPM" if r["bpm"] else "",
                r["musical_key"] or "",
                "loop" if r["is_loop"] == 1 else ("one-shot" if r["is_loop"] == 0 else "")]
        print(f"{r['name']}  [{', '.join(b for b in bits if b)}]\n    {r['path']}")
    print(f"\n{len(rows)} result(s)")


def cmd_profile(a):
    con = open_db(Path(a.db))
    prof = build_profile(con)
    Path(a.db).with_name("style_profile.json").write_text(json.dumps(prof, indent=2),
                                                          encoding="utf-8")
    print_profile(prof)


def main():
    default_db = str(app_dir() / "music_library.db")
    ap = argparse.ArgumentParser(description="Music Assistant - library indexer")
    ap.add_argument("--db", default=default_db, help="catalog file (default: %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="index samples, plugins and Ableton projects")
    s.add_argument("--samples", nargs="+", metavar="DIR", help="sample folders")
    s.add_argument("--projects", nargs="+", metavar="DIR", help="folders with .als projects")
    s.add_argument("--plugins", nargs="+", metavar="DIR", help="extra plugin folders")
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.add_argument("--max-seconds", type=float, default=30.0,
                   help="seconds of each file to analyze (default 30)")
    s.add_argument("--no-audio-analysis", action="store_true",
                   help="fast mode: names/folders only, no BPM/key detection")
    s.add_argument("--full", action="store_true", help="re-index everything")
    s.add_argument("--skip-samples", action="store_true")
    s.add_argument("--skip-plugins", action="store_true")
    s.add_argument("--skip-projects", action="store_true")
    s.set_defaults(func=cmd_scan)

    q = sub.add_parser("search", help="find samples")
    q.add_argument("words", nargs="*", help="words in the file name or folder")
    q.add_argument("--category", help="e.g. kick, snare, 808, bass, vocal, pad, fx")
    q.add_argument("--key", help="e.g. Am, 'F# minor', C")
    q.add_argument("--bpm", type=float)
    q.add_argument("--bpm-tol", type=float, default=2.0)
    g = q.add_mutually_exclusive_group()
    g.add_argument("--loops", action="store_true")
    g.add_argument("--oneshots", action="store_true")
    q.add_argument("--limit", type=int, default=30)
    q.set_defaults(func=cmd_search)

    p = sub.add_parser("profile", help="rebuild and show your style profile")
    p.set_defaults(func=cmd_profile)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
