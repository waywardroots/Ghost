"""
Phase 3, step 1: turn your Ableton projects into a MIDI training set.

Reads every Live set the indexer found (or the folders you pass) and writes one line per
unique MIDI clip to MusicAssistant/midi_dataset.jsonl:

    project, track, role (drums/bass/chords/pad/lead/arp/other), instrument, tempo, key,
    length_beats and the notes [pitch, start, duration, velocity] (beats from clip start).

Your projects are only read, never changed.

    py -3.13 midi_dataset.py                # every project in the library database
    py -3.13 midi_dataset.py --projects "C:/Users/me/Desktop/Project Files"
    py -3.13 midi_dataset.py --stats        # summary of an existing dataset
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from indexer import app_dir, key_from_profile

ROLE_WORDS = [  # checked in order; the first match wins
    ("drums", ("drum", "kick", "snare", "hat", "hihat", "clap", "perc", "tom", "ride",
               "crash", "cymbal", "shaker", "rim", "conga", "bongo", "808 kit", "909", "beat")),
    ("bass", ("bass", "sub", "reese", "808", "low end")),
    ("arp", ("arp", "seq", "sequence")),
    ("pad", ("pad", "atmos", "string", "choir", "drone", "ambient", "texture")),
    ("chords", ("chord", "stab", "keys", "piano", "rhodes", "organ", "epiano", "e-piano")),
    ("lead", ("lead", "melod", "hook", "pluck", "vox", "vocal", "bell")),
    ("fx", ("fx", "riser", "sweep", "impact", "noise")),
]
ROLES = ("drums", "bass", "chords", "pad", "lead", "arp", "fx", "other")


def _val(el, path):
    if el is None:
        return None
    f = el.find(path)
    return f.get("Value") if f is not None else None


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def instrument_of(track) -> str | None:
    """The first instrument on the track: a plugin name or a Live device."""
    for devs in track.iter("Devices"):
        for d in devs:
            if d.tag == "PluginDevice":
                for info, field in (("VstPluginInfo", "PlugName"), ("Vst3PluginInfo", "Name"),
                                    ("AuPluginInfo", "Name")):
                    el = d.find(f".//{info}/{field}")
                    if el is not None and el.get("Value"):
                        return el.get("Value")
            elif d.tag in ("InstrumentGroupDevice", "DrumGroupDevice", "OriginalSimpler",
                           "MultiSampler", "Operator", "UltraAnalog", "InstrumentVector",
                           "Collision", "LoungeLizard", "StringStudio", "Drift", "Meld",
                           "InstrumentImpulse"):
                return d.tag
        break
    return None


def clip_notes(clip) -> list[list[float]]:
    notes = []
    for kt in clip.iter("KeyTrack"):
        mk = _val(kt, "MidiKey")
        if mk is None:
            continue
        pitch = int(_f(mk))
        for ev in kt.iter("MidiNoteEvent"):
            if (ev.get("IsEnabled") or "true").lower() == "false":
                continue
            notes.append([pitch, round(_f(ev.get("Time")), 4),
                          round(_f(ev.get("Duration"), 0.25), 4),
                          int(round(_f(ev.get("Velocity"), 100)))])
    notes.sort(key=lambda n: (n[1], n[0]))
    return notes


def clip_window(clip) -> tuple[float, float]:
    """The part of the clip that plays: the loop if it's on, otherwise start..end."""
    loop = clip.find("Loop")
    if loop is not None and (_val(loop, "LoopOn") or "false").lower() == "true":
        a, b = _f(_val(loop, "LoopStart")), _f(_val(loop, "LoopEnd"))
        if b > a:
            return a, b
    a = _f(_val(clip, "CurrentStart"))
    b = _f(_val(clip, "CurrentEnd"), a + 4)
    if loop is not None:  # CurrentStart/End are timeline positions; the content starts here
        start = _f(_val(loop, "StartRelative")) + _f(_val(loop, "LoopStart"))
        return start, start + (b - a)
    return 0.0, b - a


def note_role(notes: list, length: float) -> str:
    """Guess a clip's job from its notes alone (most tracks are just 'MIDI' or '12-Diva')."""
    if not notes:
        return "other"
    starts = Counter(round(n[1], 2) for n in notes)
    poly = len(notes) / len(starts)
    mean_pitch = sum(n[0] for n in notes) / len(notes)
    durs = sorted(n[2] for n in notes)
    med_dur = durs[len(durs) // 2]
    density = len(starts) / max(length, 1.0)  # note onsets per beat
    span = max(n[0] for n in notes) - min(n[0] for n in notes)
    if span == 0 and med_dur >= 8:
        return "fx"       # one long held note: a drone, riser or noise sweep
    if poly >= 1.8:
        return "pad" if med_dur >= 2 else "chords"
    if density >= 1.5 and med_dur <= 0.5 and span >= 5:
        return "arp"
    if mean_pitch < 56:   # Live's C3 = 60; synth basses are usually written around C1-C2
        return "bass"
    return "lead"


def guess_role(name: str, instrument: str | None, notes: list, length: float = 4.0) -> str:
    text = f"{name} {instrument or ''}".lower()
    if instrument == "DrumGroupDevice":
        return "drums"
    for role, words in ROLE_WORDS:  # a name that says what it is wins
        if any(w in text for w in words):
            return role
    return note_role(notes, length)


def extract(path: Path) -> list[dict]:
    with gzip.open(path, "rb") as f:
        root = ET.parse(f).getroot()
    ls = root.find("LiveSet")
    if ls is None:
        return []
    tempo = None
    for tag in ("MainTrack", "MasterTrack"):
        v = _val(ls.find(tag), ".//Tempo/Manual")
        if v:
            tempo = round(_f(v), 2)
            break
    tracks = ls.find("Tracks")
    clips, seen, hist = [], set(), [0.0] * 12
    for tr in (tracks if tracks is not None else []):
        if tr.tag != "MidiTrack":
            continue
        name = _val(tr, "Name/EffectiveName") or _val(tr, "Name/UserName") or "MIDI"
        inst = instrument_of(tr)
        for clip in tr.iter("MidiClip"):
            a, b = clip_window(clip)
            length = round(b - a, 4)
            if length <= 0 or length > 256:
                continue
            notes = [[p, round(s - a, 4), d, v] for p, s, d, v in clip_notes(clip)
                     if a - 1e-6 <= s < b - 1e-6]
            if not notes:
                continue
            digest = hashlib.sha1(json.dumps([length, notes]).encode()).hexdigest()[:16]
            if digest in seen:  # the same clip copied around the arrangement
                continue
            seen.add(digest)
            role = guess_role(name, inst, notes, length)
            if role != "drums":
                for p, _, d, _ in notes:
                    hist[p % 12] += d
            clips.append(dict(project=str(path), track=name, clip=_val(clip, "Name") or "",
                              role=role, instrument=inst, tempo=tempo,
                              length_beats=length, notes=notes, id=digest))
    key = key_from_profile(hist)[0] if sum(hist) else None
    for c in clips:
        c["key"] = key
    return clips


def project_files(args) -> list[Path]:
    if args.projects:
        out = []
        for root in args.projects:
            for dirpath, dirnames, fnames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d.lower() != "backup"]
                out += [Path(dirpath) / f for f in fnames if f.lower().endswith(".als")]
        return out
    db = Path(args.db)
    if not db.exists():
        sys.exit(f"No library database at {db}. Run indexer.py scan first, or pass --projects.")
    con = sqlite3.connect(db)
    rows = con.execute("SELECT path FROM projects WHERE error IS NULL").fetchall()
    return [Path(r[0]) for r in rows]


def stats(out: Path):
    roles, keys, insts, n, notes = Counter(), Counter(), Counter(), 0, 0
    with open(out, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            n += 1
            notes += len(c["notes"])
            roles[c["role"]] += 1
            insts[c["instrument"] or "?"] += 1
            if c.get("key"):
                keys[c["key"]] += 1
    print(f"{n} unique MIDI clips, {notes} notes")
    print("By role:   " + ", ".join(f"{r} {k}" for r, k in roles.most_common()))
    print("By key:    " + ", ".join(f"{r} {k}" for r, k in keys.most_common(8)))
    print("By instrument: " + ", ".join(f"{r} {k}" for r, k in insts.most_common(10)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", nargs="+", metavar="DIR", help="folders of .als files "
                    "(default: every project in the library database)")
    ap.add_argument("--db", default=str(app_dir() / "music_library.db"))
    ap.add_argument("--out", default=str(app_dir() / "midi_dataset.jsonl"))
    ap.add_argument("--stats", action="store_true", help="only summarise an existing dataset")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.stats:
        return stats(out)
    files = project_files(args)
    print(f"Reading MIDI from {len(files)} Live sets...")
    total = errors = 0
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for i, p in enumerate(files, 1):
            try:
                for c in extract(p):
                    f.write(json.dumps(c) + "\n")
                    total += 1
            except Exception as e:  # a damaged or very old set: skip it
                errors += 1
                print(f"\n  skipped {p.name}: {type(e).__name__}: {e}")
            if i % 10 == 0 or i == len(files):
                print(f"\r  {i}/{len(files)} sets, {total} clips", end="", flush=True)
    os.replace(tmp, out)
    print(f"\nSaved {total} clips to {out}" + (f" ({errors} sets skipped)" if errors else ""))
    stats(out)


if __name__ == "__main__":
    main()
