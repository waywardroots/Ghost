"""
Your-style pattern library (Phase 3, step 2).

Finds MIDI clips you wrote (from midi_dataset.jsonl) that fit what's needed - a role, a key,
a tempo, a length - and turns one into a fresh part: moved into the song's key, fitted to
the length, and varied a little so it's a new take in your style rather than a copy.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

NOTE_PCS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = {"minor": [0, 2, 3, 5, 7, 8, 10], "major": [0, 2, 4, 5, 7, 9, 11]}
PITCHED = ("bass", "chords", "pad", "lead", "arp", "fx", "other")


class StyleError(Exception):
    pass


def parse_key(text) -> tuple[int, str] | None:
    """'D minor', 'Dm', 'F# major', 'Bb' -> (pitch class, 'minor'|'major')."""
    if not text:
        return None
    m = re.match(r"\s*([A-Ga-g])\s*([#b♯♭]?)\s*(.*)$", str(text))
    if not m:
        raise StyleError(f"can't read the key '{text}' (try 'D minor' or 'F# major')")
    pc = NOTE_PCS[m.group(1).lower()] + {"#": 1, "♯": 1, "b": -1, "♭": -1}.get(m.group(2), 0)
    rest = m.group(3).strip().lower()
    minor = rest.startswith("m") and not rest.startswith("maj") or rest.startswith("min")
    return pc % 12, "minor" if minor else "major"


def key_name(k) -> str:
    return f"{NAMES[k[0]]} {k[1]}"


def shift_between(src, dst) -> int:
    """Semitones that move a pattern in key src into key dst, using the relative key when the
    modes differ (a minor pattern in a major song keeps its intervals in the same scale).
    Picks the smaller move (-6..+5)."""
    s_pc, s_mode = src
    d_pc, d_mode = dst
    if s_mode != d_mode:
        # the relative minor sits 3 semitones below its major (A minor / C major)
        d_pc = (d_pc + (-3 if d_mode == "major" else 3)) % 12
    shift = (d_pc - s_pc) % 12
    return shift - 12 if shift > 5 else shift


class Library:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._clips, self._mtime = [], None

    def clips(self) -> list[dict]:
        if not self.path.exists():
            raise StyleError(f"no pattern library yet ({self.path}). Run "
                             "'py -3.13 midi_dataset.py' in the Ghost folder first.")
        mt = self.path.stat().st_mtime
        if mt != self._mtime:
            with open(self.path, encoding="utf-8") as f:
                self._clips = [json.loads(line) for line in f if line.strip()]
            self._mtime = mt
        return self._clips

    def get(self, clip_id: str) -> dict:
        for c in self.clips():
            if c["id"] == clip_id:
                return c
        raise StyleError(f"no pattern with id {clip_id}")

    def find(self, role, key=None, tempo=None, bars=None, instrument=None, query=None,
             limit=10, beats_per_bar=4, exclude=None) -> list[dict]:
        role = (role or "").lower()
        want_key = parse_key(key) if key else None
        scored = []
        for c in self.clips():
            if role and c["role"] != role:
                continue
            if instrument and instrument.lower() not in (c.get("instrument") or "").lower():
                continue
            if query and query.lower() not in f"{c['project']} {c['track']} {c['clip']}".lower():
                continue
            if exclude and exclude.lower() in project_name(c["project"]).lower():
                continue
            score = 0.0
            ck = None
            if c.get("key"):
                try:
                    ck = parse_key(c["key"])
                except StyleError:
                    ck = None
            if want_key and role in PITCHED:
                if ck is None:
                    score -= 2
                else:
                    score += 3 if ck == want_key else (2 if ck[1] == want_key[1] else 0)
                    score -= abs(shift_between(ck, want_key)) * 0.15
            if tempo and c.get("tempo"):
                score -= min(abs(c["tempo"] - tempo), 40) / 8.0
            if bars:
                cb = c["length_beats"] / beats_per_bar
                score -= 0 if cb == bars or (bars % cb == 0 if cb else False) else 1.5
            n = len(c["notes"])
            score += min(n, 64) / 64.0  # a little preference for fuller patterns
            scored.append((score, c, ck))
        scored.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for score, c, ck in scored:
            if c["id"] in seen:  # the same part saved in several versions of a song
                continue
            seen.add(c["id"])
            if len(out) >= limit:
                break
            out.append(describe(c, ck, want_key if role in PITCHED else None, beats_per_bar,
                                round(score, 2)))
        return out


def project_name(path: str) -> str:
    """'C:\\Users\\me\\Song Project\\Song.als' -> 'Song' (works for Windows paths anywhere)."""
    name = re.split(r"[\\/]", str(path).rstrip("\\/"))[-1]
    return name[:-4] if name.lower().endswith(".als") else name


def describe(c, ck=None, want_key=None, beats_per_bar=4, score=None) -> dict:
    notes = c["notes"]
    pitches = [n[0] for n in notes]
    starts = {round(n[1], 2) for n in notes}
    d = {"id": c["id"], "project": project_name(c["project"]), "track": c["track"],
         "clip": c.get("clip") or "", "role": c["role"], "instrument": c.get("instrument"),
         "key": c.get("key"), "tempo": c.get("tempo"),
         "bars": round(c["length_beats"] / beats_per_bar, 2), "notes": len(notes),
         "range": f"{NAMES[min(pitches) % 12]}{min(pitches) // 12 - 2}-"
                  f"{NAMES[max(pitches) % 12]}{max(pitches) // 12 - 2}",
         "notes_per_beat": round(len(starts) / max(c["length_beats"], 1), 2)}
    if want_key and ck:
        d["transpose_to_fit"] = shift_between(ck, want_key)
    if score is not None:
        d["match"] = score
    return d


def make_part(c, key=None, bars=None, variation=0.3, seed=None, octave=0,
              beats_per_bar=4) -> tuple[list[dict], dict]:
    """Turn a stored clip into notes for a new clip: in key, fitted to bars, varied."""
    rng = random.Random(seed if seed is not None else c["id"])
    notes = [list(n) for n in c["notes"]]
    length = float(c["length_beats"])
    info = {"from": describe(c, beats_per_bar=beats_per_bar)}
    pitched = c["role"] in PITCHED
    shift = 0
    if key and pitched:
        want = parse_key(key)
        have = parse_key(c["key"]) if c.get("key") else None
        if have:
            shift = shift_between(have, want)
            info["transposed"] = shift
        else:
            info["note"] = "the original's key is unknown, so it wasn't transposed"
    shift += 12 * int(octave)
    target = float(bars) * beats_per_bar if bars else length
    out = []
    reps = int(target // length) + (1 if target % length else 0)
    for r in range(max(1, reps)):
        for p, s, d, v in notes:
            t = s + r * length
            if t >= target - 1e-6:
                continue
            out.append([p + (shift if pitched else 0), t, min(d, target - t), v])
    variation = max(0.0, min(1.0, float(variation)))
    if variation > 0:
        out = _vary(out, c["role"], key, variation, rng, target)
    out.sort(key=lambda n: (n[1], n[0]))
    info["bars"] = round(target / beats_per_bar, 2)
    info["note_count"] = len(out)
    info["variation"] = variation
    return [{"pitch": int(p), "start": round(s, 4), "duration": round(max(d, 0.02), 4),
             "velocity": int(max(1, min(127, v)))} for p, s, d, v in out], info


def _scale_pcs(key):
    try:
        k = parse_key(key) if key else None
    except StyleError:
        k = None
    return {(k[0] + i) % 12 for i in SCALES[k[1]]} if k else None


def _vary(notes, role, key, amount, rng, target):
    """Small, musical changes: humanised velocity, a few notes dropped off the main beats,
    an occasional neighbour scale note in melodic parts, and an added ghost note or two."""
    scale = _scale_pcs(key)
    on_beat = lambda t: abs(t - round(t)) < 1e-3
    out = []
    for p, s, d, v in notes:
        v = v + rng.randint(-int(10 * amount) - 2, int(10 * amount) + 2)
        if not on_beat(s) and rng.random() < 0.25 * amount:
            continue  # thin out a little
        if role in ("lead", "arp", "bass") and scale and rng.random() < 0.15 * amount:
            step = rng.choice((-2, -1, 1, 2))
            q = p + step
            if q % 12 in scale:
                p = q
        out.append([p, s, d, v])
    # ghost notes in a few empty 16th slots, echoing a nearby note
    taken = {round(n[1] * 4) for n in out}
    slots = [i for i in range(int(target * 4)) if i not in taken]
    for i in rng.sample(slots, k=min(len(slots), int(len(out) * 0.12 * amount))):
        t = i / 4.0
        near = min(out, key=lambda n: abs(n[1] - t), default=None)
        if near:
            out.append([near[0], t, min(0.2, near[2]), int(near[3] * 0.55)])
    return out
