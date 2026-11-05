"""
Read the arrangement of any Ableton Live set (.als) without opening it in Live:
tempo, length, locators, and for every track the bar ranges where it plays.
Used to model a new song's structure on one of the user's finished songs.

    python arrangement_map.py "path/to/song.als"
"""
from __future__ import annotations

import gzip
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

KINDS = {"MidiTrack": "midi", "AudioTrack": "audio", "GroupTrack": "group", "ReturnTrack": "return"}


def _val(el, path):
    if el is None:
        return None
    f = el.find(path)
    return f.get("Value") if f is not None else None


def _spans(events, gap=0.01):
    out = []
    for s, e in sorted(events):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def read_arrangement(path: str) -> dict:
    p = Path(path)
    if p.suffix.lower() != ".als" or not p.is_file():
        raise ValueError(f"not an Ableton Live set: {path}")
    with gzip.open(p, "rb") as f:
        root = ET.parse(f).getroot()
    ls = root.find("LiveSet")
    tempo = None
    for tag in ("MainTrack", "MasterTrack"):
        t = _val(ls.find(tag), ".//Tempo/Manual")
        if t:
            tempo = round(float(t), 2)
            break
    bpb = 4.0  # 4/4
    bar = lambda beats: round(beats / bpb + 1, 2)
    groups, tracks, end = {}, [], 0.0
    for tr in ls.find("Tracks"):
        name = _val(tr, "Name/EffectiveName") or _val(tr, "Name/UserName") or tr.tag
        kind = KINDS.get(tr.tag, tr.tag)
        if kind == "group":
            groups[tr.get("Id")] = name
        events = []
        for arr in tr.iter("ArrangerAutomation"):
            evs = arr.find("Events")
            if evs is None:
                continue
            for c in evs:
                if c.tag in ("MidiClip", "AudioClip"):
                    s = float(c.get("Time") or _val(c, "CurrentStart") or 0)
                    e = float(_val(c, "CurrentEnd") or s)
                    if _val(c, "Disabled") == "true":
                        continue
                    events.append((s, e))
        spans = _spans(events)
        if spans:
            end = max(end, spans[-1][1])
        mute = _val(tr, ".//Speaker/Manual") == "false"
        tracks.append({"name": name, "type": kind, "group": groups.get(_val(tr, "TrackGroupId")),
                       "muted": mute,
                       "plays": [[bar(s), bar(e)] for s, e in spans],
                       "clips": len(events)})
    locators = sorted(((float(l.find("Time").get("Value")), _val(l, "Name"))
                       for l in ls.iter("Locator") if l.find("Time") is not None))
    return {"file": p.name, "tempo": tempo, "length_bars": round(end / bpb, 2),
            "locators": [{"bar": bar(t), "name": n} for t, n in locators],
            "tracks": [t for t in tracks if t["type"] != "group" and t["plays"]],
            "note": "plays = bar ranges (start inclusive, end exclusive) where the track has "
                    "clips on the Arrangement timeline"}


def sections_by_density(info: dict, step: int = 8) -> list[dict]:
    """Rough energy map: how many tracks play in each 8-bar block (helps spot drops/breakdowns)."""
    out, n = [], int(info["length_bars"] // step) + 1
    for i in range(n):
        a, b = 1 + i * step, 1 + (i + 1) * step
        active = [t["name"] for t in info["tracks"] if not t["muted"] and
                  any(s < b and e > a for s, e in t["plays"])]
        out.append({"bars": f"{a}-{b - 1}", "tracks_playing": len(active)})
    return out


if __name__ == "__main__":
    import json
    info = read_arrangement(sys.argv[1])
    info["energy"] = sections_by_density(info)
    print(json.dumps(info, indent=1))
