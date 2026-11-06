"""Checks midi_dataset.py against a small hand-made Live set."""
import gzip
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import midi_dataset as md  # noqa: E402

FAILS = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        FAILS.append(name)


def clip(name, notes, loop=(0, 4), loop_on=True, start=0, end=4, rel=0):
    keys = {}
    for p, t, d, v, *en in notes:
        keys.setdefault(p, []).append(
            f'<MidiNoteEvent Time="{t}" Duration="{d}" Velocity="{v}" '
            f'IsEnabled="{"false" if en and not en[0] else "true"}"/>')
    kts = "".join(f'<KeyTrack Id="{i}"><Notes>{"".join(ev)}</Notes><MidiKey Value="{p}"/></KeyTrack>'
                  for i, (p, ev) in enumerate(keys.items()))
    return (f'<MidiClip Time="{start}"><CurrentStart Value="{start}"/><CurrentEnd Value="{end}"/>'
            f'<Loop><LoopStart Value="{loop[0]}"/><LoopEnd Value="{loop[1]}"/>'
            f'<StartRelative Value="{rel}"/><LoopOn Value="{"true" if loop_on else "false"}"/></Loop>'
            f'<Name Value="{name}"/><Notes><KeyTracks>{kts}</KeyTracks></Notes></MidiClip>')


def track(name, clips, device):
    return (f'<MidiTrack><Name><EffectiveName Value="{name}"/></Name>'
            f'<DeviceChain><DeviceChain><Devices>{device}</Devices></DeviceChain></DeviceChain>'
            f'<ClipSlotList><ClipSlot><ClipSlot><Value>{clips[0]}</Value></ClipSlot></ClipSlot></ClipSlotList>'
            f'<ArrangerAutomation><Events>{"".join(clips[1:])}</Events></ArrangerAutomation></MidiTrack>')


serum = '<PluginDevice><PluginDesc><VstPluginInfo><PlugName Value="Serum_x64"/></VstPluginInfo></PluginDesc></PluginDevice>'
# D minor bassline (D, F, A, C) - the loop is the first 4 beats only; a note at beat 5 is outside it
bass_notes = [(38, 0, 0.5, 110), (41, 1, 0.5, 100), (45, 2, 0.5, 100), (49, 3, 0.5, 90),
              (38, 3.5, 0.25, 60, False), (38, 5, 0.5, 100)]
bass = clip("Roller", bass_notes, loop=(0, 4))
bass_copy = clip("Roller", bass_notes, loop=(0, 4), start=16, end=20)  # same clip on the timeline
chords = clip("Dm", [(62, 0, 0.5, 80), (65, 0, 0.5, 80), (69, 0, 0.5, 80), (67, 4, 0.5, 80),
                     (70, 4, 0.5, 80), (74, 4, 0.5, 80)], loop=(0, 8))  # Dm, Gm stabs
drums = clip("Beat", [(36, 0, 0.25, 127), (36, 1, 0.25, 127), (42, 0.5, 0.1, 90)])
xml = ('<?xml version="1.0"?><Ableton Creator="Ableton Live 12.1"><LiveSet><Tracks>'
       + track("Bass", [bass, bass_copy], serum)
       + track("MIDI 3", [chords], serum)
       + track("909", [drums], "<DrumGroupDevice/>")
       + '</Tracks><MainTrack><DeviceChain><Mixer><Tempo><Manual Value="126"/></Tempo></Mixer>'
       + '</DeviceChain></MainTrack></LiveSet></Ableton>')

with tempfile.TemporaryDirectory() as d:
    als = Path(d) / "song.als"
    with gzip.open(als, "wb") as f:
        f.write(xml.encode())
    clips = md.extract(als)
    by = {c["track"]: c for c in clips}
    check("one clip per unique pattern (timeline copy dropped)", len(clips) == 3)
    b = by["Bass"]
    check("bass role, instrument and tempo", b["role"] == "bass" and b["instrument"] == "Serum_x64"
          and b["tempo"] == 126)
    check("loop window kept, outside + disabled notes dropped",
          b["length_beats"] == 4 and [n[0] for n in b["notes"]] == [38, 41, 45, 49])
    check("unnamed polyphonic track read as chords", by["MIDI 3"]["role"] == "chords")
    check("drum rack track read as drums", by["909"]["role"] == "drums"
          and by["909"]["instrument"] == "DrumGroupDevice")
    check("project key found (D minor)", b["key"] == "D minor")
    check("held chords read as pad, a long single note as fx",
          md.note_role([[62, 0, 4, 80], [65, 0, 4, 80], [69, 0, 4, 80]], 4) == "pad"
          and md.note_role([[50, 0, 16, 100]], 16) == "fx")
    check("fast even single notes read as arp",
          md.note_role([[62 + (i % 4) * 3, i * 0.25, 0.2, 90] for i in range(16)], 4) == "arp")

    out = Path(d) / "set.jsonl"
    md.main(["--projects", d, "--out", str(out)])
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    check("dataset file written", len(lines) == 3 and all("notes" in x for x in lines))

print(f"\nAll {9 - len(FAILS)} of 9 checks passed." if not FAILS else f"\n{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
