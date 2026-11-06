"""Checks the your-style pattern library (style.py)."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import style  # noqa: E402

FAILS, N = [], [0]


def check(name, ok):
    N[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        FAILS.append(name)


check("keys read in common spellings",
      style.parse_key("D minor") == (2, "minor") and style.parse_key("Dm") == (2, "minor")
      and style.parse_key("F# major") == (6, "major") and style.parse_key("Bb") == (10, "major")
      and style.parse_key("Ebmin") == (3, "minor"))
check("smallest move between keys", style.shift_between((5, "minor"), (2, "minor")) == -3
      and style.shift_between((0, "minor"), (2, "minor")) == 2
      and style.shift_between((11, "minor"), (5, "minor")) == -6)
check("minor pattern into a major song uses the relative key",
      style.shift_between((9, "minor"), (0, "major")) == 0)

clips = [
    dict(id="bassF", project="C:/p/Night Drive.als", track="MIDI", clip="", role="bass",
         instrument="SubBoomBass2", tempo=125, key="F minor", length_beats=4,
         notes=[[41, 0, 0.5, 110], [41, 0.75, 0.25, 90], [44, 1.5, 0.5, 100], [48, 3, 0.5, 100]]),
    dict(id="bassD", project="C:/p/Deep.als", track="Bass", clip="", role="bass",
         instrument="Diva", tempo=140, key="D minor", length_beats=8,
         notes=[[38, 0, 0.5, 110], [38, 4, 0.5, 100]]),
    dict(id="kit", project="C:/p/Deep.als", track="909", clip="", role="drums",
         instrument="DrumGroupDevice", tempo=126, key="D minor", length_beats=4,
         notes=[[36, b, 0.25, 127] for b in range(4)]),
]
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "set.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in clips))
    lib = style.Library(p)
    found = lib.find("bass", key="D minor", tempo=126, bars=1)
    check("closest fit ranked first (tempo/length outweigh exact key)",
          [f["id"] for f in found] == ["bassF", "bassD"] and found[0]["transpose_to_fit"] == -3)
    check("role filter", [f["id"] for f in lib.find("drums")] == ["kit"])
    check("instrument filter", [f["id"] for f in lib.find("bass", instrument="diva")] == ["bassD"])

    notes, info = style.make_part(lib.get("bassF"), key="D minor", bars=4, variation=0)
    check("exact copy moved into D minor and tiled to 4 bars",
          info["transposed"] == -3 and info["bars"] == 4 and len(notes) == 16
          and [n["pitch"] for n in notes[:4]] == [38, 38, 41, 45] and notes[-1]["start"] == 15)
    n2, _ = style.make_part(lib.get("bassF"), key="D minor", bars=4, variation=0.6, seed=7)
    n3, _ = style.make_part(lib.get("bassF"), key="D minor", bars=4, variation=0.6, seed=7)
    n4, _ = style.make_part(lib.get("bassF"), key="D minor", bars=4, variation=0.6, seed=8)
    check("variation changes the part but the same seed repeats it",
          n2 != notes and n2 == n3 and n4 != n2)
    pcs = {n["pitch"] % 12 for n in n2}
    check("varied notes stay in D minor", pcs <= {2, 4, 5, 7, 9, 10, 0})
    check("every note inside the clip", all(0 <= n["start"] < 16 for n in n2))
    dn, dinfo = style.make_part(lib.get("kit"), key="F minor", bars=2, variation=0)
    check("drums are never transposed", "transposed" not in dinfo
          and {n["pitch"] for n in dn} == {36} and len(dn) == 8)
    cut, cinfo = style.make_part(lib.get("bassD"), bars=1, variation=0)
    check("longer pattern cropped to the requested bars",
          cinfo["bars"] == 1 and [n["start"] for n in cut] == [0])
    try:
        lib.get("nope")
        check("unknown id reported", False)
    except style.StyleError:
        check("unknown id reported", True)

missing = style.Library(Path("/nonexistent/x.jsonl"))
try:
    missing.clips()
    check("missing library explains how to build it", False)
except style.StyleError as e:
    check("missing library explains how to build it", "midi_dataset.py" in str(e))

print(f"\nAll {N[0]} checks passed." if not FAILS else f"\n{len(FAILS)} of {N[0]} FAILED")
sys.exit(1 if FAILS else 0)
