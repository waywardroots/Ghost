"""Checks arrangement_maps.py: bar-by-bar element maps from stems, and typical layouts."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arrangement_maps as am  # noqa: E402
import test_music_dataset as tmd  # noqa: E402

PASSED = 0


def check(name, cond):
    global PASSED
    if not cond:
        raise AssertionError("FAILED: " + name)
    PASSED += 1
    print("PASS ", name)


def fake_row(name, on):
    """on: element -> list of (first_bar, last_bar) where it plays, 1-based inclusive."""
    bars = 128
    lv = {}
    for el in am.ELEMENTS:
        v = np.full(bars, -90)
        for a, b in on.get(el, []):
            v[a - 1:b] = -20
        lv[el] = v.tolist()
    return {"path": f"F:/Music/Melodic Techno Playlists/x/{name}.mp3", "bpm": 125.0,
            "bars": bars, "levels": lv}


def main():
    stems, _ = tmd.synth(bars=32)               # lead plays in bars 9-16 and 25-32
    stems["vocals"] = np.zeros_like(stems["bass"])
    row = am.track_map(stems, "x.wav")
    check("map covers the track bar by bar", row and 28 <= row["bars"] <= 33 and abs(row["bpm"] - 128) < 0.5)
    act = am.activity(row)
    check("kick and bass play throughout", act["kick"][:28].mean() > 0.9 and act["bass"][:28].mean() > 0.9)
    lead = am.ranges(act["lead"])
    check("lead found in the sections where it plays", lead.startswith("9-16") and "25-" in lead)
    check("silent vocals are absent", not act["vocals"].any())

    a = {"kick": [(1, 64), (81, 128)], "bass": [(17, 64), (81, 128)], "hats": [(1, 128)],
         "lead": [(33, 64), (65, 80), (97, 128)]}
    b = {"kick": [(1, 128)], "bass": [(1, 128)], "hats": [(1, 128)], "lead": [(1, 128)]}
    rows = [fake_row(f"A{i}", a) for i in range(4)] + [fake_row(f"B{i}", b) for i in range(3)]
    rows[1]["levels"]["lead"][40] = -90         # small differences inside a group
    bp = am.blueprints(rows, k=2)
    check("two layouts found, biggest first", [x["tracks_like_it"] for x in bp] == [4, 3]
          and bp[0]["track"].startswith("A") and bp[1]["track"].startswith("B"))
    check("blueprint gives bar ranges per element",
          bp[0]["elements"]["kick"] == "1-64, 81-128" and bp[0]["elements"]["bass"] == "17-64, 81-128")
    s = am.summary(rows)
    check("folder summary: when elements enter and kick breaks",
          s["element_enters_at_bar"]["kick"] == 1 and s["kick_break_bars"] == 16
          and s["share_with_element"]["vocals"] == 0.0)
    print(f"\nAll {PASSED} checks passed.")


if __name__ == "__main__":
    main()
