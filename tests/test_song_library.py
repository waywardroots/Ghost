"""Checks song_library.py: bar-by-bar element maps from stems, and typical layouts."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import song_library as am  # noqa: E402
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
    k = row["drums"]["kick"][4]
    check("kick pattern read on the 16th grid", [i for i, c in enumerate(k) if c != "."] == [0, 4, 8, 12])
    h = row["drums"]["hats"][4]
    check("off-beat hats read", all(h[i] != "." for i in (2, 6, 10, 14)) and h[0] == ".")
    bar5 = [n for n in row["notes"]["bass"] if 64 <= n[1] < 80]
    check("bassline read bar by bar from the first downbeat",
          [(n[0], n[1] - 64) for n in bar5] == [(38, 2), (38, 6), (41, 10), (38, 14)])
    check("key found", row["key"] == "D minor")
    check("sound profile per section", row["sound"]["other"][0]["bright_hz"] > 300
          and "movement" in row["sound"]["bass"][0])
    check("echo read-out present", "space" in row["fx"]["other"])

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
    # Ghost's tools over a library file
    import json, tempfile
    import mcp_server as ms
    real = dict(row, path="F:/Music/Melodic Techno Playlists/Dec/Artist - Real Song.mp3")
    with tempfile.TemporaryDirectory() as d:
        lib = Path(d) / "song_library.jsonl"
        lib.write_text("\n".join(json.dumps(r) for r in rows + [real]))
        ms.LIBRARY_PATH = lib
        found = ms.song_library(query="real song")
        check("library search", found["songs_in_library"] == 8 and found["songs"][0]["track"] == "Artist - Real Song")
        bp = ms.song_blueprint("real song")
        check("blueprint: arrangement ranges", bp["elements"]["lead"].startswith("9-16"))
        dr = ms.song_blueprint("real song", part="drums", element="kick")
        check("blueprint: drum bars grouped", any("X...X...X...X..." in g for g in dr["drums"]["kick"]))
        nt = ms.song_blueprint("real song", part="notes", element="bass", from_bar=5, bars=1, to_key="E minor")
        check("blueprint: notes in beats, transposed", nt["transposed"] == 2
              and [n[:2] for n in nt["notes"]] == [[40, 0.5], [40, 1.5], [43, 2.5], [40, 3.5]])
        lay = ms.typical_layouts("melodic techno", k=2)
        check("typical layouts are real songs", len(lay["layouts"]) == 2 and lay["summary"]["tracks"] == 8)
        ms.LIBRARY_PATH = Path(d) / "missing.jsonl"
        check("missing library explained", "song_library.py" in ms.song_library().get("error", ""))
    print(f"\nAll {PASSED} checks passed.")


if __name__ == "__main__":
    main()
