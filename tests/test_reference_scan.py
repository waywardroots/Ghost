"""Checks reference_scan.py on a synthetic club track with a known arrangement."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import reference_scan as rs  # noqa: E402

FAILS, N = [], [0]


def check(name, ok):
    N[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        FAILS.append(name)


check("Camelot and note keys read", rs.normalize_key("8A") == "A minor"
      and rs.normalize_key("Dm") == "D minor" and rs.normalize_key("Bbm") == "A# minor"
      and rs.normalize_key("F#") == "F# major" and rs.normalize_key("junk") is None)
check("Beatport file names give a title", rs.title_from_name("17654266_Karma_(Original Mix)")
      == (None, "Karma (Original Mix)"))
check("your own tracks recognised",
      rs.is_yours(Path("F:/Music/My Tracks To Play/x.mp3"), None)
      and rs.is_yours(Path("F:/Music/Techno/a.mp3"), "James And The Cat")
      and not rs.is_yours(Path("F:/Music/Techno/a.mp3"), "Someone"))

# A 125 BPM track: 16 bars intro (hats only), 32 bars drop (kick + bass + hats + pad),
# 16 bars breakdown (pad only), 32 bars drop. Sections should come back at bars 1, 17, 49, 65.
sr, bpm = 22050, 125.0
beat = 60 / bpm
rng = np.random.default_rng(0)


def bar(kick, bass, hats, pad):
    n = int(sr * beat * 4)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for b in range(4):
        s = int(b * beat * sr)
        seg = np.arange(int(0.25 * sr)) / sr
        if kick:
            out[s:s + seg.size] += 0.9 * np.sin(2 * np.pi * (50 + 80 * np.exp(-seg * 30)) * seg) * np.exp(-seg * 12)
        if hats:
            h = int((b + 0.5) * beat * sr)
            out[h:h + 800] += 0.12 * rng.standard_normal(800) * np.exp(-np.arange(800) / 150)
    if bass:
        out += 0.25 * np.sin(2 * np.pi * 73.4 * t)
    if pad:
        out += 0.08 * (np.sin(2 * np.pi * 293.7 * t) + np.sin(2 * np.pi * 349.2 * t) + np.sin(2 * np.pi * 440 * t))
    return out


song = np.concatenate([bar(0, 0, 1, 0) for _ in range(16)] + [bar(1, 1, 1, 1) for _ in range(32)]
                      + [bar(0, 0, 0, 1) for _ in range(16)] + [bar(1, 1, 1, 1) for _ in range(32)])
stereo = np.stack([song, song * 0.95]).T

with tempfile.TemporaryDirectory() as d:
    folder = Path(d) / "Melodic Techno"
    folder.mkdir()
    f = folder / "Test Artist - Night Run.wav"
    sf.write(f, stereo, sr)
    res = rs.analyze_track((str(f), 125.0, "D minor"))
    check("analysis ran", not res.get("error"))
    secs = res.get("sections") or []
    starts = [(s["kind"], s["start_bar"]) for s in secs]
    print("   sections:", starts, "intro", res.get("intro_bars"))
    check("intro length found (16 bars)", abs((res.get("intro_bars") or 0) - 16) <= 1)
    brk = [s for s in secs if s["kind"] == "break" and s["start_bar"] > 20]
    check("breakdown found at bar 49, 16 bars long",
          len(brk) == 1 and abs(brk[0]["start_bar"] - 49) <= 1 and abs(brk[0]["bars"] - 16) <= 2)
    check("tag tempo and key used", res["bpm"] == 125.0 and res["musical_key"] == "D minor"
          and res["bpm_source"] == "tag")
    check("mix measured on the drop", res["bands"]["bass"] > 20 and 0 < res["stereo_width"] < 0.1)

    db = Path(d) / "lib.db"
    rs.main(["--music", d, "--db", str(db), "--workers", "1"])
    con = rs.open_db(str(db))
    row = con.execute("SELECT * FROM reference_tracks").fetchone()
    check("saved with genre from its folder and artist from the name",
          row["genre"] == "Melodic Techno" and row["artist"] == "Test Artist"
          and row["title"] == "Night Run" and row["yours"] == 0)
    p = rs.profile(con, genre="melodic")
    check("profile summarises arrangement and mix",
          p["tracks"] == 1 and p["arrangement"]["first_breakdown_bar"] in (48, 49, 50)
          and p["mix"]["bands_percent"]["bass"] > 20)
    import os
    os.environ["GHOST_DB"] = str(db)
    import mcp_server as ms
    found = ms.find_reference_tracks(genre="melodic", key="Dm", tempo=126)
    check("Ghost tool finds the track with its arrangement",
          len(found) == 1 and found[0]["title"] == "Night Run"
          and any(s["kind"] == "break" for s in found[0]["sections"]))
    check("Ghost tool gives the profile", ms.get_reference_profile(genre="melodic")["tracks"] == 1)
    before = row["indexed_at"]
    rs.main(["--music", d, "--db", str(db), "--workers", "1"])
    check("unchanged files are skipped next time",
          con.execute("SELECT indexed_at FROM reference_tracks").fetchone()[0] == before)

print(f"\nAll {N[0]} checks passed." if not FAILS else f"\n{len(FAILS)} of {N[0]} FAILED")
sys.exit(1 if FAILS else 0)
