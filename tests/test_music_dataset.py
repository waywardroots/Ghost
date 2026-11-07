"""Checks music_dataset.py on a synthetic 128 BPM track in D minor whose notes are known:
tempo, bar starts, key, and the bass and lead lines read back from the stems."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import music_dataset as md  # noqa: E402
import midi_model as mm  # noqa: E402

PASSED = 0
SR = md.SR


def check(name, cond):
    global PASSED
    if not cond:
        raise AssertionError("FAILED: " + name)
    PASSED += 1
    print("PASS ", name)


def saw(f, n, sr=SR):
    t = np.arange(n) / sr
    return 2 * ((t * f) % 1.0) - 1


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


BASS_A, BASS_B = [38, 38, 41, 38], [34, 34, 38, 34]      # D D F D / Bb Bb D Bb (8th offbeats)
LEAD = [74, 77, 81, 77, 74, 72, 74, 69]                  # 8th notes, one bar


def synth(bpm=128.0, bars=32, start=0.31, pickup_beats=1):
    beat = 60 / bpm
    n = int((start + (bars * 4 + pickup_beats + 2) * beat) * SR)
    drums, bass, other = np.zeros(n), np.zeros(n), np.zeros(n)
    rng = np.random.default_rng(0)
    kick_len = int(0.15 * SR)
    tk = np.arange(kick_len) / SR
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tk * 40)) * tk) * np.exp(-tk * 20)
    total_beats = bars * 4 + pickup_beats
    for b in range(total_beats):
        i = int((start + b * beat) * SR)
        drums[i:i + kick_len] += kick[:n - i]
        j = int((start + (b + 0.5) * beat) * SR)
        drums[j:j + 2000] += 0.2 * rng.normal(size=min(2000, n - j)) * np.exp(-np.arange(min(2000, n - j)) / 300)
    notes = {"bass": [], "lead": []}
    for bar in range(bars):
        pat = BASS_A if (bar // 8) % 2 == 0 else BASS_B
        b0 = pickup_beats + bar * 4
        for k, p in enumerate(pat):
            t = start + (b0 + k + 0.5) * beat
            i, ln = int(t * SR), int(0.4 * beat * SR)
            env = np.minimum(1, np.arange(ln) / 80) * np.minimum(1, (ln - np.arange(ln)) / 200)
            bass[i:i + ln] += 0.5 * saw(hz(p), ln) * env
            notes["bass"].append((bar, (k * 4 + 2), p))
        if (bar // 8) % 2 == 1:                     # the lead plays in every other section
            for k, p in enumerate(LEAD):
                t = start + (b0 + k * 0.5) * beat
                i, ln = int(t * SR), int(0.45 * beat * SR)
                env = np.minimum(1, np.arange(ln) / 60) * np.minimum(1, (ln - np.arange(ln)) / 150)
                other[i:i + ln] += 0.3 * np.sign(np.sin(2 * np.pi * hz(p) * np.arange(ln) / SR)) * env
                notes["lead"].append((bar, k * 2, p))
    return {"drums": drums, "bass": bass, "other": other}, notes


def main():
    stems, truth = synth()
    beats, bpm = md.beat_grid(stems["drums"])
    check("tempo found (128 BPM)", abs(bpm - 128) < 0.3)
    check("beat grid lines up with the kicks", abs(((beats[0] - 0.31) / (60 / 128) + 0.5) % 1 - 0.5) < 0.05)
    beats_tag, bpm_tag = md.beat_grid(stems["drums"], tag_bpm=128.0)
    check("a BPM tag locks the tempo exactly", bpm_tag == 128.0)

    phase = md.downbeat_phase(beats, stems)
    first_beat = int(round((beats[0] - 0.31) / (60 / 128)))   # which synth beat grid beat 0 is
    check("bar starts found", (first_beat + phase) % 4 == 1)
    check("key found (D minor)", md.detect_key(stems) == "D minor")

    rows = md.analyze_stems(stems, "C:/music/test.mp3", tag_key="D minor")
    bass = [r for r in rows if r["role"] == "bass"]
    lead = [r for r in rows if r["role"] in ("lead", "arp")]
    check("bass phrases found", len(bass) >= 2)
    # every bass phrase should be the pattern: notes on the 8th offbeats of each bar
    want = {(s, p) for bar in range(8) for s, p in
            [((bar * 4 + k) + 0.5, q) for k, q in enumerate(BASS_A)]}
    got = {(n[1], n[0]) for n in bass[0]["notes"]}
    check("bassline read back on the right beats and notes", len(got & want) / len(want) >= 0.9
          and len(got - want) <= 3)
    check("bass note lengths are close", all(abs(n[2] - 0.5) <= 0.25 for n in bass[0]["notes"]))
    check("lead line found in the sections where it plays", len(lead) >= 1)
    lw = {(bar * 4 + k * 0.5, p) for bar in range(8) for k, p in enumerate(LEAD)}
    lg = {(n[1], n[0]) for n in lead[0]["notes"]}
    check("lead line read back", len(lg & lw) / len(lw) >= 0.85)
    check("rows are in the training format", all(r["source"] == "music" and r["length_beats"] == 32
                                                  and r["key"] == "D minor" for r in rows))
    rows2 = md.analyze_stems(stems, "C:/music/test.mp3", tag_key="F major")
    check("a relative-major tag is corrected from the bassline (F major -> D minor)",
          rows2 and all(r["key"] == "D minor" for r in rows2))
    check("a right tag is kept", md.settle_mode("D minor", [[38, 0, 4, 100], [41, 4, 1, 100]]) == "D minor"
          and md.settle_mode("C major", [[36, 0, 4, 100], [33, 4, 1, 100]]) == "C major")
    wins = mm.clip_windows(bass[0])
    check("phrases tokenise for the model", wins and wins[0][1] == 8)
    print(f"\nAll {PASSED} checks passed.")


if __name__ == "__main__":
    main()
