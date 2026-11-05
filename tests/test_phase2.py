"""
End-to-end test of Phase 2 without Ableton:
assistant tools (mcp_server) -> TCP socket -> Ghost Remote Script -> simulated Live.

Run:  python tests/test_phase2.py
"""
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ableton"), str(ROOT / "tests")]
os.environ.setdefault("GHOST_LIVE_PORT", "9877")
import tempfile
os.environ["GHOST_SNAPSHOTS"] = os.path.join(tempfile.mkdtemp(), "snaps.json")

import fake_live  # noqa: E402

fake_live.install_fake_modules()
import Ghost  # noqa: E402  (the Remote Script package)
import mcp_server as ms  # noqa: E402

PASSED = []


def check(name, cond):
    if not cond:
        raise AssertionError("FAILED: " + name)
    PASSED.append(name)


def expect_error(name, fn, *a, contains="", **kw):
    try:
        fn(*a, **kw)
    except ms.LiveError as e:
        check(name, contains.lower() in str(e).lower())
        return str(e)
    raise AssertionError("FAILED (no error): " + name)


def main():
    # Tools fail politely while Live isn't running
    expect_error("helpful message when Live is closed", ms.get_session_info, contains="Control Surface")

    # Start "Live" with Ghost loaded; a thread plays the role of Live's main loop
    c = fake_live.CInstance(scenes=4)
    surface = Ghost.create_instance(c)
    stop = threading.Event()

    def live_main_loop():
        while not stop.is_set():
            surface.update_display()
            time.sleep(0.02)

    threading.Thread(target=live_main_loop, daemon=True).start()
    time.sleep(0.3)
    song, browser = c.song, c.app.browser

    try:
        check("ping", ms.live.send("ping")["ghost"])
        check("set tempo", ms.set_tempo(128)["tempo"] == 128 and song.tempo == 128)
        expect_error("tempo range check", ms.set_tempo, 5, contains="between")
        ms.set_time_signature(6, 8)
        check("time signature", (song.signature_numerator, song.signature_denominator) == (6, 8))

        drums = ms.create_midi_track(name="Drums")["track_index"]
        bass = ms.create_midi_track(name="Bass")["track_index"]
        vox = ms.create_audio_track(name="Vox")["track_index"]
        check("tracks created with names", [t.name for t in song.tracks] == ["Drums", "Bass", "Vox"])
        check("insert at index", ms.create_midi_track(name="Keys", index=1)["track_index"] == 1
              and song.tracks[1].name == "Keys")
        ms.delete_track(1)
        check("delete track", [t.name for t in song.tracks] == ["Drums", "Bass", "Vox"])

        info = ms.get_session_info()
        check("session info", info["tempo"] == 128 and len(info["tracks"]) == 3
              and info["tracks"][2]["type"] == "audio" and info["return_tracks"] == ["A-Reverb", "B-Delay"])

        # Clips and notes
        check("create clip", ms.create_clip(bass, 0, 16, name="Bassline")["name"] == "Bassline")
        expect_error("no silent overwrite", ms.create_clip, bass, 0, 4, contains="overwrite")
        expect_error("clips only on MIDI tracks", ms.create_clip, vox, 0, contains="not a MIDI")
        notes = [ms.Note(pitch=p, start=i * 1.0, duration=0.9, velocity=110)
                 for i, p in enumerate([45, 45, 48, 43])]
        check("add notes", ms.add_notes(bass, 0, notes)["added"] == 4)
        got = ms.get_notes(bass, 0)["notes"]
        check("read notes back", [n["pitch"] for n in got] == [45, 45, 48, 43]
              and got[0]["velocity"] == 110)
        ms.add_notes(bass, 0, [ms.Note(pitch=36, start=0, duration=4)], replace=True)
        check("replace notes", [n["pitch"] for n in ms.get_notes(bass, 0)["notes"]] == [36])
        expect_error("bad pitch rejected", ms.live.send, "add_notes", track_index=bass, clip_index=0,
                     notes=[{"pitch": 200, "start": 0, "duration": 1}], contains="0-127")
        ms.create_clip(drums, 6, 4)
        check("new scenes created on demand", len(song.scenes) == 7 and song.tracks[0].clip_slots[6].has_clip)
        ms.set_clip_loop(bass, 0, 0, 8)
        check("clip loop", song.tracks[bass].clip_slots[0].clip.loop_end == 8)
        ms.fire_clip(bass, 0)
        ms.fire_scene(0)
        ms.start_playback()
        check("transport + launching", song.is_playing and song.scenes[0].fired == 1
              and song.tracks[bass].clip_slots[0].fired == 1)
        ms.stop_playback()
        expect_error("bad track index explained", ms.get_track_info, 42, contains="3 tracks")

        # Devices and plugins through Live's browser
        r = ms.load_device(bass, "serum", kind="plugin")
        check("load VST by name", r["loaded"] == "Serum" and "Serum" in r["track_devices"]
              and song.view.selected_track is song.tracks[bass])
        ms.load_device(drums, "808 Core Kit", kind="drum_kit")
        check("load drum kit (extension ignored)", song.tracks[drums].devices[-1].name == "808 Core Kit")
        ms.load_device(vox, "EQ Eight", kind="audio_effect")
        check("load Live effect", "EQ Eight" in [d.name for d in song.tracks[vox].devices])
        expect_error("unknown device explained", ms.load_device, vox, "NotARealPlugin", contains="search_live_browser")
        hits = ms.search_live_browser("valhalla")
        check("browser search", hits and hits[0]["name"] == "ValhallaVintageVerb")
        # ---- sound design on a synth (simulated Serum) ----
        info = ms.get_device_parameters(bass, 0)
        names = [p["name"] for p in info["parameters"]]
        check("read parameters with display values", names[:2] == ["Device On", "Filter Cutoff"]
              and info["parameters"][1]["display"].endswith("Hz"))
        check("choices listed for switches",
              info["parameters"][5]["choices"] == ["Sine", "Triangle", "Saw", "Square", "Noise"])
        check("parameter search", [p["name"] for p in
              ms.get_device_parameters(bass, 0, query="amp")["parameters"]] == ["Amp Attack", "Amp Release"])
        S = ms.ParamSetting
        r = ms.set_device_parameters(bass, 0, [
            S(parameter="Filter Cutoff", value="800 Hz"),
            S(parameter="resonance", value="35 %"),
            S(parameter="Amp Attack", value="40 ms"),
            S(parameter="Amp Release", value="1.5 s"),
            S(parameter="Osc A Wave", value="saw"),
            S(parameter="Master Volume", value="-6 dB"),
            S(parameter="Darkness", value="80 %"),
            S(parameter="Unison", value="7"),
        ])
        shown = {x["parameter"]: x["display"] for x in r["set"]}
        check("no errors in a full sound design pass", r["errors"] == [])
        check("set cutoff in Hz", shown["Filter Cutoff"] in ("799 Hz", "800 Hz", "801 Hz"))
        check("set percent", shown["Filter Resonance"] == "35 %")
        check("set attack in ms", abs(float(shown["Amp Attack"].split()[0]) - 40) < 0.5)
        check("set release in seconds", shown["Amp Release"] == "1.50 s")
        check("pick waveform by name", shown["Osc A Wave"] == "Saw")
        check("set volume in dB", shown["Master Volume"] in ("-6.0 dB", "-5.9 dB", "-6.1 dB"))
        check("reverse-running parameter", shown["Darkness"] == "80 %")
        check("stepped parameter", shown["Unison"] == "7")
        r = ms.set_device_parameters(bass, 0, [S(parameter="Filter Cutoff", value="5 kHz"),
                                                S(parameter="Amp Attack", value="300 Hz"),
                                                S(parameter="Osc", value="Saw"),
                                                S(parameter="Osc B Wave", value="Wobble"),
                                                S(parameter="Nope", value=1)])
        errs = " | ".join(e["error"] for e in r["errors"])
        check("kHz works", r["set"][0]["display"] in ("4.99 kHz", "5.00 kHz", "5.01 kHz"))
        check("bad settings reported, good ones still applied", len(r["set"]) == 1 and len(r["errors"]) == 4)
        check("unit mismatch explained", "shown in s, not hz" in errs)
        check("ambiguous name explained", "several parameters" in errs)
        check("unknown option lists the options", "Options: Sine, Triangle, Saw" in errs)
        check("bad set didn't change the value",
              song.tracks[bass].devices[0].parameters[3].str_for_value(
                  song.tracks[bass].devices[0].parameters[3].value).startswith("4"))
        check("raw values still work", ms.set_device_parameter(bass, 0, "Filter Resonance", 5.0)["value"] == 1.0)
        check("set by index", ms.set_device_parameter(bass, 0, "4", "200 ms")["parameter"] == "Amp Release")

        # snapshots
        ms.save_sound_snapshot(bass, 0, "test A")
        ms.set_device_parameters(bass, 0, [S(parameter="Filter Cutoff", value="100 Hz"),
                                            S(parameter="Osc A Wave", value="Square")])
        ms.save_sound_snapshot(bass, 0, "test B")
        r = ms.recall_sound_snapshot(bass, 0, "test A")
        dev = song.tracks[bass].devices[0]
        check("recall snapshot", r["errors"] == [] and str(dev.parameters[1]).split()[0] in ("4.99", "5.00", "5.01"))
        check("recall restores switches", str(dev.parameters[5]) == "Saw")
        check("snapshots listed", {"test A", "test B"} <= {s["name"] for s in ms.list_sound_snapshots()})
        expect_error("missing snapshot explained", ms.recall_sound_snapshot, bass, 0, "nope", contains="Saved:")

        # a plugin whose controls aren't exposed yet
        ms.load_device(vox, "Pro-Q 3", kind="plugin")
        info = ms.get_device_parameters(vox, len(song.tracks[vox].devices) - 1)
        check("Configure hint for unexposed plugin", "Configure" in info.get("hint", ""))
        ms.set_track_volume(vox, 0.7)
        check("track volume", song.tracks[vox].mixer_device.volume.value == 0.7)

        # Samples
        sample_file = ROOT / "tests" / "_tmp_vox.wav"
        sample_file.write_bytes(b"RIFF")
        r = ms.load_sample(vox, str(sample_file), clip_index=1)
        check("sample into audio clip (Live 12.0.5+ path)", r["method"] == "audio clip"
              and song.tracks[vox].clip_slots[1].clip.name == "_tmp_vox.wav")
        sample_file.unlink()
        r = ms.load_sample(drums, r"D:\Samples\Drums\Kick_Hard_01.wav")
        check("sample into Simpler via browser", r["method"] == "browser" and "Simpler" in r["track_devices"])
        expect_error("sample outside browser explained", ms.load_sample, drums,
                     r"C:\nowhere\mystery.wav", contains="Places")

        ms.undo()
        check("undo", song.undo_count == 1)
        check("unknown command", "unknown command" in expect_error(
            "unknown command reported", ms.live.send, "explode", contains="unknown"))

        # Concurrency: two assistants at once
        results = []
        ts = [threading.Thread(target=lambda: results.append(ms.LiveConnection("127.0.0.1", 9877)
                                                             .send("get_session_info")["tempo"]))
              for _ in range(5)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        check("parallel clients", results == [128.0] * 5)
    finally:
        stop.set()
        surface.disconnect()

    print("\n".join("PASS  " + p for p in PASSED))
    print(f"\nAll {len(PASSED)} checks passed.")


if __name__ == "__main__":
    main()
