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
os.environ["GHOST_LISTEN_DIR"] = tempfile.mkdtemp()

import fake_live  # noqa: E402
fake_live.RECORD_DIR = __import__("tempfile").mkdtemp()

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
        r = fn(*a, **kw)
    except ms.LiveError as e:
        check(name, contains.lower() in str(e).lower())
        return str(e)
    if isinstance(r, dict) and "error" in r:  # tools report errors as results
        check(name, contains.lower() in r["error"].lower())
        return r["error"]
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
        check("read notes back as compact rows", [n[0] for n in got] == [45, 45, 48, 43]
              and got[0] == [45, 0, 0.9, 110])
        ms.add_notes(bass, 0, [ms.Note(pitch=36, start=0, duration=4)], replace=True)
        check("replace notes", [n[0] for n in ms.get_notes(bass, 0)["notes"]] == [36])
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
              and song.view.selected_track == song.tracks[bass])
        ms.load_device(drums, "808 Core Kit", kind="drum_kit")
        check("load drum kit (extension ignored)", song.tracks[drums].devices[-1].name == "808 Core Kit")
        ms.load_device(vox, "EQ Eight", kind="audio_effect")
        check("load Live effect", "EQ Eight" in [d.name for d in song.tracks[vox].devices])
        expect_error("unknown device explained", ms.load_device, vox, "NotARealPlugin", contains="search_live_browser")
        hits = ms.search_live_browser("valhalla")
        check("browser search", hits and hits[0]["name"] == "ValhallaVintageVerb")
        # ---- sound design on a synth (simulated Serum) ----
        info = ms.get_device_parameters(bass, 0)
        lines = info["parameters"]
        check("read parameters as compact lines with display values",
              lines[0].startswith("0 Device On = ") and lines[1].startswith("1 Filter Cutoff = ")
              and "Hz [" in lines[1] and " raw " in lines[1])
        check("choices listed for switches", lines[5].endswith("{Sine|Triangle|Saw|Square|Noise}"))
        amp = ms.get_device_parameters(bass, 0, query="amp")["parameters"]
        check("parameter search", [l.split(" = ")[0].split(" ", 1)[1] for l in amp]
              == ["Amp Attack", "Amp Release"])
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

        # ---- listening: record the synth, change it, hear the difference ----
        ms.set_device_parameters(bass, 0, [S(parameter="Filter Cutoff", value="300 Hz")])
        song.tracks[vox].solo = True  # user had something soloed; must be restored
        song.tracks[vox].arm = True   # ...and something armed (Ghost's arming may clear it)
        tracks_before, scenes_before = len(song.tracks), len(song.scenes)
        ms.live.timeout = 25
        orig_wait = ms.time.sleep
        ms.time.sleep = lambda s: orig_wait(min(s, 0.05))  # don't wait in real time in tests
        try:
            out = ms.listen_to_track(bass, beats=8)
            import json as _j
            dark = _j.loads(out[0])
            check("listen returns a short report and no picture by default",
                  "heard" in dark and len(out) == 1 and "volume_movement_hz" not in dark["measurements"])
            check("hears the pitch", dark["measurements"]["pitch_note"] == "C1")
            out = ms.listen_to_track(bass, beats=8, picture=True, detail=True)
            full = _j.loads(out[0])
            check("picture and full detail on request", isinstance(out[1], ms.Image)
                  and full["measurements"]["volume_movement_hz"] is not None)
            check("session restored after listening", len(song.tracks) == tracks_before
                  and len(song.scenes) == scenes_before
                  and all(t.name != "Ghost Listen" for t in song.tracks))
            check("solo states restored", song.tracks[vox].solo and not song.tracks[bass].solo)
            check("armed tracks restored", song.tracks[vox].arm is True)
            song.tracks[vox].arm = False
            ms.set_device_parameters(bass, 0, [S(parameter="Filter Cutoff", value="3 kHz")])
            bright = _j.loads(ms.listen_to_track(bass, beats=8, compare_to="last")[0])
            check("hears that it got brighter", any("brighter" in d for d in bright["differences"]))
            check("brightness rose", bright["measurements"]["brightness_hz"] > dark["measurements"]["brightness_hz"] * 1.5)
            ref = dark["audio_file"]
            r = _j.loads(ms.analyze_audio_file(ref, compare_to=bright["audio_file"])[0])
            check("analyze a file against a reference", any("darker" in d for d in r["differences"]))
            expect_error("listening to an audio track needs a clip", ms.listen_to_track, vox,
                         notes=[ms.Note(pitch=48, start=0, duration=1)], contains="MIDI track")
            check("failed listen leaves no listen track", all(t.name != "Ghost Listen" for t in song.tracks))
            song.back_to_arranger = True  # an earlier listen took a track off the timeline
            _bpb0 = 4.0 * song.signature_numerator / song.signature_denominator
            mix = _j.loads(ms.listen_to_mix(bar=5, bars=2)[0])
            check("listen to the whole mix from a bar", "heard" in mix
                  and 3 * _bpb0 <= getattr(song, "_record_began", -1) < 4 * _bpb0)
            _bpb = 4.0 * song.signature_numerator / song.signature_denominator
            check("run-up bar trimmed off", abs(mix["measurements"]["seconds"]
                                               - 2 * _bpb * 60.0 / song.tempo) < 0.05)
            check("mix listen measures loudness", isinstance(mix["measurements"].get("lufs_integrated"), float)
                  and any("LUFS" in h for h in mix["heard"]))
            check("tracks follow the Arrangement again", song.back_to_arranger is False
                  and all(t.name != "Ghost Listen" for t in song.tracks))
            class _Group:  # like Live: group tracks raise instead of having clips
                name, is_foldable, devices, can_be_armed = "grp", True, [], False
                mixer_device = song.tracks[0].mixer_device
                @property
                def arrangement_clips(self):
                    raise RuntimeError("Main, Group and Return Tracks have no arrangement clips")
                @property
                def mute(self):
                    raise RuntimeError("no mute")
                clip_slots = []
            song.tracks.insert(0, _Group())
            try:
                ok = "tracks" in ms.get_arrangement_info() and \
                     ms.get_session_info()["tracks"][0]["type"] == "group"
            finally:
                song.tracks.pop(0)
            check("group tracks don't break arrangement/session info", ok)
            g = _Group(); song.tracks.insert(0, g)

            member = song.tracks[1]; member.is_grouped = True; member.group_track = song.tracks[0]  # a fresh wrapper, like Live
            try:
                info = ms.get_session_info()["tracks"]
                ok = info[0].get("group_members") == [1] and info[1].get("in_group") == 0
            finally:
                song.tracks.pop(0); member.is_grouped = False; member.group_track = None
            check("groups list their member tracks", ok)
            song.tracks[vox].arm = True  # the user has a track armed
            out = ms.listen_to_track(bass, bar=3, beats=4)
            check("listen to a track from the Arrangement", "heard" in _j.loads(out[0])
                  and song.recorded_solo == song.tracks[bass].name
                  and song.tracks[bass].solo is False and song.back_to_arranger is False)
            check("arrangement recording leaves the loop alone, restores record and arms",
                  song.loop is True and song.loop_start == 0.0 and song.loop_length == 720.0
                  and not song.punch_in and not song.punch_out and not song.record_mode
                  and song.tracks[vox].arm is True
                  and all(t.name != "Ghost Listen" for t in song.tracks))
            song.tracks[vox].arm = False
            m = ms.get_track_info(-1)
            check("master track addressed as -1", m["type"] == "master" and m["clips"] == [])
            check("master volume", abs(ms.set_track_volume(-1, 0.8)["volume"] - 0.8) < 1e-6)
            expect_error("master can't be deleted", ms.delete_track, -1, contains="Master")
        finally:
            ms.time.sleep = orig_wait
            song.tracks[vox].solo = False

        # ---- scenes & arrangement ----
        ms.set_time_signature(4, 4)
        ms.set_scene_name(0, "Main idea")
        check("scene name", song.scenes[0].name == "Main idea")
        ms.duplicate_clip(bass, 0, 9)
        check("duplicate clip (adds scenes)", song.tracks[bass].clip_slots[9].has_clip
              and song.tracks[bass].clip_slots[9].clip is not song.tracks[bass].clip_slots[0].clip)
        expect_error("duplicate needs a free slot", ms.duplicate_clip, bass, 0, 9, contains="overwrite")
        # drums: 1-bar clip; bass: 4-bar clip (16 beats) from earlier
        ms.create_clip(drums, 0, 4, overwrite=True)
        Sec, Ref = ms.Section, ms.ClipRef
        expect_error("section must fit the clips", ms.build_arrangement,
                     [Sec(name="Odd", bars=6, clips=[Ref(track_index=bass, clip_index=0)])],
                     contains="whole number")
        check("failed build placed nothing", not getattr(song.tracks[bass], "arrangement_clips", []))
        song.is_playing = True
        r = ms.build_arrangement([
            Sec(name="Intro", bars=8, clips=[Ref(track_index=drums)]),
            Sec(name="Drop", bars=16, clips=[Ref(track_index=drums), Ref(track_index=bass)]),
        ])
        check("arrangement built", [s_["section"] for s_ in r["sections"]] == ["Intro", "Drop"]
              and r["sections"][1]["start_bar"] == 9 and r["end_bar"] == 25)
        dclips = song.tracks[drums].arrangement_clips
        check("clips repeated to fill sections", len(dclips) == 24
              and dclips[0].start_time == 0 and dclips[-1].end_time == 96)
        check("bass only in the drop", [c.start_time for c in song.tracks[bass].arrangement_clips] == [32, 48, 64, 80])
        check("locators named", sorted((c.time, c.name) for c in song.cue_points) == [(0, "Intro"), (32, "Drop")])
        check("playback stopped while building", not song.is_playing)
        r2 = ms.build_arrangement([Sec(name="Outro", bars=4, clips=[Ref(track_index=drums)])])
        check("next build goes after existing work", r2["sections"][0]["start_bar"] == 25)
        info = ms.get_arrangement_info()
        check("arrangement info", info["length_bars"] == 28 and [l["name"] for l in info["locators"]] == ["Intro", "Drop", "Outro"])
        drum_clips = next(t for t in info["tracks"] if t["index"] == drums)["clips"]
        check("arrangement clips joined into bar ranges",
              all(isinstance(v, str) and "-" in v for v in drum_clips.values())
              and any(v.startswith("1-") for v in drum_clips.values()))
        r3 = ms.build_arrangement([Sec(name="New A", bars=4, clips=[Ref(track_index=drums)]),
                                   Sec(name="New B", bars=4, clips=[Ref(track_index=drums)])], start_bar=1)
        names = sorted((c.time, c.name) for c in song.cue_points)
        check("rebuild replaces stale locators in its range", names[:2] == [(0, "New A"), (16, "New B")]
              and all(n not in ("Intro",) for _, n in names) and (32, "Drop") in names)
        # fix up locators: wrong ones removed, right ones placed
        song.cue_points.append(fake_live.types.SimpleNamespace(time=60.0, name="9"))
        locs = ms.set_locators([ms.Locator(bar=1, name="Intro"), ms.Locator(bar=9, name="Drop")],
                               remove_existing=True)
        check("set_locators replaces wrong ones", locs[0]["at"] == "Intro@1, Drop@9")
        locs = ms.set_locators([ms.Locator(bar=9, name="Drop!")])
        check("set_locators renames in place", locs[0]["at"] in ("Drop!@9", "Intro@1, Drop!@9")
              and sorted((c.time, c.name) for c in song.cue_points) == [(0, "Intro"), (32, "Drop!")])

        # ---- arrangement editing: pads, variations, audio, automation ----
        song.tracks[drums].devices.append(fake_live.DrumRack())
        pads = ms.get_drum_pads(drums)["pads"]
        check("drum pads listed", {p_["name"]: p_["note"] for p_ in pads} ==
              {"Kick": 36, "Snare": 38, "Clap": 39, "Closed Hat": 42})
        expect_error("no drum rack explained", ms.get_drum_pads, vox, contains="no Drum Rack")
        ms.create_clip(bass, 2, 16, name="Bass var", overwrite=True)
        r = ms.place_clip(bass, 2, bar=13, repeats=1)
        bclips = song.tracks[bass].arrangement_clips
        check("variation replaces the clip under it", [(c.start_time, c.name) for c in bclips
              if c.start_time >= 32] == [(32, ""), (48, "Bass var"), (64, "")] or
              any(c.name == "Bass var" and c.start_time == 48 for c in bclips))
        check("no overlap after placing", all(a.end_time <= b.start_time for a, b in zip(bclips, bclips[1:])))
        import tempfile as _tf, soundfile as _sf, numpy as _np
        fx = os.path.join(_tf.mkdtemp(), "Riser 01.wav"); _sf.write(fx, _np.zeros((4410, 2)), 44100)
        r = ms.place_audio_clip(vox, fx, bar=5)
        check("audio file onto the timeline", r["method"] == "direct" and
              any(c.name == "Riser 01.wav" and c.start_time == 16 for c in song.tracks[vox].arrangement_clips))
        expect_error("audio only on audio tracks", ms.place_audio_clip, bass, fx, 1, contains="not an audio track")
        r = ms.place_audio_clip(vox, fx, bar=21, transpose=2, repeats=3)
        loops = sorted((c for c in song.tracks[vox].arrangement_clips if c.start_time >= 80),
                       key=lambda c: c.start_time)
        check("loop placed back to back, transposed", r["copies"] == 3 and r["end_bar"] == 24
              and [c.start_time for c in loops] == [80, 84, 88]
              and all(getattr(c, "pitch_coarse", 0) == 2 for c in loops))
        expect_error("transpose range checked", ms.place_audio_clip, vox, fx, 30, transpose=60,
                     contains="between -48 and 48")
        P = ms.AutoPoint
        r = ms.automate(bass, 0, "Filter Cutoff", [P(bar=9, value="200 Hz"), P(bar=13, value="10 kHz")])
        dev = song.tracks[bass].devices[0]
        cut = dev.parameters[1]
        envs = [c.envelopes.get(id(cut)) for c in song.tracks[bass].arrangement_clips
                if c.start_time < 48 and c.end_time > 32]
        check("automation placed on the timeline", r["clips_placed"] == 1 and all(envs))
        e = envs[0]
        lo, mid, hi = e.value_at(0.1), e.value_at(8.1), e.value_at(15.9)
        check("sweep rises from 200 Hz to 10 kHz", cut.str_for_value(lo).startswith(("19", "20", "21"))
              and lo < mid < hi and cut.str_for_value(hi).endswith("kHz"))
        check("parameter rests at the end value", str(cut).startswith(("9.9", "10.0")))
        slots_before = len(song.scenes)
        check("temporary copy cleaned up", len(song.scenes) == slots_before and
              sum(1 for sl in song.tracks[bass].clip_slots if sl.has_clip) == 3)
        res = dev.parameters[2]
        r2 = ms.automate(bass, 0, "Filter Resonance", [P(bar=9, value="10 %"), P(bar=13, value="60 %")])
        clip9 = [c for c in song.tracks[bass].arrangement_clips if abs(c.start_time - 32) < 1e-3][0]
        kept_env = clip9.envelopes.get(id(cut))
        check("second lane keeps the first lane's automation",
              r2.get("kept_automation") == ["Filter Cutoff"] and kept_env is not None
              and kept_env.value_at(0.1) < kept_env.value_at(15.9)
              and clip9.envelopes.get(id(res)) is not None)
        vol = dev.parameters[7]
        r3 = ms.automate(bass, 0, "Amp Attack", [P(bar=9, value="5 ms"), P(bar=13, value="50 ms")],
                         lanes=[ms.AutoLane(device_index=0, parameter="Master Volume",
                                            points=[P(bar=9, value=0.2), P(bar=13, value=0.8)])])
        clip9 = [c for c in song.tracks[bass].arrangement_clips if abs(c.start_time - 32) < 1e-3][0]
        ve = clip9.envelopes.get(id(vol))
        check("lanes draw several parameters in one pass",
              len(r3.get("lanes", [])) == 2 and ve is not None
              and ve.value_at(0.1) < ve.value_at(15.9)
              and clip9.envelopes.get(id(dev.parameters[3])) is not None
              and "Filter Cutoff" in r3["kept_automation"] and abs(vol.value - 0.8) < 1e-6)
        expect_error("bad unit explained", ms.automate, bass, 0, "Filter Cutoff",
                     [P(bar=9, value="3 dB"), P(bar=13, value="1 kHz")], contains="not db")

        # ---- clearing the arrangement ----
        n_before = len(song.tracks[drums].arrangement_clips)
        r = ms.clear_arrangement([drums], from_bar=9, to_bar=13)
        left = song.tracks[drums].arrangement_clips
        check("clear a range", r["removed"] == 4 and len(left) == n_before - 4
              and not any(c.start_time < 48 and c.end_time > 32 for c in left))
        r = ms.clear_arrangement([drums, bass])
        check("clear whole tracks, others untouched", not song.tracks[drums].arrangement_clips
              and not song.tracks[bass].arrangement_clips and song.tracks[vox].arrangement_clips)
        check("session clips kept", song.tracks[bass].clip_slots[0].has_clip)

        # ---- old plugins whose display lags (VST2): calibrated, never pushed to extremes ----
        song.tracks[vox].devices.append(fake_live.old_plugin())
        dv = len(song.tracks[vox].devices) - 1
        r = ms.set_device_parameters(vox, dv, [S(parameter="Decay", value="3 s"),
                                                S(parameter="Style", value="N"),
                                                S(parameter="Mix", value=0.3)])
        got = {x["parameter"]: x for x in r["set"]}
        check("lagging plugin: no errors", r["errors"] == [])
        check("lagging plugin: time reached by calibration", got["Decay"].get("calibrated")
              and abs(float(got["Decay"]["display"].split()[0]) - 3.0) < 0.1)
        check("lagging plugin: named setting found", got["Style"]["display"] == "N")
        check("lagging plugin: raw values still direct", got["Mix"]["value"] == 0.3)
        before = song.tracks[vox].devices[dv].parameters[1].value
        r = ms.set_device_parameters(vox, dv, [S(parameter="Decay", value="200 Hz")])
        check("lagging plugin: wrong unit refused and value restored",
              r["errors"] and "not hz" in r["errors"][0]["error"]
              and abs(song.tracks[vox].devices[dv].parameters[1].value - before) < 1e-5)
        # a control showing bare raw numbers must never be steered by them (it once went to max)
        tone = song.tracks[vox].devices[dv].parameters[4]
        r = ms.set_device_parameters(vox, dv, [S(parameter="Tone", value="113 Hz")])
        check("raw-number display refused, value restored",
              r["errors"] and "raw numbers" in r["errors"][0]["error"] and abs(tone.value - 0.25) < 1e-6)
        # ...but one that starts reporting once moved gets calibrated normally
        r = ms.set_device_parameters(vox, dv, [S(parameter="Cutoff", value="180 Hz")])
        got = r["set"][0] if r["set"] else {}
        check("waking plugin calibrated to target", not r["errors"] and got.get("calibrated")
              and abs(float(got["display"].split()[0]) - 180) <= 10)
        # long batches stop before the caller's time limit instead of being cut off midway
        budget, ms.CALIBRATION_BUDGET_S = ms.CALIBRATION_BUDGET_S, -1
        decay = song.tracks[vox].devices[dv].parameters[1]
        before = decay.value
        r = ms.set_device_parameters(vox, dv, [S(parameter="Decay", value="1 s")])
        ms.CALIBRATION_BUDGET_S = budget
        check("calibration time budget: reports 'send it again', leaves value alone",
              r["errors"] and "send it again" in r["errors"][0]["error"] and decay.value == before)

        # restructure a song: move sections, reuse one twice, drop the rest
        ra = ms.create_midi_track(name="Rearr")["track_index"]
        ms.create_clip(ra, 0, length_beats=4, name="bar")
        ms.place_clip(ra, 0, bar=1, repeats=8)
        M = ms.Move
        r = ms.rearrange_arrangement([M(from_bar=1, to_bar=3, new_bar=1),
                                      M(from_bar=5, to_bar=7, new_bar=3),
                                      M(from_bar=5, to_bar=7, new_bar=7)], track_indices=[ra])
        starts = sorted(c.start_time for c in song.tracks[ra].arrangement_clips)
        check("sections moved, one used twice, the rest dropped",
              starts == [0, 4, 8, 12, 24, 28] and r["clips_moved"] == 6 and r["new_end_bar"] == 9)
        check("overlapping moves refused, nothing changed",
              "error" in ms.rearrange_arrangement([M(from_bar=1, to_bar=5, new_bar=1),
                                                   M(from_bar=1, to_bar=3, new_bar=3)],
                                                  track_indices=[ra])
              and sorted(c.start_time for c in song.tracks[ra].arrangement_clips) == starts)
        ms.delete_track(ra)

        # your-style patterns: one of the user's F minor basslines written into D minor
        import json as _json, tempfile as _tf
        lib_file = Path(_tf.mkdtemp()) / "set.jsonl"
        lib_file.write_text(_json.dumps(dict(
            id="b1", project="C:/p/Night.als", track="MIDI", clip="", role="bass",
            instrument="Diva", tempo=125, key="F minor", length_beats=4,
            notes=[[41, 0, 0.5, 110], [44, 1.5, 0.5, 100], [48, 3, 0.5, 100]])))
        ms.STYLE_LIB = ms.style.Library(lib_file)
        found = ms.find_my_patterns("bass", key="D minor", tempo=128)
        r = ms.write_my_pattern(bass, 6, found[0]["id"], key="D minor", bars=2, variation=0)
        got = ms.get_notes(bass, 6)
        check("your-style pattern written in the song's key",
              r["transposed"] == -3 and got["clip_length_beats"] == 8
              and [n[0] for n in got["notes"]][:3] == [38, 41, 45] and len(got["notes"]) == 6)
        check("pattern errors come back as messages",
              "error" in ms.write_my_pattern(bass, 7, "missing-id"))

        # your-style model: a new part written by the trained model (a random one here)
        from test_midi_model import random_model
        ms.MODEL_PATH = Path(_tf.mkdtemp()) / "missing.npz"
        r = ms.generate_part(bass, 8, "bass")
        check("generate_part without a model explains how to train one",
              "train_midi_model.py" in r.get("error", ""))
        ms.MODEL_PATH = ms.MODEL_PATH.with_name("m.npz")
        random_model(ms.MODEL_PATH)
        r = ms.generate_part(bass, 8, "bass", key="D minor", bars=2, seed=5, takes=2, overwrite=True)
        got = [ms.get_notes(bass, 8)["notes"], ms.get_notes(bass, 9)["notes"]]
        check("generate_part writes takes in the song's key",
              len(r["takes"]) == 2 and r["takes"][1]["clip_index"] == 9
              and all(n[1] < 8 for g in got for n in g)
              and all((n[0] - 2) % 12 in ms.midi_model.SCALES["minor"] for g in got for n in g)
              and got[0] != got[1])
        check("generate_part refuses unknown roles", "error" in ms.generate_part(bass, 10, "drums"))
        r = ms.generate_part(bass, 10, "lead", style="collection", seed=2, overwrite=True)
        check("generate_part writes in the collection style", r.get("style") == "collection")

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
        r = ms.load_sample(vox, str(sample_file), clip_index=2, transpose=2)
        check("session audio clip transposed", r.get("transpose") == 2
              and song.tracks[vox].clip_slots[2].clip.pitch_coarse == 2)
        song.tracks[vox].clip_slots[2].delete_clip()
        sample_file.unlink()
        r = ms.load_sample(drums, r"D:\Samples\Drums\Kick_Hard_01.wav")
        check("sample into Simpler via browser", r["method"] == "browser" and "Simpler" in r["track_devices"])
        r = ms.load_sample(drums, r"E:\Samples\Samples\techno\Kicks\Deep_Kick_20.wav")
        check("sample deep in a huge library found by its folder path",
              r.get("method") == "browser" and r.get("from", "").endswith("Kicks/Deep_Kick_20.wav"))
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
