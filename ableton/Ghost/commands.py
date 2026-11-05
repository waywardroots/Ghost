"""
Ghost Remote Script - command handlers.

Every method named cmd_<name> is a command the assistant can send.
These run on Ableton Live's main thread (see ghost.py), so they may
touch the Live Object Model directly.
"""
from __future__ import absolute_import, print_function, unicode_literals

import os

from . import params as params_mod

VERSION = "0.3.0"
PLUGIN_CLASSES = ("PluginDevice", "AuPluginDevice")

# Browser categories searched for "all", in this order.
BROWSER_CATEGORIES = (
    "plugins", "instruments", "audio_effects", "midi_effects", "drums", "sounds",
    "max_for_live", "user_library", "user_folders", "samples", "clips", "packs",
    "current_project",
)
DEVICE_KINDS = {
    "plugin": ("plugins",),
    "instrument": ("instruments", "plugins", "sounds"),
    "audio_effect": ("audio_effects", "plugins"),
    "midi_effect": ("midi_effects",),
    "drum_kit": ("drums",),
    "preset": ("sounds", "user_library", "instruments", "audio_effects"),
    "any": BROWSER_CATEGORIES,
}
SAMPLE_CATEGORIES = ("user_folders", "samples", "user_library", "current_project", "packs")
MAX_BROWSER_NODES = 40000
MAX_BROWSER_DEPTH = 10


class CommandError(Exception):
    """An error the assistant should see as a plain message."""


def _base_name(name):
    n = name.lower().strip()
    for ext in (".adg", ".adv", ".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".vst3",
                ".dll", ".clap", ".amxd", ".alc"):
        if n.endswith(ext):
            return n[: -len(ext)]
    return n


def _match_score(name, query):
    q = query.lower().strip()
    n = name.lower().strip()
    b = _base_name(name)
    if n == q or b == q:
        return 3
    if b.startswith(q):
        return 2
    if q in n:
        return 1
    return 0


class CommandHandler(object):
    def __init__(self, song, application, log=None):
        self.song = song
        self.app = application
        self.log = log or (lambda msg: None)
        self._commands = dict((n[4:], getattr(self, n)) for n in dir(self) if n.startswith("cmd_"))

    def handle(self, command, params):
        fn = self._commands.get(command)
        if fn is None:
            raise CommandError("unknown command '%s'" % (command,))
        return fn(**(params or {}))

    # ------------------------------------------------------------------ helpers
    def _tracks(self):
        return list(self.song.tracks)

    def _track(self, track_index):
        tracks = self._tracks()
        if not isinstance(track_index, int) or not 0 <= track_index < len(tracks):
            raise CommandError("track_index %r is out of range (this set has %d tracks, "
                               "numbered from 0)" % (track_index, len(tracks)))
        return tracks[track_index]

    def _slot(self, track, clip_index, create_scenes=False):
        if not isinstance(clip_index, int) or clip_index < 0:
            raise CommandError("clip_index must be 0 or higher")
        if create_scenes:
            while len(list(self.song.scenes)) <= clip_index:
                self.song.create_scene(-1)
        slots = list(track.clip_slots)
        if clip_index >= len(slots):
            raise CommandError("clip_index %d is out of range (%d scenes)" % (clip_index, len(slots)))
        return slots[clip_index]

    def _clip(self, track_index, clip_index):
        slot = self._slot(self._track(track_index), clip_index)
        if not slot.has_clip:
            raise CommandError("no clip at track %d, slot %d" % (track_index, clip_index))
        return slot.clip

    @staticmethod
    def _track_type(track):
        if getattr(track, "is_foldable", False):
            return "group"
        if getattr(track, "has_midi_input", False):
            return "midi"
        if getattr(track, "has_audio_input", False):
            return "audio"
        return "other"

    def _track_summary(self, index, track):
        return {
            "index": index,
            "name": track.name,
            "type": self._track_type(track),
            "mute": bool(getattr(track, "mute", False)),
            "solo": bool(getattr(track, "solo", False)),
            "arm": bool(track.arm) if getattr(track, "can_be_armed", False) else None,
            "devices": [d.name for d in track.devices],
            "clips": [i for i, s in enumerate(track.clip_slots) if s.has_clip],
        }

    def _select_track(self, track):
        self.song.view.selected_track = track

    # ------------------------------------------------------------------ session
    def cmd_ping(self):
        return {"ghost": VERSION}

    def cmd_get_session_info(self):
        s = self.song
        return {
            "tempo": round(float(s.tempo), 3),
            "time_signature": "%d/%d" % (s.signature_numerator, s.signature_denominator),
            "is_playing": bool(s.is_playing),
            "scene_count": len(list(s.scenes)),
            "tracks": [self._track_summary(i, t) for i, t in enumerate(self._tracks())],
            "return_tracks": [t.name for t in s.return_tracks],
        }

    def cmd_get_track_info(self, track_index):
        t = self._track(track_index)
        info = self._track_summary(track_index, t)
        info["devices"] = [{"index": i, "name": d.name,
                            "class_name": getattr(d, "class_name", None),
                            "parameter_count": len(list(d.parameters))}
                           for i, d in enumerate(t.devices)]
        clips = []
        for i, slot in enumerate(t.clip_slots):
            if slot.has_clip:
                c = slot.clip
                clips.append({"clip_index": i, "name": c.name, "length_beats": float(c.length),
                              "is_midi": bool(c.is_midi_clip),
                              "is_playing": bool(getattr(c, "is_playing", False))})
        info["clips"] = clips
        return info

    def cmd_set_tempo(self, bpm):
        bpm = float(bpm)
        if not 20 <= bpm <= 999:
            raise CommandError("tempo must be between 20 and 999 BPM")
        self.song.tempo = bpm
        return {"tempo": bpm}

    def cmd_set_time_signature(self, numerator, denominator):
        if int(denominator) not in (1, 2, 4, 8, 16) or not 1 <= int(numerator) <= 99:
            raise CommandError("invalid time signature")
        self.song.signature_numerator = int(numerator)
        self.song.signature_denominator = int(denominator)
        return {"time_signature": "%d/%d" % (int(numerator), int(denominator))}

    def cmd_start_playback(self):
        self.song.start_playing()
        return {"is_playing": True}

    def cmd_stop_playback(self):
        self.song.stop_playing()
        return {"is_playing": False}

    def cmd_undo(self):
        if not self.song.can_undo:
            raise CommandError("nothing to undo")
        self.song.undo()
        return {"undone": True}

    # ------------------------------------------------------------------ tracks
    def _create_track(self, kind, index, name):
        tracks_before = len(self._tracks())
        if index is None or index == -1 or index >= tracks_before:
            index = -1
        getattr(self.song, "create_%s_track" % kind)(index)
        new_index = len(self._tracks()) - 1 if index == -1 else index
        track = self._tracks()[new_index]
        if name:
            track.name = name
        return {"track_index": new_index, "name": track.name}

    def cmd_create_midi_track(self, index=-1, name=None):
        return self._create_track("midi", index, name)

    def cmd_create_audio_track(self, index=-1, name=None):
        return self._create_track("audio", index, name)

    def cmd_set_track_name(self, track_index, name):
        self._track(track_index).name = name
        return {"track_index": track_index, "name": name}

    def cmd_delete_track(self, track_index):
        name = self._track(track_index).name
        self.song.delete_track(track_index)
        return {"deleted": name}

    def cmd_set_track_volume(self, track_index, volume):
        """volume 0.0-1.0 (0.85 is 0 dB in Live)"""
        p = self._track(track_index).mixer_device.volume
        p.value = max(p.min, min(p.max, float(volume)))
        return {"volume": p.value}

    # ------------------------------------------------------------------ clips
    def cmd_create_clip(self, track_index, clip_index, length=4.0, name=None, overwrite=False):
        track = self._track(track_index)
        if self._track_type(track) != "midi":
            raise CommandError("track %d is not a MIDI track" % track_index)
        length = float(length)
        if length <= 0:
            raise CommandError("length must be greater than 0 beats")
        slot = self._slot(track, clip_index, create_scenes=True)
        if slot.has_clip:
            if not overwrite:
                raise CommandError("slot %d on track %d already has a clip (pass overwrite=true "
                                   "to replace it)" % (clip_index, track_index))
            slot.delete_clip()
        slot.create_clip(length)
        if name:
            slot.clip.name = name
        return {"track_index": track_index, "clip_index": clip_index, "length_beats": length,
                "name": slot.clip.name}

    def cmd_delete_clip(self, track_index, clip_index):
        slot = self._slot(self._track(track_index), clip_index)
        if not slot.has_clip:
            raise CommandError("no clip to delete")
        slot.delete_clip()
        return {"deleted": True}

    def cmd_set_clip_name(self, track_index, clip_index, name):
        self._clip(track_index, clip_index).name = name
        return {"name": name}

    def cmd_set_clip_loop(self, track_index, clip_index, loop_start, loop_end):
        c = self._clip(track_index, clip_index)
        if float(loop_end) <= float(loop_start):
            raise CommandError("loop_end must be after loop_start")
        c.loop_end = float(loop_end)
        c.loop_start = float(loop_start)
        return {"loop_start": c.loop_start, "loop_end": c.loop_end}

    def cmd_fire_clip(self, track_index, clip_index):
        self._slot(self._track(track_index), clip_index).fire()
        return {"fired": True}

    def cmd_stop_track_clips(self, track_index):
        self._track(track_index).stop_all_clips()
        return {"stopped": True}

    def cmd_fire_scene(self, scene_index):
        scenes = list(self.song.scenes)
        if not 0 <= scene_index < len(scenes):
            raise CommandError("scene_index out of range")
        scenes[scene_index].fire()
        return {"fired": True}

    # ------------------------------------------------------------------ notes
    @staticmethod
    def _clean_notes(notes):
        if not isinstance(notes, (list, tuple)) or not notes:
            raise CommandError("notes must be a non-empty list")
        out = []
        for i, n in enumerate(notes):
            try:
                pitch = int(n["pitch"])
                start = float(n.get("start", n.get("start_time", 0.0)))
                dur = float(n.get("duration", 0.25))
                vel = float(n.get("velocity", 100))
                mute = bool(n.get("mute", False))
            except (KeyError, TypeError, ValueError, AttributeError):
                raise CommandError("note %d needs pitch, start, duration (and optional velocity)" % i)
            if not 0 <= pitch <= 127:
                raise CommandError("note %d: pitch must be 0-127" % i)
            if start < 0 or dur <= 0:
                raise CommandError("note %d: start must be >= 0 and duration > 0" % i)
            out.append((pitch, start, dur, max(1.0, min(127.0, vel)), mute))
        return out

    def _clear_notes(self, clip):
        span = max(float(clip.length), 0.0) + 100000.0
        if hasattr(clip, "remove_notes_extended"):
            clip.remove_notes_extended(0, 128, 0.0, span)
        else:
            clip.remove_notes(0.0, 0, span, 128)

    def cmd_add_notes(self, track_index, clip_index, notes, replace=False):
        clip = self._clip(track_index, clip_index)
        if not clip.is_midi_clip:
            raise CommandError("that clip is an audio clip")
        clean = self._clean_notes(notes)
        if replace:
            self._clear_notes(clip)
        self._write_notes(clip, clean)
        return {"added": len(clean), "clip_length_beats": float(clip.length)}

    def cmd_get_notes(self, track_index, clip_index):
        clip = self._clip(track_index, clip_index)
        if not clip.is_midi_clip:
            raise CommandError("that clip is an audio clip")
        span = float(clip.length) + 100000.0
        notes = []
        if hasattr(clip, "get_notes_extended"):
            for n in clip.get_notes_extended(0, 128, 0.0, span):
                notes.append({"pitch": n.pitch, "start": float(n.start_time),
                              "duration": float(n.duration), "velocity": float(n.velocity),
                              "mute": bool(n.mute)})
        else:
            for p, s, d, v, m in clip.get_notes(0.0, 0, span, 128):
                notes.append({"pitch": p, "start": float(s), "duration": float(d),
                              "velocity": float(v), "mute": bool(m)})
        notes.sort(key=lambda n: (n["start"], n["pitch"]))
        return {"clip_length_beats": float(clip.length), "notes": notes}

    # ------------------------------------------------------------------ browser
    def _roots(self, categories):
        browser = self.app.browser
        roots = []
        for cat in categories:
            if cat == "user_folders":
                for f in getattr(browser, "user_folders", []) or []:
                    roots.append(("user_folders", f))
            else:
                item = getattr(browser, cat, None)
                if item is not None:
                    roots.append((cat, item))
        return roots

    def _walk(self, categories):
        """Breadth-first walk of Live's browser; yields (category, item, path)."""
        queue_ = [(cat, item, item.name, 0) for cat, item in self._roots(categories)]
        seen = 0
        while queue_ and seen < MAX_BROWSER_NODES:
            nxt = []
            for cat, item, path, depth in queue_:
                for child in item.children:
                    seen += 1
                    child_path = path + "/" + child.name
                    yield cat, child, child_path
                    if child.is_folder and depth < MAX_BROWSER_DEPTH:
                        nxt.append((cat, child, child_path, depth + 1))
                    if seen >= MAX_BROWSER_NODES:
                        return
            queue_ = nxt

    def _find(self, query, categories, loadable_only=True):
        best, best_score = None, 0
        for cat, item, path in self._walk(categories):
            if loadable_only and not item.is_loadable:
                continue
            score = _match_score(item.name, query)
            if score == 3:
                return cat, item, path
            if score > best_score:
                best, best_score = (cat, item, path), score
        return best

    def cmd_search_browser(self, query, category="all", limit=25):
        cats = BROWSER_CATEGORIES if category in (None, "", "all") else (category,)
        hits = []
        for cat, item, path in self._walk(cats):
            score = _match_score(item.name, query)
            if score:
                hits.append((score, cat, item, path))
        hits.sort(key=lambda h: (-h[0], not h[2].is_loadable, len(h[3])))
        return [{"name": i.name, "category": c, "path": p, "loadable": bool(i.is_loadable)}
                for _, c, i, p in hits[: int(limit)]]

    def cmd_load_device(self, track_index, name, kind="any"):
        """Load an instrument, effect, plugin, preset or drum kit onto a track by name."""
        if kind not in DEVICE_KINDS:
            raise CommandError("kind must be one of: " + ", ".join(sorted(DEVICE_KINDS)))
        track = self._track(track_index)
        found = self._find(name, DEVICE_KINDS[kind])
        if not found:
            raise CommandError("couldn't find '%s' in Live's browser. Use search_live_browser "
                               "to see the exact names available." % name)
        cat, item, path = found
        self._select_track(track)
        self.app.browser.load_item(item)
        return {"loaded": item.name, "from": path, "category": cat,
                "track_devices": [d.name for d in track.devices]}

    def cmd_load_sample(self, track_index, path, clip_index=None, overwrite=False):
        """Put an audio file into a clip slot (audio track) or a Simpler (MIDI track)."""
        track = self._track(track_index)
        ttype = self._track_type(track)
        slot = None
        if clip_index is not None:
            slot = self._slot(track, clip_index, create_scenes=True)
            if slot.has_clip:
                if not overwrite:
                    raise CommandError("that clip slot already has a clip (overwrite=true replaces it)")
                slot.delete_clip()

        # Live 12.0.5+: load any file straight into an audio clip slot
        if ttype == "audio" and slot is not None and hasattr(slot, "create_audio_clip"):
            if not os.path.isfile(path):
                raise CommandError("file not found: %s" % path)
            slot.create_audio_clip(path)
            return {"loaded": os.path.basename(path), "method": "audio clip",
                    "track_index": track_index, "clip_index": clip_index}

        # Otherwise go through Live's browser (folder must be in Places / User Library)
        file_name = os.path.basename(path.replace("\\", "/"))
        found = self._find(file_name, SAMPLE_CATEGORIES)
        if not found or _match_score(found[1].name, file_name) < 3:
            raise CommandError(
                "'%s' isn't visible in Live's browser. In Live, add its folder to Places "
                "(browser sidebar > Add Folder), or use an audio track with a clip_index on "
                "Live 12.0.5 or newer." % file_name)
        _, item, bpath = found
        self._select_track(track)
        if slot is not None:
            self.song.view.highlighted_clip_slot = slot
        self.app.browser.load_item(item)
        return {"loaded": item.name, "method": "browser", "from": bpath,
                "track_devices": [d.name for d in track.devices]}

    # ------------------------------------------------------------------ listening
    # Ghost records a track through Live's "Resampling" input so it can analyze the result.
    LISTEN_TRACK = "Ghost Listen"

    def cmd_start_audition(self, track_index, beats=8.0, notes=None, clip_index=None):
        """Play a phrase on a track (soloed) and record it on a temporary Resampling track."""
        if getattr(self, "_audition", None):
            raise CommandError("a recording is already running; call finish_audition first")
        beats = float(beats)
        if not 1 <= beats <= 64:
            raise CommandError("beats must be between 1 and 64")
        src = self._track(track_index)
        song = self.song
        state = {"src": src, "solos": [(t, bool(t.solo)) for t in self._tracks()],
                 "created_scene": None, "src_slot": None, "temp_clip": False}

        # what to play
        if clip_index is None:
            if not notes:
                raise CommandError("give notes to play, or the clip_index of an existing clip")
            if self._track_type(src) != "midi":
                raise CommandError("to play notes the track must be a MIDI track")
            clean = self._clean_notes(notes)
            song.create_scene(-1)
            row = len(list(song.scenes)) - 1
            state["created_scene"] = row
            slot = list(src.clip_slots)[row]
            slot.create_clip(beats)
            state["temp_clip"] = True
            self._write_notes(slot.clip, clean)
        else:
            slot = self._slot(src, clip_index)
            if not slot.has_clip:
                raise CommandError("no clip at that slot to play")
            if state["created_scene"] is None:
                song.create_scene(-1)
                state["created_scene"] = len(list(song.scenes)) - 1
            row = state["created_scene"]
        state["src_slot"] = slot

        # the listening track
        song.create_audio_track(-1)
        listen = self._tracks()[-1]
        listen.name = self.LISTEN_TRACK
        state["listen"] = listen
        try:
            routing = [r for r in listen.available_input_routing_types
                       if ("%s" % (r.display_name,)).strip().lower() == "resampling"]
            if not routing:
                raise CommandError("Live didn't offer a 'Resampling' input")
            listen.input_routing_type = routing[0]
            listen.current_monitoring_state = 2  # Off, so it doesn't feed back
            listen.arm = True
            for t in self._tracks():  # hear only the source; solos are restored afterwards
                if t is not src and getattr(t, "solo", False):
                    t.solo = False
            src.solo = True
            listen_slot = list(listen.clip_slots)[state["created_scene"]]
            state["listen_slot"] = listen_slot
            slot.fire()
            try:
                listen_slot.fire(record_length=beats)
                state["auto_stop"] = True
            except TypeError:  # older Live: stop it ourselves
                listen_slot.fire()
                state["auto_stop"] = False
        except Exception:
            self._audition = state
            self._cleanup_audition()
            raise
        self._audition = state
        seconds = (beats + 4.0) * 60.0 / float(song.tempo)  # + up to one bar of launch delay
        return {"recording": True, "wait_seconds": round(seconds + 0.5, 2), "beats": beats}

    def cmd_finish_audition(self):
        """Stop the recording (if it's done) and return the recorded file. Cleanup is separate
        so the file can be copied first."""
        st = getattr(self, "_audition", None)
        if not st:
            raise CommandError("no recording is running")
        slot = st["listen_slot"]
        clip = slot.clip if slot.has_clip else None
        if clip is not None and getattr(clip, "is_recording", False) and st.get("auto_stop"):
            return {"done": False}
        self.song.stop_playing()
        if not slot.has_clip:
            return {"done": True, "file": None,
                    "error": "nothing was recorded (check Live's audio device is running)"}
        return {"done": True, "file": getattr(slot.clip, "file_path", None)}

    def cmd_end_audition(self):
        if not getattr(self, "_audition", None):
            return {"cleaned": False}
        self._cleanup_audition()
        return {"cleaned": True}

    def _cleanup_audition(self):
        st, self._audition = self._audition, None
        song = self.song
        try:
            song.stop_playing()
        except Exception:
            pass
        for t, was in st.get("solos", []):
            try:
                t.solo = was
            except Exception:
                pass
        listen = st.get("listen")
        if listen is not None:
            tracks = self._tracks()
            if listen in tracks:
                song.delete_track(tracks.index(listen))
        if st.get("temp_clip") and st.get("src_slot") is not None and st["src_slot"].has_clip:
            st["src_slot"].delete_clip()
        row = st.get("created_scene")
        if row is not None and row < len(list(song.scenes)):
            empty = all(not list(t.clip_slots)[row].has_clip for t in self._tracks())
            if empty:
                song.delete_scene(row)

    def _write_notes(self, clip, clean):
        if hasattr(clip, "add_new_notes"):
            import Live
            spec = Live.Clip.MidiNoteSpecification
            clip.add_new_notes(tuple(spec(pitch=p, start_time=s, duration=d, velocity=v, mute=m)
                                     for p, s, d, v, m in clean))
        else:
            clip.set_notes(tuple(clean))

    # ------------------------------------------------------------------ device parameters
    def _device(self, track_index, device_index):
        devices = list(self._track(track_index).devices)
        if not 0 <= device_index < len(devices):
            raise CommandError("device_index out of range (track has %d devices)" % len(devices))
        return devices[device_index]

    def cmd_get_device_parameters(self, track_index, device_index, query=None):
        d = self._device(track_index, device_index)
        params = list(d.parameters)
        q = (query or "").lower().strip()
        out = []
        for i, p in enumerate(params):
            if q and q not in p.name.lower():
                continue
            e = {"index": i, "name": p.name, "value": round(float(p.value), 6),
                 "display": params_mod.display(p), "min": float(p.min), "max": float(p.max)}
            if getattr(p, "is_quantized", False):
                e["choices"] = params_mod.choices(p)
            out.append(e)
        res = {"device": d.name, "class_name": getattr(d, "class_name", None),
               "parameter_count": len(params), "parameters": out}
        if getattr(d, "class_name", "") in PLUGIN_CLASSES and len(params) <= 2:
            res["hint"] = ("This plugin isn't sharing its controls with Live yet. Ask the user to "
                           "open the device in Live, click Configure (the wrench icon), move each "
                           "control Ghost should use (up to 128), then click Configure again.")
        return res

    def _find_param(self, d, params, parameter):
        if isinstance(parameter, int) or (isinstance(parameter, str) and parameter.isdigit()):
            i = int(parameter)
            if 0 <= i < len(params):
                return params[i]
            raise CommandError("parameter index %d out of range on %s" % (i, d.name))
        q = ("%s" % (parameter,)).lower().strip()
        exact = [p for p in params if p.name.lower().strip() == q]
        if exact:
            return exact[0]
        partial = [p for p in params if q in p.name.lower()]
        if len(partial) == 1:
            return partial[0]
        if partial:
            raise CommandError("'%s' matches several parameters on %s: %s"
                               % (parameter, d.name, ", ".join(p.name for p in partial[:12])))
        raise CommandError("parameter '%s' not found on %s (use get_device_parameters)"
                           % (parameter, d.name))

    def cmd_set_device_parameters(self, track_index, device_index, settings):
        """settings: [{"parameter": name or index, "value": raw number or text like '800 Hz'}]"""
        d = self._device(track_index, device_index)
        params = list(d.parameters)
        if not isinstance(settings, (list, tuple)) or not settings:
            raise CommandError("settings must be a non-empty list")
        done, errors = [], []
        for s in settings:
            try:
                p = self._find_param(d, params, s.get("parameter"))
                done.append(params_mod.set_param(p, s.get("value")))
            except (CommandError, params_mod.ParamError) as e:
                errors.append({"parameter": s.get("parameter"), "error": str(e)})
            except Exception as e:
                errors.append({"parameter": s.get("parameter"),
                               "error": "%s: %s" % (type(e).__name__, e)})
        return {"device": d.name, "set": done, "errors": errors}

    def cmd_set_device_parameter(self, track_index, device_index, parameter, value):
        r = self.cmd_set_device_parameters(track_index, device_index,
                                           [{"parameter": parameter, "value": value}])
        if r["errors"]:
            raise CommandError(r["errors"][0]["error"])
        out = dict(r["set"][0])
        out["device"] = r["device"]
        return out
