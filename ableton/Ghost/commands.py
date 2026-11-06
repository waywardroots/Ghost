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



def _safe(obj, attr, default=None):
    """getattr that also survives Live's RuntimeErrors (group, return and master tracks raise
    instead of returning nothing for things like arrangement_clips, clip_slots or mute)."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default

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

    MASTER = -1  # track_index -1 addresses the Master track (devices, volume)

    def _track(self, track_index):
        if track_index == self.MASTER and getattr(self.song, "master_track", None) is not None:
            return self.song.master_track
        tracks = self._tracks()
        if not isinstance(track_index, int) or not 0 <= track_index < len(tracks):
            raise CommandError("track_index %r is out of range (this set has %d tracks, "
                               "numbered from 0; -1 is the Master)" % (track_index, len(tracks)))
        return tracks[track_index]

    def _back_to_arrangement(self):
        """Press Live's 'Back to Arrangement' so every track follows the timeline again
        (firing a Session clip takes that track off it)."""
        try:
            if getattr(self.song, "back_to_arranger", False):
                self.song.back_to_arranger = False
        except Exception:
            pass

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

    def _group_info(self, track):
        """Groups make one sound: list a group's member tracks, and a member's group."""
        tracks = self._tracks()
        info = {}
        if _safe(track, "is_foldable", False):
            info["group_members"] = [i for i, t in enumerate(tracks)
                                     if _safe(t, "group_track", None) is track]
        parent = _safe(track, "group_track", None) if _safe(track, "is_grouped", False) else None
        if parent is not None and parent in tracks:
            info["in_group"] = tracks.index(parent)
        return info

    def _track_summary(self, index, track):
        is_master = track is getattr(self.song, "master_track", None)
        out = self._track_summary_base(index, track, is_master)
        out.update(self._group_info(track))
        return out

    def _track_summary_base(self, index, track, is_master):
        return {
            "index": index,
            "name": track.name,
            "type": "master" if is_master else self._track_type(track),
            "mute": bool(_safe(track, "mute", False)),
            "solo": bool(_safe(track, "solo", False)),
            "arm": bool(_safe(track, "arm", False)) if _safe(track, "can_be_armed", False) else None,
            "devices": [d.name for d in track.devices],
            "clips": [i for i, s in enumerate(_safe(track, "clip_slots", []) or [])
                      if s.has_clip],
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
        for i, slot in enumerate(_safe(t, "clip_slots", []) or []):
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
        if track_index == self.MASTER:
            raise CommandError("the Master track can't be deleted")
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

    def cmd_load_sample(self, track_index, path, clip_index=None, overwrite=False, transpose=0):
        """Put an audio file into a clip slot (audio track) or a Simpler (MIDI track).
        transpose: semitones for an audio clip (e.g. to match a transposed copy on the timeline)."""
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
            transpose = int(transpose or 0)
            if not -48 <= transpose <= 48:
                raise CommandError("transpose must be between -48 and 48 semitones")
            clip = slot.create_audio_clip(path) or slot.clip
            out = {"loaded": os.path.basename(path), "method": "audio clip",
                   "track_index": track_index, "clip_index": clip_index}
            if transpose and clip is not None:
                clip.pitch_coarse = transpose
                out["transpose"] = transpose
            return out

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

    # ------------------------------------------------------------------ scenes & arrangement
    def cmd_set_scene_name(self, scene_index, name):
        scenes = list(self.song.scenes)
        if not 0 <= scene_index < len(scenes):
            raise CommandError("scene_index out of range (%d scenes)" % len(scenes))
        scenes[scene_index].name = name
        return {"scene_index": scene_index, "name": name}

    def cmd_duplicate_clip(self, track_index, clip_index, target_clip_index, overwrite=False):
        track = self._track(track_index)
        src = self._slot(track, clip_index)
        if not src.has_clip:
            raise CommandError("no clip at track %d, slot %d" % (track_index, clip_index))
        dst = self._slot(track, target_clip_index, create_scenes=True)
        if dst.has_clip:
            if not overwrite:
                raise CommandError("target slot already has a clip (overwrite=true replaces it)")
            dst.delete_clip()
        src.duplicate_clip_to(dst)
        return {"track_index": track_index, "from": clip_index, "to": target_clip_index}

    def _beats_per_bar(self):
        return self.song.signature_numerator * 4.0 / self.song.signature_denominator

    def _arrangement_end(self):
        end = 0.0
        for t in self._tracks():
            for c in _safe(t, "arrangement_clips", []) or []:
                end = max(end, float(c.end_time))
        return end

    def cmd_get_arrangement_info(self):
        bpb = self._beats_per_bar()
        tracks = []
        for i, t in enumerate(self._tracks()):
            clips = [{"name": c.name, "start_bar": round(float(c.start_time) / bpb + 1, 2),
                      "end_bar": round(float(c.end_time) / bpb + 1, 2)}
                     for c in (_safe(t, "arrangement_clips", []) or [])]
            if clips:
                tracks.append({"index": i, "name": t.name, "clips": clips})
        cues = [{"name": c.name, "bar": round(float(c.time) / bpb + 1, 2)}
                for c in (getattr(self.song, "cue_points", []) or [])]
        return {"length_bars": round(self._arrangement_end() / bpb, 2), "tracks": tracks,
                "locators": sorted(cues, key=lambda c: c["bar"])}

    def cmd_build_arrangement(self, sections, start_bar=None):
        """sections: [{"name": "Intro", "bars": 16, "clips": [{"track_index": 0, "clip_index": 0}]}]
        Copies Session clips onto the Arrangement timeline, repeating each to fill its section,
        and drops a named locator at the start of every section."""
        if not isinstance(sections, (list, tuple)) or not sections:
            raise CommandError("sections must be a non-empty list")
        song, bpb = self.song, self._beats_per_bar()
        tracks = self._tracks()
        if not tracks or not hasattr(tracks[0], "duplicate_clip_to_arrangement"):
            raise CommandError("this version of Live can't place clips in the Arrangement "
                               "(needs Live 11 or newer)")
        # check everything first, so a bad section changes nothing
        plan = []
        for n, sec in enumerate(sections):
            name = ("%s" % sec.get("name", "Section %d" % (n + 1))).strip()
            bars = sec.get("bars")
            if not isinstance(bars, (int, float)) or bars <= 0:
                raise CommandError("section '%s' needs bars > 0" % name)
            beats = float(bars) * bpb
            items = []
            for c in sec.get("clips") or []:
                ti, ci = c.get("track_index"), c.get("clip_index", 0)
                track = self._track(ti)
                slot = self._slot(track, ci)
                if not slot.has_clip:
                    raise CommandError("section '%s': no clip at track %d, slot %d" % (name, ti, ci))
                length = float(slot.clip.length)
                if length <= 0:
                    raise CommandError("section '%s': clip on track %d has no length" % (name, ti))
                reps = beats / length
                if abs(reps - round(reps)) > 1e-6:
                    raise CommandError(
                        "section '%s' (%g bars) isn't a whole number of repeats of the %g-bar clip "
                        "on track %d (%s). Use section lengths that fit every clip."
                        % (name, bars, length / bpb, ti, track.name))
                items.append((track, slot.clip, length, int(round(reps))))
            plan.append((name, beats, items))

        was_playing = bool(song.is_playing)
        if was_playing:
            song.stop_playing()
        if start_bar is None:
            t0 = self._arrangement_end()
            t0 = (int(t0 / bpb) + (1 if t0 % bpb else 0)) * bpb  # next whole bar
        else:
            t0 = (float(start_bar) - 1) * bpb
        start = t0
        placed, out = 0, []
        for name, beats, items in plan:
            for track, clip, length, reps in items:
                for k in range(reps):
                    track.duplicate_clip_to_arrangement(clip, t0 + k * length)
                    placed += 1
            out.append({"section": name, "start_bar": int(round(t0 / bpb)) + 1,
                        "start_beat": t0, "bars": beats / bpb, "tracks": len(items)})
            t0 += beats
        try:
            song.current_song_time = start
        except Exception:
            pass
        return {"sections": out, "clips_placed": placed,
                "end_bar": int(round(t0 / bpb)) + 1, "length_bars": (t0 - start) / bpb}

    # ---- arrangement editing
    def cmd_clear_arrangement(self, track_indices, from_bar=1, to_bar=None):
        """Delete Arrangement clips on the given tracks that overlap bars from_bar..to_bar
        (to_bar exclusive; None = to the end). Session clips are never touched."""
        bpb = self._beats_per_bar()
        t0 = (float(from_bar) - 1) * bpb
        t1 = None if to_bar is None else (float(to_bar) - 1) * bpb
        removed = 0
        for ti in track_indices:
            track = self._track(ti)
            if not hasattr(track, "delete_clip"):
                raise CommandError("this version of Live can't delete Arrangement clips from a "
                                   "script (needs Live 12)")
            for c in list(_safe(track, "arrangement_clips", []) or []):
                s, e = float(c.start_time), float(c.end_time)
                if e > t0 + 1e-9 and (t1 is None or s < t1 - 1e-9):
                    track.delete_clip(c)
                    removed += 1
        return {"removed": removed}

    def cmd_rearrange(self, moves, track_indices=None):
        """Move whole sections of the Arrangement: each move copies every clip that starts in
        bars from_bar..to_bar (to_bar exclusive) on the chosen tracks to new_bar, with its
        automation. A section can be used twice (e.g. a breakdown repeated), and bars that no
        move covers are dropped. Safe order: everything is copied to a spare area past the
        end first, so nothing is deleted unless all copies worked."""
        bpb = self._beats_per_bar()
        moves = [(float(m["from_bar"]), float(m["to_bar"]), float(m["new_bar"])) for m in moves]
        if not moves:
            raise CommandError("no moves given")
        for a, b, n in moves:
            if b <= a or a < 1 or n < 1:
                raise CommandError("each move needs from_bar < to_bar and bars from 1")
        new = sorted((n, n + (b - a)) for a, b, n in moves)
        for (s1, e1), (s2, e2) in zip(new, new[1:]):
            if s2 < e1 - 1e-9:
                raise CommandError("two moves land on the same bars (%g-%g and %g-%g)"
                                   % (s1, e1, s2, e2))
        tracks = ([self._track(i) for i in track_indices] if track_indices is not None
                  else list(self.song.tracks))
        for tr in tracks:
            if not hasattr(tr, "delete_clip"):
                raise CommandError("this version of Live can't move Arrangement clips from a "
                                   "script (needs Live 12)")
        song_end = 0.0
        for tr in tracks:
            for c in list(_safe(tr, "arrangement_clips", []) or []):
                song_end = max(song_end, float(c.end_time))
        stage = (int(song_end / bpb) + 64) * bpb  # spare area well past the last clip
        staged, originals = [], []
        try:
            for tr in tracks:
                clips = list(_safe(tr, "arrangement_clips", []) or [])
                originals.append((tr, clips))
                for a, b, n in moves:
                    t0, t1 = (a - 1) * bpb, (b - 1) * bpb
                    for c in clips:
                        s = float(c.start_time)
                        if t0 - 1e-6 <= s < t1 - 1e-6:
                            dest = stage + (n - 1) * bpb + (s - t0)
                            tr.duplicate_clip_to_arrangement(c, dest)
                            staged.append((tr, dest))
        except Exception as e:
            self._delete_starting_at(staged)
            raise CommandError("couldn't copy the sections (%s); nothing was changed" % e)
        removed = 0
        for tr, clips in originals:
            for c in clips:
                tr.delete_clip(c)
                removed += 1
        placed = 0
        for tr, dest in staged:
            c = self._arrangement_clip_at(tr, dest)
            if c is None:
                continue
            tr.duplicate_clip_to_arrangement(c, dest - stage)
            tr.delete_clip(c)
            placed += 1
        end_bar = max(e for _, e in new)
        return {"clips_moved": placed, "clips_before": removed, "new_end_bar": end_bar}

    def _delete_starting_at(self, staged):
        for tr, dest in staged:
            c = self._arrangement_clip_at(tr, dest)
            if c is not None:
                tr.delete_clip(c)

    def cmd_get_drum_pads(self, track_index):
        """Which sounds sit on which notes in a track's Drum Rack."""
        track = self._track(track_index)
        racks = [d for d in track.devices if getattr(d, "can_have_drum_pads", False)]
        if not racks:
            raise CommandError("track %d has no Drum Rack" % track_index)
        pads = []
        for p in racks[0].drum_pads:
            if list(getattr(p, "chains", []) or []):
                pads.append({"note": int(p.note), "name": p.name})
        return {"device": racks[0].name, "pads": pads}

    def cmd_place_clip(self, track_index, clip_index, bar, repeats=1):
        """Put one Session clip on the Arrangement timeline at a bar (replacing what's there)."""
        track = self._track(track_index)
        slot = self._slot(track, clip_index)
        if not slot.has_clip:
            raise CommandError("no clip at track %d, slot %d" % (track_index, clip_index))
        bpb, length = self._beats_per_bar(), float(slot.clip.length)
        t = (float(bar) - 1) * bpb
        for k in range(int(repeats)):
            track.duplicate_clip_to_arrangement(slot.clip, t + k * length)
        return {"placed": slot.clip.name, "start_bar": bar,
                "end_bar": round((t + int(repeats) * length) / bpb + 1, 3)}

    def _arrangement_clip_at(self, track, t):
        for c in list(_safe(track, "arrangement_clips", []) or []):
            if abs(float(c.start_time) - t) < 1e-3:
                return c
        return None

    def cmd_place_audio_clip(self, track_index, path, bar, transpose=0, repeats=1):
        """Put an audio file (riser, impact, vocal, loop...) on an audio track's timeline at a
        bar, optionally transposed (semitones) and repeated back to back."""
        track = self._track(track_index)
        if self._track_type(track) != "audio":
            raise CommandError("track %d is not an audio track" % track_index)
        if not os.path.isfile(path):
            raise CommandError("file not found: %s" % path)
        transpose, repeats = int(transpose), max(1, int(repeats))
        if not -48 <= transpose <= 48:
            raise CommandError("transpose must be between -48 and 48 semitones")
        bpb = self._beats_per_bar()
        t = (float(bar) - 1) * bpb
        if hasattr(track, "create_audio_clip"):  # Live 12: straight onto the timeline
            placed = []
            for _ in range(repeats):
                clip = track.create_audio_clip(path, t) or self._arrangement_clip_at(track, t)
                if clip is None:
                    break
                if transpose:
                    clip.pitch_coarse = transpose
                placed.append(clip)
                t = float(clip.end_time)
            out = {"placed": os.path.basename(path), "start_bar": bar, "method": "direct",
                   "copies": len(placed)}
            if placed:
                out["end_bar"] = round(float(placed[-1].end_time) / bpb + 1, 3)
            if transpose:
                out["transpose"] = transpose
            return out
        if transpose or repeats > 1:
            raise CommandError("transpose and repeats need Live 12")
        # Otherwise: load it into a spare Session slot, copy it to the timeline, clean up
        song = self.song
        made_scene = False
        slots = list(track.clip_slots)
        free = [s for s in slots if not s.has_clip]
        if not free:
            song.create_scene(-1)
            made_scene = True
            free = [list(track.clip_slots)[-1]]
        slot = free[0]
        if not hasattr(slot, "create_audio_clip"):
            raise CommandError("this version of Live can't load audio files from a script "
                               "(needs Live 12)")
        slot.create_audio_clip(path)
        try:
            track.duplicate_clip_to_arrangement(slot.clip, t)
        finally:
            slot.delete_clip()
            if made_scene:
                song.delete_scene(len(list(song.scenes)) - 1)
        return {"placed": os.path.basename(path), "start_bar": bar, "method": "via session"}

    def cmd_automate(self, track_index, device_index, parameter, points, clip_index=0,
                     step_beats=0.25, lanes=None):
        """Draw automation onto the Arrangement between the first and last point.

        Live only lets scripts draw into Session clips, so for each stretch of the timeline
        Ghost copies the track's Session clip (clip_index) into a spare slot, draws the
        automation into that copy, places the copy on the timeline (replacing what was there)
        and removes the spare copy. points: [{bar, value}], value = raw or display text.
        lanes: more [{device_index, parameter, points}] drawn into the same clips, so several
        parameters can move together. Automation already on the replaced clips is kept where
        Live lets Ghost read it."""
        song, track = self.song, self._track(track_index)
        src = self._slot(track, clip_index)
        if not src.has_clip:
            raise CommandError("track %d has no Session clip in slot %d to carry the automation"
                               % (track_index, clip_index))
        bpb = self._beats_per_bar()
        specs = [{"device_index": device_index, "parameter": parameter, "points": points}]
        specs += list(lanes or [])
        drawn = []  # (param, device, pts)
        for spec in specs:
            d = self._device(track_index, int(spec["device_index"]))
            p = self._find_param(d, list(d.parameters), spec["parameter"])
            if any(q is p for q, _, _ in drawn):
                raise CommandError("%s is listed twice" % p.name)
            pts_in = spec.get("points")
            if not isinstance(pts_in, (list, tuple)) or len(pts_in) < 2:
                raise CommandError("give at least two points for %s" % p.name)
            old = float(p.value)
            try:
                pts = sorted(((float(pt["bar"]) - 1) * bpb, params_mod.resolve(p, pt["value"]))
                             for pt in pts_in)
            except params_mod.ParamError as e:
                raise CommandError(str(e))
            finally:
                p.value = old  # resolving may probe the parameter; put it back
            if pts[-1][0] <= pts[0][0]:
                raise CommandError("points for %s must span some time" % p.name)
            drawn.append((p, d, pts))
        t0 = min(pts[0][0] for _, _, pts in drawn)
        t1 = max(pts[-1][0] for _, _, pts in drawn)

        def value_at(pts, t):
            for (a, va), (b, vb) in zip(pts, pts[1:]):
                if a <= t <= b:
                    return va if b == a else va + (vb - va) * (t - a) / (b - a)
            return pts[-1][1] if t > pts[-1][0] else pts[0][1]

        length = float(src.clip.length)
        step = max(0.0625, float(step_beats))
        made_scene, placed, steps = False, 0, 0
        mine = [q for q, _, _ in drawn]
        others = [q for q in self._automatable(track) if not any(q is m for m in mine)]
        kept, found_old = set(), 0
        free = [s for s in track.clip_slots if not s.has_clip]
        if not free:
            song.create_scene(-1)
            made_scene = True
            free = [list(track.clip_slots)[-1]]
        tmp = free[0]
        try:
            start = t0
            while start < t1 - 1e-9:
                src.duplicate_clip_to(tmp)
                clip = tmp.clip
                # Keep automation already drawn on the clip being replaced (other lanes).
                old = self._arrangement_clip_at(track, start)
                if old is not None:
                    found_old += 1
                    kept.update(self._copy_envelopes(old, clip, others, length, step))
                for p, _, pts in drawn:
                    env = clip.automation_envelope(p) or clip.create_automation_envelope(p)
                    if env is None:
                        raise CommandError("Live won't automate %s here" % p.name)
                    t = 0.0
                    while t < length - 1e-9:
                        st = min(step, length - t)
                        env.insert_step(t, st, value_at(pts, start + t + st / 2))
                        steps += 1
                        t += st
                track.duplicate_clip_to_arrangement(clip, start)
                tmp.delete_clip()
                placed += 1
                start += length
        finally:
            if tmp.has_clip:
                tmp.delete_clip()
            if made_scene:
                song.delete_scene(len(list(song.scenes)) - 1)
        for p, _, pts in drawn:
            p.value = pts[-1][1]  # rest at the end value outside the automated clips
        p, d, _ = drawn[0]
        out = {"parameter": p.name, "device": d.name, "clips_placed": placed, "steps": steps,
               "ends_at": params_mod.display(p), "start_bar": t0 / bpb + 1,
               "end_bar": (t0 + placed * length) / bpb + 1,
               "replaced_clips": found_old, "kept_automation": sorted(kept)}
        if len(drawn) > 1:
            out["lanes"] = [{"parameter": q.name, "device": dd.name,
                             "ends_at": params_mod.display(q)} for q, dd, _ in drawn]
        return out

    def _automatable(self, track):
        """Every parameter on the track a clip envelope can hold: devices plus the mixer."""
        out = []
        for d in list(track.devices):
            out.extend(list(d.parameters))
        mixer = getattr(track, "mixer_device", None)
        if mixer is not None:
            for name in ("volume", "panning"):
                q = getattr(mixer, name, None)
                if q is not None:
                    out.append(q)
            out.extend(list(getattr(mixer, "sends", []) or []))
        return out

    def _copy_envelopes(self, old, clip, params, length, step):
        """Redraw old's automation for params into clip, sampled every step. Returns names."""
        names = []
        for q in params:
            try:
                src_env = old.automation_envelope(q)
            except Exception:
                src_env = None
            if src_env is None:
                continue
            dst = clip.automation_envelope(q) or clip.create_automation_envelope(q)
            if dst is None:
                continue
            t, wrote = 0.0, False
            while t < length - 1e-9:
                st = min(step, length - t)
                v = src_env.value_at(t + st / 2)
                if v is not None:
                    dst.insert_step(t, st, v)
                    wrote = True
                t += st
            if wrote:
                names.append(q.name)
        return names

    # Locators: Live moves the playhead a moment after it's set, so placing a locator takes
    # two steps (set_song_time, then cue_at_playhead once the playhead has arrived).
    def cmd_get_song_time(self):
        return {"beat": float(self.song.current_song_time),
                "is_playing": bool(self.song.is_playing)}

    def cmd_set_song_time(self, beat):
        self.song.current_song_time = max(0.0, float(beat))
        return {"requested_beat": float(beat)}

    def _cue_at(self, beat):
        for c in self.song.cue_points:
            if abs(float(c.time) - beat) < 1e-3:
                return c
        return None

    def cmd_cue_at_playhead(self, expect_beat, name=None, delete=False):
        song = self.song
        now = float(song.current_song_time)
        if abs(now - float(expect_beat)) > 1e-3:
            return {"ready": False, "playhead_beat": now}
        cue = self._cue_at(now)
        if delete:
            if cue is not None:
                song.set_or_delete_cue()
            return {"ready": True, "deleted": cue is not None}
        created = cue is None
        if created:
            song.set_or_delete_cue()
            cue = self._cue_at(now)
        if cue is None:
            return {"ready": True, "error": "Live didn't create the locator"}
        if name:
            cue.name = name
        return {"ready": True, "created": created, "name": cue.name}

    def cmd_get_locators(self):
        bpb = self._beats_per_bar()
        return [{"name": c.name, "beat": float(c.time), "bar": round(float(c.time) / bpb + 1, 3)}
                for c in sorted(self.song.cue_points, key=lambda c: float(c.time))]

    # ------------------------------------------------------------------ listening
    # Ghost records a track through Live's "Resampling" input so it can analyze the result.
    LISTEN_TRACK = "Ghost Listen"

    def cmd_start_audition(self, track_index, beats=8.0, notes=None, clip_index=None,
                           arrangement=False):
        """Play a phrase on a track (soloed) and record it on a temporary Resampling track.
        arrangement=True plays the Arrangement from the playhead instead (set it first), with
        the track soloed - for a group that is all its member tracks, as one sound.
        track_index -1 records the whole mix."""
        if getattr(self, "_audition", None):
            raise CommandError("a recording is already running; call finish_audition first")
        beats = float(beats)
        if not 1 <= beats <= 64:
            raise CommandError("beats must be between 1 and 64")
        song = self.song
        mix = track_index == self.MASTER
        src = None if mix else self._track(track_index)
        state = {"src": src, "solos": [(t, bool(_safe(t, "solo", False))) for t in self._tracks()],
                 "created_scene": None, "src_slot": None, "temp_clip": False, "mix": mix,
                 # armed tracks, noted before Ghost arms its own (Exclusive Arm may disarm them)
                 "arms": [t for t in self._tracks()
                          if _safe(t, "can_be_armed", False) and _safe(t, "arm", False)]}

        # what to play
        if mix or arrangement:  # the Arrangement from the playhead (set it with set_song_time)
            state["mix"] = True
            self._back_to_arrangement()
            slot = None
        elif clip_index is None:
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
                if t is not src and _safe(t, "solo", False):
                    t.solo = False
            if src is not None:
                src.solo = True
            if state["mix"]:
                self._start_arrangement_recording(state, listen, beats)
            else:
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
        out = {"recording": True, "wait_seconds": round(seconds + 0.5, 2), "beats": beats}
        if state["mix"]:
            bpb = self._beats_per_bar()
            out["from_bar"] = round(state["rec_start"] / bpb + 1, 3)  # recording starts a bar in
            seconds = (beats + 2 * bpb) * 60.0 / float(song.tempo)
            out["wait_seconds"] = round(seconds + 0.5, 2)
        return out

    # Recording the Arrangement: Live's own Arrangement recording on the temporary track,
    # punched in/out to exactly the bars being measured. (Launching a Session record slot while
    # the Arrangement plays left plugin synths silent in the recording on some setups.)
    _TRANSPORT = ("loop", "loop_start", "loop_length", "punch_in", "punch_out", "record_mode")

    def _start_arrangement_recording(self, state, listen, beats):
        song, bpb = self.song, self._beats_per_bar()
        state["transport"] = dict((k, getattr(song, k)) for k in self._TRANSPORT
                                  if hasattr(song, k))
        # nothing of the user's may record: disarm every other track (restored afterwards)
        for t in self._tracks():
            if t is not listen and _safe(t, "can_be_armed", False) and _safe(t, "arm", False):
                if t not in state["arms"]:
                    state["arms"].append(t)
                t.arm = False
        play_from = float(song.current_song_time)
        start = play_from + bpb  # one bar of run-up, like pressing Play a bar early
        state["play_from"], state["rec_start"], state["rec_end"] = play_from, start, start + beats
        song.loop = False
        # Live keeps the loop inside the song: shrink it before moving it
        song.loop_length = min(float(beats), float(song.loop_length))
        song.loop_start = start
        song.loop_length = beats
        song.punch_in = True
        song.punch_out = True
        song.record_mode = True
        song.start_playing()

    def _restore_transport(self, st):
        song = self.song
        tr = st.get("transport", {})
        if "loop_length" in tr:  # shrink first so the loop can move anywhere (see above)
            try:
                song.loop_length = min(float(tr["loop_length"]), float(song.loop_length))
            except Exception:
                pass
        for k in ("record_mode", "punch_in", "punch_out", "loop_start", "loop_length", "loop"):
            if k in tr:
                try:
                    setattr(song, k, tr[k])
                except Exception:
                    pass
        st["transport"] = {}

    def _restore_arms(self, st):
        listen = st.get("listen")
        for t in st.get("arms", []):
            if t is listen or t not in self._tracks():
                continue
            try:
                t.arm = True
            except Exception:
                pass
        st["arms"] = []

    def _arrangement_recording(self, st):
        listen = st.get("listen")
        for c in list(_safe(listen, "arrangement_clips", []) or []):
            if getattr(c, "file_path", None):
                return c
        return None

    def cmd_finish_audition(self):
        """Stop the recording (if it's done) and return the recorded file. Cleanup is separate
        so the file can be copied first."""
        st = getattr(self, "_audition", None)
        if not st:
            raise CommandError("no recording is running")
        if st.get("mix"):
            song = self.song
            if song.is_playing and float(song.current_song_time) < st["rec_end"] + 0.25:
                return {"done": False, "beat": float(song.current_song_time)}
            song.stop_playing()
            try:
                song.record_mode = False
            except Exception:
                pass
            clip = self._arrangement_recording(st)
            if clip is None:
                return {"done": True, "file": None,
                        "error": "nothing was recorded (check Live's audio device is running)"}
            return {"done": True, "file": clip.file_path}
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
        self._restore_transport(st)
        listen = st.get("listen")
        if listen is not None:
            tracks = self._tracks()
            if listen in tracks:
                song.delete_track(tracks.index(listen))
        self._restore_arms(st)
        self._back_to_arrangement()
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
