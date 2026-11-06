"""A small stand-in for Ableton Live's Python API so Ghost can be tested without Live."""
import sys
import types


class Param:
    def __init__(self, name, value=0.0, mn=0.0, mx=1.0):
        self.name, self.value, self.min, self.max = name, value, mn, mx
        self.is_quantized = False


class ShownParam(Param):
    """A parameter that displays its value like a real plugin does."""

    def __init__(self, name, fmt, value=0.5, mn=0.0, mx=1.0, items=None):
        Param.__init__(self, name, value, mn, mx)
        self._fmt = fmt
        if items:
            self.value_items = items
            self.is_quantized = True

    def str_for_value(self, v):
        return self._fmt(v)

    def __str__(self):
        return self._fmt(self.value)


def _hz(v):
    f = 20.0 * (1000.0 ** v)
    return "%.0f Hz" % f if f < 1000 else "%.2f kHz" % (f / 1000)


def _ms(v):
    t = 0.5 + 9999.5 * v ** 3
    return "%.1f ms" % t if t < 1000 else "%.2f s" % (t / 1000)


def _db(v):
    import math
    return "-inf dB" if v <= 0 else "%.1f dB" % (20 * math.log10(v) + 6)


def synth_params():
    waves = ["Sine", "Triangle", "Saw", "Square", "Noise"]
    return [Param("Device On", 1.0),
            ShownParam("Filter Cutoff", _hz, 0.7),
            ShownParam("Filter Resonance", lambda v: "%d %%" % round(v * 100), 0.1),
            ShownParam("Amp Attack", _ms, 0.05),
            ShownParam("Amp Release", _ms, 0.3),
            ShownParam("Osc A Wave", lambda v: waves[int(round(v))], 0.0, 0.0, 4.0, waves),
            ShownParam("Osc B Wave", lambda v: waves[int(round(v))], 2.0, 0.0, 4.0, waves),
            ShownParam("Master Volume", _db, 0.5),
            ShownParam("Darkness", lambda v: "%d %%" % round((1 - v) * 100), 0.5),  # runs backwards
            ShownParam("Unison", lambda v: "%d" % int(round(v)), 1.0, 1.0, 16.0)]


class LaggyParam(Param):
    """Like many VST2 plugins: can't preview values, display updates on Live's next refresh."""

    def __init__(self, name, fmt, value=0.5):
        Param.__init__(self, name, value)
        self._fmt = fmt
        self._shown = fmt(value)

    def str_for_value(self, v):
        return self._shown  # VST2: only ever the current (possibly stale) text

    def __str__(self):
        return self._shown

    def refresh(self):
        self._shown = self._fmt(self.value)


class RawParam(LaggyParam):
    """A plugin control that (like SubBoomBass2 right after a set loads) hasn't reported its
    own text: Live shows the bare raw number. waking=True starts reporting once it's moved."""

    def __init__(self, name, fmt, value=0.5, waking=False):
        LaggyParam.__init__(self, name, fmt, value)
        self._waking, self._start, self._awake = waking, value, False
        self._shown = "%.2f" % value

    def str_for_value(self, v):
        # like SubBoomBass2 after a reload: the preview echoes the raw number
        return "%.2f" % v if not self._awake else self._shown

    def refresh(self):
        if self._waking and self.value != self._start:
            self._awake = True
        self._shown = self._fmt(self.value) if self._awake else "%.2f" % self.value


def old_plugin():
    styles = "AENTP"
    return Device("Old Verb", [Param("Device On", 1.0),
                              LaggyParam("Decay", lambda v: "%.2f s " % (0.3 * (233.0 ** v))),
                              LaggyParam("Style", lambda v: styles[min(4, int(v * 5))] + " ", 0.0),
                              LaggyParam("Mix", lambda v: "%.1f %% " % (v * 100), 1.0),
                              RawParam("Tone", lambda v: _hz(v) + "  ", 0.25),
                              RawParam("Cutoff", lambda v: _hz(v) + "  ", 0.25, waking=True)],
                  class_name="PluginDevice")


class Device:
    def __init__(self, name, params=("Device On", "Cutoff", "Resonance"), class_name=None):
        self.name = name
        self.class_name = class_name or name.replace(" ", "")
        self.parameters = [p if isinstance(p, Param) else Param(p) for p in params]


class Note:
    def __init__(self, pitch, start_time, duration, velocity, mute):
        self.pitch, self.start_time, self.duration = pitch, start_time, duration
        self.velocity, self.mute = velocity, mute


class Clip:
    def __init__(self, length, is_midi=True, name=""):
        self.length, self.is_midi_clip, self.name = length, is_midi, name
        self.loop_start, self.loop_end = 0.0, length
        self.is_playing = False
        self._notes = []
        self.envelopes = {}

    def automation_envelope(self, param):
        return self.envelopes.get(id(param))

    def create_automation_envelope(self, param):
        return self.envelopes.setdefault(id(param), Envelope())

    def add_new_notes(self, specs):
        for s in specs:
            self._notes.append(Note(s.pitch, s.start_time, s.duration, s.velocity, s.mute))

    def get_notes_extended(self, from_pitch, pitch_span, from_time, time_span):
        return [n for n in self._notes if from_pitch <= n.pitch < from_pitch + pitch_span
                and from_time <= n.start_time < from_time + time_span]

    def remove_notes_extended(self, from_pitch, pitch_span, from_time, time_span):
        keep = set(map(id, self.get_notes_extended(from_pitch, pitch_span, from_time, time_span)))
        self._notes = [n for n in self._notes if id(n) not in keep]


RECORD_DIR = None  # set by tests: where "Live" writes recordings
SONG = None


def render_track(track, seconds):
    """Fake 'audio' for a track: two detuned saws through a low-pass set by its synth's cutoff."""
    import numpy as np
    import soundfile as sf
    sr = 44100
    t = np.arange(int(sr * seconds)) / sr
    cutoff_raw = 0.5
    for d in track.devices:
        for p in d.parameters:
            if p.name == "Filter Cutoff":
                cutoff_raw = p.value
    fc = 20.0 * (1000.0 ** cutoff_raw)
    f = 65.41
    saw = lambda fr: 2 * ((fr * t) % 1) - 1
    x = 0.3 * (saw(f * 1.007) + saw(f / 1.007))
    y = np.zeros_like(x)
    s1 = s2 = 0.0
    g = 1 - np.exp(-2 * np.pi * min(fc, 18000) / sr)
    for i in range(x.size):
        s1 += g * (x[i] - s1)
        s2 += g * (s1 - s2)
        y[i] = s2
    import os
    path = os.path.join(RECORD_DIR, "rec_%d.wav" % len(os.listdir(RECORD_DIR)))
    sf.write(path, np.stack([y, y], 1), sr)
    return path


class ClipSlot:
    def __init__(self, track=None):
        self.clip = None
        self.fired = 0
        self.track = track

    @property
    def has_clip(self):
        return self.clip is not None

    def create_clip(self, length):
        assert self.clip is None, "Live raises if the slot is not empty"
        self.clip = Clip(length)

    def create_audio_clip(self, path):
        assert self.clip is None
        self.clip = Clip(8.0, is_midi=False, name=path.replace("\\", "/").split("/")[-1])

    def delete_clip(self):
        self.clip = None

    def duplicate_clip_to(self, target):
        import copy
        assert target.clip is None
        target.clip = copy.deepcopy(self.clip)

    def fire(self, record_length=None, launch_quantization=None):
        self.fired += 1
        tr = self.track
        if self.clip is None and tr is not None and tr.arm and tr.has_audio_input:
            assert tr.input_routing_type.display_name == "Resampling", "fake only records resampling"
            assert tr.current_monitoring_state == 2, "monitoring should be Off"
            soloed = [t for t in SONG.tracks if t.solo]
            beats = record_length or 8.0
            c = Clip(beats, is_midi=False, name="recording")
            assert len(soloed) == 1, "exactly the source track should be soloed"
            c.file_path = render_track(soloed[0], beats * 60.0 / SONG.tempo)
            c.is_recording = False
            self.clip = c


class _AudioTrackMixin:
    pass


class Routing:
    def __init__(self, name):
        self.display_name = name


class Track:
    def __init__(self, kind, name, scenes):
        self.name = name
        self.has_midi_input = kind == "midi"
        self.has_audio_input = kind == "audio"
        self.is_foldable = False
        self.can_be_armed = True
        self.arm = self.mute = self.solo = False
        self.devices = []
        self.clip_slots = [ClipSlot(self) for _ in range(scenes)]
        self.available_input_routing_types = [Routing("Ext. In"), Routing("Resampling"), Routing("No Input")]
        self.input_routing_type = self.available_input_routing_types[0]
        self.current_monitoring_state = 1
        self.mixer_device = types.SimpleNamespace(volume=Param("Volume", 0.85, 0.0, 1.0))
        self.stopped = 0

    def stop_all_clips(self):
        self.stopped += 1

    def create_audio_clip(self, path, time):
        assert self.has_audio_input, "audio clips only on audio tracks"
        import os
        assert os.path.isfile(path)
        c = Clip(4.0, is_midi=False, name=os.path.basename(path))
        self.duplicate_clip_to_arrangement(c, time)
        return self.arrangement_clips[-1]

    def delete_clip(self, clip):
        self.arrangement_clips.remove(clip)

    def duplicate_clip_to_arrangement(self, clip, time):
        if not hasattr(self, "arrangement_clips"):
            self.arrangement_clips = []
        t, length = float(time), float(clip.length)
        # like Live: a new clip replaces whatever it covers on the timeline
        self.arrangement_clips = [c for c in self.arrangement_clips
                                  if c.end_time <= t or c.start_time >= t + length]
        ac = ArrClip(clip.name, t, length)
        import copy as _copy
        ac.envelopes = _copy.deepcopy(getattr(clip, "envelopes", {}))  # envelopes travel with clips
        if hasattr(clip, "pitch_coarse"):
            ac.pitch_coarse = clip.pitch_coarse
        self.arrangement_clips.append(ac)
        self.arrangement_clips.sort(key=lambda c: c.start_time)


class Envelope:
    def __init__(self):
        self.steps = []

    def insert_step(self, time, length, value):
        self.steps.append((time, length, value))

    def value_at(self, t):
        hit = [v for (a, ln, v) in self.steps if a <= t < a + ln]
        return hit[-1] if hit else None


class ArrClip:
    def __init__(self, name, start, length):
        self.name, self.start_time, self.end_time = name, start, start + length
        self.start_marker = 0.0
        self.envelopes = {}

    @property
    def length(self):
        return self.end_time - self.start_time

    def automation_envelope(self, param):
        return self.envelopes.get(id(param))

    def create_automation_envelope(self, param):
        raise RuntimeError("Not a session clip or parameter belongs to another track.")


class Pad:
    def __init__(self, note, name, used=True):
        self.note, self.name, self.chains = note, name, ([object()] if used else [])


class DrumRack:
    def __init__(self):
        self.name = "Drum Rack"
        self.class_name = "DrumGroupDevice"
        self.can_have_drum_pads = True
        self.parameters = [Param("Device On", 1.0)]
        names = {36: "Kick", 38: "Snare", 39: "Clap", 42: "Closed Hat"}
        self.drum_pads = [Pad(n, names.get(n, ""), n in names) for n in range(128)]


class Scene:
    def __init__(self):
        self.fired = 0
        self.name = ""

    def fire(self):
        self.fired += 1


class Song:
    def __init__(self, scenes=8):
        global SONG
        SONG = self
        self.scenes = [Scene() for _ in range(scenes)]
        self.tracks = []
        self.return_tracks = [types.SimpleNamespace(name="A-Reverb"), types.SimpleNamespace(name="B-Delay")]
        self.tempo = 120.0
        self.signature_numerator, self.signature_denominator = 4, 4
        self.is_playing = False
        self.view = types.SimpleNamespace(selected_track=None, highlighted_clip_slot=None)
        self.undo_count = 0
        self.cue_points = []
        self._song_time = 0.0
        self._pending_time = None
        self.back_to_arranger = False
        self.clip_trigger_quantization = 4
        self.loop, self.loop_start, self.loop_length = True, 0.0, 720.0
        self.punch_in = self.punch_out = self.record_mode = False
        self.master_track = Track("other", "Master", 0)
        self.master_track.has_audio_input = False
        self.master_track.can_be_armed = False

    @property
    def current_song_time(self):
        return self._song_time

    @current_song_time.setter
    def current_song_time(self, v):
        self._pending_time = float(v)  # applied on the next tick, like real Live

    def _tick(self):
        if self._pending_time is not None:
            self._song_time, self._pending_time = self._pending_time, None
        for t in self.tracks:
            for d in t.devices:
                for p in getattr(d, "parameters", []):
                    if hasattr(p, "refresh"):
                        p.refresh()

    def set_or_delete_cue(self):
        hit = [c for c in self.cue_points if c.time == self._song_time]
        if hit:
            self.cue_points.remove(hit[0])
        else:
            self.cue_points.append(types.SimpleNamespace(time=self._song_time,
                                                         name=str(len(self.cue_points) + 1)))

    def _new(self, kind, index):
        t = Track(kind, "%d-%s" % (len(self.tracks) + 1, "MIDI" if kind == "midi" else "Audio"),
                  len(self.scenes))
        if index == -1:
            self.tracks.append(t)
        else:
            self.tracks.insert(index, t)

    def create_midi_track(self, index):
        self._new("midi", index)

    def create_audio_track(self, index):
        self._new("audio", index)

    def delete_track(self, index):
        del self.tracks[index]

    def create_scene(self, index):
        self.scenes.append(Scene())
        for t in self.tracks:
            t.clip_slots.append(ClipSlot(t))

    def delete_scene(self, index):
        del self.scenes[index]
        for t in self.tracks:
            del t.clip_slots[index]

    def start_playing(self):
        self.is_playing = True
        if self.record_mode:  # Arrangement recording from the playhead
            armed = [t for t in self.tracks if getattr(t, "arm", False)]
            assert len(armed) == 1, "only Ghost's own track may be armed while recording"
            rec = armed[0]
            assert rec.input_routing_type.display_name == "Resampling"
            assert not self.back_to_arranger, "tracks must follow the Arrangement"
            soloed = [t for t in self.tracks if t.solo]
            src = soloed[0] if soloed else [t for t in self.tracks if t.has_midi_input and t.devices][0]
            bpb = 4.0 * self.signature_numerator / self.signature_denominator
            beats = bpb + 64  # plays on until Ghost stops it; record plenty
            ac = ArrClip("recording", self.current_song_time, beats)
            ac.file_path = render_track(src, beats * 60.0 / self.tempo)
            if not hasattr(rec, "arrangement_clips"):
                rec.arrangement_clips = []
            rec.arrangement_clips.append(ac)
            self.mix_recorded_from = self.current_song_time + bpb  # measured after the run-up
            self.recorded_solo = src.name if soloed else None
            self._song_time = self.current_song_time + beats  # it has played through

    def stop_playing(self):
        self.is_playing = False

    @property
    def can_undo(self):
        return True

    def undo(self):
        self.undo_count += 1


class Item:
    def __init__(self, name, children=(), loadable=True):
        self.name = name
        self.children = list(children)
        self.is_folder = bool(children)
        self.is_loadable = loadable and not children
        self.uri = "query:" + name


class Browser:
    def __init__(self, song):
        self.song = song
        F = lambda n, *c: Item(n, c)  # folder
        self.instruments = F("Instruments", F("Operator", Item("Operator")), Item("Wavetable"),
                             Item("Drift"), Item("Simpler"))
        self.audio_effects = F("Audio Effects", Item("Reverb"), Item("EQ Eight"),
                               Item("Compressor"), F("Delay", Item("Echo")))
        self.midi_effects = F("MIDI Effects", Item("Arpeggiator"), Item("Chord"))
        self.plugins = F("Plug-Ins", F("VST3", F("Xfer Records", Item("Serum")),
                                       F("FabFilter", Item("FabFilter Pro-Q 3")),
                                       F("Valhalla DSP", Item("ValhallaVintageVerb"))))
        self.drums = F("Drums", Item("808 Core Kit.adg"), Item("909 Core Kit.adg"))
        self.sounds = F("Sounds", F("Bass", Item("Dark Sub Bass.adv")))
        self.samples = F("Samples", Item("Kick 808.wav"))
        self.user_library = F("User Library", F("Samples", Item("my_vocal_chop.wav")))
        self.user_folders = [F("Samples", F("Drums", Item("Kick_Hard_01.wav")),
                               F("Loops", Item("Dark_Arp_Loop.wav")))]
        self.max_for_live = F("Max for Live")
        self.clips = F("Clips")
        self.packs = F("Packs")
        self.current_project = F("Current Project")
        self.loaded = []

    def load_item(self, item):
        track = self.song.view.selected_track
        assert track is not None, "Live loads onto the selected track"
        self.loaded.append(item.name)
        name = item.name
        if name.lower().endswith((".wav", ".aif", ".mp3")):
            if track.has_midi_input:
                track.devices.append(Device("Simpler"))
            else:
                slot = self.song.view.highlighted_clip_slot or track.clip_slots[0]
                slot.create_audio_clip(name)
        elif name == "Serum":
            track.devices.append(Device("Serum", synth_params(), class_name="PluginDevice"))
        elif name == "FabFilter Pro-Q 3":
            # a big plugin whose controls haven't been exposed with Configure yet
            track.devices.append(Device(name, ("Device On",), class_name="PluginDevice"))
        else:
            track.devices.append(Device(name.rsplit(".", 1)[0]))


class Application:
    def __init__(self, song):
        self.browser = Browser(song)


class CInstance:
    def __init__(self, scenes=8):
        self.song = Song(scenes)
        self.app = Application(self.song)
        self.log = []


def install_fake_modules():
    """Register fake 'Live' and '_Framework' modules before importing Ghost."""
    live = types.ModuleType("Live")

    class MidiNoteSpecification:
        def __init__(self, pitch, start_time, duration, velocity=100, mute=False):
            self.pitch, self.start_time, self.duration = pitch, start_time, duration
            self.velocity, self.mute = velocity, mute

    live.Clip = types.SimpleNamespace(MidiNoteSpecification=MidiNoteSpecification)
    sys.modules["Live"] = live

    fw = types.ModuleType("_Framework")
    cs_mod = types.ModuleType("_Framework.ControlSurface")

    class ControlSurface:
        def __init__(self, c_instance):
            self._c = c_instance

        def song(self):
            return self._c.song

        def application(self):
            return self._c.app

        def log_message(self, msg):
            self._c.log.append(msg)

        def show_message(self, msg):
            self._c.log.append("SHOW: " + msg)

        def update_display(self):
            self._c.song._tick()

        def disconnect(self):
            pass

    cs_mod.ControlSurface = ControlSurface
    fw.ControlSurface = cs_mod
    sys.modules["_Framework"] = fw
    sys.modules["_Framework.ControlSurface"] = cs_mod
