"""A small stand-in for Ableton Live's Python API so Ghost can be tested without Live."""
import sys
import types


class Param:
    def __init__(self, name, value=0.0, mn=0.0, mx=1.0):
        self.name, self.value, self.min, self.max = name, value, mn, mx
        self.is_quantized = False


class Device:
    def __init__(self, name, params=("Device On", "Cutoff", "Resonance")):
        self.name = name
        self.class_name = name.replace(" ", "")
        self.parameters = [Param(p) for p in params]


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

    def add_new_notes(self, specs):
        for s in specs:
            self._notes.append(Note(s.pitch, s.start_time, s.duration, s.velocity, s.mute))

    def get_notes_extended(self, from_pitch, pitch_span, from_time, time_span):
        return [n for n in self._notes if from_pitch <= n.pitch < from_pitch + pitch_span
                and from_time <= n.start_time < from_time + time_span]

    def remove_notes_extended(self, from_pitch, pitch_span, from_time, time_span):
        keep = set(map(id, self.get_notes_extended(from_pitch, pitch_span, from_time, time_span)))
        self._notes = [n for n in self._notes if id(n) not in keep]


class ClipSlot:
    def __init__(self):
        self.clip = None
        self.fired = 0

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

    def fire(self):
        self.fired += 1


class Track:
    def __init__(self, kind, name, scenes):
        self.name = name
        self.has_midi_input = kind == "midi"
        self.has_audio_input = kind == "audio"
        self.is_foldable = False
        self.can_be_armed = True
        self.arm = self.mute = self.solo = False
        self.devices = []
        self.clip_slots = [ClipSlot() for _ in range(scenes)]
        self.mixer_device = types.SimpleNamespace(volume=Param("Volume", 0.85, 0.0, 1.0))
        self.stopped = 0

    def stop_all_clips(self):
        self.stopped += 1


class Scene:
    def __init__(self):
        self.fired = 0

    def fire(self):
        self.fired += 1


class Song:
    def __init__(self, scenes=8):
        self.scenes = [Scene() for _ in range(scenes)]
        self.tracks = []
        self.return_tracks = [types.SimpleNamespace(name="A-Reverb"), types.SimpleNamespace(name="B-Delay")]
        self.tempo = 120.0
        self.signature_numerator, self.signature_denominator = 4, 4
        self.is_playing = False
        self.view = types.SimpleNamespace(selected_track=None, highlighted_clip_slot=None)
        self.undo_count = 0

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
            t.clip_slots.append(ClipSlot())

    def start_playing(self):
        self.is_playing = True

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
            pass

        def disconnect(self):
            pass

    cs_mod.ControlSurface = ControlSurface
    fw.ControlSurface = cs_mod
    sys.modules["_Framework"] = fw
    sys.modules["_Framework.ControlSurface"] = cs_mod
