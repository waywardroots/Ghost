#!/usr/bin/env python3
"""
Ghost - Phase 2: assistant connector (MCP server).

Gives an AI assistant (Claude Desktop or any MCP client) tools to:
  - control Ableton Live through the Ghost Remote Script (127.0.0.1:9877)
  - search YOUR sample library, plugins and style profile built by indexer.py

Run by the assistant app (see README). To check the link to Live yourself:
    python mcp_server.py --check
"""
from __future__ import annotations

import itertools
import json
import os
import shutil
import socket
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indexer import app_dir, build_profile, normalize_key_query, open_db  # noqa: E402
import presets  # noqa: E402
import style  # noqa: E402

import listen  # noqa: E402

try:  # mcp 2.x
    from mcp.server.mcpserver import Image, MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image
from pydantic import BaseModel, Field  # installed with mcp

LIVE_HOST = os.environ.get("GHOST_LIVE_HOST", "127.0.0.1")
LIVE_PORT = int(os.environ.get("GHOST_LIVE_PORT", "9877"))
DB_PATH = Path(os.environ.get("GHOST_DB", str(app_dir() / "music_library.db")))

NOT_RUNNING = ("Can't reach Ableton Live. Make sure Live is open and 'Ghost' is selected as a "
               "Control Surface in Live's Settings > Link, Tempo & MIDI.")


# ----------------------------------------------------------------------------
# Connection to the Remote Script inside Live
# ----------------------------------------------------------------------------
class LiveError(RuntimeError):
    pass


class LiveConnection:
    def __init__(self, host: str, port: int, timeout: float = 25.0):
        self.host, self.port, self.timeout = host, port, timeout
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock, self._buf = None, b""

    def _connect(self):
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=3)
        except OSError as e:
            raise LiveError(NOT_RUNNING) from e
        self._sock.settimeout(self.timeout)
        self._buf = b""

    def _readline(self) -> bytes:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionResetError("Live closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def send(self, command: str, **params):
        params = {k: v for k, v in params.items() if v is not None}
        with self._lock:
            for attempt in (1, 2):
                if self._sock is None:
                    self._connect()
                rid = next(self._ids)
                msg = json.dumps({"id": rid, "command": command, "params": params}) + "\n"
                try:
                    self._sock.sendall(msg.encode("utf-8"))
                    resp = json.loads(self._readline().decode("utf-8"))
                    break
                except socket.timeout as e:
                    # Don't resend: Live may still carry out the command.
                    self._close()
                    raise LiveError("Live stopped responding (a dialog may be open in Live)") from e
                except (ConnectionError, OSError) as e:
                    # Stale connection (Live restarted): reconnect and try once more.
                    self._close()
                    if attempt == 2:
                        raise LiveError(NOT_RUNNING) from e
        if resp.get("status") != "ok":
            raise LiveError(resp.get("error") or "Live reported an unknown error")
        return resp.get("result")


live = LiveConnection(LIVE_HOST, LIVE_PORT)


# ----------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------
INSTRUCTIONS = """You are Ghost, a music production assistant working inside the user's Ableton Live.
Before writing music, call get_style_profile to learn the user's usual tempos, keys, plugins and
samples, and prefer their own plugins (list_my_plugins) and samples (search_samples).
Call get_session_info before editing so track and clip indexes are current. Indexes start at 0.
Clip slots are scene rows. Note timing is in beats: 1.0 = quarter note, 4.0 = one 4/4 bar.
MIDI pitch 60 = C3 in Live (middle C). Use load_device with exact names; if unsure, search_live_browser.
Make changes in small steps and tell the user what you changed; undo reverses Live's last change.

Sound design:
- Load the synth or effect with load_device, then get_device_parameters (use query to narrow
  big plugins). Set values with set_device_parameters, in real units ('800 Hz', '35 ms', '-6 dB',
  'Saw') rather than raw numbers when the parameter shows units.
- If a plugin shows a 'hint' about Configure, relay it: the user has to expose that plugin's
  controls to Live once before you can shape it.
- save_sound_snapshot before big changes and for each variation, so the user can compare and go back.
- For Serum 2 and u-he synths (Diva, Hive, Repro-1, Repro-5) you can also write complete presets:
  find_preset_files (Serum 2 and u-he; prefer an init patch or the user's own presets), read_preset_file to learn
  the layout, then write_preset_file. Ghost can't save presets for other plugins; tell the user
  to save from the plugin or with Live's Save Preset button when they like a sound.

Listening: you can't hear audio directly, but listen_to_track records a track and gives you
measurements plus a spectrogram image. Work in a loop: shape -> listen -> compare with the goal
(or a reference via analyze_audio_file / compare_to) -> adjust. Say what you measured in plain
words, and treat the user's ears as the final judge."""

server = _Server("ghost", instructions=INSTRUCTIONS)


def tool():
    """Register a tool. Ghost errors come back as {"error": ...} so every client shows the
    reason (some hide the text of failed tool calls)."""
    import functools
    import inspect

    def wrap(fn):
        if inspect.iscoroutinefunction(fn):
            return server.tool()(fn)

        @functools.wraps(fn)
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except (LiveError, presets.PresetError) as e:
                return {"error": str(e)}
            except Exception as e:  # never fail silently: say what broke
                import traceback
                where = traceback.extract_tb(e.__traceback__)[-1]
                return {"error": f"{type(e).__name__}: {e}",
                        "where": f"{Path(where.filename).name}:{where.lineno}"}
        return server.tool()(inner)
    return wrap


class Note(BaseModel):
    pitch: int = Field(description="MIDI pitch 0-127 (60 = C3 / middle C in Live, 36 = C1 kick)")
    start: float = Field(description="start time in beats from the clip start (1.0 = quarter note)")
    duration: float = Field(description="length in beats (0.25 = 16th note)")
    velocity: float = Field(100, description="1-127")
    mute: bool = False


# ---- session -----------------------------------------------------------------
@tool()
def get_session_info() -> dict:
    """Current Live set: tempo, time signature, play state, every track with its devices
    and which clip slots hold clips."""
    return live.send("get_session_info")


@tool()
def get_track_info(track_index: int) -> dict:
    """Details for one track: devices (with index) and clips (name, length, MIDI or audio)."""
    return live.send("get_track_info", track_index=track_index)


@tool()
def set_tempo(bpm: float) -> dict:
    """Set the song tempo in BPM."""
    return live.send("set_tempo", bpm=bpm)


@tool()
def set_time_signature(numerator: int, denominator: int) -> dict:
    """Set the song time signature, e.g. 4/4 or 6/8."""
    return live.send("set_time_signature", numerator=numerator, denominator=denominator)


@tool()
def start_playback() -> dict:
    """Start Live's transport."""
    return live.send("start_playback")


@tool()
def stop_playback() -> dict:
    """Stop Live's transport."""
    return live.send("stop_playback")


@tool()
def undo() -> dict:
    """Undo the last change in Live."""
    return live.send("undo")


# ---- tracks ------------------------------------------------------------------
@tool()
def create_midi_track(name: Optional[str] = None, index: int = -1) -> dict:
    """Add a MIDI track (index -1 = at the end). Returns its track_index."""
    return live.send("create_midi_track", name=name, index=index)


@tool()
def create_audio_track(name: Optional[str] = None, index: int = -1) -> dict:
    """Add an audio track (index -1 = at the end). Returns its track_index."""
    return live.send("create_audio_track", name=name, index=index)


@tool()
def set_track_name(track_index: int, name: str) -> dict:
    """Rename a track."""
    return live.send("set_track_name", track_index=track_index, name=name)


@tool()
def set_track_volume(track_index: int, volume: float) -> dict:
    """Set a track's fader, 0.0-1.0 (0.85 is about 0 dB)."""
    return live.send("set_track_volume", track_index=track_index, volume=volume)


@tool()
def delete_track(track_index: int) -> dict:
    """Delete a track. Only do this when the user asks."""
    return live.send("delete_track", track_index=track_index)


# ---- clips & notes -----------------------------------------------------------
@tool()
def create_clip(track_index: int, clip_index: int, length_beats: float = 4.0,
                name: Optional[str] = None, overwrite: bool = False) -> dict:
    """Create an empty MIDI clip in a MIDI track's clip slot (scene row). 4 beats = 1 bar of 4/4.
    New scenes are added if clip_index is past the last one."""
    return live.send("create_clip", track_index=track_index, clip_index=clip_index,
                     length=length_beats, name=name, overwrite=overwrite)


@tool()
def add_notes(track_index: int, clip_index: int, notes: list[Note], replace: bool = False) -> dict:
    """Write MIDI notes into a clip. replace=true clears the clip first."""
    return live.send("add_notes", track_index=track_index, clip_index=clip_index,
                     notes=[n.model_dump() for n in notes], replace=replace)


@tool()
def get_notes(track_index: int, clip_index: int) -> dict:
    """Read all MIDI notes in a clip (to analyze or build on the user's ideas)."""
    return live.send("get_notes", track_index=track_index, clip_index=clip_index)


@tool()
def set_clip_name(track_index: int, clip_index: int, name: str) -> dict:
    """Rename a clip."""
    return live.send("set_clip_name", track_index=track_index, clip_index=clip_index, name=name)


@tool()
def set_clip_loop(track_index: int, clip_index: int, loop_start: float, loop_end: float) -> dict:
    """Set a clip's loop region in beats."""
    return live.send("set_clip_loop", track_index=track_index, clip_index=clip_index,
                     loop_start=loop_start, loop_end=loop_end)


@tool()
def delete_clip(track_index: int, clip_index: int) -> dict:
    """Delete a clip. Only do this when the user asks."""
    return live.send("delete_clip", track_index=track_index, clip_index=clip_index)


@tool()
def fire_clip(track_index: int, clip_index: int) -> dict:
    """Launch a clip."""
    return live.send("fire_clip", track_index=track_index, clip_index=clip_index)


@tool()
def stop_track_clips(track_index: int) -> dict:
    """Stop all clips on a track."""
    return live.send("stop_track_clips", track_index=track_index)


@tool()
def fire_scene(scene_index: int) -> dict:
    """Launch a whole scene (row)."""
    return live.send("fire_scene", scene_index=scene_index)


# ---- scenes & arrangement -------------------------------------------------------
@tool()
def set_scene_name(scene_index: int, name: str) -> dict:
    """Name a Session scene (row), e.g. 'Drop' or 'Breakdown'."""
    return live.send("set_scene_name", scene_index=scene_index, name=name)


@tool()
def duplicate_clip(track_index: int, clip_index: int, target_clip_index: int,
                   overwrite: bool = False) -> dict:
    """Copy a clip to another slot on the same track (works for audio and MIDI clips), e.g. to
    make a variation without touching the original."""
    return live.send("duplicate_clip", track_index=track_index, clip_index=clip_index,
                     target_clip_index=target_clip_index, overwrite=overwrite)


class ClipRef(BaseModel):
    track_index: int
    clip_index: int = 0


class Section(BaseModel):
    name: str = Field(description="section name, used for the locator, e.g. 'Intro', 'Drop 1'")
    bars: int = Field(description="length in bars; must be a whole number of repeats of each clip")
    clips: list[ClipRef] = Field(description="the Session clips that play in this section")


@tool()
def get_arrangement_info() -> dict:
    """What's already in the Arrangement view: clips per track (start/end bar) and locators."""
    return live.send("get_arrangement_info")


@tool()
def clear_arrangement(track_indices: list[int], from_bar: float = 1,
                      to_bar: Optional[float] = None) -> dict:
    """Delete Arrangement clips on the listed tracks between from_bar and to_bar (exclusive;
    omit to_bar for 'to the end'). Session clips are kept. Use before rebuilding a structure;
    leave out tracks whose arrangement the user wants to keep. Ask the user to save first."""
    return live.send("clear_arrangement", track_indices=track_indices, from_bar=from_bar,
                     to_bar=to_bar)


class Move(BaseModel):
    from_bar: float = Field(description="first bar of the section as it is now")
    to_bar: float = Field(description="bar where it ends now (exclusive)")
    new_bar: float = Field(description="bar where it should start")


@tool()
def rearrange_arrangement(moves: list[Move], track_indices: Optional[list[int]] = None) -> dict:
    """Restructure a song: move whole sections of the Arrangement (all tracks, or the listed
    ones) with their clips and automation. Each move takes the clips that start in
    from_bar..to_bar and puts them at new_bar. A section may be used twice; bars no move
    covers are dropped. Nothing is deleted unless every copy works, and Ctrl+Z undoes it.
    Afterwards set new locators with set_locators(remove_existing=true)."""
    return live.send("rearrange", moves=[m.model_dump() for m in moves],
                     track_indices=track_indices)


@tool()
def read_project_arrangement(path: str) -> dict:
    """Map the arrangement of any of the user's Ableton sets (.als) without opening it: tempo,
    length, locators, every track's bar ranges, and an 8-bar energy map. Use it to model a new
    song's structure on one of their finished songs (find files with the user's project folders)."""
    import arrangement_map
    try:
        info = arrangement_map.read_arrangement(path)
    except (ValueError, OSError) as e:
        return {"error": str(e)}
    info["energy"] = arrangement_map.sections_by_density(info)
    return info


@tool()
def get_drum_pads(track_index: int) -> dict:
    """Which sound is on which MIDI note in a track's Drum Rack (e.g. to write a snare roll)."""
    return live.send("get_drum_pads", track_index=track_index)


@tool()
def place_clip(track_index: int, clip_index: int, bar: float, repeats: int = 1) -> dict:
    """Put a Session clip onto the Arrangement at a bar, repeated if asked. It replaces whatever
    is on that track there - use it to swap in variations or add fills."""
    return live.send("place_clip", track_index=track_index, clip_index=clip_index, bar=bar,
                     repeats=repeats)


@tool()
def place_audio_clip(track_index: int, path: str, bar: float, transpose: int = 0,
                     repeats: int = 1) -> dict:
    """Put an audio file (e.g. a riser, impact, vocal or loop found with search_samples) onto an
    audio track's Arrangement timeline at a bar. Risers usually END on the drop: start them
    their length before it. transpose: semitones (e.g. 2 to move a C loop to D).
    repeats: copies placed back to back (a loop across a whole section)."""
    return live.send("place_audio_clip", track_index=track_index, path=path, bar=bar,
                     transpose=transpose, repeats=repeats)


class AutoPoint(BaseModel):
    bar: float = Field(description="position in bars (1 = song start; 17.5 = halfway through bar 17)")
    value: float | str = Field(description="raw value, or display text like '200 Hz', '18 kHz', '-12 dB', '40 %'")


class AutoLane(BaseModel):
    device_index: int
    parameter: str
    points: list[AutoPoint]


@tool()
def automate(track_index: int, device_index: int, parameter: str, points: list[AutoPoint],
             clip_index: int = 0, step_beats: float = 0.25,
             lanes: list[AutoLane] | None = None) -> dict:
    """Draw automation (filter sweeps, volume rises, reverb throws) onto the Arrangement between
    the first and last point, joining points with straight lines. Live only allows drawing into
    Session clips, so Ghost re-places copies of the track's Session clip (clip_index) over that
    stretch with the automation inside - pick the clip that plays there. Add the device first
    (e.g. load_device 'Auto Filter'). End sweeps on the value the device should rest at,
    because the parameter stays there afterwards.
    lanes: more parameters on the same track to move in the same stretch (e.g. delay mix and
    feedback together). Re-placing clips can drop automation drawn earlier on them, so when
    several parameters move over the same bars, draw them in ONE call with lanes.
    replaced_clips/kept_automation report what was found and carried over."""
    return live.send("automate", track_index=track_index, device_index=device_index,
                     parameter=parameter, points=[p.model_dump() for p in points],
                     clip_index=clip_index, step_beats=step_beats,
                     lanes=[l.model_dump() for l in lanes] if lanes else None)


def _at_playhead(beat: float, **kw) -> dict:
    live.send("set_song_time", beat=beat)
    for _ in range(30):  # Live applies the new playhead position on its next update
        time.sleep(0.12)
        r = live.send("cue_at_playhead", expect_beat=beat, **kw)
        if r.get("ready"):
            return r
    raise LiveError(f"Live's playhead didn't move to beat {beat}; is a dialog open in Live?")


def _place_locators(items: list[tuple[float, str]], remove_existing: bool = False) -> list:
    if remove_existing:
        for cue in live.send("get_locators"):
            _at_playhead(cue["beat"], delete=True)
    for beat, name in items:
        r = _at_playhead(beat, name=name)
        if r.get("error"):
            raise LiveError(r["error"])
    live.send("set_song_time", beat=items[0][0] if items else 0.0)
    return live.send("get_locators")


class Locator(BaseModel):
    bar: float = Field(description="bar number where the locator goes (1 = start of the song)")
    name: str


@tool()
def set_locators(locators: list[Locator], remove_existing: bool = False) -> list:
    """Add or rename Arrangement locators (section markers). remove_existing=true first deletes
    every current locator - only do that when the user agrees or Ghost made them."""
    info = live.send("get_session_info")
    num, den = (int(x) for x in info["time_signature"].split("/"))
    bpb = num * 4.0 / den
    return _place_locators([((l.bar - 1) * bpb, l.name) for l in locators], remove_existing)


@tool()
def build_arrangement(sections: list[Section], start_bar: Optional[int] = None) -> dict:
    """Write a song structure into the Arrangement view: each section's Session clips are copied
    onto the timeline (repeated to fill the section) and a named locator marks each section.
    By default it starts after anything already in the Arrangement, so existing work is kept.
    Plan sections in multiples of 4 or 8 bars. Before running it, ask the user to save the set
    (each placed clip is its own undo step). Afterwards tell them to press Tab for Arrangement view."""
    r = live.send("build_arrangement", sections=[s.model_dump() for s in sections],
                  start_bar=start_bar)
    try:
        wanted = [(sec["start_beat"], sec["section"]) for sec in r["sections"]]
        lo = wanted[0][0]
        hi = lo + sum(sec["bars"] for sec in r["sections"]) * (wanted[1][0] - lo) / r["sections"][0]["bars"] \
            if len(wanted) > 1 else lo
        keep = {b for b, _ in wanted}
        # locators left inside the rebuilt range from an earlier layout would mislabel it
        for cue in live.send("get_locators"):
            if lo <= cue["beat"] < hi and cue["beat"] not in keep:
                _at_playhead(cue["beat"], delete=True)
        r["locators"] = _place_locators(wanted)
    except LiveError as e:
        r["locator_problem"] = str(e)
    return r


# ---- devices, plugins, samples -----------------------------------------------
@tool()
def search_live_browser(query: str, category: str = "all", limit: int = 25) -> list:
    """Search Live's browser for loadable items. category: all, plugins, instruments,
    audio_effects, midi_effects, drums, sounds, samples, user_library, user_folders, packs."""
    return live.send("search_browser", query=query, category=category, limit=limit)


@tool()
def load_device(track_index: int, name: str, kind: str = "any") -> dict:
    """Load an instrument, effect, VST/AU plugin, preset or drum kit onto a track by name.
    kind: any, plugin, instrument, audio_effect, midi_effect, drum_kit, preset.
    track_index -1 = the Master track (mastering chain)."""
    return live.send("load_device", track_index=track_index, name=name, kind=kind)


@tool()
def load_sample(track_index: int, path: str, clip_index: Optional[int] = None,
                overwrite: bool = False, transpose: int = 0) -> dict:
    """Load one of the user's samples (full path from search_samples). On an audio track with
    clip_index it becomes an audio clip; on a MIDI track it loads into a Simpler.
    transpose: semitones for that audio clip, e.g. to match a transposed loop on the timeline
    (a Session copy of an Arrangement loop lets automate draw fades and sweeps over it)."""
    return live.send("load_sample", track_index=track_index, path=path,
                     clip_index=clip_index, overwrite=overwrite, transpose=transpose)


@tool()
def get_device_parameters(track_index: int, device_index: int,
                          query: Optional[str] = None) -> dict:
    """List a device's parameters: raw value, what it shows (display, e.g. '800 Hz'), range,
    and choices for switches. query filters by name (e.g. 'filter', 'env', 'osc')."""
    return live.send("get_device_parameters", track_index=track_index,
                     device_index=device_index, query=query)


class ParamSetting(BaseModel):
    parameter: str = Field(description="parameter name (exact or unique part) or its index")
    value: float | str = Field(description="a number = raw Live value (min-max from "
                               "get_device_parameters); text = what the device shows, in real "
                               "units or names: '800 Hz', '1.2 kHz', '35 ms', '-6 dB', '40 %', "
                               "'-12 st', 'Saw', 'On'")


CALIBRATION_BUDGET_S = float(os.environ.get("GHOST_CALIBRATION_BUDGET", "30"))


@tool()
def set_device_parameters(track_index: int, device_index: int,
                          settings: list[ParamSetting]) -> dict:
    """Shape a sound: set many parameters on one device at once. Settings that fail are listed
    in 'errors' while the rest still apply. Each result shows the new display value."""
    started = time.monotonic()
    r = live.send("set_device_parameters", track_index=track_index, device_index=device_index,
                  settings=[s.model_dump() for s in settings])
    still = []
    for err in r.get("errors", []):
        if not str(err.get("error", "")).startswith("NEEDS_CALIBRATION"):
            still.append(err)
            continue
        want = next(s.value for s in settings if s.parameter == err["parameter"])
        if time.monotonic() - started > CALIBRATION_BUDGET_S:
            # Each calibration takes several seconds; stop before the caller's time limit
            # (about a minute) so nothing is left half-done.
            still.append({"parameter": err["parameter"], "error": "not set yet: this plugin is "
                          "set step by step and this call ran out of time; send it again"})
            continue
        try:
            r["set"].append(_calibrate(track_index, device_index, err["parameter"], want))
        except LiveError as e:
            still.append({"parameter": err["parameter"], "error": str(e)})
    r["errors"] = still
    return r


def _param_tools():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ghost_params", Path(__file__).resolve().parent / "ableton" / "Ghost" / "params.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _calibrate(track_index: int, device_index: int, parameter: str, want, settle: float = 0.15):
    """Reach a display value ('2 kHz', '3 s', 'N') on a plugin that only updates its display
    after Live refreshes: set a raw value, wait, read the display back, and narrow in."""
    pm = _param_tools()
    info = live.send("get_device_parameters", track_index=track_index, device_index=device_index)
    q = str(parameter).lower().strip()
    matches = [p for p in info["parameters"] if p["name"].lower().strip() == q] or \
        [p for p in info["parameters"] if q in p["name"].lower()]
    if not matches:
        raise LiveError(f"parameter '{parameter}' not found")
    p0 = matches[0]
    idx, lo, hi, original = p0["index"], p0["min"], p0["max"], p0["value"]

    def read():
        cur = live.send("get_device_parameters", track_index=track_index,
                        device_index=device_index)["parameters"]
        return (next(x["display"] for x in cur if x["index"] == idx) or "").strip()

    def is_raw(text, v):
        # Live shows the bare raw number ("0.25") when a plugin hasn't reported its own text,
        # e.g. right after the set is loaded.
        try:
            return abs(float(text) - float(v)) < 0.006
        except ValueError:
            return False

    def show(v, timeout=1.5):
        """Set v and return the display once it has caught up (same text twice in a row,
        and not the bare raw number)."""
        live.send("set_device_parameter", track_index=track_index, device_index=device_index,
                  parameter=str(idx), value=float(v))
        time.sleep(settle)
        last, waited = read(), settle
        while waited < timeout:
            time.sleep(0.08)
            waited += 0.08
            now = read()
            if now == last and not is_raw(now, v):
                return now
            last = now
        return last

    def done(v):
        return {"parameter": p0["name"], "value": round(v, 6), "display": show(v),
                "calibrated": True}

    text = str(want).strip()
    target = pm.parse_amount(text)
    try:
        if target is None:  # a named setting: step through the range
            norm = lambda x: " ".join(x.lower().split())
            for i in range(41):
                v = lo + (hi - lo) * i / 40
                if norm(show(v)) == norm(text):
                    out = done(v)
                    if norm(out["display"]) == norm(text):  # not a stale reading
                        return out
            raise LiveError(f"couldn't find a setting of {p0['name']} that shows '{text}'; "
                            f"use a raw value between {lo:g} and {hi:g}")
        amount, family = target
        tl, th = show(lo), show(hi)
        if family and (is_raw(tl, lo) or is_raw(th, hi)):
            raise LiveError(f"{p0['name']} only shows raw numbers right now ('{th}'), so I can't "
                            f"aim for '{text}'. Open the plugin window once, or use a raw value "
                            f"between {lo:g} and {hi:g}")
        a, b = pm.parse_amount(tl), pm.parse_amount(th)
        if a is None or b is None or a[0] == b[0]:
            raise LiveError(f"{p0['name']} doesn't show numbers I can steer by; use a raw value")
        shown = a[1] or b[1]
        if family and family != shown and not (family == "%" and shown is None):
            raise LiveError(f"{p0['name']} is shown in {shown or 'plain numbers'}, not {family}")
        up = b[0] > a[0]
        if amount <= min(a[0], b[0]):
            return done(lo if up else hi)
        if amount >= max(a[0], b[0]):
            return done(hi if up else lo)
        l, h = lo, hi
        for _ in range(12):  # 1/4096 of the range: finer than any display shows
            mid = (l + h) / 2
            m = pm.parse_amount(show(mid))
            if m is None:
                break
            if (m[0] < amount) == up:
                l = mid
            else:
                h = mid
        out = done((l + h) / 2)
        # Never keep a result that missed: a stale or odd display would otherwise leave the
        # sound somewhere unexpected.
        got = pm.parse_amount(out["display"])
        tol = max(0.15 * abs(amount), 0.002 if family == "s" else 1.0)
        if got is None or got[1] != shown or abs(got[0] - amount) > tol:
            raise LiveError(f"{p0['name']} ended up showing '{out['display']}' instead of "
                            f"'{text}', so I put it back; use a raw value")
        return out
    except LiveError:
        live.send("set_device_parameter", track_index=track_index, device_index=device_index,
                  parameter=str(idx), value=float(original))  # put it back
        raise


@tool()
def set_device_parameter(track_index: int, device_index: int, parameter: str,
                         value: float | str) -> dict:
    """Set one parameter. value: raw number, or display text like '800 Hz', '-6 dB', 'Saw'."""
    r = set_device_parameters.__wrapped__(track_index, device_index,
                                          [ParamSetting(parameter=parameter, value=value)])
    if r["errors"]:
        raise LiveError(r["errors"][0]["error"])
    out = dict(r["set"][0])
    out["device"] = r["device"]
    return out


# ---- sound snapshots (saved on disk, survive restarts) -------------------------
SNAPSHOT_FILE = Path(os.environ.get("GHOST_SNAPSHOTS", str(app_dir() / "snapshots.json")))


def _load_snaps() -> dict:
    try:
        return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_snaps(data: dict):
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(SNAPSHOT_FILE)


@tool()
def save_sound_snapshot(track_index: int, device_index: int, name: str) -> dict:
    """Save every setting of a device under a name, so you can try variations and come back.
    Use before big changes and for 'give me 3 versions' (save each as A, B, C)."""
    info = live.send("get_device_parameters", track_index=track_index, device_index=device_index)
    snaps = _load_snaps()
    snaps[name] = {"device": info["device"],
                   "values": {p["name"]: p["value"] for p in info["parameters"]
                              if p["name"] != "Device On"}}
    _save_snaps(snaps)
    return {"saved": name, "device": info["device"], "parameters": len(snaps[name]["values"])}


@tool()
def recall_sound_snapshot(track_index: int, device_index: int, name: str) -> dict:
    """Put a device back to a saved snapshot (matched by parameter name)."""
    snaps = _load_snaps()
    if name not in snaps:
        raise LiveError(f"no snapshot named '{name}'. Saved: {', '.join(snaps) or 'none'}")
    snap = snaps[name]
    current = live.send("get_device_parameters", track_index=track_index, device_index=device_index)
    names = {p["name"] for p in current["parameters"]}
    settings = [{"parameter": k, "value": v} for k, v in snap["values"].items() if k in names]
    if not settings:
        raise LiveError(f"snapshot '{name}' was taken on {snap['device']}, which shares no "
                        f"parameters with {current['device']}")
    r = live.send("set_device_parameters", track_index=track_index, device_index=device_index,
                  settings=settings)
    note = None if snap["device"] == current["device"] else \
        f"snapshot came from {snap['device']}, applied to {current['device']}"
    return {"recalled": name, "restored": len(r["set"]), "errors": r["errors"], "note": note}


@tool()
def list_sound_snapshots() -> list:
    """Saved sound snapshots and which device each came from."""
    return [{"name": k, "device": v["device"], "parameters": len(v["values"])}
            for k, v in _load_snaps().items()]


# ---- preset files (Serum 2, u-he Diva, Hive, Repro-1, Repro-5) ----------------
def _presets_call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except presets.PresetError as e:
        raise LiveError(str(e)) from e


@tool()
def find_preset_files(synth: str = "", query: str = "", limit: int = 40) -> list:
    """Find preset files Ghost can read and write: Serum 2 and u-he (Diva, Hive, Repro-1, Repro-5).
    query matches words in the name or folder (e.g. 'init', 'bass', 'pad'). The user's own
    presets are listed first."""
    return _presets_call(presets.find_presets, synth, query, limit)


@tool()
def read_preset_file(path: str) -> dict:
    """Read every setting in a Serum 2 or u-he preset (u-he: grouped by section; Serum 2: paths like
    Oscillator0/plainParams/kParamDetune). Learn the layout before writing a new one."""
    return _presets_call(presets.read_preset, path)


@tool()
def write_preset_file(template_path: str, name: str, changes: dict[str, float | str],
                      description: str = "") -> dict:
    """Make a new Serum 2 or u-he preset: copy template_path (an init patch, factory or user preset),
    change settings, and save it as a new preset in the synth's 'Ghost' folder. The template
    is never modified. changes keys: u-he 'Section:Param' (or 'Param' if unique); Serum 2 the full
    setting path. Use names
    and value scales exactly as read_preset_file shows them. The user then picks it in the
    synth's own preset browser (folder 'Ghost'; refresh the browser if needed)."""
    return _presets_call(presets.write_preset, template_path, name, changes, description)


# ---- listening ------------------------------------------------------------------
LISTEN_DIR = Path(os.environ.get("GHOST_LISTEN_DIR", str(app_dir() / "listening")))
_last_heard: dict = {}


def _wait_for_file(path: str, timeout: float = 8.0):
    p, last, stable, t0 = Path(path), -1, 0, time.time()
    while time.time() - t0 < timeout:
        size = p.stat().st_size if p.exists() else -1
        if size > 1000 and size == last:
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        last = size
        time.sleep(0.25)
    if not p.exists():
        raise LiveError(f"Live said it recorded to {path}, but that file doesn't exist")


def _report(path: str, title: str, compare_to: Optional[str]):
    m = listen.analyze(path)
    result = {"heard": m.get("summary"), "measurements": {k: v for k, v in m.items() if k != "summary"},
              "audio_file": path}
    if compare_to:
        ref_path = _last_heard.get("path") if compare_to == "last" else compare_to
        if not ref_path or not Path(ref_path).exists():
            raise LiveError("nothing to compare with - give a file path, or listen once first for 'last'")
        ref = listen.analyze(ref_path)
        result["compared_with"] = ref_path
        result["reference_heard"] = ref.get("summary")
        result["differences"] = listen.compare(m, ref)["differences"]
    out = [json.dumps(result, indent=1, default=str)]
    if not m.get("silent"):
        png = listen.spectrogram(path, str(Path(path).with_suffix(".png")), title)
        out.append(Image(path=png))
    _last_heard.update(path=path)
    return out


DEFAULT_PHRASE = [  # one long note (tone, movement, sustain), then short ones (attack, release)
    Note(pitch=48, start=0.0, duration=4.0, velocity=100),
    Note(pitch=48, start=4.5, duration=0.5, velocity=110),
    Note(pitch=55, start=5.5, duration=0.5, velocity=100),
    Note(pitch=60, start=6.5, duration=1.0, velocity=100),
]


@tool()
def listen_to_track(track_index: int, notes: Optional[list[Note]] = None,
                    clip_index: Optional[int] = None, beats: float = 8.0,
                    compare_to: Optional[str] = None, bar: Optional[float] = None):
    """Hear a track: Ghost plays a phrase on it (soloed), records the result inside Live through
    Resampling, measures it (low end per band, brightness, pitch, grit, movement/beating, stereo
    width, attack/sustain) and returns a spectrogram picture. Use after shaping a sound to check
    it. notes: the phrase to play (default: one long C2 then three short notes; use low notes
    like 36 for bass, chords for pads). clip_index: play an existing clip instead.
    bar: play the Arrangement from that bar with the track soloed instead - works for audio
    tracks without Session clips and for GROUPS, which the user treats as one sound (a group
    and its member tracks are mixed and judged together; adjust the group, not one member).
    compare_to: a reference audio file path (e.g. from search_samples) or 'last' for the
    previous listen. Note: this stops Live's playback while it records."""
    if bar is not None:
        bpb = _beats_per_bar()
        _settle_playhead(bar, bpb)
        try:
            start = live.send("start_audition", track_index=track_index, beats=beats,
                              arrangement=True)
            _check_start(start, bar)
            _arm_recording()
        except LiveError:
            live.send("end_audition")
            raise
        return _finish_listen(start, f"track{track_index}_bar{bar:g}",
                              f"Track {track_index} from bar {bar:g}", compare_to)
    if clip_index is None and not notes:
        notes = DEFAULT_PHRASE
    start = live.send("start_audition", track_index=track_index, beats=beats,
                      notes=[n.model_dump() for n in notes] if notes else None, clip_index=clip_index)
    return _finish_listen(start, f"track{track_index}", f"Track {track_index}", compare_to)


@tool()
def listen_to_mix(bar: float = 2, bars: int = 8, compare_to: Optional[str] = None):
    """Hear the whole song: Ghost plays the Arrangement from a bar (nothing soloed), records the
    master output through Resampling and measures it like listen_to_track, plus loudness
    (LUFS) and peak. Use it to mix and master: check a drop, a breakdown, the low end of the
    full mix. bars: up to 16. compare_to: a reference track path (e.g. from
    find_reference_tracks) or 'last'. Stops Live's playback while it records."""
    bars = max(1, min(16, int(bars)))
    bpb = _beats_per_bar()
    _settle_playhead(bar, bpb)
    try:
        start = live.send("start_audition", track_index=-1, beats=bars * bpb)
        _check_start(start, bar)
        _arm_recording()
    except LiveError:
        live.send("end_audition")
        raise
    return _finish_listen(start, f"mix_bar{bar:g}", f"Mix from bar {bar:g}", compare_to,
                          loudness=True)


def _beats_per_bar() -> float:
    info = live.send("get_session_info")
    num, den = (float(x) for x in str(info.get("time_signature", "4/4")).split("/"))
    return num * 4.0 / den  # Live counts quarter-note beats


def _settle_playhead(bar: float, bpb: float) -> None:
    """Put the playhead one bar before `bar` (Ghost presses Play there and records from the
    next downbeat) and wait until Live has really put it there."""
    if float(bar) < 2:
        raise LiveError("start from bar 2 or later (Ghost needs one bar of run-up)")
    live.send("stop_playback")
    live.send("prepare_audition")  # loop off while recording (restored afterwards)
    for _ in range(20):
        time.sleep(0.1)
        if not live.send("get_loop")["loop"]:
            break
    target = (float(bar) - 2) * bpb
    settled = 0
    for _ in range(40):  # Live moves the playhead on a later update; make sure it stays there
        if settled == 0:
            live.send("set_song_time", beat=target)
        time.sleep(0.12)
        now = live.send("get_song_time")["beat"]
        settled = settled + 1 if abs(now - target) < 1e-3 else 0
        if settled >= 3:
            return
    raise LiveError(f"Live's playhead didn't stay at bar {bar:g}; is a dialog open in Live?")


def _check_start(start: dict, bar: float) -> None:
    if abs(float(start.get("from_bar", bar)) - float(bar)) > 1e-3:
        raise LiveError(f"playback started at bar {start.get('from_bar')} instead of {bar:g}")


def _arm_recording() -> None:
    """Live plays, then jumps to the run-up bar; switch record on once it has landed there."""
    for _ in range(40):
        if live.send("arm_audition_record").get("armed"):
            return
        time.sleep(0.05)
    raise LiveError("Live's playhead didn't jump to the run-up bar")


def _finish_listen(start: dict, tag: str, title: str, compare_to: Optional[str],
                   loudness: bool = False):
    try:
        time.sleep(start["wait_seconds"])
        for _ in range(40):
            r = live.send("finish_audition")
            if r.get("done"):
                break
            time.sleep(0.5)
        else:
            where = f" (playhead at beat {r.get('beat')}, waiting for {r.get('until')})" \
                if r.get("beat") is not None else ""
            raise LiveError("the recording didn't finish - is Live's transport running?" + where)
        if not r.get("file"):
            raise LiveError(r.get("error") or "Live didn't report a recorded file")
        _wait_for_file(r["file"])
    finally:
        try:
            live.send("end_audition")  # removes Ghost's track, so Live lets go of the file
        except LiveError:
            pass
    dest = _copy_recording(r["file"], tag)
    if r.get("trim_seconds") is not None:  # Arrangement recordings include the run-up
        dest = _trim_recording(dest, r["trim_seconds"], r.get("keep_seconds"), tag)
    out = _report(str(dest), title, compare_to)
    if loudness:
        out = _add_loudness(out, str(dest))
    return out


def _copy_recording(src: str, tag: str, timeout: float = 15.0) -> Path:
    """Copy Live's new recording into Ghost's folder. On Windows Live can hold the file open
    for a few seconds after recording, so keep trying until it can be read."""
    import soundfile as sf
    LISTEN_DIR.mkdir(parents=True, exist_ok=True)
    dest = LISTEN_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}.wav"
    t0, err = time.time(), None
    while time.time() - t0 < timeout:
        try:
            shutil.copyfile(src, dest)
            sf.info(str(dest))  # a complete WAV header, not a half-written file
            return dest
        except (OSError, RuntimeError) as e:  # LibsndfileError is a RuntimeError
            err = e
            time.sleep(0.5)
    raise LiveError(f"Live recorded {src}, but Ghost couldn't read it after "
                    f"{timeout:.0f} seconds ({err})")


def _trim_recording(path: Path, skip: float, keep: Optional[float], tag: str) -> Path:
    """Cut the run-up off an Arrangement recording (and anything past the measured bars)."""
    import soundfile as sf
    y, sr = sf.read(str(path), always_2d=True)
    a = int(round(skip * sr))
    b = len(y) if not keep else min(len(y), a + int(round(keep * sr)))
    if a >= b:
        raise LiveError("the recording was shorter than expected - nothing left after the run-up")
    LISTEN_DIR.mkdir(parents=True, exist_ok=True)
    out = LISTEN_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}_trimmed.wav"
    sf.write(str(out), y[a:b], sr, subtype="PCM_24")
    return out


def _add_loudness(out, path: str):
    """Add integrated loudness (LUFS, ITU-R BS.1770) and true-ish peak to a listen report."""
    try:
        import json as _json
        import numpy as np
        import soundfile as sf
        y, sr = sf.read(path, always_2d=True)
        lufs = _lufs(y, sr)
        peak = 20 * np.log10(max(float(np.max(np.abs(y))), 1e-9))
        data = _json.loads(out[0])
        data.setdefault("measurements", {})["lufs_integrated"] = lufs
        data["measurements"]["sample_peak_db"] = round(peak, 2)
        data.setdefault("heard", []).append(
            f"Loudness: {lufs} LUFS integrated, peak {peak:.1f} dBFS.")
        return [_json.dumps(data, indent=1)] + list(out[1:])
    except Exception as e:  # loudness is a bonus; never fail the listen over it
        import sys
        print("loudness measurement failed: %s" % e, file=sys.stderr)
        return out


def _lufs(y, sr: int):
    """Integrated loudness per ITU-R BS.1770-4 (K-weighting, 400 ms blocks, gating)."""
    import numpy as np
    from scipy.signal import lfilter
    import math
    # K-weighting: high shelf then high-pass (coefficients derived for any sample rate)
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = math.tan(math.pi * f0 / sr)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = math.tan(math.pi * f0 / sr)
    a2 = [1, 2 * (K * K - 1) / (1 + K / Q + K * K), (1 - K / Q + K * K) / (1 + K / Q + K * K)]
    b2 = [1, -2, 1]
    z = lfilter(b2, a2, lfilter(b1, a1, y, axis=0), axis=0)
    block, step = int(0.4 * sr), int(0.1 * sr)
    if z.shape[0] < block:
        return None
    ms = np.array([np.mean(z[i:i + block] ** 2, axis=0).sum()
                   for i in range(0, z.shape[0] - block + 1, step)])
    lk = -0.691 + 10 * np.log10(np.maximum(ms, 1e-12))
    gated = ms[lk > -70]
    if not gated.size:
        return None
    rel = -0.691 + 10 * np.log10(gated.mean()) - 10
    final = ms[(lk > -70) & (lk > rel)]
    return round(float(-0.691 + 10 * np.log10(final.mean())), 1)


@tool()
def analyze_audio_file(path: str, compare_to: Optional[str] = None):
    """Hear any audio file: a sample from the user's library (search_samples), a bounce, or a
    reference. Returns the same measurements and spectrogram as listen_to_track. Use it on a
    reference sound first, then listen_to_track(..., compare_to=<that path>) to match it."""
    if not Path(path).is_file():
        raise LiveError(f"file not found: {path}")
    return _report(path, Path(path).name, compare_to)


# ---- the user's own library (from indexer.py) ----------------------------------
def _db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise LiveError("Your library hasn't been indexed yet. Run: python indexer.py scan "
                        "--samples <folders> --projects <folders>")
    return open_db(DB_PATH)


@tool()
def search_samples(query: str = "", category: Optional[str] = None, key: Optional[str] = None,
                   bpm: Optional[float] = None, bpm_tolerance: float = 2.0,
                   loops_only: bool = False, one_shots_only: bool = False,
                   limit: int = 20) -> list:
    """Search the user's own sample library. category: kick, snare, clap, hihat, cymbal, perc,
    drums, 808, bass, vocal, fx, pad, keys, guitar, strings, brass, lead, chords, melody, synth.
    key like 'Am' or 'F# minor'. Returns full paths for load_sample."""
    sql, args = "SELECT * FROM samples WHERE error IS NULL", []
    for w in query.split():
        sql += " AND lower(path) LIKE ?"
        args.append(f"%{w.lower()}%")
    if category:
        sql += " AND category=?"
        args.append(category.lower())
    if key:
        k = normalize_key_query(key)
        if not k:
            raise LiveError(f"couldn't read key '{key}'; try 'Am' or 'F# minor'")
        sql += " AND musical_key=?"
        args.append(k)
    if bpm:
        sql += " AND bpm BETWEEN ? AND ?"
        args += [bpm - bpm_tolerance, bpm + bpm_tolerance]
    if loops_only:
        sql += " AND is_loop=1"
    if one_shots_only:
        sql += " AND is_loop=0"
    sql += " ORDER BY RANDOM() LIMIT ?"  # variety instead of always the same files
    args.append(max(1, min(int(limit), 100)))
    con = _db()
    return [{"name": r["name"], "path": r["path"], "category": r["category"], "bpm": r["bpm"],
             "key": r["musical_key"], "loop": bool(r["is_loop"]), "seconds": r["duration"]}
            for r in con.execute(sql, args)]


@tool()
def list_my_plugins(query: str = "", limit: int = 60) -> list:
    """The user's installed plugins, most-used first (usage = number of their projects)."""
    con = _db()
    usage = dict(con.execute(
        "SELECT lower(plugin), COUNT(DISTINCT project) FROM project_plugins GROUP BY lower(plugin)"))
    rows = con.execute("SELECT name, format, vendor FROM plugins WHERE lower(name) LIKE ?",
                       (f"%{query.lower()}%",)).fetchall()
    out = [{"name": r["name"], "format": r["format"], "vendor": r["vendor"],
            "used_in_projects": usage.get(r["name"].lower(), 0)} for r in rows]
    out.sort(key=lambda p: (-p["used_in_projects"], p["name"].lower()))
    return out[: max(1, min(int(limit), 300))]


@tool()
def get_style_profile() -> dict:
    """How the user makes music, learned from their own projects: usual tempos and keys,
    favourite plugins and Live devices per track type, go-to samples."""
    return build_profile(_db())


# ---- the music the user owns (reference_scan.py) ----------------------------
def _ref_db():
    import reference_scan
    if not DB_PATH.exists():
        raise LiveError("no library database yet: run 'py -3.13 reference_scan.py' in the Ghost folder")
    return reference_scan.open_db(str(DB_PATH)), reference_scan


@tool()
def find_reference_tracks(query: Optional[str] = None, genre: Optional[str] = None,
                          key: Optional[str] = None, tempo: Optional[float] = None,
                          yours: Optional[bool] = None, limit: int = 15) -> list[dict]:
    """Search the finished tracks the user owns (their DJ library and their own releases) to
    use as references: query matches artist/title, genre matches the folder (e.g. 'melodic
    techno'), key/tempo rank the closest first, yours=true for their own released tracks.
    Each result has the arrangement (intro, sections by bar) and mix numbers; pass its path
    to analyze_audio_file or as compare_to in listen_to_track to match a sound or mix."""
    con, rs = _ref_db()
    q = "SELECT * FROM reference_tracks WHERE error IS NULL"
    args = []
    if query:
        q += " AND (lower(artist) LIKE ? OR lower(title) LIKE ? OR lower(path) LIKE ?)"
        args += [f"%{query.lower()}%"] * 3
    if genre:
        q += " AND lower(genre) LIKE ?"
        args.append(f"%{genre.lower()}%")
    if yours is not None:
        q += " AND yours=?"
        args.append(int(yours))
    rows = con.execute(q, args).fetchall()
    want = rs.normalize_key(key) if key else None

    def score(r):
        s = 0.0
        if want and r["musical_key"]:
            s += 2 if r["musical_key"] == want else (0.5 if r["musical_key"].split()[1] == want.split()[1] else 0)
        if tempo and r["bpm"]:
            s -= abs(r["bpm"] - tempo) / 4
        return s
    rows = sorted(rows, key=score, reverse=True)[: max(1, min(int(limit), 100))]
    out = []
    for r in rows:
        out.append({"path": r["path"], "artist": r["artist"], "title": r["title"],
                    "genre": r["genre"], "yours": bool(r["yours"]), "bpm": r["bpm"],
                    "key": r["musical_key"], "minutes": round((r["seconds"] or 0) / 60, 1),
                    "bars": r["bars"], "intro_bars": r["intro_bars"],
                    "sections": json.loads(r["sections"] or "[]"),
                    "mix": {"loudness_db": r["loudness_db"], "crest_db": r["crest_db"],
                            "bands_percent": json.loads(r["bands"] or "{}"),
                            "brightness_hz": r["brightness_hz"],
                            "stereo_width": r["stereo_width"], "low_end_width": r["low_end_width"]}})
    return out


@tool()
def get_reference_profile(genre: Optional[str] = None, yours: Optional[bool] = None) -> dict:
    """What the user's reference music has in common (medians): tempo, length, keys, how long
    intros run, how many breakdowns and where the first one lands, and the mix (loudness, band
    balance, width). Narrow to a genre folder, or yours=true for their own releases. Use it to
    set arrangement and mix targets for a new track."""
    con, rs = _ref_db()
    return rs.profile(con, genre=genre, yours=yours)


# ---- your-style patterns (Phase 3) ------------------------------------------
STYLE_LIB = style.Library(Path(os.environ.get("GHOST_MIDI_DATASET",
                                              str(app_dir() / "midi_dataset.jsonl"))))


@tool()
def find_my_patterns(role: str, key: Optional[str] = None, tempo: Optional[float] = None,
                     bars: Optional[float] = None, instrument: Optional[str] = None,
                     query: Optional[str] = None, exclude: Optional[str] = None,
                     limit: int = 10) -> list[dict]:
    """Search the parts the user wrote in their own projects (built by midi_dataset.py).
    role: drums, bass, chords, pad, lead, arp or fx. key/tempo/bars rank the closest fits
    first (pitched parts in other keys still match and say how far to transpose).
    instrument filters by plugin (e.g. 'Diva'); query by project/track/clip name; exclude
    leaves out projects whose name contains it (e.g. the song being worked on).
    Use the returned id with write_my_pattern."""
    return STYLE_LIB.find(role, key=key, tempo=tempo, bars=bars, instrument=instrument,
                          query=query, exclude=exclude, limit=max(1, min(int(limit), 50)))


@tool()
def write_my_pattern(track_index: int, clip_index: int, pattern_id: str,
                     key: Optional[str] = None, bars: Optional[float] = None,
                     variation: float = 0.3, octave: int = 0, seed: Optional[int] = None,
                     name: Optional[str] = None, overwrite: bool = False) -> dict:
    """Write a new clip in the user's style from one of their own patterns (find_my_patterns):
    moved into key (pitched parts; drums stay as they are), fitted to bars, and varied by
    variation (0 = exact copy, 0.3 = a fresh take, 1 = loose). octave shifts by octaves.
    A different seed gives a different take. overwrite replaces a clip already in the slot."""
    c = STYLE_LIB.get(pattern_id)
    notes, info = style.make_part(c, key=key, bars=bars, variation=variation, seed=seed,
                                  octave=octave, beats_per_bar=4)
    length = info["bars"] * 4
    live.send("create_clip", track_index=track_index, clip_index=clip_index, length=length,
              name=name or f"{c['role'].title()} ({style.project_name(c['project'])})",
              overwrite=overwrite)
    live.send("add_notes", track_index=track_index, clip_index=clip_index, notes=notes,
              replace=True)
    info.update(track_index=track_index, clip_index=clip_index)
    return info


# ----------------------------------------------------------------------------
def check():
    print(f"Connecting to Ableton Live on {LIVE_HOST}:{LIVE_PORT} ...")
    try:
        print("Ghost Remote Script version:", live.send("ping")["ghost"])
        info = live.send("get_session_info")
        print(f"Connected. Tempo {info['tempo']} BPM, {len(info['tracks'])} tracks.")
    except LiveError as e:
        print("Not connected:", e)
    print("Library:", DB_PATH if DB_PATH.exists() else "not indexed yet (run indexer.py scan)")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        server.run()
