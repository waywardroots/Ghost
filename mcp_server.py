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
tool = server.tool


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


# ---- devices, plugins, samples -----------------------------------------------
@tool()
def search_live_browser(query: str, category: str = "all", limit: int = 25) -> list:
    """Search Live's browser for loadable items. category: all, plugins, instruments,
    audio_effects, midi_effects, drums, sounds, samples, user_library, user_folders, packs."""
    return live.send("search_browser", query=query, category=category, limit=limit)


@tool()
def load_device(track_index: int, name: str, kind: str = "any") -> dict:
    """Load an instrument, effect, VST/AU plugin, preset or drum kit onto a track by name.
    kind: any, plugin, instrument, audio_effect, midi_effect, drum_kit, preset."""
    return live.send("load_device", track_index=track_index, name=name, kind=kind)


@tool()
def load_sample(track_index: int, path: str, clip_index: Optional[int] = None,
                overwrite: bool = False) -> dict:
    """Load one of the user's samples (full path from search_samples). On an audio track with
    clip_index it becomes an audio clip; on a MIDI track it loads into a Simpler."""
    return live.send("load_sample", track_index=track_index, path=path,
                     clip_index=clip_index, overwrite=overwrite)


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


@tool()
def set_device_parameters(track_index: int, device_index: int,
                          settings: list[ParamSetting]) -> dict:
    """Shape a sound: set many parameters on one device at once. Settings that fail are listed
    in 'errors' while the rest still apply. Each result shows the new display value."""
    return live.send("set_device_parameters", track_index=track_index, device_index=device_index,
                     settings=[s.model_dump() for s in settings])


@tool()
def set_device_parameter(track_index: int, device_index: int, parameter: str,
                         value: float | str) -> dict:
    """Set one parameter. value: raw number, or display text like '800 Hz', '-6 dB', 'Saw'."""
    return live.send("set_device_parameter", track_index=track_index,
                     device_index=device_index, parameter=parameter, value=value)


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
                    compare_to: Optional[str] = None):
    """Hear a track: Ghost plays a phrase on it (soloed), records the result inside Live through
    Resampling, measures it (low end per band, brightness, pitch, grit, movement/beating, stereo
    width, attack/sustain) and returns a spectrogram picture. Use after shaping a sound to check
    it. notes: the phrase to play (default: one long C2 then three short notes; use low notes
    like 36 for bass, chords for pads). clip_index: play an existing clip instead.
    compare_to: a reference audio file path (e.g. from search_samples) or 'last' for the
    previous listen. Note: this stops Live's playback while it records."""
    if clip_index is None and not notes:
        notes = DEFAULT_PHRASE
    start = live.send("start_audition", track_index=track_index, beats=beats,
                      notes=[n.model_dump() for n in notes] if notes else None, clip_index=clip_index)
    try:
        time.sleep(start["wait_seconds"])
        for _ in range(40):
            r = live.send("finish_audition")
            if r.get("done"):
                break
            time.sleep(0.5)
        else:
            raise LiveError("the recording didn't finish - is Live's transport running?")
        if not r.get("file"):
            raise LiveError(r.get("error") or "Live didn't report a recorded file")
        _wait_for_file(r["file"])
        LISTEN_DIR.mkdir(parents=True, exist_ok=True)
        dest = LISTEN_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_track{track_index}.wav"
        shutil.copy2(r["file"], dest)
    finally:
        try:
            live.send("end_audition")
        except LiveError:
            pass
    return _report(str(dest), f"Track {track_index}", compare_to)


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
