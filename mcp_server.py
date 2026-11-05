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
import socket
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indexer import app_dir, build_profile, normalize_key_query, open_db  # noqa: E402

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
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
Make changes in small steps and tell the user what you changed; undo reverses Live's last change."""

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
def get_device_parameters(track_index: int, device_index: int) -> dict:
    """List a device's parameters with current values and ranges."""
    return live.send("get_device_parameters", track_index=track_index, device_index=device_index)


@tool()
def set_device_parameter(track_index: int, device_index: int, parameter: str,
                         value: float) -> dict:
    """Set a device parameter by name (or index as text), clamped to its range."""
    p: object = int(parameter) if str(parameter).isdigit() else parameter
    return live.send("set_device_parameter", track_index=track_index,
                     device_index=device_index, parameter=p, value=value)


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
