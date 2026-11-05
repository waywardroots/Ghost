"""
Ghost Remote Script - runs inside Ableton Live.

Listens on 127.0.0.1:9877 (this computer only) for newline-delimited JSON:
    {"id": 1, "command": "create_midi_track", "params": {"name": "Bass"}}
and replies with
    {"id": 1, "status": "ok", "result": {...}}   or   {"id": 1, "status": "error", "error": "..."}

Live's API may only be used from Live's main thread, so network threads
queue commands and Live runs them during update_display (about 10x a second).
"""
from __future__ import absolute_import, print_function, unicode_literals

import json
import socket
import threading
import traceback

try:
    import queue
except ImportError:  # pragma: no cover
    import Queue as queue

from _Framework.ControlSurface import ControlSurface

from .commands import CommandError, CommandHandler, VERSION

HOST = "127.0.0.1"
PORT = 9877
MAX_COMMANDS_PER_TICK = 20
REPLY_TIMEOUT = 20.0


class Ghost(ControlSurface):
    def __init__(self, c_instance):
        ControlSurface.__init__(self, c_instance)
        self._queue = queue.Queue()
        self._running = True
        self._server = None
        self._handler = CommandHandler(self.song(), self.application(), self.log_message)
        thread = threading.Thread(target=self._serve, name="GhostServer")
        thread.daemon = True
        thread.start()
        self.log_message("Ghost %s: started" % VERSION)
        self.show_message("Ghost assistant ready (port %d)" % PORT)

    # ---------------------------------------------------------------- main thread
    def update_display(self):
        ControlSurface.update_display(self)
        self.process_pending()

    def process_pending(self):
        for _ in range(MAX_COMMANDS_PER_TICK):
            try:
                command, params, reply = self._queue.get_nowait()
            except queue.Empty:
                return
            reply.put(self._run(command, params))

    def _run(self, command, params):
        try:
            return {"status": "ok", "result": self._handler.handle(command, params)}
        except CommandError as e:
            return {"status": "error", "error": str(e)}
        except TypeError as e:
            return {"status": "error", "error": "bad parameters for '%s': %s" % (command, e)}
        except Exception as e:
            self.log_message("Ghost error in %s:\n%s" % (command, traceback.format_exc()))
            return {"status": "error", "error": "%s: %s" % (type(e).__name__, e)}

    # ---------------------------------------------------------------- network threads
    def _serve(self):
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((HOST, PORT))
            srv.listen(5)
            srv.settimeout(1.0)
            self._server = srv
        except Exception:
            self.log_message("Ghost: could not open port %d:\n%s" % (PORT, traceback.format_exc()))
            self.show_message("Ghost: port %d is busy - is another copy running?" % PORT)
            return
        while self._running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except (OSError, socket.error):
                break
            t = threading.Thread(target=self._client, args=(conn,), name="GhostClient")
            t.daemon = True
            t.start()

    def _client(self, conn):
        conn.settimeout(1.0)
        buf = b""
        try:
            while self._running:
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                except (OSError, socket.error):
                    break
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        reply = self._dispatch(line)
                        conn.sendall((json.dumps(reply, default=str) + "\n").encode("utf-8"))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _dispatch(self, line):
        try:
            req = json.loads(line.decode("utf-8"))
            command, params = req.get("command"), req.get("params") or {}
        except (ValueError, AttributeError):
            return {"id": None, "status": "error", "error": "invalid JSON"}
        reply = queue.Queue(1)
        self._queue.put((command, params, reply))
        try:
            resp = reply.get(timeout=REPLY_TIMEOUT)
        except queue.Empty:
            resp = {"status": "error", "error": "Live did not respond in time (busy or a dialog is open)"}
        resp["id"] = req.get("id")
        return resp

    # ---------------------------------------------------------------- shutdown
    def disconnect(self):
        self._running = False
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
        self.log_message("Ghost: stopped")
        ControlSurface.disconnect(self)
