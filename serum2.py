"""
Serum 2 preset files (.SerumPreset).

Layout (worked out from the user's own Serum 2 install):
    b"XferJson\\0"                     9-byte magic
    <uint64 LE>  n                     length of the JSON header
    n bytes      JSON                  presetName, presetAuthor, tags, ..., "hash"
    <uint32 LE>  size                  length of the decompressed body
    <uint32 LE>  2                     format flag
    zstd(CBOR body)                    every setting; "hash" = md5 of these compressed bytes

The CBOR codec below keeps each number in the width Serum wrote it (32- or 64-bit
float, etc.) so an unchanged preset re-encodes to the exact same bytes.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct

MAGIC = b"XferJson\x00"
FLAG = 2


class SerumError(Exception):
    pass


# ---------------------------------------------------------------------------- CBOR
class F16(float):
    pass


class F32(float):
    pass


class F64(float):
    pass


class _Reader:
    def __init__(self, data: bytes):
        self.d, self.i = data, 0

    def take(self, n):
        if self.i + n > len(self.d):
            raise SerumError("preset body is truncated")
        out = self.d[self.i:self.i + n]
        self.i += n
        return out

    def arg(self, info):
        if info < 24:
            return info
        size = {24: 1, 25: 2, 26: 4, 27: 8}.get(info)
        if size is None:
            raise SerumError("unsupported CBOR length encoding")
        return int.from_bytes(self.take(size), "big")

    def item(self):
        ib = self.take(1)[0]
        major, info = ib >> 5, ib & 31
        if major == 0:
            return self.arg(info)
        if major == 1:
            return -1 - self.arg(info)
        if major == 2:
            return bytes(self.take(self.arg(info)))
        if major == 3:
            return self.take(self.arg(info)).decode("utf-8")
        if major == 4:
            return [self.item() for _ in range(self.arg(info))]
        if major == 5:
            out = {}
            for _ in range(self.arg(info)):
                k = self.item()
                out[k] = self.item()
            return out
        if major == 6:
            self.arg(info)  # tag number: not used by Serum; keep the value
            return self.item()
        if info == 20:
            return False
        if info == 21:
            return True
        if info in (22, 23):
            return None
        if info == 25:
            return F16(struct.unpack(">e", self.take(2))[0])
        if info == 26:
            return F32(struct.unpack(">f", self.take(4))[0])
        if info == 27:
            return F64(struct.unpack(">d", self.take(8))[0])
        raise SerumError("unsupported CBOR value")


def cbor_loads(data: bytes):
    r = _Reader(data)
    v = r.item()
    if r.i != len(data):
        raise SerumError("unexpected data after preset body")
    return v


def _head(major, n):
    if n < 24:
        return bytes([(major << 5) | n])
    for info, size in ((24, 1), (25, 2), (26, 4), (27, 8)):
        if n < (1 << (8 * size)):
            return bytes([(major << 5) | info]) + n.to_bytes(size, "big")
    raise SerumError("number too large")


def cbor_dumps(v) -> bytes:
    if v is True:
        return b"\xf5"
    if v is False:
        return b"\xf4"
    if v is None:
        return b"\xf6"
    if isinstance(v, F16):
        return b"\xf9" + struct.pack(">e", v)
    if isinstance(v, F32):
        return b"\xfa" + struct.pack(">f", v)
    if isinstance(v, float):
        return b"\xfb" + struct.pack(">d", v)
    if isinstance(v, int):
        return _head(0, v) if v >= 0 else _head(1, -1 - v)
    if isinstance(v, str):
        b = v.encode("utf-8")
        return _head(3, len(b)) + b
    if isinstance(v, (bytes, bytearray)):
        return _head(2, len(v)) + bytes(v)
    if isinstance(v, (list, tuple)):
        return _head(4, len(v)) + b"".join(cbor_dumps(x) for x in v)
    if isinstance(v, dict):
        return _head(5, len(v)) + b"".join(cbor_dumps(k) + cbor_dumps(x) for k, x in v.items())
    raise SerumError(f"can't store {type(v).__name__} in a preset")


# ---------------------------------------------------------------------------- file
def _zstd():
    try:
        import zstandard
    except ImportError as e:
        raise SerumError("Serum 2 presets need the 'zstandard' package: pip install zstandard") from e
    return zstandard


def load(data: bytes) -> tuple[dict, object]:
    if not data.startswith(MAGIC):
        raise SerumError("not a Serum 2 preset")
    n = struct.unpack("<Q", data[9:17])[0]
    meta = json.loads(data[17:17 + n].decode("utf-8"))
    rest = data[17 + n:]
    size, _flag = struct.unpack("<II", rest[:8])
    body = _zstd().ZstdDecompressor().decompress(rest[8:], max_output_size=max(size, 1) * 2 + 1024)
    return meta, cbor_loads(body)


def dump(meta: dict, body) -> bytes:
    raw = cbor_dumps(body)
    comp = _zstd().ZstdCompressor(level=19).compress(raw)
    meta = dict(meta)
    meta["hash"] = hashlib.md5(comp).hexdigest()
    head = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return MAGIC + struct.pack("<Q", len(head)) + head + struct.pack("<II", len(raw), FLAG) + comp


# ---------------------------------------------------------------------------- settings
def _scalar(v):
    return isinstance(v, (int, float, str, bool)) and not isinstance(v, bytes)


def flatten(body, max_list=16) -> dict:
    """Every editable setting as 'Path/To/Key': value (big data like wavetables is skipped)."""
    out = {}

    def walk(v, path):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}/{k}" if path else str(k))
        elif isinstance(v, list):
            if len(v) <= max_list:
                for i, x in enumerate(v):
                    walk(x, f"{path}/{i}")
        elif _scalar(v):
            out[path] = float(v) if isinstance(v, float) else v

    walk(body, "")
    return out


def _pattern(path: str) -> str:
    return re.sub(r"\d+", "#", path)


def add_known(body, path: str, value):
    """Add a setting Serum left at its default, but only if the same setting exists on a
    matching part of this preset (e.g. kParamDetune on Oscillator0 -> allowed on Oscillator1).
    That keeps Ghost from inventing names Serum wouldn't understand."""
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise SerumError(f"'{path}' isn't in this preset")
    parent, key = "/".join(parts[:-1]), parts[-1]
    flat = flatten(body, max_list=64)
    model = next((v for k, v in flat.items()
                  if k.rsplit("/", 1)[-1] == key and _pattern(k.rsplit("/", 1)[0]) == _pattern(parent)), None)
    if model is None:
        raise SerumError(f"'{path}' isn't in this preset, and no matching part of it uses '{key}'. "
                         "Use a setting read_preset_file lists, or pick another template.")
    node = body
    for i, p in enumerate(parts[:-1]):
        nxt = node[int(p)] if isinstance(node, list) and p.isdigit() and int(p) < len(node) else \
            node.get(p) if isinstance(node, dict) else None
        if nxt == "default" and i == len(parts) - 2 and isinstance(node, dict):
            node[p] = nxt = {}  # this part was all defaults; give it its own settings
        if not isinstance(nxt, (dict, list)):
            raise SerumError(f"'{path}' isn't in this preset (no '{p}')")
        node = nxt
    if not isinstance(node, dict):
        raise SerumError(f"'{path}' isn't in this preset")
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise SerumError(f"'{path}' needs a number, got {value!r}")
    if isinstance(model, bool) or not isinstance(model, (int, float)):
        raise SerumError(f"'{path}' can only be added for number settings")
    node[key] = F64(num) if isinstance(model, float) else int(round(num))
    return None, node[key]


def set_path(body, path: str, value):
    parts = [p for p in path.split("/") if p]
    node = body
    for p in parts[:-1]:
        if isinstance(node, list) and p.isdigit() and int(p) < len(node):
            node = node[int(p)]
        elif isinstance(node, dict) and p in node:
            node = node[p]
        else:
            raise SerumError(f"'{path}' isn't in this preset (no '{p}')")
    last = parts[-1] if parts else ""
    if isinstance(node, list) and last.isdigit() and int(last) < len(node):
        holder, key = node, int(last)
    elif isinstance(node, dict) and last in node:
        holder, key = node, last
    else:
        raise SerumError(f"'{path}' isn't in this preset")
    old = holder[key]
    if not _scalar(old):
        raise SerumError(f"'{path}' is a group, not a single setting")
    if isinstance(old, bool):
        if not isinstance(value, bool):
            raise SerumError(f"'{path}' needs true or false")
        new = value
    elif isinstance(old, (int, float)):
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise SerumError(f"'{path}' needs a number, got {value!r}")
        if isinstance(old, float):
            new = type(old)(num) if isinstance(old, (F16, F32, F64)) else num
        else:
            new = int(round(num))
    else:
        if not isinstance(value, str):
            raise SerumError(f"'{path}' is a named option (now '{old}'); give a name, not a number")
        new = value
    holder[key] = new
    return old, new
