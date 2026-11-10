"""
Preset files Ghost can read and write directly.

Supported now:
- Serum 2 (.SerumPreset; see serum2.py)
- u-he synths (Diva, Hive, Repro-1, Repro-5 and other u-he plugins), whose .h2p
  presets are plain text:

    /*@Meta  ... Author / Description ... */
    #AM=Diva
    #cm=VCF1
    Freq=60.00
    Res=12.00
    ...
    // Section for ugly compressed binary Data
    // DON'T TOUCH THIS
    ...

Ghost never edits an existing preset. It copies one (a factory preset, an init
patch or one of yours), changes values, and saves the result in a "Ghost"
folder inside that synth's presets, where the synth's own browser shows it.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import serum2

PRESET_EXT = ".h2p"
SERUM_EXT = ".serumpreset"
BINARY_MARK = "// Section for ugly compressed binary Data"
MAX_FILES = 60000
GHOST_FOLDER = "Ghost"

_SECTION = re.compile(r"^#cm=(.+?)\s*$")
_KV = re.compile(r"^([^#/=\s][^=]*?)=(.*)$")
_NUMBER = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
_META = re.compile(r"/\*@Meta(.*?)\*/", re.S)
_META_FIELD = re.compile(r"([A-Za-z][A-Za-z ]*):\s*\n'(.*?)'", re.S)


class PresetError(Exception):
    pass


# ---------------------------------------------------------------------------- finding
def preset_roots() -> list[Path]:
    env = os.environ.get("GHOST_PRESET_DIRS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p and Path(p).exists()]
    home = Path.home()
    cands = [home / "Documents" / "u-he",                        # Windows (current installers)
             home / "Library" / "Audio" / "Presets" / "u-he",    # macOS
             Path("/Library/Audio/Presets/u-he"),
             home / "Documents" / "Xfer" / "Serum 2 Presets"]   # Serum 2 (Windows and macOS)
    for var, sub in (("CommonProgramFiles", "VST3"), ("CommonProgramFiles", "VST2"),
                     ("ProgramFiles", "VSTPlugins"), ("ProgramFiles", "Steinberg/VSTPlugins")):
        base = os.environ.get(var)
        if base:
            cands.append(Path(base) / sub / "u-he")
    out, seen = [], set()
    for c in cands:
        if c.exists() and str(c).lower() not in seen:
            seen.add(str(c).lower())
            out.append(c)
    return out


def _synth_from_text(text: str) -> str | None:
    m = re.search(r"^#AM=(.+?)\s*$", text, re.M)
    return m.group(1) if m else None


def _data_dir(path: Path):
    """For a u-he preset: (<Synth>.data folder, synth name, 'Presets'/'UserPresets').
    Some synths share a folder (Repro-5 lives in Repro-1.data/Presets/Repro-5)."""
    parts = path.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower().endswith(".data"):
            if i + 2 < len(parts) and parts[i + 1].lower() in ("presets", "userpresets"):
                return Path(*parts[: i + 1]), parts[i + 2], parts[i + 1]
            return Path(*parts[: i + 1]), parts[i][:-5], None
    return None, None, None


def _synth_from_path(path: Path) -> str | None:
    if path.suffix.lower() == SERUM_EXT:
        return "Serum 2"
    _, synth, _ = _data_dir(path)
    if synth:
        return synth
    for anc in path.parents:  # macOS: .../Presets/u-he/<Synth>/...
        if anc.parent.name.lower() == "u-he":
            return anc.name
    return path.parent.name if path.parent else None


def _is_preset(p: Path) -> bool:
    return p.suffix.lower() in (PRESET_EXT, SERUM_EXT)


def find_presets(synth: str = "", query: str = "", limit: int = 40) -> list[dict]:
    squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # "Repro 5" == "repro-5"
    synth_q, words = squash(synth), query.lower().split()
    hits, scanned = [], 0
    for root in preset_roots():
        for p in root.rglob("*"):
            if not _is_preset(p) or p.name.startswith("._") or "__MACOSX" in p.parts:
                continue  # macOS resource-fork leftovers from zip files, not real presets
            scanned += 1
            if scanned > MAX_FILES:
                break
            path_l = str(p).lower()
            name = _synth_from_path(p) or ""
            if synth_q and synth_q not in squash(name):
                continue
            if words and not all(w in path_l for w in words):
                continue
            seps = (os.sep, "/")
            yours = ("userpresets" in path_l or "user presets" in path_l
                     or any(f"{s}user{s}" in path_l for s in seps) or p.parent.name == GHOST_FOLDER)
            hits.append({"name": p.stem, "synth": name, "path": str(p), "yours": yours,
                         "made_by_ghost": p.parent.name == GHOST_FOLDER})
    hits.sort(key=lambda h: (not h["yours"], h["name"].lower()))
    return hits[: max(1, min(int(limit), 300))]


# ---------------------------------------------------------------------------- reading
def _split(text: str):
    """(head, body_lines, tail) where tail is the binary block that must stay untouched."""
    i = text.find(BINARY_MARK)
    if i == -1:
        return text, ""
    return text[:i], text[i:]


def parse(text: str) -> dict:
    head, _ = _split(text)
    meta = {}
    m = _META.search(head)
    if m:
        meta = {k.strip(): v for k, v in _META_FIELD.findall(m.group(1))}
    sections, current = {}, None
    for line in head.splitlines():
        s = _SECTION.match(line)
        if s:
            current = s.group(1)
            sections.setdefault(current, {})
            continue
        if current is None:
            continue
        kv = _KV.match(line.strip())
        if kv:
            sections[current][kv.group(1).strip()] = kv.group(2).strip()
    if not sections:
        raise PresetError("no parameter sections found - is this a u-he .h2p preset?")
    return {"synth": _synth_from_text(head), "meta": meta, "sections": sections}


def _read_serum(p: Path) -> dict:
    try:
        meta, body = serum2.load(p.read_bytes())
    except serum2.SerumError as e:
        raise PresetError(str(e)) from e
    flat = {k: v for k, v in serum2.flatten(body).items()
            if not (v == "default" and k.endswith("plainParams"))}
    meta = {k: v for k, v in meta.items() if k != "hash"}
    return {"name": p.stem, "path": str(p), "synth": "Serum 2", "meta": meta,
            "settings": flat, "parameter_count": len(flat),
            "note": "Sections set to 'default' are left out; they use Serum's defaults and can't "
                    "be edited from this preset. Pick a template that already uses the parts you "
                    "want to change."}


def read_preset(path: str) -> dict:
    p = Path(path)
    if not p.is_file() or not _is_preset(p):
        raise PresetError(f"not a supported preset file (.h2p or .SerumPreset): {path}")
    if p.suffix.lower() == SERUM_EXT:
        return _read_serum(p)
    info = parse(p.read_text(encoding="utf-8", errors="replace"))
    info["name"], info["path"] = p.stem, str(p)
    info["synth"] = info["synth"] or _synth_from_path(p)
    info["parameter_count"] = sum(len(v) for v in info["sections"].values())
    return info


# ---------------------------------------------------------------------------- writing
def _resolve_key(sections: dict, key: str) -> tuple[str, str]:
    if ":" in key or "/" in key:
        sec, _, param = key.replace("/", ":").partition(":")
        sec_match = [s for s in sections if s.lower() == sec.strip().lower()]
        if not sec_match:
            raise PresetError(f"section '{sec}' not in preset (sections: {', '.join(sections)})")
        params = sections[sec_match[0]]
        p_match = [k for k in params if k.lower() == param.strip().lower()]
        if not p_match:
            raise PresetError(f"'{param}' not in section {sec_match[0]}")
        return sec_match[0], p_match[0]
    found = [(s, k) for s, ps in sections.items() for k in ps if k.lower() == key.strip().lower()]
    if len(found) == 1:
        return found[0]
    if not found:
        raise PresetError(f"parameter '{key}' not found in this preset")
    raise PresetError(f"'{key}' is in several sections ({', '.join(s for s, _ in found)}); "
                      f"write it as Section:{key}")


def _format(old: str, new) -> str:
    if _NUMBER.match(old):
        try:
            num = float(new)
        except (TypeError, ValueError):
            raise PresetError(f"needs a number, got {new!r}")
        decimals = len(old.split(".")[1]) if "." in old else 0
        return f"{num:.{decimals}f}" if decimals else str(int(round(num)))
    return str(new)


def _meta_block(meta: dict, eol: str = "\n") -> str:
    lines = ["/*@Meta", ""]
    for k, v in meta.items():
        lines += [f"{k}:", "'" + str(v).replace("'", "’") + "'", ""]
    lines.append("*/")
    return eol.join(lines) + eol + eol


def _dest_folder(template: Path) -> Path:
    if template.suffix.lower() == SERUM_EXT:
        # .../Serum 2 Presets/Presets/User/Ghost (Serum's own place for user presets)
        for anc in template.parents:
            if anc.name.lower() == "presets" and anc.parent.name.lower() == "serum 2 presets":
                return anc / "User" / GHOST_FOLDER
        return template.parent / GHOST_FOLDER
    data, synth, _ = _data_dir(template)
    if data is not None and synth:
        if (data / "UserPresets").exists():
            return data / "UserPresets" / synth / GHOST_FOLDER
        return data / "Presets" / synth / GHOST_FOLDER
    for anc in template.parents:
        if anc.parent.name.lower() == "u-he":  # macOS layout
            return anc / GHOST_FOLDER
    return template.parent / GHOST_FOLDER


def _write_serum(tpl: Path, name: str, changes: dict, description: str, dest_folder) -> dict:
    try:
        meta, body = serum2.load(tpl.read_bytes())
        applied = []
        for key, value in (changes or {}).items():  # all-or-nothing: body is a fresh copy
            path = key.strip().strip("/")
            try:
                old, new = serum2.set_path(body, path, value)
            except serum2.SerumError as e:
                if "isn't in this preset" not in str(e):
                    raise
                old, new = serum2.add_known(body, path, value)
            applied.append({"parameter": key, "from": "default" if old is None else old, "to": new})
        meta["presetName"] = _safe_name(name)
        meta["presetAuthor"] = "Ghost"
        meta["presetDescription"] = (description or f"Made by Ghost from '{tpl.stem}'").strip()
        data = serum2.dump(meta, body)
    except serum2.SerumError as e:
        raise PresetError(str(e)) from e
    folder = Path(dest_folder) if dest_folder else _dest_folder(tpl)
    if folder.name != GHOST_FOLDER:
        folder = folder / GHOST_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (_safe_name(name) + tpl.suffix)
    if dest.resolve() == tpl.resolve():
        raise PresetError("refusing to overwrite the template")
    dest.write_bytes(data)
    return {"saved": str(dest), "name": dest.stem, "synth": "Serum 2", "template": tpl.stem,
            "changed": applied}


def _safe_name(name: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip().strip(".")
    if not clean:
        raise PresetError("preset name is empty")
    return clean[:80]


def write_preset(template_path: str, name: str, changes: dict, description: str = "",
                 dest_folder: str | None = None) -> dict:
    tpl = Path(template_path)
    if not tpl.is_file() or not _is_preset(tpl):
        raise PresetError(f"template is not a .h2p or .SerumPreset preset: {template_path}")
    if tpl.suffix.lower() == SERUM_EXT:
        return _write_serum(tpl, name, changes, description, dest_folder)
    text = tpl.read_bytes().decode("utf-8", errors="replace")  # keep CRLF/LF exactly
    head, tail = _split(text)
    sections = parse(text)["sections"]

    # Work out every change before touching anything, so one bad key changes nothing.
    planned: dict[tuple[str, str], str] = {}
    for key, value in (changes or {}).items():
        sec, param = _resolve_key(sections, key)
        try:
            planned[(sec, param)] = _format(sections[sec][param], value)
        except PresetError as e:
            raise PresetError(f"{sec}:{param} {e}")

    out_lines, current, applied = [], None, []
    for line in head.splitlines(keepends=True):
        s = _SECTION.match(line.rstrip("\r\n"))
        if s:
            current = s.group(1)
        elif current is not None:
            kv = _KV.match(line.strip())
            if kv and (current, kv.group(1).strip()) in planned:
                k = kv.group(1).strip()
                eol = line[len(line.rstrip("\r\n")):]
                old_val = kv.group(2).strip()
                new_val = planned[(current, k)]
                line = f"{k}={new_val}{eol}"
                applied.append({"parameter": f"{current}:{k}", "from": old_val, "to": new_val})
        out_lines.append(line)
    new_head = "".join(out_lines)

    meta = parse(text)["meta"]
    meta["Author"] = "Ghost"
    meta["Description"] = (description or f"Made by Ghost from '{tpl.stem}'").strip()
    eol = "\r\n" if "\r\n" in head else "\n"
    block = _meta_block(meta, eol)
    new_head = _META.sub(lambda _: block.rstrip("\r\n"), new_head, count=1) if _META.search(new_head) \
        else block + new_head

    folder = Path(dest_folder) if dest_folder else _dest_folder(tpl)
    if folder.name != GHOST_FOLDER:
        folder = folder / GHOST_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (_safe_name(name) + PRESET_EXT)
    if dest.resolve() == tpl.resolve():
        raise PresetError("refusing to overwrite the template")
    dest.write_text(new_head + tail, encoding="utf-8", newline="")
    return {"saved": str(dest), "name": dest.stem, "synth": _synth_from_text(text) or _synth_from_path(tpl),
            "template": tpl.stem, "changed": applied}
