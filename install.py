#!/usr/bin/env python3
"""
Ghost installer (Windows).

  python install.py            installs the Remote Script into Ableton and
                               adds Ghost to Claude Desktop's connectors
  python install.py --no-claude    only installs the Remote Script

Safe to run again after you update Ghost.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT_SRC = HERE / "ableton" / "Ghost"


def remote_scripts_dir() -> Path:
    return Path.home() / "Documents" / "Ableton" / "User Library" / "Remote Scripts"


def claude_config_paths():
    paths = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
    local = os.environ.get("LOCALAPPDATA")
    if local:  # Microsoft Store install
        pkgs = Path(local) / "Packages"
        if pkgs.exists():
            for p in pkgs.glob("Claude_*"):
                paths.append(p / "LocalCache" / "Roaming" / "Claude" / "claude_desktop_config.json")
    return paths


def install_remote_script():
    dest = remote_scripts_dir() / "Ghost"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SCRIPT_SRC, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"[ok] Remote Script installed to {dest}")


def install_claude_connector():
    entry = {"command": sys.executable, "args": [str(HERE / "mcp_server.py")]}
    targets = [p for p in claude_config_paths() if p.parent.exists()]
    if not targets:
        print("[!] Claude Desktop config folder not found. Install Claude Desktop, open it once,")
        print("    then run this again - or add this to claude_desktop_config.json yourself:")
        print(json.dumps({"mcpServers": {"ghost": entry}}, indent=2))
        return
    for cfg in targets:
        data = {}
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8") or "{}")
            except ValueError:
                print(f"[!] {cfg} isn't valid JSON - not touching it. Add this entry yourself:")
                print(json.dumps({"ghost": entry}, indent=2))
                continue
            backup = cfg.with_name(f"claude_desktop_config.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
            shutil.copy2(cfg, backup)
            print(f"[ok] Backed up your Claude config to {backup.name}")
        data.setdefault("mcpServers", {})["ghost"] = entry
        cfg.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[ok] Ghost added to Claude Desktop ({cfg})")


def main():
    if not SCRIPT_SRC.exists():
        sys.exit("Run this from the Ghost folder (ableton/Ghost not found).")
    install_remote_script()
    if "--no-claude" not in sys.argv:
        install_claude_connector()
    print("""
Next steps:
 1. Restart Ableton Live. Open Settings > Link, Tempo & MIDI and set a Control Surface
    slot to "Ghost" (Input and Output: None). Live shows "Ghost assistant ready".
 2. Fully quit and reopen Claude Desktop so it loads the Ghost connector.
 3. Test the link any time with:  python mcp_server.py --check
""")


if __name__ == "__main__":
    main()
