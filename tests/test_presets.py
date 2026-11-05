"""Tests for u-he preset reading/writing and the fallback parameter path.  Run: python tests/test_presets.py"""
import os, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ableton"), str(ROOT / "tests")]
import presets
import fake_live
fake_live.install_fake_modules()
from Ghost import params as gp

SAMPLE = ("/*@Meta\r\n\r\nAuthor:\r\n'u-he'\r\n\r\nDescription:\r\n'Init patch'\r\n\r\n*/\r\n\r\n"
          "#AM=Diva\r\n#Vers=1\r\n#Endian=little\r\n#nm=3\r\n"
          "#cm=OSC\r\nTune=0.00\r\nWave=1\r\nVolume=50.00\r\n"
          "#cm=VCF1\r\nFreq=100.00\r\nRes=0.00\r\nVolume=10.00\r\n"
          "#cm=ENV1\r\nAtk=0.00\r\nRel=20.00\r\n\r\n"
          "// Section for ugly compressed binary Data\r\n// DON'T TOUCH THIS\r\n[[[BINARY!data=]]]\r\n")
PASSED = []
def check(n, c):
    assert c, "FAILED: " + n
    PASSED.append(n)
def raises(n, fn, contains):
    try: fn()
    except presets.PresetError as e:
        check(n, contains in str(e)); return
    raise AssertionError("FAILED (no error): " + n)

root = Path(tempfile.mkdtemp()) / "u-he"
fac = root / "Diva.data" / "Presets" / "Diva" / "Basics"
usr = root / "Diva.data" / "UserPresets" / "Diva"
fac.mkdir(parents=True); usr.mkdir(parents=True)
tpl = fac / "Init.h2p"; tpl.write_bytes(SAMPLE.encode())
(usr / "My Fat Bass.h2p").write_bytes(SAMPLE.encode())
(root / "Hive.data" / "Presets" / "Hive").mkdir(parents=True)
(root / "Hive.data" / "Presets" / "Hive" / "Pad.h2p").write_bytes(SAMPLE.replace("Diva", "Hive").encode())
os.environ["GHOST_PRESET_DIRS"] = str(root)

found = presets.find_presets("diva")
check("find presets for one synth", [f["name"] for f in found] == ["My Fat Bass", "Init"])
check("user presets first", found[0]["yours"] and not found[1]["yours"])
check("query filter", [f["name"] for f in presets.find_presets(query="init")] == ["Init"])
info = presets.read_preset(str(tpl))
check("read sections", info["synth"] == "Diva" and info["sections"]["VCF1"]["Freq"] == "100.00"
      and info["parameter_count"] == 8 and info["meta"]["Description"] == "Init patch")

r = presets.write_preset(str(tpl), "Ghost Dark Bass", {"VCF1:Freq": 62.456, "Res": 30, "OSC:Wave": 3,
                         "atk": 1.5}, "Dark round bass")
out = Path(r["saved"])
check("saved into the user Ghost folder", out == usr / "Ghost" / "Ghost Dark Bass.h2p")
txt = out.read_bytes().decode()
check("values written with original precision", "Freq=62.46\r\n" in txt and "Res=30.00\r\n" in txt
      and "Wave=3\r\n" in txt and "Atk=1.50\r\n" in txt)
check("only targeted section changed", "#cm=VCF1\r\nFreq=62.46" in txt and "Volume=10.00" in txt and "Volume=50.00" in txt)
check("binary block untouched", txt.endswith("// Section for ugly compressed binary Data\r\n// DON'T TOUCH THIS\r\n[[[BINARY!data=]]]\r\n"))
check("CRLF preserved", "\n" not in txt.replace("\r\n", ""))
check("meta updated", "Author:\r\n'Ghost'" in txt and "Dark round bass" in txt)
check("template untouched", tpl.read_bytes().decode() == SAMPLE)
check("changes reported", {c["parameter"] for c in r["changed"]} == {"VCF1:Freq", "VCF1:Res", "OSC:Wave", "ENV1:Atk"})
check("re-read works", presets.read_preset(str(out))["sections"]["VCF1"]["Freq"] == "62.46")
raises("ambiguous key explained", lambda: presets.write_preset(str(tpl), "x", {"Volume": 1}), "Section:Volume")
raises("unknown param refused", lambda: presets.write_preset(str(tpl), "x", {"Cutoff": 1}), "not found")
raises("non-number refused", lambda: presets.write_preset(str(tpl), "x", {"Freq": "loud"}), "needs a number")
check("nothing written on error", not (usr / "Ghost" / "x.h2p").exists())
raises("bad file explained", lambda: presets.read_preset(str(root)), ".h2p")
r2 = presets.write_preset(str(root / "Hive.data" / "Presets" / "Hive" / "Pad.h2p"), "Soft: Pad?", {"Freq": 50})
check("no UserPresets -> Presets/Ghost, unsafe chars removed",
      Path(r2["saved"]) == root / "Hive.data" / "Presets" / "Hive" / "Ghost" / "Soft Pad.h2p")

# Fallback for plugins that only report their current value (no str_for_value)
class OnlyCurrent(fake_live.ShownParam):
    def str_for_value(self, v):
        raise RuntimeError("not supported")
p = OnlyCurrent("Cutoff", fake_live._hz, 0.3)
res = gp.set_param(p, "1.2 kHz")
check("fallback path sets real units", res["display"] in ("1.19 kHz", "1.20 kHz", "1.21 kHz"))
try:
    gp.set_param(p, "20 ms")
except gp.ParamError:
    check("fallback restores value after a failed set", str(p).startswith("1.2") or str(p).startswith("1.19"))
check("parse thousands", gp.parse_amount("1,200 Hz") == (1200.0, "hz") and gp.parse_amount("-inf dB")[0] == float("-inf"))

print("\n".join("PASS  " + x for x in PASSED)); print(f"\nAll {len(PASSED)} checks passed.")
