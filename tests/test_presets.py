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

# Repro-5 lives inside Repro-1.data (as on a real install)
r5dir = root / "Repro-1.data" / "Presets" / "Repro-5"; r5dir.mkdir(parents=True)
(root / "Repro-1.data" / "UserPresets").mkdir(parents=True)
(r5dir / "Deep.h2p").write_bytes(SAMPLE.replace("Diva", "Repro-5").encode())
check("Repro-5 detected inside Repro-1 folder", [f["synth"] for f in presets.find_presets("repro 5")] == ["Repro-5"])
check("Repro-1 search doesn't return Repro-5", presets.find_presets("repro-1") == [])
r5 = presets.write_preset(str(r5dir / "Deep.h2p"), "R5 Test", {"Freq": 70})
check("Repro-5 saved to its own user folder",
      Path(r5["saved"]) == root / "Repro-1.data" / "UserPresets" / "Repro-5" / "Ghost" / "R5 Test.h2p")

# Serum 2 (synthetic preset built with the same layout as real ones)
import serum2
F32, F64 = serum2.F32, serum2.F64
body = {"Env0": {"plainParams": {"kParamAttack": F64(0.001), "kParamRelease": F64(0.1)}},
        "Macro0": {"name": "DETUNE", "plainParams": {"kParamValue": F64(16.0)}},
        "Macro1": {"name": "WIDTH", "plainParams": "default"},
        "Oscillator0": {"plainParams": {"kParamDetune": F64(0.14), "kParamDetuneMode": "kDetuneExp",
                                        "kParamOctave": F64(-2.0)}},
        "Oscillator1": {"plainParams": {"kParamOctave": F64(-2.0)}},
        "ClipPlayer": {"kUIParamPreview": F32(1.0), "active": [0, 1], "big": list(range(100))},
        "Wave": {"data": b"\x00\x01" * 50}, "flag": True, "none": None}
raw = serum2.cbor_dumps(body)
check("CBOR round-trip keeps number widths", serum2.cbor_dumps(serum2.cbor_loads(raw)) == raw)
meta = {"fileType": "SerumPreset", "presetAuthor": "Factory", "presetName": "Base", "product": "Serum2"}
sdir = Path(tempfile.mkdtemp()) / "Serum 2 Presets" / "Presets"
(sdir / "Factory").mkdir(parents=True); (sdir / "User").mkdir()
spath = sdir / "Factory" / "Base.SerumPreset"; spath.write_bytes(serum2.dump(meta, body))
os.environ["GHOST_PRESET_DIRS"] = str(sdir.parent)
check("find Serum 2 presets", [f["synth"] for f in presets.find_presets("serum")] == ["Serum 2"])
info = presets.read_preset(str(spath))
check("read Serum 2 settings", info["settings"]["Env0/plainParams/kParamAttack"] == 0.001
      and "Macro1/plainParams" not in info["settings"] and "hash" not in info["meta"])
r = presets.write_preset(str(spath), "Ghost Reese", {"Env0/plainParams/kParamAttack": 0.2,
                          "Oscillator1/plainParams/kParamDetune": 0.3,        # known from Oscillator0
                          "Macro1/plainParams/kParamValue": 50,                # 'default' part gets settings
                          "Oscillator0/plainParams/kParamDetuneMode": "kDetuneLinear"}, "wide")
out = Path(r["saved"])
check("Serum 2 saved to Presets/User/Ghost", out == sdir / "User" / "Ghost" / "Ghost Reese.SerumPreset")
m2, b2 = serum2.load(out.read_bytes())
import hashlib, struct
n = struct.unpack("<Q", out.read_bytes()[9:17])[0]
check("hash matches compressed body", m2["hash"] == hashlib.md5(out.read_bytes()[17 + n + 8:]).hexdigest())
check("Serum 2 values written", b2["Env0"]["plainParams"]["kParamAttack"] == 0.2
      and b2["Oscillator1"]["plainParams"]["kParamDetune"] == 0.3
      and b2["Macro1"]["plainParams"] == {"kParamValue": 50.0}
      and b2["Oscillator0"]["plainParams"]["kParamDetuneMode"] == "kDetuneLinear")
check("untouched data preserved", b2["Wave"] == body["Wave"] and b2["ClipPlayer"] == body["ClipPlayer"])
check("Serum 2 name/author updated", m2["presetName"] == "Ghost Reese" and m2["presetAuthor"] == "Ghost")
check("Serum 2 template untouched", serum2.load(spath.read_bytes())[1] == body)
for bad, why in (({"Oscillator0/plainParams/kParamMadeUp": 1}, "no matching part"),
                 ({"Oscillator0/plainParams/kParamDetuneMode": 3}, "named option"),
                 ({"Env0/plainParams/kParamAttack": "slow"}, "needs a number")):
    raises("Serum 2 refuses " + why, lambda b=bad: presets.write_preset(str(spath), "bad", b), why)
check("nothing written on Serum error", not (sdir / "User" / "Ghost" / "bad.SerumPreset").exists())

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
