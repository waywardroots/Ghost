"""
Ghost's ears: measure a sound and draw it.

analyze(path)            -> measurements + a plain-language summary
spectrogram(path, png)   -> picture of the sound (frequency over time) Claude can look at
compare(a, b)            -> how sound a differs from sound b (e.g. Ghost's patch vs a reference)

Run directly to check a file:  python listen.py some_sound.wav [reference.wav]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

SR = 44100
BANDS = [("sub", 20, 60), ("bass", 60, 250), ("low_mid", 250, 800), ("mid", 800, 2500),
         ("high_mid", 2500, 6000), ("air", 6000, 20000)]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _db(x: float) -> float:
    return round(20 * math.log10(max(x, 1e-9)), 1)


def _load(path: str, max_seconds: float = 30.0):
    import librosa
    y, sr = librosa.load(path, sr=SR, mono=False, duration=max_seconds)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float64), sr


def _note(hz: float | None) -> str | None:
    if not hz or hz <= 0:
        return None
    midi = 69 + 12 * math.log2(hz / 440.0)
    n = int(round(midi))
    cents = int(round((midi - n) * 100))
    # Live names middle C (MIDI 60) "C3"
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 2}" + (f" {cents:+d}c" if abs(cents) >= 10 else "")


def analyze(path: str) -> dict:
    import librosa
    y, sr = _load(path)
    mono = y.mean(axis=0)
    dur = mono.size / sr
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-5:
        return {"file": str(path), "seconds": round(dur, 2), "silent": True,
                "summary": ["The recording is silent. Is the track muted, or is the synth not "
                            "receiving the MIDI clip?"]}
    rms = float(np.sqrt(np.mean(mono ** 2)))

    # --- spectrum by band
    n_fft = 8192
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total = float(S.sum()) or 1.0
    bands = {name: round(100 * float(S[(freqs >= lo) & (freqs < hi)].sum()) / total, 1)
             for name, lo, hi in BANDS}

    # --- brightness and noisiness over time
    cent = librosa.feature.spectral_centroid(y=mono, sr=sr, hop_length=1024)[0]
    loud_frames = librosa.feature.rms(y=mono, hop_length=1024)[0]
    active = loud_frames > (loud_frames.max() * 0.1)
    cent_a = cent[active] if active.any() else cent
    centroid = float(np.median(cent_a))
    movement = float(np.std(np.log2(np.maximum(cent_a, 20)))) if cent_a.size > 2 else 0.0
    flatness = float(np.median(librosa.feature.spectral_flatness(y=mono, hop_length=1024)[0][active]
                               if active.any() else 0.0))

    # --- pitch
    f0_hz = None
    try:
        f0, voiced, _ = librosa.pyin(mono[: sr * 6], fmin=30, fmax=1200, sr=sr, frame_length=4096)
        f0v = f0[voiced & ~np.isnan(f0)]
        if f0v.size >= 3:
            f0_hz = float(np.median(f0v))
    except Exception:
        pass

    # --- harmonics: how fast they fade (dB per octave). A clean saw falls about 6 dB/oct,
    # filtering makes it steeper, distortion makes it flatter.
    richness = harmonic_slope = None
    if f0_hz:
        richness = round(100 * float(S[freqs > f0_hz * 8].sum()) / total, 1)
        avg = S.mean(axis=1)
        pts = []
        for h in range(1, 41):
            fh = f0_hz * h
            if fh > 16000:
                break
            win = (freqs > fh * 0.97) & (freqs < fh * 1.03)
            if win.any():
                pts.append((math.log2(h), 10 * math.log10(max(float(avg[win].max()), 1e-20))))
        if len(pts) >= 6:
            xs, ys = np.array(pts).T
            ref = ys[0]
            keep = ys > ref - 90  # ignore the noise floor
            if keep.sum() >= 6:
                harmonic_slope = round(float(np.polyfit(xs[keep], ys[keep], 1)[0]), 1)

    # --- envelope
    hop = 256
    env = librosa.feature.rms(y=mono, frame_length=4096, hop_length=hop)[0]  # ~90 ms: smooths low-note ripple
    t = np.arange(env.size) * hop / sr
    peak_env = float(env.max())
    start = int(np.argmax(env > peak_env * 0.1))
    # attack = time to first reach 70% of the loudest point (beating can make the
    # absolute peak land late in a sustained note)
    reach = int(np.argmax(env > peak_env * 0.7))
    attack_ms = round(1000 * (t[reach] - t[start]), 1) if reach > start else 0.0
    tail = env[reach:]
    below = np.where(tail < peak_env * 0.25)[0]
    fall_ms = round(1000 * below[0] * hop / sr, 1) if below.size else None
    sustain = round(float(np.median(env[reach:reach + int(0.5 * sr / hop)])) / peak_env, 2) if peak_env > 0 else 0

    # --- movement in volume: wobble / phasing rate (reese beating shows up here).
    # Only the held part of the note, in dB, with the overall fade removed.
    mod_rate, mod_depth = None, 0.0
    held = np.where(env > peak_env * 0.3)[0]
    if held.size:
        a, b2 = reach + int(0.05 * sr / hop), held[-1] - int(0.05 * sr / hop)
        seg = env[a:b2]
        if seg.size * hop / sr >= 0.8:
            ldb = 20 * np.log10(np.maximum(seg, 1e-9))
            idx = np.arange(ldb.size)
            resid = ldb - np.polyval(np.polyfit(idx, ldb, 1), idx)
            mod_depth = round(float(np.std(resid)), 2)  # dB
            spec = np.abs(np.fft.rfft(resid * np.hanning(resid.size)))
            fr = np.fft.rfftfreq(resid.size, d=hop / sr)
            lo_rate = max(0.4, 2.0 / (resid.size * hop / sr))  # need ~2 cycles to call it movement
            band = (fr >= lo_rate) & (fr <= 20)
            if band.any() and mod_depth > 0.9:
                mod_rate = round(float(fr[np.argmax(spec * band)]), 2)

    # --- stereo
    L, R = y[0], y[1]
    mid, side = (L + R) / 2, (L - R) / 2
    width = round(float(np.sqrt(np.mean(side ** 2)) / max(np.sqrt(np.mean(mid ** 2)), 1e-9)), 2)
    corr = float(np.corrcoef(L, R)[0, 1]) if np.std(L) > 0 and np.std(R) > 0 else 1.0
    lo_mask = freqs < 120
    side_S = np.abs(librosa.stft(side, n_fft=n_fft, hop_length=2048)) ** 2
    mid_S = np.abs(librosa.stft(mid, n_fft=n_fft, hop_length=2048)) ** 2
    low_width = round(float(np.sqrt(side_S[lo_mask].sum() / max(mid_S[lo_mask].sum(), 1e-12))), 2)

    m = {
        "file": str(path), "seconds": round(dur, 2),
        "peak_db": _db(peak), "rms_db": _db(rms), "crest_db": round(_db(peak) - _db(rms), 1),
        "bands_percent": bands,
        "brightness_hz": round(centroid), "filter_movement": round(movement, 3),
        "noisiness": round(flatness, 4),
        "pitch_hz": round(f0_hz, 1) if f0_hz else None, "pitch_note": _note(f0_hz),
        "harmonic_richness_percent": richness, "harmonic_slope_db_per_octave": harmonic_slope,
        "attack_ms": attack_ms, "fall_to_quarter_ms": fall_ms, "sustain_ratio": sustain,
        "volume_movement_hz": mod_rate, "volume_movement_depth": mod_depth,
        "stereo_width": width, "stereo_correlation": round(corr, 2), "low_end_width": low_width,
    }
    m["summary"] = describe(m)
    return m


def describe(m: dict) -> list[str]:
    """Turn measurements into the words a producer would use. Rough guides, not rules."""
    out = []
    b = m["bands_percent"]
    low = b["sub"] + b["bass"]
    out.append(f"Pitch: {m['pitch_note'] or 'no clear pitch'}; low end {low:.0f}% "
               f"(sub {b['sub']:.0f}%, bass {b['bass']:.0f}%), low mids {b['low_mid']:.0f}%, "
               f"mids {b['mid']:.0f}%, highs {b['high_mid'] + b['air']:.0f}%.")
    c = m["brightness_hz"]
    tone = "very dark" if c < 300 else "dark" if c < 700 else "warm" if c < 1500 else \
        "bright" if c < 3500 else "very bright/harsh"
    out.append(f"Tone: {tone} (brightness {c} Hz).")
    sl = m.get("harmonic_slope_db_per_octave")
    if sl is None:
        grit = "noisy" if m["noisiness"] > 0.08 else "unclear (no steady pitch)"
    else:
        grit = "gritty/distorted, dense upper harmonics" if sl > -4.5 else \
            "rich, saw-like" if sl > -8 else "rounded, filtered" if sl > -16 else "smooth, nearly pure"
        grit += f" (harmonics fall {abs(sl):.0f} dB/octave)"
    out.append(f"Texture: {grit}.")
    mv = []
    if m["volume_movement_hz"]:
        kind = "slow phasing/beating" if m["volume_movement_hz"] < 2 else \
            "wobble" if m["volume_movement_hz"] < 8 else "fast flutter"
        mv.append(f"{kind} at about {m['volume_movement_hz']} Hz (depth {m['volume_movement_depth']} dB)")
    if m["filter_movement"] > 0.25:
        mv.append("the tone sweeps noticeably (filter or wavetable movement)")
    out.append("Movement: " + ("; ".join(mv) if mv else "static, little movement") + ".")
    w = m["stereo_width"]
    out.append(f"Stereo: {'mono' if w < 0.08 else 'slightly wide' if w < 0.3 else 'wide' if w < 0.7 else 'very wide'}"
               f" (width {w}); low end {'centered' if m['low_end_width'] < 0.15 else 'WIDE - may sound weak or phasey on club systems'}.")
    a = m["attack_ms"]
    out.append(f"Envelope: {'punchy' if a < 15 else 'soft' if a < 150 else 'slow swell'} attack "
               f"(~{a:.0f} ms), sustain {m['sustain_ratio']}.")
    if m["peak_db"] > -0.3:
        out.append("Level: peaking at 0 dB - likely clipping.")
    return out


def spectrogram(path: str, png: str, title: str | None = None) -> str:
    import librosa
    import librosa.display  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y, sr = _load(path, max_seconds=12)
    mono = y.mean(axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [4, 1]},
                                   sharex=True, dpi=100)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(mono, n_fft=4096, hop_length=512)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=ax1,
                                   cmap="magma", vmin=-80, vmax=0)
    ax1.set_ylim(20, 20000)
    ax1.set_title(title or Path(path).name)
    ax1.set_xlabel("")
    fig.colorbar(img, ax=ax1, format="%+2.0f dB", pad=0.01)
    t = np.arange(mono.size) / sr
    ax2.plot(t, y[0], lw=0.4, color="#4c9be8", label="L")
    ax2.plot(t, y[1], lw=0.4, color="#e8794c", alpha=0.7, label="R")
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("level")
    ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png)
    plt.close(fig)
    return png


def compare(a: dict, b: dict) -> dict:
    """How sound a (e.g. Ghost's patch) differs from b (the reference)."""
    notes = []
    for band in a["bands_percent"]:
        d = a["bands_percent"][band] - b["bands_percent"][band]
        if abs(d) >= 5:
            notes.append(f"{band.replace('_', ' ')}: {'more' if d > 0 else 'less'} than the reference by {abs(d):.0f} points")
    ratio = a["brightness_hz"] / max(b["brightness_hz"], 1)
    if ratio > 1.25 or ratio < 0.8:
        notes.append(f"{'brighter' if ratio > 1 else 'darker'} than the reference "
                     f"({a['brightness_hz']} vs {b['brightness_hz']} Hz)")
    ra, rb = a.get("harmonic_slope_db_per_octave"), b.get("harmonic_slope_db_per_octave")
    if ra is not None and rb is not None and abs(ra - rb) >= 2.5:
        notes.append(f"{'grittier/more harmonics' if ra > rb else 'cleaner/more filtered'} than the reference "
                     f"(harmonics fall {abs(ra):.0f} vs {abs(rb):.0f} dB/octave)")
    ma, mb = a.get("volume_movement_hz"), b.get("volume_movement_hz")
    if bool(ma) != bool(mb):
        notes.append("has movement the reference lacks" if ma else
                     f"lacks the reference's movement (~{mb} Hz); add detune/LFO")
    elif ma and mb and abs(ma - mb) / mb > 0.3:
        notes.append(f"movement is {'faster' if ma > mb else 'slower'} ({ma} vs {mb} Hz)")
    if abs(a["stereo_width"] - b["stereo_width"]) >= 0.2:
        notes.append(f"{'wider' if a['stereo_width'] > b['stereo_width'] else 'narrower'} "
                     f"({a['stereo_width']} vs {b['stereo_width']})")
    if abs(a["attack_ms"] - b["attack_ms"]) > max(20, 0.5 * b["attack_ms"]):
        notes.append(f"attack is {'slower' if a['attack_ms'] > b['attack_ms'] else 'faster'} "
                     f"({a['attack_ms']:.0f} vs {b['attack_ms']:.0f} ms)")
    if a.get("pitch_hz") and b.get("pitch_hz"):
        semis = 12 * math.log2(a["pitch_hz"] / b["pitch_hz"])
        if abs(semis) >= 0.5:
            notes.append(f"pitched {abs(semis):.1f} semitones {'higher' if semis > 0 else 'lower'} "
                         f"({a['pitch_note']} vs {b['pitch_note']})")
    return {"differences": notes or ["very close to the reference on every measure"]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python listen.py sound.wav [reference.wav]")
    a = analyze(sys.argv[1])
    print("\n".join(a.get("summary", [])))
    if len(sys.argv) > 2:
        b = analyze(sys.argv[2])
        print("\nCompared with", Path(sys.argv[2]).name)
        print("\n".join("- " + d for d in compare(a, b)["differences"]))
    print("\n" + json.dumps({k: v for k, v in a.items() if k != "summary"}, indent=1))
