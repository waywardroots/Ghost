"""
Setting device parameters in real units.

Live stores every parameter as a raw number (often 0.0-1.0) but shows it as text
("800 Hz", "-6.0 dB", "Saw"). To set "800 Hz" we search the raw range until the
displayed text matches. This works for Live's devices and for any plugin that
reports its own display values (VST3 plugins usually do).
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import re


class ParamError(Exception):
    pass


_THOUSANDS = re.compile(r"(\d),(\d{3})(?!\d)")
_NUM = re.compile(r"(-?\s*inf|[-+]?\d+(?:[.,]\d+)?)\s*([a-zA-Z%°]*)", re.I)
_UNITS = {
    "hz": ("hz", 1.0), "khz": ("hz", 1000.0), "k": ("hz", 1000.0),
    "ms": ("s", 0.001), "s": ("s", 1.0), "sec": ("s", 1.0),
    "db": ("db", 1.0), "%": ("%", 1.0),
    "st": ("st", 1.0), "semi": ("st", 1.0), "semitones": ("st", 1.0),
    "ct": ("ct", 1.0), "cent": ("ct", 1.0), "cents": ("ct", 1.0),
    "bpm": ("bpm", 1.0),
}
_INF = float("inf")


def parse_amount(text):
    """'1.2 kHz' -> (1200.0, 'hz'); '-inf dB' -> (-inf, 'db'); 'Saw' -> None."""
    if text is None:
        return None
    t = _THOUSANDS.sub(r"\1\2", ("%s" % (text,)).strip().replace("−", "-"))
    m = _NUM.search(t)
    if not m:
        return None
    num = m.group(1).replace(" ", "").replace(",", ".").lower()
    unit = m.group(2).lower()
    if num == "-inf":
        val = -_INF
    elif num == "inf":
        val = _INF
    else:
        val = float(num)
    family, mul = _UNITS.get(unit, (unit or None, 1.0))
    return val * mul, family


def display(param):
    try:
        return "%s" % (param,)
    except Exception:
        return None


def choices(param):
    try:
        return [("%s" % (i,)) for i in (param.value_items or [])]
    except Exception:
        return []


def _is_raw(text, v):
    """True when a display is just the bare raw value (no units), e.g. '0.25' for 0.25."""
    try:
        return abs(float(("%s" % (text,)).strip()) - float(v)) < 0.006
    except (TypeError, ValueError):
        return False


class _Reader(object):
    """Reads the text a parameter would show for a raw value."""

    def __init__(self, param):
        self.p = param
        self.use_str_for_value = False
        try:
            lo, hi = float(param.min), float(param.max)
            probes = [(v, param.str_for_value(v)) for v in (lo, (lo + hi) / 2.0, hi)]
            # Some plugins (often right after a set loads) only echo the raw number back
            # ("0.25"); steering by that would treat 0..1 as the real range.
            echoes = all(_is_raw(t, v) for v, t in probes)
            self.use_str_for_value = probes[0][1] != probes[-1][1] and not echoes
        except Exception:
            pass

    def text(self, v):
        if self.use_str_for_value:
            return self.p.str_for_value(v)
        # Many VST2 plugins can't preview a value's text, and only update their display on
        # Live's next refresh, so set-and-read here would read stale text and drift to the
        # wrong value. The connector calibrates these over several refreshes instead.
        raise ParamError("NEEDS_CALIBRATION: %s can't preview values; it shows '%s' now"
                         % (self.p.name, display(self.p)))


def _clamp(p, v):
    return max(float(p.min), min(float(p.max), float(v)))


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def _choose(p, name):
    want = _norm(name)
    items = choices(p)
    if items:
        names = [_norm(i) for i in items]
        exact = [i for i, n in enumerate(names) if n == want]
        if exact:
            return float(p.min) + exact[0]
        for hits in ([i for i, n in enumerate(names) if n.startswith(want)],
                     [i for i, n in enumerate(names) if want in n]):
            if len(hits) == 1:
                return float(p.min) + hits[0]
            if len(hits) > 1:
                raise ParamError("'%s' matches several options for %s: %s"
                                 % (name, p.name, ", ".join(items[i] for i in hits)))
        raise ParamError("'%s' isn't an option for %s. Options: %s" % (name, p.name, ", ".join(items)))
    # No list of options: scan the range for matching display text.
    reader = _Reader(p)
    steps = 256
    lo, hi = float(p.min), float(p.max)
    for i in range(steps + 1):
        v = lo + (hi - lo) * i / steps
        if _norm(reader.text(v) or "") == want:
            return v
    raise ParamError("couldn't find a setting of %s that shows '%s'" % (p.name, name))


def _solve(p, target, family):
    reader = _Reader(p)
    lo, hi = float(p.min), float(p.max)
    span = hi - lo
    if span <= 0:
        raise ParamError("%s can't be changed" % p.name)

    def f(v):
        return parse_amount(reader.text(v))

    a, b = f(lo), f(hi)
    if a is None:
        a = f(lo + span * 1e-3)  # e.g. a filter that shows "Off" at the bottom
    if b is None:
        b = f(hi - span * 1e-3)
    if a is None or b is None:
        raise ParamError("%s doesn't show its value as a number (it shows '%s'); "
                         "set it with a raw value between %g and %g instead"
                         % (p.name, reader.text(lo), lo, hi))
    shown = a[1] or b[1]
    if family and family != shown and not (family == "%" and shown is None):
        raise ParamError("%s is shown in %s, not %s" % (p.name, shown or "plain numbers", family))
    if a[0] == b[0]:
        raise ParamError("%s shows the same value across its range" % p.name)
    increasing = b[0] > a[0]
    if target <= min(a[0], b[0]):
        return lo if increasing else hi
    if target >= max(a[0], b[0]):
        return hi if increasing else lo
    def edge(strict):
        # Raw value where the display crosses the target (strict: first value showing more).
        l, h = float(p.min), float(p.max)
        for _ in range(48):
            mid = (l + h) / 2.0
            m = f(mid)
            if m is None:
                break
            below = m[0] <= target if strict else m[0] < target
            if below == increasing:
                l = mid
            else:
                h = mid
            if h - l < span * 1e-7:
                break
        return (l + h) / 2.0

    # Displays are rounded ("35 %" covers a small range), so aim for the middle of
    # the range that shows the target rather than its edge.
    a_edge, b_edge = edge(False), edge(True)
    return (a_edge + b_edge) / 2.0


def resolve(p, value):
    """Raw number -> clamped raw value. Text ('800 Hz', '-6 dB', 'Saw', 'On') -> raw value."""
    if isinstance(value, bool):
        return float(p.max) if value else float(p.min)
    if isinstance(value, (int, float)):
        return _clamp(p, value)
    text = ("%s" % (value,)).strip()
    if not text:
        raise ParamError("empty value for %s" % p.name)
    quantized = bool(getattr(p, "is_quantized", False))
    if quantized or parse_amount(text) is None:
        try:
            return _choose(p, text)
        except ParamError:
            if parse_amount(text) is None:
                raise
    amount, family = parse_amount(text)
    if quantized:
        return _clamp(p, round(amount))
    return _solve(p, amount, family)


def set_param(p, value):
    old = float(p.value)
    try:
        raw = resolve(p, value)
    except Exception:
        p.value = old  # undo any probing
        raise
    p.value = raw
    return {"parameter": p.name, "value": round(float(p.value), 6), "display": display(p)}
