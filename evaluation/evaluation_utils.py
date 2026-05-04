import math
import re
from typing import Optional


CLAMP_MIN = -1000.0
CLAMP_MAX = 1000.0

_KV_1PAIR_FLEX = re.compile(
    r'^\s*\{\s*(?:"([^"]+)"|([A-Za-z_]\w*))\s*:\s*(.*?)\s*\}\s*$',
    re.DOTALL,
)
_NUM_PREFIX = re.compile(
    r'^\s*([-+]?\s*(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
)


def clamp(x: float, lo: float = CLAMP_MIN, hi: float = CLAMP_MAX) -> float:
    return lo if x < lo else hi if x > hi else x


def extract_braced(text: str) -> Optional[str]:
    m = re.search(r"\{.*?\}", str(text), flags=re.DOTALL)
    return m.group(0) if m else None


def to_dict_from_braced_kv(block: str) -> Optional[dict[str, float]]:
    m = _KV_1PAIR_FLEX.match(block.strip())
    if not m:
        return None

    key_q, key_u, raw = m.groups()
    key = key_q if key_q is not None else key_u
    raw = raw.strip()

    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1].strip()

    mnum = _NUM_PREFIX.match(raw)
    if not mnum:
        return None

    num_str = mnum.group(1).replace(" ", "")
    return {str(key): float(num_str)}


def extract_property_value(
    text: str,
    *,
    clamp_min: float = CLAMP_MIN,
    clamp_max: float = CLAMP_MAX,
) -> Optional[float]:
    braced = extract_braced(text)
    if braced is None:
        return None

    kv_dict = to_dict_from_braced_kv(braced)
    if kv_dict is None:
        return None

    value = float(next(iter(kv_dict.values())))
    if not math.isfinite(value):
        return None
    return clamp(value, clamp_min, clamp_max)
