"""Bits of the ``tc_flag`` band.

Flagged pixels are reported, not corrected: pixels in cast shadow, in self-shadow or on very
steep slopes keep their input value. Low illumination is a warning only; those pixels are
corrected. "No fit" is set on every pixel of an image in which the C or SCS+C fit of at least one
band could not be used; those bands are returned unchanged (see ``tc_fit_ok_<band>``).
"""

from __future__ import annotations

CAST_SHADOW = 1
"""Shadow cast by surrounding terrain (and not self-shadowed)."""
SELF_SHADOW = 2
"""The slope faces away from the sun: ``cos i <= 0``."""
STEEP = 4
"""Slope above the steep threshold (60° by default)."""
NO_FIT = 8
"""The C or SCS+C fit of at least one band was not usable; those bands are unchanged."""
LOW_ILLUMINATION = 16
"""Barely lit: ``0 < cos i < 0.1``. Flag only; the pixel is corrected."""

FLAGS = {
    "cast_shadow": CAST_SHADOW,
    "self_shadow": SELF_SHADOW,
    "steep": STEEP,
    "no_fit": NO_FIT,
    "low_illumination": LOW_ILLUMINATION,
}
"""Flag names and their bit values."""

EXCLUDED = CAST_SHADOW | SELF_SHADOW | STEEP
"""Flags whose pixels keep their input value."""


def decode(value: int) -> list[str]:
    """Names of the flags set in a ``tc_flag`` value.

    Examples
    --------
    >>> decode(20)
    ['steep', 'low_illumination']
    >>> decode(0)
    []
    """
    value = int(value)
    if value < 0 or value >= 2 ** len(FLAGS):
        raise ValueError(f"not a tc_flag value: {value}")
    return [name for name, bit in FLAGS.items() if value & bit]
