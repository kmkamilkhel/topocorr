"""Bits of the ``tc_flag`` band.

Flagged pixels are reported, not corrected: pixels in cast shadow, in self-shadow or on very
steep slopes keep their input value. Low illumination is a warning only; those pixels are
corrected. "No fit" is set on every pixel of an image in which the C or SCS+C fit of at least one
band could not be used; those bands are returned unchanged (see ``tc_fit_ok_<band>``). With the
stack methods, "no fit" marks the pixels of a stratum whose per-image fit (``se-stack``) or c
(``scsc-stack``) could not be used in at least one band, and "no model" the pixels whose stratum
the stack model does not cover; both keep their input value in those bands.
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
NO_MODEL = 32
"""Stack methods: the pixel's stratum is not in the model or not assessable; not corrected."""

FLAGS = {
    "cast_shadow": CAST_SHADOW,
    "self_shadow": SELF_SHADOW,
    "steep": STEEP,
    "no_fit": NO_FIT,
    "low_illumination": LOW_ILLUMINATION,
    "no_model": NO_MODEL,
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
