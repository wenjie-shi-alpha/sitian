"""Deterministic units and labels used when constructing forecast evidence."""
from __future__ import annotations

import math


def wind_direction_label(degrees: float) -> str:
    """Nearest of eight compass directions for a meteorological FROM bearing.

    North is 0/360 degrees, east is 90. Half-sector ties go clockwise.
    """
    value = float(degrees)
    if not math.isfinite(value):
        raise ValueError("wind direction must be finite")
    return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[
        int(((value % 360) + 22.5) // 45) % 8
    ]
