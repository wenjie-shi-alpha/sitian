import pytest

from sitian.meteorology import wind_direction_label


@pytest.mark.parametrize("angle,label", [
    (0, "N"), (45, "NE"), (90, "E"), (135, "SE"), (180, "S"),
    (225, "SW"), (270, "W"), (315, "NW"), (360, "N"), (-45, "NW"),
    (22.49, "N"), (22.5, "NE"), (337.49, "NW"), (337.5, "N"),
])
def test_compass_bearings(angle, label):
    assert wind_direction_label(angle) == label


@pytest.mark.parametrize("angle", [float("nan"), float("inf"), -float("inf")])
def test_invalid_bearing_is_rejected(angle):
    with pytest.raises(ValueError, match="finite"):
        wind_direction_label(angle)
