from importlib import resources

import numpy
import pytest

from snail import damage_library
from snail.damages import PiecewiseLinearDamageCurve


def test_packaged_curve_resources_are_available():
    data_root = resources.files("snail.data.damage_curves")

    assert (data_root / "metadata.csv").is_file()
    assert (data_root / "curves.csv").is_file()
    assert (data_root / "NOTICE").is_file()


def test_available_curves_filtering():
    curves = damage_library.available_curves(hazard="flood", sector="Energy")
    assert not curves.empty
    assert (curves["hazard_name"] == "flood").all()
    assert (curves["sector"] == "Energy").all()


def test_available_curves_excludes_repair_and_fault_rates():
    curves = damage_library.available_curves()

    assert "V (repair rate)" not in set(curves["curve_type"])
    assert "V (faults/km)" not in set(curves["curve_type"])


def test_load_curve_returns_piecewise_curve():
    curve = damage_library.load_curve("F1.1")
    assert isinstance(curve, PiecewiseLinearDamageCurve)

    intensities = numpy.array([0.0, 0.1, 0.2])
    expected = numpy.array([0.0, 0.008202, 0.016404])

    numpy.testing.assert_allclose(
        curve.damage_fraction(intensities),
        expected,
        atol=1e-6,
    )


def test_get_metadata_content():
    metadata = damage_library.get_metadata("F1.1")

    assert metadata.hazard_name == "flood"
    assert metadata.sector == "Energy"
    assert metadata.exposed_element.startswith("Small power plants")
    assert metadata.intensity_axis.startswith("Depth")
    assert metadata.intensity_unit == "m"


def test_repair_rate_curve_is_not_packaged():
    with pytest.raises(KeyError):
        damage_library.load_curve("E16.35")
