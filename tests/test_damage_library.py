import importlib
from importlib import resources
import importlib.util
import sys
from pathlib import Path

import numpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
for key in list(sys.modules.keys()):
    if key.startswith("snail"):
        sys.modules.pop(key)

snail_spec = importlib.util.spec_from_file_location(
    "snail",
    SRC / "snail" / "__init__.py",
    submodule_search_locations=[str(SRC / "snail")],
)
snail_module = importlib.util.module_from_spec(snail_spec)
sys.modules["snail"] = snail_module
assert snail_spec.loader is not None
snail_spec.loader.exec_module(snail_module)

damage_library = importlib.import_module("snail.damage_library")
PiecewiseLinearDamageCurve = importlib.import_module(
    "snail.damages"
).PiecewiseLinearDamageCurve


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
