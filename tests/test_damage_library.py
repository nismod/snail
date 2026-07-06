import importlib
from importlib import resources
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


@contextmanager
def source_snail_modules():
    original_path = list(sys.path)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "snail" or name.startswith("snail.")
    }
    try:
        sys.path.insert(0, str(SRC))
        for name in list(sys.modules):
            if name == "snail" or name.startswith("snail."):
                sys.modules.pop(name)
        yield (
            importlib.import_module("snail.damage_library"),
            importlib.import_module("snail.damages").PiecewiseLinearDamageCurve,
        )
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name == "snail" or name.startswith("snail."):
                sys.modules.pop(name)
        sys.modules.update(original_modules)


@pytest.fixture
def damage_modules():
    with source_snail_modules() as modules:
        yield modules


def test_packaged_curve_resources_are_available(damage_modules):
    _, _ = damage_modules
    data_root = resources.files("snail.data.damage_curves")

    assert (data_root / "metadata.csv").is_file()
    assert (data_root / "curves.csv").is_file()
    assert (data_root / "NOTICE").is_file()


def test_available_curves_filtering(damage_modules):
    damage_library, _ = damage_modules
    curves = damage_library.available_curves(hazard="flood", sector="Energy")
    assert not curves.empty
    assert (curves["hazard_name"] == "flood").all()
    assert (curves["sector"] == "Energy").all()


def test_available_curves_excludes_repair_and_fault_rates(damage_modules):
    damage_library, _ = damage_modules
    curves = damage_library.available_curves()

    assert "V (repair rate)" not in set(curves["curve_type"])
    assert "V (faults/km)" not in set(curves["curve_type"])


def test_load_curve_returns_piecewise_curve(damage_modules):
    damage_library, PiecewiseLinearDamageCurve = damage_modules
    curve = damage_library.load_curve("F1.1")
    assert isinstance(curve, PiecewiseLinearDamageCurve)

    intensities = numpy.array([0.0, 0.1, 0.2])
    expected = numpy.array([0.0, 0.008202, 0.016404])

    numpy.testing.assert_allclose(
        curve.damage_fraction(intensities),
        expected,
        atol=1e-6,
    )


def test_get_metadata_content(damage_modules):
    damage_library, _ = damage_modules
    metadata = damage_library.get_metadata("F1.1")

    assert metadata.hazard_name == "flood"
    assert metadata.sector == "Energy"
    assert metadata.exposed_element.startswith("Small power plants")
    assert metadata.intensity_axis.startswith("Depth")
    assert metadata.intensity_unit == "m"


def test_repair_rate_curve_is_not_packaged(damage_modules):
    damage_library, _ = damage_modules
    with pytest.raises(KeyError):
        damage_library.load_curve("E16.35")
