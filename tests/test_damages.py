"""Test damage assessment"""
import numpy
import pandas
import pytest
from numpy.testing import assert_allclose
from snail.damages import DamageCurve, PiecewiseLinearDamageCurve


@pytest.fixture
def curve():
    curve_data = pandas.DataFrame(
        {"intensity": [0.0, 10, 20, 30], "damage": [0, 0.1, 0.2, 1.0]}
    )
    return PiecewiseLinearDamageCurve(curve_data)


def test_linear_curve(curve):
    # check inheritance
    assert isinstance(curve, DamageCurve)


def test_linear_curve_pass_through(curve):
    # check specified intensities give specified damages
    assert_allclose(curve.damage_fraction(curve.intensity), curve.damage)


def test_linear_curve_interpolation(curve):
    # sense-check interpolation
    intensities = numpy.array([5, 15, 25])
    expected = numpy.array([0.05, 0.15, 0.6])
    actual = curve.damage_fraction(intensities)
    assert_allclose(actual, expected)


def test_linear_curve_out_of_bounds(curve):
    # sense-check out-of-bounds
    intensities = numpy.array(
        [numpy.NINF, -999, numpy.NZERO, 0, 30, 999, numpy.inf, numpy.nan]
    )
    expected = numpy.array([0, 0, 0, 0, 1, 1, 1, numpy.nan])
    actual = curve.damage_fraction(intensities)
    assert_allclose(actual, expected)


def test_linear_curve_translate_y(curve):
    increased = curve.translate_y(0.1)
    expected = numpy.array([0.1, 0.2, 0.3, 1.0, 1.0])
    assert_allclose(increased.damage, expected)

    expected = numpy.array([0, 10, 20, 28.75, 30])
    assert_allclose(increased.intensity, expected)


def test_linear_curve_translate_y_down(curve):
    decreased = curve.translate_y(-0.1)
    expected = numpy.array([0, 0, 0.1, 0.9])
    assert_allclose(decreased.damage, expected)

    expected = numpy.array([0, 10, 20, 30])
    assert_allclose(decreased.intensity, expected)


def test_linear_curve_scale_y(curve):
    increased = curve.scale_y(2)
    expected = numpy.array([0, 0.2, 0.4, 1.0, 1.0])
    assert_allclose(increased.damage, expected)

    expected = numpy.array([0, 10, 20, 23.75, 30])
    assert_allclose(increased.intensity, expected)


def test_linear_curve_scale_y_down(curve):
    decreased = curve.scale_y(0.1)
    expected = numpy.array([0, 0.01, 0.02, 0.1])
    assert_allclose(decreased.damage, expected)

    expected = numpy.array([0, 10, 20, 30])
    assert_allclose(decreased.intensity, expected)


def test_linear_curve_translate_x(curve):
    increased = curve.translate_x(5)
    expected = numpy.array([0, 0.1, 0.2, 1.0])
    assert_allclose(increased.damage, expected)

    expected = numpy.array([5, 15, 25, 35])
    assert_allclose(increased.intensity, expected)


def test_linear_curve_translate_x_down(curve):
    decreased = curve.translate_x(-5)
    expected = numpy.array([0, 0.1, 0.2, 1.0])
    assert_allclose(decreased.damage, expected)

    expected = numpy.array([-5, 5, 15, 25])
    assert_allclose(decreased.intensity, expected)


def test_linear_curve_scale_x(curve):
    increased = curve.scale_x(2)
    expected = numpy.array([0, 0.1, 0.2, 1.0])
    assert_allclose(increased.damage, expected)

    expected = numpy.array([0, 20, 40, 60])
    assert_allclose(increased.intensity, expected)


def test_linear_curve_scale_x_down(curve):
    decreased = curve.scale_x(0.1)
    expected = numpy.array([0, 0.1, 0.2, 1.0])
    assert_allclose(decreased.damage, expected)

    expected = numpy.array([0, 1, 2, 3])
    assert_allclose(decreased.intensity, expected)
