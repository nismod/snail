"""Damage assessment"""
from abc import ABC
import numpy
import pandas
import pandera
from scipy.interpolate import interp1d
from pandera.typing import DataFrame, Series


class DamageCurve(ABC):
    """A damage curve"""

    def __init__(self):
        pass

    def damage_fraction(exposure: numpy.array) -> numpy.array:
        """Evaluate damage fraction for exposure to a given hazard intensity"""
        pass


class LinearDamageCurveSchema(pandera.DataFrameModel):
    intensity: Series[float]
    damage: Series[float]


class LinearDamageCurve(DamageCurve):
    """A piecewise-linear damage curve"""

    def __init__(self, curve: DataFrame[LinearDamageCurveSchema]):
        curve = curve.copy()
        self.intensity, self.damage = self.clip_curve_data(
            curve.intensity, curve.damage
        )

        bounds = (self.damage.min(), self.damage.max())
        self._interpolate = interp1d(
            self.intensity,
            self.damage,
            kind="linear",
            fill_value=bounds,
            bounds_error=False,
            copy=False,
        )

    def damage_fraction(self, exposure: numpy.array) -> numpy.array:
        """Evaluate damage fraction for exposure to a given hazard intensity"""
        return self._interpolate(exposure)

    def translate_y(self, y: float):
        damage = self.damage + y

        return LinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": self.intensity,
                    "damage": damage,
                }
            )
        )

    def scale_y(self, y: float):
        damage = self.damage * y

        return LinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": self.intensity,
                    "damage": damage,
                }
            )
        )

    def translate_x(self, x: float):
        intensity = self.intensity + x

        return LinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": intensity,
                    "damage": self.damage,
                }
            )
        )

    def scale_x(self, x: float):
        intensity = self.intensity * x

        return LinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": intensity,
                    "damage": self.damage,
                }
            )
        )

    @staticmethod
    def clip_curve_data(intensity, damage):
        if (damage.max() > 1) or (damage.min() < 0):
            # WARNING clipping out-of-bounds damage fractions
            bounds = (
                intensity.min(),
                intensity.max(),
            )
            inverse = interp1d(
                damage,
                intensity,
                kind="linear",
                fill_value=bounds,
                bounds_error=False,
            )

        if damage.max() > 1:
            one_intercept = inverse(1)
            idx = numpy.searchsorted(intensity, one_intercept)
            # if one_intercept is in our intensities
            if intensity[idx] == one_intercept:
                # no action - damage fraction will be set to 1
                pass
            else:
                # else insert new interpolation point
                damage = numpy.insert(damage, idx, 1)
                intensity = numpy.insert(intensity, idx, one_intercept)

        if damage.min() < 0:
            zero_intercept = inverse(0)
            idx = numpy.searchsorted(intensity, zero_intercept)
            # if zero_intercept is in our intensities
            if intensity[idx] == zero_intercept:
                # no action - damage fraction will be set to 0
                pass
            else:
                # else insert new interpolation point
                damage = numpy.insert(damage, idx, 0)
                intensity = numpy.insert(intensity, idx, zero_intercept)

        damage = numpy.clip(damage, 0, 1)

        return intensity, damage

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        ax.plot(self.intensity, self.damage, color="tab:blue")
        ax.set_ylim([0, 1])
        ax.set_ylabel("Damage Fraction")
        ax.set_xlabel("Hazard Intensity")

        return ax


# set thresholds - see Raghav code
