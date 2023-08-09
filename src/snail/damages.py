"""Damage assessment"""
from abc import ABC
import numpy
import pandas
import pandera
from scipy.interpolate import interp1d
from pandera.typing import DataFrame, Series


# TODO check `nismod/east-africa-transport` and `nismod/jamaica-infrastructure`
# manipulations of damage curves


class DamageCurve(ABC):
    """A damage curve"""

    def __init__(self):
        pass

    def damage_fraction(exposure: numpy.array) -> numpy.array:
        """Evaluate damage fraction for exposure to a given hazard intensity"""
        pass


class PiecewiseLinearDamageCurveSchema(pandera.DataFrameModel):
    intensity: Series[float]
    damage: Series[float]


class PiecewiseLinearDamageCurve(DamageCurve):
    """A piecewise-linear damage curve"""

    intensity: Series[float]
    damage: Series[float]

    def __init__(self, curve: DataFrame[PiecewiseLinearDamageCurveSchema]):
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

    def __eq__(self, other):
        damage_eq = (self.damage == other.damage).all()
        intensity_eq = (self.intensity == other.intensity).all()
        return damage_eq and intensity_eq

    @classmethod
    def from_csv(
        cls,
        fname,
        intensity_col="intensity",
        damage_col="damage_ratio",
        comment="#",
        **kwargs,
    ):
        """Read a damage curve from a CSV file.

        By default, the CSV should have columns named "intensity" and "damage_ratio",
        with any additional header lines commented out by "#".

        Any additional keyword arguments are passed through to ``pandas.read_csv``

        Parameters
        ----------
        fname: str, path object or file-like object
        intensity_col: str, default "intensity"
            Column name to read hazard intensity values
        damage_col: str, default "damage_ratio"
            Column name to read damage values
        comment: str, default "#"
            Indicates remainder of the line in the CSV should not be parsed.
            If found at the beginning of a line, the line will be ignored
            altogether.
        kwargs:
            see pandas.read_csv documentation

        Returns
        -------
        PiecewiseLinearDamageCurve
        """
        curve_data = pandas.read_csv(fname, comment=comment, **kwargs).rename(
            columns={
                intensity_col: "intensity",
                damage_col: "damage",
            }
        )
        return cls(curve_data)

    @classmethod
    def from_excel(
        cls,
        fname,
        sheet_name=0,
        intensity_col="intensity",
        damage_col="damage_ratio",
        comment="#",
        **kwargs,
    ):
        """Read a damage curve from an Excel file.

        By default, the file should have columns named "intensity" and "damage_ratio",
        with any additional header lines commented out by "#".

        Any additional keyword arguments are passed through to ``pandas.read_excel``

        Parameters
        ----------
        fname: str, path object or file-like object
        sheet_name: str, int
            Strings are used for sheet names. Integers are used in zero-indexed sheet
            positions (chart sheets do not count as a sheet position).
        intensity_col: str, default "intensity"
            Column name to read hazard intensity values
        damage_col: str, default "damage_ratio"
            Column name to read damage values
        comment: str, default "#"
            Indicates remainder of the line in the CSV should not be parsed.
            If found at the beginning of a line, the line will be ignored
            altogether.
        kwargs:
            see pandas.read_csv documentation

        Returns
        -------
        PiecewiseLinearDamageCurve
        """
        curve_data = pandas.read_excel(
            fname, sheet_name=sheet_name, comment=comment, **kwargs
        ).rename(
            columns={
                intensity_col: "intensity",
                damage_col: "damage",
            }
        )
        return cls(curve_data)

    @classmethod
    def interpolate(
        cls,
        a,
        b,
        factor: float,
    ):
        """Interpolate damage values between two curves

        ```
        new_curve_damage = a_damage + ((b_damage - a_damage) * factor)
        ```

        Parameters
        ----------
        a: PiecewiseLinearDamageCurve
        b: PiecewiseLinearDamageCurve
        factor: float
            Interpolation factor, used to calculate the new curve

        Returns
        -------
        PiecewiseLinearDamageCurve
        """
        # find the sorted, unique values of intensity used by both curves
        intensity = numpy.unique(numpy.concatenate((a.intensity, b.intensity)))
        # calculate the damage fraction at each intensity
        a_damage = a.damage_fraction(intensity)
        b_damage = b.damage_fraction(intensity)
        # interpolate at each point
        damage = a_damage + ((b_damage - a_damage) * factor)
        return cls(
            pandas.DataFrame(
                {
                    "intensity": intensity,
                    "damage": damage,
                }
            )
        )

    def damage_fraction(self, exposure: numpy.array) -> numpy.array:
        """Evaluate damage fraction for exposure to a given hazard intensity"""
        return self._interpolate(exposure)

    def translate_y(self, y: float):
        damage = self.damage + y

        return PiecewiseLinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": self.intensity,
                    "damage": damage,
                }
            )
        )

    def scale_y(self, y: float):
        damage = self.damage * y

        return PiecewiseLinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": self.intensity,
                    "damage": damage,
                }
            )
        )

    def translate_x(self, x: float):
        intensity = self.intensity + x

        return PiecewiseLinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": intensity,
                    "damage": self.damage,
                }
            )
        )

    def scale_x(self, x: float):
        intensity = self.intensity * x

        return PiecewiseLinearDamageCurve(
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
