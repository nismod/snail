"""snail - the spatial networks impact assessment library"""

from importlib.metadata import PackageNotFoundError, version

# Import things to define what is accessible directly on snail, when a client
# writes::
#   from snail import Something
# e.g. uncomment:
# from snail.network import Network
from snail import damage_library
from snail.damages import (
    DirectDamageArrays,
    DirectDamageResult,
    calculate_direct_damages,
    expected_annual_damage,
    simulate_direct_damages,
    summarise_samples,
)
from snail.overlay import overlay_raster, overlay_rasters, split_features

try:
    __version__ = version("nismod-snail")
except PackageNotFoundError:
    __version__ = "unknown"


# Define what should be imported as * when a client writes::
#   from snail import *
__all__ = [
    "DirectDamageArrays",
    "DirectDamageResult",
    "calculate_direct_damages",
    "damage_library",
    "expected_annual_damage",
    "overlay_raster",
    "overlay_rasters",
    "simulate_direct_damages",
    "split_features",
    "summarise_samples",
]
