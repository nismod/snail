"""snail - the spatial networks impact assessment library"""

from importlib.metadata import PackageNotFoundError, version

# Import things to define what is accessible directly on snail, when a client
# writes::
#   from snail import Something
# e.g. uncomment:
# from snail.network import Network
from snail import damage_library
from snail.overlay import (
    iter_overlay_raster_batches,
    iter_overlay_rasters_batches,
    iter_split_features_batches,
    overlay_raster,
    overlay_rasters,
    split_features,
)

try:
    __version__ = version("nismod-snail")
except PackageNotFoundError:
    __version__ = "unknown"


# Define what should be imported as * when a client writes::
#   from snail import *
__all__ = [
    "damage_library",
    "iter_overlay_raster_batches",
    "iter_overlay_rasters_batches",
    "iter_split_features_batches",
    "overlay_raster",
    "overlay_rasters",
    "split_features",
]
