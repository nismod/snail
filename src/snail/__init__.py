"""snail - the spatial networks impact assessment library
"""

# Import things to define what is accessible directly on snail, when a client
# writes::
#   from snail import Something

# e.g. uncomment:
# from snail.network import Network

try:
    from importlib.metadata import version

    __version__ = version("nismod-snail")
except Exception:
    __version__ = "unknown"


# Define what should be imported as * when a client writes::
#   from snail import *
__all__ = []
