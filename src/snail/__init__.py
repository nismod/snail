"""snail - the spatial networks impact assessment library
"""
import pkg_resources

# Import things to define what is accessible directly on snail, when a client
# writes::
#   from snail import Something

# e.g. uncomment:
# from snail.network import Network

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    __version__ = "unknown"


# Define what should be imported as * when a client writes::
#   from snail import *
__all__ = []
