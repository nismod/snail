import numpy as np
import pytest
from rasterio.transform import from_origin


@pytest.fixture
def sample_dataarray():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("rioxarray")

    data = np.arange(4, dtype=float).reshape(2, 2)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": np.array([10.5, 9.5]),
            "x": np.array([5.5, 6.5]),
        },
        name="hazard",
    )

    transform = from_origin(5.0, 11.0, 1.0, 1.0)
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.write_transform(transform, inplace=True)

    return da, transform
