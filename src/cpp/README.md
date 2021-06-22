### Minimal working program for intersections lookup (22/06/2021)

- `raster.h`: Declares `Ascii` structure responsible for reading and
  processing raster data, particularly finding intersections with
  `Line2` objects. Depends on:
  - `geom.h`: Defines `geometry` namespace containing geometry objects
    such as `Vec2` and `Line2`
  - `utils.h`: Various utilities, some of which depend on the
    `Exception` class defined in `exceptions.h`.
  - `exceptions.h`: Defines exception class `Exception`.
  
Compile the test with

```shell
g++ test_raster.cpp -o test_raster
```

Usage:

```shell
./test_raster inuncoast_rcp8p5_wtsub_2030_rp1000_0.asc
```
