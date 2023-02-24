# Designing example usage of `snail` command

snail -vv split \
    -f network_shp/Supply\ network_link.SHP \
    -r world-bank-climate-knowledge-portal/climatology-hd35-annual-mean_cmip6_annual_all-regridded-bct-historical-climatology_median_1995-2014.tif \
    -a \
    -o sample.gpkg

snail -vv split \
    -f tests/integration/lines.geojson \
    -r tests/integration/climatology-hd35-annual-mean.tif \
    -a \
    -o tmp.geojson

snail -vv split \
    -f tests/integration/lines.geojson \
    -r tests/integration/range.tif \
    -a \
    -o tmp.geojson

snail -vv split \
    -f tests/integration/lines.geojson \
    -t 0.0001 0 -1.341666 0 -0.0001 51.8083333 \
    --width 10000 \
    --height 10000 \
    -o tmp.geojson

# how do we handle multiple bands?
# - default to all, add on columns
# - option to pick

# parquet support for input features, output file

# specify output column name
# --output_column_name -ocn
# default to filename/band number


snail -vv \
    process \
    -fs network.csv \
    -rs details_fathom.csv \
    -d .

snail -vv \
    process \
    --features network.csv \
    --rasters details_heat.csv \
    --directory .

# csv
# - band column to pick from raster  `band`   - band column optional, if present and a value is not numeric >0, error
# - output labelling  `output_column_name`
# - output_path column - parquet or other geo


# optional handling of out-of-raster splits
# assign nodata, warn or allow for index <0 >width/height
