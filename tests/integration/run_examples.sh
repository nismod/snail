snail -vv process -fs features.csv -rs rasters.csv
snail -vv split -f lines.geojson -r inunriver_historical_WATCH_1980_rp01000.tif -c inunriver_historical_WATCH_1980_rp01000 -b 1 -a -o tmp.geojson
