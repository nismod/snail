# overlay all rasters with each features layer
snail -vv process -fs features.csv -rs rasters.csv

# split with different output formats
snail -vv split -f lines.geojson -r inunriver_historical_WATCH_1980_rp01000.tif -c inunriver_historical_WATCH_1980_rp01000 -b 1 -a -o tmp.geojson
snail -vv split -f lines.geojson -r inunriver_historical_WATCH_1980_rp01000.tif -c inunriver_historical_WATCH_1980_rp01000 -b 1 -a -o tmp.parquet

# error missing columns
snail -vv process -fs empty.txt -rs empty.txt
snail -vv process -fs empty.txt -rs empty.txt -c other

# error with missing "output_path"
snail -vv process -fs features_error.csv -rs rasters.csv

# success with "splits" column
snail -vv process -fs features_alt_path.csv -rs rasters.csv -c splits
