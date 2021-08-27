"""Pre-process hazard data

Purpose
-------

Convert GeoTiff raster hazard datasets to shapefiles based on masking and selecting values from
    - Single-band raster files

Input data requirements
-----------------------

1. Correct paths to all hazard datasets
2. Single-band GeoTiff hazard raster files with:
    - values - between 0 and 1000
    - raster grid geometry
    - projection systems: Default assumed = EPSG:4326

Results
-------

1. Shapefiles whose names show the hazard models and their selected range of values
    - ID - equal to 1
    - geometry - Shapely Polygon outline of selected hazard

"""

import os
import subprocess
import json
import sys

import fiona
import fiona.crs
import rasterio
import numpy as np
import pandas as pd


def check_files(root_dir):
    """Check if the shapefile was created"""
    f_all_tif = []
    f_all_shp = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".tif") or file.endswith(".tiff"):
                f_all_tif.append(file.split(".tif")[0])
            elif file.endswith(".shp"):
                f_all_shp.append(file.split(".shp")[0])

    for f in f_all_tif:
        f_exists = [file for file in f_all_shp if f in file]
        if len(f_exists) == 0:
            print("No file", f)


def glofris_data_details(root_dir):
    """Read names of GLOFRIS files and create attributes
    Based on the description of the data here:
    http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/index.html
    Latest version - Octboer 20, 2020


    Parameters
        - root_dir - String path to directory of file

    Outputs
        df - Pandas DataFrame written to csv file with columns:
            - file_name - String
            - hazard_type - String: Coastal or Fluvial flooding
            - year - Integer: 2018, 2030, 2050, 2080
            - climate_scenario - String: RCP4.5 or RCP8.5 or historical
            - model - String: Name of climate model
            - subsistence - String - For coastal flooding
            - percentile - String - For coastal flooding
            - probability - Float: 1/(return period)
    """
    f_all = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".tif") or file.endswith(".tiff"):
                fname = file.split(".tif")[0]
                # print (fname)
                percentile = "None"
                subsistence = "None"
                if "inuncoast" in fname:
                    flood_type = "coastal flooding"
                    model = "Coastal"
                    if "0_perc_05" in fname:
                        percentile = "5th"
                    elif "0_perc_50" in fname:
                        percentile = "50th"
                    else:
                        percentile = "95th"

                    if "wtsub" in fname:
                        subsistence = "with subsistence"
                    elif "nosub" in fname:
                        subsistence = "no subsistence"

                elif "inunriver" in fname:
                    flood_type = "fluvial flooding"
                    model = fname.split("_")[2]

                if "2030" in fname:
                    year = 2030
                elif "2050" in fname:
                    year = 2050
                elif "2080" in fname:
                    year = 2080
                else:
                    year = 2018
                if "rcp4p5" in fname:
                    sc = "rcp 4.5"
                elif "rcp8p5" in fname:
                    sc = "rcp 8.5"
                else:
                    sc = "historic"

                rp = fname.split("_rp")[-1]
                if "_" in rp:
                    rp = float(rp.split("_")[0])
                else:
                    rp = float(rp)
                f_all.append(
                    (
                        fname,
                        flood_type,
                        year,
                        sc,
                        model,
                        subsistence,
                        percentile,
                        1.0 / rp,
                    )
                )

    df = pd.DataFrame(
        f_all,
        columns=[
            "file_name",
            "hazard_type",
            "year",
            "climate_scenario",
            "model",
            "subsistence",
            "percentile",
            "probability",
        ],
    )
    df.to_csv(os.path.join(root_dir, "glofris_files.csv"), index=False)


def fathom_data_details(root_dir):
    """Read names of FATHOM files and create attributes
    Based on data received for our project

    Parameters
        - root_dir - String path to directory of file

    Outputs
        df - Pandas DataFrame written to csv file with columns:
            - file_name - String
            - hazard_type - String - Fluvial or Pluvial flooding
            - year - Integer: 2016 or 2050
            - climate_scenario - String: Baseline or Futue Med or Future High
            - model - String - FATHOM
            - probability - Float: 1/(return period)
    """
    model = "FATHOM"
    f_all = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # print (root,dirs,file)
            if file.endswith(".tif") or file.endswith(".tiff"):
                # fname = file.split('.tif')[0].split('1in')
                if "fluvial" in root:
                    flood_type = "fluvial flooding"
                elif "pluvial" in root:
                    flood_type = "pluvial flooding"
                if "baseline" in root.lower().strip():
                    climate_scenario = "Baseline"
                    year = 2018
                elif "future_med" in root.lower().strip():
                    climate_scenario = "Future Med"
                    year = 2050
                elif "future_high" in root.lower().strip():
                    climate_scenario = "Future High"
                    year = 2050

                f_all.append(
                    (
                        file.split(".tif")[0],
                        flood_type,
                        year,
                        climate_scenario,
                        model,
                        1.0 / float(file.split(".tif")[0].split("1in")[-1]),
                    )
                )

    df = pd.DataFrame(
        f_all,
        columns=[
            "file_name",
            "hazard_type",
            "year",
            "climate_scenario",
            "model",
            "probability",
        ],
    )
    df.to_csv(os.path.join(root_dir, "fathom_files.csv"), index=False)


def raster_rewrite(in_raster, out_raster, nodata):
    """Rewrite a raster to reproject and change no data value

    Parameters
        - in_raster - String name of input GeoTff file path
        - out_raster - String name of output GeoTff file path
        - nodata - Float value of data that is treated as no data

    Outputs
        Reproject and replace raster with nodata = -1
    """
    with rasterio.open(in_raster) as dataset:
        data_array = dataset.read()
        data_array[np.where(np.isnan(data_array))] = nodata

        with rasterio.open(
            out_raster,
            "w",
            driver="GTIff",
            height=data_array.shape[1],  # numpy of rows
            width=data_array.shape[2],  # number of columns
            count=dataset.count,  # number of bands
            dtype=data_array.dtype,  # this must match the dtype of our array
            crs=dataset.crs,
            transform=dataset.transform,
        ) as out_data:
            out_data.write(
                data_array
            )  # optional second parameter is the band number to write to
            out_data.nodata = -1  # set the raster's nodata value

    os.remove(in_raster)
    os.rename(out_raster, in_raster)


def raster_projections_and_databands(file_path):
    """Extract projection, data bands numbers and valuees from raster

    Parameters
        - file_path - String name of input GeoTff file path

    Outputs
        - counts - Number of bans in raster
        - crs - Projection system of raster
        - data_vals - Numpy array of raster values
    """
    with rasterio.open(file_path) as dataset:
        counts = dataset.count
        if dataset.crs:
            crs = dataset.crs.to_string()
        else:
            crs = "invalid/unknown"

    return counts, crs


def convert_geotiff_to_vector_with_threshold(
    from_threshold,
    to_threshold,
    infile,
    infile_epsg,
    tmpfile_1,
    tmpfile_2,
    outfile,
):
    """Convert GeoTiff raster file to Shapefile with geometries based on raster threshold ranges

    Parameters
        - from_threshold - Float value of lower bound of GeoTiff threshold value to be selected
        - to_threshold - Float value of upper bound of GeoTiff threshold value to be selected
        - infile - String name of input GeoTff file path
        - infile_epsg - Integer value of EPSG Projection number of raster
        - tmpfile_1 - Stirng name of tmp file 1
        - tmpfile_2 - Stirng name of tmp file 2
        - outfile - Stirng name of output shapefile

    Outputs
        Shapefile with Polygon geometries of rasters based on raster threshold ranges
    """
    args = [
        "gdal_calc.py",
        "-A",
        infile,
        "--outfile={}".format(tmpfile_1),
        "--calc=logical_and(A>={0}, A<{1})".format(
            from_threshold, to_threshold
        ),
        "--type=Byte",
        "--NoDataValue=0",
        "--co=SPARSE_OK=YES",
        "--co=NBITS=1",
        "--quiet",
        "--co=COMPRESS=LZW",
    ]
    subprocess.run(args)

    subprocess.run(
        ["gdal_edit.py", "-a_srs", "EPSG:{}".format(infile_epsg), tmpfile_1]
    )

    subprocess.run(
        [
            "gdal_polygonize.py",
            tmpfile_1,
            "-q",
            "-f",
            "ESRI Shapefile",
            tmpfile_2,
        ]
    )

    subprocess.run(
        [
            "ogr2ogr",
            "-a_srs",
            "EPSG:{}".format(infile_epsg),
            "-t_srs",
            "EPSG:4326",
            outfile,
            tmpfile_2,
        ]
    )

    subprocess.run(["rm", tmpfile_1])
    subprocess.run(["rm", tmpfile_2])
    subprocess.run(["rm", tmpfile_2.replace("shp", "shx")])
    subprocess.run(["rm", tmpfile_2.replace("shp", "dbf")])
    subprocess.run(["rm", tmpfile_2.replace("shp", "prj")])


def convert_geotiff_to_vector_with_multibands(
    band_colors, infile, infile_epsg, tmpfile_1, tmpfile_2, outfile
):
    """Convert multi-band GeoTiff raster file to Shapefile with geometries based on raster band color values

    Parameters
        - band_colors - Tuple with 3-values each corresponding to the values in raster bands
        - infile - String name of input GeoTff file path
        - infile_epsg - Integer value of EPSG Projection number of raster
        - tmpfile_1 - Stirng name of tmp file 1
        - tmpfile_2 - Stirng name of tmp file 2
        - outfile - Stirng name of output shapefile

    Outputs
        Shapefile with Polygon geometries of rasters based on raster band values
    """
    args = [
        "gdal_calc.py",
        "-A",
        infile,
        "--A_band=1",
        "-B",
        infile,
        "--B_band=2",
        "-C",
        infile,
        "--C_band=3",
        "--outfile={}".format(tmpfile_1),
        "--type=Byte",
        "--NoDataValue=0",
        "--calc=logical_and(A=={0}, B=={1},C=={2})".format(
            band_colors[0], band_colors[1], band_colors[2]
        ),
        "--co=SPARSE_OK=YES",
        "--co=NBITS=1",
        "--quiet",
        "--co=COMPRESS=LZW",
    ]
    subprocess.run(args)

    subprocess.run(
        ["gdal_edit.py", "-a_srs", "EPSG:{}".format(infile_epsg), tmpfile_1]
    )

    subprocess.run(
        [
            "gdal_polygonize.py",
            tmpfile_1,
            "-q",
            "-f",
            "ESRI Shapefile",
            tmpfile_2,
        ]
    )

    subprocess.run(
        [
            "ogr2ogr",
            "-a_srs",
            "EPSG:{}".format(infile_epsg),
            "-t_srs",
            "EPSG:4326",
            outfile,
            tmpfile_2,
        ]
    )

    subprocess.run(["rm", tmpfile_1])
    subprocess.run(["rm", tmpfile_2])
    subprocess.run(["rm", tmpfile_2.replace("shp", "shx")])
    subprocess.run(["rm", tmpfile_2.replace("shp", "dbf")])
    subprocess.run(["rm", tmpfile_2.replace("shp", "prj")])


def convert(threshold, infile, tmpfile_1, outfile):
    """Convert GeoTiff raster file to Shapefile with geometries based on raster threshold less that 999

    Parameters
        - threshold - Float value of lower bound of GeoTiff threshold value to be selected
        - infile - String name of input GeoTff file path
        - tmpfile_1 - Stirng name of tmp file 1
        - outfile - Stirng name of output shapefile

    Outputs
        Shapefile with Polygon geometries of rasters based on raster values above a threshold
    """
    args = [
        "gdal_calc.py",
        "-A",
        infile,
        "--outfile={}".format(tmpfile_1),
        "--calc=logical_and(A>={}, A<999)".format(threshold),
        "--type=Byte",
        "--NoDataValue=0",
        "--co=SPARSE_OK=YES",
        "--co=NBITS=1",
        "--co=COMPRESS=LZW",
    ]
    subprocess.run(args)

    subprocess.run(
        [
            "gdal_polygonize.py",
            tmpfile_1,
            "-q",
            "-f",
            "ESRI Shapefile",
            outfile,
        ]
    )


def hazard_conversion(
    thresholds, thresholds_label, root_dir, glofris=False, fathom=False
):
    """Process hazard data

    1. Specify the paths from where to read and write:
        - Input data
        - Hazard data

    2. Supply input data and parameters
        - Thresholds of flood hazards
        - Values of bands to be selected
        - Color code of multi-band rasters
        - Specific file names that might require some specific operations
    """

    if glofris is True:
        glofris_data_details(
            root_dir
        )  # This will write the names of the glofris files and their description in 1 csv file
    if fathom is True:
        fathom_data_details(
            root_dir
        )  # This will write the names of the glofris files and their description in 1 csv file

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".tif") or file.endswith(".tiff"):
                band_nums, crs = raster_projections_and_databands(
                    os.path.join(root, file)
                )
                print(root, file, band_nums, crs)
                if "epsg" in crs:
                    crs_split = crs.split(":")
                    s_crs = [int(c) for c in crs_split if c.isdigit() is True][
                        0
                    ]
                else:
                    s_crs = 4326

                # threshold based datasets
                for t in range(len(thresholds) - 1):
                    thr_1 = thresholds[t]
                    thr_2 = thresholds[t + 1]
                    in_file = os.path.join(root, file)
                    tmp_1 = os.path.join(
                        root, file.split(".tif")[0] + "_mask.tiff"
                    )
                    tmp_2 = os.path.join(
                        root, file.split(".tif")[0] + "_mask.shp"
                    )
                    out_file = os.path.join(
                        root,
                        file.split(".tif")[0]
                        + "_{0}-{1}_threshold.shp".format(
                            thresholds_label[t], thresholds_label[t + 1]
                        ),
                    )
                    convert_geotiff_to_vector_with_threshold(
                        thr_1, thr_2, in_file, s_crs, tmp_1, tmp_2, out_file
                    )
