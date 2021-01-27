"""Exposure analysis

Calculate the intersection of hazards and infrastructure networks.
"""
import csv
import glob
import json
import os

import geopandas
from shapely.wkt import dumps
from tqdm import tqdm


def main():
    """Read files and do intersections
    """
    config = load_config()
    base_path = config['base_path']
    epsg_code = 32630 # 32630 is UTM Zone 30N, suitable for Ghana

    # Setting up the paths
    hazard_paths = glob.glob(
        os.path.join(base_path, 'data', 'hazards','flood', '*.gpkg'))

    network_paths = glob.glob(
        os.path.join(base_path, 'data', 'nature','Polygons', '*.gpkg'))

    for hazard_path in hazard_paths:
        hazard_id = os.path.basename(hazard_path).replace(".gpkg", "")

         # Reading hazard outlines
        hazard_df = geopandas.read_file(hazard_path)

        # Convert to projected CRS
        hazard_df = hazard_df.to_crs(epsg=epsg_code)

        for network_path in network_paths:
            network_id = os.path.basename(network_path).replace(".gpkg", "")
        
            print("Processing", hazard_id, network_id)

            # Reading hazard outlines
            network_df = geopandas.read_file(network_path)

            # Convert to projected CRS
            network_df = network_df.to_crs(epsg=epsg_code)

            # Do intersection
            intersections = []
            csv_fname = os.path.join(
                base_path, 'results', 'exposure', f"{network_id}__{hazard_id}.csv")

            with open(csv_fname, 'w', encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=('network_id', 'hazard_id', 'name', 'length', 'geom'))
                w.writeheader()

                for hazard_n, hazard in enumerate(hazard_df.itertuples()):
                    print("considering", hazard_n)
                    if hazard.geometry.geom_type == 'MultiPolygon':
                        geoms = [p for p in hazard.geometry]
                    else:
                        geoms = [hazard.geometry]

                    for hazard_geom in geoms:
                        # Try fixing invalid geometry
                        if not hazard_geom.is_valid:
                            print("fixing", hazard_n)
                            hazard_geom = hazard_geom.buffer(0)
                            print("fixed", hazard_n)

                        # Use spatial index to find candidate network segments
                        potential_networks = network_df.iloc[
                            list(network_df.sindex.intersection(hazard_geom.bounds))]
                        print("found", len(potential_networks), "network")

                        if len(potential_networks):
                            for network in potential_networks.itertuples():
                                print(network.ID, hazard_n)
                                if network.geometry.intersects(hazard_geom):
                                    print("intersects")
                                    intersection_geom = network.geometry.intersection(hazard_geom)
                                    print("done intersection")
                                    w.writerow({
                                        'network_id': network.ID,
                                        'hazard_id': hazard_n,
                                        'name': network.NAME,
                                        'area': intersection_geom.area
                                    })
                                    intersections.append({
                                        'network_id': network.ID,
                                        'hazard_id': hazard_n,
                                        'name': network.NAME,
                                        'area': intersection_geom.area,
                                        'geometry': intersection_geom
                                    })

                            fh.flush()

         # Write intersection data
            fname = os.path.join(
                base_path, 'results', 'exposure', f"{network_id}__{hazard_id}.gpkg")
            intersections_df = geopandas.GeoDataFrame(intersections).set_crs(epsg=epsg_code)
            intersections_df.to_file(fname, driver="GPKG")


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'config.json')
    with open(config_path) as fh:
        config = json.load(fh)
    return config


if __name__ == '__main__':
    main()
