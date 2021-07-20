import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="the parser")
    parser.add_argument(
        "-r",
        "--raster",
        type=str,
        help="The path to the raster data file",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--vector",
        type=str,
        help="The path to the vector data file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to the output vector data file",
        required=True,
    )

    args = parser.parse_args()
    return args


def main(arguments=None):
    args = parse_arguments()

    print("Working with:")
    print(f"  Raster dataset: {args.raster}")
    print(f"  Vector dataset: {args.vector}")
    print(f"  Output vector dataset: {args.output}")
