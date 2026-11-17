import logging
import zipfile

import requests

logger = logging.getLogger(__name__)


def download_file(url, output_path, extract=False):
    """Download a file from url to output_path, extract if True and the file
    name has a .zip suffix
    """
    logger.info("Checking output %s", output_path)

    output_path.parent.mkdir(exist_ok=True, parents=True)

    if not output_path.exists():
        logger.info("Downloading from %s", url)
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

    if extract and output_path.suffix == ".zip":
        logger.info("Extracting from %s", output_path)
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(output_path.parent)
