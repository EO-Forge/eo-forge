# -*- coding: utf-8 -*-
import glob
import os
from datetime import datetime

from lxml import etree

from eo_forge.utils.raster_utils import (
    get_is_valid_mask,
)
from eo_forge.utils.sentinel import (
    calibrate_sentinel2,
    SENTINEL2_BANDS_RESOLUTION,
    SENTINEL2_SUPPORTED_RESOLUTIONS,
)
from eo_forge.io.GenLoader import BaseLoaderTask

######################################################################
class Sentinel2Loader(BaseLoaderTask):
    """Task for importing Sentinel SAFE data into a Single Raster (and Clouds File)."""

    _supported_resolutions = SENTINEL2_SUPPORTED_RESOLUTIONS
    _ordered_bands = tuple(SENTINEL2_BANDS_RESOLUTION.keys())
    _google_cloud_bucket = "gs://gcp-public-data-sentinel-2"
    _rasterio_driver = "JP2OpenJPEG"

    def __init__(
        self,
        folder,
        bands=None,
        resolution=20,
        bbox=None,
        **kwargs,
    ):
        """
        Contructor.

        Parameters
        ----------
        folder: str
            Path to the folder with the SAFE data.
        resolution: int
            Desired resolution.
        bands: iterable
            List of bands to process
        """
        super().__init__(
            folder, resolution=resolution, bands=bands, bbox=bbox, **kwargs
        )

    def _is_download_needed(self, filename):
        """Return True if the filename correspond to a selected band."""
        if not filename.endswith("jp2"):
            return True
        return any([filename.endswith(f"{band}.jp2") for band in self.bands])

    def _read_metadata(self, product_path):
        """
        Read the xml metadata files and return a dictionary with the following
        key-values pairs:

        - NODATA: int, value used for to represent NODATA values.
        - SATURATED: int, value used for to represent NODATA values.
        - band_files: dict with band:file_path pairs for each band.
        - quantification_value: float.
        - product_time: datetime, product time.
        """

        # Default values to be used when this information is not found in the metadata
        metadata = dict(NODATA=0, SATURATED=65535, quantification_value=10000)

        # Find metadata file
        metadata_file = glob.glob(os.path.join(product_path, "MTD_*.xml"))
        if len(metadata_file):
            metadata_file = metadata_file[0]

        if not os.path.isfile(metadata_file):
            # Try old format metadata
            metadata_file = glob.glob(os.path.join(product_path, "S2*_OPER_*.xml"))
            if len(metadata_file):
                metadata_file = os.path.join(product_path, metadata_file[0])
            else:
                raise RuntimeError(f"Metadata file not found in {product_path}")

        tree = etree.parse(metadata_file)
        root = tree.getroot()
        images_elements = root.findall(".//Granule/IMAGE_FILE")

        images_elements_txt = [element.text.strip() for element in images_elements]
        band_files = {
            element_txt.split("_")[-1]: os.path.join(product_path, f"{element_txt}.jp2")
            for element_txt in images_elements_txt
        }

        metadata["band_files"] = band_files

        # NODATA and SATURATED values
        special_value_elements = root.findall(
            ".//Product_Image_Characteristics/Special_Values"
        )
        for _element in special_value_elements:
            value_type = _element.find("SPECIAL_VALUE_TEXT")
            value_index = _element.find("SPECIAL_VALUE_INDEX")

            if (value_type is not None) and (value_index is not None):
                value_type = value_type.text.strip()
                value_index = int(value_index.text.strip())
                metadata[value_type] = value_index

        quantif_value_element = root.find(".//QUANTIFICATION_VALUE")
        if quantif_value_element is not None:
            metadata["quantification_value"] = int(quantif_value_element.text.strip())

        product_time = root.find(".//PRODUCT_START_TIME")
        if product_time is not None:
            metadata["product_time"] = datetime.strptime(
                product_time.text.strip().split(".")[0], "%Y-%m-%dT%H:%M:%S"
            )
        else:
            # If the timestamp is not present in the XML, use the SAFE dir name
            safe_dir_timestamp = os.path.basename(product_path).split("_")[3]
            metadata["product_time"] = datetime.strptime(
                safe_dir_timestamp, "%Y%m%dT%H%M%S"
            )

        return metadata

    def _get_product_path(self, product_id):
        """
        Returns the local path where the product id is stored, relative to the archive's
        root folder. This path mimics exactly the Google Cloud storage structure.
        """
        # S2A_MSIL1C_20151001T142056_N0204_R010_T20JLQ_20151001T143019.SAFE
        tile = product_id.split("_")[5][1:]  # 20JLQ
        sub_dirs = os.path.join("tiles", tile[:2], tile[2], tile[3:])  # tiles/20/J/LQ
        return os.path.join(self.archive_folder, sub_dirs, product_id)

    def post_process_band(self, raster, band):
        """
        Returns the calibrated Sentinel 2 Images to TOA-REF
        """
        return calibrate_sentinel2(
            raster,
            band,
            self.metadata_,
            close=True,
        )

    def _get_is_valid_data(self, raster):
        """"""
        return get_is_valid_mask(
            raster,
            filter_values=[
                self.metadata_["NODATA"],
                self.metadata_["SATURATED"],
            ],
        )
