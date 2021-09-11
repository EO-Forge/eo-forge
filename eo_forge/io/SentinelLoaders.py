"""
Sentinel loaders module
=======================
.. autosummary::
    :toctree: ../generated/

    Sentinel2Loader
    s2_cloud_preproc
"""
import glob
import os
from datetime import datetime
import numpy as np
import rasterio as rio
from lxml import etree

import geopandas as gpd

from eo_forge.utils.raster_utils import (
    get_is_valid_mask,
    shapes2array,
    write_mem_raster,
)
from eo_forge.utils.sentinel import (
    calibrate_sentinel2,
    SENTINEL2_BANDS_RESOLUTION,
    SENTINEL2_SUPPORTED_RESOLUTIONS,
)

from eo_forge.utils.utils import walk_dir_files
from eo_forge.io.GenLoader import BaseGenericLoader


######################################################################


def s2_cloud_preproc(base_dir, dump_file=None):
    """
    Read cloud mask file as geodataframe and write to disk (if necessary)
    :param dump_file: file to be written (if None, just return the
    geodataframe)
    """

    _, _, g = walk_dir_files(base_dir, cases=["MSK_CLOUDS_B00.gml"])

    if "MSK_CLOUDS_B00.gml" in g:
        mask_cloud_file_ = g["MSK_CLOUDS_B00.gml"][0]
    else:
        mask_cloud_file_ = None

    gpd_ = None

    if mask_cloud_file_ is None:
        gpd_ = None
        return gpd_
    else:
        try:
            gpd_ = gpd.read_file(mask_cloud_file_)
            if dump_file is None:
                pass
            else:
                gpd_.to_file(dump_file)
            return gpd_
        except:  # noqa
            print(f"FAILED to read/dump file: {mask_cloud_file_}")
            return None


class Sentinel2Loader(BaseGenericLoader):
    """
    Class for Loading Sentinel SAFE data into a single raster file (and cloud file)

    This class can only load data from a local storage (your laptop storage, a NFS storage,etc).
    The functionality to read directly from Cloud Buckets is not yet implemented.

    The particular raw data files are looked inside the archive folder based on their product ID.
    """

    _supported_resolutions = SENTINEL2_SUPPORTED_RESOLUTIONS
    _ordered_bands = tuple(SENTINEL2_BANDS_RESOLUTION.keys())
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
        self.raw_metadata = None

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

        self.raw_metadata = metadata
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

    def _clean_product_id(self, product_id):
        """Clean product id from extensions"""
        return product_id.replace(".SAFE", "")

    def post_process_band(self, raster, band):
        """Returns the calibrated Sentinel 2 Images to TOA-REF."""
        if band.upper() == "BQA":
            return raster
        else:
            return calibrate_sentinel2(
                raster,
                band,
                self.metadata_,
                close=True,
            )

    def _get_is_valid_data(self, raster):
        """Returns raster mask with valid data."""
        return get_is_valid_mask(
            raster,
            filter_values=[
                self.metadata_["NODATA"],
                self.metadata_["SATURATED"],
            ],
        )

    def _preprocess_clouds_mask(self, metadata, **kwargs):
        """Return Raster BQA."""

        raster_base = kwargs["raster_base"]
        nodata = kwargs["no_data"]
        base_dir = metadata["product_path"]

        gpd_ = s2_cloud_preproc(base_dir)
        if gpd_ is None:
            array_ = np.zeros((raster_base.height, raster_base.width), dtype=rio.uint8)

        else:
            array_ = shapes2array(gpd_, raster_base)

        profile = raster_base.profile.copy()
        profile.update({"count": 1, "nodata": nodata})
        return write_mem_raster(array_[np.newaxis, ...], **profile)
