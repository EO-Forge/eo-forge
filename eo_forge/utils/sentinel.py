"""
Sentinel2 helper functions
==========================

.. autosummary::
    :toctree: ../generated/

    calibrate_sentinel2
    get_granule_from_meta_sentinel
    get_sentinel_granule_img
    get_clouds_msil1c
"""
import glob
import os
import shutil
from collections import OrderedDict
from datetime import datetime
from subprocess import run
from tempfile import mkdtemp

import numpy as np
import rasterio as rio
from lxml import etree

from eo_forge.utils.raster_utils import write_mem_raster

###############################
# Sentinel2 General definitions

SENTINEL2_BANDS_RESOLUTION = OrderedDict(
    B01=60,
    B02=10,
    B03=10,
    B04=10,
    B05=20,
    B06=20,
    B07=20,
    B08=10,
    B8A=20,
    B09=60,
    B10=60,
    B11=20,
    B12=20,
)
SENTINEL2_SUPPORTED_RESOLUTIONS = (10, 20, 60, 120)  # in meters


def get_clouds_msil1c(file):
    """
    Get clouds from metadata file "MTD_MSIL1C.xml".

    Parameters
    ----------
    file: path
        path to metadata file

    Returns
    -------
    clouds: float
        clouds as % ([0,100] scale)
    """

    tree = etree.parse(file)
    root = tree.getroot()
    clouds_elements = root.findall(".//Cloud_Coverage_Assessment")
    clouds_elements_txt = [element.text.strip() for element in clouds_elements]
    try:
        clouds = float(clouds_elements_txt[0])
    except:
        clouds = None

    return clouds


def get_sentinel_granule_img(metadata_file):
    """
    Get granule id and image id from sentinel2 l1c products.

    Parameters
    ----------
    metadata_file: str
        path to sentinel2 l1c metadata file

    Returns
    -------
    granule_id: str
    image_id:str
    """
    tree = etree.parse(metadata_file)
    root = tree.getroot()
    images_elements = root.findall(".//Granule/IMAGE_FILE")
    images_elements_txt = [element.text.strip() for element in images_elements]
    base_ = images_elements_txt[0].split("/")
    granule = base_[1]
    image_base = "_".join(base_[-1].split("_")[:-1])
    return granule, image_base


def get_granule_from_meta_sentinel(base_dir):
    """
    Get granule id and image base for a sentinel2 L1C image.

    Parameters
    ----------
    base_dir: url
        gcp image url

    Returns
    -------
    granule: str
        granule id from base dir
    image_base: str
        image base
    """
    SENTINEL2_META = "{}/MTD_MSIL1C.xml"
    # make temp
    dir_ = mkdtemp()
    # set remote path
    remote_path = SENTINEL2_META.format(base_dir)
    # get metadata
    cmd = ["gsutil", "cp", remote_path, dir_]
    p = run(cmd, capture_output=True, text=True)
    # Now get granule id and img id
    metadata_file = SENTINEL2_META.format(dir_)
    granule, image_base = get_sentinel_granule_img(metadata_file)
    shutil.rmtree(dir_)
    return granule, image_base


def calibrate_sentinel2(
    raster,
    band,
    metadata,
    close=False,
):
    """
    Calibrate Sentinel 2.

    Parameters
    ----------
    raster:
        raster instance opened by rasterio
    band: band to calibrate (NOT USED)
    metadata: metadata dict
    close: bool
        Close the input raster dataset before returning the calibrated raster.

    Returns
    -------
    Returns a DatasetReader instance from a MemoryFile.
    """
    profile = raster.profile
    data = raster.read() / metadata["quantification_value"]
    profile.update(
        {
            "dtype": rio.float32,
            "driver": "GTiff",
        }
    )
    if close:
        raster.close()
    return write_mem_raster(data.astype(rio.float32), **profile)


def calibrate_s2_scl(
    raster,
    filter_values,
    init_value=0,
    nodata=0,
    close=False,
):
    """
    Calibrate Sentinel2 SCL band.

    Parameters
    ----------
    raster:
        raster instance opened by rasterio
    filter_values: list or tuple
        S2-SCL: [0,1,2,4,5,6,7,11]
    close: bool
        Close the input raster dataset before returning the calibrated raster.

    Returns
    -------
    returns a DatasetReader instance from a MemoryFile.

    REF: https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm

    0 	NO_DATA
    1 	SATURATED_OR_DEFECTIVE
    2 	DARK_AREA_PIXELS
    3 	CLOUD_SHADOWS
    4 	VEGETATION
    5 	NOT_VEGETATED
    6 	WATER
    7 	UNCLASSIFIED
    8 	CLOUD_MEDIUM_PROBABILITY
    9 	CLOUD_HIGH_PROBABILITY
    10 	THIN_CIRRUS
    11 	SNOW

    """
    profile = raster.profile
    profile.update({"dtype": rio.ubyte, "driver": "GTiff"})
    data = raster.read()
    mask_nodata = data == nodata
    assert len(filter_values) > 0, "At least one value should be provided for filtering"
    mask_ = data != filter_values[0]
    for value in filter_values[1:]:
        mask_ = mask_ & (data != value)
    data_cloud = np.where(mask_, True, init_value)
    data_cloud = np.where(mask_nodata, False, data_cloud)
    if close:
        raster.close()
    return write_mem_raster(data_cloud.astype(rio.ubyte), **profile)


class s2_metadata:
    @staticmethod
    def get_band_resolution_s2l2a(filelist):
        # build dict first
        base_dict = {}
        for f in filelist:
            s = f.split(os.sep)[-1].split("_")
            band = s[-2]
            band_status = base_dict.get(band)
            if band_status is None:
                base_dict[band] = {}
            resolution_band = int(s[-1].replace("m", ""))
            base_dict[band].update({resolution_band: f})
        return base_dict

    @staticmethod
    def get_nearest_resolution_s2l2a(band_dict, resolution=10):
        filt_dict = {}

        def get_min(res_list, resolution):
            result = [resolution - i for i in res_list]
            try:
                result_min = min(j for j in result if j >= 0)
            except:
                result_min = min(j for j in [abs(r) for r in result] if j >= 0)
                result = [abs(r) for r in result]
            idx = result.index(result_min)
            return res_list[idx]

        for b in band_dict:
            res_band = get_min(list(band_dict[b]), resolution)
            filt_dict[b] = band_dict[b][res_band]
        return filt_dict

    def read_metadata_l1c(self, product_path):
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

    def read_metadata_l2a(self, product_path, resolution=10):
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

        bands_resolution = self.get_band_resolution_s2l2a(images_elements_txt)
        images_elements_filt = self.get_nearest_resolution_s2l2a(
            bands_resolution, resolution
        )
        band_files = {
            k: f"{product_path}{os.sep}{v}.jp2" for k, v in images_elements_filt.items()
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

        quantif_value_element = root.find(
            ".//BOA_QUANTIFICATION_VALUE"
        )  # BOA_QUANTIFICATION_VALUE
        if quantif_value_element is not None:
            metadata["quantification_value"] = int(quantif_value_element.text.strip())

        quantif_value_element = root.find(".//AOT_QUANTIFICATION_VALUE")
        if quantif_value_element is not None:
            metadata["aot_quantification_value"] = float(
                quantif_value_element.text.strip()
            )

        quantif_value_element = root.find(".//WVP_QUANTIFICATION_VALUE")
        if quantif_value_element is not None:
            metadata["wvp_quantification_value"] = float(
                quantif_value_element.text.strip()
            )

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
