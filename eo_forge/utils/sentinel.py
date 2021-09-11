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
import numpy as np
import shutil
import os
from subprocess import run
from tempfile import mkdtemp
import rasterio as rio
from lxml import etree
from collections import OrderedDict
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
    base_ = images_elements_txt[0].split(os.sep)
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
