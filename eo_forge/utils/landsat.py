"""
Landsat helper functions
========================

.. autosummary::
    :toctree: ../generated/

    get_clouds_landsat
    calibrate_landsat_bqa
    calibrate_landsat5
    calibrate_landsat8
"""
from collections import OrderedDict

import numpy as np
import rasterio as rio

from eo_forge.utils.raster_utils import write_mem_raster

###############################
# Landsat General definitions
# Also define a default ordering for the bands.
LANDSAT8_BANDS_RESOLUTION = OrderedDict(
    B1=30,
    B2=30,
    B3=30,
    B4=30,
    B5=30,
    B6=30,
    B7=30,
    B8=15,
    B9=30,
    B10=100,
    B11=100,
)

# Landsat5 Bands resolution in meters
# Also define a default ordering for the bands.
LANDSAT5_BANDS_RESOLUTION = OrderedDict(
    B1=30,
    B2=30,
    B3=30,
    B4=30,
    B5=30,
    B6=30,  # 120m but resampled to 30
    B7=30,
)

# Resolutions in meters for landsat 5 and 8
LANDSAT_SUPPORTED_RESOLUTIONS = (30, 60, 90, 120)


def get_clouds_landsat(file):
    """purpouse: to read cloud image level from landsat metadata file

    Parameters
    ----------
        file: path
            path to metadata file

    Returns
    -------
        cloud level

    """
    with open(file, "r") as f:
        lines = f.readlines()
    cloud = None
    for line in lines:
        if "CLOUD_COVER_LAND" in line:
            cloud = line.split("=")[1].replace("\n", "").strip()
    return cloud


def calibrate_landsat_bqa(
    raster,
    filter_values,
    init_value=0,
    nodata=0,
    close=False,
):
    """
    Calibrate Landsat BQA band.

    Parameters
    ----------
    raster:
        raster instance opened by rasterio
    filter_values: list or tuple
        Landsat8: [1,2720]
        Landsat5: [1,672]
    close: bool
        Close the input raster dataset before returning the calibrated raster.

    Returns
    -------
    returns a DatasetReader instance from a MemoryFile.
    """
    profile = raster.profile
    profile.update({"dtype": rio.ubyte})
    data = raster.read()
    mask_nodata = data != nodata
    clear_value = filter_values[0]
    fill_value = filter_values[1]
    mask_ = (data != clear_value) & (data != fill_value)
    mask_full = np.logical_and(mask_nodata, mask_)
    data = np.where(mask_full, True, init_value)
    # https://www.usgs.gov/media/images/landsat-8-quality-assessment-band-pixel-value-interpretations
    if close:
        raster.close()
    return write_mem_raster(data.astype(rio.ubyte), **profile)


def calibrate_landsat5(
    raster,
    band,
    metadata,
    reflectance=True,
    chkur_or_neckel=True,
    close=False,
):
    """Calibrate landsat5 l1 images to top of atmosphere radiance (default) or
    reflectance (if `reflectance=True`).

    Parameters
    ----------
    raster:
        raster instance opened by rasterio
    band: str
        Landsat 5 band
    metadata: dict
        Landsat 5 metadata parsed file
    reflectance: bool
        If True get reflectance else radiance
        (REF-TOA B1-5,7 / TOA Brightness Temperature B6)
    close: bool
        Close the input raster dataset before returning the calibrated raster.
    Returns
    -------
    returns a DatasetReader instance from a MemoryFile.

    REF: L5-Cal.pdf and calibrado_l5bis.pdf
    """

    profile = raster.profile
    profile.update({"dtype": rio.float32})
    data = raster.read()
    band = band.upper()
    band_num = int(band.replace("B", ""))
    band_label = f"BAND_{band_num}"

    # solar models ESN
    if chkur_or_neckel:
        esun = {
            "B1": 1957,
            "B2": 1826,
            "B3": 1554,
            "B4": 1036,
            "B5": 215.0,
            "B7": 80.67,
        }
    else:
        esun = {
            "B1": 1957,
            "B2": 1829,
            "B3": 1557,
            "B4": 1047,
            "B5": 219.3,
            "B7": 74.52,
        }

    LMax = metadata[f"RADIANCE_MAXIMUM_{band_label}"]
    LMin = metadata[f"RADIANCE_MINIMUM_{band_label}"]
    QMax = float(metadata[f"QUANTIZE_CAL_MAX_{band_label}"])
    QMin = float(metadata[f"QUANTIZE_CAL_MIN_{band_label}"])
    dL = LMax - LMin
    dQ = QMax - QMin

    if reflectance:
        # calculate TOA-REF for B1-5/7
        # Top of Atmosphere Brightness Temperature B6
        data = (data - QMin) * dL / dQ + LMin

        if band != "B6":
            zA = np.radians(90 - float(metadata["SUN_ELEVATION"]))
            es_distance = float(metadata["EARTH_SUN_DISTANCE"])
            esun_band = esun[band]
            data = data * np.pi * es_distance ** 2 / (esun_band * np.cos(zA))

        else:
            # B6
            K1 = metadata[f"K1_CONSTANT_{band_label}"]
            K2 = metadata[f"K2_CONSTANT_{band_label}"]
            data = K2 / np.log(K1 / data + 1)
    else:
        # calculate radiance
        data = (data - QMin) * dL / dQ + LMin
    if close:
        raster.close()
    return write_mem_raster(data.astype(rio.float32), **profile)


def calibrate_landsat8(
    raster,
    band,
    metadata,
    reflectance=False,
    with_bounds=True,
    close=False,
):
    """Calibrate landsat8 l1 images to top of atmosphere radiance (default) or
    reflectance (if `reflectance=True`).

    Reference: Landsat8 Handbook LSDS-1574Version 5.0

    Parameters
    ----------
    raster:
        raster instance opened by rasterio
    band: str
        Landsat8 band
    metadata: dict
        Landsat 8 metadata parsed file
    reflectance: bool
        If True get radiance else reflectance
        (REF-TOA B1-9 / TOA Brightness Temperature B10-11)
    close: bool
        Close the input raster dataset before returning the calibrated raster.
    Returns
    -------
    returns a DatasetReader instance from a MemoryFile.
    """

    band = band.upper()
    if band == "BQA":
        return raster

    profile = raster.profile
    profile.update({"dtype": rio.float32})
    data = raster.read()
    band_num = int(band.replace("B", ""))
    band_label = f"BAND_{band_num}"

    if reflectance:
        # Calculate TOA Reflectance
        # OLI TOA-REF for B1-9 /
        # TIRS Top of Atmosphere Brightness Temperature B10-B11
        if band_label not in ("B10", "B11"):
            Ml = metadata[f"REFLECTANCE_MULT_{band_label}"]
            Al = metadata[f"REFLECTANCE_ADD_{band_label}"]
            zA = 90 - float(metadata["SUN_ELEVATION"])
            data = (data * Ml + Al) / np.cos(np.radians(zA))
            if with_bounds:
                max_rf = metadata[f"REFLECTANCE_MAXIMUM_{band_label}"]
                min_rf = metadata[f"REFLECTANCE_MINIMUM_{band_label}"]

                # TODO: CHECK
                data = np.where(data <= max_rf, data, max_rf)
                data = np.where(data < min_rf, min_rf, data)
        else:
            # B10 or B11
            Ml = metadata[f"RADIANCE_MULT_{band_label}"]
            Al = metadata[f"RADIANCE_ADD_{band_label}"]
            K1 = metadata[f"K1_CONSTANT_{band_label}"]
            K2 = metadata[f"K2_CONSTANT_{band_label}"]
            L = data * Ml + Al
            if with_bounds:
                max_rf = metadata[f"RADIANCE_MAXIMUM_{band_label}"]
                min_rf = metadata[f"RADIANCE_MINIMUM_{band_label}"]
                # TODO: CHECK
                L = np.where(L <= max_rf, L, max_rf)
                L = np.where(L < min_rf, min_rf, L)

            data = K2 / np.log(K1 / L + 1)
            del L
    else:
        # Calculate TOA radiance
        Ml = metadata[f"RADIANCE_MULT_{band_label}"]
        Al = metadata[f"RADIANCE_ADD_{band_label}"]
        data = data * Ml + Al
        if with_bounds:
            max_rf = metadata[f"RADIANCE_MAXIMUM_{band_label}"]
            min_rf = metadata[f"RADIANCE_MINIMUM_{band_label}"]
            # TODO: CHECK
            data = np.where(data <= max_rf, data, max_rf)
            data = np.where(data < min_rf, min_rf, data)

    if close:
        raster.close()
    return write_mem_raster(data.astype(rio.float32), **profile)
