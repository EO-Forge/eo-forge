import numpy as np
import rasterio as rio

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


def calibrate_sentinel2(
    raster,
    band,
    metadata,
    close=False,
):
    """Calibrate Sentinel 2
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
    returns a DatasetReader instance from a MemoryFile.
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
