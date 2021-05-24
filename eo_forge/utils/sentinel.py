import numpy as np
import rasterio as rio

from eo_forge.utils.raster_utils import write_mem_raster


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
