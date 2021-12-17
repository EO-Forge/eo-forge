"""
Sentinel loaders module
=======================
.. autosummary::
    :toctree: ../generated/

    Sentinel2Loader
    s2_cloud_preproc
"""
import geopandas as gpd
import numpy as np
import os
import rasterio as rio

from eo_forge.io.GenLoader import BaseGenericLoader
from eo_forge.utils.raster_utils import (
    get_is_valid_mask,
    shapes2array,
    write_mem_raster,
)
from eo_forge.utils.sentinel import (
    SENTINEL2_BANDS_RESOLUTION,
    SENTINEL2_SUPPORTED_RESOLUTIONS,
    calibrate_sentinel2,
    s2_metadata,
    calibrate_s2_scl,
)
from eo_forge.utils.utils import walk_dir_files


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
    _filter_scl_hcloud = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    _filter_scl_acloud = [1, 2, 4, 5, 6, 7, 11]

    def __init__(
        self,
        folder,
        bands=None,
        resolution=10,
        level="l1c",
        bbox=None,
        logger=None,
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
        level: str
            "l1c" or "l2a" processing levels
        """
        super().__init__(
            folder,
            resolution=resolution,
            bands=bands,
            bbox=bbox,
            logger=logger,
            **kwargs,
        )
        self.raw_metadata = None
        self.spacecraft = 2
        assert level in ["l1c", "l2a"], "Level should be one of: l1c or l2a"
        self.proc_level = level

        self.logger.info(f"Running on Sentinel {self.spacecraft} data")

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
        s2meta_reader = s2_metadata()
        if self.proc_level == "l1c":
            metadata = s2meta_reader.read_metadata_l1c(product_path)
        else:
            metadata = s2meta_reader.read_metadata_l2a(product_path, self.resolution)
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
            filter_values=(
                self.metadata_["NODATA"],
                self.metadata_["SATURATED"],
            ),
        )

    def _preprocess_clouds_mask(self, metadata, **kwargs):
        """Return Raster BQA."""

        # use legacy == True for gml or False for SCL
        legacy_ = kwargs.get("clouds_legacy", True)

        if self.proc_level == "l1c" and legacy_ == False:
            self.logger.warning(
                f"Found Level {self.proc_level} and cloud_legacy {legacy_}\n"
                + f"Forcing clouds_legacy to True (cloud mask from gml file)",
            )
            legacy_ = True

        if legacy_:
            self.logger.info(f"Pre-processing legacy cloud mask (gml file)")
            raster_base = kwargs["raster_base"]
            nodata = kwargs["no_data"]
            base_dir = metadata["product_path"]

            gpd_ = s2_cloud_preproc(base_dir)
            if gpd_ is None:
                array_ = np.zeros(
                    (raster_base.height, raster_base.width), dtype=rio.uint8
                )

            else:
                array_ = shapes2array(gpd_, raster_base)

            profile = raster_base.profile.copy()
            profile.update({"count": 1, "nodata": nodata})
            return write_mem_raster(array_[np.newaxis, ...], **profile)
        else:
            self.logger.info(f"Pre-processing SCL cloud mask (raster file)")

            raster_base = rio.open(metadata["band_files"]["SCL"])
            filter_values = kwargs.get("scl_filter", self._filter_scl_hcloud)
            self.logger.info(f"Preprocessing SCL filtering values: {filter_values}")
        return calibrate_s2_scl(raster_base, filter_values)
