"""
Lansat loaders module
=====================

.. autosummary::
    :toctree: ../generated/

    LandsatLoader
"""

import glob
import os
import warnings
from datetime import datetime

import rasterio

from eo_forge.io.GenLoader import BaseGenericLoader
from eo_forge.utils.landsat import (
    LANDSAT5_BANDS_RESOLUTION,
    LANDSAT8_BANDS_RESOLUTION,
    LANDSAT_SUPPORTED_RESOLUTIONS,
    calibrate_landsat5,
    calibrate_landsat8,
    calibrate_landsat_bqa,
)
from eo_forge.utils.raster_utils import get_is_valid_mask


class LandsatLoader(BaseGenericLoader):
    """
    Class for Loading Landsat5 and Landsat8 images into a single raster file (and cloud file)

    This class can only load data from a local storage (your laptop storage, a NFS storage,etc).
    The functionality to read directly from Cloud Buckets is not yet implemented.

    The particular raw data files are looked inside the archive folder based on their product ID.
    """

    _supported_resolutions = LANDSAT_SUPPORTED_RESOLUTIONS
    _rasterio_driver = "GTiff"

    def __init__(
        self,
        folder,
        resolution=30,
        spacecraft=5,
        bands=None,
        reflectance=True,
        logger=None,
        **kwargs,
    ):
        """
        Contructor.

        Parameters
        ----------
        folder : str
            Path to the folder with the Landsat raw data.
        resolution: int
            Desired resolution.
        spacecraft: int
            Landsat spacecraft (5 or 8).
        bands: iterable
            List of bands to process
        reflectance: bool
             If True load data as "reflectance", otherwise as "radiance".
        """
        if spacecraft not in (5, 8):
            raise ValueError(
                f"Only Landsat5 and Landsat8 are supported. "
                f'Spacecraft number received: "{spacecraft}"'
            )
        self.spacecraft = spacecraft
        if self.spacecraft == 8:
            self._ordered_bands = tuple(LANDSAT8_BANDS_RESOLUTION.keys())
            self._filter_values = [1, 2720]
            self.calibrate_func = calibrate_landsat8
        else:
            # landsat5
            self._ordered_bands = tuple(LANDSAT5_BANDS_RESOLUTION.keys())
            self._filter_values = [1, 672]
            self.calibrate_func = calibrate_landsat5

        self.reflectance = reflectance
        self.raw_metadata = {}

        super().__init__(
            folder, resolution=resolution, bands=bands, logger=logger, **kwargs
        )

        self.logger.info(f"Running on Landsat {self.spacecraft} data")

    def _read_metadata(self, product_path):
        """
        Read the txt metadata files and return a dictionary with the following
        key-values pairs:

        - product_time: datetime, product time.
        - cc: float, cloudy_percentage.
        - band_files: dict with band:file_path pairs for each band.
        """

        # metadata file
        metadata_file = glob.glob(os.path.join(product_path, "*_MTL.txt"))
        if len(metadata_file) == 0:
            raise RuntimeError(f"Metadata file not found in {product_path}")

        def cast_value(value):
            value = value.strip()
            if '"' in value:
                value = value.replace('"', "").strip()
            elif "." in value:
                value = float(value.strip())
            else:
                try:
                    value = int(value.strip())
                except ValueError:
                    pass
            return value

        # parse metadata.
        self.raw_metadata = {
            key.strip(): cast_value(val)
            for key, val in (
                line.split("=") for line in open(metadata_file[0]) if "=" in line
            )
        }

        metadata = {
            "cloud_cover": float(self.raw_metadata.get("CLOUD_COVER_LAND", None))
        }

        if all(
            key in self.raw_metadata for key in ("DATE_ACQUIRED", "SCENE_CENTER_TIME")
        ):
            _time = self.raw_metadata["SCENE_CENTER_TIME"].split(".")[0]
            _date = self.raw_metadata["DATE_ACQUIRED"]

            metadata["product_time"] = datetime.strptime(
                f"{_date}T{_time}", "%Y-%m-%dT%H:%M:%S"
            )
            # If the timestamp is not present in the metadata, we cannot use the file
            # name since it does not contains the time, only the date.

        base_bands = [band for band in self._ordered_bands if band.upper() != "BQA"]

        band_files = {
            band: os.path.join(product_path, self.raw_metadata[key])
            for band, key in (
                (_band, f"FILE_NAME_BAND_{int(_band[1:])}") for _band in base_bands
            )
            if key in self.raw_metadata
        }

        # Only keep in the metadata the files that exist.
        metadata["band_files"] = {
            band: file_path
            for band, file_path in band_files.items()
            if os.path.exists(file_path)
        }

        metadata["band_files"]["BQA"] = os.path.join(
            product_path, self.raw_metadata["FILE_NAME_BAND_QUALITY"]
        )

        return metadata

    def _get_product_path(self, product_id):
        """
        Returns the local path where the product id is stored, relative to the archive's
        root folder. This path mimics exactly the Google Cloud storage structure.
        """

        # sub_dirs: SENSOR_ID/01/PATH/ROW/SCENE_ID/
        product_id_parts = product_id.split("_")
        sub_dirs = os.path.join(
            product_id_parts[0],
            "01",
            product_id_parts[2][:3],
            product_id_parts[2][3:],
        )
        return os.path.join(self.archive_folder, sub_dirs, product_id)

    def _clean_product_id(self, product_id):
        """purpose: clean product id from extensions."""
        return product_id

    def post_process_band(self, raster, band):
        """Postprocess the band data (calibration)."""
        if band.upper() == "BQA":
            return calibrate_landsat_bqa(raster, self._filter_values, close=True)
        else:
            return self.calibrate_func(
                raster, band, self.raw_metadata, reflectance=self.reflectance
            )

    def _get_is_valid_data(self, raster):
        """Returns raster mask with valid data."""
        return get_is_valid_mask(raster, filter_values=[0, 0])

    def _preprocess_clouds_mask(self, metadata, band="BQA", **kwargs):
        """Return Raster BQA."""

        band = band.upper()
        data_file = metadata["band_files"][band]

        if len(data_file) == 0:
            warnings.warn(f"No files found for the {band} band")

        raster_dataset = rasterio.open(data_file, driver=self._rasterio_driver)

        return raster_dataset
