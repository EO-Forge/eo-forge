# -*- coding: utf-8 -*-
import gc
import glob
import os
import warnings
from abc import abstractmethod
from collections import OrderedDict
from datetime import datetime

import numpy as np
import rasterio
from eolearn.core import EOTask, EOPatch, FeatureType
from google.cloud import storage
from lxml import etree
from sentinelhub import DataCollection

from eo_forge.utils.landsat import (
    calibrate_landsat8,
    calibrate_landsat_bqa,
)
from eo_forge.utils.raster_utils import (
    resample_raster,
    clip_raster,
    check_resample,
    bbox_from_raster,
    reproject_raster_north_south,
    check_raster_clip_crs,
    check_raster_shape_match,
    get_is_valid_mask,
    reproject_raster_to_bbox,
)
from eo_forge.utils.sentinel import calibrate_sentinel2
from eo_forge.utils.shapes import bbox_to_geodataframe, set_buffer_on_gdf

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
    TCI=10,  # This is not a band, but we define it here anyways.
)
SENTINEL2_SUPPORTED_RESOLUTIONS = (10, 20, 60, 120)  # in meters

#############################
# Landsat General definitions
# Also define a default ordering for the bands.
# We use the same Bands naming convention as in landsat!
LANDSAT8_BANDS_RESOLUTION = OrderedDict(
    B01=30,
    B02=30,
    B03=30,
    B04=30,
    B05=30,
    B06=30,
    B07=30,
    B08=15,
    B09=30,
    B10=100,
    B11=100,
    BQA=30,
)

# Landsat5 Bands resolution in meters
# Also define a default ordering for the bands.
LANDSAT5_BANDS_RESOLUTION = OrderedDict(
    B01=30,
    B02=30,
    B03=30,
    B04=30,
    B05=30,
    B06=30,  # 120m but resampled to 30
    B07=30,
    BQA=30,
)

# Resolutions in meters for landsat 5 and 8
LANDSAT_SUPPORTED_RESOLUTIONS = (30, 60, 90, 120)


class BaseLoaderTask(EOTask):
    """Base class for Landsat and Sentinel loaders."""

    _supported_resolutions = tuple()
    _ordered_bands = tuple()
    _google_cloud_bucket = ""
    _rasterio_driver = None

    def __init__(
            self,
            folder,
            resolution=30,
            bands=None,
            bbox=None,
            **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        folder : str
            Path to the folder with the Landsat raw data.
        resolution: int
            Desired resolution.
        bands: iterable
            List of bands to process

        Attributes
        ----------
        archive_folder
        bands
        resolution
        """
        if not os.path.isdir(folder):
            raise ValueError(f"folder {folder} does not exist")

        self.archive_folder = folder
        self.bbox = bbox

        if resolution not in self._supported_resolutions:
            raise ValueError(
                f"Resolution '{resolution}' not supported.\n"
                f"Supported values: {str(self._supported_resolutions)}"
            )
        self.resolution = resolution
        if bands is None:
            self.bands = self._ordered_bands
        else:
            self.bands = []
            for band in bands:
                if band not in self._ordered_bands:
                    warnings.warn(f"'{band}' is not a valid band. Ignoring")
                else:
                    self.bands.append(band)

    def download_product(self, bucket_url, dest_dir, force=False):
        """
        Downloads a blob from the bucket.

        Parameters
        ----------
        bucket_url: url
            Bucket URL to download. The prefix "gs://" is removed.
        dest_dir: str
            Destination directory for the downloaded files.
        force: bool
            If true, files are downloaded even if they already exists on the destination
            folder. Otherwise, the download is skipped.
        """
        bucket_url = bucket_url.replace("gs://", "")
        bucket_name = bucket_url.split("/")[0]
        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(
            prefix=os.path.relpath(bucket_url, bucket_name)
        )

        print(f"Downloading {os.path.basename(bucket_url)} in {dest_dir}")
        for blob in blobs:
            filename = blob.name
            if filename.endswith("$folder$"):
                continue
            if self._is_download_needed(filename):
                dest_path = os.path.join(dest_dir, filename)
                if not os.path.exists(dest_path) or force:
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    blob.download_to_filename(dest_path)
        print(f"Downloading finished")

    def _is_download_needed(self, filename):
        """Return True if the filename correspond to a selected band."""
        raise NotImplementedError("_is_download_needed method not implemented.")

    @abstractmethod
    def _get_product_path(self, product_id):
        """
        Returns the local path where the product id is stored, relative to the archive's
        root folder. This path mimics exactly the Google Cloud storage structure.
        """
        raise NotImplementedError

    @abstractmethod
    def _read_metadata(self, product_path):
        """
        Returns a dictionary with the product metadata.
        """
        raise NotImplementedError

    def post_process_band(self, raster, band):
        """
        Postprocess the band data (ndarray).
        """
        return raster

    @abstractmethod
    def _get_is_valid_data(self, raster):
        """
        Returns raster mask with valid data
        """
        raise NotImplementedError

    def _get_bands_data(
            self,
            metadata,
            **kwargs,
    ):
        """
        Gets bands data.

        Parameters
        ----------
        metadata: dict
            Dictionary with the raster's metadata.

        Other Parameters
        ----------------
        bbox: sentinelhub.BBox or Geopandas Dataframe.
            Region of Interest.
        reproject: bool
            If True, reproject raster north-south if needed.
        crop: bool
            Whether to crop the raster to the extent of the roi_bbox.
        nodata: float
            No data value.
        all_touched: boolean
            If True, all pixels touched by geometries will be included.
            If false, only pixels whose center is within the polygon are included.
        close: bool
            Close the input raster dataset before returning the clipped raster.
        enable_transform: bool
            Allows coordinate reference system transformations if the target BBox's crs
            and the raster crs differ.
        nodata: float
            No data value.
        all_touched: boolean
            If True, all pixels touched by geometries will be included.
            If false, only pixels whose center is within the polygon are included.
        hard_bbox: bool
            If True, adjust the extent of raster to match the BBox. Otherwise, leave the
            raster's original shape and mask the values outside the bbox.
        calibrate: bool
            If True, apply the postprocessing (calibration).

        Returns
        -------
        base_bands_data: dict[band]=np.array
            Dictionary holding numpy arrays with the base bands data.
        extra_bands: dict[band]=np.array
            Dictionary holding numpy arrays with the TCI, BQA, or other products data
            that are not considered base bands.
        bbox: sentinelhub.BBox
            Bounding box of the returned data.
        """

        base_bands_data = {}
        extra_bands = {}
        band_masks = {}

        enable_transform = kwargs.get("enable_transform", True)
        crop = kwargs.pop("crop", True)
        hard_bbox = kwargs.get("hard_bbox", False)
        bbox = kwargs.get("bbox", self.bbox)
        nodata = kwargs.get("nodata", 0)
        reproject = kwargs.get("reproject", True)
        all_touched = kwargs.get("all_touched", True)
        calibrate = kwargs.get("calibrate", True)

        clipping_flag = bbox is not None

        raster_bbox = None
        for band in self.bands:
            band = band.upper()
            data_file = metadata["band_files"][band]

            if len(data_file) == 0:
                warnings.warn(f"No files found for the {band} band")

            raster_dataset = rasterio.open(
                data_file, driver=self._rasterio_driver
            )

            # Check if resampling is needed
            resample_flag, scale_factor, true_pixel = check_resample(
                raster_dataset, self.resolution
            )

            if clipping_flag:
                # Check BBOX
                roi_bbox = bbox_to_geodataframe(bbox)

                # check roi
                roi_check = check_raster_clip_crs(
                    raster_dataset,
                    roi_bbox,
                    enable_transform=enable_transform,
                )

                # check match
                full_match, _ = check_raster_shape_match(
                    raster_dataset, roi_check, enable_transform=enable_transform
                )

                if resample_flag:
                    # Resample taking special care of the borders.
                    # Add a small buffer to the BBox to take into account the borders.
                    roi_check_buffered = set_buffer_on_gdf(
                        roi_check, buffer=5 * true_pixel
                    )

                    # Pre-clip the raster with the buffered bbox.
                    raster_dataset = clip_raster(
                        raster_dataset,
                        roi_check_buffered,
                        crop=crop,
                        nodata=nodata,
                        close=True,
                        hard_bbox=hard_bbox,
                        all_touched=all_touched,
                    )

                    raster_dataset = resample_raster(
                        raster_dataset, scale_factor, close=True
                    )

                    raster_dataset = clip_raster(
                        raster_dataset,
                        roi_check,
                        crop=crop,
                        nodata=nodata,
                        close=True,
                        hard_bbox=hard_bbox,
                    )
                else:
                    # no resample just clip
                    raster_dataset = clip_raster(
                        raster_dataset,
                        roi_check,
                        crop=crop,
                        nodata=nodata,
                        close=True,
                        hard_bbox=hard_bbox,
                    )

                if not full_match:
                    raster_dataset = reproject_raster_to_bbox(raster_dataset, roi_bbox)

            else:
                # No BBOX
                if resample_flag:
                    raster_dataset = resample_raster(
                        raster_dataset, scale_factor, close=True
                    )

            # get raster_dataset_mask
            raster_dataset_mask = self._get_is_valid_data(raster_dataset)

            if calibrate:
                # Apply postprocessing (calibration)
                raster_dataset = self.post_process_band(raster_dataset, band)

            if reproject:
                raster_dataset = reproject_raster_north_south(
                    raster_dataset, close=True
                )

                raster_dataset_mask = reproject_raster_north_south(
                    raster_dataset_mask, close=True
                )
            # We are accepting all data and keeping the valid one in is_valid mask
            band_masks[band] = raster_dataset_mask.read().squeeze()

            # BBOX
            raster_bbox = bbox_from_raster(
                raster_dataset, epsg=raster_dataset.crs.to_epsg()
            )

            # Get Data
            data = raster_dataset.read()

            if band == "TCI":
                extra_bands[band] = data.swapaxes(0, 1).swapaxes(2, 1)[None, :]
            elif band == "BQA":
                # keep just [n x m] dimensions
                extra_bands[band] = data.squeeze()
            else:
                # keep just [n x m] dimensions
                base_bands_data[band] = data.squeeze()

        # Help python a little bit with the memory management
        gc.collect()
        return base_bands_data, extra_bands, raster_bbox, band_masks

    def execute(
            self,
            eopatch=None,
            product_id=None,
            download="auto",
            return_metadata=False,
            **kwargs,
    ):
        """
        Implement the base execute function for the loaders.

        Parameters
        ----------
        eopatch: EOPatch
            Input EOPatch or None if a new EOPatch should be created.
        product_id: str
            Tile product ID or name of the subfolder with the tile data.
        download: str
            Keyword to control the download of the data from Google Cloud.
            Allowed values:
            - skip: Do not download data
            - auto: Only download data if needed.
            - force: Always download data, even if it exists.
        return_metadata: bool
            Whether to return a dictionary with the metadata (return_metadata=True),
            or not (return_metadata=False).

        Other parameters
        ----------------
        bbox: sentinelhub.BBox or None
            Bounding box of the patch. None (default) returns the entire scene.
        crop: bool
            Whether to crop the raster (True) to the extent of the bbox, or fill the
            regions outside the bbox with NODATA.
        enable_transform: bool
            Allows coordinate reference system transformations if the target BBox's crs
            and the raster crs differ.

        Returns
        -------
        eopatch: EOPatch
            new EOPatch with added raster layer.
            The bands data dimensions are time x m x n x channels.
        metadata: dict
            Dictinary with the available metadata.
        """
        if eopatch is None:
            eopatch = EOPatch()

        bbox = kwargs.pop("bbox", self.bbox)
        if download.lower() not in ("auto", "force", "skip"):
            raise ValueError(
                f"Unexpected value for download keyword. Received: {download}.\n"
                "Allowed: 'auto','force','skip'"
            )

        if os.path.isdir(product_id):
            product_path = product_id
        else:
            product_path = self._get_product_path(product_id)

        if download in ("auto", "force"):
            # Download files from google cloud if needed.
            full_bucket_url = os.path.join(
                self._google_cloud_bucket,
                os.path.relpath(product_path, self.archive_folder),
            )
            self.download_product(
                full_bucket_url, self.archive_folder, force=(download == "force")
            )

        if not os.path.isdir(product_path):
            raise RuntimeError(
                f"The {product_id} directory does not exist or "
                f"was not download correctly."
            )
        metadata = self._read_metadata(product_path)
        # Note: we keep metadata in the class
        self.metadata_ = metadata
        metadata["product_path"] = product_path

        (
            base_bands_data,
            extra_bands,
            bbox,
            band_masks_data,
        ) = self._get_bands_data(metadata, bbox=bbox, **kwargs)

        ##
        # Store the base bands in a single array
        # The bands order is the one expected by eolearn.
        ordered_bands = [
            band for band in self._ordered_bands if band in base_bands_data
        ]

        _base_bands_data = [base_bands_data[band] for band in ordered_bands]

        _base_bands_mask = [band_masks_data[band] for band in ordered_bands]

        data_mask = np.stack(_base_bands_mask, axis=2)

        eopatch[FeatureType.MASK]["IS_VALID"] = data_mask[np.newaxis, :].all(
            axis=3, keepdims=True
        )

        # [n x m x bands]
        data = np.stack(_base_bands_data, axis=2)

        # Prepend the time dimension since FeatureType.DATA features include time.
        eopatch[FeatureType.DATA]["BANDS"] = data[np.newaxis, :]
        eopatch[FeatureType.META_INFO]["BANDS"] = ordered_bands
        eopatch["bbox"] = bbox

        if "BQA" in extra_bands:
            bqa_data = extra_bands["BQA"]
            eopatch[FeatureType.MASK]["CLM"] = bqa_data[
                                               np.newaxis, :, :, np.newaxis
                                               ]

        eopatch[FeatureType.TIMESTAMP] = [metadata["product_time"]]

        # Help python a little with the memory management
        gc.collect()

        if return_metadata:
            return eopatch, metadata
        return eopatch


class Sentinel2Loader(BaseLoaderTask):
    """Task for importing Sentinel SAFE data into an EOPatch."""

    _supported_resolutions = SENTINEL2_SUPPORTED_RESOLUTIONS
    _ordered_bands = tuple(SENTINEL2_BANDS_RESOLUTION.keys())
    _google_cloud_bucket = "gs://gcp-public-data-sentinel-2"
    _rasterio_driver = "JP2OpenJPEG"

    def __init__(
            self,
            folder,
            bands=None,
            resolution=20,
            data_collection=DataCollection.SENTINEL2_L1C,
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
        self.data_collection = data_collection

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
            metadata_file = glob.glob(
                os.path.join(product_path, "S2*_OPER_*.xml")
            )
            if len(metadata_file):
                metadata_file = os.path.join(product_path, metadata_file[0])
            else:
                raise RuntimeError(
                    f"Metadata file not found in {product_path}"
                )

        tree = etree.parse(metadata_file)
        root = tree.getroot()
        images_elements = root.findall(".//Granule/IMAGE_FILE")

        images_elements_txt = [
            element.text.strip() for element in images_elements
        ]
        band_files = {
            element_txt.split("_")[-1]: os.path.join(
                product_path, f"{element_txt}.jp2"
            )
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
            metadata["quantification_value"] = int(
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

    def _get_product_path(self, product_id):
        """
        Returns the local path where the product id is stored, relative to the archive's
        root folder. This path mimics exactly the Google Cloud storage structure.
        """
        # S2A_MSIL1C_20151001T142056_N0204_R010_T20JLQ_20151001T143019.SAFE
        tile = product_id.split("_")[5][1:]  # 20JLQ
        sub_dirs = os.path.join(
            "tiles", tile[:2], tile[2], tile[3:]
        )  # tiles/20/J/LQ
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


class LandsatLoader(BaseLoaderTask):
    """Task for Loading Landsat5 and Landsat8 images into an EOPATCH."""

    _supported_resolutions = LANDSAT_SUPPORTED_RESOLUTIONS
    _google_cloud_bucket = "gs://gcp-public-data-landsat"
    _rasterio_driver = "GTiff"

    def __init__(
            self,
            folder,
            resolution=30,
            spacecraft=5,
            bands=None,
            reflectance=True,
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
        else:
            # landsat5
            self._ordered_bands = tuple(LANDSAT5_BANDS_RESOLUTION.keys())
            self._filter_values = [1, 672]

        self.reflectance = reflectance
        self.raw_metadata = None

        super().__init__(folder, resolution=resolution, bands=bands, **kwargs)

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
                line.split("=")
                for line in open(metadata_file[0])
                if "=" in line
            )
        }

        metadata = {
            "cloud_cover": float(
                self.raw_metadata.get("CLOUD_COVER_LAND", None)
            )
        }

        if all(
                key in self.raw_metadata
                for key in ("DATE_ACQUIRED", "SCENE_CENTER_TIME")
        ):
            _time = self.raw_metadata["SCENE_CENTER_TIME"].split(".")[0]
            _date = self.raw_metadata["DATE_ACQUIRED"]

            metadata["product_time"] = datetime.strptime(
                f"{_date}T{_time}", "%Y-%m-%dT%H:%M:%S"
            )
            # If the timestamp is not present in the metadata, we cannot use the file
            # name since it does not contains the time, only the date.

        base_bands = [
            band for band in self._ordered_bands if band.upper() != "BQA"
        ]

        band_files = {
            band: os.path.join(product_path, self.raw_metadata[key])
            for band, key in (
                (_band, f"FILE_NAME_BAND_{int(_band[1:])}")
                for _band in base_bands
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

    def post_process_band(self, raster, band):
        if band.upper() == "BQA":
            return calibrate_landsat_bqa(
                raster, self._filter_values, close=True
            )
        else:
            return calibrate_landsat8(
                raster, band, self.raw_metadata, reflectance=self.reflectance
            )

    def _get_is_valid_data(self, raster):
        """"""
        return get_is_valid_mask(raster, filter_values=[0, 0])

    def _is_download_needed(self, filename):
        """Return True if the filename correspond to a selected band."""
        bands = [band.replace("B0", "B") for band in self.bands]
        return any([filename.endswith(f"{band}.TIF") for band in bands])
