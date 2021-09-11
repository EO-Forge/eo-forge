"""
Generic loaders module
======================

.. autosummary::
    :toctree: ../generated/

    BaseGenericLoader
"""
import gc
import os
import warnings
from abc import abstractmethod

import numpy as np
import rasterio

from eo_forge.utils.raster_utils import (
    resample_raster,
    clip_raster,
    check_resample,
    reproject_raster_north_south,
    check_raster_clip_crs,
    check_raster_shape_match,
    reproject_raster_to_bbox,
    apply_isvalid_mask,
    get_raster_data_and_profile,
    write_mem_raster,
    write_raster,
)
from eo_forge.utils.shapes import bbox_to_geodataframe, set_buffer_on_gdf


class BaseGenericLoader:
    """
    Base class for Landsat and Sentinel loaders.

    This class can only load data from a local storage (your laptop storage, a NFS storage,etc).
    The functionality to read directly from Cloud Buckets is not yet implemented.

    The particular raw data files are looked inside the archive folder based on their product ID.
    """

    _supported_resolutions = tuple()
    _ordered_bands = tuple()
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
            Path to the folder with the raw data.
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

    @abstractmethod
    def _get_product_path(self, product_id):
        """
        Returns the local path where the product id is stored, relative to the archive's
        root folder. This path mimics exactly the Google Cloud storage structure.
        """
        raise NotImplementedError

    @abstractmethod
    def _read_metadata(self, product_path):
        """Returns a dictionary with the product metadata."""
        raise NotImplementedError

    def post_process_band(self, raster, band):
        """Postprocess the band data (ndarray input)."""
        return raster

    @abstractmethod
    def _clean_product_id(self, product_id):
        """Cleans product id."""
        raise NotImplementedError

    @abstractmethod
    def _get_is_valid_data(self, raster):
        """Returns raster mask with valid data."""
        raise NotImplementedError

    @abstractmethod
    def _preprocess_clouds_mask(self, product_path, **kwargs):
        """Returns raster cloud mask."""
        raise NotImplementedError

    def _get_bands_data(
        self,
        metadata,
        **kwargs,
    ):
        """
        Gets data for different bands.

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
        base_bands_data_profiles = {}
        base_bands_match_ = {}

        enable_transform = kwargs.get("enable_transform", True)
        crop = kwargs.pop("crop", True)
        hard_bbox = kwargs.get("hard_bbox", False)
        bbox = kwargs.get("bbox", self.bbox)
        nodata = kwargs.get("nodata", 0)
        reproject = kwargs.get("reproject", True)
        all_touched = kwargs.get("all_touched", True)
        calibrate = kwargs.get("calibrate", True)

        clipping_flag = bbox is not None

        for band in self.bands:
            band = band.upper()
            data_file = metadata["band_files"][band]

            if len(data_file) == 0:
                warnings.warn(f"No files found for the {band} band")

            raster_dataset = rasterio.open(data_file, driver=self._rasterio_driver)

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
                full_match, area = check_raster_shape_match(
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
                    raster_dataset = reproject_raster_to_bbox(raster_dataset, roi_check)

            else:
                full_match = True
                area = 1
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

            raster_dataset = apply_isvalid_mask(raster_dataset, raster_dataset_mask)

            # Get Data
            data, data_profile = get_raster_data_and_profile(raster_dataset)

            # keep just [n x m] dimensions
            base_bands_data[band] = data.squeeze()
            base_bands_data_profiles[band] = data_profile
            base_bands_match_[band] = [full_match, area]

        # Help python a little bit with the memory management
        gc.collect()
        return base_bands_data, base_bands_data_profiles, base_bands_match_

    def _get_cloud_mask(
        self,
        raster_dataset,
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
            data: raster cloud data
            data_profile: raster cloud data profile

        """
        cloud_band = kwargs.get("cloud_band", "BQA")
        enable_transform = kwargs.get("enable_transform", True)
        crop = kwargs.pop("crop", True)
        hard_bbox = kwargs.get("hard_bbox", False)
        bbox = kwargs.get("bbox", self.bbox)
        nodata = kwargs.get("nodata", 0)
        reproject = kwargs.get("reproject", True)
        all_touched = kwargs.get("all_touched", True)
        calibrate = kwargs.get("calibrate", True)

        clipping_flag = bbox is not None

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
            full_match, area = check_raster_shape_match(
                raster_dataset, roi_check, enable_transform=enable_transform
            )

            if resample_flag:
                # Resample taking special care of the borders.
                # Add a small buffer to the BBox to take into account the borders.
                roi_check_buffered = set_buffer_on_gdf(roi_check, buffer=5 * true_pixel)

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
                raster_dataset = reproject_raster_to_bbox(raster_dataset, roi_check)

        else:
            # No BBOX
            if resample_flag:
                raster_dataset = resample_raster(
                    raster_dataset, scale_factor, close=True
                )

        if calibrate:
            # Apply postprocessing (calibration)
            raster_dataset = self.post_process_band(raster_dataset, cloud_band)

        if reproject:
            raster_dataset = reproject_raster_north_south(raster_dataset, close=True)

        # Get Data
        data, data_profile = get_raster_data_and_profile(raster_dataset)

        # Help python a little bit with the memory management
        gc.collect()
        return data.astype(rasterio.ubyte), data_profile

    def execute(
        self,
        product_id=None,
        process_clouds=True,
        write_file=None,
        raster_return_open=False,
        folder_proc_="./",
        **kwargs,
    ):
        """
        Base execute function for the loaders.

        This function loads from a local archive, the data for a given product ID.

        Parameters
        ----------
        product_id: str
            Tile product ID or name of the subfolder with the tile data.
        process_clouds: bool
            Read cloud mask from the quality data.
        write_file: str
            If not None, will be used as write suffix (before  file extension) for saving the
            processed data into a file.
        raster_return_open: bool
            If True, the opened rasterio dataset is returned in the results dictionary.
        folder_proc_: str
            Path to the folder where the processed data is saved if write_file is not None.

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
        results: python dictionary
            with raster data, cloud data and metadata, including match type
        """

        self.folder_proc_ = folder_proc_

        bbox = kwargs.pop("bbox", self.bbox)

        if os.path.isdir(product_id):
            product_path = product_id
        elif os.path.isdir(os.path.join(self.archive_folder, product_id)):
            product_path = os.path.join(self.archive_folder, product_id)
        else:
            product_path = self._get_product_path(product_id)

        if not os.path.isdir(product_path):
            raise RuntimeError(
                f"The {product_id} directory does not exist or "
                f"was not download correctly."
            )
        metadata = self._read_metadata(product_path)
        # Note: we keep metadata in the class
        metadata["product_path"] = product_path
        self.metadata_ = metadata
        self.metadata_.update(self.raw_metadata)

        (
            base_bands_data,
            base_bands_data_profiles,
            base_band_match,
        ) = self._get_bands_data(metadata, bbox=bbox, **kwargs)

        ##
        # Store the base bands in a single array
        # The bands order is the one expected by eolearn.
        ordered_bands = [
            band for band in self._ordered_bands if band in base_bands_data
        ]

        _base_bands_data = [base_bands_data[band] for band in ordered_bands]
        _base_bands_data_profiles = [
            base_bands_data_profiles[band] for band in ordered_bands
        ]
        _base_bands_match_ = [base_band_match[band] for band in ordered_bands]

        # [bands x n x m ] -> rasterio order
        data = np.stack(_base_bands_data, axis=0)
        data_dtype = _base_bands_data_profiles[0]["dtype"]

        # For export
        base_profile = _base_bands_data_profiles[0]
        base_profile.update(
            {
                "dtype": data_dtype,
                "count": len(_base_bands_data_profiles),
                "driver": "GTiff",
                "compress": "lzw",
            }
        )

        product_id_cleaned = self._clean_product_id(product_id)
        # check match
        match_ = [li[0] for li in _base_bands_match_]
        # Update write_suffix
        if all(match_):
            # all true
            match_flag = "TOTAL"
            write_end = ".TIF"
        else:
            write_end = "-PARTIAL.TIF"
            match_flag = "PARTIAL"

        if not write_file:
            raster = write_mem_raster(data, **base_profile)
            file_ = None
        else:
            file_ = os.path.join(
                self.folder_proc_,
                product_id_cleaned + write_file + write_end,
            )
            raster = write_raster(file_, data, **base_profile)
        # clean
        del data

        raster_cloud = None
        file_cloud = None
        # Quality Band
        if process_clouds:
            cloud_raster = self._preprocess_clouds_mask(
                metadata, **{"raster_base": raster, "no_data": 0}
            )
            cloud_data, cloud_profile = self._get_cloud_mask(
                cloud_raster, bbox=bbox, **kwargs
            )

            # make raster_cloud
            cloud_profile.update(
                {
                    "dtype": rasterio.uint8,
                    "count": 1,
                    "driver": "GTiff",
                    "compress": "lzw",
                }
            )
            if not write_file:
                raster_cloud = write_mem_raster(cloud_data, **cloud_profile)
            else:
                file_cloud = os.path.join(
                    self.folder_proc_,
                    product_id_cleaned + "_CLOUDS" + write_file + write_end,
                )
                raster_cloud = write_raster(file_cloud, cloud_data, **cloud_profile)
            # clean
            del cloud_data

        results_ = dict()

        if raster_return_open:
            results_.update(
                {
                    "raster": raster,
                    "raster_cloud": raster_cloud,
                    "match": match_flag,
                    "raster_path": file_,
                    "raster_cloud_path": file_cloud,
                }
            )
        else:
            raster.close()
            raster_cloud.close()
            results_.update(
                {
                    "raster": None,
                    "raster_cloud": None,
                    "match": match_flag,
                    "raster_path": file_,
                    "raster_cloud_path": file_cloud,
                }
            )
        # Help python a little with the memory management
        gc.collect()

        return results_
