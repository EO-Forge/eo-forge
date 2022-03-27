"""
Helper functions for raster datasets
====================================

.. autosummary::
    :toctree: ../generated/

    clip_raster
    reproject_raster_to_bbox
    reproject_raster_north_south
    check_resample
    get_raster_polygon
    resample_raster
    write_mem_raster
    write_raster
    check_raster_clip_crs
    check_raster_shape_match
    get_nodata_mask
    apply_nodata_mask
    apply_isvalid_mask
    get_is_valid_mask
    get_raster_data_and_profile
    shapes2array
"""
import warnings

import numpy as np
import rasterio as rio
import rasterio.mask as rasterio_mask
from geopandas import GeoDataFrame
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import Resampling, calculate_default_transform
from rasterio.warp import reproject
from rasterio.warp import reproject as rasterio_reproject
from shapely.geometry import box


def clip_raster(
    raster,
    bbox,
    enable_transform=True,
    close=False,
    crop=True,
    nodata=0,
    all_touched=True,
    filled=True,
    hard_bbox=False,
):
    """
    Clip raster to provided BBox.
    If enable_transform

    Parameters
    ----------
    raster: rasterio dataset
        Input raster.
    bbox: Geopandas Dataframe.
        Region of Interest.
    crop: bool
        Whether to crop the raster to the extent of the roi_bbox.
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
    filled: boolean
        If True, the pixels outside the features will be set to nodata. If False,
        the output array will contain the original pixel data, and only the mask
        will be based on shapes. Defaults to True.
    hard_bbox: bool
        If True, adjust the extent of raster to match the BBox. Otherwise, leave the
        raster's original shape and mask the values outside the bbox.

    Returns
    -------
    clipped raster: opened rasterio.MemoryFile
    """
    if not isinstance(bbox, GeoDataFrame):
        raise TypeError("The input BBox should be a Geopandas Dataframe.")
    profile = raster.profile
    bbox_epsg = bbox.crs.to_epsg()
    raster_epsg = raster.crs.to_epsg()
    if (bbox_epsg != raster_epsg) and enable_transform:
        warnings.warn(
            f"Transforming bbox crs from {bbox_epsg} to {raster_epsg}",
            stacklevel=0,
        )
        bbox = bbox.to_crs(raster_epsg)
    else:
        assert bbox_epsg == raster_epsg, "CRS Mismatch"

    try:
        # Mask the raster with nodata outside the roi_bbox shapes.
        data, data_transform = rasterio_mask.mask(
            raster,
            bbox.geometry.values,
            crop=crop,
            nodata=nodata,
            filled=filled,
            all_touched=all_touched,
        )
    except ValueError as e:
        if hard_bbox:
            raise e
        else:
            data, data_transform = rasterio_mask.mask(
                raster,
                bbox.geometry.values,
                crop=False,
                nodata=nodata,
                filled=filled,
                all_touched=all_touched,
            )
            # keep partial match
            profile.update({"match": "partial"})

    # update profile
    profile.update(
        {
            "driver": "GTiff",  # Update format
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": data_transform,
        }
    )
    if close:
        raster.close()

    return write_mem_raster(data, **profile)


def reproject_raster_to_bbox(raster, roi_bbox, close=False):
    """Reproject raster to a new bbox.

    Parameters
    ----------
    raster: rasterio dataset
        Input raster.
    bbox:
    close: bool
        Close the input raster dataset before returning the clipped raster.
    """

    left, bottom, right, top = roi_bbox.total_bounds
    window = rio.windows.from_bounds(
        transform=raster.transform, left=left, bottom=bottom, right=right, top=top
    )
    raster_data = raster.read()
    width = int(window.width)
    height = int(window.height)
    new_data_shape = (raster_data.shape[0], height, width)
    reprojected_data = np.zeros(new_data_shape, raster_data.dtype)

    dst_transform = rio.windows.transform(window, raster.transform)
    rasterio_reproject(
        raster_data,
        reprojected_data,
        src_transform=raster.transform,
        src_crs=raster.crs,
        dst_transform=dst_transform,
        dst_crs=raster.crs,
        resampling=Resampling.bilinear,
    )
    profile = raster.profile.copy()
    profile.update(
        {
            "crs": raster.crs,
            "transform": dst_transform,
            "width": width,
            "height": height,
        }
    )

    if close:
        raster.close()
    return write_mem_raster(reprojected_data, **profile)


def reproject_raster_north_south(raster, close=False):
    """Reproject raster north-south if needed.

    Parameters
    ----------
    raster: rasterio dataset
        Input raster.
    close: bool
        Close the input raster dataset before returning the clipped raster.
    """
    # check projection

    max_y = raster.bounds.top
    data = None  # noqa

    if max_y >= 0:
        return raster

    # max_y < 0 , requires transform
    epsg_str = str(raster.crs.to_epsg())
    epsg_str = epsg_str[:2] + "7" + epsg_str[3:]
    dst_crs = f"EPSG:{epsg_str}"

    transform, width, height = calculate_default_transform(
        raster.crs, dst_crs, raster.width, raster.height, *raster.bounds
    )
    profile = raster.profile.copy()
    profile.update(
        {
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
        }
    )

    data_raster = raster.read()
    data = np.zeros((raster.count, height, width), dtype=data_raster.dtype)

    reproject(
        source=data_raster,
        destination=data,
        src_transform=raster.transform,
        src_crs=raster.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    if close:
        raster.close()
    return write_mem_raster(data, **profile)


def check_resample(raster, target_resolution):
    """
    Check if resampling is required.

    Parameters
    ----------
    raster: rasterio dataset
        Input dataset.
    target_resolution: float
        Required resolution in meters.

    Returns
    -------
    resample: bool
        True if resample is required
    scale_factor: float
        Calculated scale factor
    true_resolution: float
        Original pixel size
    """
    raster_res = raster.res
    assert raster_res[0] == raster_res[1]  # square pixel
    true_resolution = raster_res[0]
    if not np.isclose(true_resolution, target_resolution):
        scale_factor = target_resolution / true_resolution
        resample_ = True
    else:
        scale_factor = 1
        resample_ = False
    return resample_, scale_factor, true_resolution


def get_raster_polygon(raster, ccw=True):
    """
    Return a the raster's bounding box polygon.

    Parameters
    ----------
    raster: opened raster instance
        Input raster.
    ccw: bool
        Counter-clockwise order (True) or not (False)

    Returns
    -------
    Polygon: shapely.geometry.box
    """
    return box(
        raster.bounds.left,
        raster.bounds.top,
        raster.bounds.right,
        raster.bounds.bottom,
        ccw=ccw,
    )


def resample_raster(raster, scale, close=False):
    """
    Resample a raster dataset by multiplying the pixel size by the scale factor,
    and update the dataset shape.
    For example, given a pixel size of 250m, dimensions of (1024, 1024) and
    a scale of 2, the resampled raster will have an output pixel size of 500m
    and dimensions of (512, 512) given a pixel size of 250m, dimensions of
    (1024, 1024) and a scale of 0.5, he resampled raster would have an output
    pixel size of 125m and dimensions of (2048, 2048) returns a `DatasetReader`
    instance from either a filesystem raster or MemoryFile (if out_path is None).

    Returns
    -------
    resampled raster: opened rasterio.MemoryFile
    """
    t = raster.transform
    transform = Affine(t.a * scale, t.b, t.c, t.d, t.e * scale, t.f)
    height = int(raster.height / scale)
    width = int(raster.width / scale)

    profile = raster.profile
    profile.update(transform=transform, driver="GTiff", height=height, width=width)

    data = raster.read(
        out_shape=(raster.count, height, width),
        resampling=Resampling.bilinear,
    )

    if close:
        raster.close()
    return write_mem_raster(data, **profile)


def write_mem_raster(data, **profile):
    """Create a rasterio.MemoryFile instance. Returns the opened instance dataset."""
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:  # Open as DatasetWriter
            dataset.write(data)
        return memfile.open()


def write_raster(file_path, data, **profile):
    with rio.open(file_path, "w", **profile) as dataset:  # Open as DatasetWriter
        dataset.write(data)

    return rio.open(file_path)  # Reopen as DatasetReader


def check_raster_clip_crs(
    raster_dataset, roi_bbox, enable_transform=True, verbose=False
):
    """
    Check band crs match to provided bbox

    Parameters
    ----------
    raster_dataset: raster instance opened by rasterio
    roi_bbox: geodataframe instance
    enable_transform: bool to enable trasnform or raise error

    Returns
    -------
    Returns a DatasetReader instance from either a filesystem raster
    or MemoryFile (if out_path is None)
    """

    # Check CRS Match
    roi_epsg = roi_bbox.crs.to_epsg()
    raster_crs = raster_dataset.crs.to_epsg()
    if enable_transform:
        # check crs
        if roi_epsg == raster_crs:
            # ok
            pass
        else:
            if verbose:
                # transform roi_bbox
                print(
                    f"WARNING: Transforming bbox crs from {roi_epsg} \
                        to {raster_crs}"
                )
            roi_bbox = roi_bbox.to_crs(raster_crs)

    else:
        assert roi_epsg == raster_crs, "CRS Mismatch"

    return roi_bbox


def check_raster_shape_match(raster, roi_shape, enable_transform=True):
    """Check raster shape match."""
    roi_update = check_raster_clip_crs(
        raster, roi_shape, enable_transform=enable_transform
    )
    raster_polygon = get_raster_polygon(raster)
    roi_geometry = roi_update.geometry.unary_union
    intersect_ = raster_polygon.intersects(roi_geometry)
    contains_ = raster_polygon.contains(roi_geometry)
    if intersect_ and contains_:
        full_match = True
        area = 1
    elif intersect_:
        full_match = False
        area = raster_polygon.intersection(roi_geometry).area / roi_geometry.area
    else:
        full_match = None
        area = 0
    return full_match, area


def get_nodata_mask(raster_dataset, nodata=0, out_path=None):
    """
    Get nodata_mask from raster by filtering values.

    Parameters
    ----------
    raster: raster instance (rasterio open)
    nodata: value
    out_path: file path (optional, else return in memory)

    Returns
    -------
    In-memory raster mask or wrtitten to disk
    """
    # as numpy (NxM)
    data = raster_dataset.read()
    mask = np.where(data == nodata, True, False).astype(rio.uint8)

    profile = raster_dataset.profile.copy()
    profile.update(
        {
            "count": 1,
            "dtype": rio.uint8,
        }
    )
    if out_path is None:
        return write_mem_raster(mask, **profile)
    else:
        return write_raster(out_path, mask, **profile)


def apply_nodata_mask(raster_dataset, raster_mask, nodata=0, out_path=None):
    """
    Apply nodata_mask to raster.

    Parameters
    ----------
    raster: raster instance (rasterio open)
    raster mask: raster instance (rasterio open)
    out_path: file path (optional, else return in memory)

    Returns
    -------
    In-memory raster mask or written to disk
    """
    data = raster_dataset.read()
    mask = raster_mask.read(1)
    data_masked = []
    for i, band in enumerate(data, 1):
        data_masked.append(np.where(mask > 0, nodata, band))

    data = np.stack(data_masked, axis=0)

    profile = raster_dataset.profile.copy()
    profile.update(
        {
            "nodata": nodata,
        }
    )

    if out_path is None:
        return write_mem_raster(data, **profile)
    else:
        return write_raster(out_path, data, **profile)


def apply_isvalid_mask(raster_dataset, raster_mask, nodata=0, out_path=None):
    """
    Apply_isvalid_mask to raster

    Parameters
    ----------
    raster: raster instance (rasterio open)
    raster mask: raster instance (rasterio open)
    out_path: file path (optional, else return in memory)

    Returns
    -------
    in-memory raster mask or written to disk
    """
    data = raster_dataset.read()
    mask = raster_mask.read(1)
    data_masked = []
    for i, band in enumerate(data, 1):
        data_masked.append(np.where(mask > 0, band, nodata))

    data = np.stack(data_masked, axis=0)

    profile = raster_dataset.profile.copy()
    profile.update(
        {
            "nodata": nodata,
        }
    )

    if out_path is None:
        return write_mem_raster(data, **profile)
    else:
        return write_raster(out_path, data, **profile)


def get_is_valid_mask(raster_dataset, filter_values=(0, 0), out_path=None):
    """
    Get is_valid_mask from raster by filtering values.

    Parameters
    ----------
    raster: raster instance (rasterio open)
    filter_values: tuple
    out_path: file path (optional, else return in memory)

    Returns
    -------
    In-memory raster mask or wrtitten to disk
    """
    # as numpy (NxM)
    data = raster_dataset.read()
    mask = ((data != filter_values[0]) & (data != filter_values[1])).astype(rio.ubyte)
    profile = raster_dataset.profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": rio.ubyte,
        }
    )
    if out_path is None:
        return write_mem_raster(mask, **profile)
    else:
        return write_raster(out_path, mask, **profile)


def get_raster_data_and_profile(raster):
    """get raster data and profile
    Parameters
    ----------
    raster: raster instance (rasterio open)

    Returns
    -------
    raster.read()
    raster.profile
    """
    return raster.read(), raster.profile


def shapes2array(shapes, raster):
    """
    Set shapes to raster.

    Parameters
    ----------
    shapes: geodataframe

    Returns
    -------
        array_: numpy array
    """

    # read geometries
    gpd_ = shapes
    array_ = rasterize(
        gpd_.geometry.values,
        out_shape=(raster.height, raster.width),
        fill=0,
        transform=raster.transform,
        default_value=1,
        dtype=rio.uint8,
    )
    return array_


def reproject_with_raster_template(src, template_ds, **kwargs):
    """reproject raster with template

    Parameters
    ----------
        src: rasterio dataset
            source to be updated
        template_ds: rasterio dataset
            raster used as template

    Returns
    -------
        raster updated (In-Memory file)

    """

    src_nodata = kwargs.get("src_nodata", 0)
    dst_nodata = kwargs.get("dst_nodata", 0)

    out_kwargs = template_ds.profile.copy()

    dst_crs = template_ds.crs
    dst_transform = template_ds.transform
    dst_height = template_ds.height
    dst_width = template_ds.width

    out_kwargs.update(
        crs=dst_crs,
        transform=dst_transform,
        width=dst_width,
        height=dst_height,
        nodata=dst_nodata,
    )

    # Adjust block size if necessary.
    if "blockxsize" in out_kwargs and dst_width < int(out_kwargs["blockxsize"]):
        del out_kwargs["blockxsize"]
    if "blockysize" in out_kwargs and dst_height < int(out_kwargs["blockysize"]):
        del out_kwargs["blockysize"]

    with MemoryFile() as memfile:
        with memfile.open(**out_kwargs) as out_raster:
            reproject(
                source=rio.band(src, list(range(1, src.count + 1))),
                destination=rio.band(out_raster, list(range(1, src.count + 1))),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=out_kwargs["transform"],
                dst_crs=out_kwargs["crs"],
                dst_nodata=dst_nodata,
                resampling=Resampling.nearest,
            )
        return memfile.open()
