"""
Helper functions for Bounding boxs
==================================

.. autosummary::
    :toctree: ../generated/

    set_buffer_on_gdf
"""


def set_buffer_on_gdf(
    gdf, buffer=50, convex_hull=True, to_epsg_=True, epsg_="EPSG:3857"
):
    """
    Set buffer and simplify (convexhull) geometry.

    Parameters
    ----------
    gdf: geopandas instance on a projected crs
    buffer: buffer to apply in meters
    convex_hull: bool to apply convex hull simplification or not
    epsg\_: epsg to be used in buffering (notice that buffer works with a projected crs)

    Returns
    -------
    gdf: updated geoDataframe
    """
    gdf_ = gdf.copy()

    crs_ = gdf_.crs
    if to_epsg_:
        gdf_.to_crs(epsg_, inplace=True)

    if convex_hull:
        gdf_.geometry = gdf_.buffer(buffer).convex_hull
    else:
        gdf_.geometry = gdf_.buffer(buffer)

    gdf_.to_crs(crs_, inplace=True)

    return gdf_
