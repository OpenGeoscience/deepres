import math
import os

import mercantile
import rasterio
from rasterio.warp import transform_bounds
from supermercado import burntiles


def get_tile_bounds(tile):
    return mercantile.xy_bounds(*tile)


def get_zoom_level_for_pixel_size(pixel_size):

    # Earth circumference for web mercator
    CIRCUMFERENCE = 20037508.342789244
    return math.log(CIRCUMFERENCE / pixel_size, 2) - 7


def get_bounds_from_raster(path):
    with rasterio.open(path) as raster:
        bounds = raster.bounds
        bbox = transform_bounds(raster.crs, {"init": "epsg:4326"}, *bounds)
        name = raster.name

    return [
        {
            "type": "Feature",
            "bbox": bbox,
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ],
            },
            "properties": {"title": name, "filename": os.path.basename(name)},
        }
    ]


def get_list_of_mercator_tiles(path):
    bounds = get_bounds_from_raster(path)
    with rasterio.open(path) as src:
        pixel_size = src.res[0]
        zoom_level = get_zoom_level_for_pixel_size(pixel_size)
        tile_set = burntiles.burn(bounds, zoom_level)

    return tile_set
