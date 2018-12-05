import fnmatch
import glob
import os

from affine import Affine
import luigi
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from parameters import RasterParameter
from utils import get_list_of_mercator_tiles, get_tile_bounds
import gdal_merge

TILE_SIZE = 256


class ExtractTileFromRaster(luigi.Task):
    path = RasterParameter()
    tile = luigi.Parameter()
    output_directory = luigi.Parameter()

    def _get_tile_directory(self):
        z, x, y = (
            str(round(float(self.tile[2]), 8)),
            str(int(self.tile[0])),
            str(int(self.tile[1])),
        )
        tile_directory = os.path.join(self.output_directory, z, x, y)

        return tile_directory

    def _get_tile_name(self):
        tile_directory = self._get_tile_directory()
        tile_name = os.path.join(tile_directory, os.path.basename(self.path))
        return tile_name

    def output(self):
        tile_name = self._get_tile_name()
        return luigi.LocalTarget(tile_name)

    def run(self):
        self.output().makedirs()
        with rasterio.open(self.path) as src:
            with WarpedVRT(
                src, crs="EPSG:3857", resampling=Resampling.nearest
            ) as vrt:
                bounds = get_tile_bounds(self.tile)
                left, bottom, right, top = bounds
                dst_window = vrt.window(left, bottom, right, top)
                data = vrt.read(
                    window=dst_window,
                    out_shape=(src.count, TILE_SIZE, TILE_SIZE),
                )
                profile = vrt.profile
                profile["width"] = TILE_SIZE
                profile["height"] = TILE_SIZE
                profile["blockxsize"] = TILE_SIZE / 2
                profile["blockysize"] = TILE_SIZE / 2
                profile["driver"] = "GTiff"

                dst_transform = vrt.window_transform(dst_window)
                scaling = Affine.scale(
                    dst_window.width / TILE_SIZE, dst_window.height / TILE_SIZE
                )
                dst_transform *= scaling
                profile["transform"] = dst_transform

                tile_name = self._get_tile_name()

                with rasterio.open(tile_name, "w", **profile) as dst:
                    dst.write(data)


class IngestRaster(luigi.WrapperTask):
    path = RasterParameter()
    output_directory = luigi.Parameter()
    tile_set = luigi.ListParameter(default="")

    def requires(self):
        tile_set = self.tile_set
        if not self.tile_set:
            tile_set = get_list_of_mercator_tiles(self.path)

        yield [
            ExtractTileFromRaster(
                path=self.path,
                tile=tile,
                output_directory=self.output_directory,
            )
            for tile in tile_set
        ]


class IngestRasterDirectory(luigi.WrapperTask):
    input_directory = luigi.Parameter()
    output_directory = luigi.Parameter()

    def requires(self):
        tifs = glob.glob(os.path.join(self.input_directory, "*.tif"))
        yield [
            IngestRaster(path=tif, output_directory=self.output_directory)
            for tif in tifs
        ]


class OutputMergedList(luigi.Task):
    input_rasters = luigi.ListParameter()
    output_file = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        with open(self.output_file, "w") as f:
            for item in self.input_rasters:
                f.write("{}\n".format(item))


class MergeRasters(luigi.Task):
    input_rasters = luigi.ListParameter()
    output_raster = luigi.Parameter()

    def requires(self):
        output_file = os.path.join(
            os.path.dirname(self.output_raster), "merged_list.txt"
        )
        yield OutputMergedList(
            input_rasters=sorted(list(self.input_rasters)),
            output_file=output_file,
        )

    def output(self):
        return luigi.LocalTarget(self.output_raster)

    def run(self):
        gdal_merge.main(
            ["", "-separate", "-o", self.output_raster]
            + sorted(list(self.input_rasters))
        )


class LayoutTiles(luigi.WrapperTask):
    input_directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    ground_truth = RasterParameter()

    def _get_merged_tile_sets(self):
        tifs = glob.glob(os.path.join(self.input_directory, "*.tif"))
        merged = [get_list_of_mercator_tiles(i) for i in tifs]
        return np.concatenate(merged, axis=0).tolist()

    def requires(self):

        yield IngestRasterDirectory(
            input_directory=self.input_directory,
            output_directory=self.output_directory,
        )
        yield IngestRaster(
            path=self.ground_truth,
            output_directory=self.output_directory,
            tile_set=self._get_merged_tile_sets(),
        )


class IngestPipeline(luigi.WrapperTask):
    input_directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    ground_truth = RasterParameter()

    def _get_tile_directories(self):
        matches = []
        for root, dirnames, filenames in os.walk(self.output_directory):
            for filename in fnmatch.filter(filenames, "*.tif"):
                matches.append(root)
        return list(set(matches))

    @staticmethod
    def _get_rasters(path):
        tifs = glob.glob(os.path.join(path, "*.tif"))
        imgs = glob.glob(os.path.join(path, "*.img"))
        rasters = tifs + imgs
        # Remove merged in case it is in the list
        if "merged.tif" in rasters:
            rasters.remove("merged.tif")

        return rasters

    def requires(self):
        yield LayoutTiles(
            input_directory=self.input_directory,
            output_directory=self.output_directory,
            ground_truth=self.ground_truth,
        )
        yield [
            MergeRasters(
                input_rasters=self._get_rasters(i),
                output_raster=os.path.join(i, "merged.tif"),
            )
            for i in self._get_tile_directories()
        ]


if __name__ == "__main__":
    luigi.run()
