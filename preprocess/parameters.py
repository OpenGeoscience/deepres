import luigi
import rasterio


class RasterParameter(luigi.Parameter):
    """
    Parameter whose value is a path to a raster
    """

    def parse(self, s):
        with rasterio.open(s) as dataset:
            if not dataset.crs.is_valid:
                message = "{} is not a valid geospatial raster".format(s)
                raise luigi.parameter.ParameterException(message)
        return s
