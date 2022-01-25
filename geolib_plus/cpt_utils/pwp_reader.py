"""
NetCFD
"""
import logging

# import packages
import netCDF4
import numpy as np
from pyproj import Transformer

transformer = Transformer.from_crs("epsg:28992", "epsg:4326")


class NetCDF:
    def __init__(self):
        self.NAP_water_level = 0  # default value of NAP
        self.lat = []  # latitude of dataset points
        self.lon = []  # longitude of dataset points
        self.data = []  # dataset
        return

    def read_cdffile(self, cdf_file):
        """
        Read water level from the NHI data portal

        :param cdf_file: path to the netCDF file
        """

        # open file
        dataset = netCDF4.Dataset(cdf_file)

        # read coordinates
        lat = dataset.variables["lat"][:]
        lon = dataset.variables["lon"][:]
        self.data = dataset.variables["Band1"][:]

        # only use valid data
        mlon, mlat = np.meshgrid(lon, lat)
        self.lat = mlat[~self.data.mask]
        self.lon = mlon[~self.data.mask]
        self.data = self.data[~self.data.mask]

        dataset.close()

    def query(self, X, Y):
        """
        Query data for the point X, Y

        :param X: coordinate X [RD coordinates]
        :param Y: coordinate Y [RD coordinates]
        """

        # convert to coordinate system of netCDF
        y_lat, x_lon = transformer.transform(X, Y)

        # find nearest cell
        id_min = ((self.lon - x_lon) ** 2 + (self.lat - y_lat) ** 2).argmin()
        self.NAP_water_level = self.data[id_min]
        logging.debug(
            "For given x: {} (lon: {}) and y: {} (lat: {}), nearest point with data is {} {}".format(
                X, x_lon, Y, y_lat, self.lon[id_min], self.lat[id_min]
            )
        )
