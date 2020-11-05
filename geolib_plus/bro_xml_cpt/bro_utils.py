"""
BRO XML CPT database reader, indexer and parser.

Enables searching of very large XML CPT database dumps.
In order to speed up these operations, an index will
be created by searching for `featureMember`s (CPTs) in the xml
if it does not yet exist next to the XML file.

The index is stored next to the file and stores the xml
filesize to validate the xml database is the same. If not,
we assume the database is new and a new index will be created
as well. The index itself is an array with columns that store
the x y location of the CPT and the start/end bytes in the XML file.

As of January 2019, almost a 100.000 CPTs are stored in the XML
and creating the index will take 5-10min depending on disk performance.

"""

import logging
from io import StringIO
import pickle
import mmap
from os.path import exists, splitext
from os import stat, name
from zipfile import ZipFile


# External modules
from lxml import etree
import numpy as np
from scipy.spatial import cKDTree as KDTree
import pandas as pd
import pyproj
from lxml.etree import _Element
from pathlib import Path
from typing import Dict, List, Union, Iterable, Optional
from pydantic import BaseModel
from typing import TypeVar

from geolib_plus.cpt_base_model import CptReader
from .validate_bro import validate_bro_cpt

# Types not included in typing
PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
# Constants for XML parsing
searchstring = b"<gml:featureMember>"
footer = b"</gml:FeatureCollection>"

req_columns = ["penetrationLength", "coneResistance", "localFriction", "frictionRatio"]


ns = "{http://www.broservices.nl/xsd/cptcommon/1.1}"
ns2 = "{http://www.broservices.nl/xsd/dscpt/1.1}"
ns3 = "{http://www.opengis.net/gml/3.2}"
ns4 = "{http://www.broservices.nl/xsd/brocommon/3.0}"
ns5 = "{http://www.opengis.net/om/2.0}"

nodata = -999999
to_epsg = "28992"


def query_index(index, x, y, radius=1000.0):
    """Query database for CPTs
    within radius of x, y.

    :param index: Index is a array with columns: x y begin end
    :type index: np.array
    :param x: X coordinate
    :type x: float
    :param y: Y coordinate
    :type y: float
    :param radius: Radius (m) to use for searching. Defaults to 1000.
    :type radius: float
    :return: 2d array of start/end (columns) for each location (rows).
    :rtype: np.array
    """

    # Setup KDTree based on points
    npindex = np.array(index)
    tree = KDTree(npindex[:, 0:2])

    # Query point and return slices
    points = tree.query_ball_point((float(x), float(y)), float(radius))

    # Return slices
    return npindex[points, 2:4].astype(np.int64)


class XMLBroColumnValues(BaseModel):
    penetrationLength: Union[Iterable, None] = None
    depth: Union[Iterable, None] = None
    elapsedTime: Union[Iterable, None] = None
    coneResistance: Union[Iterable, None] = None
    correctedConeResistance: Union[Iterable, None] = None
    netConeResistance: Union[Iterable, None] = None
    magneticFieldStrengthX: Union[Iterable, None] = None
    magneticFieldStrengthY: Union[Iterable, None] = None
    magneticFieldStrengthZ: Union[Iterable, None] = None
    magneticFieldStrengthTotal: Union[Iterable, None] = None
    electricalConductivity: Union[Iterable, None] = None
    inclinationEW: Union[Iterable, None] = None
    inclinationNS: Union[Iterable, None] = None
    inclinationX: Union[Iterable, None] = None
    inclinationY: Union[Iterable, None] = None
    inclinationResultant: Union[Iterable, None] = None
    magneticInclination: Union[Iterable, None] = None
    magneticDeclination: Union[Iterable, None] = None
    localFriction: Union[Iterable, None] = None
    poreRatio: Union[Iterable, None] = None
    temperature: Union[Iterable, None] = None
    porePressureU1: Union[Iterable, None] = None
    porePressureU2: Union[Iterable, None] = None
    porePressureU3: Union[Iterable, None] = None
    frictionRatio: Union[Iterable, None] = None


class XMLBroFullData(XMLBroColumnValues):
    id: Optional[str]
    location_x: Optional[float] = None
    location_y: Optional[float] = None
    offset_z: Optional[float] = None
    predrilled_z: Optional[float] = None
    a: Optional[float] = 0.80
    vertical_datum: Optional[str]
    local_reference: Optional[str]
    quality_class: Optional[str]
    cone_penetrometer_type: Optional[str]
    cpt_standard: Optional[str]
    result_time: Optional[str]
    dataframe: Optional[PandasDataFrame]

    @property
    def columns_string_list(self):
        return list(dict(XMLBroColumnValues()).keys())


class XMLBroCPTReader(CptReader):
    bro_data: XMLBroFullData = XMLBroFullData()
    water_measurement_type: List = []

    @property
    def __water_measurement_types(self):
        return [
            "porePressureU1",
            "porePressureU2",
            "porePressureU3",
        ]

    def parse_bro_to_cpt(self, xml_file_path: Path):
        # read the BRO_XML into Memory
        xml = self.xml_to_byte_string(xml_file_path)

        # parse the BRO_XML to BRO CPT Dataset
        self.parse_bro_xml(xml)

    @staticmethod
    def xml_to_byte_string(fn: Path) -> bytes:
        """
        Opens an xml-file and returns a byte-string
        :param fn: xml file name
        :return: byte-string of the xml file
        """
        ext = splitext(fn)[1]
        if ext == ".xml":
            with open(fn, "r") as f:
                # memory-map the file, size 0 means whole file
                xml_bytes = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)[:]
        return xml_bytes

    def search_values_in_root(self, root: _Element, search_item: str) -> str:
        for loc in root.iter(search_item):
            return loc.text

    def find_availed_data_columns(self, root: _Element) -> List:
        """ Find which columns are not empty. """
        avail_columns = []
        for parameters in root.iter(ns + "parameters"):
            for parameter in parameters:
                if parameter.text == "ja":
                    avail_columns.append(parameter.tag[len(ns) :])
        return avail_columns

    def parse_bro_xml(self, xml: bytes):
        """Parse bro CPT xml.
        Searches for the cpt data, but also
        - location
        - offset z
        - id
        - predrilled_z
        TODO Replace iter by single search
        as iter can give multiple results

        :param xml: XML bytes
        :returns: dict -- parsed CPT data + metadata
        """
        root = etree.fromstring(xml)

        # Location
        x, y = self.parse_xml_location(xml)
        self.bro_data.location_x = float(x)
        self.bro_data.location_y = float(y)

        # fill in the data structure from bro
        self.get_all_data_from_bro(root=root)

        # Find which columns are not empty
        avail_columns = self.find_availed_data_columns(root=root)

        # Determine if all data is available
        self.are_all_required_data_available(avail_columns=avail_columns)

        # Parse data array, replace nodata, filter and sort
        for cpt in root.iter(ns + "conePenetrationTest"):
            for element in cpt.iter(ns + "values"):
                # Load string data and parse as 2d array
                sar = StringIO(element.text.replace(";", "\n"))
                ar = np.loadtxt(sar, delimiter=",", ndmin=2)

                # Check shape of array
                found_rows, found_columns = ar.shape
                if found_columns != len(self.bro_data.columns_string_list):
                    logging.warning(
                        "Data has the wrong size! {} columns instead of {}".format(
                            found_columns, len(self.bro_data.columns_string_list)
                        )
                    )
                    return

                # Replace nodata constant with nan
                # Create a DataFrame from array
                # and sort by depth
                ar[ar == nodata] = np.nan
                df = pd.DataFrame(ar, columns=self.bro_data.columns_string_list)
                df = df[avail_columns]
                df.sort_values(by=["penetrationLength"], inplace=True)

            self.bro_data.dataframe = df
        return

    def all_single_data_available(self) -> bool:
        return None not in [
            self.bro_data.id,
            self.bro_data.location_x,
            self.bro_data.location_y,
            self.bro_data.offset_z,
            self.bro_data.predrilled_z,
            self.bro_data.a,
            self.bro_data.vertical_datum,
            self.bro_data.local_reference,
            self.bro_data.quality_class,
            self.bro_data.cone_penetrometer_type,
            self.bro_data.cpt_standard,
            self.bro_data.result_time,
        ]

    def are_all_required_data_available(self, avail_columns: List) -> None:
        """ Determine if all data is available """
        meta_usable = self.all_single_data_available()
        data_usable = all([col in avail_columns for col in req_columns])
        if not (meta_usable and data_usable):
            logging.warning("CPT with id {} misses required data.".format(data.id))
            return None

    @staticmethod
    def parse_xml_location(tdata: bytes):
        """Return x y of location.
        TODO Don't user iter
        :param tdata: XML bytes
        :returns: list -- of x y string coordinates

        Will transform coordinates not in EPSG:28992
        """
        root = etree.fromstring(tdata)
        crs = None

        for loc in root.iter(ns2 + "deliveredLocation"):
            for point in loc.iter(ns3 + "Point"):
                srs = point.get("srsName")
                if srs is not None and "EPSG" in srs:
                    crs = srs.split("::")[-1]
                break
            for pos in loc.iter(ns3 + "pos"):
                x, y = map(float, pos.text.split(" "))
                break

        if crs is not None and crs != to_epsg:
            logging.warning("Reprojecting from epsg::{}".format(crs))
            transformer = pyproj.Transformer.from_crs(f"epsg:{crs}", f"epsg:{to_epsg}")
            x, y = transformer.transform(x, y)

        return x, y

    def get_all_data_from_bro(self, root: _Element) -> None:
        """ Extract values from bro,"""
        # BRO Id
        self.bro_data.id = self.search_values_in_root(
            root=root, search_item=ns4 + "broId"
        )
        # Norm of the cpt
        self.bro_data.cpt_standard = self.search_values_in_root(
            root=root, search_item=ns2 + "cptStandard"
        )
        # Offset to reference point
        z = self.search_values_in_root(root=root, search_item=ns + "offset")
        self.bro_data.offset_z = float(z)
        # Local reference point
        self.bro_data.local_reference = self.search_values_in_root(
            root=root, search_item=ns + "localVerticalReferencePoint"
        )
        # Vertical datum
        self.bro_data.vertical_datum = self.search_values_in_root(
            root=root, search_item=ns + "verticalDatum"
        )
        # cpt class
        self.bro_data.quality_class = self.search_values_in_root(
            root=root, search_item=ns + "qualityClass"
        )
        # cpt type and serial number
        self.bro_data.cone_penetrometer_type = self.search_values_in_root(
            root=root, search_item=ns + "conePenetrometerType"
        )
        # cpt time of result
        for cpt in root.iter(ns + "conePenetrationTest"):
            for loc in cpt.iter(ns5 + "resultTime"):
                self.bro_data.result_time = loc.text
        # Pre drilled depth
        z = self.search_values_in_root(root=root, search_item=ns + "predrilledDepth")
        # if predrill does not exist it is zero
        if not z:
            z = 0.0
        self.bro_data.predrilled_z = float(z)

        # Cone coefficient - a
        a = self.search_values_in_root(
            root=root, search_item=ns + "coneSurfaceQuotient"
        )
        if a:
            self.bro_data.a = float(a)
        return None

    def read_file(self, filepath: Path) -> dict:
        # validate bro xml file
        validate_bro_cpt(filepath)

        self.parse_bro_to_cpt(filepath)

        # add the BRO_XML attributes to CPT structure
        return self.__parse_bro()

    def __parse_bro(
        self,
        minimum_length: int = 5,
        minimum_samples: int = 50,
        minimum_ratio: float = 0.1,
        convert_to_kPa: bool = True,
    ):
        """
        Parse the BRO information into the object structure

        Parameters
        ----------
        :param cpt: BRO cpt dataset
        :param minimum_length: (optional) minimum length that cpt files needs to have
        :param minimum_samples: (optional) minimum samples that cpt files needs to have
        :param minimum_ratio: (optional) minimum ratio of positive values that cpt files needs to have
        :param convert_to_kPa: (optional) convert units to kPa
        :return:
        """
        # create result dictionary
        result_dictionary = {}

        # remove NAN row from the dataframe
        for key in self.bro_data.dataframe:
            self.bro_data.dataframe = self.bro_data.dataframe.dropna(subset=[key])

        # check if file contains data
        if len(self.bro_data.dataframe.penetrationLength) == 0:
            message = "File " + self.bro_data.id + " contains no data"
            return message

        # check if data is different than zero:
        keys = ["penetrationLength", "coneResistance", "localFriction", "frictionRatio"]
        for k in keys:
            if all(self.bro_data.dataframe[k] == 0):
                message = "File " + self.bro_data.id + " contains empty data"
                return message

        # parse cpt file name
        result_dictionary["name"] = self.bro_data.id
        # parse coordinates
        result_dictionary["coordinates"] = [
            self.bro_data.location_x,
            self.bro_data.location_y,
        ]

        # parse reference datum
        result_dictionary["vertical_datum"] = (
            self.bro_data.vertical_datum if self.bro_data.vertical_datum else []
        )

        # parse local reference point
        result_dictionary["local_reference"] = (
            self.bro_data.local_reference if self.bro_data.vertical_datum else []
        )

        # parse quality class
        result_dictionary["quality_class"] = (
            self.bro_data.quality_class if self.bro_data.vertical_datum else []
        )

        # parse cone penetrator type
        result_dictionary["cpt_type"] = (
            self.bro_data.cone_penetrometer_type if self.bro_data.vertical_datum else []
        )

        # parse cpt standard
        result_dictionary["cpt_standard"] = (
            self.bro_data.cpt_standard if self.bro_data.vertical_datum else []
        )

        # parse result time
        result_dictionary["result_time"] = (
            self.bro_data.result_time if self.bro_data.vertical_datum else []
        )

        # parse measurement type of pore pressure
        result_dictionary["water_measurement_type"] = [
            water_measurement_type
            for water_measurement_type in self.__water_measurement_types
            if water_measurement_type in self.bro_data.dataframe
        ]
        if not result_dictionary["water_measurement_type"]:
            result_dictionary["water_measurement_type"] = "no_measurements"
        else:
            result_dictionary["water_measurement_type"] = result_dictionary[
                "water_measurement_type"
            ][0]

        # check criteria of minimum length
        if (
            np.max(np.abs(self.bro_data.dataframe.penetrationLength.values))
            < minimum_length
        ):
            message = (
                "File "
                + self.bro_data.id
                + " has a length smaller than "
                + str(minimum_length)
            )
            return message

        # check criteria of minimum samples
        if len(self.bro_data.dataframe.penetrationLength.values) < minimum_samples:
            message = (
                "File "
                + self.bro_data.id
                + " has a number of samples smaller than "
                + str(minimum_samples)
            )
            return message

        # check data consistency: remove doubles depth
        self.bro_data.dataframe = self.bro_data.dataframe.drop_duplicates(
            subset="penetrationLength", keep="first"
        )

        # check if there is a pre_drill. if so pad the data
        (
            depth,
            cone_resistance,
            friction_ratio,
            local_friction,
            pore_pressure,
        ) = self.__define_pre_drill(length_of_average_points=minimum_samples)

        # parse inclination resultant
        if "inclinationResultant" in self.bro_data.dataframe:
            result_dictionary["inclination_resultant"] = self.bro_data.dataframe[
                "inclinationResultant"
            ].values
        else:
            result_dictionary["inclination_resultant"] = np.empty(len(depth)) * np.nan

        # check quality of CPT
        # if more than minimum_ratio CPT is corrupted: discard CPT
        if (
            len(cone_resistance[cone_resistance <= 0]) / len(cone_resistance)
            > minimum_ratio
            or len(cone_resistance[local_friction <= 0]) / len(local_friction)
            > minimum_ratio
        ):
            message = "File " + self.bro_data.id + " is corrupted"
            return message

        # unit in kPa is required for correlations
        unit_converter = 1000.0 if convert_to_kPa else 1.0

        # parse depth
        result_dictionary["depth"] = depth
        # parse surface level
        result_dictionary["local_reference_level"] = self.bro_data.offset_z
        # parse NAP depth
        result_dictionary["depth_to_reference"] = (
            result_dictionary["local_reference_level"] - depth
        )
        # parse tip resistance
        result_dictionary["tip"] = cone_resistance * unit_converter
        result_dictionary["tip"][result_dictionary["tip"] <= 0] = 0.0
        # parse friction
        result_dictionary["friction"] = local_friction * unit_converter
        result_dictionary["friction"][result_dictionary["friction"] <= 0] = 0.0
        # parser friction number
        result_dictionary["friction_nbr"] = friction_ratio
        result_dictionary["friction_nbr"][result_dictionary["friction_nbr"] <= 0] = 0.0
        # read a
        result_dictionary["a"] = [self.bro_data.a]
        # default water is zero
        result_dictionary["water"] = np.zeros(len(result_dictionary["depth"]))
        # if water exists parse water
        if (
            result_dictionary["water_measurement_type"]
            in self.__water_measurement_types
        ):
            result_dictionary["water"] = pore_pressure * unit_converter

        return result_dictionary

    def __define_pre_drill(self, length_of_average_points: int = 3):
        """
        Checks the existence of pre-drill.
        If predrill exists it add the average value of tip, friction and friction number to the pre-drill length.
        The average is computed over the length_of_average_points.
        If pore water pressure is measured, the pwp is assumed to be zero at surface level.
        Parameters
        ----------
        :param cpt_BRO: BRO cpt dataset
        :param length_of_average_points: number of samples of the CPT to be used to fill pre-drill
        :return: depth, tip resistance, friction number, friction, pore water pressure
        """
        starting_depth = 0
        pore_pressure = None
        depth = self.__get_depth_from_bro()
        if float(self.bro_data.predrilled_z) != 0.0:
            # if there is pre-dill add the average values to the pre-dill
            # Set the discretisation
            dicretisation = np.average(np.diff(depth))
            # find the average
            average_cone_res = np.average(
                self.bro_data.dataframe["coneResistance"][:length_of_average_points]
            )
            average_fr_ratio = np.average(
                self.bro_data.dataframe["frictionRatio"][:length_of_average_points]
            )
            average_loc_fr = np.average(
                self.bro_data.dataframe["localFriction"][:length_of_average_points]
            )
            # Define all in the lists
            local_depth = np.arange(
                starting_depth, float(self.bro_data.predrilled_z), dicretisation
            )
            local_cone_res = np.repeat(average_cone_res, len(local_depth))
            local_fr_ratio = np.repeat(average_fr_ratio, len(local_depth))
            local_loc_fr = np.repeat(average_loc_fr, len(local_depth))
            # if there is pore water pressure
            # Here the endpoint is False so that for the final of
            # local_pore_pressure I don't end up with the same value
            # as the first in the Pore Pressure array.
            for water_measurement_type in self.__water_measurement_types:
                if water_measurement_type in self.bro_data.dataframe:
                    local_pore_pressure = np.linspace(
                        0,
                        self.bro_data.dataframe[water_measurement_type].values[0],
                        len(local_depth),
                        endpoint=False,
                    )
                    pore_pressure = np.append(
                        local_pore_pressure,
                        self.bro_data.dataframe[water_measurement_type].values,
                    )
            # Enrich the Penetration Length
            depth = np.append(
                local_depth, local_depth[-1] + dicretisation + depth - depth[0]
            )
            coneresistance = np.append(
                local_cone_res, self.bro_data.dataframe["coneResistance"].values
            )
            frictionratio = np.append(
                local_fr_ratio, self.bro_data.dataframe["frictionRatio"].values
            )
            localfriction = np.append(
                local_loc_fr, self.bro_data.dataframe["localFriction"].values
            )
        else:
            # No predrill existing: just parsing data
            depth = depth - depth[0]
            coneresistance = self.bro_data.dataframe["coneResistance"].values
            frictionratio = self.bro_data.dataframe["frictionRatio"].values
            localfriction = self.bro_data.dataframe["localFriction"].values
            # if there is pore water pressure
            for water_measurement_type in self.__water_measurement_types:
                if water_measurement_type in self.bro_data.dataframe:
                    pore_pressure = self.bro_data.dataframe[
                        water_measurement_type
                    ].values
        # correct for missing samples in the top of the CPT
        if depth[0] > 0:
            # add zero
            depth = np.append(0, depth)
            coneresistance = np.append(
                np.average(
                    self.bro_data.dataframe["coneResistance"][:length_of_average_points]
                ),
                coneresistance,
            )
            frictionratio = np.append(
                np.average(
                    self.bro_data.dataframe["frictionRatio"][:length_of_average_points]
                ),
                frictionratio,
            )
            localfriction = np.append(
                np.average(
                    self.bro_data.dataframe["localFriction"][:length_of_average_points]
                ),
                localfriction,
            )
            # if there is pore water pressure
            for water_measurement_type in self.__water_measurement_types:
                if water_measurement_type in self.bro_data.dataframe:
                    pore_pressure = np.append(
                        np.average(
                            self.bro_data.dataframe[water_measurement_type][
                                :length_of_average_points
                            ]
                        ),
                        pore_pressure,
                    )
        return depth, coneresistance, frictionratio, localfriction, pore_pressure

    def __get_depth_from_bro(self) -> np.ndarray:
        """
        If depth is present in the bro cpt and is valid, the depth is parsed from depth
        elseif resultant inclination angle is present and valid in the bro cpt, the penetration length is corrected with
        the inclination angle.
        if both depth and inclination angle are not present/valid, the depth is parsed from the penetration length.
        :param cpt_BRO: dataframe
        :return:
        """
        cpt_BRO = self.bro_data.dataframe
        depth = np.array([])
        if "depth" in cpt_BRO:
            if bool(cpt_BRO["depth"].values.all()):
                depth = cpt_BRO["depth"].values
        elif "inclinationResultant" in cpt_BRO:
            if bool(cpt_BRO["inclinationResultant"].values.all()):
                depth = self.calculate_corrected_depth(
                    cpt_BRO["penetrationLength"].values,
                    cpt_BRO["inclinationResultant"].values,
                )
        else:
            depth = cpt_BRO["penetrationLength"].values
        return depth

    def calculate_corrected_depth(
        self, penetration_length: Iterable, inclination: Iterable
    ) -> Iterable:
        """
        Correct the penetration length with the inclination angle


        :param penetration_length: measured penetration length
        :param inclination: measured inclination of the cone
        :return: corrected depth
        """
        corrected_d_depth = np.diff(penetration_length) * np.cos(
            np.radians(inclination[:-1])
        )
        corrected_depth = np.concatenate(
            (
                penetration_length[0],
                penetration_length[0] + np.cumsum(corrected_d_depth),
            ),
            axis=None,
        )
        return corrected_depth
