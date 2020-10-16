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
from pathlib import Path
from typing import Union, Dict, List, Iterable, Optional
from pydantic import BaseModel
from typing import TypeVar

# External modules
from lxml import etree
import numpy as np
from scipy.spatial import cKDTree as KDTree
import pandas as pd
import pyproj
from lxml.etree import _Element

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


class XMLBroCPTReader(XMLBroColumnValues):
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

    def parse_bro_xml(self, xml: bytes) -> None:
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
        self.location_x = float(x)
        self.location_y = float(y)

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
                if found_columns != len(self.columns_string_list):
                    logging.warning(
                        "Data has the wrong size! {} columns instead of {}".format(
                            found_columns, len(self.columns_string_list)
                        )
                    )
                    return None

                # Replace nodata constant with nan
                # Create a DataFrame from array
                # and sort by depth
                ar[ar == nodata] = np.nan
                df = pd.DataFrame(ar, columns=self.columns_string_list)
                df = df[avail_columns]
                df.sort_values(by=["penetrationLength"], inplace=True)

            self.dataframe = df

        return None

    def all_single_data_available(self) -> bool:
        return None not in [
            self.id,
            self.location_x,
            self.location_y,
            self.offset_z,
            self.predrilled_z,
            self.a,
            self.vertical_datum,
            self.local_reference,
            self.quality_class,
            self.cone_penetrometer_type,
            self.cpt_standard,
            self.result_time,
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
        self.id = self.search_values_in_root(root=root, search_item=ns4 + "broId")
        # Norm of the cpt
        self.cpt_standard = self.search_values_in_root(
            root=root, search_item=ns2 + "cptStandard"
        )
        # Offset to reference point
        z = self.search_values_in_root(root=root, search_item=ns + "offset")
        self.offset_z = float(z)
        # Local reference point
        self.local_reference = self.search_values_in_root(
            root=root, search_item=ns + "localVerticalReferencePoint"
        )
        # Vertical datum
        self.vertical_datum = self.search_values_in_root(
            root=root, search_item=ns + "verticalDatum"
        )
        # cpt class
        self.quality_class = self.search_values_in_root(
            root=root, search_item=ns + "qualityClass"
        )
        # cpt type and serial number
        self.cone_penetrometer_type = self.search_values_in_root(
            root=root, search_item=ns + "conePenetrometerType"
        )
        # cpt time of result
        for cpt in root.iter(ns + "conePenetrationTest"):
            for loc in cpt.iter(ns5 + "resultTime"):
                self.result_time = loc.text
        # Pre drilled depth
        z = self.search_values_in_root(root=root, search_item=ns + "predrilledDepth")
        # if predrill does not exist it is zero
        if not z:
            z = 0.0
        self.predrilled_z = float(z)

        # Cone coefficient - a
        a = self.search_values_in_root(
            root=root, search_item=ns + "coneSurfaceQuotient"
        )
        if a:
            self.a = float(a)
        return None
