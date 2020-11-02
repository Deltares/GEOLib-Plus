import pytest
from tests.utils import TestUtils
from pathlib import Path
import json
import os
from lxml import etree
import pandas as pd
import numpy as np
import mmap
import logging

from geolib_plus.bro_xml_cpt import bro_utils as bro
from geolib_plus.bro_xml_cpt.bro_utils import XMLBroCPTReader

# todo JN: write unit tests
class TestBroUtil:
    @pytest.mark.unittest
    def test_columns_string_list(self):
        # initialise inputs
        columns = [
            "penetrationLength",
            "depth",
            "elapsedTime",
            "coneResistance",
            "correctedConeResistance",
            "netConeResistance",
            "magneticFieldStrengthX",
            "magneticFieldStrengthY",
            "magneticFieldStrengthZ",
            "magneticFieldStrengthTotal",
            "electricalConductivity",
            "inclinationEW",
            "inclinationNS",
            "inclinationX",
            "inclinationY",
            "inclinationResultant",
            "magneticInclination",
            "magneticDeclination",
            "localFriction",
            "poreRatio",
            "temperature",
            "porePressureU1",
            "porePressureU2",
            "porePressureU3",
            "frictionRatio",
        ]
        cl_cpt = bro.XMLBroCPTReader()
        test_columns = cl_cpt.bro_data.columns_string_list
        assert test_columns == columns

    @pytest.mark.unittest
    def test_xml_to_byte_string(self):
        # define input path to xml
        test_folder = Path(TestUtils.get_local_test_data_dir("cpt/bro_xml"))
        filename = "CPT000000003688_IMBRO_A.xml"
        test_file = test_folder / filename
        # run test
        model = bro.XMLBroCPTReader.xml_to_byte_string(fn=test_file)
        # test results
        assert model

    @pytest.mark.unittest
    def test_xml_to_byte_string_wrong_path(self):
        # define input path to xml
        test_folder = Path(TestUtils.get_local_test_data_dir("cpt/bro_xml"))
        filename = "wrong.xml"
        test_file = test_folder / filename
        with pytest.raises(FileNotFoundError):
            bro.XMLBroCPTReader.xml_to_byte_string(fn=test_file)

    @pytest.mark.unittest
    def test_search_values_in_root(self):
        root = etree.Element("parentofall")
        root.text = "leads to"

        bold = etree.SubElement(root, "childofall")
        bold.text = "here it is"
        # initialise model
        model = bro.XMLBroCPTReader()
        # run test
        result = model.search_values_in_root(root=root, search_item="childofall")
        assert result == "here it is"

    @pytest.mark.unittest
    def test_search_values_in_root_does_not_exist(self):
        root = etree.Element("parentofall")
        root.text = "leads to"

        bold = etree.SubElement(root, "childofall")
        bold.text = "here it is"
        # initialise model
        model = bro.XMLBroCPTReader()
        # run test
        result = model.search_values_in_root(root=root, search_item="childofwrong")
        assert not (result)

    @pytest.mark.unittest
    def test_find_availed_data_columns(self):
        # set inputs
        root = etree.Element(
            "{http://www.broservices.nl/xsd/cptcommon/1.1}" + "parameters"
        )
        child_1 = etree.SubElement(root, "parameter1")
        child_1.text = "ja"
        child_2 = etree.SubElement(root, "parameter2")
        child_2.text = "nee"
        child_3 = etree.SubElement(root, "parameter3")
        child_3.text = "ja"
        child_4 = etree.SubElement(root, "parameter4")
        child_4.text = "nee"

        # set model
        model = bro.XMLBroCPTReader()
        # run test
        result_list = model.find_availed_data_columns(root=root)
        # check results
        assert len(result_list) == 2

    @pytest.mark.unittest
    def test__get_depth_from_bro_depth_is_set(self):
        # define the inputs
        d = {
            "depth": [1.5, 2.0, 2.5],
        }

        # set up the upper part of the dictionary
        df = pd.DataFrame(data=d)
        # initialise model
        cpt = XMLBroCPTReader()
        cpt.bro_data.dataframe = df
        # run test
        depth = cpt._XMLBroCPTReader__get_depth_from_bro()
        # check the results. Depth is just passed in this case.
        assert cpt
        assert d["depth"] == list(depth)

    @pytest.mark.unittest
    def test__get_depth_from_bro_no_depth_no_inclination(self):
        # define the inputs
        d = {
            "penetrationLength": [1.5, 2.0, 2.5],
        }
        # set up the upper part of the dictionary
        df = pd.DataFrame(data=d)
        # initialise model
        cpt = XMLBroCPTReader()
        cpt.bro_data.dataframe = df
        # run test
        depth = cpt._XMLBroCPTReader__get_depth_from_bro()
        # check the results. Depth is just passed in this case.
        assert depth.all()
        assert d["penetrationLength"] == list(depth)

    @pytest.mark.unittest
    def test__get_depth_from_bro_no_depth(self):
        # define the inputs
        d = {
            "penetrationLength": [1.5, 2.0, 2.5],
            "inclinationResultant": [1.5, 2.0, 2.5],
        }
        # calculate test results
        result = np.diff(d["penetrationLength"]) * np.cos(
            np.radians(d["inclinationResultant"][:-1])
        )
        result = np.concatenate(
            (
                d["penetrationLength"][0],
                d["penetrationLength"][0] + np.cumsum(result),
            ),
            axis=None,
        )
        # set up the upper part of the dictionary
        df = pd.DataFrame(data=d)
        # initialise model
        cpt = XMLBroCPTReader()
        cpt.bro_data.dataframe = df
        # run test
        depth = cpt._XMLBroCPTReader__get_depth_from_bro()
        # check the results. Depth is just passed in this case.
        assert depth.all()
        assert list(result) == list(depth)

    @pytest.mark.systemtest
    def test__pre_drill_with_predrill(self):

        # make a cpt with the pre_drill option
        d = {
            "penetrationLength": [1.5, 2.0, 2.5],
            "coneResistance": [1, 2, 3],
            "localFriction": [4, 5, 6],
            "frictionRatio": [0.22, 0.33, 0.44],
        }

        # set up the upper part of the dictionary
        df = pd.DataFrame(data=d)
        cpt_data = XMLBroCPTReader()
        cpt_data.bro_data.id = "cpt_name"
        cpt_data.bro_data.location_x = 111
        cpt_data.bro_data.location_y = 222
        cpt_data.bro_data.offset_z = 0.5
        cpt_data.bro_data.predrilled_z = 1.5
        cpt_data.bro_data.a = 0.8
        cpt_data.bro_data.dataframe = df

        # Run the function to be checked
        result = cpt_data._XMLBroCPTReader__parse_bro(
            minimum_length=0.01, minimum_samples=1
        )

        # Check the equality with the pre-given lists
        assert result["tip"].tolist() == [1000, 1000, 1000, 1000, 2000, 3000]
        assert result["friction"].tolist() == [4000, 4000, 4000, 4000, 5000, 6000]
        assert result["friction_nbr"].tolist() == [0.22, 0.22, 0.22, 0.22, 0.33, 0.44]
        assert result["depth"].tolist() == [0, 0.5, 1, 1.5, 2, 2.5]
        assert result["depth_to_reference"].tolist() == [
            cpt_data.bro_data.offset_z - i for i in [0, 0.5, 1, 1.5, 2, 2.5]
        ]
        assert result["water"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert result["coordinates"] == [
            cpt_data.bro_data.location_x,
            cpt_data.bro_data.location_y,
        ]
        assert result["name"] == "cpt_name"
        assert result["a"][0] == 0.8

    @pytest.mark.systemtest
    def test__pre_drill_with_pore_pressure(self):

        # Set the values of the cpt
        d = {
            "penetrationLength": [1.5, 2.0, 2.5],
            "coneResistance": [1, 2, 3],
            "localFriction": [4, 5, 6],
            "frictionRatio": [0.22, 0.33, 0.44],
            "porePressureU2": [1, 2, 3],
        }
        df = pd.DataFrame(data=d)

        # Build the upper part of the library
        cpt_data = XMLBroCPTReader()
        cpt_data.bro_data.id = "cpt_name"
        cpt_data.bro_data.location_x = 111
        cpt_data.bro_data.location_y = 222
        cpt_data.bro_data.offset_z = 0.5
        cpt_data.bro_data.predrilled_z = 1.5
        cpt_data.bro_data.a = 0.8
        cpt_data.bro_data.dataframe = df

        # define the pore pressure array before the predrilling
        # Here 3 values as the stepping is defined that way.
        # Define the stepping of the pore pressure
        # Then my target values
        # Finally multiply with 1000
        step = 1 / len(d["penetrationLength"])
        pore_pressure = [0, step * 1000, 2 * step * 1000, 1 * 1000, 2 * 1000, 3 * 1000]

        # run the function to be checked
        result = cpt_data._XMLBroCPTReader__parse_bro(
            minimum_length=0.01, minimum_samples=1
        )

        # Check the equality with the pre-defined values
        assert result["water"].tolist() == pore_pressure
        assert result["tip"].tolist() == [1000, 1000, 1000, 1000, 2000, 3000]
        assert result["friction"].tolist() == [4000, 4000, 4000, 4000, 5000, 6000]
        assert result["friction_nbr"].tolist() == [0.22, 0.22, 0.22, 0.22, 0.33, 0.44]
        assert result["depth"].tolist() == [0, 0.5, 1, 1.5, 2, 2.5]
        assert result["depth_to_reference"].tolist() == [
            cpt_data.bro_data.offset_z - i for i in [0, 0.5, 1, 1.5, 2, 2.5]
        ]
        assert result["coordinates"] == [
            cpt_data.bro_data.location_x,
            cpt_data.bro_data.location_y,
        ]
        assert result["name"] == "cpt_name"
        assert result["a"][0] == 0.8

    @pytest.mark.systemtest
    def test__pre_drill_Raise_Exception1(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {
            "penetrationLength": [1.5, 2.0],
            "coneResistance": [1, 2],
            "localFriction": [4, 5],
            "frictionRatio": [0.22, 0.33],
        }
        df = pd.DataFrame(data=d)
        # initialise model
        cpt_data = XMLBroCPTReader()
        cpt_data.bro_data.id = "cpt_name"
        cpt_data.bro_data.location_x = 111
        cpt_data.bro_data.location_y = 222
        cpt_data.bro_data.offset_z = 0.5
        cpt_data.bro_data.predrilled_z = 1.5
        cpt_data.bro_data.a = 0.8
        cpt_data.bro_data.dataframe = df

        # run the fuction
        aux = cpt_data._XMLBroCPTReader__parse_bro(minimum_length=10, minimum_samples=1)

        # check if the returned message is the appropriate
        assert "File cpt_name has a length smaller than 10" == aux

    @pytest.mark.systemtest
    def test__pre_drill_Raise_Exception2(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {
            "penetrationLength": [1.5, 2.0],
            "coneResistance": [1, 2],
            "localFriction": [4, 5],
            "frictionRatio": [0.22, 0.33],
        }
        df = pd.DataFrame(data=d)
        # initialise model
        cpt_data = XMLBroCPTReader()
        cpt_data.bro_data.id = "cpt_name"
        cpt_data.bro_data.location_x = 111
        cpt_data.bro_data.location_y = 222
        cpt_data.bro_data.offset_z = 0.5
        cpt_data.bro_data.predrilled_z = 1.5
        cpt_data.bro_data.a = 0.73
        cpt_data.bro_data.dataframe = df

        # run the fuction
        aux = cpt_data._XMLBroCPTReader__parse_bro(minimum_length=1, minimum_samples=10)

        # check if the returned message is the appropriate
        assert "File cpt_name has a number of samples smaller than 10" == aux

    @pytest.mark.systemtest
    def test_read_BRO_Raise_Exception1(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {
            "penetrationLength": [1.5, 20.0],
            "coneResistance": [-1, 2],
            "localFriction": [4, 5],
            "frictionRatio": [0.22, 0.33],
        }
        df = pd.DataFrame(data=d)
        # initialise model
        cpt_data = XMLBroCPTReader()
        cpt_data.bro_data.id = "cpt_name"
        cpt_data.bro_data.location_x = 111
        cpt_data.bro_data.location_y = 222
        cpt_data.bro_data.offset_z = 0.5
        cpt_data.bro_data.predrilled_z = 1.5
        cpt_data.bro_data.a = 0.73
        cpt_data.bro_data.dataframe = df

        # run the fuction
        aux = cpt_data._XMLBroCPTReader__parse_bro(minimum_length=10, minimum_samples=1)

        # check if the returned message is the appropriate
        assert "File cpt_name is corrupted" == aux

    @pytest.mark.systemtest
    def test_read_BRO_Raise_Exception2(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {
            "penetrationLength": [1.5, 20.0],
            "coneResistance": [1, 2],
            "localFriction": [-4, 5],
            "frictionRatio": [0.22, 0.33],
        }
        df = pd.DataFrame(data=d)
        # initialise model
        cpt_data = XMLBroCPTReader()
        cpt_data.bro_data.id = "cpt_name"
        cpt_data.bro_data.location_x = 111
        cpt_data.bro_data.location_y = 222
        cpt_data.bro_data.offset_z = 0.5
        cpt_data.bro_data.predrilled_z = 1.5
        cpt_data.bro_data.a = 0.73
        cpt_data.bro_data.dataframe = df

        # run the function
        aux = cpt_data._XMLBroCPTReader__parse_bro(minimum_length=10, minimum_samples=1)

        # check if the returned message is the appropriate
        assert "File cpt_name is corrupted" == aux

    @pytest.mark.systemtest
    def test_parse_bro_xml(self):
        # open xml file as byte object
        fn = ".\\tests\\test_files\\cpt\\bro_xml\\CPT000000065880_IMBRO_A.xml"
        with open(fn, "r") as f:
            # memory-map the file, size 0 means whole file
            xml_bytes = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)[:]
        # initialise model
        cpt_data = XMLBroCPTReader()
        # test initial expectations
        assert cpt_data
        # run test
        cpt_data.parse_bro_xml(xml=xml_bytes)
        # test that data are read
        assert cpt_data.bro_data.id == "CPT000000065880"
        assert cpt_data.bro_data.location_x == 108992.7
        assert cpt_data.bro_data.location_y == 433396.3

    @pytest.mark.systemtest
    def test_parse_bro_xml_warning(self, caplog):
        # xml will still be read but a warning will be logged
        LOGGER = logging.getLogger(__name__)
        # define logger
        LOGGER.info("Testing now.")
        # define warning expectation
        warning = "Data has the wrong size! 23 columns instead of 25"
        # open xml file as byte object
        fn = ".\\tests\\test_files\\cpt\\bro_xml\\unit_testing_files\\test_test_parse_bro_xml_raises.xml"
        with open(fn, "r") as f:
            # memory-map the file, size 0 means whole file
            xml_bytes = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)[:]
        # initialise model
        cpt_data = XMLBroCPTReader()
        # test initial expectations
        assert cpt_data
        # run test
        cpt_data.parse_bro_xml(xml=xml_bytes)
        # test that data are read
        assert cpt_data.bro_data.id == "CPT000000065880"
        assert cpt_data.bro_data.location_x == 108992.7
        assert cpt_data.bro_data.location_y == 433396.3
        # test warning
        assert warning in caplog.text

    @pytest.mark.systemtest
    def test_get_all_data_from_bro(self):
        # open xml file as byte object
        fn = ".\\tests\\test_files\\cpt\\bro_xml\\unit_testing_files\\test_get_all_data_from_bro.xml"
        with open(fn, "r") as f:
            # memory-map the file, size 0 means whole file
            xml = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)[:]
        # read test xml
        root = etree.fromstring(xml)
        # initialise model
        cpt_data = XMLBroCPTReader()
        # test initial expectations
        assert cpt_data
        # run test
        cpt_data.get_all_data_from_bro(root=root)
        # test that the data are read
        assert cpt_data.bro_data
        assert cpt_data.bro_data.a == 0.58  # <ns14:frictionSleeveSurfaceArea
        assert cpt_data.bro_data.id == "CPT000000064413"
        assert cpt_data.bro_data.cone_penetrometer_type == "F7.5CKEHG/B-1701-0745"
        assert cpt_data.bro_data.cpt_standard == "NEN5140"
        assert cpt_data.bro_data.offset_z == -1.530
        assert cpt_data.bro_data.local_reference == "maaiveld"
        assert cpt_data.bro_data.vertical_datum == "NAP"
        assert cpt_data.bro_data.quality_class == "klasse2"
        assert cpt_data.bro_data.result_time
        assert cpt_data.bro_data.predrilled_z == 0.01
