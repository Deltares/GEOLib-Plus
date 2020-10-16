import pytest
from tests.utils import TestUtils
from pathlib import Path
import json
import os
from lxml import etree

from geolib_plus.bro_xml_cpt import bro_utils as bro


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
        test_columns = cl_cpt.columns_string_list
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
