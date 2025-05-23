import json
import logging
import mmap
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lxml import etree

from geolib_plus.bro_xml_cpt import bro_utils as bro
from geolib_plus.bro_xml_cpt.bro_utils import XMLBroCPTReader
from tests.utils import TestUtils


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
        test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "bro_xml", "CPT000000003688_IMBRO_A.xml")
        )
        # test initial expectations
        assert test_file.is_file()
        # run test
        model = bro.XMLBroCPTReader.xml_to_byte_string(fn=test_file)
        # test results
        assert model

    @pytest.mark.unittest
    def test_xml_to_byte_string_wrong_path(self):
        # define input path to xml
        test_file = TestUtils.get_local_test_data_dir(Path("cpt", "bro_xml", "wrong.xml"))
        # final test
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

    @pytest.mark.systemtest
    def test_parse_bro_xml(self):
        # open xml file as byte object
        fn = TestUtils.get_local_test_data_dir(
            Path("cpt", "bro_xml", "CPT000000065880_IMBRO_A.xml")
        )
        # test initial expectations
        assert fn.is_file()
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
        fn = TestUtils.get_local_test_data_dir(
            Path(
                "cpt",
                "bro_xml",
                "unit_testing_files",
                "test_test_parse_bro_xml_raises.xml",
            )
        )
        # test initial expectations
        assert fn.is_file()
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
    def test_that_xml_is_passed(self):
        xml_files = [
            "CPT000000129426.xml",
            "CPT000000129429.xml",
            "CPT000000179090.xml",
            "CPT000000179092.xml",
            "CPT000000179095.xml",
            "CPT000000179099.xml",
            "CPT000000179101.xml",
            "CPT000000179103.xml",
            "CPT000000179106.xml",
            "CPT000000179107.xml",
            "CPT000000179108.xml",
            "CPT000000179109.xml",
            "CPT000000179114.xml",
            "CPT000000179122.xml",
            "CPT000000179124.xml",
        ]
        # open xml file as byte object
        for file in xml_files:
            fn = TestUtils.get_local_test_data_dir(
                Path("cpt", "bro_xml", "xmls_with_various_formats", file)
            )
            # test initial expectations
            assert fn.is_file()
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
            assert cpt_data.bro_data.id == file.split(".")[0]

    @pytest.mark.systemtest
    def test_get_all_data_from_bro(self):
        # open xml file as byte object
        fn = TestUtils.get_local_test_data_dir(
            Path(
                "cpt",
                "bro_xml",
                "unit_testing_files",
                "test_get_all_data_from_bro.xml",
            )
        )
        # test initial expectations
        assert fn.is_file()
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
