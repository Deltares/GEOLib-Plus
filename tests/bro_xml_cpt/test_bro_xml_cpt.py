import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tests.utils import TestUtils

from geolib_plus.bro_xml_cpt import bro_xml_cpt
from geolib_plus.bro_xml_cpt import BroXmlCpt
from geolib_plus.bro_xml_cpt.bro_utils import XMLBroCPTReader
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation


class TestBroXmlCpt:
    @pytest.mark.systemtest
    def test_that_preprocess_can_be_run_twice(self):
        # open the gef file
        test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "bro_xml", "CPT000000003688_IMBRO_A.xml")
        )
        assert test_file.is_file()
        # initialise models
        cpt = BroXmlCpt()
        # test initial expectations
        assert cpt
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing for the first time
        cpt.pre_process_data()
        # do pre-processing for the second time
        cpt.pre_process_data()
        # check if interpretation can be performed
        # initialise interpretation model
        robertson = RobertsonCptInterpretation
        # interpet the results
        cpt.interpret_cpt(robertson)
        assert cpt

    @pytest.mark.systemtest
    def test_read(self):
        # simple test for reading xml file from bro
        # define input path to xml
        test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "bro_xml", "CPT000000003688_IMBRO_A.xml")
        )
        # test initial expectations
        assert test_file.is_file()
        # initialise model
        cpt = bro_xml_cpt.BroXmlCpt()
        # run test
        cpt.read(test_file)
        # check expectations
        assert cpt
        assert cpt.name == "CPT000000003688"
        assert cpt.quality_class == "klasse2"
        assert cpt.cpt_type == "F7.5CKE/V-1214"
        assert cpt.local_reference_level == -1.75
        assert min(cpt.depth) == 0
        assert max(cpt.depth) == 24.56

    @pytest.mark.systemtest
    def test_read_drop_nans(self):
        # simple test for reading xml file from bro
        # define input path to xml
        test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "bro_xml", "CPT000000003688_IMBRO_A.xml")
        )
        # test initial expectations
        assert test_file.is_file()
        # initialise model
        cpt = bro_xml_cpt.BroXmlCpt()
        # run test
        cpt.read(test_file)
        cpt.drop_nan_values()
        # check expectations
        assert cpt
        assert cpt.name == "CPT000000003688"
        assert len(cpt.depth) == len(cpt.friction_nbr)
        assert len(cpt.friction_nbr) == 1215

    @pytest.mark.systemtest
    def test_drop_duplicate_depth_values(self):
        # simple test for reading xml file from bro
        # define input path to xml
        test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "bro_xml", "CPT000000003688_IMBRO_A.xml")
        )
        # test initial expectations
        assert test_file.is_file()
        # initialise model
        cpt_read = bro_xml_cpt.BroXmlCpt()
        # run test
        cpt_read.read(test_file)
        # set duplicate values
        cpt_read.penetration_length[1] = 0
        cpt_read.penetration_length[2] = 0
        cpt_read.penetration_length[3] = 0
        # save expectation
        previous_length = len(cpt_read.penetration_length)
        # run test
        cpt_read.drop_duplicate_depth_values()
        # check expectations
        assert len(cpt_read.penetration_length) == previous_length - 3
        assert len(cpt_read.friction_nbr) == len(cpt_read.penetration_length)

    @pytest.mark.systemtest
    def test__pre_drill_with_predrill(self):
        # initialize model
        cpt_data = BroXmlCpt()
        # define inputs
        cpt_data.name = "cpt_name"
        cpt_data.coordinates = [111, 222]
        cpt_data.local_reference_level = 0.5
        cpt_data.predrilled_z = 1.5
        cpt_data.a = 0.8
        cpt_data.depth = [1.5, 2.0, 2.5]
        cpt_data.tip = [1, 2, 3]
        cpt_data.friction = [4, 5, 6]
        cpt_data.friction_nbr = [0.22, 0.33, 0.44]

        # Run the function to be checked
        cpt_data.perform_pre_drill_interpretation(length_of_average_points=1)

        # Check the equality with the pre-given lists
        assert cpt_data.tip.tolist() == [1, 1, 1, 1, 2, 3]
        assert cpt_data.friction.tolist() == [4, 4, 4, 4, 5, 6]
        assert cpt_data.friction_nbr.tolist() == [0.22, 0.22, 0.22, 0.22, 0.33, 0.44]
        assert cpt_data.depth.tolist() == [0, 0.5, 1, 1.5, 2, 2.5]
        assert cpt_data.coordinates == [111, 222]
        assert cpt_data.name == "cpt_name"
        assert cpt_data.a == 0.8

    @pytest.mark.systemtest
    def test__pre_drill_with_pore_pressure(self):
        # initialize model
        cpt_data = BroXmlCpt()
        # define inputs
        cpt_data.name = "cpt_name"
        cpt_data.coordinates = [111, 222]
        cpt_data.local_reference_level = 0.5
        cpt_data.predrilled_z = 1.5
        cpt_data.a = 0.8
        cpt_data.depth = [1.5, 2.0, 2.5]
        cpt_data.tip = [1, 2, 3]
        cpt_data.friction = [4, 5, 6]
        cpt_data.friction_nbr = [0.22, 0.33, 0.44]
        cpt_data.pore_pressure_u1 = [1500, 2000, 2500]

        # run the function to be checked
        cpt_data.perform_pre_drill_interpretation(length_of_average_points=1)

        # Check the equality with the pre-defined values
        assert cpt_data.pore_pressure_u1.tolist() == [
            0.0,
            500.0,
            1000.0,
            1500.0,
            2000.0,
            2500.0,
        ]
        assert cpt_data.tip.tolist() == [1, 1, 1, 1, 2, 3]
        assert cpt_data.friction.tolist() == [4, 4, 4, 4, 5, 6]
        assert cpt_data.friction_nbr.tolist() == [0.22, 0.22, 0.22, 0.22, 0.33, 0.44]
        assert cpt_data.depth.tolist() == [0, 0.5, 1, 1.5, 2, 2.5]
        assert cpt_data.coordinates == [111, 222]
        assert cpt_data.name == "cpt_name"
        assert cpt_data.a == 0.8

    @pytest.mark.systemtest
    def test__pre_drill_is_zero(self):
        # initialize model
        cpt_data = BroXmlCpt()
        # define inputs
        cpt_data.name = "cpt_name"
        cpt_data.coordinates = [111, 222]
        cpt_data.local_reference_level = 0.5
        cpt_data.predrilled_z = 0
        cpt_data.a = 0.8
        cpt_data.depth = [1.5, 2.0, 2.5]
        cpt_data.tip = [1, 2, 3]
        cpt_data.friction = [4, 5, 6]
        cpt_data.friction_nbr = [0.22, 0.33, 0.44]
        cpt_data.pore_pressure_u1 = [1500, 2000, 2500]

        # run the function to be checked
        cpt_data.perform_pre_drill_interpretation(length_of_average_points=1)

        # nothing should be changed
        assert cpt_data.name == "cpt_name"
        assert cpt_data.coordinates == [111, 222]
        assert cpt_data.local_reference_level == 0.5
        assert cpt_data.predrilled_z == 0
        assert cpt_data.a == 0.8
        assert cpt_data.depth.tolist() == [0.0, 1.5, 2.0, 2.5]
        assert cpt_data.tip.tolist() == [1, 1, 2, 3]
        assert cpt_data.friction.tolist() == [4, 4, 5, 6]
        assert cpt_data.friction_nbr.tolist() == [0.22, 0.22, 0.33, 0.44]
        assert cpt_data.pore_pressure_u1.tolist() == [1500, 1500, 2000, 2500]
