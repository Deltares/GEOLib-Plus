# External
from pathlib import Path

import numpy as np
import pytest

from geolib_plus import __version__
from geolib_plus.bro_xml_cpt import *
from geolib_plus.gef_cpt import *
from geolib_plus.gef_cpt.validate_gef import validate_gef_cpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from tests.utils import TestUtils

version = "0.2.1"


@pytest.mark.systemtest
def test_version():
    assert __version__ == version


class TestGeolibPlusReading:
    @pytest.mark.systemtest
    def test_that_values_gef_and_cpt_are_of_the_same_type(self):
        # Read cpts
        # open the gef file
        test_file_gef = TestUtils.get_local_test_data_dir(
            Path("cpt", "gef", "CPT000000003688_IMBRO_A.gef")
        )
        assert test_file_gef.is_file()
        # open the bro file
        test_file_bro = TestUtils.get_local_test_data_dir(
            Path("cpt", "bro_xml", "CPT000000003688_IMBRO_A.xml")
        )
        assert test_file_bro.is_file()
        # initialise models
        cpt_gef = GefCpt()
        cpt_bro_xml = BroXmlCpt()
        # test initial expectations
        assert cpt_gef
        assert cpt_bro_xml
        # read gef file
        cpt_gef.read(filepath=test_file_gef)
        # read bro file
        cpt_bro_xml.read(filepath=test_file_bro)

        cpt_bro_xml = dict(cpt_bro_xml)
        cpt_gef = dict(cpt_gef)
        for key, value in cpt_bro_xml.items():
            if key not in ["predrilled_z", "undefined_depth", "water_measurement_type"]:
                assert type(cpt_bro_xml.get(key, None)) == type(cpt_gef.get(key, None))

    @pytest.mark.systemtest
    def test_has_points_with_error(self):
        # initialise models
        cpt_gef = GefCpt()
        cpt_bro_xml = BroXmlCpt()
        # fill in data
        cpt_gef.tip = np.array([1, 2, -9999.99, 4, 5])
        cpt_gef.friction = np.array([1, 2, 3, 4, 5])
        cpt_gef.water = np.array([1, 2, 3, 4, 5])
        cpt_gef.friction_nbr = np.array([1, 2, 3, 4, 5])
        cpt_gef.pwp = None
        cpt_gef.depth = np.array([1, 2, 3, 4, 5])
        cpt_gef.depth_to_reference = np.array([1, 2, 3, 4, 5])
        # set up error code for tip
        cpt_gef.error_codes = {"tip": -9999.99}

        cpt_bro_xml.tip = np.array([1, 2, np.nan, 4, 5])
        cpt_bro_xml.friction = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.water = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.friction_nbr = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.pwp = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.depth = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.depth_to_reference = None

        # run tests
        with pytest.raises(ValueError) as excinfo:
            cpt_bro_xml.has_points_with_error()
            assert "tip" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            cpt_gef.has_points_with_error()
            assert "tip" in str(excinfo.value)

    @pytest.mark.systemtest
    def test_are_data_available_interpretation_no_values_missing(self):
        # initialise models
        cpt_gef = GefCpt()
        cpt_bro_xml = BroXmlCpt()
        # fill in data
        cpt_gef.tip = np.array([1, 2, 3, 4, 5])
        cpt_gef.friction = np.array([1, 2, 3, 4, 5])
        cpt_gef.water = np.array([1, 2, 3, 4, 5])
        cpt_gef.friction_nbr = np.array([1, 2, 3, 4, 5])
        cpt_gef.pwp = np.array([1, 2, 3, 4, 5])
        cpt_gef.depth = np.array([1, 2, 3, 4, 5])
        cpt_gef.depth_to_reference = np.array([1, 2, 3, 4, 5])

        cpt_bro_xml.tip = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.friction = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.water = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.friction_nbr = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.pwp = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.depth = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.depth_to_reference = np.array([1, 2, 3, 4, 5])

        # run tests
        cpt_gef.are_data_available_interpretation()
        cpt_bro_xml.are_data_available_interpretation()
        # test
        assert cpt_gef
        assert cpt_bro_xml

    @pytest.mark.systemtest
    def test_are_data_available_interpretation_values_missing(self):
        # initialise models
        cpt_gef = GefCpt()
        cpt_bro_xml = BroXmlCpt()
        # fill in data
        cpt_gef.tip = np.array([1, 2, 3, 4, 5])
        cpt_gef.friction = None
        cpt_gef.water = np.array([1, 2, 3, 4, 5])
        cpt_gef.friction_nbr = np.array([1, 2, 3, 4, 5])
        cpt_gef.depth = np.array([1, 2, 3, 4, 5])
        cpt_gef.depth_to_reference = np.array([1, 2, 3, 4, 5])

        cpt_bro_xml.tip = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.friction = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.water = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.friction_nbr = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.pwp = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.depth = np.array([1, 2, 3, 4, 5])
        cpt_bro_xml.depth_to_reference = None

        # run tests
        with pytest.raises(ValueError) as excinfo:
            cpt_gef.are_data_available_interpretation()
            assert "friction" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            cpt_bro_xml.are_data_available_interpretation()
            assert "depth_to_reference" in str(excinfo.value)

    @pytest.mark.systemtest
    def test_has_duplicated_depth_values_duplicate(self):
        # initialise models
        cpt_gef = GefCpt()
        cpt_bro_xml = BroXmlCpt()
        # fill in data
        cpt_gef.penetration_length = np.array([1, 2, 3, 4, 5, 3])

        cpt_bro_xml.penetration_length = np.array([1, 2, 3.1, 3.1, 4, 5])

        # run tests
        with pytest.raises(ValueError) as excinfo:
            cpt_gef.has_duplicated_depth_values()
            assert "Value depth contains duplicates" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            cpt_bro_xml.has_duplicated_depth_values()
            assert "Value depth contains duplicates" in str(excinfo.value)

    @pytest.mark.systemtest
    def test_has_duplicated_depth_values_no_duplicate(self):
        # initialise models
        cpt_gef = GefCpt()
        cpt_bro_xml = BroXmlCpt()
        # fill in data
        cpt_gef.penetration_length = np.array([1, 2, 3, 4, 5])

        cpt_bro_xml.penetration_length = np.array([1, 2, 3.1, 4, 5])

        # run tests
        cpt_gef.has_duplicated_depth_values()
        cpt_bro_xml.has_duplicated_depth_values()
        # test
        assert cpt_gef
        assert cpt_bro_xml

    @pytest.mark.systemtest
    def test_check_if_lists_have_the_same_size(self):
        # initialise models
        cpt_gef = GefCpt()
        cpt_bro_xml = BroXmlCpt()
        # fill in data
        cpt_gef.tip = np.array([1, 2, 3, 4, 5])
        cpt_gef.friction = np.array([1, 2, 3, 4, 5, 6])

        cpt_bro_xml.depth = np.array([1, 2, 3, 5, 6, 7])
        cpt_bro_xml.depth_to_reference = np.array([1, 2, 3, 5])
        cpt_bro_xml.tip = np.array([1, 2, 3, 4, 5])

        # run tests
        with pytest.raises(ValueError) as excinfo:
            cpt_gef.check_if_lists_have_the_same_size()
            assert "friction does not have the same size as the other properties" in str(
                excinfo.value
            )
        with pytest.raises(ValueError) as excinfo:
            cpt_bro_xml.check_if_lists_have_the_same_size()
            assert "depth does not have the same size as the other properties" in str(
                excinfo.value
            )

    @pytest.mark.systemtest
    def test_robertson_interpretation_test(self):
        gef_file = TestUtils.get_local_test_data_dir("cpt/gef/KW19-3.gef")
        cpt = GefCpt()
        cpt.read(gef_file)
        cpt.pre_process_data()
        cpt.interpret_cpt(RobertsonCptInterpretation())
        print(cpt.lithology)

    @pytest.mark.systemtest
    def test_reading_bro(self):
        bro_file = TestUtils.get_local_test_data_dir(
            "cpt/bro_xml/CPT000000003688_IMBRO_A.xml"
        )
        assert bro_file.is_file()
        cpt = BroXmlCpt()
        cpt.read(bro_file)

        test_coord = [91931.000, 438294.000]
        test_depth = np.arange(0.0, 24.58, 0.02)

        test_NAP = -1.75 - test_depth

        test_tip_first = [
            0.196,
            1.824,
            3.446,
            5.727,
            4.226,
            2.535,
            0.845,
            0.353,
            2.19,
            4.276,
        ]
        test_tip_last = [
            29.516,
            28.695,
            27.867,
            26.742,
            25.233,
            23.683,
            22.438,
            21.518,
            20.873,
            20.404,
        ]

        test_friction_first = np.array([0.019, 0.025, 0.031, 0.038, 0.04, 0.036, 0.062])
        test_friction_last = np.array([0.264, 0.253, 0.241, 0.23, 0.219])

        test_friction_nbr_first = np.array([1.0, 1.0, 1.0])
        test_friction_nbr_last = np.array([0.8, 0.8, 0.8, 0.8, 0.8])

        np.testing.assert_array_equal("CPT000000003688", cpt.name)
        np.testing.assert_array_equal(test_coord, cpt.coordinates)
        np.testing.assert_array_almost_equal(test_depth, cpt.depth)
        np.testing.assert_array_equal(test_tip_first, cpt.tip[0:10])
        np.testing.assert_array_equal(test_tip_last, cpt.tip[-10:])
        np.testing.assert_array_equal(test_friction_first, cpt.friction[6:13])
        np.testing.assert_array_equal(
            test_friction_last, cpt.friction[-10 : len(cpt.friction) - 5]
        )
        np.testing.assert_array_equal(test_friction_nbr_first, cpt.friction_nbr[7:10])
        np.testing.assert_array_equal(
            test_friction_nbr_last, cpt.friction_nbr[-10 : len(cpt.friction) - 5]
        )

    # System Test for geolib_plus_read_GEF
    @pytest.mark.systemtest
    def test_reading_gef(self):
        test_file = TestUtils.get_local_test_data_dir(
            "cpt/gef/unit_testing/unit_testing.gef"
        )
        assert test_file.is_file()

        cpt = GefCpt()
        cpt.read(test_file)

        test_coord = [244319.00, 587520.00]
        test_penetration = np.linspace(1, 20, 20)
        test_tip = np.full(20, 1.0)
        test_friction = np.full(20, 2.0)
        test_friction_nbr = np.full(20, 5.0)
        test_inclination = np.full(20, 4.0)
        test_pore_pressure_u1 = None
        test_pore_pressure_u2 = np.full(20, 3.0)
        test_pore_pressure_u3 = None

        np.testing.assert_array_equal("DKP302", cpt.name)
        np.testing.assert_array_equal(test_coord, cpt.coordinates)
        np.testing.assert_array_equal(test_penetration, cpt.penetration_length)
        np.testing.assert_array_equal(test_tip, cpt.tip)
        np.testing.assert_array_equal(test_friction, cpt.friction)
        np.testing.assert_array_equal(test_friction_nbr, cpt.friction_nbr)
        np.testing.assert_array_equal(test_inclination, cpt.inclination_resultant)
        np.testing.assert_almost_equal(0.13, cpt.local_reference_level)
        assert test_pore_pressure_u1 == cpt.pore_pressure_u1
        np.testing.assert_array_equal(test_pore_pressure_u2, cpt.pore_pressure_u2)
        assert test_pore_pressure_u3 == cpt.pore_pressure_u3

    # System Test for geolib_plus_read_GEF & BRO based comparing result for same file
    testdata = [
        ("CPT000000063044_IMBRO_A"),
        ("CPT000000063045_IMBRO_A"),
        ("CPT000000064413_IMBRO_A"),
        ("CPT000000065880_IMBRO_A"),
        ("CPT000000003688_IMBRO_A"),
    ]

    @pytest.mark.integrationtest
    def test_pre_process_gef_data(self):
        """
        Tests pre process of gef data
        """
        test_file = TestUtils.get_local_test_data_dir(
            "cpt/gef/unit_testing/unit_testing.gef"
        )
        assert test_file.is_file()

        cpt = GefCpt()
        cpt.read(test_file)

        # check cpt before preprocess
        assert cpt.depth is None
        assert cpt.depth_to_reference is None
        assert cpt.water is None

        expected_depth = cpt.penetration_length[0] + np.cumsum(
            np.diff(cpt.penetration_length)
            * np.cos(np.radians(cpt.inclination_resultant[:-1]))
        )

        # process data
        cpt.pre_process_data()
        np.testing.assert_array_almost_equal(
            cpt.depth_to_reference, cpt.local_reference_level - cpt.depth
        )
        np.testing.assert_array_almost_equal(cpt.depth[2:], expected_depth)
        np.testing.assert_array_almost_equal(cpt.water, cpt.pore_pressure_u2)

    def test_pre_process_bro_data(self):
        """
        Tests pre process of gef data
        """
        test_file = TestUtils.get_local_test_data_dir(
            "cpt/bro_xml/CPT000000064413_IMBRO_A.xml"
        )
        assert test_file.is_file()

        cpt = BroXmlCpt()
        cpt.read(test_file)
        cpt.pore_pressure_u2 = np.array(cpt.depth * 10)

        # check cpt before preprocess
        assert cpt.depth.ndim == 1
        assert cpt.depth_to_reference is None
        assert cpt.water is None

        expected_depth = cpt.penetration_length[0] + np.cumsum(
            np.diff(cpt.penetration_length)
            * np.cos(np.radians(cpt.inclination_resultant[:-1]))
        )

        # process data
        cpt.pre_process_data()
        np.testing.assert_array_almost_equal(
            cpt.depth_to_reference, cpt.local_reference_level - cpt.depth
        )
        np.testing.assert_array_almost_equal(cpt.depth[1:100], expected_depth[0:99])
        np.testing.assert_array_almost_equal(cpt.water, cpt.pore_pressure_u2)

    def test_pre_process_bro_data_without_friction_nbr(self):
        """
        Tests pre process of bro data in case no friction number is present
        """
        test_file = TestUtils.get_local_test_data_dir(
            "cpt/bro_xml/CPT000000064413_IMBRO_A.xml"
        )
        assert test_file.is_file()

        cpt = BroXmlCpt()
        cpt.read(test_file)

        # manually remove friction number
        cpt.friction_nbr = None

        # process data
        cpt.pre_process_data()

        expected_friction_nbr = cpt.friction / cpt.tip * 100

        np.testing.assert_array_almost_equal(expected_friction_nbr, cpt.friction_nbr)

    @pytest.mark.integrationtest
    def test_calculate_friction_nbr_if_not_available(self):
        """
        Test calculate friction number if not available
        """
        test_file = TestUtils.get_local_test_data_dir(
            "cpt/gef/unit_testing/Exception_NoFrictionNumber.gef"
        )
        assert test_file.is_file()

        # read data
        cpt = GefCpt()
        cpt.read(test_file)

        # calculate friction number
        cpt.calculate_friction_nbr_if_not_available()

        # set expected friction number
        expected_friction_number = np.ones(20) * 200

        # assert friction number
        np.testing.assert_array_almost_equal(expected_friction_number, cpt.friction_nbr)

    @pytest.mark.integrationtest
    def test_calculate_friction_nbr_if_not_available_friction_is_available(self):
        """
        Test calculate friction number if not available. In this test, the friction number is available,
        so nothing should change.
        """
        test_file = TestUtils.get_local_test_data_dir(
            "cpt/gef/unit_testing/unit_testing.gef"
        )
        assert test_file.is_file()

        # read data
        cpt = GefCpt()
        cpt.read(test_file)

        # calculate friction number
        cpt.calculate_friction_nbr_if_not_available()

        # set expected friction number
        expected_friction_number = np.ones(20) * 5

        # assert friction number
        np.testing.assert_array_almost_equal(expected_friction_number, cpt.friction_nbr)


class TestGeolibPlusValidate:
    @pytest.mark.systemtest
    # Test validation of BRO-XML file structure .... with clean file
    def test_validate_bro_no_error(self):
        bro_xml_file_path = TestUtils.get_local_test_data_dir(
            "cpt/bro_xml/CPT000000003688_IMBRO_A.xml"
        )
        assert bro_xml_file_path.is_file()
        try:
            validate_bro_cpt(bro_xml_file_path)
        except:  # catch *all* exceptions
            pytest.fail("Validation Error: CPT BRO_XML without error raises error")

    @pytest.mark.systemtest
    # Test validation of BRO-XML file structure ..... with file with error
    def test_validate_bro_error(self):
        bro_xml_file_err_path = TestUtils.get_local_test_data_dir(
            "/cpt/bro_xml/CPT000000003688_IMBRO_A_err.xml"
        )
        with pytest.raises(Exception):
            validate_bro_cpt(bro_xml_file_err_path)

    @pytest.mark.systemtest
    # Test validation of gef file structure .... with usable file
    def test_validate_gef_no_error(self):
        # This file raises a warning - it is in another process so can't capture it
        gef_file = TestUtils.get_local_test_data_dir(
            "cpt/gef/CPT000000003688_IMBRO_A.gef"
        )
        try:
            validate_gef_cpt(gef_file)
        except:
            pytest.fail("GEF file without error raised Error")

    @pytest.mark.systemtest
    # Test validation of gef file structure ..... with file with error (need to add more errors)
    def test_validate_gef_error(self):
        # This file raises a warning
        gef_file = TestUtils.get_local_test_data_dir(
            "cpt/gef/CPT000000003688_IMBRO_A_err.gef"
        )
        with pytest.raises(Exception):
            validate_gef_cpt(gef_file)
