from geolib_plus import __version__
from geolib_plus.bro_xml_cpt.bro_xml_cpt import *
from geolib_plus.gef_cpt.gef_cpt import *
from geolib_plus.gef_cpt.validate_gef import validate_gef_cpt
from geolib_plus.bro_xml_cpt.validate_bro import validate_bro_cpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from tests.utils import TestUtils

# External
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.systemtest
def test_version():
    assert __version__ == "0.1.0"


class TestGeolibPlusReading:
    # System Test for geolib_plus_read_BRO

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

        cpt = GefCpt().read(test_file)

        test_coord = [244319.00, 587520.00]
        test_depth = np.linspace(1, 20, 20)
        test_nap = 0.13 - test_depth
        test_tip = np.full(20, 1000)
        test_friction = np.full(20, 2000)
        test_friction_nbr = np.full(20, 5)
        test_water = np.full(20, 3000)

        np.testing.assert_array_equal("DKP302", cpt.name)
        np.testing.assert_array_equal(test_coord, cpt.coordinates)
        np.testing.assert_array_equal(test_depth, cpt.depth)
        np.testing.assert_array_equal(test_nap, cpt.depth_to_reference)
        np.testing.assert_array_equal(test_tip, cpt.tip)
        np.testing.assert_array_equal(test_friction, cpt.friction)
        np.testing.assert_array_equal(test_friction_nbr, cpt.friction_nbr)
        np.testing.assert_array_equal(test_water, cpt.water)

    # System Test for geolib_plus_read_GEF & BRO based comparing result for same file
    testdata = [
        ("CPT000000063044_IMBRO_A"),
        ("CPT000000063045_IMBRO_A"),
        ("CPT000000064413_IMBRO_A"),
        ("CPT000000065880_IMBRO_A"),
        ("CPT000000003688_IMBRO_A"),
    ]

    @pytest.mark.systemtest
    @pytest.mark.parametrize("name", testdata, ids=testdata)
    def test_reading_compare(self, name):
        # Compare two files from bro (same CPT) in GEF and BRO Format
        # Should be comparable

        test_dir = TestUtils.get_local_test_data_dir("cpt")
        bro_file = test_dir / "bro_xml" / f"{name}.xml"
        assert bro_file.is_file()
        gef_file = test_dir / "gef" / f"{name}.gef"
        assert gef_file.is_file()

        bro_cpt = BroXmlCpt()
        bro_cpt.read(bro_file)

        gef_cpt = GefCpt()
        gef_cpt.read(gef_file)

        np.testing.assert_array_equal(bro_cpt.name, gef_cpt.name)
        np.testing.assert_array_equal(bro_cpt.coordinates, gef_cpt.coordinates)

        # todo: JN The following tests current fail, the arrays are different size as are the depths
        np.testing.assert_array_equal(bro_cpt.depth, gef_cpt.depth)
        np.testing.assert_array_equal(
            bro_cpt.depth_to_reference, gef_cpt.depth_to_reference
        )
        np.testing.assert_array_equal(bro_cpt.tip, gef_cpt.tip)
        np.testing.assert_array_equal(bro_cpt.friction, gef_cpt.friction)
        np.testing.assert_array_equal(bro_cpt.friction_nbr, gef_cpt.friction_nbr)
        np.testing.assert_array_equal(bro_cpt.water, gef_cpt.water)


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
    # Test validation of gef file structure ..... with file with error
    def test_validate_gef_error(self):
        # This file raises a warning
        gef_file = TestUtils.get_local_test_data_dir(
            "cpt/gef/CPT000000003688_IMBRO_A_err.gef"
        )
        with pytest.raises(Exception):
            validate_gef_cpt(gef_file)


    def test_robertson_interpretation_test(self):
        gef_file = TestUtils.get_local_test_data_dir(
            "cpt/gef/KW19-3.gef"
        )
        cpt = GefCpt()
        cpt.read(gef_file)
        cpt.pre_process_data()
        cpt.interpret_cpt(RobertsonCptInterpretation)
        print(cpt.lithology)
