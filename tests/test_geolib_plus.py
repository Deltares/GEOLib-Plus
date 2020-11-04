from geolib_plus import __version__
from geolib_plus.BRO_XML_CPT.bro_xml_cpt import *
from geolib_plus.GEF_CPT.gef_cpt import *
from geolib_plus.GEF_CPT.validate_gef import validate_gef_cpt
from tests.utils import TestUtils

# External
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.systemtest
def test_version():
    assert __version__ == "0.1.0"


# System Test for geolib_plus_read_BRO
@pytest.mark.systemtest
def test_reading_bro():
    bro_file = TestUtils.get_local_test_data_dir(
        "cpt/bro_xml/CPT000000003688_IMBRO_A.xml"
    )
    assert bro_file.is_file()
    cpt = BroXmlCpt()
    cpt.read(bro_file)

    test_coord = [91931.000, 438294.000]
    test_depth = np.arange(0.0, 24.341, 0.02)
    test_depth = test_depth[(test_depth < 3.82) | (test_depth > 3.86)]

    test_NAP = -1.75 - test_depth

    test_tip_first = [
        845.0,
        353.0,
        2190.0,
        4276.0,
        5663.0,
        6350.0,
        7498.0,
        8354.0,
        9148.0,
        9055.0,
    ]
    test_tip_last = [
        32677.0,
        32305.0,
        31765.0,
        31038.0,
        30235.0,
        29516.0,
        28695.0,
        27867.0,
        26742.0,
        25233.0,
    ]

    test_friction_first = np.array(
        [19.0, 25.0, 31.0, 38.0, 40.0, 36.0, 62.0, 63.0, 75.0, 91.0]
    )
    test_friction_last = np.array(
        [266.0, 273.0, 279.0, 276.0, 273.0, 264.0, 253.0, 241.0, 230.0, 219.0]
    )

    test_friction_nbr_first = np.array(
        [0.7, 1.0, 1.0, 1.0, 1.0, 0.6, 0.9, 0.9, 1.0, 1.1]
    )
    test_friction_nbr_last = np.array(
        [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    )

    test_water_first = np.full(10, 0)
    test_water_last = np.full(10, 0)

    np.testing.assert_array_equal("CPT000000003688", cpt.name)
    np.testing.assert_array_equal(test_coord, cpt.coordinates)
    np.testing.assert_array_almost_equal(test_depth, cpt.depth)
    np.testing.assert_array_almost_equal(test_NAP, cpt.depth_to_reference)
    np.testing.assert_array_equal(test_tip_first, cpt.tip[0:10])
    np.testing.assert_array_equal(test_tip_last, cpt.tip[-10:])
    np.testing.assert_array_equal(test_friction_first, cpt.friction[0:10])
    np.testing.assert_array_equal(test_friction_last, cpt.friction[-10:])
    np.testing.assert_array_equal(test_friction_nbr_first, cpt.friction_nbr[0:10])
    np.testing.assert_array_equal(test_friction_nbr_last, cpt.friction_nbr[-10:])
    np.testing.assert_array_equal(test_water_first, cpt.water[0:10])
    np.testing.assert_array_equal(test_water_last, cpt.water[-10:])


# System Test for geolib_plus_read_GEF
@pytest.mark.systemtest
def test_reading_gef():
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

@pytest.mark.integrationtest
def test_pre_process_gef_data():
    """
    Tests pre process of gef data
    """
    test_file = TestUtils.get_local_test_data_dir(
        "cpt/gef/unit_testing/unit_testing.gef"
    )
    assert test_file.is_file()

    cpt = GefCpt().read(test_file)

    # check cpt before preprocess
    assert cpt.depth.ndim == 0
    assert cpt.depth_to_reference is None
    assert cpt.water is None

    expected_depth = cpt.penetration_length[0] + \
                     np.cumsum(np.diff(cpt.penetration_length) * np.cos(np.radians(cpt.inclination_resultant[:-1])))

    # process data
    cpt.pre_process_data()
    np.testing.assert_array_almost_equal(cpt.depth_to_reference, cpt.local_reference_level - cpt.depth)
    np.testing.assert_array_almost_equal(cpt.depth[1:], expected_depth)
    np.testing.assert_array_almost_equal(cpt.water, cpt.pore_pressure_u2)

@pytest.mark.systemtest
@pytest.mark.parametrize("name", testdata, ids=testdata)
def test_reading_compare(name):
    # Compare two files from bro (same CPT) in GEF and BRO Format
    # Should be comparable

    test_dir = TestUtils.get_local_test_data_dir("cpt")
    bro_file = test_dir / "bro_xml" / f"{name}.xml"
    assert bro_file.is_file()
    gef_file = test_dir / "gef" / f"{name}.gef"
    assert gef_file.is_file()

    gef_cpt = GefCpt().read(gef_file)

    bro_cpt = BroXmlCpt()
    bro_cpt.read(bro_file)

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


@pytest.mark.systemtest
# Test validation of BRO-XML file structure .... with clean file
def test_validate_bro_no_error():
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
def test_validate_bro_error():
    bro_xml_file_err_path = TestUtils.get_local_test_data_dir(
        "/cpt/bro_xml/CPT000000003688_IMBRO_A_err.xml"
    )
    with pytest.raises(Exception):
        validate_bro_cpt(bro_xml_file_err_path)


@pytest.mark.systemtest
# Test validation of gef file structure .... with usable file
def test_validate_gef_no_error():
    # This file raises a warning - it is in another process so can't capture it
    gef_file = TestUtils.get_local_test_data_dir("cpt/gef/CPT000000003688_IMBRO_A.gef")
    try:
        validate_gef_cpt(gef_file)
    except:
        pytest.fail("GEF file without error raised Error")


@pytest.mark.systemtest
# Test validation of gef file structure ..... with file with error
def test_validate_gef_error():
    # This file raises a warning
    gef_file = TestUtils.get_local_test_data_dir(
        "cpt/gef/CPT000000003688_IMBRO_A_err.gef"
    )
    with pytest.raises(Exception):
        validate_gef_cpt(gef_file)
