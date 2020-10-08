from geolib_plus.BRO_XML_CPT import BRO_XML_CPT
from geolib_plus.GEF_CPT import GEF_CPT
import numpy as np
from pathlib import Path

def test_reading_bro():
    bro_file = Path("../tests/test_files/cpt/bro_xml/CPT000000003688_IMBRO_A.xml")
    cpt = BRO_XML_CPT()
    cpt.read(bro_file)

    test_coord = [91931.000, 438294.000]
    test_depth = np.arange(0.0, 24.341, 0.02)
    test_depth = test_depth[(test_depth < 3.82) | (test_depth > 3.86)]

    test_NAP = -1.75 - test_depth

    test_tip_first = [845.,  353., 2190., 4276., 5663., 6350., 7498., 8354., 9148., 9055.]
    test_tip_last = [32677., 32305., 31765., 31038., 30235., 29516., 28695., 27867., 26742., 25233.]

    test_friction_first = np.array([19., 25., 31., 38., 40., 36., 62., 63., 75., 91.])
    test_friction_last = np.array([266., 273., 279., 276., 273., 264., 253., 241., 230., 219.])

    test_friction_nbr_first = np.array([0.7, 1.,  1.,  1.,  1.,  0.6, 0.9, 0.9, 1.,  1.1])
    test_friction_nbr_last= np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

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






