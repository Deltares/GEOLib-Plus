from geolib_plus.GEF_CPT import GEF_CPT
import numpy as np
from pathlib import Path

def test_reading_gef():
    file = Path("../tests/test_files/cpt/gef/unit_testing/unit_testing.gef")
    gef_id = "unit_testing.gef"

    cpt = GEF_CPT()
    cpt.read(file, gef_id)

    test_coord = [244319.00, 587520.00]
    test_depth = np.linspace(1, 20, 20)
    test_NAP = 0.13 - test_depth
    test_tip = np.full(20, 1000)
    test_friction = np.full(20, 2000)
    test_friction_nbr = np.full(20, 5)
    test_water = np.full(20, 3000)

    np.testing.assert_array_equal("unit_testing.gef", cpt.name)
    np.testing.assert_array_equal(test_coord, cpt.coordinates)
    np.testing.assert_array_equal(test_depth, cpt.depth)
    np.testing.assert_array_equal(test_NAP, cpt.depth_to_reference)
    np.testing.assert_array_equal(test_tip, cpt.tip)
    np.testing.assert_array_equal(test_friction, cpt.friction)
    np.testing.assert_array_equal(test_friction_nbr, cpt.friction_nbr)
    np.testing.assert_array_equal(test_water, cpt.water)






