from geolib_plus.GEF_CPT import GEF_CPT
from geolib_plus.BRO_XML_CPT import BRO_XML_CPT

import numpy as np
from pathlib import Path

def test_reading_compare():

    # Compare two files from bro (same CPT) in GEF and BRO Format
    # Should be comparable

    bro_file = Path("../tests/test_files/cpt/bro_xml/CPT000000003688_IMBRO_A.xml")

    gef_file = Path("../tests/test_files/cpt/gef/CPT000000003688_IMBRO_A.gef")
    gef_id = "CPT000000003688"

    gef_cpt = GEF_CPT()
    gef_cpt.read(gef_file, gef_id)

    bro_cpt = BRO_XML_CPT()
    bro_cpt.read(bro_file)

    np.testing.assert_array_equal(bro_cpt.name, gef_cpt.name)
    np.testing.assert_array_equal(bro_cpt.coordinates, bro_cpt.coordinates)

    # todo: JN The following tests current fail, the arrays are different size as are the depths
    np.testing.assert_array_equal(bro_cpt.depth, gef_cpt.depth)
    np.testing.assert_array_equal(bro_cpt.depth_to_reference, gef_cpt.depth_to_reference)
    np.testing.assert_array_equal(bro_cpt.tip, gef_cpt.tip)
    np.testing.assert_array_equal(bro_cpt.friction, gef_cpt.friction)
    np.testing.assert_array_equal(bro_cpt.friction_nbr, gef_cpt.friction_nbr)
    np.testing.assert_array_equal(bro_cpt.water, gef_cpt.water)






