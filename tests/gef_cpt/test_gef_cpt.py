import pytest
<<<<<<< HEAD:tests/GEF_CPT/test_GEF_CPT.py
from geolib_plus.GEF_CPT import GEF_CPT
import numpy as np
=======
from geolib_plus.gef_cpt import GefCpt
>>>>>>> fb3e7dcdf4646e315885acbeddb67bfd1c133131:tests/gef_cpt/test_gef_cpt.py


# todo JN: write unit tests
class TestGefCpt:
    @pytest.mark.unittest
    @pytest.mark.workinprogress
    def test_gef_cpt_unit_tests(self):
        # simple read test of the cpt
        cpt = GEF_CPT()
        cpt.read(gef_file='tests\\test_files\\cpt\\gef\\unit_testing\\test_gef_cpt_unit_tests.gef', id=1)
        # check that all values are initialized 
        assert cpt
        assert max(cpt.depth) == 25.6
        assert min(cpt.depth) == 1.66
        reference_depth = -1.97
        assert min(cpt.depth_to_reference) == reference_depth- max(cpt.depth)
        assert max(cpt.depth_to_reference) == reference_depth - min(cpt.depth)
        assert cpt.tip is not []
        assert cpt.friction is not []
        assert cpt.friction_nbr is not []
        assert cpt.water is not []
        assert cpt.coordinates == [130880.66, 497632.94]

