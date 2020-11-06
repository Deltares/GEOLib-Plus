from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation

from tests.utils import TestUtils
import numpy as np
import pytest


class TestShapeFiles:
    @pytest.mark.unittest
    def test_lithology(self):
        robertson = RobertsonCptInterpretation()
        robertson.soil_types()
        coords_test = []
        Qtn = [2, 2, 10, 7, 20, 100, 900, 700, 700]
        Fr = [0.2, 9, 8, 1, 0.2, 0.5, 0.2, 3, 9]
        lithology_test = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

        [coords_test.append([Fr[i], Qtn[i]]) for i in range(len(Fr))]

        litho, coords = robertson.lithology(Qtn, Fr)
        np.testing.assert_array_equal(coords_test, coords)
        np.testing.assert_array_equal(lithology_test, litho)