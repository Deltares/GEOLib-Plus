from geolib_plus.shm.state_utils import StateUtils

from pathlib import Path
import numpy as np
import pytest

from tests.utils import TestUtils


class TestNktUtils:
    # S = S_statistiek = 0.38, sigma = 0.01, kar = 0.37
    # m = 0.8

    @pytest.mark.unittest
    def test_calculate_yield_stress(self):
        """
        Tests calculating yield stress trough shansep relation
        """

        # calculate yield stress trough shansep relation
        calculated_yield_stress = StateUtils.calculate_yield_stress(7.7,16.4,0.38,0.8)

        # set expected yield stress
        expected_yield_stress = 25.3289

        # assert
        assert pytest.approx(expected_yield_stress,1e-4) == calculated_yield_stress

        # # check nkt mean and std for saturated soil
        # is_saturated = True
        # nkt_mean, nkt_std = NktUtils.get_default_nkt(is_saturated)
        # expected_nkt_mean, expected_nkt_std = 20, 5
        #
        # assert pytest.approx(expected_nkt_mean) == nkt_mean
        # assert pytest.approx(expected_nkt_std) == nkt_std
        #
        # # check nkt mean and std for unsaturated soil
        # is_saturated = False
        # nkt_mean, nkt_std = NktUtils.get_default_nkt(is_saturated)
        # expected_nkt_mean, expected_nkt_std = 60, 15
        #
        # assert pytest.approx(expected_nkt_mean) == nkt_mean
        # assert pytest.approx(expected_nkt_std) == nkt_std