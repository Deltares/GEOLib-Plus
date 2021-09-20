from geolib_plus.shm.state_utils import StateUtils
from geolib_plus.shm.nkt_utils import NktMethod

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

    @pytest.mark.unittest
    def test_calculate_yield_stress_prob_parameters_from_cpt(self):
        """
        Tests  calculating mean and standard deviation of yield stress
        """

        mean_yield_stress, std_yield_stress = StateUtils.calculate_yield_stress_prob_parameters_from_cpt(15.8, 180,
                                                                                                         0.38, 0.8,
                                                                                                         16.01, 0.169)
        expected_mean = 36.983464282192045
        expected_std = 5.0001643709523655

        assert pytest.approx(expected_mean) == mean_yield_stress
        assert pytest.approx(expected_std) == std_yield_stress


    @pytest.mark.unittest
    def test_calculate_pop_prob_parameters_from_cpt(self):
        """
        Tests  calculating mean and standard deviation of pre overburden pressure
        """
        mean_pop, std_pop = StateUtils.calculate_pop_prob_parameters_from_cpt(15.8, 180, 0.38, 0.8, 16.01, 0.169)

        expected_mean = 21.183464282192045
        expected_std = 5.0001643709523655

        assert pytest.approx(expected_mean) == mean_pop
        assert pytest.approx(expected_std) == std_pop


    @pytest.mark.unittest
    def test_calculate_ocr_prob_parameters_from_cpt(self):
        """
        Tests  calculating mean and standard deviation of over consolidation ratio
        """
        mean_ocr, std_ocr = StateUtils.calculate_ocr_prob_parameters_from_cpt(16.4, 180, 0.38, 0.8, 16.01, 0.169)

        expected_mean = 2.255089285499515
        expected_std = 0.30488807139953444

        assert pytest.approx(expected_mean) == mean_ocr
        assert pytest.approx(expected_std) == std_ocr

    @pytest.fixture
    def su_qnet_data(self):
        """
        reads test data
        """
        import csv

        # read test data
        with open(TestUtils.get_local_test_data_dir(Path("shm", "su_qnet.csv"))) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            data = np.array([row for row in csv_reader]).astype(float)

        # transform q_net and su to kPa
        q_net = data[:, 0] * 1000
        su = data[:, 1] * 1000
        return su, q_net


