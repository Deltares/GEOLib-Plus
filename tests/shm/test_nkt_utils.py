
from geolib_plus.shm.nkt_utils import NktUtils

import numpy as np
import pytest

class TestNktUtils:

    @pytest.mark.unittest
    def test_get_default_nkt_bool(self):
        """
        Tests getting default nkt from boolean
        """

        # check nkt mean and std for saturated soil
        is_saturated = True
        nkt_mean, nkt_std = NktUtils.get_default_nkt(is_saturated)
        expected_nkt_mean, expected_nkt_std = 20, 5

        assert pytest.approx(expected_nkt_mean) == nkt_mean
        assert pytest.approx(expected_nkt_std) == nkt_std

        # check nkt mean and std for unsaturated soil
        is_saturated = False
        nkt_mean, nkt_std = NktUtils.get_default_nkt(is_saturated)
        expected_nkt_mean, expected_nkt_std = 60, 15

        assert pytest.approx(expected_nkt_mean) == nkt_mean
        assert pytest.approx(expected_nkt_std) == nkt_std

    @pytest.mark.unittest
    def test_get_default_nkt_array(self):
        """
        Tests getting default nkt value from an numpy array

        """
        # check nkt mean and std for saturated soil
        is_saturated = np.array([True, True, False])
        nkt_mean, nkt_std = NktUtils.get_default_nkt(is_saturated)
        expected_nkt_mean, expected_nkt_std = np.array([20,20,60]), np.array([5,5,15])

        np.testing.assert_array_equal(expected_nkt_mean, nkt_mean)
        np.testing.assert_array_equal(expected_nkt_std, nkt_std)

    @pytest.mark.unittest
    def test_get_nkt_stats_from_weighted_regression(self,su_qnet_data):
        """
        Tests getting nkt mean and variation coefficient from weighted regression
        """

        su, qnet = su_qnet_data

        nkt_mean, vc_qnet_nkt_tot = NktUtils.get_nkt_stats_from_weighted_regression(su, qnet)

        expected_nkt_mean = 15.12
        expected_vc = 0.171

        assert pytest.approx(expected_nkt_mean, abs=0.01) == nkt_mean
        assert pytest.approx(expected_vc, abs=0.01) == vc_qnet_nkt_tot

    @pytest.mark.unittest
    def test_get_characteristic_value_nkt_from_weighted_regression(self, su_qnet_data):
        """
        Tests getting characteristic value of nkt from weighted regression
        """
        su, qnet = su_qnet_data

        nkt_char = NktUtils.get_characteristic_value_nkt_from_weighted_regression(su, qnet)

        expected_nkt_char = 20.49

        assert pytest.approx(expected_nkt_char, abs=0.01) == nkt_char

    def test_get_prob_nkt_parameters_from_weighted_regression(self,su_qnet_data):
        """
        Tests getting prob parameters of qnet/nkt from weighted regression
        """
        su, qnet = su_qnet_data
        mu_prob, vc_prob = NktUtils.get_prob_nkt_parameters_from_weighted_regression(su, qnet)

        expected_mu, expected_vc = 15.12, 0.159

        assert pytest.approx(np.mean(qnet)/expected_mu,abs=0.01) == mu_prob
        assert pytest.approx(expected_vc, abs=0.01) == vc_prob


    def test_get_chararacteristic_value_nkt_from_statistics(self, su_qnet_data):
        """
        Tests getting characteristic values of nkt from statistics
        """
        su, qnet = su_qnet_data
        nkt_char = NktUtils.get_characteristic_value_nkt_from_statistics(su, qnet)

        expected_nkt_char = 20.80

        assert pytest.approx(nkt_char, abs=0.01) == expected_nkt_char

    def test_get_prob_nkt_parameters_from_statistics(self, su_qnet_data):
        """
        Tests getting probabilistic parameters of qnet/nkt from statistics
        """
        su, qnet = su_qnet_data
        mu_prob, vc_prob = NktUtils.get_prob_nkt_parameters_from_statistics(su, qnet)

        expected_mu, expected_vc = 16.01, 0.169

        assert pytest.approx(np.mean(qnet)/expected_mu,abs=0.01) == mu_prob
        assert pytest.approx(expected_vc, abs=0.001) == vc_prob

    @pytest.fixture
    def su_qnet_data(self):
        """
        reads test data
        """
        import csv

        # read test data
        with open(r'..\test_files\shm\su_qnet.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            data = np.array([row for row in csv_reader]).astype(float)

        # transform q_net and su to kPa
        q_net = data[:,0] * 1000
        su = data[:,1] * 1000
        return su, q_net








