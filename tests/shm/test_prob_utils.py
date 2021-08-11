from geolib_plus.shm.prob_utils import ProbUtils
import pytest
import numpy as np


class TestProbUtils:

    def setup_method(self):
        "setup random seed"
        np.random.seed(2021)

    @pytest.mark.unittest
    def test_calculate_student_t_factor(self):
        """
        tests calculate student t factor
        """

        # expected value comes from well known student-t table
        expected_value = -1.725

        # calculate
        calculated_value = ProbUtils.calculate_student_t_factor(20,0.05)

        # assert
        np.testing.assert_almost_equal(expected_value,calculated_value, 3)

    @pytest.mark.unittest
    def test_correct_std_with_student_t(self):
        """
        Tests correct standard deviation with student t factor.
        """

        # set expected value 1
        std = 2
        expected_value_upper_bound = -1.725 / -1.645 *std * np.sqrt(1+1/21)

        # calculate corrected standard deviation upper bound
        calculated_value_upper_bound = ProbUtils.correct_std_with_student_t(21, 0.05, std)

        # calculate corrected standard deviation lower bound
        calculated_value_lower_bound = ProbUtils.correct_std_with_student_t(int(1e10), 0.05, std, a=1)

        # calculate corrected standard deviation in between
        calculated_value_between = ProbUtils.correct_std_with_student_t(21, 0.05, std, a=1)

        # assert
        np.testing.assert_almost_equal(expected_value_upper_bound,calculated_value_upper_bound, 3)
        np.testing.assert_almost_equal(0, calculated_value_lower_bound, 3)

        assert expected_value_upper_bound > calculated_value_between > calculated_value_lower_bound

    @pytest.mark.unittest
    def test_get_mean_std_from_lognormal(self):

        # set normal mean and std
        mean, std = 1.53523, 0.38525
        expected_mean, expected_std = 5, 2

        # calculate log_mean and log_std
        calc_mean, calc_std = ProbUtils.get_mean_std_from_lognormal(mean, std)

        # create lognormal dist with log_mean and log_std
        normal_dist = np.random.normal(calc_mean,calc_std, 10000000)

        # approximate mean and std from distribution
        approximated_mean = np.mean(normal_dist)
        approximated_std = np.std(normal_dist)

        assert pytest.approx(approximated_mean, rel=0.01) == expected_mean
        assert pytest.approx(approximated_std, rel=0.01) == expected_std

    @pytest.mark.unittest
    def test_get_log_mean_std_from_normal(self):
        # set normal mean and std
        mean, std = 5, 2

        # calculate log_mean and log_std
        calc_mean, calc_std = ProbUtils.get_log_mean_std_from_normal(mean, std)

        # create lognormal dist with log_mean and log_std
        log_normal_dist = np.random.lognormal(calc_mean, calc_std, 10000000)

        # approximate mean and std from distribution
        approximated_mean = np.mean(log_normal_dist)
        approximated_std = np.std(log_normal_dist)

        assert pytest.approx(approximated_mean, rel=0.01) == mean
        assert pytest.approx(approximated_std, rel=0.01) == std

    @pytest.mark.unittest
    def test_convert_mean_and_std_normal_to_log_to_normal(self):
        """
        Tests if converting mean and std to log mean and std and back results in the initial values
        """

        mean, std = 5, 2
        log_mean, log_std = ProbUtils.get_log_mean_std_from_normal(mean, std)
        calc_mean, calc_std = ProbUtils.get_mean_std_from_lognormal(log_mean, log_std)

        assert pytest.approx(mean) == calc_mean
        assert pytest.approx(std) == calc_std

    @pytest.mark.unittest
    def test_calculate_log_stats(self):
        """
        Tests calculate log mean and standard deviation from lognormal distribution
        """

        # set log normal data
        log_mean, log_std = 1.53523, 0.38525
        data = np.random.lognormal(log_mean, log_std, 10000000)

        # calculate log mean and std
        calc_log_mean, calc_log_std = ProbUtils.calculate_log_stats(data)

        # assert
        assert pytest.approx(log_mean, rel=0.01) == calc_log_mean
        assert pytest.approx(log_std, rel=0.01) == calc_log_std

    @pytest.mark.unittest
    def test_calculate_normal_stats(self):
        """
       Tests calculate mean and standard deviation from normal distribution
       """

        # set log normal data
        log_mean, log_std = 5, 2
        data = np.random.normal(log_mean, log_std, 10000000)

        # calculate log mean and std
        calc_mean, calc_std = ProbUtils.calculate_normal_stats(data)

        # assert
        assert pytest.approx(log_mean, rel=0.01) == calc_mean
        assert pytest.approx(log_std, rel=0.01) == calc_std

    @pytest.mark.unittest
    def test_calculate_characteristic_value_from_dataset_log_normal(self):

        # generate lognormal dataset (mean=5, std=2)
        log_mean, log_std = 1.53523, 0.38525

        # calculate characteristic value from a large dataset
        large_local_data_set = np.random.lognormal(log_mean, log_std, 10000000)
        x_kar_large = ProbUtils.calculate_characteristic_value_from_dataset(large_local_data_set, True, True)

        # calculate characteristic value from a small dataset
        small_local_dataset = np.random.lognormal(log_mean, log_std, 10)
        x_kar_small = ProbUtils.calculate_characteristic_value_from_dataset(small_local_dataset, True, True)

        # check if exp(log_mean) is close to characteristic value of large dataset
        assert pytest.approx(np.exp(log_mean), rel=0.01) == x_kar_large

        # check if low characteristic value from a small dataset is smaller than characteristic value from a large
        # dataset
        assert x_kar_small < x_kar_large

    @pytest.mark.unittest
    def test_calculate_prob_parameters_from_lognormal(self):
        # generate lognormal dataset (mean=5, std=2)
        log_mean, log_std = 1.53523, 0.38525

        # calculate prob mean and std from a large dataset
        large_local_data_set = np.random.lognormal(log_mean, log_std, 10000000)
        mean_prob, std_prob = ProbUtils.calculate_prob_parameters_from_lognormal(large_local_data_set, True)

        # calculate prob mean and std from a small dataset
        small_local_dataset = np.random.lognormal(log_mean, log_std, 10)
        mean_prob_small, std_prob_small = ProbUtils.calculate_prob_parameters_from_lognormal(small_local_dataset, True)

        # set expected mean and std of small dataset (values are meant as a regression test
        expected_mean_small = 4.91908583464898
        expected_std_small = 0.6076886211545873

        # check if exp(log_mean) is close to mean_prob for a large dataset
        assert pytest.approx(np.exp(log_mean), rel=0.01) == mean_prob
        assert pytest.approx(0, abs=0.01) == std_prob

        # check if mean_prob and std_prob has changed (regression test)
        assert pytest.approx(mean_prob_small) == expected_mean_small
        assert pytest.approx(std_prob_small) == expected_std_small





