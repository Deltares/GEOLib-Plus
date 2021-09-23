from geolib_plus.shm.state_utils import StateUtils
import pytest


class TestNktUtils:

    @pytest.mark.unittest
    def test_calculate_yield_stress(self):
        """
        Tests calculating yield stress through Shansep relation
        """

        # calculate yield stress through Shansep relation
        calculated_yield_stress = StateUtils.calculate_yield_stress(7.7,16.4,0.38,0.8)

        # set expected yield stress
        expected_yield_stress = 21.36352674385969

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
        expected_mean = 34.610486101681566
        expected_std = 5.0001643709523655

        assert pytest.approx(expected_mean) == mean_yield_stress
        assert pytest.approx(expected_std) == std_yield_stress


    @pytest.mark.unittest
    def test_calculate_pop_prob_parameters_from_cpt(self):
        """
        Tests  calculating mean and standard deviation of pre overburden pressure
        """
        mean_pop, std_pop = StateUtils.calculate_pop_prob_parameters_from_cpt(15.8, 180, 0.38, 0.8, 16.01, 0.169)

        expected_mean = 18.810486101681565
        expected_std = 5.0001643709523655

        assert pytest.approx(expected_mean) == mean_pop
        assert pytest.approx(expected_std) == std_pop


    @pytest.mark.unittest
    def test_calculate_ocr_prob_parameters_from_cpt(self):
        """
        Tests  calculating mean and standard deviation of over consolidation ratio
        """
        mean_ocr, std_ocr = StateUtils.calculate_ocr_prob_parameters_from_cpt(15.8, 180, 0.38, 0.8, 16.01, 0.169)

        expected_mean = 2.190537095043137
        expected_std = 0.31646609942736487

        assert pytest.approx(expected_mean) == mean_ocr
        assert pytest.approx(expected_std) == std_ocr

    @pytest.mark.unittest
    def test_calculate_characteristic_yield_stress(self):
        """
        Tests calculating characteristic yield stress
        """

        char_yield_stress = StateUtils.calculate_characteristic_yield_stress(15.8, 15.8, 180, 0.37, 0.8, 20.8)

        expected_char_yield_stress = 25.798534471920842

        assert pytest.approx(expected_char_yield_stress) == char_yield_stress

    @pytest.mark.unittest
    def test_calculate_characteristic_pop(self):
        """
        Tests calculating characteristic pop
        """

        char_pop = StateUtils.calculate_characteristic_pop(15.8, 15.8, 180, 0.37, 0.8, 20.8)

        expected_char_pop = 9.998534471920841

        assert pytest.approx(expected_char_pop) == char_pop

    @pytest.mark.unittest
    def test_calculate_characteristic_pop(self):
        """
        Tests calculating characteristic pop
        """

        char_ocr = StateUtils.calculate_characteristic_ocr(15.8, 15.8, 180, 0.37, 0.8, 20.8)

        expected_char_ocr = 1.6328186374633444

        assert pytest.approx(expected_char_ocr) == char_ocr



