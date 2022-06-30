from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geolib_plus.shm.shansep_utils import ShansepUtils
from tests.utils import TestUtils


class TestShanshepUtils:
    @pytest.mark.unittest
    def test_get_shansep_prob_parameters_with_linear_regression(self):
        # denife inputs
        path_inputs = TestUtils.get_local_test_data_dir(Path("shm", "Data_KIJK_DSS.csv"))
        inputs = pd.read_csv(path_inputs, delimiter=";")
        inputs_modified = inputs.dropna(subset=["tau_40"])

        inputs_modified["OCR"] = inputs.Pc / inputs.sigma_vc_eff
        inputs_modified["OCR"][inputs_modified["OCR"] < 1] = 1

        # run tests
        (
            (S_test, s_std),
            (m_test, m_std),
            _,
        ) = ShansepUtils.get_shansep_prob_parameters_with_linear_regression(
            inputs_modified["OCR"],
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_vc_eff.to_numpy(),
        )

        # assert
        assert pytest.approx(S_test, abs=0.00051) == 0.386
        assert pytest.approx(m_test, abs=0.00051) == 0.926

        assert pytest.approx(s_std, abs=0.00051) == 0.005
        assert pytest.approx(m_std, abs=0.00051) == 0.027

        # test with a given S parameter
        (
            (S_output, s_std),
            (m_output, m_std),
            _,
        ) = ShansepUtils.get_shansep_prob_parameters_with_linear_regression(
            inputs_modified["OCR"],
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_vc_eff.to_numpy(),
            S=S_test,
        )

        assert S_output == S_test
        assert abs(m_output - m_test) < 0.0051

        # test with a given m parameter
        (
            (S_output, s_std),
            (m_output, m_std),
            _,
        ) = ShansepUtils.get_shansep_prob_parameters_with_linear_regression(
            inputs_modified["OCR"],
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_vc_eff.to_numpy(),
            m=m_test,
            S=None,
        )

        assert abs(S_output - S_test) < 0.0051
        assert m_test == m_output

    @pytest.mark.unittest
    def test_get_shansep_prob_parameters_with_linear_regression_validity(self):
        # denife inputs
        path_inputs = TestUtils.get_local_test_data_dir(Path("shm", "Data_KIJK_DSS.csv"))
        inputs = pd.read_csv(path_inputs, delimiter=";")
        mask = inputs.TestConditions == "In situ"
        inputs_modified = inputs.loc[mask].dropna(subset=["tau_40"])
        # test with a given S parameter
        (
            (S_test, s_test_std),
            (m_test, m_test_std),
            _,
        ) = ShansepUtils.get_shansep_prob_parameters_with_linear_regression(
            (inputs_modified.Pc / inputs_modified.sigma_v0_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_vc_eff.to_numpy(),
        )

        # If a higher m is inputted a smaller log(S) should be produced
        (
            (S_output, s_std),
            _,
            _,
        ) = ShansepUtils.get_shansep_prob_parameters_with_linear_regression(
            (inputs_modified.Pc / inputs_modified.sigma_v0_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_vc_eff.to_numpy(),
            m=m_test + 0.1,
        )

        assert np.log(S_output) < np.log(S_test)

        # If a lower m is inputted a higher log(S) should be produced
        (
            (S_output, s_std),
            _,
            _,
        ) = ShansepUtils.get_shansep_prob_parameters_with_linear_regression(
            (inputs_modified.Pc / inputs_modified.sigma_vc_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_vc_eff.to_numpy(),
            m=m_test - 0.1,
        )

        assert np.log(S_output) > np.log(S_test)

    @pytest.mark.unittest
    def test_calculate_characteristic_shansep_parameters_with_linear_regression(self):
        """
        Test get characteristic values of shear strength ratio S and strength increase component m
        """

        path_inputs = TestUtils.get_local_test_data_dir(Path("shm", "Data_KIJK_DSS.csv"))
        inputs = pd.read_csv(path_inputs, delimiter=";")
        inputs_modified = inputs.dropna(subset=["tau_40"])

        inputs_modified["OCR"] = inputs.Pc / inputs.sigma_vc_eff
        inputs_modified["OCR"][inputs_modified["OCR"] < 1] = 1

        (
            S_char,
            m_char,
        ) = ShansepUtils.calculate_characteristic_shansep_parameters_with_linear_regression(
            inputs_modified["OCR"],
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_vc_eff.to_numpy(),
        )

        expected_S_char = 0.377
        expected_m_char = 0.879

        assert pytest.approx(S_char, abs=1e-3) == expected_S_char
        assert pytest.approx(m_char, abs=1e-3) == expected_m_char
