from geolib_plus.shm.shansep_utils import ShansepUtils
from tests.utils import TestUtils

import pandas as pd
import pytest
from pathlib import Path
import numpy as np


class TestShanshepUtils:
    @pytest.mark.unittest
    def test_get_shansep_parameters(self):
        # denife inputs
        path_inputs = TestUtils.get_local_test_data_dir(
            Path("shm", "Data_KIJK_DSS.csv")
        )
        inputs = pd.read_csv(path_inputs, delimiter=";")
        mask = inputs.TestConditions == "In situ"
        inputs_modified = inputs.loc[mask].dropna(subset=["tau_40"])

        # run tests
        (S_test, m_test) = ShansepUtils.get_shansep_parameters(
            (inputs_modified.Pc / inputs_modified.sigma_v0_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_v0_eff.to_numpy(),
        )
        assert S_test - 0.421 < 0.00051
        assert m_test - 0.789 < 0.00051

        # test with a given S parameter
        (S_output, m_output) = ShansepUtils.get_shansep_parameters(
            (inputs_modified.Pc / inputs_modified.sigma_v0_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_v0_eff.to_numpy(),
            S=S_test,
        )

        assert S_output == S_test
        assert m_output - m_test < 0.00051

        # test with a given m parameter
        (S_output, m_output) = ShansepUtils.get_shansep_parameters(
            (inputs_modified.Pc / inputs_modified.sigma_v0_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_v0_eff.to_numpy(),
            m=m_test,
            S=None,
        )

        assert S_output - S_test < 0.00051
        assert m_test == m_output

    @pytest.mark.unittest
    def test_get_shansep_parameters_test_validity(self):
        # denife inputs
        path_inputs = TestUtils.get_local_test_data_dir(
            Path("shm", "Data_KIJK_DSS.csv")
        )
        inputs = pd.read_csv(path_inputs, delimiter=";")
        mask = inputs.TestConditions == "In situ"
        inputs_modified = inputs.loc[mask].dropna(subset=["tau_40"])
        # test with a given S parameter
        (S_test, m_test) = ShansepUtils.get_shansep_parameters(
            (inputs_modified.Pc / inputs_modified.sigma_v0_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_v0_eff.to_numpy(),
        )

        # If a higher m is inputted a smaller log(S) should be produced
        (S_output, m_output) = ShansepUtils.get_shansep_parameters(
            (inputs_modified.Pc / inputs_modified.sigma_v0_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_v0_eff.to_numpy(),
            m=m_test + 0.1,
        )

        assert np.log(S_output) < np.log(S_test)

        # If a lower m is inputted a higher log(S) should be produced
        (S_output, m_output) = ShansepUtils.get_shansep_parameters(
            (inputs_modified.Pc / inputs_modified.sigma_v0_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_v0_eff.to_numpy(),
            m=m_test - 0.1,
        )

        assert np.log(S_output) > np.log(S_test)
