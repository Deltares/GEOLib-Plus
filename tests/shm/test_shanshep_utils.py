from geolib_plus.shm.shanshep_utils import ShanshepUtils
from tests.utils import TestUtils

import pandas as pd
import pytest
from pathlib import Path


class TestShanshepUtils:
    @pytest.mark.unittest
    def test_get_shanshep_parameters(self):
        # denife inputs
        path_inputs = TestUtils.get_local_test_data_dir(
            Path("shm", "Data_KIJK_DSS.csv")
        )
        inputs = pd.read_csv(path_inputs, delimiter=";")
        mask = inputs.TestConditions == "In situ"
        inputs_modified = inputs.loc[mask].dropna(subset=["tau_40"])

        # run tests
        (S, m) = ShanshepUtils.get_shanshep_parameters(
            (inputs_modified.Pc / inputs_modified.sigma_v0_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_v0_eff.to_numpy(),
        )
        assert S - 0.136 < 0.00051
        assert m - 0.789 < 0.00051

        S = 2
        # test with a given S parameter
        (S_output, m) = ShanshepUtils.get_shanshep_parameters(
            (inputs_modified.Pc / inputs_modified.sigma_v0_eff).to_numpy(),
            inputs_modified.tau_40.to_numpy(),
            inputs_modified.sigma_v0_eff.to_numpy(),
            S=S,
        )

        assert S_output == 2
        assert m - (-1.661) < 0.00051
