import pytest
import numpy as np

from tests.utils import TestUtils
from geolib_plus.relative_density_correlated_parametes import (
    RelativeDensityCorrelatedParameters,
)


class TestRelativeDensityCorrelatedParameters:
    @pytest.mark.systemtest
    def test_calculate_using_RD(self):
        # define relative density as float
        RD = 100
        # create class
        RD_parameters = RelativeDensityCorrelatedParameters.calculate_using_RD(RD)
        # check expectations
        assert RD_parameters
        assert RD_parameters.gamma_unsat == 19
        assert RD_parameters.gamma_sat == 20.6
        assert RD_parameters.E_ur_ref == 180000
        # then define tests with test with and array input
        RD = np.array([100, 100])
        # create class
        RD_parameters_array = RelativeDensityCorrelatedParameters.calculate_using_RD(RD)
        # check expectations
        assert RD_parameters_array
        assert RD_parameters_array.gamma_unsat[0] == 19
        assert RD_parameters_array.gamma_sat[0] == 20.6
        assert RD_parameters_array.E_50_ref[0] == 60000
