import pytest
import math
import numpy as np

from tests.utils import TestUtils
from geolib_plus.hardening_soil_model_parameters import (
    HardeningSoilModelParameters,
    HardingSoilCalculationType,
)


class TestHardeningSoilParameters:
    @pytest.mark.systemtest
    def test_calculate_stiffness_compressibility_parameters_input(self):
        # define inputs for the model as floats
        eo = 0.8
        sigma_ref_v = 10
        v_ur = 0.5
        Cc = 1.15 * (eo - 0.35)
        Cs = 0.5
        m = 0.65
        # initialize class and check initial expectations
        parameters_for_HS = HardeningSoilModelParameters(
            eo=eo, sigma_ref_v=sigma_ref_v, v_ur=v_ur, Cc=Cc, Cs=Cs, m=m
        )
        assert parameters_for_HS
        # run test
        parameters_for_HS.calculate_stiffness(
            HardingSoilCalculationType.COMPRESSIBILITYPARAMETERS
        )
        # check results
        assert parameters_for_HS.E_oed_ref == math.log(10) * (1 + eo) * sigma_ref_v / Cc
        assert parameters_for_HS.E_ur_ref == (
            math.log(10) * (1 + eo) * sigma_ref_v / Cs
        ) * ((1 + v_ur) * (1 - 2 * v_ur) / (1 - v_ur))

    @pytest.mark.systemtest
    def test_calculate_stiffness_cone_resistance(self):
        # define inputs for the model as floats
        # some of the inputs could come from the cpt so they might be
        # defined in the form of arrays
        qc = np.array([1, 2, 2, 10])
        v_ur = np.array([0.5, 0.22, 0.23, 0.4])
        sigma_cpt_h = np.array([1, 10, 20, 30])
        # other inputs could be the same for each soil layer
        sigma_ref_h = 100
        m = 0.85
        # initialize class and check initial expectations
        parameters_for_HS = HardeningSoilModelParameters(
            qc=qc, sigma_cpt_h=sigma_cpt_h, v_ur=v_ur, sigma_ref_h=sigma_ref_h, m=m
        )
        assert parameters_for_HS
        # run test
        parameters_for_HS.calculate_stiffness(HardingSoilCalculationType.CONERESISTANCE)
        # check results
        assert parameters_for_HS.E_oed_ref is not None
        assert parameters_for_HS.E_ur_ref is not None
