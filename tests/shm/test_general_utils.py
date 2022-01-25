import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.shm.general_utils import GeneralUtils
from tests.utils import TestUtils


class TestGeneralUtils:
    @pytest.mark.unittest
    def test_linearise_data_over_layer(self):
        layer_data = np.ones(10)
        layer_depth = np.ones(10)
        linearised_data = GeneralUtils.linearise_data_over_layer(
            data_to_linearized=layer_data, depth=layer_depth
        )
        assert math.isclose(linearised_data[0], 1)
        assert math.isclose(linearised_data[-1], 1)

    @pytest.mark.integrationtest
    def test_linearise_data_over_layer_error_raised(self):
        # define test gef
        test_file_gef = Path(
            TestUtils.get_local_test_data_dir("cpt/gef"), "cpt_with_water.gef"
        )
        # initialize geolib plus
        cpt_gef = GefCpt()
        # check initial expectation
        assert cpt_gef
        # read the cpt for each type of file
        cpt_gef.read(test_file_gef)
        # pre-process cpt data
        cpt_gef.pre_process_data()
        # interpret data
        interpreter = RobertsonCptInterpretation()
        cpt_gef.interpret_cpt(interpreter)
        # check initial expectations
        assert cpt_gef.name == "CPT000000029380"

        layer_depth = cpt_gef.depth[:5]
        layer_data = cpt_gef.Qtn[:5]
        with pytest.raises(ValueError) as excinfo:
            GeneralUtils.linearise_data_over_layer(
                data_to_linearized=layer_data, depth=layer_depth, buffer_zone=5
            )
            assert (
                "The lenth of the arrays inputted are  smaller than the buffer zone. "
                in str(excinfo.value)
            )
