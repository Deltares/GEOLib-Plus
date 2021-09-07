from geolib_plus.shm.general_utils import GeneralUtils
from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation

from pathlib import Path
import pytest
import matplotlib.pyplot as plt

from tests.utils import TestUtils


class TestGeneralUtils:
    @pytest.mark.integrationtest
    def test_linearise_data_over_layer(self):
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

        layer_depth = cpt_gef.depth[:200]
        layer_data = cpt_gef.Qtn[:200]
        linearised_data = GeneralUtils.linearise_data_over_layer(
            data_to_linearized=layer_data, depth=layer_depth
        )
        plt.plot(layer_data, layer_depth, "-o", color="black", label="real layer data")
        plt.plot(
            linearised_data, layer_depth, "-", color="red", label="linearised data"
        )
        plt.legend()
        plt.show()

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
                data_to_linearized=layer_data, depth=layer_depth
            )
            assert (
                "The lenth of the arrays inputted are  smaller than the buffer zone. "
                in str(excinfo.value)
            )
