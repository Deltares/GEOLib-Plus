# External
from pathlib import Path

import pytest

from geolib_plus.gef_cpt import *
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from tests.utils import TestUtils


@pytest.fixture()
def test_file():
    yield TestUtils.get_local_test_data_dir(
        Path("cpt", "gef", "CPT000000003688_IMBRO_A.gef")
    )


class TestBugGeolibPlus35:
    @pytest.mark.systemtest
    def test_no_import_geolib(self, test_file):
        # initialise models
        cpt = GefCpt()
        # test initial expectations
        assert cpt
        # read gef file
        cpt.read(filepath=test_file)
        assert cpt

    @pytest.mark.systemtest
    def test_import_geolib(self, test_file):
        # import geolib
        import geolib
        # This is a workaround
        import pydantic

        pydantic.BaseModel.Config.validate_assignment = False

        # initialise models
        cpt = GefCpt()
        # read gef file
        cpt.read(filepath=test_file)
        assert cpt
