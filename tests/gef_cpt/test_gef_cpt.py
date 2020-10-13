import pytest
from pathlib import Path
from geolib_plus.GEF_CPT.gef_cpt import GefCpt


class TestGefCpt:
    @pytest.mark.integrationtest
    @pytest.mark.parametrize(
        "gef_file, arg_id, expectation",
        [
            (None, None, pytest.raises(ValueError)),
            ("path_not_found", None, pytest.raises(ValueError)),
            ("path_not_found", 42, pytest.raises(FileNotFoundError)),
            (None, 42, pytest.raises(ValueError)),
        ],
    )
    def test_gefcpt_read_given_not_valid_gef_file_throws_file_not_found(
        self, gef_file: Path, arg_id: int, expectation
    ):
        with expectation:
            gef_cpt = GefCpt()
            gef_cpt.read(gef_file, arg_id)

    @pytest.mark.unittest
    @pytest.mark.workinprogress
    def test_gef_cpt_unit_tests(self):
        raise NotImplementedError
