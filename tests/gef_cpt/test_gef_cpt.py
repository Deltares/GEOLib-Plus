import pytest
from pathlib import Path
from tests.utils import TestUtils

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

    @pytest.mark.integrationtest
    def test_gef_cpt_given_valid_arguments_throws_nothing(self):
        # 1. Set up test data
        test_dir = TestUtils.get_local_test_data_dir("cpt\\gef")
        filename = Path("CPT000000003688_IMBRO_A.gef")
        test_file = test_dir / filename

        # 2. Verify initial expectations
        assert test_file.is_file()

        # 3. Run test
        generated_output = GefCpt().read(test_file, 42)

        # 4. Verify final expectations
        assert generated_output
