import pytest
import numpy as np
from geolib_plus.gef_cpt import GefCpt
from pathlib import Path
from tests.utils import TestUtils


class TestGefCpt:
    @pytest.mark.integrationtest
    def test_gef_cpt_unit_tests(self):
        # simple read test of the cpt
        test_file = (
            TestUtils.get_local_test_data_dir("cpt/gef/unit_testing")
            / "test_gef_cpt_unit_tests.gef"
        )
        assert test_file.is_file(), f"File was not found at location {test_file}."
        cpt = GefCpt()
        cpt.read(test_file)
        # check that all values are initialized
        assert cpt
        assert max(cpt.penetration_length) == 25.52
        assert min(cpt.penetration_length) == 1.7

        assert max(cpt.depth) == 25.42
        assert min(cpt.depth) == 1.7

        # todo move depth_to_reference outside of cpt reader
        # assert min(cpt.depth_to_reference) == cpt.local_reference_level - max(cpt.depth)
        # assert max(cpt.depth_to_reference) == cpt.local_reference_level - min(cpt.depth)
        assert cpt.tip is not []
        assert cpt.friction is not []
        assert cpt.friction_nbr is not []
        assert cpt.water is not []
        assert cpt.inclination_x is not []
        assert cpt.inclination_y is not []
        assert cpt.time is not []
        assert cpt.coordinates == [130880.66, 497632.94]

        assert cpt.local_reference == ", maaiveld, vast horizontaal vlak"
        assert cpt.cpt_standard == ", ISO 22476-1 Toepassingsklasse 2, gevolgde norm"
        assert cpt.quality_class == ", ISO 22476-1 Toepassingsklasse 2, gevolgde norm"
        assert cpt.cpt_type == ", CP15-CF75PB1SN2/1701-1524, conus type/serienummer"
        assert cpt.result_time == "2017,07,03"

    @pytest.mark.integrationtest
    @pytest.mark.parametrize(
        "gef_file, expectation",
        [
            (None, pytest.raises(ValueError)),
            ("path_not_found", pytest.raises(FileNotFoundError)),
            ("path_not_found", pytest.raises(FileNotFoundError)),
            (None, pytest.raises(ValueError)),
        ],
    )
    def test_gefcpt_read_given_not_valid_gef_file_throws_file_not_found(
        self, gef_file: Path, expectation
    ):
        with expectation:
            GefCpt().read(gef_file)

    @pytest.mark.integrationtest
    def test_gef_cpt_given_valid_arguments_throws_nothing(self):
        # 1. Set up test data
        test_dir = TestUtils.get_local_test_data_dir("cpt/gef")
        filename = Path("CPT000000003688_IMBRO_A.gef")
        test_file = test_dir / filename

        # 2. Verify initial expectations
        assert test_file.is_file()

        # 3. Run test
        generated_output = GefCpt().read(test_file)

        # 4. Verify final expectations
        assert generated_output
