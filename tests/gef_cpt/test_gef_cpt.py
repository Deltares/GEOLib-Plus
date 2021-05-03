import pytest
import numpy as np
from geolib_plus.gef_cpt import GefCpt
from pathlib import Path
from tests.utils import TestUtils
import geolib


class TestGefCptGeolibPlusToGeolib:
    @pytest.mark.integrationtest
    def test_make_geolib_profile(self):
        # define test gef
        test_file_gef = Path(TestUtils.get_local_test_data_dir("cpt/gef"), "KW19-3.gef")
        # initialize geolib plus
        cpt_gef = GefCpt()
        # check initial expectation
        assert cpt_gef
        # read the cpt for each type of file
        cpt_gef.read(test_file_gef)
        # check initial expectations
        assert cpt_gef


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
        assert max(cpt.penetration_length) == 25.61
        assert min(cpt.penetration_length) == 1.62

        assert max(cpt.depth) == 25.52
        assert min(cpt.depth) == 1.62

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

        assert cpt.local_reference == "maaiveld, vast horizontaal vlak"
        assert cpt.cpt_standard == "ISO 22476-1 Toepassingsklasse 2, gevolgde norm"
        assert cpt.quality_class == "ISO 22476-1 Toepassingsklasse 2, gevolgde norm"
        assert cpt.cpt_type == "CP15-CF75PB1SN2/1701-1524, conus type/serienummer"
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
            generated_output = GefCpt()
            generated_output.read(gef_file)

    @pytest.mark.integrationtest
    def test_gef_cpt_given_valid_arguments_throws_nothing(self):
        # 1. Set up test data
        test_dir = TestUtils.get_local_test_data_dir("cpt/gef")
        filename = Path("CPT000000003688_IMBRO_A.gef")
        test_file = test_dir / filename

        # 2. Verify initial expectations
        assert test_file.is_file()

        # 3. Run test
        generated_output = GefCpt()
        generated_output.read(test_file)

        # 4. Verify final expectations
        assert generated_output


class TestCheckDataForError:
    @pytest.mark.unittest
    def test_has_points_with_error_with_error(self):
        gef_cpt = GefCpt()

        # set inputs
        gef_cpt.depth = np.linspace(-1, 12, 6)
        gef_cpt.friction = np.array([-5, -2, -999, -999, -3, -4])
        gef_cpt.pore_pressure_u2 = np.full(6, 1000)
        gef_cpt.friction_nbr = np.full(6, 5)
        gef_cpt.penetration_length = np.linspace(-1, 12, 6)

        gef_cpt.error_codes["depth"] = -1
        gef_cpt.error_codes["friction"] = -999
        gef_cpt.error_codes["pore_pressure_u2"] = -1
        gef_cpt.error_codes["friction_nbr"] = -1

        with pytest.raises(ValueError) as excinfo:
            gef_cpt.has_points_with_error()
            assert "friction" in str(excinfo.value)

    @pytest.mark.unittest
    def test_has_points_with_error_without_error(self):
        gef_cpt = GefCpt()

        # set inputs
        gef_cpt.depth = np.linspace(-1, 12, 6)
        gef_cpt.friction = np.array([-5, -2, -9, -9, -3, -4])
        gef_cpt.pore_pressure_u2 = np.full(6, 1000)
        gef_cpt.friction_nbr = np.full(6, 5)
        gef_cpt.penetration_length = np.linspace(-1, 12, 6)

        gef_cpt.error_codes["depth"] = -99
        gef_cpt.error_codes["friction"] = -999
        gef_cpt.error_codes["pore_pressure_u2"] = -99
        gef_cpt.error_codes["friction_nbr"] = -99

        gef_cpt.has_points_with_error()
        assert gef_cpt

    @pytest.mark.unittest
    def test_has_duplicated_depth_values_without_duplication(self):
        gef_cpt = GefCpt()
        # set inputs
        gef_cpt.depth = np.linspace(-1, 12, 6)
        gef_cpt.friction = np.array([-5, -2, -9, -9, -3, -4])
        gef_cpt.pore_pressure_u2 = np.full(6, 1000)
        gef_cpt.friction_nbr = np.full(6, 5)
        gef_cpt.penetration_length = np.linspace(-1, 12, 6)
        assert not gef_cpt.has_duplicated_depth_values()

    @pytest.mark.unittest
    def test_has_duplicated_depth_values_with_duplication(self):
        gef_cpt = GefCpt()
        # set inputs
        gef_cpt.depth = np.linspace(-1, 12, 6)
        gef_cpt.friction = np.array([-5, -2, -9, -9, -3, -4])
        gef_cpt.pore_pressure_u2 = np.full(6, 1000)
        gef_cpt.friction_nbr = np.full(6, 5)
        gef_cpt.penetration_length = np.linspace(-1, 12, 6)

        # create duplications
        gef_cpt.penetration_length[0] = gef_cpt.penetration_length[1]
        gef_cpt.penetration_length[-1] = gef_cpt.penetration_length[-2]

        with pytest.raises(ValueError) as excinfo:
            gef_cpt.has_duplicated_depth_values()
            assert "depth" in excinfo.value


class TestRemovePointsWithError:
    @pytest.mark.unittest
    def test_remove_points_with_error(self):
        # initialise model
        gef_cpt = GefCpt()

        # set inputs
        gef_cpt.depth = np.linspace(-1, 12, 6)
        gef_cpt.friction = np.array([-5, -2, -999, -999, -3, -4])
        gef_cpt.pore_pressure_u2 = np.full(6, 1000)
        gef_cpt.friction_nbr = np.full(6, 5)
        gef_cpt.penetration_length = np.linspace(-1, 12, 6)

        gef_cpt.error_codes["depth"] = -1
        gef_cpt.error_codes["friction"] = -999
        gef_cpt.error_codes["pore_pressure_u2"] = -1
        gef_cpt.error_codes["friction_nbr"] = -1

        # initialise the model
        gef_cpt.remove_points_with_error()
        assert (gef_cpt.friction == np.array([-2, -3, -4])).all()
        assert (gef_cpt.depth == np.array([1.6, 9.4, 12.0])).all()
        assert (gef_cpt.friction_nbr == np.full(3, 5)).all()
        assert (gef_cpt.pore_pressure_u2 == np.full(3, 1000)).all()

    @pytest.mark.unittest
    def test_remove_points_with_error_raises(self):
        # value pwp size is minimized to raise error
        # initialise model
        gef_cpt = GefCpt()
        # set inputs
        gef_cpt.depth = np.linspace(2, 20, 6)
        gef_cpt.friction = np.array([-1, -2, -3, -4, -999, -999])
        gef_cpt.pore_pressure_u2 = np.full(5, 1000)
        gef_cpt.friction_nbr = np.full(6, 5)

        gef_cpt.error_codes["depth"] = -1
        gef_cpt.error_codes["friction"] = -999
        gef_cpt.error_codes["pore_pressure_u2"] = -1
        gef_cpt.error_codes["friction_nbr"] = -1

        # Run test
        with pytest.raises(Exception) as excinfo:
            gef_cpt.remove_points_with_error()
        assert (
            "The data 'pore_pressure_u2' (length = 5) is not of the assumed data length = 6"
            == str(excinfo.value)
        )
