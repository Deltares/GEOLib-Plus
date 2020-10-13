import pytest
<<<<<<< HEAD:tests/GEF_CPT/test_gef_utils.py
from geolib_plus.GEF_CPT import gef_utils
import numpy as np
import re

=======
from geolib_plus.gef_cpt import gef_utils
>>>>>>> fb3e7dcdf4646e315885acbeddb67bfd1c133131:tests/gef_cpt/test_gef_utils.py


# todo JN: write unit tests
class TestGefUtil:
    @pytest.mark.unittest
    @pytest.mark.workinprogress
    def test_gef_util_unit_tests(self):
        raise NotImplementedError

    @pytest.mark.unittest
    def test_correct_negatives_and_zeros(self):
        # define keys that cannot be zero
        list_non_zero = ["first"]
        dictionary = {"first": [-1, -2, -9], "second": [-1, -2, -9]}
        # run the test
        dictionary = gef_utils.correct_negatives_and_zeros(
            result_dictionary=dictionary, correct_for_negatives=list_non_zero
        )
        # check the output
        assert (dictionary["first"] == np.array([0, 0, 0])).all()
        assert dictionary["second"] == [-1, -2, -9]

    @pytest.mark.unittest
    def test_read_data_no_pore_pressure(self):
        index_dictionary = {
            "depth": 0,
            "friction": 2,
            "friction_nb": 5,
            "pwp": None,
            "tip": 1,
        }

        dictionary_multiplication_factors = {
            "depth": 1.0,
            "tip": 1000.0,
            "friction": 1000.0,
            "friction_nb": 1.0,
            "pwp": 1000.0,
        }

        # read gef file
        gef_file = ".\\tests\\test_files\\cpt\\gef\\unit_testing\\test_read_data.gef"
        with open(gef_file, "r") as f:
            data = f.readlines()
        idx_EOH = [i for i, val in enumerate(data) if val.startswith(r"#EOH=")][0]
        data[idx_EOH + 1 :] = [
            re.sub("[ :,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1 :]
        ]
        # Run test
        result_dictionary = gef_utils.read_data(
            index_dictionary, data, idx_EOH, dictionary_multiplication_factors
        )
        # Check output
        assert result_dictionary["depth"][-1] == 25.61
        assert result_dictionary["tip"][-1] == 13387.0
        assert result_dictionary["friction"][-1] == -99999000.0
        assert result_dictionary["pwp"][-1] == 0.0

    @pytest.mark.unittest
    def test_read_data_error_raised(self):
        # depth input was not find in the cpt file
        index_dictionary = {
            "depth": None,
            "friction": 2,
            "friction_nb": 5,
            "pwp": None,
            "tip": 1,
        }

        dictionary_multiplication_factors = {
            "depth": 1.0,
            "tip": 1000.0,
            "friction": 1000.0,
            "friction_nb": 1.0,
            "pwp": 1000.0,
        }

        # read gef file
        gef_file = ".\\tests\\test_files\\cpt\\gef\\unit_testing\\test_read_data.gef"
        with open(gef_file, "r") as f:
            data = f.readlines()
        idx_EOH = [i for i, val in enumerate(data) if val.startswith(r"#EOH=")][0]
        data[idx_EOH + 1 :] = [
            re.sub("[ :,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1 :]
        ]
        # Run test

        with pytest.raises(Exception) as excinfo:
            gef_utils.read_data(
                index_dictionary, data, idx_EOH, dictionary_multiplication_factors
            )
        assert "CPT key: depth not part of GEF file" == str(excinfo.value)

    @pytest.mark.unittest
    def test_remove_points_with_error(self):
        # define input dictionary
        dictionary_input = {
            "depth": np.linspace(2, 20, 6),
            "friction": np.array([-1, -2, -999, -999, -3, -4]),
            "friction_nb": np.full(6, 5),
            "pwp": np.full(6, 1000),
        }
        error_values = {
            "depth": -1,
            "friction": -999,
            "friction_nb": -1,
            "pwp": -1,
        }
        dictionary_output = gef_utils.remove_points_with_error(
            result_dictionary=dictionary_input, index_error=error_values
        )
        assert (dictionary_output["friction"] == np.array([-1, -2, -3, -4])).all()
        assert (dictionary_output["depth"] == np.array([2.0, 5.6, 16.4, 20.0])).all()
        assert (dictionary_output["friction_nb"] == np.full(4, 5)).all()
        assert (dictionary_output["pwp"] == np.full(4, 1000)).all()

    @pytest.mark.unittest
    def test_remove_points_with_error_raises(self):
        # define input dictionary
        # value pwp size is minimized to raise error
        dictionary_input = {
            "depth": np.linspace(2, 20, 6),
            "friction": np.array([-1, -2, -3, -4, -999, -999]),
            "friction_nb": np.full(6, 5),
            "pwp": np.full(5, 1000),
        }
        error_values = {
            "depth": -1,
            "friction": -999,
            "friction_nb": -1,
            "pwp": -1,
        }
        # Run test
        with pytest.raises(Exception) as excinfo:
            gef_utils.remove_points_with_error(
                result_dictionary=dictionary_input, index_error=error_values
            )
        assert "Index <4> excides the length of list of key 'pwp'" == str(excinfo.value)

    @pytest.mark.unittest
    def test_read_column_index_for_gef_data(self):
        # define all inputs
        doc_snippet = [
            "#COLUMN= 10",
            "#COLUMNINFO= 1, m, Sondeerlengte, 1",
            "#COLUMNINFO= 2, MPa, Conusweerstand qc, 2",
            "#COLUMNINFO= 3, MPa, Wrijvingsweerstand fs, 3",
            "#COLUMNINFO= 4, %, Wrijvingsgetal Rf, 4",
            "#COLUMNINFO= 5, MPa, Waterspanning u2, 6",
            "#COLUMNINFO= 6, graden, Helling X, 21",
            "#COLUMNINFO= 7, graden, Helling Y, 22",
            "#COLUMNINFO= 8, -, Classificatie zone Robertson 1990, 99",
            "#COLUMNINFO= 9, m, Gecorrigeerde diepte, 11",
            "#COLUMNINFO= 10, s, Tijd, 12",
        ]
        # indexes that match columns in gef file
        indexes = [1, 2, 3, 4, 6, 21, 22, 99, 11, 12]
        # Run the test
        for counter, index in enumerate(indexes):
            assert counter == gef_utils.read_column_index_for_gef_data(
                key_cpt=index, data=doc_snippet
            )

    @pytest.mark.unittest
    def test_read_column_index_for_gef_data_error(self):
        # define all inputs
        doc_snippet = [
            "#COLUMN= 10",
            "#COLUMNINFO= 1, m, Sondeerlengte, 1",
            "#COLUMNINFO= 2, MPa, Conusweerstand qc, 2",
            "#COLUMNINFO= 3, MPa, Wrijvingsweerstand fs, 3",
            "#COLUMNINFO= 4, %, Wrijvingsgetal Rf, 4",
            "#COLUMNINFO= 5, MPa, Waterspanning u2, 6",
            "#COLUMNINFO= 6, graden, Helling X, 21",
            "#COLUMNINFO= 7, graden, Helling Y, 22",
            "#COLUMNINFO= 8, -, Classificatie zone Robertson 1990, 99",
            "#COLUMNINFO= 9, m, Gecorrigeerde diepte, 11",
            "#COLUMNINFO= 10, s, Tijd, 12",
        ]
        # indexes don't match the columns in gef file
        index = 5
        # Run the test
        assert not (
            gef_utils.read_column_index_for_gef_data(key_cpt=index, data=doc_snippet)
        )

    @pytest.mark.unittest
    def test_match_idx_with_error(self):
        # Set the inputs
        error_string_list = [
            "-1",
            "-2",
            "-3",
            "string",
            "-5",
            "-6",
            "-7",
            "-8",
            "-9",
            "-10",
        ]
        index_dictionary = {
            "depth": 0,
            "tip": 1,
            "friction": 2,
            "friction_nb": 3,
            "pwp": 4,
        }
        dictionary_multiplication_factors = {
            "depth": 1.0,
            "tip": 1000.0,
            "friction": 1000.0,
            "friction_nb": 1.0,
            "pwp": 1000.0,
        }
        # Run test
        idx_errors_dict = gef_utils.match_idx_with_error(
            error_string_list, index_dictionary, dictionary_multiplication_factors
        )
        # Check expectations
        assert (
            idx_errors_dict["depth"] == -1 * dictionary_multiplication_factors["depth"]
        )
        assert idx_errors_dict["tip"] == -2 * dictionary_multiplication_factors["tip"]
        assert (
            idx_errors_dict["friction"]
            == -3 * dictionary_multiplication_factors["friction"]
        )
        assert idx_errors_dict["friction_nb"] == "string"
        assert idx_errors_dict["pwp"] == -5 * dictionary_multiplication_factors["pwp"]

    @pytest.mark.unittest
    def test_match_idx_with_error_raises(self):
        # Set the inputs. One value is missing from the list
        error_string_list = [
            "-1",
            "-2",
            "-3",
            "string",
        ]
        index_dictionary = {
            "depth": 0,
            "tip": 1,
            "friction": 2,
            "friction_nb": 3,
            "pwp": 4,
        }
        dictionary_multiplication_factors = {
            "depth": 1.0,
            "tip": 1000.0,
            "friction": 1000.0,
            "friction_nb": 1.0,
            "pwp": 1000.0,
        }
        # Run test
        with pytest.raises(Exception) as excinfo:
            gef_utils.match_idx_with_error(
                error_string_list, index_dictionary, dictionary_multiplication_factors
            )
        assert "Key pwp not found in GEF file" in str(excinfo.value)

    @pytest.mark.intergration
    def test_read_gef_1(self):
        gef_file = "./tests/test_files/cpt/gef/unit_testing/unit_testing.gef"

        gef_id = ("unit_testing.gef",)

        key_cpt = {"depth": 1, "tip": 2, "friction": 3, "friction_nb": 4, "pwp": 6}

        cpt = gef_utils.read_gef(gef_file=gef_file, id=gef_id, key_cpt=key_cpt)
        test_coord = [244319.00, 587520.00]

        test_depth = np.linspace(1, 20, 20)
        test_NAP = -1 * test_depth + 0.13
        test_tip = np.full(20, 1000)
        test_friction = np.full(20, 2000)
        test_friction_nbr = np.full(20, 5)
        test_water = np.full(20, 3000)

        assert gef_id == cpt["name"]
        assert test_coord == cpt["coordinates"]
        assert (test_depth == cpt["depth"]).all()
        assert (test_NAP == cpt["depth_to_reference"]).all()
        assert (test_tip == cpt["tip"]).all()
        assert (test_friction == cpt["friction"]).all()
        assert (test_friction_nbr == cpt["friction_nbr"]).all()
        assert (test_water == cpt["water"]).all()

    @pytest.mark.intergration
    @pytest.mark.parametrize(
        "filename, error",
        [
            pytest.param(
                "./tests/test_files/cpt/gef/unit_testing/Exception_NoLength.gef",
                "CPT key: depth not part of GEF file",
                id="no depth",
            ),
            pytest.param(
                "./tests/test_files/cpt/gef/unit_testing/Exception_NoTip.gef",
                "CPT key: tip not part of GEF file",
                id="no tip",
            ),
            pytest.param(
                "./tests/test_files/cpt/gef/unit_testing/Exception_NoFriction.gef",
                "CPT key: friction not part of GEF file",
                id="no friction",
            ),
            pytest.param(
                "./tests/test_files/cpt/gef/unit_testing/Exception_NoFrictionNumber.gef",
                "CPT key: friction not part of GEF file",
                id="no num",
            ),
        ],
    )
    def test_read_gef_2(self, filename: str, error: str):

        key_cpt = {"depth": 1, "tip": 2, "friction": 3, "friction_nb": 4, "pwp": 6}

        gef_id = "unit_testing.gef"

        # test exceptions
        cpt = gef_utils.read_gef(gef_file=filename, id=gef_id, key_cpt=key_cpt)

        # Warning was logged no cpt is returned
        assert cpt == error

    @pytest.mark.intergration
    def test_read_gef_3(self):

        key_cpt = {"depth": 1, "tip": 2, "friction": 3, "friction_nb": 4, "pwp": 6}

        gef_id = "unit_testing.gef"
        filename = "./tests/test_files/cpt/gef/unit_testing/Exception_9999.gef"

        cpt = gef_utils.read_gef(gef_file=filename, id=gef_id, key_cpt=key_cpt)

        # define tests
        test_coord = [244319.00, 587520.00]
        test_depth = np.linspace(2, 20, 19)
        test_NAP = -1 * test_depth + 0.13
        test_tip = np.full(19, 1000)
        test_friction = np.full(19, 2000)
        test_friction_nbr = np.full(19, 5)
        test_water = np.full(19, 3000)

        # test expectations
        assert gef_id == cpt["name"]
        assert test_coord == cpt["coordinates"]
        assert (test_depth == cpt["depth"]).all()
        assert (test_NAP == cpt["depth_to_reference"]).all()
        assert (test_tip == cpt["tip"]).all()
        assert (test_friction == cpt["friction"]).all()
        assert (test_friction_nbr == cpt["friction_nbr"]).all()
        assert (test_water == cpt["water"]).all()

    @pytest.mark.unittest
    def test_read_data(self):
        index_dictionary = {
            "depth": 0,
            "friction": 2,
            "friction_nb": 5,
            "pwp": 3,
            "tip": 1,
        }

        dictionary_multiplication_factors = {
            "depth": 1.0,
            "tip": 1000.0,
            "friction": 1000.0,
            "friction_nb": 1.0,
            "pwp": 1000.0,
        }

        # read gef file
        gef_file = ".\\tests\\test_files\\cpt\\gef\\unit_testing\\test_read_data.gef"
        with open(gef_file, "r") as f:
            data = f.readlines()
        idx_EOH = [i for i, val in enumerate(data) if val.startswith(r"#EOH=")][0]
        data[idx_EOH + 1 :] = [
            re.sub("[ :,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1 :]
        ]
        # Run test
        result_dictionary = gef_utils.read_data(
            index_dictionary, data, idx_EOH, dictionary_multiplication_factors
        )
        # Check output
        assert result_dictionary["depth"][-1] == 25.61
        assert result_dictionary["tip"][-1] == 13387.0
        assert result_dictionary["friction"][-1] == -99999000.0
        assert result_dictionary["pwp"][-1] == -99999000.0
