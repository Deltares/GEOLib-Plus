import pytest
from typing import List, Dict
import numpy as np
import re
from geolib_plus.gef_cpt.gef_file_reader import GefFileReader, GefProperty
import logging
from tests.utils import TestUtils


class TestGefFileReaderInit:
    @pytest.mark.unittest
    def test_when_init_default_properties_are_set(self):
        # Define expected property_dict
        expected_dict = {
            "depth": GefProperty(gef_key=1, multiplication_factor=1.0),
            "tip": GefProperty(gef_key=2, multiplication_factor=1000.0),
            "friction": GefProperty(gef_key=3, multiplication_factor=1000.0),
            "friction_nb": GefProperty(gef_key=4, multiplication_factor=1.0),
            "pwp": GefProperty(gef_key=6, multiplication_factor=1000.0),
        }
        file_reader = GefFileReader()
        assert file_reader.name == ""
        assert file_reader.coord == []
        assert isinstance(file_reader.property_dict, dict)
        assert file_reader.__eq__(expected_dict)


class TestGetLineIndexFromDataStartsWith:

    test_cases_raise_exception = [
        pytest.param(None, None, pytest.raises(ValueError), id="None arguments"),
        pytest.param(
            None, [], pytest.raises(ValueError), id="None code_string, Empty data."
        ),
        pytest.param(
            None,
            ["alpha"],
            pytest.raises(ValueError),
            id="None code_string, Valid data.",
        ),
        pytest.param(
            "alpha", None, pytest.raises(ValueError), id="Valid code_string, None data."
        ),
        pytest.param(
            "alpha", [], pytest.raises(ValueError), id="Valid code_string, Empty data"
        ),
        pytest.param(
            "alpha",
            ["beta"],
            pytest.raises(ValueError),
            id="Valid arguments, Value not found.",
        ),
    ]

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "code_string, data, expectation",
        test_cases_raise_exception,
    )
    def test_when_data_starts_given_test_case_then_raises_exception(
        self, code_string: str, data: List[str], expectation
    ):
        with expectation:
            GefFileReader.get_line_index_from_data_starts_with(code_string, data)

    def test_when_data_starts_given_valid_arguments_then_returns_expected_line(
        self,
    ):
        # 1. Define test data
        code_string = "beta"
        data = ["24 42 beta", "42 24 alpha", "alpha 42 24", "beta 24 42"]
        expected_result = 3

        # 2. Run test
        result = GefFileReader.get_line_index_from_data_starts_with(code_string, data)

        # 3. Validate final expectation
        assert result == expected_result

    @pytest.mark.systemtest
    def test_given_real_data_returns_expected_result(self):
        # set inputs
        data = [
            "#SPECIMENVAR=  1 ,   0.00, m, ",
            "#TESTID= DKMP1_1317-0162-000",
            "#REPORTCODE= GEF-CPT-Report,1,1,0",
            "#STARTDATE= 2017,07,03",
            "#STARTTIME= 14,13,53",
            "#OS= DOS",
        ]
        code_string = r"#STARTDATE="
        # run test
        test_id = GefFileReader.get_line_index_from_data_starts_with(
            code_string=code_string, data=data
        )
        assert test_id == 3

    @pytest.mark.unittest
    def test_given_real_data_when_index_not_found_raises_error(self):
        # set inputs
        data = [
            "#SPECIMENVAR=  1 ,   0.00, m, ",
            "#TESTID= DKMP1_1317-0162-000",
            "#REPORTCODE= GEF-CPT-Report,1,1,0",
            "#STARTDATE= 2017,07,03",
            "#STARTTIME= 14,13,53",
            "#OS= DOS",
        ]
        code_string = r"#IAMNOTINTHEFILE="
        # Run test
        with pytest.raises(ValueError) as excinfo:
            GefFileReader.get_line_index_from_data_starts_with(
                code_string=code_string, data=data
            )
        assert "No values found for field #IAMNOTINTHEFILE= of the gef file." in str(
            excinfo.value
        )


class TestGetLineFromDataEndsWith:

    test_cases_raise_exception = [
        pytest.param(None, None, pytest.raises(ValueError), id="None arguments"),
        pytest.param(
            None, [], pytest.raises(ValueError), id="None code_string, Empty data."
        ),
        pytest.param(
            None,
            ["alpha"],
            pytest.raises(ValueError),
            id="None code_string, Valid data.",
        ),
        pytest.param(
            "alpha", None, pytest.raises(ValueError), id="Valid code_string, None data."
        ),
        pytest.param(
            "alpha", [], pytest.raises(ValueError), id="Valid code_string, Empty data"
        ),
    ]

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "code_string, data, expectation",
        test_cases_raise_exception,
    )
    def test_when_data_ends_given_test_case_arguments_then_raises_exception(
        self, code_string: str, data: List[str], expectation
    ):
        with expectation:
            GefFileReader.get_line_from_data_that_ends_with(code_string, data)

    def test_when_data_ends_given_valid_arguments_then_returns_expected_line(self):
        # 1. Define test data
        code_string = "beta"
        data = ["24 42 beta", "42 24 alpha", "alpha 42 24", "beta 24 42"]
        expected_result = data[0]

        # 2. Run test
        result = GefFileReader.get_line_from_data_that_ends_with(code_string, data)

        # 3. Validate final expectation
        assert result == expected_result


class TestReadColumnData:
    @pytest.mark.systemtest
    def test_read_column_data_no_pore_pressure(self):
        # initialise model
        gef_reader = GefFileReader()
        gef_reader.property_dict["penetration_length"].gef_column_index = 0
        gef_reader.property_dict["friction"].gef_column_index = 2
        gef_reader.property_dict["tip"].gef_column_index = 1
        gef_reader.property_dict["pwp_u2"].gef_column_index = None
        gef_reader.property_dict["friction_nb"].gef_column_index = 5
        # read gef file
        gef_file = (
            TestUtils.get_local_test_data_dir("cpt/gef/unit_testing")
            / "test_read_column_data.gef"
        )
        assert gef_file.is_file()
        with open(gef_file, "r") as f:
            data = f.readlines()
        idx_EOH = [i for i, val in enumerate(data) if val.startswith(r"#EOH=")][0]
        data[idx_EOH + 1 :] = [
            re.sub("[ :,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1 :]
        ]
        # Run test
        gef_reader.read_column_data(data, idx_EOH)
        # Check output
        assert (
            gef_reader.property_dict["penetration_length"].values_from_gef[-1] == 25.61
        )
        assert gef_reader.property_dict["tip"].values_from_gef[-1] == 13.387000
        assert gef_reader.property_dict["friction"].values_from_gef[-1] == -99999.0
        assert gef_reader.property_dict["pwp_u2"].values_from_gef is None

    @pytest.mark.systemtest
    def test_read_column_data_error_raised(self):
        # depth input was not find in the cpt file
        # initialise model
        gef_reader = GefFileReader()
        gef_reader.property_dict["penetration_length"].gef_column_index = None
        gef_reader.property_dict["friction"].gef_column_index = 2
        gef_reader.property_dict["tip"].gef_column_index = 1
        gef_reader.property_dict["pwp_u2"].gef_column_index = None
        gef_reader.property_dict["friction_nb"].gef_column_index = 5
        # read gef file
        gef_file = (
            TestUtils.get_local_test_data_dir("cpt/gef/unit_testing")
            / "test_read_column_data.gef"
        )
        assert gef_file.is_file()
        with open(gef_file, "r") as f:
            data = f.readlines()
        idx_EOH = [i for i, val in enumerate(data) if val.startswith(r"#EOH=")][0]
        data[idx_EOH + 1 :] = [
            re.sub("[ :,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1 :]
        ]
        # Run test

        with pytest.raises(Exception) as excinfo:
            gef_reader.read_column_data(data, idx_EOH)
        assert "CPT key: penetration_length not part of GEF file" == str(excinfo.value)

    @pytest.mark.unittest
    def test_read_column_data(self):

        # initialise model
        gef_reader = GefFileReader()
        # set inputs
        gef_reader.property_dict["penetration_length"].multiplication_factor = 1
        gef_reader.property_dict["friction"].multiplication_factor = 1
        gef_reader.property_dict["pwp_u2"].multiplication_factor = 1
        gef_reader.property_dict["friction_nb"].multiplication_factor = 1

        gef_reader.property_dict["penetration_length"].gef_column_index = 0
        gef_reader.property_dict["friction"].gef_column_index = 2
        gef_reader.property_dict["tip"].gef_column_index = 1
        gef_reader.property_dict["pwp_u2"].gef_column_index = 3
        gef_reader.property_dict["friction_nb"].gef_column_index = 5

        # read gef file
        gef_file = (
            TestUtils.get_local_test_data_dir("cpt/gef/unit_testing")
            / "test_read_column_data.gef"
        )
        with open(gef_file, "r") as f:
            data = f.readlines()
        idx_EOH = [i for i, val in enumerate(data) if val.startswith(r"#EOH=")][0]
        data[idx_EOH + 1 :] = [
            re.sub("[ :,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1 :]
        ]

        # Run test
        gef_reader.read_column_data(data, idx_EOH)
        # Check output
        assert (
            gef_reader.property_dict["penetration_length"].values_from_gef[-1] == 25.61
        )
        assert gef_reader.property_dict["tip"].values_from_gef[-1] == 13.387
        assert gef_reader.property_dict["friction"].values_from_gef[-1] == -99999.0
        assert gef_reader.property_dict["pwp_u2"].values_from_gef[-1] == -99999.0

class TestReadColumnIndexForGefData:
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
        # initialise the model
        gef_reader = GefFileReader()
        # Run the test
        for counter, index in enumerate(indexes):
            assert counter == gef_reader.read_column_index_for_gef_data(
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
        # initialise the model
        gef_reader = GefFileReader()
        # Run the test
        assert not (
            gef_reader.read_column_index_for_gef_data(key_cpt=index, data=doc_snippet)
        )


class TestMatchIdxWithError:
    @pytest.mark.unittest
    def test_match_idx_with_error(self):
        # Set the inputs
        error_string_list = [
            "-1",
            "-2",
            "-3",
            "string",
            "-5",
            "string",
            "-7",
            "-8",
            "-9",
            "-10",
        ]
        # initialise model
        gef_reader = GefFileReader()
        # set inputs
        gef_reader.property_dict["penetration_length"].gef_key = 0
        gef_reader.property_dict["tip"].gef_key = 1
        gef_reader.property_dict["friction"].gef_key = 2
        gef_reader.property_dict["pwp_u2"].gef_key = 4
        gef_reader.property_dict["friction_nb"].gef_key = 3

        gef_reader.property_dict["penetration_length"].multiplication_factor = 1
        gef_reader.property_dict["friction"].multiplication_factor = 1
        gef_reader.property_dict["pwp_u2"].multiplication_factor = 1
        gef_reader.property_dict["friction_nb"].multiplication_factor = 1

        gef_reader.property_dict["penetration_length"].gef_column_index = 4
        gef_reader.property_dict["friction"].gef_column_index = 2
        gef_reader.property_dict["tip"].gef_column_index = 1
        gef_reader.property_dict["pwp_u2"].gef_column_index = 3
        gef_reader.property_dict["friction_nb"].gef_column_index = 5

        # Run test
        gef_reader.match_idx_with_error(error_string_list)
        # Check expectations
        assert gef_reader.property_dict["penetration_length"].error_code == -5
        assert gef_reader.property_dict["tip"].error_code == -2
        assert gef_reader.property_dict["friction"].error_code == -3
        assert gef_reader.property_dict["friction_nb"].error_code == "string"
        assert gef_reader.property_dict["pwp_u2"].error_code == "string"

    @pytest.mark.unittest
    def test_match_idx_with_error_raises_1(self):
        # Set the inputs. One value is missing from the list
        error_string_list = ["-1", "-2", "-3", "string", "-4"]

        # initialise model
        gef_reader = GefFileReader()
        gef_reader.property_dict["penetration_length"].multiplication_factor = 1
        gef_reader.property_dict["friction"].multiplication_factor = 1000
        gef_reader.property_dict["pwp_u2"].multiplication_factor = 1000
        gef_reader.property_dict["friction_nb"].multiplication_factor = 1000

        gef_reader.property_dict["penetration_length"].gef_column_index = 4
        gef_reader.property_dict["friction"].gef_column_index = 2
        gef_reader.property_dict["tip"].gef_column_index = None
        gef_reader.property_dict["pwp_u2"].gef_column_index = 3
        gef_reader.property_dict["friction_nb"].gef_column_index = 5
        # Run test
        with pytest.raises(Exception) as excinfo:
            gef_reader.match_idx_with_error(error_string_list)
        assert "Key tip should be defined in the gef file" in str(excinfo.value)


class TestReadGef:
    @pytest.mark.integration
    def test_read_gef_1(self):
        # todo move calculation of depth_to_reference outside reader
        gef_file = TestUtils.get_local_test_data_dir(
            "cpt/gef/unit_testing/unit_testing.gef"
        )

        # initialise the model
        gef_reader = GefFileReader()
        # run the test
        cpt = gef_reader.read_gef(gef_file=gef_file)
        test_coord = [244319.00, 587520.00]

        test_depth = np.linspace(1, 20, 20)
        test_NAP = -1 * test_depth + 0.13
        test_tip = np.full(20, 1)
        test_friction = np.full(20, 2)
        test_friction_nbr = np.full(20, 5)
        test_water = np.full(20, 3)

        assert "DKP302" == cpt["name"]
        assert test_coord == cpt["coordinates"]
        # assert (test_depth == cpt["penetration_length"]).all()
        # assert (test_NAP == cpt["depth_to_reference"]).all()
        assert (test_tip == cpt["tip"]).all()
        assert (test_friction == cpt["friction"]).all()
        assert (test_friction_nbr == cpt["friction_nbr"]).all()
        assert (test_water == cpt["pore_pressure_u2"]).all()

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "filename, error",
        [
            pytest.param(
                "cpt/gef/unit_testing/Exception_NoLength.gef",
                "Key penetration_length should be defined in the gef file.",
                id="no penetration_length",
            ),
            pytest.param(
                "cpt/gef/unit_testing/Exception_NoTip.gef",
                "Key tip should be defined in the gef file.",
                id="no tip",
            ),
        ],
    )
    def test_read_gef_missing_field_error(self, filename: str, error: str):
        # initialise the model
        gef_reader = GefFileReader()

        filename = TestUtils.get_local_test_data_dir(filename)

        # test exceptions
        with pytest.raises(Exception) as excinfo:
            gef_reader.read_gef(gef_file=filename)
        assert error == str(excinfo.value)

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "filename, warning",
        [
            pytest.param(
                "cpt/gef/unit_testing/Exception_NoFriction.gef",
                "Key friction is not defined in the gef file",
                id="no friction",
            ),
            pytest.param(
                "cpt/gef/unit_testing/Exception_NoFrictionNumber.gef",
                "Key friction_nb is not defined in the gef file",
                id="friction_nb",
            ),
        ],
    )
    def test_read_gef_missing_field_warning(self, filename: str, warning: str, caplog):
        LOGGER = logging.getLogger(__name__)
        # define logger
        LOGGER.info("Testing now.")
        # initialise the model
        gef_reader = GefFileReader()
        # test exceptions
        filename = TestUtils.get_local_test_data_dir(filename)
        result_dictionary = gef_reader.read_gef(gef_file=filename)
        assert warning in caplog.text

    @pytest.mark.workinprogress
    @pytest.mark.integration
    def test_read_gef_3(self):
        # todo move calculation of depth_to_reference outside reader

        filename = TestUtils.get_local_test_data_dir(
            "cpt/gef/unit_testing/Exception_9999.gef"
        )
        # initialise the model
        gef_reader = GefFileReader()
        # run the test
        cpt = gef_reader.read_gef(gef_file=filename)

        # define tests
        test_coord = [244319.00, 587520.00]
        test_depth = np.linspace(1, 20, 20)
        test_depth[0] = -9999
        test_NAP = -1 * test_depth + 0.13
        test_tip = np.full(20, 1)
        test_tip[0] = -9999
        test_friction = np.full(20, 2)
        test_friction[0] = -9999
        test_friction_nbr = np.full(20, 5)
        test_friction_nbr[0] = -9999
        test_water = np.full(20, 3)
        test_water[0] = -9999
        # test expectations
        assert "DKP302" == cpt["name"]
        assert test_coord == cpt["coordinates"]
        assert (test_depth == cpt["penetration_length"]).all()
        # assert (test_NAP == cpt["depth_to_reference"]).all()
        assert (test_tip == cpt["tip"]).all()
        assert (test_friction == cpt["friction"]).all()
        assert (test_friction_nbr == cpt["friction_nbr"]).all()
        assert (test_water == cpt["pore_pressure_u2"]).all()


class TestReadInformationForGefData:
    @pytest.mark.unittest
    def test_read_information_for_gef_data(self):
        # set input
        data = [
            "#MEASUREMENTTEXT= 1, -, opdrachtgever",
            "#MEASUREMENTTEXT= 2, GRONDONDERZOEK T.B.V. NIEUW GEMAAL MONNICKENDAM WATERSTAND = NAP -0.47 m, projectnaam",
            "#MEASUREMENTTEXT= 3, Monnikendam, plaatsnaam",
            "#MEASUREMENTTEXT= 4, CP15-CF75PB1SN2/1701-1524, conus type/serienummer",
            "#MEASUREMENTTEXT= 5, RUPS3/PJW/JBK, sondeerapparaat/operator",
            "#MEASUREMENTTEXT= 6, ISO 22476-1 Toepassingsklasse 2, gevolgde norm",
            "#MEASUREMENTTEXT= 8, NAP",
            "#MEASUREMENTTEXT= 9, maaiveld, vast horizontaal vlak",
        ]

        # execute test
        gef_reader = GefFileReader()
        for key_name in gef_reader.information_dict:
            gef_reader.information_dict[
                key_name
            ].values_from_gef = gef_reader.read_information_for_gef_data(key_name, data)

        # assert
        assert (
            gef_reader.information_dict["cpt_type"].values_from_gef
            == "CP15-CF75PB1SN2/1701-1524, conus type/serienummer"
        )
        assert (
            gef_reader.information_dict["cpt_standard"].values_from_gef
            == "ISO 22476-1 Toepassingsklasse 2, gevolgde norm"
        )
        assert gef_reader.information_dict["vertical_datum"].values_from_gef == "NAP"
        assert (
            gef_reader.information_dict["local_reference"].values_from_gef
            == "maaiveld, vast horizontaal vlak"
        )

    @pytest.mark.unittest
    def test_read_information_for_empty_gef_data(self):
        # set input
        data = []

        # execute test
        gef_reader = GefFileReader()
        for key_name in gef_reader.information_dict:
            gef_reader.information_dict[
                key_name
            ].values_from_gef = gef_reader.read_information_for_gef_data(key_name, data)

        # assert
        assert gef_reader.information_dict["cpt_type"].values_from_gef == ""
        assert gef_reader.information_dict["cpt_standard"].values_from_gef == ""
        assert gef_reader.information_dict["vertical_datum"].values_from_gef == ""
        assert gef_reader.information_dict["local_reference"].values_from_gef == ""

    @pytest.mark.unittest
    def test_read_information_for_different_gef_data(self):
        # set input
        data = ["test", "test", "#EOH="]

        # execute test
        gef_reader = GefFileReader()
        for key_name in gef_reader.information_dict:
            gef_reader.information_dict[
                key_name
            ].values_from_gef = gef_reader.read_information_for_gef_data(key_name, data)

        # assert
        assert gef_reader.information_dict["cpt_type"].values_from_gef == ""
        assert gef_reader.information_dict["cpt_standard"].values_from_gef == ""
        assert gef_reader.information_dict["vertical_datum"].values_from_gef == ""
        assert gef_reader.information_dict["local_reference"].values_from_gef == ""
