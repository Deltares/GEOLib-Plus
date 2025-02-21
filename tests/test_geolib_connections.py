from pathlib import Path

import geolib
import pytest
from geolib.models.dfoundations import piles, profiles

from geolib_plus.gef_cpt import GefCpt
from geolib_plus.geolib_connections import DFoundationsConnector
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from tests.utils import TestUtils


class TestLayers:
    @pytest.mark.unittest
    def test_get_pile_tip_level_error(self):
        # initialize geolib plus
        cpt_gef = GefCpt()
        # check initial expectation
        assert cpt_gef
        # check that tip property is not initialized in the cpt
        assert cpt_gef.tip is None
        with pytest.raises(ValueError) as excinfo:
            DFoundationsConnector()._DFoundationsConnector__get_pile_tip_level(cpt_gef)
        assert "Tip is not defined in the cpt." in str(excinfo.value)

    @pytest.mark.unittest
    def test_get_pile_tip_level(self):
        # define test gef
        test_file_gef = Path(
            TestUtils.get_local_test_data_dir("cpt/gef"), "cpt_with_water.gef"
        )
        # initialize geolib plus
        cpt_gef = GefCpt()
        # read the cpt for each type of file
        cpt_gef.read(test_file_gef)
        # pre-process cpt data
        cpt_gef.pre_process_data()
        # interpret data
        interpreter = RobertsonCptInterpretation()
        cpt_gef.interpret_cpt(interpreter)
        # check initial expectation
        assert cpt_gef
        assert (
            abs(
                -13.9646
                - DFoundationsConnector()._DFoundationsConnector__get_pile_tip_level(
                    cpt_gef
                )
            )
            < 0.000051
        )

    @pytest.mark.unittest
    def test_get_phreatic_level(self):
        # define test gef
        test_file_gef = Path(
            TestUtils.get_local_test_data_dir("cpt/gef"), "cpt_with_water.gef"
        )
        # initialize geolib plus
        cpt_gef = GefCpt()
        # read the cpt for each type of file
        cpt_gef.read(test_file_gef)
        # pre-process cpt data
        cpt_gef.pre_process_data()
        # interpret data
        interpreter = RobertsonCptInterpretation()
        cpt_gef.interpret_cpt(interpreter)
        # check initial expectation
        assert cpt_gef
        assert (
            abs(
                -2.6
                - DFoundationsConnector()._DFoundationsConnector__get_phreatic_level(
                    cpt_gef
                )
            )
            < 0.000051
        )

    @pytest.mark.unittest
    def test_get_phreatic_level_no_water_defined(self):
        # define test gef
        test_file_gef = Path(
            TestUtils.get_local_test_data_dir("cpt/gef"), "cpt_with_water.gef"
        )
        # initialize geolib plus
        cpt_gef = GefCpt()
        # read the cpt for each type of file
        cpt_gef.read(test_file_gef)
        # pre-process cpt data
        cpt_gef.pre_process_data()
        # interpret data
        interpreter = RobertsonCptInterpretation()
        cpt_gef.interpret_cpt(interpreter)
        cpt_gef.water = None
        # check initial expectation
        assert cpt_gef
        assert (
            abs(
                cpt_gef.depth_to_reference[0]
                - 0.5
                - DFoundationsConnector()._DFoundationsConnector__get_phreatic_level(
                    cpt_gef
                )
            )
            < 0.000051
        )

    @pytest.mark.integrationtest
    def test_if_not_none_add_to_dict(self):
        # initialize inputs
        list_to_be_added = [1, 2, 3, 4]
        name_to_be_added = "name_to_be_added"
        not_to_be_added = None
        name_not_to_be_added = "name_not_to_be_added"
        dictionary = {}
        # run test that should fail
        dictionary = (
            DFoundationsConnector()._DFoundationsConnector__if_not_none_add_to_dict(
                dictionary, not_to_be_added, name_not_to_be_added
            )
        )
        assert dictionary == {}
        assert not (dictionary.get(name_not_to_be_added, False))
        # run test that should not fail
        dictionary = (
            DFoundationsConnector()._DFoundationsConnector__if_not_none_add_to_dict(
                dictionary, list_to_be_added, name_to_be_added
            )
        )
        assert not (dictionary == {})
        assert dictionary.get(name_to_be_added, False) == list_to_be_added

    @pytest.mark.integrationtest
    def test_define_cpt_inputs(self):
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

        cpt_dfoundations = (
            DFoundationsConnector()._DFoundationsConnector__define_cpt_inputs(cpt_gef)
        )

        assert cpt_dfoundations.cptname == "CPT000000029380"

    @pytest.mark.integrationtest
    def test_to_layers_for_d_foundations_error(self):
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
        with pytest.raises(ValueError) as excinfo:
            DFoundationsConnector()._DFoundationsConnector__to_layers_for_d_foundations(
                cpt_gef
            )
        assert (
            "Field 'depth_merged' was not defined in the inputted cpt. Interpretation of the cpt must be performed. "
            in str(excinfo.value)
        )

    @pytest.mark.integrationtest
    def test_to_layers_for_d_foundations(self):
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
        soil_layers = (
            DFoundationsConnector()._DFoundationsConnector__to_layers_for_d_foundations(
                cpt_gef
            )
        )
        assert len(soil_layers) == 35


class TestGefCptGeolibPlusToGeolib:
    @pytest.mark.integrationtest
    def test_make_geolib_profile(self):
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
        # run test
        profile, soils = DFoundationsConnector.create_profile_for_d_foundations(cpt_gef)
        assert profile
        assert soils

        # initialize geolib model
        test_dfoundations = Path(
            TestUtils.get_local_test_data_dir("input_files_geolib"), "bm1-1b.foi"
        )
        assert test_dfoundations.is_file()
        dfoundations_model = geolib.models.DFoundationsModel()
        # parse
        dfoundations_model.parse(test_dfoundations)
        # assert initial expectations
        assert dfoundations_model

        # Model and calculation settings need to be set otherwise they will be overwritten
        model_options = geolib.models.dfoundations.dfoundations_model.BearingPilesModel()
        calculation_options = geolib.models.dfoundations.dfoundations_model.CalculationOptions(
            calculationtype=geolib.models.dfoundations.dfoundations_model.CalculationType.VERIFICATION_DESIGN,
            cpt_test_level=-27,
        )
        dfoundations_model.set_model(model_options, calculation_options)
        # pile needs to be added
        # Add Bearing Pile
        location = piles.BearingPileLocation(
            point=geolib.geometry.Point(x=1.0, y=1.0),
            pile_head_level=1,
            surcharge=1,
            limit_state_str=1,
            limit_state_service=1,
        )
        geometry_pile = dict(base_width=1, base_length=1)
        parent_pile = dict(
            pile_name="test",
            pile_type=piles.BasePileType.USER_DEFINED_VIBRATING,
            pile_class_factor_shaft_sand_gravel=1,  # alpha_s
            preset_pile_class_factor_shaft_clay_loam_peat=piles.BasePileTypeForClayLoamPeat.STANDARD,
            pile_class_factor_shaft_clay_loam_peat=1,  # alpha_s
            pile_class_factor_tip=1,  # alpha_p
            load_settlement_curve=piles.LoadSettlementCurve.ONE,
            user_defined_pile_type_as_prefab=False,
            use_manual_reduction_for_qc=False,
            elasticity_modulus=1e7,
            characteristic_adhesion=10,
            overrule_pile_tip_shape_factor=False,
            overrule_pile_tip_cross_section_factors=False,
        )
        pile = piles.BearingRectangularPile(**parent_pile, **geometry_pile)
        dfoundations_model.add_pile_if_unique(pile, location)

        # add soils from cpt
        for soil in soils:
            dfoundations_model.add_soil(soil)
        # add profile to the dfoundations model
        test_name = dfoundations_model.add_profile(profile)

        # test expectations
        assert test_name == "CPT000000029380"
        # save updated file
        dfoundations_model.serialize(
            Path(
                TestUtils.get_output_test_data_dir("DFoundations"),
                "bm1-1a_added_cpt.foi",
            )
        )
