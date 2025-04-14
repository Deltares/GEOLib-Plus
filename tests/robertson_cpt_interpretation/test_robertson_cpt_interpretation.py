import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from geolib_plus.bro_xml_cpt import BroXmlCpt
from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import (
    OCRMethod,
    RelativeDensityMethod,
    RobertsonCptInterpretation,
    ShearWaveVelocityMethod,
    UnitWeightMethod,
)
from tests.utils import TestUtils


class TestShapeFiles:
    @pytest.mark.unittest
    def test_lithology(self):
        robertson = RobertsonCptInterpretation()
        robertson.soil_types()
        coords_test = []
        Qtn = [2, 2, 10, 7, 20, 100, 900, 700, 700]
        Fr = [0.2, 9, 8, 1, 0.2, 0.5, 0.2, 3, 9]
        lithology_test = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

        [coords_test.append([Fr[i], Qtn[i]]) for i in range(len(Fr))]

        litho, coords = robertson.lithology(Qtn, Fr)
        np.testing.assert_array_equal(coords_test, coords)
        np.testing.assert_array_equal(lithology_test, litho)


class Testintegration:
    @pytest.mark.integrationtest
    def test_against_bro_results(self):
        # open the gef file
        gef_test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "gef", "CPT000000003688_IMBRO_A.gef")
        )
        # open the bro-xml file
        bro_test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "bro_xml", "CPT000000003688_IMBRO_A.xml")
        )
        bro_test_file_new_version = TestUtils.get_local_test_data_dir(
            Path("cpt", "bro_xml", "CPT000000003688_IMBRO_A_new_version.xml")
        )
        assert gef_test_file.is_file()
        assert bro_test_file.is_file()
        assert bro_test_file_new_version.is_file()
        # initialise models
        gef_cpt = GefCpt()
        bro_cpt = BroXmlCpt()
        bro_cpt_new_version = BroXmlCpt()
        # test initial expectations
        assert gef_cpt
        assert bro_cpt
        # read gef file
        gef_cpt.read(filepath=gef_test_file)
        bro_cpt.read(filepath=bro_test_file)
        bro_cpt_new_version.read(filepath=bro_test_file_new_version)
        # do pre-processing
        gef_cpt.pre_process_data()
        bro_cpt.pre_process_data()
        bro_cpt_new_version.pre_process_data()
        # initialise interpretation model
        robertson = RobertsonCptInterpretation()
        robertson.unitweightmethod = UnitWeightMethod.ROBERTSON
        # interpet the results
        gef_cpt.interpret_cpt(robertson)
        bro_cpt.interpret_cpt(robertson)
        bro_cpt_new_version.interpret_cpt(robertson)

        values_to_test = [
            "friction_nbr",
            "friction_nbr",
            "qt",
            "qt",
            "rho",
            "rho",
            "tip",
            "tip",
            "depth",
            "depth",
            "depth_to_reference",
            "friction",
            "total_stress",
            "effective_stress",
            "Qtn",
            "Fr",
            "IC",
            "n",
            "vs",
            "G0",
            "E0",
            "permeability",
            "poisson",
            "damping",
        ]

        assert bro_cpt.name == gef_cpt.name == bro_cpt_new_version.name
        assert (
            bro_cpt.coordinates == gef_cpt.coordinates == bro_cpt_new_version.coordinates
        )
        assert bro_cpt.pwp == gef_cpt.pwp == bro_cpt_new_version.pwp
        assert bro_cpt.lithology == gef_cpt.lithology == bro_cpt_new_version.lithology
        assert np.allclose(bro_cpt.litho_points, gef_cpt.litho_points, atol=1e-2)
        assert np.allclose(
            bro_cpt_new_version.litho_points, gef_cpt.litho_points, atol=1e-2
        )
        for value in ["litho_NEN", "E_NEN", "cohesion_NEN", "fr_angle_NEN"]:
            for i in range(len(gef_cpt.litho_NEN)):
                assert set(getattr(bro_cpt, value)[i].split("/")) == set(
                    getattr(gef_cpt, value)[i].split("/")
                )
                assert set(getattr(bro_cpt_new_version, value)[i].split("/")) == set(
                    getattr(gef_cpt, value)[i].split("/")
                )

        for value in values_to_test:
            print(value)
            test = getattr(gef_cpt, value)
            expected_data = getattr(bro_cpt, value)
            expected_data_new_version = getattr(bro_cpt_new_version, value)
            assert np.allclose(expected_data, test, atol=1e-2)
            assert np.allclose(expected_data_new_version, test, atol=1e-2)


class TestInterpreter:
    @pytest.mark.systemtest
    def test_RobertsonCptInterpretation_1(self):
        # open the gef file
        test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "gef", "CPT000000003688_IMBRO_A.gef")
        )
        assert test_file.is_file()
        # initialise models
        cpt = GefCpt()
        # test initial expectations
        assert cpt
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing
        cpt.pre_process_data()
        # initialise interpretation model
        interpreter = RobertsonCptInterpretation()
        interpreter.unitweightmethod = UnitWeightMethod.LENGKEEK
        interpreter.shearwavevelocitymethod = ShearWaveVelocityMethod.ZANG
        interpreter.ocrmethod = OCRMethod.MAYNE
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing
        cpt.pre_process_data()
        # interpet the results
        cpt.interpret_cpt(interpreter)
        assert cpt
        assert cpt.lithology
        assert cpt.lithology_merged

    @pytest.mark.systemtest
    def test_RobertsonCptInterpretation_2(self):
        # open the gef file
        test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "gef", "CPT000000003688_IMBRO_A.gef")
        )
        assert test_file.is_file()
        # initialise models
        cpt = GefCpt()
        # test initial expectations
        assert cpt
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing
        cpt.pre_process_data()
        # initialise interpretation model
        interpreter = RobertsonCptInterpretation()
        interpreter.unitweightmethod = UnitWeightMethod.LENGKEEK
        interpreter.shearwavevelocitymethod = ShearWaveVelocityMethod.MAYNE
        interpreter.ocrmethod = OCRMethod.ROBERTSON
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing
        cpt.pre_process_data()
        # interpet the results
        cpt.interpret_cpt(interpreter)
        assert cpt
        assert cpt.lithology
        assert cpt.lithology_merged

    @pytest.mark.systemtest
    def test_RobertsonCptInterpretation_3(self):
        # open the gef file
        test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "gef", "CPT000000003688_IMBRO_A.gef")
        )
        assert test_file.is_file()
        # initialise models
        cpt = GefCpt()
        # test initial expectations
        assert cpt
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing
        cpt.pre_process_data()
        # initialise interpretation model
        interpreter = RobertsonCptInterpretation()
        interpreter.unitweightmethod = UnitWeightMethod.LENGKEEK
        interpreter.shearwavevelocitymethod = ShearWaveVelocityMethod.AHMED
        interpreter.ocrmethod = OCRMethod.MAYNE
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing
        cpt.pre_process_data()
        # interpet the results
        cpt.interpret_cpt(interpreter)
        assert cpt
        assert cpt.lithology
        assert cpt.lithology_merged

    @pytest.mark.systemtest
    def test_rho_calculation(self):
        # initialise models
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        # test initial expectations
        assert cpt
        assert interpreter
        # define inputs
        interpreter.gamma = np.ones(10)
        cpt.g = 10.0
        interpreter.data = cpt
        # run test
        interpreter.rho_calc()

        # exact solution = gamma / g
        exact_rho = np.ones(10) * 1000 / 10

        # self.assertEqual(exact_rho, self.cpt.rho)
        assert exact_rho.tolist() == cpt.rho.tolist()

    @pytest.mark.systemtest
    def test_gamma_calc(self):
        # initialise models
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Set all the values
        gamma_limit = 22
        interpreter.data.friction_nbr = np.ones(10)
        interpreter.data.qt = np.ones(10)
        interpreter.data.Pa = 100
        interpreter.data.depth_to_reference = range(10)
        interpreter.data.name = "UNIT_TEST"

        # Calculate analytically the solution
        np.seterr(divide="ignore")
        # Exact solution Robertson
        aux = 0.27 * np.log10(np.ones(10)) + 0.36 * (np.log10(np.ones(10) / 100)) + 1.236
        aux[np.abs(aux) == np.inf] = gamma_limit / 9.81
        local_gamma1 = aux * 9.81

        # call the function to be checked
        interpreter.gamma_calc()

        # Check if they are equal
        assert local_gamma1.tolist() == interpreter.gamma.tolist()

        # Exact solution Lengkeek
        local_gamma2 = 19 - 4.12 * (
            (np.log10(5000 / interpreter.data.qt))
            / (np.log10(30 / interpreter.data.friction_nbr))
        )
        interpreter.gamma_calc(gamma_max=gamma_limit, method=UnitWeightMethod.LENGKEEK)
        assert local_gamma2.tolist() == interpreter.gamma.tolist()

    @pytest.mark.systemtest
    def test_stress_calc(self):
        # initialise models
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Defining the inputs of the function
        interpreter.data.depth = np.arange(0, 2, 0.1)
        interpreter.gamma = [
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
        ]
        interpreter.data.depth_to_reference = np.zeros(20)
        interpreter.data.pwp = 0
        # run test
        interpreter.stress_calc()

        # The target list with the desired output
        effective_stress_test = [
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            21.5,
            23.0,
            24.5,
            26.0,
            27.5,
            29.0,
            30.5,
            32.0,
            33.5,
            35.0,
        ]

        # checking equality with the output
        assert effective_stress_test == list(
            np.around(interpreter.data.effective_stress, 1)
        )

    @pytest.mark.systemtest
    def test_norm_calc(self):
        # initialise models
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Empty list that will be filled by reading the csv file
        test_Qtn, total_stress, effective_stress, Pa, qt, friction = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        test_Fr = []

        # Opening and reading the csv file
        # These are also the inputs of the function
        test_path = TestUtils.get_local_test_data_dir(Path("test_norm_calc.csv"))
        # test initial expectations
        assert test_path.is_file()
        with open(test_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    total_stress.append(float(row[0]))
                    effective_stress.append(float(row[1]))
                    Pa.append(float(row[2]))
                    qt.append(float(row[3]))
                    friction.append(float(row[4]))

                    # These will be the outputs of the function
                    test_Qtn.append(float(row[5]))
                    test_Fr.append(float(row[6]))
                line_count += 1

        # Allocate them to the cpt
        interpreter.data.total_stress = np.array(total_stress)
        interpreter.data.effective_stress = np.array(effective_stress)
        interpreter.data.Pa = np.array(Pa)
        interpreter.data.qt = np.array(qt)
        interpreter.data.friction = np.array(friction)
        interpreter.data.friction_nbr = np.array(friction)
        interpreter.norm_calc(n_method=True)

        # Test the equality of the arrays
        assert test_Qtn == interpreter.data.Qtn.tolist()
        assert test_Fr == interpreter.data.Fr.tolist()

    @pytest.mark.unittest
    def test_IC_calc(self):
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Set the inputs of the values
        test_IC = [3.697093]
        interpreter.data.Qtn = [1]
        interpreter.data.Fr = [1]
        # run test
        interpreter.IC_calc()

        # Check if they are equal with the target value test_IC
        assert list(np.around(np.array(test_IC), 1)) == list(
            np.around(interpreter.data.IC, 1)
        )

    @pytest.mark.systemtest
    def test_vs_calc(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Define all the inputs
        interpreter.data.IC = np.array([1])
        interpreter.data.Qtn = np.array([1])
        interpreter.data.rho = np.array([1])
        interpreter.data.total_stress = np.array([1])
        interpreter.data.effective_stress = np.array([1])
        interpreter.data.tip = np.array([2])
        interpreter.data.qt = np.array([2])
        interpreter.data.friction = np.array([0.5])
        interpreter.data.Pa = 100
        interpreter.gamma = np.array([10])
        interpreter.data.vs = np.array([1])
        interpreter.data.depth = np.array([1])
        interpreter.data.depth_to_reference = np.array([1])
        interpreter.data.Fr = np.array([1])
        interpreter.data.name = "UNIT_TESTING"

        # Check the results for Robertson
        # Calculate analytically
        test_alpha_vs = 10 ** (0.55 * interpreter.data.IC + 1.68)
        test_vs = (
            test_alpha_vs * (interpreter.data.tip - interpreter.data.total_stress) / 100
        ) ** 0.5
        test_GO = interpreter.data.rho * test_vs**2

        # Call function
        interpreter.vs_calc(method=ShearWaveVelocityMethod.ROBERTSON)

        # Check their equality
        assert list(test_vs) == list(interpreter.data.vs)
        assert list(test_GO) == list(interpreter.data.G0)

        # Check the results for  Mayne
        # Calculate analytically
        test_vs = 118.8 * np.log10(interpreter.data.friction) + 18.5
        test_GO = interpreter.data.rho * test_vs**2

        # Call function
        interpreter.vs_calc(method=ShearWaveVelocityMethod.MAYNE)

        # Check their equality
        assert test_vs[0] == interpreter.data.vs[0]
        assert list(test_GO) == list(interpreter.data.G0)

        # Check the results for Andrus
        # Calculate analytically
        test_vs = (
            2.27
            * interpreter.data.qt**0.412
            * interpreter.data.IC**0.989
            * interpreter.data.depth**0.033
            * 1
        )
        test_GO = interpreter.data.rho * test_vs**2

        # Call function
        interpreter.vs_calc(method=ShearWaveVelocityMethod.ANDRUS)

        # Check their equality
        assert test_vs[0] == interpreter.data.vs[0]
        assert list(test_GO) == list(interpreter.data.G0)

        # Check the results for Zhang
        # Calculate analytically
        test_vs = (
            10.915
            * interpreter.data.tip**0.317
            * interpreter.data.IC**0.21
            * interpreter.data.depth**0.057
            * 0.92
        )
        test_GO = interpreter.data.rho * test_vs**2

        # Call function
        interpreter.vs_calc(method=ShearWaveVelocityMethod.ZANG)

        # Check their equality
        assert test_vs[0] == interpreter.data.vs[0]
        assert list(test_GO) == list(interpreter.data.G0)

        # Check the results for Ahmed
        # Calculate analytically
        test_vs = (
            1000
            * np.e ** (-0.887 * interpreter.data.IC)
            * (
                1
                + 0.443
                * interpreter.data.Fr
                * interpreter.data.effective_stress
                / 100
                * 9.81
                / interpreter.gamma
            )
            ** 0.5
        )
        test_GO = interpreter.data.rho * test_vs**2

        # Call the function
        interpreter.vs_calc(method=ShearWaveVelocityMethod.AHMED)

        # Check their equality
        assert test_vs[0] == interpreter.data.vs[0]
        assert list(test_GO) == list(interpreter.data.G0)

    @pytest.mark.unittest
    def test_poisson_calc(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Set the inputs
        interpreter.data.lithology = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

        # Set the target outputs
        test_poisson = [0.5, 0.5, 0.5, 0.25, 0.3, 0.3, 0.3, 0.375, 0.375]

        # Call the function
        interpreter.poisson_calc()

        # Check if they are equal
        assert test_poisson == list(interpreter.data.poisson)

    @pytest.mark.systemtest
    def test_damp_calc_1(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # all soil sensitive : damping = minimum value
        interpreter.data.lithology = ["1", "1", "1"]
        interpreter.data.effective_stress = np.ones(len(interpreter.data.lithology))
        interpreter.data.total_stress = np.ones(len(interpreter.data.lithology)) + 1
        interpreter.data.qt = np.ones(len(interpreter.data.lithology)) * 10

        # Define the target array
        test_damping = 2.512 * (interpreter.data.effective_stress / 100) ** -0.2889
        test_damping /= 100

        # Running the function
        interpreter.damp_calc()

        # Testing if the lists are equals
        assert list(test_damping) == list(interpreter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_2(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Defining the inputs
        # all soil very stiff : damping = minimum value
        interpreter.data.lithology = ["8", "8", "8"]
        interpreter.data.effective_stress = np.ones(len(interpreter.data.lithology))
        interpreter.data.total_stress = np.ones(len(interpreter.data.lithology)) + 1
        interpreter.data.qt = np.ones(len(interpreter.data.lithology)) * 10

        # The target output
        Cu = 2
        D50 = 0.02
        test_damping = (
            0.55
            * Cu**0.1
            * D50**-0.3
            * (interpreter.data.effective_stress / 100) ** -0.08
        )
        test_damping /= 100

        # Run the function to be tested
        interpreter.damp_calc(Cu=Cu, D50=D50)

        # Testing if the lists are equals
        assert list(test_damping) == list(interpreter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_3(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # all soil grained : damping = minimum value
        # Setting for testing
        interpreter.data.lithology = ["9", "9", "9"]
        interpreter.data.effective_stress = np.ones(len(interpreter.data.lithology))
        interpreter.data.total_stress = np.ones(len(interpreter.data.lithology)) + 1
        interpreter.data.qt = np.ones(len(interpreter.data.lithology)) * 10

        # Define the output
        test_damping = np.array([2, 2, 2]) / 100

        Cu = 3
        D50 = 0.025
        test_damping = (
            0.55
            * Cu**0.1
            * D50**-0.3
            * (interpreter.data.effective_stress / 100) ** -0.08
        )
        test_damping /= 100

        # Run the function to be tested
        interpreter.damp_calc(Cu=Cu, D50=D50)

        # Testing if the list are equal
        assert list(test_damping) == list(interpreter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_4(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Define the inputs
        # all soil sand
        interpreter.data.lithology = ["8", "6", "9", "7"]
        interpreter.data.effective_stress = np.ones(len(interpreter.data.lithology))
        interpreter.data.total_stress = np.ones(len(interpreter.data.lithology)) + 1
        interpreter.data.qt = np.ones(len(interpreter.data.lithology)) * 10

        # Calculate analytically for the type Meng
        Cu = 3
        D50 = 0.025
        test_damping = (
            0.55
            * Cu**0.1
            * D50**-0.3
            * (interpreter.data.effective_stress / 100) ** -0.08
        )
        test_damping /= 100

        # Run the function to be tested
        interpreter.damp_calc(Cu=Cu, D50=D50)

        # Testing if the list are equal
        assert list(test_damping) == list(interpreter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_5(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Define the inputs
        # all soil sand
        interpreter.data.lithology = ["3", "4", "3", "4"]
        interpreter.data.qt = np.ones(len(interpreter.data.lithology)) * 10
        interpreter.data.Qtn = np.ones(len(interpreter.data.lithology)) * 2
        interpreter.data.effective_stress = np.ones(len(interpreter.data.lithology))
        interpreter.data.total_stress = np.ones(len(interpreter.data.lithology)) + 1

        # Calculate analyticaly damping Darendeli - OCR according to Mayne
        Cu = 3
        D50 = 0.025
        PI = 40
        OCR = (
            0.33
            * (interpreter.data.qt - interpreter.data.total_stress)
            / interpreter.data.effective_stress
        )
        freq = 1
        test_damping = (
            (interpreter.data.effective_stress / 100) ** (-0.2889)
            * (0.8005 + 0.0129 * PI * OCR ** (-0.1069))
            * (1 + 0.2919 * np.log(freq))
        )
        test_damping /= 100

        # Call the function to be tested
        interpreter.damp_calc(Cu=Cu, D50=D50, Ip=PI, method=OCRMethod.MAYNE)

        # Test the damping Darendeli - OCR according to Mayne
        assert list(test_damping) == list(interpreter.data.damping)

        # Calculated analyticaly damping Darendeli - OCR according to robertson
        OCR = 0.25 * (interpreter.data.Qtn) ** 1.25
        test_damping = (
            (interpreter.data.effective_stress / 100) ** (-0.2889)
            * (0.8005 + 0.0129 * PI * OCR ** (-0.1069))
            * (1 + 0.2919 * np.log(freq))
        )
        test_damping /= 100

        interpreter.damp_calc(Cu=Cu, D50=D50, Ip=PI, method=OCRMethod.ROBERTSON)

        assert list(test_damping) == list(interpreter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_6(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Set the inputs
        # all soil sand
        interpreter.data.lithology = ["2", "2", "2", "2"]
        interpreter.data.effective_stress = np.ones(len(interpreter.data.lithology))
        interpreter.data.Pa = 1

        # Calculate analyticaly
        test_damping = (2.512 / 100) * np.ones(len(interpreter.data.lithology))

        # Call the function to be tested
        interpreter.damp_calc()

        # Check if they are equal
        assert list(test_damping) == list(interpreter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_7(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # stress is zero so damping is infinite
        interpreter.data.lithology = ["1", "1", "1"]
        interpreter.data.effective_stress = np.zeros(len(interpreter.data.lithology))
        interpreter.data.total_stress = np.zeros(len(interpreter.data.lithology)) + 1
        interpreter.data.qt = np.ones(len(interpreter.data.lithology)) * 10

        # Define the target array
        test_damping = [1, 1, 1]

        # Running the function
        interpreter.damp_calc()

        # Testing if the lists are equals
        assert list(test_damping) == list(interpreter.data.damping)

    @pytest.mark.unittest
    def test_permeability_calc(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Set the inputs
        # Ic below tipping point; at tipping point; above tipping point
        interpreter.data.lithology = ["7", "3", "2"]
        interpreter.data.IC = np.array([1, 3.27, 4])

        # The target list with the desired output
        test_permeability = np.array([0.008165824, 1.02612e-09, 6.19441e-12])

        # Call the function to be tested
        interpreter.permeability_calc()

        # Check if the values are almost equal
        for i in range(len(test_permeability)):
            assert abs(test_permeability[i] - interpreter.data.permeability[i]) < 0.001

    @pytest.mark.unittest
    def test_qt_calc(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Define the inputs
        interpreter.data.tip = np.array([1])
        interpreter.data.water = np.array([1])
        interpreter.data.a = np.array([1])

        # Define the target
        test_qt = np.array([1])

        # Call the function to be tested
        interpreter.qt_calc()

        # Check if the are equal
        assert list(test_qt) == list(interpreter.data.qt)

    @pytest.mark.unittest
    def test_Young_calc_first_case(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # test 1
        interpreter.data.G0 = np.ones(10) * 20.0
        interpreter.data.poisson = 0.10
        interpreter.young_calc()

        # exact solution
        exact_E = 2 * interpreter.data.G0 * (1 + interpreter.data.poisson)

        assert list(exact_E) == list(interpreter.data.E0)

    @pytest.mark.unittest
    def test_Young_calc_second_case(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # test 2
        interpreter.data.G0 = 5
        interpreter.data.poisson = 0.20
        interpreter.young_calc()

        # exact solution
        exact_E = 2 * interpreter.data.G0 * (1 + interpreter.data.poisson)

        # self.assertEqual(exact_rho, interpreter.data.rho)
        assert exact_E == interpreter.data.E0

    @pytest.mark.unittest
    def test_lithology_calc(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Define the input
        interpreter.data.tip = np.array([1])
        interpreter.data.friction_nbr = np.array([1])
        interpreter.data.friction = np.array([1])
        interpreter.data.effective_stress = np.array([2])
        interpreter.data.total_stress = np.array([2])
        interpreter.data.Qtn = np.array([1])
        interpreter.data.Fr = np.array([1])
        interpreter.data.Pa = 100
        lithology_test = ["1"]

        # Call the function to be tested
        interpreter.lithology_calc()

        # Check if results are equal
        assert list(interpreter.data.lithology) == list(lithology_test)

    @pytest.fixture
    def set_up_relative_density_test(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()

        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter

        # Define the input
        interpreter.data.tip = np.array([1, 1, 1, 1])
        interpreter.data.qt = np.array([20000, 20000, 20000, 20000])
        interpreter.data.friction_nbr = np.array([1, 1, 1, 1])
        interpreter.data.friction = np.array([1, 1, 1, 1])
        interpreter.data.effective_stress = np.array([10000, 10000, 10000, 10000])
        interpreter.data.total_stress = np.array([2, 2, 2, 2])
        interpreter.data.Qtn = np.array([1, 1, 1, 1])
        interpreter.data.Fr = np.array([1, 1, 1, 1])
        interpreter.data.Pa = 100
        interpreter.data.lithology = np.array(["1", "2", "6", "7"])

        return interpreter

    @pytest.mark.unittest
    def test_relative_density_calc_baldi(self, set_up_relative_density_test):

        # define method
        method = RelativeDensityMethod.BALDI

        # get test data
        interpreter = set_up_relative_density_test

        # calculate relative density
        interpreter.relative_density_calc(method)

        # get expected value
        expected_rd_value = 1 / 2.41 * np.log(20 / 15.7)
        expected_rd = [np.nan, np.nan, expected_rd_value, expected_rd_value]

        # assert
        np.testing.assert_array_almost_equal(
            expected_rd, interpreter.data.relative_density
        )

    @pytest.mark.unittest
    def test_relative_density_calc_kulhawy(self, set_up_relative_density_test):

        # define method
        method = RelativeDensityMethod.KULHAWY

        # get test data
        interpreter = set_up_relative_density_test

        # calculate relative density
        interpreter.relative_density_calc(method)

        # get expected value
        expected_rd_value = np.sqrt(
            20 / (305 * 1 * 1**0.18 * (1.2 + 0.05 * np.log10(10)))
        )
        expected_rd = [np.nan, np.nan, expected_rd_value, expected_rd_value]

        # assert
        np.testing.assert_array_almost_equal(
            expected_rd, interpreter.data.relative_density
        )

    @pytest.mark.unittest
    def test_relative_density_calc_kulhawy_simple(self, set_up_relative_density_test):

        # define method
        method = RelativeDensityMethod.KULHAWY_SIMPLE

        # get test data
        interpreter = set_up_relative_density_test
        interpreter.data.Qtn = np.ones(4) * 20

        # calculate relative density
        interpreter.relative_density_calc(method)

        # get expected value
        expected_rd_value = np.sqrt(20 / 350)
        expected_rd = [np.nan, np.nan, expected_rd_value, expected_rd_value]

        # assert
        np.testing.assert_array_almost_equal(
            expected_rd, interpreter.data.relative_density
        )

    @pytest.mark.unittest
    def test_norm_cone_resistance_clean_sand_calc(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Define the input
        interpreter.data.Qtn = np.array([1, 1, 1, 1, 1])
        interpreter.data.Fr = np.array([1, 1, 0.1, 1, 1])
        interpreter.data.IC = np.array([0.5, 2, 2, 2.6, 5])

        # Call the function to be tested
        interpreter.norm_cone_resistance_clean_sand_calc()

        # set expected array
        expected_K_c2 = (
            5.581 * 2**3 - 0.403 * 2**4 - 21.63 * 2**2 + 33.75 * 2 - 17.88
        )
        expected_K_c4 = 6e-7 * 2.6**16.76
        expected_Qtncs = np.array([1, expected_K_c2, 1, expected_K_c4, np.nan])

        # Check if results are equal
        np.testing.assert_array_almost_equal(interpreter.data.Qtncs, expected_Qtncs)

    @pytest.mark.unittest
    def test_state_parameter_calc(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # Define the input
        interpreter.data.Qtncs = np.array([10, 100, np.nan, -10])

        # Call the function to be tested
        interpreter.state_parameter_calc()

        # set expected array
        expected_psi = np.array([0.23, -0.1, np.nan, np.nan])

        # Check if results are equal
        np.testing.assert_array_almost_equal(interpreter.data.psi, expected_psi)

    @pytest.mark.systemtest
    def test_pwp_level_calc(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # define inputs
        interpreter.data.coordinates = [91931.0, 438294.0]
        # call test functions
        interpreter.pwp_level_calc()
        # test results
        assert math.isclose(-2.4, interpreter.data.pwp, rel_tol=0.001)

    @pytest.mark.systemtest
    def test_pwp_level_calc_wrong_path(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # define inputs
        interpreter.data.coordinates = [91931.0, 438294.0]
        interpreter.name_water_level_file = "oh_no_wrong_file.nc"
        with pytest.raises(FileNotFoundError):
            # call test functions
            interpreter.pwp_level_calc()

    @pytest.mark.systemtest
    def test_pwp_level_calc_wrong_suffix(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # define inputs
        interpreter.data.coordinates = [91931.0, 438294.0]
        interpreter.path_to_water_level_file = Path("cpt", "gef")
        interpreter.name_water_level_file = "CPT000000003688_IMBRO_A.gef"
        with pytest.raises(TypeError):
            # call test functions
            interpreter.pwp_level_calc()

    @pytest.mark.systemtest
    def test_pwp_level_calc_point_to_other_file(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # define inputs
        interpreter.data.coordinates = [91931.0, 438294.0]
        interpreter.path_to_water_level_file = Path(
            Path(__file__).parent.parent, "test_files"
        )
        interpreter.name_water_level_file = "peilgebieden_jp_250m.nc"
        # call test functions
        interpreter.pwp_level_calc()
        # test results
        assert math.isclose(-2.4, interpreter.data.pwp, rel_tol=0.001)

    @pytest.mark.systemtest
    def test_pwp_level_calc_user_already_inputted_pwp(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # define inputs
        interpreter.data.coordinates = [91931.0, 438294.0]
        interpreter.data.pwp = -3
        interpreter.user_defined_water_level = True
        # call test functions
        interpreter.pwp_level_calc()
        # test results
        assert math.isclose(-3, interpreter.data.pwp)

    @pytest.mark.systemtest
    def test_pwp_level_calc_user_already_inputted_pwp_error(self):
        # initialise model
        cpt = GefCpt()
        interpreter = RobertsonCptInterpretation()
        interpreter.data = cpt
        # test initial expectations
        assert cpt
        assert interpreter
        # define inputs
        interpreter.data.coordinates = [91931.0, 438294.0]
        interpreter.data.pwp = None
        interpreter.user_defined_water_level = True
        # call test functions
        with pytest.raises(ValueError):
            interpreter.pwp_level_calc()

    def test_stress_calc_total_stress(self, setup_interpreter):
        interpreter = RobertsonCptInterpretation()
        interpreter.data = type("test", (object,), {})()
        # case that we are bellow the surface

        interpreter.data.depth = np.array([0, 1, 2, 3, 4, 5])
        interpreter.data.depth_to_reference = np.array([0, 1, 2, 3, 4, 5])
        interpreter.data.pwp = 2.0
        interpreter.data.g = 9.81
        interpreter.data.gamma = np.array([18, 18, 18, 18, 18, 18])
        interpreter = setup_interpreter
        interpreter.stress_calc()
        expected_total_stress = np.array([0, 18, 36, 54, 72, 90])
        np.testing.assert_array_almost_equal(
            interpreter.data.total_stress, expected_total_stress
        )

    def test_stress_calc_no_pwp_effect(self):
        """
        Test when the computed pore water pressure (pwp) is zero,
        so that effective stress equals total stress.
        """
        instance = RobertsonCptInterpretation()
        instance.data = type("test", (object,), {})()
        # Set a simple depth array (in meters)
        instance.data.depth = np.array([0, 1, 2, 3, 4], dtype=float)
        # set the pwp to a very low value so that the pwp is negative which will be set to zero
        instance.data.pwp = -10  # Arbitrary; z_aux will still be 0.
        instance.data.depth_to_reference = instance.data.depth - 1
        instance.data.g = 9.81
        # Use a constant unit weight (gamma) of 18 kN/m3
        instance.gamma = np.full(instance.data.depth.shape, 18.0)

        instance.stress_calc()

        # Compute expected total stress:
        # z = diff([0,1,2,3,4]) = [1,1,1,1] then appended becomes [1,1,1,1,1].
        # Cumulative sum of (18 * 1): [18, 36, 54, 72, 90].
        expected_total = np.array([18, 36, 54, 72, 90], dtype=float)
        # Since pwp is zero, effective stress equals total stress.
        expected_effective = expected_total.copy()

        np.testing.assert_allclose(instance.data.total_stress, expected_total, rtol=1e-5)
        np.testing.assert_allclose(
            instance.data.effective_stress, expected_effective, rtol=1e-5
        )

    def _create_instance(self):
        """Helper to create an instance with a dummy 'data' attribute."""
        instance = RobertsonCptInterpretation()
        # Use a simple dummy object for data
        instance.data = type("DummyData", (), {})()
        return instance

    def _setup_stress_instance(self, depth, depth_to_reference, pwp, g, gamma):
        """
        Helper to setup an instance for stress_calc tests.

        Parameters:
            depth (array-like): Depth values.
            depth_to_reference (array-like or scalar): Depth-to-reference values.
            pwp (float): Pore water pressure.
            g (float): Gravitational acceleration.
            gamma (float): Unit weight (or a value to fill an array).
        """
        instance = self._create_instance()
        instance.data.depth = np.array(depth, dtype=float)
        if np.isscalar(depth_to_reference):
            instance.data.depth_to_reference = np.full(
                instance.data.depth.shape, depth_to_reference
            )
        else:
            instance.data.depth_to_reference = np.array(depth_to_reference, dtype=float)
        instance.data.pwp = pwp
        instance.data.g = g
        instance.gamma = (
            np.full(instance.data.depth.shape, gamma)
            if np.isscalar(gamma)
            else np.array(gamma, dtype=float)
        )
        return instance

    def _setup_norm_instance(self):
        """
        Helper to setup an instance for norm_calc tests.
        """
        instance = self._create_instance()
        instance.data.qt = np.array([10, 50, 100])
        instance.data.tip = np.array([1, 1, 1])
        instance.data.friction_nbr = np.array([1, 2, 3])
        instance.data.effective_stress = np.array([50, 50, 50])
        instance.data.total_stress = np.array([5, 10, 20])
        instance.data.Pa = 100
        instance.data.friction = np.array([1, 2, 3])
        return instance

    def test_stress_calc_positive_pwp(self):
        """
        Test a scenario where the computed pore water pressure is positive
        and reduces effective stress.
        """
        instance = self._setup_stress_instance(
            depth=[5, 6, 7, 8, 9], depth_to_reference=-2.0, pwp=10, g=9.81, gamma=18.0
        )

        instance.stress_calc()

        # Compute expected total stress:
        # The cumulative sum of (gamma * depth interval) plus an offset yields:
        expected_total = np.array([108, 126, 144, 162, 180], dtype=float)
        # For pwp, z_aux is calculated as min(10, 2 + 5)=7, so:
        pwp_val = (7 - (-2)) * 9.81  # 9 * 9.81
        expected_effective = expected_total - pwp_val

        np.testing.assert_allclose(instance.data.total_stress, expected_total, rtol=1e-5)
        np.testing.assert_allclose(
            instance.data.effective_stress, expected_effective, rtol=1e-5
        )

    def test_stress_calc_negative_pwp_clipped(self):
        """
        Test a case where the computed pore water pressure would be negative
        (indicating suction) but is then clipped to zero.
        """
        instance = self._setup_stress_instance(
            depth=[0, 1, 2, 3, 4],
            depth_to_reference=[2, 3, 3, 3, 3],
            pwp=10,
            g=9.81,
            gamma=18.0,
        )

        instance.stress_calc()

        # Expected total stress remains the same as computed from gamma:
        expected_total = np.array([18, 36, 54, 72, 90], dtype=float)
        # For pwp, z_aux = min(10, abs(2)+0)=2 leading to negative values,
        # so effective stress is clipped to be equal to total stress.
        expected_effective = expected_total.copy()

        np.testing.assert_allclose(instance.data.total_stress, expected_total, rtol=1e-5)
        np.testing.assert_allclose(
            instance.data.effective_stress, expected_effective, rtol=1e-5
        )

    def test_stress_calc_negative_effective_clipped(self):
        """
        Test a case where the pore water pressure is so high that the computed effective
        stress would be negative, and then it is clipped to zero.
        """
        instance = self._setup_stress_instance(
            depth=[0, 0.5, 1, 1.5, 2], depth_to_reference=-1.0, pwp=2, g=9.81, gamma=18.0
        )

        instance.stress_calc()

        # Calculate expected total stress from the cumulative sum:
        expected_total = np.array([9, 18, 27, 36, 45], dtype=float)
        # pwp: z_aux = min(2, abs(-1)+0)=1, so pwp = (1 - (-1)) * 9.81 = 19.62
        effective = expected_total - 19.62
        effective[effective < 0] = 0  # Clip negative effective stresses to zero

        np.testing.assert_allclose(instance.data.total_stress, expected_total, rtol=1e-5)
        np.testing.assert_allclose(instance.data.effective_stress, effective, rtol=1e-5)

    def _verify_norm_calc(self, instance, expected_n, exponent):
        """
        Helper to verify norm_calc results.

        Parameters:
            instance: The instance after norm_calc has been called.
            expected_n: Expected n values (numpy array).
            exponent: Exponent to use in the calculation of Cn.
        """
        Cn = (instance.data.Pa / instance.data.effective_stress) ** exponent
        Q_calc = (instance.data.qt - instance.data.total_stress) / instance.data.Pa * Cn
        Q_expected = np.where(Q_calc <= 1.0, 1.0, Q_calc)
        Q_expected = np.where(Q_expected >= 1000.0, 1000.0, Q_expected)

        F_calc = (
            instance.data.friction / (instance.data.qt - instance.data.total_stress) * 100
        )
        F_expected = np.where(F_calc <= 0.1, 0.1, F_calc)
        F_expected = np.where(F_expected >= 10.0, 10.0, F_expected)

        np.testing.assert_allclose(instance.data.n, expected_n, rtol=1e-6)
        np.testing.assert_allclose(instance.data.Qtn, Q_expected, rtol=1e-6)
        np.testing.assert_allclose(instance.data.Fr, F_expected, rtol=1e-6)

    def test_norm_calc_n_method_true(self):
        """
        Test norm_calc when using the simplified method (n_method=True)
        where the stress exponent n is set to 0.5.
        """
        instance = self._setup_norm_instance()
        instance.norm_calc(n_method=True)

        expected_n = np.array([0.5, 0.5, 0.5])
        self._verify_norm_calc(instance, expected_n, exponent=0.5)

    def test_norm_calc_n_method_false(self):
        """
        Test norm_calc when using the iterative n calculation (n_method=False).
        For the provided dummy data the iteration is expected to converge to n = 1.0.
        """
        instance = self._setup_norm_instance()
        instance.norm_calc(n_method=False)

        expected_n = np.array([1.0, 1.0, 1.0])
        self._verify_norm_calc(instance, expected_n, exponent=1.0)
