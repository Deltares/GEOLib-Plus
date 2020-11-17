from geolib_plus.robertson_cpt_interpretation import (
    RobertsonCptInterpretation,
    UnitWeightMethod,
    OCRMethod,
    ShearWaveVelocityMethod,
)
from geolib_plus.gef_cpt import GefCpt

from tests.utils import TestUtils
import numpy as np
import pytest
from pathlib import Path
import csv
import json


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


class TestIntergration:
    @pytest.mark.integrationtest
    def test_against_bro_results(self):
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
        robertson = RobertsonCptInterpretation
        robertson.unitweightmethod = UnitWeightMethod.ROBERTSON
        # interpet the results
        cpt.interpret_cpt(robertson)
        # read already calculated data
        benchmark_file = TestUtils.get_local_test_data_dir(
            Path("results_CPT000000003688_IMBRO_A.gef.json")
        )
        assert benchmark_file.is_file()
        with open(benchmark_file) as f:
            benchmark_data = json.load(f)
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

        assert benchmark_data["name"].split("_")[0] == cpt.name
        assert benchmark_data["coordinates"] == cpt.coordinates
        assert benchmark_data["ground_water_level"] == cpt.pwp
        assert benchmark_data["lithology"] == cpt.lithology
        assert np.allclose(benchmark_data["litho_points"], cpt.litho_points)
        for value in ["litho_NEN", "E_NEN", "cohesion_NEN", "fr_angle_NEN"]:
            for i in range(len(cpt.litho_NEN)):
                assert set(getattr(cpt, value)[i].split("/")) == set(
                    benchmark_data[value][i].split("/")
                )
        # GEOLIB-PLUS removed 3 depths
        for value in values_to_test:
            print(value)
            test = getattr(cpt, value)
            if value == "depth_to_reference":
                test = abs(test)
            assert np.allclose(benchmark_data[value], test)


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
        interpeter = RobertsonCptInterpretation
        interpeter.unitweightmethod = UnitWeightMethod.LENGKEEK
        interpeter.shearwavevelocitymethod = ShearWaveVelocityMethod.ZANG
        interpeter.ocrmethod = OCRMethod.MAYNE
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing
        cpt.pre_process_data()
        # interpet the results
        cpt.interpret_cpt(interpeter)
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
        interpeter = RobertsonCptInterpretation
        interpeter.unitweightmethod = UnitWeightMethod.LENGKEEK
        interpeter.shearwavevelocitymethod = ShearWaveVelocityMethod.MAYNE
        interpeter.ocrmethod = OCRMethod.ROBERTSON
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing
        cpt.pre_process_data()
        # interpet the results
        cpt.interpret_cpt(interpeter)
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
        interpeter = RobertsonCptInterpretation
        interpeter.unitweightmethod = UnitWeightMethod.LENGKEEK
        interpeter.shearwavevelocitymethod = ShearWaveVelocityMethod.AHMED
        interpeter.ocrmethod = OCRMethod.MAYNE
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing
        cpt.pre_process_data()
        # interpet the results
        cpt.interpret_cpt(interpeter)
        assert cpt
        assert cpt.lithology
        assert cpt.lithology_merged

    @pytest.mark.systemtest
    def test_rho_calculation(self):
        # initialise models
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        # test initial expectations
        assert cpt
        assert interpeter
        # define inputs
        interpeter.gamma = np.ones(10)
        cpt.g = 10.0
        interpeter.data = cpt
        # run test
        interpeter.rho_calc()

        # exact solution = gamma / g
        exact_rho = np.ones(10) * 1000 / 10

        # self.assertEqual(exact_rho, self.cpt.rho)
        assert exact_rho.tolist() == cpt.rho.tolist()

    @pytest.mark.systemtest
    def test_gamma_calc(self):
        # initialise models
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Set all the values
        gamma_limit = 22
        interpeter.data.friction_nbr = np.ones(10)
        interpeter.data.qt = np.ones(10)
        interpeter.data.Pa = 100
        interpeter.data.depth_to_reference = range(10)
        interpeter.data.name = "UNIT_TEST"

        # Calculate analytically the solution
        np.seterr(divide="ignore")
        # Exact solution Robertson
        aux = (
            0.27 * np.log10(np.ones(10)) + 0.36 * (np.log10(np.ones(10) / 100)) + 1.236
        )
        aux[np.abs(aux) == np.inf] = gamma_limit / 9.81
        local_gamma1 = aux * 9.81

        # call the function to be checked
        interpeter.gamma_calc()

        # Check if they are equal
        assert local_gamma1.tolist() == interpeter.gamma.tolist()

        # Exact solution Lengkeek
        local_gamma2 = 19 - 4.12 * (
            (np.log10(5000 / interpeter.data.qt))
            / (np.log10(30 / interpeter.data.friction_nbr))
        )
        interpeter.gamma_calc(gamma_max=gamma_limit, method=UnitWeightMethod.LENGKEEK)
        assert local_gamma2.tolist() == interpeter.gamma.tolist()

    @pytest.mark.systemtest
    def test_stress_calc(self):
        # initialise models
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Defining the inputs of the function
        interpeter.data.depth = np.arange(0, 2, 0.1)
        interpeter.gamma = [
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
        interpeter.data.depth_to_reference = np.zeros(20)
        interpeter.data.pwp = 0
        # run test
        interpeter.stress_calc()

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
            np.around(interpeter.data.effective_stress, 1)
        )

    @pytest.mark.systemtest
    def test_norm_calc(self):
        # initialise models
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Empty list that will be filled by reading the csv file
        test_Qtn, total_stress, effective_stress, Pa, tip, friction = (
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
                    tip.append(float(row[3]))
                    friction.append(float(row[4]))

                    # These will be the outputs of the function
                    test_Qtn.append(float(row[5]))
                    test_Fr.append(float(row[6]))
                line_count += 1

        # Allocate them to the cpt
        interpeter.data.total_stress = np.array(total_stress)
        interpeter.data.effective_stress = np.array(effective_stress)
        interpeter.data.Pa = np.array(Pa)
        interpeter.data.tip = np.array(tip)
        interpeter.data.friction = np.array(friction)
        interpeter.data.friction_nbr = np.array(friction)
        interpeter.norm_calc(n_method=True)

        # Test the equality of the arrays
        assert test_Qtn == interpeter.data.Qtn.tolist()
        assert test_Fr == interpeter.data.Fr.tolist()

    @pytest.mark.unittest
    def test_IC_calc(self):
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Set the inputs of the values
        test_IC = [3.697093]
        interpeter.data.Qtn = [1]
        interpeter.data.Fr = [1]
        # run test
        interpeter.IC_calc()

        # Check if they are equal with the target value test_IC
        assert list(np.around(np.array(test_IC), 1)) == list(
            np.around(interpeter.data.IC, 1)
        )

    @pytest.mark.systemtest
    def test_vs_calc(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Define all the inputs
        interpeter.data.IC = np.array([1])
        interpeter.data.Qtn = np.array([1])
        interpeter.data.rho = np.array([1])
        interpeter.data.total_stress = np.array([1])
        interpeter.data.effective_stress = np.array([1])
        interpeter.data.tip = np.array([2])
        interpeter.data.qt = np.array([2])
        interpeter.data.friction = np.array([0.5])
        interpeter.data.Pa = 100
        interpeter.gamma = np.array([10])
        interpeter.data.vs = np.array([1])
        interpeter.data.depth = np.array([1])
        interpeter.data.depth_to_reference = np.array([1])
        interpeter.data.Fr = np.array([1])
        interpeter.data.name = "UNIT_TESTING"

        # Check the results for Robertson
        # Calculate analytically
        test_alpha_vs = 10 ** (0.55 * interpeter.data.IC + 1.68)
        test_vs = (
            test_alpha_vs * (interpeter.data.tip - interpeter.data.total_stress) / 100
        ) ** 0.5
        test_GO = interpeter.data.rho * test_vs ** 2

        # Call function
        interpeter.vs_calc(method=ShearWaveVelocityMethod.ROBERTSON)

        # Check their equality
        assert list(test_vs) == list(interpeter.data.vs)
        assert list(test_GO) == list(interpeter.data.G0)

        # Check the results for  Mayne
        # Calculate analytically
        test_vs = 118.8 * np.log10(interpeter.data.friction) + 18.5
        test_GO = interpeter.data.rho * test_vs ** 2

        # Call function
        interpeter.vs_calc(method=ShearWaveVelocityMethod.MAYNE)

        # Check their equality
        assert test_vs[0] == interpeter.data.vs[0]
        assert list(test_GO) == list(interpeter.data.G0)

        # Check the results for Andrus
        # Calculate analytically
        test_vs = (
            2.27
            * interpeter.data.qt ** 0.412
            * interpeter.data.IC ** 0.989
            * interpeter.data.depth ** 0.033
            * 1
        )
        test_GO = interpeter.data.rho * test_vs ** 2

        # Call function
        interpeter.vs_calc(method=ShearWaveVelocityMethod.ANDRUS)

        # Check their equality
        assert test_vs[0] == interpeter.data.vs[0]
        assert list(test_GO) == list(interpeter.data.G0)

        # Check the results for Zhang
        # Calculate analytically
        test_vs = (
            10.915
            * interpeter.data.tip ** 0.317
            * interpeter.data.IC ** 0.21
            * interpeter.data.depth ** 0.057
            * 0.92
        )
        test_GO = interpeter.data.rho * test_vs ** 2

        # Call function
        interpeter.vs_calc(method=ShearWaveVelocityMethod.ZANG)

        # Check their equality
        assert test_vs[0] == interpeter.data.vs[0]
        assert list(test_GO) == list(interpeter.data.G0)

        # Check the results for Ahmed
        # Calculate analytically
        test_vs = (
            1000
            * np.e ** (-0.887 * interpeter.data.IC)
            * (
                1
                + 0.443
                * interpeter.data.Fr
                * interpeter.data.effective_stress
                / 100
                * 9.81
                / interpeter.gamma
            )
            ** 0.5
        )
        test_GO = interpeter.data.rho * test_vs ** 2

        # Call the function
        interpeter.vs_calc(method=ShearWaveVelocityMethod.AHMED)

        # Check their equality
        assert test_vs[0] == interpeter.data.vs[0]
        assert list(test_GO) == list(interpeter.data.G0)

    @pytest.mark.unittest
    def test_poisson_calc(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Set the inputs
        interpeter.data.lithology = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

        # Set the target outputs
        test_poisson = [0.5, 0.5, 0.5, 0.25, 0.3, 0.3, 0.3, 0.375, 0.375]

        # Call the function
        interpeter.poisson_calc()

        # Check if they are equal
        assert test_poisson == list(interpeter.data.poisson)

    @pytest.mark.systemtest
    def test_damp_calc_1(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # all soil sensitive : damping = minimum value
        interpeter.data.lithology = ["1", "1", "1"]
        interpeter.data.effective_stress = np.ones(len(interpeter.data.lithology))
        interpeter.data.total_stress = np.ones(len(interpeter.data.lithology)) + 1
        interpeter.data.qt = np.ones(len(interpeter.data.lithology)) * 10

        # Define the target array
        test_damping = 2.512 * (interpeter.data.effective_stress / 100) ** -0.2889
        test_damping /= 100

        # Running the function
        interpeter.damp_calc()

        # Testing if the lists are equals
        assert list(test_damping) == list(interpeter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_2(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Defining the inputs
        # all soil very stiff : damping = minimum value
        interpeter.data.lithology = ["8", "8", "8"]
        interpeter.data.effective_stress = np.ones(len(interpeter.data.lithology))
        interpeter.data.total_stress = np.ones(len(interpeter.data.lithology)) + 1
        interpeter.data.qt = np.ones(len(interpeter.data.lithology)) * 10

        # The target output
        Cu = 2
        D50 = 0.02
        test_damping = (
            0.55
            * Cu ** 0.1
            * D50 ** -0.3
            * (interpeter.data.effective_stress / 100) ** -0.08
        )
        test_damping /= 100

        # Run the function to be tested
        interpeter.damp_calc(Cu=Cu, D50=D50)

        # Testing if the lists are equals
        assert list(test_damping) == list(interpeter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_3(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # all soil grained : damping = minimum value
        # Setting for testing
        interpeter.data.lithology = ["9", "9", "9"]
        interpeter.data.effective_stress = np.ones(len(interpeter.data.lithology))
        interpeter.data.total_stress = np.ones(len(interpeter.data.lithology)) + 1
        interpeter.data.qt = np.ones(len(interpeter.data.lithology)) * 10

        # Define the output
        test_damping = np.array([2, 2, 2]) / 100

        Cu = 3
        D50 = 0.025
        test_damping = (
            0.55
            * Cu ** 0.1
            * D50 ** -0.3
            * (interpeter.data.effective_stress / 100) ** -0.08
        )
        test_damping /= 100

        # Run the function to be tested
        interpeter.damp_calc(Cu=Cu, D50=D50)

        # Testing if the list are equal
        assert list(test_damping) == list(interpeter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_4(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Define the inputs
        # all soil sand
        interpeter.data.lithology = ["8", "6", "9", "7"]
        interpeter.data.effective_stress = np.ones(len(interpeter.data.lithology))
        interpeter.data.total_stress = np.ones(len(interpeter.data.lithology)) + 1
        interpeter.data.qt = np.ones(len(interpeter.data.lithology)) * 10

        # Calculate analytically for the type Meng
        Cu = 3
        D50 = 0.025
        test_damping = (
            0.55
            * Cu ** 0.1
            * D50 ** -0.3
            * (interpeter.data.effective_stress / 100) ** -0.08
        )
        test_damping /= 100

        # Run the function to be tested
        interpeter.damp_calc(Cu=Cu, D50=D50)

        # Testing if the list are equal
        assert list(test_damping) == list(interpeter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_5(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Define the inputs
        # all soil sand
        interpeter.data.lithology = ["3", "4", "3", "4"]
        interpeter.data.qt = np.ones(len(interpeter.data.lithology)) * 10
        interpeter.data.Qtn = np.ones(len(interpeter.data.lithology)) * 2
        interpeter.data.effective_stress = np.ones(len(interpeter.data.lithology))
        interpeter.data.total_stress = np.ones(len(interpeter.data.lithology)) + 1

        # Calculate analyticaly damping Darendeli - OCR according to Mayne
        Cu = 3
        D50 = 0.025
        PI = 40
        OCR = (
            0.33
            * (interpeter.data.qt - interpeter.data.total_stress)
            / interpeter.data.effective_stress
        )
        freq = 1
        test_damping = (
            (interpeter.data.effective_stress / 100) ** (-0.2889)
            * (0.8005 + 0.0129 * PI * OCR ** (-0.1069))
            * (1 + 0.2919 * np.log(freq))
        )
        test_damping /= 100

        # Call the function to be tested
        interpeter.damp_calc(Cu=Cu, D50=D50, Ip=PI, method=OCRMethod.MAYNE)

        # Test the damping Darendeli - OCR according to Mayne
        assert list(test_damping) == list(interpeter.data.damping)

        # Calculated analyticaly damping Darendeli - OCR according to robertson
        OCR = 0.25 * (interpeter.data.Qtn) ** 1.25
        test_damping = (
            (interpeter.data.effective_stress / 100) ** (-0.2889)
            * (0.8005 + 0.0129 * PI * OCR ** (-0.1069))
            * (1 + 0.2919 * np.log(freq))
        )
        test_damping /= 100

        interpeter.damp_calc(Cu=Cu, D50=D50, Ip=PI, method=OCRMethod.ROBERTSON)

        assert list(test_damping) == list(interpeter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_6(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Set the inputs
        # all soil sand
        interpeter.data.lithology = ["2", "2", "2", "2"]
        interpeter.data.effective_stress = np.ones(len(interpeter.data.lithology))
        interpeter.data.Pa = 1

        # Calculate analyticaly
        test_damping = (2.512 / 100) * np.ones(len(interpeter.data.lithology))

        # Call the function to be tested
        interpeter.damp_calc()

        # Check if they are equal
        assert list(test_damping) == list(interpeter.data.damping)

    @pytest.mark.unittest
    def test_damp_calc_7(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # stress is zero so damping is infinite
        interpeter.data.lithology = ["1", "1", "1"]
        interpeter.data.effective_stress = np.zeros(len(interpeter.data.lithology))
        interpeter.data.total_stress = np.zeros(len(interpeter.data.lithology)) + 1
        interpeter.data.qt = np.ones(len(interpeter.data.lithology)) * 10

        # Define the target array
        test_damping = [1, 1, 1]

        # Running the function
        interpeter.damp_calc()

        # Testing if the lists are equals
        assert list(test_damping) == list(interpeter.data.damping)

    @pytest.mark.unittest
    def test_permeability_calc(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Set the inputs
        # Ic below tipping point; at tipping point; above tipping point
        interpeter.data.lithology = ["7", "3", "2"]
        interpeter.data.IC = np.array([1, 3.27, 4])

        # The target list with the desired output
        test_permeability = np.array([0.008165824, 1.02612e-09, 6.19441e-12])

        # Call the function to be tested
        interpeter.permeability_calc()

        # Check if the values are almost equal
        for i in range(len(test_permeability)):
            assert abs(test_permeability[i] - interpeter.data.permeability[i]) < 0.001

    @pytest.mark.unittest
    def test_qt_calc(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Define the inputs
        interpeter.data.tip = np.array([1])
        interpeter.data.water = np.array([1])
        interpeter.data.a = np.array([1])

        # Define the target
        test_qt = np.array([1])

        # Call the function to be tested
        interpeter.qt_calc()

        # Check if the are equal
        assert list(test_qt) == list(interpeter.data.qt)

    @pytest.mark.unittest
    def test_Young_calc_first_case(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # test 1
        interpeter.data.G0 = np.ones(10) * 20.0
        interpeter.data.poisson = 0.10
        interpeter.young_calc()

        # exact solution
        exact_E = 2 * interpeter.data.G0 * (1 + interpeter.data.poisson)

        assert list(exact_E) == list(interpeter.data.E0)

    @pytest.mark.unittest
    def test_Young_calc_second_case(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # test 2
        interpeter.data.G0 = 5
        interpeter.data.poisson = 0.20
        interpeter.young_calc()

        # exact solution
        exact_E = 2 * interpeter.data.G0 * (1 + interpeter.data.poisson)

        # self.assertEqual(exact_rho, interpeter.data.rho)
        assert exact_E == interpeter.data.E0

    @pytest.mark.unittest
    def test_lithology_calc(self):
        # initialise model
        cpt = GefCpt()
        interpeter = RobertsonCptInterpretation()
        interpeter.data = cpt
        # test initial expectations
        assert cpt
        assert interpeter
        # Define the input
        interpeter.data.tip = np.array([1])
        interpeter.data.friction_nbr = np.array([1])
        interpeter.data.friction = np.array([1])
        interpeter.data.effective_stress = np.array([2])
        interpeter.data.total_stress = np.array([2])
        interpeter.data.Qtn = np.array([1])
        interpeter.data.Fr = np.array([1])
        interpeter.data.Pa = 100
        lithology_test = ["1"]

        # Call the function to be tested
        interpeter.lithology_calc()

        # Check if results are equal
        assert list(interpeter.data.lithology) == list(lithology_test)