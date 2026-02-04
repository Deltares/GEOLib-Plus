import os
import stat
from pathlib import Path

import matplotlib
import numpy as np
import pytest

import geolib_plus.plot_cpt as plot_cpt
from geolib_plus.bro_xml_cpt import BroXmlCpt
from geolib_plus.gef_cpt import GefCpt
from geolib_plus.plot_settings import PlotSettings
from tests.utils import TestUtils

matplotlib.use("Agg")


class TestPlotCpt:
    @pytest.mark.unittest
    def test_get_ylims_greater_than_data(self, cpt_with_water):
        """
        Assert positive buffer at top and length graph larger than the cpt data
        :return:
        """
        settings = {
            "plot_size": "a4",
            "vertical_settings": "",
        }

        vertical_settings = {
            "buffer_at_top": 1,
            "length_graph": 100,
            "top_type": "relative",
            "repeated_distance": 0,
        }
        settings["vertical_settings"] = vertical_settings

        ylims = plot_cpt.get_y_lims(cpt_with_water, settings)

        assert pytest.approx(ylims[0][0]) == -1.0
        assert pytest.approx(ylims[0][1]) == -101.0

    @pytest.mark.unittest
    def test_get_ylims_negative_buffer_at_top(self, cpt_with_water):
        """
        Assert assert negative buffer at top and length graph larger than the cpt data
        :return:
        """
        settings = {
            "plot_size": "a4",
            "vertical_settings": "",
        }

        vertical_settings = {
            "buffer_at_top": -1,
            "length_graph": 100,
            "top_type": "relative",
            "repeated_distance": 0,
        }
        settings["vertical_settings"] = vertical_settings

        ylims = plot_cpt.get_y_lims(cpt_with_water, settings)

        assert pytest.approx(ylims[0][0]) == -3.0
        assert pytest.approx(ylims[0][1]) == -103.0

    @pytest.mark.unittest
    def test_get_ylims_smaller_than_data(self, cpt_with_water):
        """
        Assert length graph is smaller than the cpt data
        :return:
        """

        settings = {
            "plot_size": "a4",
            "vertical_settings": "",
        }

        vertical_settings = {
            "buffer_at_top": 0,
            "length_graph": 10,
            "top_type": "relative",
            "repeated_distance": 0,
        }
        settings["vertical_settings"] = vertical_settings

        ylims = plot_cpt.get_y_lims(cpt_with_water, settings)

        assert pytest.approx(ylims[0][0]) == -2.0
        assert pytest.approx(ylims[0][1]) == -12.0
        assert pytest.approx(ylims[1][0]) == -12.0
        assert pytest.approx(ylims[1][1]) == -22.0

    @pytest.mark.unittest
    def test_get_ylims_repeated_distance(self, cpt_with_water):
        """
        Assert length graph is smaller than the cpt data and last meter of previous graph is repeated
        :return:
        """

        settings = {
            "plot_size": "a4",
            "vertical_settings": "",
        }

        vertical_settings = {
            "buffer_at_top": 0,
            "length_graph": 10,
            "top_type": "relative",
            "repeated_distance": 1,
        }
        settings["vertical_settings"] = vertical_settings

        ylims = plot_cpt.get_y_lims(cpt_with_water, settings)

        assert pytest.approx(ylims[0][0]) == -2.0
        assert pytest.approx(ylims[0][1]) == -12.0
        assert pytest.approx(ylims[1][0]) == -11.0
        assert pytest.approx(ylims[1][1]) == -21.0

    @pytest.mark.unittest
    def test_get_ylims_absolute_top_level(self, cpt_with_water):
        """
        Assert length graph is smaller than the cpt data and top of first graph is an absolute given value
        :return:
        """

        settings = {
            "plot_size": "a4",
            "vertical_settings": "",
        }

        vertical_settings = {
            "absolute_top_level": 0,
            "length_graph": 10,
            "top_type": "absolute",
            "repeated_distance": 0,
        }

        settings["vertical_settings"] = vertical_settings

        ylims = plot_cpt.get_y_lims(cpt_with_water, settings)

        assert pytest.approx(ylims[0][0]) == 0.0
        assert pytest.approx(ylims[0][1]) == -10.0
        assert pytest.approx(ylims[1][0]) == -10.0
        assert pytest.approx(ylims[1][1]) == -20.0
        assert pytest.approx(ylims[2][0]) == -20.0
        assert pytest.approx(ylims[2][1]) == -30.0

    @pytest.mark.unittest
    def test_trim_cpt_data_within_thresholds(self, cpt_with_water):
        """
        Test trim cpt within tresholds without predrill depth
        :return:
        """
        settings = {"data_key": "qc", "threshold": [0, 32], "unit_converter": 1}

        vertical_settings = {"spacing_shown_cut_off_value": 1}

        # get actual undefined_depth, required because the scope of the test fixture is class
        undef_depth = cpt_with_water.undefined_depth

        # do not take into account predrill
        cpt_with_water.undefined_depth = 0

        # assert all data is within threshold
        (
            trimmed_values,
            shown_values,
            y_coord_shown_value,
            depth_in_range,
            inclination_in_range,
        ) = plot_cpt.trim_cpt_data(
            settings, vertical_settings, cpt_with_water, [-1, -101]
        )

        assert shown_values.size == 0
        assert y_coord_shown_value.size == 0
        for idx, data in enumerate(depth_in_range):
            assert data == cpt_with_water.depth_to_reference[idx]
            assert trimmed_values[idx] == cpt_with_water.tip[idx]

        # Set undefined depth again
        cpt_with_water.undefined_depth = undef_depth

    @pytest.mark.unittest
    def test_trim_cpt_data_within_thresholds_with_predrill(self, cpt_with_water):
        """
        Test trim cpt within thresholds with predrill depth
        :return:
        """
        settings = {"data_key": "qc", "threshold": [0, 32], "unit_converter": 1}

        vertical_settings = {"spacing_shown_cut_off_value": 1}

        # set expected result
        expected_result_depth = cpt_with_water.depth_to_reference[
            cpt_with_water.depth_to_reference
            < cpt_with_water.local_reference_level - cpt_with_water.undefined_depth
        ]

        expected_result_tip = cpt_with_water.tip[
            cpt_with_water.depth_to_reference
            < cpt_with_water.local_reference_level - cpt_with_water.undefined_depth
        ]

        # assert all data is within threshold
        (
            trimmed_values,
            shown_values,
            y_coord_shown_value,
            depth_in_range,
            inclination_in_range,
        ) = plot_cpt.trim_cpt_data(
            settings, vertical_settings, cpt_with_water, [-1, -101]
        )

        assert shown_values.size == 0
        assert y_coord_shown_value.size == 0
        for idx, data in enumerate(depth_in_range):

            assert data == expected_result_depth[idx]
            assert trimmed_values[idx] == expected_result_tip[idx]

    @pytest.mark.unittest
    def test_trim_cpt_data_partly_outside_thresholds(self):
        """
        Test trimmed cpt data where the original data falls partly outside the thresholds
        :return:
        """
        settings = {"data_key": "qc", "threshold": [0, 0.7], "unit_converter": 1}

        vertical_settings = {"spacing_shown_cut_off_value": 1}

        cpt = BroXmlCpt()

        # set up cpt
        cpt.depth_to_reference = np.linspace(0, -10, 11)
        cpt.undefined_depth = 0
        cpt.local_reference_level = 0
        cpt.tip = np.sin(cpt.depth_to_reference * 1 / 4 * np.pi)
        cpt.inclination_resultant = np.zeros(11)

        (
            trimmed_values,
            shown_values,
            y_coord_shown_value,
            depth_in_range,
            inclination_in_range,
        ) = plot_cpt.trim_cpt_data(settings, vertical_settings, cpt, [0, -11])

        # Assert if trimmed values are as expected
        expected_trimmed_values = np.array([0, 0, 0, 0, 0.7, 0.7, 0.7, 0, 0, 0])
        for idx, trimmed_value in enumerate(trimmed_values):
            assert expected_trimmed_values[idx] == pytest.approx(trimmed_value)

    @pytest.mark.integrationtest
    def test_generate_fig_with_inverse_friction_nbr(self, cpt, plot_settings):
        """
        Test plotting of an inversed friction nbr for a BroXmlCpt and a GefCpt

        :param cpt: BroXmlCpt or GefCpt
        :param plot_settings:  Settings for the plot
        :return:
        """

        plot_settings.set_inversed_friction_number_in_plot()

        output_path = Path(TestUtils._name_output)
        plot_cpt.plot_cpt_norm(cpt, output_path, plot_settings.general_settings)

        output_file_name = cpt.name + ".pdf"
        assert Path(output_path / output_file_name).is_file()
        (output_path / output_file_name).unlink()

    @pytest.mark.integrationtest
    def test_generate_fig_without_inclination(self, cpt, plot_settings):
        """
        Test plotting of a BroXmlCpt and a GefCpt without available inclination angle

        :param cpt: BroXmlCpt or GefCpt
        :param plot_settings:  Settings for the plot
        :return:
        """

        cpt.inclination_resultant = None
        output_path = Path(TestUtils._name_output)
        plot_cpt.plot_cpt_norm(cpt, output_path, plot_settings.general_settings)

        output_file_name = cpt.name + ".pdf"
        assert Path(output_path / output_file_name).is_file()
        (output_path / output_file_name).unlink()

    @pytest.mark.integrationtest
    def test_generate_fig_with_default_settings(self, cpt, plot_settings):
        """
        Test plotting of a BroXmlCpt and a GefCpt with default settings

        :param cpt: BroXmlCpt or GefCpt
        :param plot_settings:  Settings for the plot
        :return:
        """

        plot_settings.assign_default_settings()

        output_path = Path(TestUtils._name_output)
        plot_cpt.plot_cpt_norm(cpt, output_path, plot_settings.general_settings)

        output_file_name = cpt.name + ".pdf"
        assert Path(output_path / output_file_name).is_file()
        (output_path / output_file_name).unlink()

    @pytest.fixture(scope="class", params=[BroXmlCpt(), GefCpt()])
    def cpt(self, request):
        """
        Fills de cpt data class with data from a xml file and a gef file.
        :param request:
        :return:
        """
        if isinstance(request.param, BroXmlCpt):
            test_folder = Path(TestUtils.get_local_test_data_dir("cpt/bro_xml"))
            filename = "CPT000000003688_IMBRO_A.xml"
        elif isinstance(request.param, GefCpt):
            test_folder = Path(TestUtils.get_local_test_data_dir("cpt/gef"))
            filename = "CPT000000003688_IMBRO_A.gef"
        else:
            return None

        cpt = request.param
        test_file = test_folder / filename
        cpt.read(test_file)
        cpt.pre_process_data()
        return cpt

    @pytest.fixture(scope="class", params=[BroXmlCpt(), GefCpt()])
    def cpt_with_water(self, request):
        """
        Fills de cpt data class with data from a xml file and a gef file. The data includes water pressure.
        :param request:
        :return:
        """
        if isinstance(request.param, BroXmlCpt):
            test_folder = Path(TestUtils.get_local_test_data_dir("cpt/bro_xml"))
            filename = "cpt_with_water.xml"
        elif isinstance(request.param, GefCpt):
            test_folder = Path(TestUtils.get_local_test_data_dir("cpt/gef"))
            filename = "cpt_with_water.gef"
        else:
            return None
        cpt = request.param
        test_file = test_folder / filename
        cpt.read(test_file)
        cpt.pre_process_data()
        return cpt

    @pytest.fixture
    def plot_settings(self):
        """
        Sets default plot settings.
        :return:
        """
        return PlotSettings()

    @pytest.mark.unittest
    def test_generate_plot_raises_error_when_qc_tip_missing(
        self, cpt_with_water, plot_settings
    ):
        """
        Test that generate_plot raises ValueError when qc is in graph_settings but tip data is missing
        """
        test_cpt = cpt_with_water.copy()
        test_cpt.tip = None

        ylims = plot_cpt.get_y_lims(test_cpt, plot_settings.general_settings)

        with pytest.raises(
            ValueError,
            match="Tip data is not available for plotting, this is required for plotting.",
        ):
            plot_cpt.generate_plot(
                test_cpt, plot_settings.general_settings, ylims[0], ylims, 0
            )

    @pytest.mark.unittest
    def test_generate_plot_skips_unavailable_data(self, cpt_with_water, plot_settings):
        """
        Test that generate_plot successfully creates a plot while skipping unavailable data types
        """
        # Remove friction data so friction graphs are skipped
        test_cpt = cpt_with_water.copy()
        test_cpt.friction = None
        test_cpt.friction_nbr = None

        ylims = plot_cpt.get_y_lims(test_cpt, plot_settings.general_settings)

        # Should not raise an error, just skip the unavailable data
        fig = plot_cpt.generate_plot(
            test_cpt, plot_settings.general_settings, ylims[0], ylims, 0
        )

        assert fig is not None
        # 3 axes as qc, water are available, friction and friction_nbr are not
        # plus the main axis
        assert len(fig.axes) == 3

    @pytest.mark.unittest
    def test_generate_plot_all_data_available(self, cpt_with_water, plot_settings):
        """
        Test that generate_plot successfully creates a plot while all data types are available
        """

        test_cpt = cpt_with_water.copy()

        ylims = plot_cpt.get_y_lims(test_cpt, plot_settings.general_settings)

        # Should not raise an error, just skip the unavailable data
        fig = plot_cpt.generate_plot(
            test_cpt, plot_settings.general_settings, ylims[0], ylims, 0
        )

        assert fig is not None
        # 5 axes as qc, water, friction and friction_nbr are available
        # plus the main axis
        assert len(fig.axes) == 5

    @pytest.mark.unittest
    def test_check_data_availability_qc_with_tip_data(self):
        """
        Test that check_data_availability_for_plotting returns True when tip data is available for qc plotting
        """
        cpt = BroXmlCpt()
        cpt.tip = np.array([1.0, 2.0, 3.0])

        result = plot_cpt.check_data_availability_for_plotting(cpt, "qc")

        assert result is True

    @pytest.mark.unittest
    def test_check_data_availability_qc_without_tip_data(self):
        """
        Test that check_data_availability_for_plotting raises ValueError when tip data is not available for qc plotting
        """
        cpt = BroXmlCpt()
        cpt.tip = None

        with pytest.raises(
            ValueError,
            match="Tip data is not available for plotting, this is required for plotting.",
        ):
            plot_cpt.check_data_availability_for_plotting(cpt, "qc")

    @pytest.mark.unittest
    def test_check_data_availability_friction_with_data(self):
        """
        Test that check_data_availability_for_plotting returns True when friction data is available
        """
        cpt = BroXmlCpt()
        cpt.friction = np.array([0.1, 0.2, 0.3])

        result = plot_cpt.check_data_availability_for_plotting(cpt, "friction")

        assert result is True

    @pytest.mark.unittest
    def test_check_data_availability_friction_without_data(self):
        """
        Test that check_data_availability_for_plotting returns False when friction data is not available
        """
        cpt = BroXmlCpt()
        cpt.friction = None

        result = plot_cpt.check_data_availability_for_plotting(cpt, "friction")

        assert result is False

    @pytest.mark.unittest
    def test_check_data_availability_friction_nbr_with_data(self):
        """
        Test that check_data_availability_for_plotting returns True when friction_nbr data is available
        """
        cpt = BroXmlCpt()
        cpt.friction_nbr = np.array([1.5, 2.0, 2.5])

        result = plot_cpt.check_data_availability_for_plotting(cpt, "friction_nbr")

        assert result is True

    @pytest.mark.unittest
    def test_check_data_availability_friction_nbr_without_data(self):
        """
        Test that check_data_availability_for_plotting returns False when friction_nbr data is not available
        """
        cpt = BroXmlCpt()
        cpt.friction_nbr = None

        result = plot_cpt.check_data_availability_for_plotting(cpt, "friction_nbr")

        assert result is False

    @pytest.mark.unittest
    def test_check_data_availability_inv_friction_nbr_with_both_data(self):
        """
        Test that check_data_availability_for_plotting returns True when both tip and friction data are available for inv_friction_nbr
        """
        cpt = BroXmlCpt()
        cpt.tip = np.array([1.0, 2.0, 3.0])
        cpt.friction = np.array([0.1, 0.2, 0.3])

        result = plot_cpt.check_data_availability_for_plotting(cpt, "inv_friction_nbr")

        assert result is True

    @pytest.mark.unittest
    def test_check_data_availability_inv_friction_nbr_without_tip(self):
        """
        Test that check_data_availability_for_plotting returns False when tip data is missing for inv_friction_nbr
        """
        cpt = BroXmlCpt()
        cpt.tip = None
        cpt.friction = np.array([0.1, 0.2, 0.3])

        result = plot_cpt.check_data_availability_for_plotting(cpt, "inv_friction_nbr")

        assert result is False

    @pytest.mark.unittest
    def test_check_data_availability_inv_friction_nbr_without_friction(self):
        """
        Test that check_data_availability_for_plotting returns False when friction data is missing for inv_friction_nbr
        """
        cpt = BroXmlCpt()
        cpt.tip = np.array([1.0, 2.0, 3.0])
        cpt.friction = None

        result = plot_cpt.check_data_availability_for_plotting(cpt, "inv_friction_nbr")

        assert result is False

    @pytest.mark.unittest
    def test_check_data_availability_inv_friction_nbr_without_both(self):
        """
        Test that check_data_availability_for_plotting returns False when both tip and friction data are missing for inv_friction_nbr
        """
        cpt = BroXmlCpt()
        cpt.tip = None
        cpt.friction = None

        result = plot_cpt.check_data_availability_for_plotting(cpt, "inv_friction_nbr")

        assert result is False

    @pytest.mark.unittest
    def test_check_data_availability_water_with_data(self):
        """
        Test that check_data_availability_for_plotting returns True when water data is available
        """
        cpt = BroXmlCpt()
        cpt.water = np.array([10.0, 20.0, 30.0])

        result = plot_cpt.check_data_availability_for_plotting(cpt, "water")

        assert result is True

    @pytest.mark.unittest
    def test_check_data_availability_water_without_data(self):
        """
        Test that check_data_availability_for_plotting returns False when water data is not available
        """
        cpt = BroXmlCpt()
        cpt.water = None

        result = plot_cpt.check_data_availability_for_plotting(cpt, "water")

        assert result is False

    @pytest.mark.unittest
    def test_check_data_availability_unknown_key(self):
        """
        Test that check_data_availability_for_plotting returns True for unknown keys (default behavior)
        """
        cpt = BroXmlCpt()

        result = plot_cpt.check_data_availability_for_plotting(cpt, "unknown_key")

        assert result is True

    @pytest.mark.unittest
    def test_check_data_availability_qc_raises_value_error_without_tip(self):
        """
        Test that check_data_availability_for_plotting raises ValueError with correct message
        when tip data is not available for qc plotting
        """
        cpt = BroXmlCpt()
        cpt.tip = None

        with pytest.raises(
            ValueError,
            match="Tip data is not available for plotting, this is required for plotting.",
        ):
            plot_cpt.check_data_availability_for_plotting(cpt, "qc")

    @pytest.mark.unittest
    def test_plot_method_catches_value_error(self, cpt_with_water, capsys):
        """
        Test that the plot method catches ValueError and prints appropriate error message
        """
        test_cpt = cpt_with_water.copy()
        # Set tip to None to trigger ValueError
        test_cpt.tip = None

        # Call plot method - should catch ValueError and print message
        test_cpt.plot(Path("test_output"))

        # Capture printed output
        captured = capsys.readouterr()
        assert "Cpt data and/or settings are not valid for plotting." in captured.out
        assert "Please check the data and settings." in captured.out

    @pytest.mark.unittest
    def test_plot_method_catches_index_error(self, cpt_with_water, capsys):
        """
        Test that the plot method catches IndexError and prints appropriate error message
        """
        test_cpt = cpt_with_water.copy()
        test_cpt.friction = test_cpt.friction[:100]
        # Mock plot_cpt_norm to raise IndexError
        test_cpt.plot(Path("test_output"))
        # Capture printed output
        captured = capsys.readouterr()
        assert (
            "Cpt data and/or settings are not valid for plotting. Please check the data and settings."
            in captured.out
        )
        assert (
            "Property friction does not have the same size as the other properties"
            in captured.out
        )

    @pytest.mark.unittest
    def test_plot_method_catches_permission_error(self, cpt_with_water, capsys, tmp_path):
        """
        Test that the plot method catches PermissionError and prints the error
        """

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a read-only file with the expected output filename to trigger PermissionError
        output_file = output_dir / f"{cpt_with_water.name}.pdf"
        output_file.write_text("dummy")

        # Make the file read-only
        os.chmod(output_file, stat.S_IREAD)

        try:
            # Call plot method - should catch PermissionError when trying to overwrite read-only file
            cpt_with_water.plot(output_dir)

            # Capture printed output
            captured = capsys.readouterr()
            assert "Permission denied" in captured.out
        finally:
            # Restore write permissions for cleanup
            os.chmod(output_file, stat.S_IWRITE | stat.S_IREAD)

    @pytest.mark.unittest
    def test_plot_method_catches_generic_exception(self, cpt_with_water, capsys):
        """
        Test that the plot method catches unexpected exceptions and reports them
        """
        test_cpt = cpt_with_water.copy()
        # Introduce an unexpected error by setting an invalid type
        test_cpt.name = 12345  # Invalid type to trigger exception

        test_cpt.plot(Path("test_output"))

        # Capture printed output
        captured = capsys.readouterr()
        assert "An unexpected error occurred:" in captured.out

    @pytest.mark.unittest
    def test_plot_method_successful_execution(self, tmp_path):
        """
        Test that the plot method executes successfully with valid data
        """
        # Create a fresh CPT instance with all required data
        test_folder = Path(TestUtils.get_local_test_data_dir("cpt/bro_xml"))
        filename = "cpt_with_water.xml"
        test_file = test_folder / filename

        cpt = BroXmlCpt()
        cpt.read(test_file)
        cpt.pre_process_data()

        # Ensure all required data is present
        assert cpt.tip is not None
        assert cpt.water is not None
        assert cpt.depth_to_reference is not None

        # Call plot method - should execute without errors
        output_dir = tmp_path / "test_output"
        output_dir.mkdir()
        cpt.plot(output_dir)

        # Check that a PDF file was created
        pdf_files = list(output_dir.glob("*.pdf"))
        assert len(pdf_files) > 0, "No PDF file was created"
