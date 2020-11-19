import pytest
from pathlib import Path
import numpy as np

from geolib_plus.gef_cpt import GefCpt
from geolib_plus.bro_xml_cpt import BroXmlCpt
from geolib_plus.plot_settings import PlotSettings
import geolib_plus.plot_cpt as plot_cpt

from tests.utils import TestUtils


class TestPlotCpt():

    @pytest.mark.unittest
    def test_get_ylims_greater_than_data(self, cpt_with_water):
        """
        Assert positive buffer at top and length graph larger than the cpt data
        :return:
        """
        settings = {"plot_size": "a4",
                    "vertical_settings": "",
                    }

        vertical_settings = {"buffer_at_top": 1,
                             "length_graph": 100,
                             "top_type": "relative",
                             "repeated_distance": 0
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
        settings = {"plot_size": "a4",
                    "vertical_settings": "",
                    }

        vertical_settings = {"buffer_at_top": -1,
                             "length_graph": 100,
                             "top_type": "relative",
                             "repeated_distance": 0
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

        settings = {"plot_size": "a4",
                    "vertical_settings": "",
                    }

        vertical_settings = {"buffer_at_top": 0,
                             "length_graph": 10,
                             "top_type": "relative",
                             "repeated_distance": 0
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

        settings = {"plot_size": "a4",
                    "vertical_settings": "",
                    }

        vertical_settings = {"buffer_at_top": 0,
                             "length_graph": 10,
                             "top_type": "relative",
                             "repeated_distance": 1
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

        settings = {"plot_size": "a4",
                    "vertical_settings": "",
                    }

        vertical_settings = {"absolute_top_level": 0,
                             "length_graph": 10,
                             "top_type": "absolute",
                             "repeated_distance": 0
                             }

        settings["vertical_settings"] = vertical_settings

        ylims = plot_cpt.get_y_lims(cpt_with_water, settings)

        assert pytest.approx(ylims[0][0]) == 0.0
        assert pytest.approx(ylims[0][1]) == -10.0
        assert pytest.approx(ylims[1][0]) == -10.0
        assert pytest.approx(ylims[1][1]) == -20.0
        assert pytest.approx(ylims[2][0]) == -20.0
        assert pytest.approx(ylims[2][1]) == -30.0

    def test_trim_cpt_data_within_thresholds(self, cpt_with_water):
        """
        Test time cpt
        :return:
        """
        settings = {'data_key': 'qc',
                    'threshold': [0, 32],
                    'unit_converter': 1}

        vertical_settings = {"spacing_shown_cut_off_value": 1}

        # assert all data is within threshold
        trimmed_values, shown_values, y_coord_shown_value, depth_in_range, inclination_in_range = \
            plot_cpt.trim_cpt_data(settings, vertical_settings, cpt_with_water, [-1, -101])

        assert shown_values.size == 0
        assert y_coord_shown_value.size == 0
        for idx, data in enumerate(depth_in_range):
            assert data == cpt_with_water.depth_to_reference[idx]
            assert trimmed_values[idx] == cpt_with_water.tip[idx]

    def test_trim_cpt_data_partly_outside_thresholds(self):
        """
        Test trimmed cpt data where the original data falls partly outside the thresholds
        :return:
        """
        settings = {'data_key': 'qc',
                    'threshold': [0, 0.7],
                    'unit_converter': 1}

        vertical_settings = {"spacing_shown_cut_off_value": 1}

        cpt = BroXmlCpt()

        # set up cpt
        cpt.depth_to_reference = np.linspace(0, -10, 11)
        cpt.tip = np.sin(cpt.depth_to_reference * 1/4 * np.pi)
        cpt.inclination_resultant = np.zeros(11)

        trimmed_values, shown_values, y_coord_shown_value, depth_in_range, inclination_in_range = \
            plot_cpt.trim_cpt_data(settings, vertical_settings, cpt, [0, -11])

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

        output_file_name = cpt.name + '.pdf'
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

        output_file_name = cpt.name + '.pdf'
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

        output_file_name = cpt.name + '.pdf'
        assert Path(output_path / output_file_name).is_file()
        (output_path / output_file_name).unlink()


    @pytest.mark.workinprogress
    def test_run_multiple_plots(self, plot_settings):
        from os import listdir
        from os.path import isfile, join
        # cpt_folder = r"D:\software_development\geolib-plus\tests\test_files\cpt\bro_xml\XML-viewer-Foutmelding not valid\XML-viewer-Foutmelding not valid"
        cpt_folder = r"D:\software_development\geolib-plus\tests\test_files\cpt\bro_xml\tmp\tmp"

        # D:\software_development\geolib - plus\tests\test_files\cpt\bro_xml
        cpt_names = [f for f in listdir(cpt_folder) if isfile(join(cpt_folder, f)) and f.endswith('.xml')]

        # cpt = BroXmlCpt()
        # cpt.read(test_file)
        # cpt.pre_process_data()
        cpts= []
        for cpt_name in cpt_names:
            cpt = BroXmlCpt()
            cpt.read(Path(join(cpt_folder, cpt_name)))
            cpt.pre_process_data()
            plot_cpt.plot_cpt_norm(cpt, Path(cpt_folder), plot_settings.general_settings)
            cpts.append(cpt)


    @pytest.fixture(scope="session", params=[BroXmlCpt(), GefCpt()])
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

    @pytest.fixture(scope="session", params=[BroXmlCpt(), GefCpt()])
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
        plot_settings = PlotSettings()
        plot_settings.assign_default_settings()
        return plot_settings