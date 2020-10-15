import pytest
from geolib_plus import plot_cpt


# class TestPlotCpt:
#     @pytest.mark.unittest
#     def test_plot_cpt_unit_tests(self):
#         raise NotImplementedError

import unittest
import os

from pathlib import Path

from teamcity import is_running_under_teamcity
from teamcity.unittestpy import TeamcityTestRunner

import geolib_plus.plot_cpt as plot_cpt
import geolib_plus.plot_settings as plot_settings
import ours.CPTtool.bro as bro
import ours.CPTtool.cpt_module as cpt_module
import numpy as np
from geolib_plus.BRO_XML_CPT import bro_utils as bu
from geolib_plus.GEF_CPT import gef_cpt as g_cpt

class TestPlotCpt(unittest.TestCase):
    def setUp(self):

        # get cpt xml files
        cpt_name = r"D:\software_development\geolib-plus\tests\test_files\cpt\bro_xml/cpt_with_water.xml"
        cpt_name = r"D:\software_development\geolib-plus\tests\test_files\cpt\gef/cpt_with_water.gef"
        output_folder = r'D:\software_development\geolib-plus\tests\test_output'

        # read xml files in byte-string
        # cpt_byte_string = bro.xml_to_byte_string(cpt_name)

        # parse bro data
        gef_cpt = g_cpt.GefCpt()
        gef_cpt.read(Path(cpt_name))
        self.cpt = gef_cpt
        # # data_cpt = bu.parse_bro_xml(cpt_byte_string)
        # data_cpt['predrilled_z'] = 0.
        # self.cpt = cpt_module.CPT(output_folder)
        # self.cpt.parse_bro(data_cpt,convert_to_kPa=False)

        # self.gef_cpt =


    def test_get_ylims_greater_than_data(self):
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

        ylims = plot_cpt.get_y_lims(self.cpt, settings)

        self.assertAlmostEqual(ylims[0][0], -1.0)
        self.assertAlmostEqual(ylims[0][1], -101.0)

    def test_get_ylims_negative_buffer_at_top(self):
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

        ylims = plot_cpt.get_y_lims(self.cpt, settings)

        self.assertAlmostEqual(ylims[0][0], -3.0)
        self.assertAlmostEqual(ylims[0][1], -103.0)


    def test_get_ylims_smaller_than_data(self):
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

        ylims = plot_cpt.get_y_lims(self.cpt, settings)

        self.assertAlmostEqual(ylims[0][0], -2.0)
        self.assertAlmostEqual(ylims[0][1], -12.0)
        self.assertAlmostEqual(ylims[1][0], -12.0)
        self.assertAlmostEqual(ylims[1][1], -22.0)

    def test_get_ylims_repeated_distance(self):
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

        ylims = plot_cpt.get_y_lims(self.cpt, settings)

        self.assertAlmostEqual(ylims[0][0], -2.0)
        self.assertAlmostEqual(ylims[0][1], -12.0)
        self.assertAlmostEqual(ylims[1][0], -11.0)
        self.assertAlmostEqual(ylims[1][1], -21.0)


    def test_get_ylims_absolute_top_level(self):
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

        ylims = plot_cpt.get_y_lims(self.cpt, settings)

        self.assertAlmostEqual(ylims[0][0], 0.0)
        self.assertAlmostEqual(ylims[0][1], -10.0)
        self.assertAlmostEqual(ylims[1][0], -10.0)
        self.assertAlmostEqual(ylims[1][1], -20.0)
        self.assertAlmostEqual(ylims[2][0], -20.0)
        self.assertAlmostEqual(ylims[2][1], -30.0)

    def test_trim_cpt_data_within_thresholds(self):
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
            plot_cpt.trim_cpt_data(settings, vertical_settings, self.cpt, [-1, -101])

        self.assertEqual(shown_values, [])
        self.assertEqual(y_coord_shown_value, [])
        for idx, data in enumerate(depth_in_range):
            self.assertEqual(data, self.cpt.depth_to_reference[idx])
            self.assertEqual(trimmed_values[idx], self.cpt.tip[idx])

    def test_trim_cpt_data_partly_outside_thresholds(self):
        """
        Test trimmed cpt data where the original data falls partly outside the thresholds
        :return:
        """
        settings = {'data_key': 'qc',
                    'threshold': [0, 0.7],
                    'unit_converter': 1}

        vertical_settings = {"spacing_shown_cut_off_value": 1}

        output_folder = r'./res'

        self.cpt = cpt_module.CPT(output_folder)

        # set up cpt
        self.cpt.depth_to_reference = np.linspace(0, -10, 11)
        self.cpt.tip = np.sin(self.cpt.depth_to_reference * 1/4 * np.pi)
        self.cpt.inclination_resultant = np.zeros(11)

        trimmed_values, shown_values, y_coord_shown_value, depth_in_range, inclination_in_range = \
            plot_cpt.trim_cpt_data(settings, vertical_settings, self.cpt, [0, -11])

        # Assert if trimmed values are as expected
        expected_trimmed_values = np.array([0, 0, 0, 0, 0.7, 0.7, 0.7, 0, 0, 0])
        for idx, trimmed_value in enumerate(trimmed_values):
            self.assertAlmostEqual(expected_trimmed_values[idx], trimmed_value)

    def test_generate_fig_with_default_settings(self):
        settings = plot_settings.PlotSettings()
        settings.assign_default_settings()

        output_folder = r'D:\software_development\geolib-plus\tests\test_output'

        plot_cpt.plot_cpt_norm(self.cpt, settings.general_settings, output_folder)

        self.assertTrue(os.path.isfile(os.path.join(output_folder, self.cpt.name + '.pdf')))
        # os.remove(os.path.join(self.cpt.output_folder, self.cpt.name + '.pdf'))

    def test_generate_fig_with_inverse_friction_nbr(self):
        settings = plot_settings.PlotSettings()
        settings.assign_default_settings()
        settings.set_inversed_friction_number_in_plot()

        plot_cpt.plot_cpt_norm(self.cpt, settings.general_settings)

        self.assertTrue(os.path.isfile(os.path.join(self.cpt.output_folder, self.cpt.name + '.pdf')))
        # os.remove(os.path.join(self.cpt.output_folder, self.cpt.name + '.pdf'))


if __name__ == '__main__':  # pragma: no cover
    if is_running_under_teamcity():
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)