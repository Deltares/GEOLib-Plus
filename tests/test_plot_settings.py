import pytest

from geolib_plus import plot_settings


class TestPlotSettings:
    def setUp(self):
        self.settings = plot_settings.PlotSettings()

        self.settings.assign_default_settings()

    def test_set_inversed_friction_number_in_plot(self):
        """
        Test if inversed friction number key is added and in settings list. And check if friction number is removed.

        :return:
        """
        self.setUp()
        data_keys = self.settings.data_keys
        initial_keys = [data_keys[0], data_keys[1], data_keys[2], data_keys[3]]
        present_keys = [data_keys[0], data_keys[1], data_keys[3], data_keys[4]]

        for key in initial_keys:
            assert key in self.settings.general_settings["graph_settings"]
        assert data_keys[4] not in self.settings.general_settings["graph_settings"]

        self.settings.set_inversed_friction_number_in_plot()

        assert len(self.settings.general_settings["graph_settings"]) == 4
        for key in present_keys:
            assert key in self.settings.general_settings["graph_settings"]

        assert data_keys[2] not in self.settings.general_settings["graph_settings"]

    def test_reset_friction_number_in_plot(self):
        """
        Test if friction number key is added and in settings list. And check if inversed friction number is removed.
        :return:
        """

        self.setUp()
        data_keys = self.settings.data_keys
        present_keys = [data_keys[0], data_keys[1], data_keys[2], data_keys[3]]

        self.settings.set_inversed_friction_number_in_plot()
        self.settings.set_friction_number_in_plot()

        for key in present_keys:
            assert key in self.settings.general_settings["graph_settings"]
        assert data_keys[4] not in self.settings.general_settings["graph_settings"]

