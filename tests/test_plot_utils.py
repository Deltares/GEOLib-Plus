import pytest
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
from geolib_plus.plot_utils import set_x_axis, create_predrilled_depth_line_and_box
import numpy as np


class TestSetXAxis:
    """Test suite for the `set_x_axis` function."""

    @pytest.fixture
    def mock_ax(self):
        """Fixture to provide a mocked matplotlib Axes."""
        fig, ax = plt.subplots()
        return ax

    @pytest.fixture
    def sample_graph(self):
        """Fixture to provide a sample graph dictionary."""
        return {
            "label": {"en": "Sample X-Axis", "nl": "Voorbeeld X-As"},
            "shift_graph": 0,
            "unit_converter": 1.0,
            "ticks": [0, 5, 10, 15, 20],
            "x_axis_type": "primary",
            "graph_color": "blue",
            "scale_unit": 1,
            "line_style": "solid",
            "position_label": "bottom",
        }

    @pytest.fixture
    def sample_settings(self):
        """Fixture to provide sample settings dictionary."""
        return {
            "language": "en",
            "nbr_scale_units": 5,
            "secondary_top_axis_position": 1.2,
            "font_size_labels": 10,
            "extra_label_spacing": 5,
        }

    @pytest.fixture
    def ylim(self):
        """Fixture to provide a sample ylim."""
        return [0, 10]

    @pytest.fixture
    def mock_cpt(self):
        """Fixture to create a mock CPT object."""
        cpt = MagicMock()
        cpt.local_reference_level = 10.0
        cpt.predrilled_z = 1.0  # Default to a value > 0.5
        return cpt

    @pytest.mark.unittest
    def test_primary_x_axis(self, mock_ax, sample_graph, sample_settings, ylim):
        """Test `set_x_axis` with primary x-axis type."""
        ax = set_x_axis(mock_ax, sample_graph, sample_settings, ylim)

        # Check x-axis limits
        expected_xlim = [0, 20]
        assert ax.get_xlim() == pytest.approx(expected_xlim), "X-axis limits are incorrect."

        # Check x-axis ticks
        assert ax.get_xticks().tolist() == sample_graph["ticks"], "X-axis ticks are incorrect."

        # Check x-axis tick color
        for tick in ax.xaxis.get_ticklines():
            assert tick.get_color() == sample_graph["graph_color"], "Tick color is incorrect."

    @pytest.mark.unittest
    def test_secondary_x_axis(self, mock_ax, sample_graph, sample_settings, ylim):
        """Test `set_x_axis` with secondary x-axis type."""
        sample_graph["x_axis_type"] = "secondary"
        ax = set_x_axis(mock_ax, sample_graph, sample_settings, ylim)

        # Check that the top spine is set to visible
        assert ax.spines["top"].get_visible(), "Top spine should be visible for secondary axis."

        # Check x-axis limits
        expected_xlim = [0, 20]
        assert ax.get_xlim() == pytest.approx(expected_xlim), "X-axis limits are incorrect for secondary axis."

    @pytest.mark.unittest
    def test_no_overlap_ticks(self, mock_ax, sample_graph, sample_settings, ylim):
        """Test that tick labels do not overlap."""
        ax = set_x_axis(mock_ax, sample_graph, sample_settings, ylim)

        # Check if overlapping tick labels were removed
        tick_labels = [label.get_text() for label in ax.get_xticklabels()]
        assert all(label == "" or label.isspace() or label.isprintable() for label in tick_labels), \
            "Overlapping labels should be removed."

    @pytest.mark.unittest
    def test_no_overlap_ticks_fine_spacing(self, mock_ax, sample_graph, sample_settings, ylim):
        """Test that tick labels overlap when spacing is fine."""
        # set ticks
        sample_graph["ticks"] = np.arange(0, 20, 0.1).tolist()
        ax = set_x_axis(mock_ax, sample_graph, sample_settings, ylim)

        tick_labels = [label.get_text() for label in ax.get_xticklabels()]
        assert len(tick_labels) == len(sample_graph["ticks"]), "All tick labels should be shown."
        assert any(label != "" and not label.isspace() and label.isprintable() for label in tick_labels), \
            "No overlapping labels should be shown."

    @pytest.mark.unittest
    def test_create_predrilled_depth_line_and_box(self, mock_cpt, mock_ax):
        xlim = [0, 5]
        language = "English"

        # Call the function
        create_predrilled_depth_line_and_box(mock_cpt, mock_ax, xlim, language)

        # Verify that a line is added
        lines = mock_ax.get_lines()
        assert len(lines) == 1  # One line should be added

        # Verify the line's data
        line = lines[0]
        assert line.get_xdata().tolist() == [xlim[0], xlim[0]]
        assert line.get_ydata().tolist() == [
            mock_cpt.local_reference_level,
            mock_cpt.local_reference_level - mock_cpt.predrilled_z,
        ]

        # Verify that a textbox is added
        artists = mock_ax.artists
        assert len(artists) == 1  # One textbox should be added

    @pytest.mark.unittest
    def test_no_plot_for_small_predrilled_depth(self, mock_cpt, mock_ax):
        # Set predrilled_z to a value less than 0.5
        mock_cpt.predrilled_z = 0.4

        xlim = [0, 5]
        language = "English"

        # Call the function
        create_predrilled_depth_line_and_box(mock_cpt, mock_ax, xlim, language)

        # Verify that no line is added
        lines = mock_ax.get_lines()
        assert len(lines) == 0

        # Verify that no textbox is added
        artists = mock_ax.artists
        assert len(artists) == 0