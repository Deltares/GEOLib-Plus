from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from geolib_plus.plot_utils import (
    CALIBRATED_LENGTH_FIGURE_SIZE,
    calculate_top_spine_position,
    create_predrilled_depth_line_and_box,
    set_x_axis,
)


class TestSetXAxis:
    """Test suite for the `set_x_axis` function."""

    @pytest.fixture
    def mock_ax(self) -> plt.Axes:
        """Fixture to provide a mocked matplotlib Axes."""
        fig, ax = plt.subplots()
        return ax

    @pytest.fixture
    def sample_graph(self) -> dict:
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
    def sample_settings(self) -> dict:
        """Fixture to provide sample settings dictionary."""
        return {
            "language": "en",
            "nbr_scale_units": 5,
            "font_size_labels": 10,
            "extra_label_spacing": 5,
        }

    @pytest.fixture
    def ylim(self) -> list:
        """Fixture to provide a sample ylim."""
        return [0, 10]

    @pytest.fixture
    def mock_cpt(self) -> MagicMock:
        """Fixture to create a mock CPT object."""
        cpt = MagicMock()
        cpt.local_reference_level = 10.0
        cpt.predrilled_z = 1.0  # Default to a value > 0.5
        return cpt

    @pytest.mark.unittest
    def test_primary_x_axis(
        self, mock_ax: plt.Axes, sample_graph: dict, sample_settings: dict, ylim: list
    ) -> None:
        """Test `set_x_axis` with primary x-axis type."""
        set_x_axis(mock_ax, sample_graph, sample_settings, ylim)

        # Check x-axis limits
        expected_xlim = [0, 20]
        assert mock_ax.get_xlim() == pytest.approx(
            expected_xlim
        ), "X-axis limits are incorrect."

        # Check x-axis ticks
        assert (
            mock_ax.get_xticks().tolist() == sample_graph["ticks"]
        ), "X-axis ticks are incorrect."

        # Check x-axis tick color
        for tick in mock_ax.xaxis.get_ticklines():
            assert (
                tick.get_color() == sample_graph["graph_color"]
            ), "Tick color is incorrect."
        # default location of the label is bottom
        assert mock_ax.spines["top"]._position == (
            "outward",
            0,
        ), "Top spine position should be outward 0."

    @pytest.mark.unittest
    def test_secondary_x_axis(
        self, mock_ax: plt.Axes, sample_graph: dict, sample_settings: dict, ylim: list
    ) -> None:
        """Test `set_x_axis` with secondary x-axis type."""
        sample_graph["x_axis_type"] = "secondary"
        set_x_axis(mock_ax, sample_graph, sample_settings, ylim)

        # Check that the top spine is set to visible
        assert mock_ax.spines[
            "top"
        ].get_visible(), "Top spine should be visible for secondary axis."

        # Check x-axis limits
        expected_xlim = [0, 20]
        assert mock_ax.get_xlim() == pytest.approx(
            expected_xlim
        ), "X-axis limits are incorrect for secondary axis."

        # Check the position of the top spine
        test_position = 1 + 0.06 * 21 / (ylim[0] - ylim[1]) + 5 * 21 / (ylim[0] - ylim[1])
        assert mock_ax.spines["top"]._position == (
            "axes",
            pytest.approx(test_position),
        ), "Top spine position is incorrect for secondary axis."

    @pytest.mark.unittest
    def test_no_overlap_ticks(
        self, mock_ax: plt.Axes, sample_graph: dict, sample_settings: dict, ylim: list
    ) -> None:
        """Test that tick labels do not overlap."""
        set_x_axis(mock_ax, sample_graph, sample_settings, ylim)

        # Check if overlapping tick labels were removed
        tick_labels = [label.get_text() for label in mock_ax.get_xticklabels()]
        assert all(
            label == "" or label.isspace() or label.isprintable() for label in tick_labels
        ), "Overlapping labels should be removed."

    @pytest.mark.unittest
    def test_no_overlap_ticks_fine_spacing(
        self, mock_ax: plt.Axes, sample_graph: dict, sample_settings: dict, ylim: list
    ) -> None:
        """Test that tick labels overlap when spacing is fine."""
        # set ticks
        sample_graph["ticks"] = np.arange(0, 20, 0.1).tolist()
        set_x_axis(mock_ax, sample_graph, sample_settings, ylim)

        tick_labels = [label.get_text() for label in mock_ax.get_xticklabels()]
        assert len(tick_labels) == len(
            sample_graph["ticks"]
        ), "All tick labels should be shown."
        assert any(
            label != "" and not label.isspace() and label.isprintable()
            for label in tick_labels
        ), "No overlapping labels should be shown."

    @pytest.mark.unittest
    def test_create_predrilled_depth_line_and_box(
        self, mock_cpt: MagicMock, mock_ax: plt.Axes
    ) -> None:
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
    def test_create_predrilled_depth_line_and_box_position(self, mock_cpt: MagicMock, mock_ax: plt.Axes, sample_graph: dict, sample_settings: dict, ylim: list) -> None:
        xlim = [0, 5]
        language = "English"

        # Call the function to set the x-axis and create the predrilled depth line and box
        set_x_axis(mock_ax, sample_graph, sample_settings, ylim)
        create_predrilled_depth_line_and_box(mock_cpt, mock_ax, xlim, language)

        # Verify the position of the textbox
        artists = mock_ax.artists
        textbox = artists[1]
        bounds_bbox = textbox.get_bbox_to_anchor()._bbox.bounds
        expected_position_x = 5 / 16
        expected_position_y = 0.5609
        assert bounds_bbox[0] == pytest.approx(expected_position_x, rel=1e-2), "Textbox X position is incorrect."
        assert bounds_bbox[1] == pytest.approx(expected_position_y, rel=1e-2), "Textbox Y position is incorrect."


    @pytest.mark.unittest
    def test_calculate_top_spine_position_normal_range(self) -> None:
        """Test with standard y_range (50m)."""
        ylim = [10.0, -40.0]  # 50m range

        result = calculate_top_spine_position(ylim)

        y_range = ylim[0] - ylim[1]  # 50
        expected_vertical_spacing = (0.06 * CALIBRATED_LENGTH_FIGURE_SIZE) / y_range
        expected_extra_spacing = (0.02 * CALIBRATED_LENGTH_FIGURE_SIZE) / y_range
        expected = 1.0 + expected_vertical_spacing + expected_extra_spacing

        assert result == pytest.approx(expected)

    @pytest.mark.unittest
    def test_spine_position_always_greater_than_one(self) -> None:
        """Test that spine position is always > 1.0 (above the plot)."""
        test_ylims = [
            [10.0, -10.0],  # 20m
            [10.0, -40.0],  # 50m
            [5.0, -95.0],  # 100m
            [100.0, 0.0],  # 100m
            [0.0, -200.0],  # 200m
        ]

        for ylim in test_ylims:
            result = calculate_top_spine_position(ylim)
            assert (
                result > 1.0
            ), f"Spine position should be > 1.0 for ylim={ylim}, got {result}"
