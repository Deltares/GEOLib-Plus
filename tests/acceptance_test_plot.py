from pathlib import Path

import numpy as np

from geolib_plus.bro_xml_cpt import BroXmlCpt

cpt_file_xml = Path("D:\\bro_xml_viewer\\unit_testing\\unit_testing_files\\xml_example_1\\CPT000000006560_IMBRO_A.xml")
cpt = BroXmlCpt()
cpt.read(cpt_file_xml)
cpt.pre_process_data()

from geolib_plus.plot_settings import PlotSettings

plot_settings = PlotSettings()

# Set minimum and maximum threshold for shown values, data which surpasses these tresholds is cut off at the
# respective threshold. The units are equal to the units read from the input file in this case it is in MPa
cpt.plot_settings.plot_qc_settings["threshold"] = [0, 15]

# Sets the colour of the graph
cpt.plot_settings.plot_qc_settings["graph_color"] = "red"

# Sets the style of the line
cpt.plot_settings.plot_qc_settings["line_style"] = "--"

# Sets the scale of the shown graph. This value represents how much of the respective unit (in this case MPa) is shown
# per 2 grid cells.
cpt.plot_settings.plot_qc_settings["scale_unit"] = 2

# Shifts the graph away from the vertical axis. The unit is the same as read from the data, in this case MPa. Note that
# the absolute shift is based on the scale unit.
cpt.plot_settings.plot_qc_settings["shift_graph"] = 0

# Sets the ticks at the x-axis. In this case, from 0 to 30 (not including 30) with a step size of 5
cpt.plot_settings.plot_qc_settings["ticks"] = np.arange(0, 30, 5).tolist()

cpt.plot_settings.general_settings["language"] = "English"

# sets the type of the top level, top_types[0] for relative top level, top_types[1] for absolute top level
cpt.plot_settings.vertical_settings["top_type"] = plot_settings.top_types[1]

# Absolute value in meter, of the top of the first graph. This value is used if top type is absolute.
cpt.plot_settings.vertical_settings["absolute_top_level"] = 2

# Rounded down value of the distance in meter above surface level. This value is used if top type is relative.
cpt.plot_settings.vertical_settings["buffer_at_top"] = 1.5

# Vertical length of the graph in meter
#cpt.plot_settings.vertical_settings["length_graph"] = 50

# The distance in meter which is repeated from one graph to the subsequent graph (in case the data exceeds the bottom
# of the graph).
cpt.plot_settings.vertical_settings["repeated_distance"] = 1.5

# Distance in meter of the depth ticks
cpt.plot_settings.vertical_settings["depth_tick_distance"] = 1.5

# Distance in meter of the inclination ticks if present
cpt.plot_settings.vertical_settings["inclination_tick_distance"] = 2

# Minimal distance in meter between textboxes which show values which exceed the data threshold.
cpt.plot_settings.vertical_settings["spacing_shown_cut_off_value"] = 1.5

cpt.plot_settings.grid["horizontal_major_line_locations"] = -1 * np.arange(
    -1 * cpt.depth_to_reference[0], -1 * cpt.depth_to_reference[-1], 10
)
cpt.plot_settings.grid["horizontal_major_line_color"] = "black"
cpt.plot_settings.grid["horizontal_major_line_line_width"] = 2

cpt.plot_settings.general_settings["distance_from_plot"] = -0.1
cpt.plot_settings.general_settings["extra_label_spacing"] = 0.02
cpt.plot_settings.general_settings["secondary_top_axis_position"] = 1.08
cpt.plot_settings.general_settings['graph_settings']['qc']['x_axis_type'] = 'primary'
cpt.plot_settings.general_settings['graph_settings']['water']['x_axis_type'] = 'secondary'
cpt.plot_settings.general_settings['graph_settings']['friction']['x_axis_type'] = 'secondary'
cpt.plot_settings.general_settings['graph_settings']['friction_nbr']['x_axis_type'] = 'secondary'
cpt.plot_settings.general_settings['graph_settings']['qc']['position_label'] = 'top_left'
cpt.plot_settings.general_settings['graph_settings']['water']['position_label'] = 'top_middle'
cpt.plot_settings.general_settings['graph_settings']['friction']['position_label'] = 'top_left'
cpt.plot_settings.general_settings['graph_settings']['friction_nbr']['position_label'] = 'top_right'

cpt.plot(Path("test_output"))
