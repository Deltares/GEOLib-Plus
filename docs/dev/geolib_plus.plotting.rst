geolib\_plus.plot\_plotting package
===================================

geolib\_plus.plot\_cpt module
-----------------------------

.. automodule:: geolib_plus.plot_cpt
   :members:
   :undoc-members:
   :show-inheritance:




geolib\_plus.plot\_settings module
----------------------------------

.. autoclass:: geolib_plus.plot_settings.PlotSettings
   :members:


When the :class:`~geolib_plus.plot_settings.PlotSettings` class is initialised, the class is filled with default settings.
However, the user might want to change the default settings. Below an overview is given on how to change the default settings.

The plot settings are built up out of several dictionaries. Below an explanation is given
of the dictionaries.

All the settings are stored in :class:`~geolib_plus.plot_settings.PlotSettings.general_settings`. It is recommended to
not change the settings in the variable, besides the language and the font size of the labels. Below an example is given
how to change this.

.. code-block:: python

    from geolib_plus.plot_settings import PlotSettings
    plot_settings = PlotSettings()

    # set language to english
    plot_settings.general_settings["language"] = plot_settings.languages[1]
    # set font size of the labels
    plot_settings.general_settings["font_size_labels"] = 10

The cpt plotter allows for 5 different data fields: cone resistance, friction resistance, friction number,
inversed friction number and pore pressure. The settings for these data fields can be altered in the dictionaries:
:class:`~geolib_plus.plot_settings.PlotSettings.plot_qc_settings`, :class:`~geolib_plus.plot_settings.PlotSettings.plot_friction_settings`,
:class:`~geolib_plus.plot_settings.PlotSettings.plot_friction_nbr_settings`, :class:`~geolib_plus.plot_settings.PlotSettings.plot_inv_friction_nbr_settings` and
:class:`~geolib_plus.plot_settings.PlotSettings.plot_water_settings`.

Below a code snippet is presented on how to change cone resistance settings

.. code-block:: python

    from geolib_plus.plot_settings import PlotSettings
    plot_settings = PlotSettings()

    # Set minimum and maximum threshold for shown values, data which surpasses these tresholds is cut off at the
    # respective threshold. The units are equal to the units read from the input file in this case it is in MPa
    plot_settings.plot_qc_settings["threshold"] = [0, 30]

    # Sets the colour of the graph
    plot_settings.plot_qc_settings["graph_color"] = "blue"

    # Sets the style of the line
    plot_settings.plot_qc_settings["line_style"] = "--"

    # Sets the scale of the shown graph. This value represents how much of the respective unit (in this case MPa) is shown
    # per 2 grid cells.
    plot_settings.plot_qc_settings["scale_unit"] = 3

    # Shifts the graph away from the vertical axis. The unit is the same as read from the data, in this case MPa. Note that
    # the absolute shift is based on the scale unit.
    plot_settings.plot_qc_settings["shift_graph"] = 1

    # Sets the ticks at the x-axis. In this case, from 0 to 30 (not including 30) with a step size of 5
    plot_settings.plot_qc_settings["ticks"] = np.arange(0, 30, 5).tolist()

Besides the horizontal data settings, also the vertical settings related to the depth can be altered. Below a code
snipped is shown on how to alter the vertical settings:

.. code-block:: python

    from geolib_plus.plot_settings import PlotSettings
    plot_settings = PlotSettings()

    # sets the type of the top level, top_types[0] for relative top level, top_types[1] for absolute top level
    plot_settings.vertical_settings["top_type"] = plot_settings.top_types[1]

    # Absolute value in meter, of the top of the first graph. This value is used if top type is absolute.
    plot_settings.vertical_settings["absolute_top_level"] =  5

    # Rounded down value of the distance in meter above surface level. This value is used if top type is relative.
    plot_settings.vertical_settings["buffer_at_top"] =  1.5

    # Vertical length of the graph in meter
    plot_settings.vertical_settings["length_graph"] =  1.5

    # The distance in meter which is repeated from one graph to the subsequent graph (in case the data exceeds the bottom
    # of the graph).
    plot_settings.vertical_settings["repeated_distance"] =  1.5

    # Distance in meter of the depth ticks
    plot_settings.vertical_settings["depth_tick_distance"] =  1.5

    # Distance in meter of the inclination ticks if present
    plot_settings.vertical_settings["inclination_tick_distance"] =  1.5

    # Minimal distance in meter between textboxes which show values which exceed the data threshold.
    plot_settings.vertical_settings["spacing_shown_cut_off_value"] =  1.5

Lastly the grid can be altered. It is possible to alter the line distances, the line colours and the line widths. Below
a code snippet is shown which shows how to alter the grid settings. Note that the grid by default refers to the axis
related to the cone resistance data. Therefore the scales of the grid are equal to the scales of the cone resistance data.
The unit is MPa.

.. code-block:: python

    from geolib_plus.plot_settings import PlotSettings
    plot_settings = PlotSettings()


    # Sets the locations of the major grid lines, relative to the cone resistance data
    plot_settings.grid["vertical_major_line_locations"] = [10, 20, 30]

    # Sets the colour of the major grid lines
    plot_settings.grid["vertical_major_line_color"] = "black"

    # Sets the line width of the major grid lines
    plot_settings.grid["vertical_major_line_line_width"] = 1.5

    # sets the distances between minor vertical lines, relative to the cone resistance data. In this case
    # there are 2 sets off minor vertical lines
    plot_settings.grid["vertical_minor_line_distances"] = [2, 1]

    # Sets the colours of all the sets of minor vertical lines
    plot_settings.grid["vertical_minor_line_colors"] = ['black', 'black']

    # Sets the line widths of all the sets of minor vertical lines
    plot_settings.grid["vertical_minor_line_line_widths"] = [1, 0.5]

    # sets the distances between horizontal lines, relative to the depth data. In this case
    # there are 2 sets off horizontal lines
    plot_settings.grid["horizontal_line_distances"] = [2, 1]

    # Sets the colours of all the sets of minor vertical lines
    plot_settings.grid["horizontal_line_colors"] = ['black', 'black']

    # Sets the line widths of all the sets of horizontal lines
    plot_settings.grid["horizontal_line_line_widths"] = [1, 0.5]



geolib\_plus.plot\_utils module
-------------------------------

.. automodule:: geolib_plus.plot_utils
   :members:
   :undoc-members:
   :show-inheritance:

