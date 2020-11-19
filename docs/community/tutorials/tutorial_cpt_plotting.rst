.. tutorialcpt:

Tutorial Cpt plotting using GEOLIB+
=========================================

After the cpt is set up by following the steps described in:

.. toctree::
   :maxdepth: 3

   tutorial_setup_cpt

The user can plot a cpt by following the next step:


1. The plotting method  :func:`~geolib_plus.plot_cpt.plot_cpt_norm` can be called to plot and save the figure.
This function works for cpts based on xml-bro files and based on gef files.
Before the function can be called, it is required to define the directory where the figures will be saved.
Furthermore, the plot settings have to be defined. Plot settings are defined in the :class:`~geolib_plus.plot_settings.PlotSettings` class.
The class contains a lot of information, luckily there is a function to assign default settings, :func:`~geolib_plus.plot_settings.PlotSettings.assign_default_settings`.
Below an example is given on how the data of the cpt can be plotted.

.. code-block:: python

    from pathlib import Path

    import geolib_plus.plot_cpt as plot_cpt
    from geolib_plus.plot_settings import PlotSettings

    output_path = Path(".")
    plot_settings = PlotSettings()
    plot_settings.assign_default_settings()

    # plot cpt according to NEN-standards
    plot_cpt.plot_cpt_norm(cpt, output_path, plot_settings.general_settings)



