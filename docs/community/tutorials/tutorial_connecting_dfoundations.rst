.. tutorialcpt:

Create a D-Foundations soil profile using GEOLIB+
=================================================

GEOLIB+ contains functionality that lets the user create a D-Foundations profile using inputs from GEOLIB+.
To do that the user needs to read and interpretate a cpt using the GEOLIB+ module.


.. code-block:: python

    # import relevant packages
    from pathlib import Path
    from geolib_plus.gef_cpt import GefCpt
    from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation

    cpt_file_gef = Path("cpt", "gef", "test_cpt.gef")  
    # initialize model
    cpt_gef = GefCpt()
    cpt_gef.read(cpt_file_gef)
    cpt_gef.pre_process_data()
    # interpret data
    interpreter = RobertsonCptInterpretation()
    cpt_gef.interpret_cpt(interpreter)


Simply by using the function :func:`~geolib_plus.geolib_connections.DFoundationsConnector.create_profile_for_d_foundations` , the user
can produce the profile and soils, inputs that are required for setting up a DFoundations GEOLIB model. Note that:

1. The layers produced by the :func:`~geolib_plus.geolib_connections.DFoundationsConnector.create_profile_for_d_foundations` function have their depths taken from the cpt.depth_merged variable.
2. The soils procuded have names which are taken from the cpt.lkithology_merged value.
3. Function :func:`~geolib_plus.geolib_connections.DFoundationsConnector.create_profile_for_d_foundations` does not fill out soil parameters automatically.

.. code-block:: python
        
    from geolib_plus.geolib_connections import DFoundationsConnector
    # run test
    profile, soils = DFoundationsConnector.create_profile_for_d_foundations(cpt_gef)

After generating the profile, soils from the cpt the user can parse them in a GEOLIB model. 
This can be demostrated in the following example. Note that in the following code block the 
DFoundations model is not created from scratch but parsed from an arleady existing file.
The new file can be later serialized.

.. code-block:: python

        import geolib

        # initialize geolib model
        test_dfoundations = Path("geolib_example.foi")
        assert test_dfoundations.is_file()
        dfoundations_model = geolib.models.DFoundationsModel()
        # parse
        dfoundations_model.parse(test_dfoundations)
        # add soils from cpt
        for soil in soils:
            dfoundations_model.add_soil(soil)
        # add profile to the dfoundations model
        profile_name = dfoundations_model.add_profile(profile)
        # save updated file
        dfoundations_model.serialize(
            Path(
                "added_cpt_model.foi",
            )
        )

By opening the created DFoundations file the inputted profile and soils can be inspected.


.. image:: ..\\..\\_static\\DFoundations_profile.png
  :width: 400
  :alt: DFoundations_profile