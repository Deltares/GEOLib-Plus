.. tutorialcpt:

Tutorial Defining and calculating parameters for the Hardening Soil model
=========================================================================

Part of the GEOLIB+ package is also a module that can determine certain parameters
used as input of the Hardening Soil(HS) model. To do that we can use the 
:class:`~geolib_plus.hardening_soil_model_parameters.HardeningSoilModelParameters` class.
To begin with the HS model calculation the :class:`~geolib_plus.hardening_soil_model_parameters.HardeningSoilModelParameters` class
should be initialized.

    .. code-block:: python

        from geolib_plus.hardening_soil_model_parameters import HardeningSoilModelParameters
        # initialize class
        HS_model = HardeningSoilModelParameters()

Then the inputs needed for the calculation can be defined by the user. Let's assume first 
that the user wants to use the compressibility coefficient values to calculate the stiffnesses 
related to the HS model. Also let's assume that these input concern only one soil type and 
therefore can be defined as simple float numbers.

    .. code-block:: python
        
        # define inputs for the model as floats
        HS_model.eo = 0.8
        HS_model.sigma_ref_v = 10
        HS_model.v_ur = 0.5
        HS_model.Cc = 1.15 * (eo - 0.35)
        HS_model.Cs = 0.5
        HS_model.m = 0.65

After that the stiffnesses of the HS model can be calculated using the 
:func:`~geolib_plus.hardening_soil_model_parameters.HardeningSoilModelParameters.calculate_stiffness` method.
The user should also select the calculation type in this case the COMPRESSIBILITYPARAMETERS calculation type is selected.
The results are simply stored as attributes in the class.


    .. code-block:: python
        
        from geolib_plus.hardening_soil_model_parameters import HardingSoilCalculationType
        calculation_type = HardingSoilCalculationType.COMPRESSIBILITYPARAMETERS
        # calculate stiffness
        parameters_for_HS.calculate_stiffness(calculation_type)
        print(parameters_for_HS.E_oed_ref)
        print(parameters_for_HS.E_ur_ref)

However, the user can also choose to perform this calculation using the cone resistance.
Which can be a direct input from a CPT read by GEOLIB+. Let's assume then that the user
read a gef file using GEOLIB+ and then defines this array as input to the HS model.

    .. code-block:: python

        from pathlib import Path
        from geolib_plus.gef_cpt import GefCpt
        from geolib_plus.hardening_soil_model_parameters import HardeningSoilModelParameters
        from geolib_plus.hardening_soil_model_parameters import HardingSoilCalculationType

        # read cpt from gef file 
        cpt_file_gef = Path("cpt", "gef", "test_cpt.gef")
        cpt_gef = GefCpt()
        cpt_gef.read(cpt_file_gef)
        cpt_gef.pre_process_data()

        # initialize hardening soil class
        HS_model = HardeningSoilModelParameters()
        # Import cpt inputs in the hardening soil class
        HS_model.qc = cpt_gef.tip
        HS_model.sigma_cpt_h = cpt_gef.effective_pressure
        # other inputs could be the same for each soil layer
        HS_model.sigma_ref_h = 100
        HS_model.m = 0.85
        HS_model.v_ur = 0.5
        # calculate the stiffnesses based on the cone resistance
        calculation_type = HardingSoilCalculationType.CONERESISTANCE
        parameters_for_HS.calculate_stiffness(calculation_type)



