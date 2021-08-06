.. tutorialcpt:

Tutorial derive model parameters for sands using relative density input
=========================================================================

Part of the GEOLIB+ package is also a module that can determine certain parameters
used as input of Relative Density(RD). To do that the user must used the 
:func:`~geolib_plus.relative_density_correlated_parametes.RelativeDensityCorrelatedParameters.calculate_using_RD`
method to initialize the RD class and calculate all relevant parameters. The method
is pretty straight forward as it can be seen in the code block bellow.

    .. code-block:: python

        from geolib_plus.relative_density_correlated_parametes import RelativeDensityCorrelatedParameters
        # define relative density as float
        RD = 80
        # create class
        RD_parameters = RelativeDensityCorrelatedParameters.calculate_using_RD(RD)
        # check expectations
        print(RD_parameters.gamma_unsat)
        print(RD_parameters.gamma_sat)
        print(RD_parameters.E_ur_ref)

The user can also choose an array input for the value of relative density. That would lead also to array outputs.

    .. code-block:: python
    
        from geolib_plus.relative_density_correlated_parametes import RelativeDensityCorrelatedParameters
        import numpy as np
        
        # define relative density as an array input
        RD = np.array([80, 60])
        # create class
        RD_parameters_array = RelativeDensityCorrelatedParameters.calculate_using_RD(RD)
       # check expectations
        print(RD_parameters.gamma_unsat)
        print(RD_parameters.gamma_sat)
        print(RD_parameters.E_ur_ref)

