.. tutorialcpt:

Tutorial defining and calculating parameters for the Soft Soil Creep model
==========================================================================

Part of the GEOLIB+ package is also a module that can determine certain parameters
used as input of the Soft Soil Creep (SSC) model. To do that we can use the 
:class:`~geolib_plus.soft_soil_creep_parameters.SoftSoilCreepParameters` class.
In this tutorial the user reads will perform the following steps to calculate the SSC parameters:

    * read a cpt and interpret it using the :class:`~geolib_plus.robertson_cpt_interpretation.robertson_cpt_interpretation.RobertsonCptInterpretation` class.
    * calculate the OCR values in every point of the cpt
    * calculate the initial void ratio based on the lithology 
    * calculate the compression index based on Nishida (1956) :cite:`nishida_1956`

To begin with the SSC model calculation the :class:`~geolib_plus.soft_soil_creep_parameters.SoftSoilCreepParameters` class
should be initialized.

    .. code-block:: python

        from geolib_plus.soft_soil_creep_parameters import SoftSoilCreepParameters
        # initialize class
        SSC_model = SoftSoilCreepParameters()


The user will then read and interpret a gef file using GEOLIB+.

    .. code-block:: python

        from pathlib import Path
        from geolib_plus.gef_cpt import GefCpt
        from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation

        # read cpt from gef file 
        cpt_file_gef = Path("cpt", "gef", "test_cpt.gef")
        cpt_gef = GefCpt()
        cpt_gef.read(cpt_file_gef)
        cpt_gef.pre_process_data()
        # define interpreter
        interpreter = RobertsonCptInterpretation()  
        # perform interpretation
        cpt_gef.interpret_cpt(interpreter)


The data stored in the cpt_gef can be used to calculate more parameters that can be 
used as inputs of the SSC parameter calculation. For example, the OCR value can be
calculated using the following equation :cite:`robertson_cabal_2014`.

    .. code-block:: python

        import numpy as np

        OCR = np.array([ 0.25 * Qtn_value ** 1.25 for Qtn_value in cpt_gef.Qtn])


The initial void ratio is defined by using the lithology attribute.
From the initial void ratio the compression index can also be calculated.

    .. code-block:: python

        # assign size to void ratio list
        eo = np.zeros(len(cpt_gef.lithology))

        for i, lithology_index in enumerate(cpt_gef.lithology):
            if lithology_index in ["1", "2", "3"]:
                eo[i] = 0.6
            elif lithology_index == "4":
                eo[i] = 0.33
            elif lithology_index in ["5", "6", "7"]:
                eo[i] = 0.26
            else:
                eo[i] = 0.5
        Cc = 1.15 * ( eo - 0.35)

Now that all the inputs are prepared for the SSC parameter calculation
the user can input them into the class and perform the calculation.


    .. code-block:: python

        # input calculated arrays in SSC object
        SSC_model.eo = eo
        SSC_model.Cc = Cc
        SSC_model.v_ur = cpt_gef.poisson
        SSC_model.OCR = OCR
        # some of the inputs can be defined as floats
        SSC_model.Cs = 1
        SSC_model.Ca = 0.1
        SSC_model.K0_NC = 0.5
        # calculate parameters
        SSC_model.calculate_soft_soil_parameters()



