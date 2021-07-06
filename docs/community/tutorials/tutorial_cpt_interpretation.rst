.. tutorialcpt:

Tutorial Cpt interpretation using GEOLIB+
=========================================

After the cpt is set up by following the steps described in:

.. toctree::
   :maxdepth: 3

   tutorial_setup_cpt

The user can interpret a cpt by following the next step:

1. The interpretation method  :func:`~geolib_plus.cpt_base_model.AbstractCPT.interpret` can be called. This function is the same for the gef and xml bro data.
The user can use the default method of interpretation by using the :class:`~geolib_plus.robertson_cpt_interpretation.RobertsonCptInterpretation` class. The user can also 
create a custom interpretation model.
In this case the interpret can be initialized and the property unitweightmethod, shearwavevelocitymethod and ocrmethod can be modified or set to defaults.
The unitweightmethod can be defined from the IntEnum class :class:`~geolib_plus.robertson_cpt_interpretation.UnitWeightMethod`.
The shearwavevelocitymethod can be defined from the IntEnum class :class:`~geolib_plus.robertson_cpt_interpretation.OCRMethod`.
The ocrmethod can be defined from the IntEnum class :class:`~geolib_plus.robertson_cpt_interpretation.ShearWaveVelocityMethod`.

.. code-block:: python

    from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
    from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod
    from geolib_plus.robertson_cpt_interpretation import OCRMethod
    from geolib_plus.robertson_cpt_interpretation import ShearWaveVelocityMethod
    # do pre-processing
    interpreter = RobertsonCptInterpretation()
    interpreter.unitweightmethod = UnitWeightMethod.LENGKEEK
    interpreter.shearwavevelocitymethod = ShearWaveVelocityMethod.ZANG
    interpreter.ocrmethod = OCRMethod.MAYNE    

With the GEOLIB+ interpreter the water level has to be defined to calculate the pore pressures. 
The water level is defined in method :func:`~geolib_plus.robertson_cpt_interpretation.RobertsonCptInterpretation.pwp_level_calc`.
There are two options when it comes to defining the water level.

The water level can be user defined. To do that the user need to set the property to True user_defined_water_level.
After that the user can define the water level from the property pwp of the cpt.
This way the defined water level will be used during the interpretation.

.. code-block:: python

    interpreter.user_defined_water_level = True
    cpt_gef.pwp = -2.4
    cpt_xml.pwp = -4.3

The water level can be also extracted for an netcdf file. GEOLIB+ provides a default file of water level of the Netherlands in
the "tests\test_files\peilgebieden_jp_250m.nc" directory in that case the interpretation can be run directly without making any
changes. However, the user can also define their own water level file. This can be done as follows:

.. code-block:: python

    interpreter.user_defined_water_level = False
    interpreter.path_to_water_level_file = Path(
            "D:\geolib-plus", "tests", "test_files"
        )
    interpreter.name_water_level_file = "peilgebieden_jp_250m.nc"

Finally, the interpretation can be performed for both the xml and the cpt files.

.. code-block:: python

    cpt_gef.interpret_cpt(interpreter)
    cpt_xml.interpret_cpt(interpreter)