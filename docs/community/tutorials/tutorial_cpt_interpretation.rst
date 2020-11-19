.. tutorialcpt:

Tutorial Cpt interpretation using GEOLIB+
=========================================

After the cpt is set up by following the steps described in:

.. toctree::
   :maxdepth: 3

   tutorial_setup_cpt

The user can interpret a cpt by following the next step:

1. The interpretation method  :func:`~geolib_plus.cpt_base_model.AbstractCPT.interpret` can be called. This function is the same for the gef and xml bro data.
The user can use the default method of interpretation by using the :class:`~geolib_plus.robertson_cpt_interpretation.robertson_cpt_interpretation.RobertsonCptInterpretation` class. The user can also 
create a custom interpretation model.
In this case the interpret can be initialized and the property unitweightmethod, shearwavevelocitymethod and ocrmethod can be modified or set to defaults.
The unitweightmethod can be defined from the IntEnum class :class:`~geolib_plus.robertson_cpt_interpretation.robertson_cpt_interpretation.UnitWeightMethod`.
The shearwavevelocitymethod can be defined from the IntEnum class :class:`~geolib_plus.robertson_cpt_interpretation.robertson_cpt_interpretation.OCRMethod`.
The ocrmethod can be defined from the IntEnum class :class:`~geolib_plus.robertson_cpt_interpretation.robertson_cpt_interpretation.ShearWaveVelocityMethod`.

.. code-block:: python

    from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
    from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod
    from geolib_plus.robertson_cpt_interpretation import OCRMethod
    from geolib_plus.robertson_cpt_interpretation import ShearWaveVelocityMethod
    # do pre-processing
    interpreter = RobertsonCptInterpretation
    interpreter.unitweightmethod = UnitWeightMethod.LENGKEEK
    interpreter.shearwavevelocitymethod = ShearWaveVelocityMethod.ZANG
    interpreter.ocrmethod = OCRMethod.MAYNE    
    cpt_gef.interpret_cpt(interpreter)
    cpt_xml.interpret_cpt(interpreter)