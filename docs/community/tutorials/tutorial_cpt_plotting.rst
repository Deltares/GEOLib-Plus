.. tutorialcpt:

Tutorial setup Cpt class interpretation using GEOLIB+
=========================================

By following the next steps the user can read, process and interpret a cpt file. The formats that can be read by GEOLIB+ are the gef and bro xml files.
The gef format is formulated as in :cite:`CUR2006`. The bro xml  is formulated in :cite:`bro_2017`.

1. First the user must define where the cpt file is located. This process is the same for both xml and gef files.

.. code-block:: python

    from pathlib import Path
    cpt_file_gef = Path("cpt", "gef", "test_cpt.gef")
    cpt_file_xml = Path("cpt", "xml", "test_cpt.xml")    

2. Then the corresponding class should be initialized. For gef initialize class :class:`~geolib_plus.gef_cpt.gef_cpt.GefCpt`. 
For bro xml initialize the class :class:`~geolib_plus.bro_xml_cpt.bro_xml_cpt.BroXmlCpt`.

.. code-block:: python

    from geolib_plus.gef_cpt import GefCpt
    from geolib_plus.bro_xml_cpt import BroXmlCpt

    # initialize models
    cpt_gef = GefCpt()
    cpt_xml = BroXmlCpt()

3. After that, the raw data can be read for both of the cpts. 
In this case the raw data from the files are allocated into the properties of the class :class:`~geolib_plus.cpt_base_model.AbstractCPT`.
For example the penetration length of the cpt is property penetration_length of the  class :class:`~geolib_plus.cpt_base_model.AbstractCPT`.
As it is of type Iterable you can print out the first element.

.. code-block:: python

    # read the cpt for each type of file
    cpt_gef.read(cpt_file_gef)
    cpt_xml.read(cpt_file_xml)
    print(cpt_gef.penetration_length[0])
    print(cpt_xml.penetration_length[0])

HERE WE SHOULD ADD THE CLEANING OF THE DATA.

4. Until step 3 only the raw data are part of the class. With GEOLIB+ the user can also run the pre-processing methods. These methods are required before plotting and 
interpretation the results. Both the gef and the bro xml cpt classes can call the pre_process_data method. However, these are two different methods.
For the gef cpt :func:`~geolib_plus.gef_cpt.gef_cpt.GefCpt.pre_process_data` is called. Where the following processes take place:

*  The :func:`~geolib_plus.bro_xml_cpt.bro_xml_cpt.BroXmlCpt.perform_pre_drill_interpretation` method where all the values are updated if a pre-drill operation was performed in the cpt.
*  The :func:`~geolib_plus.cpt_base_model.AbstractCPT.pre_process_data` method. This includes methods that calculate the depth depending on the inputs of the cpt, correcting negative values and calculating the pore pressures.

For the bro xml cpt the :func:`~geolib_plus.bro_xml_cpt.bro_xml_cpt.BroXmlCpt.pre_process_data` method is called. Where the following process take place:

*  The :func:`~geolib_plus.cpt_base_model.AbstractCPT.pre_process_data` method. This includes methods that calculate the depth depending on the inputs of the cpt, correcting negative values and calculating the pore pressures.

.. code-block:: python

    # do pre-processing
    cpt_gef.pre_process_data()
    cpt_xml.pre_process_data()


Tutorial Cpt interpretation using GEOLIB+
=========================================
By following the next step the user interpret a cpt. Which is read from either a bro-xml file or a gef file.


5. Finally, the interpretation method  :func:`~geolib_plus.cpt_base_model.AbstractCPT.interpret` can be called. This function is the same for the gef and xml bro data.
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

Tutorial Cpt plotting using GEOLIB+
=========================================
