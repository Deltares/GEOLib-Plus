.. tutorialcpt:

Tutorial Cpt interpretation using GEOLIB+
=========================================

By following the next steps the user can read, process and interpret a cpt file. The formats that can be read by GEOLIB+ are the gef and bro xml files.
The gef format is formulated as in :cite`CUR2006`. The bro xml  is formulated in :cite`bro_2017`.

1. First the user must define where the cpt file is located. This process is the same for both xml and gef files.

.. code-block:: python

    from pathlib import Path
    cpt_file_gef = Path("cpt", "gef", "test_cpt.gef")
    cpt_file_xml = Path("cpt", "xml", "test_cpt.xml")    

2. Then the corresponding class should be initialized. For gef initialize class :class:`~geolib_plus.gef_cpt.gef_cpt.GefCpt`. 
For bro xml initialize the class :class:`~geolib_plus.bro_xml_cpt.bro_xml_cpt.BroXmlCpt`

.. code-block:: python

    from geolib_plus.gef_cpt import GefCpt
    from geolib_plus.bro_xml_cpt import BroXmlCpt

    # initialize models
    cpt_gef = GefCpt()
    cpt_xml = BroXmlCpt()

3. After that the raw data can be read for both of the cpts. 
In this case the raw data from the files are allocated into the properties of the class :class:`~geolib_plus.cpt_base_model.AbstractCPT`.
For example the penetration length of the cpt is property penetration_length of the  class :class:`~geolib_plus.cpt_base_model.AbstractCPT`.
As it is of type Iterable you can print out the first element.

.. code-block:: python

    # read the cpt for each type of file
    cpt_gef.read(cpt_file_gef)
    cpt_xml.read(cpt_file_xml)
    print(cpt_gef.penetration_length[0])
    print(cpt_xml.penetration_length[0])

4. Until step 3 only the raw data are part of the class. With GEOLIB+ we can also run the pre-process methods that is required before plotting and 
interpretation the results. Both the gef and the bro xml cpt classes the pre_process_data method can be called. However, these are two different methods.
For the gef cpt :func:`~geolib_plus.gef_cpt.gef_cpt.GefCpt.pre_process_data` is called. 
For the bro xml cpt :func:`~geolib_plus.bro_xml_cpt.bro_xml_cpt.BroXmlCpt.pre_process_data` is called.
Nonetheless both of this functions :func:`~geolib_plus..cpt_base_model.AbstractCPT.pre_process_data` is called.   
    
