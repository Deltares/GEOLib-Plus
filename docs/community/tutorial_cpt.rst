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
