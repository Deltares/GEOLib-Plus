import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tests.utils import TestUtils

from geolib_plus.bro_xml_cpt import bro_xml_cpt
from geolib_plus.bro_xml_cpt import BroXmlCpt
from geolib_plus.bro_xml_cpt.bro_utils import XMLBroCPTReader

# todo JN: write unit tests
class TestBroXmlCpt:
    @pytest.mark.systemtest
    def test_read(self):
        # simple test for reading xml file from bro
        # define input path to xml
        test_folder = Path(TestUtils.get_local_test_data_dir("cpt/bro_xml"))
        filename = "CPT000000003688_IMBRO_A.xml"
        test_file = test_folder / filename
        # initialise model
        cpt_read = bro_xml_cpt.BroXmlCpt()
        # run test
        cpt = cpt_read.read(test_file)
        # check expectations
        assert cpt
        assert cpt.name == "CPT000000003688"
        assert cpt.quality_class == "klasse2"
        assert cpt.cpt_type == "F7.5CKE/V-1214"
        assert cpt.local_reference_level == -1.75
        assert min(cpt.depth) == 0
        assert max(cpt.depth) == 24.56
