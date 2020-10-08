import unittest
from pathlib import Path
from geolib_plus.BRO_XML_CPT import validate_bro_cpt

import pytest

def test_validate_bro_noerror():
    bro_xml_file_path = Path('../tests/test_files/cpt/bro_xml/CPT000000003688_IMBRO_A.xml')
    try:
        validate_bro_cpt(bro_xml_file_path)
    except:  # catch *all* exceptions
        pytest.fail("Validation Error: CPT BRO_XML without error raises error")

def test_validate_bro_error():
    bro_xml_file_err_path = Path('../tests/test_files/cpt/bro_xml/CPT000000003688_IMBRO_A_err.xml')
    with pytest.raises(Exception):
        validate_bro_cpt(bro_xml_file_err_path)