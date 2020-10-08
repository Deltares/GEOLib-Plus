from geolib_plus.GEF_CPT import validate_gef
from pathlib import Path
import pytest

def test_validate_gef_noerror():
    # This file raises a warning - it is in another process so can't capture it
    gef_file = Path("../tests/test_files/cpt/gef/CPT000000003688_IMBRO_A.gef")
    try:
        validate_gef.ExecuteGEFValidation(gef_file)
    except:
        pytest.fail("GEF file without error raised Error")

def test_validate_gef_error():
    # This file raises a warning
    gef_file = Path("../tests/test_files/cpt/gef/CPT000000003688_IMBRO_A_err.gef")
    with pytest.raises(Exception):
        validate_gef.ExecuteGEFValidation(gef_file)

