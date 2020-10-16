import pytest
from geolib_plus.bro_xml_cpt import bro_utils as bro
from pathlib import Path
import json
import os

# todo JN: write unit tests
class TestBroUtil:
    @pytest.mark.unittest
    def test_columns_string_list(self):
        # initialise inputs
        columns = [
            "penetrationLength",
            "depth",
            "elapsedTime",
            "coneResistance",
            "correctedConeResistance",
            "netConeResistance",
            "magneticFieldStrengthX",
            "magneticFieldStrengthY",
            "magneticFieldStrengthZ",
            "magneticFieldStrengthTotal",
            "electricalConductivity",
            "inclinationEW",
            "inclinationNS",
            "inclinationX",
            "inclinationY",
            "inclinationResultant",
            "magneticInclination",
            "magneticDeclination",
            "localFriction",
            "poreRatio",
            "temperature",
            "porePressureU1",
            "porePressureU2",
            "porePressureU3",
            "frictionRatio",
        ]
        cl_cpt = bro.XMLBroCPTReader()
        test_columns = cl_cpt.columns_string_list
        assert test_columns == columns
