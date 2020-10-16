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
    @pytest.mark.system
    def test_read(self):
        # simple test for reading xml file from bro
        # define input path to xml
        test_folder = Path(TestUtils.get_local_test_data_dir("cpt/bro_xml"))
        filename = "CPT000000003688_IMBRO_A.xml"
        test_file = test_folder / filename
        # initialise model
        cpt = bro_xml_cpt.BroXmlCpt()
        # run test
        cpt.read(bro_xml_file=test_file)
        # check expectations
        assert cpt
        assert cpt.name == "CPT000000003688"
        assert cpt.quality_class == "klasse2"
        assert cpt.cpt_type == "F7.5CKE/V-1214"
        assert cpt.local_reference_level == -1.75
        assert min(cpt.depth) == 0
        assert max(cpt.depth) == 24.34

    @pytest.mark.unittest
    def test__get_depth_from_bro_depth_is_set(self):
        # define the inputs
        d = {
            "depth": [1.5, 2.0, 2.5],
        }

        # set up the upper part of the dictionary
        df = pd.DataFrame(data=d)
        # initialise model
        cpt = BroXmlCpt()
        # run test
        depth = cpt._BroXmlCpt__get_depth_from_bro(cpt_BRO=df)
        # check the results. Depth is just passed in this case.
        assert cpt
        assert d["depth"] == list(depth)

    @pytest.mark.unittest
    def test__get_depth_from_bro_no_depth_no_inclination(self):
        # define the inputs
        d = {
            "penetrationLength": [1.5, 2.0, 2.5],
        }
        # set up the upper part of the dictionary
        df = pd.DataFrame(data=d)
        # initialise model
        cpt = BroXmlCpt()
        # run test
        depth = cpt._BroXmlCpt__get_depth_from_bro(cpt_BRO=df)
        # check the results. Depth is just passed in this case.
        assert depth.all()
        assert d["penetrationLength"] == list(depth)

    @pytest.mark.unittest
    def test__get_depth_from_bro_no_depth(self):
        # define the inputs
        d = {
            "penetrationLength": [1.5, 2.0, 2.5],
            "inclinationResultant": [1.5, 2.0, 2.5],
        }
        # calculate test results
        result = np.diff(d["penetrationLength"]) * np.cos(
            np.radians(d["inclinationResultant"][:-1])
        )
        result = np.concatenate(
            (
                d["penetrationLength"][0],
                d["penetrationLength"][0] + np.cumsum(result),
            ),
            axis=None,
        )
        # set up the upper part of the dictionary
        df = pd.DataFrame(data=d)
        # initialise model
        cpt = BroXmlCpt()
        # run test
        depth = cpt._BroXmlCpt__get_depth_from_bro(cpt_BRO=df)
        # check the results. Depth is just passed in this case.
        assert depth.all()
        assert list(result) == list(depth)

    @pytest.mark.system
    def test__pre_drill_with_predrill(self):

        # make a cpt with the pre_drill option
        d = {
            "penetrationLength": [1.5, 2.0, 2.5],
            "coneResistance": [1, 2, 3],
            "localFriction": [4, 5, 6],
            "frictionRatio": [0.22, 0.33, 0.44],
        }

        # set up the upper part of the dictionary
        df = pd.DataFrame(data=d)
        cpt_data = XMLBroCPTReader()
        cpt_data.id = "cpt_name"
        cpt_data.location_x = 111
        cpt_data.location_y = 222
        cpt_data.offset_z = 0.5
        cpt_data.predrilled_z = 1.5
        cpt_data.a = 0.8
        cpt_data.dataframe = df

        # Run the function to be checked
        cpt = bro_xml_cpt.BroXmlCpt()
        cpt._BroXmlCpt__parse_bro(cpt_data, minimum_length=0.01, minimum_samples=1)

        # Check the equality with the pre-given lists
        assert cpt.tip.tolist() == [1000, 1000, 1000, 1000, 2000, 3000]
        assert cpt.friction.tolist() == [4000, 4000, 4000, 4000, 5000, 6000]
        assert cpt.friction_nbr.tolist() == [0.22, 0.22, 0.22, 0.22, 0.33, 0.44]
        assert cpt.depth.tolist() == [0, 0.5, 1, 1.5, 2, 2.5]
        assert cpt.depth_to_reference.tolist() == [
            cpt_data.offset_z - i for i in [0, 0.5, 1, 1.5, 2, 2.5]
        ]
        assert cpt.water.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert cpt.coordinates == [cpt_data.location_x, cpt_data.location_y]
        assert cpt.name == "cpt_name"
        assert cpt.a == 0.8

    @pytest.mark.system
    def test__pre_drill_with_pore_pressure(self):

        # Set the values of the cpt
        d = {
            "penetrationLength": [1.5, 2.0, 2.5],
            "coneResistance": [1, 2, 3],
            "localFriction": [4, 5, 6],
            "frictionRatio": [0.22, 0.33, 0.44],
            "porePressureU2": [1, 2, 3],
        }
        df = pd.DataFrame(data=d)

        # Build the upper part of the library
        cpt_data = XMLBroCPTReader()
        cpt_data.id = "cpt_name"
        cpt_data.location_x = 111
        cpt_data.location_y = 222
        cpt_data.offset_z = 0.5
        cpt_data.predrilled_z = 1.5
        cpt_data.a = 0.73
        cpt_data.dataframe = df

        # define the pore pressure array before the predrilling
        # Here 3 values as the stepping is defined that way.
        # Define the stepping of the pore pressure
        # Then my target values
        # Finally multiply with 1000
        step = 1 / len(d["penetrationLength"])
        pore_pressure = [0, step * 1000, 2 * step * 1000, 1 * 1000, 2 * 1000, 3 * 1000]

        # run the function to be checked
        cpt = bro_xml_cpt.BroXmlCpt()
        cpt._BroXmlCpt__parse_bro(cpt_data, minimum_length=0.01, minimum_samples=1)

        # Check the equality with the pre-defined values
        assert cpt.water.tolist() == pore_pressure
        assert cpt.tip.tolist() == [1000, 1000, 1000, 1000, 2000, 3000]
        assert cpt.friction.tolist() == [4000, 4000, 4000, 4000, 5000, 6000]
        assert cpt.friction_nbr.tolist() == [0.22, 0.22, 0.22, 0.22, 0.33, 0.44]
        assert cpt.depth.tolist() == [0, 0.5, 1, 1.5, 2, 2.5]
        assert cpt.depth_to_reference.tolist() == [
            cpt_data.offset_z - i for i in [0, 0.5, 1, 1.5, 2, 2.5]
        ]
        assert cpt.coordinates == [cpt_data.location_x, cpt_data.location_y]
        assert cpt.name == "cpt_name"
        assert cpt.a == 0.73

    @pytest.mark.system
    def test__pre_drill_Raise_Exception1(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {
            "penetrationLength": [1.5, 2.0],
            "coneResistance": [1, 2],
            "localFriction": [4, 5],
            "frictionRatio": [0.22, 0.33],
        }
        df = pd.DataFrame(data=d)
        # initialise model
        cpt_data = XMLBroCPTReader()
        cpt_data.id = "cpt_name"
        cpt_data.location_x = 111
        cpt_data.location_y = 222
        cpt_data.offset_z = 0.5
        cpt_data.predrilled_z = 1.5
        cpt_data.a = 0.73
        cpt_data.dataframe = df

        # run the fuction
        cpt = bro_xml_cpt.BroXmlCpt()
        aux = cpt._BroXmlCpt__parse_bro(cpt_data, minimum_length=10, minimum_samples=1)

        # check if the returned message is the appropriate
        assert "File cpt_name has a length smaller than 10" == aux

    @pytest.mark.system
    def test__pre_drill_Raise_Exception2(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {
            "penetrationLength": [1.5, 2.0],
            "coneResistance": [1, 2],
            "localFriction": [4, 5],
            "frictionRatio": [0.22, 0.33],
        }
        df = pd.DataFrame(data=d)
        # initialise model
        cpt_data = XMLBroCPTReader()
        cpt_data.id = "cpt_name"
        cpt_data.location_x = 111
        cpt_data.location_y = 222
        cpt_data.offset_z = 0.5
        cpt_data.predrilled_z = 1.5
        cpt_data.a = 0.73
        cpt_data.dataframe = df

        # run the fuction
        cpt = bro_xml_cpt.BroXmlCpt()
        aux = cpt._BroXmlCpt__parse_bro(cpt_data, minimum_length=1, minimum_samples=10)

        # check if the returned message is the appropriate
        assert "File cpt_name has a number of samples smaller than 10" == aux

    @pytest.mark.system
    def test_read_BRO_Raise_Exception1(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {
            "penetrationLength": [1.5, 20.0],
            "coneResistance": [-1, 2],
            "localFriction": [4, 5],
            "frictionRatio": [0.22, 0.33],
        }
        df = pd.DataFrame(data=d)
        # initialise model
        cpt_data = XMLBroCPTReader()
        cpt_data.id = "cpt_name"
        cpt_data.location_x = 111
        cpt_data.location_y = 222
        cpt_data.offset_z = 0.5
        cpt_data.predrilled_z = 1.5
        cpt_data.a = 0.73
        cpt_data.dataframe = df

        # run the fuction
        cpt = bro_xml_cpt.BroXmlCpt()
        aux = cpt._BroXmlCpt__parse_bro(cpt_data, minimum_length=10, minimum_samples=1)

        # check if the returned message is the appropriate
        assert "File cpt_name is corrupted" == aux

    @pytest.mark.system
    def test_read_BRO_Raise_Exception2(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {
            "penetrationLength": [1.5, 20.0],
            "coneResistance": [1, 2],
            "localFriction": [-4, 5],
            "frictionRatio": [0.22, 0.33],
        }
        df = pd.DataFrame(data=d)
        # initialise model
        cpt_data = XMLBroCPTReader()
        cpt_data.id = "cpt_name"
        cpt_data.location_x = 111
        cpt_data.location_y = 222
        cpt_data.offset_z = 0.5
        cpt_data.predrilled_z = 1.5
        cpt_data.a = 0.73
        cpt_data.dataframe = df

        # run the function
        cpt = bro_xml_cpt.BroXmlCpt()
        aux = cpt._BroXmlCpt__parse_bro(cpt_data, minimum_length=10, minimum_samples=1)

        # check if the returned message is the appropriate
        assert "File cpt_name is corrupted" == aux
