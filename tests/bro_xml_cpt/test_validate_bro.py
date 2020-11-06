import pytest
import logging

from geolib_plus.bro_xml_cpt.validate_bro import ValidatePreProcessing
from geolib_plus.bro_xml_cpt import BroXmlCpt

# todo JN: write unit tests
class TestValidateBro:
    @pytest.mark.systemtest
    def test_check_file_contains_data_returns_message(self, caplog):
        # xml will still be read but a warning will be logged
        LOGGER = logging.getLogger(__name__)
        # define logger
        LOGGER.info("Testing now.")
        # initialize model
        cpt_data = BroXmlCpt()
        # test initial expectations
        assert cpt_data
        # define inputs
        cpt_data.penetration_length = []
        cpt_data.name = "CPT 1"
        # run test
        ValidatePreProcessing()._ValidatePreProcessing__check_file_contains_data(
            cpt=cpt_data
        )
        # check expectations
        assert "File CPT 1 contains no data" in caplog.text

    @pytest.mark.systemtest
    def test_check_data_different_than_zero_returns_message(self, caplog):
        # xml will still be read but a warning will be logged
        LOGGER = logging.getLogger(__name__)
        # define logger
        LOGGER.info("Testing now.")
        cpt_data = BroXmlCpt()
        # test initial expectations
        assert cpt_data
        # define inputs
        cpt_data.name = "CPT 1"
        cpt_data.penetration_length = [0, 0, 0]
        cpt_data.tip = [0, 0, 0]
        cpt_data.friction_nbr = [0, 0, 0]
        cpt_data.friction = [0, 0, 0]
        # run test
        ValidatePreProcessing()._ValidatePreProcessing__check_data_different_than_zero(
            cpt=cpt_data
        )
        # test
        assert "File CPT 1 contains empty data" in caplog.text

    @pytest.mark.systemtest
    def test_check_criteria_minimum_length(self, caplog):
        # xml will still be read but a warning will be logged
        LOGGER = logging.getLogger(__name__)
        # define logger
        LOGGER.info("Testing now.")
        cpt_data = BroXmlCpt()
        # test initial expectations
        assert cpt_data
        # define inputs
        cpt_data.name = "CPT 1"
        cpt_data.penetration_length = [0, 1, 2]
        cpt_data.tip = [3, 4, 5]
        cpt_data.friction_nbr = [6, 7, 8]
        cpt_data.friction = [9, 10, 11]
        # run test
        ValidatePreProcessing()._ValidatePreProcessing__check_criteria_minimum_length(
            cpt=cpt_data, minimum_length=5
        )
        # test
        assert "File CPT 1 has a length smaller than 5" in caplog.text

    @pytest.mark.systemtest
    def test_check_minimum_sample_criteria(self, caplog):
        # xml will still be read but a warning will be logged
        LOGGER = logging.getLogger(__name__)
        # define logger
        LOGGER.info("Testing now.")
        cpt_data = BroXmlCpt()
        # test initial expectations
        assert cpt_data
        # define inputs
        cpt_data.name = "CPT 1"
        cpt_data.penetration_length = [0, 1, 2]
        cpt_data.tip = [3, 4, 5]
        cpt_data.friction_nbr = [6, 7, 8]
        cpt_data.friction = [9, 10, 11]
        # run test
        ValidatePreProcessing()._ValidatePreProcessing__check_minimum_sample_criteria(
            cpt=cpt_data, minimum_samples=5
        )
        # test
        assert "File CPT 1 has a number of samples smaller than 5" in caplog.text

    @pytest.mark.systemtest
    def test_validate_length_and_samples_cpt(self, caplog):
        # xml will still be read but a warning will be logged
        LOGGER = logging.getLogger(__name__)
        # define logger
        LOGGER.info("Testing now.")
        cpt_data = BroXmlCpt()
        validator = ValidatePreProcessing()
        # test initial expectations
        assert cpt_data
        assert validator
        # define inputs
        cpt_data.name = "CPT 1"
        cpt_data.penetration_length = [0, 0, 0]
        cpt_data.tip = [0, 0, 0]
        cpt_data.friction_nbr = [0, 0, 0]
        cpt_data.friction = [0, 0, 0]
        # run test
        validator.validate_length_and_samples_cpt(
            cpt=cpt_data, minimum_samples=5, minimum_length=6
        )
        # test
        assert "File CPT 1 has a number of samples smaller than 5" in caplog.text
        assert "File CPT 1 has a length smaller than 6" in caplog.text
        assert "File CPT 1 contains empty data" in caplog.text
