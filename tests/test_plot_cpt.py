import pytest
from pathlib import Path
from geolib_plus import plot_cpt
from tests.utils import TestUtils
from geolib_plus.GEF_CPT import GefCpt
from geolib_plus.BRO_XML_CPT import BroXmlCpt
from geolib_plus.plot_settings import PlotSettings
import os

class TestPlotCpt:
    @pytest.mark.unittest
    @pytest.mark.workinprogress
    def test_plot_cpt_unit_tests(self):
        raise NotImplementedError


    @pytest.mark.integrationtest
    def test_generate_fig_with_default_settings_from_xml(self, bro_xml_cpt, plot_settings):
        plot_settings.assign_default_settings()

        output_path = Path(TestUtils._name_output)
        plot_cpt.plot_cpt_norm(bro_xml_cpt,output_path, plot_settings.general_settings)

        output_file_name = bro_xml_cpt.name + '.pdf'
        assert Path(output_path / output_file_name).is_file()
        os.remove(Path(output_path / output_file_name))

    @pytest.mark.integrationtest
    def test_generate_fig_with_default_settings_from_gef(self, gef_cpt, plot_settings):

        plot_settings.assign_default_settings()

        output_path = Path(TestUtils._name_output)
        plot_cpt.plot_cpt_norm(gef_cpt,output_path, plot_settings.general_settings)

        output_file_name = gef_cpt.name + '.pdf'
        assert Path(output_path / output_file_name).is_file()
        os.remove(Path(output_path / output_file_name))


@pytest.fixture
def gef_cpt():
    test_folder = Path(TestUtils.get_local_test_data_dir("cpt/gef"))
    filename = "CPT000000003688_IMBRO_A.gef"
    test_file = test_folder / filename
    cpt = GefCpt().read(test_file)
    return cpt

@pytest.fixture
def bro_xml_cpt():
    test_folder = Path(TestUtils.get_local_test_data_dir("cpt/bro_xml"))
    filename = "CPT000000003688_IMBRO_A.xml"
    test_file = test_folder / filename
    cpt = BroXmlCpt().read(test_file)


    return cpt

@pytest.fixture
def plot_settings():
    plot_settings = PlotSettings()
    plot_settings.assign_default_settings()
    return plot_settings