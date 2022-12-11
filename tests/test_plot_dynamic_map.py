import pytest

from geolib_plus.plot_dynamic_map import ObjectLocationMap, Location
from tests.utils import TestUtils
from pathlib import Path


class TestObjectLocationMap:
    @pytest.mark.unittest
    def test_plot(self):
        meta_data = {'something1': 1}
        locations = [Location(x=64663.8, y=393995.8, label=f"cpt_1", meta_data=meta_data),
                     Location(x=64763.8, y=393895.8, label=f"cpt_2", meta_data=meta_data),
                     Location(x=64863.8, y=393795.8, label=f"cpt_3", meta_data=meta_data),
                     Location(x=64963.8, y=393695.8, label=f"cpt_3", meta_data=meta_data)]
        extract_map = ObjectLocationMap(object_locations=locations, results_folder=Path("test_output"))
        extract_map.plot_html_folium()
        output_test_folder = Path(TestUtils.get_output_test_data_dir(""))
        assert Path(output_test_folder, "dynamic_map_with_cpts.html").is_file()