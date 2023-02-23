import pytest

def read_incomplete_cpt_only_test():
    ##
    from tests.utils import TestUtils
    from geolib_plus.bro_xml_cpt import BroXmlCpt
    import os
    from pathlib import Path
    test_folder = Path(TestUtils.get_local_test_data_dir("cpt/bro_xml"))
    cpt_xml2 = os.path.join(test_folder,'CPT000000089280'+".xml") # this CPT data is empty
    cpt_data2 = BroXmlCpt()
    cpt_data2.read(Path(cpt_xml2))

    print(f"cpt_data2.depth {cpt_data2.depth is None}")
    assert cpt_data2.depth is None
    ##
    return True


def read_both_filled_and_incomplete_cpts_test():
    ##
    from tests.utils import TestUtils
    from geolib_plus.bro_xml_cpt import BroXmlCpt
    import os
    from pathlib import Path

    test_folder = Path(TestUtils.get_local_test_data_dir("cpt/bro_xml"))
    cpt_xml1 = os.path.join(test_folder,"CPT000000038970"+".xml")
    cpt_xml2 = os.path.join(test_folder,'CPT000000089280'+".xml") # this CPT data is empty
    cpt_data1 = BroXmlCpt()
    cpt_data1.read(Path(cpt_xml1))

    cpt_data2 = BroXmlCpt()
    cpt_data2.read(Path(cpt_xml2))

    print(f"cpt_data1.depth {cpt_data1.depth is None}")
    print(f"cpt_data2.depth {cpt_data2.depth is None}")

    assert cpt_data2.depth is None
    ##
    return True


class TestNoneValues:
    @pytest.mark.unittest
    def test_read_incomplete_cpt_only(self):
        assert read_incomplete_cpt_only_test()

    @pytest.mark.unittest
    def test_read_both_filled_and_incomplete_cpts(self):
        assert read_both_filled_and_incomplete_cpts_test()
