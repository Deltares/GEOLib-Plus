# External modules
from .bro_utils import XMLBroCPTReader
from geolib_plus.cpt_base_model import AbstractCPT, CptReader


class BroXmlCpt(AbstractCPT):
    @classmethod
    def get_cpt_reader(cls) -> CptReader:
        return XMLBroCPTReader()

    def pre_process_data(self):
        super(BroXmlCpt, self).pre_process_data()

        #todo pre process like it is now in the bro reader
        pass

