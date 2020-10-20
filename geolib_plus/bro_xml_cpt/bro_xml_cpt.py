# External modules
from .bro_utils import XMLBroCPTReader
from geolib_plus.cpt_base_model import AbstractCPT, CptReader


class BroXmlCpt(AbstractCPT):
    @classmethod
    def get_cpt_reader(cls) -> CptReader:
        return XMLBroCPTReader()
