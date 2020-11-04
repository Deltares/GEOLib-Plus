from geolib_plus.cpt_base_model import AbstractCPT, CptReader
from .gef_file_reader import GefFileReader
from pathlib import Path


class GefCpt(AbstractCPT):
    @classmethod
    def get_cpt_reader(cls) -> CptReader:
        return GefFileReader()

    def pre_process_data(self):
        super(GefCpt, self).pre_process_data()
        pa_to_mpa = 1e-6

        self.tip = self.tip * pa_to_mpa
        self.friction = self.friction * pa_to_mpa

        self.pore_pressure_u1 = self.pore_pressure_u1 * pa_to_mpa
        self.pore_pressure_u2 = self.pore_pressure_u2 * pa_to_mpa
        self.pore_pressure_u3 = self.pore_pressure_u3 * pa_to_mpa
        self.water = self.water * pa_to_mpa


        #todo remove points with error


        pass

