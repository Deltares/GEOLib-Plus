from geolib_plus.cpt_base_model import AbstractCPT, CptReader
from .gef_file_reader import GefFileReader
from pathlib import Path


class GefCpt(AbstractCPT):

    def read(self, gef_file: Path):

        # validate gef_file
        if not gef_file:
            raise ValueError(gef_file)
        if not id:
            raise ValueError(id)
        gef_file = Path(gef_file)
        if not gef_file.is_file():
            raise FileNotFoundError(gef_file)
        # validate_gef_cpt(gef_file)

        # read the gef
        gef_reader = GefFileReader()
        gef = gef_reader.read_gef(gef_file)

        # if gef is not a dictionary: returns error message
        if not isinstance(gef, dict):
            return gef

        # add the gef attributes to CPT structure

        for k in gef.keys():
            setattr(self, k, gef[k])

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


