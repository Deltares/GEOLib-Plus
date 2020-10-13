from geolib_plus import AbstractCPT
from pathlib import Path
from .gef_utils import read_gef
from .validate_gef import validate_gef_cpt


class GefCpt(AbstractCPT):
    def read(self, gef_file: Path, id, key_cpt=None):

        # validate gef_file
        validate_gef_cpt(gef_file)

        if key_cpt is None:
            key_cpt = {"depth": 1, "tip": 2, "friction": 3, "friction_nb": 4, "pwp": 6}

        # read the gef
        gef = read_gef(gef_file, id, key_cpt)

        # if gef is not a dictionary: returns error message
        if not isinstance(gef, dict):
            return gef

        # add the gef attributes to CPT structure

        for k in gef.keys():
            setattr(self, k, gef[k])
