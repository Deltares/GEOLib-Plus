from geolib_plus import AbstractCPT
from .gef_utils import GefFileReader
from pathlib import Path
from .validate_gef import validate_gef_cpt


class GefCpt(AbstractCPT):
    def __init__(self, gef_filepath: Path):
        if not gef_filepath:
            raise ValueError(gef_filepath)

        gef_file = Path(gef_filepath)
        if not gef_file.is_file():
            raise FileNotFoundError(gef_file)

        gef_data = GefFileReader().read_gef(gef_file)
        for key, value in gef_data.items():
            setattr(self, key, value)
