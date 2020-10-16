from geolib_plus.cpt_base_model import AbstractCPT
from .gef_file_reader import GefFileReader
from pathlib import Path


class GefCpt(AbstractCPT):
    def __init__(self, gef_file: Path):
        if not gef_file:
            raise ValueError(gef_file)

        gef_file = Path(gef_file)
        if not gef_file.is_file():
            raise FileNotFoundError(gef_file)

        gef_data = GefFileReader().read_gef(gef_file)
        for key, value in gef_data.items():
            setattr(self, key, value)
