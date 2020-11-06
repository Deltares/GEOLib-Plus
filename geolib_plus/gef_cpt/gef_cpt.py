from geolib_plus.cpt_base_model import AbstractCPT, CptReader
from .gef_file_reader import GefFileReader
from pathlib import Path


class GefCpt(AbstractCPT):
    @classmethod
    def get_cpt_reader(cls) -> CptReader:
        return GefFileReader()

    def pre_process_data(self):
        """
        Pre processes data which is read from gef files.

        Units are converted to MPa.
        #todo extend
        :return:
        """
        super().pre_process_data()

        #todo remove points with error


