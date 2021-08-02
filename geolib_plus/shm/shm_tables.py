
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
from pydantic import BaseModel

from geolib_plus.shm.soil import Soil

class Shm_tables(BaseModel):
    """
    "Schematiserings handleiding macrostability"  tables: 7.1, 7.2, 7.3, 7.4.

    """

    soils: List = []

    def load_shm_tables(
            self, path_table: Path = Path("resources"), filename: str = "shm_tables.json"
    ) -> None:
        """
        Function that reads the shm tables json file.

        :param path_table: Path to the tables file
        :param filename: Name of the tables file
        :return: Dictionary with the NEN data structure
        """

        # define the path for the shape file
        path_table = Path(Path(__file__).parent, path_table, filename)

        # read shapefile
        with open(path_table, "r") as f:
            soils = json.load(f)

        for k,v in soils.items():
            for sub_k, sub_v in v.items():
                soil = Soil(name=f"{k}_{sub_k}")
                soil = soil.transfer_soil_dict_to_class(sub_v, soil)
                self.soils.append(soil)

        return

tmp = Shm_tables()

tmp.load_shm_tables()