"""
NEN 9997 soil classification
"""
import json

# import packages
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from pydantic import BaseModel

# import packages
from .cpt_utils import resource_path


class NEN_classification(BaseModel):
    """
    NEN 9997 soil classification.

    """

    soil_types_list: List = []
    map_dict: Dict[int, List[List[Union[str, List[str]]]]] = {
        1: [
            ["Veen", "Leem"],
            [
                ["Niet voorbelast", "Matig voorbelast"],
                ["Zwak zandig", "Sterk zandig"],
            ],
        ],
        2: [["Klei"], [["Organisch"]]],
        3: [["Klei"], [["Schoon", "Zwak zandig"]]],
        4: [["Zand", "Klei"], [["Sterk siltig, kleiig"], ["Sterk zandig"]]],
        5: [["Zand"], [["Zwak siltig, kleiig"]]],
        6: [["Zand"], [["Schoon"]]],
        7: [["Grind"], [["Zwak siltig"]]],
        8: [["Grind"], [["Sterk siltig"]]],
        9: [["Grind"], [["Zwak siltig"]]],
    }

    def define_lithology_list(
        self, lithology: List
    ) -> List[List[Union[str, List[str]]]]:
        return [
            self.map_dict[int(lit)]
            for lit in lithology
            if int(lit) in self.map_dict.keys()
        ]

    def soil_types(
        self, path_shapefile: Path = Path("resources"), model_name: str = "NEN9997.json"
    ) -> None:
        """
        Function that reads the NEN json file.

        :param path_shapefile: Path to the classification files
        :param model_name: Name of model
        :return: Dictionary with the NEN data structure
        """

        # define the path for the shape file
        path_shapefile = Path(Path(__file__).parent, path_shapefile, model_name)

        # read shapefile
        with open(path_shapefile, "r") as f:
            self.soil_types_list = json.load(f)

        return

    def information(self, qcdq: float, lithology: List) -> Dict:
        """
        Identifies the properties from the NEN9997-1, given the soil lithology and the value of the
        tip resistance (corrected for the effective stress)

        Parameters
        ----------
        :param qcdq: Corrected tip resistance
        :param lithology: Lithology
        :return:
        """

        # redifine the lithology into NEN labels
        new_lit = self.define_lithology_list(lithology=lithology)

        # results
        result = {"litho_NEN": [], "E_NEN": [], "cohesion_NEN": [], "fr_angle_NEN": []}

        # determine into which soil type the point is
        for idx in range(len(qcdq)):

            # main name of soil family
            names = new_lit[idx][0]

            qcdq_table = []
            # for the possible soils identify which one has the closest qcdq
            for j, n in enumerate(names):
                # bijname of the soil family
                bijnames = new_lit[idx][1][j]
                for b in bijnames:
                    for consis, values in self.soil_types_list[n][b].items():
                        qcdq_table.append([n, b, consis, float(values["qcdq"]) * 1000])

            # find closest qcdq
            # convert qcdc table to array
            q = [i[3] for i in qcdq_table]
            distance = np.abs(qcdq[idx] - np.array(q))
            i = np.argmin(distance)  # index of the minimum qcdq distance

            # append to result
            result["litho_NEN"].append(
                "/".join([qcdq_table[i][0], qcdq_table[i][1], qcdq_table[i][2]])
            )
            result["E_NEN"].append(
                "/".join(
                    list(
                        set(
                            self.soil_types_list[qcdq_table[i][0]][qcdq_table[i][1]][
                                qcdq_table[i][2]
                            ]["E"]
                        )
                    )
                )
            )
            result["cohesion_NEN"].append(
                "/".join(
                    list(
                        set(
                            self.soil_types_list[qcdq_table[i][0]][qcdq_table[i][1]][
                                qcdq_table[i][2]
                            ]["Cohesie"]
                        )
                    )
                )
            )
            result["fr_angle_NEN"].append(
                "/".join(
                    list(
                        set(
                            self.soil_types_list[qcdq_table[i][0]][qcdq_table[i][1]][
                                qcdq_table[i][2]
                            ]["friction_angle"]
                        )
                    )
                )
            )

        return result
