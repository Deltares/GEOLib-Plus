"""
NEN 9997 soil classification
"""
# import packages
import os
import json
import numpy as np

# import packages
from .cpt_utils import resource_path


class NEN_classification:
    r"""
    NEN 9997 soil classification.

    """

    def __init__(self):
        # initialise variables
        self.soil_types_list = []
        return

    def soil_types(self, path_shapefile=r"./resources/", model_name="NEN9997.json"):
        r"""
        Function that reads the NEN json file.

        :param path_shapefile: Path to the classification files
        :param model_name: Name of model
        :return: Dictionary with the NEN data structure
        """

        # define the path for the shape file
        path_shapefile = resource_path(
            os.path.join(
                os.path.join(os.path.dirname(__file__), path_shapefile), model_name
            )
        )

        # read shapefile
        with open(path_shapefile, "r") as f:
            self.soil_types_list = json.load(f)

        return

    def information(self, qcdq, lithology):
        r"""
        Identifies the properties from the NEN9997-1, given the soil lithology and the value of the
        tip resistance (corrected for the effective stress)

        Parameters
        ----------
        :param qcdq: Corrected tip resistance
        :param lithology: Lithology
        :return:
        """

        # redifine the lithology into NEN labels
        new_lit = []
        for lit in lithology:
            if int(lit) == 1:
                new_lit.append(
                    [
                        ["Veen", "Leem"],
                        [
                            ["Niet voorbelast", "Matig voorbelast"],
                            ["Zwak zandig", "Sterk zandig"],
                        ],
                    ]
                )
            elif int(lit) == 2:
                new_lit.append([["Klei"], [["Organisch"]]])
            elif int(lit) == 3:
                new_lit.append([["Klei"], [["Schoon", "Zwak zandig"]]])
            elif int(lit) == 4:
                new_lit.append(
                    [["Zand", "Klei"], [["Sterk siltig, kleiig"], ["Sterk zandig"]]]
                )
            elif int(lit) == 5:
                new_lit.append([["Zand"], [["Zwak siltig, kleiig"]]])
            elif int(lit) == 6:
                new_lit.append([["Zand"], [["Schoon"]]])
            elif int(lit) == 7:
                new_lit.append([["Grind"], [["Zwak siltig"]]])
            elif int(lit) == 8:
                new_lit.append([["Grind"], [["Sterk siltig"]]])
            elif int(lit) == 9:
                new_lit.append([["Grind"], [["Zwak siltig"]]])

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
