"""Tools to read gef files."""
import re
import numpy as np
from typing import List, Dict, Union, Iterable
from pydantic import BaseModel
from logging import warning


class GefProperty(BaseModel):
    gef_key: int
    values_from_gef: Union[Iterable, None] = None
    multiplication_factor: float
    error_code: Union[str, float, None] = None
    gef_column_index: Union[int, None] = None


class GefFileReader:
    def __init__(self):
        self.property_dict = self._initialize_property_dict_()
        self.name = ""
        self.coord = []

    def _initialize_property_dict_(self) -> dict:
        return {
            "depth": GefProperty(gef_key=1, multiplication_factor=1.0),
            "tip": GefProperty(gef_key=2, multiplication_factor=1000.0),
            "friction": GefProperty(gef_key=3, multiplication_factor=1000.0),
            "friction_nb": GefProperty(gef_key=4, multiplication_factor=1.0),
            "pwp": GefProperty(gef_key=6, multiplication_factor=1000.0),
        }

    def read_gef(self, gef_file, fct_a=0.8) -> dict:
        # read gef file
        with open(gef_file, "r") as f:
            data = f.readlines()

        # search NAP
        idx_nap = [i for i, val in enumerate(data) if val.startswith(r"#ZID=")][0]
        NAP = float(data[idx_nap].split(",")[1])
        # search end of header
        idx_EOH = [i for i, val in enumerate(data) if val.startswith(r"#EOH=")][0]
        # # search for coordinates
        idx_coord = [i for i, val in enumerate(data) if val.startswith(r"#XYID=")][0]
        # search index depth
        for key_name in self.property_dict:
            self.property_dict[
                key_name
            ].gef_column_index = self.read_column_index_for_gef_data(
                self.property_dict[key_name].gef_key, data
            )

        # read error codes
        idx_errors_raw_text = [
            val.split(",")[1]
            for i, val in enumerate(data)
            if val.startswith(r"#COLUMNVOID=")
        ]
        self.match_idx_with_error(idx_errors_raw_text)
        # rewrite data with separator ;
        data[idx_EOH + 1 :] = [
            re.sub("[ :,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1 :]
        ]

        try:
            # search index coefficient a
            idx_a = [
                i
                for i, val in enumerate(data)
                if val.endswith("Netto oppervlaktequotient van de conuspunt\n")
            ][0]
            fct_a = float(data[idx_a].split(",")[1])
        except IndexError:
            fct_a = fct_a

        # remove empty lines
        data = list(filter(None, data))

        # read data & correct depth to NAP
        self.read_data(data, idx_EOH)

        # remove the points with error: value == -9999
        self.remove_points_with_error()
        # if tip / friction / friction number are negative -> zero
        correct_for_negatives = ["tip", "friction", "friction_nb"]
        self.correct_negatives_and_zeros(correct_for_negatives)

        self.name = [
            val.split("#TESTID=")[-1].strip()
            for i, val in enumerate(data)
            if val.startswith(r"#TESTID=")
        ][0]
        self.coord = list(
            map(
                float,
                re.sub("[ ,!\t]+", ";", data[idx_coord].strip())
                .split("#XYID=")[-1]
                .split(";")[2:4],
            )
        )

        depth = [i for i in self.property_dict["depth"].values_from_gef]
        z_NAP = [
            NAP - i for j, i in enumerate(self.property_dict["depth"].values_from_gef)
        ]

        res = dict(
            name=self.name,
            depth=np.array(self.property_dict["depth"].values_from_gef),
            depth_to_reference=np.array(z_NAP),
            tip=np.array(self.property_dict["tip"].values_from_gef),
            friction=np.array(self.property_dict["friction"].values_from_gef),
            friction_nbr=np.array(self.property_dict["friction_nb"].values_from_gef),
            a=fct_a,
            coordinates=self.coord,
            water=np.array(self.property_dict["pwp"].values_from_gef),
        )
        return res

    def read_column_index_for_gef_data(self, key_cpt: int, data: List[str]):
        """In the gef file '#COLUMNINFO=id , name , column_number' format is used.
        This function returns the id number. Which will be later used
        as reference for the errors.
        """
        result = None
        for i, val in enumerate(data):
            if val.startswith(r"#COLUMNINFO=") and int(val.split(",")[-1]) == int(
                key_cpt
            ):
                result = int(val.split(",")[0].split("=")[-1]) - 1
        return result

    def match_idx_with_error(self, idx_errors: List[str],) -> None:
        """
        In the gef file each of the parameters has a value that is written
        when an error in the cpt data accumulation ocurred.
        The depth and tip keys are required inputs of the gef file. If they
        are missing then the function will raise an error. If another key is
        missing then a warning should be given back.
        """
        # Check if errors if not empty
        if bool(idx_errors):
            for key in self.property_dict:
                if self.property_dict[key].gef_column_index is not None:
                    gef_property = self.property_dict[key]
                    try:
                        gef_property.error_code = (
                            float(idx_errors[int(gef_property.gef_column_index)])
                            * gef_property.multiplication_factor
                        )
                    except ValueError:
                        # Raises a ValueError at a string is returned
                        gef_property.error_code = idx_errors[
                            int(gef_property.gef_column_index)
                        ]
                else:
                    # the key is missing from the gef file
                    if key in ["depth", "tip"]:
                        raise Exception(f"Key {key} should be defined in the gef file.")
                    else:
                        warning(f"Key {key} is not defined in the gef file.")
        return None

    def remove_points_with_error(self) -> None:
        """
        Values that contain data with errors should be removed
        from the resulting dictionary
        """
        for key in self.property_dict:
            deleted_rows = 0
            if self.property_dict[key].values_from_gef is not None:
                for number, value in enumerate(self.property_dict[key].values_from_gef):
                    if (
                        self.property_dict[key].values_from_gef[number - deleted_rows]
                        == self.property_dict[key].error_code
                    ):
                        self.delete_value_for_all_keys(number=number - deleted_rows)
                        deleted_rows = deleted_rows + 1
        return None

    def delete_value_for_all_keys(self, number: int) -> None:
        """
        Deletes index of all lists contained in the dictionary.
        """
        try:
            for key in self.property_dict:
                if isinstance(self.property_dict[key].values_from_gef, list):
                    del self.property_dict[key].values_from_gef[number]
                elif isinstance(self.property_dict[key].values_from_gef, np.ndarray):
                    temp_list = self.property_dict[key].values_from_gef.tolist()
                    del temp_list[number]
                    self.property_dict[key].values_from_gef = np.array(temp_list)
        except IndexError:
            raise Exception(
                f"Index <{number}> excides the length of list of key '{key}'"
            )
        return None

    def read_data(self, data: List[str], idx_EOH: int) -> None:
        """
        Read column data from the gef file table.
        """
        for key in self.property_dict:
            if key == "pwp" and not (self.property_dict["pwp"].gef_column_index):
                # Pore pressures are not inputted
                self.property_dict["pwp"].values_from_gef = np.zeros(
                    len(self.property_dict["depth"].values_from_gef)
                )
            else:
                if self.property_dict[key].gef_column_index is not None:
                    self.property_dict[key].values_from_gef = [
                        float(
                            data[i].split(";")[self.property_dict[key].gef_column_index]
                        )
                        * self.property_dict[key].multiplication_factor
                        for i in range(idx_EOH + 1, len(data))
                    ]
                else:
                    if key in ["depth", "tip"]:
                        raise Exception(f"CPT key: {key} not part of GEF file")
                    else:
                        warning(f"Key {key} is not defined in the gef file.")
        return None

    def correct_negatives_and_zeros(self, correct_for_negatives: List[str]) -> None:
        """
        Values tip / friction / friction cannot be negative so they
        have to be zero.
        """
        for key in correct_for_negatives:
            if self.property_dict[key].gef_column_index is not None:
                self.property_dict[key].values_from_gef = np.array(
                    self.property_dict[key].values_from_gef
                )
                self.property_dict[key].values_from_gef[
                    self.property_dict[key].values_from_gef < 0
                ] = 0
        return None
