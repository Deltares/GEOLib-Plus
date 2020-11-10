"""Tools to read gef files."""
import re
import numpy as np
from .validate_gef import validate_gef_cpt
from geolib_plus.cpt_base_model import CptReader
from typing import List, Dict, Union, Iterable
from pathlib import Path
from pydantic import BaseModel
from logging import warning
import math


class GefProperty(BaseModel):
    gef_key: int
    error_code: Union[str, float, None] = None
    values_from_gef: Union[Iterable, str, None] = None


class GefColumnProperty(GefProperty):
    multiplication_factor: float = 1
    gef_column_index: Union[int, None] = None


class GefFileReader(CptReader):
    def __init__(self):

        self.property_dict = self.__get_default_property_dict()
        self.information_dict = self.__get_default_information_dict()

        self.name = ""
        self.coord = []

    def __get_default_information_dict(self) -> Dict:
        return {
            "cpt_type": GefProperty(gef_key=4),
            "cpt_standard": GefProperty(gef_key=6),  # "and class [quality_class]"
            "vertical_datum": GefProperty(gef_key=8),
            "local_reference": GefProperty(gef_key=9),
        }

    def __get_default_property_dict(self) -> Dict:
        return {
            "penetration_length": GefColumnProperty(gef_key=1),
            "tip": GefColumnProperty(gef_key=2),
            "friction": GefColumnProperty(gef_key=3),
            "friction_nb": GefColumnProperty(gef_key=4),
            "pwp_u1": GefColumnProperty(gef_key=5),
            "pwp_u2": GefColumnProperty(gef_key=6),
            "pwp_u3": GefColumnProperty(gef_key=7),
            "inclination_resultant": GefColumnProperty(gef_key=8),
            "inclination_ns": GefColumnProperty(gef_key=9),
            "inclination_ew": GefColumnProperty(gef_key=10),
            "depth": GefColumnProperty(gef_key=11),
            "time": GefColumnProperty(gef_key=12),
            "corrected_tip": GefColumnProperty(gef_key=13),
            "net_tip": GefColumnProperty(gef_key=14),
            "pore_ratio": GefColumnProperty(gef_key=15),
            "tip_nbr": GefColumnProperty(gef_key=16),
            "unit_weight": GefColumnProperty(gef_key=17),
            "pwp_ini": GefColumnProperty(gef_key=18),
            "total_pressure": GefColumnProperty(gef_key=19),
            "effective_pressure": GefColumnProperty(gef_key=20),
            "inclination_x": GefColumnProperty(gef_key=21),
            "inclination_y": GefColumnProperty(gef_key=22),
            "electric_cond": GefColumnProperty(gef_key=23),
            "magnetic_strength_x": GefColumnProperty(gef_key=31),
            "magnetic_strength_y": GefColumnProperty(gef_key=32),
            "magnetic_strength_z": GefColumnProperty(gef_key=33),
            "magnetic_strength_tot": GefColumnProperty(gef_key=34),
            "magnetic_inclination": GefColumnProperty(gef_key=35),
            "magnetic_declination": GefColumnProperty(gef_key=36),
        }

    @staticmethod
    def get_line_index_from_data_starts_with(code_string: str, data: List[str]) -> int:
        """Given a list of strings it returns the position of the first one which starts by the code string given.

        Args:
            code_string (str): The line that needs to be found.
            data (List[str]): Collection of strings representing lines.

        Raises:
            ValueError: When the code_string argument is not given or is None.
            ValueError: When the data argument is not given or is None.
            ValueError: When no values where found for the given arguments.

        Returns:
            int: Line index where the data can be found.
        """
        if not code_string:
            raise ValueError(code_string)
        if not data:
            raise ValueError(data)

        line_found = next(
            (i for i, val in enumerate(data) if val.startswith(code_string)), None
        )
        if not line_found:
            # if not having line_found IS NOT okay.
            raise ValueError(
                f"No values found for field {code_string} of the gef file."
            )
        return line_found

    @staticmethod
    def get_line_from_data_that_ends_with(code_string: str, data: List[str]) -> str:
        """Given a list of strings it returns the first one ending with the given code_string.

        Args:
            code_string (str): Code string to find at the end of each line.
            data (List[str]): List of strings to iterate through.

        Raises:
            ValueError: When no code_string argument is given.
            ValueError: When no data argument is given.

        Returns:
            str: Line ending with the requested code_string.
        """
        if not code_string:
            raise ValueError(code_string)
        if not data:
            raise ValueError(data)

        return next((line for line in data if line.endswith(code_string)), None)

    def read_file(self, filepath: Path) -> dict:
        return self.read_gef(gef_file=filepath)

    def read_gef(self, gef_file: Path, fct_a: float = 0.8) -> Dict:
        """
        Opens and reads gef file. Returns dictionary containing all possible
        inputs from gef file.
        """
        # read gef file
        with open(gef_file, "r") as f:
            data = f.readlines()

        # search NAP
        idx_nap = GefFileReader.get_line_index_from_data_starts_with(
            code_string=r"#ZID=", data=data
        )
        NAP = float(data[idx_nap].split(",")[1])
        # search end of header
        idx_EOH = GefFileReader.get_line_index_from_data_starts_with(
            code_string=r"#EOH=", data=data
        )
        # # search for coordinates
        idx_coord = GefFileReader.get_line_index_from_data_starts_with(
            code_string=r"#XYID=", data=data
        )

        # read result time
        result_time = self.read_date_cpt(data)

        # get values for information dict
        for key_name in self.information_dict:
            self.information_dict[
                key_name
            ].values_from_gef = self.read_information_for_gef_data(key_name, data)

        # search index depth
        for key_name in self.property_dict:
            gef_column_index = self.read_column_index_for_gef_data(
                self.property_dict[key_name].gef_key, data
            )
            self.property_dict[key_name].gef_column_index = gef_column_index

        # read error codes
        idx_errors_raw_text = [
            val.split(",")[1] for val in data if val.startswith(r"#COLUMNVOID=")
        ]
        self.match_idx_with_error(idx_errors_raw_text)
        # rewrite data with separator ;
        data[idx_EOH + 1 :] = [
            re.sub("[ :,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1 :]
        ]

        # search line with coefficient a
        line_found = GefFileReader.get_line_from_data_that_ends_with(
            code_string="Netto oppervlaktequotient van de conuspunt\n", data=data
        )
        if line_found:
            try:
                fct_a = float(line_found.split(",")[1])
            except (ValueError, TypeError):
                # We keep on with the default value.
                fct_a = fct_a

        # remove empty lines
        data = list(filter(None, data))

        # read data & correct depth to NAP
        self.read_column_data(data, idx_EOH)

        # remove the points with error: value == -9999
        self.remove_points_with_error()
        # if tip / friction / friction number are negative -> zero
        correct_for_negatives = ["tip", "friction", "friction_nb"]
        self.correct_negatives_and_zeros(correct_for_negatives)

        idx_name = GefFileReader.get_line_index_from_data_starts_with(
            code_string=r"#TESTID=", data=data
        )
        self.name = data[idx_name].split("#TESTID=")[-1].strip()
        # From the line of the coordinates retrieve the text line
        # and return list of coordinates [x,y]
        self.coord = list(
            map(
                float,
                re.sub("[ ,!\t]+", ";", data[idx_coord].strip())
                .split("#XYID=")[-1]
                .split(";")[2:4],
            )
        )

        # todo move calculation z_NAP outside reader
        # z_NAP = [
        #     NAP - i for j, i in enumerate(self.property_dict["depth"].values_from_gef)
        # ]

        return dict(
            name=self.name,
            penetration_length=self.get_as_np_array(
                self.property_dict["penetration_length"].values_from_gef
            ),
            depth=self.get_as_np_array(self.property_dict["depth"].values_from_gef),
            tip=self.get_as_np_array(self.property_dict["tip"].values_from_gef),
            friction=self.get_as_np_array(
                self.property_dict["friction"].values_from_gef
            ),
            friction_nbr=self.get_as_np_array(
                self.property_dict["friction_nb"].values_from_gef
            ),
            a=fct_a,
            coordinates=self.coord,
            pore_pressure_u1=self.get_as_np_array(
                self.property_dict["pwp_u1"].values_from_gef
            ),
            pore_pressure_u2=self.get_as_np_array(
                self.property_dict["pwp_u2"].values_from_gef
            ),
            pore_pressure_u3=self.get_as_np_array(
                self.property_dict["pwp_u3"].values_from_gef
            ),
            inclination_resultant=self.get_as_np_array(
                self.property_dict["inclination_resultant"].values_from_gef
            ),
            inclination_ns=self.get_as_np_array(
                self.property_dict["inclination_ns"].values_from_gef
            ),
            inclination_ew=self.get_as_np_array(
                self.property_dict["inclination_ew"].values_from_gef
            ),
            inclination_x=self.get_as_np_array(
                self.property_dict["inclination_x"].values_from_gef
            ),
            inclination_y=self.get_as_np_array(
                self.property_dict["inclination_y"].values_from_gef
            ),
            time=self.get_as_np_array(self.property_dict["time"].values_from_gef),
            qt=self.get_as_np_array(
                self.property_dict["corrected_tip"].values_from_gef
            ),
            net_tip=self.get_as_np_array(self.property_dict["net_tip"].values_from_gef),
            pore_ratio=self.get_as_np_array(
                self.property_dict["pore_ratio"].values_from_gef
            ),
            tip_nbr=self.get_as_np_array(self.property_dict["tip_nbr"].values_from_gef),
            unit_weight_measured=self.get_as_np_array(
                self.property_dict["unit_weight"].values_from_gef
            ),
            pwp_ini=self.get_as_np_array(self.property_dict["pwp_ini"].values_from_gef),
            total_pressure_measured=self.get_as_np_array(
                self.property_dict["total_pressure"].values_from_gef
            ),
            effective_pressure_measured=self.get_as_np_array(
                self.property_dict["effective_pressure"].values_from_gef
            ),
            electric_cond=self.get_as_np_array(
                self.property_dict["electric_cond"].values_from_gef
            ),
            magnetic_strength_x=self.get_as_np_array(
                self.property_dict["magnetic_strength_x"].values_from_gef
            ),
            magnetic_strength_y=self.get_as_np_array(
                self.property_dict["magnetic_strength_y"].values_from_gef
            ),
            magnetic_strength_z=self.get_as_np_array(
                self.property_dict["magnetic_strength_z"].values_from_gef
            ),
            magnetic_strength_tot=self.get_as_np_array(
                self.property_dict["magnetic_strength_tot"].values_from_gef
            ),
            magnetic_inclination=self.get_as_np_array(
                self.property_dict["magnetic_inclination"].values_from_gef
            ),
            magnetic_declination=self.get_as_np_array(
                self.property_dict["magnetic_declination"].values_from_gef
            ),
            local_reference_level=NAP,
            vertical_datum=self.information_dict["vertical_datum"].values_from_gef,
            local_reference=self.information_dict["local_reference"].values_from_gef,
            cpt_standard=self.information_dict["cpt_standard"].values_from_gef,
            quality_class=self.information_dict["cpt_standard"].values_from_gef,
            cpt_type=self.information_dict["cpt_type"].values_from_gef,
            result_time=result_time,
        )

    def get_as_np_array(self, values_from_gef: Iterable):
        """
        Converts iterable to np array
        :param values_from_gef:
        :return:
        """
        return np.array(values_from_gef)

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

    def read_date_cpt(self, data: List[str]) -> str:
        """
        Reads the date from the cpt. If startdate is present, the date is defined as the startdate.
        Else the date is equal to the filedate, which is always present.
        :return:
        """
        code_file_date = r"#FILEDATE= "
        code_start_date = r"#STARTDATE= "
        result_date = None
        for val in data:
            if val.startswith(code_file_date) and result_date is None:
                result_date = val.split(code_file_date)[-1]
            if val.startswith(code_start_date):
                result_date = val.split(code_start_date)[-1]
                return result_date.replace("\n", "")
            if val.startswith(r"#EOH="):
                return result_date.replace("\n", "")

    def read_information_for_gef_data(self, key_name: str, data: List[str]) -> str:
        """
        Reads header information from the gef data.
        """
        code_string = r"#MEASUREMENTTEXT= " + str(
            self.information_dict[key_name].gef_key
        )

        for val in data:
            if val.startswith(code_string):
                information = val.split(code_string)[-1]
                information = information.replace("\n", "")
                return information.replace(", ", "", 1)
            if val.startswith(r"#EOH="):
                return ""
        return ""

    def match_idx_with_error(
        self,
        idx_errors: List[str],
    ) -> None:
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
                    if key in ["penetration_length", "tip"]:
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
                temp_list = self.property_dict[key].values_from_gef.copy()
                for number, value in enumerate(temp_list):
                    if self.property_dict[key].values_from_gef[number - deleted_rows]:
                        if math.isclose(
                            self.property_dict[key].values_from_gef[
                                number - deleted_rows
                            ],
                            self.property_dict[key].error_code,
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

    def read_column_data(self, data: List[str], idx_EOH: int) -> None:
        """
        Read column data from the gef file table.
        """
        pore_pressure_types = ["pwp_u1", "pwp_u2", "pwp_u3"]
        for key in self.property_dict:
            if key in pore_pressure_types and not (
                self.property_dict[key].gef_column_index
            ):
                # Pore pressures are not inputted
                self.property_dict[key].values_from_gef = np.zeros(
                    len(self.property_dict["penetration_length"].values_from_gef)
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
                    if key in ["penetration_length", "tip"]:
                        raise Exception(f"CPT key: {key} not part of GEF file")
                    else:
                        warning(f"Key {key} is not defined in the gef file.")
        return None

    def correct_negatives_and_zeros(self, correct_for_negatives: List[str]):
        """
        Values tip / friction / friction cannot be negative so they
        have to be zero.
        """
        if not correct_for_negatives:
            return
        for key in correct_for_negatives:
            if self.property_dict[key].gef_column_index is not None:
                self.property_dict[key].values_from_gef = np.array(
                    self.property_dict[key].values_from_gef
                )
                self.property_dict[key].values_from_gef[
                    self.property_dict[key].values_from_gef < 0
                ] = 0
