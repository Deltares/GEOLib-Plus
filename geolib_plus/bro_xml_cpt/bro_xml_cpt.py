import pandas as pd
import numpy as np
from typing import Iterable, List
import math

# External modules
from .bro_utils import XMLBroCPTReader
from geolib_plus.cpt_base_model import AbstractCPT, CptReader


class BroXmlCpt(AbstractCPT):
    """
    Class that contains gets the reader and contains pre-processing functions.
    That can be used to modify the raw data.
    """

    @classmethod
    def get_cpt_reader(cls) -> CptReader:
        return XMLBroCPTReader()

    @property
    def __water_measurement_types(self) -> List[str]:
        return [
            "pore_pressure_u1",
            "pore_pressure_u2",
            "pore_pressure_u3",
        ]

    @property
    def __list_of_array_values(self):
        return [
            "tip",
            "friction_nbr",
            "penetration_length",
            "depth",
            "time",
            "qt",
            "Qtn",
            "net_tip",
            "magnetic_strength_x",
            "magnetic_strength_y",
            "magnetic_strength_z",
            "magnetic_strength_tot",
            "electric_cond",
            "inclination_ew",
            "inclination_ns",
            "inclination_x",
            "inclination_y",
            "inclination_resultant",
            "magnetic_inclination",
            "magnetic_declination",
            "friction",
            "pore_ratio",
            "temperature",
            "pore_pressure_u1",
            "pore_pressure_u2",
            "pore_pressure_u3",
            "IC",
            "water",
        ]

    def remove_points_with_error(self):
        """
        Updates fields by removing depths that contain values with nan values. i.e. incomplete data
        This means that all the properties in __list_of_array_values will be updated.
        """
        # transform to dataframe
        update_dict = {}
        for value in self.__list_of_array_values:
            # ignore None values
            if getattr(self, value) is not None:
                update_dict[value] = getattr(self, value)
        # perform action
        update_dict = pd.DataFrame(update_dict).dropna().to_dict("list")
        # update changed values in cpt
        for value in self.__list_of_array_values:
            setattr(self, value, np.array(update_dict.get(value)))
        return

    def has_points_with_error(self) -> bool:
        """
        A routine which checks whether the data is free of points with error.

        :return: If the gef cpt data is free of error flags
        """
        for key in self.__list_of_array_values:
            current_attribute = getattr(self, key)
            print(current_attribute)
            if current_attribute is not None and np.isnan(np.array(current_attribute)).any():
                return True
        return False

    def drop_duplicate_depth_values(self):
        """
        Updates fields by removing penetration_length that are duplicate.
        This means that all the properties in __list_of_array_values will be updated.
        """
        # TODO maybe here it makes more sense for the user to define what should not be duplicate
        # transform to dataframe
        update_dict = {}
        for value in self.__list_of_array_values:
            # ignore None values
            if getattr(self, value) is not None:
                update_dict[value] = getattr(self, value)
        # perform action
        update_dict = (
            pd.DataFrame(update_dict)
                .drop_duplicates(subset="penetration_length", keep="first")
                .to_dict("list")
        )
        # update changed values in cpt
        for value in self.__list_of_array_values:
            setattr(self, value, np.array(update_dict.get(value)))
        return

    def has_duplicated_depth_values(self) -> bool:
        """
        Check to see if there are any duplicate depth positions in the data
        :return True if has duplicate depths points based on penetration length
        """
        return len(np.unique(self.penetration_length)) != len(self.penetration_length)

    def pre_process_data(self):
        """
        Standard pre processes data which is read from bro xml files.
        #todo extend
        :return:
        """

        self.remove_points_with_error()
        self.drop_duplicate_depth_values()
        self.perform_pre_drill_interpretation()
        super().pre_process_data()





