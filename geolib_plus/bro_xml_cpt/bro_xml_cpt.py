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
        ]

    def drop_nan_values(self):
        """
        Updates fields by removing depths that contain nan values.
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

    @staticmethod
    def update_value_with_pre_drill(
        local_depth: Iterable, values: Iterable, length_of_average_points: int
    ) -> Iterable:
        """
        Appends average value defined from length_of_average_points from missing
        inputs defined from the size of the local depth input.
        """
        # calculate the average along the depth for this value
        average = np.average(values[:length_of_average_points])
        # the pre-drill part consists of repeated values of this kind
        local_values = np.repeat(average, len(local_depth))
        # new values are appended to the result
        return np.append(local_values, values)

    def __correct_missing_samples_top_CPT(self, length_of_average_points: int):
        """
        All values except from the value of depth should be updated. This function
        inserts in the beginning of the arrays, the average value of the property
        for length length_of_average_points.
        """
        # add zero
        self.depth = np.append(0, self.depth)
        for value_name in self.__list_of_array_values:
            data = getattr(self, value_name)
            if (data is not None) and (value_name is not "depth"):
                if not (all(v is None for v in data)):
                    value_to_add = np.append(
                        np.average(data[:length_of_average_points]),
                        data,
                    )
                    setattr(self, value_name, value_to_add)
        return

    def perform_pre_drill_interpretation(self, length_of_average_points: int = 3):
        """
        Is performed only if pre-drill exists. Assumes that depth is already defined.
        If predrill exists it add the average value of tip, friction and friction number to the pre-drill length.
        The average is computed over the length_of_average_points.
        If pore water pressure is measured, the pwp is assumed to be zero at surface level.
        Parameters
        ----------
        :param cpt_BRO: BRO cpt dataset
        :param length_of_average_points: number of samples of the CPT to be used to fill pre-drill
        :return: depth, tip resistance, friction number, friction, pore water pressure
        """
        starting_depth = 0
        if not (math.isclose(float(self.predrilled_z), 0.0)):
            # if there is pre-dill add the average values to the pre-dill
            # Set the discretization
            discretization = np.average(np.diff(self.depth))
            # Define local data
            local_depth = np.arange(
                starting_depth, float(self.predrilled_z), discretization
            )
            for value_name in self.__list_of_array_values:
                # depth value and Nones should be skipped
                if (getattr(self, value_name) is not None) and (
                    value_name
                    not in [
                        "depth",
                        "pore_pressure_u1",
                        "pore_pressure_u2",
                        "pore_pressure_u3",
                    ]
                ):
                    if not (all(v is None for v in getattr(self, value_name))):
                        setattr(
                            self,
                            value_name,
                            self.update_value_with_pre_drill(
                                local_depth=local_depth,
                                values=getattr(self, value_name),
                                length_of_average_points=length_of_average_points,
                            ),
                        )
            # if there is pore water pressure
            # Here the endpoint is False so that for the final of
            # local_pore_pressure I don't end up with the same value
            # as the first in the Pore Pressure array.
            for water_measurement_type in self.__water_measurement_types:
                pore_pressure_type = getattr(self, water_measurement_type)
                if pore_pressure_type is not None:
                    if not (all(v is None for v in pore_pressure_type)):
                        local_pore_pressure = np.linspace(
                            0,
                            pore_pressure_type[0],
                            len(local_depth),
                            endpoint=False,
                        )
                        setattr(
                            self,
                            water_measurement_type,
                            np.append(
                                local_pore_pressure,
                                pore_pressure_type,
                            ),
                        )
            # Enrich the depth
            self.depth = np.append(
                local_depth,
                local_depth[-1] + discretization + self.depth - self.depth[0],
            )
        # correct for missing samples in the top of the CPT
        if self.depth[0] > 0:
            self.__correct_missing_samples_top_CPT(length_of_average_points)
        return

    def pre_process_data(self):
        """
        Pre processes data which is read from bro xml files.

        Units are converted to MPa.
        #todo extend
        :return:
        """
        self.drop_nan_values()
        self.drop_duplicate_depth_values()
        self.perform_pre_drill_interpretation()

        super().pre_process_data()
