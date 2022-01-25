import math
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Optional, Type

import numpy as np
from pydantic import BaseModel

from .plot_cpt import plot_cpt_norm
from .plot_settings import PlotSettings


class AbstractInterpretationMethod:
    """Base Interpretation method for analyzing CPTs."""


class CptReader:
    @abstractmethod
    def read_file(self, filepath: Path) -> dict:
        raise NotImplementedError(
            "The method should be implemented in concrete classes."
        )


class AbstractCPT(BaseModel):
    """Base CPT class, should define abstract."""

    #  variables
    penetration_length: Optional[Iterable]
    depth: Optional[Iterable]
    coordinates: Optional[List]
    local_reference_level: Optional[float]
    depth_to_reference: Optional[Iterable]
    tip: Optional[Iterable]
    friction: Optional[Iterable]
    friction_nbr: Optional[Iterable]
    a: Optional[float]
    name: Optional[str]
    rho: Optional[Iterable]
    total_stress: Optional[Iterable]
    effective_stress: Optional[Iterable]
    pwp: Optional[Iterable]
    qt: Optional[Iterable]
    Qtn: Optional[Iterable]
    Qtncs: Optional[Iterable]
    Fr: Optional[Iterable]
    IC: Optional[Iterable]
    n: Optional[Iterable]
    vs: Optional[Iterable]
    G0: Optional[Iterable]
    E0: Optional[Iterable]
    permeability: Optional[Iterable]
    poisson: Optional[Iterable]
    damping: Optional[Iterable]
    relative_density: Optional[Iterable]
    psi: Optional[Iterable]

    pore_pressure_u1: Optional[Iterable]
    pore_pressure_u2: Optional[Iterable]
    pore_pressure_u3: Optional[Iterable]

    water: Optional[Iterable]
    lithology: Optional[Iterable]
    litho_points: Optional[Iterable]

    lithology_merged: Optional[Iterable]
    depth_merged: Optional[Iterable]
    index_merged: Optional[Iterable]
    vertical_datum: Optional[str]
    local_reference: Optional[str]
    inclination_x: Optional[Iterable]
    inclination_y: Optional[Iterable]
    inclination_ns: Optional[Iterable]
    inclination_ew: Optional[Iterable]
    inclination_resultant: Optional[Iterable]
    time: Optional[Iterable]
    net_tip: Optional[Iterable]
    pore_ratio: Optional[Iterable]
    tip_nbr: Optional[Iterable]
    unit_weight_measured: Optional[Iterable]
    pwp_ini: Optional[Iterable]
    total_pressure_measured: Optional[Iterable]
    effective_pressure_measured: Optional[Iterable]
    electric_cond: Optional[Iterable]
    magnetic_strength_x: Optional[Iterable]
    magnetic_strength_y: Optional[Iterable]
    magnetic_strength_z: Optional[Iterable]
    magnetic_strength_tot: Optional[Iterable]
    magnetic_inclination: Optional[Iterable]
    magnetic_declination: Optional[Iterable]
    temperature: Optional[Iterable]
    predrilled_z: Optional[float]
    undefined_depth: Optional[float]

    cpt_standard: Optional[str]
    quality_class: Optional[str]
    cpt_type: Optional[str]
    result_time: Optional[str]
    water_measurement_type: Optional[str]

    # NEN results
    litho_NEN: Optional[Iterable]
    E_NEN: Optional[Iterable]
    cohesion_NEN: Optional[Iterable]
    fr_angle_NEN: Optional[Iterable]

    # plot settings
    plot_settings: PlotSettings = PlotSettings()

    # fixed values
    g: float = 9.81  # gravitational constant [m/s2]
    Pa: float = 100.0  # atmospheric pressure [kPa]

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

    def check_if_attribute(self, list_to_be_checked: List, method: str):
        for value in list_to_be_checked:
            if getattr(self, value) is None:
                raise ValueError(
                    "Value {} should be defined before running the \
                        {}. Make sure that pre_process \
                            method was run.".format(
                        value, method
                    )
                )

    def are_data_available_interpretation(self):
        list_to_be_checked = [
            "tip",
            "friction",
            "water",
            "Pa",
            "g",
            "friction_nbr",
            "depth",
            "depth_to_reference",
        ]
        self.check_if_attribute(
            list_to_be_checked=list_to_be_checked, method="interpretation"
        )

    def are_data_available_plotting(self):
        list_to_be_checked = [
            "undefined_depth",
            "local_reference_level",
            "depth_to_reference",
            "tip",
            "friction",
            "friction_nbr",
            "water",
            "name",
        ]
        self.check_if_attribute(
            list_to_be_checked=list_to_be_checked, method="plotting"
        )

    def check_if_lists_have_the_same_size(self):
        same_size = []
        for list_to_check in self.__list_of_array_values:
            value = getattr(self, list_to_check)
            if value is not None:
                same_size.append(len(value))
                if not (len(list(dict.fromkeys(same_size))) == 1):
                    raise ValueError(
                        "Property {} does not have the same size as the other properties".format(
                            list_to_check
                        )
                    )

    @classmethod
    def create_from(cls, filepath: Path):
        cls().read(filepath)

    def read(self, filepath: Path):

        if not filepath:
            raise ValueError(filepath)

        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(filepath)

        cpt_reader = self.get_cpt_reader()
        cpt_data = cpt_reader.read_file(filepath)
        for cpt_key, cpt_value in cpt_data.items():
            setattr(self, cpt_key, cpt_value)

    @abstractmethod
    def remove_points_with_error(self):
        raise NotImplementedError(
            "The method should be implemented in concrete classes."
        )

    @abstractmethod
    def has_points_with_error(self) -> bool:
        raise NotImplementedError(
            "The method should be implemented in concrete classes."
        )

    @abstractmethod
    def drop_duplicate_depth_values(self):
        raise NotImplementedError(
            "The method should be implemented in concrete classes."
        )

    def has_duplicated_depth_values(self):
        """
        Check to see if there are any duplicate depth positions in the data
        :return True if has duplicate depths points based on penetration length
        """
        if len(np.unique(self.penetration_length)) != len(self.penetration_length):
            raise ValueError(
                "Value depth contains duplicates. To resolve this run the pre_process method."
            )

    @classmethod
    @abstractmethod
    def get_cpt_reader(cls) -> CptReader:
        raise NotImplementedError("Should be implemented in concrete class.")

    class Config:
        arbitrary_types_allowed = True

    def interpret_cpt(self, method: AbstractInterpretationMethod):
        method.interpret(self)

    def __calculate_corrected_depth(self) -> np.ndarray:
        """
        Correct the penetration length with the inclination angle


        :param penetration_length: measured penetration length
        :param inclination: measured inclination of the cone
        :return: corrected depth
        """
        corrected_d_depth = np.diff(self.penetration_length) * np.cos(
            np.radians(np.nan_to_num(self.inclination_resultant[:-1], nan=0.0))
        )
        corrected_depth = np.concatenate(
            (
                self.penetration_length[0],
                self.penetration_length[0] + np.cumsum(corrected_d_depth),
            ),
            axis=None,
        )
        return corrected_depth

    def calculate_depth(self):
        """
        If depth is present in the cpt and is valid, the depth is parsed from depth
        elseif resultant inclination angle is present and valid in the cpt, the penetration length is corrected with
        the inclination angle.
        if both depth and inclination angle are not present/valid, the depth is parsed from the penetration length.
        :return:
        """

        if self.depth is not None:
            # no calculations needed
            return
        if self.inclination_resultant is not None:
            self.depth = self.__calculate_corrected_depth()
        else:
            self.depth = deepcopy(self.penetration_length)

    @staticmethod
    def __correct_for_negatives(data: np.ndarray) -> np.ndarray:
        """
        Values tip / friction / friction cannot be negative so they
        have to be zero.
        """
        if data is not None:
            if data.size != 0 and not data.ndim:
                data[data < 0] = 0
            return data

    def __get_water_data(self):

        self.water = None

        pore_pressure_data = [
            self.pore_pressure_u1,
            self.pore_pressure_u2,
            self.pore_pressure_u3,
        ]

        for data in pore_pressure_data:
            if data is not None:
                if data.size and data.ndim and not np.all(data == 0):
                    self.water = deepcopy(data)
                    break

        if self.water is None:
            self.water = np.zeros(len(self.penetration_length))

    def __calculate_inclination_resultant(self):

        if self.inclination_resultant is None:
            if isinstance(self.inclination_x, np.ndarray) and isinstance(
                self.inclination_y, np.ndarray
            ):
                self.inclination_resultant = np.sqrt(
                    np.square(self.inclination_x) + np.square(self.inclination_y)
                )
            elif isinstance(self.inclination_ns, np.ndarray) and isinstance(
                self.inclination_ew, np.ndarray
            ):
                self.inclination_resultant = np.sqrt(
                    np.square(self.inclination_ns) + np.square(self.inclination_ew)
                )

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
        self.penetration_length = np.append(0, self.penetration_length)
        for value_name in self.__list_of_array_values:
            data = getattr(self, value_name)
            if (
                (data is not None)
                and (value_name != "depth")
                and (value_name != "penetration_length")
            ):
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

        # if the depth of unknown data is not set then assume it is the first sample of penetration length
        if self.undefined_depth is None:
            self.undefined_depth = self.penetration_length[0]

        if not (math.isclose(float(self.undefined_depth), 0.0)):
            # if there is pre-drill add the average values to the pre-drill
            # Set the discretization
            discretization = np.average(np.diff(self.penetration_length))
            # Define local data
            local_depth = np.arange(
                starting_depth, float(self.undefined_depth), discretization
            )
            for value_name in self.__list_of_array_values:
                # depth value and Nones should be skipped
                if (getattr(self, value_name) is not None) and (
                    value_name
                    not in [
                        "penetration_length",
                        "depth",
                        "pore_pressure_u1",
                        "pore_pressure_u2",
                        "pore_pressure_u3",
                        "water",
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
            self.penetration_length = np.append(
                local_depth,
                local_depth[-1]
                + discretization
                + self.penetration_length
                - self.penetration_length[0],
            )
        # correct for missing samples in the top of the CPT
        if self.depth[0] - 1e-9 > 0:
            self.__correct_missing_samples_top_CPT(length_of_average_points)
        return

    def calculate_friction_nbr_if_not_available(self):
        """
        Calculates friction number if it is not present in the the cpt input file. Friction number is calculated by
        applying: sleeve friction / cone resistance * 100

        """
        if (
            self.friction_nbr is None
            and self.tip is not None
            and self.friction is not None
        ):

            self.friction_nbr = np.zeros(len(self.tip))

            # find indices where both tip resistance and sleeve friction are 0
            non_zero_indices = (self.tip > 0) * (self.friction > 0)

            # if both sleeve friction and tip resistance are greater than 0, calculate friction number
            self.friction_nbr[non_zero_indices] = self.friction / self.tip * 100

            self.friction_nbr = self.friction / self.tip * 100

    def pre_process_data(self):
        """
        Standard pre-processes data which is read from a gef file or bro xml file.

        Depth is calculated based on available data.
        Relevant data is corrected for negative values.
        Pore pressure is retrieved from available data.
        #todo extend
        :return:
        """

        self.remove_points_with_error()

        self.drop_duplicate_depth_values()

        self.__calculate_inclination_resultant()
        self.calculate_depth()
        self.perform_pre_drill_interpretation()

        self.depth_to_reference = self.local_reference_level - self.depth

        # correct tip friction and friction number for negative values
        self.tip = self.__correct_for_negatives(self.tip)
        self.friction = self.__correct_for_negatives(self.friction)
        self.friction_nbr = self.__correct_for_negatives(self.friction_nbr)

        self.calculate_friction_nbr_if_not_available()

        self.__get_water_data()

    def plot(self, directory: Path):
        # plot cpt data
        try:
            plot_cpt_norm(self, directory, self.plot_settings.general_settings)

        except (ValueError, IndexError):
            print("Cpt data and/or settings are not valid")
        except PermissionError as error:
            print(error)
