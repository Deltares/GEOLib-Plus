import math
from importlib import import_module
from typing import Any, Dict, Iterable, List, Tuple, Union

from geolib_plus import AbstractCPT


def validate_that_geolib_is_installed():
    try:
        import geolib
    except ModuleNotFoundError:
        # Error handling
        raise ModuleNotFoundError(
            "To run function create_profile_for_d_foundations geolib module should be installed."
        )


class DFoundationsConnector:
    """
    Class that has a number of function connecting GEOLIB+ with GEOLIB D-Foundations.
    """

    @staticmethod
    def create_profile_for_d_foundations(
        cpt: AbstractCPT,
    ) -> Tuple[Any, List[Any]]:
        """
        Function used to transform class into a Profile that can be inputted
        in D-Foundations through GEOLIB.
        """
        validate_that_geolib_is_installed()
        from geolib.geometry import Point
        from geolib.models.dfoundations.profiles import (
            Excavation as dfoundations_excavation,
        )
        from geolib.models.dfoundations.profiles import Profile as dfoundations_profile

        dfoundations_cpt = DFoundationsConnector.__define_cpt_inputs(cpt)

        # create layers for the dfoundations profile
        soil_layers = DFoundationsConnector.__to_layers_for_d_foundations(cpt)
        # soils should also be generated so they can later be inputted in the D-Foundations model
        soils = DFoundationsConnector.__to_d_foundations_soils(soil_layers)

        profile = dfoundations_profile(
            name=cpt.name,
            location=Point(x=cpt.coordinates[0], y=cpt.coordinates[1]),
            cpt=dfoundations_cpt,
            phreatic_level=DFoundationsConnector.__get_phreatic_level(cpt),
            pile_tip_level=DFoundationsConnector.__get_pile_tip_level(cpt),
            layers=[layer for layer in soil_layers],
            excavation=dfoundations_excavation(excavation_level=0),
        )
        return (profile, soils)

    @staticmethod
    def __if_not_none_add_to_dict(
        dictionary: Dict, input_list: Union[Iterable, None], name: str
    ) -> Dict:
        if input_list is not None:
            dictionary[name] = input_list
        return dictionary

    @staticmethod
    def __define_cpt_inputs(cpt: AbstractCPT) -> Any:
        """
        Function that creates a D-Foundations CPT from a GEOLIB+ CPT.
        """
        from geolib.models.dfoundations.profiles import CPT as dfoundations_cpt

        # check that at least depth and tip is available
        if cpt.depth is None:
            raise ValueError("Depth is not defined in the cpt.")
        if cpt.tip is None:
            raise ValueError("Tip is not defined in the cpt.")
        # values that are read from d-foundations are depth, qc, water
        # pressure, friction number
        inputs_cpt = {}
        inputs_cpt = DFoundationsConnector.__if_not_none_add_to_dict(
            inputs_cpt, cpt.depth_to_reference, "z"
        )
        inputs_cpt = DFoundationsConnector.__if_not_none_add_to_dict(
            inputs_cpt, cpt.tip, "qc"
        )
        inputs_cpt = DFoundationsConnector.__if_not_none_add_to_dict(
            inputs_cpt, cpt.water, "rw"
        )
        inputs_cpt = DFoundationsConnector.__if_not_none_add_to_dict(
            inputs_cpt, cpt.friction, "GEFFrict"
        )
        # dictonary of list to list of dictionaries
        inputs_cpt_new = [dict(zip(inputs_cpt, t)) for t in zip(*inputs_cpt.values())]
        # to cpt d-foundations
        return dfoundations_cpt(
            cptname=cpt.name,
            groundlevel=cpt.local_reference_level,
            pre_excavation=cpt.predrilled_z,
            measured_data=inputs_cpt_new,
        )

    @staticmethod
    def __to_layers_for_d_foundations(cpt: AbstractCPT) -> List:
        """
        Function that transform interpreted cpt to soil layers. For each layer a specific soil is defined with default parameters.
        """
        if cpt.depth_merged is None:
            raise ValueError(
                "Field 'depth_merged' was not defined in the inputted cpt. Interpretation of the cpt must be performed. "
            )
        if cpt.lithology_merged is None:
            raise ValueError(
                "Field 'lithology_merged' was not defined in the inputted cpt. Interpretation of the cpt must be performed. "
            )

        soil_layers = []
        # cpt.depth_merged will always have len(cpt.lithology_merged) + 1 length as it includes the bottom.
        for index, depth in enumerate(cpt.depth_merged[:-1]):
            soil_layers.append(
                dict(
                    name=cpt.lithology_merged[index],
                    material=cpt.lithology_merged[index],
                    top_level=-1 * depth,
                )
            )
        return soil_layers

    @staticmethod
    def __get_pile_tip_level(cpt: AbstractCPT) -> float:
        """
        Calculates the value of the pile tip level. As default value, D-FOUNDATIONS uses the depth of the CPT point
        with the maximum cone resistance raised by 0.8 m
        """
        if cpt.tip is None:
            raise ValueError("Tip is not defined in the cpt.")
        max_value = max(cpt.tip)
        max_index = list(cpt.tip).index(max_value)
        return cpt.depth_to_reference[max_index] + 0.8

    @staticmethod
    def __get_phreatic_level(cpt: AbstractCPT) -> float:
        """
        Calculates the value of the phreatic level which should be inputted in D-Foundations.The default value used by
        D-FOUNDATIONS corresponds to the ground level of the imported CPT file lowered by 0.5 m.
        """
        if cpt.water is not None:
            for water_pressure in cpt.water:
                if not (math.isclose(water_pressure, 0.0)):
                    max_index = list(cpt.water).index(water_pressure)
                    return cpt.depth_to_reference[max_index]
        return cpt.depth_to_reference[0] - 0.5

    @staticmethod
    def __to_d_foundations_soils(soil_layers) -> Any:
        """
        Creates a list of all the soils. Each layer of the profile has a different soil defined. However, the values are set to default.
        The user can specify those for each layer before inputting them into D-Foundations.
        """
        from geolib.soils import Soil, SoilType

        soils = []
        for layer in soil_layers:
            if layer["material"] not in [soil.name for soil in soils]:
                local_soil = Soil(name=layer["material"], soil_type_nl=SoilType.SAND)
                local_soil.mohr_coulomb_parameters.cohesion.mean = 0
                local_soil.mohr_coulomb_parameters.friction_angle.mean = 0
                local_soil.undrained_parameters.undrained_shear_strength = 0
                soils.append(local_soil)
        return soils
