from pathlib import Path
from typing import Optional, Iterable, List, Type, Dict, Union
from pydantic import BaseModel
import math

from geolib_plus import AbstractCPT

from geolib.models.dfoundations.profiles import CPT as dfoundations_cpt
from geolib.models.dfoundations.profiles import Profile as dfoundations_profile
from geolib.models.dfoundations.profiles import Excavation as dfoundations_excavation
from geolib.geometry import Point
from geolib.soils import Soil, SoilType


class SoilLayer(BaseModel):
    """
    Soil layer class. Used to construct layers from soil parameters.
    """

    name: str
    material: str
    top_level: float
    excess_pore_pressure_top: float = 0
    excess_pore_pressure_bottom: float = 0
    ocr_value: float = 0
    reduction_core_resistance: float = 0


class SoilProfile(BaseModel):
    """
    Soil profile class which concists of different soil layers.
    """

    cpt: dfoundations_cpt = None
    soil_layers: List[SoilLayer] = []
    soils: List[Soil] = []
    profile: dfoundations_profile = None

    @staticmethod
    def if_not_none_add_to_dict(
        dictionary: Dict, input_list: Union[Iterable, None], name: str
    ):
        if input_list is not None:
            dictionary[name] = input_list
        return dictionary

    def define_cpt_inputs(self, cpt: AbstractCPT):
        """
        Function that creates a D-Foundations CPT from a GEOLIB+ CPT.
        """
        # check that at least depth and tip is available
        if cpt.depth is None:
            raise ValueError("Depth is not defined in the cpt.")
        if cpt.tip is None:
            raise ValueError("Tip is not defined in the cpt.")
        # values that are read from d-foundations are depth, qc, water
        # pressure, friction number
        inputs_cpt = {}
        inputs_cpt = SoilProfile.if_not_none_add_to_dict(
            inputs_cpt, cpt.depth_to_reference, "z"
        )
        inputs_cpt = SoilProfile.if_not_none_add_to_dict(inputs_cpt, cpt.tip, "qc")
        inputs_cpt = SoilProfile.if_not_none_add_to_dict(inputs_cpt, cpt.water, "rw")
        inputs_cpt = SoilProfile.if_not_none_add_to_dict(
            inputs_cpt, cpt.friction, "GEFFrict"
        )
        # dictonary of list to list of dictionaries
        inputs_cpt_new = [dict(zip(inputs_cpt, t)) for t in zip(*inputs_cpt.values())]
        # to cpt d-foundations
        self.cpt = dfoundations_cpt(
            cptname=cpt.name,
            groundlevel=cpt.local_reference_level,
            pre_excavation=cpt.predrilled_z,
            measured_data=inputs_cpt_new,
        )

    @classmethod
    def create_profile_for_d_foundations(cls, cpt: AbstractCPT):
        """
        Function used to transform class into a Profile that can be inputted
        in D-Foundations through GEOLIB.
        """
        classmethod_profile = cls()
        classmethod_profile.define_cpt_inputs(cpt)

        # create layers for the dfoundations profile
        classmethod_profile.to_layers_for_d_foundations(cpt)
        # soils should also be generated so they can later be inputted in the D-Foundations model
        classmethod_profile.to_d_foundations_soils()

        classmethod_profile.profile = dfoundations_profile(
            name=cpt.name,
            location=Point(x=cpt.coordinates[0], y=cpt.coordinates[1]),
            cpt=classmethod_profile.cpt,
            phreatic_level=SoilProfile.get_phreatic_level(cpt),
            pile_tip_level=SoilProfile.get_pile_tip_level(cpt),
            layers=[dict(layer) for layer in classmethod_profile.soil_layers],
            excavation=dfoundations_excavation(excavation_level=0),
        )
        return classmethod_profile

    def to_layers_for_d_foundations(self, cpt: AbstractCPT):
        """
        Function that transform interpreted cpt to soil layers. For each layer a specific soil is defined with default parameters.
        """
        if cpt.depth_merged is None:
            raise ValueError(
                "Field 'depth_merged' was not defined in the inputted cpt. Interpretation of the cpt must be performed. "
            )

        for index, depth in enumerate(cpt.depth_merged):
            soil_layer = SoilLayer(
                name="GEOLIB_plus_" + str(index),
                material="GEOLIB_plus_" + str(index),
                top_level=-1 * depth,
            )
            self.soil_layers.append(soil_layer)

    @staticmethod
    def get_pile_tip_level(cpt: AbstractCPT) -> float:
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
    def get_phreatic_level(cpt: AbstractCPT) -> float:
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

    def to_d_foundations_soils(self):
        """
        Creates a list of all the soils. Each layer of the profile has a different soil defined. However, the values are set to default.
        The user can specify those for each layer before inputting them into D-Foundations.
        """
        for layer in self.soil_layers:
            local_soil = Soil(name=layer.material, soil_type_nl=SoilType.SAND)
            local_soil.mohr_coulomb_parameters.cohesion.mean = 0
            local_soil.mohr_coulomb_parameters.friction_angle.mean = 0
            local_soil.undrained_parameters.undrained_shear_strength = 0
            self.soils.append(local_soil)
