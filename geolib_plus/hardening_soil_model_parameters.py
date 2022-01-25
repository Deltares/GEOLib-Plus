import math
from enum import IntEnum
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel


class HardingSoilCalculationType(IntEnum):
    COMPRESSIBILITYPARAMETERS = 1
    CONERESISTANCE = 2


class HardeningSoilModelParameters(BaseModel):
    """
    Class that calculates and stores parameters for the hardening soil model.

    Attributes
    ----------
    E_50_ref : Optional[Union[np.ndarray, float]]
        secant stiffness in a standard triaxial test
    E_oed_ref : Optional[Union[np.ndarray, float]]
        tangent stiffness for primary loadind
    m : Optional[Union[np.ndarray, float]] = None
        power for stress-level dependancy of stiffness
    v_ur : Optional[Union[np.ndarray, float]] = None
        Poisson's ratio for unloading-reloading
    E_ur_ref : Optional[Union[np.ndarray, float]] = None
        unloading/reloading stiffness
    p_ref : Optional[Union[np.ndarray, float]] = None
        reference stress for stiffness
    K0_NC : Optional[Union[np.ndarray, float]] = None
        K0-value for normal consolidation
    R_f : Optional[Union[np.ndarray, float]] = None
        failure ratio
    qc: Optional[Union[np.ndarray, float]] = None
        cone resistance
    sigma_ref_h: Optional[Union[np.ndarray, float]] = None
        horizontal reference stress
    sigma_cpt_h: Optional[Union[np.ndarray, float]] = None
        horizontal cpt stress
    Cc: Optional[Union[np.ndarray, float]] = None
        compression index
    Cs: Optional[Union[np.ndarray, float]] = None
        swelling index
    eo: Optional[Union[np.ndarray, float]] = None
        initial void ratio
    sigma_ref_v: Optional[Union[np.ndarray, float]] = None
        vertical reference stress
    """

    E_50_ref: Optional[Union[np.ndarray, float]] = None
    E_oed_ref: Optional[Union[np.ndarray, float]] = None
    E_ur_ref: Optional[Union[np.ndarray, float]] = None
    m: Optional[Union[np.ndarray, float]] = None
    v_ur: Optional[Union[np.ndarray, float]] = None
    qc: Optional[Union[np.ndarray, float]] = None
    sigma_ref_h: Optional[Union[np.ndarray, float]] = None
    sigma_cpt_h: Optional[Union[np.ndarray, float]] = None
    Cc: Optional[Union[np.ndarray, float]] = None
    Cs: Optional[Union[np.ndarray, float]] = None
    eo: Optional[Union[np.ndarray, float]] = None
    sigma_ref_v: Optional[Union[np.ndarray, float]] = None
    p_ref: Optional[Union[np.ndarray, float]] = None
    K0_NC: Optional[Union[np.ndarray, float]] = None
    R_f: Optional[Union[np.ndarray, float]] = None

    class Config:
        arbitrary_types_allowed = True

    def check_if_available(self, attribute_name: str, calculation_type: str):
        if getattr(self, attribute_name) is None:
            raise AttributeError(
                "{name} input should be defined for the {calculation_type} calculation type.".format(
                    name=attribute_name, calculation_type=calculation_type
                )
            )

    def calculate_stiffness(self, calculation_type: HardingSoilCalculationType) -> None:
        r"""
        Function that calculates hardening soil parameters based on the two following calculation types

        Based on the compressibility parameters:

        .. math::

            E_{oed,ref} = (\frac{ln(10)(1+e_{0})\sigma_{ref.v}}{C_{C}})

        .. math::

            E_{ur,ref} = (\frac{ln(10)(1+e_{0})\sigma_{ref.v}}{C_{s}})(\frac{(1+v_{ur})(1-2v_{ur})}{(1-v_{ur})})


        Based on the cone resistance:

        .. math::

            G_{0} = 10 q_{c}

        .. math::

            E_{ur,ref} = 0.5G_{0}2(1+v_{ur}) (\frac{\sigma_{ref.h}}{\sigma_{cpt.v}})^m

        .. math::

            E_{50,ref} = (\frac{E_{ur,ref}}{5})

        .. math::

            E_{oed,ref} = E_{50,ref}


        """
        if calculation_type == HardingSoilCalculationType.COMPRESSIBILITYPARAMETERS:

            attributes_to_be_checked = ["eo", "sigma_ref_v", "v_ur", "Cc", "Cs"]
            for attribute_name in attributes_to_be_checked:
                self.check_if_available(attribute_name, "COMPRESSIBILITYPARAMETERS")

            self.E_oed_ref = math.log(10) * (1 + self.eo) * self.sigma_ref_v / self.Cc

            self.E_ur_ref = (
                math.log(10) * (1 + self.eo) * self.sigma_ref_v / self.Cs
            ) * ((1 + self.v_ur) * (1 - 2 * self.v_ur) / (1 - self.v_ur))
        else:
            attributes_to_be_checked = ["qc", "v_ur", "sigma_ref_h", "sigma_cpt_h", "m"]
            for attribute_name in attributes_to_be_checked:
                self.check_if_available(attribute_name, "CONERESISTANCE")

            G_0 = 10 * self.qc

            self.E_ur_ref = (
                0.5
                * G_0
                * 2
                * (1 + self.v_ur)
                * np.power(self.sigma_ref_h * (1 / self.sigma_cpt_h), self.m)
            )

            self.E_50_ref = self.E_ur_ref / 5

            self.E_oed_ref = np.copy(self.E_50_ref)
