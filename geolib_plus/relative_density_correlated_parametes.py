import math
from enum import IntEnum
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel


class RelativeDensityCorrelatedParameters(BaseModel):
    """
    Class that uses input of relative density and empirical formulas to
    calculated numerous model parameters for sands. According to Brinkgreve :cite:`Brinkgreve_2010`

    Attributes
    ----------
    RD_percentage : Optional[Union[np.ndarray, float]]
        relative density of soil in percentage
    gamma_unsat : Optional[Union[np.ndarray, float]]
        unsaturated unit weight of soil
    gamma_sat : Optional[Union[np.ndarray, float]]
        saturated unit weight of soil
    E_50_ref : Optional[Union[np.ndarray, float]]
        secant stiffness in a standard triaxial test
    E_oed_ref : Optional[Union[np.ndarray, float]]
        tangent stiffness for primary loading
    E_ur_ref : Optional[Union[np.ndarray, float]]
        unloading/reloading stiffness
    G_o_ref : Optional[Union[np.ndarray, float]]
        maximal small-strain shear modulus
    m : Optional[Union[np.ndarray, float]]
        power for stress-level dependancy of stiffness
    R_f : Optional[Union[np.ndarray, float]]
        failure ratio
    phi: Optional[Union[np.ndarray, float]]
        friction angle
    psi: Optional[Union[np.ndarray, float]]
        dilation angle
    """

    RD_percentage: Union[np.ndarray, float]
    gamma_unsat: Union[np.ndarray, float]
    gamma_sat: Union[np.ndarray, float]
    E_50_ref: Union[np.ndarray, float]
    E_oed_ref: Union[np.ndarray, float]
    E_ur_ref: Union[np.ndarray, float]
    G_o_ref: Union[np.ndarray, float]
    m: Union[np.ndarray, float]
    R_f: Union[np.ndarray, float]
    phi: Union[np.ndarray, float]
    psi: Union[np.ndarray, float]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def calculate_using_RD(cls, RD_percentage: Union[np.ndarray, float]):
        r"""
        This method creates class that stores all parameters calculated with the input of RD,
        using the following equations :cite:`Brinkgreve_2010`

        .. math::

            \gamma_{unsat} = 15 + 4 \frac{RD}{100}

        .. math::

            \gamma_{sat} = 19 + 1.6 \frac{RD}{100}

        .. math::

            E_{50,ref} = 60000 \frac{RD}{100}

        .. math::

            E_{ur,ref} = 60000 \frac{RD}{100}

        .. math::

            G_{o,ref} = 60000 + 68000 \frac{RD}{100}

        .. math::

            m = 0.7 - \frac{RD}{320}

        .. math::

            R_{f} = 1 - \frac{RD}{800}

        .. math::

            \phi = 28 + 12.5 \frac{RD}{100}

        .. math::

            \psi = -2 + 12.5 \frac{RD}{100}


        """

        gamma_unsat = 15 + 4 * RD_percentage / 100
        gamma_sat = 19 + 1.6 * RD_percentage / 100
        E_50_ref = 60000 * RD_percentage / 100
        E_oed_ref = 60000 * RD_percentage / 100
        E_ur_ref = 180000 * RD_percentage / 100
        G_o_ref = 60000 + 68000 * RD_percentage / 100
        m = 0.7 - RD_percentage / 320
        R_f = 1 - RD_percentage / 800
        phi = 28 + 12.5 * RD_percentage / 100
        psi = -2 + 12.5 * RD_percentage / 100

        return cls(
            RD_percentage=RD_percentage,
            gamma_sat=gamma_sat,
            gamma_unsat=gamma_unsat,
            E_50_ref=E_50_ref,
            E_oed_ref=E_oed_ref,
            E_ur_ref=E_ur_ref,
            G_o_ref=G_o_ref,
            m=m,
            R_f=R_f,
            phi=phi,
            psi=psi,
        )
