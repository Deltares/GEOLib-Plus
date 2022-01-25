from enum import IntEnum
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel


class SoftSoilCreepParameters(BaseModel):
    """

    Class that calculates and stores parameters for the soft soil creep model.

    Attributes

    ----------

    c: Optional[Union[np.ndarray, float]] = None
        cohesion
    phi: Optional[Union[np.ndarray, float]] = None
        friction angle
    psi: Optional[Union[np.ndarray, float]] = None
        dilatancy angle
    kappa: Optional[Union[np.ndarray, float]] = None
        modified swelling index
    lambda_index: Optional[Union[np.ndarray, float]] = None
        modified compression index
    mi: Optional[Union[np.ndarray, float]] = None
        modified creep index
    v_ur: Optional[Union[np.ndarray, float]] = None
        poisson's ratio for unloading/reloading
    M: Optional[Union[np.ndarray, float]] = None
        slope of so-called 'critical state line'
    Cc: Optional[Union[np.ndarray, float]] = None
        compression index
    Cs: Optional[Union[np.ndarray, float]] = None
        swelling index
    eo: Optional[Union[np.ndarray, float]] = None
        initial void ratio
    OCR: Optional[Union[np.ndarray, float]] = None
        over consolidation ratio
    K0_NC: Optional[Union[np.ndarray, float]] = None
        K0-value for normal consolidation
    Ca: Optional[Union[np.ndarray, float]] = None
        material constant

    """

    c: Optional[Union[np.ndarray, float]] = None
    phi: Optional[Union[np.ndarray, float]] = None
    psi: Optional[Union[np.ndarray, float]] = None
    kappa: Optional[Union[np.ndarray, float]] = None
    lambda_index: Optional[Union[np.ndarray, float]] = None
    mu: Optional[Union[np.ndarray, float]] = None
    v_ur: Optional[Union[np.ndarray, float]] = None
    M: Optional[Union[np.ndarray, float]] = None
    Cc: Optional[Union[np.ndarray, float]] = None
    Cs: Optional[Union[np.ndarray, float]] = None
    eo: Optional[Union[np.ndarray, float]] = None
    OCR: Optional[Union[np.ndarray, float]] = None
    K0_NC: Optional[Union[np.ndarray, float]] = None
    Ca: Optional[Union[np.ndarray, float]] = None

    class Config:
        arbitrary_types_allowed = True

    def check_if_available(self, attribute_name: str):
        if getattr(self, attribute_name) is None:
            raise AttributeError(
                "{name} input should be defined to perform the soft soil parameter determination".format(
                    name=attribute_name
                )
            )

    def calculate_soft_soil_parameters(self):
        r"""

        Function that calculates soft soil parameters according to Vermeer :cite:`vermeer_neher_2019`.

        .. math::

            \lambda^{*} = (\frac{C_{c}}{ln(10)(1+e_{0})})

        .. math::

            \kappa^{*} = (\frac{C_{s}}{ln(10)(1+e_{0})}) \frac{ln(OCR)}{ln(\frac{2Ko^{NC}+1}{(2*Ko^{NC}+1) - (1 - \frac{1}{OCR})(2\frac{v_{ur}}{1-v_{ur}} + 1)})}

        .. math::

            \mu^{*} = (\frac{C_{c}}{ln(10)})


        """
        attributes_to_be_checked = ["OCR", "K0_NC", "v_ur", "Cc", "Cs", "eo"]
        for attribute_name in attributes_to_be_checked:
            self.check_if_available(attribute_name)
        self.lambda_index = self.Cc * (1 / (np.log(10) * (1 + self.eo)))
        a = 2 * self.K0_NC + 1
        b = 1 - 1 / self.OCR
        c = (2 * self.v_ur) * (1 / (1 - self.v_ur)) + 1
        d = 1 / np.log(a * (1 / (a - (b * c))))
        self.kappa = (
            (self.Cs * (1 / (np.log(10) * (1 + self.eo)))) * np.log(self.OCR) * d
        )
        self.mu = self.Ca / np.log(10)
