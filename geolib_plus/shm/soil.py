
# import packages
import os
import json
import numpy as np
from pathlib import Path
from enum import IntEnum
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, validator
from math import isfinite


class SoilBaseModel(BaseModel):
    class Config:
        extra = "forbid"

    @validator("*")
    def fail_on_infinite(cls, v, values, field):
        if isinstance(v, float) and not isfinite(v):
            raise ValueError(
                "Only finite values are supported, don't use nan, -inf or inf."
            )
        return v

class DistributionType(IntEnum):
    Undefined = 0
    Normal = 2
    LogNormal = 3
    Deterministic = 4

class StochasticParameter(SoilBaseModel):
    """
    Stochastic parameters class
    """

    is_probabilistic: bool = False
    mean: Optional[float] = None
    standard_deviation: Optional[float] = 0
    distribution_type: Optional[DistributionType] = DistributionType.Normal
    correlation_coefficient: Optional[float] = None
    low_characteristic_value: Optional[float] = None
    high_characteristic_value: Optional[float] = None
    low_design_value: Optional[float] = None
    high_design_value: Optional[float] = None
    limits: Optional[List] = None


class Soil(SoilBaseModel):
    """
    Schematisation manual macrostability soil class
    """

    name: Optional[str]
    unsaturated_weight: Optional[Union[float, StochasticParameter]] = StochasticParameter()
    saturated_weight: Optional[Union[float, StochasticParameter]] = StochasticParameter()
    shear_strength_ratio: Optional[
        Union[float, StochasticParameter]
    ] = StochasticParameter()

    strength_increase_exponent: Optional[
        Union[float, StochasticParameter]
    ] = StochasticParameter()

    cohesion: Optional[Union[float, StochasticParameter]] = StochasticParameter()
    dilatancy_angle: Optional[Union[float, StochasticParameter]] = StochasticParameter()
    friction_angle: Optional[Union[float, StochasticParameter]] = StochasticParameter()
    pop_layer: Optional[Union[float, StochasticParameter]] = StochasticParameter()



