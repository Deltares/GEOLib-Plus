__version__ = "0.1.0"

from abc import abstractmethod
from pathlib import Path
from pydantic import BaseModel

class AbstractInterpretationMethod:
    """Base Interpretation method for analyzing CPTs."""


class RobertsonMethod(AbstractInterpretationMethod):
    """Scientific explanation about this method."""


class AbstractCPT(BaseModel):

    """Base CPT class, should define abstract."""
    
    #  variables
    depth = []
    coordinates = []
    local_reference_level = []
    depth_to_reference = []
    tip = []
    friction = []
    friction_nbr = []
    a = []
    name = []
    gamma = []
    rho = []
    total_stress = []
    effective_stress = []
    pwp = []
    qt = []
    Qtn = []
    Fr = []
    IC = []
    n = []
    vs = []
    G0 = []
    E0 = []
    permeability = []
    poisson = []
    damping = []

    water = []
    lithology = []
    litho_points = []

    lithology_merged = []
    depth_merged = []
    index_merged = []
    vertical_datum = []
    local_reference = []
    inclination_resultant = []
    cpt_standard = []
    quality_class = []
    cpt_type = []
    result_time = []

    # NEN results
    litho_NEN = []
    E_NEN = []
    cohesion_NEN = []
    fr_angle_NEN = []

    # fixed values
    g = 9.81
    Pa = 100.

    # private variables
    __water_measurement_types = None

    @classmethod
    @abstractmethod
    def read(cls, file: Path):
        pass

    # @property
    # @abstractmethod
    # def valid(self) -> bool:
    #     pass
    #
    # @abstractmethod
    # def interpret(self, method: AbstractInterpretationMethod) -> "Profile":
    #     pass
    #
    # @abstractmethod
    # def plot(self):
    #     pass
