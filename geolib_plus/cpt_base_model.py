from abc import abstractmethod
from pathlib import Path
from pydantic import BaseModel

from .plot_cpt import plot_cpt_norm
from .plot_settings import PlotSettings


class AbstractInterpretationMethod:
    """Base Interpretation method for analyzing CPTs."""


class RobertsonMethod(AbstractInterpretationMethod):
    """Scientific explanation about this method."""


class CptReader:
    @abstractmethod
    def read_file(self, filepath: Path) -> dict:
        raise NotImplementedError(
            "The method should be implemented in concrete classes."
        )


class AbstractCPT(BaseModel):
    """Base CPT class, should define abstract."""

    #  variables
    penetration_length = []
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

    pore_pressure_u1 = []
    pore_pressure_u2 = []
    pore_pressure_u3 = []

    water = []
    lithology = []
    litho_points = []

    lithology_merged = []
    depth_merged = []
    index_merged = []
    vertical_datum = []
    local_reference = []
    inclination_x = []
    inclination_y = []
    inclination_ns = []
    inclination_ew = []
    inclination_resultant = []
    time = []
    net_tip = []
    pore_ratio = []
    tip_nbr = []
    unit_weight_measured = []
    pwp_ini = []
    total_pressure_measured = []
    effective_pressure_measured = []
    electric_cond = []
    magnetic_strength_x = []
    magnetic_strength_y = []
    magnetic_strength_z = []
    magnetic_strength_tot = []
    magnetic_inclination = []
    magnetic_declination = []

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
    Pa = 100.0

    # private variables
    __water_measurement_types = None

    @classmethod
    def read(cls, filepath: Path):
        if not filepath:
            raise ValueError(filepath)

        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(filepath)

        cpt_data = cls.cpt_reader().read_file(filepath)
        cls(**cpt_data)

    @classmethod
    @abstractmethod
    def get_cpt_reader(cls) -> CptReader:
        raise NotImplementedError("Should be implemented in concrete class.")

    # @property
    # @abstractmethod
    # def valid(self) -> bool:
    #     pass
    #
    # @abstractmethod
    # def interpret(self, method: AbstractInterpretationMethod) -> "Profile":
    #     pass
    #

    def plot(self, directory: Path):
        # plot cpt data
        try:
            plot_setting = PlotSettings()
            plot_setting.assign_default_settings()
            plot_cpt_norm(self, directory, plot_setting.general_settings)

        except (ValueError, IndexError):
            print("Cpt data and/or settings are not valid")
        except PermissionError as error:
            print(error)
