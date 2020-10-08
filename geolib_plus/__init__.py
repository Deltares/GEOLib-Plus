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

    @classmethod
    @abstractmethod
    def read(cls, file: Path):
        pass

    @property
    @abstractmethod
    def valid(self) -> bool:
        pass

    @abstractmethod
    def interpret(self, method: AbstractInterpretationMethod) -> "Profile":
        pass

    @abstractmethod
    def plot(self):
        pass


class GEF_CPT(AbstractCPT):
    pass


class XML_CPT(AbstractCPT):
    pass
