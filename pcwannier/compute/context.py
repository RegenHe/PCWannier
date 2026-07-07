from __future__ import annotations

from dataclasses import dataclass

from .gradient import Gradient
from .initializer import StateInitializer
from .matrix import MSet
from .state import StateCollection
from ..config import IncarConfig


@dataclass
class CalculationContext:
    config: IncarConfig
    state: StateCollection
    mset: MSet
    initializer: StateInitializer
    gradient: Gradient
