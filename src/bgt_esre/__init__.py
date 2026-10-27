"""Dataset-agnostic BGT-ESRE reference implementation."""

from .config import Config
from .model import BGTESRE

__all__ = ["BGTESRE", "Config"]
__version__ = "0.1.0"
