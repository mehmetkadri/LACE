# src/drift_detector/__init__.py
__version__ = "0.1.0"

from .detector import DriftDetector, DetectorConfig
__all__ = ["DriftDetector", "DetectorConfig"]