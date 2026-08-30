"""Qualified execution-mode selection and fail-closed fallback."""

from .fallback_decision import FailureClass, FallbackDecision, decide_fallback
from .mode_qualification import (
    ModeQualification,
    QualificationKey,
    QualificationRegistry,
)
from .mode_selection import ModeSelection, select_execution_mode

__all__ = [
    "FailureClass",
    "FallbackDecision",
    "ModeQualification",
    "ModeSelection",
    "QualificationKey",
    "QualificationRegistry",
    "decide_fallback",
    "select_execution_mode",
]
