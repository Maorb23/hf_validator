"""Optional inference optimization contracts and read-only analysis."""

from .analysis import analyze, pareto_front
from .contracts import NewStudyIntent, OptimizationResult, ReuseIntent, objectives_for

__all__ = [
    "NewStudyIntent",
    "ReuseIntent",
    "OptimizationResult",
    "analyze",
    "pareto_front",
    "objectives_for",
]
