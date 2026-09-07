"""Experiment-fold generation."""

from .folds import FoldDefinition, SCENARIOS, build_fold_definitions, generate_experiment_folds

__all__ = ["FoldDefinition", "SCENARIOS", "build_fold_definitions", "generate_experiment_folds"]
