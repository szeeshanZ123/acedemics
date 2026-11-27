"""
Machine Learning Module for Academic Performance and Examination Intelligence System.
Phase 10.1: External Examination Marks Prediction.
"""

from .predictor import (
    predict_external_marks,
    predict_total_marks,
    calculate_predicted_grade,
    classify_predicted_performance,
)


def build_regression_pipeline(*args, **kwargs):
    """Load the optional scikit-learn training helper on demand."""
    from .model import build_regression_pipeline as build_pipeline
    return build_pipeline(*args, **kwargs)

__all__ = [
    "predict_external_marks",
    "predict_total_marks",
    "calculate_predicted_grade",
    "classify_predicted_performance",
    "build_regression_pipeline",
]
