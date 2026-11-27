"""
ml/predictor.py

Provides reusable inference functions for predicting External Examination Marks,
Total Marks, and corresponding expected Grade for a student.

Phase 8.2: Academic Performance and Examination Intelligence System.
"""

from pathlib import Path
from typing import Dict, Union, Optional, Tuple, Any
import math
import importlib.util
import numpy as np
import pandas as pd

# Default model location
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "external_marks_model.joblib"

# Cache loaded model in memory for efficiency
_CACHED_MODEL = None
_COHORT_MODEL = None


def load_model(model_path: Optional[Union[str, Path]] = None):
    """
    Loads the trained scikit-learn Pipeline from disk.
    Caches the model instance in memory.
    """
    global _CACHED_MODEL
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if _CACHED_MODEL is not None and model_path is None:
        return _CACHED_MODEL

    if not target_path.exists():
        if model_path is not None:
            raise FileNotFoundError(f"Trained model file not found at: {target_path}")
        return None

    # Keep the serialized Random Forest available for local development, while
    # allowing the small Vercel runtime to use the built-in cohort estimator.
    try:
        import joblib
    except ImportError:
        if model_path is not None:
            raise FileNotFoundError("Optional Random Forest dependencies are not installed.")
        return None

    if not target_path.exists():
        if model_path is not None:
            raise FileNotFoundError(f"Trained model file not found at: {target_path}")
        return None

    model = joblib.load(target_path)
    if model_path is None:
        _CACHED_MODEL = model
    return model


def _load_cohort_model():
    """Fit a small least-squares estimator from bundled historical records."""
    global _COHORT_MODEL
    if _COHORT_MODEL is not None:
        return _COHORT_MODEL

    from utils.data_loader import load_data

    _, _, marks_df, attendance_df, _ = load_data()
    history = marks_df.merge(
        attendance_df[["Roll_No", "Semester", "Subject", "Attendance"]],
        on=["Roll_No", "Semester", "Subject"],
        how="left",
    )
    history["Attendance"] = pd.to_numeric(history["Attendance"], errors="coerce").fillna(0)
    history["Internal"] = pd.to_numeric(history["Internal"], errors="coerce").fillna(0)
    history["External"] = pd.to_numeric(history["External"], errors="coerce").fillna(0)
    history["Semester"] = pd.to_numeric(history["Semester"], errors="coerce").fillna(1)

    subjects = sorted(history["Subject"].dropna().astype(str).unique().tolist())
    subject_index = {subject: index for index, subject in enumerate(subjects[1:])}
    features = np.zeros((len(history), 4 + len(subject_index)), dtype=float)
    features[:, 0] = 1.0
    features[:, 1] = history["Internal"].to_numpy(dtype=float) / 30.0
    features[:, 2] = history["Attendance"].to_numpy(dtype=float) / 100.0
    features[:, 3] = history["Semester"].to_numpy(dtype=float) / 6.0
    for row_index, subject in enumerate(history["Subject"].astype(str)):
        column = subject_index.get(subject)
        if column is not None:
            features[row_index, 4 + column] = 1.0

    targets = history["External"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(features, targets, rcond=None)
    _COHORT_MODEL = (subjects, subject_index, coefficients)
    return _COHORT_MODEL


def _predict_from_cohort(internal_marks: float, attendance: float, semester: int, subject: str) -> float:
    """Estimate external marks from historical cohort patterns without sklearn."""
    subjects, subject_index, coefficients = _load_cohort_model()
    features = np.zeros(len(coefficients), dtype=float)
    features[0] = 1.0
    features[1] = float(internal_marks) / 30.0
    features[2] = float(attendance) / 100.0
    features[3] = int(semester) / 6.0
    column = subject_index.get(str(subject).strip())
    if column is not None:
        features[4 + column] = 1.0
    return float(np.dot(features, coefficients))


def validate_prediction_inputs(
    internal_marks: Any,
    attendance: Any,
    semester: Any,
    subject: Any,
) -> Tuple[bool, str]:
    """
    Strictly validates prediction inputs before passing to the ML model.

    Rules:
      - Internal Marks: numeric, 0 <= internal <= 30
      - Attendance: numeric, 0 <= attendance <= 100
      - Semester: integer/numeric, 1 <= semester <= 6 (valid dataset range)
      - Subject: non-empty string

    Returns:
      Tuple[bool, str]: (is_valid, error_message)
    """
    # 1. Subject validation
    if subject is None:
        return False, "Subject is required."
    subject_str = str(subject).strip()
    if not subject_str or subject_str.lower() in ("nan", "none", "null"):
        return False, "Subject name must not be empty."

    # 2. Semester validation
    if semester is None:
        return False, "Semester is required."
    try:
        sem_int = int(semester)
        if sem_int < 1 or sem_int > 6:
            return False, f"Semester {sem_int} is outside valid range (1–6)."
    except (ValueError, TypeError):
        return False, f"Invalid semester value: '{semester}'."

    # 3. Internal marks validation (0 to 30)
    if internal_marks is None:
        return False, "Internal marks are required."
    try:
        int_val = float(internal_marks)
        if math.isnan(int_val):
            return False, "Internal marks cannot be NaN."
        if int_val < 0.0 or int_val > 30.0:
            return False, f"Internal marks {int_val} outside expected range (0–30)."
    except (ValueError, TypeError):
        return False, f"Invalid internal marks: '{internal_marks}'."

    # 4. Attendance validation (0 to 100)
    if attendance is None:
        return False, "Attendance is required."
    try:
        att_val = float(attendance)
        if math.isnan(att_val):
            return False, "Attendance cannot be NaN."
        if att_val < 0.0 or att_val > 100.0:
            return False, f"Attendance {att_val}% outside expected range (0–100%)."
    except (ValueError, TypeError):
        return False, f"Invalid attendance: '{attendance}'."

    return True, ""


def predict_external_marks(
    internal_marks: float,
    attendance: float,
    semester: int,
    subject: str,
    model_path: Optional[Union[str, Path]] = None,
) -> float:
    """
    Predicts expected External Examination Marks using the trained ML model.
    Enforces strict input validation and clamping (0.0 to 70.0).

    Parameters:
        internal_marks (float): Student's internal score (0 - 30).
        attendance (float): Student's subject attendance percentage (0 - 100).
        semester (int): Academic semester (1 - 6).
        subject (str): Subject name.
        model_path (str or Path, optional): Custom path to joblib model file.

    Returns:
        float: Expected external marks clamped to [0.0, 70.0], rounded to 1 decimal place.
    """
    is_valid, err = validate_prediction_inputs(internal_marks, attendance, semester, subject)
    if not is_valid:
        raise ValueError(f"Invalid prediction input: {err}")

    model = load_model(model_path)

    if model is None:
        prediction = _predict_from_cohort(internal_marks, attendance, semester, subject)
        return round(min(70.0, max(0.0, prediction)), 1)

    # Construct feature DataFrame with EXACT column names expected by pipeline
    # Strictly avoids data leakage (no Total, Grade, SGPI, or post-exam values)
    input_df = pd.DataFrame(
        [
            {
                "Internal": float(internal_marks),
                "Attendance": float(attendance),
                "Semester": int(semester),
                "Subject": str(subject).strip(),
            }
        ]
    )

    prediction = model.predict(input_df)[0]

    # Safely clamp to realistic academic bounds for external exams (0 to 70 marks)
    prediction = min(70.0, max(0.0, float(prediction)))
    return round(prediction, 1)


def classify_predicted_performance(predicted_total: float) -> str:
    """
    Classifies student performance based on predicted total marks (0 - 100):
      >= 75: 'Excellent'
      >= 60: 'Good'
      >= 50: 'Average'
      <  50: 'Needs Improvement'

    Parameters:
        predicted_total (float): Internal + Predicted External marks.

    Returns:
        str: Performance category label.
    """
    if predicted_total >= 75.0:
        return "Excellent"
    elif predicted_total >= 60.0:
        return "Good"
    elif predicted_total >= 50.0:
        return "Average"
    else:
        return "Needs Improvement"


def predict_total_marks(
    internal_marks: float,
    attendance: float,
    semester: int,
    subject: str,
    model_path: Optional[Union[str, Path]] = None,
) -> float:
    """
    Calculates Predicted Total Marks = Actual Internal Marks + Predicted External Marks.
    Safely clamps to [0.0, 100.0].

    Parameters:
        internal_marks (float): Actual Internal marks.
        attendance (float): Attendance percentage.
        semester (int): Academic semester.
        subject (str): Subject name.
        model_path (str or Path, optional): Custom path to joblib model file.

    Returns:
        float: Predicted total marks rounded to 1 decimal place.
    """
    predicted_external = predict_external_marks(
        internal_marks=internal_marks,
        attendance=attendance,
        semester=semester,
        subject=subject,
        model_path=model_path,
    )
    predicted_total = min(100.0, max(0.0, float(internal_marks) + predicted_external))
    return round(predicted_total, 1)


def calculate_predicted_grade(total_marks: float) -> str:
    """
    Maps total score to the project's existing grade distribution:
      >= 90 : 'O'
      >= 80 : 'A+'
      >= 70 : 'A'
      >= 60 : 'B+'
      >= 50 : 'B'
      >= 40 : 'C'
      <  40 : 'F'

    Parameters:
        total_marks (float): Total marks out of 100.

    Returns:
        str: Letter grade category.
    """
    if total_marks >= 90:
        return "O"
    elif total_marks >= 80:
        return "A+"
    elif total_marks >= 70:
        return "A"
    elif total_marks >= 60:
        return "B+"
    elif total_marks >= 50:
        return "B"
    elif total_marks >= 40:
        return "C"
    else:
        return "F"


def predict_subject_performance(
    internal_marks: Any,
    attendance: Any,
    semester: Any,
    subject: Any,
    actual_external: Optional[Any] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Production-grade inference function for a single subject.
    Validates inputs, computes predicted external and total marks,
    evaluates performance category and grade, distinguishes actual external marks,
    and returns a clean structured record with error handling.

    If inputs are invalid or insufficient, returns a clean unavailable state
    without crashing the caller or failing sibling subjects.
    """
    clean_subj = str(subject).strip() if subject is not None else "Unknown"

    # Validate inputs
    is_valid, err_msg = validate_prediction_inputs(internal_marks, attendance, semester, subject)
    if not is_valid:
        return {
            "subject": clean_subj,
            "internal": internal_marks,
            "attendance": attendance,
            "semester": semester,
            "actual_external": actual_external,
            "predicted_external": None,
            "predicted_total": None,
            "performance": "Unavailable",
            "predicted_grade": "N/A",
            "available": False,
            "error_message": "Prediction unavailable — insufficient academic data.",
        }

    try:
        int_val = round(float(internal_marks), 1)
        att_val = round(float(attendance), 1)
        sem_val = int(semester)

        pred_ext = predict_external_marks(
            internal_marks=int_val,
            attendance=att_val,
            semester=sem_val,
            subject=clean_subj,
            model_path=model_path,
        )
        pred_tot = min(100.0, max(0.0, round(int_val + pred_ext, 1)))
        perf = classify_predicted_performance(pred_tot)
        grade = calculate_predicted_grade(pred_tot)

        act_ext = None
        if actual_external is not None:
            try:
                act_num = float(actual_external)
                if not math.isnan(act_num):
                    act_ext = round(act_num, 1)
            except (ValueError, TypeError):
                act_ext = None

        return {
            "subject": clean_subj,
            "internal": int_val,
            "attendance": att_val,
            "semester": sem_val,
            "actual_external": act_ext,
            "predicted_external": pred_ext,
            "predicted_total": pred_tot,
            "performance": perf,
            "predicted_grade": grade,
            "available": True,
            "error_message": None,
        }
    except FileNotFoundError:
        return {
            "subject": clean_subj,
            "internal": internal_marks,
            "attendance": attendance,
            "semester": semester,
            "actual_external": actual_external,
            "predicted_external": None,
            "predicted_total": None,
            "performance": "Unavailable",
            "predicted_grade": "N/A",
            "available": False,
            "error_message": "AI prediction is temporarily unavailable.",
        }
    except Exception as e:
        return {
            "subject": clean_subj,
            "internal": internal_marks,
            "attendance": attendance,
            "semester": semester,
            "actual_external": actual_external,
            "predicted_external": None,
            "predicted_total": None,
            "performance": "Unavailable",
            "predicted_grade": "N/A",
            "available": False,
            "error_message": "Prediction unavailable — insufficient academic data.",
        }


def get_full_prediction(
    internal_marks: float,
    attendance: float,
    semester: int,
    subject: str,
) -> Dict[str, Union[float, str, int]]:
    """
    Convenience function returning complete prediction breakdown.
    Preserved for backward compatibility.
    """
    pred_ext = predict_external_marks(internal_marks, attendance, semester, subject)
    pred_tot = min(100.0, max(0.0, round(float(internal_marks) + pred_ext, 1)))
    pred_grade = calculate_predicted_grade(pred_tot)

    return {
        "semester": semester,
        "subject": subject,
        "internal_marks": internal_marks,
        "attendance": attendance,
        "predicted_external_marks": pred_ext,
        "predicted_total_marks": pred_tot,
        "predicted_grade": pred_grade,
    }


def get_model_info() -> Dict[str, Any]:
    """
    Returns transparent metadata, architecture details, and verified evaluation metrics
    for the trained ML model pipeline.
    """
    if importlib.util.find_spec("joblib") is None or not DEFAULT_MODEL_PATH.exists():
        from utils.data_loader import load_data
        _, _, marks_df, _, _ = load_data()
        return {
            "model_name": "Historical Cohort Estimate",
            "pipeline_architecture": "Lightweight least-squares regression over bundled student records",
            "features": ["Internal", "Attendance", "Semester", "Subject"],
            "target": "External",
            "total_records": len(marks_df),
            "train_records": None,
            "test_records": None,
            "split_ratio": None,
            "baseline_mae": None,
            "model_mae": None,
            "rmse": None,
            "r2_score": None,
            "improvement_pct": None,
            "description": "Estimates use historical marks and attendance from the bundled sample dataset.",
            "r2_explanation": "This lightweight estimate is provided for prototype demonstration and has no published evaluation score.",
        }

    return {
        "model_name": "Random Forest Regressor",
        "pipeline_architecture": "ColumnTransformer + OneHotEncoder(handle_unknown='ignore') + RandomForestRegressor",
        "features": ["Internal", "Attendance", "Semester", "Subject"],
        "target": "External",
        "total_records": 1440,
        "train_records": 1152,
        "test_records": 288,
        "split_ratio": "80% Train / 20% Test",
        "baseline_mae": 8.84,
        "model_mae": 4.72,
        "rmse": 5.77,
        "r2_score": 0.71,
        "improvement_pct": 46.58,
        "description": "AI predictions are generated using internal marks, attendance, semester, and subject information.",
        "r2_explanation": "The model achieved an R² score of approximately 0.71 on the held-out test dataset (representing variance explained, rather than simple percentage accuracy).",
    }
