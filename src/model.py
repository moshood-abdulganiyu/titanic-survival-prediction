"""Load the trained pipeline and predict from a single raw passenger record.

Serving-time input is assumed complete (Step 4's FastAPI layer enforces
required fields via Pydantic), so this only needs to encode and order
columns to match training, not impute missing values. Imputation logic
in preprocess.clean() is a training-time concern only.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from .preprocess import engineer_features

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"

_model = joblib.load(_ARTIFACTS_DIR / "titanic_model.joblib")

with open(_ARTIFACTS_DIR / "titanic_columns.json") as f:
    _COLUMNS = json.load(f)["columns"]


def predict_one(passenger: dict) -> dict:
    """
    passenger keys: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
      - Sex: "male" or "female"
      - Embarked: "C", "Q", or "S"

    Returns {"survived": 0 or 1, "probability": float}
    """
    df = pd.DataFrame([passenger])
    df = engineer_features(df)

    # Reindex to the exact training column order. A single row's Embarked
    # value only produces one dummy column at most (e.g. Embarked='C'
    # produces neither Embarked_Q nor Embarked_S), so any column the
    # pipeline expects but this row didn't generate gets filled with 0 —
    # equivalent to "not that category" for a one-hot column.
    df = df.reindex(columns=_COLUMNS, fill_value=0)

    prediction = int(_model.predict(df)[0])
    probability = float(_model.predict_proba(df)[0][1])

    return {"survived": prediction, "probability": round(probability, 3)}