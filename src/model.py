"""
Loading the trained model and predicting on a single passenger.

The saved artifact is a full sklearn Pipeline (scaler + classifier), so
there's no manual scaling step here - .predict() does it internally. This
is the whole point of exporting a Pipeline instead of a bare model plus a
separate scaler file: one artifact, one call, no place for training/serving
scaling to drift apart.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from .preprocess import FEATURE_COLUMNS, engineer_features

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "titanic_model.joblib"
COLUMNS_PATH = ARTIFACTS_DIR / "titanic_columns.json"


def load_artifacts():
    """Load the fitted pipeline and the expected feature column order."""
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH) as f:
        columns = json.load(f)["columns"]
    return model, columns


def predict_one(raw: dict, model=None, columns=None) -> int:
    """Predict survival (0/1) for a single passenger.

    raw must contain: Pclass, Sex ("male"/"female"), Age, SibSp, Parch,
    Fare, Embarked ("C"/"Q"/"S"). This is exactly what a filled-in form
    submission gives you, so no imputation is needed here - that's a
    training-time concern (see preprocess.clean_data), not a serving-time
    one.
    """
    if model is None or columns is None:
        model, columns = load_artifacts()

    row = pd.DataFrame([raw])
    row = engineer_features(row)

    # A single row only produces one-hot columns for the category it
    # actually has (e.g. Embarked="S" won't create an Embarked_Q column at
    # all). Reindexing to the training column order fills the rest with 0,
    # which is exactly what "not that category" should look like.
    row = row.reindex(columns=columns, fill_value=0)

    prediction = model.predict(row)[0]
    return int(prediction)
