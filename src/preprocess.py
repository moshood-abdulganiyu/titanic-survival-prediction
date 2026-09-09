"""
Cleaning and feature engineering for the Titanic dataset.

Split into two functions on purpose:

- clean_data(): fills missing values and drops unusable columns. This only
  makes sense on a full dataset (it needs group medians / a mode to compute),
  so it's a training-time step.
- engineer_features(): encodes categoricals into the final model inputs.
  This has no dependency on the rest of the dataset, so it runs the same way
  whether you're training on 891 rows or scoring a single form submission.
  That's the function the backend will reuse at serving time.

The one-hot encoding here uses drop_first=True, so Embarked_S and Embarked_Q
are the only columns produced; Southampton is the implicit baseline. If a
single row is the "S" baseline, engineer_features() on that row alone won't
even produce an Embarked_Q/Embarked_S column, so callers must reindex to
FEATURE_COLUMNS afterwards (model.py does this).
"""

import pandas as pd

# Exact column order the trained model expects. Must match
# data/artifacts/titanic_columns.json.
FEATURE_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_Q",
    "Embarked_S",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and drop columns with no usable signal.

    Age is imputed with the median within each (Pclass, Sex) group rather
    than a single flat median, because group medians here range from 21.5
    to 40 - a flat median would systematically distort both ends (e.g.
    pulling 1st-class women's ages down, 3rd-class men's ages up).
    """
    df = df.copy()

    df = df.drop(columns=["Cabin"], errors="ignore")

    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda group: group.fillna(group.median())
    )

    if "Embarked" in df.columns and df["Embarked"].isna().any():
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df = df.drop(columns=["PassengerId", "Name", "Ticket"], errors="ignore")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns into numeric model inputs.

    No missing-value handling here - by the time this runs (training after
    clean_data, or serving on a fully-filled-in form submission) there
    shouldn't be any NaNs left in Sex/Embarked/Age/etc.
    """
    df = df.copy()

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    return df
