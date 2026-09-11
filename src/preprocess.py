"""Cleaning and feature engineering for the Titanic survival model.

Mirrors notebook cells 2d and 2e exactly, so the model that gets trained
and the code that serves it share the same logic. Both functions operate
on a copy of the input and return a new dataframe.
"""
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Cabin, impute Age by Pclass/Sex group median, impute Embarked mode."""
    df = df.drop(columns=["Cabin"], errors="ignore").copy()

    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    if df["Embarked"].isnull().any():
        embarked_mode = df["Embarked"].mode()[0]
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns, binary-encode Sex, one-hot encode Embarked.

    Embarked is encoded as two explicit boolean columns rather than
    pd.get_dummies(drop_first=True). get_dummies decides which category to
    drop based on what's present in that specific call - a single inference
    row only ever has one Embarked value, so it always got dropped, silently
    zeroing out Embarked_Q/Embarked_S regardless of the real port submitted.
    Fixed and verified in Step 5: every Q and S prediction was silently
    wrong before this, C happened to look right by accident.
    """
    df = df.drop(columns=["PassengerId", "Name", "Ticket"], errors="ignore").copy()

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df["Embarked_Q"] = df["Embarked"] == "Q"
    df["Embarked_S"] = df["Embarked"] == "S"
    df = df.drop(columns=["Embarked"])

    return df