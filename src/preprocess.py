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
    """Drop unused columns, binary-encode Sex, one-hot encode Embarked."""
    df = df.drop(columns=["PassengerId", "Name", "Ticket"], errors="ignore").copy()

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    return df