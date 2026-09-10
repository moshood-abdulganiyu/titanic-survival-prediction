import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import clean, engineer_features

df = pd.read_csv("data/TitanicDataset.csv")
df_clean = clean(df)
df_features = engineer_features(df_clean)

X = df_features.drop(columns=["Survived"])
y = df_features["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = joblib.load("data/artifacts/titanic_model.joblib")

for idx in X_test.index[:5]:
    raw_row = df.loc[idx]
    payload = {
        "Pclass": int(raw_row["Pclass"]),
        "Sex": raw_row["Sex"],
        "Age": float(df_clean.loc[idx, "Age"]),
        "SibSp": int(raw_row["SibSp"]),
        "Parch": int(raw_row["Parch"]),
        "Fare": float(raw_row["Fare"]),
        "Embarked": df_clean.loc[idx, "Embarked"],
    }
    pred = int(model.predict(X_test.loc[[idx]])[0])
    prob = round(float(model.predict_proba(X_test.loc[[idx]])[0][1]), 3)
    print(f"PassengerId {idx}: payload={payload}")
    print(f"  expected: survived={pred}, probability={prob}")