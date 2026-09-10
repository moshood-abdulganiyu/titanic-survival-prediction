"""Request/response contracts for the /predict endpoint.

Using Literal types for Sex/Embarked and ge/le bounds on the numeric
fields means bad input (a typo like "Male", a negative Fare) gets
rejected at the API boundary with a 422 and a clear error message,
instead of silently reaching engineer_features() and producing a
malformed row (e.g. an unrecognized Embarked value that get_dummies()
can't map to any of the trained columns).
"""
from typing import Literal

from pydantic import BaseModel, Field


class PassengerInput(BaseModel):
    Pclass: Literal[1, 2, 3] = Field(..., description="Passenger class: 1st, 2nd, or 3rd")
    Sex: Literal["male", "female"]
    Age: float = Field(..., ge=0, le=100)
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0)
    Embarked: Literal["C", "Q", "S"] = Field(..., description="Port of embarkation")


class PredictionResponse(BaseModel):
    survived: int
    probability: float