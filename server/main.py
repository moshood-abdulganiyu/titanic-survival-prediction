"""FastAPI backend for the Titanic survival prediction model.

Run from the project root: uvicorn server.main:app --reload
Docs (auto-generated from the schemas below): http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.schemas import PassengerInput, PredictionResponse
from src.model import predict_one

app = FastAPI(title="Titanic Survival Prediction API")

# The frontend is served separately (different origin/port), so without
# this a browser blocks the fetch with a CORS error before it even
# reaches this server. "*" matches what the old Flask version did with
# Access-Control-Allow-Origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerInput) -> PredictionResponse:
    result = predict_one(passenger.model_dump())
    return PredictionResponse(**result)