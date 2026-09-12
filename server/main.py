"""FastAPI backend for the Titanic survival prediction model.

Run from the project root: uvicorn server.main:app --reload
Docs (auto-generated from the schemas below): http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.schemas import PassengerInput, PredictionResponse
from src.model import predict_one

app = FastAPI(title="Titanic Survival Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerInput) -> PredictionResponse:
    result = predict_one(passenger.model_dump())
    return PredictionResponse(**result)


# Mounted last, on purpose. Starlette checks routes in the order they're
# registered, so /api/health and /predict get matched first. If this mount
# came before them, it would swallow every request, including /predict.
app.mount("/", StaticFiles(directory="client", html=True), name="client")