from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

pipeline = joblib.load("best_text_pipeline.joblib")

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    pred = pipeline.predict([payload.text])[0]
    return PredictionOut(label=str(pred))