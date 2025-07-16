from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load("../models/titanic_sklearn_pipeline_logreg.joblib")

# Define request schema (adjust fields as needed)
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Cabin: str = ""
    Embarked: str
    Name: str

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(passenger: Passenger):
    # Convert input to DataFrame
    df = pd.DataFrame([passenger.dict()])
    # Predict using pipeline
    try:
        pred = pipeline.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"survived": int(pred)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)