from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Optional, Union
import pandas as pd
import joblib

from data_preprocess import preprocess_titanic_survival

# Load model bundle (adjust path if needed)
bundle = joblib.load("../models/logreg_model.joblib")
model = bundle["model"]
stats = bundle["stats"]
columns = bundle["columns"]

# Define request/response schemas
class Passenger(BaseModel):
    Pclass: int
    Fare: float
    Sex: Union[str, int] = 'male'
    Name: Optional[str] = 'Mr. Unknown'
    Age: Optional[float] = None
    SibSp: Optional[int] = 0
    Parch: Optional[int] = 0
    Ticket: Optional[str] = ''
    Cabin: Optional[str] = ''
    Embarked: Optional[str] = 'S'

    @validator('Sex', pre=True)
    def parse_sex(cls, v):
        if isinstance(v, int):
            return 'male' if v == 0 else 'female'
        return v

    @validator('Embarked', pre=True)
    def parse_embarked(cls, v):
        if isinstance(v, int):
            # map 0->S, 1->C, 2->Q
            return {0:'S',1:'C',2:'Q'}.get(v, 'S')
        return v

class PredictionResponse(BaseModel):
    survived: int

# Create FastAPI app
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: Passenger):
    # Convert input to DataFrame
    df = pd.DataFrame([passenger.dict()])
    try:
        # Preprocess using saved stats
        df_proc, _ = preprocess_titanic_survival(df, stats)
        # One-hot & align features to training columns
        X = pd.get_dummies(df_proc, dtype=float)
        X = X.reindex(columns=columns, fill_value=0)
        # Predict
        prediction = model.predict(X)[0]
        return PredictionResponse(survived=int(prediction))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
