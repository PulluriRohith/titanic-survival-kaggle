from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the model
model = joblib.load("random_forest_model.pkl")  #you can change the model name from models folder

# Define the Passenger schema
class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Fare: float

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Titanic Prediction API"}

@app.post("/predict")
async def predict(passenger: Passenger):
    # Corrected variable name
    input_data = pd.DataFrame([passenger.dict()])

    # Make prediction
    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)