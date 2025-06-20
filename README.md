# Titanic Survival Predictor (Kaggle)

This project implements a full machine-learning pipeline to predict passenger survival on the Titanic using the Kaggle Titanic dataset.  
It covers data preprocessing, model training, evaluation, and FastAPI-based deployment for real-time inference.

---

## Project Structure

| Path / File                       | Description                                                      |
|------------------------------------|------------------------------------------------------------------|
| `src/model.py`                    | Trains the model (preprocessing, feature engineering, evaluation, artefact saving). |
| `src/app.py`                      | FastAPI service exposing prediction endpoints.                   |
| `src/data_preprocess.py`          | Reusable preprocessing pipeline (cleaning, encoding, feature engineering). |
| `data/`                           | Holds `train.csv`, `test.csv`, and the generated `submission.csv`.|
| `models/`                         | Stores the trained model and preprocessing artefacts (`logreg_model.joblib`). |
| `README.md`                       | Project documentation (this file).                               |

---

## Features

- Robust preprocessing & feature engineering (titles, family size, deck, bins, etc.)
- Logistic Regression (default), easily extendable to other models (RandomForest, XGBoost, SVM, etc.)
- Final retraining on the full dataset for leaderboard submission
- FastAPI endpoints for real-time survival prediction

---

## Installation & Quick Start

**1 — Clone & install dependencies**
```bash
git clone https://github.com/rohith-pulluri_sap/titanic-survival-kaggle.git
cd titanic-survival-kaggle
pip install -r requirements.txt
```

**2 — Train the model & generate submission.csv**
```bash
python src/model.py
```

**3 — Launch the FastAPI server**
```bash
python src/app.py
```
The API is then available at [http://localhost:8000](http://localhost:8000)

---

## API Endpoints

| Method | Path      | Purpose                                  |
|--------|-----------|------------------------------------------|
| GET    | `/health` | Health check                             |
| POST   | `/predict`| Predict survival from passenger features |

**Example payload for `/predict`:**
```json
{
  "Pclass": 3,
  "Fare": 7.25,
  "Sex": "male",
  "Name": "Braund, Mr. Owen Harris",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Ticket": "A/5 21171",
  "Cabin": "",
  "Embarked": "S"
}
```

---

## Outputs

- `submission.csv` – ready-to-upload Kaggle predictions
- `models/logreg_model.joblib` – trained model + preprocessing artefacts

---

## Acknowledgments

- [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic)
- Libraries: pandas, numpy, scikit-learn, joblib, fastapi, uvicorn

---