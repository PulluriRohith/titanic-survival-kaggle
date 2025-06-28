import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from src.pipeline import titanic_pipeline

if __name__ == "__main__":
    # 1) Load raw data
    df = pd.read_csv("../data/train.csv")
    y  = df["Survived"]

    # 2) Train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3) Fit pipeline
    titanic_pipeline.fit(X_tr, y_tr)

    # 4) Evaluate on hold-out
    val_acc = titanic_pipeline.score(X_val, y_val)
    print(f"Hold-out accuracy: {val_acc:.4f}")

    # 5) 5-fold cross-validation on full train
    cv_scores = cross_val_score(
        titanic_pipeline, df, y,
        cv=5, scoring="accuracy", n_jobs=-1
    )
    print(
        f"5-fold CV accuracy: {np.mean(cv_scores):.4f} "
        f"± {np.std(cv_scores):.4f}"
    )

    # 6) (Optional) Persist the trained pipeline
    joblib.dump(titanic_pipeline, "../models/titanic_sklearn_pipeline_logreg.joblib")
    print("Standalone pipeline saved to ../models/titanic_sklearn_pipeline_logreg.joblib")
