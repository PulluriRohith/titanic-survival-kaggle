import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from data_preprocess import preprocess_titanic_survival

IMPORTANT = ['Pclass', 'Sex', 'Fare']

# 0. Load raw CSVs
df_train_raw = pd.read_csv("../data/train.csv")
df_test_raw  = pd.read_csv("../data/test.csv")

# 1. Pre-process  (fit on train, reuse on test)
df_train_proc, stats = preprocess_titanic_survival(df_train_raw)
df_test_proc,  _     = preprocess_titanic_survival(df_test_raw, stats)

# 2. Set up target (use processed df to keep indices aligned)
y = df_train_proc['Survived']

# 3. Drop the label from your features
df_train_proc = df_train_proc.drop(columns=['Survived'])
df_test_proc  = df_test_proc.drop(columns=['Survived'], errors='ignore')

# 4. One-hot encode & align train/test
X      = pd.get_dummies(df_train_proc, dtype=float)
X_test = pd.get_dummies(df_test_proc,  dtype=float)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# ------------------------------------------------------------
# 5.  Fill any remaining NaNs with 0
# ------------------------------------------------------------
X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# ------------------------------------------------------------
# 6.  Train / validation split
# ------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 7.  Train the model
# ------------------------------------------------------------
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_tr, y_tr)

# ------------------------------------------------------------
# 8.  Hold‐out evaluation
# ------------------------------------------------------------
val_pred = logreg.predict(X_val)
accuracy = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_val, val_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))

# ------------------------------------------------------------
# 9.  Retrain on all data & generate submission
# ------------------------------------------------------------
logreg.fit(X, y)
test_pred = logreg.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": df_test_raw["PassengerId"],
    "Survived":    test_pred
})
submission.to_csv("../data/submission.csv", index=False)
print("../data/submission.csv written")  # ready for Kaggle

# ------------------------------------------------------------
# 10. Persist model (+ preprocessing stats + feature columns + medians)
# ------------------------------------------------------------
# To compute medians, we need numeric columns:
tmp = df_train_raw[IMPORTANT].copy()
tmp['Sex'] = tmp['Sex'].map({'male':0, 'female':1})
medians = {f: tmp[f].median() for f in IMPORTANT}

joblib.dump({
    "model":   logreg,
    "stats":   stats,
    "columns": X.columns,
    "medians": medians
}, "../models/logreg_model.joblib")

print("Model bundle saved to ../models/logreg_model.joblib")