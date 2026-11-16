import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CAN_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "feature"
RESULTS_DIR = REPO_ROOT / "results" / "dos"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt





NORMAL_PATH = FEATURE_DIR / "normal_run_data_features.csv"
DOS_PATH    = FEATURE_DIR / "DoS_dataset_features.csv"

normal = pd.read_csv(NORMAL_PATH)
dos    = pd.read_csv(DOS_PATH)

df = pd.concat([normal, dos], ignore_index=True)



df["label"] = (df["flag"] == "T").astype(int)
df = df.drop(columns=["flag"], errors="ignore")

df = df.drop(columns=["payload", "id"], errors="ignore")
df = df.dropna()

print("Dataset final:", df.shape)

  

# FINAL FEATURE SET (DOS CSVs) 


FEATURES = ["delta_time", "entropy"]

X = df[FEATURES]
y = df["label"]


# TRAIN/TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)



#  HYPERPARAMETER GRID


param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [8, 10, 12],
    "min_samples_leaf": [1, 5]
}

rf = RandomForestClassifier(
    class_weight="balanced",
    n_jobs=1,
    random_state=42
)

grid = GridSearchCV(
    rf,
    param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=2
)



# 6. GRID SEARCH

start = time.time()
grid.fit(X_train, y_train)
tuning_time = time.time() - start

best_model = grid.best_estimator_

print("\nBest Hyperparameters:")
print(grid.best_params_)
print(f"Tuning time: {tuning_time:.2f}s")


#  FINAL EVALUATION


y_pred = best_model.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("\nRandom Forest (Optimized1)")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")



# 8. CONFUSION MATRIX


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Normal (Predicted)", "DoS (Predicted)"],
    yticklabels=["Normal (Actual)", "DoS (Actual)"]
)

plt.title("Confusion Matrix — RF Optimized")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
