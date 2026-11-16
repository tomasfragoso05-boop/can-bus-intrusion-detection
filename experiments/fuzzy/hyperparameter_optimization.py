import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CAN_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "feature"
RESULTS_DIR = REPO_ROOT / "results" / "fuzzy"
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


# 1. LOAD DATA


NORMAL_PATH = FEATURE_DIR / "normal_run_data_features.csv"
FUZZY_PATH  = FEATURE_DIR / "Fuzzy_dataset_features.csv"

normal = pd.read_csv(NORMAL_PATH)
fuzzy  = pd.read_csv(FUZZY_PATH)

df = pd.concat([normal, fuzzy], ignore_index=True)


# 2. PREPROCESSING


df["label"] = (df["flag"] == "T").astype(int)
df = df.drop(columns=["flag"], errors="ignore")
df = df.drop(columns=["payload", "id"], errors="ignore")
df = df.drop_duplicates()
df = df.dropna()

print("Dataset final:", df.shape)


# 3. FINAL FEATURE SET
# entropy + byte_std + byte_range + delta_time + byte_mean (set E)
# Selected from the model comparison experiment as the strongest clean
# feature set for Fuzzy detection without structural shortcuts.

FEATURES = ["entropy", "byte_std", "byte_range", "delta_time", "byte_mean"]

X = df[FEATURES]
y = df["label"]


# 4. TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# 5. UNOPTIMIZED BASELINE
# Evaluate the default config first so the comparison is explicit.

baseline = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

start = time.time()
baseline.fit(X_train, y_train)
baseline_time = time.time() - start

y_pred_base = baseline.predict(X_test)

print("\n--- Unoptimized RF (n_estimators=50, max_depth=10) ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred_base):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_base):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_base):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred_base):.4f}")
print(f"Train time: {baseline_time:.2f}s")


# 6. HYPERPARAMETER GRID


param_grid = {
    "n_estimators":     [50, 100],
    "max_depth":        [8, 10, 12],
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


# 7. GRID SEARCH


start = time.time()
grid.fit(X_train, y_train)
tuning_time = time.time() - start

best_model = grid.best_estimator_

print("\nBest Hyperparameters:")
print(grid.best_params_)
print(f"Tuning time: {tuning_time:.2f}s")


# 8. OPTIMIZED EVALUATION


y_pred_opt = best_model.predict(X_test)

acc  = accuracy_score(y_test, y_pred_opt)
prec = precision_score(y_test, y_pred_opt)
rec  = recall_score(y_test, y_pred_opt)
f1   = f1_score(y_test, y_pred_opt)

print("\n--- Optimized RF ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")


# 9. SIDE-BY-SIDE COMPARISON


print("\n========= COMPARISON =========")
print(f"{'Metric':<12} {'Unoptimized':>14} {'Optimized':>12}")
print("-" * 40)
print(f"{'Accuracy':<12} {accuracy_score(y_test, y_pred_base):>14.4f} {acc:>12.4f}")
print(f"{'Precision':<12} {precision_score(y_test, y_pred_base):>14.4f} {prec:>12.4f}")
print(f"{'Recall':<12} {recall_score(y_test, y_pred_base):>14.4f} {rec:>12.4f}")
print(f"{'F1 Score':<12} {f1_score(y_test, y_pred_base):>14.4f} {f1:>12.4f}")


# 10. CONFUSION MATRIX — OPTIMIZED MODEL


cm = confusion_matrix(y_test, y_pred_opt)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal (Predicted)", "Fuzzy (Predicted)"],
    yticklabels=["Normal (Actual)", "Fuzzy (Actual)"]
)

plt.title("Confusion Matrix — RF Optimized (Fuzzy)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
