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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt



# 1. LOAD DATA

NORMAL_PATH = FEATURE_DIR / "normal_run_data_features.csv"
DOS_PATH    = FEATURE_DIR / "DoS_dataset_features.csv"

normal = pd.read_csv(NORMAL_PATH)
dos    = pd.read_csv(DOS_PATH)

print("Normal:", len(normal))
print("DoS:", len(dos))

df = pd.concat([normal, dos], ignore_index=True)


# 2. PREPROCESSING


# Label binário
df["label"] = (df["flag"] == "T").astype(int)
df = df.drop(columns=["flag"])

# Drop colunas que não usas
df = df.drop(columns=["payload", "id"], errors="ignore")

# Limpeza
df = df.drop_duplicates()
df = df.fillna(0)

print("Dataset final:", df.shape)


# 3. FEATURE SETS

payload_features = [c for c in df.columns if c.startswith("byte_")]

feature_sets = {
    "delta_time": ["delta_time"],

    "delta_time + id_interval": ["delta_time", "id_interval"],

    "delta_time + entropy": ["delta_time", "entropy"],

    "delta_time + entropy + id_interval": ["delta_time", "entropy", "id_interval"],

    "delta_time + payload": ["delta_time"] + payload_features,

    "entropy + payload": ["entropy"] + payload_features,
    
    "delta_time + entropy + payload": ["delta_time","entropy"] + payload_features,

    "delta_time + entropy + id_interval + payload": ["delta_time", "entropy", "id_interval"] + payload_features,

    "all_features": [c for c in df.columns if c != "label"]
}



# 4. TRAIN / TEST SPLIT


X_full = df.drop(columns=["label"])
y = df["label"]

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)



# 5. RUN EXPERIMENTS


results = []

for name, features in feature_sets.items():

    print(f"\n=== Experiência: {name} ===")

    X_train = X_train_full[features]
    X_test  = X_test_full[features]

    # Pipeline: Scaler + Logistic Regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=5000,
            n_jobs=-1
        ))
    ])

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results.append({
        "Features": name,
        "Num Features": len(features),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Train Time (s)": train_time
    })



 # 6. CONFUSION MATRIX


    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Normal (Predicted)", "DoS (Predicted)"],
        yticklabels=["Normal (Actual)", "DoS (Actual)"]
    )

    plt.title(f"Confusion Matrix — Logistic Regression — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()



# 7. RESULTS TABLE

results_df = pd.DataFrame(results)

print("\n COMPARAÇÃO FINAL — LOGISTIC REGRESSION \n")
print(
    results_df.sort_values("F1 Score", ascending=False)
              .reset_index(drop=True)
)

results_df.to_csv(RESULTS_DIR / "logistic_regression_feature_comparison_with_cm.csv", index=False)