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

from sklearn.model_selection import train_test_split
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

print("Normal:", len(normal))
print("Fuzzy:", len(fuzzy))

df = pd.concat([normal, fuzzy], ignore_index=True)


# 2. PREPROCESSING


df["label"] = (df["flag"] == "T").astype(int)
df = df.drop(columns=["flag"])

df = df.drop(columns=["payload", "id"], errors="ignore")

df = df.drop_duplicates()
df = df.dropna()


print(df)
#ataques = df[df["label"] == 1]   # filtra só ataques
#print(ataques)                   # imprime as linhas completas


print("Dataset final:", df.shape)


# 3. DEFINE FEATURE SETS
#
# Fuzzy attack injects completely random CAN IDs and random payloads every 0.5ms.
# Unlike DoS (low entropy, fixed ID), Fuzzy produces high entropy and high byte
# variance. Feature sets are built progressively from the strongest Fuzzy signals.
#
# Excluded globally: id (random value, zero signal), id_interval (random IDs
# rarely repeat — intervals become undefined noise), individual byte_0-7
# (aggregated statistics are far superior).
#
# Combination reference (fuzzy_attack_analysis.docx):
#   A — entropy only
#   B — entropy + byte_std
#   C — entropy + byte_std + byte_range
#   D — entropy + byte_std + byte_range + delta_time  ← recommended
#   E — entropy + byte_std + byte_range + delta_time + byte_mean

feature_sets = {

    # A — minimum viable baseline: single strongest Fuzzy feature
    "entropy": ["entropy"],

    # B — payload randomness core: entropy + variance
    "entropy + byte_std": ["entropy", "byte_std"],

    # C — full payload statistics: adds byte spread
    "entropy + byte_std + byte_range": ["entropy", "byte_std", "byte_range"],

    # D — recommended primary: payload statistics + temporal flooding signal
    "entropy + byte_std + byte_range + delta_time": ["entropy", "byte_std", "byte_range", "delta_time"],

    # E — full benchmark: adds byte_mean (uniform random distribution → ~127)
    "entropy + byte_std + byte_range + delta_time + byte_mean": ["entropy", "byte_std", "byte_range", "delta_time", "byte_mean"],

    # Exclusion validation — id_interval is unreliable for Fuzzy:
    # random IDs rarely repeat so per-ID intervals are sparse and noisy.
    # Including it should not improve and may degrade the result.
    "entropy + byte_std + byte_range + delta_time + id_interval": ["entropy", "byte_std", "byte_range", "delta_time", "id_interval"],

    # Upper bound reference — all available features, includes problematic ones.
    # Not a valid deployment candidate; used only to establish the metric ceiling.
    "all_features": [c for c in df.columns if c != "label"]
}


# 4. TRAIN / TEST SPLIT


X_full = df.drop(columns=["label"])
y      = df["label"]

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# 5. RUN EXPERIMENTS


results = []

for name, features in feature_sets.items():

    print(f"\n=== Experiência: {name} ===")

    X_train = X_train_full[features]
    X_test  = X_test_full[features]

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results.append({
        "Features":       name,
        "Num Features":   len(features),
        "Accuracy":       acc,
        "Precision":      prec,
        "Recall":         rec,
        "F1 Score":       f1,
        "Train Time (s)": train_time
    })


    # 6. CONFUSION MATRIX


    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (Predicted)", "Fuzzy (Predicted)"],
        yticklabels=["Normal (Actual)", "Fuzzy (Actual)"]
    )

    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# 7. RESULTS TABLE


results_df = pd.DataFrame(results)

print("\n COMPARAÇÃO FINAL \n")
print(
    results_df.sort_values("F1 Score", ascending=False)
              .reset_index(drop=True)
)

results_df.to_csv(RESULTS_DIR / "fuzzy_random_forest_feature_comparison_with_cm.csv", index=False)
