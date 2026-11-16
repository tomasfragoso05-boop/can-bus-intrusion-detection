import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CAN_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "feature"
RESULTS_DIR = REPO_ROOT / "results" / "dos"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import time
import psutil
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load data


NORMAL_PATH = FEATURE_DIR / "normal_run_data_features.csv"
DOS_PATH    = FEATURE_DIR / "DoS_dataset_features.csv"


normal = pd.read_csv(NORMAL_PATH)
dos    = pd.read_csv(DOS_PATH)

df = pd.concat([normal, dos], ignore_index=True)

df["label"] = (df["flag"] == "T").astype(int)
df = df.drop(columns=["flag", "payload", "id"], errors="ignore")
df = df.dropna()

FEATURES = ["delta_time", "entropy"]
X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Resource analysis


process = psutil.Process(os.getpid())

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

cpu_before = process.cpu_percent(interval=None)
mem_before = process.memory_info().rss / 1024**2

start = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start

cpu_after = process.cpu_percent(interval=None)
mem_after = process.memory_info().rss / 1024**2

start_inf = time.time()
rf.predict(X_test)
infer_time = time.time() - start_inf

print("\nResource Analysis — Optimized Random Forest")
print(f"Train time (s): {train_time:.2f}")
print(f"Inference time (s): {infer_time:.4f}")
print(f"CPU usage (%): {cpu_after - cpu_before:.2f}")
print(f"RAM usage (MB): {mem_after - mem_before:.2f}")
