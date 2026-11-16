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
import threading

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


process = psutil.Process(os.getpid())
cpu_readings = []
ram_readings = []
monitoring = True

def monitor_resources(interval=0.1):
    while monitoring:
        cpu_readings.append(process.cpu_percent(interval=None))
        ram_readings.append(process.memory_info().rss / 1024**2)
        time.sleep(interval)

# Model (OPTIMIZED)


rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)


# TRAINING PHASE


monitoring = True
monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

start_train = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start_train

monitoring = False
monitor_thread.join()

cpu_avg = sum(cpu_readings) / len(cpu_readings)
cpu_normalized = cpu_avg / psutil.cpu_count()
ram_peak = max(ram_readings)


# INFERENCE PHASE


start_inf = time.time()
rf.predict(X_test)
infer_time = time.time() - start_inf

# OUTPUT

print("\nResource Analysis — Optimized Random Forest")
print(f"Train time (s): {train_time:.2f}")
print(f"Inference time (s): {infer_time:.4f}")
print(f"Average CPU usage (%): {cpu_normalized:.2f}")
print(f"Peak RAM usage (MB): {ram_peak:.2f}")