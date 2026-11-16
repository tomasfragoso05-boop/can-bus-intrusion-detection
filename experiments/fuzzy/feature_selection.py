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
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt





NORMAL_PATH = FEATURE_DIR / "normal_run_data_features.csv"
FUZZY_PATH    = FEATURE_DIR / "Fuzzy_dataset_features.csv"




fuzzy =    pd.read_csv(FUZZY_PATH)
normal = pd.read_csv(NORMAL_PATH)




FEATURES = ["delta_time", "id_interval", "entropy", "id", "dlc","byte_mean","byte_std", "byte_range" ]

df = pd.concat([normal[FEATURES + ["flag"]],
                fuzzy[FEATURES + ["flag"]]], ignore_index=True)

df["label"] = (df["flag"] == "T").astype(int)
df = df.drop(columns=["flag"])

X = df[FEATURES]
y = df["label"]



# 2. CORRELAÇÃO (Pearson)

corr = X.join(y).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm",fmt=".2f",annot_kws={"size":10})
plt.title("Correlação entre Features")
plt.tight_layout()
plt.show()




# 3. MUTUAL INFORMATION


mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=FEATURES)
mi_series = mi_series.sort_values(ascending=False)

plt.figure(figsize=(6,4))
mi_series.plot(kind='bar', color='orange')
plt.title("Mutual Information Score (Feature → Label)")
plt.ylabel("MI Score")
plt.show()

print("\nMutual Information Scores:")
print(mi_series)



# 4. RANDOM FOREST FEATURE IMPORTANCE


rf = RandomForestClassifier(n_estimators=50, max_depth=8)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(6,4))
importances.plot(kind='bar', color='green')
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance")
plt.show()

print("\nRandom Forest Importances:")
print(importances)

