import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CAN_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "feature"

import os
import pandas as pd
import numpy as np



#   FEATURE ENGINEERING 

def get_byte_cols():
    return [f"byte_{i}" for i in range(8)]


def add_delta_time(df):
    df["delta_time"] = df["timestamp_ms"].diff().fillna(0)
    return df


def add_id_interval(df):
    df["id_interval"] = df.groupby("id")["timestamp_ms"].diff().fillna(0)
    return df


def add_byte_mean(df):
    df["byte_mean"] = df[get_byte_cols()].mean(axis=1)
    return df


def add_byte_std(df):
    df["byte_std"] = df[get_byte_cols()].std(axis=1).fillna(0)
    return df


def add_byte_range(df):
    df["byte_range"] = df[get_byte_cols()].max(axis=1) - df[get_byte_cols()].min(axis=1)
    return df


def add_entropy(df):

    def entropy(row):
        vals = [int(v) for v in row[get_byte_cols()] if not pd.isna(v)]
        if len(vals) == 0:
            return 0
        counts = np.bincount(vals, minlength=256)
        probs = counts / len(vals)
        return -np.sum(probs * np.log2(probs + 1e-9))

    df["entropy"] = df.apply(entropy, axis=1)
    return df


def feature_engineering(df):
    #  Converter ID para inteiro (hex → decimal)
    df["id"] = df["id"].apply(lambda x: int(str(x), 16))

    df = add_delta_time(df)
    df = add_id_interval(df)
    df = add_byte_mean(df)
    df = add_byte_std(df)
    df = add_byte_range(df)
    df = add_entropy(df)

    return df



#   APLICAR FE A TODOS OS FICHEIROS *_clean.csv

def apply_feature_engineering(processed_dir, features_dir):

    os.makedirs(features_dir, exist_ok=True)

    files = [f for f in os.listdir(processed_dir) if f.endswith("_clean.csv")]

    print(f"\n{len(files)} ficheiros encontrados para Feature Engineering.\n")

    for f in files:
        path = os.path.join(processed_dir, f)
        print(f"🔧 A criar features para: {f}")

        df = pd.read_csv(path)
        df_feat = feature_engineering(df)

        out_path = os.path.join(features_dir, f.replace("_clean", "_features"))
        df_feat.to_csv(out_path, index=False)

        print(f"Guardado em: {out_path}\n")

    print(" Feature Engineering concluído!\n")



#  MAIN

if __name__ == "__main__":
    # Reads data/processed/*_clean.csv and writes data/feature/*_features.csv
    apply_feature_engineering(PROCESSED_DIR, FEATURE_DIR)
