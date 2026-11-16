import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CAN_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "feature"

import os
import re
import pandas as pd


#  REGEX para parsing dos ficheiros TXT

LINE_PATTERN = re.compile(
    r"Timestamp:\s*([\d\.]+)\s*"
    r"ID:\s*([0-9A-Fa-f]+)\s+[0-9A-Fa-f]+\s*"
    r"DLC:\s*(\d+)\s*"
    r"([0-9A-Fa-f ]+)"
)

def parse_payload(payload_str, dlc):
    parts = payload_str.strip().split()
    if len(parts) != dlc:
        return None
    try:
        return [int(p, 16) for p in parts]
    except:
        return None


#  LIMPAR FICHEIRO TXT (normal_run_data)

def clean_txt_file(filepath):
    rows = {"timestamp_raw": [], "timestamp_ms": [], "id": [], "dlc": [], "payload": [], "flag": []}

    with open(filepath, "r") as f:
        for line in f:
            m = LINE_PATTERN.search(line)
            if not m:
                continue

            ts = float(m.group(1))
            can_id = m.group(2)
            dlc = int(m.group(3))
            payload_str = m.group(4)

            payload = parse_payload(payload_str, dlc)
            if payload is None:
                continue

            rows["timestamp_raw"].append(ts)
            rows["timestamp_ms"].append(int(ts * 1000))
            rows["id"].append(can_id)
            rows["dlc"].append(dlc)
            rows["payload"].append(payload)
            rows["flag"].append("R")  # TXT nunca tem ataque, logo R

    return pd.DataFrame(rows)


#  LIMPAR FICHEIRO CSV (universal, irregular, com FLAG R/T)

def clean_csv_file(filepath):

    rows = {"timestamp_raw": [], "timestamp_ms": [], "id": [], "dlc": [], "payload": [], "flag": []}

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            # CSV mal formatado → ignorar linha
            if len(parts) < 4:
                continue

            try:
                ts = float(parts[0])
                can_id = parts[1]
                dlc = int(parts[2])
                flag = parts[-1]  # última coluna é a flag R/T
            except:
                continue

            # Bytes = desde col3 até à penúltima coluna
            byte_cols = parts[3:-1]

            if len(byte_cols) < dlc:
                continue

            try:
                payload = [int(b, 16) for b in byte_cols[:dlc]]
            except:
                continue

            rows["timestamp_raw"].append(ts)
            rows["timestamp_ms"].append(int(ts * 1000))
            rows["id"].append(can_id)
            rows["dlc"].append(dlc)
            rows["payload"].append(payload)
            rows["flag"].append(flag)

    return pd.DataFrame(rows)


#  NORMALIZAR DATAFRAME E UNIFICAR ESTRUTURA

def normalize_clean_dataframe(df):

    if df is None or df.empty:
        return None

    df = df[df["payload"].notna()]
    df = df[df["payload"].apply(lambda x: len(x) > 0)]

    df["p_tuple"] = df["payload"].apply(tuple)
    df = df.drop_duplicates(subset=["timestamp_raw", "id", "p_tuple", "flag"])
    df = df.drop(columns=["p_tuple"])

    max_bytes = int(df["payload"].apply(len).max())

    for i in range(max_bytes):
        df[f"byte_{i}"] = df["payload"].apply(lambda x: x[i] if i < len(x) else None)

    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    
    df = df.drop(columns=["timestamp_raw"])

    return df


#  PROCESSAR TODOS OS FICHEIROS DA PASTA raw/

def clean_all_files(raw_dir, processed_dir):

    os.makedirs(processed_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir)
             if f.lower().endswith(".txt") or f.lower().endswith(".csv")]

    print(f"\n {len(files)} ficheiros encontrados.\n")

    for f in files:
        in_path = os.path.join(raw_dir, f)
        out_path = os.path.join(processed_dir, f.rsplit(".", 1)[0] + "_clean.csv")

        print(f" A limpar: {f}")

        try:
            if f.lower().endswith(".txt"):
                df = clean_txt_file(in_path)
            else:
                df = clean_csv_file(in_path)

            df = normalize_clean_dataframe(df)

            if df is None or df.empty:
                print(f"    Ficheiro vazio/ignorado.\n")
                continue

            df.to_csv(out_path, index=False)
            print(f"    Guardado como {out_path}\n")

        except Exception as e:
            print(f"    Erro ao processar {f}: {e}\n")

    print(" Terminado!\n")


#  MAIN

if __name__ == "__main__":
    # Raw Car-Hacking Dataset files (.txt / .csv) go in data/raw/
    clean_all_files(RAW_DIR, PROCESSED_DIR)
