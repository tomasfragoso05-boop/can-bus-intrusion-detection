# CAN Bus Intrusion Detection with Machine Learning

Machine learning models that detect **Denial-of-Service (DoS)** and **Fuzzy** attacks on the CAN bus of connected vehicles, trained on the public [Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) (HCRL, Korea University, more than 3.6 million real CAN frames).

Developed as my BSc final project in Electrical and Computer Engineering at ISEP (Porto), Feb to Jun 2026: *"Cybersecurity in Infotainment Systems for Cars: AI Driven Machine Learning Models"*.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B)

## Results

| Attack | Final model | Features | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|---|
| DoS | Random Forest (tuned) | 2 | **0.9992** | 0.8708 | 0.9306 | 0.9808 |
| Fuzzy | Random Forest (tuned) | 5 | **0.9926** | 0.9234 | 0.9568 | 0.9906 |

- **Recall is the main metric.** In a vehicle, a missed attack costs far more than a false alarm.
- **Sub-microsecond inference per CAN frame** on consumer hardware. The full DoS test set (about 733,000 frames) is classified in 0.29 s, about 0.4 µs per frame, which suggests the model is light enough for embedded automotive hardware.
- Training takes under 20 s on more than 4 million frames, with a peak of about 4.6 GB of RAM during training.
- The work is framed around automotive cybersecurity standards **ISO/SAE 21434** and **UN ECE WP.29 (R155)**.

## Pipeline

```
raw CAN logs ──► data_cleaning ──► feature_engineering ──► feature selection ──► model comparison ──► GridSearchCV ──► resource analysis
 (.txt/.csv)     parse, dedupe,     6 temporal and         leakage checks,        6 classifiers ×     RF tuning          CPU, RAM,
                 split payload      statistical features   importance, MI         9 feature sets      (3-fold, F1)       latency
```

### Engineered features

| Feature | Meaning |
|---|---|
| `delta_time` | Time since the previous frame on the bus |
| `id_interval` | Time since the previous frame with the same CAN ID |
| `byte_mean`, `byte_std`, `byte_range` | Statistics over the 8 payload bytes |
| `entropy` | Shannon entropy of the payload |

### Avoiding shortcut learning

Near-perfect scores (F1 ≈ 0.999) are easy to reach on this dataset with features that leak the label. I tested this explicitly and excluded those features:

- **`id_interval` and the raw CAN `id` are degenerate.** DoS frames are injected with the highest-priority ID (`0x000`) and an all-zero payload at a fixed rate (about one every 0.3 ms), so these features identify the attack *tool*, not the attack *behaviour*. Including them lifts F1 to 0.999 without learning anything that would generalise.
- **Raw payload bytes act as a rigid signature.** They reach recall 1.0 by matching the dataset's fixed attack payload, which a real attacker would simply change.

The final models therefore use only **behavioural** features: `delta_time + entropy` for DoS, and `entropy + byte_std + byte_range + delta_time + byte_mean` for Fuzzy. Full results with every feature set are in [`docs/`](docs) and [`results/`](results).

### Models compared

| DoS | Fuzzy |
|---|---|
| Random Forest, XGBoost, LightGBM, Linear SVM, Logistic Regression, KNN | Random Forest, XGBoost, LightGBM, MLP, Logistic Regression, Decision Tree |

Random Forest was selected for both attacks because it had the highest recall on the fair feature set, it is interpretable, and its behaviour stayed consistent across the two attacks. For Fuzzy, the MLP had a higher F1 (0.9755) but lower recall and about 5.5× longer training.

## Repository structure

```
preprocessing/
  data_cleaning.py          # parse raw HCRL logs into clean CSVs
  feature_engineering.py    # compute the 6 engineered features
experiments/
  dos/                      # model comparison, feature selection, tuning, resource analysis (DoS)
  fuzzy/                    # same for Fuzzy attacks
results/                    # metrics for each model and feature set (CSV)
docs/
  dos_model_comparison.md   # full analysis and reasoning, DoS
  fuzzy_model_comparison.md # full analysis and reasoning, Fuzzy
data/README.md              # how to download and prepare the dataset
```

## How to run

```bash
pip install -r requirements.txt
# download the dataset into data/raw/ (see data/README.md)
python preprocessing/data_cleaning.py
python preprocessing/feature_engineering.py

python experiments/dos/compare_random_forest.py          # compare feature sets
python experiments/dos/hyperparameter_optimization1.py   # GridSearchCV
python experiments/dos/resource_analysis2.py             # CPU / RAM / latency
```

All splits are 80/20, stratified, with `random_state=42`.

## Author

**Tomás Barreira**, BSc in Electrical and Computer Engineering (ISEP). Currently doing the MSc in Automation and Systems at ISEP.
Interested in embedded systems, edge AI and automotive cybersecurity.

## Acknowledgements

The dataset is from the HCRL (Hacking and Countermeasure Research Lab, Korea University): E. Seo, H. M. Song, H. K. Kim, *"GIDS: GAN based Intrusion Detection System for In-Vehicle Network"*, PST 2018.
Project supervised by Prof. Isabel Jesus (ISEP).
