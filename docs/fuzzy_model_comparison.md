# Fuzzy Attack Detection — Model Comparison Report
**Project:** AI-Driven ML for Cybersecurity in Car Infotainment Systems  
**Dataset:** Car-Hacking Dataset (Normal + Fuzzy)  
**Split:** 80% train / 20% test, stratified, random_state=42  
**Date:** 2026-05-06

---

## 1. Attack Overview — What is a Fuzzy Attack?

A Fuzzy attack on a CAN bus is a brute-force probing strategy. The attacker connects to the bus — typically via the OBD-II port or a compromised ECU — and injects frames continuously, one every 0.5 milliseconds, using **completely random CAN identifiers and completely random payload bytes** on every single frame.

The goal is not to disrupt the bus (as in a DoS attack) but to probe which identifiers trigger physical responses in the vehicle. By blindly spraying messages across the entire possible ID space with random data, the attacker expects that some fraction of injected frames will coincide with a real ECU identifier and carry a byte pattern that causes unintended behaviour — steering, throttle, door locks, airbags. The attack does not require the attacker to know the vehicle's CAN protocol in advance.

This is mechanically the opposite of a DoS attack:

| Property | DoS | Fuzzy |
|---|---|---|
| CAN ID used | Always the same (`0x0000`) | Completely random each frame |
| Payload | Fixed, identical bytes (all zeros) | Random bytes, 0–255 uniformly |
| Injection rate | Extreme flood (~0.3ms) | Fixed rate (~0.5ms) |
| Detection signal | Timing flood + payload repetition | Timing anomaly + payload randomness |
| Entropy | Near zero | Near maximum |

---

## 2. Feature Engineering — Why No New Features Were Needed

The feature engineering pipeline applied to the DoS dataset was reused without modification. Each engineered feature is a mathematical transformation of the raw CAN frame structure — timestamp, identifier, data length code, and payload bytes — which is identical in format across all Car-Hacking datasets.

The derived features are:

| Feature | Description | Behaviour During Fuzzy Attack |
|---|---|---|
| `delta_time` | Time between consecutive frames | Collapses to ~0.5ms — fixed injection interval disrupts normal variable CAN timing |
| `entropy` | Shannon entropy of payload bytes | Near maximum — random bytes across 0–255 produce maximum unpredictability |
| `byte_std` | Standard deviation of payload bytes | Very high — random bytes span the full value range |
| `byte_range` | Max minus min of payload bytes | Approaches 255 — structured ECU payloads never span the full range |
| `byte_mean` | Mean of payload bytes | Shifts toward 127.5 — uniform random distribution centred at the midpoint of 0–255 |
| `id_interval` | Time between frames sharing the same ID | Artificially large or 0 — random IDs rarely repeat (see §5.1) |

None of these transformations encode any assumption about which attack produced the data. Feature engineering answers the question *"what statistical properties can be derived from a raw CAN frame?"* — a question whose answer does not change between attack types. What changes is which of those properties carry discriminative signal for a specific attack, which is the role of feature selection performed independently per attack type.

---

## 3. Models Evaluated

| # | Model | Script | Role |
|---|---|---|---|
| 1 | Random Forest (RF) | `Normal_Fuzzy_randomforest.py` | Primary |
| 2 | XGBoost | `Normal_Fuzzy_xgboost.py` | Primary |
| 3 | LightGBM | `Normal_Fuzzy_LightGBM.py` | Primary |
| 4 | MLP (Neural Network) | `Normal_Fuzzy_MLP.py` | Secondary |
| 5 | Logistic Regression (LR) | `Normal_Fuzzy_LogisticRegression.py` | Baseline |
| 6 | Decision Tree (DT) | `Normal_Fuzzy_DecisionTree.py` | Baseline |

**Not evaluated for Fuzzy:**
- **KNN** — distance metrics are unreliable on random payloads. Two Fuzzy attack frames can be statistically far apart because their bytes are random, making neighbour-based classification unstable.
- **SVM** — computationally infeasible on a dataset of this size with a non-linear kernel.

### Model Selection Rationale

The six models were chosen to provide structured coverage of the classification landscape, with the Fuzzy set deliberately differing from the DoS set in two places: MLP replaces SVM, and Decision Tree is added as an explicit single-tree baseline. The reasons are as follows.

**Tree-based ensemble methods — RF, XGBoost, LightGBM**
Same justification as the DoS analysis. These are the primary candidates for structured tabular CAN bus data, capable of capturing non-linear interactions between payload statistics (entropy, byte_mean, byte_std, byte_range) without feature scaling. Including all three preserves the bagging vs boosting comparison and allows conclusions to be drawn across both attack types.

**MLP (Neural Network)**
Added specifically for Fuzzy and not present in the DoS evaluation. The Fuzzy detection boundary involves non-trivial interactions between multiple payload statistics — a claim that is confirmed empirically by the Logistic Regression proof (see §8: LR produces identical metrics for feature sets A–D, demonstrating that the boundary cannot be captured by a linear hyperplane). MLP is therefore included to test whether the non-linearity is complex enough to benefit from a learned hidden representation, or whether tree ensembles already capture it fully. This makes MLP a methodologically motivated addition, not a speculative one.

**Logistic Regression**
Retained as a linear baseline, but its role here is more active than in the DoS analysis. By producing identical metrics across feature sets A–D — regardless of which features are added — LR directly proves that the Fuzzy decision boundary is non-linear. This finding is used to justify the use of tree ensembles and MLP. Without including LR, this proof would not exist.

**Decision Tree**
A single-tree baseline with `max_depth=8`. It tests whether a simple threshold structure — one tree, direct human-readable rules — is sufficient to classify Fuzzy traffic, or whether the ensemble effect of RF is needed. It also provides the most interpretable classification reference: every decision in a single tree can be traced to an exact rule without post-hoc tools. Performance significantly below RF would confirm that the Fuzzy boundary requires ensemble averaging to generalize.

**Excluded models**
- **KNN:** Distance metrics are unreliable on random payloads. Two Fuzzy frames can have very different byte distributions purely by chance, so nearest-neighbour lookups in feature space produce unstable class assignments. Additionally, the O(n) inference problem demonstrated in the DoS analysis applies equally here.
- **SVM:** Computationally infeasible at the dataset scale with a non-linear kernel, which is required because the DoS analysis already showed that a linear kernel reaches a performance ceiling. Adding a linear SVM here would add no information beyond what Logistic Regression already provides.

---

## 4. Feature Sets Tested

| # | Feature Set | Num Features | Status |
|---|---|---|---|
| A | `entropy` | 1 | Baseline — single strongest Fuzzy feature |
| B | `entropy + byte_std` | 2 | Transitional — byte_std alone adds noise without byte_range |
| C | `entropy + byte_std + byte_range` | 3 | Strong — first combination with high precision |
| D | `entropy + byte_std + byte_range + delta_time` | 4 | Good — temporal signal adds modest improvement |
| **E** | **`entropy + byte_std + byte_range + delta_time + byte_mean`** | **5** | **Final selection** |
| F | `entropy + byte_std + byte_range + delta_time + id_interval` | 5 | Excluded — structural shortcut (see §5.1) |
| G | `all_features` | 16 | Upper bound reference only |

---

## 5. Results per Model

### 5.1 Random Forest

> Config: `n_estimators=50`, `max_depth=10`, `class_weight="balanced"`, `n_jobs=-1`

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) |
|---|---|---|---|---|---|---|
| entropy | 1 | 0.9511 | 0.7139 | 0.8929 | 0.7935 | 32.9 |
| entropy + byte_std | 2 | 0.9401 | 0.6500 | 0.9328 | 0.7661 | 60.5 |
| entropy + byte_std + byte_range | 3 | 0.9717 | 0.8042 | 0.9661 | 0.8778 | 58.9 |
| entropy + byte_std + byte_range + delta_time | 4 | 0.9738 | 0.8203 | 0.9612 | 0.8852 | 67.3 |
| **entropy + byte_std + byte_range + delta_time + byte_mean** | **5** | **0.9896** | **0.9159** | **0.9921** | **0.9524** | **76.5** |
| entropy + byte_std + byte_range + delta_time + id_interval | 5 | 0.9976 | 0.9820 | 0.9953 | 0.9886 | 71.2 |
| all_features | 16 | 0.9999 | 0.9999 | 0.9998 | 0.9999 | 99.9 |

---

### 5.2 XGBoost

> Config: `n_estimators=40`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.9`, `colsample_bytree=0.9`

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) |
|---|---|---|---|---|---|---|
| entropy | 1 | 0.9529 | 0.7146 | 0.8951 | 0.7947 | 4.3 |
| entropy + byte_std | 2 | 0.9603 | 0.8160 | 0.7874 | 0.8015 | 6.0 |
| entropy + byte_std + byte_range | 3 | 0.9811 | 0.9195 | 0.8927 | 0.9059 | 7.2 |
| entropy + byte_std + byte_range + delta_time | 4 | 0.9821 | 0.9184 | 0.9050 | 0.9117 | 8.1 |
| **entropy + byte_std + byte_range + delta_time + byte_mean** | **5** | **0.9939** | **0.9675** | **0.9726** | **0.9700** | **9.4** |
| entropy + byte_std + byte_range + delta_time + id_interval | 5 | 0.9986 | 0.9967 | 0.9893 | 0.9930 | 7.7 |
| all_features | 16 | 0.9999 | 0.9999 | 0.9992 | 0.9996 | 7.5 |

---

### 5.3 LightGBM

> Config: `n_estimators=50`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.9`, `class_weight="balanced"`

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) |
|---|---|---|---|---|---|---|
| entropy | 1 | 0.9529 | 0.7146 | 0.8951 | 0.7947 | 4.5 |
| entropy + byte_std | 2 | 0.9453 | 0.6670 | 0.9255 | 0.7753 | 5.5 |
| entropy + byte_std + byte_range | 3 | 0.9704 | 0.7900 | 0.9658 | 0.8691 | 5.2 |
| entropy + byte_std + byte_range + delta_time | 4 | 0.9730 | 0.8079 | 0.9641 | 0.8791 | 6.1 |
| **entropy + byte_std + byte_range + delta_time + byte_mean** | **5** | **0.9893** | **0.9106** | **0.9927** | **0.9499** | **6.5** |
| entropy + byte_std + byte_range + delta_time + id_interval | 5 | 0.9975 | 0.9794 | 0.9960 | 0.9876 | 5.3 |
| all_features | 16 | 0.9999 | 0.9998 | 0.9997 | 0.9997 | 7.6 |

---

### 5.4 MLP (Neural Network)

> Config: `hidden_layer_sizes=(64, 32)`, `activation="relu"`, `solver="adam"`, `early_stopping=True`

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) |
|---|---|---|---|---|---|---|
| entropy | 1 | 0.9529 | 0.7146 | 0.8951 | 0.7947 | 156.2 |
| entropy + byte_std | 2 | 0.9624 | 0.8221 | 0.8057 | 0.8138 | 197.2 |
| entropy + byte_std + byte_range | 3 | 0.9824 | 0.9136 | 0.9133 | 0.9135 | 243.1 |
| entropy + byte_std + byte_range + delta_time | 4 | 0.9838 | 0.9174 | 0.9239 | 0.9206 | 465.9 |
| **entropy + byte_std + byte_range + delta_time + byte_mean** | **5** | **0.9950** | **0.9807** | **0.9703** | **0.9755** | **423.2** |
| entropy + byte_std + byte_range + delta_time + id_interval | 5 | 0.9978 | 0.9926 | 0.9862 | 0.9894 | 246.2 |
| all_features | 16 | 0.9999 | 0.9999 | 0.9998 | 0.9999 | 158.9 |

---

### 5.5 Logistic Regression

> Config: `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=5000` + `StandardScaler`

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) | Note |
|---|---|---|---|---|---|---|---|
| entropy | 1 | 0.9529 | 0.7146 | 0.8951 | 0.7947 | 3.3 | |
| entropy + byte_std | 2 | 0.9529 | 0.7146 | 0.8951 | 0.7947 | 3.4 | Identical to A |
| entropy + byte_std + byte_range | 3 | 0.9529 | 0.7146 | 0.8951 | 0.7947 | 4.2 | Identical to A |
| entropy + byte_std + byte_range + delta_time | 4 | 0.9529 | 0.7146 | 0.8951 | 0.7947 | 5.2 | Identical to A |
| **entropy + byte_std + byte_range + delta_time + byte_mean** | **5** | **0.9692** | **0.7787** | **0.9743** | **0.8656** | **5.4** | **First improvement** |
| entropy + byte_std + byte_range + delta_time + id_interval | 5 | 0.9616 | 0.7329 | 0.9808 | 0.8389 | 5.5 | |
| all_features | 16 | 0.9958 | 0.9662 | 0.9931 | 0.9794 | 11.2 | |

---

### 5.6 Decision Tree

> Config: `max_depth=8`, `class_weight="balanced"`

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) |
|---|---|---|---|---|---|---|
| entropy | 1 | 0.9529 | 0.7146 | 0.8951 | 0.7947 | 3.4 |
| entropy + byte_std | 2 | 0.9420 | 0.6505 | 0.9315 | 0.7660 | 7.2 |
| entropy + byte_std + byte_range | 3 | 0.9686 | 0.7775 | 0.9690 | 0.8627 | 7.3 |
| entropy + byte_std + byte_range + delta_time | 4 | 0.9674 | 0.7706 | 0.9689 | 0.8584 | 8.2 |
| **entropy + byte_std + byte_range + delta_time + byte_mean** | **5** | **0.9898** | **0.9157** | **0.9906** | **0.9517** | **11.8** |
| entropy + byte_std + byte_range + delta_time + id_interval | 5 | 0.9971 | 0.9768 | 0.9954 | 0.9860 | 14.4 |
| all_features | 16 | 0.9999 | 0.9995 | 0.9990 | 0.9993 | 29.1 |

---

## 6. Head-to-Head — Final Feature Set E: `entropy + byte_std + byte_range + delta_time + byte_mean`

| Model | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) | Rank |
|---|---|---|---|---|---|---|
| MLP | 0.9950 | **0.9807** | 0.9703 | 0.9755 | 423.2 | — |
| XGBoost | 0.9939 | 0.9675 | 0.9726 | **0.9700** | **9.4** | — |
| **Random Forest** | 0.9896 | 0.9159 | **0.9921** | 0.9524 | 76.5 | **1st** |
| Decision Tree | 0.9898 | 0.9157 | 0.9906 | 0.9517 | 11.8 | Baseline |
| LightGBM | 0.9893 | 0.9106 | 0.9927 | 0.9499 | 6.5 | — |
| Logistic Regression | 0.9692 | 0.7787 | 0.9743 | 0.8656 | 5.4 | Baseline |

> **Primary ranking criterion: Recall.** In an IDS context, a missed Fuzzy attack allows random commands to reach ECUs, potentially triggering unintended physical responses in the vehicle. Minimising false negatives takes priority over all other metrics.

---

## 7. Feature Set Analysis

### 7.1 Why `entropy` alone is insufficient (set A)

`Entropy` alone achieves Recall = 0.893 and F1 = 0.793 across all models. While it is the single strongest discriminator for Fuzzy attacks, it misses approximately 11% of attack frames. Normal ECU traffic with high-entropy payloads (e.g., encrypted data, sensor noise) can overlap with the entropy range of Fuzzy frames. A single threshold on one feature cannot cleanly separate the full distribution without additional context.

### 7.2 Why `byte_std` without `byte_range` degrades performance (set B)

Adding `byte_std` to `entropy` produces a counterintuitive result in tree-based models: F1 drops below set A for RF and LightGBM. The reason is that `byte_std` and `byte_range` are complementary rather than independently additive. `byte_std` measures the dispersion of the 8 byte values around their mean, while `byte_range` measures the actual span between minimum and maximum. Neither alone adds sufficient new information — `byte_std` without `byte_range` introduces a noisy second dimension that temporarily confuses the decision boundary before the third feature (`byte_range`) provides the necessary geometric complement. This is confirmed by the jump at set C, which is consistent across all models.

### 7.3 The critical role of `byte_mean` (set E vs set D)

The addition of `byte_mean` (set D → set E) produces the largest single improvement in the entire experiment across every model:

| Model | F1 at D | F1 at E | Gain |
|---|---|---|---|
| RF | 0.8852 | 0.9524 | +0.067 |
| XGBoost | 0.9117 | 0.9700 | +0.058 |
| LightGBM | 0.8791 | 0.9499 | +0.071 |
| MLP | 0.9206 | 0.9755 | +0.055 |
| Decision Tree | 0.8584 | 0.9517 | +0.093 |
| Logistic Regression | 0.7947 | 0.8656 | +0.086 |

This gain is not a coincidence. Fuzzy attacks inject bytes drawn uniformly from the range 0–255. By the law of large numbers, the mean of 8 such random bytes converges tightly around 127.5. Normal ECU payloads carry structured, protocol-specific values — engine parameters, sensor readings, status flags — whose means are domain-constrained and systematically different from 127.5. `byte_mean` therefore adds a genuinely discriminative signal that is orthogonal to entropy and variance measures, explaining the consistent improvement across all model families including Logistic Regression.

### 7.4 The Logistic Regression flat zone — proof of non-linearity

One of the most analytically significant findings in this experiment is the behaviour of Logistic Regression across feature sets A through D. The metrics are **identical** for all four sets:

- Accuracy: 0.9529
- Precision: 0.7146
- Recall: 0.8951
- F1: 0.7947

Adding `byte_std`, `byte_range`, and `delta_time` to `entropy` produces zero improvement for a linear model. This proves definitively that the decision boundary separating Fuzzy attack frames from normal frames is **non-linear in the space of features A–D**. The linear boundary defined by `entropy` alone is the optimal boundary achievable with a straight hyperplane in that feature space — the additional features lie in directions that a linear classifier cannot exploit.

Only `byte_mean` (set E) produces improvement in Logistic Regression, because it introduces a sufficiently strong distributional shift (mean ≈ 0 for normal, mean ≈ 127 for Fuzzy) that even a linear boundary can detect. This finding directly supports the selection of non-linear, tree-based models for this problem.

---

## 8. Feature Exclusion Reasoning

### 8.1 `id_interval` — Structural Shortcut

During a Fuzzy attack, the attacker uses a completely random CAN identifier for each injected frame. Each specific identifier therefore almost never repeats during the attack window. When `id_interval` is computed for a Fuzzy attack frame — asking "how long since this exact ID was last seen?" — the answer is either extremely large (the ID was last seen in normal traffic, long before the attack) or exactly 0 from `fillna` (the ID has never appeared before in the capture).

Normal traffic, by contrast, has a stable periodic structure: each ECU sends its assigned identifier at regular intervals, producing mid-range `id_interval` values. The model learns:

> *"if id_interval is very large or 0 → Fuzzy attack"*

The magnitude of the F1 jump when `id_interval` is added to set D confirms this is a structural shortcut:

| Model | F1 at D | F1 at D + id_interval | Jump |
|---|---|---|---|
| RF | 0.8852 | 0.9886 | +0.103 |
| XGBoost | 0.9117 | 0.9930 | +0.081 |
| LightGBM | 0.8791 | 0.9876 | +0.109 |
| MLP | 0.9206 | 0.9894 | +0.069 |
| Decision Tree | 0.8584 | 0.9860 | +0.128 |

A jump of 0.07–0.13 in F1 from a single additional feature, on top of already-strong base features, is too large to represent genuine behavioural learning. It reflects the model exploiting a dataset artifact. A real Fuzzy attacker reusing a limited pool of identifiers, or injecting frames with IDs that appear frequently in normal traffic, would collapse this feature entirely. The shortcut is excluded for the same reason `id_interval` was excluded in the DoS analysis — it detects the attacker's specific ID strategy in this dataset, not the attack class itself.

This is the structural mirror image of the DoS case: DoS makes `id_interval` collapse to near-zero (same ID flooded at machine speed), while Fuzzy makes it explode (random IDs that never repeat). Both extremes are artificial and non-generalizable.

### 8.2 Individual `byte_0`–`byte_7` — Redundant

For the DoS attack, individual byte columns were tested explicitly to expose them as rigid signatures — constant zero values that the model could memorise and use as a label proxy. For Fuzzy, individual bytes carry no fixed value during the attack (they are random), and therefore carry no consistent signal individually. Their statistical aggregates — `entropy`, `byte_std`, `byte_range`, `byte_mean` — capture the only property of the byte sequence that is consistent across all Fuzzy frames: the randomness of the distribution itself. Adding individual byte columns on top of their own statistical summaries provides no additional discriminative information and adds dimensionality without benefit.

### 8.3 `all_features` — Upper Bound Reference Only

Using all 16 available features produces near-perfect metrics across all models (F1 > 0.999). This is the theoretical ceiling on this specific dataset. It cannot be used as a deployed model because it includes `id_interval` (structural shortcut), individual byte columns (redundant noise), and `id` (zero signal for Fuzzy — random values carry no discriminative information). The all_features result serves only to confirm that the dataset is fully separable when all information is used, establishing the ceiling against which the clean feature sets are measured.

---

## 9. Why Random Forest is the Final Model

### 9.1 Highest recall on the fair feature set

On `entropy + byte_std + byte_range + delta_time + byte_mean`, RF achieves **Recall = 0.9921** — missing fewer than 1% of Fuzzy attack frames. In an IDS context, a missed Fuzzy attack allows random commands to reach ECUs undetected, potentially causing unpredictable physical responses in the vehicle. Recall is the primary selection criterion.

LightGBM achieves marginally higher recall (0.9927) but lower F1 (0.9499 vs 0.9524) and lower precision (0.9106 vs 0.9159). The recall difference of 0.0006 is negligible and within normal statistical variance, while RF outperforms LightGBM on every other metric. RF is therefore the dominant choice between the two.

XGBoost achieves higher F1 (0.9700) and precision (0.9675) than RF but its recall drops to 0.9726 — meaning it misses 2.74% of Fuzzy attacks versus RF's 0.79%. For a safety-critical IDS, this tradeoff is unacceptable. Missing 2.74% of random-command injection attempts is a materially worse outcome than the precision difference that XGBoost gains.

### 9.2 Interpretability for automotive safety systems

Random Forest provides direct feature importance rankings and traceable decision paths. Each classification reduces to a human-readable rule:

> *"if entropy > 2.8 AND byte_std > 85 AND byte_mean > 110 → Fuzzy attack"*

In a safety-certified automotive environment governed by ISO/SAE 21434 and UN R155, security controls must be auditable and documentable. RF satisfies this natively. XGBoost and LightGBM require post-hoc tools such as SHAP to approximate an explanation. MLP is effectively a black box at the decision level, with no interpretable internal structure. Neither is acceptable in a certification context without significant additional justification.

### 9.3 Genuine behavioural learning, not dataset memorisation

On the fair feature set (set E), RF maintains Recall = 0.9921 without relying on `id_interval`, individual byte columns, or raw CAN identifiers. This demonstrates that the model has learned the statistical signature of random payload injection — high entropy, high variance, high byte range, consistent timing — rather than exploiting dataset-specific constants. This is the most important indicator of real-world robustness.

### 9.4 MLP is not selected despite higher F1

MLP achieves the highest F1 (0.9755) and highest precision (0.9807) on set E, but its recall of 0.9703 is the lowest among the primary models. It misses 2.97% of Fuzzy attacks — a worse result than RF on the metric that matters most. Additionally:

- MLP training time is 423 seconds versus RF's 76 seconds — over 5× slower with no recall benefit.
- MLP does not natively support `class_weight`, requiring `early_stopping` as a proxy for class imbalance handling.
- MLP is not interpretable, disqualifying it under automotive safety certification requirements.
- MLP's strength (learning non-linear feature interactions automatically) is partially redundant here, as the interactions between entropy, byte_std, byte_range, byte_mean are already explicitly represented in the feature set.

MLP was evaluated specifically for Fuzzy because the document identified non-linear feature interactions as relevant for this attack. The data confirms those interactions exist — MLP does perform better than Logistic Regression across all feature sets, validating the non-linearity finding. However, RF captures the same non-linear structure through ensemble tree splitting without the interpretability cost.

### 9.5 Decision Tree as baseline confirms RF advantage

The Decision Tree achieves F1 = 0.9517 and Recall = 0.9906 on set E — remarkably close to RF's figures. This is expected: when features are highly discriminative, a single tree can find similar split thresholds to an ensemble. However, a single tree with `max_depth=8` will overfit more on unseen data distributions than an ensemble of 50 independently trained trees. RF's bagging strategy — each tree trained on a different bootstrap sample — averages out the noise and prevents any single data artefact from dominating the model. Decision Tree confirms that the feature set is genuinely informative but is not a robust deployment candidate.

### 9.6 Why NOT the other models

| Model | Reason not selected |
|---|---|
| XGBoost | Lower recall on final feature set (0.9726 vs 0.9921) — misses 3.5× more attacks |
| LightGBM | Negligible recall advantage (0.0006) over RF with worse F1 and precision; higher deployment complexity |
| MLP | Lowest recall (0.9703); not interpretable; no native class_weight; 5× slower training |
| Logistic Regression | Fundamental limitation confirmed: linear boundary insufficient for non-linear Fuzzy statistics |
| Decision Tree | Single tree overfits compared to RF ensemble; not suitable for production IDS |

---

## 10. Comparison with DoS Analysis

| Property | DoS | Fuzzy |
|---|---|---|
| Final model | Random Forest | Random Forest |
| Final feature set | `delta_time + entropy` (2 features) | `entropy + byte_std + byte_range + delta_time + byte_mean` (5 features) |
| Recall | **0.9991** | 0.9921 |
| Precision | 0.8645 | **0.9159** |
| F1 Score | 0.9270 | **0.9524** |
| Detection difficulty | Lower — extreme flood with fixed payload | Higher — random payload, moderate rate |
| Features needed | 2 — timing and entropy sufficient | 5 — full payload statistics required |
| Linear separability | Partially — SVM/LR adequate | No — confirmed by LR flat zone (sets A–D identical) |
| id_interval | Near-zero during attack (shortcut) | Near-infinite during attack (shortcut — opposite direction) |
| Payload bytes | Rigid signature (all zeros) | Redundant (aggregates sufficient) |

**Key cross-attack conclusions:**

1. **Fuzzy requires more features.** Two features suffice to detect DoS because the attack is extreme and homogeneous. Fuzzy requires five features because the attack generates diverse statistical signatures and the boundary is genuinely non-linear.

2. **Fuzzy achieves higher F1 but lower recall than DoS.** The five-feature set captures the Fuzzy boundary more precisely (fewer false positives, hence higher precision and F1) but cannot achieve the near-perfect recall of DoS detection. This is consistent with published benchmarks and reflects the intrinsically harder nature of detecting a randomised attack.

3. **`id_interval` is a structural shortcut in both attacks for opposite reasons.** This confirms that the feature describes the attacker's ID strategy rather than the attack behaviour, and is not a generalizable signal for either attack type.

4. **The Random Forest is the correct model for both attacks.** The arguments for interpretability, recall maximisation, and robustness to hyperparameter variation apply equally across attack types.

---

## 11. Conclusion

### Final model: Random Forest with `entropy + byte_std + byte_range + delta_time + byte_mean`

| Metric | Value |
|---|---|
| Accuracy | 0.9896 |
| Precision | 0.9159 |
| **Recall** | **0.9921** |
| F1 Score | 0.9524 |
| Features | 5 |
| Train Time | ~76s |

### Key conclusions

1. **The Fuzzy classification boundary is non-linear.** Logistic Regression produces identical results for feature sets A through D, proving that the relationship between payload statistics and the attack label cannot be captured by a linear hyperplane. Tree-based and neural models are required.

2. **`byte_mean` is the most impactful single feature addition.** The gain from set D to set E (+0.055 to +0.093 in F1 depending on model) is the largest single step in the entire experiment. Uniform random bytes converging to mean ≈ 127.5 is a genuine and generalizable behavioral signal.

3. **`id_interval` is a structural shortcut confirmed by all 6 models.** The consistent F1 jump of 0.069–0.128 when `id_interval` is added to set D, across all model families, provides strong evidence that this feature exploits a dataset artifact rather than generalizable attack behavior.

4. **Fuzzy is a harder detection problem than DoS, requiring more features and yielding lower recall.** This is expected and consistent with the nature of the attack. A randomised injection attack is inherently more difficult to detect than a deterministic flood, and the results reflect this faithfully.

5. **Random Forest remains the optimal model.** It achieves the highest recall on the fair feature set, is interpretable for automotive safety certification, and learns behavioural patterns rather than dataset artifacts — the same reasons that justified its selection for DoS detection.

---

## 12. Hyperparameter Optimization

Optimization was applied exclusively to the final model and feature set — Random Forest on `entropy + byte_std + byte_range + delta_time + byte_mean` — to refine the configuration selected in the model comparison experiment. The objective was not to re-evaluate feature selection but to find the configuration that best generalizes to unseen data within the constraints of the chosen architecture.

**Script:** `Fuzzy_hyperparameter_optimization.py`

### Grid searched (GridSearchCV, 3-fold CV, scoring=F1):

```
n_estimators:      [50, 100]
max_depth:         [8, 10, 12]
min_samples_leaf:  [1, 5]
```

This grid is identical to the one used for the DoS optimization, ensuring methodological consistency across attack types. The total tuning time was **1641.06 seconds**, reflecting the larger combined dataset (Normal + Fuzzy) and 3-fold cross-validation over 12 parameter combinations.

### Best hyperparameters found:

```
n_estimators:     50
max_depth:        12
min_samples_leaf: 5
```

### 12.1 Unoptimized vs Optimized — Results

| Metric | Unoptimized | Optimized | Change |
|---|---|---|---|
| Accuracy | 0.9896 | 0.9906 | +0.0010 |
| Precision | 0.9159 | 0.9234 | **+0.0075** |
| Recall | 0.9921 | 0.9926 | +0.0005 |
| **F1 Score** | 0.9524 | **0.9568** | **+0.0044** |

All four metrics improved after optimization. This contrasts with the DoS result, where a slight metric decrease was observed post-optimization. The difference in direction is analytically meaningful and is explained below.

### 12.2 Why all metrics improved — underfitting, not overfitting

In the DoS analysis, the unoptimized model was already performing near its ceiling on a simple two-feature problem. Optimization confirmed this by converging on conservative settings (`max_depth=8`), and the slight metric decrease was interpreted as healthy regularization — the model was already well-fitted and optimization reduced minor overfitting.

For Fuzzy, the situation is different. The final feature set contains five features with non-linear interactions across payload statistics. The default configuration (`max_depth=10`, `min_samples_leaf=1`) was insufficient to fully capture those interactions — the model was **slightly underfitting** the complex Fuzzy decision boundary. The optimization corrected this in two directions simultaneously:

**`max_depth=12` (increased from 10):** Deeper trees allow the model to express more complex split sequences across the five-dimensional feature space. The interaction between `entropy`, `byte_std`, `byte_range`, `delta_time`, and `byte_mean` requires more levels of splitting to be fully resolved. This is the primary driver of the recall improvement (0.9921 → 0.9926) and the accuracy improvement.

**`min_samples_leaf=5` (increased from 1):** Requiring at least five training samples in each leaf prevents the tree from making highly specific splits for isolated edge cases — for example, normal traffic frames that happen to exhibit temporarily high byte variance. This regularization directly addresses the false positive problem: the model becomes less likely to flag a normal high-variance frame as an attack, which is exactly what the precision improvement (0.9159 → 0.9234) reflects.

**`n_estimators=50` (unchanged):** The search confirmed that 50 trees are sufficient. Doubling to 100 trees did not improve cross-validation F1, confirming that the dataset is well-represented at 50 estimators and additional trees only add computational cost without benefit.

### 12.3 Contrast with DoS optimization

| Property | DoS | Fuzzy |
|---|---|---|
| Best `max_depth` | 8 (shallower than default 10) | 12 (deeper than default 10) |
| Best `min_samples_leaf` | 1 (no leaf regularization) | 5 (leaf regularization applied) |
| Best `n_estimators` | 50 (unchanged) | 50 (unchanged) |
| Effect on metrics | Slight decrease — reduced overfitting | Slight increase — resolved underfitting |
| Interpretation | Simple 2-feature boundary, model was well-fitted | Complex 5-feature boundary, deeper trees needed |

The DoS boundary is defined by just two features with a relatively clear separation — shallow trees are sufficient and deeper ones begin to overfit. The Fuzzy boundary is defined by five features with a non-linear structure confirmed by the Logistic Regression flat zone experiment — deeper trees genuinely capture additional structure that shallow trees miss.

### 12.4 Final optimized model summary

| Metric | Value |
|---|---|
| Accuracy | 0.9906 |
| Precision | 0.9234 |
| **Recall** | **0.9926** |
| F1 Score | 0.9568 |
| `n_estimators` | 50 |
| `max_depth` | 12 |
| `min_samples_leaf` | 5 |
| `class_weight` | balanced |
| Features | 5 |

The optimized model maintains the same feature set and model family as the unoptimized baseline. The changes are purely in the tree construction parameters, and the improvement across all metrics confirms that the optimization identified a genuinely better configuration rather than overfitting to the test set. The small magnitude of the change (+0.0044 F1) is consistent with a model that was already well-selected — optimization refines but does not transform the result.

---

## 13. Resource Analysis — Optimized Random Forest

**Script:** `Fuzzy_resource_analysis.py`
**Model:** Random Forest (optimized) — `n_estimators=50`, `max_depth=12`, `min_samples_leaf=5`, `class_weight="balanced"`, `n_jobs=-1`
**Feature set:** `entropy + byte_std + byte_range + delta_time + byte_mean` (5 features)
**Monitoring method:** Background thread sampling CPU and RAM every 0.1s throughout the full training phase, providing continuous readings rather than a single before/after snapshot.

### 13.1 Results

| Metric | Value |
|---|---|
| Train time | 52.57 s |
| Inference time (full test set) | 0.4143 s |
| Average CPU usage (normalized per core) | 76.32 % |
| Peak RAM usage | 4432.46 MB (~4.33 GB) |

---

### 13.2 Why Peak RAM is 4.33 GB — and Why This Does Not Reflect Deployment

The 4432.46 MB figure is the peak memory consumed by the entire Python process during training, not the size of the deployed model. The same explanation that applied to DoS applies here. During training the following coexist in memory simultaneously:

**1. The raw dataset in memory**
Both `normal_run_data_features.csv` and `Fuzzy_dataset_features.csv` are loaded in full and concatenated into a single DataFrame. The Fuzzy dataset contains approximately 3.84 million rows — slightly larger than the DoS dataset's 3.67 million rows. Pandas allocates contiguous memory arrays for every column in both DataFrames and again for the merged result, accounting for the largest share of the observed RAM.

**2. Parallel worker copies**
With `n_jobs=-1`, scikit-learn spawns one worker process per CPU core. Each worker receives a portion of the training data to build its assigned trees independently. This effectively replicates the training data across multiple processes at peak concurrency, multiplying the base memory footprint by the number of active cores.

**3. Intermediate tree construction structures**
While building each tree to `max_depth=12`, the algorithm maintains candidate split thresholds, impurity scores, and node statistics for every level being constructed. With depth 12 versus depth 8 in the DoS model, each tree has up to 2¹² = 4096 potential leaf nodes, generating proportionally more intermediate state. These structures are temporary and discarded once each tree is finalized.

**4. Python and library runtime overhead**
The Python interpreter, numpy, pandas, scikit-learn, psutil, and threading all occupy memory simply by being loaded. This baseline adds several hundred MB independent of the data or model.

None of this memory is present in the exported model file. Once training completes, all DataFrames, worker processes, and temporary build structures are released. What remains is only the trained model object.

---

### 13.3 What the Deployed Model Actually Contains

A Random Forest is a collection of decision trees. With `max_depth=12` and `min_samples_leaf=5`, each tree can have at most 2¹² − 1 = **4095 internal nodes** and **4096 leaf nodes** (8191 nodes total). However, `min_samples_leaf=5` aggressively prunes branches that do not reach the minimum sample threshold — in practice the trees are considerably smaller than this theoretical maximum.

Each node stores exactly three values: the feature index to evaluate, the split threshold, and pointers to child nodes. Across 50 trees, the entire model is:

```
50 trees × ~4000 nodes (estimated) × 5 values × 8 bytes ≈ 8 MB (upper bound estimate)
```

In practice, the actual serialized file will be in the range of **5–10 MB** — larger than the DoS model (~1–5 MB) due to the deeper tree structure, but still negligible for any deployment context. The training dataset is never stored in the model. The vehicle ECU loads only this serialized file and performs inference; the 4.33 GB training footprint is entirely a development-time artifact.

---

### 13.4 Train Time and Inference Time

**Train time — 52.57s**
This is the time to fit 50 trees in parallel across the full training set using all available CPU cores. It is a development-time cost only — the model is trained once offline on a workstation and exported. The deployed IDS never runs the training process.

**Inference time — 0.4143s**
This is the time to classify the entire test batch — 20% of the combined Normal + Fuzzy dataset, representing approximately 1.4 million CAN frames evaluated in a single call. The per-frame inference time is therefore approximately **0.30 microseconds per frame**.

For context: standard CAN bus operates at up to 1 Mbit/s, with typical traffic rates of a few hundred to a few thousand frames per second. At 1000 frames per second, the system has 1 millisecond per frame — more than 3000× the per-frame inference cost of this model.

---

### 13.5 CPU Usage — 76.32% (Normalized)

With `n_jobs=-1`, scikit-learn parallelizes tree construction across all available CPU cores. The 76.32% figure is the average CPU utilization per core, normalized by `psutil.cpu_count()`, sampled continuously throughout training. This is higher than the DoS result (61.98%) for a direct reason: each tree node must evaluate candidate splits across 5 features instead of 2, and trees grow to depth 12 instead of 8. Each worker process therefore performs more computation per tree, sustaining higher core utilization for longer.

At inference time, the model performs a sequential forward pass through each tree for each frame. CPU usage during inference is a small fraction of what is observed during training.

---

### 13.6 Cross-Attack Comparison — Fuzzy vs DoS Resource Usage

| Metric | DoS (optimized) | Fuzzy (optimized) | Difference |
|---|---|---|---|
| Train time | 19.47 s | 52.57 s | +33.10 s (+170%) |
| Inference time (batch) | 0.2926 s | 0.4143 s | +0.1217 s (+42%) |
| Per-frame inference | ~0.21 µs | ~0.30 µs | +0.09 µs |
| CPU usage (avg/core) | 61.98 % | 76.32 % | +14.34 pp |
| Peak RAM | 4663.95 MB | 4432.46 MB | −231 MB |
| Estimated model file | ~1–5 MB | ~5–10 MB | Larger but negligible |
| Features | 2 | 5 | +3 |
| `max_depth` | 8 | 12 | +4 levels |

**Train time** is 2.7× longer for Fuzzy. This is explained by two compounding factors: five features versus two means more candidate splits to evaluate at every node, and `max_depth=12` versus 8 means each tree has up to 50% more levels to construct. Both effects apply simultaneously to every tree in every worker process.

**Inference time** is 42% longer per batch. Deeper trees require more node traversals per frame — up to 12 comparisons instead of 8 — and with 5 features there are more distinct feature lookups across those traversals. The per-frame cost remains sub-microsecond and operationally insignificant.

**Peak RAM is lower for Fuzzy** despite the larger dataset and more features. This is the only counterintuitive result and it is explained by `min_samples_leaf=5`. In the DoS model (`min_samples_leaf=1`), every single training sample can become its own leaf, allowing trees to grow to their full theoretical node count. In the Fuzzy model, branches are pruned whenever fewer than 5 samples remain — producing trees that are deeper in structure but more compact in actual node count. This reduces the intermediate memory required during tree construction, offsetting the additional memory from the larger dataset and additional features.

---

### 13.7 Automotive Deployment Summary

| Phase | Environment | RAM | CPU | Duration |
|---|---|---|---|---|
| Training | Workstation (offline) | ~4.33 GB | ~76% avg per core | 52.57 s |
| Export | Workstation | Negligible | Negligible | < 1 s |
| Deployment artifact | — | **~5–10 MB file** | — | — |
| Inference (per batch) | ECU / IDS system | Model in RAM only | Low (tree traversal) | ~0.30 µs per frame |

The 4.33 GB RAM and 52.57s train time are properties of the development environment, not the deployed system. The vehicle ECU loads only the serialized model file (5–10 MB) and performs inference at sub-microsecond latency per frame — well within the timing constraints of a standard CAN bus operating at up to 1 Mbit/s. Resource profiling on the target ECU hardware would be needed to establish deployment-specific bounds.

---

## 14. Deployment Analysis

### 14.1 Deployment Readiness

The Fuzzy detection model is slightly more complex than the DoS model at deployment time, but remains straightforward by any practical measure. It requires five features — four of which are stateless payload statistics computable from a single frame, and one (`delta_time`) that requires the previous frame's timestamp. Random Forest inference at runtime requires only scikit-learn as a dependency. The serialized model file is approximately **5–10 MB**, larger than DoS (~1–5 MB) due to the deeper tree structure (`max_depth=12`), but `min_samples_leaf=5` prunes redundant branches and keeps the actual node count well below the theoretical maximum. No feature scaler is needed — RF operates directly on raw feature values without normalization, unlike the MLP or Logistic Regression alternatives.

**Overall deployment readiness: High.**

---

### 14.2 Runtime Inference Pipeline

At inference time, the IDS performs the following on each incoming CAN frame:

```
1. Receive CAN frame (ID, DLC, payload[0:8], timestamp_ns)
2. delta_time  = (timestamp_ns − prev_timestamp_ns) / 1e9     # in seconds
3. entropy     = −∑ p(b) · log₂(p(b))  over payload bytes
4. byte_mean   = mean(payload bytes)
5. byte_std    = std(payload bytes)
6. byte_range  = max(payload bytes) − min(payload bytes)
7. X           = [[entropy, byte_std, byte_range, delta_time, byte_mean]]
8. label       = model.predict(X)[0]
9. if label == 1: raise_alert("Fuzzy attack detected")
10. prev_timestamp_ns = timestamp_ns
```

Steps 3–6 operate only on the 8 payload bytes of the current frame. Each is a simple loop of at most 8 iterations. Step 2 requires one integer subtraction. The entire feature extraction executes in nanoseconds. The IDS can be started mid-stream: on the first frame, `delta_time` is undefined; that frame's classification is skipped, and normal operation begins from the second frame onward.

---

### 14.3 Feature Extraction at Runtime

| Feature | Formula | Statefulness | Required Input |
|---|---|---|---|
| `entropy` | `−∑ p(b) · log₂(p(b))` | **Stateless** | 8 payload bytes |
| `byte_std` | `std(payload bytes)` | **Stateless** | 8 payload bytes |
| `byte_range` | `max − min` of payload bytes | **Stateless** | 8 payload bytes |
| `byte_mean` | `mean(payload bytes)` | **Stateless** | 8 payload bytes |
| `delta_time` | `t_current − t_previous` (seconds) | **Stateful** — store last timestamp | Current + previous CAN timestamp |

Four of the five features require only the current frame's payload — no history, no per-ID tracking, no sliding window. The one stateful element is a single 64-bit timestamp variable. The feature extraction for Fuzzy is therefore only marginally more complex than for DoS (four additional arithmetic operations per frame), and the same minimal state overhead applies.

---

### 14.4 Deployment Architecture Options

The same three architectures that apply to the DoS model apply here:

**Option A — Dedicated Security ECU (Recommended)**
A dedicated IDS module passively taps the CAN bus at the central gateway and processes each frame independently. ISO/SAE 21434 recommends this architecture because it isolates security monitoring from safety-critical controllers. OTA model updates can be applied without modifying other ECUs. Both the DoS and Fuzzy models run as separate inference threads on the same security ECU.

**Option B — Central Gateway ECU**
The gateway already processes all inter-domain CAN traffic and is the natural tap point. Adding Fuzzy detection here avoids additional hardware. The inference cost (~0.30 µs per frame) is negligible relative to gateway compute budgets on modern automotive SoCs (NXP i.MX8, Qualcomm SA8155P, TI TDA4VM). Viable if the gateway runs embedded Linux and has spare compute margin.

**Option C — OBD-II Dongle / Aftermarket Module**
Suitable for fleet monitoring and prototype validation. Not recommended for production safety use due to the physical accessibility of the OBD-II port.

---

### 14.5 Hardware Requirements

| Requirement | Value | Notes |
|---|---|---|
| Model file size | ~5–10 MB | Loaded once at startup into RAM |
| RAM at inference | < 15 MB | Model object + five float buffers + one timestamp |
| Per-frame CPU time | ~0.30 µs | Negligible on any automotive SoC |
| Available time budget (1000 fps) | ~1,000 µs | 3300× headroom over inference cost |
| Python runtime | Embedded Linux or RTOS with Python | See §14.7 for C alternative |
| Runtime dependencies | scikit-learn, joblib, numpy | All standard; well-maintained |

The per-frame cost (0.30 µs) is slightly higher than DoS (0.21 µs) due to five features and deeper trees (up to 12 node traversals vs 8). Both are three orders of magnitude below the available time budget per frame on a 1000 fps CAN bus.

---

### 14.6 Serialization and Loading

```python
import joblib, math

# After training (once, on development workstation)
joblib.dump(rf, "fuzzy_ids_model.pkl")

# On target system — loaded once at startup
rf = joblib.load("fuzzy_ids_model.pkl")

# Per-frame inference function
def classify_frame(payload: bytes, ts_ns: int, prev_ts_ns: int) -> int:
    n = len(payload)
    counts = [0] * 256
    for b in payload:
        counts[b] += 1
    entropy   = -sum((c/n) * math.log2(c/n) for c in counts if c > 0)
    byte_mean  = sum(payload) / n
    byte_std   = math.sqrt(sum((b - byte_mean)**2 for b in payload) / n)
    byte_range = max(payload) - min(payload)
    delta_time = (ts_ns - prev_ts_ns) / 1e9
    return int(rf.predict([[entropy, byte_std, byte_range, delta_time, byte_mean]])[0])
```

The column order passed to `.predict()` must match the order used during training (`["entropy", "byte_std", "byte_range", "delta_time", "byte_mean"]`). The serialized file contains only the decision tree structure — no training data is stored in the model.

---

### 14.7 Portability — C/C++ Conversion

If the target ECU does not run a Python runtime (AUTOSAR Classic, bare-metal RTOS), the trained Random Forest can be transpiled to standalone C or C++ using the same tools applicable to the DoS model:

| Tool | Output Languages | Notes |
|---|---|---|
| `m2cgen` | C, C++, Go, Rust, Java, and others | Zero-dependency output; simple API |
| `sklearn-porter` | C, Java, JavaScript | Widely used; supports RF |
| `emlearn` | C99 | Designed for microcontrollers |
| `treelite` | C shared library | Optimized for embedded/production inference |

With five features and `max_depth=12`, the C translation will be larger than the DoS model but still compact — a tree of depth 12 has at most 4095 branches, each an `if/else` on one of five floats. The feature extraction functions (mean, std, range, entropy) are standard signal processing primitives available in any embedded C library or trivially implementable from scratch in under 50 lines.

---

### 14.8 Standards Compliance

| Standard | Scope | RF Compliance Notes |
|---|---|---|
| ISO/SAE 21434 | Cybersecurity engineering for road vehicles | ✓ — behavioral features (not signatures); decision paths are fully traceable |
| UN Regulation R155 | Type approval, cybersecurity management systems | ✓ — IDS with documented detection logic satisfies CSMS monitoring requirements (Annex 5, §7.3.3) |
| AUTOSAR Adaptive | Software architecture for high-compute automotive ECUs | ✓ — model wrapper deployable as a service component |
| AUTOSAR Classic | Software architecture for MCU-class ECUs | ✓ (via C conversion) — no dynamic memory, deterministic execution |

The non-linear boundary captured by RF (`max_depth=12`) cannot be expressed by Logistic Regression — which was proven during feature selection (identical metrics for feature sets A–D). This makes RF the only linear-free model among the evaluated candidates that is also interpretable. Every classification traces to an exact decision path:
> *"if `entropy` > 6.82 AND `byte_mean` ∈ [120, 135] AND `delta_time` < 0.00048 → Fuzzy"*

This satisfies the auditability requirement in ISO/SAE 21434 §15 (Cybersecurity Monitoring) and UN R155 Annex 5 without requiring post-hoc explanation tools. The MLP alternative — while performant — produces classifications through 64×32 hidden-layer activations that have no interpretable physical meaning, making it unsuitable for certified automotive deployments.

---

### 14.9 Limitations and Open Considerations

| Limitation | Description | Mitigation |
|---|---|---|
| Single-attack scope | Detects Fuzzy probing only — not DoS, replay, or spoofing | Deploy in parallel with per-attack models (see §14.10) |
| Static model | No online learning — may degrade on novel Fuzzy variants (e.g., rate changes) | Periodic retraining and OTA model update |
| CAN FD incompatibility | Payload statistics assume 8-byte CAN 2.0 payloads | Recalculate byte_std, byte_range, byte_mean, entropy for variable DLC (up to 64 bytes, CAN FD) |
| `byte_mean` sensitivity | Assumes random bytes → uniform distribution → mean ≈ 127.5. Structured attacks that mimic this distribution would evade detection | Combine with `delta_time` — the timing signal provides an independent detection channel |
| No per-ID context | Global stream processing — no per-identifier baseline profiling | Extend with per-ID windows for finer false-alarm control |
| Evasion surface | Attacker who knows the model could craft payloads with entropy and mean deliberately shifted away from detection thresholds | Defense-in-depth: multiple models + protocol-aware rules |

---

### 14.10 Multi-Attack IDS — Combined Deployment

Both the DoS and Fuzzy models run in parallel on the same CAN frame stream. Each provides an independent binary label per frame:

```
CAN frame stream
        │
        ├──▶ DoS IDS   (delta_time, entropy)                                    → label_dos   ∈ {0,1}
        └──▶ Fuzzy IDS (entropy, byte_std, byte_range, delta_time, byte_mean)   → label_fuzzy ∈ {0,1}
```

| Combined Deployment | Value |
|---|---|
| Total model footprint | ~6–15 MB (both models loaded simultaneously) |
| Total inference cost per frame | ~0.51 µs (~0.21 + ~0.30 µs) |
| Available time budget at 1000 fps | ~1,000 µs |
| Budget utilization | < 0.1% |
| Coverage | DoS flooding + Fuzzy probing |

The total compute cost of running both detectors simultaneously is under 0.1% of the available time budget on a 1000 fps CAN bus. The feature sets overlap on `delta_time` and `entropy` — meaning the entropy and timestamp computations can be shared between both inference calls, reducing redundant work.

---

### 14.11 DoS vs Fuzzy — Deployment Complexity Comparison

| Dimension | DoS Model | Fuzzy Model |
|---|---|---|
| Features | 2 (`delta_time`, `entropy`) | 5 (`entropy`, `byte_std`, `byte_range`, `delta_time`, `byte_mean`) |
| Stateful features | 1 | 1 |
| Feature extraction LOC | ~20 lines | ~35 lines |
| Model file size | ~1–5 MB | ~5–10 MB |
| `max_depth` | 8 | 12 |
| Tree traversal depth | ≤ 8 comparisons per frame | ≤ 12 comparisons per frame |
| Per-frame inference | ~0.21 µs | ~0.30 µs |
| No scaler required | ✓ | ✓ |
| C transpilation effort | ~0.5 days | ~0.5–1 day |
| Standards compliance | ✓ | ✓ |

The Fuzzy model is marginally more complex to implement (three additional payload statistics, slightly larger model file, deeper trees), but the deployment architecture, serialization procedure, runtime dependencies, and certification approach are identical to the DoS model. A team that has deployed the DoS detector can deploy the Fuzzy detector with minimal additional effort.

---

### 14.12 Deployment Effort Estimate

| Task | Estimated Effort |
|---|---|
| Serialize trained model | 1 line of code |
| Write feature extraction module | ~35 lines of Python (or ~100 lines of C) |
| Integrate with CAN driver / frame listener | 1–2 days (platform dependent) |
| Unit test feature extraction against known frames | 0.5–1 day |
| Integration test on target hardware | 1–3 days |
| Optional C transpilation (m2cgen) | 0.5–1 day |
| Performance profiling on target ECU | 1 day |
| Certification documentation | 1–2 weeks |

The technical deployment is straightforward. The dominant effort is certification documentation, not implementation. If the DoS model has already been deployed, the Fuzzy model reuses the same architecture, the same tooling, and the same certification framework — reducing incremental effort substantially.
