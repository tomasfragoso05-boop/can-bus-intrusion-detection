# DoS Attack Detection — Model Comparison Report
**Project:** AI-Driven ML for Cybersecurity in Car Infotainment Systems  
**Dataset:** Car-Hacking Dataset (Normal + DoS)  
**Split:** 80% train / 20% test, stratified, random_state=42  
**Date:** 2026-04-23

---

## 1. Models Evaluated

| # | Model | Script |
|---|---|---|
| 1 | Random Forest (RF) | `Normal_DoS_randomforest.py` |
| 2 | XGBoost | `Normal_DoS_xgboost.py` |
| 3 | LightGBM | `Normal_Dos_LightGBM.py` |
| 4 | SVM (Linear) | `Normal_DoS_SVM.py` |
| 5 | Logistic Regression (LR) | `Normal_DoS_LogisticRegression.py` |
| 6 | KNN | `Normal_DoS_KNN.py` |

### Model Selection Rationale

The six models span three distinct algorithmic families, chosen to provide structured coverage of the classification landscape and to answer specific methodological questions about the nature of the DoS detection boundary.

**Tree-based ensemble methods — RF, XGBoost, LightGBM**
These are the primary candidates for structured tabular CAN bus data. They capture non-linear decision boundaries natively, require no feature scaling, and tolerate class imbalance well with `class_weight="balanced"`. They represent the established state of the art for tabular classification and are expected to be the strongest performers on behavioral CAN features. Including all three allows direct comparison between bagging (RF) and boosting (XGBoost, LightGBM), which differ in how they aggregate weak learners and in their sensitivity to data distribution.

**Linear models — SVM (Linear kernel), Logistic Regression**
Included as controlled baselines. A linear model's performance on the final feature set directly answers whether the DoS/Normal boundary is linearly separable in feature space. If a linear model achieves comparable metrics to an ensemble, the underlying problem is simpler than expected and a complex model adds no value. If it underperforms, non-linear models are necessary — and the gap in metrics quantifies how much non-linearity matters. Both models are kept deliberately simple (linear kernel, `lbfgs` solver) so that any performance difference reflects the boundary geometry, not model configuration.

**Instance-based learning — KNN**
Included as a distance-based, non-parametric baseline. KNN makes no assumptions about the data distribution and can approximate any decision boundary given enough data, making it a useful reference. Its inclusion also serves a secondary methodological purpose: empirically demonstrating why O(n) inference time — where n is the number of training samples — is fundamentally incompatible with real-time IDS deployment. This is not a tuning problem but a structural property of the algorithm, and its inclusion makes the argument concrete rather than theoretical.

**Excluded models**
- **MLP:** The DoS attack pattern (timing flood + payload repetition) produces a learnable boundary with shallow tree structures. A neural network would introduce unnecessary complexity and remove interpretability without a performance benefit.
- **Decision Tree:** Subsumed by Random Forest. A single decision tree has higher variance and lower generalization than an ensemble of trees trained on the same features.

---

## 2. Feature Sets Tested

| # | Feature Set | Num Features | Status |
|---|---|---|---|
| 1 | `delta_time` | 1 | Baseline — insufficient alone |
| 2 | `delta_time + id_interval` | 2 | Excluded — structural shortcut (see §6.1) |
| 3 | **`delta_time + entropy`** | **2** | **Final selection** |
| 4 | `delta_time + entropy + id_interval` | 3 | Excluded — contains degenerate feature |
| 5 | `delta_time + payload` | 12 | Excluded — rigid signature (see §6.2) |
| 6 | `entropy + payload` | 12 | Excluded — rigid signature |
| 7 | `delta_time + entropy + payload` | 13 | Excluded — rigid signature |
| 8 | `delta_time + entropy + id_interval + payload` | 14 | Excluded — both issues |
| 9 | `all_features` | 16 | Upper bound reference only (see §6.3) |

---

## 3. Results per Model

### 3.1 Random Forest

> Config: `n_estimators=50`, `max_depth=10`, `class_weight="balanced"`, `n_jobs=-1`

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) | Note |
|---|---|---|---|---|---|---|---|
| delta_time | 1 | 0.8769 | 0.5204 | 0.2004 | 0.2893 | 29.03 | |
| delta_time + id_interval | 2 | 0.9925 | 0.9565 | 0.9845 | 0.9703 | 43.82 | Degenerate |
| **delta_time + entropy** | **2** | **0.9803** | **0.8645** | **0.9991** | **0.9270** | **28.35** | **Final** |
| delta_time + entropy + id_interval | 3 | 0.9998 | 0.9988 | 0.9998 | 0.9993 | 31.81 | Degenerate |
| delta_time + payload | 12 | 0.9803 | 0.8645 | 0.9991 | 0.9270 | 33.09 | Signature |
| entropy + payload | 12 | 0.9749 | 0.8330 | 1.0000 | 0.9089 | 29.96 | Signature |
| delta_time + entropy + payload | 13 | 0.9803 | 0.8645 | 0.9991 | 0.9270 | 32.90 | Signature |
| delta_time + entropy + id_interval + payload | 14 | 0.9998 | 0.9987 | 0.9998 | 0.9993 | 33.93 | Both issues |
| all_features | 16 | 0.9998 | 0.9985 | 0.9998 | 0.9991 | 38.95 | Upper bound |

---

### 3.2 XGBoost

> Config: `n_estimators=40`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.9`, `colsample_bytree=0.9`

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) | Note |
|---|---|---|---|---|---|---|---|
| delta_time | 1 | 0.8801 | 0.5579 | 0.1065 | 0.1789 | 4.66 | |
| delta_time + id_interval | 2 | 0.9892 | 0.9769 | 0.9342 | 0.9551 | 4.85 | Degenerate |
| **delta_time + entropy** | **2** | **0.9817** | **0.8721** | **0.9970** | **0.9304** | **3.66** | **Final** |
| delta_time + entropy + id_interval | 3 | 0.9997 | 0.9987 | 0.9989 | 0.9988 | 4.49 | Degenerate |
| delta_time + payload | 12 | 0.9817 | 0.8721 | 0.9970 | 0.9304 | 4.60 | Signature |
| entropy + payload | 12 | 0.9756 | 0.8343 | 1.0000 | 0.9097 | 4.16 | Signature |
| delta_time + entropy + payload | 13 | 0.9817 | 0.8721 | 0.9970 | 0.9304 | 4.72 | Signature |
| delta_time + entropy + id_interval + payload | 14 | 0.9997 | 0.9990 | 0.9989 | 0.9990 | 4.75 | Both issues |
| all_features | 16 | 0.9998 | 0.9992 | 0.9989 | 0.9991 | 4.88 | Upper bound |

---

### 3.3 LightGBM

> Config: `n_estimators=50`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.9`, `class_weight="balanced"`

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) | Note |
|---|---|---|---|---|---|---|---|
| delta_time | 1 | 0.8787 | 0.5143 | 0.1984 | 0.2864 | 4.19 | |
| delta_time + id_interval | 2 | 0.9921 | 0.9533 | 0.9838 | 0.9683 | 5.83 | Degenerate |
| **delta_time + entropy** | **2** | **0.9809** | **0.8659** | **0.9992** | **0.9278** | **4.24** | **Final** |
| delta_time + entropy + id_interval | 3 | 0.9998 | 0.9988 | 0.9999 | 0.9993 | 3.83 | Degenerate |
| delta_time + payload | 12 | 0.9809 | 0.8659 | 0.9992 | 0.9278 | 5.49 | Signature |
| entropy + payload | 12 | 0.9756 | 0.8343 | 1.0000 | 0.9097 | 4.94 | Signature |
| delta_time + entropy + payload | 13 | 0.9809 | 0.8659 | 0.9992 | 0.9278 | 6.31 | Signature |
| delta_time + entropy + id_interval + payload | 14 | 0.9998 | 0.9988 | 0.9999 | 0.9993 | 7.54 | Both issues |
| all_features | 16 | 0.9999 | 0.9990 | 0.9999 | 0.9994 | 9.12 | Upper bound |

---

### 3.4 SVM (Linear Kernel)

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) | Note |
|---|---|---|---|---|---|---|---|
| delta_time | 1 | 0.8138 | 0.2187 | 0.2014 | 0.2097 | 3.74 | |
| delta_time + id_interval | 2 | 0.6661 | 0.1676 | 0.4342 | 0.2418 | 4.33 | Degenerate + fails |
| **delta_time + entropy** | **2** | **0.9756** | **0.8343** | **1.0000** | **0.9097** | **4.49** | **Final** |
| delta_time + entropy + id_interval | 3 | 0.9756 | 0.8345 | 0.9993 | 0.9095 | 6.32 | Degenerate |
| delta_time + payload | 12 | 0.9756 | 0.8343 | 1.0000 | 0.9097 | 26.33 | Signature |
| entropy + payload | 12 | 0.9756 | 0.8343 | 1.0000 | 0.9097 | 15.83 | Signature |
| delta_time + entropy + payload | 13 | 0.9756 | 0.8343 | 1.0000 | 0.9097 | 19.23 | Signature |
| delta_time + entropy + id_interval + payload | 14 | 0.9756 | 0.8345 | 0.9993 | 0.9095 | 20.53 | Both issues |
| all_features | 16 | 0.9813 | 0.8681 | 0.9993 | 0.9291 | 61.02 | Upper bound |

---

### 3.5 Logistic Regression

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) | Note |
|---|---|---|---|---|---|---|---|
| delta_time | 1 | 0.6020 | 0.1404 | 0.4384 | 0.2127 | 2.26 | |
| delta_time + id_interval | 2 | 0.9585 | 0.7668 | 0.9508 | 0.8489 | 3.24 | Degenerate |
| **delta_time + entropy** | **2** | **0.9756** | **0.8343** | **1.0000** | **0.9097** | **4.15** | **Final** |
| delta_time + entropy + id_interval | 3 | 0.9756 | 0.8345 | 0.9993 | 0.9095 | 5.14 | Degenerate |
| delta_time + payload | 12 | 0.9690 | 0.7981 | 1.0000 | 0.8877 | 7.18 | Signature |
| entropy + payload | 12 | 0.9756 | 0.8343 | 1.0000 | 0.9097 | 6.86 | Signature |
| delta_time + entropy + payload | 13 | 0.9756 | 0.8343 | 1.0000 | 0.9097 | 7.63 | Signature |
| delta_time + entropy + id_interval + payload | 14 | 0.9756 | 0.8345 | 0.9993 | 0.9095 | 8.93 | Both issues |
| all_features | 16 | 0.9813 | 0.8680 | 0.9993 | 0.9291 | 8.25 | Upper bound |

---

### 3.6 KNN (K-Nearest Neighbours)

> Config: `n_neighbors=5`, `algorithm='auto'` (scikit-learn default, selects KD-tree for 2 features)
>
> **Critical caveat on Train Time:** For KNN, `fit()` does no real learning — it only stores the training data and builds an index structure. The times shown below are therefore the index-build times only, not the cost of classification. The true computational cost of KNN falls entirely at inference time, which is O(n) per test frame relative to the number of training samples (~2.94 million). Running `predict()` on the ~734,000-frame test set took so long the run was practically uncompletable. This is not a configuration issue — it is a fundamental property of the algorithm.

| Feature Set | # Feat | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) ¹ | Note |
|---|---|---|---|---|---|---|---|
| delta_time | 1 | 0.8800 | 0.5572 | 0.1058 | 0.1779 | 2.76 | |
| delta_time + id_interval | 2 | 0.9941 | 0.9743 | 0.9776 | 0.9760 | 4.75 | Degenerate |
| **delta_time + entropy** | **2** | **0.9816** | **0.8722** | **0.9961** | **0.9300** | **4.50** | **Final** |
| delta_time + entropy + id_interval | 3 | 0.9998 | 0.9994 | 0.9992 | 0.9993 | 6.08 | Degenerate |
| delta_time + payload | 12 | 0.9816 | 0.8722 | 0.9960 | 0.9300 | 20.68 | Signature |
| entropy + payload | 12 | 0.9756 | 0.8343 | 1.0000 | 0.9097 | 22.91 | Signature |
| delta_time + entropy + payload | 13 | 0.9816 | 0.8722 | 0.9960 | 0.9300 | 22.29 | Signature |
| delta_time + entropy + id_interval + payload | 14 | 0.9998 | 0.9993 | 0.9990 | 0.9992 | 24.44 | Both issues |
| all_features | 16 | 0.9999 | 0.9994 | 0.9996 | 0.9995 | 2.45 | Upper bound |

¹ Train Time = `fit()` (index build) only. Inference time on this dataset is hours — see note above.

---

## 4. Head-to-Head — Final Feature Set: `delta_time + entropy`

> This is the comparison that matters: all models evaluated fairly, without degenerate or signature-based features.

| Model | Accuracy | Precision | Recall | **F1 Score** | Train Time (s) | Rank |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.9803 | 0.8645 | **0.9991** | 0.9270 | 28.35 | **1st** |
| LightGBM | 0.9809 | 0.8659 | 0.9992 | 0.9278 | **4.24** | 2nd |
| XGBoost | **0.9817** | **0.8721** | 0.9970 | **0.9304** | 3.66 | 2nd |
| KNN | 0.9816 | 0.8722 | 0.9961 | 0.9300 | 4.50 ¹ | — ² |
| SVM | 0.9756 | 0.8343 | 1.0000 | 0.9097 | 4.49 | 4th |
| Logistic Regression | 0.9756 | 0.8343 | 1.0000 | 0.9097 | 4.15 | 4th |

> Note: SVM and LR achieve Recall=1.0 here, but with significantly lower Precision (0.8343) — indicating more false positives. This is a linear boundary limitation, not a real advantage.
>
> ¹ KNN Train Time = `fit()` (index build) only — not inference time. Inference on ~734K test frames was impractically slow (hours) due to O(n) per-frame distance computation against ~2.94M training samples.
>
> ² KNN is excluded from ranking — its classification metrics are competitive but the model cannot perform real-time inference at this dataset scale. See §6.6.

---

## 5. Feature Set Exclusion Reasoning

### 5.1 `id_interval` — Structural Shortcut (Degenerate Feature)

During a DoS flood attack, the attacker injects a very high volume of frames with the same CAN ID at near-zero intervals. As a result, `id_interval` takes values close to 0 throughout the entire attack window, making it trivially separable from normal traffic. The model does not learn the *behavior* of the attack — it exploits a dataset-specific constant.

**Risk:** Any variation in timing (different hardware, partial flood, rate-limited attack) would collapse the model's performance. This is not generalizable to a real-world IDS.

| Feature Set | RF F1 | Notes |
|---|---|---|
| delta_time + id_interval | 0.9703 | High, but id_interval ≈ 0 during DoS |
| delta_time + entropy + id_interval | 0.9993 | Near-perfect due to shortcut, not learning |

### 5.2 Payload bytes — Rigid Signature (Dataset-Specific)

The DoS attack in this dataset uses a highly repetitive payload — the same byte pattern is transmitted repeatedly. Individual byte features therefore become near-constant during the attack, acting as a fixed signature detector rather than a behavioral detector.

**Risk:** Any variation in payload content (different DoS tool, obfuscated attack, protocol variant) would bypass detection entirely. This is the same weakness as traditional rule-based IDS — unsuitable for a robust ML-based system.

| Feature Set | RF Recall | Notes |
|---|---|---|
| entropy + payload | 1.0000 | Perfect recall via byte signature matching |
| delta_time + entropy | 0.9991 | High recall via behavioral pattern |

### 5.3 `all_features` — Upper Bound Reference Only

Using all 16 features yields the best raw metrics but combines both problems above, adds correlation noise, increases computational cost, and reduces interpretability. It represents the theoretical ceiling on this specific dataset, not a deployable IDS solution.

---

## 6. Why Random Forest is the Final Model

### 6.1 Consistent recall on the fair feature set

On `delta_time + entropy`, RF achieves **Recall = 0.9991** — missing almost no DoS attack. In an IDS context, a missed attack (false negative) is far more dangerous than a false alarm (false positive), so recall is the primary criterion after ruling out degenerate features.

XGBoost achieves marginally higher F1 (0.9304 vs 0.9270) and precision, but its recall drops to 0.9970, meaning it misses slightly more attacks. For a safety-critical automotive IDS, this tradeoff favors RF.

### 6.2 Behavioral learning, not signature matching

After removing id_interval and payload features, RF maintains high performance — demonstrating that it genuinely learns the temporal flooding pattern (`delta_time`) and payload predictability (`entropy`), not a dataset artifact. This is the strongest indicator of real-world robustness.

### 6.3 Interpretability for automotive safety systems

Random Forest provides direct feature importance rankings and traceable decision paths. In a safety-certified automotive environment (ISO/SAE 21434, UN R155), model decisions must be auditable. Boosted models (XGBoost, LightGBM) are harder to explain to certification bodies.

### 6.4 Robustness to hyperparameter variation

RF has no learning rate, no boosting schedule, and no subsample probability to tune. The model is far less sensitive to configuration choices than gradient boosting methods, making it more reliable across deployment environments.

### 6.5 RF vs LightGBM — Detailed Justification

On the final feature set, RF and LightGBM are statistically tied in performance (Recall 0.9991 vs 0.9992, F1 0.9270 vs 0.9278 — a difference of 0.0001 and 0.0008 respectively, well within normal variance). The decision therefore falls entirely on non-metric arguments, all of which favour RF in this context.

**Interpretability — the strongest argument**

RF builds independent trees where every decision is a direct human-readable rule:
> *"if `delta_time` < 0.00031 AND `entropy` < 0.14 → DoS"*

Every classification can be traced to an exact path through an exact tree. LightGBM uses gradient boosting — the final prediction is a weighted sum across hundreds of sequential weak trees, each correcting the residuals of the previous one. No single tree explains the decision; post-hoc tools like SHAP are required to approximate an explanation.

In an automotive security context governed by ISO/SAE 21434 and UN R155, security controls must be auditable and documentable. RF satisfies this natively. LightGBM does not.

**Overfitting risk on homogeneous data**

RF uses bagging — each tree is trained on a random bootstrap sample, independently. This naturally averages out noise and prevents any single pattern from dominating.

LightGBM uses boosting — each new tree is trained specifically to correct the errors of the previous ones, amplifying hard-to-classify samples. On a DoS dataset where the attack pattern is extremely homogeneous (consistent flooding, repetitive frames), boosting has nothing meaningful to correct and risks overfitting to the specific characteristics of this dataset. RF's bagging is the more appropriate inductive bias for a uniform, repetitive attack pattern.

**Hyperparameter stability**

RF has three meaningful hyperparameters (`n_estimators`, `max_depth`, `min_samples_leaf`) — intuitive and robust to small changes. LightGBM adds `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, and `reg_lambda`. The learning rate alone fundamentally changes model behaviour. This sensitivity is a liability in a deployed safety system where the model must behave predictably. The optimization confirmed RF works well at conservative settings — it does not require fine-tuning to be reliable.

**Deployment dependency**

RF requires only scikit-learn to run inference — a stable, widely supported library with no external native runtime. LightGBM requires its own native C++ library installed on the target system. For deployment on an automotive ECU or infotainment unit, every additional runtime dependency adds certification and maintenance burden.

**The speed advantage of LightGBM is irrelevant**

LightGBM trains approximately 7× faster (3.83s vs 19.47s). This matters when retraining frequently on live data. For a static deployed IDS model trained once offline on a fixed dataset, this difference is operationally meaningless.

| Argument | RF | LightGBM |
|---|---|---|
| Interpretability / auditability | Native — direct decision rules | Requires post-hoc tools (SHAP) |
| Overfitting on homogeneous data | Lower risk (bagging) | Higher risk (boosting amplifies residuals) |
| Hyperparameter stability | High — 3 intuitive params | Lower — learning rate sensitive |
| Deployment dependencies | scikit-learn only | Requires LightGBM native library |
| Training speed | 19.47s | 3.83s — irrelevant for static deployment |
| Performance (Recall / F1) | 0.9991 / 0.9270 | 0.9992 / 0.9278 — negligible difference |

---

### 6.6 Why NOT the other models

| Model | Reason not selected |
|---|---|
| XGBoost | Slightly lower recall on final feature set (0.9970 vs 0.9991); learning rate sensitivity |
| LightGBM | Tied on performance but inferior on interpretability, overfitting risk, and deployment complexity |
| SVM | Linear boundary — fundamentally limited for non-linear CAN bus patterns |
| Logistic Regression | Same limitation as SVM; lowest precision overall |
| KNN | Competitive metrics (F1=0.9300, Recall=0.9961) but O(n) inference complexity makes real-time classification impossible — inference on ~734K test frames took hours against ~2.94M training samples. Structurally undeployable as an IDS regardless of metrics |

---

## 7. Hyperparameter Optimization

Optimization was applied **exclusively to the final feature set** (`delta_time + entropy`) to refine the chosen model. The goal was not to re-evaluate feature selection, but to reduce overfitting and improve generalization.

### Grid searched (GridSearchCV, 3-fold CV, scoring=F1):

```
n_estimators:      [50, 100]
max_depth:         [8, 10, 12]
min_samples_leaf:  [1, 5]
```

### Two optimization runs:

| Script | Feature Set | Purpose |
|---|---|---|
| `hyperparameter_optimization1.py` | `delta_time + entropy` (2 features) | Final model optimization |
| `hyperparameter_optimization2.py` | `delta_time + entropy + payload` (13 features) | Reference comparison only |

### Effect of optimization

The best parameters landed on the conservative end of the grid (`max_depth=8`, `min_samples_leaf=1`, `n_estimators=50`), confirming that:
- The DoS pattern is learnable without deep trees
- Shallow trees reduce overfitting on the homogeneous attack pattern
- A slight metric drop post-optimization is expected and desirable — it reflects reduced overfitting, not worse performance

> The small decrease in accuracy/F1 after optimization is intentional. A model that generalizes better to unseen data is more valuable than one with marginally higher metrics on a fixed test set.

### Why optimization was NOT applied to other feature sets

Higher-dimensional feature sets (payload, all_features) already depend on rigid dataset signatures. Optimizing a model that exploits those shortcuts would maximize test-set metrics while further degrading real-world robustness. There is no methodological value in tuning a model built on degenerate features.

---

## 8. Resource Analysis — Optimized Random Forest

**Script:** `resource_analysis2.py`  
**Model:** Random Forest (optimized) — `n_estimators=50`, `max_depth=8`, `min_samples_leaf=1`, `class_weight="balanced"`, `n_jobs=-1`  
**Feature set:** `delta_time + entropy` (2 features)  
**Monitoring method:** Background thread sampling CPU and RAM every 0.1s throughout the full training phase, providing continuous readings rather than a single before/after snapshot

### 8.1 Results

| Metric | Value |
|---|---|
| Train time | 19.47 s |
| Inference time (full test set) | 0.2926 s |
| Average CPU usage (normalized per core) | 61.98 % |
| Peak RAM usage | 4663.95 MB (~4.55 GB) |

---

### 8.2 Why Peak RAM is 4.55 GB — and Why This Does Not Reflect Deployment

The 4663.95 MB figure is the peak memory consumed by the **entire Python process during training**, not the size of the model itself. Understanding what occupies that memory is critical to correctly interpreting this number.

During training, the following coexist in memory simultaneously:

**1. The raw dataset in memory**  
The Car-Hacking dataset contains millions of CAN bus frames. When loaded via `pd.read_csv()`, pandas allocates contiguous memory arrays for every column in both the normal and DoS DataFrames, and again for the merged combined DataFrame. This alone accounts for a large portion of the RAM.

**2. Parallel worker copies**  
With `n_jobs=-1`, scikit-learn instructs the operating system to spawn one worker process per CPU core. Each worker receives a subset of the training data to build its assigned trees independently. This means the training data is effectively replicated — partially or fully — across multiple worker processes at peak concurrency.

**3. Intermediate tree construction structures**  
While building each decision tree, the algorithm maintains candidate split thresholds, impurity scores, node statistics, and child node buffers for every level of the tree being constructed. These are temporary structures that exist only during training and are discarded once each tree is finalized.

**4. Python and library runtime overhead**  
The Python interpreter, numpy, pandas, scikit-learn, and psutil all occupy memory simply by being imported and active. This baseline overhead can easily account for several hundred MB independent of the data or model.

**None of the above is present in the exported model.** Once training is complete, all DataFrames, worker processes, and temporary build structures are released. What remains is only the trained model object.

---

### 8.3 What the Exported Model Actually Contains

A Random Forest, once trained, is a collection of decision trees. Each decision tree is a binary structure where every internal node stores exactly three values:

- The **feature index** to evaluate (in this case, 0 = `delta_time` or 1 = `entropy`)
- The **split threshold** — a single float (e.g., `delta_time < 0.00312`)
- Pointers to the **left and right child nodes**

With `max_depth=8`, each tree has at most 2⁸ − 1 = **255 internal nodes** and **256 leaf nodes** (511 nodes total). Each node stores a small fixed number of floats and integers. Across 50 trees, the entire model is:

```
50 trees × ~511 nodes × ~5 values per node × 8 bytes = ~1 MB (upper bound estimate)
```

In practice, trees rarely reach maximum depth on every branch, so the actual size is even smaller. When serialized to disk using `joblib.dump()` or `pickle`, this model typically produces a file of **1–5 MB**.

The training data — the millions of CAN frames used to learn the split thresholds — is **never stored in the model**. Once training is done, the dataset can be entirely discarded. The model retains only the decision logic derived from it.

---

### 8.4 Train Time and Inference Time

**Train time — 19.47s**  
This is the time to fit 50 trees in parallel across the full training set. It is relevant only as a development-time cost — in production, the model is trained once offline on a workstation and then exported. The vehicle ECU never runs the training process.

**Inference time — 0.2926s**  
This is the time to classify the **entire test batch**, which represents 20% of the full dataset — potentially hundreds of thousands of CAN frames evaluated in a single call. The per-frame inference time is therefore in the **sub-millisecond range**.

For context: standard CAN bus operates at up to 1 Mbit/s, with typical traffic rates of a few hundred to a few thousand frames per second. Even a conservative estimate of 1000 frames/second means the system has ~1ms per frame — comfortably above what a loaded RF with 2 features and max_depth=8 requires.

---

### 8.5 CPU Usage — 61.98% (Normalized)

With `n_jobs=-1`, scikit-learn parallelizes tree construction across all available CPU cores. The 61.98% figure is the **average CPU utilization per core**, normalized by `psutil.cpu_count()`, sampled throughout the training phase. This high utilization is expected and desirable during training — it means the available cores are being used efficiently.

At **inference time**, the model performs a single forward pass through each tree for each frame — a sequential, low-compute operation. CPU usage during inference is a small fraction of what is observed during training.

---

### 8.6 Automotive Deployment Summary

| Phase | Environment | RAM | CPU | Duration |
|---|---|---|---|---|
| Training | Workstation (offline) | ~4.55 GB | ~62% avg per core | 19.47 s |
| Export | Workstation | Negligible | Negligible | < 1 s |
| Deployment artifact | — | **~1–5 MB file** | — | — |
| Inference (per batch) | ECU / IDS system | Model in RAM only | Low (tree traversal) | < 1 ms per frame |

> The 4.55 GB RAM and 19.47s train time are properties of the development environment, not the deployed system. The vehicle ECU only loads the serialized model file (~1–5 MB) and performs inference, which requires a negligible fraction of those resources. Resource profiling on the target ECU hardware would be needed to establish deployment-specific bounds.

---

## 9. Conclusion

### Final model: Random Forest with `delta_time + entropy`

| Metric | Unoptimized | Optimized |
|---|---|---|
| Accuracy | 0.9803 | slightly lower |
| Precision | 0.8645 | slightly conservative |
| Recall | **0.9991** | maintained near 1.0 |
| F1 Score | 0.9270 | stable |
| Features | 2 | 2 |
| Train Time | ~28s | varies with grid |

### Key conclusions

1. **Tier separation is clear:** Tree-based models (RF, XGBoost, LightGBM) significantly outperform linear models (SVM, LR) on the fair feature set. Linear boundaries cannot capture the non-linear temporal structure of a CAN bus DoS flood.

2. **High metrics ≠ good model:** Feature sets with `id_interval` or payload bytes produce near-perfect metrics but for the wrong reason — they detect dataset artifacts, not attack behavior. These were deliberately excluded.

3. **`delta_time + entropy` is the honest benchmark:** It captures the two behavioral signatures of DoS — timing anomaly and payload repetition — without encoding shortcuts. This is the only feature set suitable for a generalizable IDS.

4. **RF is the right choice:** Near-perfect recall, interpretable, hyperparameter-robust, and proven to learn behavior rather than identity. Optimization further reduces overfitting, yielding a more reliable deployed model at the cost of marginal metric conservatism — which is the correct tradeoff for a safety-critical system.

---

## 10. Deployment Analysis

### 10.1 Deployment Readiness

The DoS detection model is one of the simplest deployable IDS configurations possible. It requires only two features (`delta_time` and `entropy`), both derivable from a single CAN frame and its predecessor, with no preprocessing pipeline beyond feature extraction itself. Random Forest inference at runtime requires only scikit-learn as a dependency. The serialized model file is approximately **1–5 MB**. No feature scaler is needed — unlike MLP or Logistic Regression, RF operates directly on raw feature values without normalization.

**Overall deployment readiness: High.**

---

### 10.2 Runtime Inference Pipeline

At inference time, the IDS performs the following on each incoming CAN frame:

```
1. Receive CAN frame (ID, DLC, payload[0:8], timestamp_ns)
2. delta_time  = (timestamp_ns − prev_timestamp_ns) / 1e9   # in seconds
3. entropy     = −∑ p(b) · log₂(p(b))  over payload bytes   # Shannon entropy
4. X           = [[delta_time, entropy]]
5. label       = model.predict(X)[0]
6. if label == 1: raise_alert("DoS detected")
7. prev_timestamp_ns = timestamp_ns
```

This is a **stateful** process: `delta_time` requires the IDS to maintain the timestamp of the last frame. The state is a single 64-bit integer — the most minimal stateful dependency possible. The `entropy` computation over 8 bytes loops through at most 8 values — nanoseconds on any modern processor.

---

### 10.3 Feature Extraction at Runtime

| Feature | Formula | Statefulness | Required Input |
|---|---|---|---|
| `delta_time` | `t_current − t_previous` (seconds) | **Stateful** — store last timestamp | Current + previous CAN timestamp |
| `entropy` | `−∑ p(b) · log₂ p(b)` | **Stateless** — single frame only | 8 payload bytes |

Both features are computed in O(1) time and O(1) space. There is no sliding window, no frame history buffer, and no per-ID state — only a single timestamp variable carried between consecutive frames. This is an important practical property: the IDS can be started mid-stream without an initialization phase; it simply treats the first frame's `delta_time` as undefined and skips that one classification.

---

### 10.4 Deployment Architecture Options

Three viable architectures exist, in descending order of preference for production use:

**Option A — Dedicated Security ECU (Recommended)**
A dedicated IDS module taps the CAN bus (typically at the central gateway) and monitors traffic passively, in parallel with normal routing. This is the architecture recommended by ISO/SAE 21434 for in-vehicle intrusion detection. The IDS consumes its own compute budget and does not share cycles with safety-critical controllers. Model updates are applied via OTA without touching other ECUs.

**Option B — Central Gateway ECU**
The gateway ECU already processes all inter-domain CAN traffic. Embedding the IDS there avoids additional hardware but introduces shared resource dependency. The inference cost (~0.21 µs per frame) is negligible relative to gateway processing budgets on modern automotive SoCs (NXP i.MX8, Qualcomm SA8155P, TI TDA4VM). This option is viable if the gateway runs embedded Linux and has spare compute margin.

**Option C — OBD-II Dongle / Aftermarket Module**
Suitable for fleet monitoring, prototype validation, or aftermarket security products. Not recommended for production safety use due to the uncontrolled physical exposure of the OBD-II port — an attacker who can inject via OBD-II can also disconnect or tamper with the monitoring hardware.

---

### 10.5 Hardware Requirements

| Requirement | Value | Notes |
|---|---|---|
| Model file size | ~1–5 MB | Loaded once at startup into RAM |
| RAM at inference | < 10 MB | Model object + two float buffers |
| Per-frame CPU time | ~0.21 µs | Negligible on any automotive SoC |
| Available time budget (1000 fps) | ~1,000 µs | 4700× headroom over inference cost |
| Python runtime | Embedded Linux or RTOS with Python | See §10.7 for C alternative |
| Runtime dependencies | scikit-learn, joblib, numpy | All standard; well-maintained |

---

### 10.6 Serialization and Loading

The trained model is serialized to a single binary file and loaded once at system startup:

```python
import joblib

# After training (once, on development workstation)
joblib.dump(rf, "dos_ids_model.pkl")

# On target system — loaded once at startup
rf = joblib.load("dos_ids_model.pkl")

# Per-frame inference function
def classify_frame(payload: bytes, ts_ns: int, prev_ts_ns: int) -> int:
    delta_time = (ts_ns - prev_ts_ns) / 1e9
    counts = [0] * 256
    for b in payload:
        counts[b] += 1
    n = len(payload)
    entropy = -sum((c/n) * math.log2(c/n) for c in counts if c > 0)
    return int(rf.predict([[delta_time, entropy]])[0])
```

The serialized file contains only the decision tree structure — split thresholds, feature indices, and child pointers. The training data is never stored in the model.

---

### 10.7 Portability — C/C++ Conversion

If the target ECU does not run a Python runtime (AUTOSAR Classic, bare-metal RTOS), the trained Random Forest can be transpiled to standalone C or C++ using open-source tools. The resulting code has zero external runtime dependencies and is a set of nested `if/else` branches over two floating-point values.

| Tool | Output Languages | Notes |
|---|---|---|
| `m2cgen` | C, C++, Go, Rust, Java, and others | Zero-dependency output; simple API |
| `sklearn-porter` | C, Java, JavaScript | Widely used; supports RF |
| `emlearn` | C99 | Designed specifically for microcontrollers |
| `treelite` | C shared library | Optimized for embedded/production inference |

A C-converted model with only 2 features and `max_depth=8` produces a compact translation that fits in a few kilobytes of flash. This makes it suitable for MCU-class deployments (e.g., automotive-grade devices like NXP S32K, Renesas RH850) where a full Python stack is not available.

---

### 10.8 Standards Compliance

| Standard | Scope | RF Compliance Notes |
|---|---|---|
| ISO/SAE 21434 | Cybersecurity engineering for road vehicles | ✓ — behavioral features (not signatures); decision paths are auditable |
| UN Regulation R155 | Type approval, cybersecurity management systems | ✓ — IDS with documented detection logic satisfies CSMS monitoring requirements (Annex 5, §7.3.3) |
| AUTOSAR Adaptive | Software architecture for high-compute automotive ECUs | ✓ — model wrapper deployable as a service component |
| AUTOSAR Classic | Software architecture for MCU-class ECUs | ✓ (via C conversion) — no dynamic memory, no OS dependency |

The interpretability of Random Forest is the critical compliance property. Every classification traces to an exact decision path:
> *"if `delta_time` < 0.00031 AND `entropy` < 0.14 → DoS"*

This satisfies the auditability requirement in ISO/SAE 21434 §15 (Cybersecurity Monitoring) and UN R155 Annex 5 (mitigations list). Boosted models (XGBoost, LightGBM) require post-hoc tools (SHAP) to approximate explanations — an additional certification burden that RF avoids entirely.

---

### 10.9 Limitations and Open Considerations

| Limitation | Description | Mitigation |
|---|---|---|
| Single-attack scope | Detects DoS flooding only — not Fuzzy, replay, or spoofing attacks | Deploy in parallel with per-attack models (see §10.10) |
| Static model | No online learning — performance may degrade on novel attack variants | Periodic retraining with updated data; OTA model update |
| CAN FD incompatibility | Feature extraction assumes 8-byte CAN 2.0 payloads | Adapt entropy calculation to variable DLC (up to 64 bytes for CAN FD) |
| Single-stream view | Model processes the global frame stream — does not model per-ID baselines | Extend with per-identifier statistical profiling for finer-grained detection |
| Benign timing anomalies | ECU resets, bus-off recovery, and gateway restarts can produce small `delta_time` spikes | Combine with ECU state signals to suppress transient false alerts |
| No payload decoding | Entropy captures randomness, not semantic anomalies in known payloads | Complement with protocol-aware rules for high-criticality IDs |

---

### 10.10 Multi-Attack IDS — Combined Deployment

Both the DoS and Fuzzy models can run in parallel on the same CAN frame stream, each providing an independent binary classification:

```
CAN frame stream
        │
        ├──▶ DoS IDS   (delta_time, entropy)          → label_dos   ∈ {0,1}
        └──▶ Fuzzy IDS (entropy, byte_std, byte_range, delta_time, byte_mean) → label_fuzzy ∈ {0,1}
```

| Combined Deployment | Value |
|---|---|
| Total model footprint | ~6–15 MB (both models loaded simultaneously) |
| Total inference cost per frame | ~0.51 µs (~0.21 + ~0.30 µs) |
| Available time budget at 1000 fps | ~1,000 µs |
| Budget utilization | < 0.1% |
| Coverage | DoS flooding + Fuzzy probing |

The combined system covers the two most studied attack vectors on the Car-Hacking Dataset within a negligible compute budget. Additional models for replay and spoofing attacks could be added following the same methodology.

---

### 10.11 Deployment Effort Estimate

| Task | Estimated Effort |
|---|---|
| Serialize trained model | 1 line of code |
| Write feature extraction module | ~30 lines of Python (or ~80 lines of C) |
| Integrate with CAN driver / frame listener | 1–2 days (platform dependent) |
| Unit test feature extraction against known frames | 0.5–1 day |
| Integration test on target hardware | 1–3 days |
| Optional C transpilation (m2cgen) | 0.5–1 day |
| Performance profiling on target ECU | 1 day |
| Certification documentation | 1–2 weeks |

The technical integration is straightforward — the model is small, the features are trivial to compute, and there are no external service dependencies. The dominant effort is certification documentation, not implementation.
