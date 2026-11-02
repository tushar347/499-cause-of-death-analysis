# Experiments

## Metadata-only model comparison

The prepared cohort contains 860 decedents in five imbalanced classes. Tier 1 uses five basic numerical features; Tier 1+2 uses all 70 numerical features.

Evaluation uses repeated stratified five-fold cross-validation with two repeats, seed 42, and macro-F1. Chance-level macro-F1 is about 0.20.

| Model | Tier 1 macro-F1 | Tier 1+2 macro-F1 |
|---|---:|---:|
| TabPFN | 0.2193 | 0.2868 |
| Logistic Regression | 0.2446 | 0.3156 |
| Random Forest | 0.2436 | 0.3399 |

The values are means across the 10 folds. They are preliminary cross-validation results, not held-out test performance. In particular, Tier 1+2 results require feature-source and timing verification.

## Selected leakage-scenario models

The frozen protocol uses 602 training, 129 validation, and 129 test records. The 81 scenario comparisons used validation for model selection; the independent test partition remains untouched. The selected validation configurations are:

| Task | Model | Scenario | Validation macro-F1 |
|---|---|---|---:|
| Five-class | Random Forest | S7_all_history | 0.3698 |
| Broad three-class | LightGBM | S7_all_history | 0.5753 |
| Trauma subtype | LightGBM | S6_medications | 0.4684 |

These configurations then received exploratory SHAP, probability-calibration, and split-conformal analyses. Their results and reproduction commands are documented in [reliability analysis](reliability_analysis.md). Validation metrics from that follow-up are descriptive because the same validation partition was used in the original configuration selection. The history, substance-use, lifestyle, and medication predictors remain subject to provenance and timing verification.

## Age analysis

The age-and-sex-only baseline reports macro-F1 of 0.2235 for Logistic Regression and 0.2341 for Random Forest. In the age ablation, removing age from the Tier 1+2 feature set lowers the Random Forest mean macro-F1 from 0.3399 to 0.3364.

Cardiac cases have a mean age of 51.7 years, compared with about 43 years across the other groups. Intoxication and cardiac cases are predicted better than firearm, stab, and blunt trauma. Blunt trauma is often confused with stab.

## Interpretation and caveats

Class imbalance, possible leakage from history/substance/medication features, the untouched test partition, and pending PMCT access limit interpretation. The repeated-cross-validation best five-class result remains approximately 0.34 macro-F1; the higher scores above are different selected tasks/scenarios measured on the previously used validation set, not final test performance. No image-only or multimodal results are available.

Aggregate outputs are stored in `results/`; figures are stored in `reports/figures/`. CSVs in this repository summarize folds, classes, features, or experiment settings; they do not contain one row per decedent.
