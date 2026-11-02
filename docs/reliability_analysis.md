# Explainability, calibration, and conformal analysis

## Scope and safeguards

This development stage analyzes the three configurations selected from the existing 81 validation comparisons:

| Task | Model | Scenario | Features | Previously reported validation macro-F1 |
|---|---|---|---:|---:|
| Five-class | Random Forest | S7_all_history | 67 | 0.3698 |
| Broad three-class | LightGBM | S7_all_history | 67 | 0.5753 |
| Trauma subtype | LightGBM | S6_medications | 18 | 0.4684 |

`run_reliability_stage.py` checks the stored scenario summary, per-class F1 results, and feature manifest before executing. The class order is explicitly fixed in the script. Existing model hyperparameters are reused. The independent test partition is excluded from fitting, calibration, explanation, and evaluation. No test score is reported.

The validation partition was already used to compare 81 configurations and select these models. Consequently, SHAP explanations and probability/conformal metrics on validation are descriptive results for selected models, not independent performance estimates or formal empirical coverage evidence.

History, diagnosis, substance-use, lifestyle, and medication fields still require source and timing verification. Explainability does not establish that a feature is safe, available at prediction time, or causally related to the outcome.

## Stage A: SHAP

Each selected estimator is fit on the full existing training partition only. Median imputation is fit on training data and then applied to validation; the feature order comes directly from `scenario_feature_manifest.csv`. TreeSHAP is applied to the fitted Random Forest and LightGBM estimators. Output-specific values are kept separate for multiclass explanations.

The stage generates global mean-absolute-importance charts, class-specific SHAP beeswarms, and dependence plots for the three top global features in each task. Global and class-specific numerical rankings are aggregate CSVs. The report describes model-output attribution and explicitly avoids causal or feature-safety claims.

Current global rankings highlight `med_na`, `n_meds`, and `age_years` for the five-class model; `age_years`, `med_na`, and `su_na` for broad three-class; and `age_years`, `med_na`, and `n_meds` for trauma subtype. For the two history-heavy tasks, diagnosis/history/substance/lifestyle/medication features account for 85.9% and 74.4% of aggregated absolute SHAP importance, respectively. Their source and timing uncertainty therefore remains material.

## Training-only probability and conformal subsets

To keep probability calibration and conformal scores disjoint, the 602 training records are divided once using stratification on the original five-class label. Records are sorted by decedent ID before seeded splitting; the current cohort has one record per decedent, and the script checks uniqueness, non-overlap, complete allocation, and separation from validation. IDs are used only locally and are not written to output files.

| Training subset | N | blunt_trauma | firearm | stab | cardiac | intox |
|---|---:|---:|---:|---:|---:|---:|
| Model fitting | 361 | 63 | 26 | 33 | 85 | 154 |
| Probability calibration | 120 | 21 | 9 | 11 | 28 | 51 |
| Conformal calibration | 121 | 21 | 9 | 11 | 29 | 51 |

The first split uses random state 42 and reserves 40% of training records; the second splits that remainder evenly using random state 43. The trauma-subtype task contains 122 model-fit, 41 probability-calibration, and 41 conformal-calibration records. Its two 41-record subsets contain 21 blunt-trauma, 9 firearm, and 11 stab cases each. Full task-specific counts are in `protocol/calibration/results/training_subset_class_counts.csv`.

The original validation partition remains the explanation and descriptive evaluation set. It is not used to fit either the temperature scaler or conformal quantiles.

## Stage B: temperature scaling

The calibration model is fitted only on the model-fitting subset. Its median imputer is also fitted only on that subset. For probabilities `p`, the scalar temperature `T` is optimized for calibration-subset negative log-likelihood and applied as `softmax(log(p) / T)`. Class ordering is checked against the configured model order.

Multiclass Brier score is the mean over records of the sum of squared differences between the class-probability vector and the one-hot label. Log loss uses the fixed class order. Ten-bin expected calibration error (ECE) uses equal-width bins of maximum predicted class probability and top-label accuracy. Reliability diagrams show one-vs-rest class curves.

| Task | T | Brier before → after | Log loss before → after | ECE before → after |
|---|---:|---:|---:|---:|
| Five-class | 0.960 | 0.679 → 0.678 | 1.344 → 1.341 | 0.119 → 0.102 |
| Broad three-class | 1.566 | 0.510 → 0.507 | 0.869 → 0.854 | 0.057 → 0.073 |
| Trauma subtype | 7.684 | 0.682 → 0.655 | 1.129 → 1.081 | 0.173 → 0.070 |

Metrics are calculated on the reused validation set. Temperature scaling improved Brier score and log loss for these validation predictions, while ECE increased slightly for broad three-class. The trauma-subtype scaler is estimated from only 41 records and is not reliable evidence of stable calibration.

Aggregate mean probabilities, grouped by task, true class, and probability class, are saved in `protocol/calibration/results/validation_probability_aggregates.csv`. Per-record probabilities are not exported.

## Stage C: split conformal prediction

The model and temperature scaler are fixed without using the conformal-calibration subset. Nonconformity is `1 - calibrated_probability(true class)`. For target coverage `1 - alpha` and `n` conformal scores, the threshold is the one-indexed `ceil((n + 1) * (1 - alpha))` order statistic. If this rank exceeds `n`, the threshold is infinite. A prediction set contains every class satisfying `1 - calibrated_probability <= threshold`.

| Task | Target | Validation coverage | Mean set size | Singleton | Abstention |
|---|---:|---:|---:|---:|---:|
| Five-class | 90% | 85.3% | 3.31 | 0.0% | 100.0% |
| Five-class | 80% | 80.6% | 2.72 | 1.6% | 98.4% |
| Broad three-class | 90% | 91.5% | 2.12 | 10.9% | 89.1% |
| Broad three-class | 80% | 79.8% | 1.67 | 34.9% | 65.1% |
| Trauma subtype | 90% | 95.5% | 2.77 | 0.0% | 100.0% |
| Trauma subtype | 80% | 90.9% | 2.52 | 4.5% | 95.5% |

These are validation-set empirical values, not independent coverage claims. Standard split conformal prediction targets marginal coverage for future exchangeable observations when the model, scaler, and conformal calibration procedure are fixed; it does not guarantee coverage for each class or individual. Per-class counts are small, particularly the 9 firearm observations in each trauma calibration subset. Abstention means a prediction set is not a singleton; empty and multi-class sets are reported separately. `prediction_set_composition_counts.csv` records the aggregate count for each unique class combination without recording individual examples.

Only aggregate coverage, set-size counts, class summaries, and plots are exported. No person-level NMDID IDs, per-record probabilities, or per-record prediction sets are saved.

## Reproduction

Use the existing project environment and run the stages in order from the repository root:

```powershell
.\.venv\Scripts\python.exe run_reliability_stage.py --stage shap
.\.venv\Scripts\python.exe run_reliability_stage.py --stage calibration
.\.venv\Scripts\python.exe run_reliability_stage.py --stage conformal
.\.venv\Scripts\python.exe run_reliability_stage.py --stage summary
```

Dependencies are listed in `requirements.txt`. Output locations:

- SHAP figures and report: `protocol/shap/figures/`, `protocol/shap/shap_report.md`
- SHAP rankings: `protocol/shap/shap_importance/`
- Calibration tables and reliability diagrams: `protocol/calibration/results/`, `protocol/calibration/figures/`
- Conformal tables, method note, and prediction-set plot: `protocol/conformal/results/`, `protocol/conformal/methodology.md`, `protocol/conformal/figures/`
- Supervisor summary: `protocol/supervisor_demo/`

Outputs are written to new protocol subdirectories so earlier experiments are not overwritten.
