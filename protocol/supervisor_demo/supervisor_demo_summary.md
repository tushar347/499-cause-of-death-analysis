# Supervisor demonstration: metadata reliability stage

## Scope and status

This exploratory development stage used the frozen training and validation partitions only. The independent 129-record test partition was not evaluated. The selected configurations were chosen from 81 prior comparisons on validation, so all validation metrics below are descriptive, not independent estimates of generalization or conformal coverage.

The models are metadata-only: five-class Random Forest/S7_all_history, broad three-class LightGBM/S7_all_history, and trauma-subtype LightGBM/S6_medications. Their prior validation macro-F1 values were 0.3698, 0.5753, and 0.4684, respectively. All history, substance, lifestyle, and medication fields remain subject to source/timing verification.

## SHAP summary

| Task | Model | Top global features | Unverified-group importance |
| --- | --- | --- | --- |
| five_class | Random Forest | med_na, age_years, n_meds, dx_hypertension, env_drinking status:high risk | 85.9% |
| broad_three_class | LightGBM | age_years, med_na, su_na, env_drinking status:high risk, dx_hypertension | 74.4% |
| trauma_subtype | LightGBM | age_years, med_na, n_meds, female, med_d7804.0 | 23.5% |

SHAP describes model reliance, not causal effects or feature safety. Class-output rankings and beeswarm/dependence plots are in `../shap/`; `med_na` and `n_meds` require particular interpretation against the source data dictionary.

### Strongest class-specific features

| Task | Class output | Top three SHAP features |
| --- | --- | --- |
| five_class | blunt_trauma | med_na, n_meds, age_years |
| five_class | firearm | age_years, env_drinking status:drinker, cur, su_na |
| five_class | stab | med_na, n_meds, env_drinking status:high risk |
| five_class | cardiac | age_years, med_na, dx_hypertension |
| five_class | intox | med_na, age_years, n_meds |
| broad_three_class | trauma | med_na, su_na, age_years |
| broad_three_class | cardiac | age_years, dx_hypertension, env_drinking status:drinker, cur |
| broad_three_class | intoxication | age_years, env_drinking status:high risk, n_meds |
| trauma_subtype | blunt_trauma | age_years, n_meds, female |
| trauma_subtype | firearm | age_years, n_meds, med_na |
| trauma_subtype | stab | age_years, med_na, female |

## Training-only calibration split

The original training records were stratified by original five-class label after sorting by decedent ID. The fixed seed is 42 (the second split uses 43). Each ID occurs once in this cohort; the implementation checks uniqueness, no overlap, complete partitioning, and separation from validation. Only aggregate counts are saved.

| Subset | N | Original class counts |
| --- | --- | --- |
| model_fit | 361 | blunt_trauma=63, firearm=26, stab=33, cardiac=85, intox=154 |
| probability_calibration | 120 | blunt_trauma=21, firearm=9, stab=11, cardiac=28, intox=51 |
| conformal_calibration | 121 | blunt_trauma=21, firearm=9, stab=11, cardiac=29, intox=51 |

For trauma subtype, the same grouped split yields 122 model-fit, 41 probability-calibration, and 41 conformal-calibration records (class counts are in `../calibration/results/training_subset_class_counts.csv`).

## Probability calibration

Temperature was fitted by minimizing multiclass negative log-likelihood on the probability-calibration subset only. Brier score is the multiclass sum-of-squared-probability error averaged over records; ECE uses 10 equal-width bins and top-label confidence. Validation metrics are summarized below.

| Task | Temperature | Brier before → after | Log loss before → after | ECE before → after |
| --- | --- | --- | --- | --- |
| five_class | 0.960 | 0.679 → 0.678 | 1.344 → 1.341 | 0.119 → 0.102 |
| broad_three_class | 1.566 | 0.510 → 0.507 | 0.869 → 0.854 | 0.057 → 0.073 |
| trauma_subtype | 7.684 | 0.682 → 0.655 | 1.129 → 1.081 | 0.173 → 0.070 |

Temperature scaling improved validation Brier score and log loss in all three tasks. ECE decreased for five-class and trauma subtype but increased for broad three-class; calibration is therefore not uniformly improved across metrics. The small trauma-subtype calibration subset (41 records) makes its temperature estimate especially uncertain.

## Split-conformal prediction sets

Thresholds use calibrated probabilities on the separate training-derived conformal subset and the finite-sample order statistic. Coverage and set sizes below are empirical on the reused validation partition and are descriptive only.

| Task | Target | Validation coverage | Mean set size | Singleton | Abstention |
| --- | --- | --- | --- | --- | --- |
| five_class | 90% | 85.3% | 3.31 | 0.0% | 100.0% |
| five_class | 80% | 80.6% | 2.72 | 1.6% | 98.4% |
| broad_three_class | 90% | 91.5% | 2.12 | 10.9% | 89.1% |
| broad_three_class | 80% | 79.8% | 1.67 | 34.9% | 65.1% |
| trauma_subtype | 90% | 95.5% | 2.77 | 0.0% | 100.0% |
| trauma_subtype | 80% | 90.9% | 2.52 | 4.5% | 95.5% |

Standard split conformal prediction targets marginal coverage under exchangeability assumptions; it does not ensure per-class coverage. Per-class results, set-size counts, calibration thresholds, and figures are in `../conformal/`. The large set sizes and low singleton rates show that these models often cannot support a single-class output at the selected nominal coverages.

## Limitations and next steps

- Do not treat validation metrics as independent confirmation: these models were selected using the same validation partition.
- The test partition remains untouched. Freeze this protocol and perform one prespecified test evaluation only after provenance review and supervisory approval.
- The training-derived temperature and conformal samples are modest; the trauma task has only 9 firearm cases in each calibration subset. ECE and class-conditional estimates are particularly noisy.
- Conformal marginal coverage depends on exchangeability and does not guarantee class-conditional or individual-record coverage.
- SHAP importance does not establish safety, timing, or causation. Verify source/provenance of medication, history, substance-use, and lifestyle variables before any clinical/forensic interpretation.
- Next research stage: complete feature provenance review, preserve this untouched test set for a single locked evaluation, then compare metadata with Merlin PMCT embeddings and prespecified multimodal fusion.

## Reproduction

From the repository root, activate `.venv` and run each stage in order:

```powershell
.\.venv\Scripts\python.exe run_reliability_stage.py --stage shap
.\.venv\Scripts\python.exe run_reliability_stage.py --stage calibration
.\.venv\Scripts\python.exe run_reliability_stage.py --stage conformal
.\.venv\Scripts\python.exe run_reliability_stage.py --stage summary
```

The script verifies the selected configurations against the saved scenario summary, per-class metrics, and feature manifest. Only aggregate outputs are written; no decedent-level identifiers, probabilities, or prediction sets are exported.
