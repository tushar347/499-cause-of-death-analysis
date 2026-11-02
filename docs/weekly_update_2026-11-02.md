# Weekly Project Progress Update

**Project:** Multimodal AI for Cause-of-Death Estimation from Postmortem Data
**Course:** CSE 499A
**Group:** 2
**Supervisor:** Dr. Sumaiya Tabassum Nimi
**Date:** 2 November 2026

## 1. Summary

This week, we extended our metadata-only experiments and worked on making our model evaluation more reliable and explainable.

We completed label mapping, preliminary leakage auditing, fixed data splitting, exploratory data analysis, statistical association tests, and 81 model experiments.

We also implemented SHAP explainability, probability calibration, and conformal prediction.

The independent test set remains untouched. PMCT preprocessing and multimodal fusion are still planned.

## 2. Label Mapping and Dataset Splitting

We organized the five original cause-of-death classes into three broad categories:

| Broad Category | Records |
|---|---:|
| Trauma | 292 |
| Cardiac | 202 |
| Intoxication | 366 |
| Total | 860 |

Trauma was further divided into blunt trauma, firearm, and stab wounds.

We created fixed decedent-level data partitions:

| Partition | Records |
|---|---:|
| Training | 602 |
| Validation | 129 |
| Testing | 129 |
| Total | 860 |

The test partition has not been evaluated.

## 3. Preliminary Leakage Audit

We initially categorized the dataset columns as follows:

- 2 candidate allowed features.
- 68 gray features requiring further review.
- 2 forbidden columns: ID and target label.

The source and recording time of medical history, substance use, lifestyle, and medication features still need verification.

This is a preliminary audit, not a final leakage-free feature selection.

## 4. EDA and Statistical Analysis

We performed exploratory data analysis using the 602 training records and all 70 numerical features.

The work included:
- Feature distribution and missing-value analysis.
- Fisher's exact tests for binary features.
- Mann-Whitney U tests for continuous features.
- Benjamini-Hochberg FDR correction.
- Age and sex adjusted association analysis.
- Odds-ratio and feature-correlation heatmaps.

Main results:

| Analysis | Result |
|---|---:|
| Unadjusted tests | 350 |
| Unadjusted FDR-significant | 34 |
| Adjusted models attempted | 350 |
| Adjusted models estimated | 244 |
| Adjusted FDR-significant | 25 |

These results show statistical associations, not causal relationships.

## 5. Leakage-Scenario Model Experiments

We tested nine feature scenarios using three models:

- Logistic Regression
- Random Forest
- LightGBM

We evaluated three classification tasks:
1. Five-class classification.
2. Broad three-class classification.
3. Trauma subtype classification.

This produced 81 validation experiments.

Best validation results:

| Task | Model | Scenario | Macro-F1 |
|---|---|---|---:|
| Five-class | Random Forest | S7 | 0.3698 |
| Broad three-class | LightGBM | S7 | 0.5753 |
| Trauma subtype | LightGBM | S6 | 0.4684 |

These are validation results, not final test results.

## 6. SHAP Explainability

We used SHAP to understand which features influenced the model predictions.

Important findings:

| Task | Important Features | History-related Importance |
|---|---|---:|
| Five-class | med_na, age_years, n_meds | 85.9% |
| Broad three-class | age_years, med_na, su_na | 74.4% |
| Trauma subtype | age_years, med_na, n_meds | 23.5% |

These values describe model reliance. They do not establish causality or feature safety.

## 7. Probability Calibration

We implemented temperature scaling to improve the reliability of predicted probabilities.

The results showed:
- Brier score improved for all three tasks.
- Log loss improved for all three tasks.
- ECE improved for five-class and trauma subtype.
- ECE worsened for broad three-class classification.

## 8. Conformal Prediction

We implemented split conformal prediction to allow models to return sets of possible classes when predictions are uncertain.

Validation coverage:

| Task | 90% Target | 80% Target |
|---|---:|---:|
| Five-class | 85.3% | 80.6% |
| Broad three-class | 91.5% | 79.8% |
| Trauma subtype | 95.5% | 90.9% |

Prediction sets were often multi-class.

These results are descriptive because the validation set was already used for model selection. Conformal coverage is marginal under exchangeability assumptions, not a guarantee for every class.

## 9. Project Structure

The project is organized into the following main folders:

```text
project/
|
|-- data/
|-- scripts/
|-- docs/
|-- results/
|-- reports/
|   |-- figures/
|
|-- protocol/
|   |-- eda/
|   |-- leakage_scenarios/
|   |-- shap/
|   |-- calibration/
|   |-- conformal/
|   |-- supervisor_demo/
|
|-- requirements.txt
|-- README.md
```

The scripts contain the experiment code, docs contain project documentation, and protocol stores the experimental outputs and analysis.

Raw NMDID records and local decedent identifiers are not included in public repository outputs.

## 10. Existing Project Pipeline

### Metadata Branch

NMDID Metadata
→ Feature Preparation
→ Leakage Audit
→ Feature Scenarios
→ Machine Learning Models
→ SHAP Explainability
→ Calibration
→ Conformal Prediction
→ Evaluation

### Imaging Branch (Planned)

NMDID PMCT Images
→ Image Preprocessing
→ Merlin Foundation Model
→ Image Embeddings

### Future Fusion

Reviewed Metadata + PMCT Embeddings
→ Multimodal Fusion
→ Cause-of-Death Prediction
→ Final Evaluation

The metadata branch has been implemented. The PMCT and multimodal branches have not started.

## 11. Limitations

- The five classes are imbalanced.
- Feature provenance and timing are not fully verified.
- Some models rely heavily on history-related features.
- Calibration subsets are relatively small, particularly for trauma subtype.
- Validation data was used during model selection.
- The independent test set remains untouched.
- No PMCT or multimodal results are available.

## 12. Next Steps

1. Complete the feature provenance audit.
2. Finalize the allowed, gray, and forbidden feature groups.
3. Lock the final experimental protocol.
4. Perform the prespecified independent test evaluation.
5. Obtain approved PMCT images.
6. Develop the PMCT preprocessing workflow.
7. Extract image embeddings using Merlin.
8. Compare metadata-only, PMCT-only, and multimodal models.

## 13. Conclusion

This week, we extended our project from basic metadata classification to statistical analysis, feature-scenario comparisons, explainability, and prediction reliability.

The next major objective is to verify the metadata features and prepare the PMCT image branch before combining both data sources.
