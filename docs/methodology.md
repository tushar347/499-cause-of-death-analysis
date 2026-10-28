# Methodology

## Prediction task

The current task is five-class cause-of-death estimation from a prepared NMDID metadata cohort of 860 decedents: intoxication (366), cardiac (202), blunt trauma (149), stab (79), and firearm (64).

## Feature tiers

- **Tier 1:** five basic numerical features: age, sex indicator, cadaver length, cadaver weight, and a BMI-like measurement.
- **Tier 1+2:** all 70 numerical predictors, organized as basic metadata (5), diagnoses (22), medical history (5), substance use (10), lifestyle (12), and medications (16).

Before any feature is used as a predictor, its definition, source, and timing must be checked against the NMDID data dictionary and the [leakage policy](leakage_policy.md). The history-based groups remain provisional pending provenance review.

## Models and evaluation

The current comparison includes TabPFN, Logistic Regression, and Random Forest. Evaluation uses repeated stratified five-fold cross-validation, two repeats, and random seed 42; the reported metric is macro-F1. The model comparison uses matched splits for the classical baselines and the stored TabPFN fold scores.

The Logistic Regression pipeline uses median imputation, standardization, and class-balanced fitting. The Random Forest pipeline uses median imputation and class-balanced fitting. The current TabPFN script passes missing values to the estimator. These choices describe the existing experiment, not a finalized protocol.

An age-and-sex-only baseline evaluates Logistic Regression and Random Forest with the same repeated stratified fold design. The age ablation removes age from Tier 1 and Tier 1+2 feature sets. The class-level baseline report is based on out-of-fold predictions from the first repeat.

## Evaluation limitations

The reported values are cross-validation estimates, not a final unbiased evaluation. There is no held-out test set yet, and decedent-level splits must be frozen before claims of final generalization are made. Class imbalance and unresolved feature provenance also limit interpretation. No PMCT-only or multimodal model has been evaluated.
