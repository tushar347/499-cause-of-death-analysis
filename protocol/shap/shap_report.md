# SHAP explainability results

TreeSHAP explanations were computed for models fitted on the frozen training partition and evaluated on the previously used validation partition. Values explain model outputs, not causal effects. LightGBM TreeSHAP values are on the raw model-output scale; no feature is declared safe based on SHAP.

## Five Class

- Model/scenario: Random Forest / S7_all_history (67 features); fitted on 602 training records and explained on 129 validation records.
- Top global features by mean absolute TreeSHAP: med_na, age_years, n_meds, dx_hypertension, env_drinking status:high risk.
- Features from diagnosis/history/substance/lifestyle/medication groups account for 85.9% of the aggregated mean absolute importance.
- This importance share is descriptive model reliance; it does not establish feature provenance, timing, safety, or causality.

- **blunt_trauma:** strongest class-output features: med_na, n_meds, age_years.
- **firearm:** strongest class-output features: age_years, env_drinking status:drinker, cur, su_na.
- **stab:** strongest class-output features: med_na, n_meds, env_drinking status:high risk.
- **cardiac:** strongest class-output features: age_years, med_na, dx_hypertension.
- **intox:** strongest class-output features: med_na, age_years, n_meds.

## Broad Three Class

- Model/scenario: LightGBM / S7_all_history (67 features); fitted on 602 training records and explained on 129 validation records.
- Top global features by mean absolute TreeSHAP: age_years, med_na, su_na, env_drinking status:high risk, dx_hypertension.
- Features from diagnosis/history/substance/lifestyle/medication groups account for 74.4% of the aggregated mean absolute importance.
- This importance share is descriptive model reliance; it does not establish feature provenance, timing, safety, or causality.

- **trauma:** strongest class-output features: med_na, su_na, age_years.
- **cardiac:** strongest class-output features: age_years, dx_hypertension, env_drinking status:drinker, cur.
- **intoxication:** strongest class-output features: age_years, env_drinking status:high risk, n_meds.

## Trauma Subtype

- Model/scenario: LightGBM / S6_medications (18 features); fitted on 204 training records and explained on 44 validation records.
- Top global features by mean absolute TreeSHAP: age_years, med_na, n_meds, female, med_d7804.0.
- Features from diagnosis/history/substance/lifestyle/medication groups account for 23.5% of the aggregated mean absolute importance.
- This importance share is descriptive model reliance; it does not establish feature provenance, timing, safety, or causality.

- **blunt_trauma:** strongest class-output features: age_years, n_meds, female.
- **firearm:** strongest class-output features: age_years, n_meds, med_na.
- **stab:** strongest class-output features: age_years, med_na, female.
