# Leakage policy

## Exclude from predictors

Do not use the following as model inputs:

- Cause-of-death or target/outcome fields, including direct or proxy encodings.
- Manner of death.
- Scan protocol, acquisition, or scanner settings.
- Identification method.
- Location fields.
- Decedent IDs or any record identifiers.

## Treat as nuisance or review separately

- Time delay after death is a nuisance variable, not a cause-of-death predictor. If retained for technical analysis, report it separately and do not use it to imply causal information.
- Diagnoses, medical history, substance use, lifestyle, and medication fields require verification of provenance and timing. Confirm that each represents information available before death rather than a consequence of the fatal event or postmortem examination.
- Report history-based feature results as a separate ablation from the Tier 1 basic-metadata baseline.

## Current status

The current Tier 1 feature set contains age, sex indicator, cadaver length, cadaver weight, and a BMI-like measurement. The Tier 1+2 set adds 65 diagnosis, history, substance-use, lifestyle, and medication features. Their source and timing have not yet been fully verified; performance from Tier 1+2 is therefore preliminary and may be affected by leakage.

## Before modeling

1. Map every candidate feature to the NMDID data dictionary and record its source and timing.
2. Remove excluded fields, identifiers, direct outcome proxies, and unverified postmortem-derived predictors.
3. Freeze data splits by decedent before model selection.
4. Report history-based features separately and document all exclusions and ablations.
