# Roadmap

1. **Verify feature provenance.** Map each candidate field to the NMDID data dictionary and verify its source and timing. Exclude outcome, manner, acquisition, identification, location, and ID fields under the leakage policy.
2. **Document the cohort.** Record inclusion/exclusion criteria and class counts without publishing person-level records.
3. **Freeze evaluation splits.** Create train/validation/test splits by decedent before model selection; retain the current cross-validation results as preliminary baseline evidence.
4. **Obtain approved PMCT data.** Confirm authorization and terms before local use, and test a reproducible PMCT preprocessing workflow.
5. **Extract image embeddings.** Select and document a pretrained foundation model, preprocessing, and embedding extraction configuration.
6. **Evaluate modalities.** Compare metadata-only, PMCT-only, and multimodal fusion using the frozen decedent-level evaluation design.
7. **Report limitations.** Document class imbalance, feature provenance, missingness, data access constraints, and uncertainty.

PMCT preprocessing has not started. PMCT embeddings and multimodal fusion are pending; no image or multimodal results currently exist.
