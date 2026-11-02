# Split conformal prediction

The model and temperature scaler are fitted using disjoint, stratified subsets of the original training partition. A third, disjoint training subset supplies nonconformity scores `1 - p(true class)`. For nominal coverage `1 - alpha`, the threshold is the `ceil((n + 1) * (1 - alpha))` order statistic, with an infinite threshold if the rank exceeds the calibration sample size. A class is included when `1 - calibrated_probability <= threshold`.

Reported empirical coverage and set sizes are evaluated on the existing validation partition. That partition was already used to compare 81 configurations and select these models, so these validation results are descriptive and must not be presented as an independent coverage guarantee. The independent test partition was not evaluated. Standard split conformal prediction targets marginal coverage under exchangeability assumptions; it does not guarantee coverage for each individual class. Small per-class calibration counts, especially for trauma subtype, make classwise estimates imprecise.

Prediction sets are summarized without saving person-level identifiers or per-record predictions. Abstention is defined as a set whose size is not one; empty and multi-class sets are also reported separately.
