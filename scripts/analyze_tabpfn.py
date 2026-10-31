import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load existing data and TabPFN results
df = pd.read_csv("data/metadata_feature_matrix.csv")
results = pd.read_csv("results/tabpfn_results.csv")

print("\n=== NMDID METADATA ANALYSIS ===")
print("Total decedents:", len(df))
print("Total features:", len(df.columns) - 2)

# 1. Feature categories
def feature_group(name):
    if name in [
        "age_years", "female", "cadaver_length",
        "cadaver_weight", "bmi_like"
    ]:
        return "Basic metadata"
    if name.startswith("dx_"):
        return "Medical diagnosis"
    if name.startswith("mh_"):
        return "Medical history"
    if name.startswith("su_"):
        return "Substance use"
    if name.startswith("env_"):
        return "Lifestyle history"
    if name.startswith("med_") or name == "n_meds":
        return "Medications"
    return "Other"

# 2. Flag features needing provenance review
def review_note(name):
    if name in ["age_years", "female"]:
        return "Check source and completeness"
    if name in [
        "cadaver_length", "cadaver_weight", "bmi_like"
    ]:
        return "Postmortem measurement; review proxy risk"
    if name.startswith("dx_"):
        return "Confirm diagnosis existed before death"
    if name.startswith("mh_"):
        return "Confirm historical rather than fatal injury"
    if name.startswith("su_"):
        return "Confirm history, not toxicology findings"
    if name.startswith("env_"):
        return "Confirm pre-death lifestyle information"
    if name.startswith("med_") or name == "n_meds":
        return "Verify medication source and timing"
    return "Manual review required"

# 3. Audit all numeric features
audit_rows = []

for feature in df.columns:
    if feature in ["id", "label"]:
        continue

    original = df[feature]
    values = pd.to_numeric(original, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)

    audit_rows.append({
        "feature": feature,
        "group": feature_group(feature),
        "missing_percent": round(
            values.isna().mean() * 100, 2
        ),
        "nonzero_count": int(
            ((values != 0) & values.notna()).sum()
        ),
        "unique_values": int(values.nunique()),
        "review_note": review_note(feature),
        "provenance_verified": "NO"
    })

audit = pd.DataFrame(audit_rows)
audit.to_csv("results/feature_audit.csv", index=False)

print("\n=== FEATURE GROUP COUNTS ===")
print(audit.groupby("group").size().to_string())

print("\n=== TOP 10 MISSING FEATURES ===")
print(
    audit.sort_values(
        "missing_percent", ascending=False
    )[["feature", "missing_percent"]]
    .head(10)
    .to_string(index=False)
)

# 4. Class distribution
distribution = (
    df["label"]
    .value_counts()
    .rename_axis("label")
    .reset_index(name="count")
)

distribution.to_csv(
    "results/class_distribution.csv", index=False
)

print("\n=== CLASS DISTRIBUTION ===")
print(distribution.to_string(index=False))

# 5. Summarize TabPFN scores
summary = (
    results.groupby("feature_set", sort=False)
    ["macro_f1"]
    .agg(
        mean_macro_f1="mean",
        std_macro_f1=lambda x: np.std(x, ddof=0),
        n_folds="count"
    )
    .reset_index()
)

summary.to_csv(
    "results/tabpfn_results_summary.csv", index=False
)

print("\n=== TABPFN RESULTS ===")
print(summary.to_string(index=False))

# 6. Generate performance chart
fig, ax = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(summary))

bars = ax.bar(
    x,
    summary["mean_macro_f1"],
    yerr=summary["std_macro_f1"],
    capsize=6,
    width=0.55
)

ax.set_xticks(x)
ax.set_xticklabels(summary["feature_set"])
ax.set_ylabel("Macro-F1 Score")
ax.set_title(
    "NMDID: TabPFN 3.5 Foundation Model Results"
)
ax.set_ylim(0, 0.40)
ax.grid(axis="y", alpha=0.2)

for bar, score in zip(
    bars, summary["mean_macro_f1"]
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.035,
        f"{score:.4f}",
        ha="center",
        fontsize=11
    )

plt.tight_layout()
plt.savefig(
    "reports/figures/tabpfn_feature_comparison.png",
    dpi=250,
    bbox_inches="tight"
)
plt.close()

print("\n=== FILES CREATED ===")
print("1. results/feature_audit.csv")
print("2. results/class_distribution.csv")
print("3. results/tabpfn_results_summary.csv")
print("4. reports/figures/tabpfn_feature_comparison.png")
print("\nAnalysis completed successfully.")
