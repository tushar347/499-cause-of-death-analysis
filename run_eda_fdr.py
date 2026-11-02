from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError

SEED = 42

CLASSES = [
    "blunt_trauma",
    "firearm",
    "stab",
    "cardiac",
    "intox"
]

OUT = Path("protocol/eda")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ------------------------------------------
# 1. Load dataset and frozen split assignments
# ------------------------------------------

candidates = [
    Path("data/metadata_feature_matrix.csv"),
    Path("metadata_feature_matrix.csv")
]

data_path = next(
    (p for p in candidates if p.exists()), None
)

if data_path is None:
    raise FileNotFoundError("Metadata CSV not found.")

df = pd.read_csv(
    data_path,
    dtype={"id": "string", "label": "string"}
)

assignments = pd.read_csv(
    "protocol/split_assignments_local.csv",
    dtype={"id": "string"}
)

df["id"] = df["id"].str.strip()
assignments["id"] = assignments["id"].str.strip()

if df["id"].duplicated().any():
    raise ValueError("Duplicate decedent IDs detected.")

df = df.merge(
    assignments,
    on="id",
    how="left",
    validate="one_to_one"
)

if df["split"].isna().any():
    raise ValueError("Some records have no split assignment.")

# Only training data is used below.
train = df[df["split"] == "train"].copy()

FEATURES = [
    c for c in df.columns
    if c not in ["id", "label", "split"]
]

if set(train["label"].unique()) != set(CLASSES):
    raise ValueError("Missing cause-of-death class in training data.")

print("\n=== TRAINING DATA ===")
print("Training decedents:", len(train))
print("Features:", len(FEATURES))
print(train["label"].value_counts().to_string())

# ------------------------------------------
# 2. Feature groups
# ------------------------------------------

def feature_group(name):
    if name in [
        "age_years", "female", "cadaver_length",
        "cadaver_weight", "bmi_like"
    ]:
        return "Basic metadata"
    if name.startswith("dx_"):
        return "Medical diagnoses"
    if name.startswith("mh_"):
        return "Medical history"
    if name.startswith("su_"):
        return "Substance use"
    if name.startswith("env_"):
        return "Lifestyle history"
    if name.startswith("med_") or name == "n_meds":
        return "Medications"
    return "Other"

def numeric_feature(name):
    return (
        pd.to_numeric(train[name], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

def is_binary(values):
    unique = set(values.dropna().unique())
    return bool(unique) and unique.issubset({0, 1})

# ------------------------------------------
# 3. Feature summary
# ------------------------------------------

summary_rows = []
binary_features = []
continuous_features = []

for feature in FEATURES:
    values = numeric_feature(feature)
    binary = is_binary(values)

    if binary:
        binary_features.append(feature)
    else:
        continuous_features.append(feature)

    summary_rows.append({
        "feature": feature,
        "group": feature_group(feature),
        "type": "binary" if binary else "continuous",
        "missing_percent": round(
            values.isna().mean() * 100, 2
        ),
        "unique_values": values.nunique(),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "minimum": values.min(),
        "maximum": values.max(),
        "positive_percent": (
            round((values == 1).mean() * 100, 2)
            if binary else np.nan
        )
    })

feature_summary = pd.DataFrame(summary_rows)

feature_summary.to_csv(
    OUT / "feature_summary_train.csv",
    index=False
)

# ------------------------------------------
# 4. EDA: class distribution
# ------------------------------------------

class_counts = (
    train["label"]
    .value_counts()
    .reindex(CLASSES)
)

fig, ax = plt.subplots(figsize=(9, 5))

class_counts.plot(
    kind="bar", ax=ax
)

ax.set_title("Training Set: Cause-of-Death Distribution")
ax.set_ylabel("Number of Decedents")
ax.set_xlabel("Cause-of-Death Class")
ax.grid(axis="y", alpha=0.2)

plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(FIG / "train_class_distribution.png", dpi=220)
plt.close()

# ------------------------------------------
# 5. EDA: missing-value analysis
# ------------------------------------------

missing = feature_summary.sort_values(
    "missing_percent", ascending=False
)

fig, ax = plt.subplots(figsize=(10, 7))

top = missing.head(20).sort_values("missing_percent")

ax.barh(
    top["feature"],
    top["missing_percent"]
)

ax.set_title("Top 20 Features by Missingness")
ax.set_xlabel("Missing Values (%)")
ax.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(FIG / "feature_missingness.png", dpi=220)
plt.close()

# ------------------------------------------
# 6. Unadjusted statistical associations
# Binary: Fisher exact test + odds ratio
# Continuous: Mann-Whitney U test
# ------------------------------------------

association_rows = []

for feature in FEATURES:

    values = numeric_feature(feature)
    binary = is_binary(values)

    for cls in CLASSES:

        target = train["label"].eq(cls)
        valid = values.notna()

        if binary:

            a = int(((values == 1) & target & valid).sum())
            b = int(((values == 0) & target & valid).sum())
            c = int(((values == 1) & ~target & valid).sum())
            d = int(((values == 0) & ~target & valid).sum())

            table = [[a, b], [c, d]]

            if (a + b == 0) or (c + d == 0):
                continue

            _, p = fisher_exact(table)

            # Continuity correction for finite OR and CI.
            OR = (
                (a + 0.5) * (d + 0.5)
            ) / (
                (b + 0.5) * (c + 0.5)
            )

            se = np.sqrt(
                1/(a+0.5) + 1/(b+0.5) +
                1/(c+0.5) + 1/(d+0.5)
            )

            lower = np.exp(np.log(OR) - 1.96*se)
            upper = np.exp(np.log(OR) + 1.96*se)

            association_rows.append({
                "feature": feature,
                "class": cls,
                "test": "Fisher exact",
                "effect": "Odds ratio",
                "odds_ratio": OR,
                "ci_lower": lower,
                "ci_upper": upper,
                "effect_size": np.nan,
                "p_value": p,
                "positive_count": a,
                "negative_count": b
            })

        else:

            positive = values[target & valid]
            negative = values[~target & valid]

            if len(positive) == 0 or len(negative) == 0:
                continue

            if values.nunique() < 2:
                continue

            result = mannwhitneyu(
                positive,
                negative,
                alternative="two-sided"
            )

            rank_biserial = (
                2 * result.statistic /
                (len(positive) * len(negative))
            ) - 1

            association_rows.append({
                "feature": feature,
                "class": cls,
                "test": "Mann-Whitney U",
                "effect": "Rank-biserial correlation",
                "odds_ratio": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "effect_size": rank_biserial,
                "p_value": result.pvalue,
                "positive_count": len(positive),
                "negative_count": len(negative)
            })

unadjusted = pd.DataFrame(association_rows)

# FDR correction separately within each target class.
unadjusted["q_value"] = np.nan

for cls in CLASSES:

    mask = (
        (unadjusted["class"] == cls) &
        unadjusted["p_value"].notna()
    )

    if mask.sum() > 0:
        unadjusted.loc[mask, "q_value"] = (
            multipletests(
                unadjusted.loc[mask, "p_value"],
                method="fdr_bh"
            )[1]
        )

unadjusted["significant_fdr_05"] = (
    unadjusted["q_value"] < 0.05
)

unadjusted.to_csv(
    OUT / "unadjusted_associations_fdr.csv",
    index=False
)

# ------------------------------------------
# 7. Age- and sex-adjusted logistic models
# ------------------------------------------

adjusted_rows = []

for feature in FEATURES:

    for cls in CLASSES:

        columns = list(dict.fromkeys([
            feature, "age_years", "female", "label"
        ]))

        sub = train[columns].copy()

        for col in [feature, "age_years", "female"]:
            sub[col] = pd.to_numeric(
                sub[col], errors="coerce"
            )

        sub = sub.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        y = sub["label"].eq(cls).astype(int)

        if y.nunique() < 2 or y.sum() < 5 or (1-y).sum() < 5:
            adjusted_rows.append({
                "feature": feature,
                "class": cls,
                "adjustment": "Age/sex adjusted",
                "n": len(sub),
                "odds_ratio": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "status": "Insufficient observations"
            })
            continue

        values = sub[feature]

        if feature == "age_years":
            exposure = sub["age_years"] / 10
            unit = "Per 10-year increase"

        elif feature == "female":
            exposure = sub["female"]
            unit = "Female indicator: 1 vs 0"

        elif is_binary(values):
            exposure = values
            unit = "Feature present: 1 vs 0"

        else:
            sd = values.std()

            if not np.isfinite(sd) or sd == 0:
                continue

            exposure = (
                values - values.mean()
            ) / sd

            unit = "Per 1 SD increase"

        if exposure.nunique() < 2:
            continue

        design = pd.DataFrame({
            "exposure": exposure
        }, index=sub.index)

        # Do not adjust for a variable against itself.
        if feature != "age_years":
            design["age_10y"] = sub["age_years"] / 10

        if feature != "female":
            design["female_adjustment"] = sub["female"]

        design = sm.add_constant(
            design, has_constant="add"
        )

        try:

            if np.linalg.matrix_rank(
                design.to_numpy()
            ) < design.shape[1]:
                raise ValueError("Singular design matrix")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fit = sm.GLM(
                    y,
                    design,
                    family=sm.families.Binomial()
                ).fit(maxiter=100)

            beta = fit.params["exposure"]
            se = fit.bse["exposure"]
            p = fit.pvalues["exposure"]

            # Flag unstable or separated estimates.
            if (
                not fit.converged or
                not np.isfinite(beta) or
                not np.isfinite(se) or
                abs(beta) > 15 or se > 10
            ):
                status = "Unstable / possible separation"
                OR = lower = upper = p = np.nan

            else:
                status = "Estimated"

                OR = np.exp(beta)
                lower = np.exp(beta - 1.96*se)
                upper = np.exp(beta + 1.96*se)

            adjusted_rows.append({
                "feature": feature,
                "class": cls,
                "adjustment": "Age/sex adjusted",
                "unit": unit,
                "n": len(sub),
                "odds_ratio": OR,
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": p,
                "status": status
            })

        except Exception as exc:

            adjusted_rows.append({
                "feature": feature,
                "class": cls,
                "adjustment": "Age/sex adjusted",
                "unit": unit,
                "n": len(sub),
                "odds_ratio": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "status": "Not estimated: " + type(exc).__name__
            })

adjusted = pd.DataFrame(adjusted_rows)

adjusted["q_value"] = np.nan

for cls in CLASSES:

    mask = (
        (adjusted["class"] == cls) &
        adjusted["p_value"].notna()
    )

    if mask.sum() > 0:
        adjusted.loc[mask, "q_value"] = (
            multipletests(
                adjusted.loc[mask, "p_value"],
                method="fdr_bh"
            )[1]
        )

adjusted.to_csv(
    OUT / "adjusted_associations_age_sex_fdr.csv",
    index=False
)

# ------------------------------------------
# 8. Odds-ratio heatmaps
# ------------------------------------------

def create_or_heatmap(data, filename, title):

    if data.empty:
        print("No odds ratios available for:", filename)
        return

    pivot = data.pivot(
        index="feature",
        columns="class",
        values="odds_ratio"
    )

    # Log2 OR: positive means higher odds,
    # negative means lower odds.
    matrix = np.log2(pivot.where(pivot > 0))
    matrix = matrix.clip(-4, 4)

    fig_height = max(8, len(matrix) * 0.22)

    fig, ax = plt.subplots(
        figsize=(10, fig_height)
    )

    im = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-4,
        vmax=4
    )

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)

    ax.set_title(title)
    ax.set_xlabel("Cause-of-Death Class")
    ax.set_ylabel("Feature")

    plt.colorbar(
        im, ax=ax, label="Log2(Odds Ratio)"
    )

    plt.tight_layout()
    plt.savefig(
        FIG / filename,
        dpi=220,
        bbox_inches="tight"
    )
    plt.close()

binary_unadjusted = unadjusted[
    unadjusted["test"] == "Fisher exact"
]

create_or_heatmap(
    binary_unadjusted,
    "unadjusted_or_heatmap.png",
    "Unadjusted Odds Ratios by Class"
)

create_or_heatmap(
    adjusted[adjusted["status"] == "Estimated"],
    "adjusted_or_heatmap.png",
    "Age/Sex-Adjusted Odds Ratios"
)

# ------------------------------------------
# 9. Continuous feature correlation
# ------------------------------------------

if len(continuous_features) >= 2:

    cont = pd.DataFrame({
        c: numeric_feature(c)
        for c in continuous_features
    })

    corr = cont.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        corr.columns, rotation=45, ha="right"
    )

    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    ax.set_title("Spearman Correlation of Continuous Features")

    plt.colorbar(im, ax=ax, label="Spearman correlation")
    plt.tight_layout()

    plt.savefig(
        FIG / "continuous_feature_correlation.png",
        dpi=220
    )
    plt.close()

# ------------------------------------------
# 10. Final summary
# ------------------------------------------

print("\n=== EDA COMPLETED ===")

print("\nFeature types:")
print(feature_summary["type"].value_counts())

print("\nMissingness:")
print(
    feature_summary.sort_values(
        "missing_percent", ascending=False
    )[["feature", "missing_percent"]].head(10)
)

print("\nUnadjusted associations:")
print("Total tests:", len(unadjusted))
print(
    "FDR significant:",
    int(unadjusted["significant_fdr_05"].sum())
)

print("\nAdjusted associations:")
print("Total models:", len(adjusted))
print("Estimated:", int((adjusted["status"] == "Estimated").sum()))
print(
    "FDR significant:",
    int((adjusted["q_value"] < 0.05).sum())
)

print("\nFiles created in:", OUT)

for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(path)

print("\nEDA and statistical analysis completed.")
print("All association tests used TRAINING DATA ONLY.")
print("No model training or feature selection was performed.")
