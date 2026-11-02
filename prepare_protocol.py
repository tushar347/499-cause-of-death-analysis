from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------
# 1. Load and validate the dataset
# ------------------------------------------

candidates = [
    Path("data/metadata_feature_matrix.csv"),
    Path("metadata_feature_matrix.csv")
]

data_path = next((p for p in candidates if p.exists()), None)

if data_path is None:
    raise FileNotFoundError(
        "metadata_feature_matrix.csv not found."
    )

df = pd.read_csv(
    data_path,
    dtype={"id": "string", "label": "string"}
)

df["id"] = df["id"].str.strip()
df["label"] = df["label"].str.strip()

classes = [
    "blunt_trauma",
    "firearm",
    "stab",
    "cardiac",
    "intox"
]

if df["id"].isna().any() or df["id"].duplicated().any():
    raise ValueError(
        "IDs must be unique. Resolve duplicate decedents "
        "before creating splits."
    )

if df["label"].isna().any():
    raise ValueError("Missing labels detected.")

if set(df["label"].unique()) != set(classes):
    raise ValueError("Unexpected cause-of-death labels.")

if df["label"].value_counts().min() < 5:
    raise ValueError("Insufficient samples for stratification.")

print("Dataset:", data_path)
print("Decedents:", len(df))
print("Features:", len(df.columns) - 2)

# ------------------------------------------
# 2. Freeze the label mapping
# ------------------------------------------

broad_map = {
    "blunt_trauma": "trauma",
    "firearm": "trauma",
    "stab": "trauma",
    "cardiac": "cardiac",
    "intox": "intoxication"
}

trauma_classes = [
    "blunt_trauma",
    "firearm",
    "stab"
]

label_rows = []

for label in classes:
    is_trauma = label in trauma_classes

    label_rows.append({
        "original_label": label,
        "broad_label": broad_map[label],
        "within_domain_task": (
            "trauma_subtype" if is_trauma
            else "not_applicable_single_class"
        ),
        "within_domain_label": label if is_trauma else "",
        "include_in_trauma_subtype_task": is_trauma
    })

label_map = pd.DataFrame(label_rows)

# ------------------------------------------
# 3. Create preliminary leakage audit
# ------------------------------------------

basic = [
    "age_years",
    "female",
    "cadaver_length",
    "cadaver_weight",
    "bmi_like"
]

def feature_group(name):
    if name in basic:
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
    if name == "id":
        return "Identifier"
    if name == "label":
        return "Target"
    return "Other - review"

def leakage_category(name):
    if name in ["id", "label"]:
        return "Forbidden"

    if name in ["age_years", "female"]:
        return "Allowed"

    return "Gray"

def reason(name):
    if name == "id":
        return "Identifier; use only for decedent grouping."

    if name == "label":
        return "Target variable; never use as a predictor."

    if name in ["age_years", "female"]:
        return (
            "Candidate demographic information. Verify "
            "definition, coding and availability at prediction time."
        )

    if name in [
        "cadaver_length",
        "cadaver_weight",
        "bmi_like"
    ]:
        return (
            "Postmortem measurement. Review possible "
            "shortcut or body-measurement effects."
        )

    if name.startswith("dx_"):
        return (
            "Verify that the diagnosis was recorded before death "
            "and was not derived from the fatal event."
        )

    if name.startswith("mh_"):
        return (
            "Verify historical timing, especially for injury-related fields."
        )

    if name.startswith("su_"):
        return (
            "Verify historical substance-use information versus "
            "toxicology or investigation findings."
        )

    if name.startswith("env_"):
        return "Verify source and timing of lifestyle information."

    if name.startswith("med_") or name == "n_meds":
        return (
            "Verify medication source and whether it represents "
            "pre-death medication history."
        )

    return "Manual source and timing review required."

audit_rows = []

for col in df.columns:
    category = leakage_category(col)

    audit_rows.append({
        "feature": col,
        "feature_group": feature_group(col),
        "category": category,
        "reason": reason(col),
        "source_verified": "No",
        "pre_death_timing_verified": "No",
        "final_decision": (
            "Exclude as predictor" if category == "Forbidden"
            else "Pending human review"
        ),
        "review_notes": ""
    })

audit = pd.DataFrame(audit_rows)

# ------------------------------------------
# 4. Prepare broad class counts
# ------------------------------------------

df["broad_label"] = df["label"].map(broad_map)

broad_counts = (
    df["broad_label"]
    .value_counts()
    .rename_axis("broad_label")
    .reset_index(name="count")
)

trauma_counts = (
    df[df["label"].isin(trauma_classes)]["label"]
    .value_counts()
    .rename_axis("trauma_subtype")
    .reset_index(name="count")
)

# ------------------------------------------
# 5. Create fixed decedent-level splits
# ------------------------------------------

# Each ID currently represents one decedent.
# Stratification is performed by the original five labels.

y = df["label"].map(
    {label: i for i, label in enumerate(classes)}
).to_numpy()

indices = np.arange(len(df))

# First separate 15% as the final test set.
train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# Separate 15% of the complete dataset for validation.
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.15 / 0.85,
    stratify=y[train_val_idx],
    random_state=43
)

split_names = np.empty(len(df), dtype=object)

split_names[train_idx] = "train"
split_names[val_idx] = "validation"
split_names[test_idx] = "test"

# Verify no decedent crosses partitions.
split_ids = {
    name: set(df.iloc[idx]["id"])
    for name, idx in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx)
    ]
}

assert split_ids["train"].isdisjoint(split_ids["validation"])
assert split_ids["train"].isdisjoint(split_ids["test"])
assert split_ids["validation"].isdisjoint(split_ids["test"])

assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

# ------------------------------------------
# 6. Save protocol outputs
# ------------------------------------------

out = Path("protocol")
out.mkdir(exist_ok=True)

label_map.to_csv(out / "label_map.csv", index=False)
audit.to_csv(out / "leakage_audit.csv", index=False)

broad_counts.to_csv(
    out / "broad_class_counts.csv",
    index=False
)

trauma_counts.to_csv(
    out / "trauma_subtype_counts.csv",
    index=False
)

# Local ID mapping: DO NOT upload or commit this file.
assignments = pd.DataFrame({
    "id": df["id"],
    "split": split_names
})

assignments.to_csv(
    out / "split_assignments_local.csv",
    index=False
)

# Aggregate five-class split summary.
summary_data = pd.DataFrame({
    "split": split_names,
    "label": df["label"]
})

split_summary = (
    summary_data.groupby(["split", "label"])
    .size()
    .reset_index(name="count")
)

split_summary["split_total"] = (
    split_summary.groupby("split")["count"]
    .transform("sum")
)

split_summary["percentage"] = (
    split_summary["count"] /
    split_summary["split_total"] * 100
).round(2)

split_summary.to_csv(
    out / "split_summary.csv",
    index=False
)

# Aggregate broad-class split summary.
broad_data = pd.DataFrame({
    "split": split_names,
    "broad_label": df["broad_label"]
})

broad_summary = (
    broad_data.groupby(["split", "broad_label"])
    .size()
    .reset_index(name="count")
)

broad_summary.to_csv(
    out / "broad_split_summary.csv",
    index=False
)

# Record split settings.
manifest = {
    "dataset": str(data_path),
    "decedents": len(df),
    "unique_decedents": int(df["id"].nunique()),
    "feature_count": len(df.columns) - 3,
    "group_column": "id",
    "split_ratio": {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15
    },
    "test_random_state": 42,
    "validation_random_state": 43,
    "stratification": "original five-class label",
    "note": (
        "Current matrix has one row per unique decedent. "
        "Future CT slices and images must inherit the split "
        "assigned to their decedent ID."
    )
}

with open(out / "split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=4)

# Prevent accidental upload of person-level ID assignments.
gitignore = Path(".gitignore")
entry = "protocol/split_assignments_local.csv"

existing = (
    gitignore.read_text(encoding="utf-8")
    if gitignore.exists() else ""
)

if entry not in existing.splitlines():
    with gitignore.open("a", encoding="utf-8") as f:
        f.write("\n" + entry + "\n")

# ------------------------------------------
# 7. Display results
# ------------------------------------------

print("\n=== BROAD LABEL DISTRIBUTION ===")
print(broad_counts.to_string(index=False))

print("\n=== TRAUMA SUBTYPE DISTRIBUTION ===")
print(trauma_counts.to_string(index=False))

print("\n=== SPLIT SUMMARY ===")
print(split_summary.to_string(index=False))

print("\n=== LEAKAGE CATEGORY COUNTS ===")
print(audit.groupby("category").size().to_string())

print("\n=== CREATED FILES ===")
for file in sorted(out.iterdir()):
    print(file)

print("\nProtocol preparation completed.")
print("No machine-learning models were trained.")
print("Keep split_assignments_local.csv private.")
