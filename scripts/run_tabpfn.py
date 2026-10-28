
# run_tabpfn.py
# CSE 499A — NMDID Cause-of-Death Classification
#
# Requirements:
# pip install tabpfn scikit-learn pandas numpy
#
# Run:
# python run_tabpfn.py
#
# Google Colab:
# 1. Select Runtime > Change runtime type > T4 GPU.
# 2. Upload this script and metadata_feature_matrix.csv.
# 3. Run: !pip install -q tabpfn scikit-learn pandas numpy
# 4. Run: !python run_tabpfn.py

import random

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tabpfn import TabPFNClassifier


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# --------------------------------------------------
# 2. Load and validate metadata
# --------------------------------------------------

df = pd.read_csv(
    "data/metadata_feature_matrix.csv",
    dtype={"id": "string", "label": "string"}
)

required_columns = ["id", "label"]

safe_features = [
    "age_years",
    "female",
    "cadaver_length",
    "cadaver_weight",
    "bmi_like"
]

required_columns += safe_features

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["id"].isna().any() or df["id"].duplicated().any():
    raise ValueError(
        "Each decedent must have a unique, non-missing ID."
    )

if df["label"].isna().any():
    raise ValueError("Missing target labels detected.")

# Expected five cause-of-death categories.
classes = [
    "blunt_trauma",
    "firearm",
    "stab",
    "cardiac",
    "intox"
]

df["label"] = df["label"].str.strip()

if set(df["label"].unique()) != set(classes):
    raise ValueError(
        "The dataset must contain exactly these labels: "
        f"{classes}"
    )

# Each class needs at least five samples for 5-fold CV.
class_counts = df["label"].value_counts()

if class_counts.min() < 5:
    raise ValueError(
        "Each class needs at least 5 records."
    )

# Encode target labels consistently.
label_map = {
    label: index for index, label in enumerate(classes)
}

y = df["label"].map(label_map).to_numpy(dtype=int)

print(f"Total decedents: {len(df)}")
print("Class distribution:")
print(class_counts.to_string())


# --------------------------------------------------
# 3. Define feature sets
# --------------------------------------------------

# Tier 1: Safe features only.
# Tier 1+2: All numeric predictors except id and label.

history_features = [
    col for col in df.columns
    if col not in ["id", "label"]
]

feature_sets = {
    "Tier 1 (safe)": safe_features,
    "Tier 1+2 (history)": history_features
}


# --------------------------------------------------
# 4. Cross-validation setup
# --------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=SEED
)

# Reuse identical train/test splits for both feature sets.
splits = list(cv.split(np.zeros(len(df)), y))

results = []


# --------------------------------------------------
# 5. Train and evaluate TabPFN
# --------------------------------------------------

for feature_set, feature_columns in feature_sets.items():

    # Convert every predictor to floating point.
    # NaN values are kept because TabPFN supports them.
    X = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )

    scores = []

    print(f"\nEvaluating: {feature_set}")
    print(f"Number of features: {X.shape[1]}")

    for i, (train_idx, test_idx) in enumerate(splits):

        repeat_number = (i // 5) + 1
        fold_number = (i % 5) + 1

        # A fresh classifier is created for each fold.
        # No tuning or model selection uses test-fold data.
        model = TabPFNClassifier(
            device=device,
            random_state=SEED
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        score = f1_score(
            y_test,
            predictions,
            labels=np.arange(len(classes)),
            average="macro",
            zero_division=0
        )

        scores.append(score)

        results.append({
            "feature_set": feature_set,
            "repeat": repeat_number,
            "fold": fold_number,
            "macro_f1": score
        })

        print(
            f"Repeat {repeat_number}, Fold {fold_number}: "
            f"macro-F1 = {score:.4f}"
        )

        del model

        if device == "cuda":
            torch.cuda.empty_cache()

    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=0)

    print(
        f"\n{feature_set}: macro-F1 = "
        f"{mean_score:.4f} +/- {std_score:.4f}"
    )


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

# Include overall mean and standard deviation
# alongside individual fold results.
results_df["mean_macro_f1"] = (
    results_df.groupby("feature_set")["macro_f1"]
    .transform("mean")
)

results_df["std_macro_f1"] = (
    results_df.groupby("feature_set")["macro_f1"]
    .transform(lambda s: s.to_numpy().std(ddof=0))
)

results_df.to_csv(
    "results/tabpfn_results.csv",
    index=False
)

print("\nResults saved to results/tabpfn_results.csv")
print("Evaluation completed.")
