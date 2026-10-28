import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

SEED = 42

df = pd.read_csv("data/metadata_feature_matrix.csv")

classes = [
    "blunt_trauma",
    "firearm",
    "stab",
    "cardiac",
    "intox"
]

safe_features = [
    "age_years",
    "female",
    "cadaver_length",
    "cadaver_weight",
    "bmi_like"
]

if df["id"].isna().any() or df["id"].duplicated().any():
    raise ValueError("Decedent IDs must be unique.")

if set(df["label"]) != set(classes):
    raise ValueError("Unexpected or missing target labels.")

y = df["label"].map(
    {name: i for i, name in enumerate(classes)}
).to_numpy()

feature_sets = {
    "Tier 1 (safe)": safe_features,
    "Tier 1+2 (history)": [
        c for c in df.columns
        if c not in ["id", "label"]
    ]
}

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=SEED
)

splits = list(cv.split(np.zeros(len(df)), y))

fold_results = []
class_results = []

def build_model(name):
    if name == "Logistic Regression":
        return make_pipeline(
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            ),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )
        )

    return make_pipeline(
        SimpleImputer(
            strategy="median",
            keep_empty_features=True
        ),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        )
    )

print("\n=== BASELINE MODEL COMPARISON ===")

for feature_set, columns in feature_sets.items():

    X = (
        df[columns]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )

    for model_name in [
        "Logistic Regression",
        "Random Forest"
    ]:

        print("\n", feature_set, "|", model_name)

        # One out-of-fold prediction per person
        # from the first CV repeat.
        first_repeat_predictions = np.full(
            len(df), -1, dtype=int
        )

        for i, (train_idx, test_idx) in enumerate(splits):

            model = build_model(model_name)

            model.fit(
                X.iloc[train_idx],
                y[train_idx]
            )

            pred = model.predict(X.iloc[test_idx])

            score = f1_score(
                y[test_idx],
                pred,
                average="macro",
                labels=np.arange(5),
                zero_division=0
            )

            repeat = i // 5 + 1
            fold = i % 5 + 1

            fold_results.append({
                "feature_set": feature_set,
                "model": model_name,
                "repeat": repeat,
                "fold": fold,
                "macro_f1": score
            })

            if repeat == 1:
                first_repeat_predictions[test_idx] = pred

            print(
                f"Repeat {repeat}, Fold {fold}: "
                f"{score:.4f}"
            )

        # Class-wise results from first-repeat
        # out-of-fold predictions only.
        report = classification_report(
            y,
            first_repeat_predictions,
            labels=np.arange(5),
            target_names=classes,
            output_dict=True,
            zero_division=0
        )

        for label in classes:
            class_results.append({
                "feature_set": feature_set,
                "model": model_name,
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"]
            })

        # Save confusion matrices for expanded metadata.
        if feature_set == "Tier 1+2 (history)":
            fig, ax = plt.subplots(figsize=(9, 7))

            ConfusionMatrixDisplay.from_predictions(
                y,
                first_repeat_predictions,
                labels=np.arange(5),
                display_labels=classes,
                xticks_rotation=30,
                ax=ax,
                cmap="Blues"
            )

            ax.set_title(
                f"{model_name} - First-repeat OOF"
            )

            plt.tight_layout()

            filename = (
                "confusion_" +
                model_name.lower().replace(" ", "_") +
                ".png"
            )

            plt.savefig("reports/figures/" + filename, dpi=220)
            plt.close()

# Load previously completed TabPFN fold scores.
tabpfn = pd.read_csv("results/tabpfn_results.csv")

tabpfn = tabpfn[
    ["feature_set", "repeat", "fold", "macro_f1"]
].copy()

tabpfn["model"] = "TabPFN"

# Combine all models for a fair CV comparison.
combined = pd.concat(
    [pd.DataFrame(fold_results), tabpfn],
    ignore_index=True
)

summary = (
    combined.groupby(
        ["feature_set", "model"],
        sort=False
    )["macro_f1"]
    .agg(
        mean_macro_f1="mean",
        std_macro_f1=lambda s: np.std(s, ddof=0),
        n_folds="count"
    )
    .reset_index()
)

combined.to_csv(
    "results/model_comparison_folds.csv",
    index=False
)

summary.to_csv(
    "results/model_comparison_summary.csv",
    index=False
)

pd.DataFrame(class_results).to_csv(
    "results/baseline_class_metrics.csv",
    index=False
)

# Visual comparison across both feature sets.
models = [
    "TabPFN",
    "Logistic Regression",
    "Random Forest"
]

features = list(feature_sets.keys())

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(features))
width = 0.25

for j, model_name in enumerate(models):

    means = []
    errors = []

    for feature_name in features:
        row = summary[
            (summary["feature_set"] == feature_name) &
            (summary["model"] == model_name)
        ].iloc[0]

        means.append(row["mean_macro_f1"])
        errors.append(row["std_macro_f1"])

    ax.bar(
        x + (j - 1) * width,
        means,
        width,
        yerr=errors,
        capsize=4,
        label=model_name
    )

ax.set_xticks(x)
ax.set_xticklabels(features)
ax.set_ylabel("Mean Macro-F1")
ax.set_title("NMDID: Model Comparison (860 Decedents)")
ax.set_ylim(0, max(0.45, summary["mean_macro_f1"].max() * 1.2))
ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig("reports/figures/model_comparison.png", dpi=250)
plt.close()

print("\n=== FINAL MODEL COMPARISON ===")
print(summary.to_string(index=False))

print("\nFILES CREATED:")
print("model_comparison_folds.csv")
print("model_comparison_summary.csv")
print("baseline_class_metrics.csv")
print("model_comparison.png")
print("confusion_logistic_regression.png")
print("confusion_random_forest.png")
print("\nCompleted.")
