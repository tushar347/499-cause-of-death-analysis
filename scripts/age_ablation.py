import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

SEED = 42
df = pd.read_csv("data/metadata_feature_matrix.csv")

classes = [
    "blunt_trauma", "firearm", "stab",
    "cardiac", "intox"
]

y = df["label"].map(
    {name: i for i, name in enumerate(classes)}
).to_numpy()

safe = [
    "age_years", "female", "cadaver_length",
    "cadaver_weight", "bmi_like"
]

all_features = [
    col for col in df.columns
    if col not in ["id", "label"]
]

feature_sets = {
    "Tier 1 without age": [
        c for c in safe if c != "age_years"
    ],
    "Tier 1+2 without age": [
        c for c in all_features if c != "age_years"
    ]
}

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=SEED
)

splits = list(cv.split(np.zeros(len(df)), y))

rows = []

print("\n=== AGE ABLATION EXPERIMENT ===")

for feature_set, columns in feature_sets.items():

    X = (
        df[columns]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )

    for name in ["Logistic Regression", "Random Forest"]:

        scores = []

        for train_idx, test_idx in splits:

            if name == "Logistic Regression":
                model = make_pipeline(
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
            else:
                model = make_pipeline(
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

            model.fit(X.iloc[train_idx], y[train_idx])

            predictions = model.predict(
                X.iloc[test_idx]
            )

            scores.append(
                f1_score(
                    y[test_idx],
                    predictions,
                    average="macro",
                    labels=np.arange(5),
                    zero_division=0
                )
            )

        mean = np.mean(scores)
        std = np.std(scores, ddof=0)

        rows.append({
            "feature_set": feature_set,
            "model": name,
            "mean_macro_f1": mean,
            "std_macro_f1": std
        })

        print(
            f"{feature_set} | {name}: "
            f"{mean:.4f} +/- {std:.4f}"
        )

results = pd.DataFrame(rows)

results.to_csv(
    "results/age_ablation_results.csv",
    index=False
)

# Compare against your original model results
original = pd.read_csv(
    "results/model_comparison_summary.csv"
)

original = original[
    original["model"].isin([
        "Logistic Regression", "Random Forest"
    ])
].copy()

original["experiment"] = (
    original["feature_set"] + " (with age)"
)

results["experiment"] = results["feature_set"]

combined = pd.concat(
    [
        original[[
            "experiment", "model", "mean_macro_f1"
        ]],
        results[[
            "experiment", "model", "mean_macro_f1"
        ]]
    ],
    ignore_index=True
)

combined.to_csv(
    "results/age_ablation_comparison.csv",
    index=False
)

# Visualization
fig, ax = plt.subplots(figsize=(11, 6))

models = ["Logistic Regression", "Random Forest"]

for model_name in models:
    subset = combined[
        combined["model"] == model_name
    ]

    ax.plot(
        subset["experiment"],
        subset["mean_macro_f1"],
        marker="o",
        label=model_name
    )

ax.set_ylabel("Mean Macro-F1")
ax.set_title("Age Ablation: With vs Without Age")
ax.tick_params(axis="x", rotation=25)
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("reports/figures/age_ablation.png", dpi=250)
plt.close()

print("\n=== RESULTS ===")
print(results.to_string(index=False))

print("\nSaved:")
print("results/age_ablation_results.csv")
print("results/age_ablation_comparison.csv")
print("reports/figures/age_ablation.png")
print("\nCompleted.")
