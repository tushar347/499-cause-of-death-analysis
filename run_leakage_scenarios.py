from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

from lightgbm import LGBMClassifier

SEED = 42

OUT = Path("protocol/leakage_scenarios")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ------------------------------------------
# 1. Load metadata and frozen assignments
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

if df["id"].isna().any() or df["id"].duplicated().any():
    raise ValueError("Decedent IDs must be unique.")

if assignments["id"].duplicated().any():
    raise ValueError("Duplicate split assignments.")

if set(df["id"]) != set(assignments["id"]):
    raise ValueError("Dataset and split assignments do not match.")

df = df.merge(
    assignments,
    on="id",
    how="left",
    validate="one_to_one"
)

classes5 = [
    "blunt_trauma",
    "firearm",
    "stab",
    "cardiac",
    "intox"
]

trauma_classes = [
    "blunt_trauma",
    "firearm",
    "stab"
]

broad_map = {
    "blunt_trauma": "trauma",
    "firearm": "trauma",
    "stab": "trauma",
    "cardiac": "cardiac",
    "intox": "intoxication"
}

basic = [
    "age_years",
    "female",
    "cadaver_length",
    "cadaver_weight",
    "bmi_like"
]

demographics = ["age_years", "female"]

features = [
    c for c in df.columns
    if c not in ["id", "label", "split"]
]

# ------------------------------------------
# 2. Define feature scenarios
# ------------------------------------------

dx = [c for c in features if c.startswith("dx_")]
mh = [c for c in features if c.startswith("mh_")]
su = [c for c in features if c.startswith("su_")]
env = [c for c in features if c.startswith("env_")]
med = [
    c for c in features
    if c.startswith("med_") or c == "n_meds"
]

history = dx + mh + su + env + med

scenarios = {
    "S0_demographics": demographics,

    "S1_basic_5": basic,

    "S2_diagnoses": demographics + dx,

    "S3_medical_history": demographics + mh,

    "S4_substance_use": demographics + su,

    "S5_lifestyle": demographics + env,

    "S6_medications": demographics + med,

    "S7_all_history": demographics + history,

    "S8_all_70": features
}

for name, cols in scenarios.items():
    if not cols:
        raise ValueError(f"Empty scenario: {name}")
    if not set(cols).issubset(set(features)):
        raise ValueError(f"Invalid features in {name}")

# Save feature manifest.
manifest_rows = []

for scenario, cols in scenarios.items():
    for feature in cols:
        manifest_rows.append({
            "scenario": scenario,
            "feature": feature
        })

pd.DataFrame(manifest_rows).to_csv(
    OUT / "scenario_feature_manifest.csv",
    index=False
)

# ------------------------------------------
# 3. Define three classification tasks
# ------------------------------------------

df["broad_label"] = df["label"].map(broad_map)

tasks = {
    "five_class": (
        df,
        "label",
        classes5
    ),

    "broad_three_class": (
        df,
        "broad_label",
        ["trauma", "cardiac", "intoxication"]
    ),

    "trauma_subtype": (
        df[df["label"].isin(trauma_classes)].copy(),
        "label",
        trauma_classes
    )
}

# ------------------------------------------
# 4. Model definitions
# ------------------------------------------

def build_model(name):

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    if name == "Logistic Regression":

        return make_pipeline(
            imputer,
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=SEED
            )
        )

    if name == "Random Forest":

        return make_pipeline(
            imputer,
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1
            )
        )

    if name == "LightGBM":

        return make_pipeline(
            imputer,
            LGBMClassifier(
                objective="multiclass",
                n_estimators=250,
                learning_rate=0.03,
                num_leaves=7,
                max_depth=4,
                min_child_samples=10,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1
            )
        )

    raise ValueError(name)

model_names = [
    "Logistic Regression",
    "Random Forest",
    "LightGBM"
]

# ------------------------------------------
# 5. Train and evaluate
# ------------------------------------------

results = []
class_results = []

print("\n=== LEAKAGE SCENARIO EXPERIMENTS ===")

for task_name, (task_df, target, class_order) in tasks.items():

    train = task_df[
        task_df["split"] == "train"
    ].copy()

    validation = task_df[
        task_df["split"] == "validation"
    ].copy()

    label_map = {
        label: i for i, label in enumerate(class_order)
    }

    y_train = train[target].map(label_map).to_numpy()
    y_val = validation[target].map(label_map).to_numpy()

    if pd.isna(y_train).any() or pd.isna(y_val).any():
        raise ValueError(f"Invalid labels in {task_name}")

    if set(y_train) != set(range(len(class_order))):
        raise ValueError(f"Missing training class in {task_name}")

    if set(y_val) != set(range(len(class_order))):
        raise ValueError(f"Missing validation class in {task_name}")

    print("\n====================================")
    print("TASK:", task_name)
    print("Training:", len(train))
    print("Validation:", len(validation))
    print("====================================")

    for scenario_name, columns in scenarios.items():

        X_train = (
            train[columns]
            .apply(pd.to_numeric, errors="raise")
            .replace([np.inf, -np.inf], np.nan)
            .astype(float)
            .to_numpy()
        )

        X_val = (
            validation[columns]
            .apply(pd.to_numeric, errors="raise")
            .replace([np.inf, -np.inf], np.nan)
            .astype(float)
            .to_numpy()
        )

        for model_name in model_names:

            model = build_model(model_name)

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            macro = f1_score(
                y_val,
                predictions,
                average="macro",
                labels=np.arange(len(class_order)),
                zero_division=0
            )

            accuracy = accuracy_score(
                y_val, predictions
            )

            balanced = balanced_accuracy_score(
                y_val, predictions
            )

            results.append({
                "task": task_name,
                "scenario": scenario_name,
                "model": model_name,
                "features": len(columns),
                "train_n": len(train),
                "validation_n": len(validation),
                "macro_f1": macro,
                "accuracy": accuracy,
                "balanced_accuracy": balanced
            })

            report = classification_report(
                y_val,
                predictions,
                labels=np.arange(len(class_order)),
                target_names=class_order,
                output_dict=True,
                zero_division=0
            )

            for label in class_order:
                class_results.append({
                    "task": task_name,
                    "scenario": scenario_name,
                    "model": model_name,
                    "class": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": report[label]["support"]
                })

            print(
                f"{scenario_name} | {model_name}: "
                f"Macro-F1={macro:.4f}, "
                f"Balanced Accuracy={balanced:.4f}"
            )

            # Save confusion matrices for all-70 scenario.
            if scenario_name == "S8_all_70":

                fig, ax = plt.subplots(figsize=(8, 6))

                ConfusionMatrixDisplay.from_predictions(
                    y_val,
                    predictions,
                    labels=np.arange(len(class_order)),
                    display_labels=class_order,
                    xticks_rotation=25,
                    ax=ax,
                    cmap="Blues"
                )

                ax.set_title(
                    f"{task_name} - {model_name} - Validation"
                )

                plt.tight_layout()

                filename = (
                    f"{task_name}_"
                    f"{model_name.lower().replace(' ', '_')}_"
                    "all70_confusion.png"
                )

                plt.savefig(
                    FIG / filename,
                    dpi=220
                )

                plt.close()

# ------------------------------------------
# 6. Save aggregate results
# ------------------------------------------

summary = pd.DataFrame(results)

summary.to_csv(
    OUT / "scenario_model_summary.csv",
    index=False
)

pd.DataFrame(class_results).to_csv(
    OUT / "scenario_class_metrics.csv",
    index=False
)

# ------------------------------------------
# 7. Generate one chart per task
# ------------------------------------------

for task_name in tasks:

    subset = summary[
        summary["task"] == task_name
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name in model_names:

        model_data = subset[
            subset["model"] == model_name
        ].set_index("scenario").reindex(
            list(scenarios.keys())
        )

        ax.plot(
            list(scenarios.keys()),
            model_data["macro_f1"],
            marker="o",
            label=model_name
        )

    ax.set_title(
        f"Feature Scenario Comparison: {task_name}"
    )

    ax.set_ylabel("Validation Macro-F1")
    ax.set_xlabel("Feature Scenario")
    ax.set_ylim(0, 1)

    ax.tick_params(
        axis="x",
        rotation=35
    )

    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        FIG / f"{task_name}_scenario_comparison.png",
        dpi=220
    )

    plt.close()

# ------------------------------------------
# 8. Final summary
# ------------------------------------------

print("\n=== TOP VALIDATION RESULTS BY TASK ===")

for task_name in tasks:

    print("\n", task_name)

    top = (
        summary[summary["task"] == task_name]
        .sort_values("macro_f1", ascending=False)
        .head(5)
    )

    print(
        top[
            ["scenario", "model", "features", "macro_f1",
             "balanced_accuracy"]
        ].to_string(index=False)
    )

print("\n=== OUTPUT FILES ===")

for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(path)

print("\nAll scenario experiments completed.")
print("Validation results only.")
print("The independent test partition was NOT evaluated.")
print("No hyperparameter tuning was performed.")
