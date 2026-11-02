"""SHAP, probability-calibration, and split-conformal analyses for selected models."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
SEED = 42
CLASS_ORDER = {
    "five_class": [
        "blunt_trauma",
        "firearm",
        "stab",
        "cardiac",
        "intox",
    ],
    "broad_three_class": ["trauma", "cardiac", "intoxication"],
    "trauma_subtype": ["blunt_trauma", "firearm", "stab"],
}
TASK_CONFIGS = {
    "five_class": {
        "scenario": "S7_all_history",
        "model": "Random Forest",
        "expected_macro_f1": 0.3698,
    },
    "broad_three_class": {
        "scenario": "S7_all_history",
        "model": "LightGBM",
        "expected_macro_f1": 0.5753,
    },
    "trauma_subtype": {
        "scenario": "S6_medications",
        "model": "LightGBM",
        "expected_macro_f1": 0.4684,
    },
}
BROAD_MAP = {
    "blunt_trauma": "trauma",
    "firearm": "trauma",
    "stab": "trauma",
    "cardiac": "cardiac",
    "intox": "intoxication",
}
RISKY_PREFIXES = ("dx_", "mh_", "su_", "env_", "med_")

SHAP_ROOT = ROOT / "protocol" / "shap"
CAL_ROOT = ROOT / "protocol" / "calibration"
CONFORMAL_ROOT = ROOT / "protocol" / "conformal"


def load_inputs():
    data_path = ROOT / "data" / "metadata_feature_matrix.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Metadata matrix not found: {data_path}")

    data = pd.read_csv(data_path, dtype={"id": "string", "label": "string"})
    assignments = pd.read_csv(
        ROOT / "protocol" / "split_assignments_local.csv",
        dtype={"id": "string"},
    )
    data["id"] = data["id"].str.strip()
    assignments["id"] = assignments["id"].str.strip()
    if data["id"].isna().any() or data["id"].duplicated().any():
        raise ValueError("Decedent IDs must be present and unique.")
    if assignments["id"].isna().any() or assignments["id"].duplicated().any():
        raise ValueError("Split assignment IDs must be present and unique.")
    if set(data["id"]) != set(assignments["id"]):
        raise ValueError("Metadata IDs and frozen split assignments do not match.")

    data = data.merge(assignments, on="id", how="left", validate="one_to_one")
    if data["split"].isna().any():
        raise ValueError("Some metadata rows have no frozen split assignment.")
    # Test records are excluded before any target or predictor processing.
    data = data.loc[data["split"].isin(["train", "validation"])].copy()

    summary = pd.read_csv(
        ROOT / "protocol" / "leakage_scenarios" / "scenario_model_summary.csv"
    )
    class_metrics = pd.read_csv(
        ROOT / "protocol" / "leakage_scenarios" / "scenario_class_metrics.csv"
    )
    manifest = pd.read_csv(
        ROOT / "protocol" / "leakage_scenarios" / "scenario_feature_manifest.csv"
    )

    tasks = {}
    for task, config in TASK_CONFIGS.items():
        selected = summary.loc[
            (summary["task"] == task)
            & (summary["scenario"] == config["scenario"])
            & (summary["model"] == config["model"])
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one verified result for {task}.")
        result = selected.iloc[0]
        if not np.isclose(
            float(result["macro_f1"]), config["expected_macro_f1"], atol=0.00006
        ):
            raise ValueError(f"Stored validation score does not match expected {task}.")
        features = manifest.loc[
            manifest["scenario"] == config["scenario"], "feature"
        ].tolist()
        if len(features) != int(result["features"]) or len(set(features)) != len(
            features
        ):
            raise ValueError(f"Feature manifest does not match stored result for {task}.")
        missing = sorted(set(features).difference(data.columns))
        if missing:
            raise ValueError(f"Manifest features missing from metadata for {task}: {missing}")

        stored_classes = class_metrics.loc[
            (class_metrics["task"] == task)
            & (class_metrics["scenario"] == config["scenario"])
            & (class_metrics["model"] == config["model"])
        ]
        classes = CLASS_ORDER[task]
        if set(stored_classes["class"]) != set(classes) or len(stored_classes) != len(
            classes
        ):
            raise ValueError(f"Stored per-class metrics do not match class order for {task}.")
        if not np.isclose(stored_classes["f1"].mean(), result["macro_f1"], atol=1e-10):
            raise ValueError(f"Stored per-class F1 does not reproduce macro-F1 for {task}.")
        tasks[task] = {
            **config,
            "features": features,
            "classes": classes,
            "stored_result": result,
        }
    return data, tasks


def task_frame(data, task):
    frame = data.copy()
    if task == "broad_three_class":
        frame["target"] = frame["label"].map(BROAD_MAP)
    elif task == "trauma_subtype":
        frame = frame.loc[frame["label"].isin(CLASS_ORDER[task])].copy()
        frame["target"] = frame["label"]
    else:
        frame["target"] = frame["label"]
    if frame["target"].isna().any():
        raise ValueError(f"Unexpected target labels in {task}.")
    return frame


def encode_labels(labels, classes):
    mapping = {name: index for index, name in enumerate(classes)}
    encoded = pd.Series(labels).map(mapping).to_numpy(dtype=int)
    if set(encoded) != set(range(len(classes))):
        raise ValueError("A required class is absent from a model-fitting subset.")
    return encoded


def numeric_features(frame, features):
    values = (
        frame[features]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    return values


def build_estimator(task, config):
    if task == "five_class":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=4,
        min_child_samples=10,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_preprocessor_and_model(frame, task, config, fit_mask):
    features = config["features"]
    classes = config["classes"]
    train = frame.loc[fit_mask]
    x_train = numeric_features(train, features)
    y_train = encode_labels(train["target"], classes)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train_transformed = imputer.fit_transform(x_train)
    estimator = build_estimator(task, config)
    estimator.fit(x_train_transformed, y_train)
    if not np.array_equal(estimator.classes_, np.arange(len(classes))):
        raise ValueError(f"Estimator class order differs from configured order in {task}.")
    return imputer, estimator


def predict_probabilities(frame, features, imputer, estimator):
    x = imputer.transform(numeric_features(frame, features))
    probabilities = estimator.predict_proba(x)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(estimator.classes_):
        raise ValueError("Model probabilities do not match the expected class dimension.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite probabilities.")
    return x, probabilities


def shap_values_array(explainer_output, n_rows, n_features, n_classes):
    values = explainer_output.values
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 2 and n_classes == 1:
        values = values[:, :, np.newaxis]
    if values.shape == (n_classes, n_rows, n_features):
        values = np.transpose(values, (1, 2, 0))
    elif values.shape == (n_rows, n_classes, n_features):
        values = np.transpose(values, (0, 2, 1))
    if values.shape != (n_rows, n_features, n_classes):
        raise ValueError(f"Unexpected SHAP dimensions: {values.shape}")
    return values


def run_shap(data, tasks):
    importance_dir = SHAP_ROOT / "shap_importance"
    figure_dir = SHAP_ROOT / "figures"
    importance_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    global_rows = []
    class_rows = []
    report_sections = [
        "# SHAP explainability results",
        "",
        "TreeSHAP explanations were computed for models fitted on the frozen training partition and evaluated on the previously used validation partition. Values explain model outputs, not causal effects. LightGBM TreeSHAP values are on the raw model-output scale; no feature is declared safe based on SHAP.",
        "",
    ]

    for task, config in tasks.items():
        frame = task_frame(data, task)
        train_mask = frame["split"].eq("train")
        val = frame.loc[frame["split"].eq("validation")]
        imputer, estimator = fit_preprocessor_and_model(
            frame, task, config, train_mask
        )
        x_val, _ = predict_probabilities(
            val, config["features"], imputer, estimator
        )
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(x_val)
        values = shap_values_array(
            explanation,
            len(val),
            len(config["features"]),
            len(config["classes"]),
        )
        mean_abs_class = np.abs(values).mean(axis=0)
        mean_abs_global = mean_abs_class.mean(axis=1)
        order = np.argsort(mean_abs_global)[::-1]

        for rank, feature_index in enumerate(order, start=1):
            global_rows.append(
                {
                    "task": task,
                    "scenario": config["scenario"],
                    "model": config["model"],
                    "feature": config["features"][feature_index],
                    "mean_abs_shap": float(mean_abs_global[feature_index]),
                    "rank": rank,
                    "validation_n": len(val),
                }
            )
        for class_index, class_name in enumerate(config["classes"]):
            class_order = np.argsort(mean_abs_class[:, class_index])[::-1]
            for rank, feature_index in enumerate(class_order, start=1):
                class_rows.append(
                    {
                        "task": task,
                        "scenario": config["scenario"],
                        "model": config["model"],
                        "class": class_name,
                        "feature": config["features"][feature_index],
                        "mean_abs_shap": float(
                            mean_abs_class[feature_index, class_index]
                        ),
                        "rank": rank,
                        "validation_n": len(val),
                    }
                )

        # Overall global importance bar plot.
        top = order[: min(20, len(order))][::-1]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(
            [config["features"][index] for index in top],
            mean_abs_global[top],
            color="#4776a8",
        )
        ax.set_xlabel("Mean absolute TreeSHAP value (averaged across classes)")
        ax.set_title(f"{task}: global feature importance")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task}_global_importance.png", dpi=220)
        plt.close(fig)

        # Output-specific beeswarms preserve the meaning of each class output.
        for class_index, class_name in enumerate(config["classes"]):
            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                values[:, :, class_index],
                x_val,
                feature_names=config["features"],
                max_display=min(20, len(config["features"])),
                show=False,
                plot_type="dot",
            )
            plt.title(f"{task}: SHAP summary for {class_name}")
            plt.tight_layout()
            plt.savefig(
                figure_dir / f"{task}_shap_beeswarm_{class_name}.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close()

        # Dependence plots use the class output most influenced by each feature.
        for feature_index in order[: min(3, len(order))]:
            class_index = int(np.argmax(mean_abs_class[feature_index]))
            feature_values = x_val[:, feature_index]
            shap_feature_values = values[:, feature_index, class_index]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(feature_values, shap_feature_values, alpha=0.65, s=22)
            ax.axhline(0, color="grey", linewidth=1, linestyle="--")
            ax.set_xlabel(config["features"][feature_index])
            ax.set_ylabel(f"SHAP value for {config['classes'][class_index]}")
            ax.set_title(f"{task}: dependence on {config['features'][feature_index]}")
            fig.tight_layout()
            safe_name = (
                config["features"][feature_index]
                .strip()
                .replace(" ", "_")
                .replace("/", "_")
                .replace(":", "")
            )
            fig.savefig(
                figure_dir / f"{task}_dependence_{safe_name}.png", dpi=220
            )
            plt.close(fig)

        ranked = [config["features"][index] for index in order[:5]]
        risky_importance = sum(
            mean_abs_global[index]
            for index, feature in enumerate(config["features"])
            if feature.startswith(RISKY_PREFIXES) or feature == "n_meds"
        )
        total_importance = float(mean_abs_global.sum())
        risky_share = (
            100 * risky_importance / total_importance if total_importance else 0
        )
        report_sections.extend(
            [
                f"## {task.replace('_', ' ').title()}",
                "",
                f"- Model/scenario: {config['model']} / {config['scenario']} ({len(config['features'])} features); fitted on {int(train_mask.sum())} training records and explained on {len(val)} validation records.",
                f"- Top global features by mean absolute TreeSHAP: {', '.join(ranked)}.",
                f"- Features from diagnosis/history/substance/lifestyle/medication groups account for {risky_share:.1f}% of the aggregated mean absolute importance.",
                "- This importance share is descriptive model reliance; it does not establish feature provenance, timing, safety, or causality.",
                "",
            ]
        )
        for class_index, class_name in enumerate(config["classes"]):
            class_top = np.argsort(mean_abs_class[:, class_index])[::-1][:3]
            names = [config["features"][index] for index in class_top]
            report_sections.append(
                f"- **{class_name}:** strongest class-output features: {', '.join(names)}."
            )
        report_sections.append("")
        print(
            f"SHAP completed: {task}, {len(val)} validation records, "
            f"{len(config['features'])} features, {len(config['classes'])} outputs"
        )

    global_importance = pd.DataFrame(global_rows)
    class_importance = pd.DataFrame(class_rows)
    global_importance.to_csv(
        importance_dir / "global_feature_importance.csv", index=False
    )
    class_importance.to_csv(
        importance_dir / "class_feature_importance.csv", index=False
    )
    for task in tasks:
        global_importance.loc[global_importance["task"] == task].to_csv(
            importance_dir / f"{task}_global_importance.csv", index=False
        )
        class_importance.loc[class_importance["task"] == task].to_csv(
            importance_dir / f"{task}_class_importance.csv", index=False
        )
    (SHAP_ROOT / "shap_report.md").write_text(
        "\n".join(report_sections), encoding="utf-8"
    )


def make_training_partitions(data):
    train = data.loc[data["split"].eq("train")].copy()
    if train["id"].duplicated().any():
        raise ValueError("Training decedent IDs are not unique.")
    # Stable ID ordering makes the seeded, stratified partition reproducible.
    train = train.sort_values("id", kind="stable").reset_index(drop=True)
    fit_ids, heldout_ids = train_test_split(
        train["id"],
        test_size=0.4,
        random_state=SEED,
        stratify=train["label"],
    )
    heldout = train.loc[train["id"].isin(heldout_ids)].sort_values(
        "id", kind="stable"
    )
    probability_ids, conformal_ids = train_test_split(
        heldout["id"],
        test_size=0.5,
        random_state=SEED + 1,
        stratify=heldout["label"],
    )
    partitions = {
        "model_fit": set(fit_ids),
        "probability_calibration": set(probability_ids),
        "conformal_calibration": set(conformal_ids),
    }
    all_partition_ids = set().union(*partitions.values())
    if len(all_partition_ids) != len(train) or sum(
        len(ids) for ids in partitions.values()
    ) != len(train):
        raise ValueError("New training subsets overlap or omit training decedents.")
    if all_partition_ids.intersection(set(data.loc[data["split"] != "train", "id"])):
        raise ValueError("New calibration subsets overlap the existing validation set.")
    return train, partitions


def task_partition_rows(train, partitions):
    rows = []
    for task in TASK_CONFIGS:
        frame = task_frame(train, task)
        for partition, ids in partitions.items():
            subset = frame.loc[frame["id"].isin(ids)]
            counts = subset["target"].value_counts()
            for class_name in CLASS_ORDER[task]:
                rows.append(
                    {
                        "task": task,
                        "partition": partition,
                        "class": class_name,
                        "count": int(counts.get(class_name, 0)),
                        "partition_total": len(subset),
                        "class_percentage": (
                            100 * counts.get(class_name, 0) / len(subset)
                            if len(subset)
                            else 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def fit_calibration_model(frame, task, config, partitions):
    train_frame = frame.loc[frame["id"].isin(partitions["model_fit"])]
    temperature_frame = frame.loc[
        frame["id"].isin(partitions["probability_calibration"])
    ]
    conformal_frame = frame.loc[
        frame["id"].isin(partitions["conformal_calibration"])
    ]
    imputer, estimator = fit_preprocessor_and_model(
        frame,
        task,
        config,
        frame["id"].isin(partitions["model_fit"]),
    )
    _, temperature_probabilities = predict_probabilities(
        temperature_frame, config["features"], imputer, estimator
    )
    temperature_labels = encode_labels(temperature_frame["target"], config["classes"])
    temperature = fit_temperature(temperature_probabilities, temperature_labels)
    return {
        "imputer": imputer,
        "estimator": estimator,
        "temperature": temperature,
        "temperature_frame": temperature_frame,
        "conformal_frame": conformal_frame,
    }


def apply_temperature(probabilities, temperature):
    eps = np.finfo(float).eps
    logits = np.log(np.clip(probabilities, eps, 1.0))
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    calibrated = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
    return calibrated


def fit_temperature(probabilities, labels):
    def objective(log_temperature):
        temperature = float(np.exp(log_temperature))
        scaled = apply_temperature(probabilities, temperature)
        return log_loss(
            labels,
            scaled,
            labels=np.arange(probabilities.shape[1]),
        )

    result = minimize_scalar(
        objective,
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def multiclass_brier(labels, probabilities):
    one_hot = np.eye(probabilities.shape[1])[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(labels, probabilities, bins=10):
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows = []
    for bin_index in range(bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        if bin_index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += count / len(labels) * abs(accuracy - mean_confidence)
        else:
            mean_confidence = np.nan
            accuracy = np.nan
        rows.append(
            {
                "bin": bin_index + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return float(ece), rows


def class_reliability_bins(labels, probabilities, class_index, bins=10):
    class_probability = probabilities[:, class_index]
    observed = (labels == class_index).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for bin_index in range(bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        if bin_index == bins - 1:
            mask = (class_probability >= lower) & (class_probability <= upper)
        else:
            mask = (class_probability >= lower) & (class_probability < upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin": bin_index + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": (
                    float(class_probability[mask].mean()) if count else np.nan
                ),
                "accuracy": float(observed[mask].mean()) if count else np.nan,
            }
        )
    return rows


def calibration_metrics(labels, probabilities):
    ece, bins = expected_calibration_error(labels, probabilities)
    return {
        "brier_score": multiclass_brier(labels, probabilities),
        "log_loss": float(
            log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))
        ),
        "ece_10_bins": ece,
    }, bins


def aggregate_probabilities(task, classes, labels, before, after):
    rows = []
    for true_index, true_class in enumerate(classes):
        mask = labels == true_index
        for probability_index, probability_class in enumerate(classes):
            rows.append(
                {
                    "task": task,
                    "partition": "validation",
                    "true_class": true_class,
                    "probability_class": probability_class,
                    "support": int(mask.sum()),
                    "mean_probability_before": float(before[mask, probability_index].mean()),
                    "mean_probability_after": float(after[mask, probability_index].mean()),
                }
            )
    return rows


def plot_reliability(task, classes, before_bins, after_bins, directory):
    fig, axes = plt.subplots(
        1, len(classes), figsize=(5 * len(classes), 4), squeeze=False
    )
    for class_index, class_name in enumerate(classes):
        ax = axes[0, class_index]
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Ideal")
        for name, all_bins, color in (
            ("Before", before_bins, "#4776a8"),
            ("Temperature scaled", after_bins, "#d17a22"),
        ):
            class_bins = all_bins[class_index]
            valid = [row for row in class_bins if row["count"] > 0]
            if valid:
                ax.plot(
                    [row["mean_confidence"] for row in valid],
                    [row["accuracy"] for row in valid],
                    marker="o",
                    label=name,
                    color=color,
                )
        ax.set_title(class_name)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Observed accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle(f"{task}: validation reliability diagrams")
    fig.tight_layout()
    fig.savefig(directory / f"{task}_reliability.png", dpi=220)
    plt.close(fig)


def run_calibration(data, tasks):
    result_dir = CAL_ROOT / "results"
    figure_dir = CAL_ROOT / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    train, partitions = make_training_partitions(data)
    split_summary = task_partition_rows(train, partitions)
    split_summary.to_csv(result_dir / "training_subset_class_counts.csv", index=False)
    metric_rows = []
    temperature_rows = []
    reliability_rows = []
    aggregate_rows = []
    for task, config in tasks.items():
        frame = task_frame(data, task)
        fitted = fit_calibration_model(frame, task, config, partitions)
        validation = frame.loc[frame["split"].eq("validation")]
        labels = encode_labels(validation["target"], config["classes"])
        _, before = predict_probabilities(
            validation, config["features"], fitted["imputer"], fitted["estimator"]
        )
        after = apply_temperature(before, fitted["temperature"])
        for method, probabilities in (("original", before), ("temperature", after)):
            metrics, _ = calibration_metrics(labels, probabilities)
            metric_rows.append(
                {
                    "task": task,
                    "partition": "validation",
                    "method": method,
                    "n": len(validation),
                    **metrics,
                }
            )
            for class_index, class_name in enumerate(config["classes"]):
                bins = class_reliability_bins(labels, probabilities, class_index)
                for bin_row in bins:
                    reliability_rows.append(
                        {
                            "task": task,
                            "method": method,
                            "class": class_name,
                            **bin_row,
                        }
                    )
        before_bins = [
            class_reliability_bins(labels, before, class_index)
            for class_index in range(len(config["classes"]))
        ]
        after_bins = [
            class_reliability_bins(labels, after, class_index)
            for class_index in range(len(config["classes"]))
        ]
        # The diagrams are one-vs-rest and use probabilities for the indicated class.
        plot_reliability(task, config["classes"], before_bins, after_bins, figure_dir)
        aggregate_rows.extend(
            aggregate_probabilities(
                task, config["classes"], labels, before, after
            )
        )
        temperature_rows.append(
            {
                "task": task,
                "scenario": config["scenario"],
                "model": config["model"],
                "temperature": fitted["temperature"],
                "model_fit_n": len(partitions["model_fit"]),
                "probability_calibration_n": len(fitted["temperature_frame"]),
                "conformal_calibration_n": len(fitted["conformal_frame"]),
                "validation_n": len(validation),
            }
        )
        print(
            f"Calibration completed: {task}, T={fitted['temperature']:.4f}, "
            f"validation n={len(validation)}"
        )

    pd.DataFrame(metric_rows).to_csv(result_dir / "calibration_metrics.csv", index=False)
    pd.DataFrame(temperature_rows).to_csv(
        result_dir / "temperature_parameters.csv", index=False
    )
    pd.DataFrame(reliability_rows).to_csv(
        result_dir / "reliability_bins.csv", index=False
    )
    pd.DataFrame(aggregate_rows).to_csv(
        result_dir / "validation_probability_aggregates.csv", index=False
    )


def finite_sample_quantile(scores, alpha):
    if not 0 < alpha < 1:
        raise ValueError("Conformal alpha must be between zero and one.")
    sorted_scores = np.sort(np.asarray(scores, dtype=float))
    if not len(sorted_scores):
        raise ValueError("Conformal calibration subset is empty.")
    rank = math.ceil((len(sorted_scores) + 1) * (1 - alpha))
    if rank > len(sorted_scores):
        return float("inf"), rank
    return float(sorted_scores[rank - 1]), rank


def run_conformal(data, tasks):
    result_dir = CONFORMAL_ROOT / "results"
    figure_dir = CONFORMAL_ROOT / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    train, partitions = make_training_partitions(data)
    overall_rows = []
    class_rows = []
    size_rows = []
    set_composition_rows = []

    for task, config in tasks.items():
        frame = task_frame(data, task)
        fitted = fit_calibration_model(frame, task, config, partitions)
        calibration = fitted["conformal_frame"]
        validation = frame.loc[frame["split"].eq("validation")]
        calibration_labels = encode_labels(calibration["target"], config["classes"])
        validation_labels = encode_labels(validation["target"], config["classes"])
        _, calibration_raw = predict_probabilities(
            calibration, config["features"], fitted["imputer"], fitted["estimator"]
        )
        calibration_probabilities = apply_temperature(
            calibration_raw, fitted["temperature"]
        )
        _, validation_raw = predict_probabilities(
            validation, config["features"], fitted["imputer"], fitted["estimator"]
        )
        validation_probabilities = apply_temperature(
            validation_raw, fitted["temperature"]
        )
        calibration_scores = 1 - calibration_probabilities[
            np.arange(len(calibration_labels)), calibration_labels
        ]

        for target_coverage in (0.90, 0.80):
            alpha = 1 - target_coverage
            quantile, rank = finite_sample_quantile(calibration_scores, alpha)
            included = (1 - validation_probabilities) <= quantile
            set_sizes = included.sum(axis=1)
            covered = included[np.arange(len(validation_labels)), validation_labels]
            overall_rows.append(
                {
                    "task": task,
                    "target_coverage": target_coverage,
                    "alpha": alpha,
                    "validation_n": len(validation),
                    "conformal_calibration_n": len(calibration),
                    "finite_sample_rank_1_based": rank,
                    "conformal_quantile": quantile,
                    "empirical_coverage": float(covered.mean()),
                    "average_set_size": float(set_sizes.mean()),
                    "singleton_percentage": float(100 * (set_sizes == 1).mean()),
                    "abstention_percentage": float(100 * (set_sizes != 1).mean()),
                    "empty_set_percentage": float(100 * (set_sizes == 0).mean()),
                    "multi_class_set_percentage": float(100 * (set_sizes > 1).mean()),
                    "temperature": fitted["temperature"],
                }
            )
            for class_index, class_name in enumerate(config["classes"]):
                mask = validation_labels == class_index
                class_rows.append(
                    {
                        "task": task,
                        "target_coverage": target_coverage,
                        "class": class_name,
                        "support": int(mask.sum()),
                        "empirical_coverage": (
                            float(covered[mask].mean()) if mask.any() else np.nan
                        ),
                    }
                )
            for set_size in range(len(config["classes"]) + 1):
                size_rows.append(
                    {
                        "task": task,
                        "target_coverage": target_coverage,
                        "set_size": set_size,
                        "count": int((set_sizes == set_size).sum()),
                        "validation_n": len(validation),
                    }
                )
            set_names = [
                "+".join(
                    class_name
                    for class_index, class_name in enumerate(config["classes"])
                    if included[row_index, class_index]
                )
                or "empty"
                for row_index in range(len(validation))
            ]
            set_counts = pd.Series(set_names).value_counts()
            for set_name, count in set_counts.items():
                set_composition_rows.append(
                    {
                        "task": task,
                        "target_coverage": target_coverage,
                        "prediction_set": set_name,
                        "count": int(count),
                        "validation_n": len(validation),
                    }
                )

        print(
            f"Conformal completed: {task}, calibration n={len(calibration)}, "
            f"validation n={len(validation)}"
        )

    overall = pd.DataFrame(overall_rows)
    overall.to_csv(result_dir / "prediction_set_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(
        result_dir / "per_class_coverage.csv", index=False
    )
    sizes = pd.DataFrame(size_rows)
    sizes.to_csv(result_dir / "prediction_set_size_counts.csv", index=False)
    compositions = pd.DataFrame(set_composition_rows)
    compositions.to_csv(
        result_dir / "prediction_set_composition_counts.csv", index=False
    )
    plot_prediction_set_sizes(sizes, figure_dir)
    (CONFORMAL_ROOT / "methodology.md").write_text(
        "# Split conformal prediction\n\n"
        "The model and temperature scaler are fitted using disjoint, stratified subsets of the original training partition. A third, disjoint training subset supplies nonconformity scores `1 - p(true class)`. For nominal coverage `1 - alpha`, the threshold is the `ceil((n + 1) * (1 - alpha))` order statistic, with an infinite threshold if the rank exceeds the calibration sample size. A class is included when `1 - calibrated_probability <= threshold`.\n\n"
        "Reported empirical coverage and set sizes are evaluated on the existing validation partition. That partition was already used to compare 81 configurations and select these models, so these validation results are descriptive and must not be presented as an independent coverage guarantee. The independent test partition was not evaluated. Standard split conformal prediction targets marginal coverage under exchangeability assumptions; it does not guarantee coverage for each individual class. Small per-class calibration counts, especially for trauma subtype, make classwise estimates imprecise.\n\n"
        "Prediction sets are summarized without saving person-level identifiers or per-record predictions. Abstention is defined as a set whose size is not one; empty and multi-class sets are also reported separately.\n",
        encoding="utf-8",
    )


def plot_prediction_set_sizes(sizes, directory):
    tasks = list(TASK_CONFIGS)
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), squeeze=False)
    for task_index, task in enumerate(tasks):
        ax = axes[0, task_index]
        subset = sizes.loc[sizes["task"] == task]
        labels = sorted(subset["set_size"].unique())
        width = 0.36
        for coverage_index, coverage in enumerate((0.90, 0.80)):
            values = (
                subset.loc[subset["target_coverage"] == coverage]
                .set_index("set_size")["count"]
                .reindex(labels, fill_value=0)
            )
            positions = np.arange(len(labels)) + (coverage_index - 0.5) * width
            ax.bar(
                positions,
                values,
                width=width,
                label=f"{int(coverage * 100)}% target",
            )
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_xlabel("Prediction-set size")
        ax.set_ylabel("Validation record count")
        ax.set_title(task.replace("_", " "))
        ax.legend()
    fig.suptitle("Conformal prediction-set size distributions")
    fig.tight_layout()
    fig.savefig(directory / "prediction_set_size_distribution.png", dpi=220)
    plt.close(fig)


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def write_supervisor_summary(tasks):
    demo_dir = ROOT / "protocol" / "supervisor_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    shap_global = pd.read_csv(
        SHAP_ROOT / "shap_importance" / "global_feature_importance.csv"
    )
    shap_class = pd.read_csv(
        SHAP_ROOT / "shap_importance" / "class_feature_importance.csv"
    )
    split_counts = pd.read_csv(
        CAL_ROOT / "results" / "training_subset_class_counts.csv"
    )
    calibration = pd.read_csv(CAL_ROOT / "results" / "calibration_metrics.csv")
    temperatures = pd.read_csv(
        CAL_ROOT / "results" / "temperature_parameters.csv"
    ).set_index("task")
    conformal = pd.read_csv(
        CONFORMAL_ROOT / "results" / "prediction_set_metrics.csv"
    )

    combined_rows = []
    summary_sections = [
        "# Supervisor demonstration: metadata reliability stage",
        "",
        "## Scope and status",
        "",
        "This exploratory development stage used the frozen training and validation partitions only. The independent 129-record test partition was not evaluated. The selected configurations were chosen from 81 prior comparisons on validation, so all validation metrics below are descriptive, not independent estimates of generalization or conformal coverage.",
        "",
        "The models are metadata-only: five-class Random Forest/S7_all_history, broad three-class LightGBM/S7_all_history, and trauma-subtype LightGBM/S6_medications. Their prior validation macro-F1 values were 0.3698, 0.5753, and 0.4684, respectively. All history, substance, lifestyle, and medication fields remain subject to source/timing verification.",
        "",
        "## SHAP summary",
        "",
    ]
    shap_rows = []
    for task, config in tasks.items():
        global_task = shap_global.loc[shap_global["task"] == task]
        top_features = global_task.head(5)["feature"].tolist()
        risky = global_task.loc[
            global_task["feature"].str.startswith(
                RISKY_PREFIXES, na=False
            )
            | global_task["feature"].eq("n_meds"),
            "mean_abs_shap",
        ].sum()
        total = global_task["mean_abs_shap"].sum()
        risky_share = 100 * risky / total if total else 0
        shap_rows.append(
            (
                task,
                config["model"],
                ", ".join(top_features),
                f"{risky_share:.1f}%",
            )
        )
    summary_sections.extend(
        [
            markdown_table(
                ["Task", "Model", "Top global features", "Unverified-group importance"],
                shap_rows,
            ),
            "",
            "SHAP describes model reliance, not causal effects or feature safety. Class-output rankings and beeswarm/dependence plots are in `../shap/`; `med_na` and `n_meds` require particular interpretation against the source data dictionary.",
            "",
            "### Strongest class-specific features",
            "",
        ]
    )
    class_rows = []
    for task in tasks:
        subset = shap_class.loc[shap_class["task"] == task]
        for class_name in CLASS_ORDER[task]:
            top = subset.loc[subset["class"] == class_name].head(3)["feature"]
            class_rows.append((task, class_name, ", ".join(top.tolist())))
    summary_sections.extend(
        [
            markdown_table(["Task", "Class output", "Top three SHAP features"], class_rows),
            "",
            "## Training-only calibration split",
            "",
            "The original training records were stratified by original five-class label after sorting by decedent ID. The fixed seed is 42 (the second split uses 43). Each ID occurs once in this cohort; the implementation checks uniqueness, no overlap, complete partitioning, and separation from validation. Only aggregate counts are saved.",
            "",
        ]
    )
    five_class_counts = split_counts.loc[split_counts["task"] == "five_class"]
    split_rows = []
    for partition in (
        "model_fit",
        "probability_calibration",
        "conformal_calibration",
    ):
        part = five_class_counts.loc[
            five_class_counts["partition"] == partition
        ]
        split_rows.append(
            (
                partition,
                int(part["partition_total"].iloc[0]),
                ", ".join(
                    f"{row['class']}={int(row['count'])}"
                    for _, row in part.iterrows()
                ),
            )
        )
    summary_sections.extend(
        [
            markdown_table(["Subset", "N", "Original class counts"], split_rows),
            "",
            "For trauma subtype, the same grouped split yields 122 model-fit, 41 probability-calibration, and 41 conformal-calibration records (class counts are in `../calibration/results/training_subset_class_counts.csv`).",
            "",
            "## Probability calibration",
            "",
            "Temperature was fitted by minimizing multiclass negative log-likelihood on the probability-calibration subset only. Brier score is the multiclass sum-of-squared-probability error averaged over records; ECE uses 10 equal-width bins and top-label confidence. Validation metrics are summarized below.",
            "",
        ]
    )
    calibration_rows = []
    for task in tasks:
        rows = calibration.loc[calibration["task"] == task].set_index("method")
        temp = temperatures.loc[task]
        calibration_rows.append(
            (
                task,
                f"{temp['temperature']:.3f}",
                f"{rows.loc['original', 'brier_score']:.3f} → {rows.loc['temperature', 'brier_score']:.3f}",
                f"{rows.loc['original', 'log_loss']:.3f} → {rows.loc['temperature', 'log_loss']:.3f}",
                f"{rows.loc['original', 'ece_10_bins']:.3f} → {rows.loc['temperature', 'ece_10_bins']:.3f}",
            )
        )
    summary_sections.extend(
        [
            markdown_table(
                ["Task", "Temperature", "Brier before → after", "Log loss before → after", "ECE before → after"],
                calibration_rows,
            ),
            "",
            "Temperature scaling improved validation Brier score and log loss in all three tasks. ECE decreased for five-class and trauma subtype but increased for broad three-class; calibration is therefore not uniformly improved across metrics. The small trauma-subtype calibration subset (41 records) makes its temperature estimate especially uncertain.",
            "",
            "## Split-conformal prediction sets",
            "",
            "Thresholds use calibrated probabilities on the separate training-derived conformal subset and the finite-sample order statistic. Coverage and set sizes below are empirical on the reused validation partition and are descriptive only.",
            "",
        ]
    )
    conformal_rows = []
    for _, row in conformal.iterrows():
        conformal_rows.append(
            (
                row["task"],
                f"{row['target_coverage']:.0%}",
                f"{row['empirical_coverage']:.1%}",
                f"{row['average_set_size']:.2f}",
                f"{row['singleton_percentage']:.1f}%",
                f"{row['abstention_percentage']:.1f}%",
            )
        )
        combined_rows.append(
            {
                "task": row["task"],
                "model": tasks[row["task"]]["model"],
                "scenario": tasks[row["task"]]["scenario"],
                "features": len(tasks[row["task"]]["features"]),
                "selected_validation_macro_f1": float(
                    tasks[row["task"]]["stored_result"]["macro_f1"]
                ),
                "target_coverage": float(row["target_coverage"]),
                "empirical_validation_coverage": float(row["empirical_coverage"]),
                "average_prediction_set_size": float(row["average_set_size"]),
                "singleton_percentage": float(row["singleton_percentage"]),
                "abstention_percentage": float(row["abstention_percentage"]),
                "conformal_calibration_n": int(row["conformal_calibration_n"]),
                "temperature": float(row["temperature"]),
            }
        )
    summary_sections.extend(
        [
            markdown_table(
                ["Task", "Target", "Validation coverage", "Mean set size", "Singleton", "Abstention"],
                conformal_rows,
            ),
            "",
            "Standard split conformal prediction targets marginal coverage under exchangeability assumptions; it does not ensure per-class coverage. Per-class results, set-size counts, calibration thresholds, and figures are in `../conformal/`. The large set sizes and low singleton rates show that these models often cannot support a single-class output at the selected nominal coverages.",
            "",
            "## Limitations and next steps",
            "",
            "- Do not treat validation metrics as independent confirmation: these models were selected using the same validation partition.",
            "- The test partition remains untouched. Freeze this protocol and perform one prespecified test evaluation only after provenance review and supervisory approval.",
            "- The training-derived temperature and conformal samples are modest; the trauma task has only 9 firearm cases in each calibration subset. ECE and class-conditional estimates are particularly noisy.",
            "- Conformal marginal coverage depends on exchangeability and does not guarantee class-conditional or individual-record coverage.",
            "- SHAP importance does not establish safety, timing, or causation. Verify source/provenance of medication, history, substance-use, and lifestyle variables before any clinical/forensic interpretation.",
            "- Next research stage: complete feature provenance review, preserve this untouched test set for a single locked evaluation, then compare metadata with Merlin PMCT embeddings and prespecified multimodal fusion.",
            "",
            "## Reproduction",
            "",
            "From the repository root, activate `.venv` and run each stage in order:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe run_reliability_stage.py --stage shap",
            ".\\.venv\\Scripts\\python.exe run_reliability_stage.py --stage calibration",
            ".\\.venv\\Scripts\\python.exe run_reliability_stage.py --stage conformal",
            ".\\.venv\\Scripts\\python.exe run_reliability_stage.py --stage summary",
            "```",
            "",
            "The script verifies the selected configurations against the saved scenario summary, per-class metrics, and feature manifest. Only aggregate outputs are written; no decedent-level identifiers, probabilities, or prediction sets are exported.",
            "",
        ]
    )
    pd.DataFrame(combined_rows).to_csv(
        demo_dir / "supervisor_demo_metrics.csv", index=False
    )
    (demo_dir / "supervisor_demo_summary.md").write_text(
        "\n".join(summary_sections), encoding="utf-8"
    )
    print("Supervisor demonstration summary generated.")


def verify_outputs(stage):
    if stage == "shap":
        expected = [
            SHAP_ROOT / "shap_importance" / "global_feature_importance.csv",
            SHAP_ROOT / "shap_importance" / "class_feature_importance.csv",
            SHAP_ROOT / "shap_report.md",
        ]
        figure_dir = SHAP_ROOT / "figures"
        expected_figures = 3 + sum(len(classes) for classes in CLASS_ORDER.values()) + 9
        files = list(figure_dir.glob("*.png"))
        if len(files) < expected_figures:
            raise RuntimeError(
                f"Expected at least {expected_figures} SHAP figures, found {len(files)}."
            )
    elif stage == "calibration":
        expected = [
            CAL_ROOT / "results" / "training_subset_class_counts.csv",
            CAL_ROOT / "results" / "calibration_metrics.csv",
            CAL_ROOT / "results" / "temperature_parameters.csv",
            CAL_ROOT / "results" / "reliability_bins.csv",
            CAL_ROOT / "results" / "validation_probability_aggregates.csv",
        ]
        files = list((CAL_ROOT / "figures").glob("*.png"))
        if len(files) != len(TASK_CONFIGS):
            raise RuntimeError(f"Expected 3 reliability figures, found {len(files)}.")
    elif stage == "conformal":
        expected = [
            CONFORMAL_ROOT / "results" / "prediction_set_metrics.csv",
            CONFORMAL_ROOT / "results" / "per_class_coverage.csv",
            CONFORMAL_ROOT / "results" / "prediction_set_size_counts.csv",
            CONFORMAL_ROOT / "results" / "prediction_set_composition_counts.csv",
            CONFORMAL_ROOT / "methodology.md",
        ]
        files = list((CONFORMAL_ROOT / "figures").glob("*.png"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one conformal figure, found {len(files)}.")
    else:
        expected = [
            ROOT / "protocol" / "supervisor_demo" / "supervisor_demo_metrics.csv",
            ROOT / "protocol" / "supervisor_demo" / "supervisor_demo_summary.md",
        ]
    missing = [path for path in expected if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty stage outputs: {missing}")
    print(f"Output verification passed for {stage}: {len(expected)} tables/reports")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("shap", "calibration", "conformal", "summary"),
        required=True,
    )
    args = parser.parse_args()
    data, tasks = load_inputs()
    if args.stage == "shap":
        run_shap(data, tasks)
    elif args.stage == "calibration":
        run_calibration(data, tasks)
    elif args.stage == "conformal":
        run_conformal(data, tasks)
    else:
        write_supervisor_summary(tasks)
    verify_outputs(args.stage)


if __name__ == "__main__":
    main()
