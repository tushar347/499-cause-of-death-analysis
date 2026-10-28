from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

old = pd.read_csv("results/model_comparison_summary.csv")
new = pd.read_csv("results/age_ablation_results.csv")

settings = [
    ("Tier 1 LR", "Tier 1 (safe)",
     "Tier 1 without age", "Logistic Regression"),
    ("Tier 1 RF", "Tier 1 (safe)",
     "Tier 1 without age", "Random Forest"),
    ("Tier 1+2 LR", "Tier 1+2 (history)",
     "Tier 1+2 without age", "Logistic Regression"),
    ("Tier 1+2 RF", "Tier 1+2 (history)",
     "Tier 1+2 without age", "Random Forest")
]

labels = []
with_age = []
without_age = []

for label, original_set, ablated_set, model in settings:
    before = old.loc[
        (old["feature_set"] == original_set) &
        (old["model"] == model),
        "mean_macro_f1"
    ].iloc[0]

    after = new.loc[
        (new["feature_set"] == ablated_set) &
        (new["model"] == model),
        "mean_macro_f1"
    ].iloc[0]

    labels.append(label)
    with_age.append(before)
    without_age.append(after)

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - 0.2, with_age, 0.4, label="With age"
)
bars2 = ax.bar(
    x + 0.2, without_age, 0.4, label="Without age"
)

ax.bar_label(bars1, fmt="%.4f", padding=3)
ax.bar_label(bars2, fmt="%.4f", padding=3)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.43)
ax.set_ylabel("Mean Macro-F1")
ax.set_title("NMDID Age Ablation Experiment")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

output = Path.cwd() / "reports" / "figures" / "age_ablation_fixed.png"

try:
    fig.savefig(str(output), dpi=200)
except OSError:
    output = Path.home() / "reports" / "figures" / "age_ablation_fixed.png"
    fig.savefig(str(output), dpi=200)

plt.close(fig)

print("Chart saved to:", output)
