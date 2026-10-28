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

X = df[["age_years", "female"]].astype(float)
y = df["label"].map(
    {name: i for i, name in enumerate(classes)}
).to_numpy()

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=SEED
)

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED
        )
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        )
    )
}

results = []

print("\n=== AGE + SEX BASELINE ===")

for name, model in models.items():
    scores = []

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y[train_idx])
        predictions = model.predict(X.iloc[test_idx])

        score = f1_score(
            y[test_idx],
            predictions,
            average="macro",
            labels=np.arange(5),
            zero_division=0
        )

        scores.append(score)

    mean = np.mean(scores)
    std = np.std(scores)

    results.append({
        "model": name,
        "mean_macro_f1": mean,
        "std_macro_f1": std
    })

    print(f"{name}: {mean:.4f} +/- {std:.4f}")

pd.DataFrame(results).to_csv(
    "results/age_sex_baseline.csv", index=False
)

# Age distribution by cause-of-death group
age_summary = (
    df.groupby("label")["age_years"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

age_summary.to_csv("results/age_class_summary.csv")

print("\n=== AGE DISTRIBUTION ===")
print(age_summary)

fig, ax = plt.subplots(figsize=(10, 6))

age_data = [
    df.loc[df["label"] == c, "age_years"]
    .dropna().to_numpy()
    for c in classes
]

ax.boxplot(age_data, tick_labels=classes)

ax.set_title("Age Distribution by Cause of Death")
ax.set_ylabel("Age in Years")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig("reports/figures/age_by_class.png", dpi=250)
plt.close()

print("\nSaved:")
print("results/age_sex_baseline.csv")
print("results/age_class_summary.csv")
print("reports/figures/age_by_class.png")
print("\nExperiment completed.")
