# Contributing

## Branching

- `main` is the stable, presentable branch. Changes reach it from `develop` through pull requests.
- `develop` is the integration branch. Feature work reaches it through pull requests.
- Create short-lived branches from `develop`, using one purpose per branch and one of these prefixes: `feature/<topic>`, `docs/<topic>`, `fix/<topic>`, or `experiment/<topic>`.
- Use squash merges or `--no-ff` merges. Do not commit directly to `main` or `develop` after the initial repository setup.
- Tag stable releases as `v0.1.0`, `v0.2.0`, and so on.

## Commit messages

Use Conventional Commit-style prefixes: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, and `exp:` for experiment results. Keep each commit focused and describe the user-visible or research change.

## Running scripts

Run commands from the repository root with the local metadata file at `data/metadata_feature_matrix.csv`. It is intentionally ignored by Git and is not included in this repository.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\run_tabpfn.py
python scripts\analyze_tabpfn.py
python scripts\compare_models.py
python scripts\age_analysis.py
python scripts\age_ablation.py
python scripts\fix_age_chart.py
```

The analysis scripts write aggregate outputs to `results/` and figures to `reports/figures/`. They may take time to finish, especially cross-validation experiments.

## Data safety

- Never commit person-level records or source exports, including `metadata_feature_matrix.csv`, NMDID CSV/ZIP exports, `cohort_*.csv`, or `nmdid_*.csv`.
- Never commit DICOM/NIfTI data, trained model files (`*.joblib`, `*.pkl`, `*.pt`), virtual environments, Python caches, `.env` files, or tokens.
- Only `data/README.md` belongs in the `data/` directory in Git.
- Aggregate, non-person-level result CSVs and figures may be committed. Before adding a CSV, verify that its rows are aggregate and not one row per decedent.
- Follow NMDID access terms and protect all restricted data. Do not include identifiers or protected data in issues, pull requests, logs, or reports.
