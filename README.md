# airbnb

**Airbnb listing price prediction**

Predicts listing prices from location, capacity, room type, property type, and review/availability attributes. The supplied notebook includes Los Angeles-area listing examples. This package adapts its second, MAE-focused workflow with native categorical boosting and validation-selected blending.

## Recorded notebook findings

The notebook's final CatBoost/LightGBM blend achieved **MAE 102.93067**,
**RMSE 400.27770**, and **R² 0.57826** on 7,421 labeled holdout listings.
It improved MAE by 1.51% over the earlier ensemble, with substantial remaining
underprediction of expensive listings. V2 reused an already examined holdout;
these are recorded notebook results, not independent external-test scores or
newly measured results from the modular package.

See [results and findings](data/README.md) for feature engineering, model
comparisons, blend weights, price-band errors, and evaluation limitations.
The [source notebook](airbnb.ipynb) is included for reference.

## Run the Python files

This project follows the `freight-rate-ml-assessment` layout: runnable `.py` files
in `scripts/`, reusable modules in `src/airbnb/`, CSV inputs in `data/`, and
models/reports in `outputs/`. Run the following commands from this project folder.

Python 3.11 or newer is required. Install dependencies, then put your original
training CSV at `data/listings.csv` and prediction rows at `data/new_rows.csv`.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS instead: source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/run_eda.py
python scripts/run_train.py
python scripts/run_predict.py
python scripts/build_report.py
```

The scripts import this folder's `src/` directly. An editable package install and
notebook execution are not required. The source CSVs were not supplied, so real
data must be added before running these default commands.

| File | What it does |
| --- | --- |
| `scripts/run_eda.py` | Saves data summaries, missing values, correlations, target associations, and figures. |
| `scripts/run_train.py` | Performs cross-validation, selects the model, evaluates holdout, and exports it. |
| `scripts/run_predict.py` | Loads the fitted export and predicts directly from raw CSV rows. |
| `scripts/build_report.py` | Builds `REPORT.md` and `REPORT.html` from a completed training run. |
| `scripts/make_demo_data.py` | Generates synthetic rows for checking script execution. |

## Folder structure

```text
airbnb/
├── scripts/
│   ├── run_eda.py
│   ├── run_train.py
│   ├── run_predict.py
│   ├── build_report.py
│   └── make_demo_data.py
├── src/airbnb/
│   ├── config.py       # paths, settings, default/full model profiles
│   ├── data.py         # CSV loading, validation, target parsing
│   ├── schema.py       # required raw input columns
│   ├── features.py     # feature engineering and preprocessing
│   ├── models.py       # model families, pipelines, search spaces
│   ├── validation.py   # splits, metrics, blend selection
│   ├── pipeline.py     # training, evaluation, model export
│   ├── predict.py      # saved-model inference
│   ├── artifacts.py    # persistence and model metadata
│   ├── eda.py          # analysis tables and plots
│   ├── report.py       # readable reports from saved results
│   ├── demo.py         # synthetic example generator
│   └── __init__.py
├── data/
├── outputs/
├── examples/predict.csv
├── tests/
├── .github/workflows/ci.yml
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Training settings

Edit `src/airbnb/config.py` to change defaults or pass command-line overrides.
The `TrainConfig` class contains the default settings; `FULL_PROFILE` contains
the broader notebook-derived model comparison. `AIRBNB_DATA_DIR` and
`AIRBNB_OUTPUT_DIR` environment variables can change the default folders.

Default: median baseline, ridge, and random forest. Full: also LightGBM, XGBoost, and CatBoost, with raw/log targets, base/rich features, parameter search, and optional blending.

```bash
python scripts/run_train.py --data data/listings.csv --output outputs/run
python scripts/run_train.py --profile full --data data/listings.csv --output outputs/full
python scripts/run_train.py --help
```

Available overrides include `--models`, `--feature-sets`, `--n-estimators`,
`--cv-folds`, `--search-iterations`, `--seed`, and `--n-jobs`. The full profile can
be expensive. Use a new output directory for each training/EDA run; existing run
files are preserved. Defaults resolve relative to this project directory; paths
explicitly passed on the command line resolve relative to your working directory.

## Model export and prediction

Training writes the evaluated fitted model to `outputs/run/model.joblib`.
It includes the feature transformer, learned preprocessing, estimator(s), and
blend weights and inverse target handling.
No separate manual imputation or encoding is needed for prediction.

```bash
python scripts/run_predict.py --model outputs/run/model.joblib --data data/new_rows.csv --output outputs/predictions.csv
python scripts/build_report.py --run outputs/run
```

The training output directory contains:

- `model.joblib`: complete fitted workflow.
- `metadata.json`: raw input schema, model choice, configuration, dependency versions, and data/source/model hashes.
- `requirements.lock.txt`: recorded runtime/model package versions.
- `config.json`: settings captured from this particular run.
- `cv_results.csv`, `cv_folds.csv`, and optional `search_*.csv`: development model comparisons.
- `split_assignments.csv`: original row indices, IDs, and evaluation partitions.
- `holdout_predictions.csv`, `metrics.json`: final evaluation results.
- `blend_validation.csv`: validation candidate scores when blending is enabled.

`build_report.py` adds `REPORT.md` and `REPORT.html` without retraining the model.
Only load model files you trust, using the same source and dependency versions as
the export environment. See the [scikit-learn persistence documentation](https://scikit-learn.org/stable/model_persistence.html).

## Input data

Training target: **`price`**. Prediction CSVs do not need the target.
Columns may appear in any order; extra columns are ignored unless explicitly
allowlisted. Column names are case sensitive, with surrounding CSV header spaces
trimmed. A required column may contain missing values, but must be present.

| Column | Expected value |
| --- | --- |
| `latitude` | numeric; missing allowed |
| `longitude` | numeric; missing allowed |
| `accommodates` | numeric; missing allowed |
| `bathrooms` | numeric; missing allowed |
| `bedrooms` | numeric; missing allowed |
| `beds` | numeric; missing allowed |
| `minimum_nights` | numeric; missing allowed |
| `availability_365` | numeric; missing allowed |
| `number_of_reviews` | numeric; missing allowed |
| `neighbourhood_cleansed` | category; missing allowed |
| `property_type` | category; missing allowed |
| `room_type` | category; missing allowed |

`host_id` is required for default host-grouped training. `id` is optional but preserved as a string. Use `--random-split` only when a row-based evaluation is appropriate; `--no-blend` uses the CV winner directly.

Training requires at least 100 valid listings,
plus enough rows/groups for the requested splits and CV folds. Exact duplicate
rows are removed before splitting. CSV IDs are loaded as strings to preserve
large numeric identifiers.

## Features

The base inputs are the notebook's 12 predictors. The `rich` feature set adds
per-guest capacity ratios, neighbourhood/room and property/room combinations,
rounded geographic cells, and numeric missing indicators. If present at training,
the explicit allowlist also uses `host_response_rate`, `host_listings_count`,
`host_response_time`, `host_is_superhost`, `instant_bookable`,
`neighbourhood_group_cleansed`, `amenities`, and `bathrooms_text`.
Amenity flags cover pool, hot tub, parking, air conditioning, kitchen, washer,
Wi-Fi, gym, fireplace, and waterfront references. Optional columns used by the
winning components become required inference columns, recorded in `metadata.json`.

LightGBM learns category levels on each training fold; unseen categories become
missing. CatBoost receives native categories. Ridge, random forest, and XGBoost
use fitted one-hot encoding and numeric imputation. Ridge also scales inputs.

## Evaluation

1. Deduplicate exact records; reject unresolved repeated listing IDs. Parse numeric or currency-formatted prices such as `$1,250.00`; exclude missing, nonfinite, and nonpositive targets.
2. Reserve 20% of hosts for holdout. Reserve 20% of remaining hosts for blend validation (about 16% overall); use the rest for model-selection CV. These are host fractions, so row fractions can differ.
3. Compare base/rich inputs and raw/log targets with fold-fitted preprocessing. Rank candidates by development CV MAE, or RMSE if configured.
4. Optionally shortlist two candidates per family and select nonnegative blend weights on the separate validation partition. Retain a blend only when it improves on the best validation single model by `min_blend_gain` in the chosen metric.
5. Freeze the recipe, refit its components on train + validation, evaluate the untouched holdout, and export that exact fitted workflow.

No valid expensive listings are trimmed. Predictions reverse any log transform and
are clipped at zero. Reported MAE/RMSE remain in the CSV's original price units;
no currency conversion is performed. `host_id` and `id` are never predictors.
Missing host IDs receive distinct row fallback groups, which cannot prevent
unknown repeated hosts from crossing splits.

CV scores are used for selecting candidates and are not unbiased external-test
estimates. `metrics.json` records the holdout results of this run. No previously
observed notebook metric is reused as a claimed result for these modules.

## EDA outputs

`run_eda.py` writes CSVs for numeric summaries, missing values, category counts,
correlation/covariance matrices, and target associations, plus PNGs for target and
numeric distributions and correlations. Target associations use Spearman correlation.
These summaries describe the supplied dataset; EDA results are not used to select
the model automatically.

## Try the scripts without the original dataset

```bash
python scripts/make_demo_data.py --output data/demo.csv --rows 400
python scripts/run_eda.py --data data/demo.csv --output outputs/demo_eda
python scripts/run_train.py --data data/demo.csv --output outputs/demo --n-estimators 20 --cv-folds 2
python scripts/run_predict.py --model outputs/demo/model.joblib --data examples/predict.csv --output outputs/demo/predictions.csv
python scripts/build_report.py --run outputs/demo
```

These inputs are **synthetic**. They check execution, export, and reload; their
scores do not establish real model performance.

## Tests and GitHub

```bash
python -m pip install pytest ruff
python -m pytest
ruff check src scripts tests
```

Tests cover direct script execution, fresh-process export/reload, missing/unseen
inputs, split isolation, invalid data, optional model families, and reports.
Each folder can be uploaded as its own same-named GitHub repository:

```bash
git init
git add .
git commit -m "Add airbnb Python ML project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/airbnb.git
git push -u origin main
```

Local CSVs, generated outputs, environments, and caches are ignored by Git. No
license was inferred from the supplied notebook. If you also want an installable
wheel, `pyproject.toml` remains available (`pip install -e .` or `python -m build`).

## Source and limitations

Listing examples alone do not establish dataset coverage, date, currency, or redistribution rights. The source CSV and verified data source URL were not included. The original notebook had already explored holdout results; rerunning the same data does not create an independent external test. This package does not reproduce the first notebook cell's geographic KMeans/stacking experiment or the v2 comparison against previously exported Colab models; those research recipes are preserved in the included source notebook.

Converted from `airbnb.ipynb`. This repository includes a copy of the source
notebook alongside the modular Python package and supporting files. Notebook-only installation/display cells and obsolete duplicate
experiments are not part of the runnable workflow. No previous notebook score is
presented as a result of this conversion.

Original notebook SHA-256: `eee4c792bf3a143067cccd247b568481613fe8f5b59b8e12f47a3795cfe64668`.
