"""Host-disjoint CV, optional validation blending, and untouched holdout evaluation."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_validate,
)
from threadpoolctl import threadpool_limits

from .artifacts import file_hash, load_artifact, save_artifact, write_json
from .config import DEFAULT_OUTPUT_DIR, TRAIN_PATH, TrainConfig
from .data import load_training_data, raw_schema
from .models import BlendModel, candidates, search_space
from .predict import predict_frame
from .schema import RAW_FEATURES, TARGET
from .validation import choose_blend, make_folds, metric_summary, split_data


def run(data_path=None, output_dir=None, config=None):
    data_path = TRAIN_PATH if data_path is None else data_path
    output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    config = config or TrainConfig()
    with threadpool_limits(limits=config.n_jobs):
        return _run(data_path, output_dir, config)


def _run(data_path, output_dir, config):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "Use a new or empty output directory to preserve previous run artifacts."
        )
    data = load_training_data(data_path, config.group_by_host)
    tr, va, te = split_data(data, config)
    X, y = data.X.iloc[tr], data.y.iloc[tr]
    groups = data.groups.iloc[tr] if data.groups is not None else None
    folds = make_folds(X, groups, config)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    models = candidates(config)
    out.mkdir(parents=True, exist_ok=True)
    assignments = pd.concat(
        [
            pd.DataFrame(
                {
                    "csv_row_index": data.X.iloc[ix].index,
                    "id": data.ids.iloc[ix].to_numpy(),
                    "partition": label,
                }
            )
            for ix, label in [(tr, "train"), (va, "validation"), (te, "holdout")]
        ],
        ignore_index=True,
    )
    assignments.to_csv(out / "split_assignments.csv", index=False)
    records, fold_records = [], []
    for name, model in list(models.items()):
        print(f"Cross-validating {name}", flush=True)
        start = time.perf_counter()
        params = search_space(model)
        if config.search_iterations and params:
            search = RandomizedSearchCV(
                model,
                params,
                n_iter=min(config.search_iterations, len(params["spec"])),
                scoring=scoring,
                refit=config.primary_metric,
                cv=folds,
                random_state=config.seed,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(X, y)
            models[name] = clone(search.best_estimator_)
            results, best = search.cv_results_, search.best_index_
            scores = {
                f"test_{metric}": np.array(
                    [results[f"split{i}_test_{metric}"][best] for i in range(config.cv_folds)]
                )
                for metric in scoring
            }
            pd.DataFrame(results).to_csv(out / f"search_{name}.csv", index=False)
        else:
            scores = cross_validate(
                model, X, y, cv=folds, scoring=scoring, n_jobs=1, error_score="raise"
            )
        records.append(
            {
                "candidate": name,
                "family": model.spec["family"],
                "seconds": time.perf_counter() - start,
                **{
                    m: float(np.mean(scores[f"test_{m}"])) * (1 if m == "r2" else -1)
                    for m in scoring
                },
                "primary_std": float(np.std(scores[f"test_{config.primary_metric}"], ddof=1)),
            }
        )
        fold_records.extend(
            {
                "candidate": name,
                "fold": i + 1,
                **{m: float(scores[f"test_{m}"][i]) * (1 if m == "r2" else -1) for m in scoring},
            }
            for i in range(config.cv_folds)
        )
        pd.DataFrame(records).to_csv(out / "cv_results.csv", index=False)
    comparison = pd.DataFrame(records).sort_values(config.primary_metric, kind="stable")
    comparison.to_csv(out / "cv_results.csv", index=False)
    pd.DataFrame(fold_records).to_csv(out / "cv_folds.csv", index=False)
    winner = comparison.iloc[0]["candidate"]
    validation_metrics = None
    if config.blend:
        # Shortlist up to two CV candidates per family, then learn convex weights on validation.
        shortlist = comparison.groupby("family", sort=False).head(2)["candidate"].tolist()
        validation_predictions = {
            name: clone(models[name]).fit(X, y).predict(data.X.iloc[va]) for name in shortlist
        }
        weights = choose_blend(
            data.y.iloc[va],
            validation_predictions,
            config.blend_steps,
            config.min_blend_gain,
            config.primary_metric,
        )
        blended = sum(weights[name] * validation_predictions[name] for name in weights)
        validation_metrics = metric_summary(data.y.iloc[va], blended)
        pd.DataFrame(
            [
                {"candidate": name, **metric_summary(data.y.iloc[va], p)}
                for name, p in validation_predictions.items()
            ]
        ).to_csv(out / "blend_validation.csv", index=False)
    else:
        weights = {winner: 1.0}
    # Refit selected recipes on all development rows after freezing choices.
    dev = np.sort(np.concatenate([tr, va]))
    parts = {name: clone(models[name]).fit(data.X.iloc[dev], data.y.iloc[dev]) for name in weights}
    model = BlendModel(parts, weights)
    predictions = model.predict(data.X.iloc[te])
    required = list(
        dict.fromkeys(RAW_FEATURES + [c for part in parts.values() for c in part.features_.inputs_])
    )
    metrics = {
        "holdout": metric_summary(data.y.iloc[te], predictions),
        "blend_validation": validation_metrics,
    }
    metadata = {
        "selected_model": "blend" if len(weights) > 1 else next(iter(weights)),
        "weights": weights,
        "selected_specs": {name: parts[name].spec for name in weights},
        "target": TARGET,
        "required_features": required,
        "input_schema": raw_schema(required),
        "config": config.to_dict(),
        "data": data.summary,
        "data_sha256": file_hash(data_path),
        "partitions": {"train": len(tr), "validation": len(va), "holdout": len(te)},
        "fit_partition": "train + validation",
        "metrics": metrics,
        "split_policy": "host groups" if config.group_by_host else "random rows",
    }
    path = save_artifact(model, out, metadata)
    reloaded = load_artifact(path)
    actual = predict_frame(reloaded, data.X.iloc[te])["predicted_price"].to_numpy()
    np.testing.assert_allclose(actual, predictions, rtol=1e-12, atol=1e-12)
    pd.DataFrame(
        {
            "csv_row_index": data.X.iloc[te].index,
            "id": data.ids.iloc[te].to_numpy(),
            "actual_price": data.y.iloc[te].to_numpy(),
            "predicted_price": predictions,
        }
    ).to_csv(out / "holdout_predictions.csv", index=False)
    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json", config.to_dict())
    print(f"Exported {path}; holdout MAE={metrics['holdout']['mae']:.4f}", flush=True)
    return {"model_path": path, "metadata": reloaded["metadata"]}
