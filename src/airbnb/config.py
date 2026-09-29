"""Validated, serializable training settings."""

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AIRBNB_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("AIRBNB_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
TRAIN_PATH = DATA_DIR / "listings.csv"
PREDICT_PATH = DATA_DIR / "new_rows.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "run"
MODEL_PATH = DEFAULT_OUTPUT_DIR / "model.joblib"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

MODEL_NAMES = ["dummy", "ridge", "random_forest", "lightgbm", "xgboost", "catboost"]
FEATURE_SETS = ["base", "rich"]


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2026
    models: tuple[str, ...] = ("ridge", "random_forest")
    feature_sets: tuple[str, ...] = ("base", "rich")
    cv_folds: int = 3
    n_estimators: int = 200
    n_jobs: int = 2
    search_iterations: int = 0
    holdout_size: float = 0.2
    validation_size: float = 0.16
    primary_metric: str = "mae"
    group_by_host: bool = True
    blend: bool = True
    blend_steps: int = 30
    min_blend_gain: float = 0.25

    def __post_init__(self):
        if (
            not self.models
            or len(set(self.models)) != len(self.models)
            or set(self.models) - set(MODEL_NAMES)
        ):
            raise ValueError(f"models must be a unique nonempty list from {MODEL_NAMES}")
        if (
            not self.feature_sets
            or len(set(self.feature_sets)) != len(self.feature_sets)
            or set(self.feature_sets) - set(FEATURE_SETS)
        ):
            raise ValueError(f"feature_sets must be a unique nonempty list from {FEATURE_SETS}")
        for name in [
            "seed",
            "cv_folds",
            "n_estimators",
            "n_jobs",
            "search_iterations",
            "blend_steps",
        ]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        for name in ["holdout_size", "validation_size", "min_blend_gain"]:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.seed < 0 or self.cv_folds < 2 or self.n_estimators < 1 or self.n_jobs < 1:
            raise ValueError(
                "Use a nonnegative seed, cv_folds >= 2, n_estimators >= 1 and n_jobs >= 1"
            )
        if self.search_iterations < 0 or self.blend_steps < 1 or self.min_blend_gain < 0:
            raise ValueError("Invalid search or blending settings")
        if not 0 < self.holdout_size < 1 or not 0 < self.validation_size < 1 - self.holdout_size:
            raise ValueError(
                "holdout_size and validation_size must be positive and sum to less than 1"
            )
        if self.primary_metric not in ["mae", "rmse"]:
            raise ValueError("Unsupported primary_metric")
        for name in ["group_by_host", "blend"]:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text()) if path else {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        for name in ["models", "feature_sets"]:
            if name in values:
                if not isinstance(values[name], (list, tuple)):
                    raise ValueError(f"{name} must be a list")
                values[name] = tuple(values[name])
        return cls(**values)


FULL_PROFILE = {
    "models": ["ridge", "random_forest", "lightgbm", "xgboost", "catboost"],
    "feature_sets": ["base", "rich"],
    "n_estimators": 400,
    "cv_folds": 3,
    "search_iterations": 8,
    "n_jobs": 2,
}


def make_config(profile="default", **overrides):
    """Choose the lightweight or notebook-model profile, then apply explicit overrides."""
    if profile not in {"default", "full"}:
        raise ValueError("profile must be default or full")
    values = dict(FULL_PROFILE) if profile == "full" else {}
    values.update({key: value for key, value in overrides.items() if value is not None})
    for key in ["models", "feature_sets"]:
        if key in values:
            values[key] = tuple(values[key])
    return TrainConfig(**values)
