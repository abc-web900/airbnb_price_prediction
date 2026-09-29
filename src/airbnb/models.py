"""Price estimators and nonnegative blends from the notebook's v2 workflow."""

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .features import FeatureTable


class PriceModel(RegressorMixin, BaseEstimator):
    def __init__(self, spec, seed=2026, n_jobs=2):
        self.spec = spec
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(target) != len(X) or not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError("Training prices must be finite, positive and aligned.")
        self.features_ = FeatureTable(self.spec["encoding"], self.spec["rich"]).fit(X)
        train = self.features_.transform(X)
        if self.spec["log"]:
            target = np.log1p(target)
        p = deepcopy(self.spec["params"])
        family = self.spec["family"]
        if family == "dummy":
            model = DummyRegressor(strategy="median")
        elif family == "ridge":
            model = make_pipeline(StandardScaler(), Ridge(**p))
        elif family == "random_forest":
            model = RandomForestRegressor(random_state=self.seed, n_jobs=self.n_jobs, **p)
        else:
            try:
                if family == "lightgbm":
                    from lightgbm import LGBMRegressor

                    model = LGBMRegressor(
                        random_state=self.seed,
                        n_jobs=self.n_jobs,
                        verbosity=-1,
                        deterministic=True,
                        force_col_wise=True,
                        **p,
                    )
                elif family == "xgboost":
                    from xgboost import XGBRegressor

                    model = XGBRegressor(
                        random_state=self.seed,
                        n_jobs=self.n_jobs,
                        tree_method="hist",
                        objective="reg:squarederror",
                        **p,
                    )
                elif family == "catboost":
                    from catboost import CatBoostRegressor

                    model = CatBoostRegressor(
                        random_seed=self.seed,
                        thread_count=self.n_jobs,
                        verbose=False,
                        allow_writing_files=False,
                        **p,
                    )
                else:
                    raise ValueError(f"Unknown model family: {family}")
            except ImportError as exc:
                raise ImportError(f"Model {family} needs: pip install -e '.[boosting]'") from exc
        kwargs = {"cat_features": self.features_.cat_columns_} if family == "catboost" else {}
        self.model_ = model.fit(train, target, **kwargs)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        pred = np.asarray(self.model_.predict(self.features_.transform(X)), dtype=float).reshape(-1)
        if self.spec["log"]:
            pred = np.expm1(pred)
        if not np.isfinite(pred).all():
            raise ValueError("Model produced non-finite price predictions.")
        return np.maximum(pred, 0.0)


class BlendModel:
    """Frozen fitted components and validation-selected convex weights."""

    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        return sum(self.weights[name] * self.models[name].predict(X) for name in self.weights)


def candidates(config):
    result = {}
    for family in dict.fromkeys(["dummy", *config.models]):
        for variant in ["base"] if family == "dummy" else config.feature_sets:
            for log in [False] if family == "dummy" else [False, True]:
                params = {
                    "dummy": {},
                    "ridge": {"alpha": 10.0},
                    "random_forest": {
                        "n_estimators": config.n_estimators,
                        "max_depth": 22,
                        "min_samples_leaf": 2,
                        "max_features": 0.8,
                    },
                    "lightgbm": {
                        "n_estimators": config.n_estimators,
                        "learning_rate": 0.05,
                        "num_leaves": 31,
                        "min_child_samples": 30,
                        "reg_lambda": 2.0,
                        "objective": "regression" if log else "regression_l1",
                    },
                    "xgboost": {
                        "n_estimators": config.n_estimators,
                        "learning_rate": 0.05,
                        "max_depth": 6,
                        "min_child_weight": 3,
                    },
                    "catboost": {
                        "iterations": config.n_estimators,
                        "learning_rate": 0.05,
                        "depth": 6,
                        "l2_leaf_reg": 5.0,
                        "loss_function": "RMSE" if log else "MAE",
                    },
                }[family]
                spec = {
                    "family": family,
                    "encoding": {"lightgbm": "lgb", "catboost": "cat"}.get(family, "ohe"),
                    "rich": variant == "rich",
                    "log": log,
                    "params": params,
                }
                result[f"{family}__{variant}__{'log' if log else 'raw'}"] = PriceModel(
                    spec, config.seed, config.n_jobs
                )
    return result


def search_space(model):
    from sklearn.model_selection import ParameterGrid

    grids = {
        "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        "random_forest": {"max_depth": [12, 22, None], "min_samples_leaf": [2, 5, 10]},
        "lightgbm": {"num_leaves": [15, 31, 63], "min_child_samples": [15, 30, 60]},
        "xgboost": {"max_depth": [3, 5, 7], "min_child_weight": [1, 3, 7]},
        "catboost": {"depth": [4, 6, 8], "l2_leaf_reg": [3.0, 5.0, 10.0]},
    }
    if model.spec["family"] not in grids:
        return {}
    specs = []
    for overrides in ParameterGrid(grids[model.spec["family"]]):
        spec = deepcopy(model.spec)
        spec["params"].update(overrides)
        specs.append(spec)
    return {"spec": specs}
