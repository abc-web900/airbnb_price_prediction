"""Exercise notebook model families when the boosting extra is installed."""

import importlib.util

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from airbnb.config import TrainConfig
from airbnb.data import load_training_data
from airbnb.demo import make_demo_data
from airbnb.models import candidates

BOOSTING = all(
    importlib.util.find_spec(package) is not None for package in ["lightgbm", "xgboost", "catboost"]
)


@pytest.mark.skipif(
    not BOOSTING, reason="Install the boosting extra to test notebook model families"
)
def test_optional_models_export_and_reload(tmp_path):
    source = tmp_path / "train.csv"
    make_demo_data(160).to_csv(source, index=False)
    data = load_training_data(source)
    config = TrainConfig(
        models=("lightgbm", "xgboost", "catboost"),
        feature_sets=("rich",),
        n_estimators=5,
        n_jobs=1,
        cv_folds=2,
    )
    with threadpool_limits(limits=1):
        for candidate, model in candidates(config).items():
            fitted = clone(model).fit(data.X.iloc[:120], data.y.iloc[:120])
            expected = fitted.predict(data.X.iloc[120:])
            assert np.isfinite(expected).all(), candidate
            path = tmp_path / "candidate.joblib"
            joblib.dump(fitted, path)
            actual = joblib.load(path).predict(data.X.iloc[120:])
            np.testing.assert_allclose(actual, expected)
