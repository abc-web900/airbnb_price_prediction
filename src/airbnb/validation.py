"""Price metrics in the original currency and development-only blend selection."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split


def metric_summary(y, prediction):
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
    }


def choose_blend(y, predictions, steps=30, min_gain=0.25, metric="mae"):
    names = list(predictions)
    matrix = np.column_stack([predictions[name] for name in names])

    def loss(p):
        return metric_summary(y, p)[metric]

    scores = [loss(matrix[:, i]) for i in range(len(names))]
    best_single = int(np.argmin(scores))
    best_loss = scores[best_single]
    single = np.eye(len(names))[best_single]
    best_weights = single.copy()
    counts = np.zeros(len(names))
    accumulated = np.zeros(len(y))
    for step in range(1, steps + 1):
        choice = int(
            np.argmin([loss((accumulated + matrix[:, i]) / step) for i in range(len(names))])
        )
        accumulated += matrix[:, choice]
        counts[choice] += 1
        score = loss(accumulated / step)
        if score < best_loss:
            best_loss, best_weights = score, counts.copy() / step
    if scores[best_single] - best_loss < min_gain:
        best_weights = single
    return {name: float(weight) for name, weight in zip(names, best_weights) if weight > 0}


def split_positions(X, groups, fraction, seed):
    if groups is not None:
        if groups.nunique() < 2:
            raise ValueError("At least two distinct hosts are needed for a grouped split.")
        return next(
            GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(
                X, groups=groups
            )
        )
    return train_test_split(np.arange(len(X)), test_size=fraction, random_state=seed)


def split_data(data, config):
    dev, test = split_positions(data.X, data.groups, config.holdout_size, config.seed)
    dev_groups = data.groups.iloc[dev] if data.groups is not None else None
    fit, validation = split_positions(
        data.X.iloc[dev],
        dev_groups,
        config.validation_size / (1 - config.holdout_size),
        config.seed + 31,
    )
    train, validation = dev[fit], dev[validation]
    if min(len(train), len(validation), len(test)) < 2:
        raise ValueError(
            "Every partition needs at least two rows; adjust splits or provide more data."
        )
    return train, validation, test


def make_folds(X, groups, config):
    if groups is not None:
        if groups.nunique() < config.cv_folds:
            raise ValueError("Too few distinct training hosts for cv_folds.")
        folds = list(
            GroupKFold(config.cv_folds, shuffle=True, random_state=config.seed + 1).split(
                X, groups=groups
            )
        )
    else:
        folds = list(KFold(config.cv_folds, shuffle=True, random_state=config.seed + 1).split(X))
    if any(len(va) < 2 for _, va in folds):
        raise ValueError("Every validation fold needs at least two rows.")
    return folds
