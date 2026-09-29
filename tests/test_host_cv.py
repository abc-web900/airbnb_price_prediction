from airbnb.config import TrainConfig
from airbnb.data import load_training_data
from airbnb.demo import make_demo_data
from airbnb.validation import make_folds


def test_host_groups_are_disjoint_in_every_fold(tmp_path):
    path = tmp_path / "listings.csv"
    make_demo_data(200).to_csv(path, index=False)
    data = load_training_data(path)
    for tr, va in make_folds(data.X, data.groups, TrainConfig()):
        assert set(data.groups.iloc[tr]).isdisjoint(data.groups.iloc[va])
