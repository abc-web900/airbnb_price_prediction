"""Native categories and allowlisted extra features from Airbnb notebook v2."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .schema import CAT_COLS as CAT
from .schema import NUM_COLS as NUM

EXTRA_NUM = ["host_response_rate", "host_listings_count"]


EXTRA_CAT = [
    "host_response_time",
    "host_is_superhost",
    "instant_bookable",
    "neighbourhood_group_cleansed",
]


EXTRA_TEXT = ["amenities", "bathrooms_text"]


class FeatureTable(TransformerMixin, BaseEstimator):
    def __init__(self, mode="lgb", rich=False):
        self.mode, self.rich = mode, rich

    def _make(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Pass raw inputs as a pandas DataFrame.")

        missing = sorted(set(self.inputs_) - set(X.columns))
        if missing:
            raise ValueError(f"Missing raw input columns: {missing}")

        z = pd.DataFrame(index=X.index)

        for c in self.numeric_inputs_:
            values = X[c]
            if c == "host_response_rate":
                values = values.astype("string").str.replace("%", "", regex=False)
            z[c] = pd.to_numeric(values, errors="coerce").astype(float)

        z = z.replace([np.inf, -np.inf], np.nan)

        z["latitude"] = z["latitude"].where(z["latitude"].between(-90, 90))
        z["longitude"] = z["longitude"].where(z["longitude"].between(-180, 180))

        for c in self.category_inputs_:
            z[c] = (
                X[c]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
                .fillna("Unknown")
                .astype(object)
            )

        if self.rich:
            for c in self.numeric_inputs_:
                z[c + "__missing"] = z[c].isna().astype(np.float32)

            denom = z["accommodates"].where(z["accommodates"] > 0)

            for c in ["beds", "bedrooms", "bathrooms"]:
                z[c + "_per_guest"] = z[c] / denom

            z["neighbourhood_room"] = z[CAT[0]] + "__" + z["room_type"]
            z["property_room"] = z["property_type"] + "__" + z["room_type"]
            z["geo_cell"] = (
                z["latitude"].round(2).astype(str) + "__" + z["longitude"].round(2).astype(str)
            )

            if "amenities" in self.inputs_:
                text = X["amenities"].astype("string").str.lower()
                z["amenities_missing"] = text.isna().astype(np.float32)

                # Text counts/flags, not a learned vocabulary
                # or target encoding.
                z["amenity_separator_count"] = text.str.count(",").astype(float)

                patterns = {
                    "pool": r"\bpool\b",
                    "hot_tub": r"hot tub|jacuzzi",
                    "parking": r"parking",
                    "air_conditioning": r"air conditioning",
                    "kitchen": r"kitchen",
                    "washer": r"washer|washing machine",
                    "wifi": r"wifi|wi-fi",
                    "gym": r"gym",
                    "fireplace": r"fireplace",
                    "waterfront": (r"waterfront|beachfront|beach access"),
                }

                for name, pattern in patterns.items():
                    z["amenity_" + name] = text.str.contains(pattern, regex=True, na=False).astype(
                        np.float32
                    )

            if "bathrooms_text" in self.inputs_:
                text = X["bathrooms_text"].astype("string").str.lower()
                z["shared_bathroom"] = text.str.contains("shared", na=False).astype(float)

        numeric = z.select_dtypes(include=np.number).columns

        z[numeric] = z[numeric].replace([np.inf, -np.inf], np.nan).astype(np.float32)

        return z

    def fit(self, X, y=None):
        if self.mode not in {"ohe", "lgb", "cat"}:
            raise ValueError("mode must be ohe, lgb, or cat")
        self.numeric_inputs_ = NUM + ([c for c in EXTRA_NUM if c in X] if self.rich else [])
        self.category_inputs_ = CAT + ([c for c in EXTRA_CAT if c in X] if self.rich else [])

        self.inputs_ = self.numeric_inputs_ + self.category_inputs_

        if self.rich:
            self.inputs_ += [c for c in EXTRA_TEXT if c in X]

        z = self._make(X)
        self.columns_ = list(z.columns)

        self.cat_columns_ = z.select_dtypes(exclude=np.number).columns.tolist()
        self.num_columns_ = z.select_dtypes(include=np.number).columns.tolist()

        if self.mode == "ohe":
            # Reproduces the previous baseline's
            # median/one-hot recipe.
            self.encoder_ = ColumnTransformer(
                [
                    (
                        "num",
                        SimpleImputer(
                            strategy="median",
                            keep_empty_features=True,
                        ),
                        self.num_columns_,
                    ),
                    (
                        "cat",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            min_frequency=2,
                            max_categories=128,
                            sparse_output=False,
                            dtype=np.float32,
                        ),
                        self.cat_columns_,
                    ),
                ],
                sparse_threshold=0,
            ).fit(z)

        elif self.mode == "lgb":
            self.levels_ = {c: sorted(z[c].unique().tolist()) for c in self.cat_columns_}

        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        z = self._make(X)[self.columns_]

        if self.mode == "ohe":
            return self.encoder_.transform(z)

        if self.mode == "lgb":
            # Training-fold category mapping.
            # Unseen categories become missing.
            for c in self.cat_columns_:
                known = z[c].where(z[c].isin(self.levels_[c]), np.nan)
                z[c] = pd.Categorical(known, categories=self.levels_[c])

        return z
