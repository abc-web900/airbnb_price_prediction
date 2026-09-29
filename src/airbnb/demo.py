"""Synthetic listings for execution checks, not model-quality claims."""

import numpy as np
import pandas as pd


def make_demo_data(rows=400, seed=2026):
    if rows < 100:
        raise ValueError("Use at least 100 synthetic rows.")
    rng = np.random.default_rng(seed)
    guests = rng.integers(1, 9, rows)
    frame = pd.DataFrame(
        {
            "id": [str(1000000000000000000 + i) for i in range(rows)],
            "host_id": [f"host-{i // 2}" for i in range(rows)],
            "latitude": rng.normal(34.05, 0.08, rows),
            "longitude": rng.normal(-118.25, 0.08, rows),
            "accommodates": guests,
            "bathrooms": rng.integers(1, 4, rows).astype(float),
            "bedrooms": rng.integers(1, 5, rows).astype(float),
            "beds": rng.integers(1, 6, rows),
            "minimum_nights": rng.choice([1, 2, 7, 30], rows),
            "availability_365": rng.integers(0, 366, rows),
            "number_of_reviews": rng.integers(0, 200, rows),
            "neighbourhood_cleansed": rng.choice(["Central", "Coastal", "Hills"], rows),
            "property_type": rng.choice(["Apartment", "House"], rows),
            "room_type": rng.choice(["Entire home/apt", "Private room"], rows),
            "host_response_rate": rng.choice(["80%", "95%", "100%"], rows),
            "host_listings_count": rng.integers(1, 10, rows),
            "amenities": rng.choice(['["Wifi", "Kitchen"]', '["Pool", "Wifi", "Parking"]'], rows),
            "bathrooms_text": rng.choice(["1 bath", "2 shared baths"], rows),
        }
    )
    price = np.maximum(20, 45 + guests * 24 + frame["bedrooms"] * 18 + rng.normal(0, 20, rows))
    frame["price"] = [f"${p:,.2f}" for p in price]
    frame.loc[rng.choice(rows, rows // 12, replace=False), "bathrooms"] = np.nan
    return frame
