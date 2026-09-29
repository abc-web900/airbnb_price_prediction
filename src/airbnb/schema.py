"""Raw CSV contract derived from the source notebook."""

NUM_COLS = [
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
]
CAT_COLS = ["neighbourhood_cleansed", "property_type", "room_type"]
TARGET = "price"
RAW_FEATURES = NUM_COLS + CAT_COLS
PROJECT = "airbnb"
