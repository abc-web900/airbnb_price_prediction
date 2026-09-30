
## Dataset and evaluation design

| Data stage | Rows | Finding |
| --- | ---: | --- |
| Original dataset | 45,533 | 25 input columns before feature processing |
| Exact duplicates removed | 0 | No exact duplicate rows were removed |
| Invalid or nonpositive prices excluded | 8,237 | Missing, invalid, or nonpositive targets were unsuitable for supervised training |
| Valid labeled listings | 37,296 | Valid expensive listings were retained |
| Development set | 29,875 | Model development and final component fitting |
| Labeled holdout | 7,421 | Same holdout used for the original and V2 comparisons |
| V2 CV/training subset within development | 24,306 | Three-fold candidate comparison |
| V2 blend-validation subset within development | 5,569 | Choose blend weights before final refitting |

Splits and cross-validation grouped listings by `host_id`, preventing the same
known host from appearing in both sides of a split. The split fractions applied
to hosts, so row counts were not exact percentages. Listing IDs and host IDs
were excluded from predictors. V2 reused the earlier holdout assignments and
refitted its selected components on all 29,875 development rows before scoring
the holdout.

The development price distribution was strongly skewed:

| Statistic | Price |
| --- | ---: |
| Minimum | 5 |
| Median | 156 |
| Mean | 288.272301 |
| 90th percentile | 485 |
| 95th percentile | 845 |
| 99th percentile | 2,500 |
| Maximum | 56,425 |

The large gap between median and mean, together with the high maximum, explains
why the experiments compared raw-price and log-price targets and examined
errors across price bands.

## Preprocessing and feature engineering

The 12 core predictors were:

| Type | Features |
| --- | --- |
| Numeric | `latitude`, `longitude`, `accommodates`, `bathrooms`, `bedrooms`, `beds`, `minimum_nights`, `availability_365`, `number_of_reviews` |
| Categorical | `neighbourhood_cleansed`, `property_type`, `room_type` |

Numeric parsing converted invalid or infinite values to missing values, and
invalid geographic coordinates were treated as missing. Categorical values
were trimmed and missing values became `Unknown`. Preprocessing was fitted
within each training fold.

### Original experiment

The first experiment tested these additions to the core predictors:

| Feature group | Transformation |
| --- | --- |
| Capacity ratios | Beds, bedrooms, and bathrooms divided by accommodates |
| Combined capacity | Beds + bedrooms + bathrooms |
| Review intensity | Number of reviews / (availability over 365 days + 1) |
| Category combinations | Neighbourhood × room type; property type × guest capacity |
| Geographic clusters | Up to 20 KMeans clusters learned from training coordinates |
| Missingness | Per-feature missing-value flags and a total missing count |

Numeric imputation used training-fold medians. Categories were one-hot encoded.
Undefined ratios became missing. Raw-price and `log1p(price)` targets were
compared, and log-target predictions were converted back to price units.

### V2 features retained by the final blend

The V2 rich feature set included per-guest capacity ratios, numeric missing
flags, neighbourhood/room and property/room combinations, and a geographic
cell formed by rounding latitude and longitude to two decimal places.

Five optional inputs were actually available and used by the rich candidates:
`host_response_rate`, `host_response_time`, `host_is_superhost`,
`instant_bookable`, and `neighbourhood_group_cleansed`.

The code also supported `host_listings_count`, amenity flags, and a shared-bathroom
indicator, but their source columns were absent in the recorded run. Those
features therefore cannot be credited with any observed improvement.

V2 LightGBM and CatBoost used native categorical handling. LightGBM category
levels were learned from training data, with unseen levels treated as missing.
Numeric missing values were retained for these native boosting models. The
final blend combined two rich-feature components and one core-feature component;
it did not use the earlier KMeans feature recipe.

## What the original comparisons showed

The following are development CV scores from the first experiment, with lower
MAE indicating better price predictions.

| LightGBM feature/target configuration | CV MAE |
| --- | ---: |
| Original features, raw target | 136.3654 |
| Original features, log target | **106.1724** |
| Engineered features, log target | 107.2329 |
| Engineered features + geographic clusters, log target | 107.3047 |
| Engineered features + clusters + missing flags, log target | 107.1151 |
| Log target, remove training prices above the 99th percentile | 112.3596 |
| Log target, remove training prices above the 99.5th percentile | 109.3632 |

**The log target was the clearest improvement:** it reduced original-feature
LightGBM CV MAE by about **22.14%**. The first engineered-feature variants did
not improve on original features with the log target. Their differences were
small compared with fold variability, so this does not establish that those
features are generally unhelpful.

Training-only price trimming made CV MAE worse. The retained models therefore
kept the complete valid price range, including expensive listings.

| Original model comparison | CV MAE |
| --- | ---: |
| Equal-weight voting ensemble | **105.4829** |
| LightGBM, original features + log target | 106.1724 |
| Tuned LightGBM | 106.2581 |
| Tuned CatBoost | 106.4071 |
| XGBoost + log target | 107.1161 |
| CatBoost + log target | 108.0412 |
| Tuned XGBoost | 108.1474 |
| Random forest + log target | 109.1731 |
| Stacking | 118.9273 |
| Ridge + raw target | 179.5573 |
| Linear regression + raw target | 179.7455 |
| Median baseline | 192.3543 |

The first selected model averaged LightGBM, XGBoost, tuned CatBoost, and random
forest predictions equally. All four used log targets and retained all
development rows. CatBoost tuning improved its CV MAE; LightGBM and XGBoost
tuning did not beat their initial log-target configurations in the tested search.
Stacking performed worse than simple averaging.

Log-target linear regression and ridge were unstable after converting predictions
back to price units: their CV MAEs were **3,736.1750** and **3,590.6145**,
respectively. The favorable log-target result for tree models did not extend
to those linear configurations.

## V2 model selection and final blend

V2 compared eight LightGBM and eight CatBoost configurations on the smaller
24,306-row training subset, testing core/rich features and log-L2/raw-MAE
objectives. Its CV scores should not be directly ranked against the first
experiment's scores because the training subsets and folds differed.

| V2 configuration | CV MAE |
| --- | ---: |
| `cat_02_log_L2_rich` | **111.1483** |
| `cat_04_log_L2_base` | 112.3813 |
| `lgb_02_log_L2_rich` | 112.3901 |
| `cat_00_log_L2_base` | 112.6772 |
| `lgb_00_log_L2_base` | 112.9978 |
| `cat_03_raw_MAE_rich` | 114.3873 |
| `lgb_03_raw_MAE_rich` | 114.6160 |

In the matched initial configurations, rich features improved log-target CV
MAE for CatBoost from 112.6772 to 111.1483 and for LightGBM from 112.9978 to
112.3901. These changes were modest and did not isolate the contribution of
each added feature. Direct raw-price MAE objectives did not beat the selected
log-target configurations in this search.

The two leading candidates from each family and the earlier voting recipe were
then compared on the same **5,569 blend-validation rows**:

| Candidate | Validation MAE | Validation RMSE |
| --- | ---: | ---: |
| Rich LightGBM, log target | 95.6894 | 405.8345 |
| Core LightGBM, log target | 99.0837 | 435.0466 |
| Rich CatBoost, log target | **91.5494** | **323.2076** |
| Core CatBoost, log target | 92.5820 | 344.2026 |
| Earlier equal-weight voting recipe | 95.7045 | 398.6005 |
| Selected V2 blend | **91.1381** | 333.5209 |

The blend reduced validation MAE by **0.4113** versus the best single model,
exceeding the configured minimum gain of 0.25. It had higher validation RMSE
than that single CatBoost model, illustrating the tradeoff from selecting by
MAE. The frozen nonnegative weights were:

| Component | Feature set | Weight |
| --- | --- | ---: |
| `cat_02_log_L2_rich` | Core + rich features | 56.09756% |
| `cat_04_log_L2_base` | Core features | 34.14634% |
| `lgb_02_log_L2_rich` | Core + rich features | 9.75610% |

All three components learned `log1p(price)`. Their predictions were transformed
back with `expm1`, clipped at zero, and then averaged using these weights in
original price units. CatBoost supplied about **90.24%** of the final blend.
The earlier voting recipe and core LightGBM received zero final weight.

## Final holdout results

All rows with valid prices were retained in the **same 7,421-row holdout**.

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Development-median baseline | 199.9113 | 631.5855 | -0.0500 |
| Earlier equal-weight ensemble, refitted | 104.50483 | 401.65820 | 0.57534 |
| **V2 validation-selected blend** | **102.93067** | **400.27770** | **0.57826** |

V2 reduced MAE by **1.57416 price units (1.51%)** and RMSE by **1.38050**
versus the earlier ensemble. Its MAE was about **48.51% below the median
baseline**. An MAE of 102.93067 means an average absolute prediction error of
about 102.93 price units. The much larger RMSE indicates that some listings
had very large errors.

## Error patterns in the final V2 model

Residuals below are actual price minus predicted price: positive values indicate
underprediction, and negative values indicate overprediction.

| Actual price band | Rows | MAE | Mean residual |
| --- | ---: | ---: | ---: |
| Above 0, up to 100 | 2,135 | 22.059 | -17.211 |
| Above 100, up to 200 | 2,660 | 34.690 | -13.729 |
| Above 200, up to 500 | 1,828 | 86.329 | 8.405 |
| Above 500, up to 1,000 | 484 | 266.469 | 113.781 |
| Above 1,000, up to 2,500 | 236 | 600.773 | 328.676 |
| Above 2,500 | 78 | **2,511.716** | **2,344.809** |

The model tended to overpredict cheaper listings and underpredict expensive
ones. The 78 listings above 2,500 represented only **1.05% of the holdout** but
had exceptionally large errors. The main remaining weakness was the expensive
tail, despite retaining these listings during training and evaluation.

## Feature importance and earlier segment findings

The notebook's permutation-importance and room/neighbourhood analyses apply to
the **earlier equal-weight ensemble**, not the V2 blend. Permutation importance
used a sample of up to 2,000 holdout rows and three repeats; the values are the
increase in MAE when a raw input was shuffled.

| Raw input | MAE increase after shuffling |
| --- | ---: |
| Longitude | 32.5175 |
| Bathrooms | 32.2963 |
| Bedrooms | 24.5324 |
| Accommodates | 15.6031 |
| Latitude | 15.5614 |
| Room type | 7.5105 |
| Property type | 7.4874 |
| Neighbourhood | 7.2059 |

Location and capacity-related inputs had the largest measured importance in
that earlier model. These are predictive associations, not causal effects.
Earlier room-type MAE was 298.758 for hotel rooms, 123.709 for entire homes,
39.335 for private rooms, and 16.534 for shared rooms. Hotel and shared-room
groups were small, with 34 and 54 rows respectively. Expensive neighbourhoods
such as Bel-Air and Malibu also had large errors. No corresponding V2 room-type
or neighbourhood table was recorded, so these numbers are not attributed to V2.

## What worked, what did not, and evaluation limits

- **Log-price targets helped tree models substantially** compared with the
  raw-target baseline in the first experiment.
- **The V2 rich features and native categorical boosters helped modestly** in
  the recorded comparisons. CatBoost was the largest contributor to the final blend.
- **Validation-selected blending improved MAE slightly**, with a corresponding
  small improvement on the reused holdout.
- **More complexity did not consistently help:** the first geographic/engineered
  variants, stacking, and some hyperparameter searches failed to improve CV MAE.
- **Removing expensive training listings hurt performance.** The selected
  models retained the full valid price range.
- **Log-target linear models were unstable**, and minimizing validation MAE
  did not also minimize validation RMSE.
- **High-price underprediction remained substantial.** Average scores hide
  much weaker performance for the most expensive listings.

The holdout had already been examined before V2 was developed. These results
support a comparison on that same dataset and split, not an independent
external-test claim. The notebook did not record prediction or evaluation of a
separate external test CSV. The target is a listing's advertised price; these
results do not measure future bookings or earned revenue.
