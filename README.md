# CORE-COX

This directory contains the core implementation of CORE-COX. Simulation code, private-data processing code, datasets, and experiment-specific evaluation scripts are intentionally excluded.

The code is released under the [MIT License](LICENSE).

## Method overview

CORE-COX first estimates a low-rank multi-task Cox model in a source population:

```text
B_source = U V^T
```

It then adapts that model to a target population by estimating a regularized residual coefficient matrix:

```text
B_target = B_source + Theta
```

`lambda_theta` controls the strength of transfer. A larger value keeps the target coefficients closer to the source coefficients; a smaller value permits a larger population-specific adjustment.

## Files

- `MTL_COX/MTLCoxData.py`: multi-disease survival-data container and validation.
- `MTL_COX/LRMTLCoxModel.py`: source-domain low-rank multi-task Cox model.
- `MTL_COX/LRTransferRTCoxModel.py`: CORE-COX residual transfer model.
- `example_usage.py`: minimal training and prediction example.

## Installation

From this directory, install the dependencies with:

```bash
pip install -r requirements.txt
```

## Required data format

Source and target data must be separate pandas DataFrames. Each row represents one participant. Each DataFrame must contain:

1. The same numeric predictor columns in the same meaning and units.
2. One binary event column per disease.
3. One positive follow-up-time column per disease.

Example:

| age | sex | biomarker_1 | disease_a_event | disease_a_time | disease_b_event | disease_b_time |
|---:|---:|---:|---:|---:|---:|---:|
| 55 | 0 | 1.24 | 1 | 4.2 | 0 | 8.0 |
| 61 | 1 | -0.31 | 0 | 6.7 | 1 | 3.5 |

Event values must be `0` for censored observations or `1` for observed events. Follow-up times must be greater than zero. Disease outcomes may be missing: missingness is handled independently for each disease. Rows with missing predictor values are removed.

All columns not listed as event or time columns are interpreted as predictors. Remove identifiers, population labels, free text, and other metadata before calling `prepare_mtl_data`. Predictors must be numeric.

The event-to-time mapping is supplied explicitly:

```python
diseases = {
    "disease_a_event": "disease_a_time",
    "disease_b_event": "disease_b_time",
}
```

Source and target DataFrames must use the same predictor names and disease definitions. Arrange predictor columns consistently in both files.

## Training CORE-COX

```python
import pandas as pd

from MTL_COX import LRMTLCoxModel, LRTransferRTCoxModel, prepare_mtl_data

source_df = pd.read_csv("source.csv")
target_df = pd.read_csv("target.csv")

diseases = {
    "disease_a_event": "disease_a_time",
    "disease_b_event": "disease_b_time",
}

source_data, target_data = prepare_mtl_data(source_df, target_df, diseases)

source_model = LRMTLCoxModel(
    rank=2,
    lambda_rank=0.01,
    max_iter=300,
    standardize=True,
    verbose=False,
)
source_model.fit(source_data)

model = LRTransferRTCoxModel(
    source_lr_model=source_model,
    lambda_theta=0.01,
    optimization_method="lbfgs",
    max_iter=1000,
    tol=1e-7,
    verbose=False,
)
model.fit(target_data)
```

Choose `rank` no larger than the smaller of the number of predictors and the number of diseases. Both `lambda_rank` and `lambda_theta` should normally be selected using validation data or nested cross-validation.

## Prediction and coefficients

Risk scores for new participants are obtained with:

```python
X_new = new_df[source_data.predictor_vars].to_numpy()
risk_scores = model.predict_risk_scores(X_new)
```

The result is a dictionary mapping each event-column name to a NumPy array. A larger risk score indicates greater relative hazard. Risk scores from different diseases should not be compared directly.

The target-population coefficients on the original predictor scale are available through:

```python
target_beta = model.get_original_beta_matrix()
```

The fitted residual matrix is available through:

```python
theta = model.get_residual_matrix()
```

`get_target_coefficients()` and `get_residual_matrix()` return coefficients on the standardized-feature scale when standardization is enabled. Use `get_original_beta_matrix()` when coefficients must be interpreted in the original units.

## Baseline hazard

`LRMTLCoxModel` can estimate source-domain baseline hazards with the Breslow estimator when `compute_baseline=True`. This supports source-domain survival-probability prediction.

The transfer model estimates target-domain regression coefficients and relative risk scores, but it does not currently re-estimate a target-domain baseline hazard. Consequently, its direct public prediction interface returns target risk scores rather than calibrated target survival probabilities. A target baseline hazard must be estimated separately if absolute survival probabilities are required.

## Practical checks

Before fitting, verify that:

- source and target predictor columns match exactly;
- every event column contains only `0`, `1`, or missing values;
- all observed follow-up times are positive;
- each disease has enough observed events for stable Cox estimation;
- preprocessing is learned from the training data only during external evaluation.
