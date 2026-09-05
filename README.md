# CORE-COX

Python implementation of CORE-COX for transfer learning in multi-task survival analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Input data

Prepare source and target data as pandas DataFrames. Each row is one participant. Both DataFrames should contain the same numeric predictor columns and one event/time pair for each outcome.

- Event columns contain `0` (censored) or `1` (event).
- Time columns contain positive follow-up times.

Example columns:

```text
age, sex, biomarker_1, disease_a_event, disease_a_time, disease_b_event, disease_b_time
```

## Quick start

```python
import pandas as pd

from corecox import CoreCoxModel, SourceCoxModel, prepare_survival_data

source_df = pd.read_csv("source.csv")
target_df = pd.read_csv("target.csv")

diseases = {
    "disease_a_event": "disease_a_time",
    "disease_b_event": "disease_b_time",
}

source_data, target_data = prepare_survival_data(
    source_df,
    target_df,
    diseases,
)

source_model = SourceCoxModel(
    rank=2,
    lambda_rank=0.01,
    max_iter=300,
    standardize=True,
    verbose=False,
)
source_model.fit(source_data)

model = CoreCoxModel(
    source_lr_model=source_model,
    lambda_theta=0.01,
    optimization_method="lbfgs",
    max_iter=1000,
    tol=1e-7,
    verbose=False,
)
model.fit(target_data)
```

## Results

```python
X_new = new_df[source_data.predictor_vars].to_numpy()

risk_scores = model.predict_risk_scores(X_new)
target_coefficients = model.get_original_beta_matrix()
residual_matrix = model.get_residual_matrix()
```

A complete runnable template is provided in [`example_usage.py`](example_usage.py).

## License

This project is available under the [MIT License](LICENSE).
