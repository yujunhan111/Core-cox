import pandas as pd

from MTL_COX import LRMTLCoxModel, LRTransferRTCoxModel, prepare_mtl_data


def main():
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
        tol=1e-7,
        standardize=True,
        verbose=False,
        method="alternating",
        initialization="random",
        compute_baseline=True,
    )
    source_model.fit(source_data)

    corecox_model = LRTransferRTCoxModel(
        source_lr_model=source_model,
        lambda_theta=0.01,
        optimization_method="lbfgs",
        max_iter=1000,
        tol=1e-7,
        verbose=False,
    )
    corecox_model.fit(target_data)

    feature_frame = target_df[source_data.predictor_vars]
    risk_scores = corecox_model.predict_risk_scores(feature_frame.to_numpy())
    target_coefficients = corecox_model.get_original_beta_matrix()
    residuals = corecox_model.get_residual_matrix()

    print(target_coefficients)
    print(residuals)
    for disease, scores in risk_scores.items():
        print(disease, scores[:5])


if __name__ == "__main__":
    main()
