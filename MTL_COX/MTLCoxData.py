import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

class MTLCoxData:
    """Shared predictors with disease-specific survival outcomes and masks."""

    def __init__(self):
        self.patient_ids = None
        self.predictor_vars = None
        self.X = None
        self.diseases = []
        self.T_dict = {}
        self.E_dict = {}
        self.valid_mask_dict = {}
        self.n_patients = 0
        self.n_features = 0
        self.n_diseases = 0

    def get_disease_data(self, disease_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if disease_name not in self.diseases:
            raise ValueError(f"Disease '{disease_name}' does not exist. Available diseases: {self.diseases}")

        valid_mask = self.valid_mask_dict[disease_name]

        X_valid = self.X[valid_mask]
        T_valid = self.T_dict[disease_name][valid_mask]
        E_valid = self.E_dict[disease_name][valid_mask]

        return X_valid, T_valid, E_valid

    def get_all_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        return self.X, self.T_dict, self.E_dict

    def get_patient_subset(self, patient_indices: np.ndarray) -> 'MTLCoxData':

        subset = MTLCoxData()

        subset.patient_ids = self.patient_ids[patient_indices]
        subset.predictor_vars = self.predictor_vars.copy()
        subset.diseases = self.diseases.copy()
        subset.n_patients = len(patient_indices)
        subset.n_features = self.n_features
        subset.n_diseases = self.n_diseases

        subset.X = self.X[patient_indices]

        subset.T_dict = {}
        subset.E_dict = {}
        subset.valid_mask_dict = {}

        for disease in self.diseases:

            subset.T_dict[disease] = self.T_dict[disease][patient_indices]
            subset.E_dict[disease] = self.E_dict[disease][patient_indices]

            disease_T = subset.T_dict[disease]
            disease_E = subset.E_dict[disease]

            valid_mask = (
                    ~np.isnan(disease_T) &
                    ~np.isnan(disease_E) &
                    (disease_T > 0) &
                    np.isin(disease_E, [0, 1])
            )

            subset.valid_mask_dict[disease] = valid_mask

        return subset

def prepare_mtl_data(data_eur_encoded: pd.DataFrame,
                     data_asn_encoded: pd.DataFrame,
                     diseases_config: Dict[str, str]) -> Tuple[MTLCoxData, MTLCoxData]:
    """Build source and target containers from event-to-time column mappings."""

    def create_mtl_data(data: pd.DataFrame, name: str) -> MTLCoxData:

        data = data.copy()
        data['patient_id'] = range(len(data))

        exclude_cols = ['patient_id']
        exclude_cols.extend(list(diseases_config.keys()))
        exclude_cols.extend(list(diseases_config.values()))

        predictor_vars = [col for col in data.columns if col not in exclude_cols]

        cleaned_data = data.dropna(subset=predictor_vars)

        if len(cleaned_data) == 0:
            raise ValueError(f"❌ {name} has no valid patients after data cleaning!")

        mtl_data = MTLCoxData()

        mtl_data.patient_ids = cleaned_data['patient_id'].values
        mtl_data.predictor_vars = predictor_vars
        mtl_data.X = cleaned_data[predictor_vars].values
        mtl_data.diseases = list(diseases_config.keys())
        mtl_data.n_patients = len(cleaned_data)
        mtl_data.n_features = len(predictor_vars)
        mtl_data.n_diseases = len(diseases_config)

        for disease_event, disease_time in diseases_config.items():

            full_T = np.full(mtl_data.n_patients, np.nan)
            full_E = np.full(mtl_data.n_patients, np.nan)

            disease_data = cleaned_data[[disease_event, disease_time]].copy()

            valid_mask = (
                disease_data[disease_event].notna() &
                disease_data[disease_time].notna() &
                (disease_data[disease_time] > 0) &
                disease_data[disease_event].isin([0, 1, True, False])
            )

            if valid_mask.sum() > 0:

                full_T[valid_mask] = disease_data.loc[valid_mask, disease_time].values
                full_E[valid_mask] = disease_data.loc[valid_mask, disease_event].astype(int).values

            mtl_data.T_dict[disease_event] = full_T
            mtl_data.E_dict[disease_event] = full_E
            mtl_data.valid_mask_dict[disease_event] = valid_mask

        return mtl_data

    mtl_data_eur = create_mtl_data(data_eur_encoded, "European population")
    mtl_data_asn = create_mtl_data(data_asn_encoded, "Asian population")

    return mtl_data_eur, mtl_data_asn
