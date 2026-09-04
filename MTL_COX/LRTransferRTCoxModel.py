import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
from numba import jit, prange
import time
from scipy.optimize import minimize
from MTL_COX.LRMTLCoxModel import LRMTLCoxModel
from MTL_COX.MTLCoxData import MTLCoxData

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def compute_rt_cox_loss_and_gradient(X_list, T_list, E_list, B_source, theta,
                                    lambda_theta, sample_sizes):

    n_diseases = len(X_list)
    n_features = B_source.shape[0]

    total_loss = 0.0
    theta_grad = np.zeros((n_features, n_diseases))

    MAX_EXP_ARG = 500.0

    for k in range(n_diseases):
        X_k = X_list[k]
        T_k = T_list[k]
        E_k = E_list[k]
        N_k = sample_sizes[k]

        if N_k == 0:
            continue

        beta_k = np.zeros(n_features)
        for p in range(n_features):
            beta_k[p] = B_source[p, k] + theta[p, k]

        risk_scores_k = np.zeros(N_k)
        for i in range(N_k):
            score = 0.0
            for p in range(n_features):
                score += X_k[i, p] * beta_k[p]

            if score > MAX_EXP_ARG:
                risk_scores_k[i] = MAX_EXP_ARG
            elif score < -MAX_EXP_ARG:
                risk_scores_k[i] = -MAX_EXP_ARG
            else:
                risk_scores_k[i] = score

        disease_loss = 0.0
        theta_grad_k = np.zeros(n_features)

        event_indices = []
        for i in range(N_k):
            if E_k[i] == 1:
                event_indices.append(i)

        if len(event_indices) == 0:
            continue

        for event_idx in event_indices:
            event_time = T_k[event_idx]

            risk_set_indices = []
            for j in range(N_k):
                if T_k[j] >= event_time:
                    risk_set_indices.append(j)

            if len(risk_set_indices) == 0:
                continue

            max_risk_score = -1e10
            for j in risk_set_indices:
                if risk_scores_k[j] > max_risk_score:
                    max_risk_score = risk_scores_k[j]

            sum_exp = 0.0
            for j in risk_set_indices:
                exp_arg = risk_scores_k[j] - max_risk_score
                if exp_arg > -MAX_EXP_ARG:
                    sum_exp += np.exp(exp_arg)

            if sum_exp <= 0:
                continue

            log_sum_exp = np.log(sum_exp) + max_risk_score
            partial_ll = risk_scores_k[event_idx] - log_sum_exp

            disease_loss -= partial_ll

            for p in range(n_features):
                theta_grad_k[p] -= X_k[event_idx, p]

            for j in risk_set_indices:
                exp_arg = risk_scores_k[j] - max_risk_score
                if exp_arg > -MAX_EXP_ARG:
                    weight = np.exp(exp_arg) / sum_exp
                    for p in range(n_features):
                        theta_grad_k[p] += weight * X_k[j, p]

        if N_k > 0 and len(event_indices) > 0:
            disease_loss /= N_k
            total_loss += disease_loss

            for p in range(n_features):
                theta_grad[p, k] = theta_grad_k[p] / N_k

    theta_penalty = 0.0
    for p in range(n_features):
        for k in range(n_diseases):
            theta_penalty += theta[p, k] ** 2
    total_loss += lambda_theta * theta_penalty

    for p in range(n_features):
        for k in range(n_diseases):
            theta_grad[p, k] += 2.0 * lambda_theta * theta[p, k]

    return total_loss, theta_grad

class LRTransferRTCoxModel:
    """Target-domain residual transfer model with B_target = B_source + theta."""

    def __init__(self,
                 source_lr_model: LRMTLCoxModel,
                 lambda_theta: float = 0.01,
                 optimization_method: str = 'lbfgs',
                 max_iter: int = 1000,
                 tol: float = 1e-7,
                 verbose: bool = True):

        if source_lr_model.B is None:
            raise ValueError("Source LR-MTL-Cox model must be trained!")

        self.source_lr_model = source_lr_model
        self.lambda_theta = lambda_theta
        self.optimization_method = optimization_method
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.B_source = source_lr_model.B.copy()
        self.diseases = source_lr_model.diseases.copy()
        self.feature_names = source_lr_model.feature_names.copy()
        self.n_features = source_lr_model.n_features
        self.n_diseases = source_lr_model.n_diseases
        self.scaler = source_lr_model.scaler

        self.theta = None
        self.B_target = None

        self.loss_history = []
        self.converged = False
        self.n_iter = 0
        self.training_time = 0

    def fit(self, target_mtl_data: MTLCoxData) -> 'LRTransferRTCoxModel':
        """Estimate the regularized target-domain residual coefficient matrix."""

        start_time = time.time()

        if self.verbose:
            print("🚀 Starting RT-Cox residual transfer training...")
            print("=" * 60)
            print(f"📊 RT-Cox configuration:")
            print(f"   λ_θ (residual regularization): {self.lambda_theta}")
            print(f"   Optimization method: {self.optimization_method}")
            print(f"   Source coefficient matrix: {self.B_source.shape}")

        self._prepare_target_data(target_mtl_data)

        self._initialize_theta()

        if self.optimization_method == 'lbfgs':
            self._optimize_with_lbfgs()
        else:
            self._optimize_with_gradient_descent()

        self.B_target = self.B_source + self.theta

        self.training_time = time.time() - start_time

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ RT-Cox training completed! Time: {self.training_time:.2f}s")
            print(f"🎯 Converged: {'Yes' if self.converged else 'No'} (iterations: {self.n_iter})")
            print(f"📈 Final loss: {self.loss_history[-1]:.6f}")
            self._print_residual_stats()

        return self
    def get_original_beta_matrix(self):
        """Return target coefficients expressed in the original predictor units."""

        if self.B_target is None:
            raise ValueError("Model not trained")

        if not self.source_lr_model.standardize or self.scaler is None:

            return pd.DataFrame(
                self.B_target,
                index=self.feature_names,
                columns=self.diseases
            )

        std_devs = self.scaler.scale_
        means = self.scaler.mean_

        beta_original = self.B_target / std_devs[:, np.newaxis]

        return pd.DataFrame(
            beta_original,
            index=self.feature_names,
            columns=self.diseases
        )

    def _prepare_target_data(self, target_mtl_data: MTLCoxData):

        self.X_list = []
        self.T_list = []
        self.E_list = []
        self.sample_sizes = np.zeros(self.n_diseases, dtype=np.int32)

        for i, disease in enumerate(self.diseases):
            try:
                X, T, E = target_mtl_data.get_disease_data(disease)

                if len(X) == 0:

                    X = np.zeros((0, self.n_features))
                    T = np.zeros(0)
                    E = np.zeros(0)
                else:

                    if self.source_lr_model.standardize and self.scaler is not None:
                        X = self.scaler.transform(X)

                self.X_list.append(X)
                self.T_list.append(T)
                self.E_list.append(E)
                self.sample_sizes[i] = len(X)

            except KeyError:

                self.X_list.append(np.zeros((0, self.n_features)))
                self.T_list.append(np.zeros(0))
                self.E_list.append(np.zeros(0))
                self.sample_sizes[i] = 0

        if self.verbose:
            total_samples = int(np.sum(self.sample_sizes))
            print(f"📦 Target domain data prepared:")
            print(f"   Total samples: {total_samples}")
            for i, disease in enumerate(self.diseases):
                if self.sample_sizes[i] > 0:
                    n_events = int(np.sum(self.E_list[i]))
                    print(f"   {disease}: {self.sample_sizes[i]} samples, {n_events} events")

    def _initialize_theta(self):

        self.theta = np.zeros((self.n_features, self.n_diseases))

        if self.verbose:
            print(f"\n🔧 Initialized Θ matrix: {self.theta.shape} (all zeros)")

    def _optimize_with_lbfgs(self):

        if self.verbose:
            print(f"\n⚙️  Starting L-BFGS-B optimization...")

        theta_init = self.theta.flatten()

        def objective(theta_flat):
            theta_matrix = theta_flat.reshape((self.n_features, self.n_diseases))
            loss, grad = compute_rt_cox_loss_and_gradient(
                self.X_list, self.T_list, self.E_list,
                self.B_source, theta_matrix, self.lambda_theta, self.sample_sizes
            )
            self.loss_history.append(loss)
            return loss, grad.flatten()

        result = minimize(
            objective,
            theta_init,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
                'disp': self.verbose
            }
        )

        self.theta = result.x.reshape((self.n_features, self.n_diseases))
        self.converged = result.success
        self.n_iter = result.nit

        if self.verbose:
            print(f"\n✓ L-BFGS-B optimization completed")
            print(f"  Status: {result.message}")

    def _optimize_with_gradient_descent(self, learning_rate: float = 0.01):

        if self.verbose:
            print(f"\n⚙️  Starting gradient descent optimization...")
            print(f"   Learning rate: {learning_rate}")

        prev_loss = float('inf')
        progress_bar = tqdm(range(self.max_iter),
                          desc='Gradient Descent',
                          disable=not self.verbose)

        for iteration in progress_bar:

            current_loss, theta_grad = compute_rt_cox_loss_and_gradient(
                self.X_list, self.T_list, self.E_list,
                self.B_source, self.theta, self.lambda_theta, self.sample_sizes
            )

            self.loss_history.append(current_loss)

            self.theta -= learning_rate * theta_grad

            progress_bar.set_postfix({
                'Loss': f'{current_loss:.6f}',
                'ΔLoss': f'{abs(prev_loss - current_loss):.2e}'
            })

            if abs(prev_loss - current_loss) < self.tol:
                self.converged = True
                self.n_iter = iteration + 1
                break

            prev_loss = current_loss

        progress_bar.close()
        if not self.converged:
            self.n_iter = self.max_iter

    def _print_residual_stats(self):

        if self.theta is None:
            return

        theta_norm = np.linalg.norm(self.theta)
        b_source_norm = np.linalg.norm(self.B_source)
        relative_change = theta_norm / b_source_norm if b_source_norm > 0 else 0

        print(f"\n📈 Residual statistics:")
        print(f"   ‖Θ‖_F: {theta_norm:.4f}")
        print(f"   ‖B̂^s‖_F: {b_source_norm:.4f}")
        print(f"   Relative change: {relative_change:.1%}")

    def predict_risk_scores(self, X):

        if self.B_target is None:
            raise ValueError("Model not trained")

        if self.source_lr_model.standardize and self.scaler is not None:
            X = self.scaler.transform(X)

        risk_scores = {}
        for i, disease in enumerate(self.diseases):
            risk_scores[disease] = X @ self.B_target[:, i]

        return risk_scores

    def get_target_coefficients(self):

        if self.B_target is None:
            raise ValueError("Model not trained")

        return pd.DataFrame(
            self.B_target,
            index=self.feature_names,
            columns=self.diseases
        )

    def get_residual_matrix(self):

        if self.theta is None:
            raise ValueError("Model not trained")

        return pd.DataFrame(
            self.theta,
            index=self.feature_names,
            columns=self.diseases
        )

    def get_source_coefficients(self):

        return pd.DataFrame(
            self.B_source,
            index=self.feature_names,
            columns=self.diseases
        )

    def __repr__(self):
        status = "Trained" if self.theta is not None else "Untrained"
        return f"LRTransferRTCoxModel({status}, λ_θ={self.lambda_theta})"

def create_lr_trans_rt_cox_model(source_lr_model: LRMTLCoxModel,
                                target_mtl_data: MTLCoxData,
                                lambda_theta: float = 0.01,
                                optimization_method: str = 'lbfgs',
                                max_iter: int = 1000,
                                tol: float = 1e-7,
                                verbose: bool = True) -> LRTransferRTCoxModel:

    if verbose:
        print("🌟 Creating RT-Cox residual transfer model...")
        print("=" * 50)

    if source_lr_model.B is None:
        raise ValueError("Source LR-MTL-Cox model must be trained!")

    if verbose:
        print(f"   📊 Source model: {source_lr_model.n_features} features × {source_lr_model.n_diseases} diseases")
        print(f"   🎯 RT-Cox parameters: λ_θ={lambda_theta}")

    model = LRTransferRTCoxModel(
        source_lr_model=source_lr_model,
        lambda_theta=lambda_theta,
        optimization_method=optimization_method,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose
    )

    model.fit(target_mtl_data)

    if verbose:
        print(f"\n🎯 RT-Cox model creation completed!")
        if model.converged:
            print(f"   ✅ Converged (iterations: {model.n_iter})")
        else:
            print(f"   ⚠️  Not fully converged")

    return model

def evaluate_rt_cox_model(source_lr_model: LRMTLCoxModel,
                         target_mtl_data: MTLCoxData,
                         **model_params) -> Tuple[LRTransferRTCoxModel, pd.DataFrame]:

    model = create_lr_trans_rt_cox_model(
        source_lr_model, target_mtl_data, **model_params
    )

    results_data = []

    for disease in model.diseases:
        try:
            X_test, T_test, E_test = target_mtl_data.get_disease_data(disease)

            if len(X_test) == 0:
                continue

            risk_scores = model.predict_risk_scores(X_test)
            disease_risk_scores = risk_scores[disease]

            from lifelines.utils import concordance_index
            c_index = concordance_index(T_test, -disease_risk_scores, E_test)

            results_data.append({
                'Disease': disease,
                'N_Samples': len(X_test),
                'N_Events': np.sum(E_test),
                'Event_Rate': np.mean(E_test),
                'C_Index': c_index
            })

        except Exception as e:
            print(f"⚠️  Disease {disease} evaluation failed: {e}")
            continue

    results_df = pd.DataFrame(results_data)

    if len(results_df) > 0:
        print(f"\n📊 RT-Cox evaluation results:")
        print("-" * 40)
        for _, row in results_df.iterrows():
            print(f"{row['Disease']:12s}: C-index = {row['C_Index']:.4f}")

        mean_c_index = results_df['C_Index'].mean()
        print("-" * 40)
        print(f"{'Average C-index':12s}: {mean_c_index:.4f}")

    return model, results_df
