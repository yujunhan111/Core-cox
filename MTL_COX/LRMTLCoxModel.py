import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm
from numba import jit, prange
from scipy.optimize import minimize
import time
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_cox_loss_fast(X, T, E, beta):

    n_samples = X.shape[0]
    loss = 0.0

    risk_scores = np.zeros(n_samples)
    for i in prange(n_samples):
        risk_scores[i] = np.dot(X[i], beta)

    sorted_indices = np.argsort(-T)

    cumulative_exp_score = 0.0

    i = 0
    while i < n_samples:
        idx = sorted_indices[i]
        current_time = T[idx]

        tied_start = i
        tied_end = i
        while tied_end < n_samples and T[sorted_indices[tied_end]] == current_time:
            tied_end += 1

        for j in range(tied_start, tied_end):
            tied_idx = sorted_indices[j]

            adjusted_score = risk_scores[tied_idx]
            if adjusted_score > 700:
                adjusted_score = 700
            elif adjusted_score < -700:
                adjusted_score = -700

            exp_score = np.exp(adjusted_score)
            cumulative_exp_score += exp_score

        if cumulative_exp_score > 1e-10:
            for j in range(tied_start, tied_end):
                tied_idx = sorted_indices[j]
                if E[tied_idx] == 1:
                    loss += risk_scores[tied_idx] - np.log(cumulative_exp_score)

        i = tied_end

    return -loss

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_cox_gradient_fast(X, T, E, beta):
    n_samples, n_features = X.shape
    gradient = np.zeros(n_features)

    risk_scores = np.zeros(n_samples)
    for i in prange(n_samples):
        risk_scores[i] = np.dot(X[i], beta)

    sorted_indices = np.argsort(-T)

    cumulative_exp_score = 0.0
    cumulative_weighted_features = np.zeros(n_features)

    i = 0
    while i < n_samples:
        idx = sorted_indices[i]
        current_time = T[idx]

        tied_start = i
        tied_end = i
        while tied_end < n_samples and T[sorted_indices[tied_end]] == current_time:
            tied_end += 1

        for j in range(tied_start, tied_end):
            tied_idx = sorted_indices[j]

            adjusted_score = risk_scores[tied_idx]
            if adjusted_score > 700:
                adjusted_score = 700
            elif adjusted_score < -700:
                adjusted_score = -700

            exp_score = np.exp(adjusted_score)

            cumulative_exp_score += exp_score
            for k in range(n_features):
                cumulative_weighted_features[k] += X[tied_idx, k] * exp_score

        if cumulative_exp_score > 1e-10:
            for j in range(tied_start, tied_end):
                tied_idx = sorted_indices[j]
                if E[tied_idx] == 1:

                    for k in range(n_features):
                        weighted_avg = cumulative_weighted_features[k] / cumulative_exp_score
                        gradient[k] += X[tied_idx, k] - weighted_avg

        i = tied_end

    return -gradient

@jit(nopython=True, fastmath=True, cache=True)
def compute_baseline_hazard(X, T, E, beta, event_times):

    n_samples = X.shape[0]
    n_events = len(event_times)
    baseline_hazards = np.zeros(n_events)

    risk_scores = np.zeros(n_samples)
    for i in range(n_samples):
        score = 0.0
        for j in range(X.shape[1]):
            score += X[i, j] * beta[j]

        if score > 700:
            score = 700
        elif score < -700:
            score = -700
        risk_scores[i] = score

    for k, event_time in enumerate(event_times):

        events_at_time = 0
        for i in range(n_samples):
            if T[i] == event_time and E[i] == 1:
                events_at_time += 1

        if events_at_time == 0:
            continue

        risk_set_sum = 0.0
        for i in range(n_samples):
            if T[i] >= event_time:
                risk_set_sum += np.exp(risk_scores[i])

        if risk_set_sum > 1e-10:
            baseline_hazards[k] = events_at_time / risk_set_sum
        else:
            baseline_hazards[k] = 0.0

    return baseline_hazards
class LRMTLCoxModel:
    """Source-domain multi-task Cox model with B = U @ V.T."""

    def __init__(self,
                 rank: int = 5,
                 lambda_rank: float = 0.01,
                 max_iter: int = 1000,
                 tol: float = 1e-7,
                 standardize: bool = True,
                 verbose: bool = True,
                 method: str = 'alternating',
                 n_jobs: int = -1,
                 initialization: str = 'random',
                 compute_baseline: bool = True):

        self.rank = rank
        self.lambda_rank = lambda_rank
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.verbose = verbose
        self.method = method
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.initialization = initialization
        self.compute_baseline = compute_baseline

        self.U = None
        self.V = None
        self.B = None
        self.diseases = None
        self.feature_names = None
        self.n_features = None
        self.n_diseases = None
        self.scaler = None

        self.baseline_hazards = {}
        self.baseline_computed = False
        self.disease_event_times = {}

        self.loss_history = []
        self.converged = False
        self.n_iter = 0
        self.training_time = 0

    def fit(self, mtl_data: 'MTLCoxData') -> 'LRMTLCoxModel':
        """Fit the shared low-rank coefficient matrix on source-domain data."""

        start_time = time.time()

        if self.verbose:
            print("🚀 Starting LR-MTL-Cox Model Training...")
            print("=" * 60)

        self.diseases = mtl_data.diseases.copy()
        self.feature_names = mtl_data.predictor_vars.copy()
        self.n_features = mtl_data.n_features
        self.n_diseases = mtl_data.n_diseases

        self.n_samples = mtl_data.n_patients

        max_rank = min(self.n_features, self.n_diseases)
        if self.rank > max_rank:
            warnings.warn(f"Rank {self.rank} exceeds maximum possible rank {max_rank}, adjusting to {max_rank}")
            self.rank = max_rank

        if self.verbose:
            print(f"📊 Data Overview:")
            print(f"   Feature Matrix: ({mtl_data.n_patients}, {mtl_data.n_features})")
            print(f"   Number of Diseases: {self.n_diseases}")
            print(f"   Latent Factors: {self.rank}")
            print(f"   Optimization Method: {self.method}")
            print(f"   Low-rank Regularization: {self.lambda_rank}")
            print(f"   Compute Baseline Hazards: {self.compute_baseline}")

        X = mtl_data.X

        if self.standardize:
            if self.verbose:
                print(f"\n🔧 Standardizing features...")
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        disease_data = self._prepare_disease_data_optimized(X, None, None, mtl_data)

        self._initialize_factors()

        if self.method == 'joint':
            self._fit_joint_optimization(disease_data)
        else:
            self._fit_alternating_minimization(disease_data)

        self.B = self.U @ self.V.T

        if self.compute_baseline:
            if self.verbose:
                print(f"\n🔬 Computing baseline hazard functions...")
            self._compute_baseline_hazards(disease_data)

        self.training_time = time.time() - start_time

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"✅ Training completed! Time: {self.training_time:.2f}s")
            if self.converged:
                print(f"🎯 Model converged! Iterations: {self.n_iter}")
            else:
                print(f"⚠️  Reached maximum iterations")

            print(f"📈 Low-rank Analysis:")
            print(f"   Effective Rank: {self._compute_effective_rank():.2f}")
            print(
                f"   Compression Ratio: {(self.n_features * self.rank + self.n_diseases * self.rank) / (self.n_features * self.n_diseases):.2%}")

            if self.baseline_computed:
                print(f"🔬 Baseline Hazard Statistics:")
                for disease in self.diseases:
                    if disease in self.baseline_hazards:
                        n_time_points = len(self.baseline_hazards[disease])
                        print(f"   {disease}: {n_time_points} time points")

        return self

    def get_original_beta_matrix(self):
        """Return coefficients expressed in the original predictor units."""

        if self.B is None:
            raise ValueError("Model not trained")

        if self.standardize and self.scaler is not None:

            original_beta = self.B / self.scaler.scale_.reshape(-1, 1)
            return original_beta
        else:

            return self.B.copy()

    def get_beta_matrix(self):

        if self.B is None:
            raise ValueError("Model not trained")

        return self.B.copy()

    def _prepare_disease_data_optimized(self, X, T_dict, E_dict, mtl_data):

        disease_data = {}

        for i, disease in enumerate(self.diseases):
            try:

                X_disease, T_disease, E_disease = mtl_data.get_disease_data(disease)

                if self.standardize and self.scaler is not None:
                    X_disease = self.scaler.transform(X_disease)

                X_disease = np.ascontiguousarray(X_disease.astype(np.float64))
                T_disease = np.ascontiguousarray(T_disease.astype(np.float64))
                E_disease = np.ascontiguousarray(E_disease.astype(np.int32))

                disease_data[disease] = {
                    'X': X_disease,
                    'T': T_disease,
                    'E': E_disease,
                    'task_idx': i,
                    'n_samples': len(T_disease)
                }

                if self.verbose:
                    n_events = np.sum(E_disease)
                    print(f"   📋 {disease}: {len(T_disease)} samples, {n_events} events")

            except Exception as e:
                if self.verbose:
                    print(f"   ❌ {disease}: Data retrieval failed - {e}")

                disease_data[disease] = {
                    'X': np.zeros((1, self.n_features), dtype=np.float64),
                    'T': np.array([1.0], dtype=np.float64),
                    'E': np.array([0], dtype=np.int32),
                    'task_idx': i,
                    'n_samples': 0
                }

        return disease_data

    def _compute_baseline_hazards(self, disease_data):

        self.baseline_hazards = {}
        self.disease_event_times = {}

        for disease, data in disease_data.items():
            if data['n_samples'] == 0:
                continue

            task_idx = data['task_idx']
            beta = self.B[:, task_idx]

            event_mask = data['E'] == 1
            if np.sum(event_mask) == 0:

                continue

            unique_event_times = np.unique(data['T'][event_mask])
            unique_event_times = np.sort(unique_event_times)

            baseline_increments = compute_baseline_hazard(
                data['X'], data['T'], data['E'], beta, unique_event_times
            )

            self.disease_event_times[disease] = unique_event_times
            self.baseline_hazards[disease] = {
                time: hazard for time, hazard in zip(unique_event_times, baseline_increments)
            }

        self.baseline_computed = True

        if self.verbose:
            print(f"   ✅ Baseline hazards computed for {len(self.baseline_hazards)} diseases")

    def _initialize_factors(self):

        if self.verbose:
            print(f"\n🎲 Initializing factor matrices (method: {self.initialization})...")

        if self.initialization == 'random':

            scale = 0.1 / np.sqrt(self.rank)
            self.U = np.random.normal(0, scale, (self.n_features, self.rank))
            self.V = np.random.normal(0, scale, (self.n_diseases, self.rank))

        elif self.initialization == 'svd':

            B_init = np.random.normal(0, 0.01, (self.n_features, self.n_diseases))
            U_svd, s_svd, Vt_svd = np.linalg.svd(B_init, full_matrices=False)

            r = min(self.rank, len(s_svd))
            s_sqrt = np.sqrt(s_svd[:r])

            self.U = U_svd[:, :r] * s_sqrt[np.newaxis, :]
            self.V = (Vt_svd[:r, :] * s_sqrt[:, np.newaxis]).T

            if self.rank > r:
                scale = 0.01 / np.sqrt(self.rank - r)
                U_extra = np.random.normal(0, scale, (self.n_features, self.rank - r))
                V_extra = np.random.normal(0, scale, (self.n_diseases, self.rank - r))
                self.U = np.hstack([self.U, U_extra])
                self.V = np.hstack([self.V, V_extra])

        elif self.initialization == 'zeros':

            self.U = np.zeros((self.n_features, self.rank))
            self.V = np.zeros((self.n_diseases, self.rank))

        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")

        if np.allclose(self.U, 0) and np.allclose(self.V, 0):
            warnings.warn("Initialization resulted in all zeros, may cause optimization issues")

    def _fit_alternating_minimization(self, disease_data):

        if self.verbose:
            print(f"\n🎯 Using alternating minimization training...")

        prev_loss = float('inf')
        progress_bar = tqdm(range(self.max_iter), desc="🔄 Alternating Optimization",
                          disable=not self.verbose)

        for iteration in progress_bar:

            self._update_U(disease_data)

            self._update_V(disease_data)

            current_loss = self._compute_loss(disease_data)
            self.loss_history.append(current_loss)

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

    def _update_U(self, disease_data):

        def objective_U(u_flat):
            U_new = u_flat.reshape(self.n_features, self.rank)
            return self._compute_loss_with_UV(disease_data, U_new, self.V)

        def gradient_U(u_flat):
            U_new = u_flat.reshape(self.n_features, self.rank)
            grad = self._compute_gradient_U(disease_data, U_new, self.V)
            return grad.flatten()

        result = minimize(
            objective_U,
            self.U.flatten(),
            method='L-BFGS-B',
            jac=gradient_U,
            options={'maxiter': 50, 'ftol': 1e-9}
        )

        self.U = result.x.reshape(self.n_features, self.rank)

    def _update_V(self, disease_data):

        def objective_V(v_flat):
            V_new = v_flat.reshape(self.n_diseases, self.rank)
            return self._compute_loss_with_UV(disease_data, self.U, V_new)

        def gradient_V(v_flat):
            V_new = v_flat.reshape(self.n_diseases, self.rank)
            grad = self._compute_gradient_V(disease_data, self.U, V_new)
            return grad.flatten()

        result = minimize(
            objective_V,
            self.V.flatten(),
            method='L-BFGS-B',
            jac=gradient_V,
            options={'maxiter': 50, 'ftol': 1e-9}
        )

        self.V = result.x.reshape(self.n_diseases, self.rank)

    def _compute_loss_with_UV(self, disease_data, U, V):

        total_loss = 0.0
        total_samples = 0

        for disease, data in disease_data.items():
            task_idx = data['task_idx']

            beta = np.ascontiguousarray(U @ V[task_idx, :])

            cox_loss = compute_cox_loss_fast(
                data['X'], data['T'], data['E'], beta
            )

            total_loss += cox_loss
            total_samples += data['n_samples']

        if total_samples > 0:
            total_loss /= total_samples

        if self.lambda_rank > 0:
            rank_penalty = 0.5 * (np.sum(U**2) + np.sum(V**2))
            total_loss += self.lambda_rank * rank_penalty

        return total_loss

    def _compute_gradient_U(self, disease_data, U, V):

        grad_U = np.zeros_like(U)

        for disease, data in disease_data.items():
            task_idx = data['task_idx']

            beta = np.ascontiguousarray(U @ V[task_idx, :])

            cox_grad = compute_cox_gradient_fast(
                data['X'], data['T'], data['E'], beta
            )

            grad_U += np.outer(cox_grad / data['n_samples'], V[task_idx, :])

        if self.lambda_rank > 0:
            grad_U += self.lambda_rank * U

        return grad_U

    def _compute_gradient_V(self, disease_data, U, V):

        grad_V = np.zeros_like(V)

        for disease, data in disease_data.items():
            task_idx = data['task_idx']

            beta = np.ascontiguousarray(U @ V[task_idx, :])

            cox_grad = compute_cox_gradient_fast(
                data['X'], data['T'], data['E'], beta
            )

            grad_V[task_idx, :] += (cox_grad / data['n_samples']) @ U

        if self.lambda_rank > 0:
            grad_V += self.lambda_rank * V

        return grad_V

    def _compute_loss(self, disease_data):

        return self._compute_loss_with_UV(disease_data, self.U, self.V)

    def _fit_joint_optimization(self, disease_data):

        if self.verbose:
            print(f"\n🎯 Using joint optimization training...")

        def objective(params):

            n_u_params = self.n_features * self.rank
            u_flat = params[:n_u_params]
            v_flat = params[n_u_params:]

            U = u_flat.reshape(self.n_features, self.rank)
            V = v_flat.reshape(self.n_diseases, self.rank)

            return self._compute_loss_with_UV(disease_data, U, V)

        def gradient(params):

            n_u_params = self.n_features * self.rank
            u_flat = params[:n_u_params]
            v_flat = params[n_u_params:]

            U = u_flat.reshape(self.n_features, self.rank)
            V = v_flat.reshape(self.n_diseases, self.rank)

            grad_U = self._compute_gradient_U(disease_data, U, V)
            grad_V = self._compute_gradient_V(disease_data, U, V)

            return np.concatenate([grad_U.flatten(), grad_V.flatten()])

        x0 = np.concatenate([self.U.flatten(), self.V.flatten()])

        with tqdm(total=self.max_iter, desc="🔄 Joint Optimization", disable=not self.verbose) as pbar:
            def callback(x):
                pbar.update(1)
                loss = objective(x)
                self.loss_history.append(loss)
                pbar.set_postfix({'Loss': f'{loss:.6f}'})

            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                jac=gradient,
                options={
                    'maxiter': self.max_iter,
                    'ftol': self.tol,
                    'gtol': self.tol
                },
                callback=callback
            )

        n_u_params = self.n_features * self.rank
        self.U = result.x[:n_u_params].reshape(self.n_features, self.rank)
        self.V = result.x[n_u_params:].reshape(self.n_diseases, self.rank)

        self.converged = result.success
        self.n_iter = result.nit

    def _compute_effective_rank(self):

        if self.B is None:
            return 0

        _, s, _ = np.linalg.svd(self.B, full_matrices=False)

        s_norm = s / np.sum(s)
        s_norm = s_norm[s_norm > 1e-10]
        effective_rank = np.exp(-np.sum(s_norm * np.log(s_norm)))
        return effective_rank

    def predict_risk_scores(self, X):

        if self.B is None:
            raise ValueError("Model not trained")

        if self.standardize and self.scaler is not None:
            X = self.scaler.transform(X)

        risk_scores = {}
        for i, disease in enumerate(self.diseases):
            risk_scores[disease] = X @ self.B[:, i]

        return risk_scores

    def predict_survival_probability(self, X, time_points=None):

        if self.B is None:
            raise ValueError("Model not trained")

        if not self.baseline_computed:
            raise ValueError("Baseline hazards not computed. Set compute_baseline=True during training.")

        if self.standardize and self.scaler is not None:
            X = self.scaler.transform(X)

        survival_probs = {}

        for disease in self.diseases:
            if disease not in self.baseline_hazards:
                continue

            disease_idx = self.diseases.index(disease)
            beta = self.B[:, disease_idx]
            risk_scores = X @ beta

            baseline_hazards = self.baseline_hazards[disease]
            baseline_times = np.array(sorted(baseline_hazards.keys()))

            cumulative_baseline_hazard = np.zeros(len(baseline_times))
            for i, t in enumerate(baseline_times):
                if i == 0:
                    cumulative_baseline_hazard[i] = baseline_hazards[t]
                else:
                    cumulative_baseline_hazard[i] = cumulative_baseline_hazard[i-1] + baseline_hazards[t]

            baseline_survival = np.exp(-cumulative_baseline_hazard)

            if time_points is None:
                pred_times = baseline_times
                interp_baseline_surv = baseline_survival
            else:
                pred_times = np.array(time_points)

                interp_baseline_surv = np.interp(pred_times, baseline_times, baseline_survival,
                                               left=1.0, right=baseline_survival[-1] if len(baseline_survival) > 0 else 1.0)

            n_samples = X.shape[0]
            n_times = len(pred_times)
            surv_matrix = np.zeros((n_samples, n_times))

            for j in range(n_samples):

                hazard_ratio = np.exp(risk_scores[j])

                hazard_ratio = np.clip(hazard_ratio, 1e-10, 1e10)

                surv_matrix[j, :] = np.power(interp_baseline_surv, hazard_ratio)

            survival_probs[disease] = {
                'times': pred_times,
                'probabilities': surv_matrix
            }

        return survival_probs

    def get_feature_importance(self):

        if self.B is None:
            raise ValueError("Model not trained")

        importance = np.linalg.norm(self.B, axis=1)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })

        return importance_df.sort_values('Importance', ascending=False)

    def get_latent_factors(self):

        if self.U is None:
            raise ValueError("Model not trained")

        factor_df = pd.DataFrame(
            self.U,
            index=self.feature_names,
            columns=[f'Factor_{i+1}' for i in range(self.rank)]
        )

        return factor_df

    def get_task_weights(self):

        if self.V is None:
            raise ValueError("Model not trained")

        weight_df = pd.DataFrame(
            self.V,
            index=self.diseases,
            columns=[f'Factor_{i+1}' for i in range(self.rank)]
        )

        return weight_df

    def get_coefficients(self):

        if self.B is None:
            raise ValueError("Model not trained")

        return pd.DataFrame(
            self.B,
            index=self.feature_names,
            columns=self.diseases
        )

    def get_baseline_hazards_summary(self):

        if not self.baseline_computed:
            return "Baseline hazards not computed"

        summary = {}
        for disease, hazards in self.baseline_hazards.items():
            if hazards:
                times = list(hazards.keys())
                hazard_values = list(hazards.values())
                summary[disease] = {
                    'n_time_points': len(times),
                    'time_range': (min(times), max(times)),
                    'mean_hazard': np.mean(hazard_values),
                    'max_hazard': max(hazard_values),
                    'total_cumulative_hazard': sum(hazard_values)
                }

        return pd.DataFrame(summary).T

    def analyze_latent_structure(self):

        if self.U is None or self.V is None:
            raise ValueError("Model not trained")

        print("🔍 Latent Structure Analysis:")
        print("=" * 50)

        factor_norms_U = np.linalg.norm(self.U, axis=0)
        factor_norms_V = np.linalg.norm(self.V, axis=0)
        factor_importance = factor_norms_U * factor_norms_V

        return {
            'factor_importance': factor_importance,
            'U_matrix': self.U,
            'V_matrix': self.V,
            'effective_rank': self._compute_effective_rank()
        }

    def get_performance_stats(self):

        stats = {
            'training_time_seconds': self.training_time,
            'iterations': self.n_iter,
            'converged': self.converged,
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'rank': self.rank,
            'effective_rank': self._compute_effective_rank() if self.B is not None else None,
            'optimization_method': self.method,
            'lambda_rank': self.lambda_rank,
            'baseline_computed': self.baseline_computed
        }

        if self.B is not None:

            original_params = self.n_features * self.n_diseases
            factorized_params = self.n_features * self.rank + self.n_diseases * self.rank
            stats['compression_ratio'] = factorized_params / original_params
            stats['parameter_reduction'] = 1 - stats['compression_ratio']

        if self.baseline_computed:
            stats['baseline_summary'] = {}
            for disease in self.diseases:
                if disease in self.baseline_hazards:
                    stats['baseline_summary'][disease] = len(self.baseline_hazards[disease])

        return stats

    def save_model(self, filepath):

        import pickle

        model_data = {
            'U': self.U,
            'V': self.V,
            'B': self.B,
            'diseases': self.diseases,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'n_diseases': self.n_diseases,
            'rank': self.rank,
            'lambda_rank': self.lambda_rank,
            'scaler': self.scaler,
            'standardize': self.standardize,
            'baseline_hazards': self.baseline_hazards,
            'baseline_computed': self.baseline_computed,
            'disease_event_times': self.disease_event_times,
            'loss_history': self.loss_history,
            'converged': self.converged,
            'n_iter': self.n_iter,
            'training_time': self.training_time,
            'compute_baseline': self.compute_baseline
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 Model saved to: {filepath}")
        if self.baseline_computed:
            print(f"   ✅ Baseline hazards included")

    @classmethod
    def load_model(cls, filepath):

        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(
            rank=model_data['rank'],
            lambda_rank=model_data['lambda_rank'],
            standardize=model_data['standardize'],
            compute_baseline=model_data.get('compute_baseline', True)
        )

        model.U = model_data['U']
        model.V = model_data['V']
        model.B = model_data['B']
        model.diseases = model_data['diseases']
        model.feature_names = model_data['feature_names']
        model.n_features = model_data['n_features']
        model.n_diseases = model_data['n_diseases']
        model.scaler = model_data['scaler']
        model.baseline_hazards = model_data['baseline_hazards']
        model.baseline_computed = model_data['baseline_computed']
        model.disease_event_times = model_data['disease_event_times']
        model.loss_history = model_data['loss_history']
        model.converged = model_data['converged']
        model.n_iter = model_data['n_iter']
        model.training_time = model_data['training_time']

        print(f"📁 Model loaded from {filepath}")
        if model.baseline_computed:
            print(f"   ✅ Baseline hazards loaded")

        return model

    def __repr__(self):
        status = "Trained" if self.B is not None else "Not Trained"
        baseline_status = "✅" if self.baseline_computed else "❌"
        effective_rank = self._compute_effective_rank() if self.B is not None else "N/A"
        return (f"LRMTLCoxModel({status}, rank={self.rank}, eff_rank={effective_rank:.2f}, "
                f"baseline={baseline_status}, λ_rank={self.lambda_rank})")

def evaluate_lr_mtl_cox_model(mtl_data, **model_params):

    default_params = {
        'rank': 5,
        'lambda_rank': 0.01,
        'max_iter': 1000,
        'method': 'alternating',
        'verbose': True,
        'compute_baseline': True
    }

    default_params.update(model_params)

    model = LRMTLCoxModel(**default_params)
    model.fit(mtl_data)

    if default_params['verbose']:
        print("\n🎯 LR-MTL-Cox Model Training Summary:")
        print("=" * 50)

        model.analyze_latent_structure()

        if model.baseline_computed:
            print(f"\n🔬 Baseline Hazards Summary:")
            baseline_summary = model.get_baseline_hazards_summary()
            print(baseline_summary)

    return model
