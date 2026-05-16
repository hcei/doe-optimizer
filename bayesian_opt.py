"""Bayesian optimisation engine using GP + Expected Improvement.

Uses Latin Hypercube Sampling for initial design, a Gaussian Process
with Matern-5/2 kernel as the surrogate model, and Expected Improvement
as the acquisition function. All internal GP operations use inputs
normalised to [0, 1]^d for numerical stability.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


class BayesianOptimizer:
    """Bayesian optimisation for maximising a black-box function.

    Parameters
    ----------
    bounds : np.ndarray of shape (d, 2)
        Lower and upper bounds for each dimension.
    n_init : int
        Number of initial LHS design points (default 2*d + 1).
    xi : float
        Exploration parameter for Expected Improvement (default 0.01).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        n_init: int | None = None,
        xi: float = 0.01,
        random_state: int | None = None,
    ):
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.xi = xi
        self.rng = np.random.default_rng(random_state)

        if n_init is None:
            n_init = 2 * self.dim + 1

        # Generate initial LHS design in [0, 1]^d, then scale to real bounds
        sampler = LatinHypercube(d=self.dim, seed=random_state)
        lhs_unit = sampler.random(n=n_init)
        self._X_init = self._denormalize(lhs_unit)

        # GP 核函数：ConstantKernel * Matern(nu=2.5) + WhiteKernel
        # ConstantKernel 控制整体方差，Matern52 产生平滑但非无限可微的样本，WhiteKernel 捕捉观测噪声
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
            * Matern(length_scale=[1.0]*self.dim,
                     length_scale_bounds=(1e-6, 1e3), nu=2.5)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e2))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=random_state,
            alpha=1e-6,  # small jitter for numerical stability
        )

        # Normalise y-values for better GP fitting
        self._y_scaler = StandardScaler()

        # Observation storage (in real space)
        self.X_obs: list[np.ndarray] = []
        self.y_obs: list[float] = []

        # Initial-phase tracking
        self._init_counter = 0
        self._initial_phase = True
        self._gp_fitted = False

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Map real coords -> [0, 1]^d."""
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return (x - lo) / (hi - lo)

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        """Map [0, 1]^d -> real coords."""
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return lo + x_norm * (hi - lo)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self) -> np.ndarray:
        """Return the next point to evaluate.

        During the initial phase returns LHS design points.
        阶段一：使用 Latin Hypercube Sampling 在设计空间均匀撒点，获取全局信息。
        阶段二：通过 Expected Improvement 采集函数平衡利用（已知优区采样）
               与探索（不确定区域采样），迭代收敛到全局最优。
        """
        if self._initial_phase:
            x = self._X_init[self._init_counter]
            self._init_counter += 1
            if self._init_counter >= len(self._X_init):
                self._initial_phase = False
            return x

        return self._optimize_ei()

    def update(self, x_new: np.ndarray, y_new: float):
        """Register a new observation and refit the GP model."""
        self.X_obs.append(np.asarray(x_new, dtype=float).flatten())
        self.y_obs.append(float(y_new))

        if len(self.X_obs) >= 2:
            X_arr = np.array(self.X_obs)
            y_arr = np.array(self.y_obs).reshape(-1, 1)

            # 将输入归一化到 [0,1]^d，输出归一化到零均值单位方差，使 GP 超参数与观测值量纲无关
            X_norm = self._normalize(X_arr)
            y_norm = self._y_scaler.fit_transform(y_arr).ravel()

            self.gp.fit(X_norm, y_norm)
            self._gp_fitted = True

    @property
    def best_params(self) -> np.ndarray:
        """Parameters of the best observation so far."""
        idx = np.argmax(self.y_obs)
        return self.X_obs[idx]

    @property
    def best_value(self) -> float:
        """Best observed value so far."""
        return max(self.y_obs) if self.y_obs else -np.inf

    @property
    def history(self) -> list[dict]:
        """List of {x, y} dicts for all observations."""
        return [{'x': x, 'y': y} for x, y in zip(self.X_obs, self.y_obs)]

    @property
    def n_evals(self) -> int:
        return len(self.y_obs)

    @property
    def is_initial_phase(self) -> bool:
        return self._initial_phase

    # ------------------------------------------------------------------
    # Acquisition function (operates in normalised space)
    # ------------------------------------------------------------------

    def _ei(self, x_norm: np.ndarray) -> float:
        """Expected Improvement at normalised point x_norm."""
        if not self._gp_fitted:
            return 0.0

        x_2d = np.atleast_2d(x_norm)
        mu_norm, sigma_norm = self.gp.predict(x_2d, return_std=True)
        mu_norm = mu_norm[0]
        sigma_norm = sigma_norm[0]

        if sigma_norm < 1e-12:
            return 0.0

        # Denormalise predictions
        mu = float(self._y_scaler.inverse_transform([[mu_norm]])[0, 0])
        sigma = sigma_norm * float(self._y_scaler.scale_[0])

        improvement = mu - self.best_value - self.xi
        if sigma < 1e-12:
            return 0.0

        # EI(x) = Delta_mu * Phi(Z) + sigma * phi(Z)，其中 Z = Delta_mu / sigma
        Z = improvement / sigma
        ei_value = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return float(max(ei_value, 0.0))

    def _neg_ei(self, x_norm: np.ndarray) -> float:
        """Negative EI for use with scipy minimise."""
        return -self._ei(x_norm)

    def _optimize_ei(self, n_restarts: int = 50) -> np.ndarray:
        """Maximise EI in normalised [0,1]^d space via multi-start L-BFGS-B. EI 函数通常为多峰形态，单次梯度优化容易陷入局部最优，因此采用多次随机起点取最优值。"""
        best_x_norm = None
        best_ei = -np.inf

        norm_bounds = [(0.0, 1.0)] * self.dim

        for _ in range(n_restarts):
            x0_norm = self.rng.uniform(0.0, 1.0, self.dim)

            res = minimize(
                self._neg_ei,
                x0_norm,
                method='L-BFGS-B',
                bounds=norm_bounds,
            )

            ei_val = -res.fun
            if ei_val > best_ei:
                best_ei = ei_val
                best_x_norm = res.x

        # Fallback
        if best_x_norm is None:
            best_x_norm = self.rng.uniform(0.0, 1.0, self.dim)

        # 避免推荐与已有观测点距离过近的候选点——GP 在已观测点的预测方差为零，
        # 会导致 EI 值退化为零。微扰动使优化器保持移动。
        x_real = self._denormalize(best_x_norm)
        if len(self.X_obs) > 0:
            X_arr = np.array(self.X_obs)
            dists = np.linalg.norm(X_arr - x_real, axis=1)
            if np.min(dists) < 1e-3:
                # Perturb slightly
                x_real += self.rng.normal(0, 0.01, self.dim)
                x_real = np.clip(x_real, self.bounds[:, 0], self.bounds[:, 1])

        return x_real