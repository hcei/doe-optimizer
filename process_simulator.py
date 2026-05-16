"""Simulated 4-factor industrial chemical process.

Models a chemical reaction yield as a second-order response surface
with interactions and Gaussian noise. The true optimum is computed
analytically from the known coefficients.
"""

import numpy as np


class ChemicalProcess:
    """4-factor chemical reaction yield simulator.

    Factors (real-world ranges):
        temperature :  50 – 200 °C
        pressure    :   1 –  10 atm
        time        :  10 – 120 min
        catalyst    : 0.1 – 5.0 %

    The response surface is a second-order polynomial in coded
    variables x_i ∈ [-1, 1] with Gaussian noise σ = 1.5 %.
    """

    # Real-world bounds: [low, high] for each factor
    FACTOR_BOUNDS = {
        'temperature': (50.0, 200.0),
        'pressure':    (1.0,  10.0),
        'time':        (10.0, 120.0),
        'catalyst':    (0.1,   5.0),
    }

    FACTOR_NAMES = ['temperature', 'pressure', 'time', 'catalyst']

    # Linear coefficients in coded space
    _BETA_LINEAR = np.array([4.0, -2.5, 5.0, -3.0])

    # Quadratic + interaction matrix (symmetric)
    # H_ii = 2 * beta_ii;  H_ij = beta_ij  (i != j)
    _HESSIAN = np.array([
        [-10.0,   2.0,  -1.5,   0.8],
        [  2.0,  -9.0,   1.8,  -1.2],
        [ -1.5,   1.8,  -7.0,   1.0],
        [  0.8,  -1.2,   1.0,  -8.0],
    ])

    _INTERCEPT = 80.0
    _NOISE_SIGMA = 1.5

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._true_optimum_coded = self._solve_optimum()

    def _solve_optimum(self) -> np.ndarray:
        """Solve H·x = -β for the stationary point in coded space."""
        b = -self._BETA_LINEAR
        # The stationary point is found by solving d(y)/dx = 0 => H*x = -beta
        x_opt = np.linalg.solve(self._HESSIAN, b)
        return x_opt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, params: dict[str, float]) -> float:
        """Evaluate yield at given real-world parameters (with noise).

        Parameters
        ----------
        params : dict
            Keys must include all FACTOR_NAMES; values in real units.

        Returns
        -------
        float
            Simulated yield (%) with additive Gaussian noise.
        """
        coded = np.array([self._to_coded(name, params[name])
                          for name in self.FACTOR_NAMES])
        y_true = self._response(coded)
        noise = self._rng.normal(0.0, self._NOISE_SIGMA)
        return y_true + noise

    def evaluate_noiseless(self, params: dict[str, float]) -> float:
        """Evaluate yield *without* noise (for ground-truth checks)."""
        coded = np.array([self._to_coded(name, params[name])
                          for name in self.FACTOR_NAMES])
        return self._response(coded)

    def true_optimum(self) -> tuple[dict[str, float], float]:
        """Return the true optimum (real-world params, yield)."""
        params = {}
        for i, name in enumerate(self.FACTOR_NAMES):
            params[name] = self._to_real(name, self._true_optimum_coded[i])
        y_opt = self._response(self._true_optimum_coded)
        return params, y_opt

    def evaluate_coded(self, coded: np.ndarray) -> float:
        """Evaluate with noise from coded-space values."""
        y_true = self._response(coded)
        noise = self._rng.normal(0.0, self._NOISE_SIGMA)
        return y_true + noise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _response(self, x: np.ndarray) -> float:
        # y = beta_0 + beta^T*x + 0.5*x^T*H*x  (second-order response surface)
        """Noiseless response surface in coded space."""
        linear = np.dot(self._BETA_LINEAR, x)
        quadratic = 0.5 * np.dot(x, np.dot(self._HESSIAN, x))
        return self._INTERCEPT + linear + quadratic

    def _to_coded(self, name: str, value: float) -> float:
        # Maps real-world value to [-1, 1] coded space: x_coded = (value - midpoint) / half_range
        lo, hi = self.FACTOR_BOUNDS[name]
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        return (value - mid) / half_range

    def _to_real(self, name: str, coded: float) -> float:
        # Maps coded [-1, 1] back to real-world value: value = coded * half_range + midpoint
        lo, hi = self.FACTOR_BOUNDS[name]
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        return coded * half_range + mid


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == '__main__':
    proc = ChemicalProcess(seed=42)

    # Verify optimum is interior
    opt_params, opt_yield = proc.true_optimum()
    print('True optimum (noiseless):')
    for k, v in opt_params.items():
        lo, hi = proc.FACTOR_BOUNDS[k]
        print(f'  {k:>12s}: {v:8.3f}  (range {lo}-{hi})')
    print(f'  {"yield":>12s}: {opt_yield:8.3f} %')

    # Check coded-space optimum is in [-1, 1]
    coded_opt = proc._true_optimum_coded
    print(f'\nCoded optimum: {np.round(coded_opt, 4)}')
    assert np.all(np.abs(coded_opt) <= 1.0), 'Optimum outside design space!'

    # Quick noisy evaluation
    print(f'\nNoisy evaluation at optimum: {proc.evaluate(opt_params):.3f} %')
    print('Self-test passed.')
