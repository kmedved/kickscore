from collections.abc import Sequence
from math import log, pi, sqrt  # Faster than numpy equivalents.

import numba

from ..item import Item
from .observation import Observation
from .utils import normcdf


@numba.jit(nopython=True)
def _mm_gaussian(
    mean_cav: float,
    var_cav: float,
    diff: float,
    var_obs: float,
) -> tuple[float, float, float]:
    logpart = -0.5 * (
        log(2 * pi * (var_obs + var_cav)) + (diff - mean_cav) ** 2 / (var_obs + var_cav)
    )
    dlogpart = (diff - mean_cav) / (var_obs + var_cav)
    d2logpart = -1.0 / (var_obs + var_cav)
    return logpart, dlogpart, d2logpart


@numba.njit(cache=True)
def _mm_gaussian_weighted(
    mean_cav: float,
    var_cav: float,
    diff: float,
    var_obs: float,
    weight: float,
) -> tuple[float, float, float]:
    var_eff = var_obs / weight
    logpart, dlogpart, d2logpart = _mm_gaussian(mean_cav, var_cav, diff, var_eff)
    logc = -0.5 * (weight - 1.0) * log(2.0 * pi * var_obs) - 0.5 * log(weight)
    return logpart + logc, dlogpart, d2logpart


@numba.jit(nopython=True)
def _cvi_gaussian(mean: float, var: float, diff: float, var_obs: float) -> tuple[float, float, float]:
    exp_ll = -0.5 * (log(2.0 * pi * var_obs) + (((diff - mean) ** 2) + var) / var_obs)
    alpha = (diff - mean) / var_obs
    beta = -0.5 / var_obs
    return exp_ll, alpha, beta


class GaussianObservation(Observation):
    def __init__(
        self, items: Sequence[tuple[Item, float]], diff: float, var: float, t: float, weight: float = 1.0
    ):
        if var <= 0:
            raise ValueError("GaussianObservation var must be > 0")
        super().__init__(items, t, weight=weight)
        self._diff = diff
        self._var = var

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        if self.weight == 1.0:
            return _mm_gaussian(mean_cav, var_cav, self._diff, self._var)
        return _mm_gaussian_weighted(mean_cav, var_cav, self._diff, self._var, self.weight)

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        return _cvi_gaussian(mean, var, self._diff, self._var)

    @staticmethod
    def probability(
        items: Sequence[tuple[Item, float]], threshold: float, var: float, t: float
    ) -> float:
        m, v = Observation.f_params(items, t)
        return normcdf((m - threshold) / sqrt(var + v))
