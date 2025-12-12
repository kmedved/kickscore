from collections.abc import Sequence
from math import exp, log  # Faster than numpy equivalents.

import numba

from ..item import Item
from .observation import Observation
from .utils import (
    K_POISSON,
    K_SKELLAM,
    cvi_expectations,
    gh_match_moments_weighted,
    iv,
    log_factorial,
    match_moments,
)


@match_moments
@cvi_expectations
@numba.jit(nopython=True)
def _ll_poisson(x: float, count: int) -> float:
    """Compute the log-likelihood of x under the Poisson model."""
    return float(x * count - log_factorial(count) - exp(x))


@match_moments
@cvi_expectations
@numba.njit(cache=True)
def _ll_poisson_const(x: float, count: int, log_fact: float) -> float:
    """Poisson log-likelihood using precomputed log-factorial."""
    return x * count - log_fact - exp(x)


class PoissonObservation(Observation):
    def __init__(
        self, items: Sequence[tuple[Item, float]], count: int, t: float, weight: float = 1.0
    ):
        super().__init__(items, t, weight=weight)
        self._count = int(count)
        self._log_fact = log_factorial(self._count)

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        if self.weight == 1.0:
            return _ll_poisson_const.match_moments(mean_cav, var_cav, self._count, self._log_fact)  # pyright: ignore[reportFunctionMemberAccess]
        return gh_match_moments_weighted(
            mean=mean_cav,
            var=var_cav,
            weight=self.weight,
            kind=K_POISSON,
            p1=float(self._count),
            p2=self._log_fact,
            p3=0.0,
        )

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        return _ll_poisson_const.cvi_expectations(mean, var, self._count, self._log_fact)  # pyright: ignore[reportFunctionMemberAccess]

    @staticmethod
    def probability(items: Sequence[tuple[Item, float]], count: int, t: float) -> float:
        m, v = Observation.f_params(items, t)
        log_fact = log_factorial(int(count))
        logpart, _, _ = _ll_poisson_const.match_moments(m, v, int(count), log_fact)  # pyright: ignore[reportFunctionMemberAccess]
        return exp(logpart)


@match_moments
@cvi_expectations
@numba.jit(nopython=True)
def _ll_skellam(x: float, diff: int, base_rate: float) -> float:
    """Compute the log-likelihood of x under the Skellam model."""
    base = exp(base_rate)
    return -(base * exp(x) + base * exp(-x)) + x * diff + log(iv(abs(diff), 2 * base))


@match_moments
@cvi_expectations
@numba.njit(cache=True)
def _ll_skellam_const(x: float, diff: int, base: float, log_iv_const: float) -> float:
    """Skellam log-likelihood using precomputed base and Bessel term."""
    return -(base * exp(x) + base * exp(-x)) + x * diff + log_iv_const


class SkellamObservation(Observation):
    def __init__(
        self, items: Sequence[tuple[Item, float]], diff: int, base_rate: float, t: float, weight: float = 1.0
    ):
        super().__init__(items, t, weight=weight)
        self._diff = int(diff)
        self._base_rate = base_rate
        self._base = exp(self._base_rate)
        self._log_iv = log(iv(abs(self._diff), 2.0 * self._base))

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        if self.weight == 1.0:
            return _ll_skellam_const.match_moments(mean_cav, var_cav, self._diff, self._base, self._log_iv)  # pyright: ignore[reportFunctionMemberAccess]
        return gh_match_moments_weighted(
            mean=mean_cav,
            var=var_cav,
            weight=self.weight,
            kind=K_SKELLAM,
            p1=float(self._diff),
            p2=self._base,
            p3=self._log_iv,
        )

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        return _ll_skellam_const.cvi_expectations(mean, var, self._diff, self._base, self._log_iv)  # pyright: ignore[reportFunctionMemberAccess]

    @staticmethod
    def probability(
        items: Sequence[tuple[Item, float]], diff: int, base_rate: float, t: float
    ) -> float:
        m, v = Observation.f_params(items, t)
        base = exp(base_rate)
        log_iv_const = log(iv(abs(diff), 2.0 * base))
        logpart, _, _ = _ll_skellam_const.match_moments(m, v, diff, base, log_iv_const)  # pyright: ignore[reportFunctionMemberAccess]
        return exp(logpart)
