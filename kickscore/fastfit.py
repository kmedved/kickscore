"""Packed, numba-accelerated fitting backend."""

from math import log
from typing import NamedTuple, Sequence

import numba
import numpy as np
from numba.typed import List

from .fitter.recursive import _fit as _kalman_fit
from .item import Item
from .observation.gaussian import (
    GaussianObservation,
    _cvi_gaussian,
    _mm_gaussian,
    _mm_gaussian_weighted,
)
from .observation.ordinal import (
    LogitTieObservation,
    LogitWinObservation,
    ProbitTieObservation,
    ProbitWinObservation,
    _mm_logit_win,
    _mm_probit_tie,
    _mm_probit_win,
)
from .observation.poisson import PoissonObservation, SkellamObservation
from .observation.utils import (
    K_GAUSSIAN,
    K_LOGIT_TIE,
    K_LOGIT_WIN,
    K_POISSON,
    K_PROBIT_TIE,
    K_PROBIT_WIN,
    K_SKELLAM,
    gh_cvi_expectations,
    gh_match_moments_weighted,
)


class PackedObservations(NamedTuple):
    ptr: np.ndarray
    edge_item: np.ndarray
    edge_idx: np.ndarray
    edge_coeff: np.ndarray
    kinds: np.ndarray
    weights: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray
    logpart: np.ndarray
    exp_ll: np.ndarray
    obs_ll: np.ndarray
    edge_x_cav: np.ndarray
    edge_n_cav: np.ndarray


def pack_observations(observations) -> tuple[PackedObservations, list[Item]]:
    item_ids: dict[Item, int] = {}
    items: list[Item] = []

    n_edges = 0
    for obs in observations:
        n_edges += obs._M
        for item in obs._items:
            if item not in item_ids:
                item_ids[item] = len(items)
                items.append(item)

    n_obs = len(observations)
    ptr = np.zeros(n_obs + 1, dtype=np.int64)
    edge_item = np.zeros(n_edges, dtype=np.int64)
    edge_idx = np.zeros(n_edges, dtype=np.int64)
    edge_coeff = np.zeros(n_edges, dtype=np.float64)

    kinds = np.zeros(n_obs, dtype=np.int64)
    weights = np.zeros(n_obs, dtype=np.float64)
    p1 = np.zeros(n_obs, dtype=np.float64)
    p2 = np.zeros(n_obs, dtype=np.float64)
    p3 = np.zeros(n_obs, dtype=np.float64)

    logpart = np.zeros(n_obs, dtype=np.float64)
    exp_ll = np.zeros(n_obs, dtype=np.float64)
    obs_ll = np.zeros(n_obs, dtype=np.float64)

    edge_x_cav = np.zeros(n_edges, dtype=np.float64)
    edge_n_cav = np.zeros(n_edges, dtype=np.float64)

    offset = 0
    for i, obs in enumerate(observations):
        ptr[i] = offset
        weights[i] = obs.weight
        logpart[i] = obs._logpart
        exp_ll[i] = obs._exp_ll

        if isinstance(obs, ProbitWinObservation):
            kinds[i] = K_PROBIT_WIN
            p1[i] = obs._margin
        elif isinstance(obs, ProbitTieObservation):
            kinds[i] = K_PROBIT_TIE
            p1[i] = obs._margin
        elif isinstance(obs, LogitWinObservation):
            kinds[i] = K_LOGIT_WIN
            p1[i] = obs._margin
        elif isinstance(obs, LogitTieObservation):
            kinds[i] = K_LOGIT_TIE
            p1[i] = obs._margin
        elif isinstance(obs, GaussianObservation):
            kinds[i] = K_GAUSSIAN
            p1[i] = obs._diff
            p2[i] = obs._var
        elif isinstance(obs, PoissonObservation):
            kinds[i] = K_POISSON
            p1[i] = float(obs._count)
            p2[i] = obs._log_fact
        elif isinstance(obs, SkellamObservation):
            kinds[i] = K_SKELLAM
            p1[i] = float(obs._diff)
            p2[i] = obs._base
            p3[i] = obs._log_iv
        else:
            raise ValueError(f"unsupported observation type: {type(obs)}")

        for j in range(obs._M):
            edge_item[offset + j] = item_ids[obs._items[j]]
            edge_idx[offset + j] = obs._indices[j]
            edge_coeff[offset + j] = obs._coeffs[j]
        offset += obs._M
    ptr[n_obs] = offset

    packed = PackedObservations(
        ptr=ptr,
        edge_item=edge_item,
        edge_idx=edge_idx,
        edge_coeff=edge_coeff,
        kinds=kinds,
        weights=weights,
        p1=p1,
        p2=p2,
        p3=p3,
        logpart=logpart,
        exp_ll=exp_ll,
        obs_ll=obs_ll,
        edge_x_cav=edge_x_cav,
        edge_n_cav=edge_n_cav,
    )
    return packed, items


def build_array_lists(items: Sequence[Item]):
    ms_list = List()
    vs_list = List()
    ns_list = List()
    xs_list = List()
    for item in items:
        ms_list.append(item.fitter.ms)
        vs_list.append(item.fitter.vs)
        ns_list.append(item.fitter.ns)
        xs_list.append(item.fitter.xs)
    return ms_list, vs_list, ns_list, xs_list


def build_recursive_lists(items: Sequence[Item]):
    ts_list = List()
    ms_list = List()
    vs_list = List()
    ns_list = List()
    xs_list = List()
    h_list = List()
    I_list = List()
    A_list = List()
    Q_list = List()
    m_p_list = List()
    P_p_list = List()
    m_f_list = List()
    P_f_list = List()
    m_s_list = List()
    P_s_list = List()
    for item in items:
        fitter = item.fitter
        ts_list.append(fitter.ts)
        ms_list.append(fitter.ms)
        vs_list.append(fitter.vs)
        ns_list.append(fitter.ns)
        xs_list.append(fitter.xs)
        h_list.append(fitter._h)
        I_list.append(fitter._I)
        A_list.append(fitter._A)
        Q_list.append(fitter._Q)
        m_p_list.append(fitter._m_p)
        P_p_list.append(fitter._P_p)
        m_f_list.append(fitter._m_f)
        P_f_list.append(fitter._P_f)
        m_s_list.append(fitter._m_s)
        P_s_list.append(fitter._P_s)

    return (
        ts_list,
        ms_list,
        vs_list,
        ns_list,
        xs_list,
        h_list,
        I_list,
        A_list,
        Q_list,
        m_p_list,
        P_p_list,
        m_f_list,
        P_f_list,
        m_s_list,
        P_s_list,
    )


@numba.njit(cache=True)
def _ep_sweep(
    ptr,
    edge_item,
    edge_idx,
    edge_coeff,
    kinds,
    weights,
    p1,
    p2,
    p3,
    ms_list,
    vs_list,
    ns_list,
    xs_list,
    edge_x_cav,
    edge_n_cav,
    logpart,
    lr,
):
    max_diff = 0.0
    n_obs = len(weights)
    for i in range(n_obs):
        start = ptr[i]
        end = ptr[i + 1]
        f_mean_cav = 0.0
        f_var_cav = 0.0
        for j in range(start, end):
            item_id = edge_item[j]
            idx = edge_idx[j]
            coeff = edge_coeff[j]
            x_tot = 1.0 / vs_list[item_id][idx]
            n_tot = x_tot * ms_list[item_id][idx]
            x_cav = x_tot - xs_list[item_id][idx]
            n_cav = n_tot - ns_list[item_id][idx]
            edge_x_cav[j] = x_cav
            edge_n_cav[j] = n_cav
            f_mean_cav += coeff * n_cav / x_cav
            f_var_cav += coeff * coeff / x_cav

        kind = kinds[i]
        w = weights[i]
        current_logpart = logpart[i]
        if w == 1.0 and kind == K_PROBIT_WIN:
            logpart_new, dlogpart, d2logpart = _mm_probit_win(f_mean_cav - p1[i], f_var_cav)
        elif w == 1.0 and kind == K_PROBIT_TIE:
            logpart_new, dlogpart, d2logpart = _mm_probit_tie(f_mean_cav, f_var_cav, p1[i])
        elif w == 1.0 and kind == K_LOGIT_WIN:
            logpart_new, dlogpart, d2logpart = _mm_logit_win(f_mean_cav - p1[i], f_var_cav)
        elif kind == K_GAUSSIAN and w == 1.0:
            logpart_new, dlogpart, d2logpart = _mm_gaussian(f_mean_cav, f_var_cav, p1[i], p2[i])
        elif kind == K_GAUSSIAN:
            logpart_new, dlogpart, d2logpart = _mm_gaussian_weighted(
                f_mean_cav, f_var_cav, p1[i], p2[i], w
            )
        else:
            logpart_new, dlogpart, d2logpart = gh_match_moments_weighted(
                f_mean_cav, f_var_cav, w, kind, p1[i], p2[i], p3[i]
            )

        for j in range(start, end):
            item_id = edge_item[j]
            idx = edge_idx[j]
            coeff = edge_coeff[j]
            x_cav = edge_x_cav[j]
            n_cav = edge_n_cav[j]
            denom = 1.0 + coeff * coeff * d2logpart / x_cav
            x = -coeff * coeff * d2logpart / denom
            n = coeff * (dlogpart - coeff * (n_cav / x_cav) * d2logpart) / denom
            xs_list[item_id][idx] = (1.0 - lr) * xs_list[item_id][idx] + lr * x
            ns_list[item_id][idx] = (1.0 - lr) * ns_list[item_id][idx] + lr * n

        max_diff = max(max_diff, abs(current_logpart - logpart_new))
        logpart[i] = logpart_new

    return max_diff


@numba.njit(cache=True)
def _ep_compute_obs_ll(
    ptr,
    edge_item,
    edge_idx,
    logpart,
    edge_x_cav,
    edge_n_cav,
    xs_list,
    ns_list,
    out,
):
    n_obs = len(logpart)
    for i in range(n_obs):
        start = ptr[i]
        end = ptr[i + 1]
        loglik = logpart[i]
        for j in range(start, end):
            item_id = edge_item[j]
            idx = edge_idx[j]
            x_cav = edge_x_cav[j]
            n_cav = edge_n_cav[j]
            x = xs_list[item_id][idx]
            n = ns_list[item_id][idx]
            loglik += 0.5 * log(x / x_cav + 1.0) + (
                (-(n * n)) - 2.0 * n * n_cav + x * n_cav * n_cav / x_cav
            ) / (2.0 * (x + x_cav))
        out[i] = loglik


@numba.njit(cache=True)
def _kl_sweep(
    ptr,
    edge_item,
    edge_idx,
    edge_coeff,
    kinds,
    weights,
    p1,
    p2,
    p3,
    ms_list,
    vs_list,
    ns_list,
    xs_list,
    exp_ll,
    obs_ll,
    lr,
):
    max_diff = 0.0
    n_obs = len(weights)
    for i in range(n_obs):
        start = ptr[i]
        end = ptr[i + 1]
        f_mean = 0.0
        f_var = 0.0
        for j in range(start, end):
            item_id = edge_item[j]
            idx = edge_idx[j]
            coeff = edge_coeff[j]
            f_mean += coeff * ms_list[item_id][idx]
            f_var += coeff * coeff * vs_list[item_id][idx]

        kind = kinds[i]
        w = weights[i]
        if kind == K_GAUSSIAN:
            exp_ll_new, alpha, beta = _cvi_gaussian(f_mean, f_var, p1[i], p2[i])
        else:
            exp_ll_new, alpha, beta = gh_cvi_expectations(f_mean, f_var, kind, p1[i], p2[i], p3[i])

        exp_ll_new *= w
        alpha *= w
        beta *= w

        for j in range(start, end):
            item_id = edge_item[j]
            idx = edge_idx[j]
            coeff = edge_coeff[j]
            x = -2.0 * coeff * coeff * beta
            n = coeff * (alpha - 2.0 * ms_list[item_id][idx] * coeff * beta)
            xs_list[item_id][idx] = (1.0 - lr) * xs_list[item_id][idx] + lr * x
            ns_list[item_id][idx] = (1.0 - lr) * ns_list[item_id][idx] + lr * n

        max_diff = max(max_diff, abs(exp_ll[i] - exp_ll_new))
        exp_ll[i] = exp_ll_new
        obs_ll[i] = exp_ll_new
    return max_diff


def ep_sweep(packed: PackedObservations, arrays, lr: float) -> float:
    ms_list, vs_list, ns_list, xs_list = arrays
    return _ep_sweep(
        packed.ptr,
        packed.edge_item,
        packed.edge_idx,
        packed.edge_coeff,
        packed.kinds,
        packed.weights,
        packed.p1,
        packed.p2,
        packed.p3,
        ms_list,
        vs_list,
        ns_list,
        xs_list,
        packed.edge_x_cav,
        packed.edge_n_cav,
        packed.logpart,
        lr,
    )


def kl_sweep(packed: PackedObservations, arrays, lr: float) -> float:
    ms_list, vs_list, ns_list, xs_list = arrays
    return _kl_sweep(
        packed.ptr,
        packed.edge_item,
        packed.edge_idx,
        packed.edge_coeff,
        packed.kinds,
        packed.weights,
        packed.p1,
        packed.p2,
        packed.p3,
        ms_list,
        vs_list,
        ns_list,
        xs_list,
        packed.exp_ll,
        packed.obs_ll,
        lr,
    )


def ep_compute_obs_ll(packed: PackedObservations, arrays) -> None:
    _, _, ns_list, xs_list = arrays
    _ep_compute_obs_ll(
        packed.ptr,
        packed.edge_item,
        packed.edge_idx,
        packed.logpart,
        packed.edge_x_cav,
        packed.edge_n_cav,
        xs_list,
        ns_list,
        packed.obs_ll,
    )


@numba.njit(cache=True, parallel=True)
def _fit_all_recursive(
    ts_list,
    ms_list,
    vs_list,
    ns_list,
    xs_list,
    h_list,
    I_list,
    A_list,
    Q_list,
    m_p_list,
    P_p_list,
    m_f_list,
    P_f_list,
    m_s_list,
    P_s_list,
):
    n = len(ts_list)
    for i in numba.prange(n):
        n_obs = int(ts_list[i].size)
        if n_obs == 0:
            continue
        _kalman_fit(
            ts_list[i],
            ms_list[i],
            vs_list[i],
            ns_list[i],
            xs_list[i],
            h_list[i],
            I_list[i],
            A_list[i],
            Q_list[i],
            m_p_list[i],
            P_p_list[i],
            m_f_list[i],
            P_f_list[i],
            m_s_list[i],
            P_s_list[i],
        )


def fit_all_recursive_items(recursive_arrays) -> None:
    _fit_all_recursive(*recursive_arrays)
