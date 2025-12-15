import numpy as np
import pytest

import kickscore as ks


def test_gaussian_weight_matches_scaled_variance():
    kernel = ks.kernel.Constant(1.0)

    weighted = ks.DifferenceModel(var=1.0)
    unweighted = ks.DifferenceModel(var=0.1)
    for model in (weighted, unweighted):
        model.add_item("a", kernel=kernel)
        model.add_item("b", kernel=kernel)

    weighted.observe(["a"], ["b"], diff=0.5, var=1.0, t=0.0, weight=10.0)
    unweighted.observe(["a"], ["b"], diff=0.5, var=0.1, t=0.0)

    assert weighted.fit(max_iter=50, tol=1e-8)
    assert unweighted.fit(max_iter=50, tol=1e-8)

    for name in ("a", "b"):
        _, mean_w, var_w = weighted.item[name].scores
        _, mean_u, var_u = unweighted.item[name].scores
        np.testing.assert_allclose(mean_w, mean_u)
        np.testing.assert_allclose(var_w, var_u)


def test_non_positive_weight_rejected():
    kernel = ks.kernel.Constant(1.0)
    model = ks.BinaryModel()
    model.add_item("x", kernel=kernel)
    with np.testing.assert_raises(ValueError):
        model.observe(winners=["x"], losers=[], t=0.0, weight=0.0)


OBSERVATION_SCENARIOS = (
    (
        "probit_win",
        lambda: ks.BinaryModel(obs_type="probit"),
        lambda model, weight: model.observe(["a"], ["b"], t=0.0, weight=weight),
        ("a", "b"),
    ),
    (
        "logit_win",
        lambda: ks.BinaryModel(obs_type="logit"),
        lambda model, weight: model.observe(["a"], ["b"], t=0.0, weight=weight),
        ("a", "b"),
    ),
    (
        "probit_tie",
        lambda: ks.TernaryModel(margin=0.1, obs_type="probit"),
        lambda model, weight: model.observe(
            ["a"], ["b"], t=0.0, tie=True, margin=0.1, weight=weight
        ),
        ("a", "b"),
    ),
    (
        "logit_tie",
        lambda: ks.TernaryModel(margin=0.1, obs_type="logit"),
        lambda model, weight: model.observe(
            ["a"], ["b"], t=0.0, tie=True, margin=0.1, weight=weight
        ),
        ("a", "b"),
    ),
    (
        "poisson",
        ks.CountModel,
        lambda model, weight: model.observe(["a"], ["b"], count=3, t=0.0, weight=weight),
        ("a", "b"),
    ),
    (
        "skellam",
        ks.CountDiffModel,
        lambda model, weight: model.observe(["a"], ["b"], diff=-1, t=0.0, weight=weight),
        ("a", "b"),
    ),
)


def _build_model(factory, item_names: tuple[str, ...]):
    model = factory()
    kernel = ks.kernel.Constant(1.0)
    for name in item_names:
        model.add_item(name, kernel=kernel)
    return model


@pytest.mark.parametrize("backend", ["python", "numba"])
@pytest.mark.parametrize("_, factory, observer, item_names", OBSERVATION_SCENARIOS)
def test_weighted_observations_match_repeated_events(backend, _, factory, observer, item_names):
    weighted_model = _build_model(factory, item_names)
    observer(weighted_model, weight=2.0)

    unweighted_model = _build_model(factory, item_names)
    observer(unweighted_model, weight=1.0)
    observer(unweighted_model, weight=1.0)

    fit_kwargs = {
        "backend": backend,
        "method": "kl",
        "lr": 0.5,
        "max_iter": 400,
        "tol": 1e-5,
    }
    ok_w = weighted_model.fit(**fit_kwargs)
    ok_u = unweighted_model.fit(**fit_kwargs)
    assert ok_w == ok_u

    for name in item_names:
        _, mean_w, var_w = weighted_model.item[name].scores
        _, mean_u, var_u = unweighted_model.item[name].scores
        np.testing.assert_allclose(mean_w[-1], mean_u[-1], rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(var_w[-1], var_u[-1], rtol=1e-3, atol=1e-3)

    np.testing.assert_allclose(
        weighted_model.log_likelihood, unweighted_model.log_likelihood, rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("_, factory, observer, item_names", OBSERVATION_SCENARIOS)
def test_invalid_weight_rejected_across_observations(_, factory, observer, item_names):
    model = _build_model(factory, item_names)
    with pytest.raises(ValueError):
        observer(model, weight=0.0)
