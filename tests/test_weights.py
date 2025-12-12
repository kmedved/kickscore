import numpy as np

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
