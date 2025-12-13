import numpy as np

import kickscore as ks


def _build_binary_model():
    kernel = ks.kernel.Constant(1.0)
    model = ks.BinaryModel(obs_type="probit")
    model.add_item("A", kernel=kernel)
    model.add_item("B", kernel=kernel)
    model.observe(winners=["A"], losers=["B"], t=0.0)
    model.observe(winners=["A"], losers=["B"], t=1.0)
    model.observe(winners=["B"], losers=["A"], t=2.0)
    return model


def test_numba_backend_matches_python():
    kwargs = {"max_iter": 25, "lr": 0.8, "tol": 1e-4}

    model_python = _build_binary_model()
    assert model_python.fit(backend="python", **kwargs)

    model_numba = _build_binary_model()
    assert model_numba.fit(backend="numba", **kwargs)

    for name in ["A", "B"]:
        _, ms_python, vs_python = model_python.item[name].scores
        _, ms_numba, vs_numba = model_numba.item[name].scores
        assert np.allclose(ms_python, ms_numba, rtol=1e-3)
        assert np.allclose(vs_python, vs_numba, rtol=1e-3)

    assert np.allclose(model_python.log_likelihood, model_numba.log_likelihood, rtol=1e-3)
