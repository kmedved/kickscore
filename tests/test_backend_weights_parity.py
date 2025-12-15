import numpy as np

import kickscore as ks


def test_numba_weighted_gaussian_matches_python():
    k = ks.kernel.Constant(1.0)

    m_py = ks.DifferenceModel(var=1.0)
    m_nb = ks.DifferenceModel(var=1.0)
    for m in (m_py, m_nb):
        m.add_item("a", kernel=k)
        m.add_item("b", kernel=k)

    m_py.observe(["a"], ["b"], diff=0.5, var=1.0, t=0.0, weight=10.0)
    m_nb.observe(["a"], ["b"], diff=0.5, var=1.0, t=0.0, weight=10.0)

    assert m_py.fit(backend="python", tol=1e-8, max_iter=50)
    assert m_nb.fit(backend="numba", tol=1e-8, max_iter=50)

    for name in ("a", "b"):
        _, ms_py, vs_py = m_py.item[name].scores
        _, ms_nb, vs_nb = m_nb.item[name].scores
        assert np.allclose(ms_py, ms_nb, rtol=1e-4, atol=1e-6)
        assert np.allclose(vs_py, vs_nb, rtol=1e-4, atol=1e-6)
