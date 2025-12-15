import kickscore as ks


def test_numba_fit_no_observations():
    m = ks.BinaryModel()
    m.add_item("A", kernel=ks.kernel.Constant(1.0))
    assert m.fit(backend="numba", max_iter=5)
