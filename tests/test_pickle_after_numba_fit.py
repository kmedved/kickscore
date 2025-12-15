import pickle

import kickscore as ks


def test_pickle_after_numba_fit():
    m = ks.BinaryModel()
    m.add_item("A", ks.kernel.Constant(1.0))
    m.add_item("B", ks.kernel.Constant(1.0))
    m.observe(["A"], ["B"], t=0.0)
    m.fit(backend="numba", max_iter=5)
    pickle.loads(pickle.dumps(m))
