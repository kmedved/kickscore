import json

import kickscore as ks
import numpy as np


def _load_json_no_comments(path: str) -> dict:
    with open(path) as f:
        raw = "".join(line for line in f if not line.startswith("//"))
    return json.loads(raw)


def _build_model(data: dict):
    model_class = getattr(ks, data["model_class"])
    model = model_class(**data.get("model_args", {}))

    for item in data["items"]:
        kernel_class = getattr(ks.kernel, item["kernel_class"])
        kernel = kernel_class(**item["kernel_args"])
        model.add_item(item["name"], kernel=kernel)

    for obs in data["observations"]:
        model.observe(**obs)

    return model


def test_numba_backend_matches_python_on_all_json(testcase_path: str):
    data = _load_json_no_comments(testcase_path)

    fit_args = dict(data.get("fit_args", {}))
    fit_args_python = dict(fit_args)
    fit_args_numba = dict(fit_args)

    m_py = _build_model(data)
    m_py.fit(backend="python", **fit_args_python)

    m_nb = _build_model(data)
    m_nb.fit(backend="numba", **fit_args_numba)

    for name in data["scores"].keys():
        _, ms_py, vs_py = m_py.item[name].scores
        _, ms_nb, vs_nb = m_nb.item[name].scores
        assert np.allclose(ms_py, ms_nb, rtol=1e-3, atol=1e-6)
        assert np.allclose(vs_py, vs_nb, rtol=1e-3, atol=1e-6)

    assert np.allclose(m_py.log_likelihood, m_nb.log_likelihood, rtol=1e-3, atol=1e-6)
