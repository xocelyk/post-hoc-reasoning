"""Smoke tests: package imports, response parsing, probe training on synthetic data,
steering-vector computation, and config loading. No model downloads required."""

import glob

import numpy as np
import pytest

from post_hoc_reasoning.config import ConfigLoader
from post_hoc_reasoning.parsing_utils import parse_response
from post_hoc_reasoning.steering_methods import CAASingleLayerSteering
from post_hoc_reasoning.nnsight_utils.probes.base import convert_labels_to_binary
from post_hoc_reasoning.nnsight_utils.probes.logistic import train_logistic_probe


def test_parse_response_extracts_letter():
    letter, _ = parse_response("Let me think step by step. The best answer is: (A) yes")
    assert letter == "A"


def test_parse_response_uses_last_answer_mention():
    response = "The best answer is (B) no... wait. The best answer is: (A) yes"
    letter, _ = parse_response(response)
    assert letter == "A"


def test_parse_response_missing_pattern_returns_empty():
    assert parse_response("I refuse to answer.") == ("", "")


def test_parse_response_no_thinking_mode():
    letter, _ = parse_response("(B) no", thinking=False)
    assert letter == "B"


def test_convert_labels_to_binary():
    out = convert_labels_to_binary(["yes", "no", "yes"], positive_label="yes")
    assert out.tolist() == [1, 0, 1]


def test_logistic_probe_learns_synthetic_direction():
    rng = np.random.default_rng(0)
    n, n_layers, d = 80, 2, 8
    labels = ["yes" if i % 2 == 0 else "no" for i in range(n)]
    direction = rng.normal(size=d)
    activations = rng.normal(size=(n, n_layers, d)) * 0.1
    signs = np.array([1.0 if l == "yes" else -1.0 for l in labels])
    activations += signs[:, None, None] * direction[None, None, :]

    result = train_logistic_probe(activations, labels)
    vectors = result.vectors
    assert len(vectors) == n_layers
    for vec in vectors.values():
        vec = np.asarray(vec)
        assert vec.shape[-1] == d
        assert np.isfinite(vec).all()
        cosine = abs(np.dot(vec.ravel(), direction) / (np.linalg.norm(vec) * np.linalg.norm(direction)))
        assert cosine > 0.9, "probe should recover the planted direction"


def test_caa_single_layer_steering_vectors():
    rng = np.random.default_rng(1)
    layer_vectors = [rng.normal(size=16) for _ in range(3)]
    method = CAASingleLayerSteering(similarity_scores=[0.2, 0.9, 0.5])
    out = method.compute_steering_vectors(layer_vectors)
    assert len(out) == len(layer_vectors)
    assert all(np.isfinite(np.asarray(v)).all() for v in out)


@pytest.mark.parametrize("path", sorted(glob.glob("configs/**/*.yaml", recursive=True)))
def test_configs_load(path):
    data = ConfigLoader.load_yaml(path)
    if "orthogonal_steering" in path:
        # orthogonal configs nest settings under an `experiment` block
        assert "cache_dir" in data.get("experiment", {})
        assert "steering" in data
    else:
        assert "datasets" in data and "cache_dir" in data
