"""CPU tests for the projection-figure pipeline's label and alignment math."""

import numpy as np

from scripts.generate_projection_figures import aligned_auc, semantic_label


def test_semantic_label_uses_randomized_mapping():
    # model picked the correct letter -> semantic answer is the correct answer
    assert semantic_label({"pred_letter": "A", "correct_letter": "A",
                           "correct_answer": "yes"}) == "yes"
    # model picked the other letter -> semantic answer flips
    assert semantic_label({"pred_letter": "B", "correct_letter": "A",
                           "correct_answer": "yes"}) == "no"
    assert semantic_label({"pred_letter": "A", "correct_letter": "B",
                           "correct_answer": "no"}) == "yes"


def test_aligned_auc_finds_planted_signal_after_boundary():
    rng = np.random.default_rng(0)
    n, layers, win = 40, 2, 20
    projs, metas = [], []
    for i in range(n):
        label = "yes" if i % 2 == 0 else "no"
        boundary, length = 30, 70
        P = rng.normal(size=(layers, length))
        # plant class signal at layer 1, positions AFTER the boundary only
        P[1, boundary:] += 3.0 if label == "yes" else -3.0
        projs.append(P)
        metas.append({"boundary": boundary, "label": label, "n_tokens": length})

    xs, auc_l1 = aligned_auc(projs, metas, layer=1, win=win, min_per_class=5)
    xs, auc_l0 = aligned_auc(projs, metas, layer=0, win=win, min_per_class=5)
    pre, post = xs < 0, xs >= 0
    assert np.nanmean(auc_l1[post]) > 0.95      # signal where planted
    assert abs(np.nanmean(auc_l1[pre]) - 0.5) < 0.15   # chance before boundary
    assert abs(np.nanmean(auc_l0[post]) - 0.5) < 0.15  # chance at other layer
