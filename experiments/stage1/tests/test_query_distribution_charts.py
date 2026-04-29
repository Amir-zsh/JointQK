from __future__ import annotations

import torch

from experiments.stage1.plots.query_distribution_charts import compute_skew_kurtosis


def test_query_distribution_skew_summary_does_not_cancel_opposite_sign_coordinates():
    samples = torch.tensor(
        [
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [10.0, -10.0],
                ]
            ]
        ]
    )

    skew, kurt = compute_skew_kurtosis(samples)

    assert skew.shape == (1, 1)
    assert kurt.shape == (1, 1)
    assert float(skew.item()) > 1.0
