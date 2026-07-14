import numpy as np
import pandas as pd

from transportability.run_amendment_v1_2 import aggregate_predictions, usable_columns


def test_aggregate_predictions_is_invariant_to_fold_row_order() -> None:
    rows = []
    for repeat in (0, 1):
        for participant, diagnosis, probability in (
            ("sub-001", "AD", (0.8, 0.1, 0.1)),
            ("sub-002", "CN", (0.1, 0.8, 0.1)),
        ):
            rows.append(
                {
                    "model": "paired_direct",
                    "repeat": repeat,
                    "fold": 1 - repeat,
                    "participant_id": participant,
                    "diagnosis": diagnosis,
                    "probability_AD": probability[0],
                    "probability_CN": probability[1],
                    "probability_FTD": probability[2],
                }
            )
    predictions = pd.DataFrame(rows).sample(frac=1.0, random_state=5)

    aggregated = aggregate_predictions(predictions)["paired_direct"]

    assert aggregated["participant_id"].tolist() == ["sub-001", "sub-002"]
    assert np.allclose(aggregated["probability_AD"], [0.8, 0.1])


def test_usable_columns_removes_sparse_and_constant_candidates() -> None:
    table = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0, np.nan],
            "sparse": [1.0, np.nan, np.nan, np.nan],
            "constant": [2.0, 2.0, 2.0, 2.0],
        }
    )

    assert usable_columns(table, ["good", "sparse", "constant"]) == ["good"]
