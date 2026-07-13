"""Build the locked derivation feature table from completed ds004504 outputs."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd


def build_derivation_features(regional_features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "participant_id",
        "diagnosis",
        "delta_relative_caudal",
        "alpha_relative_caudal",
        "aperiodic_exponent_global",
    }
    missing = required.difference(regional_features.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = regional_features.copy()
    output = pd.DataFrame(
        {
            "participant_id": frame["participant_id"],
            "diagnosis": frame["diagnosis"],
            "posterior_delta_alpha_ratio": frame["delta_relative_caudal"]
            / np.maximum(frame["alpha_relative_caudal"], np.finfo(float).tiny),
            "posterior_alpha_relative": frame["alpha_relative_caudal"],
            "global_aperiodic_exponent": frame["aperiodic_exponent_global"],
        }
    )
    for optional in ("Age", "Gender", "MMSE"):
        if optional in frame.columns:
            output[optional.lower()] = frame[optional]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regional", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = build_derivation_features(pd.read_csv(args.regional))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
