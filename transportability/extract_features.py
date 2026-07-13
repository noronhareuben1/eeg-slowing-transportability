"""Extract P-ADIC group-level feature tables after raw files are available."""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from .eeg_features import extract_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--sfreq", type=float, default=500.0)
    args = parser.parse_args()
    frames = []
    for filename, diagnosis in (("alz_c1_new.mat", "AD"), ("controls_c1_new.mat", "CN")):
        path = args.input / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(extract_group(path, diagnosis, sfreq=args.sfreq))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
