from __future__ import annotations

import json
from typing import Annotated

import typer

from rcd.config import ProjectPaths, load_config
from rcd.data import load_participants, load_preprocessed_raw, set_path
from rcd.download import download_dataset, iter_expected_set_files
from rcd.mechanistic import run_mechanistic
from rcd.prediction import run_classical_prediction
from rcd.reporting import generate_reporting_outputs
from rcd.reproduction import run_reproduction
from rcd.state import run_state_analysis

app = typer.Typer(no_args_is_help=True, help="Rostrocaudal dementia EEG analysis")


@app.command("download")
def download_command(
    dataset: Annotated[str, typer.Option(help="OpenNeuro dataset ID")],
    derivatives_only: Annotated[
        bool, typer.Option("--derivatives-only/--include-raw", help="Download released derivatives")
    ] = True,
) -> None:
    paths = ProjectPaths.discover()
    config = load_config(paths.config)
    if dataset not in config["datasets"]:
        raise typer.BadParameter(f"Unknown dataset {dataset}")
    settings = config["datasets"][dataset]
    downloaded = download_dataset(
        dataset_id=dataset,
        repo_url=settings["metadata_repo"],
        s3_prefix=settings["s3_prefix"],
        data_root=paths.data,
        derivatives_only=derivatives_only,
    )
    typer.echo(f"Validated {len(downloaded)} EEG files for {dataset}")


@app.command("validate-data")
def validate_data() -> None:
    paths = ProjectPaths.discover()
    config = load_config(paths.config)
    report: dict[str, dict[str, object]] = {}
    for dataset_id, settings in config["datasets"].items():
        root = paths.data / dataset_id
        participants = load_participants(root, config["study"]["labels"])
        files = list(iter_expected_set_files(root))
        missing = [
            row["participant_id"]
            for _, row in participants.iterrows()
            if not set_path(
                root,
                row["participant_id"],
                settings["task"],
                settings.get("derivative_pipeline"),
            ).exists()
        ]
        if missing:
            raise FileNotFoundError(f"{dataset_id} missing EEG for: {missing}")
        sample = set_path(
            root,
            participants.iloc[0]["participant_id"],
            settings["task"],
            settings.get("derivative_pipeline"),
        )
        raw = load_preprocessed_raw(
            sample,
            config["study"]["channels"]["canonical"],
            l_freq=config["preprocessing"]["l_freq"],
            h_freq=config["preprocessing"]["h_freq"],
            resample_hz=config["preprocessing"]["resample_hz"],
            rereference=config["preprocessing"]["rereference"],
        )
        report[dataset_id] = {
            "participants": int(participants.shape[0]),
            "set_files": len(files),
            "sample_duration_seconds": raw.times[-1],
            "sample_channels": raw.ch_names,
            "sample_sfreq": raw.info["sfreq"],
        }
    output = paths.outputs / "data_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(output)


@app.command("run-reproduction")
def reproduction_command(
    n_jobs: Annotated[int, typer.Option(min=1, help="Parallel participant workers")] = 1,
) -> None:
    paths = ProjectPaths.discover()
    config = load_config(paths.config)
    outputs = run_reproduction(
        project_root=paths.root,
        config_path=paths.config,
        config=config,
        n_jobs=n_jobs,
    )
    for output in outputs:
        typer.echo(output)


@app.command("run-mechanistic")
def mechanistic_command(
    n_surrogates: Annotated[
        int, typer.Option(min=2, help="IAAFT surrogates per participant and channel")
    ] = 20,
    n_jobs: Annotated[
        int, typer.Option(min=1, help="Parallel participant workers")
    ] = 1,
) -> None:
    paths = ProjectPaths.discover()
    config = load_config(paths.config)
    outputs = run_mechanistic(
        project_root=paths.root,
        config_path=paths.config,
        config=config,
        n_surrogates=n_surrogates,
        n_jobs=n_jobs,
    )
    for output in outputs:
        typer.echo(output)


@app.command("run-state")
def state_command(
    interval_seconds: Annotated[
        float, typer.Option(min=1.0, help="Matched state interval length in seconds")
    ] = 2.0,
    n_jobs: Annotated[int, typer.Option(min=1, help="Parallel participant workers")] = 1,
) -> None:
    paths = ProjectPaths.discover()
    config = load_config(paths.config)
    outputs = run_state_analysis(
        project_root=paths.root,
        config_path=paths.config,
        config=config,
        interval_seconds=interval_seconds,
        n_jobs=n_jobs,
    )
    for output in outputs:
        typer.echo(output)


@app.command("run-classical")
def classical_command() -> None:
    paths = ProjectPaths.discover()
    config = load_config(paths.config)
    outputs = run_classical_prediction(
        project_root=paths.root,
        config_path=paths.config,
        config=config,
    )
    for output in outputs:
        typer.echo(output)


@app.command("run-deep")
def deep_command(
    repeats: Annotated[int, typer.Option(min=1, max=10, help="Outer split repeats")] = 10,
) -> None:
    from rcd.deep_prediction import run_deep_prediction

    paths = ProjectPaths.discover()
    config = load_config(paths.config)
    outputs = run_deep_prediction(
        project_root=paths.root,
        config_path=paths.config,
        config=config,
        repeats=repeats,
    )
    for output in outputs:
        typer.echo(output)


@app.command("make-report")
def report_command() -> None:
    paths = ProjectPaths.discover()
    config = load_config(paths.config)
    outputs = generate_reporting_outputs(
        project_root=paths.root,
        config_path=paths.config,
        config=config,
    )
    for output in outputs:
        typer.echo(output)


if __name__ == "__main__":
    app()
