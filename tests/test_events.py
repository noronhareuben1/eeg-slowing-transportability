from pathlib import Path

from rcd.data import photic_open_intervals


def test_photic_open_intervals_returns_columns_for_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "events.tsv"
    path.write_text("onset\tduration\tsample\tvalue\n0\t0\t0\tclosed eyes\n")
    intervals = photic_open_intervals(path)
    assert intervals.empty
    assert intervals.columns.tolist() == ["frequency_hz", "onset", "duration"]


def test_photic_open_intervals_tracks_frequency(tmp_path: Path) -> None:
    path = tmp_path / "events.tsv"
    path.write_text(
        "onset\tduration\tsample\tvalue\n"
        "1\t0\t250\tPHOTO 10Hz\n"
        "2\t0\t500\topen eyes\n"
        "5.5\t0\t1375\tclosed eyes\n"
    )
    intervals = photic_open_intervals(path)
    assert intervals.to_dict("records") == [
        {"frequency_hz": 10.0, "onset": 2.0, "duration": 3.5}
    ]
