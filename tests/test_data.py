from pathlib import Path

from rcd.data import set_path


def test_set_path_supports_named_derivative_pipeline() -> None:
    root = Path("data/ds006036")
    expected = (
        root
        / "derivatives"
        / "eeglab"
        / "sub-001"
        / "eeg"
        / "sub-001_task-photomark_eeg.set"
    )
    assert set_path(root, "sub-001", "photomark", "eeglab") == expected


def test_set_path_supports_direct_derivatives() -> None:
    root = Path("data/ds004504")
    expected = root / "derivatives" / "sub-001" / "eeg" / "sub-001_task-eyesclosed_eeg.set"
    assert set_path(root, "sub-001", "eyesclosed") == expected
