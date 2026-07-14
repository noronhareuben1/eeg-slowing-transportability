from pathlib import Path

from rcd.download import annex_inventory, copy_metadata


def test_annex_inventory_reads_windows_pointer_file(tmp_path: Path) -> None:
    pointer = tmp_path / "derivatives" / "eeglab" / "sub-001" / "eeg" / "recording.set"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        "../../.git/annex/objects/aa/bb/"
        "SHA256E-s123--0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.set/"
        "SHA256E-s123--0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.set",
        encoding="utf-8",
    )

    inventory = annex_inventory(tmp_path)

    assert len(inventory) == 1
    assert inventory[0].relative_path == pointer.relative_to(tmp_path)
    assert inventory[0].size == 123
    assert inventory[0].sha256 == (
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )


def test_copy_metadata_does_not_copy_annex_pointer(tmp_path: Path) -> None:
    source = tmp_path / "metadata"
    destination = tmp_path / "dataset"
    pointer = source / "derivatives" / "eeglab" / "sub-001" / "eeg" / "recording.set"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        "SHA256E-s123--0123456789abcdef0123456789abcdef"
        "0123456789abcdef0123456789abcdef.set",
        encoding="utf-8",
    )
    metadata = source / "participants.tsv"
    metadata.write_text("participant_id\nsub-001\n", encoding="utf-8")

    copy_metadata(source, destination)

    assert not (destination / pointer.relative_to(source)).exists()
    assert (destination / "participants.tsv").exists()
