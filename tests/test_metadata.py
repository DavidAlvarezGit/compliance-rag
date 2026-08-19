from pathlib import Path

from src.metadata import BASE_DIR, load_metadata, resolve_local_path, validate_metadata


def test_registry_uses_portable_relative_paths() -> None:
    metadata = load_metadata()

    assert all(not Path(value).is_absolute() for value in metadata["local_path"])
    validate_metadata(metadata)


def test_relative_path_is_resolved_against_repository() -> None:
    resolved = resolve_local_path("data/raw_pdf/example.pdf")

    assert resolved == BASE_DIR / "data" / "raw_pdf" / "example.pdf"
