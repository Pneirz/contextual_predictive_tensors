"""Unpack large CSV archives into data/processed.

The uncompressed order-3 CSVs are intentionally not tracked because they exceed
GitHub's 100 MB single-file limit. This helper restores them from the ZIP files
stored under data/large_archives.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = REPO_ROOT / "data" / "large_archives"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def unpack_archive(archive: Path) -> None:
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        zip_file.extractall(PROCESSED_DIR)
    print(f"Unpacked {archive.name} -> {PROCESSED_DIR}")


def main() -> None:
    for name in [
        "null_swap_order3_tree.csv.zip",
        "null_swap_order3_logreg.csv.zip",
    ]:
        unpack_archive(ARCHIVE_DIR / name)


if __name__ == "__main__":
    main()
