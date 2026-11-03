"""
01_inspect_archive.py - look inside the CRAN package WITHOUT extracting it.

Stage: Data acquisition.
Goal : prove that the official archive we downloaded really contains the
       dataset file SMPracticals/data/limits.rda, and record its size.

Run  : python scripts/01_inspect_archive.py
"""

import sys
import tarfile
from pathlib import Path

# Make `src` importable when the script is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARCHIVE_MEMBER, ARCHIVE_PATH  # noqa: E402


def main() -> None:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"CRAN archive not found at {ARCHIVE_PATH}")

    print(f"Archive : {ARCHIVE_PATH.name}")
    print(f"Size    : {ARCHIVE_PATH.stat().st_size:,} bytes")

    # mode="r:gz" = open for READING a gzip-compressed tar archive.
    # The `with` block closes the file automatically, even on errors.
    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as archive:
        all_files = archive.getnames()            # every path inside the archive
        data_files = [n for n in all_files if "/data/" in n]
        matches = [n for n in all_files if n.endswith("/data/limits.rda")]

        print(f"Total files in package : {len(all_files)}")
        print(f"Files in data/ folder  : {len(data_files)}")

        if len(matches) != 1:
            raise ValueError(f"Expected exactly one limits.rda, found {len(matches)}")

        member = archive.getmember(matches[0])
        print(f"Dataset file found     : {member.name}")
        print(f"Dataset file size      : {member.size:,} bytes")
        assert member.name == ARCHIVE_MEMBER

        # The package also ships the R help page for the dataset - print it so
        # the documentation we cite is the one inside the exact archive we use.
        doc = archive.extractfile("SMPracticals/man/limits.Rd")
        if doc is not None:
            print("\n--- Official documentation (SMPracticals/man/limits.Rd) ---")
            print(doc.read().decode("utf-8"))

    print("Archive inspection successful!")


if __name__ == "__main__":
    main()
