"""
02_extract_dataset.py - extract ONLY limits.rda, read it with pyreadr, save CSV.

Stage: Data acquisition and understanding.

Why is the data an .rda file?
    The dataset was published inside an R package (SMPracticals, Davison 2003).
    R stores data objects in its own binary serialisation format (.rda/.RData).
    Python cannot read it with pd.read_csv, so we use `pyreadr`, a library that
    wraps the C library librdata to decode R's format into pandas DataFrames.

Output: data/raw/limits.rda       (extracted original, byte-for-byte)
        data/raw/limits_raw.csv   (same values, as CSV, NO cleaning)

Run   : python scripts/02_extract_dataset.py
"""

import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import pyreadr  # noqa: E402

from src.config import (  # noqa: E402
    ARCHIVE_MEMBER,
    ARCHIVE_PATH,
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    RAW_CSV_PATH,
    RDA_PATH,
)


def extract_rda() -> None:
    """Copy the single archive member to data/raw/limits.rda."""
    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as archive:
        member = archive.getmember(ARCHIVE_MEMBER)
        # extractfile() returns a file-like object for ONE member, so we never
        # unpack the other ~100 package files or run any R code. Writing the
        # bytes ourselves also avoids "path traversal" risks of extractall().
        source = archive.extractfile(member)
        if source is None:
            raise ValueError("limits.rda is not a regular file")
        RDA_PATH.write_bytes(source.read())
    print(f"Extracted {ARCHIVE_MEMBER} -> {RDA_PATH.relative_to(RDA_PATH.parents[2])}"
          f" ({RDA_PATH.stat().st_size:,} bytes)")


def read_rda() -> pd.DataFrame:
    """Read the R object and return it as a pandas DataFrame."""
    # read_r returns an ordered dictionary: {R object name: DataFrame}.
    result = pyreadr.read_r(str(RDA_PATH))
    print(f"R objects inside the file: {list(result.keys())}")
    df = result["limits"]
    # R data frames carry row names ("1".."92"); we use a clean 0..91 index.
    return df.reset_index(drop=True)


def main() -> None:
    extract_rda()
    df = read_rda()

    print("\nTypes exactly as decoded from R:")
    print(df.dtypes.to_string())

    assert df.shape == (EXPECTED_ROWS, len(EXPECTED_COLUMNS)), df.shape
    assert list(df.columns) == EXPECTED_COLUMNS, list(df.columns)
    print(f"\nShape check passed: {df.shape[0]} rows x {df.shape[1]} columns")

    # Save without modification. `day` is an R factor (category); CSV stores
    # its label text, which is the same 1..92 code.
    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"Saved faithful CSV copy -> {RAW_CSV_PATH.name}")

    print("\nFirst five observations:")
    print(df.head().to_string())
    print("\nLast five observations:")
    print(df.tail().to_string())


if __name__ == "__main__":
    main()
