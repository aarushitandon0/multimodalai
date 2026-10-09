"""Content hashing for dataset provenance (spec 4.6 step 5, 16).

Every audio file in the dataset carries a SHA-256 in ``LICENSES.csv`` and in
``dataset/checksums.sha256``, so a reviewer can confirm that the bytes behind a
reported metric are the bytes we downloaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["CHUNK_SIZE", "sha256_bytes", "sha256_file"]

CHUNK_SIZE = 1 << 20
"""Read size (1 MiB); large enough to keep hashing I/O-bound."""


def sha256_file(path: Path | str, *, chunk_size: int = CHUNK_SIZE) -> str:
    """Return the lower-case hex SHA-256 of a file's contents.

    Streamed, so a 100 MB download is hashed without being held in memory.

    Args:
        path: File to hash.
        chunk_size: Bytes per read.

    Raises:
        FileNotFoundError: ``path`` does not exist.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the lower-case hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()
