"""Shared parsing for the SVM-format peptide sequence files.

The format is an optional integer index, whitespace, then the sequence::

    1 MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPN
    2 GVVDSDDLPLVVAASNAGKSTVVQLLAAAG

Blank lines and ``#`` comments are skipped. A line carrying only a sequence
(no index) is assigned an auto-incremented 1-based index.

Extracted verbatim from three copies that were byte-identical (or a thin
wrapper) in ``models/batch_esmfold.py``, ``models/esm_sequence_processor.py``,
and ``models/run_esmfold_peptides.py``. Behavior is preserved exactly — see
``tests/test_imports.py::test_parse_sequence_file_*``.
"""
from __future__ import annotations

from typing import List, Tuple


def parse_sequence_file(input_file) -> List[Tuple[str, str]]:
    """Parse an SVM-format sequence file into ``(index, sequence)`` tuples.

    Both elements are ``str``. Indices present in the file are kept as-is;
    index-less lines get a ``str`` of their 1-based position among the parsed
    sequences (matching the original per-script behavior).
    """
    sequences: List[Tuple[str, str]] = []

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 1)  # split on whitespace, max 2 parts
            if len(parts) == 2:
                idx, seq = parts
                sequences.append((idx, seq.strip()))
            elif len(parts) == 1:
                seq = parts[0]
                idx = len(sequences) + 1
                sequences.append((str(idx), seq.strip()))

    return sequences
