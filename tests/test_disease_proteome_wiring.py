"""The off-target axis and the file this program downloads for it.

Two gaps sat between them, and together they meant the axis reported
``proteome_not_configured`` or ``proteome_unreadable`` on a machine that had
already fetched exactly the file it wanted:

* UniProt ships the reference proteome gzipped, and the registry fetches
  ``UP000005640_9606.fasta.gz`` unchanged. The reader called ``read_text`` on
  it, so the one shape the program downloads was the one shape it could not
  open.
* Nothing told the screen where that download was. The axis takes its FASTA on
  ``--proteome`` or from ``HUMAN_PROTEOME``, and neither knows the registry
  exists.

The axis matters more than a missing table usually would. Its clear means "no
9-mer of this design occurs anywhere in the human proteome", and an unconfigured
proteome cannot distinguish that from "nothing was compared". The reader already
guards the near misses -- a nucleotide FASTA, a fragment file, a proteome too
small to be one -- because each would read as a clear against nothing.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pytest

from rfproteina.disease.liability import _load_proteome
from rfproteina.disease.tables import downloaded_proteome

_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
#: The reader refuses a proteome under a million distinct 9-mers, because every
#: 9-mer the file lacks is an off-target it would clear a design against. So a
#: fixture that is merely "a few proteins" is refused as ``proteome_too_small``
#: and proves nothing about reading one. 2,800 varied 400-mers give ~1.1M
#: windows, which clears it; a cyclic alphabet does not, because its windows
#: repeat and almost none are distinct.
_PROTEIN_COUNT = 2_800
_PROTEIN_LENGTH = 400


def _seeded_text() -> str:
    """A proteome-sized FASTA, generated once and deterministically.

    Built at import rather than per test: three tests read the same content and
    regenerating a megabyte of sequence for each is wasted time. Seeded, so a
    failure here is reproducible.
    """
    import random

    rolls = random.Random(20260115)
    parts: list[str] = []
    for index in range(_PROTEIN_COUNT):
        body = "".join(rolls.choice(_ALPHABET) for _ in range(_PROTEIN_LENGTH))
        parts.append(
            f">sp|P{index:05d}|GENE{index}_HUMAN x OS=Homo sapiens OX=9606 " f"GN=GENE{index} PE=1 SV=1\n{body}\n"
        )
    return "".join(parts)


_FASTA_TEXT = _seeded_text()


def fasta_text() -> str:
    return _FASTA_TEXT


def test_a_gzipped_reference_proteome_is_read(tmp_path: Path) -> None:
    """The shape UniProt serves, and the registry fetches without unpacking."""
    path = tmp_path / "UP000005640_9606.fasta.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(fasta_text())
    blocker, kmers = _load_proteome(path)
    assert blocker is None, blocker
    assert kmers.size > 0


def test_a_plain_fasta_is_read_exactly_as_before(tmp_path: Path) -> None:
    """The other half of adding gzip: not breaking what already worked."""
    path = tmp_path / "proteome.fasta"
    path.write_text(fasta_text(), encoding="utf-8")
    blocker, kmers = _load_proteome(path)
    assert blocker is None, blocker
    assert kmers.size > 0


def test_the_same_content_packs_the_same_either_way(tmp_path: Path) -> None:
    """Decided on the extension, so the two paths must agree on the result.

    If they did not, whether a design read as an off-target hit would depend on
    how its proteome happened to be stored.
    """
    text = fasta_text()
    plain = tmp_path / "proteome.fasta"
    plain.write_text(text, encoding="utf-8")
    packed = tmp_path / "proteome.fasta.gz"
    with gzip.open(packed, "wt", encoding="utf-8") as handle:
        handle.write(text)

    plain_blocker, plain_kmers = _load_proteome(plain)
    gz_blocker, gz_kmers = _load_proteome(packed)
    assert plain_blocker is gz_blocker is None
    assert plain_kmers.size == gz_kmers.size
    assert (plain_kmers == gz_kmers).all()


def test_a_file_named_gz_that_is_not_gzipped_is_a_blocker_not_a_crash(tmp_path: Path) -> None:
    """``BadGzipFile`` is not an OSError subclass this already caught."""
    path = tmp_path / "broken.fasta.gz"
    path.write_bytes(b"not gzip at all, just bytes\n")
    blocker, kmers = _load_proteome(path)
    assert blocker == "proteome_unreadable"
    assert kmers.size == 0


def test_a_truncated_gzip_is_a_blocker_not_a_crash(tmp_path: Path) -> None:
    """A transfer cut short leaves a valid header and no end."""
    whole = tmp_path / "whole.fasta.gz"
    with gzip.open(whole, "wt", encoding="utf-8") as handle:
        handle.write(fasta_text())
    cut = tmp_path / "cut.fasta.gz"
    cut.write_bytes(whole.read_bytes()[: len(whole.read_bytes()) // 2])
    blocker, _kmers = _load_proteome(cut)
    assert blocker == "proteome_unreadable"


def test_a_gzipped_nucleotide_fasta_is_still_caught(tmp_path: Path) -> None:
    """The guard that stops a clear against nothing must survive gzipping.

    ``cds.all.fa.gz`` where ``pep.all.fa.gz`` was meant packs into millions of
    9-mers that are not peptides, so every query misses and the axis reads
    clear. Reading gzip must not route around that check.
    """
    path = tmp_path / "cds.fa.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index in range(200):
            handle.write(f">ENST{index:08d} cds\n{'ATGCGTACGT' * 40}\n")
    blocker, _kmers = _load_proteome(path)
    assert blocker == "proteome_not_protein"


# ------------------------------------------------------- finding the download


def test_the_downloaded_proteome_is_found_by_its_real_name(tmp_path: Path) -> None:
    (tmp_path / "UP000005640_9606.fasta.gz").write_bytes(b"x")
    assert downloaded_proteome(tmp_path) == tmp_path / "UP000005640_9606.fasta.gz"


def test_an_absent_download_is_none_rather_than_a_path_that_is_not_there(tmp_path: Path) -> None:
    """A returned path that does not exist would become `proteome_unreadable`,
    which says the file was bad rather than that it was never fetched."""
    assert downloaded_proteome(tmp_path) is None


def _args(**fields: object) -> argparse.Namespace:
    base = {"proteome": None, "stage_tables": False}
    base.update(fields)
    return argparse.Namespace(**base)


def test_an_explicit_proteome_always_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--proteome` is an instruction, not a hint, even with a download present."""
    from rfproteina.disease.cli import _resolve_proteome

    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    staging = tmp_path / "screening_tables"
    staging.mkdir()
    (staging / "UP000005640_9606.fasta.gz").write_bytes(b"x")
    named = tmp_path / "mine.fasta"
    named.write_text(">x\nMKV\n", encoding="utf-8")

    assert _resolve_proteome(_args(proteome=named, stage_tables=True)) == named


def test_the_download_stands_in_only_when_staging_was_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Opt-in, and said out loud.

    Which FASTA a clear was measured against is part of the measurement, and
    the axis records only its blocker, so the substitution is printed.
    """
    from rfproteina.disease.cli import _resolve_proteome

    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    staging = tmp_path / "screening_tables"
    staging.mkdir()
    found = staging / "UP000005640_9606.fasta.gz"
    found.write_bytes(b"x")

    assert _resolve_proteome(_args(stage_tables=False)) is None
    assert "downloaded proteome" not in capsys.readouterr().out

    assert _resolve_proteome(_args(stage_tables=True)) == found
    assert "downloaded proteome" in capsys.readouterr().out


def test_staging_without_the_download_leaves_the_existing_fallback_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None, not a guess. The reader's own ``HUMAN_PROTEOME`` fallback still
    applies, and returning a path would have overridden it with nothing."""
    from rfproteina.disease.cli import _resolve_proteome

    monkeypatch.setenv("DATA_PATH", str(tmp_path / "empty"))
    assert _resolve_proteome(_args(stage_tables=True)) is None
