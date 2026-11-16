"""What the two nucleic axes may and may not say, against their sources.

Four things this file pins, each of which the module got wrong before it
existed:

* A protein FASTA named as the transcriptome used to clear
  ``off_target_hybridization`` having matched nothing: every other check --
  size, base floor, readability -- passes for one. The reader now refuses it
  with ``transcriptome_not_nucleic``, the mirror of the protein reader's
  ``proteome_not_protein``.
* The TLR9 rule used to be the murine consensus alone, so CpG 7909 / ODN 2006
  (Hartmann et al., J Immunol 2000, 164:1617), the prototype human CpG-B
  agonist, came back ``no_pass_cut``. Both published motifs are read now and
  each hit names the one that fired.
* One IUPAC ambiguity letter used to refuse the whole design. A degenerate base
  is real in an ordered oligo: the windows that touch it are unscanned, the
  rest of the chain is still measured.
* ``TRANSCRIPTOME_FILE_LIMIT`` used to sit below the human transcriptome the
  seed width is argued against.

``tests/test_disease_nucleic_and_structure_methods.py`` pins the arithmetic --
packing, both orientations, the hexamer placement -- and is not repeated here.
"""

import random
from pathlib import Path

import pytest

from rfproteina.disease import oligo
from rfproteina.disease.oligo import (
    SEED_WIDTH,
    TRANSCRIPTOME_FILE_LIMIT,
    innate_immune_record,
    measure_nucleic_liability,
)

#: Hartmann et al., J Immunol 2000, 164:1617: CpG 7909 / ODN 2006, the
#: prototype human CpG-B ODN. Its three GTCGTT hexamers and four CG
#: dinucleotides are read off the published sequence, not off the module.
_ODN_2006 = "TCGTCGTTTTGTCGTTTTGTCGTT"
#: Krieg et al., Nature 1995, 374:546, read by the murine-optimal hexamer.
_ODN_1826 = "TCCATGACGTTCCTGACGTT"
#: Planted in the nucleotide reference below, and carried by the design.
_PLANTED = "ACGTTGCAATTCGAT"


def _wrapped(body: str, name: str) -> str:
    lines = [body[start : start + 60] for start in range(0, len(body), 60)]
    return f">{name}\n" + "\n".join(lines) + "\n"


def _protein_reference(tmp_path: Path) -> Path:
    """1.2M residues of protein: past the base floor, and a quarter ACGTN.

    A uniform twenty-letter protein alphabet shares five letters with ACGTN, so
    its base share is 5/20 = 0.25. That number is arithmetic on the alphabet,
    not a value read from the module.
    """
    rng = random.Random(0)
    body = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(1_200_000))
    path = tmp_path / "transcriptome.fa"
    path.write_text(_wrapped(body, "sp|P1|PROTEIN"), encoding="utf-8")
    return path


def _nucleotide_reference(tmp_path: Path) -> Path:
    """1.2M bases with the planted seed, and ambiguity codes among them.

    A real reference carries ambiguity codes, so this one draws from an
    alphabet of 24 ACGT letters and the two codes R and Y: a base share of
    24/26 = 0.923, above the 0.90 the reader asks for and nowhere near the
    0.25 of a protein FASTA. Both numbers are arithmetic on the alphabet.
    """
    rng = random.Random(1)
    body = [rng.choice("ACGTACGTACGTACGTACGTACGTRY") for _ in range(1_200_000)]
    body[500_000 : 500_000 + SEED_WIDTH] = list(_PLANTED)
    path = tmp_path / "transcriptome.fa"
    path.write_text(_wrapped("".join(body), "transcript"), encoding="utf-8")
    return path


def test_a_protein_fasta_is_refused_as_a_transcriptome(tmp_path: Path):
    """The expected value is known without the module: a protein FASTA holds no
    transcript, so the only honest answer is a blocker and never a clear."""
    axes, by_example = measure_nucleic_liability(
        [("e0", "GGGG" + _PLANTED + "CCCC")],
        directory=tmp_path,
        transcriptome=_protein_reference(tmp_path),
    )
    record = axes["off_target_hybridization"]
    assert record["state"] == "untested"
    assert record["blocker"] == "transcriptome_not_nucleic"
    assert by_example["e0"]["blocker"] == "transcriptome_not_nucleic"
    assert record["measurement"] is None


def test_a_nucleotide_reference_still_decides(tmp_path: Path):
    """The guard must not refuse a real reference, nor a degenerate design."""
    reference = _nucleotide_reference(tmp_path)
    axes, by_example = measure_nucleic_liability(
        [
            ("planted", "GGGG" + _PLANTED + "CCCC"),
            ("degenerate", "GGGGN" + _PLANTED + "CCCC"),
            ("absent", "TTTTTTTTTTTTTTTTTTTTAAAAAAAAAAAAAAAAAAAA"),
        ],
        directory=tmp_path,
        transcriptome=reference,
    )
    assert axes["off_target_hybridization"]["state"] == "hit"
    assert by_example["planted"]["state"] == "hit"
    assert by_example["planted"]["reference_bases"] == 1_200_000
    assert by_example["planted"]["reference_file"] == "transcriptome.fa"
    # The N costs only the windows that touch it; the planted seed is still
    # found, so the design is measured rather than discarded.
    assert by_example["degenerate"]["state"] == "hit"
    assert by_example["absent"]["state"] == "clear"


def test_odn2006_the_human_tlr9_agonist_is_a_hit():
    """Hartmann et al., J Immunol 2000, 164:1617.

    The three GTCGTT hexamers of CpG 7909 start at zero-based 2, 10 and 18,
    i.e. one-based 3, 11 and 19, and the sequence carries four CG dinucleotides
    (zero-based 1, 4, 12 and 20). Those numbers come from reading the published
    sequence.
    """
    record = innate_immune_record(_ODN_2006)
    assert record["state"] == "hit"
    assert record["measurement"] == 3
    assert record["motif_positions"] == [3, 11, 19]
    assert record["cpg_count"] == 4
    assert record["motif_sites"] == [
        {"position": 3, "motif": "GTCGTT"},
        {"position": 11, "motif": "GTCGTT"},
        {"position": 19, "motif": "GTCGTT"},
    ]
    assert [_ODN_2006[start - 1 : start + 5] for start in record["motif_positions"]] == ["GTCGTT"] * 3
    # Both papers are named, because the two rules carry different evidence.
    assert "Krieg 1995" in str(record["method"])
    assert "Hartmann 2000" in str(record["method"])


def test_the_murine_consensus_still_hits_and_says_which_rule_fired():
    record = innate_immune_record(_ODN_1826)
    assert record["state"] == "hit"
    assert record["measurement"] == 2
    assert record["motif_positions"] == [6, 15]
    assert record["motif_sites"] == [
        {"position": 6, "motif": "RRCGYY"},
        {"position": 15, "motif": "RRCGYY"},
    ]
    # GTCGTT is not an instance of RRCGYY, and GACGTT is not GTCGTT: the two
    # rules must not be reported as one.
    human = innate_immune_record("AAAAGTCGTTTTTT")
    assert human["state"] == "hit"
    assert human["motif_sites"] == [{"position": 5, "motif": "GTCGTT"}]


def test_a_cpg_in_neither_motif_is_refused_without_denying_the_cut():
    """TTCGAA: two pyrimidines before the CG, and not GTCGTT.

    Reported as ``no_pass_cut`` until the board was found stating the opposite
    of its own guide: a cut DOES apply on this axis, since one consensus hexamer
    is a hit. The measured zero is refused for the reason
    ``cpg_free_not_evidence`` gives for its own zero -- TLR9 is not the only
    innate sensor -- so the name says that instead.
    """
    record = innate_immune_record("AAAATTCGAAAAAA")
    assert record["state"] == "untested"
    assert record["blocker"] == "cpg_not_consensus"
    assert record["cpg_count"] == 1
    assert "motif_sites" not in record
    # And a chain with no CpG at all is still the other refusal, not a clear.
    assert innate_immune_record("AAAATTTTAAAA")["blocker"] == "cpg_free_not_evidence"


def test_one_ambiguous_base_does_not_discard_the_design():
    """The same chain with and without a trailing N.

    The GACGTT hexamer starts at one-based 5 in both; that number is read off
    the sequence, not from the module.
    """
    plain = innate_immune_record("AAAAGACGTTAAAA")
    with_n = innate_immune_record("AAAAGACGTTNAAAA")
    assert plain["state"] == "hit"
    assert plain["motif_positions"] == [5]
    assert with_n["state"] == "hit"
    assert with_n["measurement"] == 1
    assert with_n["motif_positions"] == [5]
    assert with_n["cpg_count"] == 1
    # Every IUPAC degenerate letter is a base, not a reason to refuse a chain.
    for letter in "NRYKMSWBDHV":
        assert innate_immune_record(f"AAAAGACGTT{letter}AAAA")["state"] == "hit", letter


def test_seeds_spanning_an_ambiguous_base_are_dropped_not_invented():
    """A window holding a degenerate base is unscanned, not joined up.

    A 31-base chain with one N at zero-based 15 has 31 - 15 + 1 = 17 windows,
    of which the 15 that cover position 15 (starts 1..15) cannot be packed, so
    2 remain. Counted by hand from the window arithmetic.
    """
    chain = "ACGTACGTACGTACG" + "N" + "TGCATGCATGCATGC"
    assert oligo._normalized(chain) == chain
    assert len(oligo._seeds(chain)) == 2
    assert oligo._pack("ACGTACGTACGTACG") in oligo._seeds(chain)
    # The degenerate letters reverse-complement to the complementary set
    # (NC-IUB 1984), so the second orientation is not a different sequence.
    assert oligo._reverse_complement("ACGTNRYKMSWBDHV") == "BDHVWSKMRYNACGT"
    assert len(oligo._seeds(oligo._reverse_complement(chain))) == 2


def test_a_real_non_nucleic_chain_is_still_refused():
    assert oligo._normalized("MKTAYIAKQRQISFVK") is None
    assert innate_immune_record("MKTAYIAKQRQISFVK")["blocker"] == "sequence_not_nucleic"
    assert innate_immune_record(None)["blocker"] == "sequence_absent"


def test_the_limit_admits_a_human_transcriptome_sized_reference():
    """A 3e8-base FASTA wrapped at sixty columns is 3e8 * 61 / 60 bytes.

    That is arithmetic on the reference size the module's own seed-width
    argument names, not a value read from the module.
    """
    human_cdna_bytes = int(3e8 * 61 / 60)
    assert human_cdna_bytes == 305_000_000
    assert human_cdna_bytes < TRANSCRIPTOME_FILE_LIMIT


def test_a_file_past_the_limit_is_still_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reference = tmp_path / "transcriptome.fa"
    reference.write_text(">t\n" + "ACGT" * 100 + "\n", encoding="utf-8")
    monkeypatch.setattr(oligo, "TRANSCRIPTOME_FILE_LIMIT", 10)
    axes, _by_example = measure_nucleic_liability([("e0", "ACGT" * 10)], directory=tmp_path, transcriptome=reference)
    assert axes["off_target_hybridization"]["blocker"] == "transcriptome_too_large"
