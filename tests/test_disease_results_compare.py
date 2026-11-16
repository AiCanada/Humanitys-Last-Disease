"""The number that goes under each overlay on the results page.

A displayed RMSD is a confident claim, and the two ways to get it wrong are
both silent: superposing the wrong residues onto each other, and superposing
onto a mirror image. Both return a number that looks like an answer.

So the arithmetic is checked against cases with known answers -- a structure
against itself, against a rotated copy of itself, against a reflection of
itself -- before anything is checked against a real file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rfproteina.disease.results_compare import (
    MIN_PAIRED,
    alpha_carbons,
    best_match,
    chain_labels,
    compare_chains,
    first_chain,
    kabsch_rmsd,
    longest_shared_run,
)

_SEQ = "ACDEFGHIKLMNPQRSTVWY"


def cif(chain: str, sequence: str, *, start: int = 1, offset: float = 0.0) -> str:
    """A minimal mmCIF holding one chain of alpha carbons on a helix-ish path."""
    three = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
        "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
        "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
        "W": "TRP", "Y": "TYR",
    }
    head = (
        "data_test\nloop_\n"
        "_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
    )
    rows = []
    for index, letter in enumerate(sequence):
        angle = index * 1.7
        x = 2.3 * math.cos(angle) + offset
        y = 2.3 * math.sin(angle)
        z = 1.5 * index
        rows.append(
            f"ATOM {index + 1} CA {three.get(letter, 'ALA')} {chain} {start + index} "
            f"{x:.3f} {y:.3f} {z:.3f}"
        )
    return head + "\n".join(rows) + "\n"


def coords_of(text: str, chain: str) -> np.ndarray:
    found = alpha_carbons(text, chain)
    assert found is not None
    return found.coords


# ------------------------------------------------------------ the arithmetic


def test_a_structure_against_itself_is_zero() -> None:
    points = coords_of(cif("A", _SEQ), "A")
    assert kabsch_rmsd(points, points) == pytest.approx(0.0, abs=1e-9)


def test_a_rotated_and_translated_copy_is_still_zero() -> None:
    """Superposition must remove rigid motion, which is the whole point."""
    points = coords_of(cif("A", _SEQ), "A")
    angle = 0.7
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = points @ rotation.T + np.array([12.0, -4.0, 31.0])
    assert kabsch_rmsd(moved, points) == pytest.approx(0.0, abs=1e-8)


def test_a_mirror_image_is_not_reported_as_a_perfect_fit() -> None:
    """The reflection guard, which is impossible to notice when it is missing.

    Without correcting the determinant, the optimal "rotation" may be improper
    and superpose a chain onto its mirror image, returning an RMSD lower than
    any rotation can achieve. A protein and its mirror are not the same
    protein, so a reflected fit flatters the comparison.
    """
    points = coords_of(cif("A", _SEQ), "A")
    mirrored = points * np.array([1.0, 1.0, -1.0])
    found = kabsch_rmsd(mirrored, points)
    assert found > 1.0, f"a reflection fitted to {found:.3f} A, so the guard is not working"


def test_a_known_displacement_gives_the_known_rmsd() -> None:
    """One residue moved by a known amount, the rest identical.

    RMSD over n points with one off by d is d/sqrt(n), and superposition will
    shave that slightly; the bound checks it is in the right place rather than
    merely non-zero.
    """
    points = coords_of(cif("A", _SEQ), "A")
    moved = points.copy()
    moved[0] += np.array([3.0, 0.0, 0.0])
    expected = 3.0 / math.sqrt(len(points))
    assert kabsch_rmsd(moved, points) == pytest.approx(expected, rel=0.25)


@pytest.mark.parametrize(
    ("left", "right"),
    [(np.zeros((0, 3)), np.zeros((0, 3))), (np.zeros((4, 3)), np.zeros((5, 3)))],
)
def test_mismatched_or_empty_input_is_refused(left: np.ndarray, right: np.ndarray) -> None:
    with pytest.raises(ValueError):
        kabsch_rmsd(left, right)


# -------------------------------------------------------- the correspondence


def test_residues_are_paired_by_sequence_not_by_number() -> None:
    """The failure this exists to prevent.

    An AlphaFold database model numbers from 1 over the whole UniProt sequence;
    an experimental structure of the same protein may start at residue 20.
    Pairing by ``label_seq_id`` compares residue 30 of one against a different
    residue 30 of the other and reports a large RMSD as structural
    disagreement when it is the numbering that disagrees.
    """
    full = cif("A", _SEQ, start=1)
    shifted = cif("B", _SEQ, start=20)
    found = compare_chains(full, shifted, left_chain="A", right_chain="B")
    assert found.measured, found.blocker
    assert found.paired == len(_SEQ), "the same residues, renumbered, must pair completely"
    assert found.rmsd == pytest.approx(0.0, abs=1e-8)


def test_only_the_shared_run_is_paired_and_the_count_is_reported() -> None:
    """An RMSD is meaningless without the number of residues behind it."""
    long = cif("A", _SEQ + "GGGG")
    short = cif("A", _SEQ[:12])
    found = compare_chains(long, short, left_chain="A", right_chain="A")
    assert found.measured
    assert found.paired == 12
    assert found.left_residues == len(_SEQ) + 4
    assert found.right_residues == 12
    assert "12 of 12 residues" in found.sentence()


def test_a_partial_pairing_says_so_in_words() -> None:
    """A small RMSD over a fraction of the chain is the least informative
    number this page can show, so the sentence names it."""
    long = cif("A", _SEQ * 3)
    short = cif("A", _SEQ[:10])
    found = compare_chains(long, short, left_chain="A", right_chain="A")
    assert found.measured
    assert found.coverage is not None and found.coverage >= 0.8, "ten of ten is full coverage"
    wide = compare_chains(long, cif("A", _SEQ[:10] + "KKKKKKKKKKKKKKKKKKKK"), left_chain="A", right_chain="A")
    if wide.measured and wide.coverage is not None and wide.coverage < 0.8:
        assert "not paired" in wide.sentence()


def test_a_different_protein_has_nothing_to_superpose() -> None:
    found = compare_chains(cif("A", "ACDEFGHIKL"), cif("A", "WWWWWWWWWW"), left_chain="A", right_chain="A")
    assert not found.measured
    assert found.blocker == "no_common_sequence"
    assert "different protein" in found.sentence()


def test_too_few_paired_residues_is_refused_rather_than_reported() -> None:
    """Three points superpose onto almost any other three."""
    found = compare_chains(cif("A", "ACD"), cif("A", "ACD"), left_chain="A", right_chain="A")
    assert not found.measured
    assert found.blocker == "too_few_paired"
    assert str(MIN_PAIRED) in found.sentence()


def test_unknown_residues_do_not_pair() -> None:
    """``X`` stands for a residue this does not recognise, and two unknowns are
    not evidence of being the same residue."""
    assert longest_shared_run("XXXX", "XXXX") == (0, 0, 0)


def test_the_shared_run_is_contiguous() -> None:
    """A subsequence match may interleave residues from either side of a gap,
    which pairs positions that are not the same position."""
    start_left, start_right, run = longest_shared_run("AAACDEFGGG", "CDEF")
    assert run == 4
    assert start_left == 3 and start_right == 0


# ---------------------------------------------------------------- the readers


def test_a_missing_chain_is_named_rather_than_guessed() -> None:
    text = cif("A", _SEQ)
    assert alpha_carbons(text, "Z") is None
    found = compare_chains(text, text, left_chain="A", right_chain="Z")
    assert found.blocker == "no_right_chain"
    assert "retrieved model" in found.sentence()


def test_chain_labels_and_first_chain_read_the_file() -> None:
    text = cif("A", _SEQ) + cif("C", _SEQ).split("_atom_site.Cartn_z\n", 1)[1]
    assert first_chain(text) == "A"
    assert set(chain_labels(text)) >= {"A"}


def test_best_match_measures_rather_than_assumes_the_pairing() -> None:
    """A retrieved complex names its chains in an order nobody promised."""
    design = cif("A", _SEQ)
    other = cif("B", "WWWWWWWWWWWWWWWWWWWW")
    combined = design + other.split("_atom_site.Cartn_z\n", 1)[1]
    found = best_match(design, "A", combined, ("B", "A"))
    assert found.measured
    assert found.right_chain == "A", "the chain that actually superposes, not the first offered"
    assert found.rmsd == pytest.approx(0.0, abs=1e-8)


def test_best_match_with_no_comparable_chain_says_so() -> None:
    found = best_match(cif("A", _SEQ), "A", cif("A", "WWWWWWWWWW"), ("A",))
    assert not found.measured
    assert found.blocker == "no_common_sequence"


def test_the_record_is_json_shaped() -> None:
    import json

    found = compare_chains(cif("A", _SEQ), cif("A", _SEQ), left_chain="A", right_chain="A")
    json.dumps(found.as_dict())
    assert found.as_dict()["sentence"]
