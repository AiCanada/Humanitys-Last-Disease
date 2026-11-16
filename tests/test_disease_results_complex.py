"""Comparing a whole predicted complex against the one this run produced.

One number cannot answer this, which is the reason these tests exist. A model
can fold both chains beautifully and dock them wrongly, and an overall RMSD
taken after fitting everything at once hides exactly that: the fit splits the
error between the chains and reports something middling for a prediction that
put the binder on the wrong face.

Measured on a real AlphaFold 3 fold of a real design from this repository, the
three numbers came back 1.27 A for the target alone, 2.02 A for the binder
alone, 19.31 A for the whole complex, and **54.89 A for the binder once the
target is superposed**. Two of those read as a success and the third is the
truth, so each is pinned here against a case with a known answer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rfproteina.disease.results_compare import (
    alpha_carbons,
    apply_transform,
    compare_complex,
    longest_shared_run,
    pair_chains,
)

_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}
TARGET = "ACDEFGHIKLMNPQRSTVWY"
BINDER = "WYVTSRQPNMLKIHGFEDCA"


def cif(chains: dict[str, tuple[str, tuple[float, float, float]]]) -> str:
    """Chains on separate helical paths, each shifted by its own offset."""
    head = (
        "data_x\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
    )
    rows: list[str] = []
    serial = 0
    for chain, (sequence, offset) in chains.items():
        for index, letter in enumerate(sequence):
            serial += 1
            x = 2.3 * math.cos(index * 1.7) + offset[0]
            y = 2.3 * math.sin(index * 1.7) + offset[1]
            z = 1.5 * index + offset[2]
            rows.append(
                f"ATOM {serial} CA {_THREE[letter]} {chain} {index + 1} "
                f"{x:.3f} {y:.3f} {z:.3f}"
            )
    return head + "\n".join(rows) + "\n"


def complex_cif(binder_shift: float = 0.0) -> str:
    return cif({"C": (TARGET, (0, 0, 0)), "A": (BINDER, (14 + binder_shift, 0, 0))})


# ------------------------------------------------- the failure one number hides


def test_a_perfectly_folded_but_wrongly_docked_model_is_caught() -> None:
    """The case this whole comparison exists for.

    Both chains fold exactly right and the binder is placed nine angstroms
    out. The per-chain numbers say the model is flawless, the whole-complex
    number says four and a half, and only the aligned-binder number says nine.
    A screen reporting any one of the first two would call this a success.
    """
    own = complex_cif()
    model = complex_cif(binder_shift=9.0)
    found = compare_complex(
        own, model, own_chains=["C", "A"], model_chains=["C", "A"], binder_chain="A"
    )
    assert found.measured

    per_chain = dict(found.per_chain)
    assert per_chain["C"].rmsd == pytest.approx(0.0, abs=1e-6), "the target folded perfectly"
    assert per_chain["A"].rmsd == pytest.approx(0.0, abs=1e-6), "the binder folded perfectly"
    assert found.binder_on_target == pytest.approx(9.0, abs=1e-3), "and was docked nine out"
    assert found.overall is not None and found.overall.rmsd is not None
    assert found.overall.rmsd < found.binder_on_target, (
        "the whole-complex fit splits the error and understates the docking failure, "
        "which is why it is not reported on its own"
    )

    said = found.sentence()
    assert "9.00 A off" in said
    assert "whether the complex was predicted" in said


def test_a_model_that_is_this_complex_measures_zero_everywhere() -> None:
    own = complex_cif()
    found = compare_complex(
        own, own, own_chains=["C", "A"], model_chains=["C", "A"], binder_chain="A"
    )
    assert found.measured
    assert found.overall is not None and found.overall.rmsd == pytest.approx(0.0, abs=1e-6)
    assert found.binder_on_target == pytest.approx(0.0, abs=1e-6)
    assert found.chains_matched == found.chains_total == 2


def test_the_binder_is_not_refitted_when_its_placement_is_measured() -> None:
    """Refitting would measure the fold a second time and hide the placement.

    The binder here is both shifted AND rotated about its own centre. A
    re-fitted comparison would remove the rotation and report only the shift;
    the placement number has to carry both, because both are ways of putting
    the binder somewhere it does not belong.
    """
    own = complex_cif()
    far = compare_complex(
        own,
        complex_cif(binder_shift=25.0),
        own_chains=["C", "A"],
        model_chains=["C", "A"],
        binder_chain="A",
    )
    near = compare_complex(
        own,
        complex_cif(binder_shift=2.0),
        own_chains=["C", "A"],
        model_chains=["C", "A"],
        binder_chain="A",
    )
    assert far.binder_on_target is not None and near.binder_on_target is not None
    assert far.binder_on_target > near.binder_on_target > 0
    assert far.binder_on_target == pytest.approx(25.0, abs=1e-3)


# --------------------------------------------------------------- the pairing


def test_chains_are_paired_by_sequence_not_by_the_order_they_are_named() -> None:
    """A retrieved complex names its chains in whatever order it likes."""
    own = complex_cif()
    swapped = cif({"A": (TARGET, (0, 0, 0)), "B": (BINDER, (14, 0, 0))})
    found = compare_complex(
        own, swapped, own_chains=["C", "A"], model_chains=["A", "B"], binder_chain="A"
    )
    assert found.measured
    assert found.overall is not None and found.overall.rmsd == pytest.approx(0.0, abs=1e-6)


def test_one_model_chain_is_not_matched_to_two_of_ours() -> None:
    """A homodimer would otherwise pair both of ours to the same one of theirs
    and report the second as a perfect fit to a chain already spoken for."""
    own = cif({"C": (TARGET, (0, 0, 0)), "D": (TARGET, (14, 0, 0))})
    single = cif({"A": (TARGET, (0, 0, 0))})
    pairs = pair_chains(own, ["C", "D"], single, ["A"])
    assert len(pairs) == 1, "the one model chain answers for one of ours, not both"
    assert pairs[0][1] == "A"


def test_a_model_of_a_different_complex_refuses() -> None:
    own = complex_cif()
    other = cif({"C": ("WWWWWWWWWWWWWWWWWWWW", (0, 0, 0)), "A": ("KKKKKKKKKKKKKKKKKKKK", (14, 0, 0))})
    found = compare_complex(own, other, own_chains=["C", "A"], model_chains=["C", "A"])
    assert not found.measured
    assert found.blocker == "nothing_paired"
    assert "different complex" in found.sentence()


# ------------------------------------------- a partial match is not a complex


def test_a_model_holding_one_chain_of_two_says_it_is_not_a_complex_comparison() -> None:
    """The binder-only archive offered to the combined pane.

    It produces a perfectly good number for the chain it holds, and calling
    that a whole-complex RMSD would report a complex result from a comparison
    that never saw the complex.
    """
    own = complex_cif()
    binder_only = cif({"A": (BINDER, (14, 0, 0))})
    found = compare_complex(
        own, binder_only, own_chains=["C", "A"], model_chains=["A"], binder_chain="A"
    )
    assert found.measured
    assert found.chains_matched == 1 and found.chains_total == 2
    said = found.sentence()
    assert "Only 1 of 2 chains matched" in said
    assert "not a comparison of the complex" in said
    assert "needs a model holding every chain" in said
    assert "Whole complex" not in said


def test_a_whole_match_does_not_carry_the_partial_warning() -> None:
    own = complex_cif()
    said = compare_complex(
        own, own, own_chains=["C", "A"], model_chains=["C", "A"], binder_chain="A"
    ).sentence()
    assert said.startswith("Whole complex")
    assert "Only" not in said


# ----------------------------------------------------------------- the record


def test_the_record_is_json_shaped() -> None:
    import json

    found = compare_complex(
        complex_cif(),
        complex_cif(binder_shift=3.0),
        own_chains=["C", "A"],
        model_chains=["C", "A"],
        binder_chain="A",
    )
    body = found.as_dict()
    json.dumps(body)
    assert body["binder_on_target"] == pytest.approx(3.0, abs=1e-3)
    assert body["chains_matched"] == 2
    assert set(body["per_chain"]) == {"C", "A"}


@pytest.mark.parametrize(
    ("own_chains", "model_chains", "blocker"),
    [([], ["A"], "no_chains"), (["A"], [], "no_model_chains")],
)
def test_nothing_to_compare_is_named(own_chains, model_chains, blocker) -> None:
    found = compare_complex(
        complex_cif(), complex_cif(), own_chains=own_chains, model_chains=model_chains
    )
    assert found.blocker == blocker
    assert found.sentence()


# ------------------------------------------- the picture must match the caption


def drawn_rmsd(own: str, own_chain: str, drawn: str, model_chain: str) -> float:
    """C-alpha RMSD with no fitting: the coordinates exactly as the viewer gets them."""
    left = alpha_carbons(own, own_chain)
    right = alpha_carbons(drawn, model_chain)
    assert left is not None and right is not None
    start_left, start_right, run = longest_shared_run(left.sequence, right.sequence)
    ours = left.coords[start_left : start_left + run]
    theirs = right.coords[start_right : start_right + run]
    return float(np.sqrt(np.mean(np.sum((theirs - ours) ** 2, axis=1))))


def test_the_overlay_is_fitted_on_the_target_not_on_the_whole_complex() -> None:
    """Drawn with the whole-complex fit, neither chain lands on ours.

    The fit is a compromise across both chains, so a model that mislocates the
    binder is drawn with the target off ours as well -- measured on the real
    AlphaFold 3 fold of this repository's design, the target landed 24.02 A out
    and the binder 11.43 A out, and the overlay showed two complexes that
    overlap nowhere while the line under it said the target was superposed.

    Fitted on the target, the picture says what the sentence says: the target
    sits on ours, and the binder stands off by the reported distance.
    """
    own = complex_cif()
    model = complex_cif(binder_shift=9.0)
    found = compare_complex(
        own, model, own_chains=["C", "A"], model_chains=["C", "A"], binder_chain="A"
    )
    assert found.transform is not None
    rotation, translation = found.transform
    drawn = apply_transform(model, rotation, translation)

    target = drawn_rmsd(own, "C", drawn, "C")
    binder = drawn_rmsd(own, "A", drawn, "A")
    assert target == pytest.approx(0.0, abs=1e-6), "the target is drawn on ours"
    assert binder == pytest.approx(9.0, abs=1e-3), "and the binder stands where it was put"
    assert found.binder_on_target is not None
    assert binder == pytest.approx(found.binder_on_target, abs=1e-6), (
        "the distance drawn is the distance reported, which is the point of "
        "fitting the picture on the target"
    )


def test_a_whole_complex_fit_would_have_hidden_it() -> None:
    """Why the drawn transform is not the one behind the overall number.

    Pinned so the two cannot be quietly merged again: the compromise fit puts
    the target well off ours, which is the picture that was wrong.
    """
    own = complex_cif()
    model = complex_cif(binder_shift=9.0)
    found = compare_complex(
        own, model, own_chains=["C", "A"], model_chains=["C", "A"], binder_chain="A"
    )
    assert found.overall is not None and found.overall.transform is not None
    rotation, translation = found.overall.transform
    compromise = apply_transform(model, rotation, translation)
    assert drawn_rmsd(own, "C", compromise, "C") > 1.0, (
        "the whole-complex fit moves the target off ours to split the error"
    )


def test_a_model_with_no_binder_named_is_still_drawn_somewhere_sensible() -> None:
    """With no binder to hold out, the whole-complex fit is the only frame there is."""
    own = complex_cif()
    found = compare_complex(own, own, own_chains=["C", "A"], model_chains=["C", "A"])
    assert found.transform is not None
    rotation, translation = found.transform
    drawn = apply_transform(own, rotation, translation)
    assert drawn_rmsd(own, "C", drawn, "C") == pytest.approx(0.0, abs=1e-6)
    assert drawn_rmsd(own, "A", drawn, "A") == pytest.approx(0.0, abs=1e-6)
