"""Condition specs, tested directly.

Both writers were only ever reached through another module, so their refusals
had no coverage at all: every ``ValueError`` below was unreachable in the suite.
A refusal that does not work is worse than no check, because the caller carries
on believing the spec it got describes the design it meant.
"""

import pytest

from rfproteina.disease.design_spec import (
    _yaml_quote,
    backbone_spans,
    write_design_spec,
    write_inverse_folding_spec,
)

_RESIDUES = ("N", "CA", "C", "O")


def _cif(chains: dict[str, int], *, drop: tuple[str, int, str] | None = None) -> str:
    """An atom_site loop with whole backbone residues per chain.

    ``drop`` removes one atom from one residue, which is how an incomplete
    residue is built: a residue missing a backbone atom is not a span endpoint.
    """
    lines = [
        "data_fixture",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    serial = 1
    for asym, count in chains.items():
        for seq in range(1, count + 1):
            for atom in _RESIDUES:
                if drop == (asym, seq, atom):
                    continue
                lines.append(f"ATOM {serial} {atom} ALA {asym} 1 {seq} {asym} " f"{float(serial):.3f} 0.000 0.000")
                serial += 1
    lines.append("#")
    return "\n".join(lines) + "\n"


def test_backbone_spans_are_the_whole_residues(tmp_path):
    """A residue missing a backbone atom is not an endpoint of the span."""
    source = tmp_path / "pair.cif"
    source.write_text(_cif({"A": 4, "B": 3}), encoding="utf-8")
    assert backbone_spans(source) == {"A": (1, 4), "B": (1, 3)}

    gapped = tmp_path / "gapped.cif"
    gapped.write_text(_cif({"A": 4, "B": 3}, drop=("A", 4, "O")), encoding="utf-8")
    assert backbone_spans(gapped) == {"A": (1, 3), "B": (1, 3)}, "the partial residue is excluded"


def test_backbone_spans_refuse_a_file_they_cannot_read(tmp_path):
    missing = tmp_path / "absent.cif"
    with pytest.raises(ValueError, match="could not be read"):
        backbone_spans(missing)


def test_a_protein_and_a_peptide_get_their_own_lengths(tmp_path):
    """The contig length is the binder kind's, and the span is the target's."""
    source = tmp_path / "target.cif"
    source.write_text(_cif({"A": 5}), encoding="utf-8")
    protein = write_design_spec(source, tmp_path / "protein", chain="A", binder_kind="protein")
    assert protein is not None
    assert "60-110" in protein.read_text(encoding="utf-8")
    peptide = write_design_spec(source, tmp_path / "peptide", chain="A", binder_kind="peptide")
    assert peptide is not None
    assert "8-20" in peptide.read_text(encoding="utf-8")
    assert "A1-5" in peptide.read_text(encoding="utf-8")


@pytest.mark.parametrize("kind", ["ligand", "nucleic", "other", ""])
def test_a_kind_with_no_protein_contig_gets_no_spec(tmp_path, kind):
    """A ligand or oligo binder is not described by a protein contig."""
    source = tmp_path / "target.cif"
    source.write_text(_cif({"A": 5}), encoding="utf-8")
    assert write_design_spec(source, tmp_path / kind, chain="A", binder_kind=kind) is None


def test_a_chain_that_is_not_there_gets_no_spec(tmp_path):
    """Naming a chain the deposit does not have must not write a spec anyway."""
    source = tmp_path / "target.cif"
    source.write_text(_cif({"A": 5}), encoding="utf-8")
    assert write_design_spec(source, tmp_path / "out", chain="Z", binder_kind="protein") is None
    assert write_design_spec(source, tmp_path / "out", chain="  ", binder_kind="protein") is None


def test_a_filename_that_cannot_be_quoted_gets_no_spec(tmp_path):
    """A quote or a newline in the name would break out of the YAML string."""
    assert _yaml_quote('say "hi".cif') is None
    assert _yaml_quote("two\nlines.cif") is None
    assert _yaml_quote("ordinary.cif") == '"ordinary.cif"'


def test_the_inverse_spec_keeps_the_backbone_and_frees_the_sequence(tmp_path):
    """The designed chain keeps coordinates only; every other chain keeps its sequence."""
    generated = tmp_path / "run" / "binder_0_generated.cif"
    generated.parent.mkdir(parents=True)
    generated.write_text(_cif({"A": 4, "B": 3}), encoding="utf-8")
    spec = write_inverse_folding_spec(generated, tmp_path / "fresh", designed_chain="A")
    text = spec.read_text(encoding="utf-8")
    assert spec.name == "inverse_folding.yaml"
    assert "C_CRDGRP" in text
    assert "C_SEQ" in text
    assert "C_HOT" not in text, "no hotspot is invented for a re-fold"


def test_the_inverse_spec_refuses_a_chain_that_is_not_designed(tmp_path):
    generated = tmp_path / "run" / "binder_0_generated.cif"
    generated.parent.mkdir(parents=True)
    generated.write_text(_cif({"A": 4, "B": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="designed chain is not in"):
        write_inverse_folding_spec(generated, tmp_path / "fresh", designed_chain="Z")


def test_the_inverse_spec_refuses_a_structure_with_nothing_to_keep(tmp_path):
    """Re-folding the only chain would leave no target to hold it in place."""
    generated = tmp_path / "run" / "binder_0_generated.cif"
    generated.parent.mkdir(parents=True)
    generated.write_text(_cif({"A": 4}), encoding="utf-8")
    with pytest.raises(ValueError, match="no target chain"):
        write_inverse_folding_spec(generated, tmp_path / "fresh", designed_chain="A")


def test_the_inverse_spec_refuses_a_missing_structure(tmp_path):
    with pytest.raises(ValueError, match="not a generated structure"):
        write_inverse_folding_spec(tmp_path / "gone.cif", tmp_path / "fresh", designed_chain="A")
