"""Chemical motifs and sequence charge on a finished protein or peptide design.

The counts are a report. They do not clear a bar and they do not add a hit.
"""

import json
from pathlib import Path

import pytest

from rfproteina.disease.chemical import (
    PI_BLOCKER,
    chemical_phrase,
    chemical_record,
    isoelectric_point,
    net_charge,
)
from rfproteina.disease.combinations import AXES, CLASS_CONDITION_DIM, PROTEIN_AXES
from rfproteina.disease.screen import format_screen_lines, screen_generation
from tests.helpers.disease import (
    design_cif,
    no_developer_proteome,  # noqa: F401
    screen_page,
    screened_design_run,
)

_NG = "ACNGAA"
_PLAIN = "AAAAAAAAAA"


def test_the_axis_list_stays_the_same_length():
    assert CLASS_CONDITION_DIM == 44
    assert "chemical_liability" not in AXES


def test_bjellqvist_charge_matches_the_expasy_examples():
    """Biopython's IsoelectricPoint doctest uses this 1994 pKa set."""
    assert round(isoelectric_point("INGAR"), 2) == 9.75
    assert round(net_charge("INGAR", 7.0), 2) == 0.76
    assert round(isoelectric_point("PETER"), 2) == 4.53
    assert abs(net_charge("PETER", 4.53)) < 0.05


def test_the_pi_bracket_refuses_at_101_arginines_and_not_at_100():
    """The worked example beside ``PI_BLOCKER`` named a chain that does solve.

    A hundred arginines are net *negative* at pH 14: the C-terminus is fully
    ionised there while each side chain two pH units above its pKa carries only
    1/101 of a charge, so a hundred of them come to 0.990 against 1.000 and the
    crossing sits just inside the bracket at 13.9956. Only at 101 does the
    bracket hold no crossing. Nothing else pins where that boundary is, and a
    worked example that is wrong by one is how the next reader widens a
    defensive branch that was right.
    """
    assert net_charge("R" * 100, 14.0) == pytest.approx(-0.0099, abs=5e-5)
    assert isoelectric_point("R" * 100) == pytest.approx(13.9956, abs=5e-5)
    hundred = chemical_record("R" * 100, partners=None)
    assert hundred is not None
    assert hundred["isoelectric_point"] == pytest.approx(13.9956, abs=5e-5)
    assert "isoelectric_point_blocker" not in hundred

    assert net_charge("R" * 101, 14.0) > 0.0
    with pytest.raises(ValueError, match=PI_BLOCKER):
        isoelectric_point("R" * 101)


def test_sequons_skip_proline_and_a_pair_can_be_two_motifs():
    skipped = chemical_record("NPSA", partners=None)
    assert skipped is not None
    assert skipped["glycosylation_count"] == 0
    assert skipped["deamidation_count"] == 0
    assert chemical_phrase(skipped) is None

    both = chemical_record("NSTA", partners=None)
    assert both is not None
    assert both["deamidation_sites"] == [{"residue": 1, "motif": "NS"}]
    assert both["glycosylation_sites"] == [{"residue": 1, "motif": "NXT"}]
    overlap = chemical_record("NGSA", partners=None)
    assert overlap is not None
    assert overlap["deamidation_sites"] == [{"residue": 1, "motif": "NG"}]
    assert overlap["glycosylation_sites"] == [{"residue": 1, "motif": "NXS"}]
    oxidation = chemical_record("AMWA", partners=None)
    assert oxidation is not None
    assert oxidation["methionine_residues"] == [2]
    assert oxidation["tryptophan_residues"] == [3]
    assert oxidation["pka_set"] == "bjellqvist_1994"
    assert "state" not in oxidation
    phrase = chemical_phrase(both)
    assert phrase is not None
    assert "deamidation 1, residue 1 motif NS" in phrase
    assert "glycosylation 1, residue 1 motif NXT" in phrase
    assert chemical_record("ACX", partners=None) is None


def test_one_asn_gly_is_stored_and_the_call_stays(tmp_path: Path):
    record = screened_design_run(tmp_path, _NG)
    chemical = record["chemical_liability"]
    assert isinstance(chemical, dict)
    assert chemical["deamidation_count"] == 1
    assert chemical["deamidation_sites"] == [{"residue": 3, "motif": "NG"}]
    assert chemical["isomerization_count"] == 0
    assert chemical["glycosylation_count"] == 0
    assert record["call"] == "untested"
    assert record["hits"] == []
    assert record["samples"][0]["chemical_liability"] == chemical
    measured = format_screen_lines("1ABC", "A", record)[1]
    assert "deamidation 1, residue 3 motif NG" in measured


def test_a_chain_with_no_motifs_stores_zeros_and_stays_untested(tmp_path: Path):
    record = screened_design_run(tmp_path, _PLAIN)
    chemical = record["chemical_liability"]
    assert isinstance(chemical, dict)
    assert chemical["deamidation_count"] == 0
    assert chemical["isomerization_count"] == 0
    assert chemical["glycosylation_count"] == 0
    assert chemical["methionine_count"] == 0
    assert chemical["tryptophan_count"] == 0
    assert chemical["deamidation_sites"] == []
    assert record["call"] == "untested"
    assert record["sequence_developability"]["sequence_pass"] is False
    measured = format_screen_lines("1ABC", "A", record)[1]
    for word in ("deamidation", "isomerization", "glycosylation", "methionine", "tryptophan", "free cysteine"):
        assert word not in measured
    page = screen_page(record)
    assert "Deamidation 0." in page
    assert "cure" not in page.casefold()


def test_a_disulfide_partner_is_not_free_and_an_unpaired_cysteine_is(tmp_path: Path):
    paired = screened_design_run(tmp_path / "paired", "CACA", chain="A_1", bonds=_bond("A", 1, "A", 3))
    paired_chemical = paired["chemical_liability"]
    assert isinstance(paired_chemical, dict)
    assert paired_chemical["free_cysteine"] == "reported"
    assert paired_chemical["free_cysteine_count"] == 0
    assert paired_chemical["free_cysteine_residues"] == []

    unpaired = screened_design_run(tmp_path / "open", "CACA", chain="A_1", bonds=_bond("A", 1, "B", 9))
    open_chemical = unpaired["chemical_liability"]
    assert isinstance(open_chemical, dict)
    assert open_chemical["free_cysteine_count"] == 1
    assert open_chemical["free_cysteine_residues"] == [3]
    assert "free cysteine 1, residue 3" in format_screen_lines("1ABC", "A", unpaired)[1]

    covale = screened_design_run(tmp_path / "covale", "CACA", chain="A", bonds=_bond("A", 1, "A", 3, kind="covale"))
    covale_chemical = covale["chemical_liability"]
    assert isinstance(covale_chemical, dict)
    assert covale_chemical["free_cysteine_count"] == 0


def test_named_bonds_without_a_disulfide_leave_every_cysteine_untested(tmp_path: Path):
    """A bond of another kind is not evidence about disulfides.

    Counting it as one turned a file that recorded no disulfide into a hard
    count of every cysteine as free.
    """
    record = screened_design_run(tmp_path, "CACA", bonds=_metal())
    chemical = record["chemical_liability"]
    assert isinstance(chemical, dict)
    assert chemical["free_cysteine"] == "untested"
    assert "free_cysteine_count" not in chemical
    assert "free_cysteine_residues" not in chemical


def test_a_cif_without_bonds_leaves_free_cysteine_untested(tmp_path: Path):
    sequence = "CNGAA"
    record = screened_design_run(tmp_path, sequence)
    chemical = record["chemical_liability"]
    assert isinstance(chemical, dict)
    assert chemical["free_cysteine"] == "untested"
    assert "free_cysteine_count" not in chemical
    assert "free_cysteine_residues" not in chemical
    assert chemical["deamidation_count"] == 1
    assert chemical["deamidation_sites"] == [{"residue": 2, "motif": "NG"}]
    stored = json.dumps(chemical)
    assert sequence not in stored
    assert "inference_samples" not in stored
    assert "\\" not in stored
    assert "/" not in stored
    page = screen_page(record)
    assert "does not name disulfide partners" in page
    assert "Residue 2, motif NG" in page
    assert sequence not in page
    assert "cure" not in page.casefold()


def test_a_partner_that_is_not_a_positive_integer_is_ignored():
    record = chemical_record("CACA", partners=[True, 0, "3", 1.5, 3])
    assert record is not None
    assert record["free_cysteine"] == "reported"
    assert record["free_cysteine_residues"] == [1]


def test_a_nonloop_disulfide_is_named_and_a_metal_bond_leaves_cysteine_untested(tmp_path: Path):
    paired = screened_design_run(tmp_path / "paired", "CACA", chain="A_1", bonds=_bond_item("A", 1, "A", 3))
    paired_chemical = paired["chemical_liability"]
    assert isinstance(paired_chemical, dict)
    assert paired_chemical["free_cysteine"] == "reported"
    assert paired_chemical["free_cysteine_count"] == 0
    assert paired_chemical["free_cysteine_residues"] == []
    assert "free cysteine" not in format_screen_lines("1ABC", "A", paired)[1]

    # A coordination bond says nothing about disulfides, and this repository's
    # own parse preset keeps metalc while dropping disulf -- so counting it as
    # an answer reported every cysteine of a metalloprotein design free, from a
    # file that had recorded no disulfide at all.
    metal = screened_design_run(tmp_path / "metal", "CACA", bonds=_bond_item("A", 2, "A", 9, kind="metalc", comp="HIS"))
    metal_chemical = metal["chemical_liability"]
    assert isinstance(metal_chemical, dict)
    assert metal_chemical["free_cysteine"] == "untested"
    assert "free_cysteine_count" not in metal_chemical
    assert "free_cysteine_residues" not in metal_chemical
    assert "free cysteine" not in format_screen_lines("1ABC", "A", metal)[1]


def test_a_bond_value_on_the_following_line_is_still_a_disulfide(tmp_path: Path):
    bonds = (
        "_struct_conn.id\n"
        "ss1\n"
        "_struct_conn.conn_type_id\n"
        "disulf\n"
        "_struct_conn.ptnr1_label_asym_id\n"
        "A\n"
        "_struct_conn.ptnr1_label_comp_id\n"
        "CYS\n"
        "_struct_conn.ptnr1_label_seq_id\n"
        "1\n"
        "_struct_conn.ptnr1_label_atom_id\n"
        "SG\n"
        "_struct_conn.ptnr2_label_asym_id\n"
        "A\n"
        "_struct_conn.ptnr2_label_comp_id\n"
        "CYS\n"
        "_struct_conn.ptnr2_label_seq_id\n"
        "3\n"
        "_struct_conn.ptnr2_label_atom_id\n"
        "SG\n"
        "_struct_conn_type.id\n"
        "disulf\n"
        "#\n"
    )
    record = screened_design_run(tmp_path, "CACA", bonds=bonds)
    chemical = record["chemical_liability"]
    assert isinstance(chemical, dict)
    assert chemical["free_cysteine"] == "reported"
    assert chemical["free_cysteine_count"] == 0
    assert chemical["free_cysteine_residues"] == []


def test_a_nonloop_category_without_a_bond_type_stays_untested(tmp_path: Path):
    record = screened_design_run(tmp_path, "CACA", bonds="_struct_conn.details note\n#\n")
    chemical = record["chemical_liability"]
    assert isinstance(chemical, dict)
    assert chemical["free_cysteine"] == "untested"
    assert "free_cysteine_count" not in chemical
    assert "free_cysteine_residues" not in chemical


def test_an_uneven_bond_loop_leaves_free_cysteine_untested(tmp_path: Path):
    bonds = (
        "loop_\n"
        "_struct_conn.conn_type_id\n"
        "_struct_conn.ptnr1_label_asym_id\n"
        "_struct_conn.ptnr1_label_seq_id\n"
        "_struct_conn.ptnr2_label_asym_id\n"
        "_struct_conn.ptnr2_label_seq_id\n"
        "disulf A 1\n"
        "#\n"
    )
    record = screened_design_run(tmp_path, "CNGAA", bonds=bonds)
    chemical = record["chemical_liability"]
    assert isinstance(chemical, dict)
    assert chemical["free_cysteine"] == "untested"
    assert "free_cysteine_count" not in chemical
    assert "free_cysteine_residues" not in chemical
    assert chemical["deamidation_sites"] == [{"residue": 2, "motif": "NG"}]
    measured = format_screen_lines("1ABC", "A", record)[1]
    assert "free cysteine" not in measured
    assert "deamidation 1, residue 2 motif NG" in measured


def test_a_pasted_motif_name_stays_off_the_line_and_the_page(tmp_path: Path):
    pasted = "ACDEFGHIKLMNPQRSTVWY" * 2
    record = screened_design_run(tmp_path, _NG)
    chemical = record["chemical_liability"]
    assert isinstance(chemical, dict)
    chemical["deamidation_count"] = 2
    chemical["deamidation_sites"] = [
        {"residue": 3, "motif": "NG"},
        {"residue": 4, "motif": pasted},
    ]
    phrase = chemical_phrase(chemical)
    assert phrase == "deamidation 2, residue 3 motif NG"
    page = screen_page(record)
    assert "Deamidation 2." in page
    assert "Residue 3, motif NG" in page
    assert "Residue 4" not in page
    assert pasted not in page
    assert "cure" not in page.casefold()


def test_a_chemical_failure_keeps_the_sequence_metrics(tmp_path: Path, monkeypatch):
    def _boom(sequence, *, partners):
        raise ValueError("unreadable partners")

    monkeypatch.setattr("rfproteina.disease.screen.chemical_record", _boom)
    record = screened_design_run(tmp_path, _NG)
    sequence = record["sequence_developability"]
    assert isinstance(sequence, dict)
    assert isinstance(sequence["gravy"], float)
    assert isinstance(sequence["instability_index"], float)
    assert isinstance(sequence["sequence_pass"], bool)
    assert "chemical_liability" not in record
    samples = record["samples"]
    assert isinstance(samples, list) and samples
    assert "chemical_liability" not in samples[0]
    assert "deamidation" not in format_screen_lines("1ABC", "A", record)[1]


def test_a_clear_bar_stays_clear_when_a_motif_is_present(tmp_path: Path):
    run = tmp_path / "clear"
    evidence = {axis: "clear" for axis in PROTEIN_AXES}
    run.mkdir()
    screened_design_run(run, _NG)
    (run / "gene_1ABC_1" / "axis_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    again = screen_generation(run / "gene_1ABC_1", binder_kind="protein", target_id="P05113")
    assert again["call"] == "screened_clear"
    assert again["hits"] == []
    chemical = again["chemical_liability"]
    assert isinstance(chemical, dict)
    assert chemical["deamidation_count"] == 1


@pytest.mark.parametrize("kind", ["protein", "peptide", "polypeptide"])
def test_protein_and_peptide_store_the_scan(tmp_path: Path, kind: str):
    record = screened_design_run(tmp_path / kind, _NG, kind=kind)
    assert "chemical_liability" in record
    assert record["call"] != "axis_hit"


@pytest.mark.parametrize("kind", ["ligand", "small_molecule", "non_polymer", "nucleic", "dna"])
def test_ligand_and_nucleic_leave_the_scan_unread(tmp_path: Path, kind: str):
    record = screened_design_run(tmp_path / kind, _NG, kind=kind)
    assert "chemical_liability" not in record
    samples = record["samples"]
    assert isinstance(samples, list) and samples
    assert "chemical_liability" not in samples[0]


def test_the_first_design_is_the_report_and_every_design_is_stored(tmp_path: Path):
    run = tmp_path / "gene_1ABC_1"
    for name, sequence in (("binder_0", _NG), ("binder_1", _PLAIN)):
        folder = run / "inference_samples" / name
        folder.mkdir(parents=True)
        (folder / f"{name}_generated.cif").write_text(design_cif(sequence), encoding="utf-8")
    metrics = run / "inference_metrics"
    metrics.mkdir()
    (metrics / "per_example_metrics.csv").write_text(
        "example_id,chain_of_interest\nbinder_0,A\nbinder_1,A\n",
        encoding="utf-8",
    )
    record = screen_generation(run, binder_kind="protein", target_id="P05113")
    samples = record["samples"]
    assert [item["example_id"] for item in samples] == ["binder_0", "binder_1"]
    assert samples[0]["chemical_liability"]["deamidation_count"] == 1
    assert samples[1]["chemical_liability"]["deamidation_count"] == 0
    assert record["chemical_liability"] == samples[0]["chemical_liability"]
    assert _NG not in json.dumps(record)
    assert _PLAIN not in json.dumps(record)


def _bond(left_chain: str, left_seq: int, right_chain: str, right_seq: int, *, kind: str = "disulf") -> str:
    return (
        "loop_\n"
        "_struct_conn.id\n"
        "_struct_conn.conn_type_id\n"
        "_struct_conn.ptnr1_label_asym_id\n"
        "_struct_conn.ptnr1_label_comp_id\n"
        "_struct_conn.ptnr1_label_seq_id\n"
        "_struct_conn.ptnr1_label_atom_id\n"
        "_struct_conn.ptnr2_label_asym_id\n"
        "_struct_conn.ptnr2_label_comp_id\n"
        "_struct_conn.ptnr2_label_seq_id\n"
        "_struct_conn.ptnr2_label_atom_id\n"
        f"ss1 {kind} {left_chain} CYS {left_seq} SG {right_chain} CYS {right_seq} SG\n"
        "#\n"
    )


def _bond_item(
    left_chain: str,
    left_seq: int,
    right_chain: str,
    right_seq: int,
    *,
    kind: str = "disulf",
    comp: str = "CYS",
) -> str:
    return (
        "_struct_conn.id ss1\n"
        f"_struct_conn.conn_type_id {kind}\n"
        f"_struct_conn.ptnr1_label_asym_id {left_chain}\n"
        f"_struct_conn.ptnr1_label_comp_id {comp}\n"
        f"_struct_conn.ptnr1_label_seq_id {left_seq}\n"
        "_struct_conn.ptnr1_label_atom_id SG\n"
        f"_struct_conn.ptnr2_label_asym_id {right_chain}\n"
        f"_struct_conn.ptnr2_label_comp_id {comp}\n"
        f"_struct_conn.ptnr2_label_seq_id {right_seq}\n"
        "_struct_conn.ptnr2_label_atom_id SG\n"
        "_struct_conn_type.id disulf\n"
        "_struct_conn_type.criteria ?\n"
        "#\n"
    )


def _metal() -> str:
    return (
        "loop_\n"
        "_struct_conn.id\n"
        "_struct_conn.conn_type_id\n"
        "_struct_conn.ptnr1_label_asym_id\n"
        "_struct_conn.ptnr1_label_comp_id\n"
        "_struct_conn.ptnr1_label_seq_id\n"
        "_struct_conn.ptnr1_label_atom_id\n"
        "_struct_conn.ptnr2_label_asym_id\n"
        "_struct_conn.ptnr2_label_comp_id\n"
        "_struct_conn.ptnr2_label_seq_id\n"
        "_struct_conn.ptnr2_label_atom_id\n"
        "m1 metalc A HIS 2 NE2 A ZN 1 ZN\n"
        "#\n"
    )
