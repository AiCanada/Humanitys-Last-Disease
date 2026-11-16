"""Drug-type reports on a finished design.

The counts are a report. They do not clear a bar and they do not add a hit.
"""

import json
from pathlib import Path

import pytest

from rfproteina.disease.combinations import AXES, CLASS_CONDITION_DIM, PROTEIN_AXES
from rfproteina.disease.modality import HYDROPHOBIC, component_label, modality_phrase, modality_record
from rfproteina.disease.screen import format_screen_lines, screen_generation
from tests.helpers.disease import (
    design_cif,
    no_developer_proteome,  # noqa: F401
    screen_page,
    screened_design_run,
)

_FURIN = "ARAKRAA"
_PLAIN = "AAAAAAAAAA"
# ExPASy ProtParam N-end rule table. Mammalian, yeast, E. coli. Proline has no E. coli estimate.
_N_END = {
    "A": ("4.4 hour", ">20 hour", ">10 hour"),
    "R": ("1 hour", "2 min", "2 min"),
    "N": ("1.4 hour", "3 min", ">10 hour"),
    "D": ("1.1 hour", "3 min", ">10 hour"),
    "C": ("1.2 hour", ">20 hour", ">10 hour"),
    "Q": ("0.8 hour", "10 min", ">10 hour"),
    "E": ("1 hour", "30 min", ">10 hour"),
    "G": ("30 hour", ">20 hour", ">10 hour"),
    "H": ("3.5 hour", "10 min", ">10 hour"),
    "I": ("20 hour", "30 min", ">10 hour"),
    "L": ("5.5 hour", "3 min", "2 min"),
    "K": ("1.3 hour", "3 min", "2 min"),
    "M": ("30 hour", ">20 hour", ">10 hour"),
    "F": ("1.1 hour", "3 min", "2 min"),
    "P": (">20 hour", ">20 hour", None),
    "S": ("1.9 hour", ">20 hour", ">10 hour"),
    "T": ("7.2 hour", ">20 hour", ">10 hour"),
    "W": ("2.8 hour", "3 min", "2 min"),
    "Y": ("2.8 hour", "10 min", "2 min"),
    "V": ("100 hour", ">20 hour", ">10 hour"),
}


def test_the_axis_list_stays_the_same_length():
    assert CLASS_CONDITION_DIM == 44
    assert "modality_screen" not in AXES


def test_furin_asp_pro_and_glutamine_are_stored_without_a_call():
    furin = modality_record("protein", _FURIN)
    assert furin is not None
    assert furin["furin_sites"] == [{"residue": 2, "motif": "RXKR"}]
    assert furin["acid_cleavage_count"] == 0
    assert furin["n_terminal_glutamine"] is False
    both = modality_record("protein", "ARARRA")
    assert both is not None
    assert both["furin_sites"] == [{"residue": 2, "motif": "RXRR"}]
    acid = modality_record("peptide", "AADPA")
    assert acid is not None
    assert acid["binder_kind"] == "peptide"
    assert acid["acid_cleavage_sites"] == [{"residue": 3, "motif": "DP"}]
    glutamine = modality_record("protein", "QAAAAA")
    assert glutamine is not None
    assert glutamine["n_terminal_glutamine"] is True
    phrase = modality_phrase(glutamine)
    assert phrase == "N-terminal glutamine"
    glutamate = modality_record("protein", "EAAAAA")
    assert glutamate is not None
    assert glutamate["n_terminal_glutamine"] is False
    wrapped = modality_record("protein", "A RAKRA A")
    assert wrapped is not None
    assert wrapped["furin_sites"] == [{"residue": 2, "motif": "RXKR"}]
    overlap = modality_record("protein", "RRRRR")
    assert overlap is not None
    assert overlap["furin_sites"] == [
        {"residue": 1, "motif": "RXRR"},
        {"residue": 2, "motif": "RXRR"},
    ]
    acid_pair = modality_record("peptide", "DPDP")
    assert acid_pair is not None
    assert acid_pair["acid_cleavage_sites"] == [
        {"residue": 1, "motif": "DP"},
        {"residue": 3, "motif": "DP"},
    ]


def test_hydrophobic_runs_use_the_positive_kyte_doolittle_set():
    assert "W" not in HYDROPHOBIC
    record = modality_record("protein", "WWWWAAAA")
    assert record is not None
    assert record["longest_hydrophobic_run"] == 4
    assert record["hydrophobic_run_start"] == 5
    assert record["longest_homopolymer"] == 4
    assert record["homopolymer_residue"] == "W"
    assert record["homopolymer_start"] == 1
    tied = modality_record("protein", "AAWWW")
    assert tied is not None
    assert tied["longest_homopolymer"] == 3
    assert tied["homopolymer_residue"] == "W"
    assert tied["homopolymer_start"] == 3


def test_one_furin_site_is_stored_and_the_call_stays(tmp_path: Path):
    record = screened_design_run(tmp_path, _FURIN)
    modality = record["modality_screen"]
    assert isinstance(modality, dict)
    assert modality["binder_kind"] == "protein"
    assert modality["furin_count"] == 1
    assert record["call"] == "untested"
    assert record["hits"] == []
    assert record["samples"][0]["modality_screen"] == modality
    measured = format_screen_lines("1ABC", "A", record)[1]
    assert "furin 1, residue 2 motif RXKR" in measured
    stored = json.dumps(record)
    assert _FURIN not in stored
    page = screen_page(record)
    assert "Protein sequence triage" in page
    assert "Residue 2, motif RXKR" in page
    assert _FURIN not in page
    assert "cure" not in page.casefold()


def test_a_clear_bar_stays_clear_when_a_furin_site_is_present(tmp_path: Path):
    run = tmp_path / "clear"
    evidence = {axis: "clear" for axis in PROTEIN_AXES}
    run.mkdir()
    screened_design_run(run, _FURIN)
    (run / "gene_1ABC_1" / "axis_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    again = screen_generation(run / "gene_1ABC_1", binder_kind="protein", target_id="P05113")
    assert again["call"] == "screened_clear"
    assert again["hits"] == []
    assert again["modality_screen"]["furin_count"] == 1


@pytest.mark.parametrize("kind", ["protein", "peptide", "polypeptide"])
def test_protein_and_peptide_store_the_report(tmp_path: Path, kind: str):
    record = screened_design_run(tmp_path / kind, _FURIN, kind=kind)
    modality = record["modality_screen"]
    assert modality["furin_count"] == 1
    expected = "peptide" if kind == "peptide" else "protein"
    assert modality["binder_kind"] == expected
    assert record["call"] != "axis_hit"


def test_a_nucleic_chain_reports_cpg_and_leaves_the_call(tmp_path: Path):
    sequence = "ACGUACGU"
    record = screened_design_run(tmp_path, sequence, kind="nucleic")
    modality = record["modality_screen"]
    assert isinstance(modality, dict)
    assert modality["binder_kind"] == "nucleic"
    assert modality["gc_fraction"] == 0.5
    assert modality["cpg_positions"] == [2, 6]
    assert modality["longest_g_run"] == 1
    assert record["sequence_developability"]["gravy"] is None
    assert record["call"] == "untested"
    measured = format_screen_lines("1ABC", "A", record)[1]
    assert "CpG 2, residues 2, 6" in measured
    assert "GC fraction" not in measured
    page = screen_page(record)
    assert "Oligonucleotide triage" in page
    assert "GC fraction 0.5" in page
    modality["gc_fraction"] = 1.5
    modality["cpg_positions"] = [0, 2, 6]
    again = screen_page(record)
    assert "GC fraction" not in again
    assert "Residues 2, 6." in again
    assert "Residue 0" not in again
    assert sequence not in page
    assert sequence not in json.dumps(record)


def test_an_amino_acid_chain_screened_as_nucleic_leaves_the_report_out(tmp_path: Path):
    record = screened_design_run(tmp_path, "ACNGAA", kind="nucleic")
    assert "modality_screen" not in record


def test_one_nonpolymer_stores_its_formula_weight(tmp_path: Path):
    record = screened_design_run(tmp_path, _PLAIN, kind="ligand", bonds=_component("ATP", "180.5"))
    modality = record["modality_screen"]
    assert isinstance(modality, dict)
    assert modality["formula_weight"] == 180.5
    assert modality["component_id"] == "ATP"
    assert record["call"] != "axis_hit"
    measured = format_screen_lines("1ABC", "A", record)[1]
    assert "formula weight" not in measured.casefold()
    page = screen_page(record)
    assert "Small-molecule composition" in page
    assert "Formula weight 180.5 for ATP." in page
    assert "cure" not in page.casefold()


def test_a_nonloop_component_is_named_and_water_is_left_out(tmp_path: Path):
    bonds = (
        "_chem_comp.id ATP\n"
        "_chem_comp.type non-polymer\n"
        "_chem_comp.formula_weight 507.181\n"
        "_chem_comp_atom.comp_id ATP\n"
        "#\n"
    )
    record = screened_design_run(tmp_path / "line", _PLAIN, kind="ligand", bonds=bonds)
    assert record["modality_screen"]["component_id"] == "ATP"
    water = _component("HOH", "18.015") + _component("ATP", "507.181")
    kept = screened_design_run(tmp_path / "water", _PLAIN, kind="ligand", bonds=water)
    assert kept["modality_screen"]["component_id"] == "ATP"
    ambiguous = _component("ATP", "507.181") + _component("GLC", "180.156")
    wide = screened_design_run(tmp_path / "two", _PLAIN, kind="ligand", bonds=ambiguous)
    assert wide["modality_screen"]["composition"] == "ambiguous"
    assert wide["modality_screen"]["component_count"] == 2
    assert "formula_weight" not in wide["modality_screen"]


def test_a_pasted_motif_name_stays_off_the_line_and_the_page(tmp_path: Path):
    pasted = "ACDEFGHIKLMNPQRSTVWY" * 2
    record = screened_design_run(tmp_path, _FURIN)
    modality = record["modality_screen"]
    modality["furin_count"] = 2
    modality["furin_sites"] = [
        {"residue": 2, "motif": "RXKR"},
        {"residue": 0, "motif": "RXKR"},
        {"residue": True, "motif": "RXRR"},
        {"residue": 4, "motif": pasted},
    ]
    modality["furin_count"] = 4
    assert modality_phrase(modality) == "furin 4, residue 2 motif RXKR"
    page = screen_page(record)
    assert "Furin cleavage motif 4." in page
    assert "Residue 2, motif RXKR" in page
    assert "Residue 0" not in page
    assert pasted not in page


def test_a_nonstandard_letter_is_counted_and_the_chain_is_still_scanned():
    chain = "XXXX"
    unusual = modality_record("protein", chain)
    assert unusual is not None
    assert unusual["nonstandard_count"] == 4
    assert unusual["nonstandard_residues"] == [1, 2, 3, 4]
    assert unusual["homopolymer_residue"] == "X"
    assert chain not in json.dumps(unusual)
    mixed = modality_record("protein", "ARAKRAX")
    assert mixed is not None
    assert mixed["furin_count"] == 1
    assert mixed["nonstandard_residues"] == [7]
    assert modality_phrase(mixed) == "furin 1, residue 2 motif RXKR; nonstandard 1, residue 7"
    assert "ARAKRAX" not in json.dumps(mixed)
    spaced = modality_record("nucleic", "AC GU AC GU")
    assert spaced is not None
    assert spaced["cpg_positions"] == [2, 6]
    assert modality_record("nucleic", "ACN G") is None


def test_a_nonstandard_letter_is_reported_and_the_call_stays(tmp_path: Path):
    sequence = "ACX"
    record = screened_design_run(tmp_path, sequence)
    modality = record["modality_screen"]
    assert isinstance(modality, dict)
    assert modality["nonstandard_count"] == 1
    assert modality["nonstandard_residues"] == [3]
    assert modality["furin_count"] == 0
    assert record["call"] == "untested"
    assert record["sequence_developability"]["gravy"] is None
    assert "chemical_liability" not in record
    measured = format_screen_lines("1ABC", "A", record)[1]
    assert "nonstandard 1, residue 3" in measured
    assert "furin" not in measured
    page = screen_page(record)
    assert "Nonstandard residue 1." in page
    assert "Residue 3." in page
    assert "was not recorded" not in page
    assert "No number yet" not in page
    assert "was not measured" not in page
    assert '<div class="metric-row">' not in page
    assert sequence not in page
    assert sequence not in json.dumps(record)
    assert "cure" not in page.casefold()


def test_a_missing_formula_weight_is_left_out():
    missing = modality_record(
        "ligand",
        components=[{"_chem_comp.id": "ATP", "_chem_comp.type": "non-polymer", "_chem_comp.formula_weight": "?"}],
    )
    assert missing is None
    blank = screen_page(
        {
            "call": "untested",
            "modality_screen": {"binder_kind": "ligand", "composition": "weight_unreadable", "component_id": "ATP"},
        }
    )
    assert "Formula weight was not read" not in blank
    assert "Small-molecule composition" not in blank
    assert "weight_unreadable" not in blank
    assert "cure" not in blank.casefold()
    quoted = modality_record(
        "ligand",
        components=[
            {
                "_chem_comp.id": "'ATP'",
                "_chem_comp.type": "'non-polymer'",
                "_chem_comp.formula_weight": "'180.5'",
            }
        ],
    )
    assert quoted is not None
    assert quoted["formula_weight"] == 180.5
    assert quoted["component_id"] == "ATP"
    pasted = "ACDEFGHIKLMNPQRSTVWY"
    assert component_label(pasted[:9]) is None
    hidden = screen_page({"call": "untested", "modality_screen": {**quoted, "component_id": pasted}})
    assert pasted not in hidden
    assert "Formula weight 180.5." in hidden
    assert modality_record("ligand", components=[]) is None
    water = modality_record(
        "ligand",
        components=[{"_chem_comp.id": "'HOH'", "_chem_comp.type": "non-polymer", "_chem_comp.formula_weight": "18"}],
    )
    assert water is None


def test_a_quoted_name_does_not_move_the_formula_weight(tmp_path: Path):
    record = screened_design_run(
        tmp_path / "zinc",
        _PLAIN,
        kind="ligand",
        bonds=_ion_loop("ZN non-polymer . 'ZINC ION' ? 'Zn 2' 40.5\n"),
    )
    modality = record["modality_screen"]
    assert modality["binder_kind"] == "ligand"
    assert modality["component_id"] == "ZN"
    assert modality["formula_weight"] == 40.5
    assert record["call"] != "axis_hit"
    stored = json.dumps(record)
    assert "ZINC ION" not in stored
    assert "C3 H7 N O2" not in stored
    assert "H2 O" not in stored
    page = screen_page(record)
    assert "Formula weight 40.5 for ZN." in page
    assert "ZINC ION" not in page
    wide = _ion_loop("CA non-polymer . 'CALCIUM ION' ? 'Ca 2' 40.5\n" "ZN non-polymer . 'ZINC ION' ? 'Zn 2' 65.5\n")
    both = screened_design_run(tmp_path / "ions", _PLAIN, kind="ligand", bonds=wide)
    assert both["modality_screen"]["composition"] == "ambiguous"
    assert both["modality_screen"]["component_count"] == 2
    assert "formula_weight" not in both["modality_screen"]
    assert "ZINC ION" not in json.dumps(both)
    assert "CALCIUM ION" not in json.dumps(both)


def test_aliphatic_index_and_mass_stay_off_the_call(tmp_path: Path):
    alanine = modality_record("protein", "A")
    assert alanine is not None
    assert f"{alanine['aliphatic_index']:.4f}" == "100.0000"
    assert f"{alanine['molecular_weight']:.4f}" == "89.0932"
    assert alanine["molecular_weight_source"] == "iupac_conventional"
    pair = modality_record("peptide", "AA")
    assert pair is not None
    assert f"{pair['molecular_weight']:.4f}" == "160.1711"
    glycine = modality_record("protein", "G")
    assert glycine is not None
    assert f"{glycine['molecular_weight']:.4f}" == "75.0666"
    assert glycine["aliphatic_index"] == 0.0
    mixed = modality_record("protein", "AVIL")
    assert mixed is not None
    assert f"{mixed['aliphatic_index']:.4f}" == "292.5000"
    assert modality_phrase(mixed) is None
    odd = modality_record("protein", "AX")
    assert odd is not None
    assert "molecular_weight" not in odd
    assert f"{odd['aliphatic_index']:.4f}" == "50.0000"
    assert odd["nonstandard_count"] == 1
    assert "AX" not in json.dumps(odd)
    record = screened_design_run(tmp_path, _PLAIN)
    modality = record["modality_screen"]
    assert f"{modality['aliphatic_index']:.4f}" == "100.0000"
    assert f"{modality['molecular_weight']:.4f}" == "728.7943"
    assert record["call"] == "untested"
    measured = format_screen_lines("1ABC", "A", record)[1]
    assert "aliphatic" not in measured.casefold()
    assert "molecular weight" not in measured.casefold()
    page = screen_page(record)
    assert "Aliphatic index 100." in page
    assert "Molecular weight 728.7943, from the amino-acid formulas." in page
    assert _PLAIN not in page
    assert "cure" not in page.casefold()
    blank = screen_page({"call": "untested", "modality_screen": {"binder_kind": "protein", "aliphatic_index": -1}})
    assert "Aliphatic index" not in blank
    assert "Protein sequence triage" not in blank
    assert "was not recorded" not in blank
    assert "No number yet" not in blank


def test_extinction_half_life_composition_and_spans_stay_off_the_call(tmp_path: Path):
    for letter, expected in _N_END.items():
        record = modality_record("protein", letter)
        assert record is not None
        mammalian, yeast, ecoli = expected
        assert record["half_life_source"] == "n_end_rule"
        assert record["half_life_mammalian"] == mammalian
        assert record["half_life_yeast"] == yeast
        if ecoli is None:
            assert "half_life_ecoli" not in record
        else:
            assert record["half_life_ecoli"] == ecoli
    alanine = modality_record("protein", "A")
    assert alanine is not None
    assert alanine["extinction_source"] == "pace_1995"
    assert alanine["extinction_280_reduced"] == 0
    assert alanine["extinction_280_cystine"] == 0
    assert "absorbance_280_reduced" not in alanine
    assert "absorbance_280_cystine" not in alanine
    assert alanine["atom_counts"] == {"C": 3, "H": 7, "N": 1, "O": 2}
    assert alanine["residue_composition"] == {"A": 1}
    assert alanine["transmembrane_count"] == 0
    assert alanine["transmembrane_spans"] == []
    assert alanine["transmembrane_method"] == "kyte_doolittle_1982"
    assert f"{alanine['sequence_entropy_bits']:.4f}" == "0.0000"
    pair = modality_record("peptide", "AA")
    assert pair is not None
    assert pair["atom_counts"] == {"C": 6, "H": 12, "N": 2, "O": 3}
    tryptophan = modality_record("protein", "W")
    assert tryptophan is not None
    assert f"{tryptophan['molecular_weight']:.4f}" == "204.2252"
    assert tryptophan["extinction_280_reduced"] == 5500
    assert tryptophan["extinction_280_cystine"] == 5500
    assert f"{tryptophan['absorbance_280_reduced']:.4f}" == "26.9311"
    assert f"{tryptophan['absorbance_280_cystine']:.4f}" == "26.9311"
    assert tryptophan["atom_counts"] == {"C": 11, "H": 12, "N": 2, "O": 2}
    tyrosine = modality_record("protein", "Y")
    assert tyrosine is not None
    assert f"{tyrosine['molecular_weight']:.4f}" == "181.1885"
    assert tyrosine["extinction_280_reduced"] == 1490
    assert f"{tyrosine['absorbance_280_reduced']:.4f}" == "8.2235"
    cysteine = modality_record("protein", "C")
    assert cysteine is not None
    assert cysteine["extinction_280_cystine"] == 0
    assert cysteine["atom_counts"]["S"] == 1
    cystine = modality_record("peptide", "CC")
    assert cystine is not None
    assert cystine["binder_kind"] == "peptide"
    assert cystine["extinction_280_reduced"] == 0
    assert cystine["extinction_280_cystine"] == 125
    assert "absorbance_280_reduced" not in cystine
    assert f"{cystine['absorbance_280_cystine']:.4f}" == "0.5573"
    assert cystine["atom_counts"] == {"C": 6, "H": 12, "N": 2, "O": 3, "S": 2}
    both = modality_record("protein", "WCC")
    assert both is not None
    assert both["extinction_280_reduced"] == 5500
    assert both["extinction_280_cystine"] == 5625
    assert both["atom_counts"] == {"C": 17, "H": 22, "N": 4, "O": 4, "S": 2}
    odd = modality_record("protein", "AX")
    assert odd is not None
    assert "extinction_280_reduced" not in odd
    assert "atom_counts" not in odd
    assert "transmembrane_count" not in odd
    assert odd["residue_composition"] == {"A": 1}
    assert odd["half_life_mammalian"] == "4.4 hour"
    assert odd["most_common_residue"] == "A"
    assert f"{odd['sequence_entropy_bits']:.4f}" == "1.0000"
    assert f"{odd['most_common_fraction']:.4f}" == "0.5000"
    assert odd["distinct_residues"] == 2
    assert "X" not in json.dumps(odd)
    assert modality_phrase(odd) == "nonstandard 1, residue 2"
    odd_page = screen_page({"call": "untested", "modality_screen": odd})
    assert "Composition A 1." in odd_page
    assert "Sequence entropy 1 bit." in odd_page
    assert "N-end rule half-life: mammalian 4.4 hour, yeast &gt;20 hour, E. coli &gt;10 hour." in odd_page
    assert "Extinction" not in odd_page
    assert "Atoms" not in odd_page
    assert "Kyte-Doolittle" not in odd_page
    assert "AX" not in odd_page
    assert "was not recorded" not in odd_page
    assert "No number yet" not in odd_page
    foreign = modality_record("protein", "XAAA")
    assert foreign is not None
    assert "half_life_mammalian" not in foreign
    assert foreign["residue_composition"] == {"A": 3}
    assert "X" not in json.dumps(foreign)
    proline = modality_record("protein", "P")
    assert proline is not None
    assert "half_life_ecoli" not in proline
    proline_page = screen_page({"call": "untested", "modality_screen": proline})
    assert "N-end rule half-life: mammalian &gt;20 hour, yeast &gt;20 hour." in proline_page
    assert "E. coli" not in proline_page
    span = modality_record("protein", "I" * 19)
    assert span is not None
    assert span["transmembrane_count"] == 1
    assert span["transmembrane_spans"][0]["residue_start"] == 1
    assert span["transmembrane_spans"][0]["residue_end"] == 19
    assert f"{span['transmembrane_spans'][0]['hydropathy']:.4f}" == "4.5000"
    merged = modality_record("protein", "I" * 20)
    assert merged is not None
    assert merged["transmembrane_spans"] == [{"residue_start": 1, "residue_end": 20, "hydropathy": 4.5}]
    alanines = modality_record("protein", "A" * 19)
    assert alanines is not None
    assert f"{alanines['transmembrane_spans'][0]['hydropathy']:.4f}" == "1.8000"
    glycines = modality_record("protein", "G" * 19)
    assert glycines is not None
    assert glycines["transmembrane_count"] == 0
    assert glycines["transmembrane_spans"] == []
    short = modality_record("protein", "A" * 10)
    assert short is not None
    assert short["transmembrane_count"] == 0
    level = modality_record("protein", "FFFFF" + ("A" * 10) + "GGGG")
    assert level is not None
    assert level["transmembrane_count"] == 0
    above = modality_record("protein", "FFFFF" + ("A" * 11) + "GGG")
    assert above is not None
    assert above["transmembrane_count"] == 1
    assert above["transmembrane_spans"][0]["residue_end"] == 19
    assert f"{above['transmembrane_spans'][0]['hydropathy']:.4f}" == "1.7158"
    split = "I" * 19 + "R" * 12 + "I" * 19
    separated = modality_record("protein", split)
    assert separated is not None
    assert separated["transmembrane_spans"] == [
        {"residue_start": 1, "residue_end": 25, "hydropathy": 4.5},
        {"residue_start": 26, "residue_end": 50, "hydropathy": 4.5},
    ]
    assert split not in json.dumps(separated)
    assert modality_phrase(tryptophan) is None
    page = screen_page({"call": "untested", "modality_screen": tryptophan})
    markers = [
        "Molecular weight 204.2252, from the amino-acid formulas.",
        "Extinction at 280 nm is 5500 with cystines reduced and 5500 with cystines paired.",
        "Absorbance 0.1% (1 g/L) is 26.9311 with cystines reduced and 26.9311 with cystines paired.",
        "N-end rule half-life: mammalian 2.8 hour, yeast 3 min, E. coli 2 min.",
        "Composition W 1.",
        "Atoms C 11, H 12, N 2, O 2.",
        "Sequence entropy 0 bits.",
        "Most common residue W, share 1.",
        "Distinct residues 1.",
        "Furin cleavage motif 0.",
        "Kyte-Doolittle span 0.",
        "These counts are a report. No pass band applies to them.",
    ]
    positions = [page.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "cure" not in page.casefold()
    span_page = screen_page({"call": "untested", "modality_screen": span})
    assert "Kyte-Doolittle span 1. Residues 1-19, hydropathy 4.5." in span_page
    split_page = screen_page({"call": "untested", "modality_screen": separated})
    assert "Kyte-Doolittle span 2. Residues 1-25, hydropathy 4.5. Residues 26-50, hydropathy 4.5." in split_page
    poison = screen_page(
        {
            "call": "untested",
            "modality_screen": {
                "binder_kind": "protein",
                "length": 1,
                "half_life_mammalian": "4.4 hour <script>",
                "half_life_yeast": ">20 hour",
                "half_life_ecoli": "?",
                "sequence_entropy_bits": -1,
                "most_common_residue": "AX",
                "most_common_fraction": 2,
                "distinct_residues": 0,
                "transmembrane_count": 0,
                "extinction_280_reduced": -5,
                "extinction_280_cystine": False,
                "residue_composition": {"A": 1, "X": 4},
                "atom_counts": {"C": 3, "Fe": 1, "H": 0},
                "aliphatic_index": -1,
            },
        }
    )
    assert "N-end rule half-life: yeast &gt;20 hour." in poison
    assert "4.4 hour" not in poison
    assert "&lt;script&gt;" not in poison
    assert "Sequence entropy" not in poison
    assert "Most common" not in poison
    assert "Distinct residues" not in poison
    assert "Extinction" not in poison
    assert "Kyte-Doolittle span 0." in poison
    assert "Composition A 1." in poison
    assert "X 4" not in poison
    assert "Atoms C 3." in poison
    assert "Fe 1" not in poison
    assert "Aliphatic index" not in poison
    assert "was not recorded" not in poison
    nucleic = modality_record("nucleic", "ACGT")
    assert nucleic is not None
    assert "extinction_280_reduced" not in nucleic
    assert "half_life_mammalian" not in nucleic
    assert "transmembrane_count" not in nucleic
    assert "residue_composition" not in nucleic
    screened = screened_design_run(tmp_path, "WWW")
    assert screened["call"] == "untested"
    stored = screened["modality_screen"]
    assert stored["extinction_280_reduced"] == 16500
    assert stored["extinction_280_cystine"] == 16500
    assert "WWW" not in json.dumps(screened)
    measured = " ".join(format_screen_lines("1ABC", "A", screened))
    for phrase in (
        "Extinction",
        "Absorbance",
        "half-life",
        "Kyte-Doolittle",
        "Sequence entropy",
        "Composition",
        "Atoms",
    ):
        assert phrase not in measured


# Aβ42, the peptide whose central cluster supplied the 2005 scale. Published
# Na4vSS values, to one decimal, are 6.4, -2.2, and -6.3 (Data Brief 2016, 7:143).
_ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def test_aggrescan_uses_the_2007_window_and_stays_off_the_call(tmp_path: Path) -> None:
    isoleucine = modality_record("peptide", "IIIII")
    assert isoleucine is not None
    score = isoleucine["aggrescan"]
    assert score["method"] == "aggrescan_2007"
    assert score["scale_source"] == "sanchez_de_groot_2005"
    assert score["window"] == 5
    assert score["hot_spot_threshold"] == -0.02
    assert f"{score['na4vss']:.4f}" == "131.3720"
    assert f"{score['a4vss']:.4f}" == "6.5686"
    assert f"{score['a3vsa']:.4f}" == "1.8220"
    assert score["hotspots"] == [{"residue_start": 1, "residue_end": 5}]
    aspartate = modality_record("peptide", "DDDDD")
    assert aspartate is not None
    assert f"{aspartate['aggrescan']['na4vss']:.4f}" == "-175.9000"
    assert aspartate["aggrescan"]["hotspot_count"] == 0
    assert aspartate["aggrescan"]["hotspots"] == []
    broken = modality_record("peptide", "IIIIIPIIIII")
    assert broken is not None
    assert f"{broken['aggrescan']['na4vss']:.4f}" == "139.4964"
    assert broken["aggrescan"]["hotspots"] == [
        {"residue_start": 1, "residue_end": 5},
        {"residue_start": 7, "residue_end": 11},
    ]
    wild = modality_record("peptide", _ABETA42)
    assert wild is not None
    assert f"{wild['aggrescan']['na4vss']:.4f}" == "6.3771"
    assert wild["aggrescan"]["hotspots"] == [
        {"residue_start": 17, "residue_end": 22},
        {"residue_start": 30, "residue_end": 42},
    ]
    phenylalanine = _ABETA42[:18] + "D" + _ABETA42[19:]
    changed = modality_record("peptide", phenylalanine)
    assert changed is not None
    assert f"{changed['aggrescan']['na4vss']:.4f}" == "-2.1705"
    doubled = phenylalanine[:33] + "P" + phenylalanine[34:]
    both = modality_record("peptide", doubled)
    assert both is not None
    assert f"{both['aggrescan']['na4vss']:.4f}" == "-6.2514"
    assert modality_record("peptide", "A" * 75)["aggrescan"]["window"] == 5
    assert modality_record("protein", "A" * 76)["aggrescan"]["window"] == 7
    assert modality_record("protein", "A" * 175)["aggrescan"]["window"] == 7
    assert modality_record("protein", "A" * 176)["aggrescan"]["window"] == 9
    assert modality_record("protein", "A" * 300)["aggrescan"]["window"] == 9
    assert modality_record("protein", "A" * 301)["aggrescan"]["window"] == 11
    assert f"{modality_record('peptide', 'A' * 75)['aggrescan']['na4vss']:.4f}" == "-5.0067"
    assert f"{modality_record('protein', 'A' * 76)['aggrescan']['na4vss']:.4f}" == "-5.0873"
    short = modality_record("peptide", "IIII")
    assert short is not None and "aggrescan" not in short
    foreign = modality_record("protein", "IIIIIX")
    assert foreign is not None and "aggrescan" not in foreign
    nucleic = modality_record("nucleic", "ACGTACGT")
    assert nucleic is not None and "aggrescan" not in nucleic
    page = screen_page({"call": "untested", "modality_screen": isoleucine})
    assert "AGGRESCAN normalized sum 131.372. Hot spots 1. Residues 1-5." in page
    assert page.index("Kyte-Doolittle span 0.") < page.index("AGGRESCAN normalized sum 131.372")
    assert page.index("AGGRESCAN normalized sum 131.372") < page.index("No pass band applies")
    quiet = screen_page({"call": "untested", "modality_screen": aspartate})
    assert "AGGRESCAN normalized sum -175.9. Hot spots 0." in quiet
    split = screen_page({"call": "untested", "modality_screen": broken})
    assert "Hot spots 2. Residues 1-5. Residues 7-11." in split
    poison = screen_page(
        {
            "call": "untested",
            "modality_screen": {
                "binder_kind": "protein",
                "length": 5,
                "aggrescan": {
                    "method": "aggrescan_2007",
                    "na4vss": "6.4 <script>",
                    "hotspot_count": 1,
                    "hotspots": [{"residue_start": "<script>", "residue_end": 5}],
                },
            },
        }
    )
    assert "AGGRESCAN" not in poison
    assert "&lt;script&gt;" not in poison
    misnamed = screen_page(
        {
            "call": "untested",
            "modality_screen": {
                "binder_kind": "protein",
                "aggrescan": {"method": "camsol", "na4vss": 0.72, "hotspot_count": 0},
            },
        }
    )
    assert "AGGRESCAN" not in misnamed
    folder = tmp_path / "gene_1ABC_1"
    folder.mkdir()
    (folder / "solubility.tsv").write_text(
        "score\tmethod\tsource\n0.72\tCamSol intrinsic\tSormanni 2015\n",
        encoding="utf-8",
    )
    screened = screened_design_run(folder, "IIIII")
    assert screened["call"] == "untested"
    assert "IIIII" not in json.dumps(screened)
    assert screened["modality_screen"]["aggrescan"]["na4vss"] == isoleucine["aggrescan"]["na4vss"]
    assert screened["solubility"]["scores"][0]["method"] == "CamSol intrinsic"
    measured = " ".join(format_screen_lines("1ABC", "A", screened))
    assert "AGGRESCAN" not in measured
    assert "CamSol intrinsic 0.72" in measured
    assert modality_phrase(isoleucine) is None


def test_a_formula_supplies_the_weight_when_the_cell_is_blank():
    atp = _ligand_row("ATP", "C10 H16 N5 O13 P3", "?")
    assert atp is not None
    assert f"{atp['formula_weight']:.3f}" == "507.181"
    assert atp["formula_weight_source"] == "iupac_conventional"
    assert atp["component_id"] == "ATP"
    assert "composition" not in atp
    stored = json.dumps(atp)
    assert "C10 H16 N5 O13 P3" not in stored
    assert "?" not in stored
    zinc = _ligand_row("ZN", "Zn 2", "?")
    assert zinc is not None
    assert f"{zinc['formula_weight']:.3f}" == "65.409"
    calcium = _ligand_row("CA", "Ca 2", ".")
    assert calcium is not None
    assert f"{calcium['formula_weight']:.3f}" == "40.078"
    alanine = _ligand_row("X", "C3 H7 N O2", "")
    assert alanine is not None
    assert f"{alanine['formula_weight']:.3f}" == "89.093"
    waterish = _ligand_row("XXX", "H2 O", "nan")
    assert waterish is not None
    assert f"{waterish['formula_weight']:.3f}" == "18.015"
    folded = _ligand_row("ZN", "ZN", "0")
    assert folded is not None
    assert f"{folded['formula_weight']:.3f}" == "65.409"
    kept = _ligand_row("ZN", "Zn 2", "40.5")
    assert kept is not None
    assert kept["formula_weight"] == 40.5
    assert "formula_weight_source" not in kept
    assert _ligand_row("QQ", "C2 Qq", "?") is None
    assert _ligand_row("QQ", "C0 H2 O", "?") is None
    assert _ligand_row("QQ", "(C2 H4)2", "?") is None
    assert _ligand_row("QQ", "C2 H6 O . H2 O", "?") is None
    assert _ligand_row("QQ", "2", "?") is None
    assert modality_phrase(atp) is None
    page = screen_page({"call": "untested", "modality_screen": atp})
    assert "Formula weight 507.181 for ATP, from the chemical formula." in page
    assert "Formula weight was not read" not in page
    assert "C10 H16 N5 O13 P3" not in page
    assert "cure" not in page.casefold()


def test_a_blank_weight_uses_the_formula_and_an_unclosed_quote_adds_no_cell(tmp_path: Path):
    record = screened_design_run(
        tmp_path / "zinc",
        _PLAIN,
        kind="ligand",
        bonds=_ion_loop("ZN non-polymer . 'ZINC ION' ? 'Zn 2' ?\n"),
    )
    modality = record["modality_screen"]
    assert modality["component_id"] == "ZN"
    assert f"{modality['formula_weight']:.3f}" == "65.409"
    assert modality["formula_weight_source"] == "iupac_conventional"
    assert record["call"] != "axis_hit"
    stored = json.dumps(record)
    assert "Zn 2" not in stored
    assert "ZINC ION" not in stored
    assert "C3 H7 N O2" not in stored
    page = screen_page(record)
    assert "Formula weight 65.409 for ZN, from the chemical formula." in page
    assert "was not read" not in page
    broken = (
        "loop_\n"
        "_chem_comp.id\n"
        "_chem_comp.type\n"
        "_chem_comp.formula_weight\n"
        "ATP non-polymer '180.5\n"
        "#\n"
        "_chem_comp.id ZN\n"
        "_chem_comp.type non-polymer\n"
        "_chem_comp.formula Zn 2\n"
        "_chem_comp.formula_weight ?\n"
        "#\n"
    )
    followed = screened_design_run(tmp_path / "open", _PLAIN, kind="ligand", bonds=broken)
    followed_text = json.dumps(followed)
    assert "180.5" not in followed_text
    assert "Zn 2" not in followed_text
    assert f"{followed['modality_screen']['formula_weight']:.3f}" == "65.409"
    assert followed["modality_screen"]["component_id"] == "ZN"
    assert followed["call"] != "axis_hit"


def test_a_value_on_the_next_line_names_one_component(tmp_path: Path):
    bonds = (
        "_chem_comp.id\n"
        "\n"
        "ATP\n"
        "_chem_comp.type\n"
        "non-polymer\n"
        "_chem_comp.formula_weight\n"
        "180.5\n"
        "_chem_comp_atom.comp_id\n"
        "ATP\n"
        "#\n"
    )
    record = screened_design_run(tmp_path / "next", _PLAIN, kind="ligand", bonds=bonds)
    modality = record["modality_screen"]
    assert modality["component_id"] == "ATP"
    assert modality["formula_weight"] == 180.5
    assert record["call"] != "axis_hit"
    quoted = (
        "loop_\n"
        "_chem_comp.id\n"
        "_chem_comp.type\n"
        "_chem_comp.formula_weight\n"
        "'ATP' 'non-polymer' '180.5'\n"
        "#\n"
    )
    named = screened_design_run(tmp_path / "quoted", _PLAIN, kind="ligand", bonds=quoted)
    assert named["modality_screen"]["component_id"] == "ATP"
    assert named["modality_screen"]["formula_weight"] == 180.5


def test_an_uneven_component_loop_does_not_add_a_component(tmp_path: Path):
    uneven = "loop_\n" "_chem_comp.id\n" "_chem_comp.type\n" "_chem_comp.formula_weight\n" "ATP non-polymer\n" "#\n"
    absent = screened_design_run(tmp_path / "uneven", _PLAIN, kind="ligand", bonds=uneven)
    assert "modality_screen" not in absent
    both = uneven + "_chem_comp.id ATP\n_chem_comp.type non-polymer\n_chem_comp.formula_weight 180.5\n#\n"
    kept = screened_design_run(tmp_path / "kept", _PLAIN, kind="ligand", bonds=both)
    modality = kept["modality_screen"]
    assert modality["component_id"] == "ATP"
    assert modality["formula_weight"] == 180.5
    assert "composition" not in modality
    unreadable = _component("ATP", "?")
    missing = screened_design_run(tmp_path / "blank", _PLAIN, kind="ligand", bonds=unreadable)
    assert "modality_screen" not in missing
    assert "formula weight" not in format_screen_lines("1ABC", "A", missing)[1].casefold()
    assert "was not read" not in screen_page(missing)


def test_the_first_design_is_the_drug_type_report(tmp_path: Path):
    run = tmp_path / "gene_1ABC_1"
    for name, sequence in (("binder_0", _FURIN), ("binder_1", _PLAIN)):
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
    assert samples[0]["modality_screen"]["furin_count"] == 1
    assert samples[1]["modality_screen"]["furin_count"] == 0
    assert record["modality_screen"] == samples[0]["modality_screen"]
    assert _FURIN not in json.dumps(record)
    assert _PLAIN not in json.dumps(record)


def test_a_modality_failure_keeps_the_sequence_metrics(tmp_path: Path, monkeypatch):
    def _boom(kind, sequence=None, *, components=None):
        raise ValueError("unreadable chain")

    monkeypatch.setattr("rfproteina.disease.screen.modality_record", _boom)
    record = screened_design_run(tmp_path, _FURIN)
    sequence = record["sequence_developability"]
    assert isinstance(sequence["gravy"], float)
    assert isinstance(sequence["instability_index"], float)
    assert "modality_screen" not in record
    assert "furin" not in format_screen_lines("1ABC", "A", record)[1]


def _ion_loop(rows: str) -> str:
    return (
        "loop_\n"
        "_chem_comp.id\n"
        "_chem_comp.type\n"
        "_chem_comp.mon_nstd_flag\n"
        "_chem_comp.name\n"
        "_chem_comp.pdbx_synonyms\n"
        "_chem_comp.formula\n"
        "_chem_comp.formula_weight\n"
        "ALA 'L-peptide linking' y ALANINE ? 'C3 H7 N O2' 89.093\n"
        + rows
        + "HOH non-polymer . WATER ? 'H2 O' 18.015\n#\n"
    )


def _ligand_row(name: str, formula: str, weight: str) -> dict[str, object] | None:
    return modality_record(
        "ligand",
        components=[
            {
                "_chem_comp.id": name,
                "_chem_comp.type": "non-polymer",
                "_chem_comp.formula": formula,
                "_chem_comp.formula_weight": weight,
            }
        ],
    )


def _component(name: str, weight: str) -> str:
    return (
        "loop_\n"
        "_chem_comp.id\n"
        "_chem_comp.type\n"
        "_chem_comp.formula_weight\n"
        f"{name} non-polymer {weight}\n"
        "#\n"
    )


def test_a_count_that_int_cannot_read_leaves_the_formula_unread():
    """``str.isdigit`` is true of a superscript; ``int`` refuses one.

    ``C²`` passed the count test and then raised ``ValueError`` out of a
    function documented to return None for a formula it cannot read -- the
    whole drug-type report went missing instead, on a module whose own rule is
    that no weight is published for a formula it has only partly understood.
    Refusing is also the right reading: a superscript after an element symbol
    is a charge, not a count. ``str.isdecimal`` is true of exactly the
    characters ``int`` reads, so an ordinary count is unaffected and a count in
    another script's decimal digits still reads.
    """
    for formula in ("C²", "C²6", "C₆", "C6 H1₂ O6", "C² H4"):
        assert _ligand_row("QQ", formula, "?") is None, formula

    glucose = _ligand_row("GLC", "C6 H12 O6", "?")
    assert glucose is not None
    assert f"{glucose['formula_weight']:.3f}" == "180.156"
    arabic = _ligand_row("GLC", "C6 H1٢ O6", "?")
    assert arabic is not None
    assert f"{arabic['formula_weight']:.3f}" == "180.156"
