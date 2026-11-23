"""Regression checks for attribution and the six-descriptor TNP result contract."""

import hashlib
import json
from pathlib import Path

import pytest

from rfproteina.disease import tnp, tnp_worker

CIF = """data_domain
_struct_asym.id H
_struct_asym.entity_id 1
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
_entity_poly.pdbx_seq_one_letter_code_can AG
_entity_poly.pdbx_seq_one_letter_code AG
loop_
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.label_comp_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
H 1 1 ALA 0 1 2 1
H 1 2 GLY 3 4 5 1
"""
SETTINGS = {"domain_type": "VHH", "label_asym_id": "H", "sequence_sha256": hashlib.sha256(b"AG").hexdigest()}


def test_sequence_is_the_declared_polymer_even_when_coordinates_are_incomplete():
    pytest.importorskip("gemmi")
    domain = tnp.Domain.parse(SETTINGS)
    assert tnp._extract(CIF, domain) == ("AG", 2)
    assert tnp._extract(CIF.replace("H 1 1 ALA 0 1 2 1", ""), domain) == ("AG", 1)
    assert tnp._extract(CIF.replace("0 1 2 1", "? 1 2 1"), domain) == ("AG", 1)
    assert tnp._extract(CIF.replace("0 1 2 1", "0 1 2 0"), domain) == ("AG", 1)
    with pytest.raises(tnp.ProfileError, match="selected_chain_coordinates_absent"):
        tnp._extract(CIF.replace("0 1 2 1", ". . . 1").replace("3 4 5 1", "3 4 5 0"), domain)


def test_generation_writer_can_declare_an_entity_in_atom_site():
    pytest.importorskip("gemmi")
    text = CIF.replace("_struct_asym.id H\n_struct_asym.entity_id 1\n", "")
    domain = tnp.Domain.parse(SETTINGS)
    assert tnp._extract(text, domain) == ("AG", 2)
    with pytest.raises(tnp.ProfileError, match="selected_label_chain_absent_or_ambiguous"):
        tnp._extract(text.replace("H 1 2 GLY", "H 2 2 GLY"), domain)


@pytest.mark.parametrize("rows", ["1 1 ALA\n1 1 GLY", "1 1 ALA", "1 1 ALA\n1 2 MSE"])
def test_duplicate_missing_or_modified_polymer_ordinals_block_modeling(rows):
    pytest.importorskip("gemmi")
    text = CIF + "loop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n" + rows + "\n"
    assert tnp.profile(text, SETTINGS)["blocker"] == "declared_polymer_sequence_mismatch"


@pytest.mark.parametrize(
    ("old", "new", "blocker"),
    [
        ("_struct_asym.id H", "_struct_asym.id B", "selected_label_chain_absent_or_ambiguous"),
        ("polypeptide(L)", "polypeptide(D)", "selected_chain_is_not_one_protein"),
        ("H 1 1 ALA", "H 2 1 ALA", "coordinate_sequence_mismatch"),
        ("H 1 2 GLY", "H 1 3 GLY", "coordinate_sequence_mismatch"),
        ("H 1 2 GLY", "H 1 2 MSE", "coordinate_sequence_mismatch"),
        ("letter_code AG", "letter_code A(MSE)", "noncanonical_domain_sequence"),
        ("H 1 1 ALA 0 1 2 1\nH 1 2 GLY", "B 1 2 GLY", "selected_chain_coordinates_absent"),
    ],
)
def test_cif_attribution_guards(old, new, blocker):
    pytest.importorskip("gemmi")
    record = tnp.profile(CIF.replace(old, new), SETTINGS)
    assert record["status"] == "untested" and record["blocker"] == blocker
    assert "measurements" not in record


@pytest.mark.parametrize(
    ("fields", "blocker"),
    [
        ({"domain_type": "VH"}, "single_vhh_declaration_required"),
        ({"sequence_sha256": "*"}, "invalid_sequence_digest"),
        ({"sequence_sha256": "0" * 64}, "declared_sequence_digest_mismatch"),
        ({"label_asym_id": "?"}, "invalid_label_asym_id"),
        ({"allow_trim": True}, "invalid_domain_settings"),
    ],
)
def test_declaration_and_digest(fields, blocker):
    pytest.importorskip("gemmi")
    assert tnp.profile(CIF, dict(SETTINGS, **fields))["blocker"] == blocker


def test_duplicate_manifest_retains_coverage(tmp_path: Path):
    assert tnp.RunProfiles.configured(tmp_path) is None
    (tmp_path / tnp.MANIFEST).write_text('{"samples":{"sample":{},"sample":{}}}')
    run = tnp.RunProfiles.configured(tmp_path)
    run.observe("sample", CIF)
    result = run.as_record()
    assert (result["attempted"], result["untested"]) == (1, 1)
    assert result["samples"][0]["blocker"] == "duplicate_json_key"


def test_counts_cover_missing_settings_and_unmatched_entries(tmp_path: Path):
    (tmp_path / tnp.MANIFEST).write_text(json.dumps({"samples": {"one": SETTINGS, "orphan": SETTINGS}}))
    run = tnp.RunProfiles.configured(tmp_path)
    run.observe("one", None)
    run.observe("two", CIF)
    result = run.as_record()
    assert [r["blocker"] for r in result["samples"]] == ["structure_unreadable", "sample_settings_absent"]
    assert result["unmatched_manifest_ids"] == ["orphan"]
    assert (result["attempted"], result["profiled"], result["partial"], result["untested"]) == (2, 0, 0, 2)


def _report():
    return {
        "name": "domain",
        "Total CDR Length": 30,
        "CDR3 Length": 12,
        "CDR3 Compactness": 1.0,
        "PSH": 100.0,
        "PPC": 0.1,
        "PNC": 0.2,
        "Flags": dict.fromkeys(tnp_worker._FIELDS, "green"),
    }


def _green(_metric, _value):
    return "green"


@pytest.mark.parametrize(
    ("field", "bad", "blocker"),
    [
        ("CDR3 Length", 12.0, "invalid_native_length"),
        ("CDR3 Length", 11, "native_cdr_length_mismatch"),
        ("PPC", False, "invalid_native_measurement"),
        ("PSH", float("inf"), "invalid_native_measurement"),
        ("PNC", -1, "invalid_native_measurement"),
        ("PNC", "0", "invalid_native_measurement"),
        ("CDR3 Compactness", 0, "invalid_native_measurement"),
        ("CDR3 Compactness", float("nan"), "native_flag_mismatch"),
        ("name", "other", "native_profile_absent"),
    ],
)
def test_worker_rejects_malformed_native_values(field, bad, blocker):
    row = _report()
    row[field] = bad
    with pytest.raises(tnp.ProfileError, match=blocker):
        tnp_worker._measurements(row, (30, 12), _green)


def test_only_unavailable_compactness_can_yield_partial():
    row = _report()
    row["CDR3 Compactness"] = float("nan")
    row["Flags"]["C"] = "grey"
    values, missing = tnp_worker._measurements(row, (30, 12), _green)
    assert missing == ["C"] and values["C"]["value"] is None and len(values) == 6
    json.dumps(values, allow_nan=False)
    row["PPC"] = None
    with pytest.raises(tnp.ProfileError, match="invalid_native_measurement"):
        tnp_worker._measurements(row, (30, 12), _green)


def test_dssp_failure_blocks_a_syntactically_successful_report(tmp_path: Path):
    with pytest.raises(tnp.ProfileError, match="native_model_annotation_incomplete"):
        tnp_worker._check_details(tmp_path / "absent", "AG", [(1, " "), (2, " ")])
