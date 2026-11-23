"""Regression cases for attributing a Vina score to the retained structure and preparation."""

import json
import os
import shutil
from pathlib import Path

import pytest

from rfproteina.disease import vina_pose as vp

CIF = """data_pose
loop_
_entity.id
_entity.type
1 polymer
2 non-polymer
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
_chem_comp.id ETO
_chem_comp.formula 'C H4 O'
loop_
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_comp_id
_atom_site.label_atom_id
_atom_site.type_symbol
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
A 1 GLY CA C 1 0 0 0 1
L 2 ETO C1 C . 4 0 0 1
L 2 ETO O1 O . 5 0 0 1
"""

VERSION = "AutoDock Vina v1.2.7\n"
SCORE = "Estimated Free Energy of Binding   : -12.513 (kcal/mol) [=(1)+(2)+(3)-(4)]"


def _atom(i: int, name: str, residue: str, x: float, typ: str) -> str:
    return f"ATOM  {i:5} {name:4} {residue:3} A   1    " f"{x:8.3f}{0:8.3f}{0:8.3f}{1:6.2f}{0:6.2f}    {0:6.3f} {typ}\n"


@pytest.fixture
def prepared(tmp_path: Path) -> vp.RunPoses:
    pytest.importorskip("gemmi")
    (tmp_path / "r.pdbqt").write_text(_atom(1, "CA", "GLY", 0, "C"))
    ligand = _atom(1, "C", "LIG", 4, "C") + _atom(2, "O", "LIG", 5, "OA")
    (tmp_path / "l.pdbqt").write_text("ROOT\n" + ligand + "ENDROOT\nTORSDOF 0\n")
    spec = {
        "receptor": "r.pdbqt",
        "ligand": "l.pdbqt",
        "center": [2, 0, 0],
        "size": [12, 12, 12],
        "receptor_chains": ["A"],
        "ligand_chain": "L",
        "ligand_component": "ETO",
    }
    (tmp_path / vp.MANIFEST).write_text(json.dumps({"samples": {"one": spec, "orphan": spec}}))
    return vp.RunPoses.read(tmp_path / vp.MANIFEST)


def test_recognises_only_one_versioned_score_in_kcal() -> None:
    assert vp._reported_score(VERSION + SCORE) == (-12.513, "1.2.7")
    for malformed in (SCORE + "\n" + SCORE, SCORE.replace("kcal", "kJ"), SCORE.replace("-12.513", "NaN"), ""):
        with pytest.raises(vp.PoseError, match="vina_score_unreadable"):
            vp._reported_score(VERSION + malformed)
    with pytest.raises(vp.PoseError, match="unsupported_vina_version"):
        vp._reported_score(VERSION.replace("1.2.7", "1.1.2") + SCORE)


def test_manifest_duplicate_keys_cannot_overwrite_a_partner(tmp_path: Path) -> None:
    path = tmp_path / vp.MANIFEST
    path.write_text('{"samples":{"same":{},"same":{}}}')
    row = vp.RunPoses.read(path).score("same", CIF)
    assert row["blocker"] == "duplicate_manifest_key" and "score_kcal_mol" not in row


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"receptor_chains": ["A", "A"]}, "invalid_partner_selection"),
        ({"receptor_chains": ["L"]}, "invalid_partner_selection"),
        ({"ligand": "../l.pdbqt"}, "invalid_prepared_path"),
        ({"receptor": "C:\\temp\\r.pdbqt"}, "invalid_prepared_path"),
        ({"size": [1e300, 10, 12]}, "scoring_grid_size_limit"),
        ({"center": [True, 0, 0]}, "invalid_scoring_box"),
        ({"center": [float("inf"), 0, 0]}, "invalid_scoring_box"),
        ({"size": [0, 3, 4]}, "invalid_scoring_box"),
        ({"extra": True}, "invalid_pose_settings"),
        ({"center": [100, 0, 0]}, "ligand_outside_scoring_box"),
        ({"ligand_component": "ATP"}, "selected_ligand_identity_mismatch"),
    ],
)
def test_invalid_preparations_withhold_the_score(prepared: vp.RunPoses, change: dict, reason: str) -> None:
    prepared.settings["one"].update(change)
    row = prepared.score("one", CIF)
    assert row["blocker"] == reason and "score_kcal_mol" not in row


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        ("5 0 0 1", "5.0006 0 0 1", "ligand_heavy_atom_mismatch"),
        ("C1 C . 4", "C1 C . 5", "ligand_heavy_atom_mismatch"),
        ("O1 O . 5", "O1 N . 5", "ligand_component_inventory_mismatch"),
        ("GLY CA", "ALA CA", "receptor_atom_identity_mismatch"),
        ("5 0 0 1", "5 0 0 0", "invalid_source_atom"),
        ("polypeptide(L)", "polydeoxyribonucleotide", "unsupported_receptor_residue"),
        ("C H4 O", "C2 H6 O", "ligand_component_inventory_mismatch"),
    ],
)
def test_source_drift_keeps_the_sample(prepared: vp.RunPoses, old: str, new: str, reason: str) -> None:
    prepared.observe("one", CIF.replace(old, new))
    prepared.observe("missing", CIF)
    prepared.observe("damaged", None)
    report = prepared.as_record()
    assert [r["blocker"] for r in report["samples"]] == [reason, "sample_settings_absent", "structure_unreadable"]
    assert (report["scored"], report["attempted"], report["untested"]) == (0, 3, 3)
    assert report["unmatched_manifest_ids"] == ["orphan"]


def test_partial_ligand_in_both_files_is_caught_by_component_inventory(prepared: vp.RunPoses) -> None:
    ligand = prepared.directory / "l.pdbqt"
    ligand.write_text("ROOT\n" + _atom(1, "C", "LIG", 4, "C") + "ENDROOT\nTORSDOF 0\n")
    text = CIF.replace("L 2 ETO O1 O . 5 0 0 1\n", "")
    assert prepared.score("one", text)["blocker"] == "ligand_component_inventory_mismatch"


def test_external_covalent_links_require_a_different_model(prepared: vp.RunPoses) -> None:
    conn = (
        "\n_struct_conn.conn_type_id covale\n_struct_conn.ptnr1_label_asym_id A\n_struct_conn.ptnr2_label_asym_id L\n"
    )
    assert prepared.score("one", CIF + conn)["blocker"] == "covalent_or_metal_linked_ligand"


def test_same_element_can_have_the_wrong_aromatic_type(prepared: vp.RunPoses) -> None:
    definition = (
        "\n_chem_comp_bond.comp_id ETO\n_chem_comp_bond.atom_id_1 C1\n_chem_comp_bond.atom_id_2 O1\n"
        "_chem_comp_bond.pdbx_aromatic_flag N\n"
    )
    ligand = prepared.directory / "l.pdbqt"
    ligand.write_text(ligand.read_text().replace("0.000 C\n", "0.000 A\n"))
    assert prepared.score("one", CIF + definition)["blocker"] == "ligand_aromatic_carbon_type_mismatch"


def test_torsion_endpoints_must_be_bonded_in_the_component() -> None:
    bonds = vp.LigandBonds(pairs={frozenset(("O1", "C1")), frozenset(("C1", "C2"))})
    names = {1: "O1", 2: "C1", 3: "C2"}
    assert bonds.verify(vp.Preparation([], [(1, 2)]), names)["heavy_torsion_bonds_checked"] == 1
    with pytest.raises(vp.PoseError, match="ligand_torsion_bond_mismatch"):
        bonds.verify(vp.Preparation([], [(1, 3)]), names)


def test_absent_executable_is_explicit(prepared: vp.RunPoses, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTODOCK_VINA_EXEC", str(prepared.directory / "missing-vina"))
    assert prepared.score("one", CIF)["blocker"] == "vina_unavailable"


def test_prepared_symlink_must_resolve_inside_the_run(prepared: vp.RunPoses, tmp_path: Path) -> None:
    dest = tmp_path / "subrun"
    dest.mkdir()
    link = dest / "ligand"
    try:
        link.symlink_to(prepared.directory / "l.pdbqt")
    except OSError:
        pytest.skip("the filesystem does not support creating test symlinks")
    with pytest.raises(vp.PoseError, match="invalid_prepared_path"):
        vp._local_file(dest, "ligand")


def test_native_fixed_pose_when_installed(prepared: vp.RunPoses) -> None:
    name = "vina.exe" if os.name == "nt" else "vina"
    if not (os.environ.get("AUTODOCK_VINA_EXEC") or vp.tool_code_path("autodock-vina", name) or shutil.which(name)):
        pytest.skip("optional AutoDock Vina executable is absent")
    row = prepared.score("one", CIF.replace("5 0 0 1", "5.00049 0 0 1"))
    assert row["status"] == "scored", row
    assert row["mode"] == "score_only" and row["provider_version"] == "1.2.7"
    assert row["ligand_inventory_basis"] == "component_formula_heavy_elements"
    assert row["ligand_match"]["max_axis_delta_angstrom"] == pytest.approx(0.00049)
    assert prepared.score("one", CIF)["score_kcal_mol"] == row["score_kcal_mol"]
