"""Regressions for partner attribution and coordinate handling in the PRODIGY screen."""

import json

import pytest

from rfproteina.disease.corpus_provenance import BENCHMARK_LABEL_CORPUS, overlap_with
from rfproteina.disease.prodigy import METHOD, PARTNERS_FILE, AffinityInputError, Partners, RunAffinity

GROUPS = {"binder_chains": ["H", "L"], "target_chains": ["Ag"]}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"binder_chains": "H,L"}, "invalid_chain_group"),
        ({"target_chains": ["H"]}, "repeated_chain"),
        ({"target_chains": []}, "invalid_chain_group"),
        ({"target_chains": ["?"]}, "invalid_chain_group"),
        ({"binder_chains": ["H", "H"]}, "repeated_chain"),
        ({"temperature_c": True}, "invalid_temperature"),
        ({"temperature_c": float("nan")}, "invalid_temperature"),
        ({"temperature_c": -273.15}, "invalid_temperature"),
        ({"model_index": 1.5}, "invalid_model_index"),
        ({"model_index": False}, "invalid_model_index"),
        ({"chain_namespace": "auto"}, "invalid_chain_namespace"),
        ({"unknown": "ignored?"}, "invalid_partner_settings"),
    ],
)
def test_invalid_settings_do_not_reach_provider(change, reason):
    with pytest.raises(AffinityInputError, match=f"^{reason}$"):
        Partners.parse({**GROUPS, **change})


def test_request_and_measurement_identity(tmp_path):
    assert RunAffinity.configured(tmp_path) is None
    path = tmp_path / PARTNERS_FILE
    path.write_text(json.dumps(GROUPS), encoding="utf-8")
    configured = RunAffinity.configured(tmp_path)
    assert configured.partners.as_record()["binder_chains"] == ["H", "L"]
    configured.observe("refused", None)
    record = configured.as_record()
    assert record["samples"]["refused"] == {"method": METHOD, "status": "untested", "blocker": "structure_unreadable"}
    assert (record["attempted"], record["predicted"], record["untested"]) == (1, 0, 1)
    assert overlap_with(METHOD, "protein_protein_binding_delta_g") == BENCHMARK_LABEL_CORPUS
    path.write_text('{"binder_chains":["H"],"binder_chains":["L"],"target_chains":["Ag"]}', encoding="utf-8")
    refused = RunAffinity.configured(tmp_path)
    assert refused.blocker == "duplicate_partner_setting"
    assert refused.settings_sha256 != record["settings_sha256"]


def test_numbering_and_extra_chains_do_not_change_the_selected_model():
    pytest.importorskip("prodigy_prot")
    import numpy as np
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue

    from rfproteina.disease.prodigy import _prepare

    model = Model(0)
    for i, label in enumerate(["long_binder", "long_target", "bystander"]):
        chain = Chain(label)
        model.add(chain)
        for j, insertion in enumerate([" ", "A"]):
            residue = Residue((" ", 20, insertion), "GLY", " ")
            chain.add(residue)
            for k, name in enumerate(["N", "CA", "C", "O"]):
                coord = np.array([2.0 * i, float(j), float(k)])
                residue.add(Atom(name, coord, 15.0, 1.0, " ", name.rjust(4), k + 1, name[0]))
    request = Partners.parse({"binder_chains": ["long_binder"], "target_chains": ["long_target"]})
    pair, groups, counts = _prepare(model, request)
    assert counts["residues_by_chain"] == {"long_binder": 2, "long_target": 2}
    assert counts["insertion_coded_residues"] == 2
    assert len(pair) == len(groups) == 2
    assert len(set(groups)) == 2
    assert [atom.coord.tolist() for atom in pair.get_atoms()] == [
        atom.coord.tolist() for chain in [model["long_binder"], model["long_target"]] for atom in chain.get_atoms()
    ]
    assert model["long_binder"][(" ", 20, "A")].id[2] == "A"


def test_generated_cif_metadata_and_auth_chain_collisions():
    pytest.importorskip("Bio.PDB")
    from rfproteina.disease.prodigy import _structure

    # The generator omits crystallographic B factors. This CIF also omits model,
    # insertion and alternate-location columns and has real fractional occupancy.
    fields = (
        "id group_PDB type_symbol label_atom_id label_comp_id auth_asym_id "
        "label_asym_id auth_seq_id Cartn_x Cartn_y Cartn_z occupancy"
    ).split()
    cif = "data_control\nloop_\n" + "".join(f"_atom_site.{key}\n" for key in fields)
    cif += "1 ATOM C CA GLY H X 1 2 3 4 0.75\n2 ATOM C CA GLY L Y 2 3 3 4 1.0\n"
    config = {"binder_chains": ["H"], "target_chains": ["L"]}
    request = Partners.parse(config)
    model, index = _structure(cif, "cif", request)
    atom = model["H"][1]["CA"]
    assert (index, atom.coord.tolist(), atom.occupancy) == (1, [2.0, 3.0, 4.0], 0.75)
    with pytest.raises(AffinityInputError, match="structure_unreadable"):
        _structure(cif.replace("_atom_site.occupancy", "_atom_site.unrelated"), "cif", request)

    colliding = cif + "3 ATOM C CA GLY H Z 3 4 3 4 1.0\n"
    with pytest.raises(AffinityInputError, match="ambiguous_chain_namespace"):
        _structure(colliding, "cif", request)
    labelled = Partners.parse({"binder_chains": ["X"], "target_chains": ["Y"], "chain_namespace": "label"})
    model, _ = _structure(colliding, "cif", labelled)
    assert set(model.child_dict) == {"X", "Y", "Z"}


def test_refused_design_keeps_distinct_id_from_readable_design(tmp_path):
    from rfproteina.disease.screen import _sample_ids

    root = tmp_path / "inference_samples"
    root.mkdir()
    good = root / "one" / "binder_0_generated.cif"
    refused = root / "two" / "binder_0_generated.cif.zst"
    readable_ids = _sample_ids(tmp_path, [good])
    combined = _sample_ids(tmp_path, [good, refused], preserved={good: readable_ids[0]})
    assert combined == ["binder_0", "two_binder_0"]
