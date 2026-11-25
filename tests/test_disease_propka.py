"""PROPKA pKa observations stay attributed to the nominated structure."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from rfproteina.disease import propka, propka_worker

_BACKBONE = ("N", "CA", "C", "O")
_SIDE_CHAINS = {
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "CYS": ("CB", "SG"),
}


def _pdb(*, residue_count=10, include_ligand=True, altloc=False):
    """A small artificial single-model structure for attribution tests."""
    atoms = []

    def atom(record, residue, chain, number, name, offset, alternate=" "):
        serial = len(atoms) + 1
        position = number if chain == "A" else 5
        x, y, z = 3.8 * (position - 1) + 0.7 * offset, float(offset % 2), float(number % 3)
        atoms.append(
            f"{record:6s}{serial:5d} {name:^4s}{alternate}{residue:3s} {chain}{number:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{18.0:6.2f}          {name[0]:>2s}\n"
        )

    for index in range(1, residue_count + 1):
        residue = {4: "ASP", 6: "ARG", 8: "CYS"}.get(index, "GLY")
        for offset, name in enumerate((*_BACKBONE, *_SIDE_CHAINS.get(residue, ()))):
            atom("ATOM", residue, "A", index, name, offset, "A" if altloc and index == 4 else " ")
    atom("ATOM", residue, "A", residue_count, "OXT", 8)
    atoms.append("TER   \n")
    if include_ligand:
        for offset, name in enumerate(("C1", "C2", "N1")):
            atom("HETATM", "ETA", "B", 201, name, offset)
    return ("".join(atoms) + "END\n").encode("ascii")


def _manifest(directory, **values):
    path = directory / propka.MANIFEST
    path.write_text(json.dumps({"version": 1, "structure": "input.pdb", **values}), encoding="utf-8")
    return path


def _group(**updates):
    return {
        "kind": "protein", "chain": "A", "residue_number": 4, "insertion_code": "",
        "residue_name": "ASP", "atom_name": "CG", "group_type": "COO",
        "pka": 5.07, "model_pka": 3.8, "coupled": False, **updates,
    }


def _ligand(**updates):
    ligand = _group(
        kind="ligand", chain="B", residue_number=201, residue_name="ETA", atom_name="N1", group_type="N31",
        pka=9.3, model_pka=10.0,
    )
    return {**ligand, **updates}


def _forbidden(*_args):
    pytest.fail("invalid or unattributed structure reached PROPKA")


@pytest.fixture
def prepared(tmp_path):
    (tmp_path / "input.pdb").write_bytes(_pdb())
    _manifest(tmp_path)
    return tmp_path


def test_predicted_and_model_pka_remain_distinct_by_structure_and_site(prepared, monkeypatch):
    _manifest(prepared, assay_ph=7.25)
    digest = hashlib.sha256(_pdb()).hexdigest()
    plan = propka.prepare(prepared)
    assert plan.sha256 == digest
    asp = _group(coupled=True)
    arg = _group(residue_number=6, residue_name="ARG", atom_name="CZ", group_type="ARG", pka=14.4, model_pka=12.5)

    def predict(nomination):
        assert nomination.sha256 == digest
        return {"groups": [_ligand(), arg, asp], "incomplete_input": False}

    monkeypatch.setattr(propka, "_predict", predict)
    result = propka.screen_run(prepared)
    assert result["status"] == "complete" and result["structure_sha256"] == digest
    assert result["assay_ph"] == 7.25
    groups = {(row["kind"], row["chain"], row["residue_number"]): row for row in result["groups"]}
    assert groups["protein", "A", 4] == asp
    assert groups["protein", "A", 6] == arg
    assert groups["ligand", "B", 201] == _ligand()
    assert {"verdict", "threshold", "populations"}.isdisjoint(result)


@pytest.mark.parametrize("warning", [False, True])
def test_warnings_or_unscored_groups_keep_reported_values_without_inventing_numbers(prepared, monkeypatch, warning):
    scored = _group()
    groups = [scored] if warning else [scored, _ligand(pka=None, model_pka=None)]
    monkeypatch.setattr(propka, "_predict", lambda _plan: {"groups": groups, "incomplete_input": warning})
    result = propka.screen_run(prepared)
    assert result["status"] == "partial" and scored in result["groups"]
    if not warning:
        (withheld,) = [row for row in result["groups"] if row["kind"] == "ligand"]
        assert withheld["pka"] is None and withheld["model_pka"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"chain": "Z"}, {"residue_number": 5}, {"residue_number": True}, {"insertion_code": "A"},
        {"residue_name": "GLU"}, {"atom_name": "OE1"}, {"kind": "ligand"},
        {"pka": False}, {"pka": "5.07"}, {"pka": float("nan")}, {"model_pka": float("inf")},
        {"pka": None}, {"model_pka": None}, {"coupled": 1},
    ],
)
def test_malformed_or_foreign_group_cannot_supply_a_pka(prepared, monkeypatch, changes):
    monkeypatch.setattr(propka, "_predict", lambda _plan: {
        "groups": [_group(**changes)], "incomplete_input": False,
    })
    result = propka.screen_run(prepared)
    assert result["status"] == "untested" and result["blocker"] and not result.get("groups")


@pytest.mark.parametrize("response", [None, [], {"groups": []}, {"groups": [_group()], "incomplete_input": 1}])
def test_malformed_worker_response_does_not_look_like_a_completed_screen(prepared, monkeypatch, response):
    monkeypatch.setattr(propka, "_predict", lambda _plan: response)
    result = propka.screen_run(prepared)
    assert result["status"] == "untested" and result["blocker"] and not result.get("groups")


def test_worker_pipe_uses_exact_pdb_bytes_and_rejects_duplicate_json_fields(prepared, monkeypatch):
    _manifest(prepared, assay_ph=7.2)
    monkeypatch.setattr(propka, "installed", lambda: True)

    def worker(argv, **kwargs):
        assert argv[1:3] == ["-I", "-B"] and argv[-1].endswith("propka_worker.py")
        assert kwargs["input"] == _pdb() and kwargs["timeout"] > 0
        return SimpleNamespace(stdout=b'{"groups":[],"groups":[],"incomplete_input":false}')

    monkeypatch.setattr(propka.subprocess, "run", worker)
    result = propka.screen_run(prepared)
    assert result["status"] == "untested" and result["blocker"] == "propka_invalid_report"
    assert result["assay_ph"] == 7.2 and result["structure_sha256"] == hashlib.sha256(_pdb()).hexdigest()


def test_duplicate_worker_identity_cannot_publish_two_scores_for_the_same_group(prepared, monkeypatch):
    monkeypatch.setattr(propka, "_predict", lambda _plan: {
        "groups": [_group(), _group(pka=6.3)], "incomplete_input": False,
    })
    result = propka.screen_run(prepared)
    assert result["status"] == "untested" and result["blocker"] and not result.get("groups")


@pytest.mark.parametrize(
    "bad_manifest",
    [
        None, b"{}", b'{"version":true,"structure":"input.pdb"}',
        b'{"version":1,"structure":"input.pdb","structure":"other.pdb"}',
        b'{"version":1,"structure":"../input.pdb"}', b'{"version":1,"structure":"C:/input.pdb"}',
        b'{"version":1,"structure":"input.cif"}', b'{"version":1,"structure":"missing.pdb"}',
        b'{"version":1,"structure":"input.pdb","assay_ph":true}',
        b'{"version":1,"structure":"input.pdb","assay_ph":NaN}',
        b'{"version":1,"structure":"input.pdb","assay_ph":14.1}',
        b'{"version":1,"structure":"input.pdb","threshold":7}',
    ],
)
def test_bad_or_missing_manifest_blocks_provider_access(prepared, monkeypatch, bad_manifest):
    manifest = prepared / propka.MANIFEST
    manifest.unlink()
    if bad_manifest is not None:
        manifest.write_bytes(bad_manifest)
    monkeypatch.setattr(propka, "_predict", _forbidden)
    result = propka.screen_run(prepared)
    assert result["status"] == "untested" and result["blocker"]


def test_candidate_scope_requires_exactly_the_name_in_the_manifest(prepared, monkeypatch):
    monkeypatch.setattr(propka, "_predict", _forbidden)
    assert propka.screen_run(prepared, candidate="target_a")["status"] == "untested"
    _manifest(prepared, candidate_id="target_a", assay_ph=0)
    assert propka.screen_run(prepared, candidate="target_b")["status"] == "untested"
    assert propka.screen_run(prepared, candidate="TARGET_A")["status"] == "untested"

    monkeypatch.setattr(propka, "_predict", lambda _plan: {"groups": [_group()], "incomplete_input": False})
    selected = propka.screen_run(prepared, candidate="target_a")
    assert selected["status"] == "complete" and selected["assay_ph"] == 0
    assert selected["structure_sha256"] == hashlib.sha256(_pdb()).hexdigest()


def test_short_ensemble_altloc_and_symlink_structures_do_not_reach_provider(prepared, monkeypatch):
    monkeypatch.setattr(propka, "_predict", _forbidden)
    path = prepared / "input.pdb"
    models = b"MODEL        1\n" + _pdb()[:-4] + b"ENDMDL\nMODEL        2\n" + _pdb()[:-4] + b"ENDMDL\nEND\n"
    for text in (_pdb(residue_count=9), _pdb(altloc=True), models):
        path.write_bytes(text)
        result = propka.screen_run(prepared)
        assert result["status"] == "untested" and result["blocker"]

    path.unlink()
    target = prepared / "other.pdb"
    target.write_bytes(_pdb())
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    result = propka.screen_run(prepared)
    assert result["status"] == "untested" and result["blocker"]


def test_unretained_ligand_cannot_supply_a_bound_ligand_group(prepared, monkeypatch):
    (prepared / "input.pdb").write_bytes(_pdb(include_ligand=False))
    monkeypatch.setattr(propka, "_predict", lambda _plan: {"groups": [_ligand()], "incomplete_input": False})
    result = propka.screen_run(prepared)
    assert result["status"] == "untested" and result["blocker"] and not result.get("groups")


def test_missing_terminal_atoms_leave_the_result_partial(prepared, monkeypatch):
    from rfproteina.disease import screen_selection

    _manifest(prepared, assay_ph=0)
    available = {propka.METHOD_ID: True}
    (complete,) = screen_selection.plan(prepared, "protein", available=available)
    assert complete.recommended and "assay pH is 0" in complete.reason
    data = b"".join(line for line in _pdb().splitlines(keepends=True) if line[12:16].strip() != b"OXT")
    (prepared / "input.pdb").write_bytes(data)
    (incomplete,) = screen_selection.plan(prepared, "protein", available=available)
    assert incomplete.applicable and not incomplete.recommended and "termini" in incomplete.reason
    monkeypatch.setattr(propka, "_predict", lambda _plan: {"groups": [_group()], "incomplete_input": False})
    assert propka.screen_run(prepared)["status"] == "partial"


def test_ambiguous_native_chain_boundary_is_rejected(prepared, monkeypatch):
    atoms = [line for line in _pdb(include_ligand=False).splitlines(keepends=True) if line.startswith(b"ATOM")]
    other_chain = [line[:21] + b"B" + line[22:] for line in atoms]
    first_chain = [line for line in atoms if line[12:16].strip() != b"OXT"]
    (prepared / "input.pdb").write_bytes(b"".join(first_chain + [b"TER\n"] + other_chain) + b"END\n")
    monkeypatch.setattr(propka, "_predict", _forbidden)
    assert propka.screen_run(prepared)["blocker"] == "propka_structure_out_of_scope"


def test_unoccupied_heavy_coordinates_do_not_produce_predictions(prepared, monkeypatch):
    data = _pdb().splitlines(keepends=True)
    data[0] = data[0][:54] + b"  0.00" + data[0][60:]
    (prepared / "input.pdb").write_bytes(b"".join(data))
    monkeypatch.setattr(propka, "_predict", _forbidden)
    assert propka.screen_run(prepared)["blocker"] == "propka_structure_invalid"


def test_disulfide_placeholder_is_not_reported_as_a_numeric_pka(prepared, monkeypatch):
    cysteine = _group(residue_number=8, residue_name="CYS", atom_name="SG", group_type="CYS", pka=99.99, model_pka=9.0)
    monkeypatch.setattr(propka, "_predict", lambda _plan: {"groups": [cysteine], "incomplete_input": False})
    result = propka.screen_run(prepared)
    assert result["status"] != "complete"
    assert all(row["pka"] is None for row in result.get("groups", ()))


@pytest.mark.parametrize("state", ["disulfide", "discarded_coupled_group", "missing_model_pka"])
def test_provider_projects_unscored_native_groups_without_pka_numbers(state):
    atom = SimpleNamespace(
        type="atom", chain_id="_", res_num=8, icode=" ", res_name="CYS", name="SG",
        cysteine_bridge=state == "disulfide",
    )
    group = SimpleNamespace(
        atom=atom, type="CYS", titratable=True, model_pka_set=state != "missing_model_pka",
        pka_value=99.99 if state == "disulfide" else 9.2, model_pka=9.0,
        covalently_coupled_groups=[], non_covalently_coupled_groups=[],
        coupled_titrating_group=object() if state == "discarded_coupled_group" else None,
    )
    value = propka_worker._row(group)
    assert value["kind"] == "protein" and value["chain"] == ""
    assert value["pka"] is value["model_pka"] is None
    assert value["coupled"] is (state == "discarded_coupled_group")


def test_saved_view_checks_provenance_and_distinguishes_incomplete_status(prepared, monkeypatch):
    monkeypatch.setattr(propka, "_predict", lambda _plan: {"groups": [_group()], "incomplete_input": False})
    report = propka.screen_run(prepared)
    assert propka.report_view({**report, "verdict": "passed"}) == report
    for damage in (
        {"parameter_sha256": "0" * 64},
        {"structure_sha256": "unverified"},
        {"incomplete_input": True},
        {"groups": [_group(pka=None, model_pka=3.8)]},
        {"groups": [_group(), _group(pka=6.0)]},
    ):
        projected = propka.report_view({**report, **damage})
        assert projected["status"] == "untested" and projected["blocker"] == "propka_invalid_report"
        assert projected["groups"] == []


def test_standalone_keeps_selected_candidate_and_pka_scale(prepared, monkeypatch, capsys):
    from scripts import screen_extras

    _manifest(prepared, candidate_id="target_a")
    monkeypatch.setattr(propka, "_predict", lambda _plan: {"groups": [_group()], "incomplete_input": False})
    assert screen_extras.main(["propka", str(prepared), "--candidate", "target_a"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == propka.screen_run(prepared, candidate="target_a")
    assert result["groups"][0]["pka"] == 5.07
