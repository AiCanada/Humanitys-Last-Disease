"""Exercise selection with table-only candidates and the evidence snapshot seen by the picker."""

import json

import pytest

from rfproteina.disease import (
    badapple2,
    fastsolv,
    jazzy,
    mcp_api,
    metfrag,
    screen,
    screen_inventory,
    screen_panel,
    screen_selection,
)
from rfproteina.disease.validation_checks import candidate_validation_plan


@pytest.mark.parametrize(("provider", "scope", "rows"), [
    (badapple2, "input_scope", "molecules"),
    (metfrag, "scope", "candidates"),
    (fastsolv, "scope", "candidates"),
    (jazzy, "scope", "candidates"),
])
def test_saved_chemical_results_preserve_the_identity_of_a_plain_smiles_file(tmp_path, provider, scope, rows):
    pytest.importorskip("rdkit")
    if provider is fastsolv:
        (tmp_path / fastsolv.MANIFEST).write_text(json.dumps({
            "version": 1, "conditions": [{"solvent": "ethanol", "temperature_k": 298}],
        }))
    elif provider is metfrag:
        (tmp_path / metfrag.MANIFEST).write_text(json.dumps({
            "version": 1, "peaks": "peaks.tsv", "adduct": "[M+H]+", "precursor_mz": 100,
            "precursor_ppm": 5, "fragment_ppm": 5, "fragment_da": 0.001,
        }))
        (tmp_path / "peaks.tsv").write_text("mz\tintensity\n30\t10\n40\t20\n50\t30\n")
    (tmp_path / "ligand_smiles.txt").write_text("[Na+]\n")
    original = provider.screen_run(tmp_path)
    assert original[scope] == "run_ligand" and original[rows][0]["candidate_id"] == "run_ligand"
    assert provider.report_view(original) == original
    renamed = {**original[rows][0], "candidate_id": "another_sample"}
    for foreign in ([renamed], [original[rows][0], renamed]):
        rejected = provider.report_view({**original, rows: foreign})
        assert rejected["blocker"] == f"{provider.METHOD_ID}_invalid_report" and rejected[rows] == []


def test_named_ligand_can_be_selected_without_coordinates(tmp_path, monkeypatch):
    (tmp_path / "generation.json").write_text('{"binder_kind":"ligand"}')
    table = tmp_path / "ligand_candidates.tsv"
    table.write_text("candidate_id\tsmiles\nfirst\tCCO\nsecond\tCO\ndupe\tO\ndupe\tCC\ninvalid\n")
    (tmp_path / screen_inventory.ATTACHMENT).write_text(json.dumps({"pmx": [
        {"candidate_id": name, "endpoint": "free energy", "unit": "kJ/mol", "value": value}
        for name, value in (("first", 1.0), ("second", 2.0))
    ]}))
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    monkeypatch.setattr(screen_selection, "installations", lambda keys=None: {
        k: True for k in ("pmx", "ligand_composition") if keys is None or k in keys
    })
    directory = {"directory": str(tmp_path)}
    assert not candidate_validation_plan((tmp_path,), None)[0]
    with pytest.raises(ValueError, match="finished run"):
        mcp_api.plan_validation(directory)
    plan = mcp_api.call_tool("plan_screening", directory)
    assert [r["label"] for r in plan["candidates"]] == ["first", "second"]
    key = plan["candidates"][1]["id"]
    scoped = {**directory, "candidate_id": key}
    choices = {r["id"]: r for r in mcp_api.call_tool("plan_screening", scoped)["methods"]}
    assert choices["pmx"]["applicable"]
    assert choices["pmx"]["performance"]["summary"]
    assert not choices["ligand_composition"]["applicable"]
    assert mcp_api.call_tool("plan_screening", {**directory, "candidate_offset": 1})["candidates"] == [
        {"id": key, "label": "second"}
    ]
    result = mcp_api.call_tool("run_selected_screens", {**scoped, "methods": ["pmx"]})["results"][0]
    assert choices["pmx"]["reason"]
    assert result["measurement"]["observations"] == [
        {"candidate_id": "second", "value": 2.0, "unit": "kJ/mol", "endpoint": "free energy",
         "source": screen_inventory.ATTACHMENT}
    ]
    table.write_text("candidate_id\tsmiles\nfirst\tCCO\nsecond\tO\n")
    with pytest.raises(ValueError, match="current screening plan"):
        mcp_api.call_tool("run_selected_screens", {**scoped, "methods": ["pmx"]})


def test_structural_and_nominated_ids_keep_separate_scopes(tmp_path):
    (tmp_path / "generation.json").write_text('{"binder_kind":"ligand"}')
    samples = tmp_path / "inference_samples"
    samples.mkdir()
    (samples / "binder_0_generated.cif").write_text("data_sample\n")
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\nbinder_0\tCCO\ncompound_2\tCO\n")
    structures, _checks = candidate_validation_plan((tmp_path,), None)
    first, second = screen_panel.candidates_for(tmp_path, structures=structures)
    assert first.key == structures[0].key and first.structure == structures[0]
    assert second.example_id == "compound_2" and second.structure is None


def test_mcp_compact_results_do_not_read_performance_and_reject_non_boolean_flag(tmp_path, monkeypatch):
    (tmp_path / "generation.json").write_text('{"binder_kind":"ligand"}')
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\na\tCCO\n")
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    monkeypatch.setattr(screen_selection, "installations", lambda keys: dict.fromkeys(keys, True))
    monkeypatch.setattr(screen_selection.Choice, "performance", property(lambda _: pytest.fail("unexpected probe")))
    args = {"directory": str(tmp_path), "methods": ["molprobity_geometry"]}
    for options in ({}, {"include_performance": False}):
        row, = mcp_api.call_tool("run_selected_screens", {**args, **options})["results"]
        assert set(row) == {"id", "status", "reason"} and row["status"] == "inapplicable"
    for value in (None, 0, 1, "true", []):
        with pytest.raises(ValueError, match="include_performance"):
            mcp_api.call_tool("run_selected_screens", {**args, "include_performance": value})


def test_nominated_input_and_attached_result_choose_the_available_operation(tmp_path, monkeypatch):
    (tmp_path / "generation.json").write_text('{"binder_kind":"ligand"}')
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\na\tCCO\nb\tCO\n")
    (tmp_path / screen_inventory.ATTACHMENT).write_text(json.dumps({"bayesherg": [
        {"candidate_id": "a", "endpoint": "saved estimate", "value": 0.72, "unit": "probability"}
    ]}))
    checked = []

    def installations(keys):
        checked.append(set(keys))
        return dict.fromkeys(keys, False)

    monkeypatch.setattr(screen_selection, "installations", installations)
    for mode, requested in (("explicit", ["bayesherg"]), ("all_installed", [])):
        choices, selected = screen_selection.execution_plan(tmp_path, "ligand", mode, requested, example="a")
        saved = next(c for c in choices if c.method.key == "bayesherg")
        assert "bayesherg" in selected and saved.method.operation == "recorded_evidence"
        assert saved.installed and not saved.producer_installed and "saved observation" in saved.reason
    assert checked[0] == {"bayesherg"}
    wrong, = screen_selection.plan(tmp_path, "ligand", example="b", available={"bayesherg": False})
    assert not wrong.installed and wrong.method.operation == "compute"
    live, = screen_selection.plan(tmp_path, "ligand", example="a", available={"bayesherg": True})
    assert live.method.operation == "compute" and live.applicable
    candidate = screen_panel.candidates_for(tmp_path)[0]
    result, = screen_panel.run(tmp_path, candidate.key, "explicit", ["bayesherg"])
    assert result["status"] == "recorded" and result["measurement"]["observations"][0]["value"] == 0.72
    (tmp_path / "ligand_candidates.tsv").unlink()
    (tmp_path / "ligand_smiles.txt").write_text("CO\n")
    unverified, = screen_selection.plan(tmp_path, "ligand", available={"bayesherg": False})
    assert unverified.method.operation == "compute" and not unverified.installed


def test_recommendation_considers_saved_pose_checks_when_fresh_inputs_cannot_run(tmp_path, monkeypatch):
    from rfproteina.utils.pose_checks import POSEBUSTERS_DOCK_CHECKS

    samples = tmp_path / "inference_samples"
    samples.mkdir()
    (samples / "binder_0_generated.cif").write_text("data_sample\n")
    (tmp_path / "ligand_candidates.sdf").write_text("unreadable fresh input\n")
    (tmp_path / screen_inventory.ATTACHMENT).write_text(json.dumps({"posebusters": [
        {"candidate_id": "binder_0", "endpoint": POSEBUSTERS_DOCK_CHECKS[0], "value": 1, "unit": "boolean"}
    ]}))
    checked = []

    def installations(keys):
        checked.append(set(keys))
        return dict.fromkeys(keys, False)

    monkeypatch.setattr(screen_selection, "installations", installations)
    choices, selected = screen_selection.execution_plan(tmp_path, "ligand", "recommended", [], example="binder_0")
    saved = next(c for c in choices if c.method.key == "posebusters")
    assert saved.recommended and saved.method.operation == "recorded_evidence" and "posebusters" in selected
    assert "posebusters" in checked[0] and "bayesherg" not in checked[0]
    live, = screen_selection.plan(tmp_path, "ligand", example="binder_0", available={"posebusters": True})
    assert live.method.operation == "compute" and not live.recommended


def test_cli_records_why_a_saved_measurement_was_selected(tmp_path, monkeypatch):
    samples = tmp_path / "inference_samples"
    samples.mkdir()
    (samples / "binder_0_generated.cif").write_text("data_sample\n")
    (tmp_path / "screen.json").write_text(json.dumps({"samples": [{"example_id": "binder_0", "ca_rmsd": 1.75}]}))
    monkeypatch.setattr(screen, "_sample_measurements", lambda *a, **kw: ([], [], []))
    monkeypatch.setattr(screen_selection, "installations", lambda keys: dict.fromkeys(keys, True))
    planned, = screen_selection.plan(tmp_path, "protein", available={"ca_rmsd": True})
    record = screen.screen_generation(tmp_path, binder_kind="protein", target_id="", screen_methods=["ca_rmsd"])
    selection = record["method_selection"]
    assert selection["selected"] == selection["attempted"] == ["ca_rmsd"]
    assert selection["choices"][0]["operation"] == "recorded_evidence"
    assert selection["choices"][0]["reason"] and selection["choices"][0]["workload"]
    assert selection["choices"] == [planned.as_record()]
    measurement = selection["recorded_evidence"]["ca_rmsd"]
    assert measurement["observations"] == [
        {"candidate_id": "binder_0", "endpoint": "ca_rmsd", "value": 1.75, "source": "screen.json"}
    ]
