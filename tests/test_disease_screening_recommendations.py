"""Selection policy for an intended exposure, measured evidence and bounded compute."""

import json

import pytest

from rfproteina.disease import screen_assays, screen_inventory, screen_selection, screening_context


def _plan(directory, methods, example=None):
    return {
        choice.method.key: choice for choice in screen_selection.plan(directory, "ligand", example=example, available=methods)
    }


def test_therapeutic_recommendations_follow_only_the_selected_molecules(tmp_path):
    table = tmp_path / "ligand_candidates.tsv"
    table.write_text("candidate_id\tsmiles\nA\tCCO\nB\tCCCO\n", encoding="utf-8")
    keys = dict.fromkeys(("ligand_dili", "ligand_caco2", "ligand_microsomal_clearance"), True)
    (tmp_path / screening_context.MANIFEST).write_text(json.dumps({"candidates": {
        "A": {"route": "oral", "systemic": True}, "B": {"route": "topical", "systemic": False},
    }}), encoding="utf-8")
    oral = _plan(tmp_path, keys, "A")
    assert all(row.recommended for row in oral.values())
    assert "Oral" in oral["ligand_caco2"].reason and "Systemic" in oral["ligand_dili"].reason
    for scope in ("B", None):
        rows = _plan(tmp_path, keys, scope)
        assert all(row.applicable and not row.recommended and "Non-systemic" in row.reason for row in rows.values())
        assert set(screen_selection.select(tuple(rows.values()), "all_installed")) == set(keys)
    unavailable = _plan(tmp_path, {"ligand_dili": False}, "A")["ligand_dili"]
    assert unavailable.applicable and not unavailable.installed and not unavailable.recommended
    table.write_text("candidate_id\tsmiles\nA\tCC[NH3+]\n", encoding="utf-8")
    assert all(not r.recommended and "charged form" in r.reason for r in _plan(tmp_path, keys, "A").values())


def test_systemic_cyp_hypothesis_and_conflicting_or_incomplete_scope(tmp_path):
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\nA\tCCO\nB\tCCC\n")
    (tmp_path / "cypstrate.json").write_text(json.dumps({"systemic": True, "candidates": {"A": ["3A4"]}}))
    methods = {"ligand_dili": True, "ligand_caco2": True}
    scoped = _plan(tmp_path, methods, "A")
    assert scoped["ligand_dili"].recommended and not scoped["ligand_caco2"].recommended
    assert not _plan(tmp_path, methods)["ligand_dili"].recommended
    assert not _plan(tmp_path, methods, "B")["ligand_dili"].recommended
    (tmp_path / screening_context.MANIFEST).write_text('{"candidates":{"A":{"systemic":false}}}')
    assert all(not r.recommended and "conflicting" in r.reason for r in _plan(tmp_path, methods, "A").values())


@pytest.mark.parametrize("content", ['{"candidates":{"A":{"systemic":"false"}}}',
                                     '{"candidates":{"A":{"route":"oral","route":"other"}}}', "[" * 1000])
def test_invalid_intent_preserves_manual_eligibility(tmp_path, content):
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\nA\tCCO\n")
    (tmp_path / screening_context.MANIFEST).write_text(content)
    row = _plan(tmp_path, {"ligand_dili": True}, "A")["ligand_dili"]
    assert row.installed and row.applicable and not row.recommended and screening_context.MANIFEST in row.reason


def test_saved_result_survives_missing_producer_in_all_plans(tmp_path, monkeypatch):
    metrics = tmp_path / "inference_samples"
    metrics.mkdir()
    (metrics / "binder_0_generated.cif").write_text("data_stub\n")
    (tmp_path / screen_inventory.ATTACHMENT).write_text(json.dumps({"dockq": [
        {"candidate_id": "binder_0", "endpoint": "DockQ", "unit": "score", "value": 0.7},
    ]}))
    monkeypatch.setattr(screen_selection, "installations", lambda keys=None: dict.fromkeys(keys or (), False))
    for mode, requested in (("explicit", ["dockq"]), ("all_installed", [])):
        choices, selected = screen_selection.execution_plan(tmp_path, "protein", mode, requested, example="binder_0")
        row = next(c for c in choices if c.method.key == "dockq")
        assert "dockq" in selected and row.applicable and row.installed and not row.recommended
        assert row.method.operation == "recorded_evidence" and row.producer_installed is False
        assert "local reader" in row.reason


def test_picker_never_refits_bar_to_render_a_checkbox(tmp_path, monkeypatch):
    for name in ("report.txt", "forward.dat", "reverse.dat"):
        (tmp_path / name).write_text("prepared input")
    (tmp_path / "pmx_inputs.json").write_text(json.dumps({
        "candidate_id": "A", "report": "report.txt", "forward": "forward.dat", "reverse": "reverse.dat",
    }))
    monkeypatch.setattr(screen_assays, "safe_measure", lambda *args: pytest.fail("BAR ran during planning"))
    row = _plan(tmp_path, {"pmx": True})["pmx"]
    assert row.applicable and not row.recommended and "BAR" in row.reason
