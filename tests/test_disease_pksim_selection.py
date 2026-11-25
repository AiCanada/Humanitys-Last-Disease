"""The same prepared PK-Sim screen is available in finished-run, candidate and standalone workflows."""

import json

import pytest

from rfproteina.disease import mcp_api, pksim, pksim_runtime, screen, screen_selection
from rfproteina.disease.validation_checks import candidate_validation_plan
from scripts import screen_extras


def test_finished_run_and_candidate_execution(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    samples = run / "inference_samples"
    samples.mkdir(parents=True)
    for name in ("candidate_a", "candidate_b"):
        (samples / f"{name}_generated.cif").write_text("data_example\n")
    (run / "generation.json").write_text('{"binder_kind":"ligand"}')
    (run / pksim.MANIFEST).write_text("{}")
    (run / "toxicology.tsv").write_text("axis\tnm\ncyp2d6\t10\n")
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    monkeypatch.setattr(pksim_runtime, "installed", lambda: True)
    monkeypatch.setattr(screen_selection, "_modules", lambda *names: names == ("rdkit",))
    calls = []

    def readiness(directory, candidate):
        assert directory == run
        if candidate in (None, "candidate_a"):
            return True, True, "Prepared human exposure model."
        return False, False, "This candidate has no PK-Sim scenario."

    def measurement(directory, candidate=None):
        assert directory == run
        calls.append(candidate)
        return {
            "method": "pksim", "scope": "run" if candidate is None else "candidate",
            "status": "simulated", "simulated": 1, "attempted": 1,
            "samples": [{
                "candidate_id": "candidate_a", "status": "simulated",
                "compound_inchikey": "IGTYILLPRJOVFY-UHFFFAOYSA-N",
                "model_sha256": "a" * 64, "export_sha256": "b" * 64,
            }],
        }

    monkeypatch.setattr(pksim, "readiness", readiness)
    monkeypatch.setattr(pksim, "measure", measurement)
    available = screen_selection.installations({"pksim"})
    assert available == {"pksim": True}
    choice, = screen_selection.plan(run, "ligand", available=available)
    assert choice.recommended
    with monkeypatch.context() as missing:
        missing.setattr(screen_selection, "_modules", lambda *_names: False)
        assert screen_selection.installations({"pksim"}) == {"pksim": False}
    wrong_kind, = screen_selection.plan(run, "protein", available=available)
    assert not wrong_kind.applicable
    assert calls == []

    baseline = screen.screen_generation(run, binder_kind="ligand", target_id="", screen_selection="explicit")
    selected = screen.screen_generation(run, binder_kind="ligand", target_id="", screen_methods=["pksim"])
    assert calls == [None]
    for key in ("call", "hits", "states", "cuts", "toxicology", "ligand_microsomal_clearance"):
        assert selected.get(key) == baseline.get(key)
    assert selected["toxicology"]["cyp2d6"]["measurement"] == 10
    assert selected["toxicology"]["cyp2d6"]["units"] == "nM"
    saved = json.loads((run / "screen.json").read_text())["method_selection"]["prepared_measurements"]["pksim"]
    assert saved["scope"] == "run" and saved["status"] == "simulated"
    # The first nine characters in this real acetaminophen sulfate InChIKey
    # trigger the generic sequence redactor. A verified chemical ID must survive.
    assert "[sequence omitted]" in mcp_api._redact(saved["samples"][0]["compound_inchikey"])

    candidates, _ = candidate_validation_plan((run,), None)
    assert {c.example_id for c in candidates} == {"candidate_a", "candidate_b"}
    for candidate in candidates:
        record, = mcp_api.call_tool(
            "run_selected_screens", {"directory": str(run), "candidate_id": candidate.key, "methods": ["pksim"]}
        )["results"]
        if candidate.example_id == "candidate_a":
            assert record["status"] == "simulated" and record["measurement"]["scope"] == "candidate"
            assert record["measurement"]["samples"] == saved["samples"]
        else:
            assert record["status"] == "inapplicable"
    assert calls == [None, "candidate_a"]

    capsys.readouterr()
    assert screen_extras.main(["pksim", str(run), "--candidate", "candidate_a"]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == saved["samples"]
    (run / pksim.MANIFEST).unlink()
    screen.screen_generation(run, binder_kind="ligand", target_id="", screen_methods=["pksim"])
    selection = json.loads((run / "screen.json").read_text())["method_selection"]
    assert "pksim" in selection["skipped"] and "pksim" not in selection["prepared_measurements"]
    assert calls == [None, "candidate_a", "candidate_a"]


@pytest.mark.parametrize("status", ["partial", "untested"])
def test_standalone_returns_unsuccessful_status(status, tmp_path, monkeypatch, capsys):
    def incomplete(directory, candidate):
        assert directory == tmp_path and candidate is None
        return {"method": "pksim", "status": status}

    monkeypatch.setattr(pksim, "measure", incomplete)
    assert screen_extras.main(["pksim", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == status
