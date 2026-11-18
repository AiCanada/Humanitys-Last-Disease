"""Tests for the ThermoMPNN stability (ddG) screen. CPU-only, no weights, no network."""

from __future__ import annotations

import pytest

from rfproteina.disease import stability


def _fake_ssm_rows():
    # Two positions, a few mutations each. Negative ddG = stabilising (ThermoMPNN convention).
    return [
        {"position": 1, "wildtype": "A", "mutation": "A", "ddG_pred": 0.0},  # self, dropped
        {"position": 1, "wildtype": "A", "mutation": "V", "ddG_pred": -2.5},  # strongly stabilising
        {"position": 1, "wildtype": "A", "mutation": "G", "ddG_pred": 1.2},
        {"position": 2, "wildtype": "L", "mutation": "I", "ddG_pred": -0.3},
        {"position": 2, "wildtype": "L", "mutation": "P", "ddG_pred": 3.0},
    ]


def test_summarize_core_metrics():
    report = stability.summarize(_fake_ssm_rows())
    assert report.n_positions == 2
    assert report.n_mutations == 4  # the self-mutation is dropped
    assert report.most_stabilizing_ddg == -2.5
    assert report.most_destabilizing_ddg == 3.0
    assert report.n_strongly_stabilizing == 1  # only -2.5 <= -1.0
    # position 1 best is -2.5 (improvable), position 2 best is -0.3 (not <= -1.0)
    assert report.fraction_positions_improvable == 0.5


def test_summarize_empty():
    report = stability.summarize([])
    assert report.n_positions == 0
    assert report.most_stabilizing_ddg is None
    assert report.fraction_positions_improvable is None


def test_normalize_accepts_lowercase_ddg_key():
    row = {"position": 5, "wildtype": "k", "mutation": "r", "ddg": -1.1}
    m = stability._normalize(row)
    assert m.position == 5 and m.wildtype == "K" and m.mutation == "R" and m.ddg_pred == -1.1


def test_normalize_rejects_missing_ddg():
    with pytest.raises(ValueError, match="no ddG column"):
        stability._normalize({"position": 1, "wildtype": "A", "mutation": "V"})


def test_run_site_saturation_uses_injected_predict_fn(tmp_path):
    structure = tmp_path / "design.pdb"
    structure.write_text("ATOM\n")
    seen = {}

    def fake_predict(path, chain):
        seen["path"] = path
        seen["chain"] = chain
        return _fake_ssm_rows()

    rows = stability.run_site_saturation(structure, chain="A", predict_fn=fake_predict)
    assert seen["chain"] == "A"
    assert len(rows) == 5
    assert all(isinstance(r, stability.MutationDdg) for r in rows)


def test_run_site_saturation_missing_file():
    with pytest.raises(FileNotFoundError):
        stability.run_site_saturation("/no/such/file.pdb", predict_fn=lambda p, c: [])


def test_the_checkpoint_is_found_under_data_path(tmp_path, monkeypatch):
    monkeypatch.delenv("THERMOMPNN_CHECKPOINT", raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    checkpoint = tmp_path / "thermompnn" / "ThermoMPNN" / "models" / "thermoMPNN_default.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    assert stability.checkpoint_path() == checkpoint


def test_run_site_saturation_unavailable_raises_clearly(tmp_path, monkeypatch):
    structure = tmp_path / "design.pdb"
    structure.write_text("ATOM\n")
    monkeypatch.setattr(stability, "predictor_available", lambda: False)
    with pytest.raises(stability.ThermoMPNNUnavailableError, match="not installed"):
        stability.run_site_saturation(structure)


def test_verdicts_flag_sources_and_states():
    report = stability.summarize(_fake_ssm_rows())
    v = stability.verdicts(report)
    assert v["fraction_positions_improvable"]["state"] in {"clear", "hit"}
    # 0.5 > 0.25 max => hit
    assert v["fraction_positions_improvable"]["state"] == "hit"
    for rule in v.values():
        assert "pending calibration" in rule["threshold_source"] or "informational" in rule["threshold_source"]


def test_screen_structure_end_to_end(tmp_path):
    structure = tmp_path / "design.pdb"
    structure.write_text("ATOM\n")
    report = stability.screen_structure(structure, predict_fn=lambda p, c: _fake_ssm_rows())
    assert report.n_positions == 2
