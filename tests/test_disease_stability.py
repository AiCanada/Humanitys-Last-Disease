"""Tests for the ThermoMPNN stability (ddG) screen. CPU-only, no weights, no network."""

from __future__ import annotations

from pathlib import Path

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


def _install_weights(data_path):
    """Both weight files, in upstream's layout, under a temporary DATA_PATH."""
    root = data_path / "thermompnn" / "ThermoMPNN"
    checkpoint = root / "models" / "thermoMPNN_default.pt"
    vanilla = root.joinpath(*stability.PROTEINMPNN_WEIGHTS)
    for path in (checkpoint, vanilla):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
    return root, checkpoint


def _install_source(tree):
    """A directory that counts as a ThermoMPNN source tree: it has the entry point."""
    entry = tree / "analysis" / "custom_inference.py"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("# upstream entry point\n", encoding="utf-8")
    return tree


def test_code_comes_from_the_checkout_and_weights_from_data_path(tmp_path, monkeypatch):
    # The point of the split: neither root is asked for the other's half.
    monkeypatch.delenv(stability.THERMOMPNN_CODE_ENV, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("SCREENING_CODE_ROOT", str(tmp_path / "subpackage"))
    vendored = _install_source(tmp_path / "subpackage" / "thermompnn" / "ThermoMPNN")
    weights, _ = _install_weights(tmp_path / "data")
    assert stability.code_root() == vendored
    assert stability.weights_root() == weights
    assert not (vendored / "models").exists()


def test_the_code_root_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("SCREENING_CODE_ROOT", str(tmp_path / "subpackage"))
    _install_source(tmp_path / "subpackage" / "thermompnn" / "ThermoMPNN")
    elsewhere = _install_source(tmp_path / "elsewhere")
    monkeypatch.setenv(stability.THERMOMPNN_CODE_ENV, str(elsewhere))
    assert stability.code_root() == elsewhere


def test_a_pre_vendoring_clone_under_data_path_still_runs(tmp_path, monkeypatch):
    # The layout before the source was vendored: code and weights in one tree.
    monkeypatch.delenv(stability.THERMOMPNN_CODE_ENV, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("SCREENING_CODE_ROOT", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    legacy = _install_source(tmp_path / "data" / "thermompnn" / "ThermoMPNN")
    assert stability.code_root() == legacy


def test_an_empty_directory_is_not_a_source_tree(tmp_path, monkeypatch):
    # A relocation leaves the old directory behind; it must not shadow a real one.
    monkeypatch.delenv(stability.THERMOMPNN_CODE_ENV, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("SCREENING_CODE_ROOT", str(tmp_path / "subpackage"))
    (tmp_path / "subpackage" / "thermompnn" / "ThermoMPNN").mkdir(parents=True)
    (tmp_path / "data" / "thermompnn" / "ThermoMPNN").mkdir(parents=True)
    assert stability.code_root() is None


def test_the_proteinmpnn_weights_are_required_too(tmp_path, monkeypatch):
    # Absent, it fails inside the model constructor rather than at the
    # checkpoint load, so it is checked up front and named.
    monkeypatch.delenv("THERMOMPNN_CHECKPOINT", raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    checkpoint = tmp_path / "thermompnn" / "ThermoMPNN" / "models" / "thermoMPNN_default.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    assert stability.checkpoint_path() == checkpoint
    assert stability.proteinmpnn_weights_path() is None
    assert stability.predictor_available() is False


def test_predictor_available_wants_both_weights_source_and_runner(tmp_path, monkeypatch):
    monkeypatch.delenv("THERMOMPNN_CHECKPOINT", raising=False)
    monkeypatch.delenv(stability.THERMOMPNN_CODE_ENV, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("SCREENING_CODE_ROOT", str(tmp_path / "subpackage"))
    _install_source(tmp_path / "subpackage" / "thermompnn" / "ThermoMPNN")
    _install_weights(tmp_path / "data")
    python = tmp_path / "data" / "thermompnn" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    assert stability.predictor_available() is True


def test_the_runner_is_called_with_both_roots(tmp_path, monkeypatch):
    """The command carries the two roots, and ``-I`` so this package cannot shadow upstream."""
    monkeypatch.delenv("THERMOMPNN_CHECKPOINT", raising=False)
    monkeypatch.delenv(stability.THERMOMPNN_CODE_ENV, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("SCREENING_CODE_ROOT", str(tmp_path / "subpackage"))
    vendored = _install_source(tmp_path / "subpackage" / "thermompnn" / "ThermoMPNN")
    weights, checkpoint = _install_weights(tmp_path / "data")
    python = tmp_path / "data" / "thermompnn" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    structure = tmp_path / "design.pdb"
    structure.write_text("ATOM\n")
    seen = {}

    class _Finished:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["timeout"] = kwargs.get("timeout")
        out_dir = Path(command[command.index("--out-dir") + 1])
        (out_dir / "ThermoMPNN_inference_design.csv").write_text(
            "position,wildtype,mutation,ddG_pred\n1,A,V,-2.5\n", encoding="utf-8"
        )
        return _Finished()

    monkeypatch.setattr(stability.subprocess, "run", fake_run)
    rows = stability.run_site_saturation(structure, chain="A")

    command = seen["command"]
    assert command[0] == str(python)
    assert command[1] == "-I"
    assert command[2].endswith("thermompnn_runner.py")
    assert command[command.index("--code") + 1] == str(vendored)
    assert command[command.index("--weights") + 1] == str(weights)
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert seen["timeout"] is not None  # an SSM scan must not wait forever
    assert [(r.position, r.mutation, r.ddg_pred) for r in rows] == [(1, "V", -2.5)]


def test_a_timeout_is_reported_as_unavailable(tmp_path, monkeypatch):
    structure = tmp_path / "design.pdb"
    structure.write_text("ATOM\n")
    monkeypatch.setattr(stability, "predictor_available", lambda: True)
    monkeypatch.setattr(stability, "checkpoint_path", lambda: tmp_path / "ckpt.pt")
    monkeypatch.setattr(stability, "code_root", lambda: tmp_path)
    monkeypatch.setattr(stability, "_inference_python", lambda: tmp_path / "python")

    def fake_run(command, **kwargs):
        raise stability.subprocess.TimeoutExpired(command, kwargs.get("timeout", 1))

    monkeypatch.setattr(stability.subprocess, "run", fake_run)
    with pytest.raises(stability.ThermoMPNNUnavailableError, match="TimeoutExpired"):
        stability.run_site_saturation(structure)


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
