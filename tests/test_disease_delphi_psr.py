"""DELPHI custody, scoring direction, and failure behavior without proprietary data."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rfproteina.disease import delphi_psr as psr
from rfproteina.disease.delphi_inputs import Pair


@pytest.fixture
def verified_bytes(tmp_path, monkeypatch):
    data = b"local-only checkpoint fixture"
    weights = tmp_path / "pinned.pt"
    weights.write_bytes(data)
    monkeypatch.setattr(psr, "CHECKPOINT_BYTES", len(data))
    monkeypatch.setattr(psr, "CHECKPOINT_SHA256", hashlib.sha256(data).hexdigest())
    return weights, data


def _provider():
    return {
        "device": "cpu", "release": psr.RELEASE, "checkpoint_sha256": psr.CHECKPOINT_SHA256,
        "torch": "2.14.1+cu132", "anarcii": "2.0.8",
    }


def test_verified_checkpoint_rejects_wrong_bytes_size_and_missing_path(verified_bytes):
    path, raw = verified_bytes
    assert psr.verified_checkpoint(path) == raw
    path.write_bytes(b"x" * len(raw))
    with pytest.raises(psr.DelPhiError, match="delphi_checkpoint_invalid"):
        psr.verified_checkpoint(path)
    path.write_bytes(raw + b"x")
    with pytest.raises(psr.DelPhiError, match="delphi_checkpoint_invalid"):
        psr.verified_checkpoint(path)
    with pytest.raises(psr.DelPhiError, match="delphi_checkpoint_missing"):
        psr.verified_checkpoint(path.with_suffix(".missing"))


def test_verified_checkpoint_has_no_symlink_fallback(verified_bytes):
    path, _ = verified_bytes
    link = path.with_suffix(".link")
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(psr.DelPhiError, match="delphi_checkpoint_invalid"):
        psr.verified_checkpoint(link)


def test_asset_discovery_requires_both_local_dependencies_and_verified_bytes(verified_bytes, monkeypatch):
    path, _ = verified_bytes
    monkeypatch.setattr(psr.importlib.util, "find_spec", lambda module: object() if module == "torch" else None)
    monkeypatch.setenv("RFD4_DELPHI_PSR_CHECKPOINT", str(path))
    assert not psr.asset_available()
    monkeypatch.setattr(psr.importlib.util, "find_spec", lambda module: object())
    assert psr.asset_available()
    path.write_bytes(b"not the pinned bytes")
    assert not psr.asset_available()


def test_installer_is_idempotent_and_never_replaces_other_bytes(verified_bytes, tmp_path):
    source, raw = verified_bytes
    script = Path(__file__).resolve().parents[1] / "scripts" / "install_delphi_psr.py"
    install = runpy.run_path(str(script))["install"]
    destination = tmp_path / "installed" / psr.CHECKPOINT_NAME
    assert install(source, destination)
    assert destination.read_bytes() == raw
    assert not install(source, destination)
    destination.write_bytes(b"different local file")
    with pytest.raises(psr.DelPhiError, match="delphi_checkpoint_invalid"):
        install(source, destination)
    assert destination.read_bytes() == b"different local file"


def test_pairs_keep_candidate_coverage_and_named_blockers(monkeypatch, tmp_path):
    vh = "V" * 100
    vl = "L" * 98
    (tmp_path / "delphi_pairs.tsv").write_text(
        f"candidate_id\tvh\tvl\nzero\t{vh}\t{vl}\none\t{vh}\t{vl}\n"
        f"bad\t{vh}\t\nswapped\t{vl}\t{vh}\n",
        encoding="utf-8",
    )

    def worker(pairs):
        values = {"zero": 0., "one": 1., "swapped": "delphi_chain_roles_invalid"}
        assert all(pair.candidate_id in values for pair in pairs)
        return {"values": {pair.candidate_id: values[pair.candidate_id] for pair in pairs}, "provider": _provider()}

    monkeypatch.setattr(psr, "_local_predictions", worker)
    report = psr.screen_run(tmp_path)
    assert report["status"] == "partial" and report["attempted"] == 4 and report["predicted"] == 2
    assert report["scope"] == "run" and report["provider"] == _provider()
    assert report["input_sha256"] == hashlib.sha256((tmp_path / "delphi_pairs.tsv").read_bytes()).hexdigest()
    rows = {row["candidate_id"]: row for row in report["pairs"]}
    assert rows["zero"][psr.VALUE_FIELD] == 0. and rows["one"][psr.VALUE_FIELD] == 1.
    assert rows["bad"]["blocker"] == "delphi_domains_required"
    assert rows["swapped"]["blocker"] == "delphi_chain_roles_invalid"
    assert vh not in json.dumps(report) and vl not in json.dumps(report)
    assert not {"call", "threshold", "verdict"} & report.keys()
    assert psr.screen_run(tmp_path, candidate="one")["pairs"] == [rows["one"]]
    assert psr.screen_run(tmp_path, candidate="absent")["blocker"] == "delphi_candidate_not_nominated"


@pytest.mark.parametrize("response", [
    {"values": {"one": 0.7, "unexpected": 0.9}, "provider": _provider()},
    {"values": {"one": 0.7}},
    {"values": {"one": True}, "provider": _provider()},
    {"values": {"one": float("nan")}, "provider": _provider()},
    {"values": {"one": 1.1}, "provider": _provider()},
    {"blocker": "delphi_worker_timeout"},
])
def test_malformed_timeout_or_unattributed_outputs_do_not_become_predictions(tmp_path, monkeypatch, response):
    (tmp_path / "delphi_pairs.tsv").write_bytes(b"candidate_id\tvh\tvl\none\tA\tQ\n")
    monkeypatch.setattr(psr, "_local_predictions", lambda _: response)
    report = psr.screen_run(tmp_path)
    row, = report["pairs"]
    assert report["status"] == "untested" and report["predicted"] == 0 and "provider" not in report
    assert psr.VALUE_FIELD not in row and row["blocker"] == response.get("blocker", "delphi_inference_failed")


def test_worker_uses_cpu_and_isolated_private_input(monkeypatch):
    def fake(cmd, **options):
        assert cmd[1:] == ["-c", psr._WORKER_CALL]
        assert json.loads(options["input"]) == [["one", "A", "Q"]]
        assert options["env"]["CUDA_VISIBLE_DEVICES"] == options["env"]["HIP_VISIBLE_DEVICES"] == ""
        assert options["env"]["HF_HUB_OFFLINE"] == "1" and options["timeout"] > 0
        return SimpleNamespace(stdout=" " * (psr._WORKER_LIMIT + 1))

    monkeypatch.setattr(psr.subprocess, "run", fake)
    assert psr._local_predictions([Pair("one", "A", "Q")]) == {"blocker": "delphi_inference_failed"}

    def timeout(*_args, **_options):
        raise subprocess.TimeoutExpired("local worker", timeout=1)

    monkeypatch.setattr(psr.subprocess, "run", timeout)
    assert psr._local_predictions([Pair("one", "A", "Q")]) == {"blocker": "delphi_worker_timeout"}


def test_worker_request_bounds_and_distinguishes_same_named_pairs():
    assert psr._worker_pairs([["pair", "AC", "WY"]]) == [Pair("pair", "AC", "WY")]
    assert not psr._worker_pairs([["p", "AC", "WY"], ["p", "WY", "AC"]])
    assert not psr._worker_pairs([["p", "A" * 136, "WY"]])
    assert not psr._worker_pairs([["p", "AX", "WY"]])
    assert not psr._worker_pairs([[f"p{i}", "A", "Y"] for i in range(65)])


def test_pooling_preserves_independent_chain_offsets_and_positive_class():
    torch = pytest.importorskip("torch")
    from rfproteina.disease import delphi_psr_model

    def logits(paired, cdr3):
        assert paired.shape == (1, 270, 20) and cdr3.shape == (1, 25, 20)
        assert paired[0, 0, psr.AMINO_ACIDS.index("A")] == 1
        assert paired[0, 135, psr.AMINO_ACIDS.index("Y")] == 1
        assert cdr3[0, 2, psr.AMINO_ACIDS.index("P")] == 1
        assert paired[0, 2:135].count_nonzero() == paired[0, 137:].count_nonzero() == 0
        return torch.tensor([[-1.0, 1.0]])

    score, = delphi_psr_model.predict(logits, [("AC", "YW", "ARP")])
    assert score == pytest.approx(0.880797, abs=1e-6)
