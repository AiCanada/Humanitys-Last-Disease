"""Tests for the Boltz-2 oracle adapter. No boltz import, no weights, no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfproteina.metrics.oracles import boltz_oracle
from rfproteina.metrics.oracles.boltz_oracle import Boltz2Oracle


def test_to_yaml_format():
    y = Boltz2Oracle.to_yaml({"A": "ACD", "B": "EF"})
    assert "version: 1" in y
    assert "- protein:" in y
    assert "id: A" in y
    assert 'sequence: "ACD"' in y


def _write_predictions(predictions_dir, stem, chain_key_a, chain_key_b):
    predictions_dir.mkdir(parents=True)
    (predictions_dir / f"{stem}_model_0.cif").write_text("# cif\n")
    conf = {
        "confidence_score": 0.84,
        "ptm": 0.85,
        "iptm": 0.82,
        "complex_plddt": 0.88,
        "chains_ptm": {chain_key_a: 0.86, chain_key_b: 0.84},
        "pair_chains_iptm": {
            chain_key_a: {chain_key_a: 1.0, chain_key_b: 0.82},
            chain_key_b: {chain_key_a: 0.82, chain_key_b: 1.0},
        },
    }
    (predictions_dir / f"confidence_{stem}_model_0.json").write_text(json.dumps(conf))


def test_build_output_with_chain_id_keys(tmp_path):
    pred = tmp_path / "predictions" / "design"
    _write_predictions(pred, "design", "A", "B")
    out = Boltz2Oracle.build_output(pred, "design", ["A", "B"])
    s = out.summary_confidences
    assert s["overall_plddt"] == 0.88
    assert s["ptm"] == 0.85 and s["iptm"] == 0.82
    assert s["chain_ptm"] == [0.86, 0.84]
    assert s["chain_pair_iptm"][0][1] == 0.82
    assert s["chain_pair_iptm"][0][0] is None  # diagonal


def test_build_output_with_index_keys(tmp_path):
    # Boltz may key per-chain dicts by positional index strings.
    pred = tmp_path / "predictions" / "design"
    _write_predictions(pred, "design", "0", "1")
    out = Boltz2Oracle.build_output(pred, "design", ["A", "B"])
    assert out.summary_confidences["chain_ptm"] == [0.86, 0.84]


def test_build_output_missing_cif(tmp_path):
    pred = tmp_path / "predictions" / "design"
    pred.mkdir(parents=True)
    try:
        Boltz2Oracle.build_output(pred, "design", ["A"])
    except FileNotFoundError as error:
        assert "no top-model CIF" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_parse_affinity_units(tmp_path):
    pred = tmp_path / "predictions" / "design"
    pred.mkdir(parents=True)
    (pred / "affinity_design.json").write_text(
        json.dumps({"affinity_pred_value": -1.5, "affinity_probability_binary": 0.9})
    )
    aff = boltz_oracle.parse_affinity(pred, "design")
    assert aff["affinity_log10_ic50_um"] == -1.5
    assert aff["affinity_binder_probability"] == 0.9


def test_parse_affinity_absent(tmp_path):
    pred = tmp_path / "predictions" / "design"
    pred.mkdir(parents=True)
    assert boltz_oracle.parse_affinity(pred, "design") is None


# ------------------------------------------------- the MSA declaration
#
# The shipped configuration could not return a refold: no `msa` key was written
# and `use_msa_server` is false, which Boltz rejects outright. These pin the
# declaration rather than the symptom, because the symptom was a warning line.


def test_the_default_input_declares_an_msa_source():
    """An omitted `msa` key means "generate one", not "do not use one"."""
    rendered = Boltz2Oracle.to_yaml({"A": "ACD"})
    assert "msa: empty" in rendered


def test_every_chain_carries_its_own_msa_declaration():
    """Boltz reads the key per protein entry, so one for the file will not do."""
    rendered = Boltz2Oracle.to_yaml({"A": "ACD", "B": "EF"})
    assert rendered.count("msa: empty") == 2


def test_opting_into_the_server_leaves_the_msa_key_out():
    """With --use_msa_server, Boltz builds the alignment and the key must be absent."""
    rendered = Boltz2Oracle.to_yaml({"A": "ACD"}, single_sequence=False)
    assert "msa:" not in rendered
    assert 'sequence: "ACD"' in rendered


def test_the_oracle_picks_the_declaration_from_its_own_setting(tmp_path, monkeypatch):
    """`_predict` must tie the YAML to `use_msa_server`, not to a constant."""
    seen: dict[str, str] = {}

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **_kwargs):
        seen["cmd"] = " ".join(cmd)
        seen["yaml"] = Path(cmd[2]).read_text(encoding="utf-8")
        raise RuntimeError("stop after the input is written")

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)

    offline_dir = tmp_path / "offline"
    offline_dir.mkdir()
    offline = Boltz2Oracle()
    with pytest.raises(RuntimeError):
        offline._predict({"A": "ACD"}, offline_dir)
    assert "msa: empty" in seen["yaml"]
    assert "--use_msa_server" not in seen["cmd"]

    online_dir = tmp_path / "online"
    online_dir.mkdir()
    online = Boltz2Oracle(use_msa_server=True)
    with pytest.raises(RuntimeError):
        online._predict({"A": "ACD"}, online_dir)
    assert "msa:" not in seen["yaml"]
    assert "--use_msa_server" in seen["cmd"]
