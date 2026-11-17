"""Tests for the Boltz-2 oracle adapter. No boltz import, no weights, no GPU."""

from __future__ import annotations

import json

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
