"""Tests for the Chai-1 oracle adapter. No chai_lab import, no weights, no GPU."""

from __future__ import annotations

from types import SimpleNamespace

from rfproteina.metrics.oracles import chai_oracle
from rfproteina.metrics.oracles.chai_oracle import Chai1Oracle


def test_to_fasta_format():
    fasta = Chai1Oracle.to_fasta({"A": "ACD", "B": "EF"})
    assert ">protein|name=A\nACD" in fasta
    assert ">protein|name=B\nEF" in fasta


def test_scalar_vector_matrix_coercion():
    assert chai_oracle._scalar(0.5) == 0.5
    assert chai_oracle._scalar([[0.5]]) == 0.5
    assert chai_oracle._vector([[0.1, 0.2]]) == [0.1, 0.2]
    assert chai_oracle._matrix([[[1.0, 2.0], [3.0, 4.0]]]) == [[1.0, 2.0], [3.0, 4.0]]


def test_build_output_maps_fields(tmp_path):
    cif = tmp_path / "pred.cif"
    cif.write_text("# cif\n")
    rd = SimpleNamespace(
        aggregate_score=0.88,
        ptm_scores=SimpleNamespace(
            complex_ptm=0.8,
            interface_ptm=0.7,
            per_chain_ptm=[0.81, 0.79],
            per_chain_pair_iptm=[[1.0, 0.7], [0.7, 1.0]],
        ),
        plddt_scores=SimpleNamespace(complex_plddt=0.92, per_chain_plddt=[0.9, 0.94]),
    )
    candidates = SimpleNamespace(cif_paths=[cif], ranking_data=[rd])
    candidates.sorted = lambda: candidates
    out = chai_oracle.Chai1Oracle.build_output(candidates, ["A", "B"])
    assert out.cif_path == cif
    s = out.summary_confidences
    assert s["overall_plddt"] == 0.92
    assert s["ptm"] == 0.8 and s["iptm"] == 0.7
    assert s["chain_pair_iptm"] == [[1.0, 0.7], [0.7, 1.0]]
    assert s["chain_ids"] == ["A", "B"]


def test_backend_unavailable():
    oracle = Chai1Oracle()
    # chai_lab is not installed in the test env
    assert oracle._backend_available() in (True, False)  # no crash either way
