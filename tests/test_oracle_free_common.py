"""Tests for the shared free-oracle base (sequence extraction + fold/persist flow)."""

from __future__ import annotations

import json
from pathlib import Path

from biotite.structure import Atom, array

from rfproteina.metrics.oracles.free_oracle_common import (
    OracleBackendUnavailableError,
    PredictOutput,
    SequenceRefoldOracle,
    extract_protein_sequences,
)
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput


def _protein_two_chains():
    atoms = []
    # chain A: ALA, GLY ; chain B: LEU ; chain L: a ligand-only residue (dropped)
    layout = [("A", 1, "ALA"), ("A", 2, "GLY"), ("B", 1, "LEU"), ("L", 1, "LIG")]
    for chain, res_id, res_name in layout:
        atoms.append(
            Atom([0.0, 0.0, 0.0], chain_id=chain, res_id=res_id, res_name=res_name, atom_name="CA", element="C")
        )
    return array(atoms)


def test_extract_protein_sequences_drops_ligand_chain():
    seqs = extract_protein_sequences(_protein_two_chains())
    assert seqs == {"A": "AG", "B": "L"}


def test_extract_protein_sequences_nonstandard_becomes_x():
    atoms = [
        Atom([0.0, 0.0, 0.0], chain_id="A", res_id=1, res_name="ALA", atom_name="CA", element="C"),
        Atom([0.0, 0.0, 0.0], chain_id="A", res_id=2, res_name="SEP", atom_name="CA", element="C"),
    ]
    seqs = extract_protein_sequences(array(atoms))
    assert seqs == {"A": "AX"}


class _FakeOracle(SequenceRefoldOracle):
    def __init__(self, available=True, **kwargs):
        super().__init__(oracle_name="fake", **kwargs)
        self._available = available

    def _backend_available(self):
        return self._available

    def _predict(self, sequences, work_dir):
        cif = Path(work_dir) / "pred.cif"
        cif.write_text("# fake cif\n")
        summary = {
            "overall_plddt": 0.9,  # -> 90.0 after scaling
            "ptm": 0.8,
            "iptm": 0.7,
            "chain_ids": list(sequences.keys()),
            "_extra": {"backend_note": "ok", "mean_err": 1.25},
        }
        return PredictOutput(cif_path=cif, summary_confidences=summary)


def test_fold_batch_writes_refold_and_confidences(tmp_path):
    oracle = _FakeOracle()
    job = OracleJobInput(atom_array=_protein_two_chains(), example_id="ex1", out_dir=tmp_path)
    n_ok = oracle.fold_batch([job])
    assert n_ok == 1
    assert oracle.already_refolded(tmp_path, "ex1")
    metrics = json.loads(oracle.confidence_path(tmp_path, "ex1").read_text())
    assert metrics["overall_plddt"] == 90.0
    assert metrics["overall_ptm"] == 80.0
    assert metrics["backend_note"] == "ok"  # _extra passthrough merged
    assert metrics["mean_err"] == 1.25


def test_fold_batch_counts_cached(tmp_path):
    oracle = _FakeOracle()
    job = OracleJobInput(atom_array=_protein_two_chains(), example_id="ex1", out_dir=tmp_path)
    assert oracle.fold_batch([job]) == 1
    # second call: cached, still counted
    assert oracle.fold_batch([job]) == 1


def test_fold_batch_raises_when_backend_absent(tmp_path):
    oracle = _FakeOracle(available=False)
    job = OracleJobInput(atom_array=_protein_two_chains(), example_id="ex1", out_dir=tmp_path)
    try:
        oracle.fold_batch([job])
    except OracleBackendUnavailableError as error:
        assert "not installed" in str(error)
    else:
        raise AssertionError("expected OracleBackendUnavailableError")
