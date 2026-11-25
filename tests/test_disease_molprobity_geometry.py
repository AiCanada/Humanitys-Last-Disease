"""MolProbity result attribution, private worker execution, and native denominators."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rfproteina.disease import molprobity_geometry as mp
from rfproteina.disease import molprobity_worker as worker
from rfproteina.disease.molprobity_inputs import ModelInput


def _torsion(total: int = 10, outliers: int = 2) -> dict:
    return {"status": "reported", "evaluated": total, "outliers": outliers, "favored": total - outliers, "allowed": 0}


def test_native_counts_are_authoritative_and_true_zeros_survive() -> None:
    assert mp.validate_metric("ramachandran", {**_torsion(10, 0), "outlier_percent": 93})["outlier_percent"] == 0
    assert mp.validate_metric("rotamers", _torsion())["outlier_percent"] == 20
    result = mp.validate_metric("clashscore", {"status": "reported", "atoms": 500, "serious_clashes": 1})
    assert result["clashes_per_1000_atoms"] == 2
    for malformed in (_torsion(0, 0), {**_torsion(), "outliers": True}, {**_torsion(), "allowed": 7}):
        assert mp.validate_metric("ramachandran", malformed)["status"] == "untested"
    for atoms, clashes in ((0, 0), (False, 0), (8, 29), (20, -1), (100, 2.0)):
        result = mp.validate_metric("clashscore", {"status": "reported", "atoms": atoms, "serious_clashes": clashes})
        assert result == {"status": "untested", "blocker": "molprobity_metric_failed"}


def test_model_coverage_retains_hash_and_independent_missing_metric(monkeypatch, tmp_path):
    paths = [tmp_path / f"{name}_generated.cif" for name in ("first", "second")]
    records = [ModelInput("first", paths[0]), ModelInput("second", paths[1])]
    monkeypatch.setattr(mp.molprobity_inputs, "resolve_inputs", lambda *_: (records, None))
    monkeypatch.setattr(mp.molprobity_inputs, "read_structure", lambda path: path.stem.encode("utf-8"))
    monkeypatch.setattr(mp, "runtime_path", lambda: Path("isolated-interpreter"))

    def measure(_python, action, data, timeout):
        assert action == "measure" and 0 < timeout <= 120
        if data.startswith(b"second"):
            return {"blocker": "molprobity_timeout"}
        return {"provider": mp.PROVIDER, "metrics": {
            "ramachandran": _torsion(), "rotamers": _torsion(10, 0),
            "clashscore": {"status": "untested", "blocker": "molprobity_incomplete_heavy_atoms"},
        }}

    monkeypatch.setattr(mp, "_worker_call", measure)
    result = mp.screen_run(tmp_path)
    assert result["status"] == "partial" and result["model_scope"] == "whole_model"
    assert (result["attempted"], result["reported"], result["complete"]) == (2, 1, 0)
    first, second = result["samples"]
    assert first["input_sha256"] == hashlib.sha256(paths[0].stem.encode()).hexdigest()
    assert first["metrics"]["clashscore"]["blocker"] == "molprobity_incomplete_heavy_atoms"
    assert second["blocker"] == "molprobity_timeout"
    assert "first_generated" not in json.dumps(result) and "verdict" not in result


def test_worker_runs_in_private_cpu_scratch_and_rejects_garbage(monkeypatch):
    scratch = None

    def fake(command, **options):
        nonlocal scratch
        scratch = Path(options["cwd"])
        assert scratch.is_dir() and not tuple(scratch.iterdir())
        assert command[-3:] == ["-I", str(mp._WORKER), "measure"]
        assert options["input"] == b"cif snapshot" and options["timeout"] == 30
        assert options["env"]["CUDA_VISIBLE_DEVICES"] == options["env"]["HIP_VISIBLE_DEVICES"] == ""
        assert options["env"]["TMPDIR"] == str(scratch)
        return SimpleNamespace(stdout=b"not json")

    monkeypatch.setattr(mp.subprocess, "run", fake)
    assert mp._worker_call(Path("isolated-python"), "measure", b"cif snapshot", 30) == {
        "blocker": "molprobity_provider_failure",
    }
    assert scratch is not None and not scratch.exists()

    def timeout(*_args, **_options):
        raise subprocess.TimeoutExpired("local worker", timeout=1)

    monkeypatch.setattr(mp.subprocess, "run", timeout)
    assert mp._worker_call(Path("isolated-python"), "measure", b"cif snapshot", 1) == {"blocker": "molprobity_timeout"}


def _model(*, chain_id: str = "A", resname: str = "ALA", remove: str = "", x: float = 0, model_count: int = 1):
    names = [name for name in ("N", "CA", "C", "O", "CB") if name != remove]
    atoms = [SimpleNamespace(name=name, occ=1, b=0, element=name[0], xyz=(x, 0., 0.)) for name in names]
    group = SimpleNamespace(altloc="", resname=resname, atoms=lambda: atoms)
    residue = SimpleNamespace(resseq="   1", icode=" ", atom_groups=lambda: [group])
    chain = SimpleNamespace(id=chain_id, residue_groups=lambda: [residue])
    model = SimpleNamespace(chains=lambda: [chain])
    return SimpleNamespace(models=lambda: [model] * model_count, atoms=lambda: atoms)


def test_full_atom_and_native_identifier_checks_are_independent_of_torsions():
    assert worker._inspect(_model()) is None
    assert worker._inspect(_model(remove="CB")) == "molprobity_incomplete_heavy_atoms"
    assert worker._inspect(_model(chain_id="long")) == "molprobity_unsupported_model"
    with pytest.raises(worker.MolProbityWorkerError, match="molprobity_structure_invalid"):
        worker._inspect(_model(x=float("nan")))
    for excluded in (_model(resname="LIG"), _model(model_count=2)):
        with pytest.raises(worker.MolProbityWorkerError, match="molprobity_unsupported_model"):
            worker._inspect(excluded)
    assert worker._torsion(SimpleNamespace(n_total=0))["blocker"] == "molprobity_no_eligible_residues"
