import hashlib
import subprocess
import sys

import pytest

from rfproteina.disease import ligand_safety, octant, screen_assays, screen_criteria, screen_inventory


def _table(header):
    return header + "\n" + "\t".join("value" for _ in header.split("\t")) + "\n"


def test_measured_lookup_inventory_checks_the_actual_reader_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.setattr(screen_inventory, "_modules", lambda *modules: True)
    subset = tmp_path / "bindingdb" / "bindingdb_panel_subset.tsv"
    subset.parent.mkdir()
    keys = {"bindingdb_exact", "bindingdb_fingerprint"}
    required = "\t".join(ligand_safety._BINDINGDB_NEEDED)
    subset.write_text(required + "\n")
    assert not any(screen_inventory.installations(keys).values())
    subset.write_text(required + "\nrow\n")
    assert not any(screen_inventory.installations(keys).values())
    subset.write_text(_table(required))
    assert screen_inventory.installations(keys) == {"bindingdb_exact": True, "bindingdb_fingerprint": False}
    subset.write_text(_table(required + "\tsmiles"))
    assert all(screen_inventory.installations(keys).values())
    subset.write_text(_table(required + "\tsmiles\tsmiles"))
    assert not any(screen_inventory.installations(keys).values())


def test_octant_inventory_detects_an_unconverted_source(tmp_path, monkeypatch):
    subset = tmp_path / "corpus.tsv"
    monkeypatch.setattr(octant, "installed_subset", lambda: subset)
    monkeypatch.setattr(screen_inventory, "_modules", lambda *modules: True)
    required = "\t".join(octant.SUBSET_REQUIRED_COLUMNS)
    for extra, installed in (("", True), ("\tCYP3A4_pIC50", False)):
        subset.write_text(_table(required + extra))
        assert screen_inventory.installations({"octant"}) == {"octant": installed}


def test_cath_requires_configured_executable_and_payload(tmp_path, monkeypatch):
    db = tmp_path / "release.3"
    domains = tmp_path / "domains"
    domains.write_text("domains\n")
    (tmp_path / "release.3.dbtype").write_bytes(b"dbtype")
    monkeypatch.setenv("CATH_FOLDSEEK_DB", str(db))
    monkeypatch.setenv("CATH_DOMAIN_LIST", str(domains))
    monkeypatch.delenv("FOLDSEEK_EXEC", raising=False)
    assert not screen_inventory.installations({"cath_s95"})["cath_s95"]
    monkeypatch.setenv("FOLDSEEK_EXEC", sys.executable)
    assert not screen_inventory.installations({"cath_s95"})["cath_s95"]
    db.write_bytes(b"data")
    assert screen_inventory.installations({"cath_s95"})["cath_s95"]


def test_bayesherg_installer_rejects_substituted_weights_before_checking_worker(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    monkeypatch.setenv("BAYESHERG_ROOT", str(tmp_path))
    monkeypatch.setenv("BAYESHERG_PYTHON", sys.executable)
    (tmp_path / "model").mkdir()
    (tmp_path / "main.py").write_text("pass\n")
    (tmp_path / "model/BayeshERG_model.py").write_text("pass\n")
    (tmp_path / "model/model_weights.pth").write_bytes(b"x" * ligand_safety._BAYESHERG_WEIGHTS_BYTES)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("invalid weights launched worker"))
    assert not ligand_safety.bayesherg_installed()
    # When only the checkpoint is accepted, the substituted code must still fail discovery.
    monkeypatch.setattr(ligand_safety, "_BAYESHERG_WEIGHTS_SHA256",
                        hashlib.sha256(b"x" * ligand_safety._BAYESHERG_WEIGHTS_BYTES).hexdigest())
    assert not ligand_safety.bayesherg_installed()


def test_bayesherg_preserves_nominations_and_model_units(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    inputs = tmp_path / "ligand_candidates.tsv"
    inputs.write_text("candidate_id\tsmiles\nethanol\tCCO\nsame-graph\tOCC\nsingle\tC\n")
    calls = []

    def probability(smiles):
        calls.append(smiles)
        return ligand_safety.PanelProbabilities(
            smiles=smiles, model="BayeshERG", method=ligand_safety.BAYESHERG_METHOD,
            probabilities={"herg_prob": 0.2}, uncertainty_variances={"aleatoric": 0.1, "epistemic": 0.06},
            provenance={"dropout_samples": 30, "device": "cpu"},
        )

    monkeypatch.setattr(ligand_safety, "bayesherg_probability", probability)
    decision = screen_criteria.bayesherg_nominations(tmp_path, None)
    assert decision.applicable and not decision.recommended and "2 of 3" in decision.reason
    assert calls == []
    result = screen_assays.safe_measure(tmp_path, "bayesherg", None)
    assert calls == ["CCO"]
    assert result["status"] == "partial" and result["method"] == ligand_safety.BAYESHERG_METHOD
    assert result["unique_predicted"] == result["unique_compounds"] == 1
    first, duplicate, no_bonds = result["inputs"]
    assert first["candidate_id"] == "ethanol" and duplicate["candidate_id"] == "same-graph"
    (compound,) = result["compounds"]
    assert compound["compound_id"] == first["compound_id"] == duplicate["compound_id"]
    assert compound["unit"] == "probability" and compound["herg_prob"] == 0.2
    assert compound["uncertainty_variances"]["unit"] == "probability_squared"
    assert no_bonds["blocker"] == "bayesherg_requires_a_molecular_bond"
    assert not screen_criteria.bayesherg_nominations(tmp_path, "single").applicable
    assert screen_assays.safe_measure(tmp_path, "bayesherg", "absent")["status"] == "untested"
    assert calls == ["CCO"]
