"""DILI outputs stay attached to the human classification head and its nominations."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
from importlib.metadata import PathDistribution

import pandas as pd
import pytest

from rfproteina.disease import corpus_provenance, ligand_admet
from rfproteina.disease import ligand_dili as dili


@pytest.fixture
def local_release(tmp_path, monkeypatch):
    info = tmp_path / "admet_ai-2.0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: admet-ai\nVersion: 2.0.1\n", encoding="utf-8")
    root = tmp_path / "admet_ai/resources"
    ensemble = root / "models/admet_classification"
    ensemble.mkdir(parents=True)
    for index in range(5):
        (ensemble / f"model_{index}.pt").write_bytes(b"checkpoint fixture")
    definition = root / "data/admet.csv"
    definition.parent.mkdir()

    def write_definition(*rows):
        definition.write_text(
            "id,task_type,units,minimum,maximum,species\n" + "\n".join(rows) + "\n", encoding="utf-8",
        )
        with (info / "RECORD").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            for path in [definition, *sorted(ensemble.glob("*.pt"))]:
                data = path.read_bytes()
                checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                writer.writerow((path.relative_to(tmp_path).as_posix(), "sha256=" + checksum, len(data)))

    monkeypatch.setattr("importlib.metadata.distribution", lambda _name: PathDistribution(info))
    return write_definition, ensemble


_DILI_ROW = "DILI,classification,-,0.0,1.0,human"


def test_installed_assets_attest_the_exact_head_and_classification_members(local_release):
    write_definition, ensemble = local_release
    write_definition("AMES,classification,-,0.0,1.0,salmonella typhimurium", _DILI_ROW)
    version, members, hashes = dili.verified_assets()
    assert version.version == "2.0.1"
    assert len(members) == 5 and all(member.parent == ensemble for member in members)
    assert set(hashes) == {"admet.csv", *(member.name for member in members)}
    members[0].write_bytes(b"checkpoint tamper!")
    with pytest.raises(dili.DiliError, match="installed_artifact_digest_mismatch"):
        dili.verified_assets()


@pytest.mark.parametrize("rows", [
    ("DILI,regression,-,0.0,1.0,human",),
    ("DILI,classification,percent,0.0,1.0,human",),
    ("DILI,classification,-,0.0,100.0,human",),
    ("DILI,classification,-,-1.0,1.0,human",),
    ("DILI,classification,-,0.0,1.0,mouse",),
    ("ClinTox,classification,-,0.0,1.0,human",),
    (_DILI_ROW, _DILI_ROW),
])
def test_wrong_or_ambiguous_reverified_manifests_do_not_resolve_a_dili_head(local_release, rows):
    write_definition, _ensemble = local_release
    write_definition(*rows)
    with pytest.raises(dili.DiliError, match="endpoint_definition_mismatch"):
        dili.verified_assets()


def test_extract_only_the_absolute_dili_class_score_by_identity():
    frame = pd.DataFrame(
        {"DILI": [0., 1., -0.1, 1.1, True, "0.4", float("nan"), float("inf")],
         "DILI_drugbank_approved_percentile": [4.] * 8, "ClinTox": [0.8] * 8},
        index=["a", "b", "c", "d", "e", "f", "g", "h"], dtype=object,
    )
    assert ligand_admet.values(dili.SPEC, frame, ["missing", *reversed(frame.index)]) == {
        "missing": "missing_provider_row", "a": 0., "b": 1.,
        **dict.fromkeys("cdefgh", "invalid_provider_result"),
    }
    with pytest.raises(dili.DiliError, match="missing_or_duplicate_endpoint"):
        ligand_admet.values(dili.SPEC, frame.iloc[:, [1, 2]], list(frame.index))


def test_record_keeps_training_provenance_and_counts_candidate_coverage(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    (tmp_path / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\nfirst\tCCO\nalias\tOCC\nsalt\tCCO.[Na+]\nfailed\tCC(=O)O\n", encoding="utf-8",
    )

    def predict(keys):
        assert set(keys) == {"CCO", "CC(=O)O"}
        return {"values": {"CCO": 0., "CC(=O)O": "missing_provider_row"}}

    monkeypatch.setattr(dili, "_local_predictions", predict)
    report = dili.screen_run(tmp_path)
    assert report["status"] == "partial" and report["attempted"] == 4
    assert report["unique_compounds"] == 2 and report["unique_predicted"] == 1
    one, alias, salt, failed = report["inputs"]
    assert one["compound_id"] == alias["compound_id"] != failed["compound_id"]
    assert salt == {"candidate_id": "salt", "blocker": "dili_structure_unsupported"}
    by_id = {entry["compound_id"]: entry for entry in report["compounds"]}
    assert by_id[one["compound_id"]] == {
        "compound_id": one["compound_id"], "method": dili.METHOD, "endpoint": dili.ENDPOINT,
        "unit": dili.UNIT, "status": "predicted", dili.VALUE_FIELD: 0.,
    }
    assert by_id[failed["compound_id"]]["blocker"] == "missing_provider_row"
    assert dili.VALUE_FIELD not in by_id[failed["compound_id"]]
    provenance = report["corpus_provenance"]
    assert provenance["overlap"] == corpus_provenance.BENCHMARK_LABEL_CORPUS
    assert provenance["overlap_against"] == "DILI"
    assert provenance["corpora"][0]["rows"] == 475
    assert "CCO" not in json.dumps(report) and "CC(=O)O" not in json.dumps(report)
    assert not {"call", "threshold", "verdict"} & report.keys()


def test_excess_nominations_cannot_be_bypassed_by_selecting_one_candidate(tmp_path, monkeypatch):
    (tmp_path / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\n" + "".join(f"mol{i}\tCCO\n" for i in range(ligand_admet.MAX_CANDIDATES + 1)),
        encoding="utf-8",
    )
    monkeypatch.setattr(dili, "_local_predictions", lambda _keys: pytest.fail("oversized source reached inference"))
    report = dili.screen_run(tmp_path, candidate="mol0")
    assert report["status"] == "untested" and report["blocker"] == "dili_inputs_over_limit"
    assert not report["inputs"] and not report["compounds"]


@pytest.mark.skipif(os.environ.get("RFD4_DILI_RELEASE_SMOKE") != "1", reason="requires ADMET-AI v2.0.1 weights")
def test_release_cpu_predictions_match_the_v2_0_1_reference_scores(tmp_path):
    # Published DrugBank reference predictions in the ADMET-AI v2.0.1 wheel.
    # These are a mapping and scale check, not independent clinical validation.
    reference = {
        "acetaminophen": ("CC(=O)Nc1ccc(O)cc1", 0.5796205),
        "aspirin": ("CC(=O)Oc1ccccc1C(=O)O", 0.6746291),
        "caffeine": ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", 0.9320737),
    }
    (tmp_path / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\n" + "".join(f"{name}\t{smiles}\n" for name, (smiles, _) in reference.items()),
        encoding="utf-8",
    )
    report = dili.screen_run(tmp_path)
    assert report["status"] == "predicted", report.get("blocker")
    assert report["attempted"] == report["unique_compounds"] == report["unique_predicted"] == len(reference)
    assert report["provider"]["admet_ai"] == "2.0.1"
    assert report["provider"]["device"] == "cpu" and report["provider"]["ensemble_members"] == 5
    compounds = {entry["compound_id"]: entry[dili.VALUE_FIELD] for entry in report["compounds"]}
    for row in report["inputs"]:
        assert compounds[row["compound_id"]] == pytest.approx(reference[row["candidate_id"]][1], abs=1e-4)
