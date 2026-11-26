"""The ToxScan reader attributes original, hash-bound service exports to exact chemical identities."""

import csv
import json
from io import StringIO

import pytest

from rfproteina.disease import corpus_provenance, toxscan, toxscan_csv
from rfproteina.disease import toxscan_inputs as inputs


def nominate(directory, smiles="C[C@@H](O)F", *, payload=b"source", candidate_id="compound"):
    (directory / "ligand_candidates.tsv").write_text(
        f"candidate_id\tsmiles\n{candidate_id}\t{smiles}\n", encoding="utf-8",
    )
    (directory / inputs.SOURCE_FILE).write_bytes(payload)
    manifest = {"version": 1, "candidate_id": candidate_id, "source": inputs.SOURCE, "export_sha256": inputs.sha256(payload)}
    (directory / inputs.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_nomination_binds_source_and_stereochemistry(tmp_path):
    nominate(tmp_path)
    target = inputs.nomination(tmp_path, "compound")
    assert inputs.export_bytes(tmp_path, target) == b"source"
    assert inputs.canonical("F[C@H](O)C") == target.canonical_smiles
    assert inputs.canonical("C[C@H](O)F") != target.canonical_smiles
    assert inputs.canonical("C[13CH3]") != inputs.canonical("CC")
    assert inputs.canonical("[Na+].[O-]C=O") != inputs.canonical("OC=O")
    assert inputs.canonical("[CH3:1][CH3:2]") == inputs.canonical("CC")
    (tmp_path / inputs.SOURCE_FILE).write_bytes(b"other")
    with pytest.raises(inputs.ToxScanError, match="toxscan_export_digest_mismatch"):
        inputs.export_bytes(tmp_path, target)


@pytest.mark.parametrize("smiles", ("*", "CC |anything|", "not-a-structure"))
def test_ambiguous_input_is_rejected(tmp_path, smiles):
    nominate(tmp_path, smiles)
    with pytest.raises(inputs.ToxScanError, match="toxscan_invalid_structure"):
        inputs.nomination(tmp_path, None)


def test_manifest_requires_author_export_and_unique_named_candidate(tmp_path):
    good = nominate(tmp_path)
    manifest = tmp_path / inputs.MANIFEST
    for change in ({"source": "custom"}, {"version": True}, {"units": "custom"}, {"export_sha256": "unknown"}):
        manifest.write_text(json.dumps({**good, **change}), encoding="utf-8")
        with pytest.raises(inputs.ToxScanError, match="toxscan_invalid_manifest"):
            inputs.nomination(tmp_path, None)
    manifest.write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(inputs.ToxScanError, match="toxscan_candidate_not_nominated"):
        inputs.nomination(tmp_path, "other")
    with (tmp_path / "ligand_candidates.tsv").open("a") as stream:
        stream.write("compound\tCCO\n")
    with pytest.raises(inputs.ToxScanError, match="toxscan_ambiguous_nomination"):
        inputs.nomination(tmp_path, None)


def author_shape(smiles="C[C@@H](O)F"):
    """Synthetic values test the official service's wide CSV layout, never a model performance control."""
    return {
        "SMILES": smiles, toxscan_csv.AD_HEADER: "In AD", toxscan_csv.SIMILARITY_HEADER: "0.85",
        **dict.fromkeys(toxscan.REGRESSIONS, "1.25mg/kg"),
        **dict.fromkeys(toxscan.CLASSIFICATIONS, "NO"), "Ames Mutagenicity": "YES",
    }


def csv_bytes(rows, headers=toxscan_csv.HEADERS):
    stream = StringIO(newline="")
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow(headers)
    for row in rows:
        writer.writerow([row[key] for key in headers])
    return stream.getvalue().encode("utf-8")


@pytest.mark.parametrize(("flag", "domain", "recommend"), [
    ("In AD", "in_domain", True), ("Out of AD", "out_of_domain", False), ("", "unreported", False),
])
def test_service_row_matches_graph_and_preserves_applicability(tmp_path, flag, domain, recommend):
    target = author_shape("F[C@H](O)C")
    target[toxscan_csv.AD_HEADER] = flag
    wrong_isomer = {**author_shape("C[C@H](O)F"), "Ames Mutagenicity": "NO"}
    if not flag:
        target[toxscan_csv.SIMILARITY_HEADER] = ""
    payload = csv_bytes([wrong_isomer, target], tuple(reversed(toxscan_csv.HEADERS)))
    nominate(tmp_path, payload=payload)
    assert toxscan.installed()
    report = toxscan.screen_run(tmp_path, "compound")
    assert report["status"] == "complete", report.get("blocker")
    assert report["source_sha256"] == inputs.sha256(payload)
    assert report["structure_sha256"] == inputs.sha256(inputs.canonical("C[C@@H](O)F").encode())
    assert report["applicability_domain"] == domain
    assert report["similarity_score"] == (0.85 if flag else None)
    assert report["results"]["classifications"]["Ames Mutagenicity"] == "YES"
    assert report["results"]["regression_scale"] == "unverified_source_display"
    assert set(report["results"]["source_displays"]) == set(toxscan.REGRESSIONS)
    assert toxscan.readiness(tmp_path, "compound")[:2] == (True, recommend)
    assert toxscan.report_view(report) == report
    assert toxscan.report_view({**report, "candidate_id": "another"})["blocker"] == "toxscan_invalid_report"
    report["results"]["classifications"]["Ames Mutagenicity"] = "NO"
    assert toxscan.report_view(report)["blocker"] == "toxscan_invalid_report"


@pytest.mark.parametrize(("rows", "blocker"), [
    ([author_shape("CCO")], "toxscan_structure_not_found"),
    ([author_shape(), author_shape("F[C@H](O)C")], "toxscan_ambiguous_export"),
    ([{**author_shape(), "SR-p53": "negative?"}], "toxscan_invalid_export_value"),
    ([{**author_shape(), toxscan_csv.SIMILARITY_HEADER: "nan"}], "toxscan_invalid_export_value"),
    ([{**author_shape(), toxscan_csv.SIMILARITY_HEADER: "1.01"}], "toxscan_invalid_export_value"),
    ([{**author_shape(), toxscan_csv.AD_HEADER: "YES"}], "toxscan_invalid_domain"),
    ([{**author_shape(), "Rat oral LD50": "0.2 mg/L"}], "toxscan_invalid_export_value"),
    ([{**author_shape(), "Rat oral LD50": "1e1000mg/kg"}], "toxscan_invalid_export_value"),
])
def test_invalid_export_never_clears_an_assay(tmp_path, rows, blocker):
    nominate(tmp_path, payload=csv_bytes(rows))
    report = toxscan.screen_run(tmp_path)
    assert report["status"] == "untested" and report["blocker"] == blocker
    assert report["results"] == {}
    assert toxscan.report_view(report) == report


def test_refuses_benchmark_tables_partial_study_figures_and_duplicate_headings(tmp_path):
    # Figure S4 reports Ames results for these two named graphs. Other endpoint calls and
    # applicability values are absent, so a transcription cannot stand in for a service export.
    figure = b"SMILES,Ames Mutagenicity\nNCCNCCN,NO\nNCCNCCNCCNCCN,YES\n"
    duplicate = (*toxscan_csv.HEADERS[:-1], toxscan_csv.HEADERS[-2])
    for raw in (b"endpoint,AUROC\nAmes,0.879\n", figure, csv_bytes([author_shape()], duplicate)):
        nominate(tmp_path, smiles="NCCNCCN", payload=raw)
        assert toxscan.screen_run(tmp_path)["blocker"] == "toxscan_invalid_export_schema"


@pytest.mark.parametrize(("smiles", "call"), [("NCCNCCN", "NO"), ("NCCNCCNCCNCCN", "YES")])
def test_published_ames_pair_in_reconstructed_export(tmp_path, smiles, call):
    # Fig S4, DOI 10.1021/acs.est.5c04614.s001. These are transcribed categorical
    # references in the portal format; the study did not report the remaining fields.
    rows = [
        {**dict.fromkeys(toxscan_csv.HEADERS, ""), "SMILES": graph, "Ames Mutagenicity": label}
        for graph, label in (("NCCNCCN", "NO"), ("NCCNCCNCCNCCN", "YES"))
    ]
    nominate(tmp_path, smiles, payload=csv_bytes(rows))
    reference = toxscan.screen_run(tmp_path)
    assert reference["status"] == "complete" and toxscan.report_view(reference) == reference
    calls = reference["results"]["classifications"]
    assert [(key, value) for key, value in calls.items() if value is not None] == [("Ames Mutagenicity", call)]
    assert reference["applicability_domain"] == "unreported" and reference["similarity_score"] is None
    assert all(value is None for value in reference["results"]["source_displays"].values())
    assert reference["classification_validation"] == {
        "Ames Mutagenicity": {"assay": "Bacterial reverse-mutation assay", "auroc": 0.879, "accuracy": 0.876,
                             "auroc_sd": 0.016, "accuracy_sd": 0.012},
    }
    older = {k: v for k, v in reference.items() if not k.startswith("classification_validation")}
    assert toxscan.report_view(older) == reference
    corrupt = {**reference, "classification_validation": {}}
    assert toxscan.report_view(corrupt)["blocker"] == "toxscan_invalid_report"


def test_benchmark_values_match_export_columns_by_name(tmp_path):
    nominate(tmp_path, payload=csv_bytes([author_shape()], tuple(reversed(toxscan_csv.HEADERS))))
    report = toxscan.screen_run(tmp_path)
    benchmarks = report["classification_validation"]
    assert set(benchmarks) == set(toxscan.CLASSIFICATIONS)
    assert all(
        corpus_provenance.overlap_with(toxscan.METHOD, key) == corpus_provenance.BENCHMARK_LABEL_CORPUS
        for key in (*toxscan.CLASSIFICATIONS, *toxscan.REGRESSIONS)
    )
    assert benchmarks["Cardiotoxicity5"]["auroc"] == 0.692
    assert benchmarks["Cardiotoxicity10"]["accuracy"] == 0.655
    assert benchmarks["NR-AR-LBD"]["accuracy"] == 0.983
    assert benchmarks["NR-AR-LBD"]["auroc"] == 0.805
    assert report["classification_validation_source"].endswith("acs.est.5c04614.s001")


def test_historical_and_current_identity_checks_remain_distinguishable(tmp_path):
    nominate(tmp_path, payload=csv_bytes([author_shape()]))
    report = toxscan.screen_run(tmp_path)
    assert report["identity_check_version"] == 2
    assert toxscan.report_view({**report, "identity_check_version": 1})["blocker"] == "toxscan_invalid_report"

    old = {k: v for k, v in report.items() if k != "identity_check_version" and not k.startswith("classification_")}
    old_payload = {k: old[k] for k in (
        "candidate_id", "source_sha256", "structure_sha256", "applicability_domain", "similarity_score", "results",
    )}
    old["result_sha256"] = inputs.sha256(json.dumps(old_payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    restored = toxscan.report_view(old)
    assert restored["status"] == "complete" and restored["identity_check_version"] == 1
    assert restored["result_sha256"] == old["result_sha256"]
    assert restored["classification_validation"] == report["classification_validation"]
    assert toxscan.report_view(restored) == restored
    assert toxscan.report_view({**old, "identity_check_version": 2})["blocker"] == "toxscan_invalid_report"


@pytest.mark.parametrize(("similarity", "recommend"), [(0.0, False), (0.679, False), (0.680, True)])
def test_in_domain_nomination_with_discordant_displayed_score_requires_review(tmp_path, similarity, recommend):
    row = {**author_shape(), toxscan_csv.SIMILARITY_HEADER: str(similarity)}
    nominate(tmp_path, payload=csv_bytes([row]))
    report = toxscan.screen_run(tmp_path)
    assert report["status"] == "complete"
    assert report["applicability_domain"] == "in_domain" and report["similarity_score"] == similarity
    applicable, recommended, reason = toxscan.readiness(tmp_path)
    assert applicable and recommended is recommend
    if not recommend:
        assert "0.679" in reason


def test_blank_assays_stay_unreported_and_similarity_cannot_select_domain(tmp_path):
    row = dict.fromkeys(toxscan_csv.HEADERS, "")
    row.update({"SMILES": "NCCNCCN", "Ames Mutagenicity": "NO", toxscan_csv.SIMILARITY_HEADER: "0.86"})
    nominate(tmp_path, "NCCNCCN", payload=csv_bytes([row]))
    report = toxscan.screen_run(tmp_path)
    assert report["status"] == "complete"
    assert report["applicability_domain"] == "unreported" and report["similarity_score"] == 0.86
    assert toxscan.readiness(tmp_path)[:2] == (True, False)
    calls = report["results"]["classifications"]
    assert {key: call for key, call in calls.items() if call is not None} == {"Ames Mutagenicity": "NO"}
    assert all(value is None for value in report["results"]["source_displays"].values())
    assert toxscan.report_view(report) == report
    row["Ames Mutagenicity"] = ""
    nominate(tmp_path, "NCCNCCN", payload=csv_bytes([row]))
    assert toxscan.screen_run(tmp_path)["blocker"] == "toxscan_invalid_export_value"
    row.update({toxscan_csv.AD_HEADER: "In AD", "Rat oral LD50": "3.5mg/kg"})
    nominate(tmp_path, "NCCNCCN", payload=csv_bytes([row]))
    assert toxscan.screen_run(tmp_path)["status"] == "complete"
    assert toxscan.readiness(tmp_path)[:2] == (True, False)


def test_oversized_exports_and_missing_files_have_named_blockers(tmp_path, monkeypatch):
    assert toxscan.screen_run(tmp_path)["blocker"] == "toxscan_manifest_missing"
    nominate(tmp_path, payload=csv_bytes([author_shape(), author_shape("CCO")]))
    monkeypatch.setattr(toxscan_csv, "MAX_EXPORT_ROWS", 1)
    assert toxscan.screen_run(tmp_path)["blocker"] == "toxscan_invalid_export"
    (tmp_path / inputs.SOURCE_FILE).unlink()
    assert toxscan.screen_run(tmp_path)["blocker"] == "toxscan_export_missing"
