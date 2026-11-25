"""Require an exact run-target identity for local OpenAntigens evidence."""

from __future__ import annotations

import json

import pytest

from rfproteina.disease import mcp_api, openantigens, screen, screen_panel, screen_selection

ACCESSION = "P00533"
OTHER_ACCESSION = "P02768"
TARGET = {
    "entry_name": "EGFR_HUMAN",
    "topology": "Single-pass",
    "has_alphafold_structure": True,
    "has_pdb": True,
    "has_interpro": True,
    "has_family_context": True,
    "has_assembly_requirements": False,
}


def _write(path, **record):
    path.write_text(json.dumps(record), encoding="utf-8")


def _manifest(directory, accession=ACCESSION):
    _write(directory / openantigens.MANIFEST, version=1, target_accession=accession)


def _forbid_lookup(_accession):
    pytest.fail("a run target outside the input contract reached the snapshot")


@pytest.fixture
def local_snapshot(monkeypatch):
    def lookup(accession):
        return {**TARGET, "internal_data": "/private/example.fasta"} if accession == ACCESSION else None

    monkeypatch.setattr(openantigens.snapshot, "target", lookup)
    monkeypatch.setattr(openantigens.snapshot, "installed", lambda: True)


@pytest.mark.parametrize("source", ["live_job", "archived_job", "archived_generation", "standalone_manifest"])
def test_report_uses_the_qualified_run_target(tmp_path, local_snapshot, source):
    run = tmp_path / "run"
    run.mkdir()
    if source == "live_job":
        _write(tmp_path / "run.job.json", target_id=ACCESSION, binder_kind="protein")
    elif source == "archived_job":
        _write(run / "original-stem.job.json", target_id=ACCESSION, binder_kind="protein")
    elif source == "archived_generation":
        _write(run / "generation.json", target_id=ACCESSION, binder_kind="protein")
    else:
        _manifest(run)

    record = openantigens.screen_run(run, expected_target=ACCESSION)
    assert record["status"] == "reported"
    assert record["scope"] == "run_target"
    assert record["target_accession"] == ACCESSION
    assert record["target"] == TARGET
    assert "candidate_id" not in record and "threshold" not in record


@pytest.mark.parametrize(
    "settings",
    [
        '{"version": 1, "target_accession": "P00533", "candidate_id": "lead_A"}',
        '{"version": 1, "target_accession": "P00533", "target_accession": "P00533"}',
        '{"version": true, "target_accession": "P00533"}',
        '{"version": 1, "target_accession": "P00533-2"}',
        '{"version": 1, "target_accession": "EGFR_HUMAN"}',
    ],
)
def test_invalid_manifest_takes_priority_over_run_metadata(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(openantigens.snapshot, "target", _forbid_lookup)
    _write(tmp_path / "generation.json", target_id=ACCESSION)
    (tmp_path / openantigens.MANIFEST).write_text(settings, encoding="utf-8")

    report = openantigens.screen_run(tmp_path)
    assert report["status"] == "untested"
    assert report["blocker"] == "invalid_openantigens_manifest"


@pytest.mark.parametrize("conflict", ["manifest", "job", "expected_target", "candidate"])
def test_disagreeing_target_or_candidate_scope_cannot_reach_snapshot(tmp_path, monkeypatch, conflict):
    monkeypatch.setattr(openantigens.snapshot, "target", _forbid_lookup)
    _write(tmp_path / "generation.json", target_id=ACCESSION)
    _manifest(tmp_path, OTHER_ACCESSION if conflict == "manifest" else ACCESSION)
    if conflict == "job":
        _write(tmp_path / f"{tmp_path.name}.job.json", target_id=OTHER_ACCESSION)
    report = openantigens.screen_run(
        tmp_path,
        candidate="lead_A" if conflict == "candidate" else None,
        expected_target=OTHER_ACCESSION if conflict == "expected_target" else None,
    )
    assert report["status"] == "untested"
    expected = "openantigens_requires_run_scope" if conflict == "candidate" else "openantigens_target_mismatch"
    assert report["blocker"] == expected
    assert "target" not in report


@pytest.mark.parametrize(
    "metadata, blocker",
    [
        ({}, "openantigens_target_missing"),
        ({"target_id": "P00533-2"}, "openantigens_target_unresolved"),
        ({"target_id": "EGFR"}, "openantigens_target_unresolved"),
        ({"target_id": 7}, "openantigens_invalid_job"),
    ],
)
def test_missing_noncanonical_or_malformed_run_identity_is_named(tmp_path, monkeypatch, metadata, blocker):
    monkeypatch.setattr(openantigens.snapshot, "target", _forbid_lookup)
    _write(tmp_path / "generation.json", **metadata)

    assert openantigens.screen_run(tmp_path)["blocker"] == blocker


def test_ambiguous_unattributed_archive_jobs_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(openantigens.snapshot, "target", _forbid_lookup)
    _write(tmp_path / "first.job.json", target_id=ACCESSION)
    _write(tmp_path / "second.job.json", target_id=ACCESSION)

    assert openantigens.screen_run(tmp_path)["blocker"] == "openantigens_invalid_job"


def test_broken_primary_job_link_cannot_fall_back_to_another_job(tmp_path, monkeypatch):
    monkeypatch.setattr(openantigens.snapshot, "target", _forbid_lookup)
    _write(tmp_path / "fallback.job.json", target_id=ACCESSION)
    primary = tmp_path / f"{screen.base_name(tmp_path)}.job.json"
    try:
        primary.symlink_to(tmp_path / "absent.job.json")
    except OSError:
        pytest.skip("symlink creation is unavailable")

    assert openantigens.screen_run(tmp_path)["blocker"] == "openantigens_invalid_job"


def test_snapshot_failures_do_not_look_like_detected_evidence(tmp_path, local_snapshot, monkeypatch):
    _manifest(tmp_path, OTHER_ACCESSION)
    missing = openantigens.screen_run(tmp_path)
    assert missing["status"] == "untested" and missing["blocker"] == "openantigens_no_target_evidence"

    def fail(_accession):
        raise openantigens.snapshot.SnapshotError("openantigens_snapshot_unavailable")

    monkeypatch.setattr(openantigens.snapshot, "target", fail)
    unavailable = openantigens.screen_run(tmp_path)
    assert unavailable["status"] == "untested" and unavailable["blocker"] == "openantigens_snapshot_unavailable"
    assert "target" not in unavailable


def test_projection_retains_only_typed_public_summary_fields(tmp_path, local_snapshot):
    _manifest(tmp_path)
    result = openantigens.screen_run(tmp_path)
    augmented = {**result, "private_path": "/private/sequence.fasta", "target": {**TARGET, "sequence": "raw"}}
    assert openantigens.report_view(augmented) == result
    for invalid in (
        {**result, "index_sha256": "0" * 64},
        {**result, "target_accession": ACCESSION + "-2"},
        {**result, "target": {**TARGET, "entry_name": "EGFR_MOUSE"}},
        {**result, "target": {**TARGET, "has_pdb": 1}},
        {**result, "blocker": "openantigens_no_target_evidence"},
    ):
        report = openantigens.report_view(invalid)
        assert report["status"] == "untested"
        assert report["blocker"] == "openantigens_invalid_report"
        assert "target" not in report


def test_picker_recommends_attributable_local_evidence_for_run_scope(tmp_path, local_snapshot):
    _write(tmp_path / "generation.json", target_id=ACCESSION, binder_kind="protein")

    def choice(example=None, installed=True):
        return next(
            row
            for row in screen_selection.plan(
                tmp_path,
                "protein",
                example=example,
                available={openantigens.METHOD_ID: installed},
            )
            if row.method.key == openantigens.METHOD_ID
        )

    valid = choice()
    assert valid.installed and valid.applicable and valid.recommended
    assert not choice("lead_A").applicable
    assert not choice(installed=False).recommended
    _manifest(tmp_path, OTHER_ACCESSION)
    assert not choice().applicable and not choice().recommended


def test_picker_and_saved_report_use_the_same_target_context(tmp_path, local_snapshot, monkeypatch):
    _write(tmp_path / "generation.json", target_id=ACCESSION, binder_kind="protein")
    monkeypatch.setattr(screen_selection, "installations", lambda keys: dict.fromkeys(keys, True))
    monkeypatch.setattr(screen_panel, "_kind", lambda _directory: "protein")
    monkeypatch.setattr(screen, "_sample_measurements", lambda *args, **kwargs: ([], [], []))
    requested = [openantigens.METHOD_ID]

    (picked,) = screen_panel.run(tmp_path, None, "explicit", requested)
    assert picked["status"] == "reported"
    interpreted = screen.screen_generation(
        tmp_path,
        binder_kind="protein",
        target_id=ACCESSION,
        screen_methods=requested,
    )
    saved = interpreted["method_selection"]["prepared_measurements"][openantigens.METHOD_ID]
    assert picked["measurement"] == mcp_api._redact(mcp_api._openantigens_view(saved))
    assert saved["target"] == TARGET

    overridden = screen.screen_generation(
        tmp_path,
        binder_kind="protein",
        target_id=OTHER_ACCESSION,
        screen_methods=requested,
    )
    saved = overridden["method_selection"]["prepared_measurements"][openantigens.METHOD_ID]
    assert saved["status"] == "untested" and saved["blocker"] == "openantigens_target_mismatch"
