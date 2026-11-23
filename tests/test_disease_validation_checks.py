"""Validation selections control which local output files are read."""

from pathlib import Path

import pytest

from rfproteina.disease import validation_checks
from rfproteina.disease.cli import main
from rfproteina.disease.ui import (
    VALIDATION_ACTION,
    VALIDATION_CANDIDATE_FIELD,
    VALIDATION_FIELD,
    VALIDATION_SCOPE_ACTION,
    VALIDATION_SCOPE_FIELD,
    BrowserSession,
    wants_option_ui,
)


def test_validation_panel_checks_only_the_submitted_files(tmp_path: Path, monkeypatch) -> None:
    """An installed check can stay unchecked even when its input exists."""
    run = tmp_path / "example"
    samples = run / "inference_samples"
    samples.mkdir(parents=True)
    sample = samples / "example_generated.cif"
    sample.write_text("data_example\n", encoding="utf-8")
    packed = samples / "packed_generated.cif.zst"
    packed.touch()
    monkeypatch.setattr(validation_checks, "_zstandard", lambda: None)

    panel = BrowserSession(opener=lambda _url: None)
    panel.note_validation_runs((run,))
    page = panel.done_page()
    assert 'value="structure_files" data-recommended checked' in page
    assert 'value="metrics_table">' in page
    assert 'value="zstd_files" disabled' in page
    assert 'value="checksums">' in page

    read = []

    def record_read(path: Path) -> str:
        read.append(path)
        return "data_example\n"

    monkeypatch.setattr(validation_checks, "_design_text", record_read)
    with pytest.raises(ValueError, match="Unavailable"):
        panel.run_results_action(VALIDATION_ACTION, [f"{VALIDATION_FIELD}=zstd_files"])
    assert not read

    panel.run_results_action(VALIDATION_ACTION, [f"{VALIDATION_FIELD}=structure_files"])
    assert read == [sample]
    assert "1 of 1 files readable as UTF-8." in panel.done_page()

    panel.run_results_action(VALIDATION_ACTION, [])
    assert 'value="structure_files" data-recommended>' in panel.done_page()
    assert read == [sample]


def test_validate_output_opens_the_panel_for_an_existing_run(tmp_path: Path) -> None:
    """The standalone entry point sends selected results back through the form."""
    run = tmp_path / "example"
    metrics = run / "inference_metrics"
    metrics.mkdir(parents=True)
    table = metrics / "per_example_metrics.csv"
    contents = "sample,value\nexample,0.2,extra\n"
    table.write_text(contents, encoding="utf-8")
    pages = []

    def submit(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if len(pages) == 1:
            return {"__action": [VALIDATION_ACTION], VALIDATION_FIELD: ["metrics_table"]}
        return {}

    assert main(["--validate-output", str(run), "--log-file", str(tmp_path / "test.log")], exchange=submit) == 0
    assert 'value="metrics_table" data-recommended checked' in pages[0]
    assert "Return to start page" not in pages[0]
    assert "Metrics row too long." in pages[1]
    assert table.read_text(encoding="utf-8") == contents
    assert sorted(p.name for p in run.iterdir()) == ["inference_metrics"]


def test_validate_output_never_enters_the_generation_wizard() -> None:
    """Let the real parser reject a request for both mutually exclusive modes."""
    assert not wants_option_ui(["--ui", "--validate-output=run"])
    assert not wants_option_ui(["--ui", "--val", "run"])


def test_experimental_check_reports_completeness_without_reading_a_receipt(tmp_path: Path, monkeypatch) -> None:
    """Bulk selection keeps reviewer receipt contents out of evidence validation."""
    bench = tmp_path / validation_checks.VALIDATION_FILE
    bench.write_text('{"one":{"status":"example observation"},"two":{}}', encoding="utf-8")
    receipt = tmp_path / validation_checks.SYNTHESIS_REVIEW_RECEIPT
    receipt.write_bytes(b"receipt bytes for local records only")

    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path != receipt
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    choices = {choice.key: choice for choice in validation_checks.plan_validation_checks((tmp_path,))}
    assert choices["experimental_record"].recommended
    assert choices["synthesis_receipt"].installed and not choices["synthesis_receipt"].recommended
    results = validation_checks.run_validation_checks((tmp_path,), tuple(key for key, c in choices.items() if c.installed))
    assert "1 of 2 sample records contain a status." in results["experimental_record"]
    assert "example observation" not in results["experimental_record"]
    assert results["synthesis_receipt"].endswith("Attachment present.")


def test_malformed_experimental_record_and_missing_synthesis_receipt(tmp_path: Path) -> None:
    """A broken bench record never looks like a complete one."""
    bench = tmp_path / validation_checks.VALIDATION_FILE
    bench.write_text('{"status":"first","status":"second"}', encoding="utf-8")
    results = validation_checks.run_validation_checks((tmp_path,), ("experimental_record", "synthesis_receipt"))
    assert "unique keys" in results["experimental_record"]
    assert results["synthesis_receipt"].endswith("No attachment found.")


def test_synthesis_receipt_requires_a_regular_file_in_the_run(tmp_path: Path) -> None:
    target = tmp_path / "other.json"
    target.touch()
    receipt = tmp_path / validation_checks.SYNTHESIS_REVIEW_RECEIPT
    receipt.symlink_to(target)
    results = validation_checks.run_validation_checks((tmp_path,), ("synthesis_receipt",))
    assert results["synthesis_receipt"].endswith("No attachment found.")
    receipt.unlink()
    receipt.mkdir()
    results = validation_checks.run_validation_checks((tmp_path,), ("synthesis_receipt",))
    assert results["synthesis_receipt"].endswith("No attachment found.")


def test_scope_switch_recommends_only_checks_for_the_new_run(tmp_path: Path, monkeypatch) -> None:
    """A Run click cannot carry old checkboxes across to a different run."""
    first, second = tmp_path / "first", tmp_path / "second"
    sample_dir = first / "inference_samples"
    sample_dir.mkdir(parents=True)
    (sample_dir / "sample_generated.cif").write_bytes(b"data_example\n")
    second.mkdir()
    record = b'{"status":"recorded"}'
    (second / validation_checks.VALIDATION_FILE).write_bytes(record)
    pages = []

    def reject_checksum(_path):
        pytest.fail("The old scope's selected checksum was executed.")

    monkeypatch.setattr(validation_checks, "file_sha256", reject_checksum)

    def submit(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if len(pages) == 1:
            return {
                "__action": [VALIDATION_ACTION],
                VALIDATION_SCOPE_FIELD: ["1"],
                VALIDATION_FIELD: ["checksums"],
            }
        if len(pages) == 2:
            return {
                "__action": [VALIDATION_ACTION],
                VALIDATION_SCOPE_FIELD: ["1"],
                VALIDATION_FIELD: ["experimental_record"],
            }
        return {}

    args = ["--validate-output", str(first), "--validate-output", str(second), "--log-file", str(tmp_path / "test.log")]
    assert main(args, exchange=submit) == 0
    assert 'value="0" selected' in pages[0]
    assert 'value="structure_files" data-recommended checked' in pages[0]
    assert 'value="experimental_record">' in pages[0]
    assert 'value="1" selected' in pages[1]
    assert 'value="structure_files">' in pages[1]
    assert 'value="experimental_record" data-recommended checked' in pages[1]
    assert f"1 file, {len(record)} bytes on disk." in pages[1]
    assert "second — The run-level bench record contains a status." in pages[2]
    assert "first —" not in pages[2]


@pytest.mark.parametrize("scope_fields", [[], ["0", "1"], ["2"], ["/other/directory"]])
def test_multi_run_validation_rejects_unknown_or_ambiguous_scope(tmp_path: Path, scope_fields: list[str]) -> None:
    session = BrowserSession(opener=lambda _url: None)
    session.note_validation_runs((tmp_path / "first", tmp_path / "second"))
    fields = [f"{VALIDATION_SCOPE_FIELD}={value}" for value in scope_fields]
    fields.append(f"{VALIDATION_FIELD}=checksums")
    with pytest.raises(ValueError, match="Choose a finished run"):
        session.run_results_action(VALIDATION_ACTION, fields)


def test_refresh_picks_up_new_evidence_without_running_old_choices(tmp_path: Path, monkeypatch) -> None:
    panel = BrowserSession(opener=lambda _url: None)
    panel.note_validation_runs((tmp_path,))
    assert 'value="experimental_record">' in panel.done_page()

    def reject_run(*args, **kwargs):
        pytest.fail("Refreshing recommendations must not run any validation checks.")

    monkeypatch.setattr("rfproteina.disease.ui.run_validation_checks", reject_run)
    (tmp_path / validation_checks.VALIDATION_FILE).write_text('{"status":"recorded"}', encoding="utf-8")
    panel.run_results_action(VALIDATION_SCOPE_ACTION, [f"{VALIDATION_FIELD}=checksums"])
    assert 'value="experimental_record" data-recommended checked' in panel.done_page()
    assert 'value="checksums">' in panel.done_page()


def test_candidate_recommendations_use_its_format_and_bench_entry(tmp_path: Path, monkeypatch) -> None:
    samples = tmp_path / "inference_samples"
    samples.mkdir()
    first = samples / "binder_0_generated.cif"
    first.write_text("data_example\n", encoding="utf-8")
    (samples / "binder_1_generated.cif.zst").touch()
    (tmp_path / validation_checks.VALIDATION_FILE).write_text(
        '{"binder_0":{"status":"example observation"}}', encoding="utf-8"
    )
    monkeypatch.setattr(validation_checks, "_zstandard", lambda: object())
    candidates, _ = validation_checks.candidate_validation_plan((tmp_path,), None)
    first_key, second_key = (candidate.key for candidate in candidates)

    def recommended(key: str) -> set[str]:
        _, choices = validation_checks.candidate_validation_plan((tmp_path,), key)
        return {choice.key for choice in choices if choice.recommended}

    assert recommended(first_key) == {"structure_files", "experimental_record"}
    assert recommended(second_key) == {"zstd_files"}

    panel = BrowserSession(opener=lambda _url: None)
    panel.note_validation_runs((tmp_path,))
    pick_first = f"{VALIDATION_CANDIDATE_FIELD}={first_key}"

    def reject_checksum(_path):
        pytest.fail("A check from the previous scope ran during candidate selection.")

    monkeypatch.setattr(validation_checks, "file_sha256", reject_checksum)
    panel.run_results_action(VALIDATION_ACTION, [pick_first, f"{VALIDATION_FIELD}=checksums"])
    assert f'value="{first_key}" selected' in panel.done_page()
    assert 'value="zstd_files">' in panel.done_page()
    panel.run_results_action(
        VALIDATION_ACTION, [pick_first, f"{VALIDATION_FIELD}=structure_files", f"{VALIDATION_FIELD}=experimental_record"]
    )
    page = panel.done_page()
    assert "1 of 1 files readable as UTF-8." in page
    assert "The selected candidate's bench record contains a status." in page.replace("&#x27;", "'")
    assert "example observation" not in page

    first.unlink()
    assert "The selected candidate is unavailable." in panel.done_page()
    with pytest.raises(ValueError, match="Choose an available candidate"):
        panel.run_results_action(VALIDATION_ACTION, [pick_first, f"{VALIDATION_FIELD}=structure_files"])


def test_bench_record_is_not_matched_by_a_duplicate_basename(tmp_path: Path) -> None:
    for run_name in ("left", "right"):
        sample = tmp_path / "inference_samples" / run_name / "binder_0_generated.cif"
        sample.parent.mkdir(parents=True)
        sample.touch()
    archive = tmp_path / "inference_samples" / "left" / "binder_0_generated.cif.gz"
    archive.touch()
    (tmp_path / validation_checks.VALIDATION_FILE).write_text(
        '{"binder_0":{"status":"recorded"}}', encoding="utf-8"
    )
    candidates, _ = validation_checks.candidate_validation_plan((tmp_path,), None)
    assert len(candidates) == 2
    for candidate in candidates:
        _, choices = validation_checks.candidate_validation_plan((tmp_path,), candidate.key)
        assert all(c.key != "experimental_record" or not c.recommended for c in choices)
