"""Prepared image comparisons use the same candidate scope in the picker, CLI and MCP."""

import json

import pytest

from rfproteina.disease import cellprofiler_runtime, cellprofiler_stain, mcp_api, screen, screen_panel, screen_selection
from rfproteina.disease.validation_checks import candidate_validation_plan
from scripts import screen_extras


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    image = pytest.importorskip("PIL.Image")
    run = tmp_path / "run"
    structures = run / "inference_samples"
    structures.mkdir(parents=True)
    for name in ("candidate_a", "candidate_b"):
        (structures / f"{name}_generated.cif").write_text("data_example\n")
    (run / "generation.json").write_text('{"binder_kind":"protein"}')
    image.new("L", (4, 4), 255).save(run / "roi.png")
    fields = {}
    for index, (name, role) in enumerate((("test", "candidate"), ("negative", "isotype_control")), start=1):
        image.new("RGB", (4, 4), (index * 50, 115, 223)).save(run / f"{name}.png")
        fields[name] = {
            "material": "human_frozen_tissue",
            "specimen_id": "section_1",
            "tissue": "Kidney",
            "region": "Cortex",
            "assay_batch": "batch_1",
            "acquisition_id": "scan_1",
            "role": role,
            "reagent_id": "ACDEFGHIKL" if role == "candidate" else "control_1",
            "image": f"{name}.png",
            "mask": "roi.png",
        }
    fields["test"]["candidate_id"] = "candidate_a"
    pair = {"candidate_id": "candidate_a", "comparison_id": "cmp_1", "test": "test", "negative": "negative"}
    manifest = {"stains": "hematoxylin_dab", "fields": fields, "comparisons": [pair]}
    (run / cellprofiler_stain.MANIFEST).write_text(json.dumps(manifest))
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    monkeypatch.setattr(cellprofiler_runtime, "installed", lambda: True)
    calls = []

    def measure_regions(regions):
        # Stand in for the external process. Attribution, image decoding, and
        # the reporter still operate on real PNG images and their mask.
        calls.append(len(regions))
        return [
            {"roi_pixels": int(mask.sum()), "mean_dab": 0.1 * i, "median_dab": 0.1 * i}
            for i, (_rgb, mask) in enumerate(regions, start=1)
        ]

    monkeypatch.setattr(cellprofiler_runtime, "measure_regions", measure_regions)
    return run, calls


def test_picker_and_finished_run_keep_staining_outside_the_safety_verdict(prepared, monkeypatch):
    run, calls = prepared
    available = screen_selection.installations({"cellprofiler"})
    assert available == {"cellprofiler": True}
    for example, expected in ((None, True), ("candidate_a", True), ("candidate_b", False)):
        (choice,) = screen_selection.plan(run, "protein", example=example, available=available)
        assert choice.recommended is expected
        assert choice.applicable is expected
    assert not calls
    with monkeypatch.context() as missing:
        missing.setattr(screen_selection, "_modules", lambda *_names: False)
        assert screen_selection.installations({"cellprofiler"}) == {"cellprofiler": False}
    monkeypatch.setattr(cellprofiler_runtime, "installed", lambda: False)
    assert screen_selection.installations({"cellprofiler"}) == {"cellprofiler": False}
    monkeypatch.setattr(cellprofiler_runtime, "installed", lambda: True)
    baseline = screen.screen_generation(run, binder_kind="protein", target_id="", screen_selection="explicit")
    result = screen.screen_generation(run, binder_kind="protein", target_id="", screen_methods=["cellprofiler"])
    for field in ("call", "states", "hits", "cuts", "untested"):
        assert result[field] == baseline[field]
    assert calls == [2]
    saved = json.loads((run / "screen.json").read_text())["method_selection"]["prepared_measurements"]["cellprofiler"]
    assert saved["scope"] == "run" and saved["status"] == "reported"
    (row,) = saved["comparisons"]
    assert row["test"]["region"] == "Cortex" and row["negative"]["role"] == "isotype_control"
    assert row["test_minus_negative_mean_dab"] == pytest.approx(-0.1)
    (run / cellprofiler_stain.MANIFEST).unlink()
    result = screen.screen_generation(run, binder_kind="protein", target_id="", screen_methods=["cellprofiler"])
    selected = result["method_selection"]
    assert "cellprofiler" in selected["skipped"]
    assert "cellprofiler" not in selected["prepared_measurements"]
    assert calls == [2]


def test_standalone_and_candidate_mcp_read_the_same_fields(prepared, capsys):
    run, calls = prepared
    assert screen_extras.main(["cellprofiler", str(run), "--candidate", "candidate_a"]) == 0
    source = json.loads(capsys.readouterr().out)
    arguments = {"directory": str(run), "methods": ["cellprofiler"]}
    candidates, _ = candidate_validation_plan((run,), None)
    assert {c.example_id for c in candidates} == {"candidate_a", "candidate_b"}
    for candidate in candidates:
        response = mcp_api.call_tool("run_selected_screens", {**arguments, "candidate_id": candidate.key})
        (record,) = response["results"]
        if candidate.example_id == "candidate_b":
            assert record["status"] == "inapplicable" and "measurement" not in record
            continue
        measured = record["measurement"]
        assert measured["scope"] == "candidate" and record["status"] == "reported"
        assert measured["manifest_sha256"] == source["manifest_sha256"]
        assert measured["provider_image_digest"] == cellprofiler_runtime.DIGEST
        (local,) = source["comparisons"]
        (public,) = measured["comparisons"]
        assert public["test_minus_negative_mean_dab"] == local["test_minus_negative_mean_dab"]
        assert public["negative"] == local["negative"]
        assert public["test"]["image_sha256"] == local["test"]["image_sha256"]
        assert public["test"]["reagent_id"] == "[sequence omitted]"
        assert local["test"]["reagent_id"] == "ACDEFGHIKL"
    assert screen_extras.main(["cellprofiler", str(run), "--candidate", "candidate_b"]) == 2
    assert json.loads(capsys.readouterr().out)["blocker"] == "cellprofiler_candidate_not_nominated"
    assert calls == [2, 2]


def test_saved_image_projection_excludes_unknown_payloads_from_both_mcp_views(prepared):
    run, _ = prepared
    record = cellprofiler_stain.measure(run, "candidate_a")
    (row,) = record["comparisons"]
    record["raw_images"] = [12345678]
    row["test"]["export_path"] = "secret-file.png"
    row["negative"]["raw_mask"] = {"patient_name": "Private Name"}
    payload = {"method_selection": {"prepared_measurements": {"cellprofiler": record}}}
    response = mcp_api._verdict(payload, include_record=True)
    for surface in (response, response["record"]):
        saved = surface["method_selection"]["prepared_measurements"]["cellprofiler"]
        assert saved["reported"] == saved["attempted"] == 1
        assert saved["provider_image_digest"] == cellprofiler_runtime.DIGEST
        assert saved["comparisons"][0]["test_minus_negative_mean_dab"] == pytest.approx(-0.1)
        assert saved["comparisons"][0]["test"]["reagent_id"] == "[sequence omitted]"
        for value in ("secret-file.png", "patient_name", "Private Name", "12345678"):
            assert value not in json.dumps(saved)
    row["test"]["mean_dab"] = {"pixel_data": [12345678]}
    assert mcp_api._cellprofiler_view(record)["blocker"] == "cellprofiler_record_invalid"


def test_only_the_computed_image_identity_fields_get_the_digest_exemption():
    # A digest made entirely of allowed amino-acid letters trips the generic
    # sequence redactor. The exception only belongs to freshly hashed bytes.
    digest = "c" * 64
    field = {"image_sha256": digest, "mask_sha256": digest, "reagent_id": digest}
    record = {"manifest_sha256": digest, "comparisons": [{"test": field, "negative": field, "positive": field}]}
    result = screen_panel._redact_fresh_assay("cellprofiler", record)
    assert result["manifest_sha256"] == digest
    for role in ("test", "negative", "positive"):
        verified = result["comparisons"][0][role]
        assert verified["image_sha256"] == verified["mask_sha256"] == digest
        assert verified["reagent_id"] == "[sequence omitted]"
    selection = {"prepared_measurements": {"cellprofiler": record}}
    stored = mcp_api._verdict({"method_selection": selection}, include_record=True)
    assert digest not in json.dumps(stored)


def test_finished_run_inventory_refuses_a_linked_image_manifest(tmp_path, monkeypatch):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    directory = tmp_path / "run"
    directory.mkdir()
    path = directory / cellprofiler_stain.MANIFEST
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symbolic links is unavailable on this filesystem")
    monkeypatch.setattr(screen, "_file_digest", lambda path: pytest.fail(f"Read external manifest bytes from {path}"))
    (entry,) = screen._supplied_inputs(directory)
    assert entry["link"] is True and entry["file"] == cellprofiler_stain.MANIFEST
    assert entry["digest"] == "cellprofiler_invalid_manifest"
