"""Attribution, positive and negative controls, and native H-DAB intensity parity."""

import hashlib
import json

import numpy as np
import pytest

from rfproteina.disease import cellprofiler_stain as stain

Image = pytest.importorskip("PIL.Image")


def write_panel(run, panel):
    (run / stain.MANIFEST).write_text(json.dumps(panel), encoding="utf-8")


@pytest.fixture
def panel(tmp_path):
    base = {
        "material": "human_frozen_tissue",
        "specimen_id": "D01",
        "tissue": "Kidney",
        "region": "Cortex",
        "assay_batch": "B01",
        "acquisition_id": "scanner01",
        "mask": "cortex.png",
    }
    test = {**base, "role": "candidate", "candidate_id": "A", "reagent_id": "lot17", "image": "test.png"}
    neg = {**base, "role": "isotype_control", "reagent_id": "iso1", "image": "negative.png"}
    pos = {
        **base,
        "role": "positive_control",
        "reagent_id": "lot17",
        "image": "positive.png",
        "specimen_id": "control01",
        "material": "assay_control",
        "tissue": "Control pellet",
        "region": "Pellet",
    }
    row = {"candidate_id": "A", "comparison_id": "slide_pair", "test": "t", "negative": "n", "positive": "p"}
    doc = {"stains": "hematoxylin_dab", "fields": {"t": test, "n": neg, "p": pos}, "comparisons": [row]}
    mask = np.array([[255, 255, 0], [0, 0, 0]], dtype=np.uint8)
    Image.fromarray(mask).save(tmp_path / "cortex.png")
    # Beer–Lambert RGB transmission with a constant hematoxylin contribution and
    # a known DAB contribution at the two selected pixels. Other pixels vary independently.
    hema = np.array([0.644, 0.717, 0.267])
    dab = np.array([0.268, 0.570, 0.776])
    for name, amount in [("test", 0.8), ("negative", 0.2), ("positive", 1.3)]:
        optical_density = 0.3 * hema / np.linalg.norm(hema) + amount * dab / np.linalg.norm(dab)
        pixels = np.zeros((2, 3, 3), dtype=np.uint8)
        pixels[0, :2, :] = np.rint(np.exp(-optical_density) * 255).astype(np.uint8)
        pixels[1, :, :] = 250
        Image.fromarray(pixels).save(tmp_path / f"{name}.png")
    write_panel(tmp_path, doc)
    return tmp_path, doc


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = []

    def measure(regions):
        # A predictable scalar for attribution tests. Scientific parity is exercised separately.
        calls.append(len(regions))
        return [
            {"roi_pixels": int(mask.sum()), "mean_dab": float(rgb[mask].mean() / 255), "median_dab": 0.5}
            for rgb, mask in regions
        ]

    monkeypatch.setattr(stain.runtime, "measure_regions", measure)
    return calls


def test_each_field_has_its_own_measurement_and_origin(panel, fake_runtime):
    run, _ = panel
    assert stain.readiness(run, "A")[:2] == (True, True)
    assert fake_runtime == []
    result = stain.measure(run, "A")
    (row,) = result["comparisons"]
    assert result["reported"] == result["attempted"] == 1 and result["status"] == "reported"
    assert (
        result["scope"] == "candidate"
        and result["manifest_sha256"] == hashlib.sha256((run / stain.MANIFEST).read_bytes()).hexdigest()
    )
    assert fake_runtime == [3] and row["positive_control_status"] == "measured"
    assert row["positive"]["specimen_id"] == "control01" and row["test"]["specimen_id"] == "D01"
    assert row["negative"]["role"] == "isotype_control" and row["test"]["roi_pixels"] == 2
    delta = row["test"]["mean_dab"] - row["negative"]["mean_dab"]
    assert row["test_minus_negative_mean_dab"] == delta and delta < 0
    assert row["test"]["image_sha256"] == hashlib.sha256((run / "test.png").read_bytes()).hexdigest()
    assert "test.png" not in json.dumps(result) and "verdict" not in row


@pytest.mark.parametrize(
    "field,value",
    [
        ("specimen_id", "D02"),
        ("tissue", "Liver"),
        ("region", "Medulla"),
        ("assay_batch", "B02"),
        ("acquisition_id", "scanner02"),
        ("material", "human_chip"),
        ("reagent_id", "lot17"),
        ("role", "positive_control"),
    ],
)
def test_negative_control_must_match_by_metadata(panel, fake_runtime, field, value):
    run, doc = panel
    doc["fields"]["n"][field] = value
    write_panel(run, doc)
    assert not stain.readiness(run, "A")[0]
    result = stain.measure(run)
    assert result["blocker"] == "cellprofiler_control_mismatch" and result["reported"] == 0 and not fake_runtime


def test_separate_chip_control_and_candidate_scope(panel, fake_runtime):
    run, doc = panel
    for key in ("t", "n"):
        doc["fields"][key]["material"] = "human_chip"
    doc["fields"]["n"]["role"] = "no_primary_control"
    doc["fields"]["n"]["reagent_id"] = "no-primary"
    doc["comparisons"][0].pop("positive")
    doc["comparisons"].append({**doc["comparisons"][0], "candidate_id": "B"})
    write_panel(run, doc)
    result = stain.measure(run)
    assert result["status"] == "partial" and result["reported"] == 1
    valid, invalid = result["comparisons"]
    assert invalid["blocker"] == "cellprofiler_control_mismatch"
    assert valid["test"]["material"] == "human_chip" and valid["positive_control_status"] == "not_supplied"
    assert "positive" not in valid
    assert stain.measure(run, "B")["status"] == "untested" and fake_runtime == [2]
    assert stain.measure(run, "unknown")["blocker"] == "cellprofiler_candidate_not_nominated"


def test_duplicate_pair_with_new_id_is_not_a_second_observation(panel, fake_runtime):
    run, doc = panel
    doc["comparisons"].append({**doc["comparisons"][0], "comparison_id": "replicate"})
    write_panel(run, doc)
    result = stain.measure(run)
    assert result["attempted"] == 2 and result["blocker"] == "cellprofiler_duplicate_pair" and not fake_runtime


def test_alias_of_pixels_cannot_act_as_negative_control(panel, fake_runtime):
    run, doc = panel
    with Image.open(run / "test.png") as img:
        img.save(run / "negative.tif")
    doc["fields"]["n"]["image"] = "negative.tif"
    write_panel(run, doc)
    (row,) = stain.measure(run)["comparisons"]
    assert row["status"] == "untested" and row["blocker"] == "cellprofiler_image_conflict"
    assert row["field_ids"]["negative"] == "n" and "test_minus_negative_mean_dab" not in row


def test_missing_named_positive_withholds_pair(panel, fake_runtime):
    run, doc = panel
    doc["fields"]["p"]["image"] = "missing.png"
    write_panel(run, doc)
    result = stain.measure(run)
    assert result["status"] == "untested" and "test_minus_negative_mean_dab" not in result["comparisons"][0]
    assert not fake_runtime


def test_reencoded_field_cannot_be_claimed_for_a_different_candidate(panel, fake_runtime):
    run, doc = panel
    with Image.open(run / "test.png") as img:
        img.save(run / "another_candidate.tif")
    doc["fields"]["t2"] = {**doc["fields"]["t"], "candidate_id": "B", "image": "another_candidate.tif"}
    doc["comparisons"].append({**doc["comparisons"][0], "candidate_id": "B", "test": "t2"})
    write_panel(run, doc)
    for name in ("A", "B"):
        result = stain.measure(run, name)
        assert result["blocker"] == "cellprofiler_image_conflict" and result["attempted"] == 1
        assert result["comparisons"][0]["candidate_id"] == name
    assert sum(fake_runtime) == 6  # Three nominated roles per call. The other candidate is never scored.


@pytest.mark.parametrize(
    "stats",
    [
        [{"roi_pixels": 2}],
        [{"roi_pixels": 10, "mean_dab": 0.4, "median_dab": 0.5}] * 3,
        [{"roi_pixels": 2, "mean_dab": float("nan"), "median_dab": 0.1}] * 3,
    ],
)
def test_incomplete_native_measurements_cannot_make_a_comparison(panel, monkeypatch, stats):
    run, _ = panel
    monkeypatch.setattr(stain.runtime, "measure_regions", lambda batch: stats)
    assert stain.measure(run)["blocker"] == "cellprofiler_measurement_failed"


def test_storage_limits_and_malformed_keys(panel, fake_runtime):
    run, doc = panel
    doc["comparisons"] *= stain.MAX_COMPARISONS + 1
    write_panel(run, doc)
    assert stain.measure(run)["blocker"] == "cellprofiler_budget_exceeded"
    (run / stain.MANIFEST).write_text('{"stains":"hematoxylin_dab","fields":{},"fields":{}}')
    assert stain.measure(run)["blocker"] == "cellprofiler_invalid_manifest"
    (run / stain.MANIFEST).unlink()
    assert stain.measure(run)["blocker"] == "cellprofiler_manifest_missing"


def test_native_cellprofiler_against_known_two_stain_transmission(panel):
    if not stain.runtime.installed():
        pytest.skip("Requires the pinned local CellProfiler image")
    run, _ = panel
    report = stain.measure(run, "A")
    assert report["status"] == "reported", report
    (row,) = report["comparisons"]
    for role, strength in (("test", 0.8), ("negative", 0.2), ("positive", 1.3)):
        expected = 1 - np.exp(-strength)
        assert row[role]["roi_pixels"] == 2
        assert row[role]["mean_dab"] == pytest.approx(expected, abs=0.012)
        assert row[role]["median_dab"] == pytest.approx(expected, abs=0.012)
    assert row["test_minus_negative_mean_dab"] == pytest.approx(np.exp(-0.2) - np.exp(-0.8), abs=0.015)


def test_one_image_can_support_two_distinct_annotated_regions(panel, fake_runtime):
    run, doc = panel
    second_mask = np.zeros((2, 3), dtype=np.uint8)
    second_mask[1] = 255
    Image.fromarray(second_mask).save(run / "medulla.png")
    for key in ("t", "n"):
        doc["fields"][key + "2"] = {**doc["fields"][key], "mask": "medulla.png", "region": "Medulla"}
    doc["comparisons"].append({"candidate_id": "A", "comparison_id": "adjacent", "test": "t2", "negative": "n2"})
    write_panel(run, doc)
    report = stain.measure(run, "A")
    assert report["reported"] == 2 and report["status"] == "reported"
    first, second = report["comparisons"]
    assert first["test"]["image_sha256"] == second["test"]["image_sha256"]
    assert first["test"]["mask_sha256"] != second["test"]["mask_sha256"]
    assert second["test"]["region"] == "Medulla" and second["test"]["roi_pixels"] == 3


def test_total_file_and_decode_budget_never_return_a_partial_positive(panel, fake_runtime, monkeypatch):
    run, _ = panel
    with monkeypatch.context() as limited:
        limited.setattr(stain, "MAX_PANEL_BYTES", 200)
        assert not stain.readiness(run, "A")[0]
        assert stain.measure(run)["blocker"] == "cellprofiler_budget_exceeded"
    monkeypatch.setattr(stain, "MAX_PANEL_PIXELS", 12)
    assert stain.readiness(run, "A")[0]  # The picker checks source sizes without decoding.
    result = stain.measure(run, "A")
    assert result["reported"] == 0 and result["blocker"] == "cellprofiler_budget_exceeded"
    assert not fake_runtime
