"""Boundary and parity checks for the pinned native brightfield stain modules."""

import hashlib
import json
import struct
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from rfproteina.disease import cellprofiler_runtime as cp


def _field():
    rgb = np.array([[[255, 255, 255], [175, 120, 80]], [[150, 160, 190], [240, 230, 220]]], dtype=np.uint8)
    return rgb, np.array([[False, True], [False, False]])


def _report(regions):
    _, expected = cp._pack(regions)
    rows = [
        {"region_sha256": digest, "roi_pixels": count, "mean_dab": 0.3, "median_dab": 0.25}
        for digest, count in expected
    ]
    return {"version": cp.VERSION, "regions": rows}, expected


def test_wire_identity_binds_rectangle_decoded_rgb_and_mask_in_order():
    rgb, mask = _field()
    message, expected = cp._pack([(rgb[:, ::-1], mask[:, ::-1]), (rgb, ~mask)])
    fields = [
        struct.pack(">II", 2, 2) + data.tobytes() + region.tobytes()
        for data, region in [(rgb[:, ::-1], mask[:, ::-1]), (rgb, ~mask)]
    ]
    assert message == struct.pack(">I", 2) + b"".join(fields)
    assert expected == [(hashlib.sha256(raw).hexdigest(), count) for raw, count in zip(fields, (1, 3), strict=True)]


@pytest.mark.parametrize("problem", ["dtype", "rgba", "mask_shape", "mask_type", "empty", "pixels", "batch", "none"])
def test_invalid_fields_rejected_before_engine(monkeypatch, problem):
    rgb, mask = _field()
    if problem == "dtype":
        rgb = rgb.astype(np.uint16)
    elif problem == "rgba":
        rgb = np.zeros((2, 2, 4), np.uint8)
    elif problem == "mask_shape":
        mask = mask[0:1, :]
    elif problem == "mask_type":
        mask = mask.astype(np.uint8)
    elif problem == "empty":
        mask.fill(False)
    elif problem == "pixels":
        monkeypatch.setattr(cp, "MAX_PIXELS", 3)
    batch = [] if problem == "none" else [(rgb, mask)] * (5 if problem == "batch" else 1)
    monkeypatch.setattr(cp, "_engine", lambda: pytest.fail("invalid data reached the engine"))
    with pytest.raises(cp.CellProfilerError, match="cellprofiler_measurement_failed"):
        cp.measure_regions(batch)


@pytest.mark.parametrize(
    "problem", ["order", "pixels", "fractional_count", "version", "missing", "extra", "bool", "nan", "range", "json"]
)
def test_malformed_or_misattributed_native_reply_is_withheld(problem):
    rgb, mask = _field()
    record, expected = _report([(rgb, mask), (rgb, ~mask)])
    rows = record["regions"]
    if problem == "order":
        rows.reverse()
    elif problem == "pixels":
        rows[0]["roi_pixels"] = 4
    elif problem == "fractional_count":
        rows[0]["roi_pixels"] = 1.0
    elif problem == "version":
        record["version"] = "4.2.7"
    elif problem == "missing":
        rows.pop()
    elif problem == "extra":
        rows += rows
    elif problem == "bool":
        rows[0]["mean_dab"] = True
    elif problem == "nan":
        rows[0]["median_dab"] = float("nan")
    elif problem == "range":
        rows[0]["mean_dab"] = 1.01
    raw = b"broken" if problem == "json" else json.dumps(record).encode()
    with pytest.raises(cp.CellProfilerError, match="cellprofiler_measurement_failed"):
        cp._decode(raw, expected)


def test_engine_requires_pinned_repository_digest_for_linux_amd64(monkeypatch):
    monkeypatch.setattr(cp, "local_engines", lambda: iter([["docker", "a"], ["docker", "b"], ["docker", "c"]]))
    outputs = iter(
        [
            f'linux/arm64 ["{cp.IMAGE}"]',
            'linux/amd64 ["other/image@sha256:1234"]',
            f'linux/amd64 ["docker.io/{cp.IMAGE}"]',
        ]
    )
    monkeypatch.setattr(
        cp.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=next(outputs).encode())
    )
    assert cp._engine() == ["docker", "c"]


def test_missing_engine_abstains(monkeypatch):
    monkeypatch.setattr(cp, "_engine", lambda: None)
    with pytest.raises(cp.CellProfilerError, match="cellprofiler_runtime_unavailable"):
        cp.measure_region(*_field())


def test_worker_timeout_removes_only_its_own_container(monkeypatch):
    monkeypatch.setattr(cp, "_engine", lambda: ["docker"])
    commands = []

    def invoke(cmd, **kwargs):
        commands.append(cmd)
        if len(commands) == 1:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cp.subprocess, "run", invoke)
    with pytest.raises(cp.CellProfilerError, match="cellprofiler_execution_timeout"):
        cp.measure_region(*_field())
    run, cleanup = commands
    assert cp.IMAGE in run and "--network=none" in run and "--pull=never" in run
    assert cleanup == ["docker", "rm", "--force", run[run.index("--name") + 1]]


def test_native_modules_agree_with_independent_hdab_reference_for_two_regions():
    if not cp.installed():
        pytest.skip("Requires the pinned CellProfiler image on a local Docker engine")
    rgb, mask = _field()
    rows = cp.measure_regions([(rgb, mask), (rgb, ~mask)])
    # Independently resolve the published two-dye optical-density basis by least squares.
    dyes = np.array([[0.644, 0.717, 0.267], [0.268, 0.570, 0.776]])
    dyes /= np.linalg.norm(dyes, axis=1, keepdims=True)
    eps = 1 / 512
    log_rgb = np.log(rgb.astype(float) / 255 + eps)
    _, dab_log = np.linalg.lstsq(dyes.T, log_rgb.reshape(-1, 3).T, rcond=None)[0]
    dab = 1 - np.clip(np.exp(dab_log).reshape(mask.shape) - eps, 0, 1)
    for row, region in zip(rows, (mask, ~mask), strict=True):
        assert row["roi_pixels"] == np.count_nonzero(region)
        assert row["mean_dab"] == pytest.approx(np.mean(dab[region]), abs=3e-7)
        assert row["median_dab"] == pytest.approx(np.median(dab[region]), abs=3e-7)
