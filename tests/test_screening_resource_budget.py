import json
import os
import sys
from pathlib import Path

import pytest

from scripts import check_screening_resources as resources
from scripts.check_screening_resources import DISK_LIMIT, VRAM_LIMIT, footprint, main, verdict


def test_limits_are_decimal_and_have_different_endpoints():
    assert verdict(24_999_999_999, 96_000_000_000) == "within_limits"
    assert verdict(DISK_LIMIT, 0) == "over_limit"
    assert verdict(1, VRAM_LIMIT + 1) == "over_limit"
    assert verdict(DISK_LIMIT, None) == "over_limit"


def test_count_complete_roots_once_including_linked_weights(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    data = tmp_path / "weights"
    data.write_bytes(b"12345")
    (package / "model").symlink_to(data)
    (package / "self").symlink_to(package, target_is_directory=True)
    os.link(data, package / "weights_alias")
    assert footprint([package, data, package]) == 5


def test_missing_asset_or_peak_cannot_certify_a_program(tmp_path, capsys):
    args = ["example", "--root", str(tmp_path)]
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "unverified"
    assert main([*args, "--cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["vram_bytes"] == 0
    assert main([*args, "--cpu", "--root", str(tmp_path / "missing")]) == 1
    assert json.loads(capsys.readouterr().out)["installed_bytes"] is None


def test_partial_inventory_preserves_proven_disk_violation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(resources, "DISK_LIMIT", 4)
    known = tmp_path / "known"
    known.write_bytes(b"1234")
    assert main(["partial", "--root", str(known), "--root", str(tmp_path / "missing"), "--cpu"]) == 1
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "over_limit" and record["installed_bytes"] is None
    assert record["installed_bytes_lower_bound"] == 4


def test_cpu_command_is_run_with_cuda_hidden_and_charged_for_created_assets(tmp_path, monkeypatch, capfd):
    executable_size = Path(sys.executable).stat().st_size
    monkeypatch.setattr(resources, "DISK_LIMIT", executable_size + 3)
    command = [
        sys.executable, "-c", "import os, pathlib, sys; assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'; "
        "pathlib.Path(sys.argv[1]).write_bytes(b'123'); print('workload output')", str(tmp_path / "new-asset"),
    ]
    args = ["example", "--root", str(tmp_path), "--cpu", "--run", *command]
    assert main(args) == 1
    out, err = capfd.readouterr()
    record = json.loads(out)
    assert "workload output" in err
    assert record["installed_bytes_before_run"] == executable_size
    assert record["installed_bytes"] == executable_size + 3
    assert record["workload_exit_code"] == 0 and record["status"] == "over_limit"
    assert record["vram_evidence"] == "cpu_cuda_hidden"
    assert main(args) == 1
    assert json.loads(capfd.readouterr().out)["workload_exit_code"] is None


@pytest.mark.parametrize(("capacity", "exit_code", "status"), [
    (VRAM_LIMIT, 0, "within_limits"), (VRAM_LIMIT, 3, "unverified"), (VRAM_LIMIT + 1, None, "unverified"),
])
def test_cuda_bound_requires_a_completed_workload(tmp_path, monkeypatch, capfd, capacity, exit_code, status):
    monkeypatch.setattr(resources, "_cuda_capacity", lambda: capacity)
    result = main(["example", "--root", str(tmp_path), "--cuda", "--run", sys.executable,
                   "-c", f"raise SystemExit({exit_code or 0})"])
    record = json.loads(capfd.readouterr().out)
    assert result == (status != "within_limits")
    assert record["status"] == status and record["workload_exit_code"] == exit_code
    assert record["vram_bytes"] is None and record["vram_upper_bound_bytes"] == capacity
    assert record["vram_evidence"] == "visible_cuda_capacity"


def test_missing_assets_or_device_prevent_workload_start(tmp_path, monkeypatch, capfd):
    def unavailable():
        raise ValueError("no visible devices")

    monkeypatch.setattr(resources, "_cuda_capacity", unavailable)
    for root, reason in ((tmp_path, "cuda_capacity_unavailable"),
                         (tmp_path / "missing", "installed_footprint_unverified")):
        assert main(["example", "--root", str(root), "--cuda", "--run", sys.executable, "-c", "pass"]) == 1
        record = json.loads(capfd.readouterr().out)
        assert record["blocker"] == reason and record["workload_exit_code"] is None


def test_cuda_contact_geometry_on_two_point_grids():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A local CUDA device is required for the geometry workload.")
    from rfproteina.metrics.physics.contact_ms.gpu_cms import TorchCMS

    with torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()
        ticks = torch.arange(4, device="cuda", dtype=torch.float32) * 2
        plane = torch.cartesian_prod(ticks, ticks)
        points = torch.cat([torch.zeros((len(plane), 1), device="cuda"), plane], dim=1)
        radii = torch.full((len(points),), 1.7, device="cuda")
        metric = TorchCMS(device="cuda")
        totals = []
        for separation in (3.0, 40.0):
            target = points + points.new_tensor([separation, 0.0, 0.0])
            total, per_point = metric.compute(points, radii, target, radii)
            assert total.is_cuda and per_point.is_cuda and torch.isfinite(per_point).all()
            torch.testing.assert_close(total, per_point.sum(), rtol=1e-4, atol=1e-5)
            totals.append(total.item())
        assert totals[0] > 0 and totals[1] == 0
        print(json.dumps({"contact": totals[0], "separated": totals[1],
                          "torch_reserved_peak_bytes": torch.cuda.max_memory_reserved()}))
