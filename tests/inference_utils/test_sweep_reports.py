"""End-to-end integrity of post-generation reports across configurations."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import script_utils.concatenate_sweep_results as sweep
from rfproteina.inference_utils.metrics_tables import locked_metrics_directory, write_metrics_tables
from rfproteina.utils.metrics_files import RECEIPT, file_sha256, verified_sources, wide_csv_rows
from script_utils.concatenate_sweep_results import concatenate_sweep_results

PRIMARY = "per_example_metrics.csv"
TASK = "per_task_metrics.csv"
AGGREGATE = "aggregated_metrics.csv"


def _write(path, header, *rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, strict=True))


def _config(root, name="001", stage="inference_metrics"):
    directory = root / name / stage
    _write(directory / PRIMARY, ["example_id", "dataset_name", "task", "score", "note"],
           ["0001", "NA", "null", "0.99999999999999989", 'Line one, "quoted"\nLine two'])
    for filename in (TASK, AGGREGATE):
        _write(directory / filename, ["dataset_name", "task", "n", "p.pass"], ["NA", "null", "1", "1.0000"])
    return directory


def _published(root):
    return {path.name: path.read_bytes() for path in root.glob("sweep_*.csv")}


def test_sweep_preserves_every_cell_and_configuration(tmp_path):
    _config(tmp_path)
    other = _config(tmp_path, "two")
    _write(other / PRIMARY, ["example_id", "extra", "score"], ["0001", "NA", "1e-21"])
    concatenate_sweep_results(str(tmp_path))
    rows = _read(tmp_path / ("sweep_" + PRIMARY))
    assert rows == [
        {"sweep_config": "001", "example_id": "0001", "dataset_name": "NA", "task": "null",
         "score": "0.99999999999999989", "note": 'Line one, "quoted"\nLine two', "extra": ""},
        {"sweep_config": "two", "example_id": "0001", "dataset_name": "", "task": "",
         "score": "1e-21", "note": "", "extra": "NA"},
    ]
    assert _read(tmp_path / ("sweep_" + TASK))[0]["p.pass"] == "1.0000"
    source_columns = {r["sweep_config"]: json.loads(r["source_columns"])
                      for r in _read(tmp_path / "sweep_coverage.csv") if r["table"] == PRIMARY}
    assert source_columns == {"001": ["example_id", "dataset_name", "task", "score", "note"],
                              "two": ["example_id", "extra", "score"]}


def test_malformed_late_table_preserves_the_previous_sweep(tmp_path):
    directory = _config(tmp_path)
    concatenate_sweep_results(str(tmp_path))
    before = _published(tmp_path)
    _write(directory / PRIMARY, ["example_id", "score"], ["retry", "0"])
    _write(directory / AGGREGATE, ["task", "p.pass"], ["wrong", "0", "1"])
    with pytest.raises(ValueError, match=AGGREGATE):
        concatenate_sweep_results(str(tmp_path))
    assert _published(tmp_path) == before


@pytest.mark.parametrize("unfinished", ["job", "journal"])
def test_sweep_refuses_old_results_during_retry(tmp_path, unfinished):
    directory = _config(tmp_path)
    concatenate_sweep_results(str(tmp_path))
    before = _published(tmp_path)
    if unfinished == "job":
        _write(directory / "per_example_metrics_job0.csv", ["example_id", "score"], ["retry", "0"])
    else:
        (directory / ".rfproteina-metrics-pending").mkdir()
    with pytest.raises((ValueError, RuntimeError), match="001|inference_metrics"):
        concatenate_sweep_results(str(tmp_path))
    assert _published(tmp_path) == before


def test_offline_only_sweep_and_missing_optional_summary(tmp_path):
    directory = _config(tmp_path, stage="evaluation_metrics")
    concatenate_sweep_results(str(tmp_path))
    assert _read(tmp_path / ("sweep_" + PRIMARY))[0]["example_id"] == "0001"
    (directory / TASK).unlink()
    concatenate_sweep_results(str(tmp_path))
    assert _read(tmp_path / ("sweep_" + TASK)) == []
    row = next(r for r in _read(tmp_path / "sweep_coverage.csv") if r["table"] == TASK)
    assert (row["sweep_config"], row["metrics_directory"], row["status"]) == ("001", "evaluation_metrics", "missing")


def test_an_empty_sweep_returns_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        concatenate_sweep_results(str(tmp_path))


def test_receipts_bind_each_summary_to_the_current_primary(tmp_path):
    directory = _config(tmp_path)
    primary = pd.DataFrame([{"example_id": "0001", "task": "t", "dataset_name": "ds", "score": 11}])
    summary = pd.DataFrame([{"n": 1}])
    with locked_metrics_directory(directory):
        write_metrics_tables(directory, {PRIMARY: primary, TASK: summary, AGGREGATE: summary})
    concatenate_sweep_results(str(tmp_path), expected_configs=["001"])
    before = _published(tmp_path)
    for row in _read(tmp_path / "sweep_coverage.csv"):
        assert (row["status"], row["rows"], row["sha256"]) == ("verified", "1", file_sha256(directory / row["table"]))
    data = directory / TASK
    old_time = data.stat().st_mtime_ns
    data.write_bytes(data.read_bytes().replace(b"1", b"7"))
    os.utime(data, ns=(old_time, old_time))
    with pytest.raises(ValueError, match="digest mismatch"):
        concatenate_sweep_results(str(tmp_path))
    assert _published(tmp_path) == before
    assert verified_sources(directory, file_sha256(directory / PRIMARY)) == ([], None)

    with locked_metrics_directory(directory):
        write_metrics_tables(directory, {PRIMARY: primary.assign(score=12)})
    with pytest.raises(ValueError, match="not_published"):
        concatenate_sweep_results(str(tmp_path), expected_configs=["001"])
    assert _published(tmp_path) == before
    concatenate_sweep_results(str(tmp_path))
    statuses = {r["table"]: r["status"] for r in _read(tmp_path / "sweep_coverage.csv")}
    assert statuses == {TASK: "not_published", AGGREGATE: "not_published", PRIMARY: "verified"}
    assert _read(tmp_path / ("sweep_" + AGGREGATE)) == []


def test_repeat_aggregation_carries_job_receipts_into_fresh_summaries(tmp_path):
    from rfproteina.inference_utils.aggregate_metrics import concatenate_results

    directory = tmp_path / "one" / "evaluation_metrics"
    job = directory / "per_example_metrics_job1.csv"
    _write(job, ["example_id", "dataset_name", "task", "v"], ["001", "ds", "t", "1.5"])
    concatenate_results(str(directory))
    prior, blocker = verified_sources(directory, file_sha256(directory / PRIMARY))
    assert blocker is None and len(prior) == 1
    concatenate_results(str(directory / PRIMARY))
    concatenate_sweep_results(str(tmp_path), expected_configs=["one"])
    assert verified_sources(directory, file_sha256(directory / PRIMARY)) == (prior, None)
    assert {r["status"] for r in _read(tmp_path / "sweep_coverage.csv")} == {"verified"}


def test_normal_metric_writers_keep_precision_before_sweep_export(tmp_path):
    from rfproteina.inference_utils.aggregate_metrics import concatenate_results
    from rfproteina.proteina import Proteina

    directory = tmp_path / "one" / "inference_metrics"
    writer = SimpleNamespace(_output_prefix_suffix=lambda: ("", ""), is_inference=True, design_metric_manager=None,
                             oracle_metric_manager=None, refolding_metric_manager=None, pass_criteria_metric_manager=None)
    value = 0.9999999999999999
    rows = [{"example_id": "0001", "task": "t", "dataset_name": "ds", "score": value, "note": "a\rb"}]
    Proteina._save_and_log_csv(writer, rows, directory, "per_example_metrics")
    assert next(r for r in _read(directory / AGGREGATE) if r["metric"] == "score")["value"] == str(value)
    assert _read(directory / PRIMARY)[0]["note"] == "a\rb"
    concatenate_results(str(directory))
    concatenate_sweep_results(str(tmp_path), expected_configs=["one"])
    for name in (PRIMARY, TASK, AGGREGATE):
        assert _read(tmp_path / ("sweep_" + name))[0]["score"] == str(value)


def test_explicit_job_exports_retain_source_identity(tmp_path):
    from rfproteina.inference_utils.aggregate_metrics import concatenate_results

    directory = _config(tmp_path)
    job = directory / "per_example_metrics_job7.csv"
    _write(job, ["example_id", "dataset_name", "task", "score"], ["0001", "ds", "t", "5"])
    expected = [(job.name, str(job.stat().st_mtime_ns), file_sha256(job))]
    concatenate_results(str(job))
    assert verified_sources(directory, file_sha256(directory / PRIMARY)) == (expected, None)
    exported = tmp_path / "copy" / "evaluation_metrics"
    concatenate_results(str(directory / PRIMARY), output_dir=str(exported))
    assert verified_sources(exported, file_sha256(exported / PRIMARY)) == (expected, None)
    concatenate_sweep_results(str(tmp_path), expected_configs=["001", "copy"])
    assert {r["status"] for r in _read(tmp_path / "sweep_coverage.csv")} == {"verified"}


def test_original_receipt_certifies_only_its_primary(tmp_path):
    directory = _config(tmp_path)
    digest = file_sha256(directory / PRIMARY)
    with (directory / RECEIPT).open("w", newline="") as out:
        csv.writer(out).writerows([["source", "mtime_ns", "sha256"], [PRIMARY, "", digest]])
    concatenate_sweep_results(str(tmp_path), expected_configs=["001"])
    rows = {r["table"]: r for r in _read(tmp_path / "sweep_coverage.csv")}
    assert {name: r["status"] for name, r in rows.items()} == {PRIMARY: "verified", TASK: "legacy", AGGREGATE: "legacy"}
    assert rows[TASK]["sha256"] == hashlib.sha256((directory / TASK).read_bytes()).hexdigest()
    before = _published(tmp_path)
    (directory / PRIMARY).write_bytes((directory / PRIMARY).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="metrics_provenance_mismatch"):
        concatenate_sweep_results(str(tmp_path))
    assert _published(tmp_path) == before


def test_standalone_primary_can_coexist_with_historical_reevaluation(tmp_path):
    directory = _config(tmp_path)
    primary = pd.DataFrame([{"example_id": "current"}])
    with locked_metrics_directory(directory):
        write_metrics_tables(directory, {PRIMARY: primary})
    _write(directory / "per_example_metrics_reeval.csv", ["example_id"], ["fallback"])
    concatenate_sweep_results(str(tmp_path))
    assert _read(tmp_path / ("sweep_" + PRIMARY)) == [{"sweep_config": "001", "example_id": "current"}]
    _write(directory / "per_example_metrics_job0.csv", ["example_id"], ["unpublished"])
    with pytest.raises(ValueError, match="aggregation_required"):
        concatenate_sweep_results(str(tmp_path))


def test_explicit_membership_stage_and_discovered_missing_config(tmp_path):
    _config(tmp_path)
    offline = _config(tmp_path, stage="evaluation_metrics")
    _write(offline / PRIMARY, ["example_id"], ["offline"])
    _config(tmp_path, "older")
    (tmp_path / "waiting" / "evaluation_samples").mkdir(parents=True)
    concatenate_sweep_results(str(tmp_path))
    rows = _read(tmp_path / "sweep_coverage.csv")
    assert [r["status"] for r in rows if r["sweep_config"] == "waiting"] == ["missing"] * 3
    assert all(r["rows"] == r["sha256"] == r["source_columns"] == "" for r in rows if r["sweep_config"] == "waiting")
    assert {r["example_id"] for r in _read(tmp_path / ("sweep_" + PRIMARY))} == {"0001"}
    concatenate_sweep_results(str(tmp_path), metrics_dir="evaluation_metrics", expected_configs=["001"])
    assert _read(tmp_path / ("sweep_" + PRIMARY)) == [{"sweep_config": "001", "example_id": "offline"}]
    before = _published(tmp_path)
    with pytest.raises(ValueError, match="waiting"):
        concatenate_sweep_results(str(tmp_path), expected_configs=["001", "waiting"])
    assert _published(tmp_path) == before
    _write(offline / PRIMARY, ["example_id"])
    with pytest.raises(ValueError, match=PRIMARY):
        concatenate_sweep_results(str(tmp_path), metrics_dir="evaluation_metrics", expected_configs=["001"])
    assert _published(tmp_path) == before
    concatenate_sweep_results(str(tmp_path), metrics_dir="evaluation_metrics")
    assert next(r["rows"] for r in _read(tmp_path / "sweep_coverage.csv") if r["sweep_config"] == "001"
                and r["table"] == PRIMARY) == "0"


def test_planned_sweep_rejects_empty_summary_and_duplicate_configuration(tmp_path):
    directory = _config(tmp_path)
    _write(directory / TASK, ["task", "n"])
    concatenate_sweep_results(str(tmp_path))
    before = _published(tmp_path)
    with pytest.raises(ValueError, match=TASK):
        concatenate_sweep_results(str(tmp_path), expected_configs=["001"])
    with pytest.raises(ValueError, match="unique directory names"):
        concatenate_sweep_results(str(tmp_path), expected_configs=["001", "001"])
    assert _published(tmp_path) == before


def test_empty_inference_directory_does_not_hide_offline_results(tmp_path):
    _config(tmp_path, stage="evaluation_metrics")
    inference = tmp_path / "001" / "inference_metrics"
    inference.mkdir()
    (inference / ".rfproteina-metrics.lock").touch()
    concatenate_sweep_results(str(tmp_path), expected_configs=["001"])
    assert {r["metrics_directory"] for r in _read(tmp_path / "sweep_coverage.csv")} == {"evaluation_metrics"}
    before = _published(tmp_path)
    _write(inference / "per_example_metrics_job1.csv", ["example_id"], ["incomplete"])
    with pytest.raises(ValueError, match="primary table missing"):
        concatenate_sweep_results(str(tmp_path))
    assert _published(tmp_path) == before
    concatenate_sweep_results(str(tmp_path), metrics_dir="evaluation_metrics", expected_configs=["001"])
    assert _published(tmp_path) == before


def test_sweep_keeps_wide_multiline_and_single_field_records(tmp_path):
    directory = _config(tmp_path)
    values = ["NA", "", "   ", "00000000000000000000000000000003", "L\rR\n雪" * 32000, "null", "tail"]
    _write(directory / PRIMARY, ["text"], *([value] for value in values))
    with (directory / PRIMARY).open("ab") as stream:
        stream.write(b"\r\n")
    concatenate_sweep_results(str(tmp_path))
    rows = wide_csv_rows((tmp_path / ("sweep_" + PRIMARY)).read_bytes())
    assert rows == [["sweep_config", "text"], *(["001", value] for value in values)]


def test_concurrent_source_change_and_publication_failure_preserve_all_reports(tmp_path, monkeypatch):
    directory = _config(tmp_path)
    concatenate_sweep_results(str(tmp_path))
    before = _published(tmp_path)
    original = sweep._append_table

    def change_source(table, columns, output):
        original(table, columns, output)
        if table.path.name == PRIMARY:
            _write(directory / TASK, ["n"], ["2"])

    with monkeypatch.context() as patch:
        patch.setattr(sweep, "_append_table", change_source)
        with pytest.raises(RuntimeError, match="changed during sweep"):
            concatenate_sweep_results(str(tmp_path))
    assert _published(tmp_path) == before
    replace = os.replace

    def fail_last(source, destination):
        if Path(destination) == tmp_path / ("sweep_" + PRIMARY) and Path(source).parent.name == "new":
            raise OSError("injected failure")
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_last)
        with pytest.raises(OSError, match="injected failure"):
            concatenate_sweep_results(str(tmp_path))
    assert _published(tmp_path) == before
    assert not (tmp_path / ".rfproteina-metrics-pending").exists()
    concatenate_sweep_results(str(tmp_path))
    assert _read(tmp_path / ("sweep_" + TASK)) == [{"sweep_config": "001", "n": "2"}]


@pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt", reason="POSIX batch launcher")
def test_launcher_propagates_report_failures_and_limits_membership(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    fixture = tmp_path / "local checkout"
    scripts = fixture / "scripts"
    utilities = fixture / "src" / "script_utils"
    scripts.mkdir(parents=True)
    utilities.mkdir(parents=True)
    launcher = scripts / "launch_sweep.sh"
    shutil.copyfile(repo / "scripts" / launcher.name, launcher)
    shutil.copyfile(Path(sweep.__file__), utilities / "concatenate_sweep_results.py")
    # The generated report batch runs the actual exporter. The scheduler and
    # both upstream job producers are private local stand-ins.
    (scripts / "common.sh").write_text(
        'strip_inherited_slurm_env() { :; }\nload_user_info() { SKIP_ACCOUNT=true; PARTITION=fixture; }\n'
        'find_repo_root() { dirname "$1"; }\nsource_venv() { :; }\nsetup_pythonpath() { :; }\n'
        'setup_env_vars() { :; }\nprint_header() { :; }\n'
    )
    (utilities / "sweep.py").write_text(
        'import os, sys, json\nfrom pathlib import Path\n'
        'args = sys.argv[1:]\nplan = json.loads(Path(args[args.index("--sweep-config")+1]).read_text())\n'
        'out = Path(args[args.index("--output-dir")+1] if "--output-dir" in args else plan["output_dir"])\n'
        'assert out.is_relative_to(Path(os.environ["FIXTURE_ROOT"]))\nout.mkdir(exist_ok=True, parents=True)\n'
        'for name in plan["names"]:\n    (out / (name + ".yaml")).write_text("{}")\n'
    )
    producer = scripts / "launch_generate.sh"
    producer.write_text(
        '#!/usr/bin/env python\nimport os, sys\nargs = sys.argv[1:]\nname = args[args.index("--config-name")+1]\n'
        'assert "note=two words" in args\n'
        'if name == "second" and os.environ.get("MISSING_ID"): sys.exit(0)\n'
        'print("LAUNCH_INFER_CONCAT_JOB_ID=" + ("101" if name == "first local" else "102"))\n'
    )
    producer.chmod(0o755)
    submit = scripts / "sbatch"
    submit.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$SUBMIT_LOG"\necho "Submitted batch job 103"\n')
    submit.chmod(0o755)
    base = tmp_path / "sweep results"
    base.mkdir()
    plan = tmp_path / "plan file.yaml"
    names = ["first local", "second"]
    plan.write_text(json.dumps({"output_dir": str(tmp_path / "from yaml"), "names": names}))
    log = tmp_path / "submissions"
    env = dict(os.environ, FIXTURE_ROOT=str(tmp_path), SUBMIT_LOG=str(log),
               PATH=os.pathsep.join([str(scripts), str(Path(sys.executable).parent), os.environ["PATH"]]),
               PYTHONPATH=str(repo / "src"))
    command = ["bash", str(launcher), "local", "--sweep-config", str(plan), "--delay", "0", "note=two words"]

    def run(args, **extra):
        return subprocess.run(args, cwd=fixture, env={**env, **extra}, capture_output=True, text=True, timeout=90)

    default = run(command)
    assert default.returncode == 0, default.stderr + default.stdout
    assert (tmp_path / "from yaml" / "first local.yaml").is_file() and not log.exists()
    with_base = [*command, "--base-out-dir", str(base)]
    incomplete = run(with_base, MISSING_ID="yes")
    assert incomplete.returncode != 0 and "Incomplete sweep submission" in incomplete.stderr
    assert not log.exists()
    submitted = run(with_base)
    assert submitted.returncode == 0, submitted.stderr + submitted.stdout
    batch = base / "slurm_sweep_concat_script.sh"
    assert log.read_text().splitlines() == ["--dependency=afterok:101:102", str(batch)]
    for name in [*names, "previous"]:
        _config(base, name)
    published = run(["bash", str(batch)])
    assert published.returncode == 0, published.stderr + published.stdout
    assert [r["sweep_config"] for r in _read(base / ("sweep_" + PRIMARY))] == names
    before = _published(base)
    (base / "second" / "inference_metrics" / TASK).unlink()
    failed = run(["bash", str(batch)])
    assert failed.returncode != 0 and "Incomplete planned sweep" in failed.stderr
    assert _published(base) == before
    (base / "second" / "inference_metrics").rename(base / "second" / "previous reports")
    _config(base, "second", stage="evaluation_metrics")
    wrong_stage = run(["bash", str(batch)])
    assert wrong_stage.returncode != 0 and "Incomplete planned sweep" in wrong_stage.stderr
    assert _published(base) == before


@pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt", reason="POSIX batch launcher")
@pytest.mark.parametrize(("runtime", "profile_exports"), [("none", False), ("docker", False), ("docker", True)])
def test_per_config_report_batch_returns_the_program_status(tmp_path, runtime, profile_exports):
    common = Path(__file__).resolve().parents[2] / "scripts" / "common.sh"
    checkout = tmp_path / "code tree"
    program = checkout / "src" / "rfproteina" / "inference_utils" / "aggregate_metrics.py"
    program.parent.mkdir(parents=True)
    program.write_text(
        '#!/usr/bin/env python\nimport json, os, sys\nfrom pathlib import Path\n'
        'Path(os.environ["CALL"]).write_text(json.dumps([sys.argv[1:], os.getenv("FIXTURE_MARK")]))\n'
        'sys.exit(int(os.environ["EXIT_CODE"]))\n'
    )
    program.chmod(0o755)
    # Only this executable handles the container branch. It runs the generated
    # inner shell locally after checking the boundaries of each argument.
    srun = tmp_path / "srun"
    srun.write_text(
        '#!/usr/bin/env python\nimport os, sys\na = sys.argv[1:]\ni = a.index("bash")\n'
        'assert a[:i] == ["--ntasks=1", "--container-image=image path", '
        '"--container-workdir=" + os.environ["RUN_DIR"]]\nos.execvp("bash", a[i:])\n'
    )
    srun.chmod(0o755)
    directory = tmp_path / "reports here"
    report = tmp_path / "batch.sh"
    call = tmp_path / "arguments"
    env = dict(os.environ, CONTAINER_TYPE=runtime, CLUSTER="ord", CONTAINER_IMAGE="image path",
               RUN_DIR=str(tmp_path), RUN_NAME="config with spaces", SKIP_ACCOUNT="true",
               CPU_PARTITION="report", WITH_EXPORTS=str(int(profile_exports)), CALL=str(call),
               PATH=os.pathsep.join([str(tmp_path), str(Path(sys.executable).parent), os.environ["PATH"]]))
    script = ('source "$1"\nCONTAINER_MOUNTS=()\nCONTAINER_ENV=()\n'
              'if [[ "$WITH_EXPORTS" == 1 ]]; then CONTAINER_ENV+=(\'export FIXTURE_MARK="$RUN_NAME"\'); fi\n'
              'write_metrics_batch "$2" "$3" true > "$4"\n')
    subprocess.run(["bash", "-c", script, "fixture", str(common), str(checkout), str(directory), str(report)],
                   env=env, check=True, timeout=30)
    for status in (0, 19):
        finished = subprocess.run(["bash", str(report)], env={**env, "EXIT_CODE": str(status)},
                                  capture_output=True, text=True, timeout=30)
        assert finished.returncode == status, finished.stderr + finished.stdout
        assert ("Concatenation complete!" in finished.stdout) == (status == 0)
        args, mark = json.loads(call.read_text())
        assert args == ["--input", str(directory), "--cluster-with-foldseek"]
        assert mark == ("config with spaces" if profile_exports else None)
