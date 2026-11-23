"""Post-generation CSV integrity under retries, failed writes, and exports."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from textwrap import dedent
from threading import Barrier
from types import SimpleNamespace

import pandas as pd
import pytest

from rfproteina.inference_utils import aggregate_metrics as aggregation
from rfproteina.inference_utils.metrics_tables import locked_metrics_directory, write_metrics_tables


def _job(path, example, score):
    pd.DataFrame([{"dataset_name": "ds", "task": "t", "example_id": example, "score": score}]).to_csv(path, index=False)


def _outputs(directory):
    return {file.name: file.read_bytes() for file in directory.glob("*.csv") if "_job" not in file.name}


@pytest.mark.parametrize("failure", ["calculation", "serialization", "replacement"])
def test_failed_refresh_preserves_published_tables_and_pending_job(tmp_path, monkeypatch, failure):
    _job(tmp_path / "per_example_metrics_job0.csv", "old", 1)
    aggregation.concatenate_results(str(tmp_path))
    previous = _outputs(tmp_path)
    new_job = tmp_path / "per_example_metrics_job1.csv"
    _job(new_job, "new", 9)
    contents = new_job.read_bytes()

    if failure == "calculation":
        original = aggregation.aggregate_metrics_from_per_example_df

        def calculate(frame, **kwargs):
            if kwargs.get("group_by") == "task":
                raise ValueError("test interrupted summary")
            return original(frame, **kwargs)

        monkeypatch.setattr(aggregation, "aggregate_metrics_from_per_example_df", calculate)
    elif failure == "serialization":
        original = pd.DataFrame.to_csv

        def write(frame, path, **kwargs):
            if Path(path).name == "per_task_metrics.csv":
                Path(path).write_bytes(b"incomplete")
                raise OSError("test interrupted CSV write")
            return original(frame, path, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_csv", write)
    else:
        import os

        original = os.replace

        def replace(source, destination):
            if Path(destination) == tmp_path / "per_example_metrics.csv":
                raise OSError("test interrupted replacement")
            return original(source, destination)

        monkeypatch.setattr(os, "replace", replace)

    with pytest.raises((ValueError, OSError), match="test interrupted"):
        aggregation.concatenate_results(str(tmp_path))
    assert _outputs(tmp_path) == previous
    assert new_job.read_bytes() == contents
    assert not (tmp_path / "individual_job_results" / new_job.name).exists()

    monkeypatch.undo()
    aggregation.concatenate_results(str(tmp_path))
    assert pd.read_csv(tmp_path / "aggregated_metrics.csv").at[0, "n"] == 2
    assert (tmp_path / "individual_job_results" / new_job.name).read_bytes() == contents


def test_export_preserves_source_shards_archives_chains_and_destination_jobs(tmp_path, monkeypatch):
    from rfproteina.inference_utils import foldseek

    source, export = tmp_path / "source", tmp_path / "export"
    archive = source / "individual_job_results"
    archive.mkdir(parents=True)
    export.mkdir()
    external_job = export / "aggregated_metrics_job99.csv"
    external_job.write_bytes(b"unrelated")
    chains = foldseek.chain_cif_dir(source, "ds", "t")
    chains.mkdir(parents=True)
    for example, location in (("old", archive), ("new", source)):
        chain = chains / f"{example}_chain.cif"
        chain.write_text(f"data_{example}\n")
        pd.DataFrame([{"example_id": example, "dataset_name": "ds", "task": "t", "output_base_dir": str(source),
                       "designed_chain_path": str(chain)}]).to_csv(location / "per_example_metrics_job0.csv", index=False)
    monkeypatch.setattr(foldseek, "cluster_structures", lambda *a, **kw: {"old": 0, "new": 1})
    prior = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}

    for _ in range(2):
        aggregation.concatenate_results(str(source), output_dir=str(export), cluster_with_foldseek=True)
        assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == prior
        assert external_job.read_bytes() == b"unrelated"
        result = pd.read_csv(export / "per_example_metrics.csv").set_index("example_id")
        assert result.foldseek_cluster_id.to_dict() == {"old": 0, "new": 1}


def test_literal_identifiers_and_output_trees_survive_retries(tmp_path):
    rows = [{"task": "NA", "dataset_name": "null", "example_id": example, "output_base_dir": tree, "score": score}
            for example, tree, score in [("NA", "one", 1), ("NA", "two", 2), ("0001", "one", 3), ("", "one", 4)]]
    pd.DataFrame(rows).to_csv(tmp_path / "per_example_metrics_job0.csv", index=False)
    aggregation.concatenate_results(str(tmp_path))
    pd.DataFrame([dict(rows[0], score=5)]).to_csv(tmp_path / "per_example_metrics_job1.csv", index=False)
    for _ in range(2):
        aggregation.concatenate_results(str(tmp_path))
        result = pd.read_csv(tmp_path / "per_example_metrics.csv", keep_default_na=False, dtype=str)
        assert sorted(zip(result.example_id, result.output_base_dir, result.score, strict=True)) == [
            ("", "one", "4"), ("0001", "one", "3"), ("NA", "one", "5"), ("NA", "two", "2")]
        tasks = pd.read_csv(tmp_path / "per_task_metrics.csv", keep_default_na=False)
        assert tasks[["dataset_name", "task", "n"]].values.tolist() == [["null", "NA", 4]]


def test_equivalent_metric_tags_keep_missing_attempt_in_denominator(tmp_path):
    rows = [{"dataset_name": "ds", "task": "t", "example_id": str(i), "metrics_tags": tags, "p.pass": flag}
            for i, (tags, flag) in enumerate([("b,a", 1), ("['a', 'b', 'a']", None), ("c", None)])]
    pd.DataFrame(rows).to_csv(tmp_path / "per_example_metrics_job0.csv", index=False)
    aggregation.concatenate_results(str(tmp_path))
    for name in ("aggregated_metrics.csv", "per_task_metrics.csv"):
        assert pd.read_csv(tmp_path / name).at[0, "p.pass"] == 0.5


def test_empty_probability_report_replaces_stale_values(tmp_path):
    _job(tmp_path / "per_example_metrics_job0.csv", "new", 1)
    report = tmp_path / "sequence_probability_errors.csv"
    report.write_text("dataset_name,task,n,absolute_mass_error,mean_absolute_error,median_absolute_error\nds,t,1,0,0,0\n")
    aggregation.concatenate_results(str(tmp_path))
    frame = pd.read_csv(report)
    assert frame.empty
    assert {"dataset_name", "task", "n", "mean_absolute_error"} <= set(frame)


def test_probability_errors_include_partial_and_zero_coverage(monkeypatch):
    def parse(job):
        dataset, task, path = job
        if path == "unreadable":
            return [], "test read failure"
        if path == "soft":
            return [(dataset, task, "A", 1, "", "X", 0.25, 1)], None
        return [], None

    monkeypatch.setattr(aggregation, "_parse_sequence_probability_observations", parse)
    rows = [("a", "soft"), ("a", "unreadable"), ("a", None), ("a", "plain"), ("b", "unreadable")]
    report = aggregation.aggregate_sequence_probability_errors(
        pd.DataFrame(rows, columns=["task", "designed_structure_path"]).assign(dataset_name="ds"), workers=1
    ).set_index("task")
    cols = ["n_designs", "n_paths", "n_readable", "n_with_soft_conditions", "n"]
    assert report.loc["a", cols].tolist() == [4, 3, 2, 1, 1]
    assert report.loc["a", "mean_absolute_error"] == pytest.approx(0.75)
    assert report.loc["b", cols].tolist() == [1, 1, 0, 0, 0]
    assert pd.isna(report.loc["b", "mean_absolute_error"])


@pytest.mark.parametrize("stop_at", ["old.csv", "committed"])
def test_terminated_writer_recovers_on_next_lock(tmp_path, stop_at):
    for name in ("old.csv", "primary.csv"):
        (tmp_path / name).write_text("value\n1\n")
    code = dedent("""
        import os, sys
        from pathlib import Path
        import pandas as pd
        from rfproteina.inference_utils.metrics_tables import locked_metrics_directory, write_metrics_tables

        directory, stop_at = Path(sys.argv[1]), sys.argv[2]
        original = os.replace
        def replace(src, dst):
            original(src, dst)
            if Path(dst).name == stop_at:
                os._exit(73)
        with locked_metrics_directory(directory):
            os.replace = replace
            frame = pd.DataFrame({"value": [2]})
            write_metrics_tables(directory, dict.fromkeys(("new.csv", "old.csv", "primary.csv"), frame),
                                 archive_replaced="individual_job_results")
    """)
    proc = subprocess.run([sys.executable, "-c", code, str(tmp_path), stop_at], timeout=90)
    assert proc.returncode == 73
    with locked_metrics_directory(tmp_path):
        files = {p.name: pd.read_csv(p).value.item() for p in tmp_path.glob("*.csv")}
        archive = tmp_path / "individual_job_results"
        if stop_at == "committed":
            assert files == {"old.csv": 2, "new.csv": 2, "primary.csv": 2}
            assert {p.name: pd.read_csv(p).value.item() for p in archive.glob("*.csv")} == {
                "old.csv": 1, "primary.csv": 1,
            }
        else:
            assert files == {"old.csv": 1, "primary.csv": 1}
            assert not archive.exists()


def test_concurrent_job_replacements_retain_each_revision(tmp_path):
    barrier = Barrier(2)
    name = "per_example_metrics_job0.csv"
    _job(tmp_path / name, "initial", 1)

    def write(example, score):
        frame = pd.DataFrame([{"dataset_name": "ds", "task": "t", "example_id": example, "score": score}])
        barrier.wait(timeout=20)
        with locked_metrics_directory(tmp_path):
            write_metrics_tables(tmp_path, {name: frame}, archive_replaced="individual_job_results")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(write, "first", 2)
        second = pool.submit(write, "second", 3)
        first.result(timeout=30)
        second.result(timeout=30)
    unrelated = tmp_path / "aggregated_metrics_job99.csv"
    unrelated.write_bytes(b"separate job")
    aggregation.concatenate_results(str(tmp_path))
    result = pd.read_csv(tmp_path / "per_example_metrics.csv").set_index("example_id")
    assert result.score.to_dict() == {"initial": 1, "first": 2, "second": 3}
    assert len(list((tmp_path / "individual_job_results").glob("*.csv"))) == 3
    assert unrelated.read_bytes() == b"separate job"


def test_metadata_only_job_retry_keeps_design_and_clears_stale_summary(tmp_path):
    from rfproteina.proteina import Proteina

    model = SimpleNamespace(is_inference=True, design_metric_manager=None, oracle_metric_manager=None,
                            refolding_metric_manager=None, pass_criteria_metric_manager=None,
                            _output_prefix_suffix=lambda: ("", "_job0"))
    first = {"dataset_name": "ds", "task": "t", "example_id": "old"}
    Proteina._save_and_log_csv(model, [dict(first, score=2)], tmp_path, "per_example_metrics")
    Proteina._save_and_log_csv(model, [dict(first, example_id="new")], tmp_path, "per_example_metrics")
    assert pd.read_csv(tmp_path / "aggregated_metrics_job0.csv").empty
    assert pd.read_csv(tmp_path / "per_example_metrics_job0.csv").example_id.tolist() == ["new"]
    aggregation.concatenate_results(str(tmp_path))
    assert sorted(pd.read_csv(tmp_path / "per_example_metrics.csv").example_id) == ["new", "old"]


def test_export_detects_changing_source(tmp_path, monkeypatch):
    source, export = tmp_path / "source", tmp_path / "export"
    source.mkdir()
    _job(source / "per_example_metrics_job0.csv", "old", 1)
    aggregation.concatenate_results(str(source), output_dir=str(export))
    previous = _outputs(export)
    original = aggregation.aggregate_sequence_probability_errors

    def new_job_during_export(frame, **kwargs):
        _job(source / "per_example_metrics_job1.csv", "new", 3)
        return original(frame, **kwargs)

    monkeypatch.setattr(aggregation, "aggregate_sequence_probability_errors", new_job_during_export)
    with pytest.raises(RuntimeError, match="changed during export"):
        aggregation.concatenate_results(str(source), output_dir=str(export))
    assert _outputs(export) == previous


def test_legacy_anonymous_jobs_and_metadata_are_not_averaged_as_metrics(tmp_path):
    primary = tmp_path / "per_example_metrics.csv"
    pd.DataFrame([{"score": 7, "seed": 42, "generation.recycles": 2}]).to_csv(primary, index=False)
    pd.DataFrame([{"score": 9, "seed": 45, "generation.recycles": 8}] * 2).to_csv(
        tmp_path / "per_example_metrics_job0.csv", index=False
    )
    for _ in range(2):
        aggregation.concatenate_results(str(tmp_path))
        assert sorted(pd.read_csv(primary).score) == [7, 9, 9]
        summary = pd.read_csv(tmp_path / "aggregated_metrics.csv")
        assert summary.at[0, "n"] == 3
        assert "seed" not in summary
        assert "generation.recycles" not in summary
        assert pd.read_csv(tmp_path / "per_task_metrics.csv").at[0, "n"] == 3


def test_offline_evaluator_reused_job_id_and_literal_task(tmp_path):
    from omegaconf import OmegaConf

    from rfproteina.evaluate import run_evaluate

    out = tmp_path / "out"
    task_csv = tmp_path / "tasks.csv"
    for index in range(2):
        folder = tmp_path / str(index) / "inputs"
        folder.mkdir(parents=True)
        path = folder / f"input{index}.cif"
        path.write_text("invalid test structure")
        pd.DataFrame([{"designed_structure_path": str(path), "task": "NA"}] * 2).to_csv(task_csv, index=False)
        cfg = OmegaConf.create({"inputs": str(folder), "out_dir": str(out), "metrics": {}, "verbose": False,
                                "task_csv": str(task_csv), "job_id": 0, "gen_njobs": 1})
        run_evaluate(cfg)
    directory = out / "evaluation_metrics"
    aggregation.concatenate_results(str(directory), sequence_probability_workers=1)
    rows = pd.read_csv(directory / "per_example_metrics.csv", keep_default_na=False)
    assert len(rows) == 2
    assert set(rows.task) == {"NA"}
    assert set(rows.evaluation_error) == {"structure_unreadable"}
    assert pd.read_csv(directory / "sequence_probability_errors.csv").at[0, "n_readable"] == 0
    before = _outputs(directory)
    pd.DataFrame([{"designed_structure_path": str(path), "task": label} for label in ["NA", "conflict"]]).to_csv(
        task_csv, index=False
    )
    with pytest.raises(ValueError, match="one nonempty task label"):
        run_evaluate(cfg)
    assert _outputs(directory) == before


def test_independent_job_cluster_labels_cannot_establish_joint_diversity(tmp_path):
    for job, example in enumerate(["one", "", ""]):
        pd.DataFrame([{"dataset_name": "ds", "task": "t", "example_id": example, "foldseek_cluster_id": 0,
                       "cluster_coverage": 1.0, "cluster_diversity": 1.0}]).to_csv(
            tmp_path / f"per_example_metrics_job{job}.csv", index=False
        )
    for _ in range(2):
        aggregation.concatenate_results(str(tmp_path))
        rows = pd.read_csv(tmp_path / "per_example_metrics.csv")
        assert len(rows) == 3
        assert rows.foldseek_cluster_id.isna().all()
        report = pd.read_csv(tmp_path / "aggregated_metrics.csv")
        assert report.at[0, "cluster_coverage"] == 0
        assert pd.isna(report.at[0, "cluster_diversity"])
    for job in range(3):
        path = tmp_path / "individual_job_results" / f"per_example_metrics_job{job}.csv"
        assert pd.read_csv(path).at[0, "foldseek_cluster_id"] == 0


def test_joint_cluster_evidence_survives_cleanup_and_superseded_cohort(tmp_path, monkeypatch):
    from rfproteina.inference_utils import foldseek

    chains = foldseek.chain_cif_dir(tmp_path, "ds", "t")
    chains.mkdir(parents=True)
    row = {"dataset_name": "ds", "task": "t", "output_base_dir": str(tmp_path), "p.pass": 1}
    for job in range(2):
        path = chains / f"{job}_chain.cif"
        path.write_text(f"data_{job}\n")
        pd.DataFrame([dict(row, example_id=str(job), designed_chain_path=str(path))]).to_csv(
            tmp_path / f"per_example_metrics_job{job}.csv", index=False
        )
    monkeypatch.setattr(foldseek, "cluster_structures", lambda directory, **kw: {
        p.name.split("_")[0]: i for i, p in enumerate(sorted(directory.glob("*_chain.cif")))
    })
    aggregation.concatenate_results(str(tmp_path), cluster_with_foldseek=True)
    assert not list(chains.glob("*.cif"))
    aggregation.concatenate_results(str(tmp_path))
    primary = tmp_path / "per_example_metrics.csv"
    assert sorted(pd.read_csv(primary).foldseek_cluster_id) == [0, 1]
    before = _outputs(tmp_path)
    with pytest.raises(ValueError, match="previously clustered run"):
        aggregation.concatenate_results(str(tmp_path), cluster_with_foldseek=True)
    assert _outputs(tmp_path) == before

    pd.DataFrame([dict(row, example_id="extra")]).to_csv(tmp_path / "per_example_metrics_job2.csv", index=False)
    for _ in range(2):
        aggregation.concatenate_results(str(tmp_path))
        rows = pd.read_csv(primary)
        assert len(rows) == 3 and rows.foldseek_cluster_id.isna().all()
        history = tmp_path / "previous_reports" / primary.name
        assert history.read_bytes() == before[primary.name]
        assert list(history.parent.glob("*per_example_metrics*.csv")) == [history]


def test_two_output_trees_keep_coverage_but_have_no_common_cluster_scale():
    from rfproteina.inference_utils.foldseek import cluster_summary_for_group

    rows = pd.DataFrame({"output_base_dir": ["first", "second"], "task": ["t", "t"],
                         "foldseek_cluster_id": [0, 0], "p.clustered_pass_rate": [1, 1],
                         "cluster_diversity": [1, 1]})
    result = cluster_summary_for_group(rows)
    assert result["cluster_coverage"] == 1
    assert pd.isna(result["cluster_diversity"])
    assert pd.isna(result["p.clustered_pass_rate"])


def test_fresh_complete_cluster_observation_replaces_previous_labels(tmp_path):
    import os

    primary = tmp_path / "per_example_metrics.csv"
    job = tmp_path / "per_example_metrics_job0.csv"
    for index, assignment in enumerate((4, 2, None)):
        pd.DataFrame([{"dataset_name": "ds", "task": "t", "example_id": "one", "score": 1,
                       "foldseek_cluster_id": assignment, "cluster_coverage": float(assignment is not None)}]).to_csv(
            job, index=False
        )
        # Some mounted filesystems give successive writes the same timestamp.
        timestamp = max(job.stat().st_mtime_ns, primary.stat().st_mtime_ns if primary.exists() else 0) + index
        os.utime(job, ns=(timestamp, timestamp))
        aggregation.concatenate_results(str(tmp_path))
        observed = pd.read_csv(primary).at[0, "foldseek_cluster_id"]
        if assignment is None:
            assert pd.isna(observed)
        else:
            assert observed == assignment


@pytest.mark.parametrize("empty_directory", [False, True])
def test_missing_inputs_cannot_report_a_successful_refresh(tmp_path, empty_directory):
    source = tmp_path / "source"
    if empty_directory:
        source.mkdir()
    output = tmp_path / "export"
    output.mkdir()
    _job(output / "per_example_metrics.csv", "previous", 1)
    before = _outputs(output)
    with pytest.raises(FileNotFoundError):
        aggregation.concatenate_results(str(source), output_dir=str(output))
    assert _outputs(output) == before


@pytest.mark.parametrize("payload", [
    "example_id,task,p.pass\nbad,t,0,1\n",
    "example_id,task,p.pass\nfirst,t,1\nsecond,t\n",
    "example_id,task,p.pass,p.pass\nfirst,t,1,0\n",
    'example_id,task,p.pass\nfirst,"closed"junk,1\n',
    "example_id,task,p.pass\nfirst,t,1\x00unknown\n",
])
def test_malformed_shard_cannot_publish_or_displace_valid_evidence(tmp_path, payload):
    _job(tmp_path / "per_example_metrics_job0.csv", "first", 1)
    aggregation.concatenate_results(str(tmp_path))
    previous = _outputs(tmp_path)
    shard = tmp_path / "per_example_metrics_job1.csv"
    shard.write_bytes(payload.encode())
    with pytest.raises(ValueError, match="per_example_metrics_job1"):
        aggregation.concatenate_results(str(tmp_path))
    assert _outputs(tmp_path) == previous
    assert shard.read_bytes() == payload.encode()


def test_valid_quoted_and_wide_cells_keep_identifiers_blanks_and_precision(tmp_path):
    from rfproteina.inference_utils.metrics_tables import read_metrics_csv

    path = tmp_path / "table.csv"
    note = 'embedded, quote " and a newline\n' + "x" * 150_000
    pd.DataFrame([{"example_id": "0001", "task": "NA", "note": note,
                   "score": 1.0000000000000002, "missing": None}]).to_csv(path, encoding="utf-8-sig", index=False)
    row = read_metrics_csv(path).iloc[0]
    assert (row.example_id, row.task, row.note, row.score) == ("0001", "NA", note, 1.0000000000000002)
    assert pd.isna(row["missing"])


@pytest.mark.parametrize("cleanup", [False, True])
def test_screen_requires_current_shards_even_with_same_size_and_timestamp(tmp_path, cleanup):
    import os

    from rfproteina.disease.screen import screen_generation

    directory = tmp_path / "inference_metrics"
    directory.mkdir()
    shard = directory / "per_example_metrics_job0.csv"
    content = "example_id,dataset_name,task,ppi.pass\nsample,ds,t,1\n"
    shard.write_text(content)

    def screen():
        return screen_generation(tmp_path, binder_kind="other", target_id="")

    assert screen()["metrics_blocker"] == "metrics_aggregation_required"
    aggregation.concatenate_results(str(directory), cleanup=cleanup)
    assert screen()["ppi_pass"] is True
    original = directory / "individual_job_results" / shard.name if cleanup else shard
    mtime = original.stat().st_mtime_ns
    original.write_text(content.replace(",1\n", ",0\n"))
    os.utime(original, ns=(mtime, mtime))
    refused = screen()
    assert refused["metrics_blocker"] == "metrics_aggregation_required"
    assert refused["ppi_pass"] == "absent"
    aggregation.concatenate_results(str(directory), cleanup=cleanup)
    assert screen()["ppi_pass"] is False


def test_screen_withholds_tables_during_interrupted_publication(tmp_path):
    from rfproteina.disease.screen import screen_generation

    directory = tmp_path / "evaluation_metrics"
    directory.mkdir()
    (directory / "per_example_metrics.csv").write_text("example_id,ppi.pass\nsample,1\n")
    (directory / ".rfproteina-metrics-pending").mkdir()
    record = screen_generation(tmp_path, binder_kind="other", target_id="")
    assert record["metrics_blocker"] == "metrics_publication_pending"
    assert record["ppi_blocker"] == record["refold"]["refold_blocker"] == record["metrics_blocker"]
    assert record["ppi_pass"] == "absent"


def test_receipt_checks_primary_and_shard_multiplicity_and_travels_with_table(tmp_path):
    import shutil

    from rfproteina.disease.screen import screen_generation
    from rfproteina.utils.metrics_files import PRIMARY, RECEIPT

    source = tmp_path / "source" / "inference_metrics"
    source.mkdir(parents=True)
    job = source / "per_example_metrics_job0.csv"
    job.write_text("example_id,task,ppi.pass\nx,t,1\n")
    aggregation.concatenate_results(str(source))
    portable = tmp_path / "portable" / "evaluation_metrics"
    portable.mkdir(parents=True)
    for name in (PRIMARY, RECEIPT):
        shutil.copyfile(source / name, portable / name)

    def screen(directory):
        return screen_generation(directory.parent, binder_kind="other", target_id="")

    assert screen(portable)["ppi_pass"] is True
    archived = source / "individual_job_results" / job.name
    shutil.copy2(archived, archived.with_stem(archived.stem + "_revision1"))
    assert screen(source)["metrics_blocker"] == "metrics_aggregation_required"
    with pytest.raises(ValueError, match="metrics_aggregation_required"):
        aggregation.concatenate_results(str(source / PRIMARY), output_dir=str(tmp_path / "export"))
    aggregation.concatenate_results(str(source))
    assert screen(source)["ppi_pass"] is True
    altered = (source / PRIMARY).read_text().replace("x,t,1", "x,t,0")
    (source / PRIMARY).write_text(altered)
    assert screen(source)["metrics_blocker"] == "metrics_provenance_mismatch"
    with pytest.raises(ValueError, match="metrics_provenance_mismatch"):
        aggregation.concatenate_results(str(source))
    assert (source / PRIMARY).read_text() == altered
    (portable / RECEIPT).write_text("a,b,c\n")
    assert screen(portable)["metrics_blocker"] == "metrics_provenance_unreadable"


def test_archiving_job_files_with_tied_times_cannot_reverse_the_observation(tmp_path):
    import os

    earlier = tmp_path / "individual_job_results" / "per_example_metrics_job9.csv"
    earlier.parent.mkdir()
    live = tmp_path / "per_example_metrics_job0.csv"
    _job(earlier, "same", 9)
    _job(live, "same", 0)
    stamp = earlier.stat().st_mtime_ns
    os.utime(live, ns=(stamp, stamp))
    aggregation.concatenate_results(str(tmp_path))
    published = _outputs(tmp_path)
    aggregation.concatenate_results(str(tmp_path))
    assert _outputs(tmp_path) == published
