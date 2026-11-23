"""Regression tests for resuming partially evaluated generation runs."""

import os
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from atomworks.io.utils.compression import open_compressed

import rfproteina.generate as generate
from rfproteina.inference_utils.aggregate_metrics import concatenate_results
from rfproteina.metrics.utils.metrics_utils import _write_chain_only_cif_if_unambiguous
from rfproteina.utils.structure_files import iter_structure_files


def _write_metrics(path, rows) -> None:
    frame = pd.DataFrame(rows).assign(task="test", dataset_name="test")
    frame.to_csv(path, index=False)


@pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".cif.zst"])
@pytest.mark.parametrize("subdir", ["batch", ".", "archive/batch"])
def test_partition_existing_outputs_routes_existing_designs_to_evaluation(
    tmp_path, monkeypatch, suffix, subdir
) -> None:
    output_dir = tmp_path / "inference_samples" / subdir
    output_dir.mkdir(parents=True)
    for stem in ["second_job4", "first_job5"]:
        with open_compressed(output_dir / f"{stem}_generated{suffix}", "wt") as stream:
            stream.write(f"data_{stem}\n")
    (output_dir / f"third_job4_generated{suffix}").touch()
    staging_dir = tmp_path / "inference_samples" / "batch" / ".generated-cif-interrupted"
    staging_dir.mkdir(parents=True)
    (staging_dir / f"third_job4_generated{suffix}").write_text("partial staged CIF")

    dataset = torch.utils.data.TensorDataset(torch.arange(3))
    monkeypatch.setattr(generate, "_get_example_ids_from_dataset", lambda _: ["first", "second", "third"])

    to_generate, to_evaluate = generate._partition_existing_outputs(
        dataset,
        SimpleNamespace(store_dir=tmp_path, job_id=4),
    )

    if subdir != "batch":
        assert to_generate is dataset
        assert to_evaluate is None
        return
    assert to_generate.indices == [0, 2]
    assert to_evaluate is not None
    assert to_evaluate.indices == [1]


@pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".cif.zst"])
def test_resumed_chain_export_preserves_encoding(tmp_path, lysozyme, suffix) -> None:
    original = _write_chain_only_cif_if_unambiguous(lysozyme, "design", tmp_path, suffix)
    assert original is not None
    new_suffix = ".cif.zst" if suffix == ".cif" else ".cif"
    assert _write_chain_only_cif_if_unambiguous(lysozyme, "design", tmp_path, new_suffix) == original
    assert iter_structure_files(tmp_path, stem_suffix="_chain") == [original]


@pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".cif.zst"])
@pytest.mark.parametrize("job_id", [None, 4])
def test_partition_existing_outputs_regenerates_empty_cifs(tmp_path, monkeypatch, job_id, suffix) -> None:
    output_dir = tmp_path / "inference_samples" / "batch"
    output_dir.mkdir(parents=True)
    job_suffix = f"_job{job_id}" if job_id is not None else ""
    (output_dir / f"first{job_suffix}_generated{suffix}").touch()
    dataset = torch.utils.data.TensorDataset(torch.arange(1))
    monkeypatch.setattr(generate, "_get_example_ids_from_dataset", lambda _: ["first"])

    to_generate, to_evaluate = generate._partition_existing_outputs(
        dataset, SimpleNamespace(store_dir=tmp_path, job_id=job_id)
    )

    assert to_generate is dataset
    assert to_evaluate is None


def test_concatenate_results_merges_archived_and_newest_metrics(tmp_path) -> None:
    archive = tmp_path / "individual_job_results"
    archive.mkdir()
    old_path = archive / "per_example_metrics_job0.csv"
    new_path = tmp_path / "per_example_metrics_job0_reeval.csv"
    _write_metrics(old_path, [{"example_id": "old-only", "score": 1.0}, {"example_id": "rerun", "score": 2.0}])
    _write_metrics(new_path, [{"example_id": "rerun", "score": 3.0}])
    old_mtime = old_path.stat().st_mtime
    os.utime(new_path, (old_mtime + 1, old_mtime + 1))

    concatenate_results(str(tmp_path), cleanup=False)

    result = pd.read_csv(tmp_path / "per_example_metrics.csv").set_index("example_id")
    assert result["score"].to_dict() == {"old-only": 1.0, "rerun": 3.0}


def test_concatenation_preserves_target_identity_and_retry_missingness(tmp_path) -> None:
    archive = tmp_path / "individual_job_results"
    archive.mkdir()
    old = archive / "per_example_metrics_job0.csv"
    new = tmp_path / old.name
    pd.DataFrame([
        {"dataset_name": "first", "task": "t", "example_id": "0", "p.pass": 1, "score": 1.5},
        {"dataset_name": "second", "task": "t", "example_id": "0", "p.pass": 1, "score": 8},
        {"dataset_name": "first", "task": "t", "example_id": "1", "p.pass": 1, "score": 1.50004},
    ]).to_csv(old, index=False)
    pd.DataFrame([
        {"dataset_name": "first", "task": "t", "example_id": "0", "p.pass": None, "score": None},
        {"dataset_name": "first", "task": "t", "example_id": None, "p.pass": 0, "score": None},
    ]).to_csv(new, index=False)
    os.utime(new, (old.stat().st_mtime + 1, old.stat().st_mtime + 1))

    for _ in range(2):
        concatenate_results(str(tmp_path))
        result = pd.read_csv(tmp_path / "per_example_metrics.csv")
        assert len(result) == 4
        first = result[result.dataset_name == "first"]
        assert first[first.example_id == 0].score.isna().all()
        assert first[first.example_id == 1].score.item() == 1.50004
        tasks = pd.read_csv(tmp_path / "per_task_metrics.csv").set_index("dataset_name")
        assert tasks.loc["first", "n"] == 3
        assert tasks.loc["first", "p.pass"] == pytest.approx(1 / 3, abs=0.0001)
        assert tasks.loc["second", "p.pass"] == 1


def test_unreadable_generated_structure_stays_in_posthoc_evaluation(tmp_path) -> None:
    from omegaconf import OmegaConf

    from rfproteina.evaluate import run_evaluate

    inputs = tmp_path / "inference_samples"
    inputs.mkdir()
    (inputs / "one_generated.cif").write_text("data_bad\n", encoding="utf-8")
    (inputs / "one_input.cif").write_text("data_template\n", encoding="utf-8")
    cfg = OmegaConf.create({"inputs": str(inputs), "out_dir": str(tmp_path / "out"), "metrics": {},
                            "metrics_tags": ["design"], "verbose": False})
    run_evaluate(cfg)
    frame = pd.read_csv(tmp_path / "out/evaluation_metrics/per_example_metrics.csv")
    assert list(frame.evaluation_error) == ["structure_unreadable"]
    assert frame.designed_structure_path.str.endswith("one_generated.cif").all()


def test_nonfinite_oracle_values_are_indeterminate_before_reduction() -> None:
    from rfproteina.metrics.oracles.pass_criterion_metric import PassCriterionMetric

    metric = PassCriterionMetric(metric_thresholds={"pair": {"min_of": ["a", "b"], "max": 2}})
    for missing in [None, float("nan"), float("inf"), -float("inf"), "bad"]:
        assert metric.compute({"a": 1, "b": missing}) == {"pass": 0.0, "all_metrics_present": 0.0}
    assert metric.compute({"a": 1, "b": 3}) == {"pass": 1.0, "all_metrics_present": 1.0}


def test_offline_evaluation_to_screen_on_lysozyme(tmp_path, lysozyme_cif_path):
    from shutil import copyfile

    from omegaconf import OmegaConf

    from rfproteina.disease.screen import screen_generation
    from rfproteina.evaluate import run_evaluate

    inputs = tmp_path / "run" / "inference_samples"
    inputs.mkdir(parents=True)
    copyfile(lysozyme_cif_path, inputs / "lysozyme_generated.cif")
    out = tmp_path / "evaluation"
    cfg = OmegaConf.create({
        "inputs": str(inputs.parent), "out_dir": str(out), "metrics_tags": ["design"], "verbose": False,
        "metrics": {"inline_metrics": {"composition": {
            "_target_": "rfproteina.metrics.amino_acid_composition.AminoAcidCompositionMetric",
        }}},
    })
    run_evaluate(cfg)
    concatenate_results(str(out / "evaluation_metrics"))
    row = pd.read_csv(out / "evaluation_metrics" / "per_example_metrics.csv").iloc[0]
    assert row["composition.n_standard_residues"] == 129
    assert str(row["chain_of_interest"]).startswith("A")
    record = screen_generation(out, binder_kind="other", target_id="")
    assert record["metrics_file"] == "evaluation_metrics/per_example_metrics.csv"
    assert len(record["samples"]) == 1
    assert "row_blocker" not in record["samples"][0]


def test_prior_standalone_observations_survive_new_shards_without_multiplying(tmp_path):
    old_rows = [{"example_id": "old", "score": 1.5000000000000002}, {"example_id": None, "score": 7}]
    _write_metrics(tmp_path / "per_example_metrics.csv", old_rows)
    _write_metrics(tmp_path / "per_example_metrics_job0.csv", [{"example_id": "new", "score": 3}])
    for _ in range(2):
        concatenate_results(str(tmp_path))
        frame = pd.read_csv(tmp_path / "per_example_metrics.csv", float_precision="round_trip")
        assert sorted(frame.score) == [1.5000000000000002, 3, 7]


def test_partial_clustering_preserves_exported_chains(tmp_path, monkeypatch):
    from rfproteina.inference_utils import foldseek

    directory = foldseek.chain_cif_dir(tmp_path, "dataset", "task")
    directory.mkdir(parents=True)
    rows = []
    for name in ("first", "second"):
        chain = directory / f"{name}_chain.cif"
        chain.write_text(f"data_{name}\n")
        rows.append({"example_id": name, "dataset_name": "dataset", "task": "task", "output_base_dir": str(tmp_path),
                     "designed_chain_path": str(chain), "x.pass": 1})
    monkeypatch.setattr(foldseek, "cluster_structures", lambda *args, **kwargs: {"first": 0})
    pd.DataFrame(rows).to_csv(tmp_path / "per_example_metrics.csv", index=False)
    concatenate_results(str(tmp_path), cluster_with_foldseek=True)
    assert sorted(p.name for p in directory.iterdir()) == ["first_chain.cif", "second_chain.cif"]
    records = pd.read_csv(tmp_path / "per_example_metrics.csv")
    assert records["cluster_coverage"].eq(0.5).all()
    assert records["cluster_diversity"].isna().all()
    assert records["x.clustered_pass_rate"].isna().all()


def test_single_job_directory_and_clustered_rates_keep_dataset_scope(tmp_path, monkeypatch) -> None:
    from rfproteina.inference_utils import foldseek

    rows = []
    for ds in ("one", "two"):
        for i in range(2):
            files = foldseek.chain_cif_dir(tmp_path, ds, "target")
            files.mkdir(parents=True, exist_ok=True)
            rows.append({"example_id": str(i), "dataset_name": ds, "task": "target", "output_base_dir": str(tmp_path),
                         "x.pass": 1, "x.all_metrics_present": int(i == 0 or ds == "two")})
    first_dir = foldseek.chain_cif_dir(tmp_path, "one", "target")
    monkeypatch.setattr(foldseek, "cluster_structures", lambda path, **kw: {"0": 0, "1": int(path != first_dir)})
    pd.DataFrame(rows).to_csv(tmp_path / "per_example_metrics.csv", index=False)
    concatenate_results(str(tmp_path), cluster_with_foldseek=True, cleanup_chains=False)
    report = pd.read_csv(tmp_path / "per_task_metrics.csv").set_index("dataset_name")
    assert report["x.clustered_pass_rate"].to_dict() == {"one": 0.5, "two": 1.0}
    assert report["x.pass"].to_dict() == {"one": 0.5, "two": 1.0}
