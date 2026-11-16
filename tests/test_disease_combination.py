"""The combination label is a zero-init conditioning path, not a side record."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from rfproteina.datapipes.pipeline import build_design_train_pipeline
from rfproteina.datapipes.schema.proteina import SPECS
from rfproteina.disease.combinations import AXES, CLASS_CONDITION_DIM
from rfproteina.disease.training import AttachDiseaseCombination, prepare_disease_training
from rfproteina.nn.disease_combination import (
    CombinationProjection,
    add_disease_combination,
    build_disease_combination,
    combination_enabled,
    take_disease_options,
)
from rfproteina.nn.local_latents_transformer import LocalLatentsTransformer
from rfproteina.utils.weights import WeightLoadingPolicy
from tests.helpers import CONFIGS_DIR
from tests.helpers.fixtures import data_path  # noqa: F401


def _probe(projection, column):
    weight = torch.zeros_like(projection.proj.weight.detach())
    weight[0, column] = 1
    projection.proj.weight = torch.nn.Parameter(weight)


def _cfg(directory, *, crop=389, bucketing=False):
    """A released-style mixture. The replacement copies this dataset's transform settings."""
    return OmegaConf.create(
        {
            "datasets": {
                "train": {
                    "pdb_interfaces": {
                        "probability": 1.0,
                        "dataset": {
                            "transform": {
                                "_target_": "rfproteina.datapipes.pipeline.build_design_train_pipeline",
                                "design_tasks": {"unconditional": {"frequency": 1.0}},
                                "crop_size_range": [64, 256],
                                "max_crop_size": crop,
                                "precompute_atom_permutation": True,
                                "loss_weight_cfg": {"non_standard": 2.5},
                                "centroid_kwargs": {"p_add_centroid_condition": 0.2},
                                "max_atoms_in_crop": 10000,
                                "path_length_search_cutoff": 64,
                            }
                        },
                    }
                }
            },
            "dataloaders": {"train": {"length_bucketing": {"enabled": bucketing}}},
            "disease_class_data": {"directory": directory, "file_limit": None},
            "_max_tokens": crop,
        }
    )


def test_a_false_string_does_not_build_the_projection():
    assert not combination_enabled("false")
    assert combination_enabled("true")
    assert build_disease_combination(False, 8) is None
    assert build_disease_combination("false", 8) is None
    built = build_disease_combination(True, 8, request_kind="protein")
    assert built is not None
    assert torch.count_nonzero(built.proj.weight) == 0
    assert torch.count_nonzero(built.proj.bias) == 0
    assert "clear_request" not in built.state_dict()
    options = {"disease_combination": "false", "disease_binder_kind": "ligand", "dim_cond": 8}
    enabled, kind = take_disease_options(options)
    assert not combination_enabled(enabled)
    assert kind == "ligand"
    assert "disease_combination" not in options
    assert "disease_binder_kind" not in options


def test_zero_weights_leave_conditioning_unchanged_and_a_real_weight_does_not():
    projection = CombinationProjection(4, request_kind="protein")
    conditioning = torch.ones(1, 2, 4)
    combination = torch.zeros(1, 2, CLASS_CONDITION_DIM)
    combination[..., AXES.index("herg")] = -1
    mask = torch.tensor([[True, False]])
    unchanged = projection(conditioning, combination, mask)
    assert torch.equal(unchanged, conditioning)
    assert add_disease_combination(None, conditioning, combination, mask) is conditioning

    _probe(projection, AXES.index("herg"))
    changed = projection(conditioning, combination, mask)
    assert changed[0, 0, 0].item() == 0
    assert torch.equal(changed[0, 1], conditioning[0, 1])


def test_eval_fills_an_unlabeled_row_and_training_leaves_it_zero():
    projection = CombinationProjection(3, request_kind="protein")
    index = AXES.index("immunogenicity")
    _probe(projection, index)
    conditioning = torch.zeros(1, 2, 3)
    mask = torch.tensor([[True, False]])

    projection.eval()
    filled = projection(conditioning, None, mask)
    assert filled[0, 0, 0].item() == 1
    assert filled[0, 1, 0].item() == 0

    labeled = torch.zeros(1, 2, CLASS_CONDITION_DIM)
    labeled[..., AXES.index("herg")] = -1
    _probe(projection, AXES.index("herg"))
    kept = projection(conditioning, labeled, mask)
    assert kept[0, 0, 0].item() == -1

    projection.train()
    _probe(projection, index)
    untouched = projection(conditioning, None, mask)
    assert torch.equal(untouched, conditioning)


def test_the_batch_spec_zero_fills_a_missing_combination():
    dtype, default, dims = SPECS["disease_combination"]
    assert dtype is torch.float32
    assert default == 0.0
    assert dims == f"I {CLASS_CONDITION_DIM}"


def test_the_attach_transform_requires_a_matching_example_id():
    row = [0.0] * CLASS_CONDITION_DIM
    row[AXES.index("herg")] = -1.0
    transform = AttachDiseaseCombination({"high_affinity_ligand_herg_toxic": row})
    data = {
        "example_id": "high_affinity_ligand_herg_toxic",
        "feats": {"residue_type": torch.zeros(3, dtype=torch.int64)},
    }
    written = transform.forward(data)
    label = written["feats"]["disease_combination"]
    assert label.shape == (3, CLASS_CONDITION_DIM)
    assert label[1, AXES.index("herg")].item() == -1

    # Every example of a disease run is labeled, so a miss is a wiring fault. Zero-filling it
    # would train the example as untested with nothing to show that its label was lost.
    with pytest.raises(ValueError, match="No combination label"):
        transform.forward({"example_id": "herg", "feats": {"tok_idx": torch.zeros(2, dtype=torch.int64)}})

    # A replicate still finds its parent's label.
    replicated = transform.forward(
        {
            "example_id": "high_affinity_ligand_herg_toxic_rep3",
            "feats": {"tok_idx": torch.zeros(2, dtype=torch.int64)},
        }
    )
    assert replicated["feats"]["disease_combination"].shape == (2, CLASS_CONDITION_DIM)


def test_a_null_directory_refuses_the_release_mix():
    with pytest.raises(ValueError, match="release mix"):
        prepare_disease_training(_cfg(None))
    with pytest.raises(ValueError, match="release mix"):
        prepare_disease_training(_cfg("null"))
    plain = OmegaConf.create({"datasets": {"train": {"keep": {"probability": 1.0}}}})
    prepare_disease_training(plain)
    assert "keep" in plain.datasets.train


def test_coarse_tables_and_tsv_files_are_not_a_contrast(tmp_path: Path):
    coarse = tmp_path / "coarse"
    coarse.mkdir()
    (coarse / "high_affinity_toxic.cif").write_text("coarse", encoding="utf-8")
    with pytest.raises(ValueError, match="binder kind"):
        prepare_disease_training(_cfg(str(coarse)))

    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "high_affinity_ligand_herg_toxic.tsv").write_text("table", encoding="utf-8")
    with pytest.raises(ValueError, match="coordinates"):
        prepare_disease_training(_cfg(str(tables)))


def test_an_axis_structure_replaces_the_released_train_mix(tmp_path: Path):
    ready = tmp_path / "ready"
    ready.mkdir()
    path = ready / "high_affinity_ligand_herg_toxic.cif"
    path.write_text("herg", encoding="utf-8")
    earlier = ready / "a_high_affinity_ligand_nav1_5_toxic.tsv"
    earlier.write_text("table", encoding="utf-8")
    cfg = _cfg(str(ready))
    cfg.disease_class_data.file_limit = 1
    prepare_disease_training(cfg)
    leaf = cfg.datasets.train.disease_classes
    assert float(leaf.probability) == 1.0
    assert leaf.dataset._target_ == "rfproteina.inference_utils.datasets.DesignProblemDataset"
    assert list(leaf.dataset.dir_path_or_file_paths) == [str(path)]
    assert "pdb_interfaces" not in cfg.datasets.train
    transform = leaf.dataset.transform
    assert int(transform.crop_size) == 389
    assert transform.crop_size_range is None
    row = list(transform.disease_class_lookup["high_affinity_ligand_herg_toxic"])
    assert row[AXES.index("herg")] == -1.0
    assert row[len(AXES) + AXES.index("herg")] == 1.0

    # The released transform settings must survive. A pipeline with no design_tasks routes no
    # design task and enforces no crop budget; dropping the rest would optimize a different
    # objective than the checkpoint being continued, and the autoencoder loss refuses to run
    # without precomputed permutations.
    assert transform.design_tasks is not None
    assert transform.precompute_atom_permutation is True
    assert float(transform.loss_weight_cfg.non_standard) == 2.5
    assert float(transform.centroid_kwargs.p_add_centroid_condition) == 0.2
    assert int(transform.max_atoms_in_crop) == 10000
    assert int(transform.path_length_search_cutoff) == 64


def test_a_mixture_without_design_tasks_is_refused(tmp_path: Path):
    ready = tmp_path / "ready"
    ready.mkdir()
    (ready / "high_affinity_ligand_herg_toxic.cif").write_text("herg", encoding="utf-8")
    cfg = _cfg(str(ready))
    del cfg.datasets.train.pdb_interfaces.dataset.transform.design_tasks
    with pytest.raises(ValueError, match="design_tasks"):
        prepare_disease_training(cfg)


def test_length_bucketing_is_turned_off_for_a_file_list_dataset(tmp_path: Path):
    ready = tmp_path / "ready"
    ready.mkdir()
    (ready / "high_affinity_ligand_herg_toxic.cif").write_text("herg", encoding="utf-8")
    cfg = _cfg(str(ready), bucketing=True)
    prepare_disease_training(cfg)
    # Bucketing reads a length-metadata table aligned with the dataset indices, and a list of
    # class files carries none, so the sampler would fail on its first batch.
    assert cfg.dataloaders.train.length_bucketing.enabled is False


def test_a_pdb_class_file_is_named_rather_than_silently_dropped(tmp_path: Path):
    ready = tmp_path / "ready"
    ready.mkdir()
    (ready / "high_affinity_ligand_herg_toxic.cif").write_text("herg", encoding="utf-8")
    (ready / "high_affinity_protein_ada_clear.pdb").write_text("ada", encoding="utf-8")
    # The dataset's own filter drops PDB encodings. Keeping them here would both lose those
    # files and shift the shortest-unique-suffix ids of the files that remain, so every
    # lookup could miss at once.
    with pytest.raises(ValueError, match="high_affinity_protein_ada_clear.pdb"):
        prepare_disease_training(_cfg(str(ready)))


def test_file_limit_counts_only_files_that_can_train(tmp_path: Path):
    ready = tmp_path / "ready"
    ready.mkdir()
    (ready / "high_affinity_clean_batch.cif").write_text("coarse", encoding="utf-8")
    (ready / "high_affinity_ligand_herg_toxic.cif").write_text("herg", encoding="utf-8")
    cfg = _cfg(str(ready))
    cfg.disease_class_data.file_limit = 1
    prepare_disease_training(cfg)
    kept = list(cfg.datasets.train.disease_classes.dataset.dir_path_or_file_paths)
    assert [Path(path).name for path in kept] == ["high_affinity_ligand_herg_toxic.cif"]

    zero = _cfg(str(ready))
    zero.disease_class_data.file_limit = 0
    with pytest.raises(ValueError, match="file_limit is 0"):
        prepare_disease_training(zero)


def test_declared_class_constants_must_match_the_module(tmp_path: Path):
    ready = tmp_path / "ready"
    ready.mkdir()
    (ready / "high_affinity_ligand_herg_toxic.cif").write_text("herg", encoding="utf-8")
    wrong_width = _cfg(str(ready))
    wrong_width.disease_class_condition_dim = CLASS_CONDITION_DIM - 1
    with pytest.raises(ValueError, match="must agree"):
        prepare_disease_training(wrong_width)
    wrong_name = _cfg(str(ready))
    wrong_name.disease_class_data.good_class = "something_else"
    with pytest.raises(ValueError, match="good_class"):
        prepare_disease_training(wrong_name)


def test_the_train_pipeline_accepts_the_lookup_after_featurization():
    pipeline = build_design_train_pipeline(
        crop_size=16,
        disease_class_lookup={"example": [0.0] * CLASS_CONDITION_DIM},
    )
    assert any(isinstance(step, AttachDiseaseCombination) for step in pipeline.transforms)


def _compose(config_name, *overrides):
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides))


def test_disease_training_configs_zero_init_only_the_new_projection():
    denoiser = _compose("training_proteina", "experiment=disease_affinity_tox_denoiser")
    assert denoiser.lora is None
    assert denoiser.nn.disease_combination is True
    assert denoiser.nn.disease_binder_kind == "protein"
    assert denoiser.nn.use_torch_compile is False
    assert OmegaConf.select(denoiser, "weight_loading.fallback_policy") is None
    assert denoiser.weight_loading.param_policies["nn.disease_combination.*"] == "zero_init"
    assert str(denoiser.pretrain_ckpt_path).endswith(
        "flow_checkpoints/rfd4-proteina-flow-pretrain-release-2026-10-19.ckpt"
    )
    assert int(denoiser._max_tokens) == 389
    policy = instantiate(denoiser.weight_loading)
    assert policy.fallback_policy is None
    assert policy.get_policy("nn.disease_combination.proj.weight") is WeightLoadingPolicy.ZERO_INIT
    assert policy.get_policy("nn.transition_c_1.linear.weight") is WeightLoadingPolicy.COPY
    with pytest.raises(ValueError, match="release mix"):
        prepare_disease_training(denoiser)

    autoencoder = _compose("training_ae", "experiment_ae=disease_affinity_tox_ae")
    assert autoencoder.nn_ae.disease_combination is True
    assert autoencoder.nn_ae.use_torch_compile is False
    assert OmegaConf.select(autoencoder, "weight_loading.fallback_policy") is None
    policies = autoencoder.weight_loading.param_policies
    assert policies["encoder.disease_combination.*"] == "zero_init"
    assert policies["decoder.disease_combination.*"] == "zero_init"
    assert str(autoencoder.pretrain_ckpt_path).endswith("ae_checkpoints/rfd4-proteina-ae-release-2026-10-19.ckpt")
    assert int(autoencoder._ae_crop_size) == 261
    ae_policy = instantiate(autoencoder.weight_loading)
    assert ae_policy.get_policy("encoder.disease_combination.proj.bias") is WeightLoadingPolicy.ZERO_INIT
    assert ae_policy.get_policy("decoder.disease_combination.proj.weight") is WeightLoadingPolicy.ZERO_INIT
    assert ae_policy.get_policy("encoder.transition_c_1.linear.weight") is WeightLoadingPolicy.COPY

    inference = _compose("inference", "inference_experiment=disease_binder", "inputs=x")
    assert inference.bioir.cuda_graph_max_tokens == 516
    assert OmegaConf.select(inference, "nn.disease_combination") is None
    assert OmegaConf.select(inference, "nn_ae.disease_combination") is None


def test_the_binder_kind_can_be_chosen_at_inference():
    # Without this the kind is fixed by the checkpoint, so a request for a ligand screen would
    # silently be conditioned on the protein-sequence axes instead.
    assert "disease_binder_kind" in LocalLatentsTransformer.RUNTIME_OPTIONS
    assert "disease_combination" not in LocalLatentsTransformer.RUNTIME_OPTIONS
    default = _compose("inference", "inputs=x")
    assert OmegaConf.select(default, "nn.disease_binder_kind") is None
    chosen = _compose("inference", "inputs=x", "nn.disease_binder_kind=ligand")
    assert chosen.nn.disease_binder_kind == "ligand"
