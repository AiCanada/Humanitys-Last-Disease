"""Point a disease training run at structure files that name a combination.

The directory is caller-supplied. This module does not search the disk for
unfinished class files. A null directory is a hard stop, so the disease
experiment cannot silently sample the released mixture. Tables without
coordinates and coarse clean/toxic names are not a training contrast.

The replacement keeps the released transform settings and changes only the file
list, the crop, and the combination lookup, so the fine-tune optimizes the same
objective as the checkpoint it continues.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from atomworks.ml.transforms.base import Transform
from omegaconf import DictConfig, OmegaConf, open_dict

from rfproteina.disease.classes import ClassFile, load_class_files
from rfproteina.disease.combinations import CLASS_CONDITION_DIM, combination_condition, judge_combination
from rfproteina.disease.tokens import BAD_CLASS, GOOD_CLASS, check_budget
from rfproteina.utils.structure_files import is_structure_file, structure_file_stem

_MISSING = {"", "null", "none"}

#: ``DesignProblemDataset`` appends this when ``num_replicates`` is above one.
_REPLICATE_SUFFIX = re.compile(r"_rep\d+$")


def prepare_disease_training(cfg: DictConfig) -> None:
    """Replace ``datasets.train`` when this config is a disease run.

    Configs without ``disease_class_data`` are unchanged. A disease config
    whose directory is null, empty, 'null', or 'None' raises ``ValueError``
    before any checkpoint is loaded.
    """
    block = cfg.get("disease_class_data")
    if block is None:
        return
    _check_declared_constants(cfg, block)
    directory = block.get("directory")
    if _directory_missing(directory):
        raise ValueError(
            "The disease run does not sample the release mix. "
            "Set disease_class_data.directory to structure files that name a binder kind and an axis."
        )
    file_limit = block.get("file_limit")
    usable = _usable_structures(directory, file_limit)
    lookup = _class_lookup(usable)
    crop = cfg.get("_max_tokens")
    if crop is None:
        crop = cfg.get("_ae_crop_size")
    if crop is None:
        raise ValueError("The disease run needs _max_tokens or _ae_crop_size for the structure crop.")

    transform = _released_transform(cfg)
    transform.update(
        {
            "_target_": "rfproteina.datapipes.pipeline.build_design_train_pipeline",
            # One fixed crop at the pad width, so the crop and the collate cannot disagree.
            "crop_size": int(crop),
            "crop_size_range": None,
            "crop_contiguous_probability": 0.0,
            "crop_spatial_probability": 1.0,
            "is_inference": False,
            "disease_class_lookup": lookup,
        }
    )
    replacement = OmegaConf.create(
        {
            "disease_classes": {
                "probability": 1.0,
                "dataset": {
                    "_target_": "rfproteina.inference_utils.datasets.DesignProblemDataset",
                    "dir_path_or_file_paths": [str(item.path) for item in usable],
                    "transform": transform,
                },
            }
        }
    )
    with open_dict(cfg):
        if "datasets" not in cfg or cfg.datasets is None:
            cfg.datasets = {}
    with open_dict(cfg.datasets):
        cfg.datasets.train = replacement
    _disable_length_bucketing(cfg)


class AttachDiseaseCombination(Transform):
    """Write the combination row onto ``feats`` after featurization.

    Every example of a disease run carries a label, so a lookup miss is a wiring
    fault rather than a normal case. Zero-filling a miss would train the example
    as untested with no sign that its label was lost, so a miss raises.
    """

    def __init__(self, lookup: Mapping[str, Sequence[float]] | None = None) -> None:
        super().__init__()
        self._rows = _rows(lookup)

    def forward(self, data: dict) -> dict:
        if not self._rows:
            return data
        example_id = data.get("example_id")
        if not isinstance(example_id, str):
            raise ValueError(
                "The combination label matches on example_id, which the dataset sets before the "
                f"transform runs; got {type(example_id).__name__}."
            )
        row = self._rows.get(example_id)
        if row is None:
            row = self._rows.get(_REPLICATE_SUFFIX.sub("", example_id))
        if row is None:
            known = sorted(self._rows)[:3]
            raise ValueError(
                f"No combination label for example {example_id!r}. The lookup holds {len(self._rows)} "
                f"keys, for example {known}. The dataset's example ids and the lookup keys were built "
                "from different file lists."
            )
        feats = data.get("feats")
        if not isinstance(feats, dict):
            raise ValueError("The combination label is written onto feats after featurization.")
        n_tokens = _token_count(feats)
        feats["disease_combination"] = row.unsqueeze(0).expand(n_tokens, -1).contiguous()
        return data


def _check_declared_constants(cfg: DictConfig, block: DictConfig) -> None:
    """Keep the numbers a disease config declares and the module constants from drifting apart."""
    check_budget()
    declared_dim = cfg.get("disease_class_condition_dim")
    if declared_dim is not None and int(declared_dim) != CLASS_CONDITION_DIM:
        raise ValueError(
            f"disease_class_condition_dim is {int(declared_dim)}, but a combination row is "
            f"{CLASS_CONDITION_DIM} wide. The axis list and the config must agree."
        )
    for key, expected in (("good_class", GOOD_CLASS), ("bad_class", BAD_CLASS)):
        declared = block.get(key)
        if declared is not None and str(declared) != expected:
            raise ValueError(f"disease_class_data.{key} is {declared!r}, but this build names that class {expected!r}.")


def _released_transform(cfg: DictConfig) -> dict[str, Any]:
    """Transform settings of the mixture this run replaces, resolved to plain values.

    The released datasets carry the design tasks, the loss weights, the centroid
    conditioning, the atom cap, and the bond-graph cutoff. Dropping them would
    optimize a different objective than the checkpoint being continued, and a
    pipeline with no design task emits no conditioning at all. The dataset with
    the largest sampling probability stands in for the mixture.
    """
    datasets = cfg.get("datasets")
    train = None if datasets is None else datasets.get("train")
    if train is None:
        raise ValueError(
            "The disease run copies its transform settings from the released training mixture, "
            "but this config has no datasets.train to copy from."
        )
    best: tuple[float, str, Any] | None = None
    for name, entry in train.items():
        if entry is None:
            continue
        dataset = entry.get("dataset")
        candidate = None if dataset is None else dataset.get("transform")
        if candidate is None or candidate.get("design_tasks") is None:
            continue
        probability = float(entry.get("probability") or 0.0)
        if best is None or probability > best[0]:
            best = (probability, name, candidate)
    if best is None:
        raise ValueError(
            "No dataset in datasets.train supplies a transform with design_tasks, so this run has no "
            "released design-task preset to keep. Without design tasks the pipeline trains no conditioning."
        )
    return OmegaConf.to_container(best[2], resolve=True)


def _disable_length_bucketing(cfg: DictConfig) -> None:
    """Turn length bucketing off for a file-list dataset.

    Bucketing reads a length-metadata table aligned with the dataset's indices.
    A list of class files carries none, so the sampler would fail on its first
    batch instead of training.
    """
    dataloaders = cfg.get("dataloaders")
    train = None if dataloaders is None else dataloaders.get("train")
    bucketing = None if train is None else train.get("length_bucketing")
    if bucketing is None or not bucketing.get("enabled", False):
        return
    with open_dict(bucketing):
        bucketing["enabled"] = False


def _directory_missing(directory: object) -> bool:
    if directory is None:
        return True
    return str(directory).strip().lower() in _MISSING


def _usable_structures(directory: object, file_limit: object) -> list[ClassFile]:
    if file_limit is not None and int(file_limit) < 0:
        raise ValueError(f"file_limit must be >= 0, got {file_limit}")
    found = load_class_files(str(directory))
    if not found:
        raise ValueError(
            "The disease run does not sample the release mix. No file name in "
            f"{directory} declares a binder kind and a liability axis, so there is no contrast to train on."
        )
    # The dataset filters this list with ``is_structure_file``, whose default drops every PDB
    # encoding. Selecting here with a wider rule would silently drop those files *and* shift the
    # shortest-unique-suffix example ids of the files that remain, so select with the same rule
    # and name the files that cannot be loaded.
    pdb_only = [
        item.path
        for item in found
        if is_structure_file(item.path, include_pdb=True) and not is_structure_file(item.path)
    ]
    if pdb_only:
        names = ", ".join(sorted(path.name for path in pdb_only))
        raise ValueError(
            "DesignProblemDataset does not load PDB-format structures, so these class files would be "
            f"dropped and every example id would shift: {names}. Convert them to CIF before training."
        )
    structures = [item for item in found if is_structure_file(item.path)]
    if not structures:
        suffixes = ", ".join(sorted({"".join(item.path.suffixes) or "(none)" for item in found}))
        raise ValueError(
            f"No class file in {directory} is a structure the denoiser can train on; the names found "
            f"end in {suffixes}. Tables carry no coordinates. The disease run does not sample the release mix."
        )
    usable = [item for item in structures if not item.coarse and item.binder_kind and item.axis and item.state]
    if not usable:
        raise ValueError(
            "A training contrast must name the binder kind and the axis. "
            "A coarse high_affinity_clean or high_affinity_toxic file does not set an axis."
        )
    # Limit after the filters, so files that could never train do not consume the budget.
    if file_limit is not None:
        usable = usable[: int(file_limit)]
        if not usable:
            raise ValueError("disease_class_data.file_limit is 0, so no class structure would train.")
    return usable


def _class_lookup(usable: list[ClassFile]) -> dict[str, list[float]]:
    """Map every id the dataset might use onto that file's combination row."""
    rows = [_row(item) for item in usable]
    example_ids = _example_ids([item.path for item in usable])
    lookup: dict[str, list[float]] = dict(zip(example_ids, rows, strict=True))
    # An alias two class files share must resolve to neither, not to whichever came first.
    by_alias: dict[str, list[list[float]]] = {}
    for item, row in zip(usable, rows, strict=True):
        for alias in (item.path.name, item.path.stem, structure_file_stem(item.path)):
            seen = by_alias.setdefault(alias, [])
            if row not in seen:
                seen.append(row)
    for alias, candidates in by_alias.items():
        if len(candidates) == 1:
            lookup.setdefault(alias, candidates[0])
    return lookup


def _row(item: ClassFile) -> list[float]:
    if item.binder_kind is None or item.axis is None or item.state is None:
        raise ValueError(f"{item.path.name} does not name a binder kind and an axis.")
    combination = judge_combination(item.binder_kind, {item.axis: item.state})
    return [float(value) for value in combination_condition(1, combination)[0]]


def _example_ids(paths: list[Path]) -> list[str]:
    """Match DesignProblemDataset: the shortest unique path suffix, joined by ``-``.

    Mirrors ``DesignProblemDataset._get_example_id``. The two must agree on the
    same file list, which is why ``_usable_structures`` filters exactly as the
    dataset does.
    """
    all_parts = [(*path.parent.parts, structure_file_stem(path)) for path in paths]
    max_parts = max(len(parts) for parts in all_parts)
    for num_parts in range(1, max_parts + 1):
        candidate_ids = ["-".join(parts[-num_parts:]) for parts in all_parts]
        if len(set(candidate_ids)) == len(candidate_ids):
            return candidate_ids
    raise ValueError("Could not build unique example ids for the class structures.")


def _rows(lookup: Mapping[str, Sequence[float]] | None) -> dict[str, torch.Tensor]:
    rows: dict[str, torch.Tensor] = {}
    if not lookup:
        return rows
    for key, value in lookup.items():
        row = torch.tensor([float(item) for item in value], dtype=torch.float32)
        if row.numel() != CLASS_CONDITION_DIM:
            raise ValueError(
                f"The combination row for {key!r} has {row.numel()} values, but a row is "
                f"{CLASS_CONDITION_DIM} wide."
            )
        rows[str(key)] = row
    return rows


def _token_count(feats: dict) -> int:
    for key in ("residue_type", "tok_idx"):
        value = feats.get(key)
        shape = getattr(value, "shape", None)
        if shape:
            count = int(shape[0])
            if count < 1:
                raise ValueError("The combination marks real tokens, so the crop must contain at least one.")
            return count
    raise ValueError("The combination label needs residue_type or tok_idx to know the token count.")
