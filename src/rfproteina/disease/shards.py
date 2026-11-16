"""Split a selection across local processes or other machines.

Planning a shard does not start generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.structures import StructureOption

ORIGINAL_EXPERIMENT = "inference_on_user_inputs"
MODIFIED_EXPERIMENT = "disease_binder"
#: The two models a run can generate with. ``original`` is the release pair;
#: ``modified`` is the disease checkpoint. One of them generates, never both.
GENERATION_MODELS = ("original", "modified")


@dataclass(frozen=True)
class WorkerShard:
    """Structures assigned to one process or one host."""

    host: str
    structures: tuple[StructureOption, ...]
    generation_started: bool = False


def shard_for_hosts(structures: list[StructureOption], hosts: list[str]) -> list[WorkerShard]:
    """Deal structures round-robin across ``hosts``.

    Hosts are addressed by position, so a repeated name is a second worker on
    that machine rather than a duplicate of the same shard.
    """
    if not hosts:
        raise ValueError("At least one host is required to shard work.")
    buckets: list[list[StructureOption]] = [[] for _ in hosts]
    for index, structure in enumerate(structures):
        buckets[index % len(hosts)].append(structure)
    return [
        WorkerShard(host=host, structures=tuple(bucket), generation_started=False)
        for host, bucket in zip(hosts, buckets, strict=True)
    ]


def model_overrides(modified_ckpt: Path | None, binder_kind: str | None = None) -> list[str]:
    """Select the release model until a modified checkpoint file is present.

    The modified file replaces the release pair. A missing path stays on the
    original inference experiment and the checkpoint named by ``inference.yaml``.
    ``binder_kind`` picks which liability axes an unlabeled batch asks to be
    clear on, and only a checkpoint carrying the combination projection reads it.
    """
    if modified_ckpt is not None and modified_ckpt.is_file():
        overrides = [
            f"inference_experiment={MODIFIED_EXPERIMENT}",
            "hf_ckpt_path=null",
            f"ckpt_path={modified_ckpt.parent}",
            f"ckpt_name={modified_ckpt.name}",
        ]
        if binder_kind:
            overrides.append(f"nn.disease_binder_kind={binder_kind}")
        return overrides
    return [f"inference_experiment={ORIGINAL_EXPERIMENT}"]


def model_checkpoint(model: str, modified_ckpt: Path | None) -> Path | None:
    """The checkpoint the named model generates from, or None for the release pair.

    A request for the modified model without a checkpoint file raises rather than
    quietly running the release model: the request named which model to use, and
    answering it with the other one is not a smaller version of that answer.
    """
    name = str(model).strip().casefold()
    if name not in GENERATION_MODELS:
        listed = ", ".join(GENERATION_MODELS)
        raise ValueError(f"Unknown generation model {model!r}. Expected one of {listed}.")
    if name == "original":
        return None
    if modified_ckpt is None or not modified_ckpt.is_file():
        raise ValueError("The modified model needs a checkpoint file. Name one with --modified-ckpt.")
    return modified_ckpt


def generation_commands(
    shards: list[WorkerShard],
    out_dir: str,
    modified_ckpt: Path | None = None,
    binder_kind: str | None = None,
) -> list[list[str]]:
    """Argv lists for ``rfd4-generate``. Nothing is executed.

    A condition spec written for this structure replaces the annotated CIF as
    ``inputs``. A structure with no spec keeps the annotated CIF.
    """
    overrides = model_overrides(modified_ckpt, binder_kind)
    commands: list[list[str]] = []
    for shard in shards:
        if shard.generation_started:
            raise RuntimeError(f"Shard for {shard.host} already marks generation as started.")
        for structure in shard.structures:
            commands.append(
                [
                    "rfd4-generate",
                    *overrides,
                    _inputs_token(out_dir, structure),
                    # The group belongs to the identity here as it does in the input name, so two
                    # polymer groups of one deposit cannot write into the same output tree.
                    f"out_dir={out_dir}/{structure.group}_{structure.entry_id}_{structure.chain}",
                    f"+disease.host={shard.host}",
                ]
            )
    return commands


def _inputs_token(out_dir: str, structure: StructureOption) -> str:
    stem = f"{structure.group}_{structure.entry_id}_{structure.chain}"
    if (Path(out_dir) / f"{stem}.yaml").is_file():
        name = f"{stem}.yaml"
    else:
        name = f"{stem}.annotated.cif"
    return f"inputs={out_dir}/{name}"
