"""Write the generation input for one downloaded structure."""

from __future__ import annotations

import json
from pathlib import Path

from rfproteina.disease.class_tokens import CLASS_CONDITION_DIM, write_annotated_cif
from rfproteina.disease.design_spec import write_design_spec
from rfproteina.disease.publish import repository_relative
from rfproteina.disease.tokens import INFERENCE_CLASS


def write_job(
    source_cif: Path,
    work_dir: Path,
    *,
    binder_kind: str = "protein",
    target_id: str = "",
    chain: str = "",
) -> dict:
    """Write an annotated CIF that asks for a clear combination and no search reward.

    Generation is not started. The source coordinates are copied unchanged.
    ``binder_kind`` selects which liability axes the request marks clear.
    A protein or peptide binder also gets a condition spec when ``chain`` is a
    parsed protein polymer. ``design_spec`` is null when that spec was not written.
    """
    annotated = work_dir / f"{source_cif.stem}.annotated.cif"
    write_annotated_cif(
        source_cif,
        annotated,
        INFERENCE_CLASS,
        binder_kind=binder_kind,
        target_id=target_id,
    )
    resolved_chain = chain.strip() or _chain_from_name(source_cif)
    design = write_design_spec(
        source_cif,
        work_dir,
        chain=resolved_chain,
        binder_kind=binder_kind,
    )
    spec = {
        "annotated_cif": repository_relative(annotated),
        "design_spec": None if design is None else repository_relative(design),
        "class_name": "requested_clear",
        "binder_kind": binder_kind,
        "target_id": target_id,
        "class_condition_dim": CLASS_CONDITION_DIM,
        "search": None,
        "generation_started": False,
    }
    (work_dir / f"{source_cif.stem}.job.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec


def _chain_from_name(source_cif: Path) -> str:
    _prefix, separator, chain = source_cif.stem.rpartition("_")
    if not separator:
        return ""
    return chain
