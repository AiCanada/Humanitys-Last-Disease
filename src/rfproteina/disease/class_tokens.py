"""Write the combination label onto a real structure file.

Inference copies the source and appends a requested-clear record for one binder
kind. A hit, an untested screen, a coarse toxic name, and a safety-panel target
are training records. The record has no atom sites.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rfproteina.disease.combinations import (
    CLASS_CONDITION_DIM,
    Combination,
    combination_condition,
    combination_record,
    requested_clear,
)
from rfproteina.disease.tokens import BAD_CLASS, GOOD_CLASS, INFERENCE_CLASS

__all__ = [
    "CLASS_CONDITION_DIM",
    "class_condition",
    "class_record",
    "write_annotated_cif",
]


def class_condition(n_tokens: int, class_name: str) -> np.ndarray:
    """Broadcast a protein requested-clear label. A coarse toxic name is rejected."""
    if class_name == GOOD_CLASS:
        return combination_condition(n_tokens, requested_clear("protein"))
    if class_name == BAD_CLASS:
        raise ValueError(
            f"{class_name!r} names no binder kind and no axis. "
            "Name the combination that failed. It is a training contrast, not a generation request."
        )
    raise ValueError(f"Unknown class {class_name!r}. Expected {GOOD_CLASS!r} or a judged combination.")


def class_record(class_name: str) -> str:
    """mmCIF text for the protein requested-clear label. It contains no atom sites."""
    if class_name != GOOD_CLASS:
        raise ValueError(
            f"{class_name!r} is a training contrast, not a generation request. "
            "A clear inference record is the requested-clear protein combination."
        )
    return combination_record(requested_clear("protein"))


def write_annotated_cif(
    source: Path,
    dest: Path,
    class_name: str | Combination = INFERENCE_CLASS,
    *,
    binder_kind: str = "protein",
    target_id: str = "",
    target_is_panel: bool = False,
) -> Path:
    """Copy ``source`` and append a requested-clear record to its data block.

    The record is appended as another mmCIF category inside the source's single
    data block. It never opens a ``data_`` block of its own, because a
    multi-block CIF cannot be loaded.
    """
    if isinstance(class_name, Combination):
        combination = class_name
    elif class_name == INFERENCE_CLASS:
        combination = requested_clear(
            binder_kind,
            target_id=target_id,
            target_is_panel=target_is_panel,
        )
    else:
        raise ValueError(
            "Inference attaches a requested-clear combination. "
            f"{class_name!r} is a training contrast, not a generation request."
        )
    if combination.call != "requested_clear" or combination.target_is_panel:
        raise ValueError(
            "Inference attaches a requested-clear combination. "
            f"{combination.call!r} is a training contrast, not a generation request."
        )
    body = source.read_text(encoding="utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body.rstrip() + "\n" + combination_record(combination), encoding="utf-8")
    return dest
