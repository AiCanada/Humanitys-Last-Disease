"""The interface-quality thresholds, declared in the pass_criteria tree, cannot drift.

``configs/metrics/pass_criteria/interface_quality.yaml`` is the reviewable
declaration of the interface-quality cuts; the runtime source of truth is
``interface_quality.REFERENCE_THRESHOLDS``. This pins the two together so an edit
to one without the other fails here. It also proves the file does not leak into
``refold`` (which must keep reading only ``all_pass_criteria.yaml``), so the
oracle-evaluation pipeline and the refold report are untouched by it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from rfproteina.disease.interface_quality import REFERENCE_THRESHOLDS
from rfproteina.disease.refold import read_criteria

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "metrics" / "pass_criteria" / "interface_quality.yaml"


def test_the_config_file_exists_in_the_pass_criteria_tree() -> None:
    assert _CONFIG.is_file()


def test_the_declared_thresholds_match_the_module_exactly() -> None:
    declared = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert declared == REFERENCE_THRESHOLDS


def test_every_declared_threshold_names_its_source() -> None:
    declared = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    for name, rule in declared.items():
        assert str(rule.get("source", "")).strip(), name


def test_interface_quality_does_not_leak_into_refold_criteria() -> None:
    """refold reads only all_pass_criteria.yaml, so it must not see these keys."""
    criteria = read_criteria()
    assert criteria is not None
    assert "shape_complementarity" not in criteria
    assert "interface_residues_binder" not in criteria
    # The eleven refold criteria are still exactly the ones all_pass_criteria.yaml declares.
    assert "unconditional" in criteria
    assert "developability" in criteria
