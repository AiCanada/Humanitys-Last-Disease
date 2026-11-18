"""The screening config must say what the code says.

configs/metrics/antibody_developability/antibody_developability.yaml exists so
a reviewer can read the motifs, licences and thresholds without reading four
modules. A config that has drifted from the code is worse than no config: it
reads as a specification and is a stale copy. This file is the guard.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from rfproteina.disease import deepviscosity as dv
from rfproteina.disease import developability as dev
from rfproteina.disease import humanness as hum
from rfproteina.disease import immunogenicity as imm
from rfproteina.disease import mhc_tables as mhc

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "metrics" / "antibody_developability" / (
    "antibody_developability.yaml"
)


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_the_config_exists_and_says_it_is_not_a_hydra_target() -> None:
    document = _config()
    assert "not a hydra metric target" in document["_description"]


def test_the_imgt_spans_match_the_module() -> None:
    numbering = _config()["numbering"]
    assert {name: tuple(span) for name, span in numbering["cdr_spans_imgt"].items()} == dev.IMGT_CDR_SPANS
    assert numbering["canonical_cysteines"] == list(dev.IMGT_CANONICAL_CYSTEINES)
    assert numbering["scheme"] == dev.SCHEME
    assert numbering["provenance"] == "cited"


def test_every_motif_in_the_config_matches_the_module_and_names_its_source() -> None:
    configured = {item["name"]: item for item in _config()["liability_motifs"]}
    assert set(configured) == {motif.name for motif in dev.LIABILITY_MOTIFS}
    for motif in dev.LIABILITY_MOTIFS:
        item = configured[motif.name]
        assert tuple(item["motifs"]) == motif.motifs
        assert item["source"] == motif.source
        assert item["provenance"] == "cited"


def test_no_pass_cut_is_smuggled_into_the_config() -> None:
    document = _config()
    assert document["pass_criteria"]["note"].startswith("None.")
    assert document["humanness"]["primary"]["pass_cut"] is None
    assert document["humanness"]["antibody_likeness"]["pass_cut"] is None
    # The 20 cP figure is the classifier's class boundary, not a pass mark, so
    # it lives beside a null pass_cut rather than standing in for one.
    assert document["viscosity"]["pass_cut"] is None


def test_the_viscosity_section_matches_the_module() -> None:
    section = _config()["viscosity"]
    assert section["method"] == dv.VISCOSITY_METHOD
    assert section["ensemble_size"] == dv.ENSEMBLE_SIZE
    assert section["class_boundary_cp"] == dv.CLASS_BOUNDARY_CP
    assert section["concentration_mg_per_ml"] == dv.CONCENTRATION_MG_PER_ML
    assert section["decision_probability"] == dv.DECISION_PROBABILITY
    assert section["formulation"] == dv.FORMULATION
    assert section["training_antibodies"] == dv.TRAINING_ANTIBODIES
    assert section["licence"] == "MIT"
    assert len(section["caveats"]) >= len(dv.CAVEATS)


def test_the_class_ii_section_matches_the_registry_and_stays_unavailable() -> None:
    section = _config()["mhc_class_ii"]
    assert section["available"] is False
    assert section["panel"] == list(imm.DR_PANEL)
    assert section["median_rank_max"] == imm.DR_MEDIAN_MAX
    assert {item["name"] for item in section["options"]} == {option.name for option in imm.CLASS_II_OPTIONS}


def test_the_humanness_section_matches_the_module() -> None:
    section = _config()["humanness"]
    assert section["primary"]["method"] == hum.OASIS_METHOD
    assert section["primary"]["subject_threshold_percent"] == hum.OASIS_SUBJECT_THRESHOLD_PERCENT
    assert section["antibody_likeness"]["method"] == hum.ABLANG2_METHOD
    assert {item["name"] for item in section["options"]} == {option.name for option in hum.humanness_options()}


def test_the_class_i_section_records_the_scale_and_the_unit() -> None:
    section = _config()["mhc_class_i"]
    assert section["affinity_unit"] == "nM"
    assert section["percentile_scale"] == "percent_0_100"
    assert section["command_environment_variable"] == mhc.MHCFLURRY_COMMAND_ENV
    assert section["downloads_directory_environment_variable"] == mhc.MHCFLURRY_DOWNLOADS_ENV
    assert section["alleles_environment_variable"] == mhc.ALLELES_ENV
    assert section["alleles"] == list(mhc.CLASS_I_ALLELES)
    assert section["models_bundle"] == mhc.MODELS_BUNDLE
    assert section["models_release"] == mhc.MODELS_RELEASE
