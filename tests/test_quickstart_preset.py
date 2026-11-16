"""The quickstart preset composes the same config as the manual first-run incantation. CPU only."""

import pytest
from omegaconf import OmegaConf

from tests.helpers import CONFIGS_DIR
from tests.helpers.fixtures import data_path  # noqa: F401
from tests.helpers.inference import compose_inference


def test_quickstart_matches_the_manual_incantation():
    preset = OmegaConf.to_container(compose_inference("inference_experiment=quickstart", "inputs=x"), resolve=True)
    manual = OmegaConf.to_container(compose_inference("inputs=x", "~metrics", "compile_lengths=null"), resolve=True)
    preset.pop("inference_experiment", None)
    manual.pop("inference_experiment", None)
    assert preset == manual


def test_quickstart_needs_no_checkpoint_arguments_and_keeps_the_pin():
    cfg = compose_inference("inference_experiment=quickstart", "inputs=x")
    base = compose_inference("inputs=x")
    assert "metrics" not in cfg
    assert cfg.compile_lengths is None
    assert cfg.hf_ckpt_path == "nvidia/RFD4-Proteina"
    assert cfg.hf_ckpt_revision == base.hf_ckpt_revision  # one source of truth: inference.yaml
    assert "hf_ckpt_revision" not in (CONFIGS_DIR / "inference_experiment" / "quickstart.yaml").read_text().replace(
        "`hf_ckpt_revision`", ""
    )


def test_quickstart_requires_inputs_and_accepts_overrides():
    with pytest.raises(Exception, match="inputs"):
        OmegaConf.to_container(compose_inference("inference_experiment=quickstart"), resolve=True)
    cfg = compose_inference("inference_experiment=quickstart", "inputs=x", "num_replicates=3", "hf_ckpt_path=null")
    assert cfg.generation.dataset.num_replicates == 3
    assert cfg.hf_ckpt_path is None
