"""Composing an inference config, for the tests that assert on one.

Its own module rather than ``tests/helpers/__init__.py``: the root conftest
imports that package on every pytest invocation, and ``import hydra`` costs
about three seconds. Here, only the two files that compose a config pay for it.
"""

from __future__ import annotations

from hydra import compose, initialize_config_dir

from tests.helpers import CONFIGS_DIR


def compose_inference(*overrides: str):
    """The inference config with ``overrides`` applied, composed as the entry point does."""
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base="1.3"):
        return compose(config_name="inference", overrides=list(overrides))
