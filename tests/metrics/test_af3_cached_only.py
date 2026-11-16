"""Reading a refold that was produced somewhere this machine cannot fold.

``imported_refolds`` files a fold computed elsewhere -- the hosted AlphaFold
Server, another machine -- at the paths the oracle would have written, so that
every refolding metric reads it unchanged. That is true of the *files*, and it
was not true of getting the pipeline to open them.

``AF3Oracle.setup`` is called for any oracle whose tags match the validation set,
before a single design is looked at, and it launches the warm AF3 daemon: a
~10 GB worker that needs the AlphaFold 3 image and weights. So turning a
refolding tag on merely to **read** an imported fold launched that worker and
failed for want of the exact runtime the import existed to avoid. The import was
usable only on a machine that did not need it.

``cached_only`` starts nothing. These tests pin both halves of it: no daemon, and
a design with no refold on disk raises and is named rather than being counted as
folded -- a short denominator reported quietly would read as a pass rate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput
from rfproteina.utils.common import design_output_dir

_SOCKET_ENV = "AF3_LAUNCHER_SOCKET_TEMPLATE"


@pytest.fixture
def no_launcher_socket(monkeypatch):
    """A clean environment, so a pre-spawned daemon is not mistaken for ours."""
    monkeypatch.delenv(_SOCKET_ENV, raising=False)
    return monkeypatch


def cached(oracle: AF3Oracle, out_dir: Path, example_id: str) -> None:
    """Put a refold where the oracle looks for one, as place_refold would."""
    directory = design_output_dir(out_dir, example_id)
    directory.mkdir(parents=True, exist_ok=True)
    oracle.refolded_cif_path(out_dir, example_id).write_text("data_x\n#\n", encoding="utf-8")
    oracle.confidence_path(out_dir, example_id).write_text(
        json.dumps({"imported_refold": True, "oracle_name": oracle.oracle_name, "overall_ptm": 61.0}) + "\n",
        encoding="utf-8",
    )


def job(out_dir: Path, example_id: str) -> OracleJobInput:
    """A queued design. ``atom_array`` is never touched on the cached path."""
    return OracleJobInput(atom_array=None, example_id=example_id, out_dir=out_dir)


class TestSetup:
    def test_cached_only_starts_no_daemon(self, tmp_path: Path, no_launcher_socket, monkeypatch) -> None:
        """The defect: setup used to launch a ~10 GB worker before reading anything."""
        import subprocess

        def refuse(*args, **kwargs):
            raise AssertionError("cached_only must not spawn anything")

        monkeypatch.setattr(subprocess, "Popen", refuse)
        oracle = AF3Oracle(oracle_name="af3", cached_only=True)
        oracle.setup()
        assert _SOCKET_ENV not in os.environ

    def test_cached_only_teardown_is_safe_with_nothing_started(self, tmp_path: Path, no_launcher_socket) -> None:
        """Tearing down what was never set up must not raise."""
        oracle = AF3Oracle(oracle_name="af3", cached_only=True)
        oracle.setup()
        oracle.teardown()

    def test_the_default_is_unchanged(self) -> None:
        """Off by default, so no existing run behaves differently."""
        assert AF3Oracle(oracle_name="af3").cached_only is False


class TestFoldBatch:
    def test_a_fully_cached_batch_is_counted_without_folding(self, tmp_path: Path, no_launcher_socket) -> None:
        oracle = AF3Oracle(oracle_name="af3", cached_only=True)
        out = tmp_path / "run"
        for name in ("binder_0", "binder_1"):
            cached(oracle, out, name)
        assert oracle.fold_batch([job(out, "binder_0"), job(out, "binder_1")]) == 2

    def test_a_design_with_no_refold_raises_and_is_named(self, tmp_path: Path, no_launcher_socket) -> None:
        """Named, because which designs are missing is the thing the caller needs."""
        oracle = AF3Oracle(oracle_name="af3", cached_only=True)
        out = tmp_path / "run"
        cached(oracle, out, "binder_0")
        with pytest.raises(RuntimeError) as raised:
            oracle.fold_batch([job(out, "binder_0"), job(out, "binder_1")])
        message = str(raised.value)
        assert "1 of 2 design(s) have no refold on disk" in message
        assert "binder_1" in message
        assert "rfd4-af3server retrieve" in message

    def test_the_message_does_not_print_a_whole_campaign(self, tmp_path: Path, no_launcher_socket) -> None:
        """Ten names is enough to act on; four hundred is a wall of text."""
        oracle = AF3Oracle(oracle_name="af3", cached_only=True)
        out = tmp_path / "run"
        jobs = [job(out, f"binder_{index}") for index in range(25)]
        with pytest.raises(RuntimeError) as raised:
            oracle.fold_batch(jobs)
        message = str(raised.value)
        assert "25 of 25 design(s)" in message
        assert message.count("binder_") == 10
        assert "..." in message

    def test_an_empty_batch_is_not_an_error(self, tmp_path: Path, no_launcher_socket) -> None:
        oracle = AF3Oracle(oracle_name="af3", cached_only=True)
        assert oracle.fold_batch([]) == 0


class TestCompute:
    def test_a_cached_refold_is_returned_with_its_canonical_paths(self, tmp_path: Path, no_launcher_socket) -> None:
        """The contract the import rests on, reached with no oracle runtime at all.

        ``compute`` takes the cache fast path before it ever looks at the atom
        array, which is why ``None`` is a legitimate argument here.
        """
        oracle = AF3Oracle(oracle_name="af3", cached_only=True)
        out = tmp_path / "run"
        cached(oracle, out, "binder_0")
        result = oracle.compute(None, "binder_0", out)
        assert result["overall_ptm"] == pytest.approx(61.0)
        assert result["imported_refold"] is True
        assert result["confidence_json_path"] == str(oracle.confidence_path(out, "binder_0"))
        assert result["refolded_structure_path"] == str(oracle.refolded_cif_path(out, "binder_0"))
