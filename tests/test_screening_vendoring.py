"""The vendored screening sources stay source: no nested repository, no weights.

Both properties are enforced by ``.gitignore`` and both are easy to lose there,
because a ``!<tree>/**`` negation re-includes everything beneath it and silently
outranks the top-level ``**.pt`` and ``.git/`` rules. These tests fail on the
file in the tree rather than on the ignore rule, so they catch the mistake
whichever end it was made from -- an edited rule, or a weight file copied in
beside the code that loads it.

Read-only: no network, no weights, nothing written.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest
import tomllib

from rfproteina.disease import stability

REPO = Path(__file__).resolve().parents[1]
SUBPACKAGE = REPO / "screening subpackage"
VENDORED = {
    "thermompnn": SUBPACKAGE / "thermompnn" / "ThermoMPNN",
    "crossabsense": SUBPACKAGE / "crossabsense" / "CrossAbSense",
}
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}


def test_no_vendored_tree_carries_a_nested_repository():
    nested = [str(p.relative_to(REPO)) for p in SUBPACKAGE.rglob(".git")]
    assert nested == [], f"a nested repository would be recorded as a bare gitlink: {nested}"


@pytest.mark.parametrize("tool", sorted(VENDORED))
def test_no_weights_sit_in_the_vendored_source(tool):
    tree = VENDORED[tool]
    if not tree.is_dir():
        pytest.skip(f"{tool} is not vendored in this checkout")
    found = [str(p.relative_to(REPO)) for p in tree.rglob("*") if p.is_file() and p.suffix in WEIGHT_SUFFIXES]
    assert found == [], f"weights belong under DATA_PATH, not in the checkout: {found}"


#: Paths that must stay ignored inside a vendored tree, whether or not they
#: exist today. ``.pyc`` is here because it is the one that actually reached a
#: staged state: the tool's own interpreter imports the vendored source and
#: writes bytecode beside it, so running a single scan creates these.
MUST_BE_IGNORED = (
    "__pycache__/cache.cpython-310.pyc",
    "analysis/__pycache__/SSM.cpython-310.pyc",
    "models/thermoMPNN_default.pt",
    "final.ckpt",
    "model.safetensors",
    ".git/HEAD",
    ".git/config",
)


@pytest.mark.parametrize("relative", MUST_BE_IGNORED)
def test_git_would_not_track_bytecode_weights_or_a_nested_repo(relative):
    """Asks git what it would do, which is the property the ignore rules exist for.

    A presence check cannot cover these: bytecode is recreated by every run, and
    a ``.git`` only appears if someone re-clones. The rules are what has to hold,
    and a ``!<tree>/**`` negation silently outranks the top-level rules, so this
    checks each case against the real resolver.
    """
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on PATH")
    for tree in VENDORED.values():
        if not tree.is_dir():
            continue
        candidate = tree / relative
        finished = subprocess.run(
            [git, "check-ignore", "-q", str(candidate)],
            cwd=REPO,
            capture_output=True,
            check=False,
        )
        assert finished.returncode == 0, (
            f"{candidate.relative_to(REPO)} would be tracked; a negation is outranking " "the rule meant to exclude it"
        )


def test_the_thermompnn_manifest_pins_what_the_installer_fetches():
    with (SUBPACKAGE / "thermompnn" / "manifest.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    commit = manifest["upstream"]["commit"]
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    assert manifest["location"]["weights_tracked"] is False
    paths = {entry["path"] for entry in manifest["weights"]}
    assert "models/thermoMPNN_default.pt" in paths
    assert "/".join(stability.PROTEINMPNN_WEIGHTS) in paths
    for entry in manifest["weights"]:
        assert entry["bytes"] > 0
        assert len(entry["sha256"]) == 64 and all(c in "0123456789abcdef" for c in entry["sha256"])


def test_the_vendored_thermompnn_tree_is_complete_enough_to_install():
    tree = VENDORED["thermompnn"]
    # What install_thermompnn.py and thermompnn_runner.py each need present.
    for relative in ("analysis/custom_inference.py", "pyproject.toml", "uv.lock", "local.yaml", "LICENSE"):
        assert (tree / relative).is_file(), f"{relative} is missing from the vendored tree"


def test_the_screen_resolves_its_source_from_this_checkout(monkeypatch):
    monkeypatch.delenv(stability.THERMOMPNN_CODE_ENV, raising=False)
    monkeypatch.delenv("SCREENING_CODE_ROOT", raising=False)
    assert stability.code_root() == VENDORED["thermompnn"]


def test_the_runner_is_valid_python_and_needs_nothing_from_this_package():
    """It runs under ThermoMPNN's own interpreter, so this suite never imports it.

    Nothing else would catch a syntax error in it, or an ``import rfproteina``
    that cannot resolve in an environment where only ThermoMPNN is installed.
    """
    runner = REPO / "src" / "rfproteina" / "disease" / "thermompnn_runner.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"), filename=str(runner))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert [name for name in imported if name.split(".")[0] == "rfproteina"] == []
