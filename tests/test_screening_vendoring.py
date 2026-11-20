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
    # This project's own program rather than an upstream mirror, so its
    # manifest records a tree digest instead of a reproducing commit. The two
    # properties these tests are about hold either way: source stays source,
    # and the payload stays under DATA_PATH.
    "confrover": SUBPACKAGE / "confrover" / "rbase",
    "confrover_viewer": SUBPACKAGE / "confrover" / "viz",
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
    # RBase's SE(3) diffuser writes its IGSO3 lookup tables into a .cache
    # directory beside its own source the first time it runs -- three .npy
    # files, 8 MB each. They are the one payload a vendored tree here produces
    # by being *run* rather than by being copied into, which makes them the
    # easiest to commit by accident.
    ".cache/eps_1000_omega_1000_min_sigma_0_1_max_sigma_1_5_schedule_logarithmic/cdf_vals.npy",
    "score_norms.npy",
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


def test_the_confrover_manifest_records_what_cannot_be_checked_against_upstream():
    """This tree is the project's own, and the manifest has to say so.

    The other two promise that a clone at the commit they name reproduces them.
    This one cannot: the package lives in a working tree that was never
    committed, and it is expected to keep changing. A manifest that made the
    same promise would be wrong rather than strict, so it carries a tree digest
    and says plainly that no commit reproduces the tree.
    """
    manifest = SUBPACKAGE / "confrover" / "manifest.toml"
    if not manifest.is_file():
        pytest.skip("confrover is not vendored in this checkout")
    with manifest.open("rb") as handle:
        found = tomllib.load(handle)
    origin = found["origin"]
    assert origin["reproducible_from_commit"] is False
    assert len(origin["tree_sha256"]) == 64
    assert origin["diverges_from"]
    # The limitation that decides what the tab can offer.
    assert found["scope"]["single_chain_only"] is True
    # Both checkpoints, each with the digest the installer checks.
    weights = {str(entry["path"]): entry for entry in found["weights"]}
    assert len(weights) == 2
    for entry in weights.values():
        assert len(str(entry["sha256"])) == 64
        assert int(entry["bytes"]) > 0
    assert found["location"]["weights_tracked"] is False


def test_the_confrover_notice_states_the_change_made_to_apache_code():
    """Apache-2.0 section 4(b) asks a modified file to say it was changed.

    The vendored tree carries OpenFold and Ligo-Biosciences AlphaFold3, and
    their import paths were rewritten when the package was renamed. That is a
    modification, so it is stated -- once, at the tree level, rather than as a
    banner on 92 files that would read as a claim over code that is not ours.
    """
    notice = SUBPACKAGE / "confrover" / "NOTICE"
    if not notice.is_file():
        pytest.skip("confrover is not vendored in this checkout")
    said = notice.read_text(encoding="utf-8")
    assert "Apache License 2.0" in said
    assert "import paths" in said
    assert "OpenFold" in said and "Ligo" in said
    tree = SUBPACKAGE / "confrover" / "rbase"
    if not tree.is_dir():
        return
    # And no file under the third-party trees carries this project's banner.
    branded = [
        str(path.relative_to(REPO))
        for path in (tree / "_ext").rglob("*.py")
        if "Artificial Intelligence Canada" in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert branded == [], f"third-party files carrying this project's banner: {branded[:5]}"


def test_the_confrover_tree_keeps_the_file_the_generator_would_otherwise_write():
    """``stereo_chemical_props.txt`` has to stay, or a run writes into the checkout.

    The generator's own setup step copies that file into
    ``_ext/openfold/resources/`` when it is absent. Vendored, the copy is
    skipped; dropped, every run would modify the tree whose digest the manifest
    records.
    """
    tree = SUBPACKAGE / "confrover" / "rbase"
    if not tree.is_dir():
        pytest.skip("confrover is not vendored in this checkout")
    assert (tree / "_ext" / "openfold" / "resources" / "stereo_chemical_props.txt").is_file()


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
