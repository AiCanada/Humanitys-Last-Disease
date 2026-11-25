"""The shape of ``screen.json``, pinned beside the version that describes it.

``SCREEN_VERSION`` exists so a reader can tell which record it is holding
rather than inferring it from which keys happen to be present. That only works
if the number moves when the shape does, and it did not: the record gained
roughly fourteen top-level keys while the version stayed at 1, so screens of
two different shapes both claimed to be shape 1.

Remembering to raise it is the discipline that failed, so this file does not
rely on memory. Both key sets are pinned here as literals beside the version
number. Add a key and the pinned set fails; the fix puts `SCREEN_VERSION` in
front of whoever is editing it.

If you are here because one of these failed: that is the test working. Add the
key to `SCREEN_REQUIRED_KEYS` or `SCREEN_OPTIONAL_KEYS` in `screen.py`, add it
to the matching literal below, and raise `SCREEN_VERSION`.
"""

import json
from pathlib import Path

import pytest

from rfproteina.disease.screen import (
    SCREEN_OPTIONAL_KEYS,
    SCREEN_REQUIRED_KEYS,
    SCREEN_VERSION,
    screen_generation,
)
from tests.helpers.disease import (
    UBIQUITIN,
    binder_cif,
    no_developer_files,  # noqa: F401
)

#: Raised whenever either set below changes. See the module docstring.
_EXPECTED_VERSION = 19

_REQUIRED = (
    "bar",
    "bar_axes",
    "call",
    "code",
    "cuts",
    "designed_interface",
    "designed_interface_quality",
    "hits",
    "inputs",
    "metrics_file",
    "ppi_blocker",
    "ppi_pass",
    "pre_oracle_pass",
    "refold",
    "requested_class",
    "samples",
    "screen_version",
    "sequence_developability",
    "solubility",
    "states",
    "untested",
)
_OPTIONAL = (
    "chain_blocker",
    "chemical_liability",
    "delphi_psr",
    "design_diversity",
    "job_mismatch",
    "ligand_alerts",
    "ligand_ames",
    "ligand_caco2",
    "ligand_dili",
    "ligand_microsomal_clearance",
    "ligand_solubility",
    "method_selection",
    "metrics_blocker",
    "mhc_class_i",
    "modality_screen",
    "molprobity_geometry",
    "nucleic_liability",
    "peptide_rbc_evidence",
    "prodigy_affinity",
    "protein_sol",
    "qupkake",
    "replaced",
    "sequence_liability",
    "sygma",
    "synthesis",
    "target_accession",
    "target_liability",
    "tnp_profile",
    "toxicology",
    "vina_pose",
)

_PROTEIN = UBIQUITIN
_PEPTIDE = "DEKRDEKRDEKRDEKRDEKR"
_OLIGO = "ATGCATGCATGCATGCATGCATGCATGCATGC"


def _build(
    root: Path,
    name: str,
    designs: dict[str, str],
    *,
    files: dict[str, str] | None = None,
    metrics: bool = True,
) -> Path:
    output = root / name
    output.mkdir(parents=True, exist_ok=True)
    for design, sequence in designs.items():
        folder = output / "inference_samples" / design
        folder.mkdir(parents=True)
        (folder / f"{design}_generated.cif").write_text(binder_cif(sequence), encoding="utf-8")
    if metrics:
        table = output / "inference_metrics" / "per_example_metrics.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        rows = "".join(f"{design},A\n" for design in designs)
        table.write_text("example_id,chain_of_interest\n" + rows, encoding="utf-8")
    for filename, text in (files or {}).items():
        (output / filename).write_text(text, encoding="utf-8")
    return output


def test_the_version_matches_the_shape_pinned_here() -> None:
    """Raise SCREEN_VERSION in the same edit that changes either set below."""
    assert SCREEN_VERSION == _EXPECTED_VERSION


def test_the_declared_required_keys_are_the_ones_pinned_here() -> None:
    assert tuple(sorted(SCREEN_REQUIRED_KEYS)) == _REQUIRED


def test_the_declared_optional_keys_are_the_ones_pinned_here() -> None:
    assert tuple(sorted(SCREEN_OPTIONAL_KEYS)) == _OPTIONAL


def test_no_key_is_both_required_and_optional() -> None:
    assert not SCREEN_REQUIRED_KEYS & SCREEN_OPTIONAL_KEYS


@pytest.mark.parametrize(
    ("kind", "sequence"),
    [("protein", _PROTEIN), ("peptide", _PEPTIDE), ("nucleic", _OLIGO), ("other", _PROTEIN)],
)
def test_every_kind_writes_the_required_keys_and_nothing_undeclared(tmp_path: Path, kind: str, sequence: str) -> None:
    record = screen_generation(
        _build(tmp_path, kind, {"binder_0": sequence}),
        binder_kind=kind,
        target_id="P05113",
    )
    assert set(record) >= SCREEN_REQUIRED_KEYS
    undeclared = set(record) - SCREEN_REQUIRED_KEYS - SCREEN_OPTIONAL_KEYS
    assert not undeclared, f"undeclared key(s) {sorted(undeclared)}: raise SCREEN_VERSION"


def test_a_run_with_nothing_supplied_still_writes_the_required_keys(tmp_path: Path) -> None:
    """The floor of the shape: no designs, no metrics, no tables."""
    output = tmp_path / "bare"
    output.mkdir()
    record = screen_generation(output, binder_kind="protein", target_id="P05113")
    assert set(record) >= SCREEN_REQUIRED_KEYS
    assert record["screen_version"] == SCREEN_VERSION


def test_a_fully_loaded_run_writes_nothing_undeclared(tmp_path: Path) -> None:
    """Every optional key this fixture can reach, at once."""
    output = _build(
        tmp_path,
        "loaded",
        {"binder_0": _PROTEIN, "binder_1": _PEPTIDE},
        files={
            "axis_evidence.json": json.dumps({"paralog": "clear"}),
            "solubility.tsv": "score\tmethod\tsource\n0.72\tCamSol\tSormanni 2015\n",
        },
    )
    record = screen_generation(output, binder_kind="protein", target_id="P05113")
    undeclared = set(record) - SCREEN_REQUIRED_KEYS - SCREEN_OPTIONAL_KEYS
    assert not undeclared, f"undeclared key(s) {sorted(undeclared)}: raise SCREEN_VERSION"
    # The optional keys this run should have reached, so the fixture cannot
    # quietly stop exercising them.
    assert {"chemical_liability", "modality_screen", "design_diversity"} <= set(record)


def test_the_version_is_written_into_the_file(tmp_path: Path) -> None:
    output = _build(tmp_path, "written", {"binder_0": _PROTEIN})
    screen_generation(output, binder_kind="protein", target_id="P05113")
    stored = json.loads((output / "screen.json").read_text(encoding="utf-8"))
    assert stored["screen_version"] == _EXPECTED_VERSION
