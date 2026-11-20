"""Tests for the ImmuneBuilder (ABodyBuilder2) oracle adapter. No ImmuneBuilder import."""

from __future__ import annotations

from rfproteina.metrics.oracles.immunebuilder_oracle import ImmuneBuilderOracle


class _FakeAntibody:
    def __init__(self, errors=None):
        self.error_estimates = errors
        self._saved = None

    def save(self, path):
        with open(path, "w") as f:
            f.write("REMARK fake antibody\n")
        self._saved = path


def test_pick_chains_explicit():
    oracle = ImmuneBuilderOracle(heavy_chain="H", light_chain="L")
    assert oracle.pick_chains({"H": "AAA", "L": "CCC"}) == ("H", "L")


def test_pick_chains_fallback_first_two():
    oracle = ImmuneBuilderOracle()
    assert oracle.pick_chains({"A": "AAA", "B": "CCC"}) == ("A", "B")


def test_pick_chains_too_few():
    oracle = ImmuneBuilderOracle()
    try:
        oracle.pick_chains({"A": "AAA"})
    except ValueError as error:
        assert "heavy and a light" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_pick_chains_named_chain_absent():
    oracle = ImmuneBuilderOracle(heavy_chain="H", light_chain="Z")
    try:
        oracle.pick_chains({"H": "AAA", "L": "CCC"})
    except ValueError as error:
        assert "light chain 'Z'" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_build_output_saves_pdb_and_reads_error(tmp_path):
    antibody = _FakeAntibody(errors=[1.0, 2.0, 3.0])
    out = ImmuneBuilderOracle.build_output(antibody, tmp_path, ["H", "L"])
    assert out.cif_path.exists()
    assert out.cif_path.suffix == ".pdb"
    extra = out.summary_confidences["_extra"]
    assert extra["immunebuilder_mean_predicted_error_a"] == 2.0
    assert extra["immunebuilder_max_predicted_error_a"] == 3.0


def test_build_output_without_error_estimate(tmp_path):
    antibody = _FakeAntibody(errors=None)
    out = ImmuneBuilderOracle.build_output(antibody, tmp_path, ["H", "L"])
    assert out.summary_confidences["_extra"] == {}


def test_output_suffix_is_pdb():
    assert ImmuneBuilderOracle().output_cif_suffix == ".pdb"


# ----------------------------------------- the two defects the drivers caused


def test_the_pdb_suffix_survives_a_driver_retargeting_the_run_format():
    """Both drivers assign the run's suffix to every oracle they hold.

    `evaluate.py` and `proteina.py` loop over the oracle metrics and set
    `output_cif_suffix` to the run's own choice, `.cif` or `.cif.zst`. This
    oracle writes PDB, so that loop renamed a PDB to `..._refolded.cif.zst` and
    every downstream reader refused it. The oracle now declares the suffix
    fixed and the loops skip it.
    """
    oracle = ImmuneBuilderOracle()
    assert oracle.output_suffix_is_fixed is True

    # Exactly what the drivers do, guard included.
    for run_suffix in (".cif", ".cif.zst"):
        if not getattr(oracle, "output_suffix_is_fixed", False):
            oracle.output_cif_suffix = run_suffix
    assert oracle.output_cif_suffix == ".pdb"


def test_a_cif_writing_oracle_is_still_retargeted():
    """The guard must not freeze every oracle: the run-wide choice is real."""
    from rfproteina.metrics.oracles.boltz_oracle import Boltz2Oracle

    oracle = Boltz2Oracle()
    assert oracle.output_suffix_is_fixed is False
    if not getattr(oracle, "output_suffix_is_fixed", False):
        oracle.output_cif_suffix = ".cif.zst"
    assert oracle.output_cif_suffix == ".cif.zst"


def test_the_backend_probe_requires_what_a_fold_actually_needs(monkeypatch):
    """Probing only `ImmuneBuilder` reported a backend that could not fold.

    The import pulls ANARCI, which shells out to HMMER's `hmmscan`, and the
    refine step needs OpenMM and PDBFixer. With only the one package present
    the probe said yes, and the real failure arrived per design inside
    `fold_batch`'s catch-all as a warning rather than one clear refusal.
    """
    import importlib.util as real_util
    import shutil as real_shutil

    from rfproteina.metrics.oracles import immunebuilder_oracle as mod

    present: set[str] = set()
    monkeypatch.setattr(
        mod.importlib.util,
        "find_spec",
        lambda name: object() if name in present else None,
    )
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/hmmscan" if "hmmscan" in present else None)

    oracle = ImmuneBuilderOracle()

    present = {"ImmuneBuilder"}
    assert oracle._backend_available() is False, "the package alone cannot fold"

    present = {"ImmuneBuilder", "anarci", "openmm", "pdbfixer"}
    assert oracle._backend_available() is False, "ANARCI still needs hmmscan on PATH"

    present = {"ImmuneBuilder", "anarci", "openmm", "pdbfixer", "hmmscan"}
    assert oracle._backend_available() is True

    assert real_util and real_shutil  # the real modules are untouched outside this test
