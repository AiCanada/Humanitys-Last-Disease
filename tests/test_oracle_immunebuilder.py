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
