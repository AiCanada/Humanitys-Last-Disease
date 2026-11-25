"""Keep picker guidance tied to the quantity each named screen actually reports."""

import pytest

from rfproteina.disease import pksim_curves, screen_performance, toxicology


@pytest.mark.parametrize(
    ("method", "operation", "basis", "readout", "application"),
    [
        ("cellprofiler", "compute", "matched staining controls", "mean DAB", "pathologist"),
        ("peptide_rbc_evidence", "native_result_reader", "red-cell assay", "hemolysis percentage", "species and matrix"),
        ("octant", "native_result_reader", "CYP3A4 inhibition", "median IC50", "metabolizes"),
        ("chembl_panel", "recorded_evidence", "by assay and endpoint", "determinate nM", "activity records"),
        ("bindingdb_exact", "native_result_reader", "full InChIKey", "study source", "target preparation"),
        ("openmm", "recorded_evidence", "Single-structure", "changed the sign", "component energies"),
        ("dynamics", "recorded_evidence", "ensemble frames", "radius of gyration", "independent samples"),
    ],
)
def test_screen_specific_evidence_survives_serialization(method, operation, basis, readout, application):
    record = screen_performance.for_method(method, operation)
    assert basis in record["basis"]
    assert readout in record["summary"]
    assert application in record["interpretation"]


def test_toxicology_guidance_describes_how_the_reader_handles_bounds(tmp_path):
    table = tmp_path / "toxicology.tsv"
    table.write_text("axis\tnm\nherg\t>5000\n", encoding="utf-8")
    unresolved = toxicology.measure_toxicology(tmp_path)["herg"]
    assert unresolved["state"] == "untested" and unresolved["blocker"] == "reading_not_decisive"
    table.write_text("axis\tnm\nherg\t>20000\n", encoding="utf-8")
    resolved = toxicology.measure_toxicology(tmp_path)["herg"]
    assert (resolved["state"], resolved["measurement"], resolved["qualifier"]) == ("clear", 20000, ">")
    evidence = screen_performance.for_method("toxicology_table", "native_result_reader")
    assert "qualifier" in evidence["summary"] and "bound spans the cutoff" in evidence["interpretation"]


def test_pksim_guidance_identifies_the_effect_of_the_exported_time_grid():
    observer = "Organism|Liver|Analyte|Tissue"
    header = f"IndividualId,Time [min],{observer} [mg/l]\n".encode()
    coarse, = pksim_curves.extract(header + b"0,0,1\n0,2,1\n", (observer,))
    fine, = pksim_curves.extract(header + b"0,0,1\n0,1,3\n0,2,1\n", (observer,))
    assert (coarse["grid_cmax"], fine["grid_cmax"]) == (1, 3)
    assert (coarse["auc_trapezoidal"], fine["auc_trapezoidal"]) == (2, 4)
    evidence = screen_performance.for_method("pksim", "compute")
    assert "sampled interval" in evidence["summary"] and "coarse time grid" in evidence["interpretation"]
