"""Identity, scope, settings and end-to-end exposure attribution at the PK-Sim boundary."""

import hashlib
import json

import pytest

from rfproteina.disease import pksim

pytest.importorskip("rdkit")

DRUG = "Parent"
SIM = "Infusion control, individual 1"
PATH = "Organism|Kidney|Parent|Tissue"
SECOND = "Organism|Liver|Product|Tissue"
KEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def project():
    return {
        "ApplicationName": "PK-Sim",
        "Name": "Example",
        "Compounds": [{"Name": DRUG, "IsSmallMolecule": True}, {"Name": "Product", "IsSmallMolecule": True}],
        "Individuals": [{"Name": "I", "OriginData": {"Species": "Human"}}],
        "Protocols": [{"Name": "single", "ApplicationType": "Intravenous"}],
        "Simulations": [
            {
                "Name": SIM,
                "Individual": "I",
                "Compounds": [{"Name": DRUG, "Protocol": {"Name": "single"}}, {"Name": "Product"}],
                "OutputSelections": [PATH, SECOND],
                "OutputSchema": [
                    {
                        "Parameters": [
                            {"Name": "Start time", "Value": 0, "Unit": "h"},
                            {"Name": "End time", "Value": 1, "Unit": "h"},
                        ]
                    }
                ],
            }
        ],
    }


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    calls = []
    model = project()

    def configure(model, *, candidate="a", compound=DRUG, extra=None, csv=None):
        raw = json.dumps(model).encode()
        digest = hashlib.sha256(raw).hexdigest()
        case = {"model_sha256": digest, "simulation": SIM, "compound": compound, "outputs": [SECOND, PATH]}
        entries = {candidate: case, **(extra or {})}
        (tmp_path / pksim.MANIFEST).write_text(json.dumps({"samples": entries}))
        table = "candidate_id\tsmiles\na\tCCO\nb\tCCC\n"
        (tmp_path / "ligand_candidates.tsv").write_text(table)
        monkeypatch.setattr(pksim.pksim_runtime, "load_model", lambda d, c: (raw, KEY))

        def solve(data, sim):
            assert data == raw and sim == SIM
            calls.append(digest)
            return (
                csv
                if csv is not None
                else (
                    f"IndividualId,Time [min],{PATH} [µmol/l],{SECOND} [mg/l]\n0,0,0,1\n0,30,7,1\n0,60,3,2\n"
                ).encode()
            )

        monkeypatch.setattr(pksim.pksim_runtime, "run", solve)
        return case

    case = configure(model)
    return tmp_path, configure, model, calls, case


def test_ready_without_solver_then_report_native_parent_and_product_separately(prepared):
    run, _, _, calls, _ = prepared
    assert pksim.readiness(run, "a")[:2] == (True, True) and not calls
    report = pksim.measure(run)
    assert report["attempted"] == report["simulated"] == 1 and report["status"] == "simulated"
    row = report["samples"][0]
    assert len(calls) == 1 and row["simulation"] == SIM and row["compound_inchikey"] == KEY
    assert row["species"] == "Human" and row["protocol"] == "single" and row["configured_window_min"] == [0, 60]
    product, parent = row["curves"]
    assert (product["analyte"], product["concentration_unit"], product["grid_cmax"]) == ("Product", "mg/l", 2)
    assert (parent["analyte"], parent["organ_or_blood_pool"], parent["auc_trapezoidal"]) == (DRUG, "Kidney", 255)
    assert all(c["method"] == pksim.METHOD_ID for c in row["curves"])


def test_mismatched_compound_and_absent_candidate_cannot_receive_another_curve(prepared):
    run, config, model, calls, case = prepared
    config(model, extra={"b": case})
    record = pksim.measure(run)
    assert record["status"] == "partial" and record["attempted"] == 2 and record["simulated"] == 1
    assert record["samples"][1]["blocker"] == "pksim_molecule_mismatch"
    assert pksim.measure(run, "outside")["blocker"] == "pksim_candidate_not_nominated"
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["species", "population", "protocol", "duplicate", "model", "observer", "coadmin"])
def test_invalid_scenario_fails_before_solver(prepared, change):
    run, config, model, calls, _ = prepared
    sim = model["Simulations"][0]
    if change == "species":
        model["Individuals"][0]["OriginData"]["Species"] = "Rat"
    elif change == "population":
        sim["Population"] = "cohort"
    elif change == "protocol":
        model["Protocols"].clear()
    elif change == "duplicate":
        model["Simulations"].append(sim)
    elif change == "model":
        model["ApplicationName"] = "MoBi"
    elif change == "observer":
        sim["OutputSelections"] = [PATH]
    else:
        sim["Compounds"][1]["Protocol"] = {"Name": "single"}
    config(model)
    record = pksim.measure(run)
    assert record["status"] == "untested" and record["blocker"] == "pksim_invalid_model"
    assert not calls


def test_missing_mismatched_or_unreadable_inputs(prepared, monkeypatch):
    run, _, _, calls, _ = prepared
    monkeypatch.setattr(pksim.pksim_runtime, "load_model", lambda d, c: (b"{}", KEY))
    assert pksim.measure(run)["blocker"] == "pksim_model_mismatch"
    (run / pksim.MANIFEST).write_text('{"samples":{},"samples":{}}')
    assert pksim.measure(run)["blocker"] == "pksim_invalid_manifest"
    (run / pksim.MANIFEST).unlink()
    assert pksim.measure(run)["blocker"] == "pksim_manifest_missing"
    assert not calls


def test_scenario_local_overrides_require_the_effective_subject_and_regimen(prepared):
    run, configure, model, calls, _ = prepared
    model["Simulations"][0]["AlteredBuildingBlocks"] = [{"Name": "I", "Type": "Individual"}]
    configure(model)
    assert pksim.readiness(run, "a")[0] is False
    assert pksim.measure(run, "a")["blocker"] == "pksim_unsupported_snapshot"
    assert not calls


def test_submitted_table_is_authoritative_and_single_smiles_remains_run_scoped(prepared):
    run, config, model, calls, _ = prepared
    (run / "ligand_smiles.txt").write_text("CCO")
    (run / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\na\tCCO\na\tCCO\n")
    assert pksim.measure(run, "a")["blocker"] == "pksim_molecule_mismatch"
    (run / "ligand_candidates.tsv").unlink()
    assert pksim.measure(run, "a")["blocker"] == "pksim_molecule_mismatch"
    config(model, candidate="run_ligand")
    (run / "ligand_candidates.tsv").unlink()
    assert pksim.measure(run, "run_ligand")["blocker"] == "pksim_molecule_mismatch"
    assert pksim.measure(run)["status"] == "simulated" and len(calls) == 1


def test_truncated_valid_numeric_export_does_not_complete_scenario(prepared):
    run, config, model, _, _ = prepared
    config(model, csv=f"IndividualId,Time [h],{PATH} [mg/l],{SECOND} [mg/l]\n0,0,1,2\n0,0.5,2,3\n".encode())
    row = pksim.measure(run)["samples"][0]
    assert row["blocker"] == "pksim_invalid_export" and "curves" not in row


def test_scope_respects_input_and_result_budgets(prepared):
    run, config, model, calls, case = prepared
    config(model, extra={f"other{i}": case for i in range(8)})
    assert pksim.measure(run)["blocker"] == "pksim_budget_exceeded" and not calls
    header = f"IndividualId,Time [min],{PATH} [mg/l],{SECOND} [mg/l]\n"
    data = "".join(f"0,{i},1,2\n" for i in range(4097))
    config(model, csv=(header + data).encode())
    assert pksim.measure(run)["blocker"] == "pksim_budget_exceeded" and len(calls) == 1


def test_documented_cimetidine_project_matches_native_reference(tmp_path):
    """Published OSP project, plasma observer and concentration grid from the native control."""
    digest = "314c835349119455bc5c60c16febbf301b55de97eb0f3e4279313949b4b6f8aa"
    if not pksim.pksim_runtime.installed():
        pytest.skip("Requires the pinned portable PK-Sim release")
    if not (pksim.pksim_runtime.cache_path() / "models" / digest).exists():
        pytest.skip("Register the published cimetidine project from the screen input guide")
    observer = "Organism|PeripheralVenousBlood|Cimetidine|Plasma (Peripheral Venous Blood)"
    case = {
        "model_sha256": digest,
        "simulation": "iv 100 mg (5 min), Grahnen 1979, n=3",
        "compound": "Cimetidine",
        "outputs": [observer],
    }
    (tmp_path / pksim.MANIFEST).write_text(json.dumps({"samples": {"control": case}}))
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\ncontrol\tCC1=C(N=CN1)CSCCNC(=NC)NC#N\n")
    report = pksim.measure(tmp_path, candidate="control")
    assert report["status"] == "simulated", report
    (curve,) = report["samples"][0]["curves"]
    assert curve["output_path"] == observer
    assert (curve["concentration_unit"], curve["time_unit"]) == ("µmol/l", "min")
    assert curve["time"] == list(range(0, 1441, 6))
    assert curve["grid_cmax"] == pytest.approx(11.01352, rel=1e-6)
    assert curve["auc_trapezoidal"] == pytest.approx(541.294066098, rel=1e-6)
