"""The scientific picker accounts for installation, candidate identity, compute and evidence provenance."""

import json
from pathlib import Path

import pytest

from rfproteina.disease import mcp_api, screen_inventory, screen_panel, screen_selection
from rfproteina.disease.ui import (
    SCREENING_ACTION,
    SCREENING_FIELD,
    VALIDATION_CANDIDATE_FIELD,
    BrowserSession,
    OptionPanel,
)
from rfproteina.disease.validation_checks import candidate_validation_plan


@pytest.fixture
def finished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "run"
    samples = directory / "inference_samples"
    samples.mkdir(parents=True)
    for name in ("binder_0", "binder_1"):
        (samples / f"{name}_generated.cif").write_text("data_example\n")
    (directory / "generation.json").write_text('{"binder_kind":"ligand"}')
    (directory / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\nbinder_0\tCCO\n")
    (directory / screen_inventory.ATTACHMENT).write_text(
        json.dumps(
            {
                "pmx": [{"candidate_id": "binder_0", "endpoint": "free energy", "unit": "kJ/mol", "value": 0.0}],
            }
        )
    )
    available = dict.fromkeys(screen_selection.METHOD_IDS, True)
    available["tnp_profile"] = False
    monkeypatch.setattr(screen_selection, "installations", lambda keys=None: dict(available))
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    return directory


def test_coverage_has_a_unique_selection_for_every_reviewed_method() -> None:
    assert len(screen_selection.METHOD_IDS) == len(set(screen_selection.METHOD_IDS)) == 60
    assert set(screen_selection.METHOD_IDS) == set(screen_selection.NATIVE_IDS) | screen_inventory.BY_KEY.keys()


def test_stereochemistry_preserves_the_producers_fraction_and_scope(finished):
    folder = finished / "inference_metrics"
    folder.mkdir()
    metrics = folder / "per_example_metrics.csv"
    endpoint = "chiral_accuracy.tetrahedral_percent_correct_chirality"

    def choice(example):
        result, = screen_selection.plan(finished, "ligand", example=example, available={"stereochemistry": True})
        return result

    metrics.write_text(f"example_id,{endpoint}\nbinder_0,0\n")
    assert choice("binder_0").recommended
    assert choice(None).applicable and not choice(None).recommended
    metrics.write_text(f"example_id,{endpoint}\nbinder_0,90\n")
    assert choice("binder_0").applicable and not choice("binder_0").recommended


def test_clashes_and_bond_lengths_need_different_atom_pairs():
    import numpy as np
    from atomworks.ml.conditions import C_CRD
    from biotite.structure import AtomArray, BondList, BondType

    from rfproteina.disease.screen_physical import has_clash_pair, has_measurable_bonds

    atoms = AtomArray(2)
    atoms.coord = np.zeros((2, 3), dtype=np.float32)
    assert has_clash_pair(atoms) and not has_measurable_bonds(atoms)
    atoms.bonds = BondList(2, np.array([[0, 1, BondType.SINGLE]]))
    assert has_measurable_bonds(atoms) and not has_clash_pair(atoms)
    atoms.bonds = None
    C_CRD.set_annotation_from_ground_truth(atoms, np.array([True, True]))
    assert not has_clash_pair(atoms)


def test_two_supplied_assays_for_one_compound_stay_measurable(tmp_path):
    from rfproteina.disease import screen_assays

    (tmp_path / "ligand_smiles.txt").write_text("CCO\n")
    path = tmp_path / "toxicology.tsv"
    header = "protein\tstandard_value\tstandard_units\tstandard_type\tassay_id\tsource\tinchikey\n"
    identity = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    path.write_text(header + f"herg\t120\tnM\tIC50\ta\tstudy\t{identity}\nherg\t240\tnM\tIC50\tb\tstudy\t{identity}\n")
    choice, = screen_selection.plan(tmp_path, "ligand", available={"toxicology_table": True})
    assert choice.recommended
    result = screen_assays.measure(tmp_path, "toxicology_table", None)
    assert result["status"] == "reported" and result["measured_axes"] == 1
    assert {r["measurement"] for r in result["axes"]["herg"]["opinions"]} == {120, 240}
    (tmp_path / "ligand_smiles.txt").write_text("CC(=O)O\n")
    choice, = screen_selection.plan(tmp_path, "ligand", available={"toxicology_table": True})
    assert not choice.applicable


@pytest.mark.parametrize(
    ("key", "file", "settings"),
    [
        ("tnp_profile", "tnp_domains.json", {"samples": {"foreign": {}}}),
        ("vina_pose", "vina_poses.json", {"samples": {"foreign": {}}}),
        ("prodigy_affinity", "prodigy_partners.json", {"binder_chains": ["X"], "target_chains": ["Y"]}),
    ],
)
def test_prepared_manifests_require_matching_structure_inputs(finished, key, file, settings):
    (finished / file).write_text(json.dumps(settings))
    kind = "ligand" if key == "vina_pose" else "protein"
    choices = screen_selection.plan(finished, kind, example="binder_0", available={key: True})
    choice, = choices
    assert not choice.applicable and not choice.recommended
    assert screen_selection.select(choices, "all_installed") == (key,)


def test_ligand_component_picker_keeps_formula_and_stated_weight_distinct(tmp_path):
    import gzip

    from rfproteina.disease import screen_criteria, screen_physical

    folder = tmp_path / "inference_samples"
    folder.mkdir()
    path = folder / "binder_0_generated.cif.gz"
    common = "data_ligand\n_chem_comp.id EOH\n_chem_comp.type non-polymer\n_chem_comp.formula 'C2 H6 O'\n"
    path.write_bytes(gzip.compress((common + "_chem_comp.formula_weight 46.068\n").encode()))
    structures = screen_physical.Structures(tmp_path)
    assert screen_criteria.ligand_component(structures, 1024).recommended
    smaller = screen_criteria.ligand_component(structures, 4)
    assert smaller.applicable and not smaller.recommended
    path.write_bytes(gzip.compress((common + "_chem_comp.formula_weight 99\n").encode()))
    choice, = screen_selection.plan(tmp_path, "ligand", available={"ligand_composition": True})
    assert choice.applicable and not choice.recommended
    measured, = structures.measure("ligand_composition")["samples"]
    assert measured["component_id"] == "EOH"
    assert measured["formula_weight_Da"] == 99 and measured["calculated_formula_weight_Da"] == 46.068


def test_protein_composition_uses_the_assigned_polypeptide(tmp_path):
    from rfproteina.disease.screen_physical import Structures

    folder = tmp_path / "inference_samples"
    folder.mkdir()
    metrics = tmp_path / "inference_metrics"
    metrics.mkdir()
    (metrics / "per_example_metrics.csv").write_text("example_id,chain_of_interest\nbinder_0,A\n")
    path = folder / "binder_0_generated.cif"
    card = (
        "data_sample\n_entity_poly.pdbx_strand_id A\n_entity_poly.type '{}'\n"
        "_entity_poly.pdbx_seq_one_letter_code_can {}\n_entity_poly.pdbx_seq_one_letter_code '{}'\n"
    )
    available = {"protein_composition": True}
    path.write_text(card.format("polyribonucleotide", "AG", "AG"))
    assert not screen_selection.plan(tmp_path, "protein", available=available)[0].applicable
    path.write_text(card.format("polypeptide(L)", "AS", "A(SEP)"))
    choice, = screen_selection.plan(tmp_path, "protein", available=available)
    assert choice.applicable and not choice.recommended
    row, = Structures(tmp_path).measure("protein_composition")["samples"]
    assert row["mass_basis"] == "unmodified_linear_chain"
    assert row["input_warning"] == "modified_form_requires_mass_confirmation"
    assert row["residue_counts"] == {"A": 1, "S": 1}
    path.write_text(card.format("polypeptide(L)", "R" * 101, "R" * 101))
    choice, = screen_selection.plan(tmp_path, "peptide", available={"protein_charge": True})
    assert choice.applicable and not choice.recommended
    row, = Structures(tmp_path).measure("protein_charge")["samples"]
    assert row["pka_set"] == "bjellqvist_1994" and "isoelectric_point" not in row
    assert row["net_charge_at_pH_7_4"] > 0 and row["isoelectric_point_blocker"]


def test_sequence_diversity_requires_two_resolved_chains(finished):
    from rfproteina.disease import screen_physical

    folder = finished / "inference_samples"
    metrics = finished / "inference_metrics"
    metrics.mkdir(exist_ok=True)
    (metrics / "per_example_metrics.csv").write_text("example_id,chain_of_interest\nbinder_0,A\nbinder_1,A\n")
    card = "data_poly\n_entity_poly.pdbx_strand_id A\n_entity_poly.pdbx_seq_one_letter_code_can {}\n"
    for path in folder.glob("*_generated.cif"):
        path.write_text(card.format("AA"))
    available = {"sequence_diversity": True}
    assert screen_selection.plan(finished, "protein", available=available)[0].recommended
    assert not screen_selection.plan(finished, "protein", example="binder_0", available=available)[0].applicable
    report = screen_physical.Structures(finished).measure("sequence_diversity")
    assert report["cohort"]["distinct"] == 1 and report["cohort"]["compared_pairs"] == 1
    (folder / "binder_1_generated.cif").write_text(card.format("AX"))
    assert not screen_selection.plan(finished, "protein", available=available)[0].applicable


def test_saved_observations_exclude_candidates_from_other_runs(finished):
    rows = [
        {"candidate_id": name, "endpoint": "DockQ", "unit": "dimensionless", "value": 0.0}
        for name in ("binder_0", "from_another_run")
    ]
    path = finished / screen_inventory.ATTACHMENT
    path.write_text(json.dumps({"dockq": rows}))
    record = screen_inventory.Evidence(finished, None).coverage(screen_inventory.BY_KEY["dockq"], include_values=True)
    assert record["unmatched_observations"] == 1
    assert [row["candidate_id"] for row in record["observations"]] == ["binder_0"]
    path.write_text(json.dumps({"dockq": rows[1:]}))
    choice, = screen_selection.plan(finished, "protein", available={"dockq": True})
    assert not choice.applicable and not choice.recommended


def test_rdkit_default_tracks_molecular_identity_without_running_alerts(finished, monkeypatch):
    from rfproteina.disease import ligand_alerts

    (finished / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\nbinder_0\tCCO\nbinder_1\tCCO.O\ndup\tC\ndup\tCO\n"
        "bad\t*CC\nextra\tC\tC\nlabel\tCCO appended-name\n"
    )
    monkeypatch.setattr(ligand_alerts, "screen_smiles", lambda *a, **kw: pytest.fail("picker executed a screen"))

    def choice(example):
        return next(c for c in screen_selection.plan(finished, "ligand", example=example)
                    if c.method.key == "ligand_alerts")

    assert choice("binder_0").recommended
    for example in ("bad", "extra", "label", "dup", "missing"):
        assert not choice(example).applicable
    for example in (None, "binder_1"):
        row = choice(example)
        assert row.applicable and not row.recommended
        assert screen_selection.select([row], "all_installed") == ("ligand_alerts",)
    monkeypatch.setattr(ligand_alerts, "CATALOGUES", (*ligand_alerts.CATALOGUES, "ABSENT"))
    partial = choice("binder_0")
    assert partial.applicable and not partial.recommended and "ABSENT" in partial.reason


def test_rdkit_legacy_input_is_run_scoped_and_cannot_hide_a_second_ligand(finished):
    from rfproteina.disease import screen

    (finished / "ligand_candidates.tsv").unlink()
    path = finished / "ligand_smiles.txt"
    path.write_text("CCO ethanol\n", encoding="utf-8-sig")
    assert screen._ligand_alerts_block(finished)["mol_weight"] == pytest.approx(46.07, abs=0.01)
    assert not next(c for c in screen_selection.plan(finished, "ligand", example="run_ligand")
                    if c.method.key == "ligand_alerts").applicable
    path.write_text("CCO ethanol\nCO methanol\n")
    assert screen._ligand_alerts_block(finished) == {"ligand_alerts_blocker": "expected_one_run_ligand"}
    assert not next(c for c in screen_selection.plan(finished, "ligand") if c.method.key == "ligand_alerts").applicable


def test_rdkit_manual_run_retains_unusable_rows_and_catalogue_gaps(finished, monkeypatch):
    from rfproteina.disease import ligand_alerts

    path = finished / "ligand_candidates.tsv"
    path.write_text("candidate_id\tsmiles\nvalid\tOc1ccccc1O\ninvalid\tCCO title\nsalt\tCC(=O)[O-].[Na+]\n")
    (result,) = screen_panel.run(finished, None, "explicit", ["ligand_alerts"])
    assert result["status"] == "partial"
    rows = {s["candidate_id"]: s for s in result["measurement"]["samples"]}
    assert rows["valid"]["alerts"]
    assert rows["invalid"]["blocker"] == "invalid_nominated_smiles"
    assert rows["salt"]["mol_weight"] == pytest.approx(82.034, abs=0.01)
    path.write_text("candidate_id\tsmiles\nbinder_0\tCCO\n")
    monkeypatch.setattr(ligand_alerts, "CATALOGUES", (*ligand_alerts.CATALOGUES, "ABSENT"))
    (partial,) = screen_panel.run(finished, None, "explicit", ["ligand_alerts"])
    assert partial["status"] == "partial"
    assert partial["measurement"]["samples"][0]["verdicts"]["structure_alerts"]["state"] == "untested"


def test_admet_prefill_checks_the_form_and_counts_distinct_model_inputs(finished, monkeypatch):
    from rfproteina.disease import ligand_solubility

    monkeypatch.setattr(ligand_solubility, "_model", lambda: pytest.fail("selection loaded the ensemble"))
    path = finished / "ligand_candidates.tsv"

    def choice(example=None):
        return next(c for c in screen_selection.plan(finished, "ligand", example=example)
                    if c.method.key == "ligand_solubility")

    path.write_text("candidate_id\tsmiles\nneutral\tCC(=O)O\nion\tCC(=O)[O-]\nsalt\tCC(=O)[O-].[Na+]\n"
                    "gas\t[Ar]\nwildcard\tC*\n")
    assert choice("neutral").recommended
    for name in (None, "ion", "salt", "gas"):
        result = choice(name)
        assert result.applicable and not result.recommended
    assert not choice("wildcard").applicable
    size = ligand_solubility.BATCH_SIZE
    path.write_text("candidate_id\tsmiles\n" + "".join(f"mol{i}\t{'C' * (i+1)}\n" for i in range(size)))
    assert choice().recommended
    with path.open("a") as stream:
        stream.write("same\t[CH4:5]\n")
    assert choice().recommended
    with path.open("a") as stream:
        stream.write("additional\tCCO\n")
    run = choice()
    assert run.applicable and not run.recommended and f"{size + 1} distinct" in run.reason
    assert choice("additional").recommended


def test_solubility_table_prefill_respects_attribution_and_scale(finished):
    header = "candidate_id\tscore\tmethod\tsource\tunit\n"
    one = "binder_0\t0\tshake-flask\tassay A\tmg/mL\n"
    path = finished / "solubility.tsv"

    def choice(example=None):
        (item,) = screen_selection.plan(finished, "ligand", example=example, available={"solubility": True})
        return item

    path.write_text(header + one + "unrelated\t0.2\tmodel\tstudy B\tindex\n")
    assert choice("binder_0").recommended
    assert not choice("binder_1").applicable
    assert choice().applicable and not choice().recommended
    path.write_text(header + one.replace("mg/mL", ""))
    assert choice().applicable and not choice().recommended
    path.write_text(header + one)
    assert choice().recommended
    path.write_text(header)
    assert not choice().applicable


def test_protein_sol_prefill_uses_the_same_sequence_identity_as_the_reader(tmp_path):
    import hashlib

    from rfproteina.disease import screen_criteria, screen_physical
    from tests.helpers.disease import UBIQUITIN, finished_run

    run = finished_run(tmp_path)
    header = "HEADERS PREDICTIONS LINE,ID,percent-sol,scaled-sol,population-sol,pI\n"
    prediction = "SEQUENCE PREDICTIONS,>binder_0,40,0.3,0.45,5.0\n"
    report = run / "seq_prediction.txt"
    report.write_text(header + prediction)

    def choice(kind="protein"):
        (result,) = screen_selection.plan(run, kind, example="binder_0", available={"protein_sol": True})
        return result

    assert not choice().applicable
    fasta = run / "protein_sol.fasta"
    fasta.write_text(">binder_0\n" + UBIQUITIN + "\n")
    assert choice().recommended
    assert choice("peptide").applicable and not choice("peptide").recommended
    limited = screen_criteria.protein_sol(run, "protein", screen_physical.Structures(run), 1)
    assert limited.applicable and not limited.recommended
    digest = "sha256:" + hashlib.sha256(UBIQUITIN.encode()).hexdigest()
    report.write_text(header + prediction + prediction.replace("binder_0", digest))
    assert not choice().applicable
    report.write_text(header + prediction.replace("binder_0", digest))
    fasta.unlink()
    assert choice().recommended


def test_default_and_all_installed_respect_compute_and_input_scope(finished: Path) -> None:
    choices = screen_panel.choices_for(finished, None)
    default = screen_selection.select(choices, "recommended")
    available = screen_selection.select(choices, "all_installed")
    assert "ligand_solubility" in default
    assert "qupkake" in available and "qupkake" not in default
    assert "sygma" in available and "sygma" not in default
    assert "tnp_profile" not in available
    assert "pmx" in available
    assert set(default) < set(available)
    assert all(c.installed and c.applicable for c in choices if c.recommended)
    assert not next(c for c in choices if c.method.key == "clashes").applicable


def test_only_the_nominated_candidate_gets_its_method_and_evidence(finished: Path) -> None:
    candidates, _ = candidate_validation_plan((finished,), None)
    for candidate in candidates:
        applicable = candidate.example_id == "binder_0"
        plan = {item.method.key: item for item in screen_panel.choices_for(finished, candidate.key)}
        assert plan["ligand_solubility"].applicable == applicable
        assert plan["pmx"].applicable == applicable
        (result,) = screen_panel.run(finished, candidate.key, "explicit", ["pmx"])
        if applicable:
            assert result["status"] == "recorded"
            assert result["measurement"]["attached_observations"] == 1
        else:
            assert result["status"] == "inapplicable"
            assert "measurement" not in result


def test_missing_weights_and_invalid_selection_are_rejected_before_running(finished: Path, monkeypatch) -> None:
    monkeypatch.setattr(screen_panel, "_measure", lambda *args: pytest.fail("unexpected dispatch"))
    for keys in (["tnp_profile"], ["ligand_solubility", "unknown"], ["sygma", "sygma"]):
        with pytest.raises(ValueError):
            screen_panel.run(finished, None, "explicit", keys)
    assert screen_panel.run(finished, None, "explicit", []) == []


def test_an_explicit_solubility_request_only_plans_that_method(finished, monkeypatch):
    monkeypatch.setitem(screen_selection._CRITERIA, "ligand_alerts", lambda *args: pytest.fail("planned RDKit alerts"))
    monkeypatch.setattr(screen_selection.screen_physical, "_coordinate_atoms", lambda *args: pytest.fail("read geometry"))
    choices, selected = screen_selection.execution_plan(finished, "ligand", "explicit", ["ligand_solubility"])
    assert selected == ("ligand_solubility",)
    assert [choice.method.key for choice in choices] == list(selected)


def test_browser_can_clear_recommendations_before_the_first_calculation(finished: Path) -> None:
    def submit(page):
        assert "Continue with selected screens" in page
        assert 'value="ligand_solubility" data-recommended checked' in page
        assert "Run selected screens</button>" not in page
        return {}

    assert OptionPanel(submit).choose_screens(finished, "ligand") == ()


def test_invalid_manifest_and_another_candidates_evidence_are_withheld(finished: Path) -> None:
    (finished / "sygma.json").write_text('{"phase1_cycles":9999}')
    plan = {c.method.key: c for c in screen_panel.choices_for(finished, None)}
    assert plan["sygma"].installed and not plan["sygma"].applicable and not plan["sygma"].recommended
    assert "invalid settings" in plan["sygma"].reason
    (row,) = screen_panel.run(finished, None, "explicit", ["pmx"])
    assert row["measurement"]["observations"] == [
        {
            "candidate_id": "binder_0",
            "endpoint": "free energy",
            "unit": "kJ/mol",
            "value": 0.0,
            "source": screen_inventory.ATTACHMENT,
        }
    ]


def test_all_failed_samples_cannot_report_a_completed_measurement() -> None:
    assert screen_panel._status("vina_pose", {"attempted": 2, "scored": 0}) == "untested"
    assert screen_panel._status("prodigy_affinity", {"attempted": 3, "predicted": 1}) == "partial"
    assert screen_panel._status("tnp_profile", {"attempted": 2, "profiled": 0, "partial": 1}) == "partial"
    assert screen_panel._status("protein_sol", {"samples": [{"blocker": "missing"}]}) == "untested"
    assert screen_panel._status("protein_sol", {"samples": [{"score": 0.0}]}) == "reported"


def test_each_method_finishes_before_the_next_progress_event(finished: Path) -> None:
    events = []
    rows = screen_panel.run(finished, None, "explicit", ["pmx", "clashes"], on_line=events.append)
    assert [r["status"] for r in rows] == ["recorded", "inapplicable"]
    assert [event.split()[0] for event in events] == ["Running", "Finished", "Skipped"]


def test_explicit_empty_cli_selection_remains_empty(finished: Path, monkeypatch) -> None:
    observed = []

    def invoke(argv, **kwargs):
        observed.extend(argv)
        return 0, []

    monkeypatch.setattr(mcp_api, "_call_main", invoke)
    mcp_api.screen_directory({"directory": str(finished), "screen_methods": []})
    assert observed[observed.index("--screen-selection") + 1] == "explicit"
    for value in ("sygma", ["typo"], ["sygma", "sygma"]):
        with pytest.raises(ValueError):
            mcp_api.screen_directory({"directory": str(finished), "screen_methods": value})


def test_ui_scope_switch_does_not_apply_previous_selection(finished: Path) -> None:
    panel = BrowserSession(opener=lambda _url: None)
    panel.note_validation_runs((finished,))
    page = panel.done_page()
    assert "Select all installed screens" in page
    assert "of 59 installed screens selected" in page and "60 listed" in page
    assert 'value="ligand_solubility" data-recommended checked' in page
    assert 'value="tnp_profile" disabled' in page
    candidates, _ = candidate_validation_plan((finished,), None)
    second = next(c for c in candidates if c.example_id == "binder_1")
    fields = [f"{VALIDATION_CANDIDATE_FIELD}={second.key}", f"{SCREENING_FIELD}=pmx"]
    assert "refreshed" in panel.run_results_action(SCREENING_ACTION, fields)
    assert "1 screening result." in panel.run_results_action(SCREENING_ACTION, fields)
    assert "inapplicable" in panel.done_page()


def test_mcp_select_all_and_empty_explicit_work_independently(finished: Path) -> None:
    prepared = mcp_api.call_tool("plan_screening", {"directory": str(finished)})
    assert len(prepared["methods"]) == 60
    assert mcp_api.call_tool("run_selected_screens", {"directory": str(finished), "methods": []})["results"] == []
    for args in ({}, {"methods": [], "selection": "recommended"}, {"selection": []}, {"methods": "sygma"}):
        with pytest.raises(ValueError):
            mcp_api.call_tool("run_selected_screens", {"directory": str(finished), **args})
