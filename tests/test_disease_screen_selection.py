"""The scientific picker accounts for installation, candidate identity, compute and evidence provenance."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert len(screen_selection.METHOD_IDS) == len(set(screen_selection.METHOD_IDS)) == 83
    assert "dimorphite_dl" in screen_selection.NATIVE_IDS
    assert "p2rank" in screen_selection.NATIVE_IDS
    assert "badapple2" in screen_selection.NATIVE_IDS
    assert "propka" in screen_selection.NATIVE_IDS
    assert "crest" in screen_selection.NATIVE_IDS
    assert "openantigens" in screen_selection.NATIVE_IDS
    assert "metfrag" in screen_selection.NATIVE_IDS
    assert "fastsolv" in screen_selection.NATIVE_IDS
    assert "jazzy" in screen_selection.NATIVE_IDS
    assert "espsim" in screen_selection.NATIVE_IDS
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


@pytest.mark.parametrize("key", ["badapple2", "fastsolv", "jazzy"])
def test_partial_chemical_batches_stay_selectable_and_explain_their_coverage(tmp_path, key):
    pytest.importorskip("rdkit")
    (tmp_path / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\nacetaminophen\tCC(=O)Nc1ccc(O)cc1\nsodium\t[Na+]\n", encoding="utf-8",
    )
    (tmp_path / "fastsolv.json").write_text(json.dumps({
        "version": 1, "conditions": [{"solvent": "ethanol", "temperature_k": 298.15}],
    }))
    availability = {key: True}
    (batch,) = screen_selection.plan(tmp_path, "ligand", available=availability)
    assert batch.applicable and "1 of 2" in batch.reason
    assert screen_selection.select([batch], "recommended") == ()
    assert screen_selection.select([batch], "all_installed") == (key,)
    assert screen_selection.select([batch], "explicit", [key]) == (key,)

    (single,) = screen_selection.plan(tmp_path, "ligand", example="acetaminophen", available=availability)
    assert single.recommended and "1 of 1" in single.reason
    (unsupported,) = screen_selection.plan(tmp_path, "ligand", example="sodium", available=availability)
    assert not unsupported.applicable and not unsupported.recommended


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
    assert "ligand_ames" in default
    assert "ligand_microsomal_clearance" in default
    assert "ligand_caco2" in available and "ligand_caco2" not in default
    assert "ligand_dili" in available and "ligand_dili" not in default
    assert "epa_httk" in available and "epa_httk" not in default
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
        assert plan["ligand_ames"].applicable == applicable
        assert plan["ligand_caco2"].applicable == applicable
        assert not plan["ligand_caco2"].recommended
        assert plan["ligand_microsomal_clearance"].applicable == applicable
        assert plan["pmx"].applicable == applicable
        (result,) = screen_panel.run(finished, candidate.key, "explicit", ["pmx"])
        if applicable:
            assert result["status"] == "recorded"
            assert result["measurement"]["attached_observations"] == 1
        else:
            assert result["status"] == "inapplicable"
            assert "measurement" not in result


def test_ames_picker_handle_forms_and_bounded_workloads(finished: Path) -> None:
    from rfproteina.disease import ligand_ames

    def choice(kind="ligand", example=None):
        result, = screen_selection.plan(finished, kind, example=example, available={ligand_ames.METHOD_ID: True})
        return result

    table = finished / "ligand_candidates.tsv"
    assert choice().recommended and not choice("protein").applicable
    table.write_text("candidate_id\tsmiles\nbinder_0\tCC[NH3+]\nbinder_1\tCCO.[Na+]\n")
    assert choice(example="binder_0").applicable and not choice(example="binder_0").recommended
    assert not choice(example="binder_1").applicable
    table.write_text("candidate_id\tsmiles\nnot a valid id\tCCO\n")
    assert not choice().applicable
    table.write_text("candidate_id\tsmiles\n" + "".join(f"n{i}\t{'C' * i}\n" for i in range(1, 66)))
    assert choice().applicable and not choice().recommended
    assert "multiple CPU batches" in choice().reason


def test_selected_ames_runs_for_its_candidate_and_preserves_the_legacy_verdict(finished, monkeypatch):
    from rfproteina.disease import ligand_ames, ligand_solubility, screen

    calls = []

    def predict(keys):
        calls.append(keys)
        return {"values": dict.fromkeys(keys, 0.0)}

    monkeypatch.setattr(ligand_ames, "_local_predictions", predict)
    monkeypatch.setattr(ligand_solubility, "screen_run", lambda *_args, **_kwargs: None)
    old = screen.screen_generation(finished, binder_kind="ligand", target_id="")
    assert "ligand_ames" not in old and not calls
    new = screen.screen_generation(finished, binder_kind="ligand", target_id="", screen_methods=["ligand_ames"])
    assert calls == [["CCO"]]
    assert new["ligand_ames"]["compounds"][0]["predicted_ames_probability"] == 0.0
    assert all(old[key] == new[key] for key in ("call", "hits", "states", "cuts"))
    candidates, _ = candidate_validation_plan((finished,), None)
    candidate = next(c for c in candidates if c.example_id == "binder_0")
    result, = screen_panel.run(finished, candidate.key, "explicit", ["ligand_ames"])
    assert result["status"] == "predicted"
    assert result["measurement"]["unique_predicted"] == 1
    assert result["measurement"]["inputs"][0]["candidate_id"] == "binder_0"


def test_microsomal_clearance_recommendations_respect_installation_form_and_scope(finished: Path) -> None:
    from rfproteina.disease.ligand_microsomal_clearance import METHOD_ID

    def choice(kind="ligand", example=None, installed=True):
        result, = screen_selection.plan(finished, kind, example=example, available={METHOD_ID: installed})
        return result

    assert choice().recommended and not choice("protein").applicable
    assert not choice(installed=False).recommended
    table = finished / "ligand_candidates.tsv"
    table.write_text("candidate_id\tsmiles\nbinder_0\tCC[NH3+]\nbinder_1\tCCO.[Na+]\n")
    charged = choice(example="binder_0")
    assert charged.applicable and not charged.recommended
    assert not choice(example="binder_1").applicable
    table.write_text("candidate_id\tsmiles\n" + "".join(f"n{i}\t{'C' * i}\n" for i in range(1, 66)))
    large = choice()
    assert large.applicable and not large.recommended and "multiple CPU batches" in large.reason
    table.unlink()
    (finished / "ligand_smiles.txt").write_text("CCO\n")
    assert choice().recommended and not choice(example="binder_0").applicable


def test_selected_microsomal_clearance_does_not_start_ames_or_change_verdict(finished, monkeypatch):
    from rfproteina.disease import ligand_ames, ligand_microsomal_clearance, ligand_solubility, screen

    method_id = ligand_microsomal_clearance.METHOD_ID
    calls = []

    def predict(keys):
        calls.append(keys)
        return {"values": {key: {"CCO": 0.0, "CCCO": 10.5}[key] for key in keys}}

    monkeypatch.setattr(ligand_microsomal_clearance, "_local_predictions", predict)
    monkeypatch.setattr(ligand_ames, "screen_run", lambda *_a, **_kw: pytest.fail("Ames unexpectedly selected"))
    monkeypatch.setattr(ligand_solubility, "screen_run", lambda *_a, **_kw: None)
    (finished / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\nbinder_0\tCCO\nbinder_1\tCCCO\n")
    old = screen.screen_generation(finished, binder_kind="ligand", target_id="")
    assert method_id not in old and not calls
    new = screen.screen_generation(finished, binder_kind="ligand", target_id="", screen_methods=[method_id])
    assert calls == [["CCO", "CCCO"]]
    assert "ligand_ames" not in new
    assert [row["predicted_clint_ul_min_mg"] for row in new[method_id]["compounds"]] == [0.0, 10.5]
    assert all(old[key] == new[key] for key in ("call", "hits", "states", "cuts"))
    assert "human liver microsomal intrinsic clearance predicted for 2/2" in screen._measured_line("", "", new)
    before = (finished / "screen.json").read_bytes()
    candidates, _ = candidate_validation_plan((finished,), None)
    candidate = next(c for c in candidates if c.example_id == "binder_0")
    payload = {"directory": str(finished), "candidate_id": candidate.key, "methods": [method_id]}
    result, = mcp_api.call_tool("run_selected_screens", payload)["results"]
    assert result["status"] == "predicted" and calls[-1] == ["CCO"]
    assert result["measurement"]["inputs"][0]["candidate_id"] == "binder_0"
    assert result["measurement"]["compounds"][0]["predicted_clint_ul_min_mg"] == 0.0
    assert (finished / "screen.json").read_bytes() == before


def test_dili_installation_checks_its_own_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    from rfproteina.disease import ligand_ames, ligand_dili

    checked = []
    installed = True
    resources_valid = True

    def verify():
        checked.append(ligand_dili.ENDPOINT)
        if not resources_valid:
            raise ligand_dili.DiliError("trained_model_unavailable")

    def unexpected():
        pytest.fail("DILI discovery must not verify a different endpoint")

    monkeypatch.setattr(screen_selection, "_modules", lambda *_names: installed)
    monkeypatch.setattr(ligand_dili, "verified_assets", verify)
    monkeypatch.setattr(ligand_ames, "verified_assets", unexpected)
    key = ligand_dili.METHOD_ID
    assert screen_selection.installations({key}) == {key: True}
    resources_valid = False
    assert screen_selection.installations({key}) == {key: False}
    installed = False
    assert screen_selection.installations({key}) == {key: False}
    assert checked == ["DILI", "DILI"]


def test_dili_is_optional_for_systemically_exposed_ligands(finished: Path) -> None:
    from rfproteina.disease.ligand_dili import METHOD_ID

    def choice(kind="ligand", example=None, installed=True):
        result, = screen_selection.plan(finished, kind, example=example, available={METHOD_ID: installed})
        return result

    eligible = choice()
    assert eligible.installed and eligible.applicable and not eligible.recommended
    assert "systemically exposed" in eligible.reason and "explicitly" in eligible.reason
    assert not choice("protein").applicable
    assert not choice(example="binder_1").applicable
    with pytest.raises(ValueError, match="Unavailable screening methods"):
        screen_selection.select((choice(installed=False),), "explicit", (METHOD_ID,))
    table = finished / "ligand_candidates.tsv"
    table.write_text("candidate_id\tsmiles\nbinder_0\tCC[NH3+]\nbinder_1\tCCO.[Na+]\n")
    charged = choice(example="binder_0")
    assert charged.applicable and not charged.recommended and "charged form" in charged.reason
    assert not choice(example="binder_1").applicable
    table.unlink()
    (finished / "ligand_smiles.txt").write_text("CCO\n")
    assert choice().applicable and not choice(example="binder_0").applicable


def test_selected_dili_is_candidate_scoped_and_does_not_start_other_admet_screens(finished, monkeypatch):
    from rfproteina.disease import ligand_ames, ligand_dili, ligand_microsomal_clearance, ligand_solubility, screen

    method_id = ligand_dili.METHOD_ID
    calls = []

    def predict(keys):
        calls.append(keys)
        scores = {"CCO": 0.0, "CCCO": 1.0}
        return {"values": {key: scores[key] for key in keys}}

    def unexpected(*_args, **_kwargs):
        pytest.fail("Another ADMET-AI ensemble unexpectedly selected")

    monkeypatch.setattr(ligand_dili, "_local_predictions", predict)
    monkeypatch.setattr(ligand_ames, "screen_run", unexpected)
    monkeypatch.setattr(ligand_microsomal_clearance, "screen_run", unexpected)
    monkeypatch.setattr(ligand_solubility, "screen_run", lambda *_a, **_kw: None)
    (finished / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\nbinder_0\tCCO\nbinder_1\tCCCO\n")
    old = screen.screen_generation(finished, binder_kind="ligand", target_id="")
    assert method_id not in old and not calls
    monkeypatch.setattr(ligand_solubility, "screen_run", unexpected)
    new = screen.screen_generation(finished, binder_kind="ligand", target_id="", screen_methods=[method_id])
    assert calls == [["CCO", "CCCO"]]
    assert all(old[key] == new[key] for key in ("call", "hits", "states", "cuts"))
    report = new[method_id]
    assert report["status"] == "predicted"
    assert [row["predicted_dili_probability"] for row in report["compounds"]] == [0.0, 1.0]
    line = screen._measured_line("", "", new)
    assert "drug-induced liver injury model scores for 2/2" in line and "no clinical risk estimate" in line

    before = (finished / "screen.json").read_bytes()
    candidates, _ = candidate_validation_plan((finished,), None)
    candidate = next(c for c in candidates if c.example_id == "binder_0")
    payload = {"directory": str(finished), "candidate_id": candidate.key, "methods": [method_id]}
    result, = mcp_api.call_tool("run_selected_screens", payload)["results"]
    assert result["status"] == "predicted" and calls[-1] == ["CCO"]
    measurement = result["measurement"]
    assert measurement["inputs"][0]["candidate_id"] == "binder_0"
    assert (measurement["endpoint"], measurement["unit"]) == ("DILI", "probability")
    assert measurement["compounds"][0]["predicted_dili_probability"] == 0.0
    assert (finished / "screen.json").read_bytes() == before
    monkeypatch.setattr(ligand_dili, "_local_predictions", lambda _keys: {"blocker": "provider_timeout"})
    timed_out, = mcp_api.call_tool("run_selected_screens", payload)["results"]
    assert timed_out["status"] == "untested" and timed_out["measurement"]["blocker"] == "provider_timeout"
    assert "untested (provider_timeout)" in screen._measured_line("", "", {method_id: timed_out["measurement"]})


_DELPHI_HEADER = "candidate_id\tvh\tvl\n"
# Public assay-independent sequence fixtures. The workflow tests do not run
# antibody numbering or use these structures as predictions of model quality.
_DELPHI_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYY"
    "CSRWGGDGFYAMDYWGQGTLVTVSS"
)
_DELPHI_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


def test_delphi_recommends_one_explicit_pair_and_reviews_mixed_or_large_scopes(finished, monkeypatch) -> None:
    from rfproteina.disease import delphi_inputs, delphi_psr

    method_id = delphi_psr.METHOD_ID
    manifest = finished / delphi_inputs.MANIFEST
    monkeypatch.setattr(delphi_psr, "screen_run", lambda *a, **k: pytest.fail("planning ran the DELPHI provider"))

    def choice(kind="protein", example=None, installed=True):
        selected, = screen_selection.plan(finished, kind, example=example, available={method_id: installed})
        return selected

    assert not choice().applicable  # A run with generated structures is not a paired nomination.
    good = f"binder_0\t{_DELPHI_VH}\t{_DELPHI_VL}\n"
    manifest.write_text(_DELPHI_HEADER + good)
    assert choice().applicable and choice().recommended
    assert choice(example="binder_0").recommended
    assert not choice(example="binder_1").applicable
    assert not choice("peptide").applicable and not choice("ligand").applicable
    assert choice(installed=False).applicable and not choice(installed=False).recommended
    assert "checkpoint" in choice(installed=False).reason
    assert screen_selection.select((choice(),), "all_installed") == (method_id,)

    def submit(page):
        assert 'value="delphi_psr" data-recommended checked' in page
        return {}

    assert OptionPanel(submit).choose_screens(finished, "protein") == ()

    manifest.write_text(_DELPHI_HEADER + good + f"binder_1\t{_DELPHI_VH}\t\n")
    assert choice().applicable and not choice().recommended and "1 of 2" in choice().reason
    assert choice(example="binder_0").recommended  # Other rows do not veto a selected pair.
    assert not choice(example="binder_1").applicable and "VHH" in choice(example="binder_1").reason
    manifest.write_text(_DELPHI_HEADER + good + f"binder_1\t{_DELPHI_VH}\t{_DELPHI_VL}\n")
    assert choice().applicable and not choice().recommended and "Select explicitly" in choice().reason


def test_delphi_explicit_cli_and_mcp_stay_in_scope_and_do_not_change_the_verdict(finished, monkeypatch) -> None:
    from rfproteina.disease import delphi_inputs, delphi_psr, ligand_dili, screen

    method_id = delphi_psr.METHOD_ID
    (finished / "generation.json").write_text('{"binder_kind":"protein"}')
    manifest = finished / delphi_inputs.MANIFEST
    manifest.write_text(
        _DELPHI_HEADER + "".join(f"{name}\t{_DELPHI_VH}\t{_DELPHI_VL}\n" for name in ("binder_0", "binder_1"))
    )
    calls = []

    def predict(pairs):
        calls.append([pair.candidate_id for pair in pairs])
        scores = {"binder_0": 0.0, "binder_1": 1.0}
        return {
            "values": {pair.candidate_id: scores[pair.candidate_id] for pair in pairs},
            "provider": {"release": delphi_psr.RELEASE, "checkpoint_sha256": delphi_psr.CHECKPOINT_SHA256,
                         "device": "cpu", "torch": "2.8.0", "anarcii": "2.0.8"},
        }

    monkeypatch.setattr(delphi_psr, "_local_predictions", predict)
    monkeypatch.setattr(ligand_dili, "screen_run", lambda *a, **k: pytest.fail("ran an unrelated provider"))
    monkeypatch.setattr(screen, "_sample_measurements", lambda *a, **k: ([], [], []))
    old = screen.screen_generation(finished, binder_kind="protein", target_id="")
    assert method_id not in old and not calls
    new = screen.screen_generation(finished, binder_kind="protein", target_id="", screen_methods=[method_id])
    assert calls == [["binder_0", "binder_1"]]
    assert all(old[key] == new[key] for key in ("call", "hits", "states", "cuts"))
    assert [row[delphi_psr.VALUE_FIELD] for row in new[method_id]["pairs"]] == [0.0, 1.0]
    assert "DELPHI PSR pass-class scores for 2/2" in screen._measured_line("", "", new)
    attested, = (entry for entry in new["inputs"] if entry["file"] == delphi_inputs.MANIFEST)
    raw = manifest.read_bytes()
    assert attested == {"file": delphi_inputs.MANIFEST, "bytes": len(raw), "digest": hashlib.sha256(raw).hexdigest()[:16]}

    before = (finished / "screen.json").read_bytes()
    candidates, _ = candidate_validation_plan((finished,), None)
    candidate = next(candidate for candidate in candidates if candidate.example_id == "binder_0")
    payload = {"directory": str(finished), "candidate_id": candidate.key, "methods": [method_id]}
    result, = mcp_api.call_tool("run_selected_screens", payload)["results"]
    assert result["status"] == "predicted" and calls[-1] == ["binder_0"]
    measurement = result["measurement"]
    assert measurement["scope"] == "candidate"
    assert measurement["pairs"] == [
        {"candidate_id": "binder_0", "status": "predicted", delphi_psr.VALUE_FIELD: 0.0}
    ]
    assert _DELPHI_VH not in json.dumps(result) and _DELPHI_VL not in json.dumps(result)
    assert (finished / "screen.json").read_bytes() == before


@pytest.mark.parametrize("as_link", [True, False])
def test_delphi_provenance_does_not_hash_a_link_or_over_budget_input(tmp_path, monkeypatch, as_link) -> None:
    from rfproteina.disease import delphi_inputs, screen

    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "outside"
    outside.write_text(_DELPHI_HEADER)
    manifest = run / delphi_inputs.MANIFEST
    monkeypatch.setattr(screen, "_file_digest", lambda *_args: pytest.fail("used an unrestricted input digest"))
    if as_link:
        try:
            manifest.symlink_to(outside)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        expected = {"file": manifest.name, "digest": "delphi_pairs_unreadable", "link": True}
    else:
        size = delphi_inputs.MAX_BYTES + 1
        manifest.write_bytes(b"A" * size)
        expected = {"file": manifest.name, "digest": "delphi_pairs_over_limit", "bytes": size}
    attested, = screen._supplied_inputs(run)
    assert attested == expected


def _rbc_observation(name="binder_0", *, species="human", qc="accepted") -> str:
    return f"{name}\tobs-1\tform-1\treport-1\t{species}\twashed_rbc\tcontext-1\t1\tuM\t1\t0\t{qc}\n"


def test_peptide_rbc_picker_checks_species_qc_modality_and_candidate_identity(finished, monkeypatch):
    from rfproteina.disease import peptide_rbc

    method = peptide_rbc.METHOD_ID
    manifest = finished / peptide_rbc.MANIFEST
    header = "\t".join(peptide_rbc.HEADER) + "\n"
    monkeypatch.setattr(peptide_rbc, "screen_run", lambda *a, **k: pytest.fail("planning executed the reader"))

    def choice(example=None, kind="peptide"):
        item, = screen_selection.plan(finished, kind, example=example, available={method: True})
        return item

    assert not choice().applicable
    manifest.write_text(header + _rbc_observation(), encoding="utf-8")
    assert choice().recommended and choice("binder_0").recommended
    assert not choice("binder_1").applicable and not choice(kind="protein").applicable
    assert screen_selection.select((choice(),), "all_installed") == (method,)

    def render(page):
        assert f'value="{method}" data-recommended checked' in page
        return {}

    assert OptionPanel(render).choose_screens(finished, "peptide") == ()
    manifest.write_text(header + _rbc_observation(species="rabbit"), encoding="utf-8")
    assert choice().applicable and not choice().recommended
    manifest.write_text(header + _rbc_observation() + _rbc_observation("binder_1", qc="failed"), encoding="utf-8")
    assert choice().applicable and not choice().recommended and "1 of 2" in choice().reason
    assert choice("binder_1").applicable and not choice("binder_1").recommended
    manifest.write_text(header + _rbc_observation("unknown-candidate"), encoding="utf-8")
    assert not choice().applicable and not choice("unknown-candidate").applicable


def test_peptide_rbc_refuses_a_counterfeit_candidate_alias_and_ambiguous_bare_id(finished):
    import gzip

    from rfproteina.disease import peptide_rbc, screen_criteria

    samples = finished / "inference_samples"
    alias = samples / "binder_2_generated.cif"
    try:
        alias.symlink_to(samples / "binder_0_generated.cif")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    # Do not fall back to another format when the preferred path is an alias.
    (samples / "binder_2_generated.cif.gz").write_bytes(gzip.compress(b"data_secondary_copy\n"))
    duplicate = samples / "parallel"
    duplicate.mkdir()
    (duplicate / "binder_0_generated.cif").write_text("data_parallel\n", encoding="utf-8")
    ids = screen_criteria.peptide_rbc_known_candidates(finished)
    assert "binder_1" in ids and "binder_2" not in ids and "binder_0" not in ids
    path = finished / peptide_rbc.MANIFEST
    path.write_text("\t".join(peptide_rbc.HEADER) + "\n" + _rbc_observation(), encoding="utf-8")
    item, = screen_selection.plan(finished, "peptide", available={peptide_rbc.METHOD_ID: True})
    assert not item.applicable and not item.recommended
    path.write_text("\t".join(peptide_rbc.HEADER) + "\n" + _rbc_observation("parallel_binder_0"), encoding="utf-8")
    item, = screen_selection.plan(finished, "peptide", available={peptide_rbc.METHOD_ID: True})
    assert item.applicable and item.recommended


def test_peptide_rbc_keeps_cli_and_mcp_supplemental_verdict_scope(finished, monkeypatch):
    from rfproteina.disease import peptide_rbc, screen

    method = peptide_rbc.METHOD_ID
    (finished / "generation.json").write_text('{"binder_kind":"peptide"}', encoding="utf-8")
    manifest = finished / peptide_rbc.MANIFEST
    manifest.write_text(
        "\t".join(peptide_rbc.HEADER) + "\n" + _rbc_observation() + _rbc_observation("binder_1", qc="failed"),
        encoding="utf-8",
    )
    original = peptide_rbc.screen_run
    calls = []

    def measured(path, *, candidate=None, known_candidates=None):
        calls.append((candidate, known_candidates))
        return original(path, candidate=candidate, known_candidates=known_candidates)

    monkeypatch.setattr(peptide_rbc, "screen_run", measured)
    monkeypatch.setattr(screen, "_sample_measurements", lambda *a, **k: ([], [], []))
    baseline = screen.screen_generation(finished, binder_kind="peptide", target_id="")
    assert method not in baseline and not calls
    verdict = screen.screen_generation(finished, binder_kind="peptide", target_id="", screen_methods=[method])
    known = frozenset({"binder_0", "binder_1"})
    assert calls == [(None, known)]
    assert all(baseline[key] == verdict[key] for key in ("call", "hits", "states", "cuts"))
    record = verdict[method]
    assert record["status"] == "partial" and record["reported"] == record["human_reported"] == 1
    assert record["input_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert "1/2 observations reported, 1 human" in screen._measured_line("", "", verdict)

    before = (finished / "screen.json").read_bytes()
    candidates, _ = candidate_validation_plan((finished,), None)
    keys = {candidate.example_id: candidate.key for candidate in candidates}

    def run(name):
        args = {"directory": str(finished), "candidate_id": keys[name], "methods": [method]}
        result, = mcp_api.call_tool("run_selected_screens", args)["results"]
        return result

    supported = run("binder_0")
    assert supported["status"] == "reported" and calls[-1] == ("binder_0", known)
    assert supported["measurement"]["scope"] == "candidate"
    record, = supported["measurement"]["observations"]
    assert record["candidate_id"] == "binder_0" and record["hemolysis_percent"] == 0.0
    refused = run("binder_1")
    assert refused["status"] == "untested" and calls[-1] == ("binder_1", known)
    assert refused["measurement"]["observations"][0]["blocker"] == "peptide_rbc_controls_failed"
    assert (finished / "screen.json").read_bytes() == before


@pytest.mark.parametrize("as_link", [True, False])
def test_peptide_rbc_attestation_refuses_links_and_over_budget_inputs(tmp_path, monkeypatch, as_link):
    from rfproteina.disease import peptide_rbc, screen

    run = tmp_path / "run"
    run.mkdir()
    manifest = run / peptide_rbc.MANIFEST
    monkeypatch.setattr(screen, "_file_digest", lambda *_: pytest.fail("used the generic file digest"))
    if as_link:
        outside = tmp_path / "outside"
        outside.write_text("test\n", encoding="utf-8")
        try:
            manifest.symlink_to(outside)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        expected = {"file": manifest.name, "link": True, "digest": "peptide_rbc_unreadable"}
    else:
        size = peptide_rbc.MAX_BYTES + 1
        manifest.write_bytes(b"A" * size)
        expected = {"file": manifest.name, "bytes": size, "digest": "peptide_rbc_over_limit"}
    assert screen._supplied_inputs(run) == [expected]


def test_caco2_is_optional_for_installed_ligand_scopes(finished: Path) -> None:
    from rfproteina.disease.ligand_caco2 import METHOD_ID

    def choice(kind="ligand", example=None, installed=True):
        result, = screen_selection.plan(finished, kind, example=example, available={METHOD_ID: installed})
        return result

    eligible = choice()
    assert eligible.installed and eligible.applicable and not eligible.recommended
    assert "oral" in eligible.reason and "explicitly" in eligible.reason
    assert not choice("protein").applicable
    assert not choice(example="binder_1").applicable
    assert not choice(installed=False).recommended
    with pytest.raises(ValueError, match="Unavailable screening methods"):
        screen_selection.select((choice(installed=False),), "explicit", (METHOD_ID,))
    table = finished / "ligand_candidates.tsv"
    table.write_text("candidate_id\tsmiles\nbinder_0\tCC[NH3+]\nbinder_1\tCCO.[Na+]\n")
    charged = choice(example="binder_0")
    assert charged.applicable and not charged.recommended and "charged form" in charged.reason
    assert not choice(example="binder_1").applicable
    table.write_text("candidate_id\tsmiles\n" + "".join(f"n{i}\t{'C' * i}\n" for i in range(1, 66)))
    large = choice()
    assert large.applicable and not large.recommended and "multiple CPU batches" in large.reason
    table.unlink()
    (finished / "ligand_smiles.txt").write_text("CCO\n")
    assert choice().applicable and not choice(example="binder_0").applicable


def test_molprobity_picker_recommends_one_local_model_without_evaluating(finished, monkeypatch):
    from rfproteina.disease import molprobity_geometry, molprobity_inputs

    method_id = molprobity_geometry.METHOD_ID
    monkeypatch.setattr(molprobity_geometry, "screen_run", lambda *a, **k: pytest.fail("picker ran MolProbity"))

    def choice(kind="protein", example=None, installed=True):
        selected, = screen_selection.plan(finished, kind, example=example, available={method_id: installed})
        return selected

    multiple = choice()
    assert multiple.installed and multiple.applicable and not multiple.recommended
    assert "2 of 2" in multiple.reason and "whole model" in multiple.reason
    single = choice(example="binder_0")
    assert single.applicable and single.recommended and "partner chains" in single.reason
    assert choice(kind="peptide", example="binder_1").recommended
    assert not choice(kind="ligand", example="binder_0").applicable
    assert not choice(kind="nucleic", example="binder_0").applicable
    assert not choice(example="unavailable-example").applicable
    assert not choice(example="").applicable
    assert not choice(example="binder_0", installed=False).recommended
    assert "local MolProbity CCTBX" in choice(example="binder_0", installed=False).reason

    for index in range(2, molprobity_inputs.MAX_MODELS + 1):
        (finished / "inference_samples" / f"binder_{index}_generated.cif").write_text("data_over_limit\n")
    assert not choice().applicable and "molprobity_too_many_models" in choice().reason
    assert choice(example="binder_0").recommended


def test_molprobity_cli_is_advisory_and_mcp_runs_only_the_selected_model(finished, monkeypatch):
    from rfproteina.disease import corpus_provenance, molprobity_geometry, screen

    method_id = molprobity_geometry.METHOD_ID
    (finished / "generation.json").write_text('{"binder_kind":"protein"}', encoding="utf-8")
    called = []

    def fake_run(directory, *, candidate=None):
        assert directory == finished
        called.append(candidate)
        model_ids = (candidate,) if candidate is not None else ("binder_0", "binder_1")
        models = [{
            "candidate_id": name,
            "status": "reported",
            "input_sha256": "7" * 64,
            "metrics": {
                "ramachandran": {"status": "reported", "evaluated": 10, "outliers": 1, "allowed": 2,
                                "favored": 7, "outlier_percent": 10.0},
                "rotamers": {"status": "reported", "evaluated": 20, "outliers": 0, "allowed": 0,
                             "favored": 20, "outlier_percent": 0.0},
                "clashscore": {"status": "reported", "atoms": 800, "serious_clashes": 1,
                               "clashes_per_1000_atoms": 1.25},
            },
        } for name in model_ids]
        return {
            "method": molprobity_geometry.METHOD, "model_scope": "whole_model", "status": "reported",
            "scope": "run" if candidate is None else "candidate", "attempted": len(models),
            "reported": len(models), "complete": len(models), "samples": models,
            "provider": {"cctbx_base": "2025.11", "device": "cpu", "clash_preparation": "reduce2_no_flips",
                         "clash_method": "probe2"},
            "corpus_provenance": corpus_provenance.declare(molprobity_geometry.METHOD),
        }

    monkeypatch.setattr(molprobity_geometry, "screen_run", fake_run)
    monkeypatch.setattr(screen, "_sample_measurements", lambda *a, **k: ([], [], []))
    baseline = screen.screen_generation(finished, binder_kind="protein", target_id="")
    assert method_id not in baseline and not called
    new = screen.screen_generation(finished, binder_kind="protein", target_id="", screen_methods=[method_id])
    assert called == [None]
    assert all(baseline[key] == new[key] for key in ("call", "hits", "states", "cuts"))
    assert new[method_id]["model_scope"] == "whole_model" and new[method_id]["complete"] == 2
    assert "MolProbity whole-model geometry has 2/2 models" in screen._measured_line("", "", new)

    before = (finished / "screen.json").read_bytes()
    candidates, _ = candidate_validation_plan((finished,), None)
    second = next(candidate for candidate in candidates if candidate.example_id == "binder_1")
    args = {"directory": str(finished), "candidate_id": second.key, "methods": [method_id]}
    result, = mcp_api.call_tool("run_selected_screens", args)["results"]
    assert called == [None, "binder_1"] and result["status"] == "reported"
    measurement = result["measurement"]
    assert measurement["scope"] == "candidate" and measurement["model_scope"] == "whole_model"
    report, = measurement["samples"]
    assert report["candidate_id"] == "binder_1"
    assert report["metrics"]["clashscore"]["clashes_per_1000_atoms"] == 1.25
    assert (finished / "screen.json").read_bytes() == before


def test_httk_picker_checks_installation_and_scope_without_calculating(finished, monkeypatch):
    from rfproteina.disease import httk

    method = httk.METHOD_ID
    prepared = []

    def prepare(directory, candidate=None):
        assert directory == finished
        prepared.append(candidate)
        members = [] if candidate == "binder_1" else [SimpleNamespace(blocker=None)]
        return SimpleNamespace(members=members, blocker=None)

    monkeypatch.setattr(httk, "prepare", prepare)
    monkeypatch.setattr(httk, "measure", lambda *a, **k: pytest.fail("the picker ran EPA httk"))

    def choice(*, kind="ligand", example=None, installed=True):
        item, = screen_selection.plan(finished, kind, example=example, available={method: installed})
        return item

    assert not choice().applicable and not prepared
    (finished / httk.MANIFEST).write_text("{}", encoding="utf-8")
    ready = choice()
    assert ready.installed and ready.applicable and not ready.recommended
    assert "Select explicitly" in ready.reason and "per unit intake" in ready.reason
    assert screen_selection.select((ready,), "recommended") == ()
    assert screen_selection.select((ready,), "explicit", (method,)) == (method,)
    assert not choice(example="binder_1").applicable
    assert not choice(kind="protein").applicable
    assert not choice(installed=False).installed
    assert prepared == [None, "binder_1", None]


def test_httk_installation_probe_does_not_trigger_other_providers(monkeypatch):
    from rfproteina.disease import httk, pksim_runtime

    monkeypatch.setattr(pksim_runtime, "installed", lambda: pytest.fail("probed unrelated PK-Sim"))
    monkeypatch.setattr(httk, "asset_available", lambda: True)
    assert screen_selection.installations({httk.METHOD_ID}) == {httk.METHOD_ID: True}
    monkeypatch.setattr(httk, "asset_available", lambda: False)
    assert screen_selection.installations({httk.METHOD_ID}) == {httk.METHOD_ID: False}


@pytest.mark.parametrize("as_link", [False, True])
def test_httk_input_provenance_uses_the_bounded_manifest_reader(tmp_path, monkeypatch, as_link):
    from rfproteina.disease import httk, screen

    run = tmp_path / "run"
    run.mkdir()
    manifest = run / httk.MANIFEST
    if as_link:
        target = tmp_path / "external.json"
        target.write_text("{}", encoding="utf-8")
        try:
            manifest.symlink_to(target)
        except OSError:
            pytest.skip("symlinks are unavailable in this environment")
        metadata = {"link": True, "digest": "httk_manifest_unreadable"}
    else:
        size = httk.MAX_MANIFEST_BYTES + 1
        manifest.write_bytes(b" " * size)
        metadata = {"bytes": size, "digest": "httk_manifest_over_limit"}
    monkeypatch.setattr(screen, "_file_digest", lambda _path: pytest.fail("used the generic unbounded reader"))
    assert screen._supplied_inputs(run) == [{"file": httk.MANIFEST, **metadata}]


def test_httk_is_advisory_prepared_evidence_and_mcp_projects_candidate_scope(finished, monkeypatch):
    from rfproteina.disease import httk, screen

    method = httk.METHOD_ID
    (finished / httk.MANIFEST).write_text("{}", encoding="utf-8")
    scoped = []
    projected = []

    def prepare(directory, candidate=None):
        assert directory == finished
        members = [SimpleNamespace(blocker=None)] if candidate in (None, "binder_0") else []
        return SimpleNamespace(members=members, blocker=None)

    def measure(directory, candidate=None):
        assert directory == finished
        scoped.append(candidate)
        return {"method": method, "status": "reported", "sample": candidate}

    def project(report):
        projected.append(report["sample"])
        return {"method": report["method"], "status": "untested", "blocker": "input_unavailable"}

    monkeypatch.setattr(httk, "prepare", prepare)
    monkeypatch.setattr(httk, "measure", measure)
    monkeypatch.setattr(mcp_api, "_httk_view", project, raising=False)
    monkeypatch.setattr(screen, "_sample_measurements", lambda *a, **k: ([], [], []))
    baseline = screen.screen_generation(finished, binder_kind="ligand", target_id="", screen_selection="explicit")
    assert not scoped and method not in baseline["method_selection"]["prepared_measurements"]
    new = screen.screen_generation(finished, binder_kind="ligand", target_id="", screen_methods=[method])
    assert all(baseline[key] == new[key] for key in ("call", "hits", "states", "cuts"))
    assert method not in new and new["method_selection"]["prepared_measurements"][method]["sample"] is None

    candidates, _ = candidate_validation_plan((finished,), None)
    previous = (finished / "screen.json").read_bytes()
    for candidate in candidates:
        args = {"directory": str(finished), "candidate_id": candidate.key, "methods": [method]}
        result, = mcp_api.call_tool("run_selected_screens", args)["results"]
        if candidate.example_id == "binder_0":
            assert result == {"id": method, "status": "untested", "measurement": {
                "method": method, "status": "untested", "blocker": "input_unavailable",
            }}
        else:
            assert result["status"] == "inapplicable"
    assert scoped == [None, "binder_0"] and projected == ["binder_0"]
    assert (finished / "screen.json").read_bytes() == previous


def test_selected_caco2_uses_both_native_heads_without_changing_verdict(finished, monkeypatch):
    from rfproteina.disease import ligand_ames, ligand_caco2, ligand_microsomal_clearance, ligand_solubility, screen

    method_id = ligand_caco2.METHOD_ID
    calls = []

    def predict(keys):
        calls.append(keys)
        scores = {"CCO": [-0.2, 1.2], "CCCO": [0.0, "invalid_provider_result"]}
        return {"values": {key: scores[key] for key in keys}}

    monkeypatch.setattr(ligand_caco2, "_local_predictions", predict)
    monkeypatch.setattr(ligand_ames, "screen_run", lambda *_a, **_kw: pytest.fail("Ames unexpectedly selected"))
    monkeypatch.setattr(
        ligand_microsomal_clearance, "screen_run", lambda *_a, **_kw: pytest.fail("clearance unexpectedly selected")
    )
    monkeypatch.setattr(ligand_solubility, "screen_run", lambda *_a, **_kw: None)
    (finished / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\nbinder_0\tCCO\nbinder_1\tCCCO\n")
    old = screen.screen_generation(finished, binder_kind="ligand", target_id="")
    assert method_id not in old and not calls
    new = screen.screen_generation(finished, binder_kind="ligand", target_id="", screen_methods=[method_id])
    assert calls == [["CCO", "CCCO"]]
    assert all(old[key] == new[key] for key in ("call", "hits", "states", "cuts"))
    report = new[method_id]
    assert report["status"] == "partial"
    assert (report["unique_predicted"], report["unique_complete"]) == (2, 1)
    second = report["compounds"][1]["heads"]
    assert second["efflux_ratio"]["model_score"] == 0.0
    assert second["apparent_permeability"]["status"] == "untested"
    assert "model_score" not in second["apparent_permeability"]
    assert "both heads for 1/2" in screen._measured_line("", "", new)

    before = (finished / "screen.json").read_bytes()
    candidates, _ = candidate_validation_plan((finished,), None)
    candidate = next(c for c in candidates if c.example_id == "binder_0")
    payload = {"directory": str(finished), "candidate_id": candidate.key, "methods": [method_id]}
    result, = mcp_api.call_tool("run_selected_screens", payload)["results"]
    assert result["status"] == "predicted" and calls[-1] == ["CCO"]
    measurement = result["measurement"]
    assert measurement["inputs"][0]["candidate_id"] == "binder_0"
    assert measurement["scale"] == "native_model_score" and "unit" not in measurement
    heads = measurement["compounds"][0]["heads"]
    assert heads["efflux_ratio"]["model_score"] == -0.2
    assert heads["apparent_permeability"]["model_score"] == 1.2
    assert (finished / "screen.json").read_bytes() == before


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
        assert 'value="ligand_microsomal_clearance" data-recommended checked' in page
        assert 'value="ligand_caco2"' in page
        assert 'value="ligand_caco2" data-recommended checked' not in page
        assert 'value="ligand_dili"' in page
        assert 'value="ligand_dili" data-recommended checked' not in page
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
    total = len(screen_selection.METHOD_IDS)
    assert f"of {total - 1} installed screens selected" in page and f"{total} listed" in page
    assert 'value="dimorphite_dl"' in page
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
    assert tuple(row["id"] for row in prepared["methods"]) == screen_selection.METHOD_IDS
    assert mcp_api.call_tool("run_selected_screens", {"directory": str(finished), "methods": []})["results"] == []
    for args in ({}, {"methods": [], "selection": "recommended"}, {"selection": []}, {"methods": "sygma"}):
        with pytest.raises(ValueError):
            mcp_api.call_tool("run_selected_screens", {"directory": str(finished), **args})
