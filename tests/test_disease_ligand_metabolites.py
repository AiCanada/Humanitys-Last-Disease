"""Regression contracts for phase-ordered, structure-resolved SyGMa screening."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest
from rdkit.Chem import rdChemReactions

from rfproteina.disease import ligand_metabolites as screen
from rfproteina.disease import sygma_engine as engine


def _rule(phase: str, line: int, weight: str, smarts: str) -> engine.Rule:
    return engine.Rule(phase, line, f"rule{line}", Decimal(weight), rdChemReactions.ReactionFromSmarts(smarts))


def _toy() -> dict:
    # Isotopic labels make legal and phase-reversed paths distinguishable using
    # real RDKit reactions, independently of the scientific rules installation.
    return {
        "phase1": [
            _rule("phase1", 1, "0.10", "[12CH4:1]>>[13CH4:1]"),
            _rule("phase1", 2, "0.90", "[14CH4:1]>>[13CH4:1]"),
        ],
        "phase2": [
            _rule("phase2", 1, "0.80", "[12CH4:1]>>[14CH4:1]"),
            _rule("phase2", 2, "0.70", "[13CH4:1]>>[15CH4:1]"),
        ],
    }


def test_best_path_obeys_phase_order_and_cycle_depth() -> None:
    rules = _toy()
    result = engine._enumerate("[12CH4]", 1, 1, rules)
    values = {result["structures"][row["product_id"]]: row for row in result["products"]}
    assert {s: r["path_score"] for s, r in values.items()} == {"[14CH4]": 0.8, "[13CH4]": 0.1, "[15CH4]": 0.07}
    assert values["[13CH4]"]["molecular_formula"] == "[13C]H4"
    assert [v["phase"] for v in values["[15CH4]"]["pathway"]] == ["phase1", "phase2"]
    assert engine.structure_id("[12CH4]") not in {r["product_id"] for r in result["products"]}
    truncated = engine._enumerate("[12CH4]", 0, 1, rules)
    assert [truncated["structures"][x["product_id"]] for x in truncated["products"]] == ["[14CH4]"]


def test_strict_fragment_boundary_uses_original_parent_atoms() -> None:
    p1 = _rule("phase1", 1, "0.5", "[C:1]-[C:2]>>([C:1].[C:2])")
    # Four newly attached atoms cannot rescue a fragment that retained just one.
    p2 = _rule("phase2", 1, "0.5", "[CH4:1]>>[C:1]OCCO")
    result = engine._enumerate("C" * 20, 1, 1, {"phase1": [p1], "phase2": [p2]})
    rows = {result["structures"][x["product_id"]]: x for x in result["products"]}
    assert set(rows) == {"C" * i for i in range(4, 20)}
    assert rows["CCCC"]["parent_heavy_atoms_retained"] == 4 and result["size_filtered_structures"] == 4
    assert {row["rank"] for row in rows.values()} == {1}


def test_equal_score_chooses_a_reproducible_pathway() -> None:
    first = _rule("phase1", 10, "0.5", "[12CH4:1]>>[13CH4:1]")
    second = _rule("phase1", 22, "0.5", "[12CH4:1]>>[13CH4:1]")
    tie = _rule("phase1", 23, "0.5", "[12CH4:1]>>[14CH4:1]")
    low = _rule("phase1", 24, "0.2", "[12CH4:1]>>[15CH4:1]")
    forward = engine._enumerate("[12CH4]", 1, 0, {"phase1": [first, second, tie, low], "phase2": []})
    reverse = engine._enumerate("[12CH4]", 1, 0, {"phase1": [low, tie, second, first], "phase2": []})
    assert forward["products"] == reverse["products"]
    a, b, c = forward["products"]
    assert [a["rank"], b["rank"], c["rank"]] == [1, 1, 3] and a["pathway"][0]["rule_line"] == 10


def test_bounded_enumeration_never_publishes_a_truncated_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "MAX_STATES", 1)
    with pytest.raises(engine.ScreenError, match="sygma_state_budget"):
        engine._enumerate("[12CH4]", 1, 1, _toy())
    r = _rule("phase1", 1, "0.5", "[C:1]-[C:2]>>([C:1].[C:2])")
    monkeypatch.setattr(engine, "MAX_MATCHES", 1)
    with pytest.raises(engine.ScreenError, match="sygma_rule_match_limit"):
        engine._enumerate("CCCC", 1, 0, {"phase1": [r], "phase2": []})


@pytest.mark.parametrize(
    "manifest",
    [
        '{"phase1_cycles":true}',
        '{"input":"sdf","input":"smiles"}',
        "[]",
        '{"other":0}',
        '{"phase1_cycles":0,"phase2_cycles":0}',
        '{"phase2_cycles":-1}',
        '{"phase1_cycles":5}',
    ],
)
def test_bad_manifest_cannot_fall_back(tmp_path: Path, manifest: str) -> None:
    (tmp_path / "ligand_smiles.txt").write_text("C")
    (tmp_path / screen.MANIFEST).write_text(manifest)
    q = screen.screen_run(tmp_path, kind="ligand")
    assert q["blocker"] == "invalid_sygma_manifest" and q["scenario"]["phase1_cycles"] is None


def test_nomination_coverage_and_structure_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(s: str, p1: int, p2: int) -> dict:
        calls.append(s)
        return {"compound_id": engine.structure_id(s), "status": "no_products", "products": []}

    monkeypatch.setattr(screen, "_run", run)
    assert screen.screen_run(tmp_path, kind="ligand") is None
    (tmp_path / "ligand_smiles.txt").write_text("C")
    (tmp_path / screen.MANIFEST).write_text("{}")
    assert screen.screen_run(tmp_path, kind="protein")["blocker"] == "ligand_binder_kind_required"
    (tmp_path / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\na\tC[C@H](O)F\nb\tF[C@@H](O)C\nmirror\tC[C@@H](O)F\n"
        "salt\tCC.[Cl-]\nradical\t[CH3]\nbad\t*\ndup\tC\ndup\tCO\nshort\n"
    )
    q = screen.screen_run(tmp_path, kind="ligand")
    assert q["status"] == "partial" and len(calls) == 2 and q["parents_without_products"] == 2 and q["attempted"] == 9
    a, b, c, *bad = q["inputs"]
    assert a["compound_id"] == b["compound_id"] != c["compound_id"]
    assert [v["blocker"] for v in bad] == [
        "disconnected_structure_requires_nominated_parent",
        "open_shell_parent_unsupported",
        "query_or_dummy_atoms_unsupported",
        "duplicate_candidate_id",
        "duplicate_candidate_id",
        "invalid_candidate_table_width",
    ]


def test_provider_identity_and_coverage_validation() -> None:
    key = engine.structure_id("[12CH4]")
    good = {
        "compound_id": key,
        "provider": {
            "rule_sha256": engine.RULE_HASHES,
            "rule_release": "1.1.0",
            "device": "cpu",
            "rules_per_phase": {"phase1": 148, "phase2": 27},
        },
        **engine._enumerate("[12CH4]", 1, 1, _toy()),
    }
    assert screen._valid_report(good, key, 1, 1)
    for field, value in (
        ("path_score", float("nan")),
        ("rank", 10),
        ("parent_heavy_atoms_retained", 0),
        ("exact_mass_da", -1),
        ("product_id", key),
        ("inchikey", ""),
        ("unspecified_stereoelements", -1),
    ):
        bad = deepcopy(good)
        bad["products"][0][field] = value
        assert not screen._valid_report(bad, key, 1, 1)
    bad = deepcopy(good)
    bad["products"][-1]["pathway"].reverse()
    assert not screen._valid_report(bad, key, 1, 1)
    assert not screen._valid_report({**good, "structures": {}}, key, 1, 1)


def test_missing_and_corrupt_archive_are_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bad.tar.gz"
    monkeypatch.setenv("SYGMA_RULE_ARCHIVE", str(path))
    with pytest.raises(engine.ScreenError, match="sygma_rule_archive_unavailable"):
        engine.load_rules()
    path.write_bytes(b"truncated")
    with pytest.raises(engine.ScreenError, match="sygma_archive_digest_mismatch"):
        engine.load_rules()


def test_picker_budgets_one_identified_parent_without_enumerating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from rfproteina.disease.screen_selection import plan

    monkeypatch.setattr(screen, "_run", lambda *_: pytest.fail("The picker ran the reaction rules"))
    available = {"sygma": True}
    table = tmp_path / "ligand_candidates.tsv"
    table.write_text("candidate_id\tsmiles\na\tC[C@H](O)F\nb\tF[C@@H](O)C\n")

    def states(example=None):
        choice, = plan(tmp_path, "ligand", example=example, available=available)
        return choice.applicable, choice.recommended

    assert states() == (True, False)
    (tmp_path / screen.MANIFEST).write_text("{}")
    assert states() == (True, True)
    table.write_text(table.read_text() + "c\tCC(O)F\nunusable\tC*\n")
    assert states() == (True, False)
    assert states("a") == (True, True)
    assert states("c") == (True, False)
    assert states("unusable") == (False, False)
    (tmp_path / screen.MANIFEST).write_text('{"phase1_cycles":2, "phase2_cycles":0}')
    assert states("a") == (True, False)
    (tmp_path / screen.MANIFEST).write_text('{"phase1_cycles":0, "phase2_cycles":0}')
    assert states("a") == (False, False)


def test_legacy_run_molecule_cannot_be_screened_as_a_named_design(tmp_path: Path):
    (tmp_path / "ligand_smiles.txt").write_text("CCO ethanol\n")
    result = screen.screen_run(tmp_path, kind="ligand", use_defaults=True, candidate="run_ligand")
    assert result["blocker"] == "candidate_requires_named_nomination"


def test_reference_stereochemistry_and_acetaminophen_conjugates() -> None:
    try:
        result = engine.enumerate_products("CC(=O)Nc1ccc(O)cc1", 1, 1)
    except engine.ScreenError as error:
        if str(error) == "sygma_rules_unavailable":
            pytest.skip("Optional official SyGMa rules are not installed.")
        raise
    rows = {r["inchikey"]: r for r in result["products"]}
    assert (len(rows), rows["IGTYILLPRJOVFY-UHFFFAOYSA-N"]["rank"]) == (40, 3)
    gluc = rows["IPROLSVTVHAQLE-UHFFFAOYSA-N"]
    assert gluc["rank"] == 1 and gluc["unspecified_stereoelements"] == 5
    assert "IPROLSVTVHAQLE-BYNIDDHOSA-N" not in rows


def test_service_preserves_verified_chemical_keys_and_withholds_structures() -> None:
    from rfproteina.disease.mcp_api import _verdict

    smiles = "CC(=O)Nc1ccc(OS(=O)(=O)O)cc1"
    key = "IGTYILLPRJOVFY-UHFFFAOYSA-N"
    identifier = engine.structure_id(smiles)
    product = {"inchikey": key, "product_id": identifier}
    unverified = {**product, "product_id": engine.structure_id("C")}
    forged = {**product, "inchikey": "ACDEFGHIKLMNPQ-UHFFFAOYSA-N"}
    block = {"structures": {identifier: smiles}, "parents": [{"products": [product, unverified, forged]}]}
    verdict = _verdict({"sygma": block}, include_record=True)
    assert verdict["record"]["sygma"] == verdict["sygma"]
    assert "structures" not in verdict["sygma"]
    returned = verdict["sygma"]["parents"][0]["products"]
    assert returned[0]["inchikey"] == key
    assert all("[sequence omitted]" in row["inchikey"] for row in returned[1:])
