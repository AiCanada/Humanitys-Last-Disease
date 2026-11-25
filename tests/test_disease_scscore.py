"""SCScore release parity, comparison scope, source integrity and review decisions."""

import io
import json

import pytest

from rfproteina.disease import corpus_provenance, scscore, scscore_model
from scripts import install_scscore

pytest.importorskip("rdkit.Chem")


def _table(directory, rows):
    (directory / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\n" + "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8"
    )


def _comparison(directory, names, evidence=None):
    body = {"candidates": names}
    if evidence is not None:
        body["decision_evidence"] = evidence
    (directory / scscore.MANIFEST).write_text(json.dumps(body), encoding="utf-8")


@pytest.fixture
def nominations(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    _table(directory, [("binder_0", "CCO"), ("binder_1", "c1ccccc1"), ("binder_2", "CCC")])
    _comparison(directory, ["binder_0", "binder_1"])
    return directory


@pytest.fixture
def release():
    path = scscore_model.weights_path()
    if not path.is_file():
        pytest.skip("Install the released SCScore checkpoint for upstream parity tests")
    assert scscore_model.installed()
    return path


def _choice(directory, candidate=None, *, kind="ligand"):
    from rfproteina.disease import screen_selection

    (result,) = screen_selection.plan(directory, kind, example=candidate, available={scscore.METHOD_ID: True})
    return result


def _fixed_scores(monkeypatch):
    calls = []

    def score(molecules):
        calls.append(tuple(molecules))
        values = {"CCO": 1.25, "c1ccccc1": 2.5, "CCC": 1.25}
        return {identity: values[identity] for identity in molecules}, {"model": scscore_model.MODEL}

    monkeypatch.setattr(scscore_model, "score", score)
    return calls


def test_pinned_release_matches_upstream_standalone_numpy(nominations, release):
    # Generated independently with SCScorer.restore() and get_score_from_smi()
    # at revision 37090a6. The first two are the authors' demonstration inputs.
    expected = {
        "CCCOCCC": 1.4322608053642631,
        "CCCNc1ccccc1": 1.8913059057522013,
        "CCO": 1.0000023556761728,
        "C[C@H](O)C(=O)O": 1.4786884564584466,
        "C[C@@H](O)C(=O)O": 1.7629709868089813,
        "c1ccccc1": 1.191511968296806,
    }
    names = [f"candidate_{i}" for i in range(len(expected))]
    _table(nominations, zip(names, expected, strict=True))
    _comparison(nominations, names)
    report = scscore.measure(nominations)
    assert report["status"] == "scored" and report["scored"] == len(expected)
    assert [row["scscore"] for row in report["samples"]] == pytest.approx(list(expected.values()), abs=1e-10, rel=0)
    assert [row["comparison_rank"] for row in report["samples"]] == [3, 6, 1, 4, 5, 2]
    assert report["provider"]["weights_sha256"] == scscore_model.SHA256
    assert report["provider"]["device"] == "cpu" and report["scale"] == [1, 5]
    provenance = report["corpus_provenance"]
    assert provenance["model"] == scscore.METHOD and provenance["overlap"] == corpus_provenance.SIMULATED_LABELS
    assert "Reaxys" in provenance["corpora"][0]["name"]


def test_only_declared_members_contribute_and_aliases_share_rank(nominations, monkeypatch):
    calls = _fixed_scores(monkeypatch)
    _table(
        nominations,
        [
            ("a", "CCO"),
            ("alias", "[CH3:7][CH2:1]O"),
            ("b", "c1ccccc1"),
            ("c", "CCC"),
            ("outside", "CCCC"),
        ],
    )
    _comparison(nominations, ["b", "alias", "c", "a"])
    assert _choice(nominations).recommended and not calls
    report = scscore.measure(nominations)
    rows = report["samples"]
    assert calls == [("c1ccccc1", "CCO", "CCC")]
    assert [r["candidate_id"] for r in rows] == ["b", "alias", "c", "a"]
    assert [r["comparison_rank"] for r in rows] == [2, 1, 1, 1]
    assert rows[1]["compound_id"] == rows[3]["compound_id"] != rows[2]["compound_id"]
    assert all(r["method"] == scscore.METHOD for r in rows)
    one = scscore.measure(nominations, "alias")
    assert calls[-1] == ("CCO",) and len(one["samples"]) == 1
    assert one["samples"][0]["scscore"] == 1.25 and "comparison_rank" not in one["samples"][0]
    assert "CCO" not in json.dumps(one)
    assert not _choice(nominations, "outside").applicable
    assert not _choice(nominations, kind="protein").applicable


def test_resolved_comparison_skips_before_opening_molecules_or_weights(nominations, monkeypatch):
    evidence = {"verified_route_ref": "ELN-418", "procurement_costs_ref": "QUOTE-27", "resolves_comparison": True}
    _comparison(nominations, ["binder_0", "binder_1"], evidence)
    (nominations / "ligand_candidates.tsv").unlink()
    monkeypatch.setattr(scscore_model, "score", lambda *_: pytest.fail("Resolved decision invoked SCScore"))
    monkeypatch.setattr(
        scscore.ligand_inputs, "bounded_smiles_candidates", lambda *a, **kw: pytest.fail("Read nominees")
    )
    for candidate in (None, "binder_0"):
        choice = _choice(nominations, candidate)
        assert not choice.applicable and "procurement" in choice.reason
        report = scscore.measure(nominations, candidate)
        assert report["status"] == "skipped" and report["reason"] == scscore.RESOLVED
        assert report["decision_evidence"] == evidence and report["attempted"] == report["scored"] == 0
    outside = scscore.measure(nominations, "outsider")
    assert outside["blocker"] == "candidate_outside_comparison"
    assert "decision_evidence" not in outside and "comparison_size" not in outside


def test_two_nominees_can_share_a_table_with_a_large_unrelated_library(nominations, monkeypatch):
    calls = _fixed_scores(monkeypatch)
    _table(nominations, [("binder_0", "CCO"), *((f"other_{i}", "unparsed") for i in range(300)), ("binder_1", "CCC")])
    result = scscore.measure(nominations)
    assert result["scored"] == result["comparison_size"] == 2 and calls == [("CCO", "CCC")]


def test_explicit_selection_skips_resolved_evidence_without_installed_weights(nominations, monkeypatch):
    from rfproteina.disease import screen, screen_selection

    monkeypatch.setattr(screen_selection, "installations", lambda keys: dict.fromkeys(keys, False))
    monkeypatch.setattr(scscore_model, "score", lambda *_: pytest.fail("Attempted inference after a resolved decision"))
    with pytest.raises(ValueError, match="Unavailable screening methods"):
        screen_selection.execution_plan(nominations, "ligand", "explicit", ["scscore"])
    evidence = {"verified_route_ref": "ELN-418", "procurement_costs_ref": "QUOTE-27", "resolves_comparison": True}
    _comparison(nominations, ["binder_0", "binder_1"], evidence)
    report = screen.screen_generation(nominations, binder_kind="ligand", target_id="", screen_methods=["scscore"])
    selection = report["method_selection"]
    assert selection["attempted"] == []
    assert all(ref in selection["skipped"]["scscore"] for ref in ("ELN-418", "QUOTE-27"))
    assert "scscore" not in selection["prepared_measurements"]


@pytest.mark.parametrize(
    "evidence",
    [
        {"verified_route_ref": "ELN-418"},
        {"procurement_costs_ref": "QUOTE-27"},
        {"verified_route_ref": "ELN-418", "procurement_costs_ref": "QUOTE-27", "resolves_comparison": False},
    ],
)
def test_open_decision_still_scores(nominations, evidence, monkeypatch):
    _comparison(nominations, ["binder_0", "binder_1"], evidence)
    calls = _fixed_scores(monkeypatch)
    assert _choice(nominations).recommended
    assert scscore.measure(nominations)["status"] == "scored" and len(calls) == 1


@pytest.mark.parametrize(
    ("evidence", "blocker"),
    [
        ({"resolves_comparison": True, "procurement_costs_ref": "Q1"}, "incomplete_route_and_cost_resolution"),
        ({"resolves_comparison": "true"}, "invalid_resolution_declaration"),
        ({"verified_route_ref": True}, "invalid_decision_evidence_reference"),
        ({"verified_route_ref": "pending"}, "invalid_decision_evidence_reference"),
    ],
)
def test_resolution_claims_require_both_evidence_records(nominations, evidence, blocker):
    _comparison(nominations, ["binder_0", "binder_1"], evidence)
    assert scscore.measure(nominations)["blocker"] == blocker
    assert not _choice(nominations).applicable


def test_distinct_graphs_required_and_no_implicit_comparison_or_run_ligand(nominations, monkeypatch):
    monkeypatch.setattr(scscore_model, "score", lambda *_: pytest.fail("No molecular tie exists"))
    _table(nominations, [("binder_0", "CCO"), ("binder_1", "OCC")])
    assert scscore.measure(nominations)["blocker"] == "two_distinct_compounds_required"
    assert not _choice(nominations).applicable
    (nominations / "ligand_smiles.txt").write_text("CCO")
    (nominations / "ligand_candidates.tsv").unlink()
    missing = scscore.measure(nominations)
    assert missing["blocker"] == "named_candidate_table_required" and missing["attempted"] == 2
    assert all(row["blocker"] == missing["blocker"] for row in missing["samples"])
    (nominations / scscore.MANIFEST).unlink()
    assert not _choice(nominations).applicable and scscore.measure(nominations)["scored"] == 0


@pytest.mark.parametrize(
    ("text", "blocker"),
    [
        ('{"candidates":["a","b"], "candidates":["a","c"]}', "duplicate_comparison_field"),
        ('{"candidates":["a","a"]}', "invalid_or_duplicate_comparison_candidate"),
        ('{"candidates":["a"]}', "comparison_requires_two_to_256_candidates"),
        ('{"candidates":["a","../b"]}', "invalid_or_duplicate_comparison_candidate"),
    ],
)
def test_invalid_manifest_is_a_named_blocker(nominations, text, blocker):
    (nominations / scscore.MANIFEST).write_text(text)
    assert scscore.measure(nominations)["blocker"] == blocker


def test_partial_coverage_preserves_failed_members_and_withholds_group_ranks(nominations, monkeypatch):
    _table(
        nominations,
        [
            ("ok", "CCO"),
            ("ok2", "CCC"),
            ("duplicate", "CCO"),
            ("duplicate", "CO"),
            ("bad", "wrong"),
            ("salt", "CC.[Cl-]"),
            ("query", "*C"),
            ("broken", "CO", "extra"),
        ],
    )
    _comparison(nominations, ["duplicate", "ok", "bad", "ok2", "salt", "missing", "query", "broken"])
    calls = _fixed_scores(monkeypatch)
    assert _choice(nominations).applicable and not _choice(nominations).recommended
    report = scscore.measure(nominations)
    assert report["status"] == "partial" and report["scored"] == 2 and report["attempted"] == 8
    assert all("comparison_rank" not in r for r in report["samples"])
    blockers = {r["candidate_id"]: r["blocker"] for r in report["samples"] if r["status"] == "untested"}
    assert blockers == {
        "duplicate": "duplicate_candidate_nomination",
        "bad": "invalid_molecular_nomination",
        "salt": "requires_one_supported_organic_molecule",
        "missing": "candidate_missing_from_table",
        "query": "invalid_molecular_nomination",
        "broken": "invalid_molecular_nomination",
    }
    assert calls == [("CCO", "CCC")]


@pytest.mark.parametrize("invalid", [0.0, 5.1, float("nan"), True, None])
def test_invalid_or_absent_score_cannot_receive_a_rank(nominations, monkeypatch, invalid):
    values = {"CCO": 2.25}
    if invalid is not None:
        values["c1ccccc1"] = invalid
    monkeypatch.setattr(scscore_model, "score", lambda _: (values, {}))
    report = scscore.measure(nominations)
    first, second = report["samples"]
    assert report["status"] == "partial" and first["scscore"] == 2.25
    assert second["blocker"] == "invalid_scscore_result" and "scscore" not in second
    assert all("comparison_rank" not in row for row in report["samples"])
    values.clear()
    assert scscore.measure(nominations)["blocker"] == "no_valid_scscore_results"


def test_uninstalled_or_partial_weights_block_offline(nominations, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "models"))
    assert not scscore_model.installed()
    assert scscore.measure(nominations)["blocker"] == "scscore_weights_missing"
    path = scscore_model.weights_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"incomplete checkpoint")
    assert not scscore_model.installed()
    assert scscore.measure(nominations)["blocker"] == "scscore_weights_digest_mismatch"


def test_cached_weights_checked_again_and_bad_installs_preserve_existing(nominations, tmp_path, monkeypatch, release):
    original = release.read_bytes()
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "models"))
    source = tmp_path / "model.json.gz"
    source.write_bytes(original)
    target = install_scscore.install(source)
    assert scscore.measure(nominations)["scored"] == 2
    damaged = bytes([original[0] ^ 1]) + original[1:]
    target.write_bytes(damaged)
    assert not scscore_model.installed()
    report = scscore.measure(nominations)
    assert report["scored"] == 0 and all(r["blocker"] == "scscore_weights_digest_mismatch" for r in report["samples"])
    source.write_bytes(b"partial copy")
    with pytest.raises(scscore.SCScoreError, match="digest_mismatch"):
        install_scscore.install(source)
    assert target.read_bytes() == damaged
    monkeypatch.setattr(install_scscore, "urlopen", lambda *_a, **_kw: io.BytesIO(original))
    assert install_scscore.install() == target and target.read_bytes() == original
    monkeypatch.setattr(install_scscore, "urlopen", lambda *_a, **_kw: pytest.fail("Re-downloaded valid checkpoint"))
    assert install_scscore.install() == target


def test_selected_screen_persists_off_axis_and_mcp_scopes_to_candidate(nominations, monkeypatch, capsys):
    from rfproteina.disease import mcp_api, screen, screen_selection
    from rfproteina.disease.validation_checks import candidate_validation_plan
    from scripts.screen_extras import main

    samples = nominations / "inference_samples"
    samples.mkdir()
    for name in ("binder_0", "binder_1", "binder_2"):
        (samples / f"{name}_generated.cif").write_text("data_example\n")
    (nominations / "generation.json").write_text('{"binder_kind":"ligand"}')
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(nominations.parent))
    monkeypatch.setattr(screen_selection, "installations", lambda keys=None: dict.fromkeys(keys or (), True))
    calls = _fixed_scores(monkeypatch)
    baseline = screen.screen_generation(nominations, binder_kind="ligand", target_id="", screen_selection="explicit")
    assert not calls
    selected = screen.screen_generation(nominations, binder_kind="ligand", target_id="", screen_methods=["scscore"])
    assert all(selected[k] == baseline[k] for k in ("call", "states", "hits", "cuts"))
    stored = json.loads((nominations / "screen.json").read_bytes())["method_selection"]["prepared_measurements"][
        "scscore"
    ]
    assert stored["samples"][0]["scscore"] == 1.25 and stored["samples"][1]["comparison_rank"] == 2
    candidates, _ = candidate_validation_plan((nominations,), None)
    for candidate in candidates:
        reply = mcp_api.call_tool(
            "run_selected_screens",
            {
                "directory": str(nominations),
                "candidate_id": candidate.key,
                "methods": ["scscore"],
            },
        )["results"][0]
        if candidate.example_id == "binder_2":
            assert reply["status"] == "inapplicable"
        else:
            (row,) = reply["measurement"]["samples"]
            assert reply["status"] == "scored" and row["candidate_id"] == candidate.example_id
            assert "comparison_rank" not in row
    capsys.readouterr()
    assert main(["scscore", str(nominations)]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == stored["samples"]
