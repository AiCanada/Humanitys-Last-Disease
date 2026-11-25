"""Release parity, isoform attribution and isolation of the CYP substrate endpoint."""

import json
import subprocess

import pytest

from rfproteina.disease import corpus_provenance, cypstrate, cypstrate_runtime

pytest.importorskip("rdkit")

ACICLOVIR = "C1=NC2=C(N1COCCO)N=C(NC2=O)N"
AMIODARONE = "CCN(CC)CCOc1c(I)cc(cc1I)C(=O)c2c3ccccc3oc2CCCC"
ISOFORMS = cypstrate.CYPSTRATE_ISOFORMS


def manifest(directory, candidates, *, mode="best_performance", systemic=True):
    (directory / cypstrate.MANIFEST).write_text(
        json.dumps({"systemic": systemic, "prediction_mode": mode, "candidates": candidates}), encoding="utf-8"
    )


def table(directory, rows):
    (directory / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\n" + "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8"
    )


@pytest.fixture
def run(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    table(directory, [("a", "CCO"), ("b", "CCC"), ("outside", "CCCC")])
    manifest(directory, {"a": ["CYP3A4", "CYP1A2"], "b": ["2D6"]})
    return directory


def choice(run, candidate=None, kind="ligand"):
    from rfproteina.disease import screen_selection

    (item,) = screen_selection.plan(run, kind, example=candidate, available={"cypstrate": True})
    return item


def returned(entry):
    return {
        "id": entry["id"],
        "preprocessed_id": "a" * 64,
        "upstream_smiles_changed": False,
        "predictions": {
            iso: {"label": "Substrate" if iso == "CYP3A4" else "Non-substrate", "similarity": 0.84}
            for iso in entry["isoforms"]
        },
    }


@pytest.fixture
def predictor(monkeypatch):
    calls = []

    def predict(inputs, mode):
        calls.append((inputs, mode))
        return {"results": [returned(entry) for entry in reversed(inputs)]}

    monkeypatch.setattr(cypstrate_runtime, "predict", predict)
    return calls


@pytest.mark.parametrize("mode", ["best_performance", "full_coverage"])
def test_installed_release_matches_authors_reference_compounds(run, mode):
    if not cypstrate_runtime.installed():
        pytest.skip("Requires the authors' pinned CYPstrate image on the local Docker engine")
    table(
        run,
        [
            ("a", ACICLOVIR),
            ("b", AMIODARONE),
            ("excluded", "C[Ge](C)(C)C"),
            ("left", "C[C@H](O)C(=O)O"),
            ("right", "C[C@@H](O)C(=O)O"),
        ],
    )
    manifest(run, {"a": ISOFORMS, "b": ISOFORMS, "excluded": ["2D6"], "left": ["2D6"], "right": ["2D6"]}, mode=mode)
    report = cypstrate.measure(run)
    assert report["status"] == "partial" and report["attempted"] == 21
    by_pair = {(r["candidate_id"], r["isoform"]): r for r in report["samples"]}
    assert by_pair["excluded", "CYP2D6"]["blocker"] == "cypstrate_preprocessing_failed"
    a, b = by_pair["a", "CYP1A2"], by_pair["b", "CYP3A4"]
    assert a["nearest_training_maccs_tanimoto"] == 0.75 and b["nearest_training_maccs_tanimoto"] == 1
    assert b["classification"] == "substrate" and by_pair["a", "CYP3A4"]["classification"] == "non_substrate"
    if mode == "best_performance":
        assert a["status"] == "abstained" and "classification" not in a
        assert [by_pair["b", iso].get("classification") for iso in ISOFORMS] == [
            "substrate",
            "non_substrate",
            None,
            None,
            "non_substrate",
            "substrate",
            "substrate",
            "non_substrate",
            "substrate",
        ]
    else:
        assert a["classification"] == "non_substrate" and report["abstained"] == 0
    left, right = by_pair["left", "CYP2D6"], by_pair["right", "CYP2D6"]
    assert (
        left["compound_id"] != right["compound_id"]
        and left["preprocessed_compound_id"] == right["preprocessed_compound_id"]
    )
    assert all(row["method"] == cypstrate.MODES[mode] for row in report["samples"])
    assert report["provider"]["image"] == cypstrate_runtime.IMAGE
    assert report["corpus_provenance"]["overlap"] == corpus_provenance.BENCHMARK_LABEL_CORPUS
    assert report["corpus_provenance"]["corpora"][0]["rows"] == 1831


def test_scope_and_deduplication_preserve_only_nominated_pairs(run, predictor):
    table(run, [("a", "CCO"), ("alias", "OCC"), ("b", "CCC"), ("outside", "CCCC")])
    manifest(run, {"alias": ["2D6"], "b": ["1A2"], "a": ["3A4", "2D6"]})
    assert choice(run).recommended and not predictor
    report = cypstrate.measure(run)
    assert report["status"] == "classified" and report["classified"] == 4
    assert [(r["candidate_id"], r["isoform"]) for r in report["samples"]] == [
        ("alias", "CYP2D6"),
        ("b", "CYP1A2"),
        ("a", "CYP3A4"),
        ("a", "CYP2D6"),
    ]
    assert len(predictor[0][0]) == 2
    assert report["samples"][0]["compound_id"] == report["samples"][2]["compound_id"]
    scoped = cypstrate.measure(run, "b")
    assert [e["smiles"] for e in predictor[-1][0]] == ["CCC"] and scoped["attempted"] == 1
    assert cypstrate.measure(run, "outside")["samples"] == [] and len(predictor) == 2
    assert not choice(run, kind="protein").applicable and not choice(run, "outside").applicable


@pytest.mark.parametrize(
    ("content", "blocker"),
    [
        ('{"candidates":{"a":["2D6"]}}', "cypstrate_exposure_unspecified"),
        ('{"systemic":"true","candidates":{"a":["2D6"]}}', "cypstrate_exposure_unspecified"),
        ('{"systemic":true,"candidates":{"a":["2D6"],"a":["2B6"]}}', "cypstrate_invalid_manifest"),
        ('{"systemic":true,"candidates":{"a":["2C18"]}}', "cypstrate_invalid_isoforms"),
        ('{"systemic":true,"candidates":{"a":["CYP2D6","2d6"]}}', "cypstrate_invalid_isoforms"),
        ('{"systemic":true,"candidates":{"a":["2D6"]},"prediction_mode":"fast"}', "cypstrate_invalid_mode"),
    ],
)
def test_invalid_declarations_never_run_a_model(run, predictor, content, blocker):
    (run / cypstrate.MANIFEST).write_text(content)
    assert cypstrate.measure(run)["blocker"] == blocker
    assert not choice(run).applicable and not predictor


def test_systemic_and_isoform_hypotheses_required_and_run_ligand_never_broadcast(run, predictor):
    (run / cypstrate.MANIFEST).unlink()
    assert not choice(run).applicable and cypstrate.measure(run)["blocker"] == "cypstrate_manifest_missing"
    manifest(run, {"a": ["3A4"]}, systemic=False)
    assert cypstrate.measure(run)["status"] == "skipped" and not choice(run).applicable
    (run / "ligand_candidates.tsv").unlink()
    (run / "ligand_smiles.txt").write_text("CCO\n")
    manifest(run, {"a": ["3A4"]})
    assert cypstrate.measure(run)["blocker"] == "candidate_requires_named_nomination" and not predictor
    manifest(run, {"run_ligand": ["3A4"]})
    assert cypstrate.measure(run)["classified"] == 1
    assert cypstrate.measure(run, "run_ligand")["blocker"] == "candidate_requires_named_nomination"


def test_mixed_input_keeps_missing_and_duplicate_candidates_in_the_denominator(run, predictor):
    table(run, [("good", "CCC"), ("dup", "CCC"), ("dup", "CO"), ("bad", "wrong"), ("broken", "CC", "extra")])
    manifest(run, {n: ["3A4"] for n in ["missing", "good", "dup", "bad", "broken"]})
    assert choice(run).applicable and not choice(run).recommended
    report = cypstrate.measure(run)
    assert report["status"] == "partial" and report["attempted"] == 5 and report["classified"] == 1
    assert [r.get("blocker") for r in report["samples"]] == [
        "candidate_missing_from_table",
        None,
        "duplicate_candidate_nomination",
        "invalid_nominated_smiles",
        "invalid_molecular_nomination",
    ]


@pytest.mark.parametrize(
    ("label", "similarity"),
    [
        ("No prediction", 0.8),
        (0.9, 1.0),
        ("Non-substrate", True),
        ("Substrate", float("nan")),
        ("Substrate", 1.1),
        ("Inhibitor", 0.8),
    ],
)
def test_invalid_or_abstaining_results_never_become_negative_classifications(run, monkeypatch, label, similarity):
    def predict(inputs, mode):
        row = returned(inputs[0])
        row["predictions"]["CYP3A4"] = {"label": label, "similarity": similarity}
        return {"results": [row]}

    monkeypatch.setattr(cypstrate_runtime, "predict", predict)
    both = cypstrate.measure(run, "a")
    first = both["samples"][0]
    assert both["status"] == "partial" and "classification" not in first
    assert first["blocker"] == ("cypstrate_no_consensus" if label == "No prediction" else "cypstrate_invalid_output")
    if label == "No prediction":
        manifest(run, {"a": ["3A4"]})
        assert cypstrate.measure(run)["status"] == "abstained"
        manifest(run, {"a": ["3A4"]}, mode="full_coverage")
        assert cypstrate.measure(run)["samples"][0]["blocker"] == "cypstrate_invalid_output"


def test_foreign_duplicate_and_missing_results_are_never_reassigned(run, monkeypatch):
    monkeypatch.setattr(cypstrate_runtime, "predict", lambda inputs, _: {"results": [returned(inputs[0])] * 2})
    report = cypstrate.measure(run)
    assert report["classified"] == 0 and all(r["blocker"] == "cypstrate_unattributed_result" for r in report["samples"])
    monkeypatch.setattr(cypstrate_runtime, "predict", lambda *_: {"results": [{"id": "unrequested"}]})
    assert cypstrate.measure(run)["blocker"] == "cypstrate_invalid_output"


def test_only_a_verified_local_image_can_supply_a_record(run, monkeypatch):
    engines = [["docker", "--host", "unix:///first.sock"], ["docker", "--host", "unix:///second.sock"]]
    monkeypatch.setattr(cypstrate_runtime, "_local_engines", lambda: iter(engines))
    fake = subprocess.CompletedProcess([], 0, stdout=b"sha256:wrong")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake)
    assert not cypstrate_runtime.installed()
    report = cypstrate.measure(run)
    assert report["classified"] == 0 and report["blocker"] == "cypstrate_image_unavailable"
    good = subprocess.CompletedProcess([], 0, stdout=cypstrate_runtime.DIGEST.encode())
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: fake if args[:3] == engines[0] else good)
    assert cypstrate_runtime._engine() == engines[1]


def test_rejected_manifest_or_table_keeps_model_idle(run, predictor):
    table_path = run / "ligand_candidates.tsv"
    table_path.write_bytes(b"x" * (1024 * 1024 + 1))
    report = cypstrate.measure(run)
    assert report["attempted"] == 3 and report["blocker"] == "ligand_input_exceeds_interactive_size"
    assert all(r["status"] == "untested" for r in report["samples"])
    (run / cypstrate.MANIFEST).unlink()
    (run / cypstrate.MANIFEST).symlink_to(table_path)
    assert cypstrate.measure(run)["blocker"] == "cypstrate_manifest_not_regular" and not predictor


def test_timeout_cleans_only_the_owned_container_and_never_pulls_or_sends_inputs_as_args(monkeypatch):
    calls = []
    monkeypatch.setattr(cypstrate_runtime, "_engine", lambda: ["docker", "--host", "unix:///local.sock"])

    def invoke(args, **kwargs):
        calls.append((args, kwargs))
        if "run" in args:
            raise subprocess.TimeoutExpired(args, 5)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", invoke)
    with pytest.raises(cypstrate.CypstrateError, match="cypstrate_execution_timeout"):
        cypstrate_runtime.predict([{"id": "abc", "smiles": ACICLOVIR, "isoforms": ["CYP1A2"]}], "best_performance")
    (command, options), (cleanup, _) = calls
    assert "--network=none" in command and "--pull=never" in command and cypstrate_runtime.IMAGE in command
    assert ACICLOVIR.encode() in options["input"] and ACICLOVIR not in str(command)
    assert cleanup[-3:] == ["rm", "--force", command[command.index("--name") + 1]]


def test_selected_cli_and_mcp_leave_cyp_inhibition_and_clearance_independent(run, predictor, monkeypatch, capsys):
    from rfproteina.disease import mcp_api, screen, screen_selection
    from rfproteina.disease.validation_checks import candidate_validation_plan
    from scripts.screen_extras import main

    samples = run / "inference_samples"
    samples.mkdir()
    for name in ("a", "b", "outside"):
        (samples / f"{name}_generated.cif").write_text("data_example\n")
    (run / "generation.json").write_text('{"binder_kind":"ligand"}')
    inhibition = run / "toxicology.tsv"
    inhibition.write_text("axis\tnm\ncyp3a4\t25\ncyp2d6\t10\n")
    original = inhibition.read_bytes()
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(run.parent))
    monkeypatch.setattr(screen_selection, "installations", lambda keys=None: dict.fromkeys(keys or (), True))
    base = screen.screen_generation(run, binder_kind="ligand", target_id="", screen_selection="explicit")
    selected = screen.screen_generation(run, binder_kind="ligand", target_id="", screen_methods=["cypstrate"])
    assert inhibition.read_bytes() == original
    assert base["toxicology"]["cyp2d6"]["measurement"] == 10 and base["toxicology"]["cyp2d6"]["units"] == "nM"
    assert all(
        selected.get(k) == base.get(k)
        for k in ("call", "states", "hits", "cuts", "toxicology", "ligand_microsomal_clearance")
    )
    record = json.loads((run / "screen.json").read_bytes())["method_selection"]["prepared_measurements"]["cypstrate"]
    assert record["classified"] == 3 and record["samples"][0]["endpoint"] == "cyp3a4_substrate"
    assert record["samples"][-1]["endpoint"] == "cyp2d6_substrate"
    assert record["samples"][-1]["classification"] == "non_substrate"
    candidates, _ = candidate_validation_plan((run,), None)
    assert {c.example_id for c in candidates} == {"a", "b", "outside"}
    for candidate in candidates:
        (result,) = mcp_api.call_tool(
            "run_selected_screens",
            {
                "directory": str(run),
                "candidate_id": candidate.key,
                "methods": ["cypstrate"],
            },
        )["results"]
        if candidate.example_id == "outside":
            assert result["status"] == "inapplicable"
        else:
            assert result["status"] == "classified"
            assert {row["candidate_id"] for row in result["measurement"]["samples"]} == {candidate.example_id}
    capsys.readouterr()
    assert main(["cypstrate", str(run), "--candidate", "b"]) == 0
    assert json.loads(capsys.readouterr().out)["samples"][0]["isoform"] == "CYP2D6"
    calls = len(predictor)
    manifest(run, {"a": ["CYP3A4"]}, systemic=False)
    monkeypatch.setattr(screen_selection, "installations", lambda keys: dict.fromkeys(keys, False))
    screen.screen_generation(run, binder_kind="ligand", target_id="", screen_methods=["cypstrate"])
    updated = json.loads((run / "screen.json").read_bytes())["method_selection"]
    assert "non-systemic" in updated["skipped"]["cypstrate"]
    assert "cypstrate" not in updated["prepared_measurements"] and len(predictor) == calls
