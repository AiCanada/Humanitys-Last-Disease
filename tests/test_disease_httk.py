"""EPA httk fixes its scenario and publishes only attributable population summaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rfproteina.disease import httk

_REF = ("58-08-2", "DTXSID0020232")
_OTHER = ("103-90-2", "DTXSID2020006")
_CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
_ACETAMINOPHEN = "CC(=O)Nc1ccc(O)cc1"
_KEYS = {_REF: "RYYVLZVUVIJVGH-UHFFFAOYSA-N", _OTHER: "RZVAJINKPMORJF-UHFFFAOYSA-N"}


@pytest.fixture
def identity_index(monkeypatch):
    monkeypatch.setattr(httk.httk_identity, "load_index", lambda: _KEYS)


def _write(directory, binding=None, rows=None):
    binding = {"candidate_a": _REF} if binding is None else binding
    rows = [("candidate_a", _CAFFEINE)] if rows is None else rows
    raw = json.dumps({"scenario": httk.SCENARIO, "candidates": {
        k: {"cas": p[0], "dtxsid": p[1]} for k, p in binding.items()
    }}).encode("utf-8")
    (directory / "httk.json").write_bytes(raw)
    (directory / "ligand_candidates.tsv").write_text(
        "candidate_id\tsmiles\n" + "".join(f"{name}\t{smiles}\n" for name, smiles in rows), encoding="utf-8",
    )
    return raw


def test_requires_valid_exact_ids_and_no_arbitrary_scenario_parameters(tmp_path):
    assert httk.valid_cas(_REF[0]) and httk.valid_cas(_OTHER[0])
    assert not httk.valid_cas("50-78-3") and not httk.valid_cas("50-78-2\t")
    assert httk.valid_dtxsid(_REF[1]) and not httk.valid_dtxsid("DTXCID1234567")
    path = tmp_path / "httk.json"
    path.write_text('{"scenario":"human_oral_unit_intake","scenario":"other","candidates":{}}')
    assert httk.prepare(tmp_path).blocker == "httk_invalid_manifest"
    value = json.loads(_write(tmp_path))
    value["scenario"] = "intravenous"
    path.write_text(json.dumps(value))
    assert httk.prepare(tmp_path).blocker == "httk_scenario_unsupported"
    value["scenario"] = httk.SCENARIO
    value["dose"] = 1
    path.write_text(json.dumps(value))
    assert httk.prepare(tmp_path).blocker == "httk_invalid_manifest"


def test_nominations_are_scoped_and_no_run_fallback(tmp_path, identity_index):
    pytest.importorskip("rdkit")
    raw = _write(tmp_path)
    plan = httk.prepare(tmp_path, "candidate_a")
    assert plan.manifest_sha256 == hashlib.sha256(raw).hexdigest() and plan.blocker is None
    pair, = plan.members
    assert pair.name == "candidate_a" and pair.reference == _REF and pair.compound_id.startswith("smiles_sha256_")
    assert pair.blocker is None and pair.reference_inchikey == _KEYS[_REF]
    assert httk.prepare(tmp_path, "candidate_b").blocker == "httk_candidate_not_nominated"
    (tmp_path / "ligand_candidates.tsv").unlink()
    (tmp_path / "ligand_smiles.txt").write_text("CCO\n")
    assert httk.prepare(tmp_path).blocker == "httk_named_nominations_required"


def test_conflicting_nomination_cannot_be_bypassed_by_candidate_scope(tmp_path, identity_index):
    pytest.importorskip("rdkit")
    _write(tmp_path, {"candidate_a": _REF, "candidate_b": _REF}, [("candidate_a", "CCO"), ("candidate_b", "CCCO")])
    assert {item.blocker for item in httk.prepare(tmp_path).members} == {"httk_ambiguous_ligand"}
    assert httk.prepare(tmp_path, "candidate_a").members[0].blocker == "httk_ambiguous_ligand"
    _write(tmp_path, {"candidate_a": _REF, "candidate_b": _OTHER}, [("candidate_a", "CCO"), ("candidate_b", "OCC")])
    assert {item.blocker for item in httk.prepare(tmp_path).members} == {"httk_ambiguous_ligand"}


def test_manifest_sources_are_bounded_and_not_symlinked(tmp_path):
    assert httk.prepare(tmp_path).blocker == "httk_manifest_missing"
    path = tmp_path / "httk.json"
    path.write_bytes(b" " * (httk.MAX_MANIFEST_BYTES + 1))
    assert httk.prepare(tmp_path).blocker == "httk_manifest_over_limit"
    path.unlink()
    unsafe = tmp_path / "foreign.json"
    unsafe.write_text("{}")
    try:
        path.symlink_to(unsafe)
    except OSError:
        pytest.skip("symlinks not available")
    assert httk.prepare(tmp_path).blocker == "httk_manifest_unreadable"


def test_read_numeric_population_summary_rejects_unattributed_or_corrupt_outputs():
    inputs = ("0.2855", "Paini 2020", "1", "Paini 2020")
    legacy = "\t".join((*_REF, "reported", "0", "2.5", "4", ""))
    as_text = legacy + "\t" + "\t".join(inputs) + "\n"
    assert httk._read_results(as_text, (_REF,))[_REF] == {
        "status": "reported", "p05": 0, "median": 2.5, "p95": 4, "human_in_vitro": {
            "clint_ul_min_per_million_hepatocytes": 0.2855, "clint_reference": "Paini 2020",
            "fraction_unbound_plasma": 1, "fraction_unbound_plasma_reference": "Paini 2020",
        },
    }
    for values in (("NaN", "2", "4"), ("0", "6", "4"), ("0", "2", "inf"), ("-1", "2", "5")):
        bad = "\t".join((*_REF, "reported", *values, "", *inputs))
        assert httk._read_results(bad, (_REF,))[_REF]["blocker"] == "httk_invalid_draws"
    with pytest.raises(httk.HttkError, match="httk_worker_failed"):
        httk._read_results(legacy, (_REF,))
    with pytest.raises(httk.HttkError, match="httk_worker_failed"):
        httk._read_results(as_text, (_OTHER,))
    with pytest.raises(httk.HttkError, match="httk_worker_failed"):
        httk._read_results(as_text + as_text, (_REF,))


@pytest.mark.parametrize("inputs", [
    ("NaN", "Paini 2020", "1", "Paini 2020"), ("-1", "Paini 2020", "1", "Paini 2020"),
    ("1", "Paini 2020", "0", "Paini 2020"), ("1", "Paini 2020", "1.001", "Paini 2020"),
    ("1", "Paini 2020", "1", ""), ("1", "Paini 2020", "1", "Zanelli 2012"),
    ("1", "Lombardo 2018", "1", "Paini 2020"), ("1", "/private/source", "1", "Paini 2020"),
])
def test_worker_rejects_missing_values_and_sources_from_another_assay(inputs):
    line = "\t".join((*_REF, "reported", "1", "2", "3", "", *inputs))
    assert httk._read_results(line, (_REF,))[_REF] == {"status": "untested", "blocker": "httk_human_inputs_unavailable"}


def test_installed_native_worker_reports_published_assay_inputs():
    executable = httk.runtime_path()
    if executable is None or not httk._available(executable, executable.stat().st_mtime_ns, 0):
        pytest.skip("The pinned EPA httk R package is optional")
    caffeine = ("58-08-2", "DTXSID0020232")
    aspirin = ("50-78-2", "DTXSID5020108")
    pairs = (caffeine, aspirin)
    payload = "".join(f"{cas}\t{dtxsid}\n" for cas, dtxsid in pairs).encode("ascii")
    records = httk._read_results(httk._run(executable, "run", payload, timeout=240), pairs)
    assert records[caffeine]["human_in_vitro"] == {
        "clint_ul_min_per_million_hepatocytes": 0.2855, "clint_reference": "Paini 2020",
        "fraction_unbound_plasma": 1.0, "fraction_unbound_plasma_reference": "Paini 2020",
    }
    # Published clearance p = 0.0924 exceeds the worker's prespecified 0.05 cutoff.
    assert records[aspirin] == {"status": "untested", "blocker": "httk_human_inputs_unavailable"}

    # One missing label would suppress every assay attributed to that source;
    # the two controls alone exercise only a small part of the release.
    script = (
        "d <- httk::chem.physical_and_invitro.data; "
        "for (k in c('Human.Clint.Reference', 'Human.Funbound.plasma.Reference')) "
        "{ x <- unique(stats::na.omit(d[[k]])); writeLines(paste(k, x, sep='\\t')) }"
    )
    labels = subprocess.run(
        [str(executable), "--vanilla", "-e", script], capture_output=True, text=True, timeout=30, check=True,
    ).stdout.splitlines()
    native_refs = {tuple(row.split("\t")) for row in labels}
    assert native_refs == (
        {("Human.Clint.Reference", source) for source in httk._CLINT_REFERENCES}
        | {("Human.Funbound.plasma.Reference", source) for source in httk._FUP_REFERENCES}
    )


def test_batch_reuses_exact_reference_without_emitting_structures_or_a_verdict(tmp_path, monkeypatch, identity_index):
    pytest.importorskip("rdkit")
    _write(tmp_path, {"candidate_a": _REF, "alias": _REF, "other": _OTHER},
           [("candidate_a", _CAFFEINE), ("alias", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"), ("other", _ACETAMINOPHEN)])
    monkeypatch.setattr(httk, "runtime_path", lambda: Path("trusted-Rscript"))
    monkeypatch.setattr(httk, "asset_available", lambda: True)

    def run(_exe, action, data, *, timeout):
        assert action == "run" and timeout == 240
        assert data.decode() == "\t".join(_REF) + "\n" + "\t".join(_OTHER) + "\n"
        return (
            "\t".join((*_REF, "reported", "1", "2", "3", "", "0.2855", "Paini 2020", "1", "Paini 2020")) + "\n"
            + "\t".join((*_OTHER, "untested", "", "", "", "httk_human_inputs_unavailable", "", "", "", "")) + "\n"
        )

    monkeypatch.setattr(httk, "_run", run)
    report = httk.measure(tmp_path)
    assert report["status"] == "partial" and (report["attempted"], report["reported"]) == (3, 2)
    a, b, c = report["members"]
    assert a["compound_id"] == b["compound_id"] and a["median"] == b["median"] == 2
    assert a["human_in_vitro"] == b["human_in_vitro"] and "human_in_vitro" not in c
    assert c["blocker"] == "httk_human_inputs_unavailable" and "p05" not in c
    assert report["provider"] == httk.PROVIDER and report["reference_intake_mg_kg_day"] == 1
    assert _CAFFEINE not in json.dumps(report) and not {"call", "threshold", "verdict"} & report.keys()


def test_singly_mislabeled_reference_and_unmapped_pair_do_not_receive_a_result(tmp_path, identity_index, monkeypatch):
    pytest.importorskip("rdkit")
    monkeypatch.setattr(httk, "_run", lambda *a, **kw: pytest.fail("Launched the solver for an unverified structure"))
    _write(tmp_path, rows=[("candidate_a", "CCO")])
    record = httk.measure(tmp_path, "candidate_a")
    row, = record["members"]
    assert row["blocker"] == "httk_structure_mismatch" and "median" not in row and "reference_inchikey" not in row
    _write(tmp_path, binding={"candidate_a": (_REF[0], _OTHER[1])})
    assert httk.prepare(tmp_path).members[0].blocker == "httk_structure_unmapped"


def test_key_match_includes_the_stereochemical_block(tmp_path, monkeypatch):
    chem = pytest.importorskip("rdkit.Chem")
    left, right = (chem.MolToInchiKey(chem.MolFromSmiles(s)) for s in ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"))
    assert left[:14] == right[:14] and left != right
    monkeypatch.setattr(httk.httk_identity, "load_index", lambda: {_REF: left})
    _write(tmp_path, rows=[("candidate_a", "C[C@@H](O)C(=O)O")])
    assert httk.prepare(tmp_path).members[0].blocker == "httk_structure_mismatch"


def test_missing_crosswalk_abstains_before_running_r(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    def unavailable():
        raise httk.httk_identity.IdentityReferenceError("httk_identity_reference_unavailable")
    monkeypatch.setattr(httk.httk_identity, "load_index", unavailable)
    _write(tmp_path)
    assert httk.prepare(tmp_path).members[0].blocker == "httk_identity_reference_unavailable"


def test_worker_launch_is_private_fixed_cpu_and_bounded(monkeypatch):
    directory = None

    def launch(argv, **options):
        nonlocal directory
        directory = Path(options["cwd"])
        assert directory.is_dir() and str(directory) == options["env"]["TMPDIR"]
        assert argv[1:] == ["--vanilla", str(httk._WORKER), "probe"]
        assert options["env"]["CUDA_VISIBLE_DEVICES"] == "" and options["env"]["R_DATATABLE_NUM_THREADS"] == "1"
        return SimpleNamespace(stdout=b"httk\t2.7.4\n")

    monkeypatch.setattr(httk.subprocess, "run", launch)
    assert httk._run(Path("local-Rscript"), "probe", b"", timeout=12).strip() == "httk\t2.7.4"
    assert directory is not None and not directory.exists()

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("Rscript", timeout=12)

    monkeypatch.setattr(httk.subprocess, "run", timeout)
    with pytest.raises(httk.HttkError, match="httk_worker_timeout"):
        httk._run(Path("local-Rscript"), "probe", b"", timeout=12)
