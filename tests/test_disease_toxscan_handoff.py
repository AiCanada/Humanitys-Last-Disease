"""A manually returned author CSV can be attached only to the nominated molecular structure."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from rfproteina.disease import toxscan
from rfproteina.disease import toxscan_handoff as handoff
from rfproteina.disease import toxscan_inputs as inputs
from scripts import screen_extras
from tests.test_disease_toxscan import author_shape, csv_bytes

pytest.importorskip("rdkit")


def table(directory, text="one\tOCC\ntwo\tCC\n"):
    (directory / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\n" + text, encoding="utf-8")


def test_cli_prepares_one_smiles_then_imports_a_source_without_manual_hashing(tmp_path, capsys):
    table(tmp_path)
    assert screen_extras.main(["toxscan-request", str(tmp_path), "--candidate", "one"]) == 0
    request = json.loads(capsys.readouterr().out)
    assert request["canonical_smiles"] == "CCO" and request["source"] == inputs.SOURCE
    assert "shares this molecular structure" in request["notice"]
    assert {p.name for p in tmp_path.iterdir()} == {"ligand_candidates.tsv"}

    payload = csv_bytes([author_shape("CC"), author_shape("CCO")])
    source = tmp_path / "downloaded.csv"
    source.write_bytes(payload)
    command = ["toxscan-import", str(tmp_path), str(source), "--candidate", "one"]
    for digest, blocker in (("bad", "toxscan_invalid_request_digest"), ("0" * 64, "toxscan_nomination_changed")):
        assert screen_extras.main([*command, "--expected-structure-sha256", digest]) == 2
        assert json.loads(capsys.readouterr().out)["blocker"] == blocker
        assert not (tmp_path / toxscan.MANIFEST).exists() and not (tmp_path / inputs.SOURCE_FILE).exists()
    command.extend(("--expected-structure-sha256", request["structure_sha256"]))
    assert screen_extras.main(command) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == toxscan.screen_run(tmp_path, "one")
    assert report["source_sha256"] == inputs.sha256(payload)
    assert report["structure_sha256"] == request["structure_sha256"]
    manifest = tmp_path / toxscan.MANIFEST
    original = manifest.read_bytes()
    assert json.loads(original)["structure_sha256"] == report["structure_sha256"]
    receipt_time = manifest.stat().st_mtime_ns
    assert screen_extras.main(command) == 0
    assert manifest.read_bytes() == original and manifest.stat().st_mtime_ns == receipt_time
    assert json.loads(capsys.readouterr().out) == report

    # A multi-row download must never silently rebind if the original ID later names another graph in the CSV.
    table(tmp_path, "one\tCC\ntwo\tCCO\n")
    assert toxscan.screen_run(tmp_path, "one")["blocker"] == "toxscan_nomination_changed"
    with pytest.raises(inputs.ToxScanError, match="toxscan_nomination_changed"):
        handoff.import_export(tmp_path, "one", payload, expected_structure_sha256=request["structure_sha256"])


def test_invalid_upload_or_ambiguous_candidate_creates_no_artifacts(tmp_path, capsys):
    table(tmp_path)
    for data, blocker in ((b"test", "toxscan_invalid_export_schema"),
                          (csv_bytes([author_shape("CCN")]), "toxscan_structure_not_found"),
                          (csv_bytes([author_shape("CCO"), author_shape("OCC")]), "toxscan_ambiguous_export"),
                          (b"a" * (inputs.MAX_EXPORT_BYTES + 1), "toxscan_invalid_export")):
        with pytest.raises(inputs.ToxScanError, match=blocker):
            handoff.import_export(tmp_path, "one", data)
    assert screen_extras.main(["toxscan-import", str(tmp_path), str(tmp_path / "absent"), "--candidate", "one"]) == 2
    assert json.loads(capsys.readouterr().out)["blocker"] == "toxscan_export_missing"
    table(tmp_path, "one\tOCC\none\tCCO\n")
    with pytest.raises(inputs.ToxScanError, match="toxscan_ambiguous_nomination"):
        handoff.prepare_request(tmp_path, "one")
    assert {p.name for p in tmp_path.iterdir()} == {"ligand_candidates.tsv"}


@pytest.mark.parametrize("conflict_file", [inputs.SOURCE_FILE, inputs.MANIFEST])
def test_existing_conflicts_are_preserved_on_either_side_of_the_pair(tmp_path, conflict_file):
    table(tmp_path)
    conflict = tmp_path / conflict_file
    conflict.write_bytes(b"existing operator data")
    with pytest.raises(inputs.ToxScanError, match="toxscan_import_conflict"):
        handoff.import_export(tmp_path, "one", csv_bytes([author_shape("CCO")]))
    assert conflict.read_bytes() == b"existing operator data"
    assert not (tmp_path / (inputs.MANIFEST if conflict_file == inputs.SOURCE_FILE else inputs.SOURCE_FILE)).exists()


def test_second_link_failure_rolls_back_only_files_created_in_that_transaction(tmp_path, monkeypatch):
    table(tmp_path)
    payload = csv_bytes([author_shape("CCO")])
    link = handoff.os.link

    def fail_on_manifest(source, destination):
        if destination.name == inputs.MANIFEST:
            raise OSError("simulated write failure")
        return link(source, destination)

    monkeypatch.setattr(handoff.os, "link", fail_on_manifest)
    for existing in (False, True):
        if existing:
            (tmp_path / inputs.SOURCE_FILE).write_bytes(payload)
        with pytest.raises(inputs.ToxScanError, match="toxscan_import_write_failed"):
            handoff.import_export(tmp_path, "one", payload)
        assert (tmp_path / inputs.SOURCE_FILE).exists() == existing
        assert not (tmp_path / inputs.MANIFEST).exists()
    monkeypatch.setattr(handoff.os, "link", link)
    assert handoff.import_export(tmp_path, "one", payload)["status"] == "complete"


def test_equivalent_legacy_manifest_keeps_original_bytes(tmp_path):
    table(tmp_path)
    payload = csv_bytes([author_shape("CCO")])
    manifest = {"source": inputs.SOURCE, "candidate_id": "one", "version": 1, "export_sha256": inputs.sha256(payload)}
    original = json.dumps(manifest, indent=4).encode("utf-8-sig")
    (tmp_path / inputs.MANIFEST).write_bytes(original)
    assert handoff.import_export(tmp_path, "one", payload)["status"] == "complete"
    assert (tmp_path / inputs.MANIFEST).read_bytes() == original


def test_nomination_change_during_publication_rolls_back_the_pair(tmp_path, monkeypatch):
    table(tmp_path)
    link = handoff.os.link

    def edit_table_after_publish(source, destination):
        link(source, destination)
        if destination.name == inputs.MANIFEST:
            table(tmp_path, "one\tCC\ntwo\tCCO\n")

    monkeypatch.setattr(handoff.os, "link", edit_table_after_publish)
    with pytest.raises(inputs.ToxScanError, match="toxscan_nomination_changed"):
        handoff.import_export(tmp_path, "one", csv_bytes([author_shape("CCO"), author_shape("CC")]))
    assert not (tmp_path / inputs.MANIFEST).exists() and not (tmp_path / inputs.SOURCE_FILE).exists()
    assert inputs.candidate_structure(tmp_path, "one") == "CC"


def test_concurrent_nominations_can_publish_only_one_consistent_pair(tmp_path):
    table(tmp_path)
    payload = csv_bytes([author_shape("CCO"), author_shape("CC")])
    barrier = Barrier(2)

    def import_for(candidate):
        barrier.wait(timeout=10)
        try:
            return handoff.import_export(tmp_path, candidate, payload)["candidate_id"]
        except inputs.ToxScanError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(import_for, ("one", "two")))
    winner = toxscan.screen_run(tmp_path)["candidate_id"]
    assert winner in ("one", "two") and set(outcomes) == {winner, "toxscan_import_conflict"}
    assert handoff.import_export(tmp_path, winner, payload)["candidate_id"] == winner


def test_existing_symlink_is_never_followed_on_import(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    table(run)
    payload = csv_bytes([author_shape("CCO")])
    elsewhere = tmp_path / "elsewhere.csv"
    elsewhere.write_bytes(payload)
    try:
        (run / inputs.SOURCE_FILE).symlink_to(elsewhere)
    except OSError:
        pytest.skip("requires symlinks")
    with pytest.raises(inputs.ToxScanError, match="toxscan_import_conflict"):
        handoff.import_export(run, "one", payload)
    assert elsewhere.read_bytes() == payload and not (run / inputs.MANIFEST).exists()
