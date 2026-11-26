"""Local browser handoff from a nominated molecule to its verified author-platform result."""

import re
from html import unescape

import pytest

from rfproteina.disease import logs, screen_panel, screen_selection, toxscan, ui, ui_toxscan
from tests.helpers.disease import finished_run
from tests.test_disease_toxscan import author_shape, csv_bytes

pytest.importorskip("rdkit")


def hidden(page, name):
    match = re.search(rf'name="{re.escape(name)}" value="([^"]+)"', page)
    assert match is not None
    return unescape(match.group(1))


@pytest.fixture
def picker(tmp_path, monkeypatch):
    (tmp_path / "generation.json").write_text('{"binder_kind":"ligand"}')
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\none\tOCC\ntwo\tCCN\n")
    monkeypatch.setattr(screen_selection, "installations", lambda keys=None: {
        key: key in {"toxscan", "ligand_alerts"} for key in (screen_selection.METHOD_IDS if keys is None else keys)
    })
    session = ui.BrowserSession(opener=lambda _: pytest.fail("Preparing a request must stay local"))
    session.note_validation_runs((tmp_path,))
    keys = {c.example_id: c.key for c in screen_panel.candidates_for(tmp_path)}
    return session, keys


def select(session, key):
    message = session.run_results_action(ui.VALIDATION_SCOPE_ACTION, [f"{ui.VALIDATION_CANDIDATE_FIELD}={key}"])
    assert "refreshed" in message


def preparation(session, key):
    page = session.done_page()
    assert ui_toxscan.PREPARE in page and ui_toxscan.IMPORT not in page
    fields = {ui.SCREENING_SCOPE_FIELD: [hidden(page, ui.SCREENING_SCOPE_FIELD)],
              ui.VALIDATION_CANDIDATE_FIELD: [key], ui.SCREENING_FIELD: ["ligand_alerts"]}
    assert "ready to copy" in session.run_results_action(ui_toxscan.PREPARE, ui.flat_fields(fields))
    prepared = session.done_page()
    assert f'href="{toxscan.SOURCE}"' in prepared and f'name="{ui_toxscan.FILE}"' in prepared
    assert 'readonly spellcheck="false" rows="2">CCO</textarea>' in prepared
    fields[ui_toxscan.TOKEN] = [hidden(prepared, ui_toxscan.TOKEN)]
    return fields


def upload(data, filename, uploads):
    content_type = "multipart/form-data; boundary=toxscan-browser-test"
    head = (f'--toxscan-browser-test\r\nContent-Disposition: form-data; name="{ui_toxscan.FILE}"; '
            f'filename="{filename}"\r\nContent-Type: text/csv\r\n\r\n').encode("ascii")
    return ui.parse_submission(head + data + b"\r\n--toxscan-browser-test--\r\n", content_type, uploads)


def test_local_multipart_handoff_preserves_bytes_selection_and_result(picker, tmp_path):
    session, keys = picker
    page = session.done_page()
    assert "Choose one named ligand" in page and ui_toxscan.PREPARE not in page
    select(session, keys["one"])
    fields = preparation(session, keys["one"])
    filename = "private-download-name.csv"
    # A file with this name exists locally but the uploader must use the posted bytes instead.
    (tmp_path / filename).write_bytes(b"private unrelated content")
    payload = b"\xef\xbb\xbf" + csv_bytes([author_shape("CCN"), author_shape("CCO")])
    assert b"\r\n" in payload
    uploads = tmp_path / "uploads"
    fields.update(upload(payload, filename, uploads))
    assert not uploads.exists() and ui_toxscan.decode_upload(fields[ui_toxscan.FILE]) == payload
    redacted = logs.redact_fields(fields)
    assert redacted[ui_toxscan.FILE] == redacted[ui_toxscan.TOKEN] == "[redacted]"
    assert fields[ui_toxscan.FILE][0] not in str(redacted) and fields[ui_toxscan.TOKEN][0] not in str(redacted)

    assert "matched the nominated structure" in session.run_results_action(ui_toxscan.IMPORT, ui.flat_fields(fields))
    assert (tmp_path / "toxscan.csv").read_bytes() == payload
    assert (tmp_path / filename).read_bytes() == b"private unrelated content"
    page = session.done_page()
    assert 'value="ligand_alerts" data-recommended checked' in page
    assert 'value="toxscan" data-recommended checked' not in page
    assert fields[ui_toxscan.TOKEN][0] not in page and fields[ui_toxscan.FILE][0] not in page
    assert "Returned 1 screening result" in session.run_results_action(
        ui.SCREENING_ACTION, [f"{ui.VALIDATION_CANDIDATE_FIELD}={keys['one']}", f"{ui.SCREENING_FIELD}=toxscan"]
    )
    reviewed, = session._screen_method_results
    assert reviewed["status"] == "complete" and reviewed["measurement"]["candidate_id"] == "one"
    assert reviewed["measurement"]["source_sha256"] == toxscan.screen_run(tmp_path, "one")["source_sha256"]


def test_wrong_token_and_stale_candidate_scope_never_attach_results(picker, tmp_path):
    session, keys = picker
    select(session, keys["one"])
    fields = preparation(session, keys["one"])
    fields.update(upload(csv_bytes([author_shape("OCC"), author_shape("CCN")]), "data.csv", None))
    wrong = {**fields, ui_toxscan.TOKEN: ["unrelated"]}
    message = session.run_results_action(ui_toxscan.IMPORT, ui.flat_fields(wrong))
    assert "prepare" in message.lower() and not (tmp_path / toxscan.MANIFEST).exists()
    select(session, keys["two"])
    assert "scope changed" in session.run_results_action(ui_toxscan.IMPORT, ui.flat_fields(fields))
    assert not (tmp_path / toxscan.SOURCE_FILE).exists() and not (tmp_path / toxscan.MANIFEST).exists()


def test_reused_id_with_another_graph_rejects_original_request(picker, tmp_path):
    session, keys = picker
    select(session, keys["one"])
    fields = preparation(session, keys["one"])
    fields.update(upload(csv_bytes([author_shape("OCC"), author_shape("CCN")]), "data.csv", None))
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\none\tCCN\n")
    with pytest.raises(ValueError, match="candidate"):
        session.run_results_action(ui_toxscan.IMPORT, ui.flat_fields(fields))
    assert not (tmp_path / toxscan.SOURCE_FILE).exists() and not (tmp_path / toxscan.MANIFEST).exists()


def test_text_field_or_duplicate_upload_cannot_supply_a_local_path(tmp_path):
    source = tmp_path / "data.csv"
    source.write_bytes(b"private")
    for parts in ([str(source)], [ui_toxscan.encode_upload(b"a"), ui_toxscan.encode_upload(b"b")], [""]):
        with pytest.raises(ValueError, match="Choose"):
            ui_toxscan.decode_upload(parts)


def test_structure_backed_selector_stays_bound_to_the_graph_prepared_for_submission(picker, tmp_path):
    session, _ = picker
    run = finished_run(tmp_path, "structured_ligand")
    (run / "generation.json").write_text('{"binder_kind":"ligand"}')
    candidate, = screen_panel.candidates_for(run)
    assert candidate.structure is not None and candidate.example_id
    table = run / "ligand_candidates.tsv"
    table.write_text(f"candidate_id\tsmiles\n{candidate.example_id}\tCCO\n")
    session.note_validation_runs((run,))
    select(session, candidate.key)
    fields = preparation(session, candidate.key)
    table.write_text(f"candidate_id\tsmiles\n{candidate.example_id}\tCCN\n")
    current, = screen_panel.candidates_for(run)
    assert current.key == candidate.key
    fields.update(upload(csv_bytes([author_shape("CCO"), author_shape("CCN")]), "returned.csv", None))
    message = session.run_results_action(ui_toxscan.IMPORT, ui.flat_fields(fields))
    assert "differs from the prepared ToxScan request" in message
    assert not (run / toxscan.MANIFEST).exists() and not (run / toxscan.SOURCE_FILE).exists()
