"""A saved Ames score stays attached to its original nomination and checkpoint."""

import pytest

from rfproteina.disease import screen_panel, screen_selection, toxscan_ames, toxscan_ames_model, ui
from tests.helpers.disease import finished_run
from tests.test_disease_toxscan_local_ames import METHOD, PRIVATE_ID, SMILES
from tests.test_disease_toxscan_local_ames import local_fit as local_fit

pytestmark = pytest.mark.usefixtures("local_fit")


@pytest.fixture
def run_prediction(monkeypatch):
    monkeypatch.setattr(toxscan_ames_model, "installed_sha256", lambda: "8" * 64)
    monkeypatch.setattr(screen_selection, "installations", lambda keys=None: {
        key: key == METHOD for key in (screen_selection.METHOD_IDS if keys is None else keys)
    })

    def run(directory, key=None):
        browser = ui.BrowserSession(opener=lambda _: pytest.fail("The Ames picker must stay local"))
        browser.note_validation_runs((directory,))
        scope = [f"{ui.VALIDATION_CANDIDATE_FIELD}={key}"] if key else []
        if scope:
            assert "refreshed" in browser.run_results_action(ui.VALIDATION_SCOPE_ACTION, scope)
        message = browser.run_results_action(ui.SCREENING_ACTION, [*scope, f"{ui.SCREENING_FIELD}={METHOD}"])
        assert "Returned 1 screening result" in message
        result, = browser._screen_method_results
        assert result["status"] == "complete"
        monkeypatch.setattr(toxscan_ames_model, "predict", lambda _: pytest.fail("A redraw ran inference"))
        page = browser.done_page()
        assert "<summary>complete</summary>" in page and "Previous local Ames prediction" not in page
        return browser, result["measurement"]["result_sha256"]

    return run


def assert_historical(browser, digest):
    page = browser.done_page()
    assert "<summary>Previous local Ames prediction" in page and "<summary>complete</summary>" not in page
    assert digest in page and "predicted YES" in page and "Validation evidence at execution" in page


@pytest.mark.parametrize("change", ["graph", "candidate", "second_candidate", "missing_model", "different_model"])
def test_whole_run_rechecks_only_nomination_and_verified_weights(run_prediction, tmp_path, monkeypatch, change):
    browser, digest = run_prediction(tmp_path)
    if change in {"missing_model", "different_model"}:
        current = None if change == "missing_model" else "9" * 64
        monkeypatch.setattr(toxscan_ames_model, "installed_sha256", lambda: current)
    else:
        rows = {
            "graph": f"{PRIVATE_ID}\tCCN",
            "candidate": f"replacement\t{SMILES}",
            "second_candidate": f"{PRIVATE_ID}\t{SMILES}\nsecond\tCCN",
        }
        (tmp_path / toxscan_ames.TABLE).write_text(f"candidate_id\tsmiles\n{rows[change]}\n")
    assert_historical(browser, digest)


def test_structured_candidate_uses_selected_nomination_on_each_redraw(run_prediction, tmp_path):
    run = finished_run(tmp_path, "structures")
    (run / "generation.json").write_text('{"binder_kind":"ligand"}')
    candidate, = screen_panel.candidates_for(run)
    assert candidate.structure and candidate.example_id
    table = run / toxscan_ames.TABLE
    table.write_text(f"candidate_id\tsmiles\nother\tClCCl\n{candidate.example_id}\t{SMILES}\n")
    browser, digest = run_prediction(run, candidate.key)
    table.write_text(f"candidate_id\tsmiles\nother\t{SMILES}\n{candidate.example_id}\tCCN\n")
    assert candidate.key in {c.key for c in screen_panel.candidates_for(run)}
    assert_historical(browser, digest)


def test_disappearing_table_selector_keeps_the_historical_prediction(run_prediction, tmp_path):
    candidate, = screen_panel.candidates_for(tmp_path)
    browser, digest = run_prediction(tmp_path, candidate.key)
    (tmp_path / toxscan_ames.TABLE).write_text(f"candidate_id\tsmiles\n{PRIVATE_ID}\tCCN\n")
    assert candidate.key not in {c.key for c in screen_panel.candidates_for(tmp_path)}
    assert_historical(browser, digest)
