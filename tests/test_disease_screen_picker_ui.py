"""The browser preserves human choices and separates file checks from scientific screens."""

import json
import re
from html import escape

import pytest

from rfproteina.disease import immuscope_study, screen_panel, screen_selection, ui


def test_screen_picker_explains_each_recommendation_and_cost():
    def choice(key, *, installed=True, applicable=True, recommended=False):
        method = screen_selection.Method(key, key.title(), (), (), "CPU ensemble", "Prioritize an assay.")
        return screen_selection.Choice(method, installed, applicable, recommended, "Scope <reason>.")

    choices = (
        choice("missing", installed=False), choice("unsupported", applicable=False),
        choice("optional"), choice("ready", recommended=True),
    )
    html = ui._scientific_screen_panel(choices, ("optional", "missing"), [])
    groups = dict(re.findall(r"<fieldset><legend>([^<]+)</legend>(.*?)</fieldset>", html, re.DOTALL))
    assert list(groups) == [
        "Recommended (1)", "Other applicable screens (1)", "Installed but inapplicable (1)", "Unavailable (1)",
    ]
    assert (
        'value="ready" data-recommended aria-describedby="screen-reason-ready screen-performance-ready"'
        in groups["Recommended (1)"]
    )
    assert 'value="optional" checked' in groups["Other applicable screens (1)"]
    assert 'value="unsupported" aria-describedby=' in groups["Installed but inapplicable (1)"]
    assert 'value="missing" disabled aria-describedby=' in groups["Unavailable (1)"]
    assert 'id="screen-reason-ready">Prioritize an assay. Scope &lt;reason&gt;.' in html
    assert "CPU ensemble" in html
    assert "<span data-screening-count>1</span> of 3 installed screens selected" in html


def test_picker_shows_performance_and_sources_without_changing_selection(monkeypatch):
    methods = screen_selection.METHODS[:2]
    first, second = (method.key for method in methods)
    evidence = {
        first: {"status": "published_benchmark", "basis": "Comparison with <control> & reference",
                "summary": "Sensitivity < 0.80 on the held-out set.",
                "interpretation": "Compare <result> & controls.",
                "source": "https://example.org/paper?a=1&b=2"},
        second: {"status": "unknown", "summary": "No estimate for this endpoint.",
                 "interpretation": "Measure against a matched control."},
    }
    monkeypatch.setattr(screen_selection.Choice, "performance", property(lambda option: evidence[option.method.key]))
    choices = [
        screen_selection.Choice(method, True, True, i == 0, "Candidate reason.") for i, method in enumerate(methods)
    ]
    for before_first_run in (False, True):
        html = ui._scientific_screen_panel(choices, None, [], submit_button=not before_first_run)
        assert html.count("<strong>Evidence basis</strong>") == 2
        assert (f'id="screen-performance-{first}"><strong>Evidence basis</strong> '
                'Comparison with &lt;control&gt; &amp; reference<br>') in html
        assert (f'id="screen-performance-{second}"><strong>Evidence basis</strong> '
                'Unspecified in this record.<br>') in html
        assert html.count("<strong>Validation performance</strong>") == 2
        assert html.count("<strong>What this means</strong>") == 2
        assert "Compare &lt;result&gt; &amp; controls." in html and "Measure against a matched control." in html
        assert html.count("Validation source</a>") == 1
        assert 'href="https://example.org/paper?a=1&amp;b=2"' in html
        assert "Sensitivity &lt; 0.80 on the held-out set." in html and "No estimate for this endpoint." in html
        assert f'value="{first}" data-recommended checked' in html and f'value="{second}" checked' not in html


@pytest.mark.parametrize("kind,show_study", [
    ("polypeptide", True), ("peptide", True), ("small molecule", False), ("rna", False), ("", False),
])
def test_reference_is_scoped_and_separate_from_both_selection_forms(tmp_path, monkeypatch, kind, show_study):
    (tmp_path / "generation.json").write_text(json.dumps({"binder_kind": kind}))
    choices = tuple(
        screen_selection.Choice(method, True, True, method.key == "qupkake", "Experimental follow-up.")
        for method in screen_selection.METHODS if method.key in {"qupkake", "ligand_caco2"}
    )
    monkeypatch.setattr(screen_selection, "plan", lambda *_, **kwargs: choices)
    monkeypatch.setattr(ui, "load_key_slots", lambda: {})
    pages = []

    def submit(page):
        pages.append(page)
        return {ui.SCREENING_FIELD: ["ligand_caco2"]}

    panel = ui.OptionPanel(submit)
    assert panel.choose_screens(tmp_path, kind) == ("ligand_caco2",)
    session = ui.BrowserSession(opener=lambda _: None)
    session.note_validation_runs((tmp_path,))
    pages.append(session.done_page())
    study = immuscope_study.published_study()
    for page in pages:
        assert page.count(f'type="checkbox" name="{ui.SCREENING_FIELD}"') == 2
        assert '<span data-screening-count>1</span> of 2 installed screens selected' in page
        assert 'value="qupkake" data-recommended checked' in page and 'value="ligand_caco2" checked' not in page
        for choice in choices:
            assert escape(choice.performance["interpretation"]) in page
            assert escape(choice.performance["basis"]) in page
        assert "Near the planned assay pH" in page and "bidirectional Caco-2 assay" in page
        reference = re.search(r'<details class="results-tools" data-study-reference>(.*?)</details>', page, re.DOTALL)
        assert bool(reference) == show_study
        if reference:
            body = reference.group(1)
            assert "<input" not in body and f'href="{study["source"]}"' in body
            for endpoint in study["endpoints"].values():
                assert escape(endpoint["endpoint"]) in body
                for value in endpoint["metrics"].values():
                    numbers = value if isinstance(value, list) else [value]
                    assert all(str(number) in body for number in numbers)


def test_scientific_selection_survives_a_file_validation_post_including_clear(tmp_path, monkeypatch):
    choices = (
        screen_selection.Choice(screen_selection.METHODS[0], True, True, True, "Assay coverage."),
        screen_selection.Choice(screen_selection.METHODS[1], True, True, False, "Optional follow-up."),
    )
    monkeypatch.setattr(screen_panel, "choices_for", lambda *_: choices)
    session = ui.BrowserSession(opener=lambda _: None)
    session.note_validation_runs((tmp_path,))
    old = session.done_page()
    stamp = re.search(r'name="screening_scope_version" value="(\d+)"', old).group(1)
    token = f"{ui.SCREENING_SCOPE_FIELD}={stamp}"
    optional = choices[1].method.key

    session.run_results_action(ui.VALIDATION_ACTION, [token, f"{ui.SCREENING_FIELD}={optional}"])
    assert f'value="{optional}" checked' in session.done_page()
    session.run_results_action(ui.VALIDATION_ACTION, [token])
    assert '<span data-screening-count>0</span>' in session.done_page()

    session.note_validation_runs((tmp_path,))
    monkeypatch.setattr(screen_panel, "run", lambda *_a, **_k: pytest.fail("stale dispatch"))
    assert "scope changed" in session.run_results_action(ui.SCREENING_ACTION, [token, f"{ui.SCREENING_FIELD}={optional}"])
    assert '<span data-screening-count>1</span>' in session.done_page()


def test_result_shows_the_evidence_requested_at_execution_after_a_model_change(tmp_path, monkeypatch):
    method = screen_selection.METHODS[0]
    choice = screen_selection.Choice(method, True, True, False, "Optional follow-up.")
    monkeypatch.setattr(screen_panel, "choices_for", lambda *_: (choice,))
    live = {"basis": "Newer local evaluation", "summary": "Updated score", "interpretation": "Updated application"}
    monkeypatch.setattr(screen_selection.Choice, "performance", property(lambda _: live))
    original = {"basis": "Original <evaluation>", "summary": "Earlier error < 0.5",
                "interpretation": "Original conditions & controls", "source": "https://example.org/older?a=1&b=2"}

    def run(directory, candidate, mode, methods, *, on_line, include_performance):
        assert directory == tmp_path and candidate is None and mode == "explicit" and methods == (method.key,)
        assert callable(on_line) and include_performance is True
        return [{"id": method.key, "status": "reported", "measurement": {"reading": 10}, "performance": original}]

    monkeypatch.setattr(screen_panel, "run", run)
    session = ui.BrowserSession(opener=lambda _: None)
    session.note_validation_runs((tmp_path,))
    assert session.run_results_action(ui.SCREENING_ACTION, [f"{ui.SCREENING_FIELD}={method.key}"]) == (
        "Returned 1 screening result."
    )
    page = session.done_page()
    assert live["summary"] in page
    previous = re.search(r'<details><summary>reported</summary>(.*?)</details>', page, re.DOTALL).group(1)
    assert "Validation evidence at execution" in previous and "&quot;reading&quot;: 10" in previous
    assert all(escape(original[key]) in previous for key in ("basis", "summary", "interpretation", "source"))
    assert all(text not in previous for text in live.values())


def test_legacy_result_with_no_evidence_and_unusable_source_link():
    old = {"id": "old", "status": "inapplicable", "reason": "Missing <control>"}
    detail = ui._scientific_result_detail(old)
    assert "inapplicable" in detail and "Missing &lt;control&gt;" in detail
    assert "Validation evidence for this result is unavailable" in detail and "Evidence basis" not in detail
    old["performance"] = {"summary": "Reported.", "interpretation": "Check.", "source": "javascript:example()"}
    detail = ui._scientific_result_detail(old)
    assert "Validation evidence at execution" in detail and "Unspecified in this record." in detail
    assert "href=" not in detail and "javascript:" not in detail


def test_browser_offers_scientific_screening_for_a_ligand_without_generated_files(tmp_path, monkeypatch):
    (tmp_path / "generation.json").write_text('{"binder_kind":"ligand"}')
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\nethanol\tCCO\n")
    available = dict.fromkeys(screen_selection.METHOD_IDS, True)
    monkeypatch.setattr(screen_selection, "installations", lambda keys=None: available)
    session = ui.BrowserSession(opener=lambda _: None)
    session.note_validation_runs((tmp_path,))
    candidate, = screen_panel.candidates_for(tmp_path)
    assert candidate.structure is None
    assert f'value="{candidate.key}"' in session.done_page()
    fields = [f"{ui.VALIDATION_CANDIDATE_FIELD}={candidate.key}"]
    assert "refreshed" in session.run_results_action(ui.VALIDATION_SCOPE_ACTION, fields)
    page = session.done_page()
    assert "Output-file checks require generated files attributable to this candidate." in page
    assert 'formnovalidate disabled>Run selected checks' in page
    assert 'value="ligand_alerts" data-recommended checked' in page
    assert session.run_results_action(ui.SCREENING_ACTION, fields) == "No scientific screens selected."


def test_finished_screen_does_not_overwrite_a_new_run_scope(tmp_path, monkeypatch):
    session = ui.BrowserSession(opener=lambda _: None)
    session.note_validation_runs((tmp_path,))
    replacement = tmp_path / "replacement"
    replacement.mkdir()

    def finish_in_previous_scope(*_args, **_kwargs):
        session.note_validation_runs((replacement,))
        return [{"id": "ligand_alerts", "status": "reported", "measurement": {"old": True}}]

    monkeypatch.setattr(screen_panel, "run", finish_in_previous_scope)
    message = session.run_results_action(ui.SCREENING_ACTION, [f"{ui.SCREENING_FIELD}=ligand_alerts"])
    assert "previous scope" in message
    assert session._screen_method_results == [] and session._screen_method_selection is None
