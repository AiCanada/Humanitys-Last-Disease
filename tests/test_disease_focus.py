"""Pathway, mechanism, and optional target before the structure list."""

from rfproteina.disease.cli import main as cure_main
from rfproteina.disease.focus import (
    DISTINCTION,
    BiologicalFocus,
    display_rank,
    focus_from_fields,
    focus_summary,
    row_role,
    search_additions,
)
from rfproteina.disease.lookup import Disease
from rfproteina.disease.sources import (
    GENE_ATTRIBUTE,
    OPENTARGETS_URL,
    RCSB_SEARCH_URL,
    UNIPROT_ATTRIBUTE,
    collect_structure_options,
    pathway_uniprot_members,
    query_parts,
    reactome_pathway_list,
)
from rfproteina.disease.structures import StructureOption
from rfproteina.disease.ui import OptionForm, render_confirm, render_focus, render_options, render_structures


def _row(accession: str, entry: str, chain: str = "A", group: str = "protein") -> StructureOption:
    return StructureOption(
        group=group,
        accession=accession,
        entry_id=entry,
        chain=chain,
        title=accession,
        predicted=False,
        source="fixture",
    )


def _focus(**kwargs: object) -> BiologicalFocus:
    base = BiologicalFocus(
        disease_accessions=("P11111",),
        disease_symbols=("DIS",),
        pathway_accessions=("P22222",),
        interaction_symbols=("PART",),
    )
    return BiologicalFocus(**{**base.__dict__, **kwargs})


def test_the_landing_input_is_a_disease_or_biological_condition():
    page = render_options(OptionForm())
    assert "Disease or Biological Condition" in page
    assert page.index('name="disease"') < page.index('name="target_structure"')
    assert "<h2>Disease</h2>" in page
    assert "Scientific or medical name." in page


def test_the_confirm_page_still_asks_to_use_this_disease():
    page = render_confirm(Disease("Asthma", "EFO_0000270"))
    assert "<h2>Confirm the disease</h2>" in page
    assert "Use this disease" in page
    assert "This list is used to help confirm Disease" in page
    assert 'name="pathway"' not in page
    assert 'name="mechanism"' not in page


def test_the_focus_page_asks_for_pathway_mechanism_and_an_optional_target():
    page = render_focus(
        Disease("Asthma", "EFO_0000270"),
        [("R-HSA-177929", "EGFR signaling")],
        [("IL13", "P35225")],
    )
    assert "<h2>Relevant biology</h2>" in page
    assert "Relevant biological pathway" in page
    assert "Potential therapeutic mechanism" in page
    assert "Optional target protein or gene" in page
    assert 'name="pathway"' in page
    assert 'name="mechanism"' in page
    assert 'name="target_menu"' in page
    assert 'name="target_name"' in page
    assert "Any pathway" in page
    assert "EGFR signaling" in page
    assert "Protein associated with the disease" in page
    assert "Protein in the selected pathway" in page
    assert "Protein that interacts with a disease-associated protein" in page
    assert "IL13 (P35225)" in page
    assert "not necessarily equivalent" in page
    assert "initiating the disease" in page
    assert "does not decide which protein starts the disease" in page
    chosen = focus_from_fields(
        {
            "pathway": ["R-HSA-177929"],
            "mechanism": ["pathway"],
            "target_menu": ["IL13|P35225"],
            "target_name": ["EGFR"],
        },
        [("R-HSA-177929", "EGFR signaling")],
    )
    assert chosen.pathway_id == "R-HSA-177929"
    assert chosen.pathway_name == "EGFR signaling"
    assert chosen.mechanism == "pathway"
    assert chosen.target == "EGFR"
    blank = focus_from_fields({}, [("R-HSA-177929", "EGFR signaling")])
    assert blank.pathway_id == "any"
    assert blank.mechanism == "any"
    assert blank.target == ""
    refused = focus_from_fields({"pathway": ["R-HSA-999"], "mechanism": ["cure"]}, [("R-HSA-177929", "EGFR signaling")])
    assert refused.pathway_id == "any"
    assert refused.mechanism == "any"


def test_a_typed_target_then_the_chosen_mechanism_then_the_other_roles():
    rows = [
        _row("P11111", "1DIS"),
        _row("P22222", "2PTH"),
        _row("PART", "3INT", group="gene"),
        _row("P99999", "4NAM"),
    ]
    focus = _focus(
        pathway_id="R-HSA-1",
        pathway_name="Chosen pathway",
        mechanism="pathway",
        target="P99999",
        target_accession="P99999",
    )
    order = [number for number, _row in display_rank(rows, focus)]
    assert order == [4, 2, 1, 3]
    additions = search_additions(focus)
    assert additions[0][0] == "P99999"
    assert "P22222" in additions[0]
    assert additions[1] == []
    interacting = _focus(mechanism="interaction", target="")
    assert search_additions(interacting)[1] == ["PART"]
    assert "Named target" in focus_summary(focus) or "Target: P99999" in focus_summary(focus)
    assert DISTINCTION in focus_summary(focus)
    # A chosen pathway is the preference when the mechanism is left on Any.
    # An explicit disease-association mechanism still puts that set first.
    chosen_pathway = _focus(pathway_id="R-HSA-1", pathway_name="Chosen pathway", mechanism="any", target="")
    assert [number for number, _item in display_rank(rows, chosen_pathway)] == [2, 1, 3, 4]
    assert row_role(rows[1], chosen_pathway) == "Pathway member"
    assert row_role(rows[0], chosen_pathway) == "Disease-associated"
    disease_first = _focus(pathway_id="R-HSA-1", mechanism="disease_associated", target="")
    assert [number for number, _item in display_rank(rows, disease_first)][0] == 1


def test_the_structure_page_ranks_without_moving_the_checkbox_numbers():
    rows = [_row("P11111", "1DIS"), _row("P99999", "4NAM")]
    focus = _focus(mechanism="disease_associated", target="P99999", target_accession="P99999")
    page = render_structures(rows, focus=focus)
    assert page.index('name="pick" value="2"') < page.index('name="pick" value="1"')
    assert "Named target" in page
    assert "Disease-associated" in page
    assert "not necessarily equivalent" in page
    assert "initiating the disease" in page
    assert 'value="biological" selected' in page
    assert 'data-rank="400"' in page
    ranking = [{"number": 1, "value": 90, "ratings": {"grok grok-4": 90}}]
    ranked = render_structures(rows, ranking=ranking, focus=focus)
    assert ranked.index('name="pick" value="1"') < ranked.index('name="pick" value="2"')
    assert "Named target" in ranked
    assert 'value="biological" selected' not in ranked


def test_query_parts_read_a_menu_value_and_a_typed_name():
    assert query_parts("IL13|P35225") == ("P35225", "IL13")
    assert query_parts("p35225") == ("P35225", "")
    assert query_parts("IL13") == ("", "IL13")
    assert query_parts("") == ("", "")
    assert query_parts("1ABC") == ("", "")


def test_the_pathway_menu_keeps_every_pathway_and_a_chosen_one_is_searched():
    def read_value(url: str) -> object:
        if url.endswith("/pathways"):
            return [
                {"stId": "R-HSA-1", "displayName": "First pathway"},
                {"stId": "R-HSA-2", "displayName": "Second pathway"},
            ]
        if "participants/R-HSA-2" in url:
            return [{"databaseName": "UniProt", "identifier": "Q9Y6K9", "schemaClass": "ReferenceGeneProduct"}]
        raise AssertionError(url)

    assert reactome_pathway_list(read_value, ["P00533"]) == [
        ("R-HSA-1", "First pathway"),
        ("R-HSA-2", "Second pathway"),
    ]
    assert pathway_uniprot_members(read_value, "R-HSA-2", 5) == ["Q9Y6K9"]
    assert pathway_uniprot_members(read_value, "not-a-pathway", 5) == []


def test_extra_accessions_are_searched_after_the_association_cap():
    seen: list[str] = []

    def post(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL:
            rows = [
                {
                    "target": {
                        "approvedSymbol": f"G{index}",
                        "proteinIds": [{"id": f"P0000{index}", "source": "uniprot"}],
                    }
                }
                for index in range(4)
            ]
            return {"data": {"disease": {"associatedTargets": {"rows": rows}}}}
        value = payload["query"]["parameters"]["value"]
        attribute = payload["query"]["parameters"]["attribute"]
        assert attribute in {UNIPROT_ATTRIBUTE, GENE_ATTRIBUTE}
        seen.append(value)
        return {"result_set": []}

    found = collect_structure_options(
        "EFO_1",
        post=post,
        get=lambda _url: {},
        association_limit=1,
        row_limit=1,
        extra_accessions=["Q9Y6K9"],
        extra_symbols=["EGFR"],
    )
    assert found == []
    assert seen[0] == "P00000"
    assert "Q9Y6K9" in seen
    assert "EGFR" in seen
    assert seen.index("P00000") < seen.index("Q9Y6K9") < seen.index("G0")


def test_the_panel_ranks_a_chosen_pathway_before_the_disease_proteins(tmp_path, monkeypatch, capsys):
    """The pathway chosen on the new page is searched and listed ahead of the disease proteins."""

    def fail_open(*_args, **_kwargs):
        raise AssertionError("a live call was made")

    monkeypatch.setattr("rfproteina.disease.sources.urlopen", fail_open)
    monkeypatch.setattr("rfproteina.disease.research.urlopen", fail_open)
    monkeypatch.setattr("subprocess.run", fail_open)
    monkeypatch.setattr("subprocess.Popen", fail_open)
    catalog = tmp_path / "catalog.json"
    catalog.write_text('[{"name": "Asthma", "disease_id": "EFO_0000270"}]', encoding="utf-8")
    out = tmp_path / "ranked"
    searched: list[str] = []

    def post(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL:
            return {
                "data": {
                    "disease": {
                        "associatedTargets": {
                            "rows": [
                                {
                                    "target": {
                                        "approvedSymbol": "DIS",
                                        "proteinIds": [{"id": "P11111", "source": "uniprot"}],
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        assert url == RCSB_SEARCH_URL
        value = payload["query"]["parameters"]["value"]
        searched.append(value)
        entry = {"P11111": "1DIS_A", "Q9Y6K9": "2PTH_A", "DIS": "3GEN_A"}.get(value)
        if entry is None:
            return {"result_set": []}
        return {"result_set": [{"identifier": entry}]}

    def read_value(url: str) -> object:
        if url.endswith("/pathways"):
            return [
                {"stId": "R-HSA-1", "displayName": "First pathway"},
                {"stId": "R-HSA-9", "displayName": "Chosen pathway"},
            ]
        if url.endswith("/participants/R-HSA-9"):
            return [{"databaseName": "UniProt", "identifier": "Q9Y6K9", "schemaClass": "ReferenceGeneProduct"}]
        if url.endswith("/participants/R-HSA-1"):
            return [{"databaseName": "UniProt", "identifier": "P11111", "schemaClass": "ReferenceGeneProduct"}]
        raise AssertionError(url)

    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return {
                "disease": ["Asthma"],
                "catalog": [str(catalog)],
                "binder_kind": ["protein"],
                "out_dir": [str(out)],
                "where": ["local"],
            }
        if "<h2>Confirm the disease</h2>" in page:
            return {"accept": ["yes"]}
        if 'name="pathway"' in page and 'name="mechanism"' in page:
            assert "Chosen pathway" in page
            assert "Potential therapeutic mechanism" in page
            assert "Optional target protein or gene" in page
            return {"pathway": ["R-HSA-9"], "mechanism": ["any"]}
        if 'name="pick"' in page:
            assert page.index('name="pick" value="2"') < page.index('name="pick" value="1"')
            assert "Pathway member" in page
            assert "Disease-associated" in page
            assert "Chosen pathway" in page
            assert "not necessarily equivalent" in page
            assert 'value="biological" selected' in page
            return {"pick": ["2"]}
        assert '<p class="step-title">Confirm inference</p>' in page
        return {}

    code = cure_main(
        ["--ui"],
        post=post,
        get=lambda _url: {},
        read_value=read_value,
        fetch=lambda _option: b"data_2PTH\n#\n",
        read_json=lambda _url: {},
        exchange=exchange,
        read_line=lambda prompt: (_ for _ in ()).throw(AssertionError(prompt)),
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert len(pages) == 5, "landing, confirm, biological target, ranked structures, inference"
    assert "Biological focus: pathway Chosen pathway, mechanism any, target none." in captured.out
    assert "Inference was not confirmed. Generation was not started." in captured.out
    assert searched.index("P11111") < searched.index("Q9Y6K9")
