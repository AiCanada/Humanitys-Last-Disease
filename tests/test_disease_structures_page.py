"""Download with nothing ticked comes back to the structures page.

On the command line an empty selection ends the run, and the printed line says
which flag to pass instead. On the page there is no flag to pass, and the panel
simply closed: the run ended on the structures page having downloaded nothing,
with hundreds of structures listed and the only explanation on a terminal the
user was not looking at.

A ranking is the one case that still leaves the page, because the ranked
targets are offered next and the choice is made there instead.
"""

import json
from pathlib import Path

from rfproteina.disease.cli import main as cure_main
from tests.helpers.disease import ROW_P00533, asthma_catalog, block_live_calls, refuse_prompt

_PICK = 'type="checkbox" name="pick"'


def _two_rows(directory: Path) -> Path:
    """Two deposited structures, so the page has something to tick."""
    path = directory / "structures.json"
    path.write_text(
        json.dumps(
            [
                dict(ROW_P00533),
                {
                    "group": "protein",
                    "accession": "P35225",
                    "entry_id": "2IL13",
                    "chain": "B",
                    "title": "interleukin 13",
                    "predicted": False,
                    "source": "fixture",
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def _landing(catalog: Path, records: Path, out: Path) -> dict[str, list[str]]:
    return {
        "disease": ["Asthma"],
        "catalog": [str(catalog)],
        "accept_only": ["yes"],
        "structures_json": [str(records)],
        "binder_kind": ["protein"],
        "out_dir": [str(out)],
        "where": ["local"],
        "remote_dir": ["disease_runs"],
        "confirm_inference": ["yes"],
    }


def test_an_empty_download_shows_the_page_again_instead_of_closing(tmp_path, monkeypatch, capsys) -> None:
    """The reported bug: the panel closed, and said why only on the terminal."""
    block_live_calls(monkeypatch)
    catalog = asthma_catalog(tmp_path)
    records = _two_rows(tmp_path)
    out = tmp_path / "empty-download"
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _landing(catalog, records, out)
        if _PICK in page:
            shown = sum(1 for seen in pages if _PICK in seen)
            # Press Download with nothing ticked, exactly as the user did.
            return {"pick": ["1"]} if shown > 1 else {}
        return {}

    code = cure_main(
        ["--ui"],
        fetch=lambda _option: b"data_X\n#\n",
        read_json=lambda _url: {},
        exchange=exchange,
        read_line=refuse_prompt,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err

    structures = [page for page in pages if _PICK in page]
    assert len(structures) >= 2, "the empty download ended the run instead of asking again"
    assert "Tick a structure to download" in structures[1]
    assert "Tick a structure to download" not in structures[0], "the first page carries no complaint"
    # And the run goes on from the second answer rather than stopping there.
    assert "Selected 1 structure(s) for download." in captured.out


def test_a_flag_run_still_says_which_flag_to_pass(tmp_path, monkeypatch, capsys) -> None:
    """With no page there is nothing to come back to, so the printed line stands."""
    block_live_calls(monkeypatch)
    catalog = asthma_catalog(tmp_path)
    records = _two_rows(tmp_path)
    out = tmp_path / "flag-run"

    code = cure_main(
        [
            "--disease",
            "Asthma",
            "--catalog",
            str(catalog),
            "--accept-only",
            "--structures-json",
            str(records),
            "--binder-kind",
            "protein",
            "--out-dir",
            str(out),
        ],
        fetch=lambda _option: b"data_X\n#\n",
        read_json=lambda _url: {},
        read_line=refuse_prompt,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Pass --all, --pick, or --select-structure before download." in captured.out
