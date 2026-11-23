"""Supplied solubility and aggregation scores on the post-generation screen.

CamSol, Aggrescan3D, and Protein-Sol are not run here, so those numbers come
from a table beside the run. Every row names its method, because a solubility
score has no scale without one. Nothing here is judged and the screen's call
does not move. The sequence AGGRESCAN is a separate modality report.
"""

from pathlib import Path

import pytest

from rfproteina.disease.screen import format_screen_lines, screen_generation
from rfproteina.disease.screen_board import render_closing
from rfproteina.disease.solubility import SOLUBILITY_ROW_LIMIT, read_solubility, solubility_phrase
from tests.helpers.disease import new_folder

_HEAD = "score\tmethod\tsource\n"
_GOOD = _HEAD + "0.72\tCamSol intrinsic\tSormanni 2015\n"


def _table(folder: Path, text: str, *, name: str = "solubility.tsv") -> Path:
    (folder / name).write_text(text, encoding="utf-8")
    return folder


def _run(tmp_path: Path, text: str | None, *, name: str = "solubility.tsv", kind: str = "protein") -> dict:
    output = tmp_path / "protein_1ABC_1"
    sample = output / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text("data_binder\n#\n", encoding="utf-8")
    if text is not None:
        (output / name).write_text(text, encoding="utf-8")
    return screen_generation(output, binder_kind=kind, target_id="P05113")


def test_a_supplied_score_is_stored_with_its_method_and_study(tmp_path: Path) -> None:
    found = read_solubility(_table(new_folder(tmp_path, "run"), _GOOD))
    assert found["rows"] == 1
    assert found["scores"] == [{"score": 0.72, "method": "CamSol intrinsic", "source": "Sormanni 2015"}]
    assert "solubility_blocker" not in found


def test_several_tools_are_each_kept_in_file_order(tmp_path: Path) -> None:
    text = _HEAD + "0.72\tCamSol intrinsic\tSormanni 2015\n-1.8\tAggrescan3D a3d\tKuriata 2019\n"
    found = read_solubility(_table(new_folder(tmp_path, "run"), text))
    assert [item["method"] for item in found["scores"]] == ["CamSol intrinsic", "Aggrescan3D a3d"]
    assert [item["score"] for item in found["scores"]] == [0.72, -1.8]


def test_a_unit_and_a_note_are_kept_when_given(tmp_path: Path) -> None:
    text = "score\tmethod\tsource\tunit\tnote\n4.1\tProtein-Sol\tHebditch 2017\tscaled\tpH 7\n"
    stored = read_solubility(_table(new_folder(tmp_path, "run"), text))["scores"][0]
    assert stored["unit"] == "scaled"
    assert stored["note"] == "pH 7"


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("no method", "score\tsource\n0.72\tSormanni 2015\n"),
        ("no study", "score\tmethod\n0.72\tCamSol\n"),
    ],
)
def test_a_row_without_its_method_or_its_study_is_not_stored(tmp_path: Path, label: str, text: str) -> None:
    """A bare number carries no scale, and one with no study cannot be traced."""
    found = read_solubility(_table(new_folder(tmp_path, label), text))
    assert found["solubility_blocker"] == "solubility_not_reported"
    assert "scores" not in found


def test_a_dropped_row_is_counted_beside_the_ones_that_stood(tmp_path: Path) -> None:
    text = _HEAD + "0.72\tCamSol\tSormanni 2015\n0.5\t\t\n"
    found = read_solubility(_table(new_folder(tmp_path, "run"), text))
    assert found["rows"] == 1
    assert found["incomplete_rows"] == 1


def test_a_missing_table_says_so(tmp_path: Path) -> None:
    assert read_solubility(new_folder(tmp_path, "run"))["solubility_blocker"] == "solubility_not_configured"


def test_explicit_candidate_attribution_keeps_each_method_and_ignores_unrelated_rows(tmp_path):
    header = "candidate_id\tscore\tmethod\tsource\tunit\n"
    _table(tmp_path, header + "one\t0\tassay A\tstudy A\tmg/mL\none\t-2\tmodel B\tstudy B\tlogS\n"
           "two\tbad\tassay A\tstudy A\tmg/mL\n")
    report = read_solubility(tmp_path, candidate="one")
    assert [(r["candidate_id"], r["score"], r["unit"]) for r in report["scores"]] == [
        ("one", 0, "mg/mL"), ("one", -2, "logS")
    ]
    assert "one assay A 0 mg/mL" in solubility_phrase(report)
    assert read_solubility(tmp_path)["solubility_blocker"] == "cell_unparsed"
    assert read_solubility(tmp_path, candidate="other")["solubility_blocker"] == "solubility_candidate_missing"
    _table(tmp_path, _GOOD)
    assert read_solubility(tmp_path, candidate="one")["solubility_blocker"] == "solubility_run_only"


def test_duplicate_identity_headers_cannot_assign_an_observation(tmp_path):
    _table(tmp_path, "candidate_id\tcandidate id\tscore\tmethod\tsource\none\ttwo\t1\tmethod\tsource\n")
    assert read_solubility(tmp_path, candidate="two")["solubility_blocker"] == "solubility_unreadable"


@pytest.mark.parametrize("fields,values", [
    ("score,value", "0.000019,0.000011"), ("score,value", "0.7,bad"), ("method,tool", "CamSol,A3D"),
])
def test_conflicting_or_invalid_aliases_do_not_supply_a_result(tmp_path, fields, values):
    tail = "method,source\nCamSol,paper" if fields.startswith("score") else "score,source\n0.7,paper"
    headings, data = tail.split("\n")
    directory = _table(tmp_path, f"{fields},{headings}\n{values},{data}\n", name="solubility.csv")
    assert read_solubility(directory)["solubility_blocker"] == "cell_unparsed"


def test_small_nonzero_readings_and_last_line_quotes(tmp_path):
    directory = _table(tmp_path, _HEAD + "0.000019\tCamSol\tpaper\n")
    assert read_solubility(directory)["scores"][0]["score"] == 0.000019
    assert "0.0000" not in solubility_phrase(read_solubility(directory))
    _table(tmp_path, _HEAD + '0.7\tCamSol\t"unclosed')
    assert read_solubility(directory)["solubility_blocker"] == "solubility_unreadable"


def test_a_cell_that_is_not_a_number_names_its_column(tmp_path: Path) -> None:
    found = read_solubility(_table(new_folder(tmp_path, "run"), _HEAD + "high\tCamSol\tSormanni 2015\n"))
    assert found["solubility_blocker"] == "cell_unparsed"
    assert found["unparsed_columns"] == ["score"]


@pytest.mark.parametrize(
    ("label", "text", "name"),
    [
        ("a semicolon export", "score;method;source\n0.72;CamSol;S 2015\n", "solubility.csv"),
        ("a row with a surplus field", "score,method,source\n0.72,CamSol, intrinsic,S 2015\n", "solubility.csv"),
        ("a repeated score column", "score\tscore\tmethod\tsource\n0.1\t0.9\tCamSol\tS 2015\n", "solubility.tsv"),
        ("an unterminated quote", 'score\tmethod\tsource\n0.7\tCamSol\t"S\n0.9\tAggrescan\tK 2019\n', "solubility.tsv"),
    ],
)
def test_a_table_that_cannot_be_read_in_full_is_refused_in_full(
    tmp_path: Path, label: str, text: str, name: str
) -> None:
    """Each of these moves a value under another column's name."""
    found = read_solubility(_table(new_folder(tmp_path, label.replace(" ", "_")), text, name=name))
    assert found["solubility_blocker"] == "solubility_unreadable"
    assert "scores" not in found


def test_a_table_longer_than_the_bound_is_withheld_whole(tmp_path: Path) -> None:
    rows = "".join(f"{index / 100}\tCamSol\tS 2015\n" for index in range(SOLUBILITY_ROW_LIMIT + 1))
    found = read_solubility(_table(new_folder(tmp_path, "run"), _HEAD + rows))
    assert found["solubility_blocker"] == "solubility_truncated"
    assert "scores" not in found


def test_a_banner_above_the_header_is_skipped(tmp_path: Path) -> None:
    text = "# exported 2026-11-12\n" + _GOOD
    assert read_solubility(_table(new_folder(tmp_path, "run"), text))["rows"] == 1


def test_the_screen_reports_the_score_and_the_call_does_not_move(tmp_path: Path) -> None:
    record = _run(tmp_path, _GOOD)
    assert record["solubility"]["scores"][0]["method"] == "CamSol intrinsic"
    assert record["call"] == "untested", "a report is not an axis"
    assert "solubility" not in record["states"]
    assert "solubility supplied: CamSol intrinsic 0.72" in format_screen_lines("1ABC", "1", record)[1]


def test_a_run_with_no_table_still_records_why(tmp_path: Path) -> None:
    record = _run(tmp_path, None)
    assert record["solubility"]["solubility_blocker"] == "solubility_not_configured"
    assert "solubility supplied" not in format_screen_lines("1ABC", "1", record)[1]


def test_a_ligand_run_reads_the_table_too(tmp_path: Path) -> None:
    """Aggregation is not a protein-only question."""
    record = _run(tmp_path, _GOOD, kind="ligand")
    assert record["solubility"]["rows"] == 1


def test_the_closed_page_explains_a_missing_table(tmp_path: Path) -> None:
    record = _run(tmp_path, None)
    page = render_closing([], [{"entry_id": "1ABC", "chain": "1", "screen": record}])
    assert "No solubility table was found in the output folder." in page
    assert "Add solubility.tsv to the output folder." in page


def test_the_phrase_is_none_without_a_score() -> None:
    assert solubility_phrase({"solubility_blocker": "solubility_not_configured"}) is None
    assert solubility_phrase({"scores": []}) is None
