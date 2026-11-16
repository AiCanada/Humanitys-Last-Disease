"""Co-evolution notes stored beside the paralog screen.

The four notes are the words in the table. A number is not classified as high
or low. The paralog state stays the UniRef50 result.
"""

import json
from pathlib import Path

import pytest

from rfproteina.disease import coevolution
from rfproteina.disease.coevolution import COEVOLUTION_SENTENCES
from rfproteina.disease.screen import format_screen_lines, screen_generation
from rfproteina.disease.screen_board import render_closing
from rfproteina.disease.target_liability import measure_target_liability
from tests.helpers.disease import new_folder

_IL5 = "P05113"
_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"


def _table(folder: Path, text: str, *, name: str = "paralog_coevolution.tsv") -> None:
    (folder / name).write_text(text, encoding="utf-8")


def _paralog(record: dict[str, object]) -> dict[str, object]:
    target = record["target_liability"]
    assert isinstance(target, dict)
    item = target["paralog"]
    assert isinstance(item, dict)
    return item


def _reader(*members: tuple[str, str]):
    def read(url: str) -> object:
        if "/uniref/search" in url:
            return {"results": [{"id": "UniRef50_P05113"}]}
        if "/uniprotkb/search" in url:
            return {
                "results": [
                    {"primaryAccession": accession, "genes": [{"geneName": {"value": symbol}}]}
                    for accession, symbol in members
                ]
            }
        return []

    return read


@pytest.mark.parametrize(
    ("level", "toxicity", "signal"),
    [
        ("high", "toxic", "most_likely_toxic"),
        ("low", "toxic", "lowest_signal_toxic"),
        ("high", "non-toxic", "probably_non_toxic"),
        ("low", "non-toxic", "lowest_signal_non_toxic"),
    ],
)
def test_the_four_transfer_notes_leave_the_paralog_state(
    tmp_path: Path, level: str, toxicity: str, signal: str
) -> None:
    output = new_folder(tmp_path, signal)
    _table(output, f"same_shape\tco-evolution\ttoxicity\nyes\t{level}\t{toxicity}\n")
    paralog = measure_target_liability(_IL5, directory=output)["paralog"]
    assert paralog["state"] == "untested"
    assert paralog["blocker"] == "network_not_permitted"
    assert paralog["coevolution_signal"] == signal
    assert paralog["coevolution_level"] == ("high" if level == "high" else "low")
    assert paralog["other_species_toxicity"] == ("toxic" if toxicity == "toxic" else "non_toxic")
    assert paralog["same_shape"] is True
    assert "cure" not in COEVOLUTION_SENTENCES[signal].casefold()


def test_a_number_is_not_classified_as_high_or_low(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "number")
    _table(output, "same_shape\tcoevolution\ttoxicity\nyes\t0.9\ttoxic\n")
    paralog = measure_target_liability(_IL5, directory=output)["paralog"]
    assert paralog["blocker"] == "network_not_permitted"
    assert paralog["coevolution_blocker"] == "coevolution_not_classified"
    assert "coevolution_signal" not in paralog


def test_no_same_shape_and_a_blank_token_leave_the_note_unset(tmp_path: Path) -> None:
    absent = new_folder(tmp_path, "absent")
    _table(absent, "same_shape\tco_evolution\ttoxicity\nno\thigh\ttoxic\n")
    paralog = measure_target_liability(_IL5, directory=absent)["paralog"]
    assert paralog["coevolution_blocker"] == "same_shape_absent"
    assert paralog["same_shape"] is False
    assert "coevolution_signal" not in paralog

    blank = new_folder(tmp_path, "blank")
    _table(blank, "same_shape\tco_evolution\ttoxicity\nn.a.\tinconclusive\t\n")
    assert measure_target_liability(_IL5, directory=blank)["paralog"]["coevolution_blocker"] == (
        "coevolution_not_reported"
    )


def test_a_bad_row_is_not_rescued_by_a_neighbouring_note(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "mixed")
    _table(
        output,
        "same_shape\tco_evolution\ttoxicity\nyes\thigh\ttoxic\nyes\t0.2\ttoxic\n",
    )
    paralog = measure_target_liability(_IL5, directory=output)["paralog"]
    assert paralog["coevolution_blocker"] == "coevolution_not_classified"
    assert "coevolution_signal" not in paralog


def test_disagreeing_rows_stay_unset_and_agreeing_rows_count(tmp_path: Path) -> None:
    clash = new_folder(tmp_path, "clash")
    _table(
        clash,
        "same_shape\tco_evolution\ttoxicity\tspecies\n" "yes\thigh\ttoxic\tmouse\n" "yes\tlow\tnon-toxic\tzebrafish\n",
    )
    paralog = measure_target_liability(_IL5, directory=clash)["paralog"]
    assert paralog["coevolution_blocker"] == "coevolution_disagreement"
    assert "coevolution_signal" not in paralog
    assert paralog["coevolution_signals"] == ["lowest_signal_non_toxic", "most_likely_toxic"]
    assert paralog["coevolution_species"] == ["mouse", "zebrafish"]

    agreed = new_folder(tmp_path, "agreed")
    _table(
        agreed,
        "shape\tcoevolution\tother_species_toxicity\torganism\tnotes\tnotes\n"
        "yes\thigh\ttoxic\tMus musculus\ta\tb\n"
        "true\thigh\ttoxic\tDanio rerio\t\t\n"
        f"no\tlow\ttoxic\t{_SEQUENCE}\t\t\n",
    )
    kept = measure_target_liability(_IL5, directory=agreed)["paralog"]
    assert kept["coevolution_signal"] == "most_likely_toxic"
    assert kept["coevolution_rows"] == 2
    assert kept["coevolution_species"] == ["Danio rerio", "Mus musculus"]
    assert _SEQUENCE not in json.dumps(kept)


def test_a_repeated_used_column_is_unreadable_and_a_missing_file_stays_quiet(tmp_path: Path) -> None:
    missing = measure_target_liability(_IL5, directory=new_folder(tmp_path, "missing"))["paralog"]
    assert missing["coevolution_blocker"] == "coevolution_not_configured"
    assert "coevolution_signal" not in missing

    doubled = new_folder(tmp_path, "doubled")
    _table(doubled, "same_shape\tsame_shape\tco_evolution\ttoxicity\nyes\tyes\thigh\ttoxic\n")
    assert measure_target_liability(_IL5, directory=doubled)["paralog"]["coevolution_blocker"] == (
        "coevolution_unreadable"
    )

    broken = new_folder(tmp_path, "broken")
    (broken / "paralog_coevolution.tsv").write_bytes(b"\xff\xfe")
    assert measure_target_liability(_IL5, directory=broken)["paralog"]["coevolution_blocker"] == (
        "coevolution_unreadable"
    )


def test_tsv_wins_and_a_banner_is_skipped(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "banner")
    _table(output, "same_shape,co_evolution,toxicity\nyes,low,toxic\n", name="paralog_coevolution.csv")
    _table(output, "# exported\n\nsame_shape\tco_evolution\ttoxicity\nyes\thigh\tnontoxic\n")
    paralog = measure_target_liability(_IL5, directory=output)["paralog"]
    assert paralog["coevolution_signal"] == "probably_non_toxic"
    assert paralog["coevolution_level"] == "high"


@pytest.mark.parametrize("kind", ["protein", "peptide"])
def test_the_note_is_on_the_screen_and_does_not_change_the_call(tmp_path: Path, kind: str) -> None:
    output = new_folder(tmp_path, kind)
    _table(output, "same_shape\tco_evolution\ttoxicity\tspecies\nyes\thigh\ttoxic\tMus musculus\n")
    record = screen_generation(output, binder_kind=kind, target_id=_IL5)
    paralog = _paralog(record)
    assert paralog["state"] == "untested"
    assert paralog["blocker"] == "network_not_permitted"
    assert paralog["coevolution_signal"] == "most_likely_toxic"
    assert record["call"] == "untested"
    assert record["hits"] == []
    text = json.dumps(record)
    assert str(output) not in text
    assert _SEQUENCE not in text
    lines = "\n".join(format_screen_lines("1ABC", "A", record))
    assert "paralog coevolution most_likely_toxic." in lines
    assert "most likely toxic in people." in lines
    page = render_closing(
        format_screen_lines("1ABC", "A", record),
        [{"entry_id": "1ABC", "chain": "A", "screen": record}],
    )
    assert "most likely toxic in people." in page
    assert "This note is stored beside the UniRef50 count." in page
    assert "Mus musculus" in page
    assert "cure" not in page.casefold()


def test_a_non_toxic_note_does_not_clear_a_paralog_hit(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "hit")
    _table(output, "same_shape\tco_evolution\ttoxicity\nyes\thigh\tnon-toxic\n")
    record = screen_generation(
        output,
        binder_kind="protein",
        target_id=_IL5,
        read_json=_reader((_IL5, "IL5"), ("P05112", "IL4")),
    )
    paralog = _paralog(record)
    assert paralog["state"] == "hit"
    assert paralog["coevolution_signal"] == "probably_non_toxic"
    assert record["call"] == "axis_hit"
    assert record["hits"] == ["paralog"]


def test_a_curated_paralog_call_keeps_the_note_beside_it(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "curated")
    _table(output, "same_shape\tco_evolution\ttoxicity\nyes\tlow\ttoxic\n")
    (output / "axis_evidence.json").write_text(json.dumps({"paralog": "clear"}), encoding="utf-8")
    record = screen_generation(output, binder_kind="protein", target_id=_IL5)
    paralog = _paralog(record)
    assert paralog["state"] == "clear"
    assert paralog["source"] == "axis_evidence"
    assert paralog["blocker"] is None
    assert paralog["coevolution_signal"] == "lowest_signal_toxic"
    assert record["hits"] == []


def test_a_missing_table_stays_out_of_the_measured_line(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "quiet")
    record = screen_generation(output, binder_kind="protein", target_id=_IL5)
    assert _paralog(record)["coevolution_blocker"] == "coevolution_not_configured"
    lines = "\n".join(format_screen_lines("1ABC", "A", record))
    assert "paralog coevolution" not in lines


def test_alias_columns_that_disagree_leave_the_note_unset(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "alias")
    _table(output, "same_shape\tcoevolution\tco_evolution\ttoxicity\nyes\thigh\tlow\ttoxic\n")
    paralog = measure_target_liability(_IL5, directory=output)["paralog"]
    assert paralog["state"] == "untested"
    assert paralog["blocker"] == "network_not_permitted"
    assert paralog["coevolution_blocker"] == "coevolution_unparsed"
    assert paralog["unparsed_columns"] == ["co_evolution", "coevolution"]
    assert "coevolution_signal" not in paralog
    page = render_closing(
        format_screen_lines("1ABC", "A", {"call": "untested", "target_liability": {"paralog": paralog}}),
        [{"entry_id": "1ABC", "chain": "A", "screen": {"call": "untested", "target_liability": {"paralog": paralog}}}],
    )
    assert "Columns that did not parse: co_evolution, coevolution." in page
    assert "cure" not in page.casefold()

    agreed = new_folder(tmp_path, "alias-agreed")
    _table(agreed, "same_shape\tshape\tcoevolution\tco_evolution\ttoxicity\nyes\tyes\thigh.\tHigh\ttoxic\n")
    kept = measure_target_liability(_IL5, directory=agreed)["paralog"]
    assert kept["coevolution_signal"] == "most_likely_toxic"
    assert kept["coevolution_level"] == "high"


def test_excel_padding_and_the_organism_column(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "excel")
    _table(output, "same_shape\tco_evolution\ttoxicity\tother_species\nyes\u00a0\thigh\tnon-toxic.\tmouse\n")
    paralog = measure_target_liability(_IL5, directory=output)["paralog"]
    assert paralog["coevolution_signal"] == "probably_non_toxic"
    assert paralog["coevolution_species"] == ["mouse"]
    assert paralog["other_species_toxicity"] == "non_toxic"

    spaced = new_folder(tmp_path, "spaced")
    _table(spaced, "same_shape\tco_evolution\ttoxicity\tspecies\nyes\thigh\ttoxic\tACDE FGHI KLMN\n")
    kept = measure_target_liability(_IL5, directory=spaced)["paralog"]
    assert kept["coevolution_signal"] == "most_likely_toxic"
    assert "coevolution_species" not in kept
    assert "ACDEFGHIKLMN" not in json.dumps(kept)

    clash = new_folder(tmp_path, "two-names")
    _table(clash, "species\torganism\tsame_shape\tco_evolution\ttoxicity\nmouse\trat\tyes\thigh\ttoxic\n")
    unnamed = measure_target_liability(_IL5, directory=clash)["paralog"]
    assert unnamed["coevolution_signal"] == "most_likely_toxic"
    assert "coevolution_species" not in unnamed


def test_a_table_past_either_reader_bound_is_withheld(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coevolution, "COEVOLUTION_ROW_LIMIT", 2)
    wide = new_folder(tmp_path, "wide")
    _table(wide, "same_shape\tco_evolution\ttoxicity\n" + "yes\thigh\ttoxic\n" * 3)
    paralog = measure_target_liability(_IL5, directory=wide)["paralog"]
    assert paralog["blocker"] == "network_not_permitted"
    assert paralog["coevolution_blocker"] == "coevolution_truncated"
    assert "coevolution_signal" not in paralog

    monkeypatch.setattr(coevolution, "COEVOLUTION_ROW_LIMIT", 2)
    inside = new_folder(tmp_path, "inside")
    _table(inside, "same_shape\tco_evolution\ttoxicity\n" + "yes\thigh\ttoxic\n" * 2)
    kept = measure_target_liability(_IL5, directory=inside)["paralog"]
    assert kept["coevolution_signal"] == "most_likely_toxic"
    assert kept["coevolution_rows"] == 2

    monkeypatch.setattr(coevolution, "COEVOLUTION_FILE_LIMIT", 8)
    huge = new_folder(tmp_path, "huge")
    _table(huge, "same_shape\tco_evolution\ttoxicity\nyes\thigh\ttoxic\n")
    blocked = measure_target_liability(_IL5, directory=huge)["paralog"]
    assert blocked["coevolution_blocker"] == "coevolution_too_large"
    assert "coevolution_signal" not in blocked


@pytest.mark.parametrize("kind", ["ligand", "small_molecule", "nucleic", "dna"])
def test_kinds_without_paralog_leave_the_table_unread(tmp_path: Path, kind: str) -> None:
    output = new_folder(tmp_path, kind)
    _table(output, "same_shape\tco_evolution\ttoxicity\nyes\thigh\ttoxic\n")
    record = screen_generation(output, binder_kind=kind, target_id=_IL5)
    assert "coevolution_signal" not in json.dumps(record)
    target = record.get("target_liability")
    if isinstance(target, dict):
        assert "paralog" not in target
