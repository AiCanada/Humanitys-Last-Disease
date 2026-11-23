"""Ligand safety-panel readings on the post-generation screen.

The cut is the existing 10_000 nM panel rule. A missing table stays untested.
Protein screens do not take panel axes from the table.
"""

import json
from pathlib import Path

import pytest

from rfproteina.disease.screen import format_screen_lines, screen_generation
from rfproteina.disease.screen_board import render_closing
from rfproteina.disease.toxicology import measure_toxicology
from tests.helpers.disease import new_folder

_CLEAR = "axis\tnm\nherg\t20000\npgp\t20000\nbcrp\t20000\n"
_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" * 2


def _table(folder: Path, text: str, *, name: str = "toxicology.tsv") -> None:
    (folder / name).write_text(text, encoding="utf-8")


def _tox(record: dict[str, object], axis: str) -> dict[str, object]:
    group = record["toxicology"]
    assert isinstance(group, dict)
    item = group[axis]
    assert isinstance(item, dict)
    return item


@pytest.mark.parametrize("kind", ["ligand", "small_molecule", "non_polymer"])
def test_a_missing_toxicology_table_stays_untested(tmp_path: Path, kind: str) -> None:
    output = new_folder(tmp_path, kind)
    record = screen_generation(output, binder_kind=kind, target_id="P05113")
    assert record["call"] == "untested"
    assert record["bar"] == "coverage"
    assert record["hits"] == []
    herg = _tox(record, "herg")
    assert herg["state"] == "untested"
    assert herg["blocker"] == "toxicology_not_configured"
    assert herg["measurement"] is None
    assert herg["source"] == "untested"
    assert herg["threshold"] == 10_000.0
    assert herg["units"] == "nM"
    assert record["states"]["herg"] == "untested"
    assert "immunogenicity" not in record["states"]
    assert "sequence_liability" not in record


def test_coverage_clears_when_herg_pgp_and_bcrp_are_above_the_cut(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "clear")
    _table(output, _CLEAR)
    record = screen_generation(output, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "screened_clear"
    assert record["bar"] == "coverage"
    assert record["hits"] == []
    assert record["states"]["herg"] == "clear"
    assert record["states"]["pgp"] == "clear"
    assert record["states"]["bcrp"] == "clear"
    assert "cyp3a4" in record["untested"]
    assert "nav1_5" in record["untested"]
    assert "bsep" in record["untested"]
    assert "plasma_abundance" in record["untested"]
    herg = _tox(record, "herg")
    assert herg["state"] == "clear"
    assert herg["source"] == "measured"
    assert herg["measurement"] == 20_000.0
    assert herg["blocker"] is None
    lines = "\n".join(format_screen_lines("1ABC", "A", record))
    assert "herg clear 20000.0, nM panel reading" in lines
    assert "cyp3a4 clear" not in lines


def test_a_reading_at_the_cut_is_a_hit_and_one_above_it_is_clear(tmp_path: Path) -> None:
    hit = new_folder(tmp_path, "hit")
    _table(hit, "axis\tnm\nherg\t10000\npgp\t10001\nbcrp\t10001\n")
    record = screen_generation(hit, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "axis_hit"
    assert record["hits"] == ["herg"]
    assert _tox(record, "herg")["state"] == "hit"
    assert _tox(record, "herg")["measurement"] == 10_000.0
    assert _tox(record, "pgp")["state"] == "clear"
    assert _tox(record, "pgp")["measurement"] == 10_001.0
    lines = "\n".join(format_screen_lines("1ABC", "A", record))
    assert "herg hit 10000.0, nM panel reading" in lines


def test_the_lowest_reading_is_kept_and_a_panel_hit_outside_coverage_counts(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "lowest")
    _table(
        output,
        "axis\tnm\nherg\t50000\nherg\t10\npgp\t20000\nbcrp\t20000\nnav1_5\t10\n",
    )
    record = screen_generation(output, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "axis_hit"
    assert record["hits"] == ["herg", "nav1_5"]
    assert _tox(record, "herg")["measurement"] == 10.0


def test_a_wide_table_and_a_banner_and_a_byte_order_mark(tmp_path: Path) -> None:
    wide = new_folder(tmp_path, "wide")
    # ``bcrp`` names the protein and no unit, so the table has to say nM
    # somewhere for its number to be read at all: see
    # tests/test_disease_panel_methods.py, where a bare column with no unit
    # cell used to turn a clean 50 uM reading into a 50 nM hit.
    _table(wide, "herg (nM)\tpgp_nm\tbcrp\tunit\n20000\t20000\t20000\tnM\n")
    assert screen_generation(wide, binder_kind="ligand", target_id="P05113")["call"] == "screened_clear"

    banner = new_folder(tmp_path, "banner")
    _table(banner, "# panel export\n\naxis\tnm\nherg\t20000\npgp\t20000\nbcrp\t20000\n")
    assert screen_generation(banner, binder_kind="ligand", target_id="P05113")["call"] == "screened_clear"

    marked = new_folder(tmp_path, "marked")
    (marked / "toxicology.tsv").write_bytes(_CLEAR.encode("utf-8-sig"))
    record = screen_generation(marked, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "screened_clear"
    assert _tox(record, "herg")["source"] == "measured"


def test_tsv_wins_over_csv_and_csv_is_read_when_it_is_the_only_file(tmp_path: Path) -> None:
    both = new_folder(tmp_path, "both")
    _table(both, "axis\tnm\nherg\t10\npgp\t20000\nbcrp\t20000\n")
    _table(both, "axis,nm\nherg,20000\npgp,20000\nbcrp,20000\n", name="toxicology.csv")
    record = screen_generation(both, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "axis_hit"
    assert _tox(record, "herg")["measurement"] == 10.0

    csv_only = new_folder(tmp_path, "csv")
    _table(csv_only, "axis,nm\nherg,20000\npgp,20000\nbcrp,20000\n", name="toxicology.csv")
    assert screen_generation(csv_only, binder_kind="ligand", target_id="P05113")["call"] == "screened_clear"


def test_a_bad_cell_stays_untested_and_does_not_borrow_another_number(tmp_path: Path) -> None:
    negative = new_folder(tmp_path, "negative")
    _table(negative, "axis\tnm\nherg\t-1\npgp\t20000\nbcrp\t20000\n")
    record = screen_generation(negative, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "untested"
    assert record["hits"] == []
    assert _tox(record, "herg")["state"] == "untested"
    assert _tox(record, "herg")["blocker"] == "reading_not_nm"
    assert _tox(record, "herg")["measurement"] == -1.0
    lines = "\n".join(format_screen_lines("1ABC", "A", record))
    assert "herg untested -1.0, nM panel reading" in lines

    comma = new_folder(tmp_path, "comma")
    _table(comma, "herg\tpgp\tbcrp\n10,5\t20000\t20000\n")
    comma_record = screen_generation(comma, binder_kind="ligand", target_id="P05113")
    assert comma_record["call"] == "untested"
    assert _tox(comma_record, "herg")["blocker"] == "cell_unparsed"
    assert _tox(comma_record, "herg")["measurement"] is None
    assert _tox(comma_record, "herg")["unparsed_columns"] == ["herg"]
    assert "herg" not in comma_record["hits"]
    page = render_closing(
        format_screen_lines("1ABC", "A", comma_record),
        [{"entry_id": "1ABC", "chain": "A", "screen": comma_record}],
    )
    assert "Columns that did not parse: herg." in page

    empty = new_folder(tmp_path, "empty")
    _table(empty, "")
    assert _tox(screen_generation(empty, binder_kind="ligand", target_id="P05113"), "pgp")["blocker"] == (
        "toxicology_not_reported"
    )

    broken = new_folder(tmp_path, "broken")
    (broken / "toxicology.tsv").write_bytes(b"\xff\xfe")
    assert _tox(screen_generation(broken, binder_kind="ligand", target_id="P05113"), "herg")["blocker"] == (
        "toxicology_unreadable"
    )

    doubled = new_folder(tmp_path, "doubled")
    _table(doubled, "nm\tnm\n10\t20\n")
    assert _tox(screen_generation(doubled, binder_kind="ligand", target_id="P05113"), "bsep")["blocker"] == (
        "toxicology_unreadable"
    )


def test_axis_evidence_wins_over_the_table(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "curated")
    (output / "axis_evidence.json").write_text(
        json.dumps({"herg": "clear", "pgp": "clear", "bcrp": "clear"}),
        encoding="utf-8",
    )
    _table(output, "axis\tnm\nherg\t10\n")
    record = screen_generation(output, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "screened_clear"
    assert record["hits"] == []
    assert record["states"]["herg"] == "clear"
    herg = _tox(record, "herg")
    assert herg["source"] == "axis_evidence"
    assert herg["state"] == "clear"
    assert herg["measured_state"] == "hit"
    assert herg["measurement"] == 10.0
    lines = "\n".join(format_screen_lines("1ABC", "A", record))
    assert "herg clear, source axis_evidence, measured hit 10.0" in lines

    overruled = new_folder(tmp_path, "overruled")
    (overruled / "axis_evidence.json").write_text(json.dumps({"herg": "hit"}), encoding="utf-8")
    _table(overruled, _CLEAR)
    hit = screen_generation(overruled, binder_kind="ligand", target_id="P05113")
    assert hit["call"] == "axis_hit"
    assert hit["hits"] == ["herg"]
    assert _tox(hit, "herg")["source"] == "axis_evidence"
    assert _tox(hit, "herg")["measured_state"] == "clear"
    assert _tox(hit, "herg")["measurement"] == 20_000.0


def test_a_protein_screen_does_not_gain_panel_axes(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "protein")
    _table(output, f"axis\tnm\tnotes\nherg\t10\t{_SEQUENCE}\n")
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "untested"
    assert "toxicology" not in record
    assert "herg" not in record["states"]
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert _SEQUENCE not in text
    assert "herg" not in text
    page = render_closing(
        format_screen_lines("8TLD", "2", record), [{"entry_id": "8TLD", "chain": "2", "screen": record}]
    )
    assert ">Immunogenicity</label>" in page
    assert ">hERG</label>" not in page
    assert "cure" not in page.casefold()


@pytest.mark.parametrize("kind", ["peptide", "nucleic"])
def test_peptide_and_nucleic_screens_omit_the_panel(tmp_path: Path, kind: str) -> None:
    output = new_folder(tmp_path, kind)
    _table(output, "axis\tnm\nherg\t10\n")
    record = screen_generation(output, binder_kind=kind, target_id="IL5")
    assert "toxicology" not in record
    assert "herg" not in record["states"]
    assert record["call"] == "untested"


def test_a_ligand_screen_keeps_the_table_out_of_the_stored_text(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "scrub")
    _table(
        output,
        "axis\tnm\tnotes\n" f"herg\t20000\t{_SEQUENCE}\n" "pgp\t20000\t\n" "bcrp\t20000\t\n" "immunogenicity\t0\t\n",
    )
    record = screen_generation(output, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "screened_clear"
    assert "immunogenicity" not in record["states"]
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert _SEQUENCE not in text
    assert str(output) not in text
    # The table's contents and this machine's path stay out, which is what this
    # test is for. Its name does not: the name of a supplied table is a constant
    # of the program, identical for every user and already printed on the closed
    # page, and a verdict that cannot name its inputs cannot be checked against
    # them.
    named = {item["file"] for item in record["inputs"]}
    assert named == {"toxicology.tsv"}
    assert len(record["inputs"][0]["digest"]) == 16


def test_the_ligand_board_shows_panel_tabs(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "board")
    record = screen_generation(output, binder_kind="ligand", target_id="P05113")
    page = render_closing(
        format_screen_lines("1ABC", "A", record),
        [{"entry_id": "1ABC", "chain": "A", "screen": record}],
    )
    assert ">hERG</label>" in page
    assert ">P-gp</label>" in page
    assert ">BCRP</label>" in page
    assert ">CYP3A4</label>" in page
    assert ">Plasma abundance</label>" in page
    assert ">Immunogenicity</label>" not in page
    assert "No toxicology table was found in the output folder." in page
    assert "Add toxicology.tsv to the output folder." in page
    assert "At or below 10000 nM is a hit on this panel protein." in page
    assert "cure" not in page.casefold()


def test_a_declared_unit_other_than_nm_is_not_rescaled(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "micromolar")
    _table(output, "axis\tnm\tunit\nherg\t15\tuM\npgp\t20000\tnM\nbcrp\t20000\tnM\n")
    record = screen_generation(output, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "untested"
    assert record["hits"] == []
    herg = _tox(record, "herg")
    assert herg["state"] == "untested"
    assert herg["blocker"] == "reading_not_nm"
    assert herg["measurement"] is None

    named = new_folder(tmp_path, "named-nm")
    _table(named, "axis\tnm\tunits\nherg\t15\tnM\npgp\t20000\tnanomolar\nbcrp\t20000\tnmol/L\n")
    judged = screen_generation(named, binder_kind="ligand", target_id="P05113")
    assert judged["call"] == "axis_hit"
    assert _tox(judged, "herg")["measurement"] == 15.0
    assert _tox(judged, "herg")["n_readings"] == 1


def test_blank_tokens_are_absent_and_a_repeated_notes_column_stays_readable(tmp_path: Path) -> None:
    blanks = new_folder(tmp_path, "blanks")
    _table(blanks, "axis\tnm\nherg\ttrue\npgp\tn.a.\nbcrp\tfalse\n")
    record = screen_generation(blanks, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "untested"
    assert record["hits"] == []
    assert _tox(record, "herg")["blocker"] == "toxicology_not_reported"
    assert _tox(record, "pgp")["blocker"] == "toxicology_not_reported"

    zero = new_folder(tmp_path, "zero")
    _table(zero, "axis\tnm\nherg\t0\npgp\t20000\nbcrp\t20000\n")
    zero_record = screen_generation(zero, binder_kind="ligand", target_id="P05113")
    assert zero_record["call"] == "axis_hit"
    assert _tox(zero_record, "herg")["measurement"] == 0.0

    notes = new_folder(tmp_path, "notes")
    _table(notes, "axis\tnm\tnotes\tnotes\nherg\t20000\ta\tb\npgp\t20000\t\t\nbcrp\t20000\t\t\n")
    assert screen_generation(notes, binder_kind="ligand", target_id="P05113")["call"] == "screened_clear"

    doubled_axis = new_folder(tmp_path, "doubled-axis")
    _table(doubled_axis, "herg\therg\n10\t20000\n")
    assert _tox(screen_generation(doubled_axis, binder_kind="ligand", target_id="P05113"), "herg")["blocker"] == (
        "toxicology_unreadable"
    )


def test_different_panel_methods_retain_both_results(tmp_path: Path) -> None:
    (tmp_path / "toxicology.tsv").write_text(
        "axis\tnm\tmethod\nherg\t50\tassay A\nherg\t50000\tassay B\n", encoding="utf-8"
    )
    panel = measure_toxicology(tmp_path)["herg"]
    assert panel["measurement"] is None
    assert panel["state"] == "hit" and panel["disagreement"]
    assert [(entry["method"], entry["measurement"]) for entry in panel["opinions"]] == [
        ("assay A", 50), ("assay B", 50000)
    ]


def test_the_lowest_of_several_cells_records_how_many(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "count")
    _table(output, "axis\tnm\nherg\t50000\nherg\t10\npgp\t20000\nbcrp\t20000\n")
    record = screen_generation(output, binder_kind="ligand", target_id="P05113")
    assert _tox(record, "herg")["measurement"] == 10.0
    assert _tox(record, "herg")["n_readings"] == 2
    page = render_closing(
        format_screen_lines("1ABC", "A", record),
        [{"entry_id": "1ABC", "chain": "A", "screen": record}],
    )
    assert "The reading is the lowest of 2 cells." in page
    assert "cure" not in page.casefold()


def test_a_quote_that_swallows_a_row_refuses_the_whole_table(tmp_path: Path) -> None:
    """``csv`` reports nothing when a field eats the rest of its row.

    An open quote above the cyp3a4 row makes that row part of hERG's own cell.
    The reader then held one row where the file had three: hERG was
    ``cell_unparsed``, and cyp3a4 -- a 20 nM hit the file states plainly --
    came back ``toxicology_not_reported``, a claim the file contradicts. Where
    a later quote closes the field, the rows it crossed are gone just the same
    and the rows after it still parse, so the table published a clear while
    silently holding a hit. A cell can only span physical lines this way, so
    the table is refused in full, loudly.
    """
    eaten = new_folder(tmp_path, "unclosed")
    _table(eaten, 'axis,nm\nherg,"150\ncyp3a4,20\npgp,20000\n', name="toxicology.csv")
    found = measure_toxicology(eaten)
    for axis in ("herg", "cyp3a4", "pgp"):
        assert found[axis]["state"] == "untested", axis
        assert found[axis]["blocker"] == "toxicology_unreadable", axis
        assert found[axis]["measurement"] is None, axis

    reclosed = new_folder(tmp_path, "reclosed")
    _table(reclosed, 'axis,nm\nherg,"150\ncyp3a4,20"\npgp,20000\nbcrp,20000\n', name="toxicology.csv")
    reopened = measure_toxicology(reclosed)
    assert reopened["pgp"]["blocker"] == "toxicology_unreadable"
    assert reopened["pgp"]["state"] == "untested"
    assert reopened["cyp3a4"]["blocker"] == "toxicology_unreadable"

    record = screen_generation(reclosed, binder_kind="ligand", target_id="P05113")
    assert record["call"] == "untested"
    assert record["states"]["pgp"] == "untested"

    # The same quote on the header line eats the rows under the column names,
    # where no row is left to carry the evidence: a panel whose hERG row read
    # 10 nM came back with no rows at all and every axis "not reported".
    header = new_folder(tmp_path, "header")
    _table(header, 'axis,"nm\nherg,10"\npgp,20000\nbcrp,20000\n', name="toxicology.csv")
    headed = measure_toxicology(header)
    assert headed["herg"]["blocker"] == "toxicology_unreadable"
    assert headed["pgp"]["blocker"] == "toxicology_unreadable"

    whole = new_folder(tmp_path, "header-all")
    _table(whole, 'axis,"nm\nherg,10\npgp,20000\n', name="toxicology.csv")
    assert measure_toxicology(whole)["herg"]["blocker"] == "toxicology_unreadable"

    # A quote that closes at the end of its own row loses nothing, and the
    # reading it holds is still read.
    tail = new_folder(tmp_path, "tail")
    _table(tail, 'axis,nm\nherg,"20000"\npgp,20000\nbcrp,20000\n', name="toxicology.csv")
    assert measure_toxicology(tail)["herg"]["measurement"] == 20_000.0

    # And a header that quotes one column name on its own line is ordinary.
    quoted = new_folder(tmp_path, "quoted-header")
    _table(quoted, 'axis,"nm"\nherg,20000\npgp,20000\nbcrp,20000\n', name="toxicology.csv")
    assert measure_toxicology(quoted)["herg"]["measurement"] == 20_000.0


def test_a_value_column_named_only_for_the_analyte_needs_the_row_to_say_nm(tmp_path: Path) -> None:
    """ChEMBL writes the number under ``standard_value``.

    Its ``standard_units`` and ``standard_relation`` were already read here and
    its value column was not, so a panel exported from ChEMBL reported every
    axis ``toxicology_not_reported``: a 50 nM hERG hit dropped, and the record
    saying the table had not reported hERG when it had. The column names only
    the analyte, so the row must still state nM -- the rule the bare ``herg``
    column is read under, now one rule for both column shapes.
    """
    stated = new_folder(tmp_path, "chembl")
    _table(stated, "protein\tstandard_value\tstandard_units\nherg\t50\tnM\n")
    herg = measure_toxicology(stated)["herg"]
    assert herg["state"] == "hit"
    assert herg["measurement"] == 50.0
    assert herg["blocker"] is None
    assert herg["n_readings"] == 1

    silent = new_folder(tmp_path, "chembl-silent")
    _table(silent, "protein\tstandard_value\nherg\t50\n")
    unstated = measure_toxicology(silent)["herg"]
    assert unstated["state"] == "untested"
    assert unstated["blocker"] == "reading_not_nm"
    assert unstated["measurement"] is None

    micromolar = new_folder(tmp_path, "chembl-um")
    _table(micromolar, "protein\tstandard_value\tstandard_units\nherg\t15\tuM\n")
    refused = measure_toxicology(micromolar)["herg"]
    assert refused["state"] == "untested"
    assert refused["blocker"] == "reading_not_nm"
    assert refused["measurement"] is None

    bounded = new_folder(tmp_path, "chembl-bound")
    _table(bounded, "protein\tstandard_value\tstandard_units\tstandard_relation\nherg\t50000\tnM\t>\n")
    clear = measure_toxicology(bounded)["herg"]
    assert clear["state"] == "clear"
    assert clear["measurement"] == 50_000.0
    assert clear["qualifier"] == ">"

    analyte = new_folder(tmp_path, "ic50")
    _table(analyte, "protein\tic50\tunit\nherg\t50\tnM\n")
    assert measure_toxicology(analyte)["herg"]["state"] == "hit"
    bare = new_folder(tmp_path, "ic50-bare")
    _table(bare, "protein\tic50\nherg\t50\n")
    assert measure_toxicology(bare)["herg"]["blocker"] == "reading_not_nm"

    # A column whose own name states nM still needs no unit cell.
    named = new_folder(tmp_path, "named")
    _table(named, "protein\tnm\nherg\t50\n")
    assert measure_toxicology(named)["herg"]["state"] == "hit"


def test_two_unit_cells_that_disagree_are_not_a_reading_in_the_earlier_one(tmp_path: Path) -> None:
    """ChEMBL carries ``units`` and ``standard_units`` and they differ by design.

    Reading the first unit column that held anything made the thousandfold
    error reachable through the second one: 15 under a ``nM`` ``unit`` cell was
    judged a hit while ``standard_units`` beside it said uM, and an ``n/a`` in
    the earlier column hid a uM declaration in the later one outright. Two
    relation columns had the same fault, and there it cleared an axis: ``>``
    in ``relation`` on a number ``standard_relation`` called ``<``.
    """
    split = new_folder(tmp_path, "two-units")
    _table(split, "protein\tnm\tunit\tstandard_units\nherg\t15\tnM\tuM\n")
    herg = measure_toxicology(split)["herg"]
    assert herg["state"] == "untested"
    assert herg["blocker"] == "reading_not_nm"
    assert herg["measurement"] is None

    masked = new_folder(tmp_path, "masked-unit")
    _table(masked, "protein\tnm\tunit\tunits\nherg\t15\tn/a\tuM\n")
    assert measure_toxicology(masked)["herg"]["blocker"] == "reading_not_nm"

    agreeing = new_folder(tmp_path, "agreeing-units")
    _table(agreeing, "protein\tnm\tunit\tstandard_units\nherg\t15\tnM\tnmol/L\n")
    assert measure_toxicology(agreeing)["herg"]["state"] == "hit"

    disputed = new_folder(tmp_path, "two-relations")
    _table(disputed, "protein\tnm\trelation\tstandard_relation\nherg\t50000\t>\t<\n")
    bounded = measure_toxicology(disputed)["herg"]
    assert bounded["state"] == "untested"
    assert bounded["blocker"] == "reading_not_decisive"
    assert "qualifier" not in bounded

    one_bound = new_folder(tmp_path, "one-relation")
    _table(one_bound, "protein\tnm\trelation\tstandard_relation\nherg\t50000\t>\t>\n")
    assert measure_toxicology(one_bound)["herg"]["state"] == "clear"
    assert measure_toxicology(one_bound)["herg"]["qualifier"] == ">"


def test_a_row_naming_two_panel_proteins_cannot_file_its_concentration(tmp_path: Path) -> None:
    """hERG in ``protein`` beside NAV1.5 in ``panel`` is one number, two names.

    Every axis column is read so that a gene name beside a panel id still
    reaches the panel id. Returning the first match filed the concentration
    under hERG and reported NAV1.5 as never tested, which is one protein judged
    on another's number at even odds -- what this module refuses a misaligned
    row for. It cannot be filed, so it is refused like any other unattributed
    reading.
    """
    two = new_folder(tmp_path, "two-names")
    _table(two, "protein\tpanel\tnm\nherg\tnav1.5\t50\n")
    found = measure_toxicology(two)
    for axis in ("herg", "nav1_5"):
        assert found[axis]["state"] == "untested", axis
        assert found[axis]["blocker"] == "toxicology_unreadable", axis
        assert found[axis]["measurement"] is None, axis

    one = new_folder(tmp_path, "one-name")
    _table(one, "protein\tpanel\tnm\nhERG\therg\t50\n")
    assert measure_toxicology(one)["herg"]["measurement"] == 50.0

    gene = new_folder(tmp_path, "gene")
    _table(gene, "protein\tpanel\tnm\nKCNH2\therg\t50\n")
    assert measure_toxicology(gene)["herg"]["measurement"] == 50.0

    outside = new_folder(tmp_path, "outside")
    _table(outside, "protein\tnm\nKCNQ1\t50\n")
    assert measure_toxicology(outside)["herg"]["blocker"] == "toxicology_not_reported"


def test_a_relation_this_cannot_read_is_undecided_and_not_a_unit_fault(tmp_path: Path) -> None:
    """``~50`` is in nM and is still not a reading, and the unit was fine.

    Recorded as ``reading_not_nm`` it sent the owner to the unit column, where
    nothing was wrong, and its board remedy reads "Write the reading in nM" on
    a reading already written in nM. It is the undecided case, whose remedy
    already asks for what this row is missing.
    """
    approximate = new_folder(tmp_path, "approximate")
    _table(approximate, "protein\tnm\trelation\nherg\t50\t~\n")
    herg = measure_toxicology(approximate)["herg"]
    assert herg["state"] == "untested"
    assert herg["blocker"] == "reading_not_decisive"
    assert herg["measurement"] is None

    record = screen_generation(approximate, binder_kind="ligand", target_id="P05113")
    page = render_closing(
        format_screen_lines("1ABC", "A", record),
        [{"entry_id": "1ABC", "chain": "A", "screen": record}],
    )
    assert "Record the measured concentration, or a bound that falls clear of the cut." in page
    assert "Write the reading in nM." not in page

    decisive = new_folder(tmp_path, "decisive")
    _table(decisive, "protein\tnm\tstandard_relation\nherg\t15000\t>\n")
    assert measure_toxicology(decisive)["herg"]["state"] == "clear"


def test_measure_toxicology_uses_the_existing_cut(tmp_path: Path) -> None:
    output = new_folder(tmp_path, "direct")
    # ``nav1.5`` names the protein and no unit, so the row declares nM. The
    # cut and the alias are what this test is about, and both still hold.
    _table(output, "nav1.5\tcyp3a4_nm\tunit\n10000\t10001\tnM\n")
    found = measure_toxicology(output)
    assert found["nav1_5"]["state"] == "hit"
    assert found["nav1_5"]["measurement"] == 10_000.0
    assert found["cyp3a4"]["state"] == "clear"
    assert found["cyp3a4"]["measurement"] == 10_001.0
    assert found["herg"]["blocker"] == "toxicology_not_reported"
    assert found["herg"]["threshold"] == 10_000.0
