"""Five sequence-liability axes. A missing input stays untested."""

import csv
import json
from pathlib import Path

import pytest

from rfproteina.disease import liability
from rfproteina.disease.combinations import PLASMA_AXES, PROTEIN_AXES
from rfproteina.disease.liability import (
    _CALIS_WEIGHT,
    _DR_PANEL,
    _calis_score,
    _load_proteome,
    measure_sequence_liability,
)
from rfproteina.disease.screen import format_screen_lines, screen_generation
from tests.helpers.disease import UBIQUITIN, no_developer_proteome  # noqa: F401


@pytest.fixture
def tiny_proteome_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the proteome size floor aside, for tests that are not about scale.

    The floor is a policy: a FASTA an order of magnitude below the human
    reference cannot support a clear, because every 9-mer it lacks is an
    off-target the axis cannot see. The tests requesting this fixture cover
    k-mer packing, FASTA parsing, the cache and the rollup, none of which
    depends on how large the file is. The floor has its own tests, which do not
    request it, so lowering it here cannot hide a regression in it.
    """
    monkeypatch.setattr(liability, "_PROTEOME_MIN_KMERS", 1)


_UBIQUITIN = UBIQUITIN
_NINE = "MQIFVKTLT"
_DECOY = "ACDEFGHIKLMNPQRSTVWY"
_POLY_LEU = "LLLLLLLLLL"
#: Published DR-restricted epitopes. A class II fixture uses these so that no
#: designed sequence ever appears in one. HA 306-318 is the canonical
#: promiscuous class II binder; OVA 323-339 is the published DR-restricted one.
_HA_306_318 = "PKYVKQNTLKLAT"

#: The panel in the order liability.py declares it, for building a table by hand.
_DR_PANEL_NAMES = _DR_PANEL
_OVA_323_339 = "ISQAVHAAHAEINEAGR"
#: Constructed, not from the literature: a stand-in for a peptide binding nothing.
_NON_BINDER = "EEEEEEEEEEEEE"


def _ada_design(peptide: str = _HA_306_318) -> list[tuple[str, str]]:
    """A design containing the panel's peptide, which is what makes it this design's.

    A class II table is only this design's scan if its peptides are windows of
    this design's chain. Passing no sample at all leaves the axis unable to
    establish that, which is the honest answer and not what these tests are about.
    """
    return [("binder_0", peptide)]


def _panel_rows(
    peptide: str,
    ranks: list[float],
    *,
    alleles: tuple[str, ...] = _DR_PANEL,
) -> list[tuple[str, str, float]]:
    """One row per allele for one peptide, in panel order."""
    return [(allele, peptide, rank) for allele, rank in zip(alleles, ranks, strict=True)]


_SPAN = "ACDEFGHIKLMNPQRSTVWY"


def test_a_design_that_deserves_it_can_reach_screened_clear(tmp_path: Path, tiny_proteome_ok: None) -> None:
    """The acceptance test for the whole screen: the gate can open.

    Before this work every axis in the protein bar was unreachable, so ten real
    runs on disk all read ``untested`` with no axis clear and no design could be
    advanced however good it was. Each input below is one a user supplies, and
    together they clear all six axes of the sequence bar:

    * a class I table naming one presented 9-mer that scores at or below zero
    * a reference table whose binding density is higher than the design's
    * a DR panel whose medians are all above the promiscuity threshold
    * a proteome sharing no 9-mer with the design
    * a curated paralog judgement in ``axis_evidence.json``

    That last one is not a measurement, and it is the honest state of this axis:
    a UniRef50 cluster singleton was the only route to a measured paralog clear,
    and a singleton is not evidence of having no paralogs -- KRAS and EGFR are
    both singletons by that measure. So the six-axis protein bar needs a curated
    paralog call, and the screen records that its source was the evidence file
    rather than a measurement.

    The plasma axes stay untested, which is correct: they are applicable to a
    protein but they are not in the ``sequence`` bar.
    """
    output = tmp_path / "deserving"
    # Fifteen residues, so the design holds seven distinct 9-mer windows and one
    # of them binding is a low density. A nine-residue design has exactly one
    # window, and a count over windows then makes its single binder 100 per 100
    # -- every window it has -- which no reference protein can be worse than.
    design = "LLLLLLLLL" + "GGGGGG"
    _write_cif(output, "binder_0", design)
    _write_metrics(output, ["binder_0"])

    proteome = tmp_path / "unrelated.fasta"
    proteome.write_text(">other\nWWWWWWWWWW\n", encoding="utf-8")

    # One presented 9-mer, and it scores -0.05184, so immunogenicity clears.
    # The design's other six windows are scored and not presented, which is
    # what a complete scan looks like.
    _write_class_i(output, ["LLLLLLLLL"], covers=design)
    # Twenty distinct binding peptides over a 100-residue reference's 9-mer
    # windows is 21.74 per 100. The design's one over its seven windows is
    # 14.29, so it is not worse.
    rows = [("P02768", f"REF{index:06d}", 100, 0.2) for index in range(20)]
    assert len({peptide for _name, peptide, _length, _rank in rows}) == 20
    _write_reference(output, "mhc_i", rows)

    # Every median is above 20, so no peptide is a promiscuous binder. Every
    # window is scored, not just one: a clear on either class II axis is a
    # statement about the whole chain, so the whole chain has to be in the file.
    panel = [
        row
        for window in sorted({design[at : at + 9] for at in range(len(design) - 9 + 1)})
        for row in _panel_rows(window, [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0])
    ]
    _write_dr_panel(output, panel)
    # The class II design table is the DR panel, so the bar must be scored over
    # the same seven alleles or it is a count over a different set of questions.
    _write_reference(output, "mhc_ii", [("P02768", "KKKKKKKKK", 100, 5.0)], alleles=_DR_PANEL)

    # No measurement can establish the absence of a paralog family, so the one
    # axis that cannot clear on its own is supplied as curated evidence.
    (output / "axis_evidence.json").write_text(json.dumps({"paralog": "clear"}), encoding="utf-8")

    def read(url: str) -> object:
        if "/uniref/search" in url:
            return {"results": [{"id": "UniRef50_P05113"}]}
        if "/uniprotkb/search" in url:
            return {"results": [{"primaryAccession": "P05113", "genes": [{"geneName": {"value": "IL5"}}]}]}
        return []

    record = screen_generation(
        output,
        binder_kind="protein",
        target_id="P05113",
        proteome=proteome,
        read_json=read,
    )
    liability = record["sequence_liability"]
    assert liability["immunogenicity"]["state"] == "clear"
    assert liability["proteome_offtarget"]["state"] == "clear"
    assert liability["mhc_i"]["state"] == "clear"
    assert liability["mhc_ii"]["state"] == "clear"
    assert liability["ada"]["state"] == "clear"
    paralog = record["target_liability"]["paralog"]
    assert paralog["state"] == "clear"
    assert paralog["source"] == "axis_evidence", "not a measurement, and it says so"
    assert paralog["measured_blocker"] == "cluster_singleton_not_evidence"
    assert record["call"] == "screened_clear", record["untested"]
    assert record["bar"] == "sequence"
    assert record["hits"] == []
    # Every axis that is not clear names what to supply, and none contradicts
    # itself by carrying a blocker beside a decided state.
    for axis, item in {**liability, **record["target_liability"]}.items():
        if item["state"] == "clear":
            assert item["blocker"] is None, axis
        else:
            assert item["blocker"], f"{axis} is untested and says nothing"


def test_the_proteome_rollup_reports_the_worst_sample(tmp_path: Path, tiny_proteome_ok: None) -> None:
    """The axis number is the highest count, and every failed sample is named.

    No fixture reached two matches before this, so the maximum over several
    failures was never taken. Here the counts are 2, 5 and 0 by construction:
    a 10-residue substring of the proteome has two 9-mers, a 13-residue one has
    five, and a sequence sharing nothing has none.
    """
    fasta = tmp_path / "span.fasta"
    fasta.write_text(f">span\n{_SPAN}\n", encoding="utf-8")
    samples = [
        ("binder_0", _SPAN[:10]),
        ("binder_1", _SPAN[:13]),
        ("binder_2", "WWWWWWWWW"),
    ]
    axes, by_example = measure_sequence_liability(samples, directory=tmp_path, proteome=fasta)
    assert by_example["binder_0"]["measurement"] == 2
    assert by_example["binder_1"]["measurement"] == 5
    assert by_example["binder_2"]["measurement"] == 0
    record = axes["proteome_offtarget"]
    assert record["state"] == "hit"
    assert record["measurement"] == 5, "the worst sample, not the first or the mean"
    assert record["failed"] == ["binder_0", "binder_1"]


def test_a_sample_without_a_sequence_does_not_clear_the_rollup(tmp_path: Path, tiny_proteome_ok: None) -> None:
    """One clear sample and one unscorable one is not a clear.

    The proteome loaded and nothing matched, but a sample that could not be
    scored is not evidence that it would have passed.
    """
    fasta = tmp_path / "span.fasta"
    fasta.write_text(f">span\n{_SPAN}\n", encoding="utf-8")
    samples = [("binder_0", "WWWWWWWWW"), ("binder_1", None)]
    axes, by_example = measure_sequence_liability(samples, directory=tmp_path, proteome=fasta)
    assert by_example["binder_0"]["state"] == "clear"
    assert by_example["binder_1"]["blocker"] == "sequence_absent"
    record = axes["proteome_offtarget"]
    assert record["state"] == "untested"
    assert record["blocker"] == "sequence_absent"
    assert record["failed"] == []


def test_a_multi_record_fasta_is_every_record(tmp_path: Path, tiny_proteome_ok: None) -> None:
    """Every fixture was one record, so a second header was never parsed."""
    fasta = tmp_path / "two.fasta"
    fasta.write_text(f">one\n{_SPAN}\n>two\nWWWWWWWWWW\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability(
        [("binder_0", "WWWWWWWWW")],
        directory=tmp_path,
        proteome=fasta,
    )
    assert axes["proteome_offtarget"]["state"] == "hit", "the second record counts"
    assert axes["proteome_offtarget"]["measurement"] == 1


def test_a_parsed_proteome_is_reused_until_the_file_changes(tmp_path: Path, tiny_proteome_ok: None) -> None:
    """Reading a whole proteome costs seconds and a run screens several designs."""
    fasta = tmp_path / "cached.fasta"
    fasta.write_text(f">one\n{_SPAN}\n", encoding="utf-8")
    first = _load_proteome(fasta)
    again = _load_proteome(fasta)
    assert again[1] is first[1], "the same file is parsed once"
    fasta.write_text(f">one\n{_SPAN}\n>two\nWWWWWWWWWW\n", encoding="utf-8")
    changed = _load_proteome(fasta)
    assert changed[1] is not first[1], "a changed file is parsed again"
    assert changed[1].size > first[1].size


def test_an_unreadable_predictor_table_is_not_a_verdict(tmp_path: Path) -> None:
    """Neither of these two paths had any test, so both were dead code.

    A table that is not valid UTF-8, and one the CSV reader refuses, each leave
    the axis untested rather than silently reading no rows and clearing.
    """
    (tmp_path / "mhc_i.tsv").write_bytes(b"percentile_rank\n\xff\xfe\n")
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "predictor_unreadable"
    assert axes["mhc_i"]["measurement"] is None

    # A field past the CSV field limit is what actually raises csv.Error here. A
    # NUL byte does not: it decodes, and the cell simply is not a number, which
    # is reported as score_absent rather than as an unreadable table.
    huge = "x" * (csv.field_size_limit() + 1)
    (tmp_path / "mhc_i.tsv").write_text(f"percentile_rank\n{huge}\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    assert axes["mhc_i"]["blocker"] == "predictor_unreadable"

    # A NUL decodes, so the file is readable, but "1.0\x00" is not a number and
    # must not pass as an empty cell: skipping what will not parse is how a
    # table of binders read as a table with none. Reported as the unreadable
    # cell it is, naming the column, rather than as a missing score.
    (tmp_path / "mhc_i.tsv").write_bytes(b"percentile_rank\n1.0\x00\n")
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    assert axes["mhc_i"]["blocker"] == "cell_unparsed"
    assert axes["mhc_i"]["unparsed_columns"] == ["percentile_rank"]


def test_the_calis_anchor_positions_carry_no_weight(tmp_path: Path) -> None:
    """The published mask, asserted as the property it is rather than by snapshot.

    Positions 1, 2 and the C-terminus drive MHC binding rather than T-cell
    recognition, so Calis 2013 gives them zero weight. Mutating only those three
    must therefore leave the score untouched, and a snapshot of one peptide
    would not have caught a mask applied to the wrong positions.
    """
    assert _CALIS_WEIGHT[0] == 0.0
    assert _CALIS_WEIGHT[1] == 0.0
    assert _CALIS_WEIGHT[8] == 0.0
    assert len(_CALIS_WEIGHT) == 9
    plain = _calis_score("AAAAAAAAA")
    assert _calis_score("WWAAAAAAW") == plain, "only the masked positions changed"
    assert _calis_score("KKAAAAAAK") == plain
    assert _calis_score("AAWAAAAAA") != plain, "an unmasked position does change it"


def _write_class_i(folder: Path, peptides: list[str], *, rank: float = 0.5, covers: str | None = None) -> None:
    """A class I table naming which 9-mers it calls presented.

    THE RANKS HERE ARE CONSTRUCTED, well inside the IEDB class I cutoff, so that
    every peptide listed counts as presented. They are not predictor output.

    ``covers`` is the design the table was scanned from: its remaining 9-mer
    windows are written well above the cutoff, so they are scored and not
    presented. That is what a complete scan looks like, and a density only
    clears on one.
    """
    lines = ["allele\tpeptide\tmethod\tpercentile_rank"]
    lines += [f"HLA-A*02:01\t{peptide}\tconsensus\t{rank}" for peptide in peptides]
    if covers is not None:
        windows = {covers[at : at + 9] for at in range(len(covers) - 9 + 1)}
        lines += [f"HLA-A*02:01\t{window}\tconsensus\t90.0" for window in sorted(windows - set(peptides))]
    (folder / "mhc_i.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dr_panel(
    folder: Path,
    rows: list[tuple[str, str, float]],
    *,
    method: str = "consensus",
) -> None:
    """A class II table in the shape a predictor writes one.

    THE RANKS HERE ARE CONSTRUCTED to fall either side of the published cutoffs.
    They are not any tool's output and must not be read as a prediction. Only
    the peptides are real, and they are published epitopes rather than designs.
    """
    lines = ["allele\tpeptide\tmethod\tpercentile_rank"]
    lines += [f"{allele}\t{peptide}\t{method}\t{rank}" for allele, peptide, rank in rows]
    (folder / "mhc_ii.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_missing_inputs_stay_unconfigured(tmp_path: Path) -> None:
    axes, by_example = measure_sequence_liability([], directory=tmp_path)
    assert by_example == {}
    assert axes["proteome_offtarget"] == {
        "state": "untested",
        "measurement": None,
        "blocker": "proteome_not_configured",
        "failed": [],
    }
    for axis in ("mhc_i", "mhc_ii", "immunogenicity", "ada"):
        assert axes[axis] == {"state": "untested", "measurement": None, "blocker": "predictor_not_configured"}
    assert list(axes) == [
        "immunogenicity",
        "proteome_offtarget",
        "mhc_i",
        "mhc_ii",
        "ada",
    ]


def test_mhc_i_cutoffs(tmp_path: Path) -> None:
    assert _mhc(tmp_path / "low", "mhc_i", "percentile_rank", "0.4") == ("hit", 0.4)
    assert _mhc(tmp_path / "edge", "mhc_i", "rank", "1.0") == ("hit", 1.0)
    assert _mhc(tmp_path / "over", "mhc_i", "percentile_rank", "1.1") == ("clear", 1.1)
    assert _mhc(tmp_path / "strong", "mhc_i", "ic50_nm", "500") == ("hit", 500.0)
    # ``ic50_nm`` rather than a bare ``ic50``: the cutoff is nanomolar, and a
    # column named only for the analyte states no unit, so it is refused
    # instead of being read on the cut this test is about.
    assert _mhc(tmp_path / "weak", "mhc_i", "ic50_nm", "500.1") == ("clear", 500.1)


def test_mhc_ii_cutoffs(tmp_path: Path) -> None:
    assert _mhc(tmp_path / "low", "mhc_ii", "percentile_rank", "10") == ("hit", 10.0)
    assert _mhc(tmp_path / "over", "mhc_ii", "rank", "10.1") == ("clear", 10.1)
    assert _mhc(tmp_path / "strong", "mhc_ii", "ic50_nm", "1000") == ("hit", 1000.0)
    assert _mhc(tmp_path / "weak", "mhc_ii", "ic50_nm", "1000.1") == ("clear", 1000.1)


def test_one_mhc_hit_wins_and_the_stored_number_is_the_best_rank(tmp_path: Path) -> None:
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "mhc_i.csv").write_text("Percentile_Rank,ic50_nm\n2.0,100\n0.4,2000\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([], directory=folder, binder_kind="peptide")
    assert axes["mhc_i"]["state"] == "hit"
    assert axes["mhc_i"]["measurement"] == 0.4
    assert axes["mhc_i"]["blocker"] is None


def test_a_score_is_stored_and_stays_untested(tmp_path: Path) -> None:
    (tmp_path / "immunogenicity.tsv").write_text("score\n0.2\n1.5\n", encoding="utf-8")
    (tmp_path / "ada.csv").write_text("score\n2.25\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    assert axes["immunogenicity"] == {
        "state": "untested",
        "measurement": 1.5,
        "blocker": "no_pass_cut",
        # The number came from the table, so it is not tagged with the name of a
        # formula this module implements.
        "method": "supplied_table",
    }
    assert axes["ada"]["state"] == "untested"
    assert axes["ada"]["measurement"] == 2.25
    assert axes["ada"]["blocker"] == "no_pass_cut"
    assert axes["ada"]["method"] == "supplied_table"


def test_a_table_without_a_number_stays_untested(tmp_path: Path) -> None:
    (tmp_path / "mhc_i.tsv").write_text("percentile_rank\nnan\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "score_absent"
    assert axes["mhc_i"]["measurement"] is None


def test_exact_9mer_identity_counts_distinct_windows(tmp_path: Path, tiny_proteome_ok: None) -> None:
    proteome = tmp_path / "proteome.fasta"
    proteome.write_text(">one\nAAAAAAAAA\n", encoding="utf-8")
    axes, by_example = measure_sequence_liability(
        [("binder_0", "AAAAAAAAAA"), ("binder_1", "AAAAAAAAXQQQQQQQQQ")],
        directory=tmp_path,
        proteome=proteome,
    )
    assert by_example["binder_0"]["state"] == "hit"
    assert by_example["binder_0"]["measurement"] == 1
    assert by_example["binder_1"]["state"] == "clear"
    assert by_example["binder_1"]["measurement"] == 0
    assert axes["proteome_offtarget"]["state"] == "hit"
    assert axes["proteome_offtarget"]["measurement"] == 1
    assert axes["proteome_offtarget"]["failed"] == ["binder_0"]
    dumped = json.dumps(axes)
    assert "AAAAAAAAA" not in dumped
    assert str(proteome) not in dumped


def test_a_headerless_fasta_is_one_record_and_an_empty_file_is_not_clear(
    tmp_path: Path, tiny_proteome_ok: None
) -> None:
    readable = tmp_path / "plain.fasta"
    readable.write_text("ACDEFGHIK\n", encoding="utf-8")
    axes, by_example = measure_sequence_liability(
        [("binder_0", "ACDEFGHIKLM")],
        directory=tmp_path,
        proteome=readable,
    )
    assert by_example["binder_0"]["state"] == "hit"
    assert by_example["binder_0"]["measurement"] == 1
    empty = tmp_path / "empty.fasta"
    empty.write_text("\n", encoding="utf-8")
    empty_axes, _by_example = measure_sequence_liability(
        [("binder_0", _DECOY)],
        directory=tmp_path,
        proteome=empty,
    )
    assert empty_axes["proteome_offtarget"]["state"] == "untested"
    assert empty_axes["proteome_offtarget"]["blocker"] == "proteome_empty"
    missing = tmp_path / "missing.fasta"
    missing_axes, _by_example = measure_sequence_liability(
        [("binder_0", _DECOY)],
        directory=tmp_path,
        proteome=missing,
    )
    assert missing_axes["proteome_offtarget"]["blocker"] == "proteome_unreadable"
    assert missing_axes["proteome_offtarget"]["state"] == "untested"


def test_no_samples_do_not_clear_a_configured_proteome(tmp_path: Path, tiny_proteome_ok: None) -> None:
    proteome = tmp_path / "proteome.fasta"
    proteome.write_text(">one\nAAAAAAAAA\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([], directory=tmp_path, proteome=proteome)
    assert axes["proteome_offtarget"]["state"] == "untested"
    assert axes["proteome_offtarget"]["blocker"] == "sequence_absent"


def test_a_screen_without_inputs_stays_untested(tmp_path: Path) -> None:
    output = tmp_path / "bare"
    output.mkdir()
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "untested"
    assert record["requested_class"] == "requested_clear"
    assert record["untested"] == [*PROTEIN_AXES, *PLASMA_AXES]
    liability = record["sequence_liability"]
    assert isinstance(liability, dict)
    assert liability["proteome_offtarget"]["blocker"] == "proteome_not_configured"
    assert liability["proteome_offtarget"]["source"] == "untested"
    assert liability["mhc_i"]["blocker"] == "predictor_not_configured"
    assert liability["immunogenicity"]["source"] == "untested"
    lines = format_screen_lines("8TLD", "2", record)
    assert "proteome_not_configured" not in lines[1]
    assert "predictor_not_configured" not in lines[1]
    assert "Samples " not in lines[1]


def test_a_proteome_match_is_a_hit_and_the_screen_stores_no_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_proteome_ok: None
) -> None:
    output = tmp_path / "hit"
    fasta = tmp_path / "mnt" / "leak" / "human.fasta"
    fasta.parent.mkdir(parents=True)
    fasta.write_text(f">piece\n{_NINE}\n", encoding="utf-8")
    monkeypatch.setenv("HUMAN_PROTEOME", str(fasta))
    _write_cif(output, "binder_0", _UBIQUITIN)
    _write_metrics(output, ["binder_0"])
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "axis_hit"
    assert record["hits"] == ["proteome_offtarget"]
    assert record["requested_class"] == "requested_clear"
    assert "paralog" in record["untested"]
    liability = record["sequence_liability"]
    assert liability["proteome_offtarget"]["state"] == "hit"
    assert liability["proteome_offtarget"]["source"] == "measured"
    assert liability["proteome_offtarget"]["measurement"] == 1
    assert liability["proteome_offtarget"]["failed"] == ["binder_0"]
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert _UBIQUITIN not in text
    assert _NINE not in text
    assert str(fasta) not in text
    assert "/mnt/" not in text
    assert "sk-" not in text
    assert "exact 9-mer identity" in format_screen_lines("8TLD", "2", record)[1]
    assert "Failed proteome_offtarget: binder_0" in format_screen_lines("8TLD", "2", record)[1]


def test_axis_evidence_wins_over_a_proteome_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_proteome_ok: None
) -> None:
    output = tmp_path / "evidence"
    fasta = tmp_path / "human.fasta"
    fasta.write_text(f">piece\n{_NINE}\n", encoding="utf-8")
    monkeypatch.setenv("HUMAN_PROTEOME", str(fasta))
    _write_cif(output, "binder_0", _UBIQUITIN)
    _write_metrics(output, ["binder_0"])
    (output / "axis_evidence.json").write_text(json.dumps(dict.fromkeys(PROTEIN_AXES, "clear")), encoding="utf-8")
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "screened_clear"
    assert record["untested"] == list(PLASMA_AXES)
    liability = record["sequence_liability"]
    assert liability["proteome_offtarget"]["state"] == "clear"
    assert liability["proteome_offtarget"]["source"] == "axis_evidence"
    assert liability["proteome_offtarget"]["measurement"] == 1
    assert record["samples"][0]["proteome_offtarget"]["state"] == "hit"
    measured = format_screen_lines("8TLD", "2", record)[1]
    assert "proteome_offtarget hit" not in measured
    assert "exact 9-mer identity" not in measured


def test_a_proteome_clear_does_not_screen_the_sequence_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_proteome_ok: None
) -> None:
    output = tmp_path / "clear"
    fasta = tmp_path / "human.fasta"
    fasta.write_text(">other\nWWWWWWWWWW\n", encoding="utf-8")
    monkeypatch.setenv("HUMAN_PROTEOME", str(fasta))
    _write_cif(output, "binder_0", _UBIQUITIN)
    _write_metrics(output, ["binder_0"])
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "untested"
    assert record["call"] != "screened_clear"
    assert "paralog" in record["untested"]
    assert "proteome_offtarget" not in record["untested"]
    liability = record["sequence_liability"]
    assert liability["proteome_offtarget"]["state"] == "clear"
    assert liability["proteome_offtarget"]["measurement"] == 0
    assert liability["proteome_offtarget"]["source"] == "measured"
    assert "proteome_offtarget clear 0, exact 9-mer identity" in format_screen_lines("8TLD", "2", record)[1]


def test_mhc_i_rank_hits_through_the_screen(tmp_path: Path) -> None:
    output = tmp_path / "mhc"
    output.mkdir()
    (output / "mhc_i.tsv").write_text("percentile_rank\n0.4\n", encoding="utf-8")
    record = screen_generation(output, binder_kind="peptide", target_id="IL5")
    assert record["call"] == "axis_hit"
    assert record["hits"] == ["mhc_i"]
    assert record["sequence_liability"]["mhc_i"] == {
        "state": "hit",
        "measurement": 0.4,
        "blocker": None,
        "source": "measured",
        # The verdict names which number decided it, and the other stays visible.
        "rank": 0.4,
        "ic50": None,
        "trigger": "rank",
        # Windows the table declined to score are not non-binders, so a verdict
        # says how many there were. This table names no peptide at all.
        "unscored_peptides": 0,
    }
    assert "mhc_i hit 0.4" in format_screen_lines("8TLD", "2", record)[1]
    assert "predictor_not_configured" not in format_screen_lines("8TLD", "2", record)[1]


def test_mhc_ii_above_the_cutoff_is_clear_and_the_call_stays_untested(tmp_path: Path) -> None:
    output = tmp_path / "mhc2"
    output.mkdir()
    # The row names the peptide it scored, and that peptide is the design. An
    # affinity above the cutoff is what this test is about; a table naming no
    # peptide is separately refused for a clear, because nothing then ties the
    # number to this molecule, and that refusal is what would be measured here.
    peptide = "MQIFVKTLT"
    sample = output / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text(
        "data_design\n#\nloop_\n_entity_poly.entity_id\n"
        "_entity_poly.pdbx_seq_one_letter_code_can\n_entity_poly.pdbx_strand_id\n"
        f"0\n;{peptide}\n;\nA\n#\n",
        encoding="utf-8",
    )
    _write_metrics(output, ["binder_0"])
    (output / "mhc_ii.tsv").write_text(f"peptide\tic50_nm\n{peptide}\t1000.1\n", encoding="utf-8")
    record = screen_generation(output, binder_kind="peptide", target_id="IL5")
    assert record["call"] == "untested"
    assert record["sequence_liability"]["mhc_ii"]["state"] == "clear"
    assert record["sequence_liability"]["mhc_ii"]["measurement"] == 1000.1
    assert "mhc_ii" not in record["untested"]
    assert "paralog" in record["untested"]
    assert "mhc_ii clear 1000.1" in format_screen_lines("8TLD", "2", record)[1]


def test_immunogenicity_and_ada_scores_do_not_change_the_call(tmp_path: Path) -> None:
    output = tmp_path / "scores"
    output.mkdir()
    (output / "immunogenicity.tsv").write_text("score\n1.5\n", encoding="utf-8")
    (output / "ada.tsv").write_text("score\n0.5\n", encoding="utf-8")
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "untested"
    assert record["hits"] == []
    assert record["sequence_liability"]["immunogenicity"]["measurement"] == 1.5
    assert record["sequence_liability"]["immunogenicity"]["blocker"] == "no_pass_cut"
    assert record["sequence_liability"]["immunogenicity"]["state"] == "untested"
    assert record["sequence_liability"]["ada"]["measurement"] == 0.5
    assert record["sequence_liability"]["ada"]["blocker"] == "no_pass_cut"
    measured = format_screen_lines("8TLD", "2", record)[1]
    assert "no_pass_cut" not in measured, "a blocker name is not the readout"
    # A supplied number is reported rather than dropped, and it is attributed to
    # the table it came from. Silently omitting it hid a number the user provided.
    assert "immunogenicity untested 1.5, supplied table" in measured
    assert "ada untested 0.5, supplied table" in measured
    assert "Calis" not in measured, "the sequence formula did not produce this"
    assert "Kolaskar" not in measured, "the sequence formula did not produce this"


def test_a_later_proteome_hit_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_proteome_ok: None) -> None:
    output = tmp_path / "two"
    fasta = tmp_path / "human.fasta"
    fasta.write_text(">q\nQQQQQQQQQ\n", encoding="utf-8")
    monkeypatch.setenv("HUMAN_PROTEOME", str(fasta))
    _write_cif(output, "binder_0", _DECOY)
    _write_cif(output, "binder_1", "QQQQQQQQQQ")
    _write_metrics(output, ["binder_0", "binder_1"])
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "axis_hit"
    assert record["samples"][0]["proteome_offtarget"]["state"] == "clear"
    assert record["samples"][1]["proteome_offtarget"]["state"] == "hit"
    assert record["sequence_liability"]["proteome_offtarget"]["failed"] == ["binder_1"]
    assert record["sequence_developability"]["gravy"] == record["samples"][0]["gravy"]
    measured = format_screen_lines("8TLD", "2", record)[1]
    assert "Failed proteome_offtarget: binder_1" in measured
    assert "Samples 2, sequence_pass" in measured
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert _DECOY not in text
    assert "QQQQQQQQQ" not in text


def test_gravy_is_not_an_immunogenicity_hit(tmp_path: Path) -> None:
    output = tmp_path / "hydrophobic"
    _write_cif(output, "binder_0", _POLY_LEU)
    _write_metrics(output, ["binder_0"])
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "untested"
    assert record["call"] != "screened_clear"
    assert record["hits"] == []
    assert "paralog" in record["untested"]
    assert "immunogenicity" not in record["untested"]
    assert record["samples"][0]["sequence_pass"] is False
    immunogenicity = record["sequence_liability"]["immunogenicity"]
    assert immunogenicity["state"] == "clear"
    assert immunogenicity["measurement"] == -0.05184
    assert immunogenicity["blocker"] is None
    assert immunogenicity["source"] == "measured"
    assert immunogenicity["positive_windows"] == 0
    assert immunogenicity["method"] == "calis_2013_class_i_9mer"
    assert immunogenicity["measurement"] != record["samples"][0]["gravy"]
    assert record["sequence_liability"]["mhc_i"]["blocker"] == "predictor_not_configured"
    assert record["sequence_liability"]["mhc_ii"]["blocker"] == "predictor_not_configured"
    # ada reads the class II table now, and there is none here.
    assert record["sequence_liability"]["ada"]["state"] == "untested"
    assert record["sequence_liability"]["ada"]["blocker"] == "predictor_not_configured"
    measured = format_screen_lines("8TLD", "2", record)[1]
    assert "immunogenicity clear -0.05184, Calis 2013 class I 9-mer, every 9-mer <= 0" in measured
    assert "cure" not in measured.casefold()
    assert "ada" not in measured, "an unconfigured axis is not a readout"
    assert "Kolaskar" not in measured, "the antigenic-propensity scale is no longer the ada axis"


@pytest.mark.parametrize(
    ("peptide", "score", "state"),
    [
        ("FIAGLIAIV", 0.27206, "untested"),
        ("KAVYNFATC", 0.16928, "untested"),
        ("RLNEVAKNL", -0.0101, "clear"),
        ("LITGRLQSL", -0.10776, "clear"),
        ("FQPQNGQFI", -0.12392, "clear"),
        ("SLFNTVATL", 0.16385, "untested"),
        ("LLLLLLLLL", -0.05184, "clear"),
        ("QQQQQQQQQ", -0.54144, "clear"),
    ],
)
def test_calis_class_i_9mer_matches_the_iedb_scores(tmp_path: Path, peptide: str, score: float, state: str) -> None:
    axes, _by_example = measure_sequence_liability([("binder_0", peptide)], directory=tmp_path)
    record = axes["immunogenicity"]
    assert record["measurement"] == score
    assert record["state"] == state
    assert record["method"] == "calis_2013_class_i_9mer"
    assert record["state"] != "hit"
    if state == "clear":
        assert record["blocker"] is None
        assert record["positive_windows"] == 0
    else:
        assert record["blocker"] == "no_pass_cut"
        assert record["positive_windows"] == 1
    assert peptide not in json.dumps(axes)
    assert axes["mhc_i"]["blocker"] == "predictor_not_configured"
    assert axes["mhc_ii"]["measurement"] is None


def test_a_positive_calis_score_is_named_and_is_not_a_hit(tmp_path: Path) -> None:
    output = tmp_path / "positive"
    _write_cif(output, "binder_0", _UBIQUITIN)
    _write_metrics(output, ["binder_0"])
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "untested"
    assert record["hits"] == []
    assert "immunogenicity" in record["untested"]
    assert "ada" in record["untested"]
    immunogenicity = record["sequence_liability"]["immunogenicity"]
    ada = record["sequence_liability"]["ada"]
    assert immunogenicity["measurement"] == 0.29826
    assert immunogenicity["positive_windows"] == 32
    # No class II table, so ada has nothing to read and says which input is missing.
    assert ada["measurement"] is None
    assert ada["blocker"] == "predictor_not_configured"
    measured = format_screen_lines("8TLD", "2", record)[1]
    assert "immunogenicity untested 0.29826, Calis 2013 class I 9-mer, positive 9-mers 32" in measured
    assert "immunogenicity hit" not in measured
    assert "immunogenicity clear" not in measured
    assert "cure" not in measured.casefold()
    assert "no_pass_cut" not in measured
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert _UBIQUITIN not in text


def test_a_positive_calis_window_is_stored_and_is_not_a_protein_hit(tmp_path: Path) -> None:
    axes, _by_example = measure_sequence_liability([("binder_0", _UBIQUITIN)], directory=tmp_path)
    record = axes["immunogenicity"]
    assert record["state"] == "untested"
    assert record["blocker"] == "no_pass_cut"
    assert record["measurement"] == 0.29826
    assert record["positive_windows"] == 32
    assert record["positive_examples"] == ["binder_0"]
    assert record["method"] == "calis_2013_class_i_9mer"
    decoy, _by_example = measure_sequence_liability([("binder_0", _DECOY)], directory=tmp_path)
    assert decoy["immunogenicity"]["state"] == "untested"
    assert decoy["immunogenicity"]["measurement"] == 0.35891
    assert decoy["immunogenicity"]["positive_windows"] == 4
    assert _UBIQUITIN not in json.dumps(axes)


def test_one_positive_sample_keeps_immunogenicity_untested(tmp_path: Path) -> None:
    axes, _by_example = measure_sequence_liability(
        [("binder_0", "LLLLLLLLL"), ("binder_1", "FIAGLIAIV")],
        directory=tmp_path,
    )
    record = axes["immunogenicity"]
    assert record["state"] == "untested"
    assert record["blocker"] == "no_pass_cut"
    assert record["measurement"] == 0.27206
    assert record["positive_windows"] == 1
    assert record["positive_examples"] == ["binder_1"]
    clear, _by_example = measure_sequence_liability(
        [("binder_0", "LLLLLLLLL"), ("binder_1", "QQQQQQQQQ")],
        directory=tmp_path,
    )
    assert clear["immunogenicity"]["state"] == "clear"
    assert clear["immunogenicity"]["measurement"] == -0.05184
    assert clear["immunogenicity"]["positive_windows"] == 0
    assert clear["immunogenicity"]["positive_examples"] == []
    short, _by_example = measure_sequence_liability(
        [("binder_0", "LLLLLLLLL"), ("binder_1", "AAAA")],
        directory=tmp_path,
    )
    assert short["immunogenicity"]["state"] == "untested"
    assert short["immunogenicity"]["blocker"] == "sequence_absent"
    assert short["immunogenicity"]["measurement"] == -0.05184


def test_a_nonstandard_residue_splits_calis_windows(tmp_path: Path) -> None:
    axes, _by_example = measure_sequence_liability(
        [("binder_0", "LLLLLLLLLXFIAGLIAIV")],
        directory=tmp_path,
    )
    assert axes["immunogenicity"]["state"] == "untested"
    assert axes["immunogenicity"]["measurement"] == 0.27206
    assert axes["immunogenicity"]["positive_windows"] == 1
    broken, _by_example = measure_sequence_liability([("binder_0", "LLLLLLLLLX")], directory=tmp_path)
    assert broken["immunogenicity"]["state"] == "clear"
    assert broken["immunogenicity"]["measurement"] == -0.05184


def _write_reference(
    folder: Path,
    stem: str,
    rows: list[tuple[str, str, int, float]],
    *,
    alleles: tuple[str, ...] = ("HLA-A*02:01",),
) -> None:
    """A reference table: protein, peptide, protein length, rank.

    THE RANKS HERE ARE CONSTRUCTED. The proteins are the tolerated human ones
    the density bar is taken from, and no design sequence appears.

    The allele set must equal the design table's, because a scan over one
    allele is not a bar for a scan over seven. One row per peptide per allele
    keeps the distinct peptide count, and so every density, unchanged.
    """
    lines = ["protein\tpeptide\tlength\tallele\tmethod\tpercentile_rank"]
    lines += [
        f"{name}\t{peptide}\t{length}\t{allele}\tconsensus\t{rank}"
        for name, peptide, length, rank in rows
        for allele in alleles
    ]
    (folder / f"{stem}_reference.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_design_table(
    folder: Path,
    stem: str,
    peptides: list[str],
    *,
    rank: float = 0.5,
    alleles: tuple[str, ...] = ("HLA-A*02:01",),
    covers: str | None = None,
) -> None:
    """A class table for the design, naming the peptides it scored.

    The allele column is here because a real IEDB or NetMHCpan export carries
    one, and the density comparison refuses a table that does not say which
    alleles it was scored over. One row per peptide per allele, so the distinct
    peptide count -- the density's numerator -- does not depend on the panel.

    ``covers`` is the design the table was scanned from. Every remaining window
    of it is written as a non-binder, which is what a complete scan looks like.
    Without it the table names binders alone, and a count of those over the
    whole chain is the understatement that only ever errs toward clear, so the
    density refuses to clear on one.
    """
    lines = ["peptide\tallele\tmethod\tpercentile_rank"]
    lines += [f"{peptide}\t{allele}\tconsensus\t{rank}" for peptide in peptides for allele in alleles]
    if covers is not None:
        width = len(peptides[0])
        windows = {covers[at : at + width] for at in range(len(covers) - width + 1)}
        lines += [
            f"{window}\t{allele}\tconsensus\t90.0" for window in sorted(windows - set(peptides)) for allele in alleles
        ]
    (folder / f"{stem}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_density_at_or_below_the_reference_clears(tmp_path: Path) -> None:
    """Albumin, IgG and ubiquitin set the bar, not an absolute cutoff.

    The design is 100 residues with two binding peptides, so 2.0 per 100. The
    reference protein is 100 residues with three, so 3.0. Two is not worse than
    three, so the axis clears.
    """
    # The peptides are substrings of the design, as a real scan's peptides are.
    design = "A" * 45 + "K" * 45 + "G" * 10
    assert len(design) == 100
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA", "KKKKKKKKK"], covers=design)
    _write_reference(
        tmp_path,
        "mhc_i",
        [("P02768", "KKKKKKKKK", 100, 0.2), ("P02768", "KKKKKKKKR", 100, 0.3), ("P02768", "KKKKKKKRR", 100, 0.4)],
    )
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "clear"
    # Both sides counted over 9-mer windows: two binders in the design's
    # 100 - 9 + 1, three in the reference protein's.
    assert record["measurement"] == _round(2 * 100 / (100 - 9 + 1))
    assert record["reference_density"] == _round(3 * 100 / (100 - 9 + 1))
    assert record["binding_peptides"] == 2
    assert record["method"] == "binding_peptides_per_100_windows"
    assert record["reference_proteins"] == ["P02768"]


def test_a_filtered_reference_export_cannot_set_the_bar(tmp_path: Path) -> None:
    """The reviewed blocker: a binders-only export inflates the bar sixfold.

    A reference table's peptide ends only reach the protein's length when the
    scan covered the whole protein. A binders-only export stops at the last
    binder, and dividing by that measured 6.25 binders per 100 residues where
    the complete scan of the same protein measured 0.49. An inflated bar always
    errs toward clearing the design, so an incomplete scan is refused.
    """
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    # Three binders ending at 28, 38 and 48 of a 609-residue protein, with the
    # first row starting well inside it: that is a filtered export.
    (tmp_path / "mhc_i_reference.tsv").write_text(
        "protein\tpeptide\tstart\tend\tallele\tmethod\tpercentile_rank\n"
        "P02768\tKKKKKKKKK\t20\t28\tHLA-A*02:01\tconsensus\t0.2\n"
        "P02768\tKKKKKKKKR\t30\t38\tHLA-A*02:01\tconsensus\t0.3\n"
        "P02768\tKKKKKKKRR\t40\t48\tHLA-A*02:01\tconsensus\t0.4\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    assert axes["mhc_i"]["blocker"] == "reference_length_unknown"
    assert "reference_density" not in axes["mhc_i"]

    # The same rows with a scan that starts at the N-terminus are usable.
    (tmp_path / "mhc_i_reference.tsv").write_text(
        # Both rows are 9-mers, because the design table is: a count over one
        # peptide width is not a bar for a count over several.
        "protein\tpeptide\tstart\tend\tallele\tmethod\tpercentile_rank\n"
        "P02768\tMKWVTFISL\t1\t9\tHLA-A*02:01\tconsensus\t0.2\n"
        "P02768\tKKKKKKKKR\t30\t600\tHLA-A*02:01\tconsensus\t0.3\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    assert axes["mhc_i"]["state"] in {"clear", "hit"}
    # Two binders over the 9-mer windows of a 600-residue protein, which is
    # 600 - 9 + 1 of them. A count of 9-mers belongs over the windows a 9-mer
    # can occupy, not over the residues.
    assert axes["mhc_i"]["reference_density"] == _round(2 * 100 / (600 - 9 + 1))


def test_a_peptide_length_column_is_not_a_protein_length(tmp_path: Path) -> None:
    """Some exports call the peptide's own length ``length``.

    Dividing by nine instead of by the protein's length inflates the bar by
    nearly two orders of magnitude, so a stated length no larger than the
    longest peptide is ignored rather than used.
    """
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    (tmp_path / "mhc_i_reference.tsv").write_text(
        "protein\tpeptide\tlength\tallele\tmethod\tpercentile_rank\n"
        "P02768\tKKKKKKKKK\t9\tHLA-A*02:01\tconsensus\t0.2\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    assert axes["mhc_i"]["blocker"] == "reference_length_unknown"


def test_a_reference_scanned_on_other_alleles_is_not_a_bar(tmp_path: Path) -> None:
    """A count over one allele against a count over seven is not a comparison.

    The wider scan carries more binders purely by having been asked more
    questions, which inflates the bar and errs toward clearing.
    """
    (tmp_path / "mhc_i.tsv").write_text(
        "allele\tpeptide\tmethod\tpercentile_rank\nHLA-A*02:01\tAAAAAAAAA\tconsensus\t0.5\n",
        encoding="utf-8",
    )
    rows = "".join(f"P02768\tKKKKKKKK{index}\t{index}\t100\tconsensus\t0.2\n" for index in range(7))
    (tmp_path / "mhc_i_reference.tsv").write_text(
        "protein\tpeptide\tstart\tlength\tmethod\tpercentile_rank\n" + rows,
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    # The reference names no allele column. Reading that as agreement is how
    # deleting one column turned this refusal into a measured clear, so an
    # undeclared set is its own refusal rather than a pass.
    assert axes["mhc_i"]["blocker"] == "reference_allele_unknown"
    assert axes["mhc_i"]["reference_alleles"] == []
    (tmp_path / "mhc_i_reference.tsv").write_text(
        "protein\tpeptide\tallele\tstart\tlength\tmethod\tpercentile_rank\n"
        + "".join(
            f"P02768\tKKKKKKKK{index}\tHLA-DRB1*03:0{index}\t{index}\t100\tconsensus\t0.2\n" for index in range(7)
        ),
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    assert axes["mhc_i"]["blocker"] == "reference_allele_mismatch"
    assert axes["mhc_i"]["design_alleles"] == ["A0201"], "normalised, so HLA- is dropped"


def test_a_byte_order_mark_does_not_delete_a_column(tmp_path: Path) -> None:
    """Excel's "CSV UTF-8" writes one by default, and it is not whitespace.

    It stayed on the first column's name, so no lookup could reach that column
    and a table plainly holding an allele column reported having none.
    """
    rows = _panel_rows(_HA_306_318, [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0])
    lines = ["allele\tpeptide\tmethod\tpercentile_rank"]
    lines += [f"{allele}\t{peptide}\tconsensus\t{rank}" for allele, peptide, rank in rows]
    (tmp_path / "mhc_ii.tsv").write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability(_ada_design(), directory=tmp_path)
    assert axes["ada"]["state"] == "clear", "the mark must not hide the allele column"
    assert axes["ada"]["measurement"] == 40.0


def test_a_leading_banner_line_does_not_defeat_the_header(tmp_path: Path) -> None:
    """A predictor's export often opens with a generated-by line."""
    rows = _panel_rows(_HA_306_318, [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0])
    lines = ["# generated by a predictor", "", "allele\tpeptide\tmethod\tpercentile_rank"]
    lines += [f"{allele}\t{peptide}\tconsensus\t{rank}" for allele, peptide, rank in rows]
    (tmp_path / "mhc_ii.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability(_ada_design(), directory=tmp_path)
    assert axes["ada"]["state"] == "clear"


def test_the_iedb_consensus_rank_column_is_read(tmp_path: Path) -> None:
    """IEDB's consensus export has no bare rank column at all.

    It names the column after the method it combined, so a reader wanting only
    ``percentile_rank`` saw a table full of ranks as a table with none -- and
    then listed every allele in the file as missing.
    """
    rows = _panel_rows(_HA_306_318, [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0])
    lines = ["allele\tpeptide\tconsensus_percentile_rank"]
    lines += [f"{allele}\t{peptide}\t{rank}" for allele, peptide, rank in rows]
    (tmp_path / "mhc_ii.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability(_ada_design(), directory=tmp_path)
    record = axes["ada"]
    assert record["state"] == "clear", record.get("blocker")
    assert record["measurement"] == 40.0
    assert "missing_alleles" not in record


def test_a_fragment_fasta_is_not_a_proteome(tmp_path: Path, tiny_proteome_ok: None) -> None:
    """Records that parse but hold no 9-mer find nothing, which read as a clear."""
    fasta = tmp_path / "fragments.fasta"
    fasta.write_text(">sp|P1|F\nMKV\n>sp|P2|F\nACDE\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)],
        directory=tmp_path,
        proteome=fasta,
    )
    assert axes["proteome_offtarget"]["state"] == "untested"
    assert axes["proteome_offtarget"]["blocker"] == "proteome_empty"


def test_a_class_ii_table_from_another_design_does_not_clear_ada(tmp_path: Path) -> None:
    """Nothing tied the class II table to the design before this."""
    _write_dr_panel(tmp_path, _panel_rows("WWWWWWWWWWWWW", [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]))
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)],
        directory=tmp_path,
    )
    assert axes["ada"]["state"] == "untested"
    assert axes["ada"]["blocker"] == "table_not_from_this_design"


def _round(value: float) -> float:
    return float(f"{value:.4f}")


def test_a_table_scanned_from_another_design_is_refused(tmp_path: Path) -> None:
    """A stale or reused table would give a number about two different proteins.

    The peptides in a class table are substrings of the chain it was scanned
    from. If none of them occur in any sample here, the count belongs to some
    other molecule and dividing it by this design's length is meaningless.
    """
    _write_design_table(tmp_path, "mhc_i", ["WWWWWWWWW", "WWWWWWWWK"])
    _write_reference(tmp_path, "mhc_i", [("P02768", "KKKKKKKKK", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "table_not_from_this_design"


def test_a_nonbinding_table_from_another_design_does_not_clear(tmp_path: Path) -> None:
    """Peptides above the cutoff still have to be this design's windows.

    An empty binding set used to become a density of zero, and zero is not
    above a reference, so a table scanned from another protein cleared this one.
    """
    _write_design_table(tmp_path, "mhc_i", ["WWWWWWWWW", "WWWWWWWWK"], rank=50.0)
    _write_reference(tmp_path, "mhc_i", [("P02768", "KKKKKKKKK", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "table_not_from_this_design"

    peptide = tmp_path / "peptide"
    peptide.mkdir()
    _write_design_table(peptide, "mhc_i", ["WWWWWWWWW"], rank=0.4)
    window, _by_example = measure_sequence_liability(
        [("binder_0", "AAAAAAAAA")],
        directory=peptide,
        binder_kind="peptide",
    )
    assert window["mhc_i"]["state"] == "untested"
    assert window["mhc_i"]["blocker"] == "table_not_from_this_design"


def test_a_scan_of_this_design_with_nothing_binding_is_a_real_zero(tmp_path: Path) -> None:
    """Nothing bound, and the peptides are this chain's, so the density is zero."""
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"], rank=50.0)
    _write_reference(tmp_path, "mhc_i", [("P02768", "KKKKKKKKK", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "clear"
    assert record["measurement"] == 0.0
    assert record["blocker"] is None


def test_an_unnamed_binding_rank_does_not_clear_a_protein(tmp_path: Path) -> None:
    """A binding rank with no peptide is not a density of zero.

    One non-binding window of this design used to make that unnamed rank
    disappear, and the axis cleared against the reference.
    """
    (tmp_path / "mhc_i.tsv").write_text(
        "peptide\tpercentile_rank\nAAAAAAAAA\t50\n\t0.2\n",
        encoding="utf-8",
    )
    _write_reference(tmp_path, "mhc_i", [("P02768", "KKKKKKKKK", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "peptide_column_absent"


def test_the_density_denominator_is_the_chain_that_was_scanned(tmp_path: Path) -> None:
    """Numerator and denominator have to be the same molecule.

    Two samples of different lengths, and the table's peptides occur only in the
    shorter one. Dividing by the longer chain would understate the density of
    the chain actually scanned.
    """
    scanned = "A" * 50
    other = "W" * 200
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    _write_reference(tmp_path, "mhc_i", [("P02768", "KKKKKKKKK", 100, 0.2)])
    axes, _by_example = measure_sequence_liability(
        [("binder_0", other), ("binder_1", scanned)],
        directory=tmp_path,
    )
    record = axes["mhc_i"]
    assert record["binding_peptides"] == 1
    assert record["measurement"] == _round(100 / (50 - 9 + 1)), "the fifty-residue chain's windows, not the other's"


def test_a_density_above_the_reference_is_a_hit(tmp_path: Path) -> None:
    """More MHC-visible than a protein people tolerate is the liability."""
    design = "A" * 25 + "K" * 25 + "G" * 25 + "W" * 25
    _write_design_table(
        tmp_path,
        "mhc_i",
        ["AAAAAAAAA", "KKKKKKKKK", "GGGGGGGGG", "WWWWWWWWW"],
    )
    _write_reference(tmp_path, "mhc_i", [("P0CG47", "KKKKKKKKK", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "hit"
    # Four binders over the design's 9-mer windows against one over the
    # reference protein's, both counted the same way.
    assert record["measurement"] == _round(4 * 100 / (100 - 9 + 1))
    assert record["reference_density"] == _round(100 / (100 - 9 + 1))


def test_a_metadata_export_is_not_a_proteome(tmp_path: Path) -> None:
    """A UniProtKB download with the Sequence column unticked still parses.

    No ``>`` line anywhere, so the whole file is one headerless record of
    protein names, and its handful of 9-mers answer every query with "no
    match". That read as a measured clear against the human proteome.
    """
    rows = ["Entry\tEntry Name\tProtein names\tGene Names\tOrganism\tLength"]
    rows += [f"P0{index:04d}\tNAME{index}\tATP synthase subunit alpha\tATP5F1A\tHuman\t553" for index in range(400)]
    fasta = tmp_path / "metadata.tsv"
    fasta.write_text("\n".join(rows) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", _DECOY * 5)], directory=tmp_path, proteome=fasta)
    record = axes["proteome_offtarget"]
    assert record["state"] == "untested"
    assert record["blocker"] == "proteome_too_small"


def test_a_coding_fasta_is_not_a_proteome(tmp_path: Path) -> None:
    """``cds.all.fa`` where ``pep.all.fa`` was meant.

    A coding FASTA packs into millions of 9-mers that are not peptides, so it
    is large enough to pass a size floor and still matches nothing.
    """
    fasta = tmp_path / "cds.fa"
    fasta.write_text("".join(f">tx{index}\n{'ACGT' * 1000}\n" for index in range(400)), encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", _DECOY * 5)], directory=tmp_path, proteome=fasta)
    record = axes["proteome_offtarget"]
    assert record["state"] == "untested"
    assert record["blocker"] == "proteome_not_protein"


def test_a_single_protein_is_not_a_proteome(tmp_path: Path) -> None:
    """One albumin fragment carries 52 9-mers, and clearing against it is not
    clearing against the proteome."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(f">P02768 fragment\n{_UBIQUITIN}\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", _DECOY * 5)], directory=tmp_path, proteome=fasta)
    assert axes["proteome_offtarget"]["blocker"] == "proteome_too_small"


def test_a_proteome_large_enough_still_measures(tmp_path: Path) -> None:
    """The floor must admit a real proteome, so the passing side is pinned too.

    A pseudo-random protein of 1.2 million residues carries about as many
    distinct 9-mers, which is the scale of the human reference. The design's
    own windows are astronomically unlikely to occur in it, so this clears.
    """
    import random

    source = random.Random(20261108)
    residues = "".join(source.choice(_DECOY) for _ in range(1_200_000))
    fasta = tmp_path / "big.fasta"
    fasta.write_text(">synthetic\n" + residues + "\n", encoding="utf-8")
    blocker, kmers = _load_proteome(fasta)
    assert blocker is None
    assert kmers.size >= 1_000_000
    axes, _by_example = measure_sequence_liability(
        [("binder_0", "WWWWWWWWWWWWWWWWWWWW")], directory=tmp_path, proteome=fasta
    )
    record = axes["proteome_offtarget"]
    assert record["state"] == "clear"
    # The size is part of the claim, so a clear records what it was measured against.
    assert record["proteome_kmers"] == int(kmers.size)


def test_a_comma_decimal_is_refused_rather_than_dropped(tmp_path: Path) -> None:
    """A dropped cell always loses a binder, so it cannot pass as absent.

    German and French Excel write 0.5 as ``0,5``. Skipping what will not parse
    made that indistinguishable from an empty cell: every fractional rank went,
    the integral non-binders stayed, and a table of three binders reported a
    best rank of 5.0 and cleared at a density of zero.
    """
    design = "A" * 50 + "K" * 50
    lines = ["peptide\tallele\tmethod\tpercentile_rank"]
    lines.append("AAAAAAAAA\tHLA-A*02:01\tconsensus\t0,5")
    lines.append("KKKKKKKKK\tHLA-A*02:01\tconsensus\t5")
    (tmp_path / "mhc_i.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_reference(tmp_path, "mhc_i", [("P02768", "RRRRRRRRR", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "cell_unparsed"
    assert record["unparsed_columns"] == ["percentile_rank"]


def test_a_recognised_no_value_cell_is_still_not_a_refusal(tmp_path: Path) -> None:
    """``n/a`` and a bare dash say "no number", which is not an unreadable one.

    Refusing these would refuse real exports, so the two cases are kept apart.
    """
    # The two no-value rows name peptides the design does not contain, so the
    # scan still covers every window of it. A declined window that IS in the
    # chain is a different matter and is reported as unscored_windows: the
    # predictor having skipped it is not the same as it holding no binder.
    design = "A" * 50 + "W" * 50
    lines = ["peptide\tallele\tmethod\tpercentile_rank"]
    lines.append("AAAAAAAAA\tHLA-A*02:01\tconsensus\t0.5")
    lines.append("KKKKKKKKK\tHLA-A*02:01\tconsensus\tn/a")
    lines.append("GGGGGGGGG\tHLA-A*02:01\tconsensus\t-")
    # The rest of the chain's windows, scored and not binding, so the file is a
    # complete scan: a density that clears has to rest on one.
    named = {"AAAAAAAAA", "KKKKKKKKK", "GGGGGGGGG"}
    for at in range(len(design) - 9 + 1):
        window = design[at : at + 9]
        if window not in named:
            named.add(window)
            lines.append(f"{window}\tHLA-A*02:01\tconsensus\t90.0")
    (tmp_path / "mhc_i.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_reference(tmp_path, "mhc_i", [("P02768", "RRRRRRRRR", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["blocker"] is None
    assert record["rank"] == 0.5


def test_a_repeated_rank_column_is_refused(tmp_path: Path) -> None:
    """csv keeps the last of two columns with one name, and says nothing.

    A header carrying the consensus rank twice -- the real ranks in one column
    and another method's in the next -- lost the column holding the binders,
    and the four binders at 0.4 were reported as non-binders at 8.0.
    """
    design = "A" * 50 + "K" * 50
    lines = ["peptide\tallele\tmethod\tpercentile_rank\tpercentile_rank"]
    lines.append("AAAAAAAAA\tHLA-A*02:01\tconsensus\t0.4\t8.0")
    (tmp_path / "mhc_i.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_reference(tmp_path, "mhc_i", [("P02768", "RRRRRRRRR", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "predictor_unreadable"


def test_an_absent_method_or_allele_column_is_not_agreement(tmp_path: Path) -> None:
    """Deleting one column must not turn a refusal into a measured clear.

    Both guards read an absent set as agreement, so dropping the design table's
    method column compared an IEDB consensus density against a NetMHCpan bar,
    and dropping its allele column compared one allele against seven. Neither
    comparison means anything, and both cleared.
    """
    design = "A" * 50 + "K" * 50
    reference = [("P02768", f"REF{index:06d}", 100, 0.2) for index in range(3)]

    # No method column on the design side.
    folder = tmp_path / "nomethod"
    folder.mkdir()
    (folder / "mhc_i.tsv").write_text(
        "peptide\tallele\tpercentile_rank\nAAAAAAAAA\tHLA-A*02:01\t0.6\n", encoding="utf-8"
    )
    _write_reference(folder, "mhc_i", reference)
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=folder)
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "reference_method_unknown"

    # No allele column on the design side.
    folder = tmp_path / "noallele"
    folder.mkdir()
    (folder / "mhc_i.tsv").write_text("peptide\tmethod\tpercentile_rank\nAAAAAAAAA\tconsensus\t0.6\n", encoding="utf-8")
    _write_reference(folder, "mhc_i", reference)
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=folder)
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "reference_allele_unknown"


def test_a_reference_over_an_untolerated_protein_is_not_a_bar(tmp_path: Path) -> None:
    """The bar is what tolerated proteins carry, so only they may set it.

    A density over some other protein is a bar with no meaning, and one over a
    protein with high epitope content inflates it, which errs toward clearing.
    Q9Y6K9 is a real accession and not one of the three, which is the point.
    """
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    _write_reference(tmp_path, "mhc_i", [("Q9Y6K9", f"REF{index:06d}", 100, 0.2) for index in range(9)])
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_not_tolerated_protein"
    # The refusal names what was found, so the remedy is to score the three.
    assert record["reference_proteins"] == ["Q9Y6K9"]


def test_an_untolerated_protein_cannot_raise_the_bar(tmp_path: Path) -> None:
    """A stray row is dropped from the bar, not allowed to lift it.

    The design's 1.0 per 100 is above albumin's 0.0 and below the other
    protein's 9.0. Were the stray protein counted, the highest density would be
    the bar and the design would clear on a protein nobody tolerates.
    """
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    rows = [("P02768", "KKKKKKKKK", 100, 5.0)]
    rows += [("Q9Y6K9", f"REF{index:06d}", 100, 0.2) for index in range(9)]
    _write_reference(tmp_path, "mhc_i", rows)
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "hit"
    assert record["reference_density"] == 0.0
    assert record["reference_proteins"] == ["P02768"]


def test_an_isoform_suffix_and_a_fasta_header_still_name_albumin(tmp_path: Path) -> None:
    """``P02768-1`` and ``sp|P02768|ALBU_HUMAN`` are the same protein.

    Both are what a real export writes, so neither may be read as a protein
    outside the tolerated set.
    """
    design = "A" * 50 + "K" * 50
    for label in ("P02768-1", "sp|P02768|ALBU_HUMAN"):
        folder = tmp_path / label.replace("|", "_")
        folder.mkdir()
        _write_design_table(folder, "mhc_i", ["AAAAAAAAA"], covers=design)
        _write_reference(folder, "mhc_i", [(label, "KKKKKKKKK", 100, 0.2)])
        axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=folder)
        record = axes["mhc_i"]
        assert record["blocker"] is None, label
        assert record["reference_density"] == _round(100 / (100 - 9 + 1)), label


def test_rows_pooled_under_one_accession_are_refused(tmp_path: Path) -> None:
    """More distinct windows than the protein has is more than one protein.

    A real export that scored albumin, transferrin and C3 and tagged every row
    with the first accession divides three molecules' binders by one molecule's
    residues. Albumin holds 92 distinct 9-mers at 100 residues; 150 of them did
    not come from one albumin.
    """
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    _write_reference(tmp_path, "mhc_i", [("P02768", f"REF{index:06d}", 100, 0.2) for index in range(150)])
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_rows_exceed_protein"
    assert record["reference_proteins_pooled"] == ["P02768"]


def test_a_reference_row_beyond_the_protein_is_refused(tmp_path: Path) -> None:
    """A peptide cannot end past the protein it is said to be a window of.

    This catches the pooled export that a window count cannot: a binders-only
    scan of three proteins under one label carries few rows, but the rows from
    the longer proteins start beyond the first one's last residue.
    """
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    lines = ["protein\tpeptide\tlength\tstart\tallele\tmethod\tpercentile_rank"]
    lines.append("P02768\tKKKKKKKKK\t609\t101\tHLA-A*02:01\tconsensus\t0.2")
    # Albumin's last residue is 609, so a window starting at 1200 is another protein's.
    lines.append("P02768\tRRRRRRRRR\t609\t1200\tHLA-A*02:01\tconsensus\t0.2")
    (tmp_path / "mhc_i_reference.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    # Two rows at two distinct starts are one scan, so the stated length is what
    # is wrong. Pooling repeats positions, and gets its own blocker.
    assert record["blocker"] == "reference_length_inconsistent"
    assert record["observed_extent"]["P02768"] == 1208


def test_rows_repeating_a_position_are_pooled_not_mislabelled(tmp_path: Path) -> None:
    """Several proteins under one accession each start at residue one.

    So a position appears more than once at a width, which one scan of one
    protein cannot do. That is pooling, and it keeps the pooling blocker.
    """
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    lines = ["protein\tpeptide\tlength\tstart\tallele\tmethod\tpercentile_rank"]
    for index in range(40):
        # Two molecules scanned from residue one, both tagged P02768.
        start = index + 1
        lines.append(f"P02768\tAA{index:07d}\t50\t{start}\tHLA-A*02:01\tconsensus\t0.2")
        lines.append(f"P02768\tBB{index:07d}\t50\t{start}\tHLA-A*02:01\tconsensus\t0.2")
    (tmp_path / "mhc_i_reference.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_rows_exceed_protein"


def test_a_scan_over_several_peptide_widths_is_not_pooled(tmp_path: Path) -> None:
    """A class I scan covers 8- to 11-mers, and each width has its own windows.

    Counting every width together would make one honest albumin scan look like
    four proteins pooled, so the bound is per width. A 100-residue protein holds
    93 8-mers and 90 11-mers; all four widths at once is 366 rows and no pooling.
    """
    design = "A" * 50 + "K" * 50
    # The design is scanned at the same four widths, because a count over four
    # widths is not comparable with a count over one.
    _write_design_table(tmp_path, "mhc_i", ["A" * width for width in (8, 9, 10, 11)])
    lines = ["protein\tpeptide\tlength\tallele\tmethod\tpercentile_rank"]
    for width in (8, 9, 10, 11):
        windows = 100 - width + 1
        for index in range(windows):
            # Exactly ``width`` characters, so the row's width is what it claims.
            peptide = f"{index:0{width}d}"
            assert len(peptide) == width
            lines.append(f"P02768\t{peptide}\t100\tHLA-A*02:01\tconsensus\t5.0")
    (tmp_path / "mhc_i_reference.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["blocker"] is None
    assert record["reference_proteins"] == ["P02768"]


def test_a_complete_scan_to_the_last_residue_is_not_pooled(tmp_path: Path) -> None:
    """The window at the very end is inside the protein, under either indexing.

    A guard that refused this would refuse every complete scan, so it is pinned
    for both the 1-based and the 0-based export.
    """
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"], covers=design)
    for first, last in ((1, 92), (0, 91)):
        folder = tmp_path / f"scan{first}"
        folder.mkdir()
        _write_design_table(folder, "mhc_i", ["AAAAAAAAA"], covers=design)
        lines = ["protein\tpeptide\tlength\tstart\tallele\tmethod\tpercentile_rank"]
        lines.append(f"P02768\tKKKKKKKKK\t100\t{first}\tHLA-A*02:01\tconsensus\t0.2")
        lines.append(f"P02768\tRRRRRRRRR\t100\t{last}\tHLA-A*02:01\tconsensus\t5.0")
        (folder / "mhc_i_reference.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=folder)
        assert axes["mhc_i"]["blocker"] is None, (first, last)


def test_without_a_reference_the_density_is_recorded_and_not_judged(tmp_path: Path) -> None:
    """No bar means no verdict. The number is still reported.

    This is the honest state of every run until the reference set is scored, and
    it is why these axes read untested rather than hit on a real design.
    """
    _write_design_table(tmp_path, "mhc_ii", ["AAAAAAAAA"], rank=5.0)
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    record = axes["mhc_ii"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_not_configured"
    # One binder over the 100-residue chain's 9-mer windows.
    assert record["measurement"] == _round(100 / (100 - 9 + 1))
    assert "reference_density" not in record


def test_a_reference_from_another_tool_is_not_a_bar(tmp_path: Path) -> None:
    """Two predictors disagree about which peptides bind, so one is not a bar for the other."""
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    (tmp_path / "mhc_i_reference.tsv").write_text(
        "protein\tpeptide\tlength\tmethod\tpercentile_rank\nP02768\tKKKKKKKKK\t100\tnetmhcpan_el\t0.2\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_method_mismatch"
    assert record["design_method"] == ["consensus"]
    assert record["reference_method"] == ["netmhcpan_el"]


def test_a_reference_without_a_length_cannot_be_a_density(tmp_path: Path) -> None:
    """A count is not a density until it is divided by a length."""
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    (tmp_path / "mhc_i_reference.tsv").write_text(
        "protein\tpeptide\tallele\tmethod\tpercentile_rank\n" "P02768\tKKKKKKKKK\tHLA-A*02:01\tconsensus\t0.2\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", "A" * 100)], directory=tmp_path)
    assert axes["mhc_i"]["blocker"] == "reference_length_unknown"


def test_a_window_that_is_not_presented_is_not_scored(tmp_path: Path) -> None:
    """The whole point of restricting the scan, on one sequence.

    ``LLLLLLLLLXFIAGLIAIV`` has two 9-mer runs: one scoring -0.05184 and one
    scoring 0.27206. Unrestricted, that positive window keeps the axis untested
    whatever else is true. With a class I table that presents only the negative
    one, the positive window is not a presented ligand and the design clears.
    """
    design = [("binder_0", "LLLLLLLLLXFIAGLIAIV")]
    loose, _by_example = measure_sequence_liability(design, directory=tmp_path)
    assert loose["immunogenicity"]["state"] == "untested"
    assert loose["immunogenicity"]["blocker"] == "no_pass_cut"
    assert loose["immunogenicity"]["presentation"] == "not_configured"

    _write_class_i(tmp_path, ["LLLLLLLLL"])
    restricted, _by_example = measure_sequence_liability(design, directory=tmp_path)
    record = restricted["immunogenicity"]
    assert record["state"] == "clear"
    assert record["measurement"] == -0.05184
    assert record["blocker"] is None
    assert record["presentation"] == "restricted"
    assert record["presented_ligands"] == 1
    assert record["positive_windows"] == 0


def test_a_presented_ligand_above_zero_is_a_hit(tmp_path: Path) -> None:
    """Restricting the scan makes the axis two-sided, not permissive."""
    _write_class_i(tmp_path, ["FIAGLIAIV"])
    axes, _by_example = measure_sequence_liability(
        [("binder_0", "LLLLLLLLLXFIAGLIAIV")],
        directory=tmp_path,
    )
    record = axes["immunogenicity"]
    assert record["state"] == "hit"
    assert record["measurement"] == 0.27206
    assert record["positive_windows"] == 1
    assert record["positive_examples"] == ["binder_0"]
    assert record["presented_ligands"] == 1


def test_the_immunogenicity_line_matches_what_was_scored(tmp_path: Path) -> None:
    """A hit is not described as every 9-mer being at most 0.

    The unrestricted clear is the sentence that names every 9-mer. A restricted
    clear only scored the presented ligands, so it says so.
    """
    hit_dir = tmp_path / "hit"
    _write_cif(hit_dir, "binder_0", "LLLLLLLLLXFIAGLIAIV")
    _write_metrics(hit_dir, ["binder_0"])
    _write_class_i(hit_dir, ["FIAGLIAIV"])
    hit = screen_generation(hit_dir, binder_kind="protein", target_id="IL5")
    hit_line = format_screen_lines("8TLD", "2", hit)[1]
    assert "immunogenicity hit 0.27206, Calis 2013 class I 9-mer, positive 9-mers 1" in hit_line
    assert "every 9-mer <= 0" not in hit_line

    clear_dir = tmp_path / "restricted"
    _write_cif(clear_dir, "binder_0", "LLLLLLLLLXFIAGLIAIV")
    _write_metrics(clear_dir, ["binder_0"])
    _write_class_i(clear_dir, ["LLLLLLLLL"])
    restricted = screen_generation(clear_dir, binder_kind="protein", target_id="IL5")
    clear_line = format_screen_lines("8TLD", "2", restricted)[1]
    assert "immunogenicity clear -0.05184, Calis 2013 class I 9-mer, every presented 9-mer <= 0" in clear_line
    assert "every 9-mer <= 0" not in clear_line


def test_a_presented_9mer_outside_the_design_is_not_a_clear(tmp_path: Path) -> None:
    """A presented 9-mer that is not in the design was not scored.

    Clearing on the ligand that is in the design would say every presented
    9-mer is at most 0, including the one the scan never reached.
    """
    output = tmp_path / "partial"
    _write_cif(output, "binder_0", "LLLLLLLLL")
    _write_metrics(output, ["binder_0"])
    _write_class_i(output, ["LLLLLLLLL", "FIAGLIAIV"])
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    immunogenicity = record["sequence_liability"]["immunogenicity"]
    assert immunogenicity["state"] == "untested"
    assert immunogenicity["blocker"] == "presented_not_in_design"
    text = "\n".join(format_screen_lines("8TLD", "2", record))
    assert "every presented 9-mer <= 0" not in text
    assert "immunogenicity clear" not in text


def test_nothing_to_score_names_which_of_the_three_reasons(tmp_path: Path) -> None:
    """Nothing for the model to score is not the same as nothing to worry about.

    Three different situations produced one blocker, which sent the user looking
    for data they had already supplied. Each now says which it is.
    """
    design = [("binder_0", _POLY_LEU)]

    # The table is this design's and named a 9-mer, but it binds nothing.
    _write_class_i(tmp_path, ["LLLLLLLLL"], rank=50.0)
    axes, _by_example = measure_sequence_liability(design, directory=tmp_path)
    assert axes["immunogenicity"]["blocker"] == "no_presented_ligand"
    assert axes["immunogenicity"]["measurement"] is None

    # It presented a ligand, just not one these 9-mer weights can score.
    _write_class_i(tmp_path, ["LLLLLLLLLL"])
    axes, _by_example = measure_sequence_liability(design, directory=tmp_path)
    assert axes["immunogenicity"]["blocker"] == "no_nine_mer_ligand"

    # The peptides are not windows of this design, so it is another design's scan.
    _write_class_i(tmp_path, ["YYYYYYYYY", "YYYYYYYYK"])
    axes, _by_example = measure_sequence_liability(design, directory=tmp_path)
    assert axes["immunogenicity"]["blocker"] == "table_not_from_this_design"


def test_a_class_i_table_without_peptides_cannot_restrict(tmp_path: Path) -> None:
    """Ranks for peptides the table does not name restrict nothing.

    That is the shape the table had before this, so it must not read as an empty
    presented set, which would clear every design by having nothing to score.
    """
    (tmp_path / "mhc_i.tsv").write_text("percentile_rank\n0.4\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", _UBIQUITIN)], directory=tmp_path)
    record = axes["immunogenicity"]
    assert record["presentation"] == "not_configured"
    assert record["state"] == "untested"
    assert record["positive_windows"] == 32, "the whole-chain scan, as information"
    # For a protein the same table also cannot give a density, because a rank
    # without the peptide it belongs to has no numerator to count.
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "peptide_column_absent"
    assert axes["mhc_i"]["measurement"] == 0.4, "the number it did hold"


def test_a_promiscuous_dr_binder_is_an_ada_hit(tmp_path: Path) -> None:
    """The Paul 2015 rule: median rank over the seven-allele panel at or below 20.

    Reaching every allele in the panel is the risk signal, because such a peptide
    finds a restricting allele in most people. The sorted ranks below are
    2/5/8/12/15/18/19, so the median is the fourth, 12.0.
    """
    _write_dr_panel(tmp_path, _panel_rows(_HA_306_318, [2.0, 5.0, 8.0, 12.0, 15.0, 18.0, 19.0]))
    axes, _by_example = measure_sequence_liability(_ada_design(), directory=tmp_path)
    record = axes["ada"]
    assert record["state"] == "hit"
    assert record["measurement"] == 12.0
    assert record["blocker"] is None
    assert record["method"] == "mhc_ii_dr_breadth_paul_2015"
    assert record["alleles"] == 7
    assert record["peptides_scored"] == 1
    assert record["promiscuous_peptides"] == 1
    assert record["binding_alleles"] == 7


def test_no_promiscuous_dr_binder_clears_ada(tmp_path: Path) -> None:
    """Above the median cutoff on every peptide is the clear, not an absence of data."""
    rows = _panel_rows(_HA_306_318, [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0])
    rows += _panel_rows(_NON_BINDER, [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0])
    _write_dr_panel(tmp_path, rows)
    axes, _by_example = measure_sequence_liability(_ada_design(), directory=tmp_path)
    record = axes["ada"]
    assert record["state"] == "clear"
    assert record["measurement"] == 40.0, "the best median across both peptides"
    assert record["peptides_scored"] == 2
    assert record["promiscuous_peptides"] == 0
    assert record["binding_alleles"] == 0


def test_an_allele_spelling_is_not_a_different_allele(tmp_path: Path) -> None:
    """A predictor writes the panel three different ways; it is still the panel."""
    spelled = ("HLA-DRB1*03:01", "DRB1_0701", "drb1*15:01", "DRB3*01:01", "DRB3_0202", "DRB4*01:01", "DRB5*01:01")
    _write_dr_panel(
        tmp_path,
        _panel_rows(_OVA_323_339, [2.0, 5.0, 8.0, 12.0, 15.0, 18.0, 19.0], alleles=spelled),
    )
    axes, _by_example = measure_sequence_liability(_ada_design(_OVA_323_339), directory=tmp_path)
    assert axes["ada"]["state"] == "hit"
    assert axes["ada"]["measurement"] == 12.0


def test_a_panel_with_no_rank_does_not_blame_the_alleles(tmp_path: Path) -> None:
    """The median rule reads the rank column, so affinity alone is not enough.

    Reporting that as an incomplete panel named the exact seven alleles the user
    had just scored, and the remedy for that blocker is to go and score them --
    a loop with no way out. The number is missing, not the alleles.
    """
    lines = ["allele\tpeptide\tmethod\tic50_nm"]
    lines += [f"{allele}\t{_HA_306_318}\tconsensus\t120\n" for allele in _DR_PANEL]
    (tmp_path / "mhc_ii.tsv").write_text(
        "".join(line if line.endswith("\n") else line + "\n" for line in lines), encoding="utf-8"
    )
    axes, _by_example = measure_sequence_liability(_ada_design(), directory=tmp_path)
    record = axes["ada"]
    assert record["state"] == "untested"
    assert record["blocker"] == "rank_absent"
    assert "missing_alleles" not in record, "the alleles are present; the rank is not"


def test_a_partial_dr_panel_cannot_be_a_median(tmp_path: Path) -> None:
    """A median over seven is not a median over four, so it stays untested.

    The blocker names the alleles that were missing, because the fix is to run
    the predictor over them rather than to guess at the rest.
    """
    short = _DR_PANEL[:4]
    _write_dr_panel(tmp_path, _panel_rows(_HA_306_318, [1.0, 2.0, 3.0, 4.0], alleles=short))
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    record = axes["ada"]
    assert record["state"] == "untested"
    assert record["blocker"] == "allele_panel_incomplete"
    assert record["measurement"] is None
    assert record["missing_alleles"] == ["DRB30202", "DRB40101", "DRB50101"]


def test_an_unknown_rank_scale_is_not_the_paul_threshold(tmp_path: Path) -> None:
    """20 is an IEDB consensus percentile. On a NetMHCIIpan EL rank it is not.

    That tool's own bands are 1 per cent strong and 5 per cent weak, so reading a
    median of 20 as permissive would invert the verdict. The table has to say
    which scale its ranks are on.
    """
    _write_dr_panel(
        tmp_path,
        _panel_rows(_HA_306_318, [2.0, 5.0, 8.0, 12.0, 15.0, 18.0, 19.0]),
        method="netmhciipan_el",
    )
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    assert axes["ada"]["state"] == "untested"
    assert axes["ada"]["blocker"] == "rank_scale_unknown"
    assert axes["ada"]["measurement"] is None


@pytest.mark.parametrize(
    "method",
    [
        "consensus",
        "consensus (comblib/smm/nn)",
        "IEDB recommended (Consensus)",
        "consensus3",
    ],
)
def test_the_real_consensus_method_names_are_accepted(tmp_path: Path, method: str) -> None:
    """The tool writes the components it combined into the method name.

    An exact match against a fixed set refused genuine IEDB output as an unknown
    scale, which is a false blocker on the one input the Paul threshold was
    calibrated for. The test is parameterised over the spellings the tool
    actually produces.
    """
    _write_dr_panel(
        tmp_path,
        _panel_rows(_HA_306_318, [2.0, 5.0, 8.0, 12.0, 15.0, 18.0, 19.0]),
        method=method,
    )
    axes, _by_example = measure_sequence_liability(_ada_design(), directory=tmp_path)
    assert axes["ada"]["state"] == "hit", method
    assert axes["ada"]["measurement"] == 12.0


@pytest.mark.parametrize("method", ["netmhciipan_el", "NetMHCIIpan 4.3 EL", "nn_align", ""])
def test_a_scale_that_is_not_consensus_is_still_refused(tmp_path: Path, method: str) -> None:
    """Widening the match must not turn into accepting any scale at all."""
    _write_dr_panel(
        tmp_path,
        _panel_rows(_HA_306_318, [2.0, 5.0, 8.0, 12.0, 15.0, 18.0, 19.0]),
        method=method,
    )
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    assert axes["ada"]["blocker"] == "rank_scale_unknown", method


def test_the_proteome_cache_does_not_grow_without_bound(tmp_path: Path, tiny_proteome_ok: None) -> None:
    """Each entry is tens of megabytes and the panel is a long-lived process."""
    from rfproteina.disease.liability import _PROTEOME_CACHE, _PROTEOME_CACHE_MAX

    fasta = tmp_path / "rolling.fasta"
    for index in range(_PROTEOME_CACHE_MAX + 3):
        fasta.write_text(f">r{index}\n{_SPAN}{'A' * index}\n", encoding="utf-8")
        blocker, table = _load_proteome(fasta)
        assert blocker is None
        assert table.size > 0
    assert len(_PROTEOME_CACHE) <= _PROTEOME_CACHE_MAX


def test_a_class_ii_table_without_the_columns_the_rule_needs(tmp_path: Path) -> None:
    """The old table shape had ranks but no allele or peptide, so it cannot be scored."""
    (tmp_path / "mhc_ii.tsv").write_text("percentile_rank\n5.0\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    assert axes["ada"]["blocker"] == "allele_column_absent"
    (tmp_path / "mhc_ii.tsv").write_text("allele\tpercentile_rank\nDRB1*03:01\t5.0\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    assert axes["ada"]["blocker"] == "peptide_column_absent"


def test_the_ada_record_stores_no_peptide(tmp_path: Path) -> None:
    """A peptide of a designed chain is sequence, so only counts are stored."""
    _write_dr_panel(tmp_path, _panel_rows(_HA_306_318, [2.0, 5.0, 8.0, 12.0, 15.0, 18.0, 19.0]))
    axes, _by_example = measure_sequence_liability([], directory=tmp_path)
    written = json.dumps(axes)
    assert _HA_306_318 not in written
    assert "PKYVKQ" not in written, "not even a fragment of it"
    assert axes["ada"]["peptides_scored"] == 1, "the count is what is kept"


def test_a_score_table_wins_over_the_sequence_formula(tmp_path: Path) -> None:
    (tmp_path / "immunogenicity.tsv").write_text("score\n1.5\n", encoding="utf-8")
    (tmp_path / "ada.tsv").write_text("score\n0.5\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", "LLLLLLLLL")], directory=tmp_path)
    assert axes["immunogenicity"] == {
        "state": "untested",
        "measurement": 1.5,
        "blocker": "no_pass_cut",
        "method": "supplied_table",
    }
    assert axes["ada"]["measurement"] == 0.5
    assert axes["ada"]["state"] == "untested"
    # The table's number is tagged as the table's, not as Calis or Kolaskar: the
    # sequence formula did not produce it and must not be credited with it.
    assert axes["immunogenicity"]["method"] == "supplied_table"
    assert axes["ada"]["method"] == "supplied_table"
    assert "calis" not in str(axes["immunogenicity"]).casefold()
    assert "kolaskar" not in str(axes["ada"]).casefold()


def test_an_anchor_motif_is_not_an_mhc_call(tmp_path: Path) -> None:
    axes, _by_example = measure_sequence_liability([("binder_0", "ALAAAAAAV")], directory=tmp_path)
    assert axes["mhc_i"] == {"state": "untested", "measurement": None, "blocker": "predictor_not_configured"}
    assert axes["mhc_ii"] == {"state": "untested", "measurement": None, "blocker": "predictor_not_configured"}
    assert axes["mhc_i"]["state"] != "hit"
    assert axes["mhc_i"]["state"] != "clear"


def test_a_peptide_proteome_hit_is_an_axis_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_proteome_ok: None
) -> None:
    output = tmp_path / "peptide"
    fasta = tmp_path / "human.fasta"
    fasta.write_text(f">piece\n{_NINE}\n", encoding="utf-8")
    monkeypatch.setenv("HUMAN_PROTEOME", str(fasta))
    _write_cif(output, "binder_0", _UBIQUITIN)
    _write_metrics(output, ["binder_0"])
    record = screen_generation(output, binder_kind="peptide", target_id="IL5")
    assert record["call"] == "axis_hit"
    assert record["hits"] == ["proteome_offtarget"]
    assert "sequence_liability" in record


def test_a_ligand_screen_omits_sequence_liability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "ligand"
    fasta = tmp_path / "human.fasta"
    fasta.write_text(f">piece\n{_NINE}\n", encoding="utf-8")
    monkeypatch.setenv("HUMAN_PROTEOME", str(fasta))
    (output / "mhc_i.tsv").parent.mkdir(parents=True)
    (output / "mhc_i.tsv").write_text("percentile_rank\n0.4\n", encoding="utf-8")
    _write_cif(output, "binder_0", _UBIQUITIN)
    _write_metrics(output, ["binder_0"])
    record = screen_generation(output, binder_kind="ligand", target_id="IL5")
    assert "sequence_liability" not in record
    assert record["call"] == "untested"
    assert "proteome_offtarget" not in record["states"]
    assert "mhc_i" not in record["states"]
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert _NINE not in text
    assert str(fasta) not in text


def test_empty_and_unreadable_proteome_files_stay_untested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty = tmp_path / "empty.fasta"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("HUMAN_PROTEOME", str(empty))
    job = tmp_path / "empty.job.json"
    job.write_text(json.dumps({"class_name": "requested_clear"}), encoding="utf-8")
    before = job.read_bytes()
    _write_cif(empty_dir, "binder_0", _DECOY)
    _write_metrics(empty_dir, ["binder_0"])
    empty_record = screen_generation(empty_dir, binder_kind="protein", target_id="IL5")
    assert job.read_bytes() == before
    assert empty_record["call"] == "untested"
    assert empty_record["sequence_liability"]["proteome_offtarget"]["blocker"] == "proteome_empty"
    assert empty_record["sequence_liability"]["proteome_offtarget"]["state"] == "untested"
    missing = tmp_path / "missing.fasta"
    monkeypatch.setenv("HUMAN_PROTEOME", str(missing))
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    missing_record = screen_generation(missing_dir, binder_kind="protein", target_id="IL5")
    assert missing_record["call"] == "untested"
    assert missing_record["sequence_liability"]["proteome_offtarget"]["blocker"] == "proteome_unreadable"


def _mhc(folder: Path, stem: str, column: str, value: str) -> tuple[str, float]:
    """One row at one cutoff, read as a peptide binder.

    The kind is peptide because that is where the per-row window rule is the
    verdict. For a protein the same cutoff is only the binder call that feeds a
    density, since every human protein has rows under it.

    The row names the peptide it scored, and the design is that peptide. Where
    the cutoff falls is what this helper is for, and a table naming no peptide
    is separately refused for a clear: nothing ties such a file to the
    molecule, so it cannot report an absence of binders in it. Without the
    name, every above-the-cutoff case here went through that refusal instead
    of through the cutoff it is meant to pin.
    """
    folder.mkdir()
    peptide = "MQIFVKTLT"
    (folder / f"{stem}.tsv").write_text(f"peptide\t{column}\n{peptide}\t{value}\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", peptide)], directory=folder, binder_kind="peptide")
    record = axes[stem]
    return str(record["state"]), float(record["measurement"])


def _write_cif(directory: Path, example_id: str, sequence: str) -> None:
    folder = directory / "inference_samples" / example_id
    folder.mkdir(parents=True, exist_ok=True)
    text = (
        "data_design\n#\nloop_\n"
        "_entity_poly.entity_id\n"
        "_entity_poly.pdbx_seq_one_letter_code_can\n"
        "_entity_poly.pdbx_strand_id\n"
        "0\n"
        f";{sequence}\n"
        ";\n"
        "A\n"
        "1\n"
        ";AAAAAAAAAA\n"
        ";\n"
        "B\n"
    )
    (folder / f"{example_id}_generated.cif").write_text(text, encoding="utf-8")


def _write_metrics(directory: Path, example_ids: list[str]) -> None:
    rows = ["example_id,chain_of_interest"]
    rows.extend(f"{example_id},A_1" for example_id in example_ids)
    path = directory / "inference_metrics" / "per_example_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# The zero-density guard and the reference bar. Mutation testing found both
# mechanisms unprotected: six single-line mutations of the zero-density guard
# and one of the bar cap all left the suite green, including replacing the
# coverage call with `if False`, which turns an incomplete scan into a measured
# `clear` at 0.0 per 100 residues. The guards are correct; what was missing is
# anything that notices their removal.
# ---------------------------------------------------------------------------

#: Fifty distinct two-letter blocks in a fixed order. Every 9-mer of this chain
#: is therefore distinct: a 9-mer spans five consecutive blocks, the blocks are
#: all different, and so the block sequence fixes the position. A chain of
#: repeats would hide the coverage guard entirely -- "A" * 100 holds ninety-two
#: 9-mer positions and one distinct 9-mer, so naming that one peptide is a
#: complete scan and neither refusal branch can be reached.
_REPEAT_FREE = "".join(f"{_SPAN[index // 20]}{_SPAN[index % 20]}" for index in range(50))


def _nine_mers(chain: str) -> list[str]:
    return sorted({chain[start : start + 9] for start in range(len(chain) - 8)})


def _write_rank_rows(folder: Path, stem: str, rows: list[tuple[str, str]]) -> None:
    """A design table with the rank cell written exactly as given.

    The helpers above format a float. These tests need a cell that is not a
    number at all -- what a predictor writes for a window it did not score.
    """
    lines = ["peptide\tallele\tmethod\tpercentile_rank"]
    lines += [f"{peptide}\tHLA-A*02:01\tconsensus\t{rank}" for peptide, rank in rows]
    (folder / f"{stem}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_repeat_free_chain_has_a_distinct_window_everywhere() -> None:
    """The premise of the two tests below, pinned so they cannot pass vacuously."""
    assert len(_REPEAT_FREE) == 100
    assert len(_nine_mers(_REPEAT_FREE)) == 92


def test_a_partly_scanned_chain_cannot_report_a_real_zero(tmp_path: Path) -> None:
    """Zero binders found is only a zero if every window was looked at.

    A positive density claims "these binders are in the chain", which a
    binders-only export supports. A zero claims there are none anywhere, which
    is a statement about every window. This table holds 40 of the chain's 92
    distinct 9-mers, all non-binding, and the four binders could be among the
    52 it does not name.
    """
    windows = _nine_mers(_REPEAT_FREE)
    _write_design_table(tmp_path, "mhc_i", windows[:40], rank=50.0)
    _write_reference(tmp_path, "mhc_i", [("P02768", "KKKKKKKKK", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", _REPEAT_FREE)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "scan_incomplete"


def test_a_window_left_unscored_in_a_complete_scan_is_refused(tmp_path: Path) -> None:
    """Named is not scored: a cell reading "-" or "n.d." is a window with no answer.

    The file covers the whole chain, so the coverage share is satisfied, and
    without this the four unanswered windows would count as non-binding and the
    density would read as a measured zero.
    """
    windows = _nine_mers(_REPEAT_FREE)
    rows = [(peptide, "50.0") for peptide in windows[:-4]]
    rows += [(windows[-4], "-"), (windows[-3], "n.d."), (windows[-2], "nd"), (windows[-1], "NA")]
    _write_rank_rows(tmp_path, "mhc_i", rows)
    _write_reference(tmp_path, "mhc_i", [("P02768", "KKKKKKKKK", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", _REPEAT_FREE)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "unscored_windows"


def test_a_width_the_cutoffs_do_not_cover_does_not_decide_coverage(tmp_path: Path) -> None:
    """Only widths a binding call is defined for can earn or deny the credential.

    This file scans every 9-mer of the chain and, as a real multi-length export
    does, carries a handful of longer peptides as well. Twelve is outside the
    calibrated widths, so those rows neither complete nor spoil the scan: the
    9-mers cover the chain and the zero stands. Counting them would turn this
    complete scan into ``scan_incomplete``.
    """
    windows = _nine_mers(_REPEAT_FREE)
    twelves = sorted({_REPEAT_FREE[start : start + 12] for start in range(40)})
    _write_design_table(tmp_path, "mhc_i", [*windows, *twelves[:5]], rank=50.0)
    _write_reference(tmp_path, "mhc_i", [("P02768", "KKKKKKKKK", 100, 0.2)])
    axes, _by_example = measure_sequence_liability([("binder_0", _REPEAT_FREE)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "clear"
    assert record["measurement"] == 0.0
    assert record["blocker"] is None


def test_a_reference_claiming_most_of_a_protein_binds_is_not_a_bar(tmp_path: Path) -> None:
    """A bar far above what the cutoff can produce is a bad table, not a high bar.

    Percentile rank <= 1 is the top one per cent of peptides for an allele, so a
    complete scan yields binders of that order; albumin's real measured density
    is 0.49 per 100 residues. This file says all 92 windows of a 100-residue
    albumin bind, which is 92 per 100 and clears almost any design -- and an
    inflated bar always errs toward clearing.
    """
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    rows = [("P02768", f"REF{index:06d}", 100, 0.2) for index in range(92)]
    _write_reference(tmp_path, "mhc_i", rows)
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_density_implausible"
    assert record["reference_proteins"] == ["P02768"]


def test_a_reference_scanned_at_one_width_cannot_bar_a_scan_of_four(tmp_path: Path) -> None:
    """Binders of four widths over one length is not a density the bar can meet.

    The design table here names 8-, 9-, 10- and 11-mers; the reference names
    only 9-mers. Comparing them counts four widths of binders against a bar
    built from one.
    """
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["A" * width for width in (8, 9, 10, 11)])
    _write_reference(tmp_path, "mhc_i", [("P02768", f"REF{index:06d}", 100, 0.2) for index in range(9)])
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_width_mismatch"
    assert record["design_widths"] == [8, 9, 10, 11]
    assert record["reference_widths"] == [9]


def test_a_reference_with_no_rows_is_not_a_bar(tmp_path: Path) -> None:
    """A header with nothing under it is a file, not a scan."""
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    (tmp_path / "mhc_i_reference.tsv").write_text(
        "protein\tpeptide\tlength\tallele\tmethod\tpercentile_rank\n",
        encoding="utf-8",
    )
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_unreadable"


def test_a_reference_cell_that_is_not_a_number_is_refused(tmp_path: Path) -> None:
    """A comma decimal is a locale, not a rank, and reading it as blank loses a binder."""
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    lines = ["protein\tpeptide\tlength\tallele\tmethod\tpercentile_rank"]
    lines.append("P02768\tKKKKKKKKK\t100\tHLA-A*02:01\tconsensus\t0,2")
    (tmp_path / "mhc_i_reference.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_cell_unparsed"
    assert record["unparsed_columns"] == ["percentile_rank"]


@pytest.mark.parametrize(
    "label",
    [
        "sp|Q9Y6K9|TNF_HUMAN,P02768",
        "P02768,P01857",
        "P02768-FRAGMENT-1-76",
    ],
)
def test_a_reference_label_that_is_not_one_tolerated_protein_is_refused(tmp_path: Path, label: str) -> None:
    """One label, one tolerated accession, or the density is over something else.

    A label naming two proteins pools their binders over one length. A fragment
    label is a slice of albumin scored as if it were albumin, which is where the
    85.5-against-0.49 inflation came from. Neither may set the bar, and both
    spellings are ones an export really writes.
    """
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    _write_reference(tmp_path, "mhc_i", [(label, f"REF{index:06d}", 100, 0.2) for index in range(9)])
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_not_tolerated_protein"


def test_a_reference_row_ending_beyond_the_protein_is_refused(tmp_path: Path) -> None:
    """The end column is read, not only the start column.

    Every other reference fixture supplies ``start``, so the bound on ``end``
    never ran: it could be raised by ten thousand residues and the suite stayed
    green, while the two bounds either side of it in the same function were
    both caught by name. An export carrying end instead of start is one the
    module's own _END_COLUMNS exists to read.

    With no start column the rows cannot be shown to be one scan rather than
    several proteins pooled, so this is the pooling refusal rather than the
    "the stated length is wrong" one the start-column test gets.
    """
    design = "A" * 50 + "K" * 50
    _write_design_table(tmp_path, "mhc_i", ["AAAAAAAAA"])
    lines = ["protein\tpeptide\tlength\tend\tallele\tmethod\tpercentile_rank"]
    lines.append("P02768\tKKKKKKKKK\t609\t109\tHLA-A*02:01\tconsensus\t0.2")
    # Albumin's last residue is 609, so a window ending at 1208 is another protein's.
    lines.append("P02768\tRRRRRRRRR\t609\t1208\tHLA-A*02:01\tconsensus\t0.2")
    (tmp_path / "mhc_i_reference.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("binder_0", design)], directory=tmp_path)
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "reference_rows_exceed_protein"
    assert record["reference_proteins_pooled"] == ["P02768"]


def test_a_mixed_scale_table_does_not_lend_consensus_to_foreign_rows(tmp_path: Path) -> None:
    """`_declares_consensus` answers for the whole table; the Paul median must not.

    A file merged from two exports -- four rows naming the IEDB consensus and
    three naming NetMHCIIpan EL -- published a Paul 2015 hit three sevenths of
    which sat on a scale that rule is not defined on. NetMHCIIpan's 5 is not the
    paper's 5, and the paper's median is over seven consensus ranks.

    Confining the median to the rows that declare the scale leaves the panel
    short, which is reported as the gap it is, naming the alleles missing.
    """
    lines = ["allele\tpeptide\tmethod\tpercentile_rank"]
    for allele in ("DRB1*03:01", "DRB1*07:01", "DRB1*15:01", "DRB3*01:01"):
        lines.append(f"{allele}\t{_HA_306_318}\tconsensus\t5.0")
    for allele in ("DRB3*02:02", "DRB4*01:01", "DRB5*01:01"):
        lines.append(f"{allele}\t{_HA_306_318}\tnetmhciipan_el\t5.0")
    (tmp_path / "mhc_ii.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    axes, _by_example = measure_sequence_liability(_ada_design(), directory=tmp_path)
    record = axes["ada"]
    assert record["state"] == "untested", "a median across two scales is not a measurement"
    assert record["blocker"] == "allele_panel_incomplete"
    assert record["measurement"] is None, "the misattributed 5.0 must be withdrawn, not published"


def test_a_single_method_table_reads_exactly_as_before(tmp_path: Path) -> None:
    """The row filter must be inert unless a row names a different method.

    Two tables have to behave exactly as they did before the filter existed. One
    declares consensus in every method cell. The other declares the scale by its
    rank column alone and carries no method column at all, which is how the
    export writes it -- that is the case the filter could silently have emptied,
    because every row of it answers ``_declares_consensus`` false on its own and
    is kept only by having no method cell to contradict the table.
    """
    ranks = [2.0, 5.0, 8.0, 12.0, 15.0, 18.0, 19.0]
    _write_dr_panel(tmp_path, _panel_rows(_HA_306_318, ranks))
    declared = measure_sequence_liability(_ada_design(), directory=tmp_path)[0]["ada"]
    assert declared["state"] == "hit"
    assert declared["measurement"] == 12.0

    by_column = ["allele\tpeptide\tconsensus_percentile_rank"]
    by_column += [f"{a}\t{_HA_306_318}\t{r}" for a, r in zip(_DR_PANEL_NAMES, ranks, strict=True)]
    (tmp_path / "mhc_ii.tsv").write_text("\n".join(by_column) + "\n", encoding="utf-8")
    column_only = measure_sequence_liability(_ada_design(), directory=tmp_path)[0]["ada"]
    assert column_only["state"] == declared["state"], "a table with no method column is not foreign"
    assert column_only["measurement"] == declared["measurement"]


def test_a_table_declaring_no_scale_is_still_refused_before_the_filter(tmp_path: Path) -> None:
    """Blank method cells against a plain rank column declare nothing, so the
    axis refuses at ``rank_scale_unknown`` and never reaches the row filter.

    Pinned because it is what makes the filter safe: a table that does not say
    what scale its numbers are on is already untested, so the filter can only
    ever see a table that has declared one.
    """
    ranks = [2.0, 5.0, 8.0, 12.0, 15.0, 18.0, 19.0]
    lines = ["allele\tpeptide\tmethod\tpercentile_rank"]
    lines += [f"{a}\t{_HA_306_318}\t\t{r}" for a, r in zip(_DR_PANEL_NAMES, ranks, strict=True)]
    (tmp_path / "mhc_ii.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = measure_sequence_liability(_ada_design(), directory=tmp_path)[0]["ada"]
    assert record["state"] == "untested"
    assert record["blocker"] == "rank_scale_unknown"


def test_an_incomplete_presented_set_still_counts_what_it_scored(tmp_path: Path) -> None:
    """The record contradicted itself: one window scored, and none positive.

    When a presented 9-mer is not a window of any sample the axis cannot clear,
    because clearing would say every presented 9-mer is at most 0 while one of
    them was never scored. That part is right. Overwriting the counts with 0 and
    [] was not: `presented_ligands` said one window had been scored and
    `positive_windows` said none of the scored ones reached the cut, while the
    scored window was FIAGLIAIV at 0.272 and it is in the design.

    `presented_complete` is what says the set is short. The counts say what was
    actually found, and a reader needs them precisely because the scan is short.
    """
    in_design = "FIAGLIAIV"
    not_in_design = "SIINFEKLV"
    design = [("binder_0", "LLLLLLLLL" + in_design)]

    rows = ["allele\tpeptide\tmethod\tpercentile_rank"]
    for peptide in (in_design, not_in_design):
        rows.append(f"HLA-A*02:01\t{peptide}\tmhcflurry\t0.4")
    (tmp_path / "mhc_i.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    axes, _by_example = measure_sequence_liability(design, directory=tmp_path)
    record = axes["immunogenicity"]

    assert record["state"] == "untested", "a short presented set cannot clear"
    assert record["blocker"] == "presented_not_in_design"
    assert record["presented_complete"] is False
    assert record["presented_ligands"] == 1, "one presented 9-mer is a window of the design"
    assert record["positive_windows"] == 1, "and it scores above zero, so the count is one"
    assert record["positive_examples"] == ["binder_0"]
