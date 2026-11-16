"""Target-level axes: paralog, and the three plasma axes.

Every test here supplies its own reader, so nothing reaches the network. A
reader that is asked for a URL the test did not plan for fails the test rather
than returning nothing, because an unplanned lookup returning nothing is how an
axis quietly becomes untested without anyone noticing.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from rfproteina.disease.screen import screen_generation
from rfproteina.disease.target_liability import (
    TARGET_LIABILITY_AXES,
    TargetLimits,
    measure_target_liability,
)
from tests.helpers.disease import DEPOSIT_CIF, condition_spec, url_reader

_IL5 = "P05113"
_CLUSTER = "UniRef50_P05113"


def _uniref(cluster: str = _CLUSTER) -> dict[str, object]:
    return {"results": [{"id": cluster}]}


def _members(*entries: tuple[str, str]) -> dict[str, object]:
    return {
        "results": [
            {"primaryAccession": accession, "genes": [{"geneName": {"value": symbol}}]} for accession, symbol in entries
        ]
    }


def _hpa(accession: str, *, immunoassay: object = None, mass_spec: object = None) -> list[dict[str, object]]:
    return [
        {
            "Gene": "IL5",
            "Uniprot": [accession],
            "Blood concentration - Conc. blood IM [pg/L]": immunoassay,
            "Blood concentration - Conc. blood MS [pg/L]": mass_spec,
        }
    ]


def _string(*names: str) -> list[dict[str, object]]:
    return [{"preferredName_A": "IL5", "preferredName_B": name} for name in names]


def _all(**extra: object) -> dict[str, object]:
    """A reader answering every lookup the four axes make."""
    answers: dict[str, object] = {
        "/uniref/search": _uniref(),
        "/uniprotkb/search": _members((_IL5, "IL5")),
        "proteinatlas.org": _hpa(_IL5, immunoassay=1.0e9),
        "string-db.org": _string("IL5RA"),
    }
    answers.update(extra)
    return answers


def test_a_uniref50_singleton_is_not_evidence_of_no_paralog(tmp_path: Path) -> None:
    """This axis has no clear of its own, and that is the honest answer.

    A singleton cluster was the only route to a measured clear, and it does not
    support one: a UniRef50 cluster is built around a representative and does
    not gather a protein's near-identical relatives, so KRAS and EGFR both come
    back as singletons while having exactly the families a binder would
    cross-react with. Absence of a listed relative is not absence of a family.

    The measurement is still reported, and a curated judgement goes in
    axis_evidence.json.
    """
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(_all()))
    record = axes["paralog"]
    assert record["state"] == "untested"
    assert record["blocker"] == "cluster_singleton_not_evidence"
    assert record["measurement"] == 0
    assert record["cluster"] == _CLUSTER
    assert record["paralogs"] == []


def test_a_shared_uniref50_cluster_is_a_paralog_hit(tmp_path: Path) -> None:
    """Another human protein at 50 per cent identity is the cross-reactivity risk."""
    answers = _all(**{"/uniprotkb/search": _members((_IL5, "IL5"), ("P05112", "IL4"))})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    record = axes["paralog"]
    assert record["state"] == "hit"
    assert record["measurement"] == 1
    assert record["paralogs"] == ["IL4"]


def test_a_gene_symbol_target_cannot_be_a_paralog_lookup(tmp_path: Path) -> None:
    """The lookup is by accession, so a symbol row says so instead of guessing."""
    answers = {"proteinatlas.org": [], "string-db.org": _string()}
    axes = measure_target_liability("IL5", directory=tmp_path, read_json=url_reader(answers))
    assert axes["paralog"]["state"] == "untested"
    assert axes["paralog"]["blocker"] == "target_not_accession"
    assert axes["plasma_abundance"]["blocker"] == "target_not_accession"


def test_a_failed_paralog_lookup_is_not_a_clear(tmp_path: Path) -> None:
    """A service that could not answer leaves the axis untested, never clear."""
    answers = _all(**{"/uniref/search": OSError("no route to host")})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    assert axes["paralog"]["state"] == "untested"
    assert axes["paralog"]["blocker"] == "lookup_unavailable"


def test_a_cluster_with_no_id_is_not_a_cluster(tmp_path: Path) -> None:
    answers = _all(**{"/uniref/search": {"results": []}})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    assert axes["paralog"]["blocker"] == "cluster_not_found"


def test_plasma_abundance_prefers_the_immunoassay_value(tmp_path: Path) -> None:
    """Both columns are reported; the directly calibrated one is the measurement."""
    answers = _all(**{"proteinatlas.org": _hpa(_IL5, immunoassay=2.0e9, mass_spec=7.0e9)})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    record = axes["plasma_abundance"]
    assert record["measurement"] == 2.0e9
    assert record["assay"] == "immunoassay"
    assert record["units"] == "pg/L"
    # No published number makes a concentration a liability, so no verdict yet.
    assert record["state"] == "untested"
    assert record["blocker"] == "threshold_not_set"


def test_plasma_abundance_falls_back_to_mass_spectrometry(tmp_path: Path) -> None:
    answers = _all(**{"proteinatlas.org": _hpa(_IL5, mass_spec=5.0e8)})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    assert axes["plasma_abundance"]["measurement"] == 5.0e8
    assert axes["plasma_abundance"]["assay"] == "mass_spectrometry"


def test_plasma_abundance_judges_against_a_named_threshold(tmp_path: Path) -> None:
    answers = _all(**{"proteinatlas.org": _hpa(_IL5, immunoassay=2.0e12)})
    limits = TargetLimits(abundance_max=1.0e12)
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers), limits=limits)
    assert axes["plasma_abundance"]["state"] == "hit"
    assert axes["plasma_abundance"]["threshold"] == 1.0e12
    answers = _all(**{"proteinatlas.org": _hpa(_IL5, immunoassay=1.0e9)})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers), limits=limits)
    assert axes["plasma_abundance"]["state"] == "clear"


def test_the_abundance_row_is_matched_on_the_accession(tmp_path: Path) -> None:
    """The atlas search is fuzzy, so the first row is often a different protein.

    Searching one symbol returns its neighbours too, and taking row zero would
    report another protein's blood concentration against this target.
    """
    wrong = _hpa("P43652", immunoassay=4.0e13)[0]
    wrong["Gene"] = "ALB"
    answers = _all(**{"proteinatlas.org": [wrong, *_hpa(_IL5, immunoassay=3.0e8)]})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    assert axes["plasma_abundance"]["measurement"] == 3.0e8


def test_a_target_the_atlas_does_not_report(tmp_path: Path) -> None:
    answers = _all(**{"proteinatlas.org": []})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    assert axes["plasma_abundance"]["blocker"] == "target_not_reported"
    answers = _all(**{"proteinatlas.org": _hpa(_IL5)})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    assert axes["plasma_abundance"]["blocker"] == "concentration_not_reported"


def test_plasma_hub_counts_distinct_high_confidence_partners(tmp_path: Path) -> None:
    """The degree is the measurement; no published cut makes it a verdict."""
    answers = _all(**{"string-db.org": _string("IL5RA", "CSF2RB", "IL5RA")})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    record = axes["plasma_hub"]
    assert record["measurement"] == 2, "the repeat is one partner"
    assert record["state"] == "untested"
    assert record["blocker"] == "threshold_not_set"
    assert record["confidence"] == 700


def test_plasma_hub_judges_against_a_named_threshold(tmp_path: Path) -> None:
    answers = _all(**{"string-db.org": _string("A", "B", "C", "D")})
    limits = TargetLimits(hub_max=3)
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers), limits=limits)
    assert axes["plasma_hub"]["state"] == "hit"
    limits = TargetLimits(hub_max=4)
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers), limits=limits)
    assert axes["plasma_hub"]["state"] == "clear"


def test_a_fold_change_without_a_study_is_a_rumour(tmp_path: Path) -> None:
    """The table has to say where the number came from."""
    (tmp_path / "plasma_fold.tsv").write_text("fold\n3.4\n", encoding="utf-8")
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(_all()))
    record = axes["plasma_fold"]
    assert record["state"] == "untested"
    assert record["blocker"] == "source_not_named"
    assert record["measurement"] == 3.4


def test_a_fold_change_with_a_study_and_a_threshold(tmp_path: Path) -> None:
    (tmp_path / "plasma_fold.tsv").write_text(
        "fold\tsource\n3.4\tSun 2023 UKB-PPP\n",
        encoding="utf-8",
    )
    axes = measure_target_liability(
        _IL5,
        directory=tmp_path,
        read_json=url_reader(_all()),
        limits=TargetLimits(fold_max=2.0),
    )
    record = axes["plasma_fold"]
    assert record["state"] == "hit"
    assert record["measurement"] == 3.4
    # Under its own key, not "source": screen._stamp_source owns that one and
    # overwrote the study with where the state came from.
    assert record["fold_source"] == "Sun 2023 UKB-PPP"
    assert record["threshold"] == 2.0


def test_a_fold_change_is_judged_on_magnitude(tmp_path: Path) -> None:
    """A target that falls in disease moves the sink as much as one that rises."""
    (tmp_path / "plasma_fold.csv").write_text("fold,source\n-4.0,a study\n", encoding="utf-8")
    axes = measure_target_liability(
        _IL5,
        directory=tmp_path,
        read_json=url_reader(_all()),
        limits=TargetLimits(fold_max=2.0),
    )
    assert axes["plasma_fold"]["state"] == "hit"
    assert axes["plasma_fold"]["measurement"] == -4.0


def test_a_log2_fold_is_judged_as_a_linear_magnitude(tmp_path: Path) -> None:
    """``log2_fold_change`` of 1.5 is about 2.8-fold, not a fold of 1.5.

    The stored measurement stays the cell. A linear ``fold_change`` of 1.5
    against a threshold of 2 stays clear.
    """

    def judge(header: str, cell: str) -> dict[str, object]:
        folder = tmp_path / f"{header}-{cell.replace('-', 'neg').replace('.', 'p')}"
        folder.mkdir()
        (folder / "plasma_fold.tsv").write_text(f"{header}\tsource\n{cell}\ta study\n", encoding="utf-8")
        axes = measure_target_liability(
            _IL5,
            directory=folder,
            read_json=url_reader(_all()),
            limits=TargetLimits(fold_max=2.0),
        )
        return axes["plasma_fold"]

    linear = judge("fold_change", "1.5")
    assert linear["state"] == "clear"
    assert linear["measurement"] == 1.5
    logged = judge("log2_fold_change", "1.5")
    assert logged["state"] == "hit"
    assert logged["measurement"] == 1.5
    exact = judge("log2_fold_change", "1")
    assert exact["state"] == "hit"
    assert exact["measurement"] == 1.0
    small = judge("log2_fold_change", "0.5")
    assert small["state"] == "clear"
    assert small["measurement"] == 0.5
    fallen = judge("log2_fold_change", "-1.5")
    assert fallen["state"] == "hit"
    assert fallen["measurement"] == -1.5


def test_a_fold_cites_the_study_on_its_own_row(tmp_path: Path) -> None:
    """The study has to sit on the row that supplied the number.

    An unsourced log2 of 3 is about 8-fold. Citing an earlier row's study for
    it would publish a hit that row did not report. A sourced log2 cites its
    own study.

    Nor does dropping that row leave a clear behind. This axis used to answer
    the unsourced 8-fold by publishing ``clear`` on the sourced 1.1 beside it,
    which is the table's smaller number standing for the whole table: a clear
    speaks for every row, and this one could not. The number without a study
    is reported and the verdict is withheld.
    """
    unsourced = tmp_path / "unsourced"
    unsourced.mkdir()
    (unsourced / "plasma_fold.tsv").write_text(
        "fold\tsource\tlog2_fold_change\n1.1\tSun 2023\t\n\t\t3\n",
        encoding="utf-8",
    )
    quiet = measure_target_liability(
        _IL5,
        directory=unsourced,
        read_json=url_reader(_all()),
        limits=TargetLimits(fold_max=2.0),
    )["plasma_fold"]
    assert quiet["state"] == "untested"
    assert quiet["blocker"] == "source_not_named"
    assert quiet["measurement"] == 3.0
    # The study stayed on its own row: it was not lent to the 8-fold cell, and
    # it did not buy a clear for the table either.
    assert "fold_source" not in quiet

    credited = tmp_path / "credited"
    credited.mkdir()
    (credited / "plasma_fold.tsv").write_text(
        "fold\tsource\tlog2_fold_change\n1.1\tSun 2023\t\n\tOther study\t3\n",
        encoding="utf-8",
    )
    hit = measure_target_liability(
        _IL5,
        directory=credited,
        read_json=url_reader(_all()),
        limits=TargetLimits(fold_max=2.0),
    )["plasma_fold"]
    assert hit["state"] == "hit"
    assert hit["measurement"] == 3.0
    assert hit["fold_source"] == "Other study"


def test_no_fold_table_is_not_a_clear(tmp_path: Path) -> None:
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(_all()))
    assert axes["plasma_fold"]["state"] == "untested"
    assert axes["plasma_fold"]["blocker"] == "fold_not_configured"


def test_every_target_axis_is_reported_in_order(tmp_path: Path) -> None:
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(_all()))
    assert list(axes) == list(TARGET_LIABILITY_AXES)


def test_a_screen_without_a_reader_looks_nothing_up(tmp_path: Path) -> None:
    """The default is no network. An unlooked-up axis says so.

    A screen that reached the network by default would make every run depend on
    four services being up, and would do it without the caller asking.
    """
    output = tmp_path / "quiet"
    output.mkdir()
    record = screen_generation(output, binder_kind="protein", target_id=_IL5)
    target = record["target_liability"]
    assert list(target) == list(TARGET_LIABILITY_AXES)
    for axis in ("paralog", "plasma_abundance", "plasma_hub"):
        assert target[axis]["state"] == "untested"
        assert target[axis]["blocker"] == "network_not_permitted"
        assert target[axis]["source"] == "untested"
    # The fold axis reads a table in the output directory and wants no network,
    # so reporting that a network was unavailable would have been wrong.
    assert target["plasma_fold"]["blocker"] == "fold_not_configured"
    assert record["call"] == "untested"


def test_a_measured_paralog_hit_reaches_the_judge(tmp_path: Path) -> None:
    """A target axis is evidence like any other, not a separate report.

    The direction this axis can measure is the hit: another human protein in
    the cluster is positive evidence of a family to cross-react with.
    """
    output = tmp_path / "judged"
    output.mkdir()
    answers = _all(**{"/uniprotkb/search": _members((_IL5, "IL5"), ("P05112", "IL4"))})
    record = screen_generation(
        output,
        binder_kind="protein",
        target_id=_IL5,
        read_json=url_reader(answers),
    )
    assert record["target_liability"]["paralog"]["state"] == "hit"
    assert record["target_liability"]["paralog"]["source"] == "measured"
    assert record["states"]["paralog"] == "hit"
    assert record["call"] == "axis_hit"
    assert record["hits"] == ["paralog"]


def test_a_nucleic_screen_has_no_target_liability(tmp_path: Path) -> None:
    """The judge refuses an axis foreign to the kind, so none is written.

    ``AXES_BY_KIND`` gives a nucleic binder only the nucleic axes, so neither
    paralog nor the plasma axes apply to one.
    """
    output = tmp_path / "oligo"
    output.mkdir()
    record = screen_generation(
        output,
        binder_kind="nucleic",
        target_id=_IL5,
        read_json=url_reader(_all()),
    )
    assert "target_liability" not in record
    assert "sequence_liability" not in record


def test_a_target_record_holds_no_url_and_no_key(tmp_path: Path) -> None:
    """What is written is a measurement, not the request that produced it."""
    output = tmp_path / "scrubbed"
    output.mkdir()
    record = screen_generation(
        output,
        binder_kind="protein",
        target_id=_IL5,
        read_json=url_reader(_all()),
    )
    written = json.dumps(record["target_liability"])
    assert "http" not in written
    assert "proteinatlas" not in written
    assert "rest.uniprot" not in written


@pytest.mark.parametrize("kind", ["protein", "peptide"])
def test_paralog_applies_to_both_sequence_kinds(tmp_path: Path, kind: str) -> None:
    output = tmp_path / kind
    output.mkdir()
    record = screen_generation(output, binder_kind=kind, target_id=_IL5, read_json=url_reader(_all()))
    assert record["target_liability"]["paralog"]["blocker"] == "cluster_singleton_not_evidence"


# The receptor leads the file, and the designed chain has its own accession.
# The sequence-fixed chain is the only one the lookup may use.
# The same deposit in the other spelling mmCIF allows. A category with one row
# is written as plain item/value lines, not a loop_, and a deposit with a single
# UniProt reference -- a monomeric target, which is the common case -- is written
# exactly like this. 117 of the 261 target structures in this checkout are.
_FLAT_DEPOSIT = """\
data_flat
#
_struct_asym.id          C
_struct_asym.entity_id   2
#
_struct_ref.id                           1
_struct_ref.db_name                      UNP
_struct_ref.db_code                      TARGET_HUMAN
_struct_ref.pdbx_db_accession            P05113
_struct_ref.pdbx_db_isoform              ?
_struct_ref.entity_id                    2
_struct_ref.pdbx_seq_one_letter_code
;PPPP
;
_struct_ref.pdbx_align_begin             20
#
_struct_ref_seq.align_id            1
_struct_ref_seq.ref_id              1
_struct_ref_seq.pdbx_strand_id      C
_struct_ref_seq.pdbx_db_accession   P05113
#
"""

_TWO_ACCESSIONS = """\
data_two
#
loop_
_struct_asym.id
_struct_asym.entity_id
C 2
#
loop_
_struct_ref.id
_struct_ref.db_name
_struct_ref.pdbx_db_accession
_struct_ref.entity_id
1 UNP P05113 2
2 UNP P05112 2
#
"""

_DISAGREE = """\
data_disagree
#
loop_
_struct_asym.id
_struct_asym.entity_id
C 2
#
loop_
_struct_ref.id
_struct_ref.db_name
_struct_ref.pdbx_db_accession
_struct_ref.entity_id
1 UNP P05113 2
2 UNP Q01344 3
#
loop_
_struct_ref_seq.align_id
_struct_ref_seq.ref_id
_struct_ref_seq.pdbx_strand_id
_struct_ref_seq.pdbx_db_accession
1 2 C Q01344
#
"""

_NO_UNP = """\
data_plain
#
loop_
_struct_asym.id
_struct_asym.entity_id
C 2
#
"""


def _watching(answers: dict[str, object], seen: list[str]) -> Callable[[str], object]:
    inner = url_reader(answers)

    def read(url: str) -> object:
        seen.append(url)
        return inner(url)

    return read


def test_a_gene_symbol_uses_the_fixed_chain_on_the_campaign_deposit(tmp_path: Path) -> None:
    """The generated CIF has no UNP row. The deposit's fixed chain does.

    The receptor is the first UNP row and the designed chain has another
    accession. Neither is the sequence-fixed chain. The job keeps the gene
    symbol the caller passed.
    """
    root = tmp_path / "runs"
    campaign = root / "gene"
    generated = campaign / "inference_samples" / "binder_0"
    inverse = root / "inverse"
    generated.mkdir(parents=True)
    inverse.mkdir()
    (root / "deposit.cif").write_text(DEPOSIT_CIF, encoding="utf-8")
    (root / "gene.yaml").write_text(
        condition_spec("deposit.cif", "C", backbone=("A", "C")),
        encoding="utf-8",
    )
    (generated / "binder_0_generated.cif").write_text("data_generated\n#\n", encoding="utf-8")
    (inverse / "inverse_folding.yaml").write_text(
        condition_spec(
            "../gene/inference_samples/binder_0/binder_0_generated.cif",
            "C",
            backbone=("A", "C"),
        ),
        encoding="utf-8",
    )
    job = root / "inverse.job.json"
    job.write_text(
        json.dumps({"class_name": "requested_clear", "binder_kind": "protein", "target_id": "IL5"}),
        encoding="utf-8",
    )
    before = job.read_bytes()
    seen: list[str] = []
    record = screen_generation(
        inverse,
        binder_kind="protein",
        target_id="IL5",
        read_json=_watching(_all(), seen),
    )
    assert job.read_bytes() == before
    assert "job_mismatch" not in record
    assert record["target_accession"] == _IL5
    assert record["call"] == "untested"
    assert record["target_liability"]["paralog"]["blocker"] == "cluster_singleton_not_evidence"
    assert record["target_liability"]["plasma_fold"]["blocker"] == "fold_not_configured"
    text = (inverse / "screen.json").read_text(encoding="utf-8")
    assert "http" not in text
    assert "Q01344" not in text
    assert "P99999" not in text
    assert "deposit.cif" not in text
    assert seen
    assert all(_IL5 in url for url in seen)


def test_a_run_reads_the_condition_spec_beside_it(tmp_path: Path) -> None:
    """The original campaign spec is the sibling of the run directory."""
    campaign = tmp_path / "gene"
    campaign.mkdir()
    (tmp_path / "deposit.cif").write_text(DEPOSIT_CIF, encoding="utf-8")
    (tmp_path / "gene.yaml").write_text(
        condition_spec("deposit.cif", "C", backbone=("A", "C")),
        encoding="utf-8",
    )
    seen: list[str] = []
    record = screen_generation(campaign, binder_kind="protein", target_id="IL5", read_json=_watching(_all(), seen))
    assert record["target_accession"] == _IL5
    assert all(_IL5 in url and "Q01344" not in url for url in seen)


def test_a_deposit_written_without_loops_still_names_its_target(tmp_path: Path) -> None:
    """The accession reader's own loop-only gap, which left every target unnamed.

    ``_struct_ref``, ``_struct_asym``, ``_struct_ref_seq`` and ``_entity_poly``
    were all read from ``loop_`` blocks only. A deposit with one UniProt
    reference writes them flat, so no accession was found and the chains then
    "disagreed" -- every one of the 261 target structures in this checkout
    resolved to ``target_accession_ambiguous``, including the ones naming their
    protein in plain sight. ``gene_4ZW9_1.cif`` carries ``P11169`` this way.

    It erred toward untested rather than toward a clear, so it was never a false
    clear; it meant the target-liability axes had no accession to measure
    against, which is a different kind of nothing.
    """
    campaign = tmp_path / "gene"
    campaign.mkdir()
    (tmp_path / "deposit.cif").write_text(_FLAT_DEPOSIT, encoding="utf-8")
    (tmp_path / "gene.yaml").write_text(condition_spec("deposit.cif", "C"), encoding="utf-8")
    record = screen_generation(campaign, binder_kind="protein", target_id="IL5")
    assert record["target_accession"] == _IL5


def test_both_spellings_of_one_deposit_name_the_same_target(tmp_path: Path) -> None:
    """The loop path is untouched, so the two spellings must not diverge.

    Written as one assertion over both files rather than two tests, because the
    thing that matters is that they agree: a reader that resolved one spelling
    to a different protein than the other would be worse than one that resolved
    neither.
    """
    found: list[str | None] = []
    for index, deposit in enumerate((DEPOSIT_CIF, _FLAT_DEPOSIT)):
        root = tmp_path / f"case{index}"
        campaign = root / "gene"
        campaign.mkdir(parents=True)
        (root / "deposit.cif").write_text(deposit, encoding="utf-8")
        (root / "gene.yaml").write_text(condition_spec("deposit.cif", "C"), encoding="utf-8")
        record = screen_generation(campaign, binder_kind="protein", target_id="IL5")
        found.append(record.get("target_accession"))
    assert found == [_IL5, _IL5], found


def test_a_flat_deposit_naming_two_accessions_for_one_chain_stays_untested(tmp_path: Path) -> None:
    """Reading more does not mean deciding more: a real disagreement still refuses.

    Two references cannot be written as a single flat category, so this is a
    flat ``_struct_asym`` joined to a two-row ``_struct_ref`` loop -- the mixed
    case, which is also the one where reading both spellings could have let a
    row in twice and turned a disagreement into a single confident answer.
    """
    output = tmp_path / "mixed"
    output.mkdir()
    mixed = (
        "data_mixed\n#\n_struct_asym.id          C\n_struct_asym.entity_id   2\n#\n"
        "loop_\n_struct_ref.id\n_struct_ref.db_name\n_struct_ref.pdbx_db_accession\n"
        "_struct_ref.entity_id\n1 UNP P05113 2\n2 UNP P05112 2\n#\n"
    )
    (output / "mixed.cif").write_text(mixed, encoding="utf-8")
    (output / "mixed.yaml").write_text(condition_spec("mixed.cif", "C"), encoding="utf-8")

    def read(url: str) -> object:
        raise AssertionError(url)

    record = screen_generation(output, binder_kind="protein", target_id="IL5", read_json=read)
    # No accession is stored at all, rather than one of the two being picked.
    assert "target_accession" not in record
    assert record["call"] == "untested"
    for axis in ("paralog", "plasma_abundance", "plasma_hub"):
        assert record["target_liability"][axis]["state"] == "untested"


def test_a_resolved_accession_is_stored_and_not_looked_up_without_a_reader(tmp_path: Path) -> None:
    """Reading the structure is not permission to call the public services."""
    campaign = tmp_path / "gene"
    campaign.mkdir()
    (tmp_path / "deposit.cif").write_text(DEPOSIT_CIF, encoding="utf-8")
    (tmp_path / "gene.yaml").write_text(condition_spec("deposit.cif", "C"), encoding="utf-8")
    record = screen_generation(campaign, binder_kind="protein", target_id="IL5")
    assert record["target_accession"] == _IL5
    assert record["target_liability"]["paralog"]["blocker"] == "network_not_permitted"
    assert record["target_liability"]["plasma_hub"]["blocker"] == "network_not_permitted"
    assert record["call"] == "untested"


def test_an_accession_already_passed_in_is_not_replaced(tmp_path: Path) -> None:
    """A structure that names a different protein does not override the caller."""
    output = tmp_path / "held"
    output.mkdir()
    (output / "deposit.cif").write_text(DEPOSIT_CIF, encoding="utf-8")
    (output / "held.yaml").write_text(condition_spec("deposit.cif", "C"), encoding="utf-8")
    passed = "P12345"
    seen: list[str] = []
    answers = _all(
        **{
            "/uniref/search": _uniref("UniRef50_P12345"),
            "/uniprotkb/search": _members((passed, "KEPT")),
        }
    )
    record = screen_generation(output, binder_kind="protein", target_id=passed, read_json=_watching(answers, seen))
    assert "target_accession" not in record
    assert record["target_liability"]["paralog"]["blocker"] == "cluster_singleton_not_evidence"
    assert seen
    assert all(passed in url for url in seen)
    assert all(_IL5 not in url for url in seen)


def test_two_accessions_on_the_fixed_chain_stay_untested(tmp_path: Path) -> None:
    """Two proteins on the chain is not a reason to pick one."""
    output = tmp_path / "split"
    output.mkdir()
    (output / "split.cif").write_text(_TWO_ACCESSIONS, encoding="utf-8")
    (output / "split.yaml").write_text(condition_spec("split.cif", "C"), encoding="utf-8")

    def read(url: str) -> object:
        raise AssertionError(url)

    record = screen_generation(output, binder_kind="protein", target_id="IL5", read_json=read)
    assert "target_accession" not in record
    assert record["call"] == "untested"
    for axis in ("paralog", "plasma_abundance", "plasma_hub"):
        assert record["target_liability"][axis]["state"] == "untested"
        assert record["target_liability"][axis]["blocker"] == "target_accession_ambiguous"
    assert record["target_liability"]["plasma_fold"]["blocker"] == "fold_not_configured"


def test_a_label_and_author_accession_that_disagree_stay_untested(tmp_path: Path) -> None:
    output = tmp_path / "disagree"
    output.mkdir()
    (output / "disagree.cif").write_text(_DISAGREE, encoding="utf-8")
    (output / "disagree.yaml").write_text(condition_spec("disagree.cif", "C"), encoding="utf-8")

    def read(url: str) -> object:
        raise AssertionError(url)

    record = screen_generation(output, binder_kind="protein", target_id="IL5", read_json=read)
    assert "target_accession" not in record
    assert record["target_liability"]["paralog"]["blocker"] == "target_accession_ambiguous"


def test_two_fixed_chains_with_different_accessions_stay_untested(tmp_path: Path) -> None:
    output = tmp_path / "both"
    output.mkdir()
    (output / "deposit.cif").write_text(DEPOSIT_CIF, encoding="utf-8")
    (output / "both.yaml").write_text(condition_spec("deposit.cif", "C", "E"), encoding="utf-8")

    def read(url: str) -> object:
        raise AssertionError(url)

    record = screen_generation(output, binder_kind="protein", target_id="IL5", read_json=read)
    assert "target_accession" not in record
    assert record["target_liability"]["paralog"]["blocker"] == "target_accession_ambiguous"


def test_a_gene_symbol_with_no_structure_accession_stays_untested(tmp_path: Path) -> None:
    """No spec, and a spec whose chain has no UNP row, both stay on the symbol."""
    bare = tmp_path / "bare"
    bare.mkdir()

    def read(url: str) -> object:
        if "string-db.org" not in url:
            raise AssertionError(url)
        assert _IL5 not in url
        return _string()

    record = screen_generation(bare, binder_kind="protein", target_id="IL5", read_json=read)
    assert "target_accession" not in record
    assert record["target_liability"]["paralog"]["blocker"] == "target_not_accession"
    assert record["target_liability"]["plasma_abundance"]["blocker"] == "target_not_accession"

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "plain.cif").write_text(_NO_UNP, encoding="utf-8")
    (plain / "plain.yaml").write_text(condition_spec("plain.cif", "C"), encoding="utf-8")
    again = screen_generation(plain, binder_kind="protein", target_id="IL5", read_json=read)
    assert "target_accession" not in again
    assert again["target_liability"]["paralog"]["blocker"] == "target_not_accession"
    assert again["target_liability"]["plasma_abundance"]["blocker"] == "target_not_accession"


# ---------------------------------------------------------------------------
# The blockers below had no assertion anywhere, and two of the guards that
# emit them survived mutation: the degree cap could be raised by ten thousand
# and the unreadable-fold branch could be made dead, with the suite staying
# green either way. The guards work; nothing held them in place.
# ---------------------------------------------------------------------------


def test_a_fold_written_as_a_ratio_below_one_is_still_a_swing(tmp_path: Path) -> None:
    """A plain ratio is symmetric about 1, not 0.

    A disease/healthy column writes a fivefold fall as 0.2. Read as an absolute
    value that is 0.2, which clears every threshold above it -- so the axis
    reported ``clear`` on the same biology that reads ``hit`` when the table
    happens to spell it 5.0. Both are a fivefold change.
    """
    (tmp_path / "plasma_fold.tsv").write_text("fold\tsource\n0.2\tSun 2023\n", encoding="utf-8")
    axes = measure_target_liability(
        _IL5,
        directory=tmp_path,
        read_json=url_reader(_all()),
        limits=TargetLimits(fold_max=2.0),
    )
    record = axes["plasma_fold"]
    assert record["state"] == "hit"
    assert record["measurement"] == 0.2


def test_the_largest_swing_in_a_fold_table_is_the_one_judged(tmp_path: Path) -> None:
    """Row selection is by magnitude, so a downward row cannot be passed over.

    With 0.2 and 1.5 in one table the maximum used to be 1.5, and the fivefold
    swing on the other row was never judged at all -- the axis cleared on the
    smaller number.
    """
    (tmp_path / "plasma_fold.tsv").write_text(
        "fold\tsource\n0.2\tSun 2023\n1.5\tSun 2023\n",
        encoding="utf-8",
    )
    axes = measure_target_liability(
        _IL5,
        directory=tmp_path,
        read_json=url_reader(_all()),
        limits=TargetLimits(fold_max=2.0),
    )
    record = axes["plasma_fold"]
    assert record["measurement"] == 0.2
    assert record["state"] == "hit"


def test_a_fold_table_that_cannot_be_read_is_not_a_clear(tmp_path: Path) -> None:
    """Unreadable is not absent, and it is certainly not no change."""
    (tmp_path / "plasma_fold.tsv").write_bytes(b"\xff\xfe\xfd fold\n")
    axes = measure_target_liability(
        _IL5,
        directory=tmp_path,
        read_json=url_reader(_all()),
        limits=TargetLimits(fold_max=2.0),
    )
    record = axes["plasma_fold"]
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_unreadable"


def test_a_fold_table_with_no_rows_reports_no_fold(tmp_path: Path) -> None:
    """A header with nothing under it has not reported a fold change."""
    (tmp_path / "plasma_fold.tsv").write_text("fold\tsource\n", encoding="utf-8")
    axes = measure_target_liability(
        _IL5,
        directory=tmp_path,
        read_json=url_reader(_all()),
        limits=TargetLimits(fold_max=2.0),
    )
    record = axes["plasma_fold"]
    assert record["state"] == "untested"
    assert record["blocker"] == "fold_not_reported"


def test_a_degree_at_the_api_limit_is_censored_not_measured(tmp_path: Path) -> None:
    """A count that stopped at the limit is a floor, not a degree.

    There is no absolute hub threshold in this module, so a censored degree
    compared against a user's threshold can read ``clear`` while the real
    degree is far higher. The number is still reported, as a floor.
    """
    from rfproteina.disease.target_liability import _STRING_LIMIT

    answers = _all(**{"string-db.org": _string(*[f"PARTNER{index:05d}" for index in range(_STRING_LIMIT)])})
    axes = measure_target_liability(
        _IL5,
        directory=tmp_path,
        read_json=url_reader(answers),
        limits=TargetLimits(hub_max=5000),
    )
    record = axes["plasma_hub"]
    assert record["state"] == "untested"
    assert record["blocker"] == "degree_censored"
    assert record["measurement"] == _STRING_LIMIT


def test_a_target_with_no_name_cannot_be_looked_up(tmp_path: Path) -> None:
    """No identifier, no lookup, and no clear either."""
    axes = measure_target_liability("", directory=tmp_path, read_json=url_reader({}))
    assert axes["plasma_hub"]["state"] == "untested"
    assert axes["plasma_hub"]["blocker"] == "target_not_named"


def test_a_target_with_no_listed_partners_is_not_a_clear(tmp_path: Path) -> None:
    """STRING having no record for the target is not a degree of zero."""
    from rfproteina.disease.research import LookupMissError

    answers = _all(**{"string-db.org": LookupMissError("string")})
    axes = measure_target_liability(
        _IL5,
        directory=tmp_path,
        read_json=url_reader(answers),
        limits=TargetLimits(hub_max=3),
    )
    record = axes["plasma_hub"]
    assert record["state"] == "untested"
    assert record["blocker"] == "partners_not_listed"


def test_a_target_absent_from_its_own_cluster_listing_is_not_a_singleton(tmp_path: Path) -> None:
    """Being the only name in an incomplete list is not evidence of being alone."""
    answers = _all(**{"/uniprotkb/search": _members(("P05112", "IL4"))})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    record = axes["paralog"]
    assert record["state"] == "untested"
    assert record["blocker"] == "members_not_listed"


def test_a_cluster_listing_at_the_page_limit_is_truncated_not_counted(tmp_path: Path) -> None:
    """A page that filled up is a floor on the family, so the count is not a verdict."""
    from rfproteina.disease.target_liability import _MEMBER_LIMIT

    entries = [(f"P{index:05d}", f"GENE{index}") for index in range(_MEMBER_LIMIT - 1)]
    answers = _all(**{"/uniprotkb/search": _members((_IL5, "IL5"), *entries)})
    axes = measure_target_liability(_IL5, directory=tmp_path, read_json=url_reader(answers))
    record = axes["paralog"]
    assert record["state"] == "untested"
    assert record["blocker"] == "members_truncated"
    assert record["measurement"] == _MEMBER_LIMIT
