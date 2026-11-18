"""The class II predictor fitted from IEDB's open measurements.

``immunogenicity`` reports the axis unavailable because every published class
II tool is licence-blocked. The measurements are not, so this module fits a
predictor from them. These tests hold three things: that the corpus filters
drop what cannot be fitted rather than fitting it anyway, that the fit learns a
signal when one is there, and -- the one that matters most -- that the rank it
produces can never be mistaken for an IEDB consensus percentile rank, because
the ADA threshold of 20 is defined on that scale and on no other.

Measured live against the real corpus: 20,087 rows, 14,739 (allele, peptide)
pairs, all seven panel alleles, mean held-out AUC 0.735.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import pytest

from rfproteina.disease import class_ii as ci
from rfproteina.disease import immunogenicity as ig

_MOTIF_START = "Y"
_MOTIF_END = "W"
_OTHER = "ACDEFGHIKLMNPQRSTV"


@pytest.fixture(autouse=True)
def _no_backends():
    """Registration is process-wide, so each test starts and ends with none."""
    ig._BACKENDS.clear()
    yield
    ig._BACKENDS.clear()


def _corpus(path: Path, rows, columns=ci.CORPUS_COLUMNS) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})
    return path


def _row(peptide, nanomolar, allele="DRB1*15:01", units="nM", inequality=""):
    return {
        "allele": allele,
        "peptide": peptide,
        "measurement": str(nanomolar),
        "units": units,
        "inequality": inequality,
    }


def _motif_rows(count=400, alleles=("DRB1*15:01",), seed=7):
    """A corpus with a real signal: a core starting Y and ending W binds.

    Without this, a test can only check that fitting runs. With it, the AUC is
    a claim about whether the fit found something, which is the only reason to
    have a model at all.
    """
    generator = random.Random(seed)
    rows = []
    for index in range(count):
        filler = "".join(generator.choices(_OTHER, k=15))
        binder = index % 2 == 0
        if binder:
            start = generator.randrange(0, 15 - ci.CORE_WIDTH + 1)
            core = _MOTIF_START + "".join(generator.choices(_OTHER, k=7)) + _MOTIF_END
            peptide = filler[:start] + core + filler[start + ci.CORE_WIDTH :]
        else:
            peptide = filler.replace(_MOTIF_START, "A").replace(_MOTIF_END, "A")
        nanomolar = generator.uniform(1, 200) if binder else generator.uniform(5000, 40000)
        for allele in alleles:
            rows.append(_row(peptide, round(nanomolar, 2), allele=allele))
    return rows


# --------------------------------------------------------------- the scale rule


def test_the_method_name_never_claims_the_consensus_scale() -> None:
    """The whole safety argument in one line.

    ``convert_class_ii_rows`` decides whether the ADA axis trusts a table by
    looking for ``CONSENSUS_MARK`` in its method name. A name containing
    "consensus" would silently license the threshold of 20 against a scale it
    was never calibrated on.
    """
    assert ig.CONSENSUS_MARK not in ci.CLASS_II_METHOD.casefold()
    _table, warnings = ig.convert_class_ii_rows(
        [{"peptide": "A" * 15, "allele": "DRB1*15:01", "percentile_rank": 1.0}],
        method=ci.CLASS_II_METHOD,
    )
    assert any("ada_axis_will_report_rank_scale_unknown" in warning for warning in warnings)


def test_the_option_registry_carries_this_route_with_its_licence() -> None:
    options = [option for option in ig.CLASS_II_OPTIONS if "class_ii" in option.name]
    assert options, "the fitted route belongs in the registry beside the blocked ones"
    (option,) = options
    assert "creativecommons.org/licenses/by/4.0" in option.licence_quote
    assert "2026-11-17" in option.licence_read_on
    assert "NOT an IEDB consensus" in option.rank_scale
    assert option.licence.strip() and option.commercial_use.strip()


# ------------------------------------------------------------------- the corpus


def test_the_transform_pins_one_nanomolar_to_one_and_fifty_thousand_to_zero() -> None:
    assert ci.transformed_target(1.0) == pytest.approx(1.0)
    assert ci.transformed_target(ci.MAX_NM) == pytest.approx(0.0)
    assert ci.transformed_target(1_000_000.0) == 0.0, "clipped, so one weak outlier cannot dominate"
    assert 0.0 < ci.transformed_target(100.0) < 1.0
    with pytest.raises(ValueError, match="positive"):
        ci.transformed_target(0.0)


def test_a_half_life_in_minutes_is_dropped_not_read_as_a_concentration(tmp_path) -> None:
    """IEDB's units column also holds ``min`` and ``angstroms``."""
    path = _corpus(
        tmp_path / "c.tsv",
        [
            _row("YAAAAAAAW" + "AAAAAA", 10.0),
            _row("YCCCCCCCW" + "CCCCCC", 240.0, units="min"),
            _row("YDDDDDDDW" + "DDDDDD", 8.0, units="angstroms"),
            _row("YEEEEEEEW" + "EEEEEE", 12.0, units=""),
        ],
    )
    report = ci.load_corpus(path)
    assert len(report.rows) == 1
    assert report.dropped["units_not_nm"] == 3
    assert report.rows[0].nanomolar == pytest.approx(10.0)


def test_a_censored_measurement_is_dropped_not_taken_at_its_bound(tmp_path) -> None:
    """``>50000`` means weaker than the assay saw, not 50,000 nM."""
    path = _corpus(
        tmp_path / "c.tsv",
        [
            _row("YAAAAAAAWAAAAAA", 10.0),
            _row("YCCCCCCCWCCCCCC", 50000.0, inequality=">"),
            _row("YDDDDDDDWDDDDDD", 1.0, inequality="<"),
            _row("YEEEEEEEWEEEEEE", 20.0, inequality="="),
        ],
    )
    report = ci.load_corpus(path)
    assert report.dropped["measurement_censored"] == 2
    assert {row.nanomolar for row in report.rows} == {10.0, 20.0}


def test_repeats_are_aggregated_by_geometric_mean(tmp_path) -> None:
    """Affinities live on a log scale, and the real corpus repeats one pair
    seven times across 6.5 to 75 nM. An arithmetic mean would follow the
    largest reading."""
    peptide = "YAAAAAAAWAAAAAA"
    path = _corpus(tmp_path / "c.tsv", [_row(peptide, 10.0), _row(peptide, 1000.0)])
    report = ci.load_corpus(path)
    (row,) = report.rows
    assert row.readings == 2
    assert row.nanomolar == pytest.approx(100.0), "geometric, not the arithmetic 505"
    assert report.aggregated == 1


def test_unfittable_peptides_are_dropped_with_a_reason(tmp_path) -> None:
    path = _corpus(
        tmp_path / "c.tsv",
        [
            _row("YAAAAAAAWAAAAAA", 10.0),
            _row("YAAAW", 10.0),
            _row("YAAXAAAAWAAAAAA", 10.0),
            _row("YAAAAAAAWAAAAAE", 0.0),
            _row("YAAAAAAAWAAAAAC", "not-a-number"),
            {"allele": "", "peptide": "YAAAAAAAWAAAAAD", "measurement": "10", "units": "nM", "inequality": ""},
        ],
    )
    report = ci.load_corpus(path)
    assert len(report.rows) == 1
    assert report.dropped["peptide_shorter_than_core"] == 1
    assert report.dropped["non_standard_residue"] == 1
    assert report.dropped["measurement_not_positive"] == 1
    assert report.dropped["measurement_unparsable"] == 1
    assert report.dropped["allele_absent"] == 1
    assert report.read == 6


def test_a_partial_panel_is_reported_rather_than_passed_off(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", [_row("YAAAAAAAWAAAAAA", 10.0, allele="DRB1*15:01")])
    report = ci.load_corpus(path)
    assert report.covers_panel is False
    assert set(report.per_allele) == {"DRB1*15:01"}


def test_a_corpus_that_cannot_be_read_raises_rather_than_returning_nothing(tmp_path) -> None:
    with pytest.raises(ci.ClassIICorpusError, match="corpus_not_found"):
        ci.load_corpus(tmp_path / "absent.tsv")
    bare = _corpus(tmp_path / "bare.tsv", [], columns=("allele", "peptide"))
    with pytest.raises(ci.ClassIICorpusError, match="corpus_column_absent"):
        ci.load_corpus(bare)


# -------------------------------------------------------------------- the model


def test_the_fit_finds_a_motif_that_is_really_there(tmp_path) -> None:
    """The claim a model has to earn. Fitting that runs is not fitting that learns."""
    path = _corpus(tmp_path / "c.tsv", _motif_rows(400))
    model = ci.train(ci.load_corpus(path), background=200)
    (allele,) = model.alleles
    assert model.alleles[allele].auc is not None
    assert model.alleles[allele].auc > 0.9, f"a planted motif should be easy, got {model.alleles[allele].auc}"
    strong = ci.score(model, "YAAAAAAAW" + "AAAAAA", allele)
    weak = ci.score(model, "AAAAAAAAA" + "AAAAAA", allele)
    assert strong > weak


def test_the_matrix_is_nine_positions_of_twenty_residues(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240))
    model = ci.train(ci.load_corpus(path), background=100)
    (fitted,) = model.alleles.values()
    assert len(fitted.weights) == ci.CORE_WIDTH * 20
    matrix = fitted.matrix()
    assert len(matrix) == ci.CORE_WIDTH
    assert all(len(row) == 20 for row in matrix)


def test_training_reports_which_panel_alleles_it_could_not_cover(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240, alleles=("DRB1*15:01", "DRB3*02:02")))
    model = ci.train(ci.load_corpus(path), background=100)
    assert set(model.covered) == {"DRB1*15:01", "DRB3*02:02"}
    assert len(model.missing) == 5
    assert any("leaving the axis partial" in note for note in model.notes)
    assert any("not a neural predictor" in note for note in model.notes)


def test_an_empty_corpus_refuses_to_produce_a_model() -> None:
    with pytest.raises(ci.ClassIICorpusError, match="no_training_rows"):
        ci.train([])


# --------------------------------------------------------------------- the rank


def test_a_rank_is_a_percentage_and_lower_binds_more_tightly(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", _motif_rows(400))
    model = ci.train(ci.load_corpus(path), background=400)
    (allele,) = model.alleles
    good = ci.percentile_rank(model, "YAAAAAAAW" + "AAAAAA", allele)
    bad = ci.percentile_rank(model, "AAAAAAAAA" + "AAAAAA", allele)
    assert 0.0 <= good <= 100.0 and 0.0 <= bad <= 100.0
    assert good < bad, "a motif match must rank ahead of a peptide without it"


def test_scoring_refuses_what_it_cannot_score(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240))
    model = ci.train(ci.load_corpus(path), background=100)
    (allele,) = model.alleles
    with pytest.raises(ci.ClassIIModelError, match="allele_not_in_model"):
        ci.score(model, "A" * 15, "DRB4*01:01")
    with pytest.raises(ValueError, match="peptide_shorter_than_core"):
        ci.score(model, "AAAA", allele)
    with pytest.raises(ValueError, match="non_standard_residue"):
        ci.score(model, "AAAXAAAAAAAAAAA", allele)


def test_the_best_core_decides_the_score_not_the_average(tmp_path) -> None:
    """A peptide binds in its best register.

    The property that follows: a peptide's windows are a subset of any longer
    peptide containing it, so padding one can only ever raise its score. An
    averaging score would fall instead, diluting a strong core in a long weak
    peptide -- which is the failure this guards.
    """
    path = _corpus(tmp_path / "c.tsv", _motif_rows(400))
    model = ci.train(ci.load_corpus(path), background=200)
    (allele,) = model.alleles
    core = "YAAAAAAAW"
    alone = ci.score(model, core, allele)
    assert ci.score(model, core + "AAAAAA", allele) >= alone
    assert ci.score(model, "AAAAAA" + core, allele) >= alone
    assert ci.score(model, "AAA" + core + "AAA", allele) >= alone


# -------------------------------------------------------------- save and reload


def test_a_saved_model_reloads_to_the_same_ranks(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240))
    model = ci.train(ci.load_corpus(path), background=200)
    saved = ci.save(model, tmp_path / "model")
    again = ci.load_model(saved)
    (allele,) = model.alleles
    peptide = "YAAAAAAAW" + "AAAAAA"
    assert ci.percentile_rank(again, peptide, allele) == ci.percentile_rank(model, peptide, allele)
    assert again.method == model.method
    assert again.mean_auc == pytest.approx(model.mean_auc)
    assert ci.load_model(tmp_path / "model").method == model.method, "a directory resolves to the model file"


def test_a_model_written_by_a_different_layout_is_refused(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240))
    saved = ci.save(ci.train(ci.load_corpus(path), background=100), tmp_path / "m")
    payload = json.loads(saved.read_text(encoding="utf-8"))

    for mutate, match in (
        (lambda d: d.update(version=99), "model_version_unsupported"),
        (lambda d: d.update(alphabet="ACDEF"), "model_encoding_mismatch"),
        (lambda d: d.update(core_width=11), "model_encoding_mismatch"),
        (lambda d: [v.update(weights=[0.0]) for v in d["alleles"].values()], "model_weights_malformed"),
        (lambda d: d.update(alleles={}), "model_has_no_alleles"),
    ):
        broken = json.loads(json.dumps(payload))
        mutate(broken)
        target = tmp_path / "broken" / ci._MODEL_FILE
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ci.ClassIIModelError, match=match):
            ci.load_model(target)

    with pytest.raises(ci.ClassIIModelError, match="model_not_found"):
        ci.load_model(tmp_path / "nowhere" / "class_ii_pssm.json")
    unreadable = tmp_path / "bad" / ci._MODEL_FILE
    unreadable.parent.mkdir(exist_ok=True)
    unreadable.write_text("{not json", encoding="utf-8")
    with pytest.raises(ci.ClassIIModelError, match="model_unreadable"):
        ci.load_model(unreadable)


# ------------------------------------------------------- into the screen's table


def test_predictions_never_carry_an_invented_concentration(tmp_path) -> None:
    """The matrix is fitted to a transformed affinity. Inverting it would put a
    model output in a column the table labels ``nM``."""
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240))
    model = ci.train(ci.load_corpus(path), background=100)
    rows = ci.predict(model, ["YAAAAAAAW" + "AAAAAA"], list(model.covered))
    assert rows
    for row in rows:
        assert "ic50_nm" not in row
        assert not [key for key in row if key.endswith("_nm")]
        assert 0.0 <= float(row["percentile_rank"]) <= 100.0
    with pytest.raises(ci.ClassIIModelError, match="allele_not_in_model"):
        ci.predict(model, ["A" * 15], ["DRB4*01:01"])


def test_the_backend_contract_matches_what_register_backend_expects(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240))
    model = ci.train(ci.load_corpus(path), background=100)
    text = ci.predict_csv(model, ["YAAAAAAAW" + "AAAAAA"], list(model.covered))
    rows = ci.rows_from_csv(text)
    assert {"peptide", "allele", "percentile_rank"} <= set(rows[0])
    assert len(rows) == len(model.covered)


def test_registering_it_turns_the_axis_from_unavailable_to_available(tmp_path) -> None:
    """The gap this module exists to close."""
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240, alleles=ig.DR_PANEL))
    model = ci.train(ci.load_corpus(path), background=100)
    before = ig.class_ii_status()
    assert before["available"] is False
    assert before["reason"] == "no_permissively_licensed_class_ii_predictor"

    name = ci.register(model)
    after = ig.class_ii_status()
    assert after["available"] is True
    assert after["reason"] is None
    assert name in ig.registered_backends()

    rows = ci.rows_from_csv(ig.predict_class_ii("YAAAAAAAW" + "AAAAAA" * 4, backend=name))
    assert rows
    assert {row["allele"] for row in rows} == set(ig.DR_PANEL)


def test_the_written_table_says_whose_scale_it_is(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240, alleles=ig.DR_PANEL))
    model = ci.train(ci.load_corpus(path), background=100)
    ci.save(model, tmp_path / "model")
    sequence = "YAAAAAAAW" + "".join(random.Random(3).choices(_OTHER, k=60))
    report = ci.class_ii_report(sequence, directory=tmp_path / "model", out=tmp_path / "run")

    assert "NOT an IEDB consensus" in str(report["scale"])
    assert report["panel_alleles_missing"] == []
    assert any("ada_axis_will_report_rank_scale_unknown" in w for w in report["warnings"])
    ranks = [row["percentile_rank"] for row in report["strongest"]]
    assert ranks == sorted(ranks), "strongest first, which for a rank means ascending"

    written = list((tmp_path / "run" / "mhc_ii.tsv").open(encoding="utf-8"))
    rows = list(csv.DictReader(written, delimiter="\t"))
    assert rows
    assert all(row["method"] == ci.CLASS_II_METHOD for row in rows)
    assert all(not row["ic50_nm"] and not row["unit"] for row in rows)
    assert all(row["start"] for row in rows), "the window offset travels with the peptide"


def test_a_model_without_a_background_cannot_be_asked_for_a_rank(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240))
    model = ci.train(ci.load_corpus(path), background=100)
    (allele,) = model.alleles
    stripped = ci.ClassIIModel(
        alleles={
            allele: ci.AlleleModel(allele=allele, weights=model.alleles[allele].weights, trained_rows=1, binders=1)
        },
    )
    with pytest.raises(ci.ClassIIModelError, match="no_background_for_allele"):
        ci.percentile_rank(stripped, "YAAAAAAAW" + "AAAAAA", allele)


def test_validate_refuses_a_holdout_that_is_not_a_fraction(tmp_path) -> None:
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240))
    rows = ci.load_corpus(path).rows
    for holdout in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="holdout"):
            ci.validate(rows, holdout=holdout)


def test_the_held_out_split_never_puts_a_peptide_on_both_sides(tmp_path) -> None:
    """The same peptide is measured against several alleles and repeatedly, so
    a row split would score the model on peptides it trained on."""
    path = _corpus(tmp_path / "c.tsv", _motif_rows(400, alleles=("DRB1*15:01", "DRB3*02:02")))
    rows = ci.load_corpus(path).rows
    scores = ci.validate(rows, holdout=0.25, seed=11)
    assert set(scores) == {"DRB1*15:01", "DRB3*02:02"}
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert all(value > 0.8 for value in scores.values()), "the planted motif survives a clean split"


def test_the_default_model_directory_needs_data_path(monkeypatch) -> None:
    monkeypatch.delenv("RFD4_CLASS_II_MODEL", raising=False)
    monkeypatch.delenv("DATA_PATH", raising=False)
    with pytest.raises(ci.ClassIIModelError, match="data_path_unset"):
        ci.default_model_dir()
    monkeypatch.setenv("DATA_PATH", str(Path("x") / "y"))
    assert ci.default_model_dir().name == "iedb"
    monkeypatch.setenv("RFD4_CLASS_II_MODEL", str(Path("over") / "ride"))
    assert ci.default_model_dir() == Path("over") / "ride"


def test_the_background_is_drawn_from_the_corpus_composition_and_is_reproducible(tmp_path) -> None:
    """A rank is only meaningful against a background, and the same model must
    always give the same rank."""
    path = _corpus(tmp_path / "c.tsv", _motif_rows(240))
    report = ci.load_corpus(path)
    first = ci.train(report, background=300)
    second = ci.train(report, background=300)
    (allele,) = first.alleles
    assert first.alleles[allele].background == second.alleles[allele].background
    assert len(first.alleles[allele].background) == ci.BACKGROUND_GRID
    ordered = list(first.alleles[allele].background)
    assert ordered == sorted(ordered), "a quantile grid is ascending by construction"
    assert math.isfinite(ordered[0]) and math.isfinite(ordered[-1])
