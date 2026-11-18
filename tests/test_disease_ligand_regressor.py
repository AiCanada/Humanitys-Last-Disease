"""The three panel axes nothing else can answer, fitted from BindingDB.

``ligand_safety`` leaves ``nav1_5``, ``bcrp`` and ``bsep`` with no predictor:
ADMET-AI has no endpoint for them. These tests hold the module that fills them
to three things -- that the corpus filters drop what cannot be fitted, that the
fit learns a signal when one is present, and above all that a predicted
concentration can never be mistaken for a measured one, which is the error the
naming and :func:`assert_prediction_not_measured` exist to prevent.

Measured live on the real subset: 0.59, 0.64 and 0.49 RMSE in log10 units.
"""

from __future__ import annotations

import csv
import json
import sys

import numpy as np
import pytest

from rfproteina.disease import ligand_regressor as lr
from rfproteina.disease import ligand_safety as ls

_RINGS = ("c1ccccc1", "c1ccncc1", "c1ccsc1", "C1CCCCC1", "C1CCNCC1")
_LINKERS = ("", "C", "O")


@pytest.fixture(autouse=True)
def _small_forest(monkeypatch):
    """Sixty trees instead of a thousand.

    The production count made no measurable difference on validation, and a
    thousand trees over 2048 features turns a unit test into a minute. The code
    paths are identical.
    """
    monkeypatch.setattr(lr, "FOREST_TREES", 60)


def _learnable(axis="nav1_5", chains=3):
    """Molecules whose affinity really does depend on their structure.

    A Murcko scaffold keeps ring systems and the linkers between them and
    strips side chains, so varying only a side chain gives one scaffold, and a
    scaffold-balanced split over a handful of scaffolds leaves the validation
    and test folds empty. Two rings joined by a linker gives 75 distinct
    scaffolds here, while the target still rises with the side chain -- which a
    Morgan count fingerprint can see. Without a real signal a test can only
    show that fitting runs, not that it learns.
    """
    rows = []
    for first in _RINGS:
        for second in _RINGS:
            for linker in _LINKERS:
                for n in range(1, chains + 1):
                    smiles = "C" * n + first + linker + second
                    rows.append(lr.TrainingRow(axis=axis, smiles=smiles, log10_nanomolar=1.0 + 0.6 * n))
    return rows


def _subset(path, records, columns=lr.SUBSET_COLUMNS):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in columns})
    return path


def _record(smiles, nanomolar, axis="nav1_5", relation=""):
    return {
        "smiles": smiles,
        "panel_axis": axis,
        "affinity_nanomolar": str(nanomolar),
        "relation": relation,
    }


# ------------------------------------------------------- the coupling it closes


def test_the_axes_are_exactly_the_ones_admet_ai_cannot_answer() -> None:
    """If ADMET-AI ever gains an endpoint, this module's purpose shrinks.

    Asserted rather than commented so the two tuples cannot drift apart and
    leave this training a gap that is no longer there.
    """
    assert lr.UNCOVERED_AXES == ls.ADMET_AI_UNCOVERED_AXES
    for axis in lr.UNCOVERED_AXES:
        assert lr.AXIS_ACCESSIONS[axis] == ls.BINDINGDB_PANEL_ACCESSIONS[axis]
        assert axis not in ls.ADMET_AI_ENDPOINTS


def test_the_predicted_key_is_not_a_measured_one() -> None:
    assert lr.PREDICTED_KEY == "predicted_nanomolar"
    assert not lr.PREDICTED_KEY.endswith("_nm"), "the suffix that means somebody measured it"


# ------------------------------------------------------------------- the corpus


def test_a_censored_measurement_is_dropped_not_taken_at_its_bound(tmp_path) -> None:
    """``>50000`` means weaker than the assay saw, not 50,000 nM."""
    path = _subset(
        tmp_path / "s.tsv",
        [
            _record("c1ccccc1", 10.0),
            _record("Cc1ccccc1", 50000.0, relation=">"),
            _record("CCc1ccccc1", 1.0, relation="<"),
            _record("CCCc1ccccc1", 20.0, relation="="),
        ],
    )
    report = lr.load_corpus(path)
    assert report.dropped["measurement_censored"] == 2
    assert sorted(round(row.nanomolar, 3) for row in report.rows) == [10.0, 20.0]
    assert report.read == 4


def test_repeats_are_aggregated_by_geometric_mean(tmp_path) -> None:
    """Log-scale affinities, so the arithmetic mean would follow the weakest."""
    path = _subset(tmp_path / "s.tsv", [_record("c1ccccc1", 10.0), _record("c1ccccc1", 1000.0)])
    report = lr.load_corpus(path)
    (row,) = report.rows
    assert row.readings == 2
    assert row.nanomolar == pytest.approx(100.0), "geometric, not the arithmetic 505"
    assert report.aggregated == 1


def test_unusable_rows_are_dropped_with_a_reason(tmp_path) -> None:
    path = _subset(
        tmp_path / "s.tsv",
        [
            _record("c1ccccc1", 10.0),
            _record("", 10.0),
            _record("Cc1ccccc1", "not-a-number"),
            _record("CCc1ccccc1", 0.0),
            _record("CCCc1ccccc1", 10.0, axis="herg"),
        ],
    )
    report = lr.load_corpus(path)
    assert len(report.rows) == 1
    assert report.dropped["smiles_absent"] == 1
    assert report.dropped["measurement_unparsable"] == 1
    assert report.dropped["measurement_not_positive"] == 1
    assert report.read == 4, "an axis this module does not fit is not even read"


def test_a_partial_corpus_is_reported_rather_than_passed_off(tmp_path) -> None:
    path = _subset(tmp_path / "s.tsv", [_record("c1ccccc1", 10.0)])
    report = lr.load_corpus(path)
    assert report.covers_uncovered_axes is False
    assert set(report.per_axis) == {"nav1_5"}
    assert report.for_axis("bcrp") == []


def test_a_corpus_that_cannot_be_read_raises(tmp_path) -> None:
    with pytest.raises(lr.RegressorCorpusError, match="subset_not_found"):
        lr.load_corpus(tmp_path / "absent.tsv")
    bare = _subset(tmp_path / "bare.tsv", [], columns=("smiles", "panel_axis"))
    with pytest.raises(lr.RegressorCorpusError, match="subset_column_absent"):
        lr.load_corpus(bare)


# -------------------------------------------------------------- the units guard


def test_a_prediction_may_never_sit_under_a_measured_nm_key() -> None:
    """The one error this module's naming exists to prevent.

    ``ligand_safety.assert_no_probability_in_nm`` cannot catch this: a
    predicted nanomolar is a well-formed concentration, and the only thing
    wrong with it under an ``*_nm`` key is that nobody measured it.
    """
    lr.assert_prediction_not_measured({"nav1_5": {lr.PREDICTED_KEY: 420.0}})
    for bad in (
        {"nav1_5": {"herg_nm": 10.0}},
        {"a": {"b": [{"nav1_5_nm": 1.0}]}},
        {"nested": {"deeper": {"bsep_nm": 99.0}}},
    ):
        with pytest.raises(ValueError, match="measured-nM key"):
            lr.assert_prediction_not_measured(bad)


# -------------------------------------------------------------------- the model


def test_the_fit_learns_a_signal_that_is_really_there() -> None:
    """The claim a model has to earn: fitting that runs is not fitting that learns."""
    model = lr.train(_learnable(), replicates=2)
    (axis,) = model.axes
    score = model.axes[axis].score
    assert score is not None
    assert score.spearman > 0.8, f"a planted monotone signal should be easy, got {score.spearman}"
    assert score.rmse_log10 < 0.5
    assert score.replicates == 2
    assert score.test_rows > 0 and score.train_rows > 0


def test_the_score_reports_its_error_as_a_fold_not_only_a_logarithm() -> None:
    score = lr.AxisScore(
        axis="nav1_5",
        backend="forest",
        rmse_log10=1.0,
        mae_log10=0.8,
        r2=0.5,
        spearman=0.7,
        test_rows=100,
        train_rows=800,
        replicates=3,
    )
    assert score.fold_error == pytest.approx(10.0), "one log10 unit is a factor of ten"


def test_training_reports_which_axes_are_still_unanswered() -> None:
    model = lr.train(_learnable(axis="bcrp"), replicates=1)
    assert model.covered == ("bcrp",)
    assert set(model.missing) == {"nav1_5", "bsep"}
    assert any("still unanswered" in note for note in model.notes)
    assert any("not a measurement" in note for note in model.notes)


def test_an_empty_corpus_and_an_unknown_backend_are_refused() -> None:
    with pytest.raises(lr.RegressorCorpusError, match="no_training_rows"):
        lr.train([])
    with pytest.raises(ValueError, match="unknown backend"):
        lr.train(_learnable(), backend="magic")
    with pytest.raises(ValueError, match="unknown backend"):
        lr.evaluate(_learnable(), axis="nav1_5", backend="magic")


def test_too_little_data_is_refused_rather_than_fitted() -> None:
    rows = _learnable()[:20]
    with pytest.raises(lr.RegressorModelError, match="too_few_trainable_rows"):
        lr.train(rows, evaluate_first=False)


# --------------------------------------------------------------- the prediction


def test_a_prediction_carries_its_error_and_its_reach() -> None:
    model = lr.train(_learnable(), replicates=1)
    out = lr.predict(model, "CCCCc1ccccc1")
    values = out.predictions["nav1_5"]
    assert values[lr.PREDICTED_KEY] > 0
    assert values["accession"] == "Q14524"
    assert values["model_rmse_log10"] is not None
    assert values["model_fold_error"] >= 1.0
    assert 0.0 <= values["nearest_training_similarity"] <= 1.0
    assert values["within_applicability"] is True, "a training molecule is inside the domain"
    assert out.backend == "forest"
    assert "not a measurement" in out.unit


def test_a_prediction_record_holds_no_measured_key() -> None:
    model = lr.train(_learnable(), replicates=1)
    record = lr.predict(model, "CCCc1ccncc1").as_record()
    flat = json.dumps(record)
    assert f'"{lr.PREDICTED_KEY}"' in flat
    assert '_nm"' not in flat
    lr.assert_prediction_not_measured(record["predictions"])
    assert any("not a measurement" in note for note in record["notes"])
    assert any("BindingDB" in note for note in record["notes"]), "the licence travels with the prediction"


def test_chemistry_the_model_never_saw_is_flagged_as_extrapolation() -> None:
    """A model that answers confidently for anything is the failure mode here."""
    model = lr.train(_learnable(), replicates=1)
    # A platinum complex shares no substructure with substituted ring systems.
    far = lr.predict(model, "O=[Pt](=O)(N)N")
    values = far.predictions["nav1_5"]
    assert values["within_applicability"] is False
    assert far.extrapolated == ("nav1_5",)
    assert far.notes[0].startswith("EXTRAPOLATION")


def test_predicting_refuses_what_it_cannot_read() -> None:
    model = lr.train(_learnable(), replicates=1)
    with pytest.raises(ValueError, match="empty_smiles"):
        lr.predict(model, "   ")
    with pytest.raises(ValueError, match="smiles_unparsable"):
        lr.predict(model, "ThisIsNotSmiles(((")
    with pytest.raises(lr.RegressorModelError, match="axis_not_in_model"):
        lr.predict(model, "c1ccccc1", axes=["bsep"])


def test_the_nanomolar_shortcut_matches_the_record() -> None:
    model = lr.train(_learnable(), replicates=1)
    out = lr.predict(model, "CCc1ccccc1")
    assert out.nanomolar["nav1_5"] == out.predictions["nav1_5"][lr.PREDICTED_KEY]


# ------------------------------------------------------------- save and reload


def test_a_saved_regressor_reloads_to_the_same_predictions(tmp_path) -> None:
    model = lr.train(_learnable(), replicates=1)
    manifest = lr.save(model, tmp_path / "model")
    again = lr.load_regressor(manifest)
    smiles = "CCCCCc1ccsc1"
    assert lr.predict(again, smiles).nanomolar == lr.predict(model, smiles).nanomolar
    assert again.method == model.method
    assert again.backend == model.backend
    assert again.scores["nav1_5"].rmse_log10 == pytest.approx(model.scores["nav1_5"].rmse_log10)
    assert lr.load_regressor(tmp_path / "model").backend == "forest", "a directory resolves to the manifest"


def test_a_regressor_written_by_another_layout_is_refused(tmp_path) -> None:
    model = lr.train(_learnable(), replicates=1)
    manifest = lr.save(model, tmp_path / "model")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    for mutate, match in (
        (lambda d: d.update(version=99), "manifest_version_unsupported"),
        (lambda d: d["morgan"].update(radius=7), "manifest_fingerprint_mismatch"),
        (lambda d: d["morgan"].update(bits=64), "manifest_fingerprint_mismatch"),
        (lambda d: d.update(backend="magic"), "unknown_backend"),
        (lambda d: d.update(axes={}), "manifest_has_no_axes"),
    ):
        broken = json.loads(json.dumps(payload))
        mutate(broken)
        target = tmp_path / "model" / lr._MANIFEST
        target.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(lr.RegressorUnavailableError, match=match):
            lr.load_regressor(target)

    target = tmp_path / "model" / lr._MANIFEST
    target.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "model" / "nav1_5.joblib").unlink()
    with pytest.raises(lr.RegressorUnavailableError, match="estimator_not_found"):
        lr.load_regressor(target)
    with pytest.raises(lr.RegressorUnavailableError, match="manifest_not_found"):
        lr.load_regressor(tmp_path / "nowhere")


def test_a_missing_model_is_reported_against_the_directory_asked_for(tmp_path) -> None:
    """Not against its parent.

    Resolving the directory by ``is_dir()`` reads a model directory that does
    not exist yet as a file path, then looks for the manifest one level up --
    so pointing at an untrained ``regressor/`` complained that the manifest was
    missing from its parent, which is a different and existing directory.
    """
    asked = tmp_path / "regressor"
    with pytest.raises(lr.RegressorUnavailableError, match="manifest_not_found") as caught:
        lr.load_regressor(asked)
    assert str(asked) in str(caught.value)
    assert str(tmp_path / lr._MANIFEST) not in str(caught.value)

    # And a manifest named explicitly still resolves to its own directory.
    model = lr.train(_learnable(), replicates=1)
    manifest = lr.save(model, tmp_path / "saved")
    assert lr.load_regressor(manifest).backend == "forest"
    assert lr.load_regressor(tmp_path / "saved").backend == "forest"


# ------------------------------------------------------------------- the paths


def test_the_default_locations_need_data_path(monkeypatch) -> None:
    monkeypatch.delenv("RFD4_LIGAND_REGRESSOR", raising=False)
    monkeypatch.delenv("DATA_PATH", raising=False)
    with pytest.raises(lr.RegressorCorpusError, match="data_path_unset"):
        lr.default_subset()
    with pytest.raises(lr.RegressorUnavailableError, match="data_path_unset"):
        lr.default_model_dir()
    monkeypatch.setenv("DATA_PATH", "/tmp/x")
    assert lr.default_subset().name == "bindingdb_panel_subset.tsv"
    assert lr.default_model_dir().name == "regressor"
    monkeypatch.setenv("RFD4_LIGAND_REGRESSOR", "/tmp/over")
    assert lr.default_model_dir().name == "over"


def test_the_chemprop_backend_refuses_loudly_when_absent(monkeypatch) -> None:
    """Held whether or not chemprop is installed, so it tests the guard.

    The chemprop path's real behaviour is covered by the live run; what matters
    here is that its absence is reported rather than silently downgraded to the
    forest, which would change the method id under a caller's feet.
    """
    monkeypatch.setitem(sys.modules, "chemprop", None)
    with pytest.raises(lr.RegressorUnavailableError, match="chemprop_unavailable"):
        lr._fit_chemprop(_learnable(), list(range(100)), list(range(100, 120)))
    with pytest.raises(lr.RegressorUnavailableError, match="chemprop_unavailable"):
        lr._split_indices(_learnable(), 1, 0)


def test_fingerprints_keep_track_of_what_did_not_parse() -> None:
    matrix, kept = lr.fingerprints(["c1ccccc1", "ThisIsNotSmiles(((", "CCO"])
    assert kept == [0, 2], "the broken one is dropped and its index is not claimed"
    assert matrix.shape == (2, lr.MORGAN_BITS)
    empty, none = lr.fingerprints(["nonsense((("])
    assert none == [] and empty.shape == (0, lr.MORGAN_BITS)


def test_similarity_is_one_against_itself_and_low_against_something_else() -> None:
    matrix, _kept = lr.fingerprints(["CCCCc1ccccc1"])
    assert lr._similarity(matrix, matrix[0]) == pytest.approx(1.0)
    other, _kept = lr.fingerprints(["O=[Pt](=O)(N)N"])
    assert lr._similarity(matrix, other[0]) < lr.APPLICABILITY_SIMILARITY
    assert np.isnan(lr._similarity(None, matrix[0]))
