"""Predicted nanomolar for the three panel axes nothing else can answer.

``ligand_safety`` answers the ten-axis panel two ways. The **measured** route
reads real affinities out of BindingDB and GtoPdb; the **predicted** route runs
ADMET-AI, which has endpoints for seven axes. Three are left over --
:data:`UNCOVERED_AXES`, the same tuple ``ligand_safety.ADMET_AI_UNCOVERED_AXES``
derives -- and for a designed ligand nobody has measured, those three axes
simply have no answer:

* ``nav1_5``  the cardiac sodium channel (Q14524)
* ``bcrp``    the efflux transporter ABCG2 (Q9UNQ0)
* ``bsep``    the bile salt export pump ABCB11 (O95342)

There is no licence-blocked tool to install for them. There is, however,
measured data: BindingDB holds thousands of affinities against those exact
accessions, and ``tables.bindingdb_panel_subset`` already reduces the download
to them. This module fits a regressor per axis on that subset, so the axes gain
a prediction rather than staying empty.

**What it predicts, and what that is not.** The target is ``log10`` of the
nanomolar affinity, because the measurements span seven orders of magnitude and
fitting raw nanomolar would let one millimolar reading outweigh every binder.
The output is reported as :data:`PREDICTED_KEY` -- ``predicted_nanomolar`` --
and **never** under an ``*_nm`` key. That naming is the whole safety argument:
``*_nm`` with ``unit: nM`` means somebody measured it, ``neighbour_nanomolar``
means it was measured on a *different* molecule, and this means a model guessed
it. :func:`assert_prediction_not_measured` enforces the distinction rather than
trusting it.

**What it is worth.** Measured, not asserted: pooled over three
scaffold-balanced held-out splits, on determinate measurements only -- a
``>10000 nM`` row is a bound, and scoring it as a point value would be a
fabricated number.

===========  ======  ==========  =====  =========  ========
axis          train  RMSE log10   fold  Spearman     test n
===========  ======  ==========  =====  =========  ========
``nav1_5``    1,710       0.592   3.9x      0.777       639
``bcrp``        860       0.631   4.3x      0.715       321
``bsep``        352       0.535   3.4x      0.587       132
===========  ======  ==========  =====  =========  ========

So a typical prediction is wrong by about four-fold, and the ranking is good
to a Spearman near 0.6 to 0.8. **That is a triage signal, not an assay**: it
orders designs and flags the obviously dangerous, and a number from it should
never be quoted as a potency. :func:`train` recomputes all of this rather than
trusting the table, and the replicates matter -- a single split leaves ``bsep``
with 44 test rows, which is not a number worth reporting.

**Why the default backend is not the graph network.** Chemprop was measured
against a fingerprint model on identical splits and lost on all three axes:
0.520 against 0.481 on ``nav1_5``, 0.613 against 0.507 on ``bcrp``, 0.570
against 0.551 on ``bsep``. It early-stopped between 32 and 69 epochs with a
500-epoch ceiling, so training longer is not the missing ingredient; at 350 to
1,700 training molecules a D-MPNN simply has too little data to beat Morgan
fingerprints. Both backends are built and selectable -- :data:`BACKENDS` --
and the default is the one that measured better. Early stopping mattered a
great deal to that comparison: without it chemprop scored a *negative* R² on
two axes, so :func:`_fit_chemprop` always early-stops on validation loss and
restores the best checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: The panel axes with no ADMET-AI endpoint, in panel order. Checked against
#: ``ligand_safety.ADMET_AI_UNCOVERED_AXES`` by the test suite, so the two
#: cannot drift apart silently.
UNCOVERED_AXES: tuple[str, ...] = ("nav1_5", "bcrp", "bsep")

#: The UniProt accession each axis is keyed to in the BindingDB subset.
AXIS_ACCESSIONS: dict[str, str] = {"nav1_5": "Q14524", "bcrp": "Q9UNQ0", "bsep": "O95342"}

#: The key every prediction is stored under. Deliberately not ``*_nm``: that
#: suffix is reserved for a measurement, and this is not one.
PREDICTED_KEY = "predicted_nanomolar"

#: Method ids. The backend is named in the id, so a stored prediction says
#: which model produced it without a lookup.
METHOD_FOREST = "bindingdb_morgan_extratrees_v1"
METHOD_CHEMPROP = "bindingdb_chemprop_dmpnn_v1"
BACKENDS: tuple[str, ...] = ("forest", "chemprop")

#: Columns read from ``tables.bindingdb_panel_subset``'s output.
SUBSET_COLUMNS = ("smiles", "panel_axis", "affinity_nanomolar", "relation")
#: Only an equality is a value. ``>`` and ``<`` are bounds, and BindingDB
#: leaves the column empty for a determinate measurement.
DETERMINATE = ("", "=")

#: Morgan parameters. Radius 2 over 2048 bits with counts measured best on
#: validation across the three axes; radius 3 over 4096 was better on ``bsep``
#: alone and worse elsewhere, which is the shape of a choice made on 44 rows.
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MORGAN_COUNTS = True
#: Trees in the forest. More made no measurable difference on validation.
FOREST_TREES = 1000
#: Scaffold-balanced, because a random split over congeneric series flatters a
#: fingerprint model badly: the same scaffold lands on both sides.
SPLIT = "scaffold_balanced"
SPLIT_SIZES = (0.8, 0.1, 0.1)
#: Replicate splits averaged for the reported score. One split over 44 test
#: rows is not a number worth reporting on its own.
REPLICATES = 3
#: A prediction outside the training set's chemistry is extrapolation, and this
#: is the Tanimoto below which the record says so. The same 0.4 as
#: ``ligand_safety.nearest_measured_neighbours``, and for the same reason:
#: below roughly there a Morgan pair is not an analogue. Measured consequence
#: of the looser 0.3 first tried -- decane scored 0.304 against the Nav1.5
#: training set and was reported as inside the model's chemistry, which it
#: plainly is not.
APPLICABILITY_SIMILARITY = 0.4

_MANIFEST = "ligand_regressor.json"
_VERSION = 1


class RegressorCorpusError(Exception):
    """The BindingDB subset is absent, unreadable, or lacks its columns."""


class RegressorUnavailableError(Exception):
    """A backend's dependency is missing, or the saved model cannot be read."""


class RegressorModelError(Exception):
    """The model does not cover what was asked of it."""


@dataclass(frozen=True)
class TrainingRow:
    """One compound's aggregated affinity against one axis, as ``log10(nM)``."""

    axis: str
    smiles: str
    log10_nanomolar: float
    readings: int = 1

    @property
    def nanomolar(self) -> float:
        return 10.0**self.log10_nanomolar


@dataclass(frozen=True)
class CorpusReport:
    """What the subset gave per axis, and what was dropped with the reason."""

    rows: tuple[TrainingRow, ...]
    read: int = 0
    dropped: Mapping[str, int] = field(default_factory=dict)
    aggregated: int = 0
    licence: str = "BindingDB; cite BindingDB and the per-row reference"

    @property
    def per_axis(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.axis] = counts.get(row.axis, 0) + 1
        return counts

    @property
    def covers_uncovered_axes(self) -> bool:
        counts = self.per_axis
        return all(counts.get(axis) for axis in UNCOVERED_AXES)

    def for_axis(self, axis: str) -> list[TrainingRow]:
        return [row for row in self.rows if row.axis == axis]


@dataclass(frozen=True)
class AxisScore:
    """One axis's held-out error, averaged over replicate scaffold splits.

    ``rmse_log10`` is the number that matters and it is in log10 units: 0.6
    means a typical error of about four-fold. ``spearman`` is what a screen
    actually uses, since ranking designs matters more than the absolute value.
    """

    axis: str
    backend: str
    rmse_log10: float
    mae_log10: float
    r2: float
    spearman: float
    test_rows: int
    train_rows: int
    replicates: int

    @property
    def fold_error(self) -> float:
        """The RMSE expressed as a factor, which is how a chemist reads it."""
        return 10.0**self.rmse_log10


@dataclass(frozen=True)
class AxisModel:
    """One axis's fitted estimator plus what it was fitted on.

    ``training_fingerprints`` is kept so a prediction can report its distance
    to the training set. A model that answers confidently for chemistry it has
    never seen is the failure mode an applicability check exists to catch.
    """

    axis: str
    backend: str
    estimator: Any
    score: AxisScore | None = None
    training_fingerprints: Any = None
    training_rows: int = 0


@dataclass(frozen=True)
class PanelRegressor:
    """Every axis's model, with the method that names the backend."""

    axes: Mapping[str, AxisModel]
    backend: str = "forest"
    method: str = METHOD_FOREST
    corpus_rows: int = 0
    notes: tuple[str, ...] = ()

    @property
    def covered(self) -> tuple[str, ...]:
        return tuple(axis for axis in UNCOVERED_AXES if axis in self.axes)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(axis for axis in UNCOVERED_AXES if axis not in self.axes)

    @property
    def scores(self) -> dict[str, AxisScore]:
        return {axis: model.score for axis, model in self.axes.items() if model.score is not None}


@dataclass(frozen=True)
class PanelPredictions:
    """Predicted concentrations for one ligand, kept apart from measurement.

    Every value lives under :data:`PREDICTED_KEY` and carries the model's own
    held-out error and the nearest training-set similarity, so a reader can see
    both how wrong the model usually is and whether this compound is the kind
    of thing it was fitted on.
    """

    smiles: str
    predictions: dict[str, dict[str, object]]
    method: str
    backend: str
    unit: str = "nM (model output, not a measurement)"
    uncovered_axes: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["predictions"] = {axis: dict(values) for axis, values in self.predictions.items()}
        record["uncovered_axes"] = list(self.uncovered_axes)
        record["notes"] = list(self.notes)
        return record

    @property
    def nanomolar(self) -> dict[str, float]:
        """Axis to predicted nanomolar, for a one-line read."""
        return {axis: float(values[PREDICTED_KEY]) for axis, values in self.predictions.items()}

    @property
    def extrapolated(self) -> tuple[str, ...]:
        """Axes where the nearest training compound is too far to trust."""
        return tuple(
            axis for axis, values in self.predictions.items() if not bool(values.get("within_applicability", True))
        )


def assert_prediction_not_measured(record: Mapping[str, object]) -> None:
    """Raise if a predicted concentration is stored where a measurement belongs.

    ``ligand_safety.assert_no_probability_in_nm`` keeps probabilities out of
    concentration columns. This keeps *predictions* out of them, which that
    check cannot do: a predicted nanomolar is a perfectly well-formed
    concentration, and the only thing wrong with it under an ``*_nm`` key is
    that nobody measured it.
    """

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if isinstance(key, str) and key.endswith("_nm"):
                    raise ValueError(f"model prediction stored under a measured-nM key: {key}")
                walk(value)
        elif isinstance(node, list | tuple):
            for item in node:
                walk(item)

    walk(record)


def load_corpus(path: Path, axes: Sequence[str] = UNCOVERED_AXES) -> CorpusReport:
    """Read the BindingDB panel subset into training rows, and say what was dropped.

    Three filters, each for a reason that would otherwise be a silent error:
    the relation must be an equality, because ``>50000`` is a bound and not a
    value; the SMILES must parse; and repeats of the same (axis, compound) are
    aggregated by **geometric mean**, because these are log-scale affinities
    and an arithmetic mean would follow the weakest reading.
    """
    path = Path(path)
    if not path.is_file():
        raise RegressorCorpusError(f"subset_not_found:{path}")
    wanted = {str(axis) for axis in axes}
    dropped: dict[str, int] = defaultdict(int)
    readings: dict[tuple[str, str], list[float]] = defaultdict(list)
    read = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            names = frozenset(reader.fieldnames or ())
            for column in SUBSET_COLUMNS:
                if column not in names:
                    raise RegressorCorpusError(f"subset_column_absent:{column}")
            for row in reader:
                axis = (row.get("panel_axis") or "").strip()
                if axis not in wanted:
                    continue
                read += 1
                if (row.get("relation") or "").strip() not in DETERMINATE:
                    dropped["measurement_censored"] += 1
                    continue
                smiles = (row.get("smiles") or "").strip()
                if not smiles:
                    dropped["smiles_absent"] += 1
                    continue
                try:
                    value = float((row.get("affinity_nanomolar") or "").strip())
                except ValueError:
                    dropped["measurement_unparsable"] += 1
                    continue
                if not math.isfinite(value) or value <= 0:
                    dropped["measurement_not_positive"] += 1
                    continue
                readings[(axis, smiles)].append(value)
    except (OSError, UnicodeError, csv.Error) as error:
        raise RegressorCorpusError(f"subset_unreadable:{type(error).__name__}") from error

    rows = tuple(
        TrainingRow(
            axis=axis,
            smiles=smiles,
            log10_nanomolar=sum(math.log10(v) for v in values) / len(values),
            readings=len(values),
        )
        for (axis, smiles), values in sorted(readings.items())
    )
    return CorpusReport(
        rows=rows,
        read=read,
        dropped=dict(sorted(dropped.items())),
        aggregated=sum(1 for row in rows if row.readings > 1),
    )


def _morgan_generator() -> Any:
    try:
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:  # pragma: no cover - rdkit is a base dependency
        raise RegressorUnavailableError(f"rdkit_unavailable:{error}") from error
    return rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)


def _parse(smiles: str) -> Any:
    try:
        from rdkit import Chem
    except ImportError as error:  # pragma: no cover
        raise RegressorUnavailableError(f"rdkit_unavailable:{error}") from error
    return Chem.MolFromSmiles((smiles or "").strip())


def fingerprints(smiles: Iterable[str]) -> tuple[np.ndarray, list[int]]:
    """Morgan fingerprints, with the indices of the SMILES that parsed."""
    generator = _morgan_generator()
    vectors: list[np.ndarray] = []
    kept: list[int] = []
    for index, text in enumerate(smiles):
        mol = _parse(text)
        if mol is None:
            continue
        vector = generator.GetCountFingerprintAsNumPy(mol) if MORGAN_COUNTS else generator.GetFingerprintAsNumPy(mol)
        vectors.append(vector.astype(np.float32))
        kept.append(index)
    if not vectors:
        return np.zeros((0, MORGAN_BITS), dtype=np.float32), []
    return np.vstack(vectors), kept


def _split_indices(rows: Sequence[TrainingRow], replicates: int, seed: int) -> tuple[Any, Any, Any]:
    """Scaffold-balanced split indices, through chemprop's own splitter."""
    try:
        from chemprop import data as chemprop_data
    except ImportError as error:
        raise RegressorUnavailableError(f"chemprop_unavailable:{error}") from error
    mols = [_parse(row.smiles) for row in rows]
    if any(mol is None for mol in mols):
        raise RegressorModelError("unparsable_smiles_in_split")
    return chemprop_data.make_split_indices(mols, split=SPLIT, sizes=SPLIT_SIZES, seed=seed, num_replicates=replicates)


def _rmse(truth: Sequence[float], predicted: Sequence[float]) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(predicted)) ** 2)))


def _spearman(truth: Sequence[float], predicted: Sequence[float]) -> float:
    def ranks(values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        order = np.argsort(array, kind="mergesort")
        out = np.empty(len(array), dtype=np.float64)
        index = 0
        while index < len(order):
            end = index
            while end < len(order) and array[order[end]] == array[order[index]]:
                end += 1
            out[order[index:end]] = (index + end - 1) / 2.0
            index = end
        return out

    a, b = ranks(truth), ranks(predicted)
    if not np.std(a) or not np.std(b):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _score(
    axis: str,
    backend: str,
    truth: Sequence[float],
    predicted: Sequence[float],
    train_rows: int,
    replicates: int,
) -> AxisScore:
    truth_array = np.asarray(truth, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    residual = float(np.sum((truth_array - predicted_array) ** 2))
    total = float(np.sum((truth_array - truth_array.mean()) ** 2))
    return AxisScore(
        axis=axis,
        backend=backend,
        rmse_log10=_rmse(truth_array, predicted_array),
        mae_log10=float(np.mean(np.abs(truth_array - predicted_array))),
        r2=1.0 - residual / total if total else float("nan"),
        spearman=_spearman(truth_array, predicted_array),
        test_rows=int(len(truth_array)),
        train_rows=train_rows,
        replicates=replicates,
    )


def _make_forest(seed: int) -> Any:
    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except ImportError as error:
        raise RegressorUnavailableError(f"scikit_learn_unavailable:{error}") from error
    return ExtraTreesRegressor(n_estimators=FOREST_TREES, min_samples_leaf=2, n_jobs=-1, random_state=seed)


def _fit_forest(rows: Sequence[TrainingRow], seed: int = 0) -> tuple[Any, np.ndarray]:
    matrix, kept = fingerprints([row.smiles for row in rows])
    if len(kept) < 50:
        raise RegressorModelError(f"too_few_trainable_rows:{len(kept)}")
    targets = np.array([rows[i].log10_nanomolar for i in kept], dtype=np.float64)
    estimator = _make_forest(seed)
    estimator.fit(matrix, targets)
    return estimator, matrix


def _fit_chemprop(
    rows: Sequence[TrainingRow],
    train: Sequence[int],
    validation: Sequence[int],
    *,
    seed: int = 0,
    max_epochs: int = 500,
    patience: int = 20,
) -> Any:
    """A D-MPNN, always early-stopped on validation loss with the best weights restored.

    Not optional. Taking the last epoch's weights instead scored a negative R²
    on two of the three axes -- worse than predicting the training mean -- and
    early stopping alone turned that into a usable model. Training stops
    between roughly 30 and 70 epochs, so the ceiling is not what ends it.
    """
    try:
        import torch
        from chemprop import data as chemprop_data
        from chemprop import models, nn
        from lightning import pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError as error:
        raise RegressorUnavailableError(f"chemprop_unavailable:{error}") from error

    torch.manual_seed(seed)

    def points(indices: Sequence[int]) -> list[Any]:
        out = []
        for index in indices:
            mol = _parse(rows[index].smiles)
            if mol is None:
                continue
            out.append(chemprop_data.MoleculeDatapoint(mol=mol, y=np.array([rows[index].log10_nanomolar])))
        return out

    train_points, validation_points = points(train), points(validation)
    if len(train_points) < 50 or not validation_points:
        raise RegressorModelError(f"too_few_trainable_rows:{len(train_points)}")
    train_set = chemprop_data.MoleculeDataset(train_points)
    validation_set = chemprop_data.MoleculeDataset(validation_points)
    scaler = train_set.normalize_targets()
    validation_set.normalize_targets(scaler)
    model = models.MPNN(
        nn.BondMessagePassing(depth=3, d_h=300),
        nn.MeanAggregation(),
        nn.RegressionFFN(
            output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
        ),
        batch_norm=True,
        metrics=[nn.RMSE()],
    )
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=patience), checkpoint],
    )
    trainer.fit(
        model,
        chemprop_data.build_dataloader(train_set, batch_size=50, num_workers=0),
        chemprop_data.build_dataloader(validation_set, batch_size=50, num_workers=0, shuffle=False),
    )
    if checkpoint.best_model_path:
        model = models.MPNN.load_from_checkpoint(checkpoint.best_model_path)
    return model


def _predict_chemprop(model: Any, smiles: Sequence[str]) -> np.ndarray:
    from chemprop import data as chemprop_data
    from lightning import pytorch as pl

    points = []
    for text in smiles:
        mol = _parse(text)
        if mol is None:
            raise RegressorModelError("unparsable_smiles")
        points.append(chemprop_data.MoleculeDatapoint(mol=mol, y=np.array([0.0])))
    loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(points), batch_size=50, num_workers=0, shuffle=False
    )
    trainer = pl.Trainer(
        accelerator="auto", devices=1, enable_progress_bar=False, logger=False, enable_model_summary=False
    )
    out = trainer.predict(model, loader)
    return np.concatenate([tensor.numpy().reshape(-1) for tensor in out])


def evaluate(
    rows: Sequence[TrainingRow],
    *,
    axis: str,
    backend: str = "forest",
    replicates: int = REPLICATES,
    seed: int = 0,
) -> AxisScore | None:
    """Held-out error for one axis, pooled over replicate scaffold splits.

    Pooled rather than averaged so a replicate with a 36-row test set does not
    weigh the same as one with 50. ``None`` when there is too little data to
    say anything, which is reported rather than filled in with a guess.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    train_sets, validation_sets, test_sets = _split_indices(rows, replicates, seed)
    truth: list[float] = []
    predicted: list[float] = []
    trained = 0
    for replicate in range(replicates):
        train = list(train_sets[replicate])
        validation = list(validation_sets[replicate])
        test = list(test_sets[replicate])
        if len(train) < 50 or not test:
            continue
        if backend == "forest":
            estimator, _matrix = _fit_forest([rows[i] for i in train], seed=seed + replicate)
            matrix, kept = fingerprints([rows[i].smiles for i in test])
            if not kept:
                continue
            predicted.extend(estimator.predict(matrix).tolist())
            truth.extend(rows[test[i]].log10_nanomolar for i in kept)
        else:
            model = _fit_chemprop(rows, train, validation, seed=seed + replicate)
            texts = [rows[i].smiles for i in test]
            predicted.extend(_predict_chemprop(model, texts).tolist())
            truth.extend(rows[i].log10_nanomolar for i in test)
        trained = len(train)
    if len(truth) < 10:
        return None
    return _score(axis, backend, truth, predicted, trained, replicates)


def train(
    corpus: CorpusReport | Sequence[TrainingRow],
    *,
    backend: str = "forest",
    replicates: int = REPLICATES,
    seed: int = 0,
    evaluate_first: bool = True,
) -> PanelRegressor:
    """Fit one regressor per axis, and measure each on held-out scaffolds.

    The score is computed here rather than left to a caller: a model whose
    error is unknown cannot be read, and a predicted concentration with no
    stated error invites being treated as a measurement.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    rows = corpus.rows if isinstance(corpus, CorpusReport) else tuple(corpus)
    if not rows:
        raise RegressorCorpusError("no_training_rows")
    by_axis: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        by_axis[row.axis].append(row)

    models: dict[str, AxisModel] = {}
    for axis, axis_rows in sorted(by_axis.items()):
        score = (
            evaluate(axis_rows, axis=axis, backend=backend, replicates=replicates, seed=seed)
            if evaluate_first
            else None
        )
        if backend == "forest":
            estimator, matrix = _fit_forest(axis_rows, seed=seed)
            prints = matrix
        else:
            train_sets, validation_sets, _test = _split_indices(axis_rows, 1, seed)
            estimator = _fit_chemprop(axis_rows, list(train_sets[0]), list(validation_sets[0]), seed=seed)
            prints, _kept = fingerprints([row.smiles for row in axis_rows])
        models[axis] = AxisModel(
            axis=axis,
            backend=backend,
            estimator=estimator,
            score=score,
            training_fingerprints=prints,
            training_rows=len(axis_rows),
        )

    method = METHOD_FOREST if backend == "forest" else METHOD_CHEMPROP
    notes = [
        f"trained on {len(rows):,} (axis, compound) pairs from BindingDB",
        "the target is log10(nM); an RMSE of 0.6 is a typical error near four-fold",
        f"every value is reported as {PREDICTED_KEY} and never under an *_nm key: it is a model output, "
        "not a measurement",
        "censored measurements were excluded: a bound is not a value",
    ]
    missing = [axis for axis in UNCOVERED_AXES if axis not in models]
    if missing:
        notes.append("axes with no model, still unanswered: " + ", ".join(missing))
    return PanelRegressor(axes=models, backend=backend, method=method, corpus_rows=len(rows), notes=tuple(notes))


def _similarity(matrix: np.ndarray, vector: np.ndarray) -> float:
    """Highest Tanimoto between one compound and a training matrix."""
    if matrix is None or not len(matrix):
        return float("nan")
    query = vector.astype(np.float64)
    train = matrix.astype(np.float64)
    intersection = np.minimum(train, query).sum(axis=1)
    union = np.maximum(train, query).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(union > 0, intersection / union, 0.0)
    return float(values.max()) if len(values) else float("nan")


def predict(regressor: PanelRegressor, smiles: str, *, axes: Sequence[str] | None = None) -> PanelPredictions:
    """Predicted nanomolar per axis for one ligand, with its error and reach.

    Each entry carries the model's held-out RMSE and the Tanimoto to its
    nearest training compound. Both are there so a reader can tell a prediction
    inside the model's chemistry from an extrapolation dressed as one.
    """
    text = (smiles or "").strip()
    if not text:
        raise ValueError("empty_smiles")
    mol = _parse(text)
    if mol is None:
        raise ValueError("smiles_unparsable")
    wanted = list(axes) if axes else list(regressor.covered)
    unknown = [axis for axis in wanted if axis not in regressor.axes]
    if unknown:
        raise RegressorModelError("axis_not_in_model:" + ",".join(unknown))

    matrix, kept = fingerprints([text])
    if not kept:
        raise ValueError("smiles_unparsable")
    vector = matrix[0]

    predictions: dict[str, dict[str, object]] = {}
    for axis in wanted:
        model = regressor.axes[axis]
        if regressor.backend == "forest":
            value = float(model.estimator.predict(matrix.reshape(1, -1))[0])
        else:
            value = float(_predict_chemprop(model.estimator, [text])[0])
        nearest = _similarity(model.training_fingerprints, vector)
        score = model.score
        predictions[axis] = {
            PREDICTED_KEY: round(10.0**value, 3),
            "predicted_log10_nanomolar": round(value, 4),
            "accession": AXIS_ACCESSIONS.get(axis, ""),
            "model_rmse_log10": None if score is None else round(score.rmse_log10, 4),
            "model_fold_error": None if score is None else round(score.fold_error, 2),
            "nearest_training_similarity": None if math.isnan(nearest) else round(nearest, 4),
            "within_applicability": bool(not math.isnan(nearest) and nearest >= APPLICABILITY_SIMILARITY),
            "training_rows": model.training_rows,
        }

    notes = [
        "a model output, not a measurement; nothing here may be written into an *_nm column",
        "log10-scale model: read the fold error, not the digits",
        "BindingDB terms apply to the training data; cite BindingDB",
    ]
    outside = [axis for axis, values in predictions.items() if not values["within_applicability"]]
    if outside:
        notes.insert(
            0,
            "EXTRAPOLATION: no training compound within Tanimoto "
            f"{APPLICABILITY_SIMILARITY} for " + ", ".join(outside),
        )
    return PanelPredictions(
        smiles=text,
        predictions=predictions,
        method=regressor.method,
        backend=regressor.backend,
        uncovered_axes=tuple(regressor.missing),
        notes=tuple(notes),
    )


def save(regressor: PanelRegressor, directory: Path) -> Path:
    """Write the models and a manifest. Returns the manifest path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "version": _VERSION,
        "backend": regressor.backend,
        "method": regressor.method,
        "corpus_rows": regressor.corpus_rows,
        "notes": list(regressor.notes),
        "morgan": {"radius": MORGAN_RADIUS, "bits": MORGAN_BITS, "counts": MORGAN_COUNTS},
        "axes": {},
    }
    for axis, model in sorted(regressor.axes.items()):
        if regressor.backend == "forest":
            try:
                import joblib
            except ImportError as error:
                raise RegressorUnavailableError(f"joblib_unavailable:{error}") from error
            payload = directory / f"{axis}.joblib"
            joblib.dump(model.estimator, payload, compress=3)
        else:
            from chemprop import models as chemprop_models

            payload = directory / f"{axis}.pt"
            chemprop_models.save_model(payload, model.estimator)
        prints = directory / f"{axis}.fingerprints.npy"
        np.save(prints, np.asarray(model.training_fingerprints, dtype=np.float32))
        manifest["axes"][axis] = {
            "estimator": payload.name,
            "fingerprints": prints.name,
            "training_rows": model.training_rows,
            "score": None if model.score is None else asdict(model.score),
        }
    path = directory / _MANIFEST
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return path


def load_regressor(path: Path) -> PanelRegressor:
    """Read a saved regressor, refusing one written by another layout."""
    path = Path(path)
    # Decided by name, not by what exists. Testing ``path.is_dir()`` reads a
    # model directory that has not been created yet as a file path and then
    # hunts for the manifest in its parent, so a missing model is reported
    # against the wrong directory.
    directory = path.parent if path.name == _MANIFEST else path
    manifest_path = directory / _MANIFEST
    if not manifest_path.is_file():
        raise RegressorUnavailableError(f"manifest_not_found:{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegressorUnavailableError(f"manifest_unreadable:{type(error).__name__}") from error
    if manifest.get("version") != _VERSION:
        raise RegressorUnavailableError(f"manifest_version_unsupported:{manifest.get('version')}")
    morgan = manifest.get("morgan") or {}
    if (
        morgan.get("radius") != MORGAN_RADIUS
        or morgan.get("bits") != MORGAN_BITS
        or morgan.get("counts") != MORGAN_COUNTS
    ):
        raise RegressorUnavailableError("manifest_fingerprint_mismatch")
    backend = str(manifest.get("backend") or "forest")
    if backend not in BACKENDS:
        raise RegressorUnavailableError(f"unknown_backend:{backend}")

    axes: dict[str, AxisModel] = {}
    for axis, record in (manifest.get("axes") or {}).items():
        payload = directory / str(record.get("estimator"))
        if not payload.is_file():
            raise RegressorUnavailableError(f"estimator_not_found:{payload.name}")
        if backend == "forest":
            try:
                import joblib
            except ImportError as error:
                raise RegressorUnavailableError(f"joblib_unavailable:{error}") from error
            estimator = joblib.load(payload)
        else:
            from chemprop import models as chemprop_models

            estimator = chemprop_models.load_model(payload)
        prints_path = directory / str(record.get("fingerprints"))
        prints = np.load(prints_path) if prints_path.is_file() else None
        score = record.get("score")
        axes[axis] = AxisModel(
            axis=axis,
            backend=backend,
            estimator=estimator,
            score=AxisScore(**score) if score else None,
            training_fingerprints=prints,
            training_rows=int(record.get("training_rows") or 0),
        )
    if not axes:
        raise RegressorUnavailableError("manifest_has_no_axes")
    return PanelRegressor(
        axes=axes,
        backend=backend,
        method=str(manifest.get("method") or METHOD_FOREST),
        corpus_rows=int(manifest.get("corpus_rows") or 0),
        notes=tuple(manifest.get("notes") or ()),
    )


def default_subset() -> Path:
    """``DATA_PATH/bindingdb/bindingdb_panel_subset.tsv``."""
    data_path = os.environ.get("DATA_PATH", "").strip()
    if not data_path:
        raise RegressorCorpusError("data_path_unset")
    return Path(data_path) / "bindingdb" / "bindingdb_panel_subset.tsv"


def default_model_dir() -> Path:
    """Where the trained regressors live."""
    override = os.environ.get("RFD4_LIGAND_REGRESSOR", "").strip()
    if override:
        return Path(override)
    data_path = os.environ.get("DATA_PATH", "").strip()
    if not data_path:
        raise RegressorUnavailableError("data_path_unset")
    return Path(data_path) / "bindingdb" / "regressor"


def regressor_report(smiles: str, *, directory: Path | None = None) -> dict[str, object]:
    """What ``scripts/screen_ligand.py`` prints for one ligand."""
    regressor = load_regressor(directory or default_model_dir())
    predictions = predict(regressor, smiles)
    record = predictions.as_record()
    assert_prediction_not_measured(record["predictions"])
    return {
        "method": regressor.method,
        "backend": regressor.backend,
        "axes": list(regressor.covered),
        "axes_still_unanswered": list(regressor.missing),
        "scores": {
            axis: {"rmse_log10": round(score.rmse_log10, 4), "spearman": round(score.spearman, 4)}
            for axis, score in regressor.scores.items()
        },
        "extrapolated_axes": list(predictions.extrapolated),
        **record,
    }
