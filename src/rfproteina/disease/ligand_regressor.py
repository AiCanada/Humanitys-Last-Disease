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

**Historical forest results.** These values came from three scaffold splits
of an earlier corpus that pooled assay types and target preparations.
Fresh fits on the full release subset use single-chain IC50 values with
the matching accession. Each fresh fit has its own validation statistics.

===========  ======  ==========  =====  =========  ========
axis          train  RMSE log10   fold  Spearman     test n
===========  ======  ==========  =====  =========  ========
``nav1_5``    1,710       0.592   3.9x      0.777       639
``bcrp``        860       0.631   4.3x      0.715       321
``bsep``        352       0.535   3.4x      0.587       132
===========  ======  ==========  =====  =========  ========

The reported errors describe the original partitions. :func:`train` recomputes
held-out errors for a new corpus. The default backend uses Morgan fingerprints
and ExtraTrees. The optional Chemprop backend selects its checkpoint by
validation loss and uses only its training partition for similarity estimates.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
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
_VERSION = 2


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
    measurement_scope: str = "assay and target preparation unreported"

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
    r2: float | None
    spearman: float | None
    test_rows: int
    train_rows: int
    replicates: int

    @property
    def fold_error(self) -> float:
        """The RMSE expressed as a factor, which is how a chemist reads it."""
        return 10.0**self.rmse_log10

    def reported_metrics(self) -> dict[str, float | int | None]:
        return {
            "train_rows": self.train_rows, "test_rows": self.test_rows, "replicates": self.replicates,
            "rmse_log10": round(self.rmse_log10, 4), "fold_error": round(self.fold_error, 2),
            "spearman": round(self.spearman, 4) if self.spearman is not None and math.isfinite(self.spearman) else None,
            "r2": round(self.r2, 4) if self.r2 is not None and math.isfinite(self.r2) else None,
        }


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
    training_rows: int | None = 0


@dataclass(frozen=True)
class PanelRegressor:
    """Every axis's model, with the method that names the backend."""

    axes: Mapping[str, AxisModel]
    backend: str = "forest"
    method: str = METHOD_FOREST
    corpus_rows: int = 0
    notes: tuple[str, ...] = ()
    measurement_scope: str = "assay and target preparation unreported"

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
        return tuple(axis for axis, values in self.predictions.items() if values.get("within_applicability") is False)


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

    The released schema identifies IC50, accession and chain count. Older
    four-column extracts retain their unreported measurement scope. Within
    each axis, determinate positive readings for the same stereochemical
    identity are aggregated by geometric mean.
    """
    path = Path(path)
    if not path.is_file():
        raise RegressorCorpusError(f"subset_not_found:{path}")
    wanted = {str(axis) for axis in axes}
    dropped: dict[str, int] = defaultdict(int)
    readings: dict[tuple[str, str], list[float]] = defaultdict(list)
    canonical: dict[str, str] = {}
    from rdkit import Chem

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
                if "assay" in names and (row.get("assay") or "").strip().upper() != "IC50":
                    dropped["assay_other_than_ic50"] += 1
                    continue
                if "chains" in names and (row.get("chains") or "").strip() != "1":
                    dropped["target_not_single_chain"] += 1
                    continue
                if "accession" in names and (row.get("accession") or "").strip() != AXIS_ACCESSIONS.get(axis):
                    dropped["target_accession_mismatch"] += 1
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
                if smiles not in canonical:
                    mol = _parse(smiles)
                    canonical[smiles] = Chem.MolToSmiles(mol) if mol is not None else ""
                if not canonical[smiles]:
                    dropped["smiles_unparsable"] += 1
                    continue
                readings[(axis, canonical[smiles])].append(value)
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
        measurement_scope=", ".join((
            "IC50" if "assay" in names else "assay unreported",
            "single chain" if "chains" in names else "chain count unreported",
            "matched accession" if "accession" in names else "accession unreported",
        )),
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
    text = (smiles or "").strip()
    if not text or any(char in text for char in "\r\n\0"):
        return None
    params = Chem.SmilesParserParams()
    params.parseName = False
    mol = Chem.MolFromSmiles(text, params)
    return mol if mol is not None and mol.GetNumAtoms() else None


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


def _spearman(truth: Sequence[float], predicted: Sequence[float]) -> float | None:
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
        return None
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
        r2=1.0 - residual / total if total else None,
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
    with tempfile.TemporaryDirectory(prefix="ligand-chemprop-") as work:
        checkpoint = ModelCheckpoint(dirpath=work, monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(
            max_epochs=max_epochs, accelerator="auto", devices=1, default_root_dir=work,
            enable_progress_bar=False, enable_model_summary=False, logger=False,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=patience), checkpoint],
        )
        trainer.fit(
            model,
            chemprop_data.build_dataloader(train_set, batch_size=50, num_workers=0, seed=seed),
            chemprop_data.build_dataloader(
                validation_set, batch_size=50, num_workers=0, shuffle=False, drop_last=False
            ),
        )
        if (
            not checkpoint.best_model_path or checkpoint.best_model_score is None
            or not math.isfinite(float(checkpoint.best_model_score))
        ):
            raise RegressorModelError("chemprop_no_finite_validation_checkpoint")
        return models.MPNN.load_from_checkpoint(checkpoint.best_model_path, map_location="cpu")


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
        chemprop_data.MoleculeDataset(points), batch_size=50, num_workers=0, shuffle=False, drop_last=False
    )
    trainer = pl.Trainer(
        accelerator="auto", devices=1, enable_progress_bar=False, logger=False, enable_model_summary=False,
        enable_checkpointing=False,
    )
    out = trainer.predict(model, loader)
    if not out:
        raise RegressorModelError("chemprop_returned_no_predictions")
    values = np.concatenate([tensor.detach().cpu().numpy().reshape(-1) for tensor in out])
    if len(values) != len(smiles) or not np.isfinite(values).all():
        raise RegressorModelError("chemprop_incomplete_or_nonfinite_predictions")
    return values


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
    if replicates < 1:
        raise ValueError("replicates_must_be_positive")
    return _score_splits(rows, axis, backend, seed, _split_indices(rows, replicates, seed))


def _score_splits(
    rows: Sequence[TrainingRow], axis: str, backend: str, seed: int, splits: tuple[Any, Any, Any],
    *, first_chemprop_model: Any = None,
) -> AxisScore | None:
    truth: list[float] = []
    predicted: list[float] = []
    trained = 0
    scored_replicates = 0
    for replicate, (train, validation, test) in enumerate(zip(*splits, strict=True)):
        if len(train) < 50 or len(test) == 0:
            continue
        if backend == "forest":
            estimator, _matrix = _fit_forest([rows[i] for i in train], seed=seed + replicate)
            matrix, kept = fingerprints([rows[i].smiles for i in test])
            if not kept:
                continue
            predicted.extend(estimator.predict(matrix).tolist())
            truth.extend(rows[test[i]].log10_nanomolar for i in kept)
        else:
            model = first_chemprop_model if replicate == 0 else None
            if model is None:
                model = _fit_chemprop(rows, train, validation, seed=seed + replicate)
            texts = [rows[i].smiles for i in test]
            predicted.extend(_predict_chemprop(model, texts).tolist())
            truth.extend(rows[i].log10_nanomolar for i in test)
        trained = len(train)
        scored_replicates += 1
    if len(truth) < 10:
        return None
    return _score(axis, backend, truth, predicted, trained, scored_replicates)


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
    if replicates < 1:
        raise ValueError("replicates_must_be_positive")
    rows = corpus.rows if isinstance(corpus, CorpusReport) else tuple(corpus)
    if not rows:
        raise RegressorCorpusError("no_training_rows")
    by_axis: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        if row.axis not in UNCOVERED_AXES:
            raise RegressorModelError(f"unknown_panel_axis:{row.axis}")
        by_axis[row.axis].append(row)

    models: dict[str, AxisModel] = {}
    for axis, axis_rows in sorted(by_axis.items()):
        score = None
        if backend == "forest":
            if evaluate_first:
                score = evaluate(axis_rows, axis=axis, backend=backend, replicates=replicates, seed=seed)
            estimator, matrix = _fit_forest(axis_rows, seed=seed)
            prints = matrix
        else:
            splits = _split_indices(axis_rows, replicates if evaluate_first else 1, seed)
            fit, validation = splits[0][0], splits[1][0]
            estimator = _fit_chemprop(axis_rows, fit, validation, seed=seed)
            if evaluate_first:
                score = _score_splits(axis_rows, axis, backend, seed, splits, first_chemprop_model=estimator)
            prints, _kept = fingerprints([axis_rows[i].smiles for i in fit])
        models[axis] = AxisModel(
            axis=axis,
            backend=backend,
            estimator=estimator,
            score=score,
            training_fingerprints=prints,
            training_rows=len(prints),
        )

    method = METHOD_FOREST if backend == "forest" else METHOD_CHEMPROP
    notes = [
        f"corpus contained {len(rows):,} (axis, compound) pairs before splitting",
        "The target is log10(nM). Exponentiating the held-out RMSE expresses that error on a fold scale.",
        f"every value is reported as {PREDICTED_KEY} and never under an *_nm key: it is a model output, "
        "not a measurement",
        "censored measurements were excluded: a bound is not a value",
    ]
    missing = [axis for axis in UNCOVERED_AXES if axis not in models]
    if missing:
        notes.append("axes with no model, still unanswered: " + ", ".join(missing))
    scope = corpus.measurement_scope if isinstance(corpus, CorpusReport) else "caller-supplied training rows"
    return PanelRegressor(axes=models, backend=backend, method=method, corpus_rows=len(rows),
                          notes=tuple(notes), measurement_scope=scope)


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
        try:
            concentration = 10.0**value
        except OverflowError as error:
            raise RegressorModelError("nonfinite_predicted_concentration") from error
        if not math.isfinite(concentration) or concentration <= 0.0:
            raise RegressorModelError("nonfinite_predicted_concentration")
        predictions[axis] = {
            PREDICTED_KEY: concentration,
            "predicted_log10_nanomolar": round(value, 4),
            "training_measurements": regressor.measurement_scope,
            "accession": AXIS_ACCESSIONS.get(axis, ""),
            "model_rmse_log10": None if score is None else round(score.rmse_log10, 4),
            "model_fold_error": None if score is None else round(score.fold_error, 2),
            "model_test_rows": None if score is None else score.test_rows,
            "model_r2": None if score is None else score.reported_metrics()["r2"],
            "nearest_training_similarity": None if math.isnan(nearest) else round(nearest, 4),
            "within_applicability": None if math.isnan(nearest) else nearest >= APPLICABILITY_SIMILARITY,
            "training_rows": model.training_rows,
        }

    notes = [
        "a model output, not a measurement; nothing here may be written into an *_nm column",
        "log10-scale model: read the fold error, not the digits",
        "BindingDB terms apply to the training data; cite BindingDB",
    ]
    outside = [axis for axis, values in predictions.items() if values["within_applicability"] is False]
    unknown_reference = [axis for axis, values in predictions.items() if values["within_applicability"] is None]
    if unknown_reference:
        notes.append("Training-set similarity unavailable for " + ", ".join(unknown_reference))
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


def _write_payloads(regressor: PanelRegressor, directory: Path) -> dict[str, Any]:
    manifest: dict[str, object] = {
        "version": _VERSION,
        "backend": regressor.backend,
        "method": regressor.method,
        "corpus_rows": regressor.corpus_rows,
        "measurement_scope": regressor.measurement_scope,
        "notes": list(regressor.notes),
        "morgan": {"radius": MORGAN_RADIUS, "bits": MORGAN_BITS, "counts": MORGAN_COUNTS},
        "axes": {},
    }
    for axis, model in sorted(regressor.axes.items()):
        if axis not in UNCOVERED_AXES or model.axis != axis or model.backend != regressor.backend:
            raise RegressorModelError("estimator_axis_or_backend_mismatch")
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
        if model.training_fingerprints is not None:
            matrix = np.asarray(model.training_fingerprints, dtype=np.float32)
            if matrix.shape != (model.training_rows, MORGAN_BITS) or not np.isfinite(matrix).all():
                raise RegressorModelError("training_fingerprints_mismatch")
            np.save(prints, matrix)
        manifest["axes"][axis] = {
            "estimator": payload.name,
            "fingerprints": prints.name if model.training_fingerprints is not None else None,
            "training_rows": model.training_rows,
            "score": None if model.score is None else asdict(model.score),
        }
    return manifest


def check_output_backend(directory: Path, backend: str) -> None:
    path = directory / _MANIFEST
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("backend", "forest") != backend:
            raise RegressorModelError("output_directory_contains_a_different_backend")


def save(regressor: PanelRegressor, directory: Path) -> Path:
    """Publish a complete local model. Existing readers can finish using the previous files."""
    directory = Path(directory)
    method = {"forest": METHOD_FOREST, "chemprop": METHOD_CHEMPROP}.get(regressor.backend)
    if method is None or regressor.method != method or not regressor.axes:
        raise RegressorModelError("invalid_regressor_manifest")
    directory.parent.mkdir(parents=True, exist_ok=True)
    check_output_backend(directory, regressor.backend)
    with tempfile.TemporaryDirectory(prefix=".ligand-fit-", dir=directory.parent) as name:
        staging = Path(name)
        payloads = staging / "payloads"
        payloads.mkdir()
        manifest = _write_payloads(regressor, payloads)
        staged_manifest = staging / _MANIFEST
        staged_manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        if not directory.exists():
            # A fresh model keeps the original flat file layout.
            staged_manifest.replace(payloads / _MANIFEST)
            try:
                payloads.rename(directory)
            except OSError:
                if not directory.is_dir():
                    raise
                (payloads / _MANIFEST).replace(staged_manifest)
            else:
                return directory / _MANIFEST
        check_output_backend(directory, regressor.backend)
        revision = staging.name.lstrip(".")
        for record in manifest["axes"].values():
            for key in ("estimator", "fingerprints"):
                if record[key]:
                    record[key] = f"{revision}/{record[key]}"
        staged_manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        payloads.rename(directory / revision)
        staged_manifest.replace(directory / _MANIFEST)
    return directory / _MANIFEST


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
    version = manifest.get("version")
    if type(version) is not int or version not in (1, _VERSION):
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
    expected_method = METHOD_CHEMPROP if backend == "chemprop" else METHOD_FOREST
    if manifest.get("method", expected_method) != expected_method:
        raise RegressorUnavailableError("manifest_method_backend_mismatch")

    axes: dict[str, AxisModel] = {}
    for axis, record in (manifest.get("axes") or {}).items():
        if axis not in UNCOVERED_AXES:
            raise RegressorUnavailableError(f"unknown_panel_axis:{axis}")
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
        # A v1 graph checkpoint indexed its entire corpus, including held-out compounds.
        reference_verified = backend == "forest" or version >= 2
        prints = np.load(prints_path) if reference_verified and prints_path.is_file() else None
        training_rows = record.get("training_rows") if reference_verified else None
        if prints is not None and (prints.ndim != 2 or prints.shape != (training_rows, MORGAN_BITS)):
            raise RegressorUnavailableError("manifest_training_fingerprints_mismatch")
        score = record.get("score")
        if score and (score.get("axis") != axis or score.get("backend") != backend):
            raise RegressorUnavailableError("manifest_score_axis_or_backend_mismatch")
        if score:
            for metric in ("r2", "spearman"):
                if score.get(metric) is not None and not math.isfinite(score[metric]):
                    score[metric] = None
        axes[axis] = AxisModel(
            axis=axis,
            backend=backend,
            estimator=estimator,
            score=AxisScore(**score) if score else None,
            training_fingerprints=prints,
            training_rows=training_rows,
        )
    if not axes:
        raise RegressorUnavailableError("manifest_has_no_axes")
    return PanelRegressor(
        axes=axes,
        backend=backend,
        method=expected_method,
        measurement_scope=str(manifest.get("measurement_scope") or "assay and target preparation unreported"),
        corpus_rows=int(manifest.get("corpus_rows") or 0),
        notes=tuple(manifest.get("notes") or ()),
    )


def default_subset() -> Path:
    """``DATA_PATH/bindingdb/bindingdb_panel_subset.tsv``."""
    data_path = os.environ.get("DATA_PATH", "").strip()
    if not data_path:
        raise RegressorCorpusError("data_path_unset")
    return Path(data_path) / "bindingdb" / "bindingdb_panel_subset.tsv"


def default_model_dir(backend: str = "forest") -> Path:
    """Where the trained regressors live."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    override_key = "RFD4_CHEMPROP_REGRESSOR" if backend == "chemprop" else "RFD4_LIGAND_REGRESSOR"
    override = os.environ.get(override_key, "").strip()
    if override:
        return Path(override)
    data_path = os.environ.get("DATA_PATH", "").strip()
    if not data_path:
        raise RegressorUnavailableError("data_path_unset")
    return Path(data_path) / "bindingdb" / ("regressor_chemprop" if backend == "chemprop" else "regressor")


def regressor_report(smiles: str, *, directory: Path | None = None) -> dict[str, object]:
    """What ``scripts/screen_ligand.py`` prints for one ligand."""
    if not (smiles or "").strip():
        raise ValueError("empty_smiles")
    if _parse(smiles) is None:
        raise ValueError("smiles_unparsable")
    regressor = load_regressor(directory or default_model_dir())
    predictions = predict(regressor, smiles)
    record = predictions.as_record()
    assert_prediction_not_measured(record["predictions"])
    return {
        "method": regressor.method,
        "backend": regressor.backend,
        "axes": list(regressor.covered),
        "axes_still_unanswered": list(regressor.missing),
        "scores": {axis: score.reported_metrics() for axis, score in regressor.scores.items()},
        "extrapolated_axes": list(predictions.extrapolated),
        **record,
    }
