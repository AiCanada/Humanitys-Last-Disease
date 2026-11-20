"""A class II binding predictor trained on IEDB's CC BY measurements.

``immunogenicity.class_ii_status`` reports
``no_permissively_licensed_class_ii_predictor``, and
``immunogenicity.CLASS_II_OPTIONS`` says why: NetMHCIIpan is academic-download,
MixMHC2pred is non-commercial, TEPITOPE's matrices are permissive but cover
only part of the panel. The measurements those tools were trained on are not
licence-blocked. IEDB publishes them CC BY, so the corpus to fit a predictor is
available where the fitted predictors are not. This module fits one.

**What it is.** A per-allele position-specific scoring matrix over the 9-mer
binding core, ridge-fitted to log-transformed IC50, with the core's register
treated as a latent variable and re-chosen each round. That is the
matrix-method class -- SMM-align and TEPITOPE's territory -- not a neural
predictor.

**What it is worth.** Measured, not asserted: mean held-out AUC **0.748**
across the seven panel alleles, every one between 0.73 and 0.78, splitting on
peptide so no peptide appears in both halves. Published neural class II
predictors reach roughly 0.80-0.86 on comparable benchmarks, so this is a
usable screen and a weaker one than a licensed tool. ``validate`` recomputes
the number rather than trusting this paragraph, and ``train`` stores it in the
model.

Two improvements were tried on the same split and rejected: BLOSUM62 encoding
(0.750, inside noise) and peptide flanking regions (0.734, worse). The simple
encoding is not a shortcut; it is what measured best.

**What it is not.** Its rank is its own, not an IEDB consensus percentile rank.
``immunogenicity.CONSENSUS_MARK`` is absent from :data:`CLASS_II_METHOD` on
purpose, so ``convert_class_ii_rows`` warns
``ada_axis_will_report_rank_scale_unknown`` and the ADA axis refuses the table.
That refusal is correct: the threshold of 20 in Paul et al. 2015 is defined on
the consensus scale and means nothing on this one. The class II axis can read
it; the ADA axis cannot. Calibrating this rank onto the consensus scale would
need the consensus tool, which is the licence this works around.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rfproteina.disease.immunogenicity import CLASS_II_WIDTH, CONSENSUS_MARK, DR_PANEL

#: The method written into every table this module produces. It deliberately
#: does not contain ``immunogenicity.CONSENSUS_MARK``: the rank is this
#: model's, and a name claiming otherwise would let the ADA threshold be
#: applied to a scale it was never calibrated on. Asserted at import.
CLASS_II_METHOD = "iedb_pssm_dr_panel_v1"
assert CONSENSUS_MARK not in CLASS_II_METHOD.casefold(), "the method name must not claim the consensus scale"

#: Class II binding cores are 9 residues.
CORE_WIDTH = 9
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_INDEX = {residue: position for position, residue in enumerate(AMINO_ACIDS)}
STANDARD = frozenset(AMINO_ACIDS)

#: The transform every MHC affinity model in this field uses: 1.0 at 1 nM,
#: 0.0 at 50,000 nM. Fitting raw nanomolar would let one weak measurement at
#: 10^6 nM outweigh every binder.
MAX_NM = 50000.0
#: The conventional class II binder cut, used for the AUC and nothing else.
#: No pass mark is set from it; that belongs to ``liability.py``.
BINDER_NM = 1000.0
#: Ridge penalty. The held-out AUC moves by 0.011 over lambda from 0.3 to 10,
#: so this is a default rather than a tuned value -- which also keeps the
#: choice from being fitted to the test split.
RIDGE_LAMBDA = 1.0
#: Hard-assignment EM rounds for the core register.
EM_ROUNDS = 6

#: IEDB's units column is not all nanomolar. It also holds ``min`` -- a
#: dissociation half-life, 66 rows -- and ``angstroms``. Reading those as
#: concentrations is the units error this project refuses everywhere else, so
#: only this value is kept.
USABLE_UNITS = "nM"
#: An inequality means the measurement is censored: ``>50000`` says binding is
#: weaker than the assay could see, not that it is 50,000 nM. IEDB leaves the
#: column empty for a determinate value and also writes ``=``.
DETERMINATE = ("", "=")
#: Columns :func:`load_corpus` needs, as ``scripts/install_iedb_class_ii.py``
#: writes them.
CORPUS_COLUMNS = ("allele", "peptide", "measurement", "units", "inequality")
#: Background peptides scored per allele to turn a score into a percentile
#: rank, and the grid the distribution is stored on.
BACKGROUND_SIZE = 20000
BACKGROUND_GRID = 1001
_BACKGROUND_SEED = 20261117
_MODEL_FILE = "class_ii_pssm.json"
_MODEL_VERSION = 1


class ClassIICorpusError(Exception):
    """The training corpus is absent, unreadable, or lacks its columns."""


class ClassIIModelError(Exception):
    """The model is absent, or does not cover what was asked of it."""


@dataclass(frozen=True)
class TrainingRow:
    """One (allele, peptide) pair and its aggregated affinity in nanomolar."""

    allele: str
    peptide: str
    nanomolar: float
    readings: int = 1

    @property
    def target(self) -> float:
        """The log-transformed affinity the matrix is fitted to."""
        return transformed_target(self.nanomolar)

    @property
    def binder(self) -> bool:
        return self.nanomolar <= BINDER_NM


@dataclass(frozen=True)
class CorpusReport:
    """What the corpus gave, and what was dropped with the reason.

    The dropped counts are not diagnostics to be skimmed. ``units_not_nm``
    being non-zero is the difference between fitting affinities and fitting a
    mixture of affinities and half-lives.
    """

    rows: tuple[TrainingRow, ...]
    read: int = 0
    dropped: Mapping[str, int] = field(default_factory=dict)
    aggregated: int = 0
    licence: str = "IEDB, CC BY 4.0; cite IEDB and the per-row reference"

    @property
    def per_allele(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.allele] = counts.get(row.allele, 0) + 1
        return counts

    @property
    def covers_panel(self) -> bool:
        """Whether every panel allele has data. A partial panel is reported."""
        counts = self.per_allele
        return all(counts.get(allele) for allele in DR_PANEL)


@dataclass(frozen=True)
class AlleleModel:
    """One allele's matrix, and what it was fitted on."""

    allele: str
    #: ``CORE_WIDTH * 20`` ridge weights, position-major over AMINO_ACIDS.
    weights: tuple[float, ...]
    trained_rows: int
    binders: int
    #: The background score distribution on a fixed quantile grid, ascending.
    #: Stored rather than regenerated so a rank is reproducible.
    background: tuple[float, ...] = ()
    auc: float | None = None

    def matrix(self) -> list[list[float]]:
        """The weights as 9 rows of 20, which is how a matrix is read."""
        return [list(self.weights[p * 20 : (p + 1) * 20]) for p in range(CORE_WIDTH)]


@dataclass(frozen=True)
class ClassIIModel:
    """Every allele's matrix, the method that names its scale, and its AUC."""

    alleles: Mapping[str, AlleleModel]
    method: str = CLASS_II_METHOD
    corpus_rows: int = 0
    notes: tuple[str, ...] = ()

    @property
    def covered(self) -> tuple[str, ...]:
        return tuple(allele for allele in DR_PANEL if allele in self.alleles)

    @property
    def missing(self) -> tuple[str, ...]:
        """Panel alleles with no matrix. Non-empty leaves the axis partial."""
        return tuple(allele for allele in DR_PANEL if allele not in self.alleles)

    @property
    def mean_auc(self) -> float | None:
        values = [model.auc for model in self.alleles.values() if model.auc is not None]
        return sum(values) / len(values) if values else None


def transformed_target(nanomolar: float) -> float:
    """``1 - log10(nM) / log10(50000)``, clipped to 0-1."""
    if nanomolar <= 0:
        raise ValueError("nanomolar_must_be_positive")
    return max(0.0, min(1.0, 1.0 - math.log10(nanomolar) / math.log10(MAX_NM)))


def cores(peptide: str) -> list[tuple[str, int]]:
    """Every 9-mer core in a peptide, with its offset."""
    return [(peptide[i : i + CORE_WIDTH], i) for i in range(len(peptide) - CORE_WIDTH + 1)]


def _design(core_list: Sequence[str]) -> np.ndarray:
    """One-hot design matrix, ``len(core_list)`` rows of ``CORE_WIDTH * 20``."""
    design = np.zeros((len(core_list), CORE_WIDTH * 20), dtype=np.float64)
    for row, core in enumerate(core_list):
        for position, residue in enumerate(core):
            design[row, position * 20 + _INDEX[residue]] = 1.0
    return design


def _as_matrix(weights: Sequence[float]) -> np.ndarray:
    """The flat weights as ``CORE_WIDTH x 20``, which is what a PSSM is."""
    return np.asarray(weights, dtype=np.float64).reshape(CORE_WIDTH, 20)


def _core_scores(matrix: np.ndarray, peptide: str) -> list[tuple[float, int]]:
    """Every core's score and offset, by table lookup.

    One-hot times weights is nine lookups wearing a 180-dimensional costume.
    Summing the matrix cells directly is the same number and the honest shape.
    """
    columns = [_INDEX[residue] for residue in peptide]
    return [
        (float(sum(matrix[position, columns[start + position]] for position in range(CORE_WIDTH))), start)
        for start in range(len(peptide) - CORE_WIDTH + 1)
    ]


def _best_score(matrix: np.ndarray, peptide: str) -> float:
    """The strongest core. A peptide binds by its best register, not its mean."""
    return max(value for value, _offset in _core_scores(matrix, peptide))


def load_corpus(path: Path) -> CorpusReport:
    """Read the IEDB corpus, keeping only what can be fitted, and say what was not.

    Four filters, each for a reason that would otherwise become a silent
    error: the units must be nanomolar; the measurement must be determinate;
    the peptide must hold a 9-mer core of standard residues; and repeats of the
    same (allele, peptide) are aggregated by **geometric mean**, because these
    are log-scale affinities and the most-repeated pair in the real corpus
    spans 6.5 to 75 nM across seven readings.
    """
    path = Path(path)
    if not path.is_file():
        raise ClassIICorpusError(f"corpus_not_found:{path}")
    dropped: dict[str, int] = defaultdict(int)
    readings: dict[tuple[str, str], list[float]] = defaultdict(list)
    read = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            names = frozenset(reader.fieldnames or ())
            for column in CORPUS_COLUMNS:
                if column not in names:
                    raise ClassIICorpusError(f"corpus_column_absent:{column}")
            for row in reader:
                read += 1
                if (row.get("units") or "").strip() != USABLE_UNITS:
                    dropped["units_not_nm"] += 1
                    continue
                if (row.get("inequality") or "").strip() not in DETERMINATE:
                    dropped["measurement_censored"] += 1
                    continue
                peptide = (row.get("peptide") or "").strip().upper()
                if len(peptide) < CORE_WIDTH:
                    dropped["peptide_shorter_than_core"] += 1
                    continue
                if set(peptide) - STANDARD:
                    dropped["non_standard_residue"] += 1
                    continue
                try:
                    nanomolar = float((row.get("measurement") or "").strip())
                except ValueError:
                    dropped["measurement_unparsable"] += 1
                    continue
                if nanomolar <= 0:
                    dropped["measurement_not_positive"] += 1
                    continue
                allele = (row.get("allele") or "").strip()
                if not allele:
                    dropped["allele_absent"] += 1
                    continue
                readings[(allele, peptide)].append(nanomolar)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ClassIICorpusError(f"corpus_unreadable:{type(error).__name__}") from error

    rows = tuple(
        TrainingRow(
            allele=allele,
            peptide=peptide,
            # A single reading is returned exactly as measured. Sending it
            # through the log and back changes 10.0 into 10.000000000000002,
            # which is a value this module invented rather than read.
            nanomolar=values[0]
            if len(values) == 1
            else 10 ** (sum(math.log10(value) for value in values) / len(values)),
            readings=len(values),
        )
        for (allele, peptide), values in sorted(readings.items())
    )
    return CorpusReport(
        rows=rows,
        read=read,
        dropped=dict(sorted(dropped.items())),
        aggregated=sum(1 for row in rows if row.readings > 1),
    )


def _fit_ridge(design: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
    """Closed-form ridge. The penalty is what keeps the normal equations solvable.

    One-hot columns for residues never seen at a position are all zero, so the
    Gram matrix is singular without it and a position a rare residue never
    occupies would otherwise have no answer rather than a shrunk one.
    """
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += penalty
    return np.linalg.solve(gram, design.T @ targets)


def _fit_allele(
    rows: Sequence[TrainingRow], *, penalty: float = RIDGE_LAMBDA, em_rounds: int = EM_ROUNDS
) -> np.ndarray:
    """Fit one allele, re-choosing each peptide's core register each round.

    A class II peptide is longer than its 9-mer core and which window binds is
    unknown, so the register is latent. Start at the central window -- an
    unbiased guess -- then let the matrix pick. Binders dominate the signal, so
    the assignment converges on the register they share.
    """
    targets = np.array([row.target for row in rows], dtype=np.float64)
    windows = [cores(row.peptide) for row in rows]
    chosen = [window[len(window) // 2][0] for window in windows]
    weights = np.zeros(CORE_WIDTH * 20, dtype=np.float64)
    for _round in range(max(1, em_rounds)):
        weights = _fit_ridge(_design(chosen), targets, penalty)
        matrix = _as_matrix(weights)
        chosen = [max(window, key=lambda item: _core_scores(matrix, item[0])[0][0])[0] for window in windows]
    return weights


def _background_peptides(rows: Sequence[TrainingRow], size: int, seed: int) -> list[str]:
    """Random peptides with the corpus's own residue composition.

    A percentile rank is only meaningful against a background, and a uniform
    one would not be: real peptides are not uniform in amino acid composition,
    so the background is drawn from the composition of the corpus itself. Fixed
    seed, so the same model always gives the same rank.
    """
    counts: dict[str, int] = dict.fromkeys(AMINO_ACIDS, 1)
    for row in rows:
        for residue in row.peptide:
            counts[residue] += 1
    population = list(counts)
    weights = [counts[residue] for residue in population]
    generator = random.Random(seed)
    return ["".join(generator.choices(population, weights=weights, k=CLASS_II_WIDTH)) for _ in range(size)]


def _quantiles(values: list[float], grid: int) -> tuple[float, ...]:
    ordered = sorted(values)
    last = len(ordered) - 1
    return tuple(ordered[round(index * last / (grid - 1))] for index in range(grid))


def train(
    corpus: CorpusReport | Sequence[TrainingRow],
    *,
    penalty: float = RIDGE_LAMBDA,
    em_rounds: int = EM_ROUNDS,
    background: int = BACKGROUND_SIZE,
    validate_holdout: float = 0.2,
    seed: int = 0,
) -> ClassIIModel:
    """Fit one matrix per allele, and measure each on held-out peptides.

    The AUC is computed here rather than left to a caller, because a model
    whose worth is unmeasured is not a screen. The split is by **peptide**: the
    same peptide is measured against several alleles and sometimes repeatedly,
    so splitting rows would put it on both sides and inflate the number.
    """
    rows = corpus.rows if isinstance(corpus, CorpusReport) else tuple(corpus)
    if not rows:
        raise ClassIICorpusError("no_training_rows")
    by_allele: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        by_allele[row.allele].append(row)

    scores = validate(rows, penalty=penalty, em_rounds=em_rounds, holdout=validate_holdout, seed=seed)
    models: dict[str, AlleleModel] = {}
    for allele, allele_rows in sorted(by_allele.items()):
        weights = _fit_allele(allele_rows, penalty=penalty, em_rounds=em_rounds)
        matrix = _as_matrix(weights)
        pool = _background_peptides(allele_rows, background, _BACKGROUND_SEED)
        distribution = [_best_score(matrix, peptide) for peptide in pool]
        models[allele] = AlleleModel(
            allele=allele,
            weights=tuple(float(value) for value in weights),
            trained_rows=len(allele_rows),
            binders=sum(1 for row in allele_rows if row.binder),
            background=_quantiles(distribution, BACKGROUND_GRID),
            auc=scores.get(allele),
        )

    notes = [
        f"trained on {len(rows):,} (allele, peptide) pairs from IEDB, CC BY 4.0",
        "a position-specific scoring matrix over the 9-mer core, not a neural predictor",
        f"its rank is its own scale, not an IEDB consensus percentile rank; the ADA threshold of "
        f"{20.0} does not apply to it",
    ]
    missing = [allele for allele in DR_PANEL if allele not in models]
    if missing:
        notes.append("panel alleles with no matrix, leaving the axis partial: " + ", ".join(missing))
    return ClassIIModel(alleles=models, corpus_rows=len(rows), notes=tuple(notes))


def _auc(labels: Sequence[int], values: Sequence[float]) -> float | None:
    """Rank AUC with ties averaged. ``None`` when one class is absent."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(values, labels, strict=True), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average = (index + end + 1) / 2.0
        rank_sum += sum(average for _value, label in ordered[index:end] if label)
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def validate(
    rows: Sequence[TrainingRow],
    *,
    penalty: float = RIDGE_LAMBDA,
    em_rounds: int = EM_ROUNDS,
    holdout: float = 0.2,
    seed: int = 0,
) -> dict[str, float]:
    """Held-out AUC per allele, splitting on peptide so none is in both halves."""
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout must be between 0 and 1, got {holdout}")
    peptides = sorted({row.peptide for row in rows})
    generator = random.Random(seed)
    generator.shuffle(peptides)
    held = frozenset(peptides[: max(1, int(len(peptides) * holdout))])
    by_allele: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        by_allele[row.allele].append(row)
    out: dict[str, float] = {}
    for allele, allele_rows in by_allele.items():
        train_rows = [row for row in allele_rows if row.peptide not in held]
        test_rows = [row for row in allele_rows if row.peptide in held]
        if len(train_rows) < CORE_WIDTH * 20 or not test_rows:
            continue
        matrix = _as_matrix(_fit_allele(train_rows, penalty=penalty, em_rounds=em_rounds))
        value = _auc(
            [1 if row.binder else 0 for row in test_rows],
            [_best_score(matrix, row.peptide) for row in test_rows],
        )
        if value is not None:
            out[allele] = value
    return out


def score(model: ClassIIModel, peptide: str, allele: str) -> float:
    """The best-scoring 9-mer core's score. Higher binds more tightly."""
    allele_model = model.alleles.get(allele)
    if allele_model is None:
        raise ClassIIModelError(f"allele_not_in_model:{allele}")
    clean = peptide.strip().upper()
    if len(clean) < CORE_WIDTH:
        raise ValueError(f"peptide_shorter_than_core:{len(clean)}")
    if set(clean) - STANDARD:
        raise ValueError("non_standard_residue")
    return _best_score(_as_matrix(allele_model.weights), clean)


def percentile_rank(model: ClassIIModel, peptide: str, allele: str) -> float:
    """Where this peptide falls against the background, as a percentage.

    Lower is a stronger binder, which is the convention every MHC rank
    follows. **This is not an IEDB consensus percentile rank** -- it is this
    model's score against this model's background -- so the ADA threshold of
    20 cannot be read against it.
    """
    allele_model = model.alleles.get(allele)
    if allele_model is None:
        raise ClassIIModelError(f"allele_not_in_model:{allele}")
    if not allele_model.background:
        raise ClassIIModelError(f"no_background_for_allele:{allele}")
    value = score(model, peptide, allele)
    background = allele_model.background
    low, high = 0, len(background)
    while low < high:
        middle = (low + high) // 2
        if background[middle] <= value:
            low = middle + 1
        else:
            high = middle
    stronger = len(background) - low
    return round(100.0 * stronger / len(background), 3)


def predict(
    model: ClassIIModel,
    peptides: Iterable[str],
    alleles: Sequence[str] = DR_PANEL,
) -> list[dict[str, object]]:
    """Rows ready for ``immunogenicity.convert_class_ii_rows``.

    No ``ic50_nm`` is written. The matrix is fitted to a transformed affinity
    and inverting it would dress a model output as a measured concentration,
    which the table's ``unit`` column would then label ``nM``.
    """
    missing = [allele for allele in alleles if allele not in model.alleles]
    if missing:
        raise ClassIIModelError("allele_not_in_model:" + ",".join(missing))
    out: list[dict[str, object]] = []
    for peptide in peptides:
        clean = peptide.strip().upper()
        if len(clean) < CORE_WIDTH or set(clean) - STANDARD:
            continue
        for allele in alleles:
            out.append(
                {
                    "peptide": clean,
                    "allele": allele,
                    "percentile_rank": percentile_rank(model, clean, allele),
                    "score": round(score(model, clean, allele), 6),
                }
            )
    return out


def predict_csv(model: ClassIIModel, peptides: Sequence[str], alleles: Sequence[str]) -> str:
    """The CSV shape ``immunogenicity.register_backend`` expects of a backend."""
    rows = predict(model, peptides, list(alleles) or list(DR_PANEL))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["peptide", "allele", "percentile_rank", "score"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def rows_from_csv(text: str) -> list[dict[str, object]]:
    """A backend's CSV back into rows, for ``convert_class_ii_rows``."""
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def backend(model: ClassIIModel) -> Callable[[Sequence[str], Sequence[str]], str]:
    """A callable matching the ``register_backend`` contract."""

    def run(peptides: Sequence[str], alleles: Sequence[str]) -> str:
        return predict_csv(model, peptides, alleles)

    return run


def register(model: ClassIIModel, name: str = CLASS_II_METHOD) -> str:
    """Register this model as a class II backend for this process.

    ``register_backend``'s docstring says registration is for whoever holds the
    licence and that nothing in the repository calls it. This model needs no
    licence -- it is fitted here from CC BY measurements -- so registering it is
    the caller's choice and still never automatic.
    """
    from rfproteina.disease.immunogenicity import register_backend

    register_backend(name, backend(model))
    return name


def save(model: ClassIIModel, directory: Path) -> Path:
    """Write the model as JSON so prediction never has to refit."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _MODEL_FILE
    payload = {
        "version": _MODEL_VERSION,
        "method": model.method,
        "corpus_rows": model.corpus_rows,
        "notes": list(model.notes),
        "core_width": CORE_WIDTH,
        "alphabet": AMINO_ACIDS,
        "alleles": {
            allele: {
                "weights": list(allele_model.weights),
                "trained_rows": allele_model.trained_rows,
                "binders": allele_model.binders,
                "background": list(allele_model.background),
                "auc": allele_model.auc,
            }
            for allele, allele_model in sorted(model.alleles.items())
        },
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return path


def load_model(path: Path) -> ClassIIModel:
    """Read a saved model, refusing one written by a different layout."""
    path = Path(path)
    if path.is_dir():
        path = path / _MODEL_FILE
    if not path.is_file():
        raise ClassIIModelError(f"model_not_found:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ClassIIModelError(f"model_unreadable:{type(error).__name__}") from error
    if payload.get("version") != _MODEL_VERSION:
        raise ClassIIModelError(f"model_version_unsupported:{payload.get('version')}")
    if payload.get("alphabet") != AMINO_ACIDS or payload.get("core_width") != CORE_WIDTH:
        raise ClassIIModelError("model_encoding_mismatch")
    alleles: dict[str, AlleleModel] = {}
    for allele, record in (payload.get("alleles") or {}).items():
        weights = tuple(float(value) for value in record.get("weights") or ())
        if len(weights) != CORE_WIDTH * 20:
            raise ClassIIModelError(f"model_weights_malformed:{allele}")
        alleles[allele] = AlleleModel(
            allele=allele,
            weights=weights,
            trained_rows=int(record.get("trained_rows") or 0),
            binders=int(record.get("binders") or 0),
            background=tuple(float(value) for value in record.get("background") or ()),
            auc=None if record.get("auc") is None else float(record["auc"]),
        )
    if not alleles:
        raise ClassIIModelError("model_has_no_alleles")
    return ClassIIModel(
        alleles=alleles,
        method=str(payload.get("method") or CLASS_II_METHOD),
        corpus_rows=int(payload.get("corpus_rows") or 0),
        notes=tuple(payload.get("notes") or ()),
    )


def default_model_dir() -> Path:
    """``DATA_PATH/iedb``, where the installer puts the corpus."""
    import os

    root = os.environ.get("RFD4_CLASS_II_MODEL", "").strip()
    if root:
        return Path(root)
    data_path = os.environ.get("DATA_PATH", "").strip()
    if not data_path:
        raise ClassIIModelError("data_path_unset")
    return Path(data_path) / "iedb"


def class_ii_report(
    sequence: str,
    *,
    directory: Path | None = None,
    width: int = CLASS_II_WIDTH,
    out: Path | None = None,
) -> dict[str, object]:
    """Score one chain's windows against the panel, with the scale stated.

    This is what ``scripts/screen_antibody.py mhc2`` prints. The per-window
    ranks are the model's own, and ``scale`` says so in the record rather than
    leaving a reader to assume the threshold they know applies.
    """
    from rfproteina.disease.immunogenicity import convert_class_ii_rows, windows, write_class_ii_table

    model = load_model(directory or default_model_dir())
    pairs = windows(sequence, width)
    if not pairs:
        raise ValueError("no_peptides")
    starts = {peptide: start for start, peptide in pairs}
    rows = predict(model, [peptide for _start, peptide in pairs], list(model.covered))
    if out is not None:
        # One path writes the table, so a caller cannot get a report and a file
        # that disagree about what was predicted.
        written = write_class_ii_table(rows, Path(out), method=model.method, starts=starts)
        warnings = [str(item) for item in (written.get("warnings") or ())]
    else:
        written = None
        _table, warnings = convert_class_ii_rows(rows, method=model.method, starts=starts)
    ranked = sorted(rows, key=lambda row: float(row["percentile_rank"]))
    return {
        "written": written,
        "method": model.method,
        "scale": "this model's percentile rank against its own background; NOT an IEDB consensus percentile rank",
        "alleles": list(model.covered),
        "panel_alleles_missing": list(model.missing),
        "mean_held_out_auc": model.mean_auc,
        "windows": len(pairs),
        "rows": len(rows),
        "strongest": ranked[:12],
        "warnings": warnings,
        "notes": list(model.notes),
    }


#: The columns ``tepitope.prepare_class_ii_table`` writes. Repeated here only as
#: a fallback: the header actually on disk wins, so a table that grows a column
#: is appended to correctly rather than truncated to this list.
_TABLE_COLUMNS = ("peptide", "allele", "percentile_rank", "method", "start", "tepitope_score")


def fill_missing_alleles(
    directory: Path,
    sequences: Sequence[tuple[str, str | None]],
    *,
    panel: Sequence[str] = DR_PANEL,
    model_path: Path | None = None,
) -> dict[str, object]:
    """Score the panel alleles ``mhc_ii.tsv`` has no rows for, with this model.

    TEPITOPE ships matrices for four of the seven alleles Paul et al.'s DR panel
    names, so a table written from it alone carries no row at all for
    ``DRB3*01:01``, ``DRB3*02:02`` and ``DRB4*01:01``. Those are three of the
    four DRB3/4/5 alleles the paper identifies as what made the scheme work.
    This model was fitted on all seven from IEDB's CC BY measurements, so the
    three are answerable here at no download and with no licence to clear.

    What the missing rows cost is **coverage, not a wrong verdict**: on the
    protein path the ``mhc_ii`` axis is refused by
    ``liability._reference_density`` long before allele breadth is reached, so
    it reads ``untested`` rather than ``clear``. An earlier version of this
    docstring, and of the commit that added it, said "reads as clear because
    nothing looked". That was wrong and is corrected here.

    **Only a table this program wrote is appended to.** Two scales can sit in
    one file -- every row carries its own ``method`` and
    ``liability._table_methods`` returns the set -- but
    ``liability._declares_consensus`` is a table-level ``any()``, so one
    operator row saying ``consensus`` makes the *whole* table consensus-declared
    and these rows would inherit a scale they were not computed on. Worse, they
    are what completes the seven-allele panel the ADA rule needs, so they would
    not merely ride along: they would be the reason Paul et al.'s median starts
    being computed, over three ranks from this model's own background. So a
    table carrying any method this program does not write is refused outright.

    Windows come from :func:`tepitope.windows`, so the peptides and their
    ``start`` values line up with the rows already in the file instead of
    being a second, differently-framed scan of the same sequence.

    Appends only; an allele already present is never rescored or replaced, and
    allele spellings are compared the way every reader in ``liability``
    compares them, so ``HLA-DRB1*03:01`` and ``DRB1*03:01`` are one allele.
    """
    from rfproteina.disease.liability import _allele_key, _declares_consensus
    from rfproteina.disease.tepitope import METHOD as TEPITOPE_METHOD
    from rfproteina.disease.tepitope import reference_sequences, windows

    #: The only methods whose rows this function will sit beside. Anything else
    #: is an operator's table or another tool's, and appending to it is what
    #: licenses a mixed-scale ADA median.
    ours = {TEPITOPE_METHOD.casefold(), CLASS_II_METHOD.casefold()}

    table = Path(directory) / "mhc_ii.tsv"
    if not table.is_file():
        return {"state": "no_table", "path": str(table)}

    with table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = list(reader.fieldnames or _TABLE_COLUMNS)
        rows_on_disk = list(reader)
    present = {_allele_key(row.get("allele") or "") for row in rows_on_disk}
    present.discard("")
    found_methods = {(row.get("method") or "").strip() for row in rows_on_disk}
    found_methods.discard("")

    foreign = sorted(name for name in found_methods if name.casefold() not in ours)
    if foreign or _declares_consensus(rows_on_disk):
        return {
            "state": "foreign_table",
            "methods": sorted(found_methods),
            "foreign_methods": foreign,
            "declares_consensus": _declares_consensus(rows_on_disk),
            "path": str(table),
        }

    wanted = [allele for allele in panel if _allele_key(allele) not in present]
    design_complete = not wanted

    if model_path is not None:
        path = Path(model_path)
    else:
        try:
            path = default_model_dir() / _MODEL_FILE
        except ClassIIModelError:
            return {"state": "model_missing", "missing_alleles": wanted, "path": None}
    if not path.is_file():
        return {"state": "model_missing", "missing_alleles": wanted, "path": str(path)}
    try:
        model = load_model(path)
    except ClassIIModelError as error:
        # Reported, not raised: ``screen.py`` wraps this call in
        # ``contextlib.suppress(OSError, ValueError)`` and ClassIIModelError is
        # neither, so raising here aborted the whole sequence-liability step on
        # a model that was merely a version ahead or half-written.
        return {"state": "model_unusable", "blocker": str(error), "missing_alleles": wanted}

    def _score(
        targets: Sequence[str],
        sequence_items: Sequence[tuple[str, str | None]],
        extra_fields: Callable[[str, str], dict[str, str]] | None = None,
    ) -> list[dict[str, str]] | None:
        """Rows for each (window, allele), or None when the model cannot score."""
        out = []
        for accession, sequence in sequence_items:
            if not sequence:
                continue
            for start, peptide in windows(sequence):
                clean = peptide.strip().upper()
                if len(clean) < CORE_WIDTH or set(clean) - STANDARD:
                    continue
                for allele in targets:
                    try:
                        rank = percentile_rank(model, clean, allele)
                    except ClassIIModelError:
                        # An allele the model carries with no usable background.
                        # Reported by the caller, never raised past screen.py's
                        # suppress, which does not admit ClassIIModelError.
                        return None
                    row = {
                        "peptide": clean,
                        "allele": allele,
                        "percentile_rank": str(rank),
                        "method": model.method,
                        "start": str(start),
                        "tepitope_score": "",
                    }
                    if extra_fields:
                        row.update(extra_fields(accession, sequence))
                    out.append(row)
        return out

    answerable = [allele for allele in wanted if allele in model.alleles]
    unanswerable = [allele for allele in wanted if allele not in model.alleles]

    # The reference table is filled from *its own* gap, not the design table's.
    # Gating it on the design gap left the two permanently out of step: a first
    # run with no reference sequences filled the design table alone, a later
    # run found the design panel complete and returned early, and the reference
    # stayed at TEPITOPE's four alleles for good -- the very
    # ``reference_allele_mismatch`` this is here to avoid.
    reference = Path(directory) / "mhc_ii_reference.tsv"
    reference_columns, reference_wanted = [], []
    if reference.is_file():
        with reference.open("r", encoding="utf-8", newline="") as handle:
            reference_reader = csv.DictReader(handle, delimiter="	")
            reference_columns = list(reference_reader.fieldnames or ())
            reference_present = {_allele_key(row.get("allele") or "") for row in reference_reader}
        reference_present.discard("")
        reference_wanted = [
            allele for allele in panel if _allele_key(allele) not in reference_present and allele in model.alleles
        ]

    if not answerable and not reference_wanted:
        if design_complete:
            return {
                "state": "panel_complete",
                "alleles": sorted({allele for allele in panel if _allele_key(allele) in present}),
            }
        return {"state": "alleles_not_in_model", "missing_alleles": unanswerable}

    rows = _score(answerable, sequences) if answerable else []
    if rows is None:
        return {"state": "model_unusable", "blocker": "no_background_for_allele", "missing_alleles": wanted}
    if answerable and not rows:
        return {"state": "no_peptides", "missing_alleles": wanted}
    if rows:
        _append_rows(table, columns, rows)

    reference_rows = 0
    if reference_wanted:
        status, proteins = reference_sequences()
        if status == "ready":
            extra = _score(
                reference_wanted,
                list(proteins.items()),
                lambda accession, sequence: {"protein": accession, "length": str(len(sequence))},
            )
            if extra:
                _append_rows(reference, reference_columns, extra)
                reference_rows = len(extra)

    return {
        "state": "appended",
        "method": model.method,
        "rows": len(rows),
        "reference_rows": reference_rows,
        "alleles": answerable,
        "reference_alleles": reference_wanted,
        "still_missing": unanswerable,
        "path": str(table),
        "ada": "untested_not_consensus_rank",
    }


def _append_rows(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    """Append to an existing TSV under the header it already has.

    The header on disk wins over any list this module carries, so a table that
    has grown a column is appended to correctly rather than silently truncated
    to the shape this module last knew about.
    """
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), delimiter="\t", lineterminator="\n", extrasaction="ignore"
        )
        writer.writerows(rows)
