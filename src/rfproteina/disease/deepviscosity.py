"""Run DeepViscosity: a sequence-only high-concentration viscosity class.

This is the screen's first viscosity answer. Nothing else here predicted one:
DeepSCM, the other sequence surrogate for the same endpoint, is CC BY-NC and so
is excluded on licence grounds, and the motif and oracle readers in
:mod:`rfproteina.disease.developability` say nothing about how a formulation
flows. DeepViscosity is MIT for both code and weights.

What it answers, and in what terms. An ensemble of 102 small feed-forward
networks reads DeepSP's 30 spatial descriptors and returns the probability that
an antibody's viscosity exceeds 20 cP **at 150 mg/mL in 20 mM histidine-HCl,
pH 6.0**. Those conditions are not decoration: they are the only conditions the
measurements behind the model were taken at, and a class carried away from them
is an extrapolation. The authors report 87.5 per cent accuracy on a 16-antibody
independent test set and 89.5 per cent on a 38-antibody one, against training
and leave-one-group-out validation accuracies of 80.7 and 88.2 per cent.

**It does not predict a viscosity.** There is no centipoise number in this
model's output and none is invented here; it emits a class and a probability.
:func:`assert_no_predicted_centipoise` enforces that rather than trusting this
paragraph, because a predicted probability written into a ``_cp`` field would
read downstream as a measured viscosity.

Two things are done differently from upstream, both verified rather than
assumed:

* **No TensorFlow.** DeepViscosity ships Keras 2.11 models, and this
  environment has no TensorFlow. The forward pass -- four tanh dense layers of
  128, 64, 32 and 16 units and a sigmoid output -- is evaluated in torch, the
  same choice and for the same reason as :mod:`rfproteina.disease.deepsp`. The
  port reproduces the probabilities the upstream predictor notebook published
  for all 16 antibodies of its test set; ``tests/test_disease_deepviscosity.py``
  keeps that comparison.
* **Nothing is unpickled.** Upstream ships its input scaler as an sklearn
  ``StandardScaler`` inside a joblib pickle, and loading a downloaded pickle
  runs whatever that file says. ``scripts/install_deepviscosity.py`` verifies a
  pinned SHA-256, reads it once through an unpickler whose allowlist can reach
  no callable of an attacker's choosing, and writes the 30 means and 30 scales
  as plain JSON. This module reads only that JSON, so neither joblib nor
  scikit-learn is needed to predict, and no pickle is ever opened here.

The descriptors come from :func:`rfproteina.disease.deepsp.run_deepsp`, so both
DeepSP's models and DeepViscosity's must be installed; which one is missing is
reported rather than guessed around.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rfproteina.disease.deepsp import (
    BLOCKER_TEXT as DEEPSP_BLOCKER_TEXT,
)
from rfproteina.disease.deepsp import (
    DEEPSP_METHOD,
    DeepSPResult,
    DeepSPUnavailableError,
    deepsp_available,
    keras_weights,
    run_deepsp,
)
from rfproteina.disease.deepsp import default_models_dir as deepsp_models_dir
from rfproteina.disease.developability import DEEPSP_COLUMNS, NumberingUnavailableError

#: Bumped when the architecture, the scaling or the aggregation changes, so a
#: stored record says which rule produced it.
VISCOSITY_METHOD = "deepviscosity_ann_ensemble_torch_v1"
VISCOSITY_SOURCE = "DeepViscosity (MIT), https://github.com/Lailabcode/DeepViscosity"

#: One sentence per blocker this module can put on a record.
#:
#: Three of the four are DeepSP's own and are inherited rather than restated:
#: this module looks for the SAP weights too, and reports the same names when
#: they are missing. A ``data_path_unset`` here can come from either weights
#: directory, which is why that sentence names neither.
#:
#: As in :py:data:`rfproteina.disease.deepsp.BLOCKER_TEXT`, nothing renders
#: these today -- ``viscosity_blocker`` has no reader -- so this is the words
#: for when a surface does show them.
BLOCKER_TEXT = {
    **DEEPSP_BLOCKER_TEXT,
    "deepviscosity_unavailable": (
        "The DeepViscosity weights, or the libraries that run them, are not on this "
        "machine. scripts/install_deepviscosity.py fetches them."
    ),
}
VISCOSITY_PAPER = (
    "Kalejaye et al., 'Accelerating high-concentration monoclonal antibody development with "
    "large-scale viscosity data and ensemble deep learning', mAbs 17(1) (2025), "
    "doi:10.1080/19420862.2025.2483944"
)

#: Where ``scripts/install_deepviscosity.py`` writes, under ``DATA_PATH``.
VISCOSITY_SUBDIR = Path("deepviscosity") / "models"

#: One model per leave-one-group-out split of the 229 training antibodies,
#: which were clustered into 102 groups by Levenshtein distance. The count is
#: part of the published model: averaging a subset is a different predictor.
ENSEMBLE_SIZE = 102
#: ``ANN_logo_0`` .. ``ANN_logo_101``, as upstream names them.
MODEL_STEM = "ANN_logo_{index}"

#: The converted scaler, written by the installer and read here.
SCALER_NAME = "deepviscosity_scaler.json"
#: Upstream's pickle, named so the installer and this module agree on it.
SCALER_SOURCE_NAME = "DeepViscosity_scaler.joblib"
#: Pinned because this file is a pickle. The installer refuses to read bytes
#: that do not match, instead of unpickling something nobody has looked at.
#: Measured 2026-11-18 on the ``main`` branch; 1,335 bytes.
SCALER_SHA256 = "2e9a5f56fc9006594e3f013c40dc66d2e641552daf62d01432f992a163587cd0"

#: The published class boundary, in centipoise, and the single concentration it
#: is defined at. Both are the model's own definition of its two classes, not a
#: pass mark chosen by this project.
CLASS_BOUNDARY_CP = 20.0
CONCENTRATION_MG_PER_ML = 150.0
FORMULATION = "20 mM histidine/histidine-HCl, pH 6.0"
#: Mean ensemble probability at or above this is the high class. Upstream uses
#: ``>= 0.5``, so a probability of exactly 0.5 is high, not low.
DECISION_PROBABILITY = 0.5
LOW_CLASS = "low"
HIGH_CLASS = "high"
#: 162 low and 67 high, measured at AstraZeneca and not published.
TRAINING_ANTIBODIES = 229

#: The 30 features, in the order upstream's ``features`` list gives them after
#: ``Name``. Written out rather than imported so that a reordering of
#: :data:`~rfproteina.disease.developability.DEEPSP_COLUMNS` cannot silently
#: feed this model's scaler the wrong column: the two are asserted equal below
#: and ``tests/test_disease_deepviscosity.py`` checks it independently.
DEEPVISCOSITY_FEATURES: tuple[str, ...] = (
    "SAP_pos_CDRH1",
    "SAP_pos_CDRH2",
    "SAP_pos_CDRH3",
    "SAP_pos_CDRL1",
    "SAP_pos_CDRL2",
    "SAP_pos_CDRL3",
    "SAP_pos_CDR",
    "SAP_pos_Hv",
    "SAP_pos_Lv",
    "SAP_pos_Fv",
    "SCM_neg_CDRH1",
    "SCM_neg_CDRH2",
    "SCM_neg_CDRH3",
    "SCM_neg_CDRL1",
    "SCM_neg_CDRL2",
    "SCM_neg_CDRL3",
    "SCM_neg_CDR",
    "SCM_neg_Hv",
    "SCM_neg_Lv",
    "SCM_neg_Fv",
    "SCM_pos_CDRH1",
    "SCM_pos_CDRH2",
    "SCM_pos_CDRH3",
    "SCM_pos_CDRL1",
    "SCM_pos_CDRL2",
    "SCM_pos_CDRL3",
    "SCM_pos_CDR",
    "SCM_pos_Hv",
    "SCM_pos_Lv",
    "SCM_pos_Fv",
)
FEATURE_COUNT = len(DEEPVISCOSITY_FEATURES)
assert (
    tuple(DEEPSP_COLUMNS) == DEEPVISCOSITY_FEATURES
), "DeepViscosity's feature order must match DEEPSP_COLUMNS: the scaler is positional"

#: The layer kinds :func:`_forward` implements. ``Flatten`` is a no-op on the
#: (n, 30) input but upstream's graph contains one, so it is accepted and
#: checked rather than ignored.
_SUPPORTED_LAYERS = frozenset({"InputLayer", "Flatten", "Dense"})
#: ``relu`` is accepted because the published hyperparameter search chose
#: between ReLU and tanh, so a later release could ship either.
_SUPPORTED_ACTIVATIONS = frozenset({"tanh", "sigmoid", "relu", "linear", None, ""})

#: Keys that may end in ``_cp`` because they are published constants rather
#: than anything this model computed, mapped to the value they must carry.
_PUBLISHED_CENTIPOISE = {"class_boundary_cp": CLASS_BOUNDARY_CP}

_ENSEMBLE: dict[str, tuple[tuple[Any, ...], ...]] = {}
_ENSEMBLE_LOCK = threading.Lock()


class DeepViscosityUnavailableError(Exception):
    """DeepViscosity cannot be run here. Its absence is reported, never worked around."""


@dataclass(frozen=True)
class Scaler:
    """The published input scaling: ``(x - mean) / scale``, column by column."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    #: 229, the training set size. Carried so a record can show the scaler
    #: belongs to the published model rather than to something refitted.
    samples: int
    sklearn_version: str

    def __post_init__(self) -> None:
        if len(self.mean) != FEATURE_COUNT or len(self.scale) != FEATURE_COUNT:
            raise DeepViscosityUnavailableError(
                f"deepviscosity_scaler_width:{len(self.mean)},{len(self.scale)}!={FEATURE_COUNT}"
            )
        if any(value <= 0.0 for value in self.scale):
            # A zero scale would divide a real descriptor to infinity; a
            # negative one would flip its sign. Neither is a StandardScaler.
            raise DeepViscosityUnavailableError("deepviscosity_scaler_nonpositive_scale")


@dataclass(frozen=True)
class ViscosityPrediction:
    """One antibody's ensemble verdict."""

    name: str
    #: Mean over the ensemble of P(viscosity > 20 cP at 150 mg/mL).
    probability: float
    #: Spread across the ensemble, population standard deviation as upstream
    #: reports it. This is model disagreement, not experimental error.
    probability_sd: float
    label: int
    models: int

    @property
    def viscosity_class(self) -> str:
        return HIGH_CLASS if self.label else LOW_CLASS

    @property
    def margin(self) -> float:
        """How far the mean probability sits from the decision boundary."""
        return abs(self.probability - DECISION_PROBABILITY)

    @property
    def near_boundary(self) -> bool:
        """Whether the ensemble disagrees by more than the call is clear by.

        Compared against this antibody's own spread rather than a fixed number,
        because the spread is what says how much the 102 models agree. Four of
        the sixteen published test antibodies land within 0.02 of the boundary
        with spreads near 0.13, so a class alone would hide a coin toss.
        """
        return self.margin < self.probability_sd


@dataclass(frozen=True)
class ViscosityResult:
    """Predictions for a set of antibodies, or why they are missing."""

    rows: tuple[ViscosityPrediction, ...]
    method: str = VISCOSITY_METHOD
    source: str = VISCOSITY_SOURCE
    #: Antibodies left out, name -> reason, as DeepSPResult reports them.
    skipped: Mapping[str, str] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.rows)


def default_models_dir() -> Path | None:
    """``DATA_PATH/deepviscosity/models``, or ``None`` where ``DATA_PATH`` is unset."""
    root = os.environ.get("DATA_PATH", "").strip()
    return Path(root) / VISCOSITY_SUBDIR if root else None


def model_names() -> tuple[str, ...]:
    """Every file the runner opens, in a stable order."""
    names: list[str] = []
    for index in range(ENSEMBLE_SIZE):
        stem = MODEL_STEM.format(index=index)
        names.extend((f"{stem}.json", f"{stem}.h5"))
    return tuple(names)


def viscosity_available(models: Path) -> tuple[bool, str | None]:
    """Whether the ensemble, the scaler and the libraries are all here."""
    try:
        import h5py  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment dependent
        return False, f"deepviscosity_dependency_missing:{error.name or error}"
    if not (models / SCALER_NAME).is_file():
        return False, f"deepviscosity_scaler_absent:{SCALER_NAME}"
    missing = [name for name in model_names() if not (models / name).is_file()]
    if missing:
        # 102 pairs, so the names are counted rather than listed: a blocker
        # naming 204 files is unreadable and says no more than the count.
        return False, f"deepviscosity_models_absent:{len(missing)}_of_{2 * ENSEMBLE_SIZE}"
    return True, None


def load_scaler(path: Path) -> Scaler:
    """Read the installer's JSON. Nothing is unpickled."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DeepViscosityUnavailableError(f"deepviscosity_scaler_unreadable:{type(error).__name__}") from error
    if not isinstance(document, dict):
        raise DeepViscosityUnavailableError("deepviscosity_scaler_malformed")
    try:
        mean = tuple(float(value) for value in document["mean"])
        scale = tuple(float(value) for value in document["scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise DeepViscosityUnavailableError(f"deepviscosity_scaler_malformed:{type(error).__name__}") from error
    return Scaler(
        mean=mean,
        scale=scale,
        samples=int(document.get("samples", 0)),
        sklearn_version=str(document.get("sklearn_version", "")),
    )


def _architecture(path: Path) -> list[dict[str, Any]]:
    """The Keras layer configs, checked against the architecture we implement."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DeepViscosityUnavailableError(f"deepviscosity_architecture_unreadable:{type(error).__name__}") from error
    layers = (config.get("config") or {}).get("layers") or []
    if not layers:
        raise DeepViscosityUnavailableError(f"deepviscosity_architecture_empty:{path.name}")
    width: int | None = None
    for layer in layers:
        kind, settings = layer.get("class_name", ""), layer.get("config") or {}
        if kind not in _SUPPORTED_LAYERS:
            raise DeepViscosityUnavailableError(f"deepviscosity_layer_unsupported:{path.name}:{kind}")
        if kind == "Dense" and settings.get("activation") not in _SUPPORTED_ACTIVATIONS:
            raise DeepViscosityUnavailableError(
                f"deepviscosity_activation_unsupported:{path.name}:{settings.get('activation')}"
            )
        shape = settings.get("batch_input_shape")
        if width is None and isinstance(shape, list) and len(shape) == 2:
            width = shape[1]
    if width is not None and width != FEATURE_COUNT:
        raise DeepViscosityUnavailableError(f"deepviscosity_input_width:{path.name}:{width}!={FEATURE_COUNT}")

    # The head decides what the number means. A softmax, or two output units,
    # would still run and still produce something in [0, 1] that is not
    # P(high), so the shape of the head is checked rather than assumed.
    head = layers[-1]
    settings = head.get("config") or {}
    if head.get("class_name") != "Dense":
        raise DeepViscosityUnavailableError(f"deepviscosity_head_not_dense:{path.name}:{head.get('class_name')}")
    if settings.get("units") != 1:
        raise DeepViscosityUnavailableError(f"deepviscosity_head_units:{path.name}:{settings.get('units')}")
    if settings.get("activation") != "sigmoid":
        raise DeepViscosityUnavailableError(f"deepviscosity_head_activation:{path.name}:{settings.get('activation')}")
    return layers


def _load_model(architecture: Path, weights: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """One model's checked layers and its weights."""
    layers = _architecture(architecture)
    store = keras_weights(weights)
    expected = FEATURE_COUNT
    for layer in layers:
        if layer.get("class_name") != "Dense":
            continue
        name = (layer.get("config") or {}).get("name", "")
        held = store.get(name)
        if held is None or "kernel:0" not in held or "bias:0" not in held:
            raise DeepViscosityUnavailableError(f"deepviscosity_weights_absent:{weights.name}:{name}")
        kernel = held["kernel:0"]
        if kernel.ndim != 2 or kernel.shape[0] != expected:
            # Every consecutive width here differs (30, 128, 64, 32, 16, 1), so
            # a transposed kernel cannot line up by coincidence.
            raise DeepViscosityUnavailableError(
                f"deepviscosity_kernel_shape:{weights.name}:{name}:{kernel.shape}!=({expected}, *)"
            )
        expected = kernel.shape[1]
    if expected != 1:
        raise DeepViscosityUnavailableError(f"deepviscosity_output_width:{weights.name}:{expected}")
    return layers, store


def _ensemble_models(models: Path) -> tuple[tuple[Any, ...], ...]:
    """The 102 checked models, loaded once per directory and kept.

    Reading 204 files for every antibody would dominate the run time, and the
    files do not change under us: the installer writes them whole.
    """
    key = str(Path(models).resolve())
    with _ENSEMBLE_LOCK:
        held = _ENSEMBLE.get(key)
        if held is not None:
            return held
    loaded: list[tuple[Any, ...]] = []
    for index in range(ENSEMBLE_SIZE):
        stem = MODEL_STEM.format(index=index)
        loaded.append(_load_model(models / f"{stem}.json", models / f"{stem}.h5"))
    built = tuple(loaded)
    with _ENSEMBLE_LOCK:
        _ENSEMBLE[key] = built
    return built


def reset_ensemble_cache() -> None:
    """Forget loaded models, so a test can install different weights."""
    with _ENSEMBLE_LOCK:
        _ENSEMBLE.clear()


def feature_matrix(rows: Sequence[Mapping[str, object]]) -> Any:
    """``(n, 30)`` float32 of the descriptors, in :data:`DEEPVISCOSITY_FEATURES` order."""
    import numpy as np

    if not rows:
        raise DeepViscosityUnavailableError("deepviscosity_no_rows")
    out = np.zeros((len(rows), FEATURE_COUNT), dtype=np.float32)
    for position, row in enumerate(rows):
        for index, column in enumerate(DEEPVISCOSITY_FEATURES):
            if column not in row:
                raise DeepViscosityUnavailableError(f"deepviscosity_feature_absent:{column}")
            try:
                out[position, index] = float(row[column])  # type: ignore[arg-type]
            except (TypeError, ValueError) as error:
                raise DeepViscosityUnavailableError(f"deepviscosity_feature_unparsed:{column}") from error
    return out


def scale_features(matrix: Any, scaler: Scaler) -> Any:
    """Apply the published scaling the way scikit-learn applied it.

    ``StandardScaler.transform`` subtracts and divides in place on a float32
    array, so each step rounds back to float32 before the next one. Doing it as
    one expression in float64 and casting at the end gives slightly different
    last bits, and four of the published test antibodies sit within 0.02 of the
    decision boundary, so the order is reproduced rather than tidied.
    """
    import numpy as np

    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT:
        raise DeepViscosityUnavailableError(f"deepviscosity_matrix_shape:{matrix.shape}")
    scaled = np.array(matrix, dtype=np.float32, copy=True)
    scaled -= np.asarray(scaler.mean, dtype=np.float64)
    scaled /= np.asarray(scaler.scale, dtype=np.float64)
    return scaled


def _activate(tensor: Any, activation: object, where: str) -> Any:
    import torch

    if activation in (None, "", "linear"):
        return tensor
    if activation == "tanh":
        return torch.tanh(tensor)
    if activation == "sigmoid":
        return torch.sigmoid(tensor)
    if activation == "relu":
        return torch.relu(tensor)
    raise DeepViscosityUnavailableError(f"deepviscosity_activation_unsupported:{where}:{activation}")


def _forward(matrix: Any, layers: Sequence[Mapping[str, Any]], store: Mapping[str, Mapping[str, Any]]) -> Any:
    """Run one model over an ``(n, 30)`` scaled batch, in torch.

    A plain dense stack, so the two traps of the DeepSP port do not arise here:
    there is no convolution whose activation ordering matters, and the
    ``Flatten`` sits on an already flat input, so it must **not** transpose.
    That it is a no-op is checked rather than assumed, because a rank other
    than two here would mean the caller built the wrong matrix.
    """
    import torch

    tensor = torch.from_numpy(matrix).float()
    for layer in layers:
        kind, config = layer["class_name"], layer.get("config") or {}
        if kind == "InputLayer":
            continue
        if kind == "Flatten":
            if tensor.dim() != 2:
                raise DeepViscosityUnavailableError(f"deepviscosity_flatten_rank:{tensor.dim()}")
            continue
        if kind == "Dense":
            name = config.get("name", "")
            held = store[name]
            tensor = tensor @ torch.from_numpy(held["kernel:0"]).float() + torch.from_numpy(held["bias:0"]).float()
            tensor = _activate(tensor, config.get("activation"), name)
            continue
        raise DeepViscosityUnavailableError(f"deepviscosity_layer_unsupported:{kind}")  # pragma: no cover
    return tensor.detach().numpy()


def predict_probabilities(matrix: Any, models: Path, scaler: Scaler | None = None) -> tuple[Any, Any]:
    """Mean and spread of P(high) over the ensemble, for an ``(n, 30)`` batch.

    ``matrix`` holds raw DeepSP descriptors; the scaling is applied here so a
    caller cannot forget it.
    """
    import numpy as np

    models = Path(models)
    held = scaler if scaler is not None else load_scaler(models / SCALER_NAME)
    scaled = scale_features(matrix, held)
    ensemble = _ensemble_models(models)
    stack = np.zeros((len(ensemble), scaled.shape[0]), dtype=np.float64)
    for position, (layers, store) in enumerate(ensemble):
        out = _forward(scaled, layers, store)
        if out.shape != (scaled.shape[0], 1):
            raise DeepViscosityUnavailableError(f"deepviscosity_output_shape:{out.shape}")
        stack[position] = out[:, 0]
    if not ((stack >= 0.0).all() and (stack <= 1.0).all()):
        # The head is a sigmoid, so anything outside [0, 1] means the graph we
        # ran was not the graph the weights belong to.
        raise DeepViscosityUnavailableError("deepviscosity_probability_out_of_range")
    # Population standard deviation, which is what numpy's default gives and
    # what upstream reported as Prob_Std.
    return stack.mean(axis=0), stack.std(axis=0)


def predict_from_descriptors(
    rows: Sequence[Mapping[str, object]],
    models: Path,
    *,
    names: Sequence[str] | None = None,
) -> ViscosityResult:
    """Classify antibodies whose DeepSP descriptors are already in hand.

    Separate from :func:`run_deepviscosity` on purpose: it is the half that
    needs no numbering and no DeepSP models, so a caller holding a DeepSP CSV
    -- or a test holding the published descriptors -- can exercise the ensemble
    alone.
    """
    models = Path(models)
    available, reason = viscosity_available(models)
    if not available:
        raise DeepViscosityUnavailableError(reason or "deepviscosity_unavailable")
    if names is not None:
        labels = list(names)
    else:
        labels = [str(row.get("Name", f"row{index + 1}")) for index, row in enumerate(rows)]
    if len(labels) != len(rows):
        raise DeepViscosityUnavailableError(f"deepviscosity_name_count:{len(labels)}!={len(rows)}")
    matrix = feature_matrix(rows)
    mean, spread = predict_probabilities(matrix, models)
    predictions = tuple(
        ViscosityPrediction(
            name=labels[index],
            probability=float(mean[index]),
            probability_sd=float(spread[index]),
            label=int(mean[index] >= DECISION_PROBABILITY),
            models=ENSEMBLE_SIZE,
        )
        for index in range(len(rows))
    )
    return ViscosityResult(predictions)


def run_deepviscosity(
    antibodies: Mapping[str, tuple[str, str]],
    models: Path,
    deepsp_models: Path,
    *,
    cpu: bool = True,
    ncpu: int = 1,
) -> tuple[ViscosityResult, DeepSPResult]:
    """Sequences to viscosity classes, through DeepSP.

    Both chains are required, for DeepSP's reason: its light half is 127 of the
    272 input columns, so a lone heavy chain would be scored against 127 gaps.
    The DeepSP result is returned alongside, because the descriptors are the
    evidence for the class and are worth keeping with it.

    Raises :class:`DeepViscosityUnavailableError` or
    :class:`~rfproteina.disease.deepsp.DeepSPUnavailableError` when either
    model set or the libraries are absent, and
    :class:`~rfproteina.disease.developability.NumberingUnavailableError` when
    ANARCII is.
    """
    models = Path(models)
    available, reason = viscosity_available(models)
    if not available:
        raise DeepViscosityUnavailableError(reason or "deepviscosity_unavailable")
    descriptors = run_deepsp(antibodies, Path(deepsp_models), cpu=cpu, ncpu=ncpu)
    if not descriptors.rows:
        return ViscosityResult((), skipped=dict(descriptors.skipped)), descriptors
    result = predict_from_descriptors(list(descriptors.rows), models, names=list(descriptors.names))
    return ViscosityResult(result.rows, skipped=dict(descriptors.skipped)), descriptors


def deepviscosity_report(
    sequences: Mapping[str, str],
    models: Path | None = None,
    deepsp_models: Path | None = None,
    *,
    name: str = "design",
    cpu: bool = True,
    ncpu: int = 1,
) -> dict[str, object]:
    """One JSON-shaped record for one antibody.

    The shape the other developability readers use: a method, its provenance,
    and a named blocker instead of a result when something is missing. Nothing
    raises -- a caller printing a record wants a record even when the answer is
    "not here", and which piece is missing is the useful part.

    ``pass_cut`` stays ``None``. The 20 cP boundary is the classifier's own
    definition of its two classes, not a mark this project has decided an
    antibody fails: a high call is information for a formulator, and no
    publication says a design is unusable because a sequence model put it above
    20 cP at one concentration in one buffer.
    """
    directory = Path(models) if models is not None else default_models_dir()
    if directory is None:
        return {"viscosity_blocker": "data_path_unset", "method": VISCOSITY_METHOD}
    available, reason = viscosity_available(directory)
    if not available:
        return {"viscosity_blocker": reason or "deepviscosity_unavailable", "method": VISCOSITY_METHOD}

    sap_models = Path(deepsp_models) if deepsp_models is not None else deepsp_models_dir()
    if sap_models is None:
        return {"viscosity_blocker": "data_path_unset", "method": VISCOSITY_METHOD}
    ready, why = deepsp_available(sap_models)
    if not ready:
        return {"viscosity_blocker": why or "deepsp_unavailable", "method": VISCOSITY_METHOD}

    pairs = {name: (sequences.get("H", ""), sequences.get("L", ""))}
    if not all(pairs[name]):
        # ``{"H": ..., "L": ...}`` is the convention and is taken by name. Any
        # other two keys fall back to insertion order, which can put the light
        # chain first -- and that is safe rather than silently wrong, because
        # run_deepsp numbers each half and refuses a pair whose first chain
        # ANARCII does not call heavy. The blocker then names the chain type it
        # actually found. test_a_swapped_pair_is_refused_not_scored pins it.
        chains = [value for value in sequences.values() if value]
        if len(chains) != 2:
            return {
                "viscosity_blocker": "both_chains_required",
                "method": VISCOSITY_METHOD,
                "chains_given": sorted(key for key, value in sequences.items() if value),
            }
        pairs = {name: (chains[0], chains[1])}

    try:
        result, descriptors = run_deepviscosity(pairs, directory, sap_models, cpu=cpu, ncpu=ncpu)
    except (DeepViscosityUnavailableError, DeepSPUnavailableError) as error:
        return {"viscosity_blocker": str(error), "method": VISCOSITY_METHOD}
    except NumberingUnavailableError as error:
        return {"viscosity_blocker": str(error), "method": VISCOSITY_METHOD}
    if not result.rows:
        return {
            "viscosity_blocker": result.skipped.get(name, "not_scored"),
            "method": VISCOSITY_METHOD,
        }

    prediction = result.rows[0]
    row = descriptors.rows[0]
    record: dict[str, object] = {
        "method": VISCOSITY_METHOD,
        "source": VISCOSITY_SOURCE,
        "paper": VISCOSITY_PAPER,
        "name": prediction.name,
        "predicted_class": prediction.viscosity_class,
        "class_label": prediction.label,
        "probability_high": prediction.probability,
        "probability_sd": prediction.probability_sd,
        "margin": prediction.margin,
        "near_boundary": prediction.near_boundary,
        "models": prediction.models,
        "decision_probability": DECISION_PROBABILITY,
        "class_boundary_cp": CLASS_BOUNDARY_CP,
        "concentration_mg_per_ml": CONCENTRATION_MG_PER_ML,
        "formulation": FORMULATION,
        "descriptors": {column: float(row[column]) for column in DEEPVISCOSITY_FEATURES},  # type: ignore[arg-type]
        "descriptor_method": DEEPSP_METHOD,
        "training_antibodies": TRAINING_ANTIBODIES,
        "pass_cut": None,
        "pass_cut_reason": "published_class_boundary_is_the_models_own_class_definition_not_a_project_pass_mark",
        "caveats": list(CAVEATS),
    }
    assert_no_predicted_centipoise(record)
    return record


#: What the authors say the model cannot do, kept with the number rather than
#: in a document nobody reads next to it.
CAVEATS: tuple[str, ...] = (
    f"a class at {CONCENTRATION_MG_PER_ML:.0f} mg/mL in {FORMULATION}; another concentration or buffer is an "
    "extrapolation, and the authors' own 38-antibody test set, measured in histidine/sucrose at pH 5.8, was "
    "called high viscosity throughout",
    "Fv sequence only, so isotype and subclass effects (IgA, IgM, IgG1 against IgG2 or IgG4) are not represented",
    "trained on sequence-diverse antibodies, so closely related variants of one lineage are its weakest case",
    f"{TRAINING_ANTIBODIES} training antibodies, 162 low and 67 high, so the classes are imbalanced",
    "a class and a probability, never a viscosity in centipoise",
)


def assert_no_predicted_centipoise(record: Mapping[str, object]) -> None:
    """Refuse a record that puts a model output in a centipoise field.

    DeepViscosity produces a probability and a class. It does not produce a
    viscosity, and a probability of 0.55 written into a ``viscosity_cp`` field
    would be read downstream as 0.55 cP -- a measurement, and a thousand times
    away from the boundary it was meant to express. The only ``_cp`` key
    allowed is the published boundary, and it must still carry the published
    value.
    """
    for key, value in record.items():
        folded = key.casefold()
        if not (folded.endswith("_cp") or folded.endswith("_centipoise") or folded.endswith("_cps")):
            continue
        if key not in _PUBLISHED_CENTIPOISE:
            raise DeepViscosityUnavailableError(f"centipoise_key_is_not_a_measurement:{key}")
        if float(value) != _PUBLISHED_CENTIPOISE[key]:  # type: ignore[arg-type]
            raise DeepViscosityUnavailableError(f"published_centipoise_changed:{key}:{value}")


def write_viscosity_csv(result: ViscosityResult, path: Path) -> Path:
    """Write upstream's own output columns, so the file is interchangeable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("Name", "Prob_Mean", "Prob_Std", "DeepViscosity_classes")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in result.rows:
            writer.writerow(
                {
                    "Name": row.name,
                    "Prob_Mean": row.probability,
                    "Prob_Std": row.probability_sd,
                    "DeepViscosity_classes": row.label,
                }
            )
    return path
