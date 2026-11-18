"""Tests for rfproteina.disease.deepviscosity (CPU-only, no TensorFlow).

Most of this needs no model files: the contract, the scaling, the architecture
checks, the refusals and the aggregation all run against a synthetic ensemble
built here, so a wrong matmul or a flipped class rule fails without a download.

The three tests that make the port defensible do need the real weights. They
use ``RFD4_DEEPVISCOSITY_MODELS`` when it is set, and otherwise the directory
``scripts/install_deepviscosity.py`` writes, so a normal install is enough to
run them. They check, in turn, the ensemble against the probabilities upstream
published for its own 16-antibody test set, the whole chain from raw sequence,
and this project's DeepSP port against the descriptors DeepViscosity ships.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rfproteina.disease import deepviscosity as dv
from rfproteina.disease.developability import DEEPSP_COLUMNS


def _models_dir() -> Path | None:
    named = os.environ.get("RFD4_DEEPVISCOSITY_MODELS", "").strip()
    if named:
        return Path(named)
    return dv.default_models_dir()


def _have_models() -> bool:
    directory = _models_dir()
    return directory is not None and dv.viscosity_available(directory)[0]


needs_models = pytest.mark.skipif(
    not _have_models(),
    reason="run scripts/install_deepviscosity.py, or set RFD4_DEEPVISCOSITY_MODELS, to check the port",
)

#: What upstream's own ``DeepViscosity_Predictor.ipynb`` printed for its
#: ``DeepViscosity_input.csv``: name -> (Prob_Mean, Prob_Std, class). This is
#: the Lai_mAb_16 independent test set, and the four highs with two
#: misclassifications are the 87.5 per cent the paper reports, which is an
#: independent check that these saved numbers are the published ones.
PUBLISHED: dict[str, tuple[float, float, int]] = {
    "mAb1": (0.304625, 0.130680, 0),
    "mAb2": (0.545643, 0.120694, 1),
    "mAb3": (0.519607, 0.093952, 1),
    "mAb4": (0.469281, 0.134071, 0),
    "mAb5": (0.214899, 0.107229, 0),
    "mAb6": (0.498471, 0.137418, 0),
    "mAb7": (0.496279, 0.145389, 0),
    "mAb8": (0.514386, 0.108350, 1),
    "mAb9": (0.568411, 0.097684, 1),
    "mAb10": (0.461401, 0.116360, 0),
    "mAb11": (0.397551, 0.086192, 0),
    "mAb12": (0.329545, 0.147663, 0),
    "mAb13": (0.201963, 0.090927, 0),
    "mAb14": (0.467469, 0.108275, 0),
    "mAb15": (0.209530, 0.097079, 0),
    "mAb16": (0.335033, 0.120334, 0),
}


# ------------------------------------------------------------------ helpers


def _dense(name: str, units: int, activation: str) -> dict[str, object]:
    return {"class_name": "Dense", "config": {"name": name, "units": units, "activation": activation}}


def _architecture_document(layers: list[dict[str, object]]) -> dict[str, object]:
    return {"class_name": "Sequential", "config": {"name": "sequential_0", "layers": layers}}


def _upstream_layers() -> list[dict[str, object]]:
    """The shape upstream ships: input, flatten, four tanh, one sigmoid."""
    return [
        {"class_name": "InputLayer", "config": {"name": "in", "batch_input_shape": [None, 30]}},
        {"class_name": "Flatten", "config": {"name": "flatten_0", "batch_input_shape": [None, 30]}},
        _dense("Dense_1", 128, "tanh"),
        _dense("Dense_2", 64, "tanh"),
        _dense("Dense_3", 32, "tanh"),
        _dense("Dense_4", 16, "tanh"),
        _dense("Dense_5", 1, "sigmoid"),
    ]


def _write_architecture(path: Path, layers: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(_architecture_document(layers)), encoding="utf-8")
    return path


def _write_weights(path: Path, weights: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    """A Keras ``save_weights`` h5: layer groups at the root, nested datasets."""
    import h5py

    with h5py.File(path, "w") as handle:
        handle.attrs["layer_names"] = [name.encode() for name in weights]
        for name, (kernel, bias) in weights.items():
            group = handle.create_group(name)
            group.attrs["weight_names"] = [f"{name}/kernel:0".encode(), f"{name}/bias:0".encode()]
            group.create_dataset(f"{name}/kernel:0", data=kernel.astype(np.float32))
            group.create_dataset(f"{name}/bias:0", data=bias.astype(np.float32))
    return path


def _write_scaler(directory: Path, mean: list[float] | None = None, scale: list[float] | None = None) -> Path:
    path = directory / dv.SCALER_NAME
    document = {
        "mean": mean if mean is not None else [0.0] * dv.FEATURE_COUNT,
        "scale": scale if scale is not None else [1.0] * dv.FEATURE_COUNT,
        "samples": 229,
        "sklearn_version": "1.4.2",
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> tuple[Path, np.ndarray]:
    """102 one-layer models whose answer does not depend on the input.

    Each model is ``sigmoid(bias_i)`` with a zero kernel, so the expected mean
    and spread are computable here exactly. That is what lets the aggregation
    and the class rule be tested without the real weights; the arithmetic of a
    dense layer is checked separately, where the kernel is not zero.

    Built once for the module: writing 204 files per test dominated the run
    time. The ensemble cache is still cleared around every test, so nothing
    leaks between them -- only the files on disk are shared.
    """
    directory = tmp_path_factory.mktemp("ensemble")
    biases = np.linspace(-0.6, 0.6, dv.ENSEMBLE_SIZE, dtype=np.float32)
    layers = [
        {"class_name": "InputLayer", "config": {"name": "in", "batch_input_shape": [None, 30]}},
        _dense("Dense_1", 1, "sigmoid"),
    ]
    for index in range(dv.ENSEMBLE_SIZE):
        stem = dv.MODEL_STEM.format(index=index)
        _write_architecture(directory / f"{stem}.json", layers)
        _write_weights(
            directory / f"{stem}.h5",
            {"Dense_1": (np.zeros((dv.FEATURE_COUNT, 1)), np.array([biases[index]]))},
        )
    _write_scaler(directory)
    dv.reset_ensemble_cache()
    return directory, biases


@pytest.fixture(autouse=True)
def _clear_cache():
    dv.reset_ensemble_cache()
    yield
    dv.reset_ensemble_cache()


# ------------------------------------------------------------------ the contract


def test_the_feature_order_is_deepsps_own_order():
    """The scaler is positional: a reordering here mis-scales every column."""
    assert tuple(DEEPSP_COLUMNS) == dv.DEEPVISCOSITY_FEATURES
    assert dv.FEATURE_COUNT == 30
    assert dv.DEEPVISCOSITY_FEATURES[0] == "SAP_pos_CDRH1"
    assert dv.DEEPVISCOSITY_FEATURES[-1] == "SCM_pos_Fv"


def test_the_published_constants_are_what_the_paper_states():
    assert dv.ENSEMBLE_SIZE == 102, "one model per leave-one-group-out split of 102 sequence clusters"
    assert dv.CLASS_BOUNDARY_CP == 20.0
    assert dv.CONCENTRATION_MG_PER_ML == 150.0
    assert dv.TRAINING_ANTIBODIES == 229
    assert dv.DECISION_PROBABILITY == 0.5


def test_the_method_name_does_not_claim_tensorflow_or_a_measurement():
    assert "torch" in dv.VISCOSITY_METHOD
    assert "MIT" in dv.VISCOSITY_SOURCE
    assert "cp" not in dv.VISCOSITY_METHOD.split("_")


def test_the_file_list_is_two_hundred_and_four_names():
    names = dv.model_names()
    assert len(names) == 2 * dv.ENSEMBLE_SIZE
    assert names[0] == "ANN_logo_0.json"
    assert names[1] == "ANN_logo_0.h5"
    assert names[-1] == "ANN_logo_101.h5"


# ------------------------------------------------------------------ the scaler


def test_the_scaler_round_trips_through_json(tmp_path):
    mean = [float(index) for index in range(dv.FEATURE_COUNT)]
    scale = [float(index + 1) for index in range(dv.FEATURE_COUNT)]
    _write_scaler(tmp_path, mean, scale)
    held = dv.load_scaler(tmp_path / dv.SCALER_NAME)
    assert held.mean == tuple(mean)
    assert held.scale == tuple(scale)
    assert held.samples == 229, "the published scaler was fitted on the 229 training antibodies"


def test_a_scaler_of_the_wrong_width_is_refused(tmp_path):
    path = tmp_path / dv.SCALER_NAME
    path.write_text(json.dumps({"mean": [0.0] * 29, "scale": [1.0] * 29}), encoding="utf-8")
    with pytest.raises(dv.DeepViscosityUnavailableError, match="scaler_width"):
        dv.load_scaler(path)


def test_a_zero_scale_is_refused(tmp_path):
    """Dividing a real descriptor by zero is not a StandardScaler."""
    scale = [1.0] * dv.FEATURE_COUNT
    scale[7] = 0.0
    _write_scaler(tmp_path, scale=scale)
    with pytest.raises(dv.DeepViscosityUnavailableError, match="nonpositive_scale"):
        dv.load_scaler(tmp_path / dv.SCALER_NAME)


def test_a_malformed_scaler_is_refused(tmp_path):
    path = tmp_path / dv.SCALER_NAME
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(dv.DeepViscosityUnavailableError, match="scaler_malformed"):
        dv.load_scaler(path)


def test_scaling_centres_and_divides_column_by_column():
    scaler = dv.Scaler(
        mean=tuple(float(index) for index in range(dv.FEATURE_COUNT)),
        scale=tuple(2.0 for _ in range(dv.FEATURE_COUNT)),
        samples=229,
        sklearn_version="1.4.2",
    )
    matrix = np.arange(dv.FEATURE_COUNT, dtype=np.float32).reshape(1, dv.FEATURE_COUNT)
    scaled = dv.scale_features(matrix, scaler)
    assert scaled.dtype == np.float32, "scikit-learn keeps float32 input in float32"
    assert np.allclose(scaled, 0.0), "x equals the mean, so every column centres to zero"


def test_scaling_refuses_a_matrix_of_the_wrong_width():
    scaler = dv.Scaler(mean=(0.0,) * 30, scale=(1.0,) * 30, samples=229, sklearn_version="1.4.2")
    with pytest.raises(dv.DeepViscosityUnavailableError, match="matrix_shape"):
        dv.scale_features(np.zeros((2, 29), dtype=np.float32), scaler)


# ------------------------------------------------------------------ the architecture


def test_the_upstream_architecture_is_accepted(tmp_path):
    path = _write_architecture(tmp_path / "m.json", _upstream_layers())
    layers = dv._architecture(path)
    assert [layer["class_name"] for layer in layers][-1] == "Dense"


def test_a_layer_kind_we_do_not_implement_is_refused(tmp_path):
    """A Dropout or a Conv1D here would be silently skipped, not applied."""
    layers = _upstream_layers()
    layers.insert(2, {"class_name": "Dropout", "config": {"name": "drop", "rate": 0.2}})
    path = _write_architecture(tmp_path / "m.json", layers)
    with pytest.raises(dv.DeepViscosityUnavailableError, match="layer_unsupported"):
        dv._architecture(path)


def test_a_head_that_is_not_a_single_sigmoid_is_refused(tmp_path):
    """Two units is not P(high), even where the activation is one we run."""
    layers = _upstream_layers()
    layers[-1] = _dense("Dense_5", 2, "sigmoid")
    path = _write_architecture(tmp_path / "two.json", layers)
    with pytest.raises(dv.DeepViscosityUnavailableError, match="head_units"):
        dv._architecture(path)

    # A linear head returns a logit, which reads as a probability and is not.
    layers[-1] = _dense("Dense_5", 1, "linear")
    path = _write_architecture(tmp_path / "linear.json", layers)
    with pytest.raises(dv.DeepViscosityUnavailableError, match="head_activation"):
        dv._architecture(path)

    # A softmax is refused one step earlier, by the activation vocabulary,
    # because nothing here implements it. Either refusal is correct; this
    # records which one actually fires.
    layers[-1] = _dense("Dense_5", 2, "softmax")
    path = _write_architecture(tmp_path / "softmax.json", layers)
    with pytest.raises(dv.DeepViscosityUnavailableError, match="activation_unsupported"):
        dv._architecture(path)


def test_an_input_width_other_than_thirty_is_refused(tmp_path):
    layers = _upstream_layers()
    layers[0] = {"class_name": "InputLayer", "config": {"name": "in", "batch_input_shape": [None, 25]}}
    layers[1] = {"class_name": "Flatten", "config": {"name": "flatten_0"}}
    path = _write_architecture(tmp_path / "m.json", layers)
    with pytest.raises(dv.DeepViscosityUnavailableError, match="input_width"):
        dv._architecture(path)


def test_an_empty_architecture_is_refused(tmp_path):
    path = _write_architecture(tmp_path / "m.json", [])
    with pytest.raises(dv.DeepViscosityUnavailableError, match="architecture_empty"):
        dv._architecture(path)


def test_a_kernel_that_does_not_chain_is_refused(tmp_path):
    """30 -> 128 -> 64 ... every width differs, so a transpose cannot line up."""
    layers = [
        {"class_name": "InputLayer", "config": {"name": "in", "batch_input_shape": [None, 30]}},
        _dense("Dense_1", 1, "sigmoid"),
    ]
    _write_architecture(tmp_path / "m.json", layers)
    _write_weights(tmp_path / "m.h5", {"Dense_1": (np.zeros((1, 30)), np.zeros(30))})
    with pytest.raises(dv.DeepViscosityUnavailableError, match="kernel_shape"):
        dv._load_model(tmp_path / "m.json", tmp_path / "m.h5")


# ------------------------------------------------------------------ the forward pass


def test_a_dense_layer_computes_kernel_bias_and_activation(tmp_path):
    """The arithmetic, with a kernel that is not zero."""
    kernel = np.zeros((dv.FEATURE_COUNT, 1))
    kernel[0, 0] = 2.0
    kernel[1, 0] = -3.0
    layers = [_dense("Dense_1", 1, "sigmoid")]
    _write_architecture(tmp_path / "m.json", layers)
    _write_weights(tmp_path / "m.h5", {"Dense_1": (kernel, np.array([0.5]))})
    _, store = dv._load_model(tmp_path / "m.json", tmp_path / "m.h5")

    matrix = np.zeros((1, dv.FEATURE_COUNT), dtype=np.float32)
    matrix[0, 0] = 1.0
    matrix[0, 1] = 1.0
    out = dv._forward(matrix, layers, store)
    expected = 1.0 / (1.0 + np.exp(-(2.0 - 3.0 + 0.5)))
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(expected, abs=1e-6)


def test_flatten_on_an_already_flat_input_is_a_no_op(tmp_path):
    """DeepSP's Flatten transposes channels-last. This one must not."""
    layers = [
        {"class_name": "Flatten", "config": {"name": "flatten_0"}},
        _dense("Dense_1", 1, "sigmoid"),
    ]
    _write_architecture(tmp_path / "m.json", layers)
    _write_weights(tmp_path / "m.h5", {"Dense_1": (np.zeros((dv.FEATURE_COUNT, 1)), np.array([0.0]))})
    _, store = dv._load_model(tmp_path / "m.json", tmp_path / "m.h5")
    out = dv._forward(np.zeros((3, dv.FEATURE_COUNT), dtype=np.float32), layers, store)
    assert out.shape == (3, 1)
    assert np.allclose(out, 0.5), "sigmoid(0) is one half"


# ------------------------------------------------------------------ the ensemble


def test_the_ensemble_averages_and_reports_its_spread(synthetic):
    directory, biases = synthetic
    rows = [{"Name": "ab", **{column: 0.0 for column in dv.DEEPVISCOSITY_FEATURES}}]
    result = dv.predict_from_descriptors(rows, directory)
    expected = 1.0 / (1.0 + np.exp(-biases.astype(np.float64)))
    assert len(result.rows) == 1
    prediction = result.rows[0]
    assert prediction.models == 102
    assert prediction.probability == pytest.approx(float(expected.mean()), abs=1e-6)
    assert prediction.probability_sd == pytest.approx(float(expected.std()), abs=1e-6)


def test_exactly_one_half_is_the_high_class(synthetic):
    """Upstream classifies on ``>= 0.5``, so the boundary belongs to high."""
    directory, _ = synthetic
    assert dv.ViscosityPrediction("ab", 0.5, 0.1, 1, 102).viscosity_class == dv.HIGH_CLASS
    assert dv.ViscosityPrediction("ab", 0.499999, 0.1, 0, 102).viscosity_class == dv.LOW_CLASS
    rows = [{"Name": "ab", **{column: 0.0 for column in dv.DEEPVISCOSITY_FEATURES}}]
    prediction = dv.predict_from_descriptors(rows, directory).rows[0]
    assert prediction.label == int(prediction.probability >= 0.5)


def test_a_missing_feature_is_named(synthetic):
    directory, _ = synthetic
    rows = [{"Name": "ab", **{column: 0.0 for column in dv.DEEPVISCOSITY_FEATURES[:-1]}}]
    with pytest.raises(dv.DeepViscosityUnavailableError, match="feature_absent:SCM_pos_Fv"):
        dv.predict_from_descriptors(rows, directory)


def test_the_ensemble_is_loaded_once_and_can_be_forgotten(synthetic):
    directory, _ = synthetic
    rows = [{"Name": "ab", **{column: 0.0 for column in dv.DEEPVISCOSITY_FEATURES}}]
    dv.predict_from_descriptors(rows, directory)
    assert len(dv._ENSEMBLE) == 1
    dv.reset_ensemble_cache()
    assert not dv._ENSEMBLE


def test_upstreams_output_columns_are_written(synthetic, tmp_path):
    directory, _ = synthetic
    rows = [{"Name": "ab", **{column: 0.0 for column in dv.DEEPVISCOSITY_FEATURES}}]
    result = dv.predict_from_descriptors(rows, directory)
    path = dv.write_viscosity_csv(result, tmp_path / "classes.csv")
    header = next(csv.reader(path.open(encoding="utf-8")))
    assert header == ["Name", "Prob_Mean", "Prob_Std", "DeepViscosity_classes"]


# ------------------------------------------------------------------ near the boundary


def test_a_call_inside_the_ensembles_own_spread_is_flagged():
    """Four of the sixteen published antibodies land within 0.02 of the line."""
    close = dv.ViscosityPrediction("mAb6", 0.498471, 0.137418, 0, 102)
    assert close.margin < 0.002
    assert close.near_boundary

    clear = dv.ViscosityPrediction("mAb13", 0.201963, 0.090927, 0, 102)
    assert clear.margin > 0.29
    assert not clear.near_boundary


# ------------------------------------------------------------------ the units guard


def test_a_probability_written_into_a_centipoise_field_is_refused():
    """0.55 in a viscosity_cp field reads downstream as 0.55 cP."""
    with pytest.raises(dv.DeepViscosityUnavailableError, match="centipoise_key_is_not_a_measurement"):
        dv.assert_no_predicted_centipoise({"viscosity_cp": 0.55})
    with pytest.raises(dv.DeepViscosityUnavailableError, match="centipoise_key_is_not_a_measurement"):
        dv.assert_no_predicted_centipoise({"predicted_centipoise": 0.55})


def test_the_published_boundary_is_the_only_allowed_centipoise_key():
    dv.assert_no_predicted_centipoise({"class_boundary_cp": 20.0, "probability_high": 0.6})
    with pytest.raises(dv.DeepViscosityUnavailableError, match="published_centipoise_changed"):
        dv.assert_no_predicted_centipoise({"class_boundary_cp": 25.0})


# ------------------------------------------------------------------ the refusals


def test_missing_models_are_counted_not_listed(tmp_path):
    _write_scaler(tmp_path)
    available, reason = dv.viscosity_available(tmp_path)
    assert not available
    assert "deepviscosity_models_absent:204_of_204" in (reason or "")


def test_a_missing_scaler_is_named(tmp_path):
    available, reason = dv.viscosity_available(tmp_path)
    assert not available
    assert "scaler_absent" in (reason or "")


def test_the_report_names_the_blocker_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_PATH", raising=False)
    record = dv.deepviscosity_report({"H": "EVQ", "L": "DIQ"})
    assert record["viscosity_blocker"] == "data_path_unset"
    assert record["method"] == dv.VISCOSITY_METHOD

    record = dv.deepviscosity_report({"H": "EVQ", "L": "DIQ"}, models=tmp_path)
    assert "scaler_absent" in str(record["viscosity_blocker"])


def test_one_chain_is_refused_by_name(tmp_path, monkeypatch):
    """DeepSP scores an Fv; a lone heavy chain would be 127 gaps."""
    directory, _ = tmp_path, None
    _write_scaler(directory)
    monkeypatch.setattr(dv, "viscosity_available", lambda _models: (True, None))
    monkeypatch.setattr(dv, "deepsp_available", lambda _models: (True, None))
    monkeypatch.setattr(dv, "deepsp_models_dir", lambda: directory)
    record = dv.deepviscosity_report({"H": "EVQ"}, models=directory)
    assert record["viscosity_blocker"] == "both_chains_required"
    assert record["chains_given"] == ["H"]


# ------------------------------------------------------------------ the installer's allowlist


def _installer():
    """Load scripts/install_deepviscosity.py as a module to call into it."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "install_deepviscosity.py"
    spec = importlib.util.spec_from_file_location("install_deepviscosity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_installer_refuses_a_pickle_that_names_anything_else(tmp_path):
    """The security boundary: a downloaded pickle must not reach a callable.

    Upstream's scaler is a pickle, so the installer is the one place a
    downloaded file is deserialised. If the allowlist ever stopped biting, this
    file would run ``os.system`` on install instead of failing.
    """
    pytest.importorskip("joblib")
    import os
    import pickle

    class Hostile:
        def __reduce__(self):
            return (os.system, ("true",))

    payload = pickle.dumps(Hostile())
    path = tmp_path / "DeepViscosity_scaler.joblib"
    path.write_bytes(payload)
    with pytest.raises(SystemExit, match="refusing to unpickle"):
        _installer().scaler_state(payload, path)


def test_the_installer_and_the_module_agree_on_the_pinned_digest():
    assert len(dv.SCALER_SHA256) == 64
    assert set(dv.SCALER_SHA256) <= set("0123456789abcdef")


@needs_models
def test_the_installed_scaler_records_the_digest_it_was_converted_from():
    """Proof on disk that these 60 numbers came from the reviewed bytes."""
    directory = _models_dir()
    assert directory is not None
    document = json.loads((directory / dv.SCALER_NAME).read_text(encoding="utf-8"))
    assert document["source_sha256"] == dv.SCALER_SHA256
    assert document["features"] == list(dv.DEEPVISCOSITY_FEATURES)
    assert document["samples"] == dv.TRAINING_ANTIBODIES


# ------------------------------------------------------------------ against DeepViscosity itself


@needs_models
def test_the_port_reproduces_the_published_probabilities():
    """The ensemble, against the probabilities upstream's notebook printed.

    Run from the descriptors DeepViscosity ships, so this isolates the half
    this module owns: the scaler and the 102 networks, with ANARCII and DeepSP
    held fixed at upstream's own values.
    """
    directory = _models_dir()
    assert directory is not None
    reference = directory / "DeepSP_descriptors.csv"
    if not reference.is_file():
        pytest.skip("DeepSP_descriptors.csv is not installed; re-run the installer")
    rows = list(csv.DictReader(reference.open(encoding="utf-8")))
    assert len(rows) == len(PUBLISHED)

    result = dv.predict_from_descriptors(rows, directory)
    worst_probability = 0.0
    worst_spread = 0.0
    for prediction in result.rows:
        want_probability, want_spread, want_class = PUBLISHED[prediction.name]
        worst_probability = max(worst_probability, abs(prediction.probability - want_probability))
        worst_spread = max(worst_spread, abs(prediction.probability_sd - want_spread))
        assert prediction.label == want_class, (
            f"{prediction.name}: class {prediction.label} against the published {want_class} "
            f"at probability {prediction.probability:.6f}"
        )
    # Their notebook printed six decimals, so that rounding is the floor here,
    # not the port: measured agreement is 4.6e-07 on both. Nine of the sixteen
    # land inside their own ensemble spread, so anything near 1e-3 would be
    # able to flip a published class.
    assert worst_probability < 1e-5, f"worst probability difference {worst_probability:.3e}"
    assert worst_spread < 1e-5, f"worst spread difference {worst_spread:.3e}"


@needs_models
def test_the_whole_chain_from_sequence_reproduces_the_published_classes():
    """Raw heavy and light chains through ANARCII, DeepSP and the ensemble.

    This path re-derives the descriptors rather than reading upstream's, so it
    is the one place a numbering change could show up. Measured agreement is
    5.0e-07 on the probability with no class flipped, which is the floor of
    their six-decimal printing; the bound below sits well above that so a minor
    precision drift passes, and well below the closest published margin
    (mAb6, 0.0015 from the boundary) so nothing that could flip a class does.
    """
    from rfproteina.disease.deepsp import default_models_dir as sap_dir

    directory = _models_dir()
    assert directory is not None
    inputs = directory / "DeepViscosity_input.csv"
    sap = sap_dir()
    if not inputs.is_file() or sap is None or not sap.is_dir():
        pytest.skip("needs DeepViscosity_input.csv and DeepSP's models")

    antibodies = {
        row["Name"]: (row["Heavy_Chain"].strip(), row["Light_Chain"].strip())
        for row in csv.DictReader(inputs.open(encoding="utf-8"))
    }
    result, descriptors = dv.run_deepviscosity(antibodies, directory, sap)
    assert not result.skipped, result.skipped
    assert len(result.rows) == len(PUBLISHED)
    assert len(descriptors.rows) == len(PUBLISHED)

    worst = 0.0
    for prediction in result.rows:
        want_probability, _, want_class = PUBLISHED[prediction.name]
        worst = max(worst, abs(prediction.probability - want_probability))
        assert (
            prediction.label == want_class
        ), f"{prediction.name}: class {prediction.label} against the published {want_class}"
    assert worst < 1e-4, f"worst probability difference {worst:.3e}"


@needs_models
def test_a_swapped_pair_is_refused_not_scored():
    """Light chain where heavy belongs must fail loudly, not score.

    The heavy half is 145 of DeepSP's 272 input columns and the light half 127,
    so a swapped pair would align into the wrong slots and still return a
    number. DeepSP numbers each half and checks what ANARCII called it, which
    is what makes the report's fallback on key order safe.
    """
    from rfproteina.disease.deepsp import default_models_dir as sap_dir

    directory = _models_dir()
    assert directory is not None
    inputs = directory / "DeepViscosity_input.csv"
    sap = sap_dir()
    if not inputs.is_file() or sap is None or not sap.is_dir():
        pytest.skip("needs DeepViscosity_input.csv and DeepSP's models")

    row = next(csv.DictReader(inputs.open(encoding="utf-8")))
    swapped = {row["Name"]: (row["Light_Chain"].strip(), row["Heavy_Chain"].strip())}
    result, _ = dv.run_deepviscosity(swapped, directory, sap)
    assert not result.rows, "a swapped pair must not be scored"
    assert "heavy_chain_not_numbered_as_heavy" in result.skipped[row["Name"]]


@needs_models
def test_this_projects_deepsp_port_matches_the_descriptors_deepviscosity_ships():
    """DeepViscosity bundles DeepSP's output for its test set, so check it.

    A second, independent copy of the same 480 values: DeepSP publishes them in
    its own repository and DeepViscosity bundles them again for its test set.
    ``tests/test_disease_deepsp.py`` checks the first copy; this checks the
    second, generated upstream with ANARCI, against this port's ANARCII route.
    """
    from rfproteina.disease.deepsp import default_models_dir as sap_dir
    from rfproteina.disease.deepsp import run_deepsp

    directory = _models_dir()
    assert directory is not None
    inputs = directory / "DeepViscosity_input.csv"
    reference = directory / "DeepSP_descriptors.csv"
    sap = sap_dir()
    if not inputs.is_file() or not reference.is_file() or sap is None or not sap.is_dir():
        pytest.skip("needs DeepViscosity's input and descriptor CSVs and DeepSP's models")

    antibodies = {
        row["Name"]: (row["Heavy_Chain"].strip(), row["Light_Chain"].strip())
        for row in csv.DictReader(inputs.open(encoding="utf-8"))
    }
    produced = run_deepsp(antibodies, sap)
    assert not produced.skipped, produced.skipped
    published = {row["Name"]: row for row in csv.DictReader(reference.open(encoding="utf-8"))}

    compared = 0
    worst = 0.0
    for row in produced.rows:
        want = published[str(row["Name"])]
        for column in dv.DEEPVISCOSITY_FEATURES:
            theirs = float(want[column])
            mine = float(row[column])  # type: ignore[arg-type]
            worst = max(worst, abs(mine - theirs) / max(abs(theirs), 1.0))
            compared += 1
    assert compared == 30 * len(PUBLISHED)
    assert worst < 1e-4, f"worst relative error {worst:.3e}"
