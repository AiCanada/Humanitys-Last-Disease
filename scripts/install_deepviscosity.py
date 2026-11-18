#!/usr/bin/env python3
"""Download the MIT DeepViscosity ensemble into ``DATA_PATH/deepviscosity/models``.

DeepViscosity classifies a monoclonal antibody's viscosity at 150 mg/mL as low
(<= 20 cP) or high (> 20 cP) from its Fv sequence alone, by reading DeepSP's 30
spatial descriptors. The runner is ``rfproteina.disease.deepviscosity`` and it
needs no TensorFlow: the dense stack is evaluated in torch, checked against the
probabilities upstream's own predictor notebook published.

About 8.6 MB over 204 files -- 102 architecture/weight pairs, one per
leave-one-group-out split of the training set -- plus the scaler.

**DeepSP's models are also required** and are not fetched here: they are the
same three files ``scripts/install_deepsp.py`` already installs, and one copy
with one source beats two that can drift. Run that first, or this script will
tell you to.

The scaler needs a word. Upstream ships it as an sklearn ``StandardScaler``
inside a joblib pickle, and unpickling a downloaded file runs whatever that file
says. So it is handled once, here, under three constraints: the bytes must match
a pinned SHA-256; the unpickler's allowlist can reach no callable an attacker
could choose; and what lands on disk is plain JSON. The library never opens a
pickle, and predicting needs neither joblib nor scikit-learn.

    python scripts/install_deepviscosity.py
    python scripts/install_deepviscosity.py --have
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfproteina.disease.deepsp import deepsp_available
from rfproteina.disease.deepsp import default_models_dir as deepsp_models_dir
from rfproteina.disease.deepviscosity import (
    DEEPVISCOSITY_FEATURES,
    ENSEMBLE_SIZE,
    FEATURE_COUNT,
    MODEL_STEM,
    SCALER_NAME,
    SCALER_SHA256,
    SCALER_SOURCE_NAME,
    VISCOSITY_SUBDIR,
    model_names,
)

#: Pinned to a branch rather than a tag because DeepViscosity publishes none.
_BASE = "https://raw.githubusercontent.com/Lailabcode/DeepViscosity/main"
_MODEL_DIR = "DeepViscosity_ANN_ensemble_models"
_SCALER_DIR = "DeepViscosity_scaler"
#: The first eight bytes of any HDF5 file. Checked because a redirect to an
#: error page is still a 200 with a body, and a file that looks installed and
#: fails at load is worse than one that was never written.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_SMALLEST = 512

#: Upstream's own test set and the descriptors it produced for it: 16
#: antibodies, about 9 KB. Not needed to predict, which is why
#: ``viscosity_available`` does not ask for them -- but they are what
#: ``tests/test_disease_deepviscosity.py`` checks the port against, and a port
#: nobody checked against the original is a guess with a confident interface.
_REFERENCE = ("DeepViscosity_input.csv", "DeepSP_descriptors.csv")


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _ensure_data_path() -> None:
    if os.environ.get("DATA_PATH", "").strip():
        return
    env_file = Path(__file__).resolve().parent.parent / ".env.local"
    if not env_file.is_file():
        raise SystemExit("DATA_PATH is unset and .env.local does not name one.")
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "DATA_PATH":
            os.environ["DATA_PATH"] = _unquote(value)
            return
    raise SystemExit("DATA_PATH is unset and .env.local does not name one.")


def _destination() -> Path:
    return Path(os.environ["DATA_PATH"]) / VISCOSITY_SUBDIR


def scaler_state(payload: bytes, path: Path) -> dict[str, object]:
    """Read upstream's joblib pickle without reaching any callable it names.

    joblib writes each array's bytes raw into the pickle stream, so a plain
    ``pickle.load`` cannot walk past them -- joblib's own
    ``NumpyUnpickler.load_build`` does. Subclassing it keeps that array reader
    while ``find_class`` refuses every global but the handful this file
    legitimately needs, so ``StandardScaler`` is answered with a class that
    only stores state and scikit-learn is never imported.
    """
    try:
        import numpy as np
        from joblib.numpy_pickle import NumpyUnpickler
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"converting the scaler once needs joblib and numpy ({error.name} is missing). "
            "Install the ligand-regressor extra, or any environment with joblib, and re-run."
        ) from error

    multiarray = getattr(np, "_core", np.core).multiarray

    class Held:
        """Stands in for sklearn's StandardScaler: stores state, runs nothing."""

        def __setstate__(self, state: dict[str, object]) -> None:
            self.__dict__.update(state)

    allowed = {
        ("sklearn.preprocessing._data", "StandardScaler"): Held,
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy.core.multiarray", "scalar"): multiarray.scalar,
        ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct,
        ("numpy._core.multiarray", "scalar"): multiarray.scalar,
        ("numpy._core.multiarray", "_reconstruct"): multiarray._reconstruct,
    }

    class Restricted(NumpyUnpickler):
        def find_class(self, module: str, name: str):  # noqa: ANN202 - pickle's own signature
            if (module, name) in allowed:
                return allowed[(module, name)]
            if module == "joblib.numpy_pickle":
                # joblib's own array wrappers, which is how the payload is read.
                return super().find_class(module, name)
            raise SystemExit(
                f"refusing to unpickle {module}.{name} from {path.name}: it is not one of the "
                "globals a StandardScaler needs. The file upstream publishes has changed shape."
            )

    with path.open("rb") as handle:
        names = set(inspect.signature(NumpyUnpickler.__init__).parameters)
        if "ensure_native_byte_order" in names:
            held = Restricted(str(path), handle, ensure_native_byte_order=True).load()
        else:  # pragma: no cover - older joblib
            held = Restricted(str(path), handle).load()

    state = held.__dict__
    for key in ("mean_", "scale_", "n_features_in_"):
        if key not in state:
            raise SystemExit(f"{path.name} carries no {key}; it is not the scaler this model needs.")
    if int(state["n_features_in_"]) != FEATURE_COUNT:
        raise SystemExit(f"{path.name} scales {state['n_features_in_']} features, not {FEATURE_COUNT}.")
    if not state.get("with_mean", True) or not state.get("with_std", True):
        # A scaler that centres but does not divide, or the reverse, is a
        # different transform; applying ours would shift every descriptor.
        raise SystemExit(
            f"{path.name} was fitted with with_mean={state.get('with_mean')} "
            f"and with_std={state.get('with_std')}, which is a different transform."
        )
    mean = [float(value) for value in state["mean_"]]
    scale = [float(value) for value in state["scale_"]]
    if len(mean) != FEATURE_COUNT or len(scale) != FEATURE_COUNT:
        raise SystemExit(f"{path.name} holds {len(mean)} means and {len(scale)} scales, not {FEATURE_COUNT} of each.")
    if any(value <= 0.0 for value in scale):
        raise SystemExit(f"{path.name} holds a non-positive scale, which no StandardScaler produces.")
    return {
        "mean": mean,
        "scale": scale,
        "samples": int(state.get("n_samples_seen_", 0)),
        "sklearn_version": str(state.get("_sklearn_version", "")),
        "features": list(DEEPVISCOSITY_FEATURES),
        "source": SCALER_SOURCE_NAME,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "note": (
            "Converted from upstream's joblib pickle by scripts/install_deepviscosity.py so that "
            "rfproteina.disease.deepviscosity never unpickles. The transform is (x - mean) / scale, "
            "column by column, in the feature order above."
        ),
    }


def _fetch(url: str, timeout: int = 300) -> bytes:
    # The URL is built from _BASE above, not from anything a caller passes.
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def _check(name: str, payload: bytes) -> None:
    if len(payload) < _SMALLEST:
        raise SystemExit(f"{name} came back as {len(payload)} bytes, which is not a model.")
    if name.endswith(".h5") and not payload.startswith(_HDF5_MAGIC):
        raise SystemExit(f"{name} does not start with the HDF5 signature; it is not a weight file.")
    if name.endswith(".json"):
        try:
            json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"{name} is not readable JSON: {type(error).__name__}.") from error


def _install_scaler(destination: Path, force: bool) -> bool:
    target = destination / SCALER_NAME
    if target.is_file() and not force:
        print(f"have     {SCALER_NAME}")
        return False
    url = f"{_BASE}/{_SCALER_DIR}/{SCALER_SOURCE_NAME}"
    print(f"fetching {url}")
    payload = _fetch(url)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SCALER_SHA256:
        raise SystemExit(
            f"{SCALER_SOURCE_NAME} hashes to {digest}, not the pinned {SCALER_SHA256}.\n"
            "This file is a pickle, so it is not read unless its bytes are the ones that were "
            "reviewed. If upstream published a new scaler, check it and update SCALER_SHA256 in "
            "src/rfproteina/disease/deepviscosity.py -- every prediction depends on these numbers."
        )
    source = destination / SCALER_SOURCE_NAME
    source.write_bytes(payload)
    try:
        state = scaler_state(payload, source)
    finally:
        # The pickle itself is not left lying in the model directory: nothing
        # reads it again, and a stray pickle beside the weights invites one.
        source.unlink(missing_ok=True)
    target.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"  {len(payload):>9,} B  sha256 {digest[:16]}  ->  {SCALER_NAME} ({state['samples']} samples)")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--have", action="store_true", help="report what is installed, then stop")
    parser.add_argument("--force", action="store_true", help="fetch even where a file is already present")
    args = parser.parse_args(argv)
    _ensure_data_path()
    destination = _destination()
    names = model_names()

    if args.have:
        present = [name for name in names if (destination / name).is_file()]
        scaler = (destination / SCALER_NAME).is_file()
        print(f"models    {destination}")
        print(f"ensemble  {len(present)} of {len(names)} files ({ENSEMBLE_SIZE} pairs)")
        print(f"scaler    {SCALER_NAME}: {'present' if scaler else 'absent'}")
        for name in _REFERENCE:
            print(f"reference {name}: {'present' if (destination / name).is_file() else 'absent'}")
        sap = deepsp_models_dir()
        if sap is None:
            print("deepsp    DATA_PATH unset")
        else:
            ready, why = deepsp_available(sap)
            print(f"deepsp    {'ready' if ready else why}  ({sap})")
        if present and len(present) < len(names):
            missing = [name for name in names if not (destination / name).is_file()]
            print(f"missing   {len(missing)}, first few: {', '.join(missing[:4])}")
        return 0 if scaler and len(present) == len(names) else 1

    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    for position in range(ENSEMBLE_SIZE):
        stem = MODEL_STEM.format(index=position)
        for name in (f"{stem}.json", f"{stem}.h5"):
            path = destination / name
            if path.is_file() and not args.force:
                continue
            payload = _fetch(f"{_BASE}/{_MODEL_DIR}/{name}")
            _check(name, payload)
            path.write_bytes(payload)
            written += 1
        if (position + 1) % 20 == 0 or position + 1 == ENSEMBLE_SIZE:
            print(f"  {position + 1:>3} of {ENSEMBLE_SIZE} pairs")
    if _install_scaler(destination, args.force):
        written += 1

    for name in _REFERENCE:
        path = destination / name
        if path.is_file() and not args.force:
            print(f"have     {name}")
            continue
        payload = _fetch(f"{_BASE}/{name}")
        _check(name, payload)
        path.write_bytes(payload)
        written += 1
        print(f"  {len(payload):>9,} B  {name}")

    total = sum((destination / name).stat().st_size for name in names if (destination / name).is_file())
    total += (destination / SCALER_NAME).stat().st_size
    print(f"ready  {destination}  ({written} fetched, {total / 1024 / 1024:.1f} MB total)")
    sap = deepsp_models_dir()
    if sap is not None:
        ready, why = deepsp_available(sap)
        if not ready:
            print(f"note   DeepSP's models are also required and are {why}. Run scripts/install_deepsp.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
