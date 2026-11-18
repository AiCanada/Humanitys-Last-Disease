#!/usr/bin/env python3
"""Install ThermoMPNN and its default checkpoint under ``DATA_PATH/thermompnn``.

The weights are ``models/thermoMPNN_default.pt`` in the MIT repository
https://github.com/Kuhlman-Lab/ThermoMPNN. A shallow clone brings that file
and the ProteinMPNN weights the model loads beside it. The Python packages go
in a separate virtual environment. The clone's own lockfile selects Python
3.10 and the CPU build of torch 2.5.1. A newer torch defaults ``torch.load``
to ``weights_only=True`` and refuses these checkpoints.

    python scripts/install_thermompnn.py
    python scripts/install_thermompnn.py --have

``stability.py`` reads ``DATA_PATH/thermompnn/ThermoMPNN/models/thermoMPNN_default.pt``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

_REPO = "https://github.com/Kuhlman-Lab/ThermoMPNN.git"
_CHECKPOINT = "thermoMPNN_default.pt"


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


def _root() -> Path:
    return Path(os.environ["DATA_PATH"]) / "thermompnn"


def _have(repo: Path) -> int:
    checkpoint = repo / "models" / _CHECKPOINT
    vanilla = repo / "vanilla_model_weights" / "v_48_020.pt"
    print(f"checkpoint  {checkpoint}")
    print(f"status      {'present' if checkpoint.is_file() else 'absent'}")
    if checkpoint.is_file():
        print(f"bytes       {checkpoint.stat().st_size}")
    print(f"proteinmpnn {'present' if vanilla.is_file() else 'absent'}")
    return 0 if checkpoint.is_file() and vanilla.is_file() else 1


def _venv_python(venv: Path) -> str:
    """The interpreter version recorded in ``pyvenv.cfg``, or an empty string."""
    cfg = venv / "pyvenv.cfg"
    if not cfg.is_file():
        return ""
    found = ""
    for line in cfg.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "version_info":
            return value.strip()
        if sep and key.strip() == "version":
            found = value.strip()
    return found


# The same model block ``analysis/custom_inference.py`` builds before it loads
# the checkpoint. Kept here so the installer proves the weights unpickle.
_LOAD_CHECK = r"""
import os
import sys

root = os.environ["THERMOMPNN_REPO"]
sys.path[:0] = [os.path.join(root, "analysis"), root]
from omegaconf import OmegaConf
from thermompnn_benchmarking import get_trained_model

cfg = OmegaConf.load(os.path.join(root, "local.yaml"))
config = {
    "training": {
        "num_workers": 8,
        "learn_rate": 0.001,
        "epochs": 100,
        "lr_schedule": True,
    },
    "model": {
        "hidden_dims": [64, 32],
        "subtract_mut": True,
        "num_final_layers": 2,
        "freeze_weights": True,
        "load_pretrained": True,
        "lightattn": True,
        "lr_schedule": True,
    },
}
merged = OmegaConf.merge(cfg, OmegaConf.create(config))
merged.platform.thermompnn_dir = root
model = get_trained_model(
    model_name=os.path.join(root, "models", "thermoMPNN_default.pt"),
    config=merged,
    override_custom=True,
)
print(f"loaded {type(model).__name__}")
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--have", action="store_true", help="report the checkpoint, then stop")
    args = parser.parse_args(argv)
    _ensure_data_path()
    root = _root()
    repo = root / "ThermoMPNN"
    if args.have:
        return _have(repo)
    root.mkdir(parents=True, exist_ok=True)
    if not (repo / "models" / _CHECKPOINT).is_file():
        if repo.exists():
            shutil.rmtree(repo)
        print(f"cloning {_REPO}")
        subprocess.run(["git", "clone", "--depth", "1", _REPO, str(repo)], check=True)
    checkpoint = repo / "models" / _CHECKPOINT
    vanilla = repo / "vanilla_model_weights" / "v_48_020.pt"
    if not checkpoint.is_file() or not vanilla.is_file():
        raise SystemExit(f"Clone finished but {checkpoint.name} or v_48_020.pt is absent.")
    venv = root / "venv"
    recorded = _venv_python(venv)
    if venv.exists() and not recorded.startswith("3.10"):
        print(f"replacing virtual environment (found Python {recorded or 'unknown'})")
        shutil.rmtree(venv)
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    env["UV_LINK_MODE"] = "copy"
    print("installing the locked CPU stack (Python 3.10, torch 2.5.1)")
    subprocess.run(
        ["uv", "sync", "--frozen", "--extra", "cpu", "--python", "3.10"],
        cwd=repo,
        env=env,
        check=True,
    )
    python = venv / "bin" / "python"
    if not python.is_file():
        python = venv / "Scripts" / "python.exe"
    env["THERMOMPNN_REPO"] = str(repo)
    print("loading the checkpoint")
    subprocess.run([str(python), "-c", _LOAD_CHECK], env=env, check=True)
    print(f"ready  {checkpoint}  ({checkpoint.stat().st_size} bytes)")
    print(f"ready  {vanilla}  ({vanilla.stat().st_size} bytes)")
    print(f"ready  {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
