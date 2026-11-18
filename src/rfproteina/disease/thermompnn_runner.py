"""Site-saturation ddG inference for ThermoMPNN, with its code and weights apart.

Upstream's own entry point, ``analysis/custom_inference.py``, resolves the
ProteinMPNN weights it builds the model around relative to its own location:
``transfer_model.get_protein_mpnn`` reads
``cfg.platform.thermompnn_dir/vanilla_model_weights``, and upstream's ``main``
sets that key to the directory holding the script. That is right for a clone
where code and weights sit in one tree, and wrong here, where the code is
vendored in the repository and the weights stay under ``DATA_PATH``.

This script bridges the two. It puts the vendored tree on ``sys.path``, points
``platform.thermompnn_dir`` at the weights root, then calls upstream's ``main``,
so the inference itself -- the SSM mutation list, the model call, the CSV it
writes -- stays upstream's code rather than a reimplementation of it that could
drift from the convention :mod:`rfproteina.disease.stability` reads.

Run by the ThermoMPNN virtual environment's interpreter (Python 3.10, torch
2.5.1), which has neither this package nor its dependencies installed. Nothing
here imports ``rfproteina``, and ``rfproteina`` never imports this: it is a
script that happens to be version-controlled beside its caller.

Pass ``-I``. That keeps the directory this file sits in off ``sys.path``, where
it would otherwise shadow upstream's ``datasets``, ``cache`` and ``model_utils``
with same-named modules of this package's own::

    python -I thermompnn_runner.py --code <vendored tree> --weights <data root> \
        --checkpoint <ckpt> --pdb <design.pdb> --chain A --out-dir <dir>

:func:`rfproteina.disease.stability._run_thermompnn` is the only caller.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


def _place_upstream_on_path(code: Path) -> None:
    """Prepend the vendored tree's two import roots.

    ``analysis`` first because that is where upstream's scripts import each
    other from (``from SSM import ...``), relying on it being ``sys.path[0]``
    as the script's own directory. Under ``-I`` it is not added, so it is added
    here.
    """
    sys.path[:0] = [str((code / "analysis").resolve()), str(code.resolve())]


def _point_proteinmpnn_at(weights: Path) -> None:
    """Redirect ``get_protein_mpnn`` to load ProteinMPNN's weights from ``weights``.

    Wrapping that one function, rather than editing the vendored file, is what
    keeps every vendored file byte-identical to the upstream commit
    ``manifest.toml`` pins -- which is the property that makes the tree
    checkable against upstream at all. The wrapper rewrites one config key and
    delegates; it does not reimplement the loader.

    Raises:
        SystemExit: If the function is absent, which means the vendored tree is
            not the pinned commit. Named loudly here rather than surfacing as a
            weight file "not found" under a path nobody configured.
    """
    import transfer_model
    from omegaconf import OmegaConf

    original = getattr(transfer_model, "get_protein_mpnn", None)
    if original is None:
        raise SystemExit(
            "transfer_model.get_protein_mpnn is absent, so the ProteinMPNN weight directory "
            "cannot be redirected at the weights root. The vendored ThermoMPNN tree is not "
            "the commit screening subpackage/thermompnn/manifest.toml pins."
        )
    root = OmegaConf.create({"platform": {"thermompnn_dir": str(weights)}})

    def with_weights_root(cfg: Any, *args: Any, **kwargs: Any) -> Any:
        return original(OmegaConf.merge(cfg, root), *args, **kwargs)

    transfer_model.get_protein_mpnn = with_weights_root


#: The model block upstream's ``main`` merges over ``local.yaml`` before it
#: loads a checkpoint. Needed only by ``--check-only``, which builds the model
#: without going through ``main``. Kept beside the redirection it checks.
_MODEL_CONFIG: dict[str, dict[str, object]] = {
    "training": {"num_workers": 8, "learn_rate": 0.001, "epochs": 100, "lr_schedule": True},
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


def _check_only(code: Path, weights: Path, checkpoint: Path) -> int:
    """Build the model from the two trees and report it, scanning nothing.

    What the installer runs to prove the split layout works: this is the one
    path that loads both weight files -- the ddG checkpoint by the path given,
    and ProteinMPNN's through the redirection above -- so a wrong weights root
    fails here, at install time, rather than on a design months later.
    """
    from omegaconf import OmegaConf
    from thermompnn_benchmarking import get_trained_model

    merged = OmegaConf.merge(OmegaConf.load(str(code / "local.yaml")), OmegaConf.create(_MODEL_CONFIG))
    merged.platform.thermompnn_dir = str(weights)
    model = get_trained_model(model_name=str(checkpoint), config=merged, override_custom=True)
    print(f"loaded {type(model).__name__} from {checkpoint.name} with weights under {weights}")
    return 0


def _parse(argv: list[str] | None) -> Namespace:
    parser = argparse.ArgumentParser(description="ThermoMPNN site-saturation scan of one structure.")
    parser.add_argument("--code", required=True, type=Path, help="the vendored ThermoMPNN tree")
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="directory holding models/ and vanilla_model_weights/ (under DATA_PATH)",
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="the ddG checkpoint to load")
    parser.add_argument("--pdb", type=Path, help="structure to scan; required unless --check-only")
    parser.add_argument("--chain", default="", help="chain to scan; empty selects the structure's first chain")
    parser.add_argument("--out-dir", type=Path, help="directory the inference CSV is written to")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="build the model from both trees, then stop without scanning",
    )
    args = parser.parse_args(argv)
    if not args.check_only:
        missing = [name for name in ("pdb", "out_dir") if getattr(args, name) is None]
        if missing:
            parser.error("--pdb and --out-dir are required unless --check-only is given")
    return args


def main(argv: list[str] | None = None) -> int:
    """Load the model from two trees and run upstream's inference over one structure."""
    args = _parse(argv)
    _place_upstream_on_path(args.code)
    _point_proteinmpnn_at(args.weights)

    import torch
    from omegaconf import OmegaConf

    if args.check_only:
        with torch.no_grad():
            return _check_only(args.code, args.weights, args.checkpoint)

    import custom_inference

    # Upstream's main reloads this itself; passed for its signature. Its values
    # are the upstream authors' own cluster paths and none of them is read on
    # the inference path: platform.thermompnn_dir is overwritten by main (code
    # root) and then by the wrapper above (weights root), and data_loc is for
    # the training datasets, which are not vendored.
    cfg = OmegaConf.load(str(args.code / "local.yaml"))
    upstream = Namespace(
        pdb=str(args.pdb),
        chain=args.chain,
        model_path=str(args.checkpoint),
        out_dir=str(args.out_dir),
    )
    with torch.no_grad():
        custom_inference.main(cfg, upstream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
