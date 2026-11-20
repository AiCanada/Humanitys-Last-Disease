"""Run the vendored RBase generator under its own interpreter.

Executed as a script, never imported by the package: the generator needs
mdtraj, ml-collections and dm-tree, which this project's environment does not
carry, so it runs under whichever interpreter the installer built. Nothing here
imports ``rfproteina`` -- ``tests/test_screening_vendoring.py`` asserts that,
because an import would tie the tool's environment to this one and defeat the
point of running it elsewhere.

Run with ``-I``. That keeps this file's own directory off ``sys.path``, so a
module sitting beside it cannot shadow a standard-library name the generator
imports, and it ignores ``PYTHONPATH`` -- which is why the vendored tree is put
on the path here, explicitly, rather than through the environment.

Three things this does that the generator would otherwise do by itself, and
must not:

``--check-only``
    Report what is installed and exit. The screen asks this before it asks for
    an ensemble, so a missing piece is a named decline rather than a download.

cache redirection
    ``CachePaths`` derives every path it uses -- the CUDA clone, the SO(3)
    tables, the alignment cache, the folding parameters -- from one root. That
    root is set to the weights directory, so a run writes nothing into the
    checkout and nothing beside the vendored source.

no network
    The generator's ``check_and_install_dependencies`` clones a CUDA library,
    and its representation loader queries a public MSA server. Both are
    refused here: each is checked for, and a missing one exits with the name
    the screen reports instead of being fetched. A design's sequence does not
    leave the machine because somebody pressed a button.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Printed as the last line of a successful run, so the caller parses one line
#: rather than the generator's own logging.
RESULT_PREFIX = "CONFROVER_RESULT "
#: The pieces of the install, and the name reported when one is absent. Kept in
#: step with ``dynamics.REQUIRED_PAYLOAD`` by a test.
PAYLOAD = {
    "weights": "weights_not_installed",
    "cutlass": "cutlass_not_installed",
    "openfold_params": "openfold_params_not_installed",
    "msa": "msa_not_cached",
}


def installed(path: Path) -> bool:
    """Whether a piece of the install is really there, not just its directory.

    Emptiness counts as absent. An interrupted install leaves behind the
    directory it was writing into, and a bare ``exists()`` read that as done.
    Kept in step with ``dynamics._installed`` by a test.
    """
    if path.is_file():
        return True
    return path.is_dir() and any(path.iterdir())


def payload_state(weights_root: Path) -> dict[str, bool]:
    """Which pieces of the install are present under the weights root."""
    return {name: installed(weights_root / name) for name in PAYLOAD}


def first_missing(weights_root: Path) -> str | None:
    """The name the screen reports for the first absent piece, or None."""
    if not weights_root.is_dir():
        return PAYLOAD["weights"]
    for name, blocker in PAYLOAD.items():
        if not installed(weights_root / name):
            return blocker
    return None


def checkpoint_for(weights_root: Path, task_mode: str) -> Path | None:
    """The checkpoint a task mode needs, or None when it is not there.

    Interpolation has its own weights and the base model refuses that task, so
    the mode decides the file rather than a caller naming it.
    """
    name = "confrover_interp_20m_v1_0.pt" if task_mode == "interp" else "confrover_base_20m_v1_0.pt"
    found = weights_root / "weights" / name
    return found if found.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an RBase ensemble for one chain.")
    parser.add_argument("--code", required=True, type=Path, help="the vendored tree's parent")
    parser.add_argument("--weights", required=True, type=Path, help="DATA_PATH/confrover")
    parser.add_argument("--design", type=Path, help="the starting conformation")
    parser.add_argument("--seqres", default="", help="the chain's amino acid sequence")
    parser.add_argument("--case-id", default="design", help="a name for the output")
    parser.add_argument("--task", default="forward", choices=("forward", "iid", "interp"))
    parser.add_argument("--out", type=Path, help="where the conformers are written")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--stride-in-10ps", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--end", type=Path, help="the ending conformation, for interp")
    parser.add_argument("--check-only", action="store_true", help="report the install and exit")
    args = parser.parse_args(argv)

    weights_root = Path(args.weights)
    if args.check_only:
        state = payload_state(weights_root)
        base = checkpoint_for(weights_root, "forward")
        interp = checkpoint_for(weights_root, "interp")
        importable, why = _importable(Path(args.code))
        print(
            RESULT_PREFIX
            + json.dumps(
                {
                    "payload": state,
                    "missing": first_missing(weights_root),
                    "base_checkpoint": str(base) if base else None,
                    "interp_checkpoint": str(interp) if interp else None,
                    "importable": importable,
                    "import_error": why,
                }
            )
        )
        return 0

    missing = first_missing(weights_root)
    if missing is not None:
        # Every one of these the generator would fetch or build on first call.
        # Exiting with the name is the whole reason this wrapper exists.
        print(f"{RESULT_PREFIX}{json.dumps({'blocker': missing})}")
        return 1

    checkpoint = checkpoint_for(weights_root, args.task)
    if checkpoint is None:
        print(f"{RESULT_PREFIX}{json.dumps({'blocker': 'weights_not_installed'})}")
        return 1
    if args.task != "iid" and args.design is None:
        print(f"{RESULT_PREFIX}{json.dumps({'blocker': 'conditions_required'})}")
        return 1
    if args.task == "interp" and args.end is None:
        print(f"{RESULT_PREFIX}{json.dumps({'blocker': 'conditions_required'})}")
        return 1
    if not args.seqres.strip():
        print(f"{RESULT_PREFIX}{json.dumps({'blocker': 'design_unreadable'})}")
        return 1
    if args.out is None:
        print(f"{RESULT_PREFIX}{json.dumps({'blocker': 'generator_failed'})}")
        return 1

    sys.path.insert(0, str(Path(args.code)))
    try:
        from rbase.model import RBase
    except Exception as error:
        print(
            f"{RESULT_PREFIX}"
            f"{json.dumps({'blocker': 'confrover_not_installed', 'detail': f'{type(error).__name__}: {error}'})}"
        )
        return 1

    conditions: object = None
    if args.task == "forward":
        conditions = str(args.design)
    elif args.task == "interp":
        conditions = [str(args.design), str(args.end)]

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    try:
        model = RBase.from_pretrained(str(checkpoint))
        model.eval()
        # One root for everything the generator caches, so nothing is written
        # beside the vendored source or into the checkout.
        model.generate(
            case_id=args.case_id,
            seqres=args.seqres.strip(),
            task_mode=args.task,
            output_dir=str(output),
            conditions=conditions,
            n_frames=args.frames if args.task != "iid" else None,
            stride_in_10ps=args.stride_in_10ps if args.task != "iid" else None,
            n_replicates=args.replicates,
            cache_dir=str(weights_root),
            msa_root=str(weights_root / "msa"),
            folding_repr=str(weights_root / "folding_repr"),
            seed=args.seed,
            diffusion_steps=args.diffusion_steps,
        )
    except Exception as error:
        print(
            f"{RESULT_PREFIX}"
            f"{json.dumps({'blocker': 'generator_failed', 'detail': f'{type(error).__name__}: {error}'})}"
        )
        return 1

    written = sorted(str(p) for p in output.rglob("*.pdb"))
    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "output_dir": str(output),
                "frames": written,
                "task_mode": args.task,
                "checkpoint": str(checkpoint),
            }
        )
    )
    return 0


def _importable(code: Path) -> tuple[bool, str]:
    """Whether this interpreter can import the vendored package, and why not."""
    sys.path.insert(0, str(code))
    try:
        import rbase.model  # noqa: F401

        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


if __name__ == "__main__":
    raise SystemExit(main())
