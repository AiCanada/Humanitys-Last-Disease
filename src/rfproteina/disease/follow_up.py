"""Build the next local sample after a finished screen.

An interface whose sequence failed is inverse-folded. A measured contact surface
of zero gets more replicates of the original PPI spec. An unmeasured surface, or
an interface whose sequence already passed, builds no command. Nothing here
starts ``rfd4-generate``: these are argv lists for a caller to print, write down
or run, which is the same division every sibling module keeps.

:py:func:`follow_up_commands` is the one to call. It reads the per-sample list,
because the top-level fields a screen writes repeat its first CIF only: a run
whose first sample passed and whose fourth and seventh failed produced no
follow-up at all while this read that top-level field.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rfproteina.disease.design_spec import write_inverse_folding_spec
from rfproteina.disease.screen import measured_contact_ms
from rfproteina.disease.shards import ORIGINAL_EXPERIMENT

_REPLICATES = 8


def follow_up_command(
    screen: Mapping[str, object],
    *,
    generation_dir: Path,
    generated_cif: Path,
    original_spec: Path,
    fresh_dir: Path,
    designed_chain: str,
    host: str = "local",
    failed: bool | None = None,
    contact_ms: float | None = None,
) -> list[str] | None:
    """Argv for one follow-up, or None when this sample is not due one.

    ``fresh_dir`` is the output directory. It must be a new folder, so the
    finished generation and its condition spec stay where they are.

    ``failed`` says whether the sample being re-folded failed its sequence. A
    caller working one sample at a time passes it, because asking the screen
    would answer for the whole run. Left None, any failed sample counts.

    ``contact_ms`` is that sample's own surface. The top-level field repeats
    the first CIF, so a run of several samples passes the one it is judging.
    """
    contact = measured_contact_ms(screen) if contact_ms is None else contact_ms
    if contact is None or contact < 0:
        return None
    _require_fresh(fresh_dir, generation_dir)
    name = host.strip()
    if not name:
        raise ValueError("The follow-up needs a host.")
    if contact > 0:
        if not (_sequence_failed(screen) if failed is None else failed):
            return None
        if not generated_cif.is_file():
            raise ValueError(f"{generated_cif.name} is not a generated structure.")
        spec = write_inverse_folding_spec(generated_cif, fresh_dir, designed_chain=designed_chain)
        if spec.resolve() == original_spec.resolve():
            raise ValueError("The inverse-folding spec would overwrite the original PPI spec.")
        return _command(spec, fresh_dir, name, inverse=True)
    if not original_spec.is_file():
        raise ValueError(f"{original_spec.name} is not a condition spec.")
    return _command(original_spec, fresh_dir, name, inverse=False)


def follow_up_commands(
    screen: Mapping[str, object],
    *,
    generation_dir: Path,
    generated_cifs: Mapping[str, Path],
    original_spec: Path,
    fresh_root: Path,
    designed_chain: str,
    host: str = "local",
) -> list[list[str]]:
    """One command per sample that needs one, in listed order.

    ``generated_cifs`` maps a screen's example id to its file, which is what
    ``screen.sample_cifs`` returns, so the sample a command re-folds is the
    sample the screen judged.

    Each failed sample gets its own directory under ``fresh_root``, because two
    inverse-folding runs writing one output directory would overwrite each
    other's designs. A sample whose own surface is zero asks for one more draw
    of the original spec, once for the run. A sample whose surface was not
    measured is skipped, and does not decide the others.
    """
    samples = screen.get("samples")
    if not isinstance(samples, list):
        return []
    found: list[list[str]] = []
    sent_replicates = False
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        contact = _sample_contact(sample.get("contact_ms"))
        if contact is None or contact < 0:
            continue
        if contact == 0:
            if sent_replicates:
                continue
            command = follow_up_command(
                screen,
                generation_dir=generation_dir,
                generated_cif=original_spec,
                original_spec=original_spec,
                fresh_dir=fresh_root,
                designed_chain=designed_chain,
                host=host,
                contact_ms=0.0,
            )
            if command:
                found.append(command)
            sent_replicates = True
            continue
        example_id = sample.get("example_id")
        failed = sample.get("sequence_pass") is False
        if not failed or not isinstance(example_id, str) or not example_id:
            continue
        generated = generated_cifs.get(example_id)
        if generated is None or not generated.is_file():
            continue
        command = follow_up_command(
            screen,
            generation_dir=generation_dir,
            generated_cif=generated,
            original_spec=original_spec,
            fresh_dir=fresh_root / example_id,
            designed_chain=designed_chain,
            host=host,
            failed=True,
            contact_ms=contact,
        )
        if command is not None:
            found.append(command)
    return found


def _sample_contact(value: object) -> float | None:
    """One sample's contact, rounded the way the screen stores it.

    ``None`` was not measured. A boolean is not a surface.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return float(f"{number:.4f}")


def _failed_samples(screen: Mapping[str, object]) -> list[str]:
    """Example ids whose sequence did not pass, in listed order.

    The per-sample list is the authority. The top-level field repeats the first
    CIF, so reading it missed every run whose first sample happened to pass.
    """
    samples = screen.get("samples")
    if not isinstance(samples, list):
        return []
    found: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        example_id = sample.get("example_id")
        if isinstance(example_id, str) and example_id and sample.get("sequence_pass") is False:
            found.append(example_id)
    return found


def _sequence_failed(screen: Mapping[str, object]) -> bool:
    """Whether any sample's sequence failed.

    Any sample, not the first one: the top-level ``sequence_developability``
    block repeats the first CIF, so a run whose first sample passed reported
    nothing to follow up even when later samples failed.
    """
    if _failed_samples(screen):
        return True
    sequence = screen.get("sequence_developability")
    if not isinstance(sequence, dict):
        return False
    return sequence.get("sequence_pass") is False


def _require_fresh(fresh_dir: Path, generation_dir: Path) -> None:
    fresh = fresh_dir.resolve()
    live = generation_dir.resolve()
    if fresh == live or live in fresh.parents or fresh in live.parents:
        raise ValueError("The follow-up needs a fresh output directory.")


def _command(spec: Path, fresh_dir: Path, host: str, *, inverse: bool) -> list[str]:
    command = [
        "rfd4-generate",
        f"inference_experiment={ORIGINAL_EXPERIMENT}",
        f"inputs={spec.as_posix()}",
        f"out_dir={fresh_dir.as_posix()}",
        f"+disease.host={host}",
    ]
    if inverse:
        command.append("generation=inverse_folding")
        command.append("recycle_strategy=medium/flat")
    else:
        command.append("recycle_strategy=medium/front_loaded")
    command.append(f"num_replicates={_REPLICATES}")
    command.append("batch_size=1")
    return command
