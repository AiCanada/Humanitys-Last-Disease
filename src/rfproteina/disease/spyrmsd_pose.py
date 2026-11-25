"""Heavy-atom pose recovery for explicitly paired redocking controls in a common receptor frame."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from numbers import Real
from pathlib import Path, PureWindowsPath

import numpy as np

from rfproteina.disease.corpus_provenance import declare
from rfproteina.disease.pose_validation import PoseUnreadableError, read_nominated_pose

METHOD = "spyrmsd"
ENDPOINT = "redocking_pose_rmsd"
UNIT = "angstrom"
MANIFEST = "spyrmsd_controls.json"
_FIELDS = {"pose", "reference", "experimental_reference_id", "pose_frame", "reference_frame"}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127}")
_MAX_CONTROLS = 128
_MAX_HEAVY_ATOMS = 256
_MAX_MAPPINGS = 4096


class ControlError(ValueError):
    """A named blocker that can be returned without disclosing a submitted path or structure."""


@dataclass(frozen=True)
class Pose:
    identity: str
    coordinates: np.ndarray
    atom_labels: np.ndarray
    adjacency: np.ndarray
    sha256: str
    file_stat: os.stat_result


@dataclass(frozen=True)
class Control:
    pose: Pose
    reference: Pose
    reference_id: str
    frame: str


def installed() -> bool:
    from rdkit import Chem
    from rustworkx import graph_vf2_mapping
    from spyrmsd import get_backend, rmsd

    return bool(get_backend()) and all(callable(f) for f in (Chem.MolFromMolBlock, graph_vf2_mapping, rmsd.symmrmsd))


def _read(path: Path, limit: int, role: str) -> tuple[bytes, os.stat_result]:
    """Hash and parse one bounded snapshot of the nominated regular file."""
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ControlError(f"{role}_not_regular")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
                raise ControlError(f"{role}_changed")
            if opened.st_size > limit:
                raise ControlError(f"{role}_size_limit")
            data = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        if len(data) > limit:
            raise ControlError(f"{role}_size_limit")
        if len(data) != opened.st_size or opened.st_mtime_ns != after.st_mtime_ns:
            raise ControlError(f"{role}_changed")
    except FileNotFoundError as exc:
        raise ControlError(f"{role}_missing") from exc
    except OSError as exc:
        raise ControlError(f"{role}_unreadable") from exc
    return data, opened


def _unique(pairs: list[tuple[str, object]]) -> dict:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ControlError("duplicate_manifest_key")
    return fields


def _samples(directory: Path, example: str | None) -> dict:
    data, _ = _read(directory / MANIFEST, 64 * 1024, "manifest")
    try:
        settings = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique)
    except ControlError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ControlError("invalid_redocking_manifest") from exc
    if not isinstance(settings, dict) or settings.keys() != {"samples"} or not isinstance(settings["samples"], dict):
        raise ControlError("invalid_redocking_manifest")
    samples = settings["samples"]
    if not samples or (example is not None and example not in samples):
        raise ControlError("no_nominated_redocking_control")
    if len(samples) > _MAX_CONTROLS:
        raise ControlError("redocking_control_limit")
    return samples if example is None else {example: samples[example]}


def _pose(directory: Path, name: object, role: str) -> Pose:
    from rdkit import Chem

    if not isinstance(name, str) or not name or not name.isprintable() or len(name) > 255:
        raise ControlError(f"invalid_{role}_filename")
    path = PureWindowsPath(name)
    if path.name != name or ":" in name or path.is_reserved() or path.suffix.lower() not in {".sdf", ".mol"}:
        raise ControlError(f"invalid_{role}_filename")
    data, file_stat = _read(directory / name, 1024 * 1024, role)
    try:
        block = data.decode("utf-8-sig")
        lines = block.rstrip().splitlines()
        if path.suffix.lower() == ".sdf":
            if lines[-1:] != ["$$$$"] or lines.count("$$$$") != 1:
                raise ControlError(f"{role}_requires_one_sdf_record")
        elif lines[-1:] != ["M  END"]:
            raise ControlError(f"{role}_requires_one_mol_record")
        if sum(line.rstrip() == "M  END" for line in lines) != 1:
            raise ControlError(f"{role}_requires_one_molecule")
        mol = read_nominated_pose(block, "")
    except (UnicodeError, RuntimeError, OverflowError) as exc:
        raise ControlError(f"{role}_unreadable") from exc
    except PoseUnreadableError as exc:
        raise ControlError(f"{role}_{exc}") from exc

    if len(Chem.GetMolFrags(mol)) != 1:
        raise ControlError(f"{role}_disconnected_ligand")
    if not 1 <= mol.GetNumHeavyAtoms() <= _MAX_HEAVY_ATOMS:
        raise ControlError(f"{role}_heavy_atom_count_unsupported")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if any(group.GetGroupType() != Chem.StereoGroupType.STEREO_ABSOLUTE for group in mol.GetStereoGroups()):
        raise ControlError(f"{role}_ambiguous_stereochemistry")
    molecule = Chem.RemoveHs(mol)
    if any(item.specified != Chem.StereoSpecified.Specified for item in Chem.FindPotentialStereo(molecule)):
        raise ControlError(f"{role}_ambiguous_stereochemistry")
    identity = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    heavy = Chem.RemoveAllHs(mol, sanitize=False)
    if len(Chem.GetMolFrags(heavy)) != 1:
        raise ControlError(f"{role}_disconnected_heavy_atoms")
    coordinates = heavy.GetConformer().GetPositions()
    if len(coordinates) > 1 and np.all(coordinates == coordinates[0]):
        raise ControlError(f"{role}_missing_coordinates")
    # Atomic number labels give the upstream heavy-atom graph convention. Preserve
    # heavy isotope labels too, so a tracer cannot exchange with an unlabelled atom.
    labels = np.array([a.GetAtomicNum() + 128 * a.GetIsotope() for a in heavy.GetAtoms()])
    return Pose(
        identity=identity,
        coordinates=coordinates,
        atom_labels=labels,
        adjacency=Chem.GetAdjacencyMatrix(heavy),
        sha256=hashlib.sha256(data).hexdigest(),
        file_stat=file_stat,
    )


def _source_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and _IDENTIFIER.fullmatch(value) is not None
        and value.casefold() not in {"none", "null", "na", "nan", "unknown", "unspecified"}
    )


def _control(directory: Path, entry: object) -> Control:
    from rdkit import rdBase

    if not isinstance(entry, dict) or entry.keys() - _FIELDS:
        raise ControlError("invalid_redocking_control")
    reference_id = entry.get("experimental_reference_id")
    if not _source_id(reference_id) or not entry.get("reference"):
        raise ControlError("experimental_reference_required")
    frame = entry.get("pose_frame")
    if not _source_id(frame) or not _source_id(entry.get("reference_frame")):
        raise ControlError("receptor_frame_required")
    if frame != entry["reference_frame"]:
        raise ControlError("receptor_frame_mismatch")
    with rdBase.BlockLogs():
        pose = _pose(directory, entry.get("pose"), "pose")
        reference = _pose(directory, entry["reference"], "reference")
    if os.path.samestat(pose.file_stat, reference.file_stat):
        raise ControlError("pose_and_reference_are_same_file")
    if pose.identity != reference.identity or pose.coordinates.shape != reference.coordinates.shape:
        raise ControlError("ligand_identity_mismatch")
    return Control(pose, reference, reference_id, frame)


def controls(
    directory: Path, example: str | None, *, standalone: bool = False
) -> Iterator[tuple[str | None, Control | None, str | None]]:
    """Resolve only nominated candidates and preserve each unusable pair in its denominator."""
    samples = _samples(directory, example)
    candidates = None
    if not standalone:
        from rfproteina.disease.validation_checks import _candidates, _inventory

        candidates = {c.example_id for c in _candidates(_inventory(directory)) if c.example_id}
    for name, entry in samples.items():
        if not _IDENTIFIER.fullmatch(name):
            yield None, None, "invalid_candidate_id"
            continue
        if candidates is not None and name not in candidates:
            yield name, None, "unknown_generated_candidate"
            continue
        try:
            control = _control(directory, entry)
        except ControlError as exc:
            yield name, None, str(exc)
        except (ImportError, OSError):
            yield name, None, "pose_reader_unavailable"
        except (ValueError, RuntimeError, OverflowError):
            yield name, None, "ligand_unreadable"
        else:
            yield name, control, None


def _rmsd(control: Control) -> float:
    import rustworkx as rx
    from spyrmsd import rmsd
    from spyrmsd.exceptions import NonIsomorphicGraphs
    from spyrmsd.graphs.rx import graph_from_adjacency_matrix

    ref, pose = control.reference, control.pose
    # sPyRMSD 0.9 materializes all graph mappings before measuring. Isomorphic
    # graphs have exactly as many mappings as automorphisms of the reference.
    graph = graph_from_adjacency_matrix(ref.adjacency, ref.atom_labels)
    mappings = rx.graph_vf2_mapping(graph, graph, node_matcher=operator.eq, id_order=False)
    if next(islice(mappings, _MAX_MAPPINGS, None), None) is not None:
        raise ControlError("symmetry_mapping_limit")
    try:
        value = rmsd.symmrmsd(
            ref.coordinates,
            pose.coordinates,
            ref.atom_labels,
            pose.atom_labels,
            ref.adjacency,
            pose.adjacency,
            center=False,
            minimize=False,
        )
    except NonIsomorphicGraphs as exc:
        raise ControlError("ligand_graph_mismatch") from exc
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ControlError("invalid_rmsd_result")
    return float(value)


def measure(directory: Path, example: str | None = None, *, standalone: bool = False) -> dict:
    samples = []
    report = {
        "method": METHOD,
        "endpoint": ENDPOINT,
        "unit": UNIT,
        "status": "untested",
        "attempted": 0,
        "measured": 0,
        "samples": samples,
        "scope": "standalone_controls" if standalone else "generated_candidates",
        "corpus_provenance": declare(METHOD, against=ENDPOINT),
    }
    try:
        for name, control, blocker in controls(directory, example, standalone=standalone):
            row = {"method": METHOD, "candidate_id": name, "status": "untested"}
            samples.append(row)
            if control is None:
                row["blocker"] = blocker
                continue
            row.update(
                experimental_reference_id=control.reference_id,
                receptor_frame_id=control.frame,
                pose_sha256=control.pose.sha256,
                reference_sha256=control.reference.sha256,
                heavy_atoms=len(control.reference.coordinates),
            )
            try:
                import spyrmsd
                from rdkit import rdBase

                value = _rmsd(control)
            except ControlError as exc:
                row["blocker"] = str(exc)
            except (ImportError, OSError):
                row["blocker"] = "spyrmsd_unavailable"
            except (ValueError, RuntimeError, OverflowError):
                row["blocker"] = "spyrmsd_calculation_failed"
            else:
                row.update(status="reported", rmsd_angstrom=value)
                report["measured"] += 1
                report["provider"] = {
                    "spyrmsd": spyrmsd.__version__,
                    "backend": spyrmsd.get_backend(),
                    "rdkit": rdBase.rdkitVersion,
                    "atom_selection": "heavy",
                    "center": False,
                    "minimize": False,
                }
    except ControlError as exc:
        report["blocker"] = str(exc)
    report["attempted"] = len(samples)
    if report["measured"]:
        report["status"] = "reported" if report["measured"] == len(samples) else "partial"
    return report
