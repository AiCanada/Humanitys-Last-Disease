"""Post-generation Therapeutic Nanobody Profiler for explicitly nominated, complete VHH domains."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

from rfproteina.disease.corpus_provenance import declare
from rfproteina.disease.screening_tools import STANDARD_RESIDUES, tool_code_path
from rfproteina.disease.tnp_worker import ProfileError, unique_fields

METHOD = "therapeutic_nanobody_profiler_0.1.0"
MANIFEST = "tnp_domains.json"
_TIMEOUT_SECONDS = 600


def provenance() -> dict:
    return {
        **declare(METHOD, against="clinical_vhh_descriptor_distribution"),
        "caveat": (
            "The 36 calibration domains also occurred in the 108-domain Fc-fusion assay panel. "
            "The flags describe computed VHH properties. The release and the published reference table "
            "gave different positive-charge flags on the archived caplacizumab VHH1 model."
        ),
    }


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Domain:
    label_asym_id: str
    sequence_sha256: str

    @classmethod
    def parse(cls, entry: object) -> Domain:
        if not isinstance(entry, dict) or entry.keys() != {"label_asym_id", "sequence_sha256", "domain_type"}:
            raise ProfileError("invalid_domain_settings")
        if entry["domain_type"] != "VHH":
            raise ProfileError("single_vhh_declaration_required")
        chain, digest = entry["label_asym_id"], entry["sequence_sha256"]
        if not isinstance(chain, str) or not chain.isprintable() or chain.strip() != chain or chain in ("", ".", "?"):
            raise ProfileError("invalid_label_asym_id")
        if not isinstance(digest, str) or not re.fullmatch("[a-fA-F0-9]{64}", digest):
            raise ProfileError("invalid_sequence_digest")
        return cls(chain, digest.lower())


def _extract(text: str, domain: Domain) -> tuple[str, int]:
    import gemmi

    try:
        document = gemmi.cif.read_string(text)
        if len(document) != 1:
            raise ProfileError("single_structure_required")
        block = document.sole_block()
        as_string = gemmi.cif.as_string
        atom_tags = [
            "label_asym_id",
            "label_entity_id",
            "label_seq_id",
            "label_comp_id",
            "?Cartn_x",
            "?Cartn_y",
            "?Cartn_z",
            "?occupancy",
        ]
        atoms = block.find("_atom_site.", atom_tags)
        categories = block.get_mmcif_category_names()
        if "_struct_asym." in categories:
            entities = [
                as_string(r[1])
                for r in block.find("_struct_asym.", ["id", "entity_id"])
                if as_string(r[0]) == domain.label_asym_id
            ]
        else:
            # AtomWorks' generation writer declares the chain-to-entity relationship in atom_site.
            entities = {as_string(r[1]) for r in atoms if as_string(r[0]) == domain.label_asym_id}
        if len(entities) != 1 or not all(entities):
            raise ProfileError("selected_label_chain_absent_or_ambiguous")
        ent = next(iter(entities))
        tags = ["entity_id", "type", "pdbx_seq_one_letter_code_can", "?pdbx_seq_one_letter_code"]
        rows = [r for r in block.find("_entity_poly.", tags) if as_string(r[0]) == ent]
        if len(rows) != 1 or as_string(rows[0][1]) != "polypeptide(L)":
            raise ProfileError("selected_chain_is_not_one_protein")
        row = rows[0]
        sequence = "".join(as_string(row[2]).split())
        original = "".join(as_string(row[3]).split()) if row.has(3) else ""
        if not sequence or not set(sequence) <= STANDARD_RESIDUES or (original and original != sequence):
            raise ProfileError("noncanonical_domain_sequence")
        if _sha(sequence.encode("ascii")) != domain.sequence_sha256:
            raise ProfileError("declared_sequence_digest_mismatch")
        if "_entity_poly_seq." in categories:
            monomers = {}
            for r in block.find("_entity_poly_seq.", ["entity_id", "num", "mon_id"]):
                if as_string(r[0]) == ent:
                    pos = int(as_string(r[1]))
                    residue = gemmi.find_tabulated_residue(as_string(r[2]))
                    if pos in monomers or not residue.is_standard():
                        raise ProfileError("declared_polymer_sequence_mismatch")
                    monomers[pos] = residue.one_letter_code
            if monomers != dict(enumerate(sequence, 1)):
                raise ProfileError("declared_polymer_sequence_mismatch")
        observed = set()
        for row in atoms:
            if as_string(row[0]) != domain.label_asym_id:
                continue
            idx = int(as_string(row[2]))
            aa = gemmi.find_tabulated_residue(as_string(row[3]))
            if (
                as_string(row[1]) != ent
                or idx < 1
                or idx > len(sequence)
                or not aa.is_standard()
                or aa.one_letter_code != sequence[idx - 1]
            ):
                raise ProfileError("coordinate_sequence_mismatch")
            has_xyz = all(row.has(i) and math.isfinite(gemmi.cif.as_number(row[i])) for i in (4, 5, 6))
            occupied = not row.has(7) or not as_string(row[7]) or gemmi.cif.as_number(row[7]) > 0
            if has_xyz and occupied:
                observed.add(idx)
        if not observed:
            raise ProfileError("selected_chain_coordinates_absent")
        return sequence, len(observed)
    except ProfileError:
        raise
    except (RuntimeError, ValueError, IndexError, TypeError, OverflowError) as error:
        raise ProfileError("structure_unreadable") from error


def _interpreter() -> Path:
    explicit = os.environ.get("TNP_PYTHON")
    if explicit:
        candidate = Path(explicit)
    else:
        installed = shutil.which("TNP")
        candidate = (
            Path(installed).with_name("python")
            if installed
            else tool_code_path("tnp", "env/bin/python")
            or Path.home() / ".local/share/rfproteina/tnp-0.1.0-cpu/runtime/bin/python"
        )
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ProfileError("tnp_provider_unavailable")
    # Resolving the python symlink can leave a venv and select its base interpreter.
    return candidate.absolute()


def _native(sequence: str) -> dict:
    if os.name != "posix":
        raise ProfileError("posix_tnp_runtime_required")
    interpreter = _interpreter()
    env = {
        **os.environ,
        "PATH": os.pathsep.join((str(interpreter.parent), os.defpath)),
        "CUDA_VISIBLE_DEVICES": "-1",
        "OPENMM_DEFAULT_PLATFORM": "CPU",
        "OPENMM_CPU_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    worker = Path(__file__).with_name("tnp_worker.py").resolve()
    # The upstream compactness routine passes its temporary path through a shell.
    # Give it only a private, shell-safe path, including when the caller has spaces in TMPDIR.
    temp_root = gettempdir()
    if not re.fullmatch(r"[/A-Za-z0-9._-]+", temp_root):
        temp_root = "/tmp"
    with TemporaryDirectory(prefix="tnp-profile-", dir=temp_root) as tmp:
        env.update(TMPDIR=tmp, TEMP=tmp, TMP=tmp)
        with subprocess.Popen(
            [str(interpreter), "-I", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
            cwd=tmp,
        ) as process:
            try:
                stdout, _ = process.communicate(sequence.encode("ascii"), timeout=_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                # TNP launches NanoBodyBuilder2 and HMMER as child processes.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise ProfileError("tnp_timeout") from error
    if process.returncode or len(stdout) > 65536:
        raise ProfileError("tnp_worker_failed")
    try:
        record = json.loads(stdout, object_pairs_hook=unique_fields)
    except ValueError as error:
        raise ProfileError("tnp_worker_failed") from error
    if not isinstance(record, dict) or record.get("status") not in ("profiled", "partial", "untested"):
        raise ProfileError("tnp_worker_failed")
    return record


def profile(text: str | None, settings: object) -> dict:
    result = {"method": METHOD}
    try:
        if text is None:
            raise ProfileError("structure_unreadable")
        domain = Domain.parse(settings)
        result.update(
            label_asym_id=domain.label_asym_id,
            sequence_sha256=domain.sequence_sha256,
            structure_text_sha256=_sha(text.encode("utf-8")),
            declared_domain_type="VHH",
        )
        sequence, observed = _extract(text, domain)
        result.update(domain_length=len(sequence), observed_residues=observed)
        result.update(_native(sequence))
    except ProfileError as error:
        result.update(status="untested", blocker=str(error))
    except ImportError:
        result.update(status="untested", blocker="structure_reader_unavailable")
    except OSError:
        result.update(status="untested", blocker="tnp_runtime_io_error")
    return result


@dataclass
class RunProfiles:
    settings: dict = field(default_factory=dict)
    manifest_sha256: str | None = None
    blocker: str | None = None
    results: list[dict] = field(default_factory=list)

    @classmethod
    def configured(cls, directory: Path) -> RunProfiles | None:
        path = directory / MANIFEST
        if not path.exists() and not path.is_symlink():
            return None
        run = cls()
        try:
            if not path.is_file():
                raise ProfileError("domain_manifest_unreadable")
            with path.open("rb") as stream:
                data = stream.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ProfileError("domain_manifest_size_limit")
            run.manifest_sha256 = _sha(data)
            obj = json.loads(data, object_pairs_hook=unique_fields)
            if (
                not isinstance(obj, dict)
                or obj.keys() != {"samples"}
                or not isinstance(obj["samples"], dict)
                or any(not k or k.strip() != k or not k.isprintable() for k in obj["samples"])
            ):
                raise ProfileError("invalid_domain_manifest")
            run.settings = obj["samples"]
        except ProfileError as error:
            run.blocker = str(error)
        except (ValueError, RecursionError):
            run.blocker = "invalid_domain_manifest"
        except OSError:
            run.blocker = "domain_manifest_unreadable"
        return run

    def observe(self, example: str, text: str | None) -> None:
        reason = self.blocker
        if reason is None and text is None:
            reason = "structure_unreadable"
        if reason is None and example not in self.settings:
            reason = "sample_settings_absent"
        result = (
            {"method": METHOD, "status": "untested", "blocker": reason}
            if reason
            else profile(text, self.settings[example])
        )
        self.results.append({"example_id": example, **result})

    def as_record(self) -> dict:
        counts = {s: sum(row["status"] == s for row in self.results) for s in ("profiled", "partial", "untested")}
        missing = self.settings.keys() - {row["example_id"] for row in self.results}
        return {
            "method": METHOD,
            "manifest_sha256": self.manifest_sha256,
            "corpus_provenance": provenance(),
            "attempted": len(self.results),
            **counts,
            "unmatched_manifest_ids": sorted(missing),
            "blocker": self.blocker or (None if self.results else "designs_absent"),
            "samples": self.results,
        }
