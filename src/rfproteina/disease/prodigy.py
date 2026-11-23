"""Post-generation PRODIGY affinity estimates for two explicitly named protein partners."""

from __future__ import annotations

import hashlib
import io
import json
import math
import string
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from rfproteina.disease.corpus_provenance import declare

if TYPE_CHECKING:
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue

METHOD = "prodigy_ic_nis"
_ENDPOINT = "protein_protein_binding_delta_g"
PARTNERS_FILE = "prodigy_partners.json"
_WORKER = "rfproteina.disease.prodigy"
_CHAIN_ALPHABET = string.ascii_letters + string.digits
_CONTACT_DISTANCE = 5.5  # PRODIGY's interpartner heavy-atom contact radius in angstroms.


class AffinityInputError(ValueError):
    """A named reason why the requested pair cannot be measured."""


@dataclass(frozen=True)
class Partners:
    binder_chains: tuple[str, ...]
    target_chains: tuple[str, ...]
    temperature_c: float
    chain_namespace: str
    model_index: int | None
    altloc: str | None

    @classmethod
    def parse(cls, value: object) -> Partners:
        if not isinstance(value, dict) or value.keys() - cls.__dataclass_fields__.keys():
            raise AffinityInputError("invalid_partner_settings")
        groups = []
        for key in ("binder_chains", "target_chains"):
            group = value.get(key)
            if (
                not isinstance(group, list)
                or not group
                or any(
                    not isinstance(c, str) or c in ("", ".", "?") or c.strip() != c or not c.isprintable()
                    for c in group
                )
            ):
                raise AffinityInputError("invalid_chain_group")
            groups.append(tuple(group))
        chains = groups[0] + groups[1]
        if len(set(chains)) != len(chains):
            raise AffinityInputError("repeated_chain")
        if len(chains) > len(_CHAIN_ALPHABET):
            raise AffinityInputError("too_many_selected_chains")
        temp = value.get("temperature_c", 25.0)
        if type(temp) not in (int, float) or not -273.15 < temp <= sys.float_info.max:
            raise AffinityInputError("invalid_temperature")
        namespace = value.get("chain_namespace", "author")
        if namespace not in ("author", "label"):
            raise AffinityInputError("invalid_chain_namespace")
        model = value.get("model_index")
        if model is not None and (type(model) is not int or model < 1):
            raise AffinityInputError("invalid_model_index")
        altloc = value.get("altloc")
        if altloc is not None and (not isinstance(altloc, str) or len(altloc) != 1 or not altloc.isalnum()):
            raise AffinityInputError("invalid_altloc")
        return cls(*groups, float(temp), namespace, model, altloc)

    def as_record(self) -> dict:
        return {**asdict(self), "binder_chains": list(self.binder_chains), "target_chains": list(self.target_chains)}


def _untested(reason: str) -> dict:
    return {"method": METHOD, "status": "untested", "blocker": reason}


def predict_file(path: Path, partners: Partners) -> dict:
    """Run the same screen used by the generation report on one local structure."""
    from rfproteina.disease.screen import _design_text

    name = path.with_suffix("") if path.suffix in (".gz", ".gzip", ".zst") else path
    fmt = {".cif": "cif", ".mmcif": "cif", ".pdbx": "cif", ".pdb": "pdb", ".ent": "pdb"}.get(name.suffix.lower())
    if fmt is None:
        result = _untested("unsupported_structure_format_or_namespace")
    elif (text := _design_text(path)) is None:
        result = _untested("structure_unreadable")
    else:
        result = predict(text, fmt, partners)
    return {**partners.as_record(), **result, "corpus_provenance": declare(METHOD, against=_ENDPOINT)}


def _unique_keys(pairs: list) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise AffinityInputError("duplicate_partner_setting")
        obj[key] = value
    return obj


@dataclass
class RunAffinity:
    partners: Partners | None
    blocker: str | None = None
    settings_sha256: str | None = None
    results: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def configured(cls, directory: Path) -> RunAffinity | None:
        digest = None
        try:
            with (directory / PARTNERS_FILE).open("rb") as stream:
                data = stream.read(65537)
            if len(data) > 65536:
                raise AffinityInputError("partner_settings_too_large")
            digest = hashlib.sha256(data).hexdigest()
            return cls(
                Partners.parse(json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_keys)),
                settings_sha256=digest,
            )
        except FileNotFoundError:
            return cls(None, "partner_settings_unreadable") if (directory / PARTNERS_FILE).is_symlink() else None
        except AffinityInputError as error:
            return cls(None, str(error), digest)
        except (OSError, ValueError, RecursionError):
            return cls(None, "partner_settings_unreadable", digest)

    def observe(self, sample: str, text: str | None, *, blocker: str | None = None) -> None:
        if self.blocker or self.partners is None or blocker or text is None:
            result = _untested(self.blocker or blocker or "structure_unreadable")
        else:
            result = predict(text, "cif", self.partners)
        self.results[sample] = result

    def as_record(self) -> dict:
        measured = sum(row["status"] == "predicted" for row in self.results.values())
        return {
            "method": METHOD,
            "corpus_provenance": declare(METHOD, against=_ENDPOINT),
            "partners": self.partners.as_record() if self.partners is not None else None,
            "settings_sha256": self.settings_sha256,
            "attempted": len(self.results),
            "predicted": measured,
            "untested": len(self.results) - measured,
            "blocker": self.blocker or ("designs_absent" if not self.results else None),
            "samples": self.results,
        }


def predict(text: str, fmt: str, partners: Partners) -> dict:
    """Isolate the native FreeSASA calculation so a failed structure cannot stop a run."""
    data = text.encode("utf-8")
    identity = {"structure_text_sha256": hashlib.sha256(data).hexdigest(), **partners.as_record()}
    try:
        run = subprocess.run(
            [sys.executable, "-m", _WORKER, fmt, json.dumps(partners.as_record(), allow_nan=False)],
            input=data,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result = _untested("provider_timeout")
    except OSError:
        result = _untested("provider_start_failed")
    else:
        try:
            result = json.loads(run.stdout) if run.returncode == 0 else _untested("provider_failed")
            if not isinstance(result, dict) or result.get("status") not in ("predicted", "untested"):
                result = _untested("provider_output_unreadable")
        except (ValueError, UnicodeError):
            result = _untested("provider_output_unreadable")
    return {**identity, **result}


def _cif_input(text: str) -> tuple[dict, io.StringIO]:
    from Bio.PDB import MMCIFIO
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    data = io.StringIO(text)
    table = MMCIF2Dict(data)
    count = len(table["_atom_site.id"])
    # AtomWorks writes model coordinates without crystallographic B factors. Bio.PDB
    # requires a numeric placeholder for these, although PRODIGY never reads them.
    # Missing conformer, insertion and model columns denote ordinary single entries.
    defaults = {
        "_atom_site.B_iso_or_equiv": "0",
        "_atom_site.label_alt_id": ".",
        "_atom_site.pdbx_PDB_ins_code": "?",
        "_atom_site.pdbx_PDB_model_num": "1",
    }
    missing = {key: [value] * count for key, value in defaults.items() if key not in table}
    b_key = "_atom_site.B_iso_or_equiv"
    if b_key in table and any(v in (".", "?") for v in table[b_key]):
        missing[b_key] = ["0" if v in (".", "?") else v for v in table[b_key]]
    if missing:
        table.update(missing)
        writer = MMCIFIO()
        writer.set_dict({"data_": "complex", **{k: v for k, v in table.items() if k.startswith("_atom_site.")}})
        data = io.StringIO()
        writer.save(data)
    data.seek(0)
    return table, data


def _structure(text: str, fmt: str, request: Partners) -> tuple[Model, int]:
    from Bio.PDB import MMCIFParser, PDBParser

    table = None
    try:
        if fmt == "cif":
            table, data = _cif_input(text)
            parser = MMCIFParser(auth_chains=request.chain_namespace == "author", QUIET=True)
        elif fmt == "pdb" and request.chain_namespace == "author":
            data = io.StringIO(text)
            parser = PDBParser(PERMISSIVE=False, QUIET=True)
        else:
            raise AffinityInputError("unsupported_structure_format_or_namespace")
        structure = parser.get_structure("complex", data)
    except AffinityInputError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        raise AffinityInputError("structure_unreadable") from error
    models = list(structure)
    if len(models) > 1 and request.model_index is None:
        raise AffinityInputError("model_selection_required")
    index = request.model_index or 1
    if index > len(models):
        raise AffinityInputError("model_absent")
    model = models[index - 1]
    if table is not None:
        # The author chain can span multiple labelled polymer chains in an assembly.
        # Bio.PDB may merge their residues under one ID when numbering does not collide.
        groups = table["_atom_site.group_PDB"]
        authors = table.get("_atom_site.auth_asym_id", [None] * len(groups))
        labels = table["_atom_site.label_asym_id"]
        serials = table.get("_atom_site.pdbx_PDB_model_num", [None] * len(groups))
        numbers = table.get("_atom_site.auth_seq_id", table.get("_atom_site.label_seq_id", []))
        chosen = set(request.binder_chains + request.target_chains)
        seen = {}
        for group, author, label, serial, number in zip(groups, authors, labels, serials, numbers, strict=True):
            key, alias = (author, label) if request.chain_namespace == "author" else (label, author)
            if group != "ATOM" or key not in chosen or (serial is not None and int(serial) != model.serial_num):
                continue
            if number in (".", "?"):
                raise AffinityInputError("residue_identifier_absent")
            if seen.setdefault(key, alias) != alias:
                raise AffinityInputError("ambiguous_chain_namespace")
    return model, index


def _residue_atoms(residue: Residue, aa: str, altloc: str | None) -> list[Atom]:
    from Bio.PDB.ic_data import residue_atom_bond_state

    if residue.is_disordered() == 2:
        raise AffinityInputError("ambiguous_residue_identity")
    heavy = [atom for atom in residue.get_unpacked_list() if atom.element not in ("H", "D")]
    variants = {atom.altloc for atom in heavy if atom.altloc != " "}
    chosen = altloc
    if variants:
        if len(variants) == 1 and chosen is None:
            chosen = next(iter(variants))
        if chosen not in variants:
            raise AffinityInputError("alternate_conformer_required")
    atoms = [atom for atom in heavy if atom.altloc == " " or atom.altloc == chosen]
    names = {atom.name for atom in atoms}
    expected = {"N", "CA", "C", "O", *residue_atom_bond_state.get(aa, {})}
    if aa != "G":
        expected.add("CB")
    if expected - names:
        raise AffinityInputError("incomplete_protein_residue")
    if len(names) != len(atoms) or names - expected - {"OXT"}:
        raise AffinityInputError("unsupported_protein_atoms")
    return atoms


def _prepare(model: Model, request: Partners) -> tuple[Model, list[str], dict]:
    from Bio.Data.PDBData import protein_letters_3to1
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Polypeptide import is_aa, is_nucleic
    from Bio.PDB.Residue import Residue

    # FreeSASA uses single-character chains. Assign unique temporary identifiers and
    # residue numbers in memory, including for residues with a PDB insertion code.
    pair = Model(0)
    chains = request.binder_chains + request.target_chains
    mapped = dict(zip(chains, _CHAIN_ALPHABET[: len(chains)], strict=True))
    seen_coords = set()
    included = {}
    insertions = omitted = 0
    for label, short in mapped.items():
        if label not in model:
            raise AffinityInputError("selected_chain_absent")
        chain = Chain(short)
        for res in model[label]:
            if res.id[0] != " ":
                if is_aa(res) or is_nucleic(res) or {"N", "CA", "C"}.issubset(res.child_dict):
                    raise AffinityInputError("unsupported_polymer_residue")
                omitted += 1
                continue
            aa = protein_letters_3to1.get(res.resname)
            if aa is None:
                raise AffinityInputError("unsupported_polymer_residue")
            normalized = Residue((" ", len(chain) + 1, " "), res.resname, " ")
            for atom in _residue_atoms(res, aa, request.altloc):
                coord = tuple(float(n) for n in atom.coord)
                if len(coord) != 3 or not all(math.isfinite(n) for n in coord) or coord in seen_coords:
                    raise AffinityInputError("invalid_coordinates")
                if atom.occupancy == 0:
                    raise AffinityInputError("zero_occupancy_atom")
                if atom.element != atom.name[0] or atom.occupancy is None or not 0 < atom.occupancy <= 1:
                    raise AffinityInputError("invalid_protein_atom")
                seen_coords.add(coord)
                copy = atom.copy()
                copy.altloc = " "
                copy.disordered_flag = 0
                normalized.add(copy)
            chain.add(normalized)
            insertions += res.id[2] != " "
        if not len(chain):
            raise AffinityInputError("selected_chain_has_no_protein")
        included[label] = len(chain)
        pair.add(chain)
    groups = [",".join(mapped[c] for c in group) for group in (request.binder_chains, request.target_chains)]
    counts = {
        "residues_by_chain": included,
        "heavy_atoms": len(seen_coords),
        "insertion_coded_residues": insertions,
        "excluded_nonpolymer_residues": omitted,
    }
    return pair, groups, counts


def prepared_pair(text: str, fmt: str, request: Partners) -> tuple[Model, list[str], dict, int]:
    """Resolve the requested model and both complete protein partners without evaluating affinity."""
    from Bio.PDB.PDBExceptions import PDBConstructionException

    try:
        model, index = _structure(text, fmt, request)
    except PDBConstructionException as error:
        raise AffinityInputError("structure_unreadable") from error
    pair, groups, counts = _prepare(model, request)
    return pair, groups, counts, index


def verify_interface(text: str, request: Partners) -> None:
    """Check one interpartner contact without enumerating contacts or calculating surface areas."""
    from Bio.PDB.NeighborSearch import NeighborSearch

    pair, groups, _, _ = prepared_pair(text, "cif", request)
    left, right = ([atom for label in group.split(",") for atom in pair[label].get_atoms()] for group in groups)
    neighbours = NeighborSearch(right)
    if not any(neighbours.search(atom.coord, _CONTACT_DISTANCE) for atom in left):
        raise AffinityInputError("no_interpartner_contacts")


def _calculate(text: str, fmt: str, request: Partners) -> dict:
    from prodigy_prot.modules.prodigy import Prodigy

    pair, groups, counts, index = prepared_pair(text, fmt, request)
    predictor = Prodigy(pair, selection=groups, temp=request.temperature_c)
    try:
        predictor.predict(distance_cutoff=_CONTACT_DISTANCE)
    except ValueError as error:
        if str(error) == "No contacts found for selection":
            raise AffinityInputError("no_interpartner_contacts") from error
        raise
    dg, kd = float(predictor.ba_val), float(predictor.kd_val)
    if not math.isfinite(dg) or not math.isfinite(kd) or kd <= 0:
        raise AffinityInputError("unrepresentable_prediction")
    return {
        "method": METHOD,
        "status": "predicted",
        "provider_version": version("prodigy-prot"),
        "freesasa_version": version("freesasa"),
        "biopython_version": version("biopython"),
        "model_index": index,
        **counts,
        "interpartner_residue_contacts": len(predictor.ic_network),
        "predicted_delta_g_kcal_mol": dg,
        "derived_kd_mol_l": kd,
    }


def _worker() -> None:
    try:
        request = Partners.parse(json.loads(sys.argv[2]))
        result = _calculate(sys.stdin.buffer.read().decode("utf-8"), sys.argv[1], request)
    except AffinityInputError as error:
        result = _untested(str(error))
    except (ImportError, OSError):
        result = _untested("provider_unavailable")
    except Exception:
        # Numerical and parser failures may originate in optional compiled dependencies.
        result = _untested("provider_failed")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    _worker()
