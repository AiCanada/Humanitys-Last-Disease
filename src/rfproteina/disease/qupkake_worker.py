"""Isolated CPU worker for the released QupKake micro-pKa models."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from contextlib import redirect_stdout
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit import Chem
    from torch_geometric.data import Data

METHOD = "qupkake_gfn2_xtb_micro_pka"
REVISION = "f7a294e91929e4e4f0a7100ad5cf5a64a34a7d4f"
SEED = 42
_VERSIONS = {
    "torch": "2.1.2+cpu",
    "torch-geometric": "2.4.0",
    "pytorch-lightning": "2.1.0",
    "torchmetrics": "1.2.1",
    "rdkit": "2022.9.5",
    "numpy": "1.26.4",
}
_CODE = (
    "featurizer.py",
    "models/__init__.py",
    "mol_dataset.py",
    "mol_utils.py",
    "pka_models.py",
    "predict.py",
    "sites_models.py",
    "transforms.py",
    "xtbp.py",
)
_CODE_DIGEST = "a253aa2045630fde0ad5603ad530daf0001996cfbde775f4fc7b548bca63b77d"
_ASSETS = {
    "models/deprotonation_model.ckpt": "888f2b2259d509871569188f4bcf3f23741c3de8e141220dcc60797e6dce8fc0",
    "models/protonation_model.ckpt": "ea4e30ac225183dd1bc452cfe748eea7fcf16083ada9e07a4dd50e49652e8a2b",
    "models/pka_model.ckpt": "04848ca4faf8d6a0d7c88dd253edf7411fdd239d7e204ef3b90592b4c594decb",
    "xtb-641/bin/xtb": "54f55ee7f62033c09355144d013cdc02ed97f0e94fe445f5accf3e8bc1046715",
    "xtb-641/share/xtb/param_gfn2-xtb.txt": "f6f05c6c5264cb2f75c54f8ac6838044541c1e493b16f1a4d4ee4fa64de5f545",
}


class ProfileError(ValueError):
    """A scientific or provider condition that prevents attribution."""


def _load_release(*, inspect_only: bool = False) -> dict:
    spec = importlib.util.find_spec("qupkake")
    if spec is None or not spec.origin:
        raise ProfileError("qupkake_not_installed")
    root = Path(spec.origin).parent
    try:
        code = hashlib.sha256()
        for name in sorted(_CODE):
            code.update((root / name).read_bytes())
        if code.hexdigest() != _CODE_DIGEST:
            raise ProfileError("qupkake_source_revision_mismatch")
        for name, expected in _ASSETS.items():
            digest = hashlib.sha256()
            with (root / name).open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ProfileError("qupkake_asset_digest_mismatch")
    except OSError as error:
        raise ProfileError("qupkake_asset_unavailable") from error
    executable = root / "xtb-641/bin/xtb"
    if not os.access(executable, os.X_OK):
        raise ProfileError("xtb_not_executable")
    versions = {name: metadata.version(name) for name in _VERSIONS}
    if versions != _VERSIONS:
        raise ProfileError("qupkake_dependency_version_mismatch")
    if inspect_only:
        return {"installed": True}
    # Release __init__ calls conda even for an explicitly selected binary. Bind the
    # verified executable to the same package attribute used by its native modules.
    package = importlib.util.module_from_spec(spec)
    package.XTB_LOCATION = str(executable)
    sys.modules["qupkake"] = package
    os.environ["XTBPATH"] = str(root / "xtb-641/share/xtb")
    return {
        "revision": REVISION,
        "xtb": "6.4.1",
        "device": "cpu",
        "packages": versions,
        "code_sha256": _CODE_DIGEST,
        "asset_sha256": _ASSETS,
    }


def _check_xtb(output: str, charge: int, *, geometry: bool) -> None:
    match = re.search(r"^\s*charge\s*:\s*(-?\d+)\s*$", output, re.MULTILINE)
    if match is None or int(match[1]) != charge:
        raise ProfileError("xtb_charge_mismatch")
    if "* finished run on" not in output or (geometry and "GEOMETRY OPTIMIZATION CONVERGED AFTER" not in output):
        raise ProfileError("xtb_calculation_incomplete")


def _bind_xtb_checks() -> None:
    from qupkake import featurizer, mol_utils, xtbp
    from rdkit import Chem

    class CheckedXTB(xtbp.RunXTB):
        def _run_xtb(self) -> str:
            mol = Chem.MolFromMolFile(self.mol, removeHs=False) if isinstance(self.mol, str) else self.mol
            output = super()._run_xtb()
            _check_xtb(output, Chem.GetFormalCharge(mol), geometry="--opt" in self.options.split())
            return output

    # Both the tautomer search and the graph featurizer imported this class by name.
    mol_utils.RunXTB = featurizer.RunXTB = CheckedXTB


def _geometry(smiles: str, tautomerize: bool) -> tuple[Chem.Mol, Chem.Mol, int | None]:
    from qupkake.mol_utils import Tautomerize
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    class CheckedTautomers(Tautomerize):
        def set_tautomers(self) -> list[Chem.Mol]:
            result = rdMolStandardize.TautomerEnumerator().Enumerate(Chem.RemoveHs(self.mol))
            if result.status != rdMolStandardize.TautomerEnumeratorStatus.Completed:
                raise ProfileError("tautomer_enumeration_incomplete")
            return [Chem.AddHs(t, addCoords=True) for t in result]

    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        raise ProfileError("provider_cannot_read_canonical_smiles")
    for atom in parent.GetAtoms():
        atom.SetIntProp("_q_parent", atom.GetIdx() + 1)
    mol = Chem.AddHs(parent)
    if AllChem.EmbedMolecule(mol, randomSeed=SEED, useRandomCoords=True, ignoreSmoothingFailures=True) < 0:
        raise ProfileError("conformer_embedding_failed")
    if not AllChem.MMFFHasAllMoleculeParams(mol) or AllChem.MMFFOptimizeMolecule(mol, maxIters=1000) != 0:
        raise ProfileError("mmff_geometry_unavailable")
    count = None
    if tautomerize:
        t = CheckedTautomers(mol=mol, name="candidate", keep_mol=False, check_exists=False, num_processes=1)
        mol, count = t.lowest_tautomer, len(t.tautomers)
        if count > 1 and not all(math.isfinite(energy) for _, energy in t.sorted_energies):
            raise ProfileError("nonfinite_tautomer_energy")
    return parent, mol, count


def _atoms(mol: Chem.Mol) -> dict:
    # Native conjugation removes and restores hydrogens. Heavy-atom identity must
    # survive both that operation and the optional tautomer enumeration.
    atoms = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if not atom.HasProp("_q_parent"):
            raise ProfileError("atom_correspondence_lost")
        key = atom.GetIntProp("_q_parent")
        if key in atoms:
            raise ProfileError("duplicate_atom_correspondence")
        atoms[key] = (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            tuple(sorted(a.GetIntProp("_q_parent") for a in atom.GetNeighbors() if a.GetAtomicNum() != 1)),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(includeNeighbors=True),
        )
    return atoms


def _same_skeleton(before: dict, after: dict) -> None:
    if before.keys() != after.keys() or any(before[i][:3] != after[i][:3] for i in before):
        raise ProfileError("heavy_atom_skeleton_changed")


def _reaction_site(before: dict, after: dict, kind: str) -> int:
    _same_skeleton(before, after)
    changed = [key for key in before if before[key][3:] != after[key][3:]]
    if len(changed) == 1:
        i = changed[0]
        step = 1 if kind == "basic" else -1
        if after[i][3] - before[i][3] == after[i][4] - before[i][4] == step:
            return i
    raise ProfileError("conjugate_is_not_one_site_one_proton")


def _check_valence(core: Chem.Mol) -> None:
    """Check the closed-shell Lewis electron count for second-period main-group atoms."""
    from rdkit import Chem

    kekule = Chem.Mol(core)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    for atom in kekule.GetAtoms():
        z = atom.GetAtomicNum()
        if 5 <= z <= 9:
            bonds = atom.GetTotalValence()
            nonbonding = (z - 2) - atom.GetFormalCharge() - bonds
            if nonbonding < 0 or nonbonding % 2 or 2 * bonds + nonbonding > 8:
                raise ProfileError("invalid_closed_shell_valence")


def _state(mol: Chem.Mol, states: dict) -> str:
    from rdkit import Chem

    # Validate a copy so sanitization cannot change the native graph sent to the model.
    try:
        core = Chem.RemoveHs(mol, sanitize=True)
    except ValueError as error:
        raise ProfileError("invalid_microstate_valence") from error
    if any(a.GetNumRadicalElectrons() for a in core.GetAtoms()):
        raise ProfileError("open_shell_microstate_unsupported")
    _check_valence(core)
    smiles = Chem.MolToSmiles(core)
    # Include the parent atom map in the identifier so symmetry-equivalent sites
    # can share an unlabelled SMILES without overwriting their site assignments.
    for atom in core.GetAtoms():
        atom.SetAtomMapNum(atom.GetIntProp("_q_parent"))
    mapped = Chem.MolToSmiles(core)
    key = "state_" + hashlib.sha256(mapped.encode("ascii")).hexdigest()
    states[key] = {"smiles": smiles, "atom_mapped_smiles": mapped, "formal_charge": Chem.GetFormalCharge(core)}
    return key


def _reason(error: Exception) -> str:
    if isinstance(error, ProfileError):
        return str(error)
    if isinstance(error, ImportError | metadata.PackageNotFoundError):
        return "provider_dependency_unavailable"
    return "provider_failed_" + type(error).__name__


def _profile(smiles: str, tautomerize: bool) -> dict:
    provider = _load_release()
    import torch
    from qupkake.featurizer import Featurizer
    from qupkake.mol_dataset import MolPairDataset
    from qupkake.predict import load_models
    from qupkake.transforms import IncludeEnergy, ToTensor
    from torch_geometric.data import Batch
    from torch_geometric.transforms import Compose, NormalizeFeatures

    class PairBuilder:
        """Use the two molecule operations of the release without its file/row cache."""

        type_values = ("basic", "acidic")
        conjugate = MolPairDataset._conjugate_mol
        pair = MolPairDataset._graphs_to_molpair

    class CheckedFeatures(Featurizer):
        def _handle_processing_error(self, error: Exception) -> None:
            raise ProfileError(_reason(error)) from error

    def features(mol: Chem.Mol) -> Data:
        graph = CheckedFeatures(mol=mol, num_processes=1).data
        if not all(torch.isfinite(v).all() for v in (graph.x, graph.edge_attr, graph.global_attr, graph.energy)):
            raise ProfileError("nonfinite_quantum_features")
        return graph

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ProfileError("cpu_isolation_failed")
    _bind_xtb_checks()
    parent, selected, count = _geometry(smiles, tautomerize)
    before = _atoms(selected)
    _same_skeleton(_atoms(parent), before)
    states = {}
    original, current = _state(parent, states), _state(selected, states)
    graph = features(selected)
    basic, acidic, regression = load_models()
    regression.eval()
    pair_ops = PairBuilder()
    globals_ = ["global_attr_prot", "global_attr_deprot"]
    transform = Compose(
        [ToTensor(["y", "d_energy"]), NormalizeFeatures(globals_), IncludeEnergy(globals_, "d_energy", -1)]
    )
    sites, rejected = {}, []
    failed_heads = []
    with torch.inference_mode():
        for kind, head in (("basic", basic), ("acidic", acidic)):
            # The released SitesPrediction.predict_step uses a strict zero-logit cut.
            try:
                logits = (
                    head.eval().model(x=graph.x, edge_attr=graph.edge_attr, edge_index=graph.edge_index).reshape(-1)
                )
                if logits.numel() != selected.GetNumAtoms() or not torch.isfinite(logits).all():
                    raise ProfileError("invalid_site_head_output")
                indices = (logits > 0).nonzero().flatten().tolist()
            except Exception as error:
                failed_heads.append({"kind": kind, "blocker": _reason(error)})
                continue
            for index in indices:
                atom = selected.GetAtomWithIdx(index)
                nomination = {"kind": kind, "provider_atom_index": index}
                try:
                    if atom.GetAtomicNum() == 1 or not atom.HasProp("_q_parent"):
                        raise ProfileError("site_is_not_a_mapped_heavy_atom")
                    nominated = atom.GetIntProp("_q_parent")
                    nomination["nominated_atom_map"] = nominated
                    other = pair_ops.conjugate(selected, index, kind)
                    actual = _reaction_site(before, _atoms(other), kind)
                    other_id = _state(other, states)
                    key = (kind, actual, other_id)
                    if key in sites:
                        sites[key]["nominated_atom_maps"].append(nominated)
                        continue
                    prot_id, deprot_id = (other_id, current) if kind == "basic" else (current, other_id)
                    row = {
                        "method": METHOD,
                        "kind": kind,
                        "site_atom_map": actual,
                        "endpoint": "aqueous_microscopic_pKa",
                        "unit": "pKa",
                        "atomic_number": before[actual][0],
                        "nominated_atom_maps": [nominated],
                        "protonated_state_id": prot_id,
                        "deprotonated_state_id": deprot_id,
                    }
                    sites[key] = row
                    try:
                        conjugate = features(other)
                        prot, deprot = (conjugate, graph) if kind == "basic" else (graph, conjugate)
                        pair = transform(pair_ops.pair(prot, deprot, "candidate", None))
                        batch = Batch.from_data_list([pair], follow_batch=["x_prot", "x_deprot"])
                        value = regression.predict_step(batch, 0)
                        if value.numel() != 1 or not math.isfinite(value.item()):
                            raise ProfileError("invalid_regression_output")
                        row.update(status="predicted", micro_pka=float(value.item()))
                    except Exception as error:
                        row.update(status="untested", blocker=_reason(error))
                except Exception as error:
                    rejected.append({**nomination, "blocker": _reason(error)})
    rows = list(sites.values())
    predicted = sum(r["status"] == "predicted" for r in rows)
    status = "predicted" if rows else "no_sites"
    if failed_heads or rejected or predicted < len(rows):
        status = "partial" if predicted else "untested"
    result = {
        "status": status,
        "provider": provider,
        "sites": rows,
        "rejected_nominations": rejected,
        "failed_site_heads": failed_heads,
        "input_state_id": original,
        "selected_state_id": current,
        "states": states,
        "tautomers_considered": count,
    }
    if status == "untested":
        result["blocker"] = (
            "site_head_incomplete" if failed_heads and not rows and not rejected else "no_attributable_site_prediction"
        )
    return result


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-installation"]:
        try:
            _load_release(inspect_only=True)
        except (OSError, ValueError, ImportError):
            sys.exit(1)
        sys.exit(0)
    request = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        try:
            result = _profile(request["smiles"], request["tautomerize"])
        except Exception as exc:
            result = {"status": "untested", "blocker": _reason(exc), "sites": []}
    print(json.dumps({**result, "input": request["smiles"]}, allow_nan=False))
