"""Tests for rfproteina.disease.interactions (CPU-only, fast)."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from rfproteina.disease import interactions as ix


def _write_pdb(path, atoms):
    lines = []
    for i, (name, resname, chain, resid, x, y, z, element) in enumerate(atoms, start=1):
        lines.append(
            f"ATOM  {i:>5} {name:<4} {resname:>3} {chain}{resid:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _mol_at(smiles, centroid):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    heavy = [conf.GetAtomPosition(a.GetIdx()) for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    cx = np.mean([p.x for p in heavy])
    cy = np.mean([p.y for p in heavy])
    cz = np.mean([p.z for p in heavy])
    shift = (centroid[0] - cx, centroid[1] - cy, centroid[2] - cz)
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        conf.SetAtomPosition(a.GetIdx(), (p.x + shift[0], p.y + shift[1], p.z + shift[2]))
    return mol


def _protein(tmp_path):
    ring = [
        ("CG", "PHE", "A", 1, 1.4, 0.0, 0.0, "C"),
        ("CD1", "PHE", "A", 1, 0.7, 1.2, 0.0, "C"),
        ("CD2", "PHE", "A", 1, -0.7, 1.2, 0.0, "C"),
        ("CE1", "PHE", "A", 1, -1.4, 0.0, 0.0, "C"),
        ("CE2", "PHE", "A", 1, -0.7, -1.2, 0.0, "C"),
        ("CZ", "PHE", "A", 1, 0.7, -1.2, 0.0, "C"),
        ("OD1", "ASP", "A", 2, 20.0, 0.0, 0.0, "O"),
        ("OD2", "ASP", "A", 2, 21.0, 0.0, 0.0, "O"),
        ("OG", "SER", "A", 3, 40.0, 0.0, 0.0, "O"),
        ("O", "GLY", "A", 4, 60.0, 0.0, 0.0, "O"),
    ]
    return _write_pdb(tmp_path / "prot.pdb", ring)


def test_pi_stacking_and_hydrophobic_when_benzene_sits_over_phe(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("c1ccccc1", (0.0, 0.0, 4.0))
    fp = ix.native_fingerprint(protein, mol)
    assert fp.counts["pi_stacking"] > 0
    assert fp.counts["hydrophobic"] > 0


def test_everything_zero_when_ligand_is_far(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("c1ccccc1", (0.0, 0.0, 100.0))
    fp = ix.native_fingerprint(protein, mol)
    assert sum(fp.counts.values()) == 0


def test_hbond_proxy_counts_oxygen_pair(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("CCO", (40.0, 0.0, 2.0))
    fp = ix.native_fingerprint(protein, mol)
    assert fp.counts["hbond"] > 0


def test_salt_bridge_between_cation_and_aspartate(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("C[NH3+]", (20.5, 0.0, 2.0))
    fp = ix.native_fingerprint(protein, mol)
    assert fp.counts["salt_bridge"] > 0


def test_halogen_contact(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("Clc1ccccc1", (60.0, 0.0, 2.0))
    fp = ix.native_fingerprint(protein, mol)
    assert fp.counts["halogen"] > 0


def test_native_fingerprint_refuses_mol_without_conformer(tmp_path):
    protein = _protein(tmp_path)
    flat = Chem.MolFromSmiles("CCO")
    with pytest.raises(ix.StructureUnreadableError):
        ix.native_fingerprint(protein, flat)


def test_native_fingerprint_rejects_invalid_coordinates_and_excludes_cofactors(tmp_path):
    protein = _write_pdb(tmp_path / "cofactor.pdb", [
        ("CA", "ALA", "A", 1, 100, 0, 0, "C"), ("C1", "LIG", "L", 1, 0, 0, 0, "C")])
    mol = _mol_at("CC", (0, 0, 0))
    assert sum(ix.native_fingerprint(protein, mol).counts.values()) == 0
    mol.GetConformer().SetAtomPosition(0, (float("nan"), 0, 0))
    with pytest.raises(ix.StructureUnreadableError, match="nonfinite"):
        ix.native_fingerprint(protein, mol)


def test_native_verdicts_are_all_no_cut(tmp_path):
    protein = _protein(tmp_path)
    mol = _mol_at("c1ccccc1", (0.0, 0.0, 4.0))
    fp = ix.native_fingerprint(protein, mol)
    verdicts = ix.native_verdicts(fp)
    assert {v["state"] for v in verdicts.values()} == {"no_cut"}


def test_fnat_identical_is_one_and_partial_is_fractional(tmp_path):
    reference = _write_pdb(
        tmp_path / "ref.pdb",
        [
            ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            ("CA", "ALA", "A", 2, 0.0, 0.0, 10.0, "C"),
            ("CA", "ALA", "B", 1, 3.0, 0.0, 0.0, "C"),
            ("CA", "ALA", "B", 2, 3.0, 0.0, 10.0, "C"),
        ],
    )
    same = ix.compare_to_reference(reference, reference, "A", "B")
    assert same.fnat == 1.0
    assert same.fnonnat == 0.0

    model = _write_pdb(
        tmp_path / "model.pdb",
        [
            ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            ("CA", "ALA", "A", 2, 0.0, 0.0, 10.0, "C"),
            ("CA", "ALA", "B", 1, 3.0, 0.0, 0.0, "C"),
            ("CA", "ALA", "B", 2, 3.0, 0.0, 100.0, "C"),
        ],
    )
    partial = ix.compare_to_reference(model, reference, "A", "B")
    assert partial.fnat == 0.5


def test_comparison_verdicts_gate_only_when_bar_supplied(tmp_path):
    reference = _write_pdb(
        tmp_path / "r.pdb",
        [
            ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            ("CA", "ALA", "B", 1, 3.0, 0.0, 0.0, "C"),
        ],
    )
    comp = ix.compare_to_reference(reference, reference, "A", "B")
    assert ix.comparison_verdicts(comp)["fnat"]["state"] == "no_cut"
    assert ix.comparison_verdicts(comp, fnat_min=0.5)["fnat"]["state"] == "clear"
    assert ix.comparison_verdicts(comp, fnat_min=1.5)["fnat"]["state"] == "hit"


def test_fnat_distinguishes_insertions_and_has_no_fraction_with_zero_model_contacts(tmp_path):
    ref = _write_pdb(tmp_path / "ref.pdb", [
        ("CA", "ALA", "A", 1, 0, 0, 0, "C"), ("CA", "ALA", "A", 1, 0, 1, 0, "C"),
        ("CA", "ALA", "B", 2, 3, 0, 0, "C")])
    lines = ref.read_text().splitlines()
    lines[1] = lines[1][:26] + "A" + lines[1][27:]
    ref.write_text("\n".join(lines) + "\n")
    model = tmp_path / "model.pdb"
    model.write_text(ref.read_text().replace("ALA", "GLY"))
    same = ix.compare_to_reference(model, ref, "A", "B")
    assert same.recovered_contacts == same.native_contacts == 2 and same.fnat == 1
    model.write_text(model.read_text().replace("   3.000", "  93.000"))
    absent = ix.compare_to_reference(model, ref, "A", "B")
    assert absent.fnat == 0 and absent.fnonnat is None
    with pytest.raises(ix.StructureUnreadableError, match="distinct_chains"):
        ix.compare_to_reference(model, ref, "A", "A")


def _no_dockq(monkeypatch, tmp_path):
    """Make every route to a DockQ executable come up empty."""
    monkeypatch.setenv(ix.DOCKQ_COMMAND_ENV, str(tmp_path / "not-here"))
    monkeypatch.delenv("DATA_PATH", raising=False)
    monkeypatch.setattr(ix.shutil, "which", lambda _name: None)


def test_prolif_and_dockq_refuse_loudly_when_absent(tmp_path, monkeypatch):
    """The refusal path, held whether or not either engine is installed.

    Asserting the raise on a machine that merely lacks ProLIF tests the
    machine, not the code: once ProLIF is installed the same call fails much
    later and with a different exception, and the test keeps passing for the
    wrong reason. Blocking the import is what makes this about the guard.

    DockQ needs the same treatment for a different reason: it is no longer an
    import at all but an executable, so blocking ``sys.modules["DockQ"]``
    would prove nothing once it is installed. Emptying every route to the
    command is the equivalent.
    """
    protein = _protein(tmp_path)
    monkeypatch.setitem(sys.modules, "prolif", None)
    with pytest.raises(ix.ProLIFUnavailableError, match="prolif_unavailable"):
        ix.prolif_fingerprint(protein, protein)

    _no_dockq(monkeypatch, tmp_path)
    assert ix.dockq_command() is None
    available, reason = ix.dockq_available()
    assert not available
    assert "dockq_unavailable" in (reason or "")
    with pytest.raises(ix.DockQUnavailableError, match="dockq_unavailable"):
        ix.dockq(protein, protein)


def test_a_stale_dockq_variable_reads_as_absent(tmp_path, monkeypatch):
    """A variable pointing at a file that is not executable is not a command.

    Handing it to subprocess would turn a stale environment into a crash; the
    resolver drops it so the answer is "not installed".
    """
    dead = tmp_path / "DockQ"
    dead.write_text("not an executable\n", encoding="utf-8")
    dead.chmod(0o644)
    monkeypatch.setenv(ix.DOCKQ_COMMAND_ENV, str(dead))
    monkeypatch.delenv("DATA_PATH", raising=False)
    monkeypatch.setattr(ix.shutil, "which", lambda _name: None)
    assert ix.dockq_command() is None


def test_the_variable_wins_over_the_installed_environment(tmp_path, monkeypatch):
    chosen = tmp_path / "DockQ"
    chosen.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    chosen.chmod(0o755)
    monkeypatch.setenv(ix.DOCKQ_COMMAND_ENV, str(chosen))
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "elsewhere"))
    assert ix.dockq_command() == str(chosen)


def test_the_install_root_follows_data_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    assert ix.dockq_install_root() == tmp_path / ix.DOCKQ_SUBDIR
    monkeypatch.delenv("DATA_PATH", raising=False)
    assert ix.dockq_install_root() is None


def test_a_missing_structure_is_named_before_dockq_runs(tmp_path, monkeypatch):
    """Not DockQ's job to report a path the caller got wrong."""
    stub = tmp_path / "DockQ"
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    monkeypatch.setenv(ix.DOCKQ_COMMAND_ENV, str(stub))
    real = _protein(tmp_path)
    with pytest.raises(ix.StructureUnreadableError, match="dockq_model_absent"):
        ix.dockq(tmp_path / "ghost.pdb", real)
    with pytest.raises(ix.StructureUnreadableError, match="dockq_reference_absent"):
        ix.dockq(real, tmp_path / "ghost.pdb")


def test_the_quality_bands_are_the_authors_own():
    """0.23, 0.49 and 0.80, as the DockQ banner prints them."""
    assert ix.dockq_quality(1.0) == "high"
    assert ix.dockq_quality(0.80) == "high"
    assert ix.dockq_quality(0.799) == "medium"
    assert ix.dockq_quality(0.49) == "medium"
    assert ix.dockq_quality(0.489) == "acceptable"
    assert ix.dockq_quality(0.23) == "acceptable"
    assert ix.dockq_quality(0.229) == "incorrect"
    assert ix.dockq_quality(0.0) == "incorrect"


def test_verdicts_report_the_band_and_gate_only_on_a_caller_bar():
    record = {"global_dockq": 0.6, "quality": "medium"}
    loose = ix.dockq_verdicts(record)
    assert loose["dockq"]["state"] == "no_cut"
    assert loose["dockq"]["band"] == "medium"
    assert "btae586" in loose["dockq"]["band_source"]
    assert ix.dockq_verdicts(record, dockq_min=0.5)["dockq"]["state"] == "clear"
    assert ix.dockq_verdicts(record, dockq_min=0.7)["dockq"]["state"] == "hit"
    assert ix.dockq_verdicts(record, dockq_min=0.7)["dockq"]["threshold_source"].startswith("caller_supplied")


@pytest.mark.parametrize("status,score", [(0, 0.4899999), (0, float("nan")), (0, 1.01), (2, 0.8)])
def test_dockq_requires_successful_process_and_valid_unrounded_score(tmp_path, monkeypatch, status, score):
    import json
    import subprocess
    from pathlib import Path

    model = tmp_path / "model.pdb"
    model.touch()
    monkeypatch.setattr(ix, "dockq_command", lambda: "DockQ")
    monkeypatch.setattr(ix, "dockq_version", lambda _: "fixture")

    def run(argv, **kwargs):
        report = {"GlobalDockQ": score, "best_dockq": score, "best_result": {"AB": {"DockQ": score}},
                  "best_mapping_str": "AB:AB"}
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(report))
        return subprocess.CompletedProcess(argv, status, "", "")

    monkeypatch.setattr(ix.subprocess, "run", run)
    if status or not 0 <= score <= 1:
        with pytest.raises(ix.DockQFailedError):
            ix.dockq(model, model)
    else:
        result = ix.dockq(model, model)
        assert ix.dockq_verdicts(result, 0.49)["dockq"]["state"] == "hit"
    assert ix.dockq_verdicts({"global_dockq": None}, 0.49)["dockq"]["state"] == "untested"


# --------------------------------------------------------------- DockQ, really run

_BACKBONE = (("N", -0.5, 0.0, 0.0), ("CA", 0.0, 0.0, 0.0), ("C", 0.7, 0.9, 0.0), ("O", 0.7, 2.1, 0.0))

needs_dockq = pytest.mark.skipif(
    not ix.dockq_available()[0],
    reason="run scripts/install_dockq.py, or set DOCKQ_EXECUTABLE, to run DockQ itself",
)


def _complex_pdb(path, offsets, residues=12):
    """Poly-alanine strands, one per chain offset, as a PDB DockQ can score.

    A single CA per chain is not enough for DockQ -- it needs residues it can
    align and an interface to find -- so each chain gets a real N/CA/C/O
    backbone running along x, and the offsets put the strands in contact.
    """
    atoms = []
    for chain, (ox, oy, oz) in offsets.items():
        for index in range(residues):
            for name, dx, dy, dz in _BACKBONE:
                atoms.append(
                    (name, "ALA", chain, index + 1, index * 3.8 + dx + ox, dy + oy, dz + oz, name[0])
                )
    return _write_pdb(path, atoms)


@needs_dockq
def test_dockq_scores_an_identical_complex_as_one(tmp_path):
    native = _complex_pdb(tmp_path / "native.pdb", {"A": (0.0, 0.0, 0.0), "B": (0.0, 5.0, 0.0)})
    record = ix.dockq(native, native, no_align=True)
    assert record["method"] == ix.METHOD_DOCKQ
    assert record["interface_count"] == 1
    assert record["global_dockq"] == pytest.approx(1.0, abs=1e-4)
    assert record["quality"] == "high"
    assert record["mapping"] == "AB:AB"
    assert record["interfaces"]["AB"]["fnat"] == pytest.approx(1.0, abs=1e-6)
    assert record["interfaces"]["AB"]["iRMSD"] == pytest.approx(0.0, abs=1e-3)
    assert "unknown" not in str(record["version"]), "the version should be readable beside the executable"


@needs_dockq
def test_dockq_marks_a_displaced_chain_down(tmp_path):
    native = _complex_pdb(tmp_path / "native.pdb", {"A": (0.0, 0.0, 0.0), "B": (0.0, 5.0, 0.0)})
    moved = _complex_pdb(tmp_path / "moved.pdb", {"A": (0.0, 0.0, 0.0), "B": (1.5, 5.0, 1.0)})
    record = ix.dockq(moved, native, no_align=True)
    assert record["global_dockq"] < 1.0
    assert record["interfaces"]["AB"]["fnat"] < 1.0
    assert record["interfaces"]["AB"]["iRMSD"] > 0.0


@needs_dockq
def test_the_sum_passes_one_on_three_interfaces_and_the_global_score_does_not(tmp_path):
    """The distinction the record is named for.

    DockQ's JSON ``best_dockq`` adds the interfaces up, so a three-interface
    identity comparison gives 3.0 -- measured, not supposed. Reporting that as
    "the DockQ score" would read as a scale error. ``global_dockq`` is the
    0-to-1 figure the CLI prints, and it stays at 1.0.
    """
    triangle = {"A": (0.0, 0.0, 0.0), "B": (0.0, 5.0, 0.0), "C": (0.0, 2.5, 4.33)}
    native = _complex_pdb(tmp_path / "trimer.pdb", triangle)
    record = ix.dockq(native, native, no_align=True)
    assert record["interface_count"] == 3
    assert record["global_dockq"] == pytest.approx(1.0, abs=1e-4)
    assert record["dockq_sum"] == pytest.approx(3.0, abs=1e-3)
    assert record["dockq_sum"] > 1.0
    assert set(record["interfaces"]) == {"AB", "AC", "BC"}


@needs_dockq
def test_a_single_chain_input_fails_loudly_rather_than_silently(tmp_path):
    """DockQ exits 0 and writes no JSON here, so the status cannot be trusted.

    Measured: given one chain it prints "Need at least two chains in the two
    inputs", writes nothing, and returns 0. A wrapper reading the return code
    would report success with no score.
    """
    lone = _complex_pdb(tmp_path / "lone.pdb", {"A": (0.0, 0.0, 0.0)})
    with pytest.raises(ix.DockQFailedError, match="dockq_no_result"):
        ix.dockq(lone, lone, no_align=True)


@needs_dockq
def test_the_record_is_json_serialisable(tmp_path):
    """The old in-process route returned tuple-keyed interfaces, which are not.

    ``run_on_all_native_interfaces`` keys its result by a ``(chain, chain)``
    tuple; ``json.dumps`` refuses those as keys, so a record built from it
    could not be written to a report. DockQ's own ``--json`` uses "AB".
    """
    import json as _json

    native = _complex_pdb(tmp_path / "native.pdb", {"A": (0.0, 0.0, 0.0), "B": (0.0, 5.0, 0.0)})
    record = ix.dockq(native, native, no_align=True)
    text = _json.dumps(record)
    assert '"global_dockq"' in text
    assert all(isinstance(key, str) for key in record["interfaces"])


@needs_dockq
def test_a_mapping_limits_the_comparison_to_one_interface(tmp_path):
    triangle = {"A": (0.0, 0.0, 0.0), "B": (0.0, 5.0, 0.0), "C": (0.0, 2.5, 4.33)}
    native = _complex_pdb(tmp_path / "trimer.pdb", triangle)
    record = ix.dockq(native, native, mapping=":AB", no_align=True)
    assert record["interface_count"] == 1
    assert set(record["interfaces"]) == {"AB"}
    assert record["dockq_sum"] == pytest.approx(1.0, abs=1e-3)


def _stacked_over_phe(tmp_path, seed=11):
    """A benzene parallel to the only Phe ring, 3.6 A along that ring's normal.

    The geometry has to be built, not guessed. ProLIF applies real distance and
    angle criteria, so a ligand merely dropped near a ring -- which the native
    heavy-atom proxy happily counts -- is correctly rejected by it.
    """
    peptide = Chem.AddHs(Chem.MolFromSequence("AFA"))
    AllChem.EmbedMolecule(peptide, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(peptide)
    conformer = peptide.GetConformer()
    ring = [atom.GetIdx() for atom in peptide.GetAtoms() if atom.GetIsAromatic()]
    coords = np.array([list(conformer.GetAtomPosition(i)) for i in ring])
    centre = coords.mean(axis=0)
    normal = np.linalg.svd(coords - centre)[2][2]

    ligand = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(ligand, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(ligand)
    lconf = ligand.GetConformer()
    positions = np.array([list(lconf.GetAtomPosition(i)) for i in range(ligand.GetNumAtoms())])
    lcentre = positions.mean(axis=0)
    lnormal = np.linalg.svd(positions - lcentre)[2][2]
    axis = np.cross(lnormal, normal)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(lnormal, normal))
    if sine < 1e-8:
        rotation = np.eye(3)
    else:
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / sine
        rotation = np.eye(3) + skew * sine + skew @ skew * (1 - cosine)
    target = centre + normal * 3.6
    for index in range(ligand.GetNumAtoms()):
        moved = rotation @ (positions[index] - lcentre) + target
        lconf.SetAtomPosition(index, Point3D(*[float(value) for value in moved]))

    protein_path = tmp_path / "stack_protein.pdb"
    ligand_path = tmp_path / "stack_ligand.pdb"
    Chem.MolToPDBFile(peptide, str(protein_path))
    Chem.MolToPDBFile(ligand, str(ligand_path))
    return protein_path, ligand_path


def test_prolif_reads_a_real_stacked_pose_when_installed(tmp_path):
    """ProLIF's own typed interactions, where ProLIF is available.

    Skipped rather than faked when it is absent: the native proxy is a
    different measurement, and standing in for ProLIF here would test nothing.
    """
    pytest.importorskip("prolif")
    protein, ligand = _stacked_over_phe(tmp_path)
    fingerprint = ix.prolif_fingerprint(protein, ligand)
    assert fingerprint.engine == "prolif"
    assert fingerprint.method == ix.METHOD_PROLIF
    assert fingerprint.counts["pi_stacking"] > 0, "a ring parallel at 3.6 A is a stack"
    assert fingerprint.counts["hydrophobic"] > 0
    assert set(fingerprint.counts) == set(
        ix.INTERACTION_TYPES
    ), "ProLIF also reports VdWContact; only the five tracked types may appear"


def test_the_two_engines_are_never_reported_as_the_same_measurement(tmp_path):
    """They count different things, so they must not share a method id.

    On the same stacked pose ProLIF reports one typed hydrophobic interaction
    where the native engine reports dozens of heavy-atom pairs. Both are
    defensible; reading one as the other is not, which is why the method id
    travels with every count.
    """
    pytest.importorskip("prolif")
    protein, ligand = _stacked_over_phe(tmp_path)
    typed = ix.prolif_fingerprint(protein, ligand)
    native = ix.native_fingerprint(protein, Chem.MolFromPDBFile(str(ligand), removeHs=False))
    assert typed.method != native.method
    assert typed.engine != native.engine
    assert (
        native.counts["hydrophobic"] > typed.counts["hydrophobic"]
    ), "the native proxy counts atom pairs, so it should be the larger number"


def test_the_prolif_entry_point_rejects_sdf_plainly(tmp_path):
    """A real limit of this entry point, recorded so it is not rediscovered.

    ``prolif_fingerprint`` opens the ligand with ``mda.Universe``, and
    MDAnalysis ships no SDF reader -- so the ligand must be PDB, MOL2 or PDBQT
    even though ProLIF itself reads SDF through its own supplier. What matters
    for safety is that it fails loudly rather than returning empty counts.
    """
    pytest.importorskip("prolif")
    protein, ligand = _stacked_over_phe(tmp_path)
    sdf = tmp_path / "ligand.sdf"
    Chem.MolToMolFile(Chem.MolFromPDBFile(str(ligand), removeHs=False), str(sdf))
    with pytest.raises(Exception) as caught:  # noqa: PT011 - MDAnalysis raises ValueError
        ix.prolif_fingerprint(protein, sdf)
    assert not isinstance(
        caught.value, ix.ProLIFUnavailableError
    ), "an unreadable ligand must not be reported as a missing ProLIF"


def test_prolif_cli_accepts_one_rdkit_pose_and_honours_the_engine(tmp_path, capsys):
    import json

    from scripts.screen_pose_safety import main

    pytest.importorskip("prolif")
    protein, ligand = _stacked_over_phe(tmp_path)
    mol = Chem.MolFromPDBFile(str(ligand), removeHs=False)
    sdf = tmp_path / "pose.sdf"
    Chem.MolToMolFile(mol, str(sdf))
    assert main(["ifp", str(protein), str(sdf), "--engine", "prolif"]) == 0
    record = json.loads(capsys.readouterr().out)["measured"]
    assert record["method"] == ix.METHOD_PROLIF and record["counts"]["pi_stacking"] == 1
    assert ix.fingerprint(protein, mol, prefer_prolif=True).engine == "prolif"


def test_prolif_retains_directional_interactions_and_rejects_missing_results(tmp_path, monkeypatch):
    import pandas as pd

    prolif = pytest.importorskip("prolif")
    protein, ligand = _stacked_over_phe(tmp_path)
    names = ("Anionic", "Cationic", "XBAcceptor", "XBDonor")
    frame = pd.DataFrame([[True, False, True, True]],
                         columns=pd.MultiIndex.from_tuples([("lig", "res", name) for name in names]))
    monkeypatch.setattr(prolif.Fingerprint, "to_dataframe", lambda _: frame)
    counts = ix.prolif_fingerprint(protein, ligand).counts
    assert counts["salt_bridge"] == 1 and counts["halogen"] == 2
    monkeypatch.setattr(prolif.Fingerprint, "to_dataframe", lambda _: frame.iloc[:0])
    with pytest.raises(ix.StructureUnreadableError, match="missing_pose_result"):
        ix.prolif_fingerprint(protein, ligand)
