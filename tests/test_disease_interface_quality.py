"""Interface terms the pass criteria do not carry, measured without PyRosetta.

Buried area, interface residue counts, polar satisfaction and Lawrence-Colman
shape complementarity are computed from geometry alone, so these run on a
hand-written PDB in about a second and need no model, no network and no weights.
What is checked is that the numbers move the way geometry says they must: two
chains pushed apart bury nothing and have no interface, two chains in contact
bury something, the KD-tree contact counts equal a brute-force count, and every
threshold reported carries the source it came from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfproteina.disease.interface_quality import (
    REFERENCE_THRESHOLDS,
    InterfaceUnreadableError,
    _angle_ok,
    _contacts,
    _polar_type,
    measure_interface,
    verdicts,
)

_RESIDUES = ("GLY", "SER", "ASN")


def _chain(chain: str, x_offset: float, serial: int) -> tuple[list[str], int]:
    """Three residues laid along y, each with N, CA, C and O."""
    lines: list[str] = []
    for index, name in enumerate(_RESIDUES):
        for atom, element, dx, dy, dz in (
            ("N", "N", 0.0, 0.0, 0.0),
            ("CA", "C", 1.4, 0.3, 0.0),
            ("C", "C", 2.4, -0.6, 0.4),
            ("O", "O", 3.0, -0.3, 1.4),
        ):
            x = x_offset + dx
            y = index * 3.5 + dy
            z = dz
            lines.append(
                f"ATOM  {serial:>5} {atom:<4}{name:>4} {chain}{index + 1:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}"
            )
            serial += 1
    return lines, serial


def _slab(chain: str, x0: float, serial: int, nx: int = 6, ny: int = 6) -> tuple[list[str], int]:
    """A flat grid of carbon atoms in the y-z plane at x=x0, for a packed interface."""
    lines: list[str] = []
    res = 1
    for iy in range(ny):
        for iz in range(nx):
            x, y, z = x0, iy * 2.0, iz * 2.0
            lines.append(
                f"ATOM  {serial:>5} {'CA':<4}{'GLY':>4} {chain}{res:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C"
            )
            serial += 1
            res += 1
    return lines, serial


def _complex(path: Path, separation: float) -> Path:
    first, serial = _chain("A", 0.0, 1)
    second, _ = _chain("B", separation, serial)
    path.write_text("\n".join([*first, *second, "TER", "END", ""]), encoding="utf-8")
    return path


def _slab_complex(path: Path, gap: float) -> Path:
    first, serial = _slab("A", 0.0, 1)
    second, _ = _slab("B", gap, serial)
    path.write_text("\n".join([*first, *second, "TER", "END", ""]), encoding="utf-8")
    return path


def test_two_chains_in_contact_bury_surface_and_name_interface_residues(tmp_path: Path) -> None:
    measured = measure_interface(_complex(tmp_path / "close.pdb", 4.5), "A")
    assert measured.dsasa_total_a2 > 0.0
    assert measured.interface_residues_binder >= 1
    assert measured.target_chains == ("B",)


def test_chains_pulled_far_apart_bury_nothing(tmp_path: Path) -> None:
    measured = measure_interface(_complex(tmp_path / "far.pdb", 60.0), "A")
    assert measured.dsasa_total_a2 == pytest.approx(0.0, abs=1e-6)
    assert measured.interface_residues_binder == 0
    assert measured.interface_polar_contacts == 0
    assert measured.buried_unsatisfied_polar_atoms == 0


def test_contact_counts_fall_as_the_chains_separate(tmp_path: Path) -> None:
    close = measure_interface(_complex(tmp_path / "a.pdb", 4.5), "A")
    apart = measure_interface(_complex(tmp_path / "b.pdb", 9.0), "A")
    assert close.interface_atom_contacts > apart.interface_atom_contacts
    assert close.dsasa_total_a2 > apart.dsasa_total_a2


def test_a_chain_that_is_not_there_is_refused_not_guessed(tmp_path: Path) -> None:
    structure = _complex(tmp_path / "close.pdb", 4.5)
    with pytest.raises(InterfaceUnreadableError, match="binder_chain_absent"):
        measure_interface(structure, "Z")
    with pytest.raises(InterfaceUnreadableError, match="target_chain_absent"):
        measure_interface(structure, "A", ["Q"])


def test_an_unreadable_file_raises_rather_than_reporting_zero(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdb"
    broken.write_text("not a structure at all\n", encoding="utf-8")
    with pytest.raises(InterfaceUnreadableError):
        measure_interface(broken, "A")


# --- shape complementarity ---------------------------------------------------


def test_shape_complementarity_is_a_number_in_range_or_none_with_reason(tmp_path: Path) -> None:
    record = measure_interface(_complex(tmp_path / "close.pdb", 4.5), "A").as_record()
    sc = record["shape_complementarity"]
    if sc is None:
        assert record["shape_complementarity_reason"]
    else:
        assert -1.0 <= float(sc) <= 1.0


def test_far_apart_chains_have_no_shape_complementarity(tmp_path: Path) -> None:
    measured = measure_interface(_complex(tmp_path / "far.pdb", 60.0), "A")
    assert measured.shape_complementarity is None
    assert measured.shape_complementarity_reason


def test_a_packed_slab_interface_has_a_positive_defined_sc(tmp_path: Path) -> None:
    """Two flat complementary layers 3.8 A apart should give a real, positive Sc."""
    measured = measure_interface(_slab_complex(tmp_path / "slab.pdb", 3.8), "A")
    assert measured.shape_complementarity is not None
    assert measured.shape_complementarity > 0.0
    assert measured.interface_dots_binder >= 10
    assert measured.interface_dots_target >= 10


# --- KD-tree contacts equal a brute-force count ------------------------------


def test_kdtree_contacts_match_a_brute_force_count(tmp_path: Path) -> None:
    from rfproteina.disease.interface_quality import CONTACT_A, POLAR_CONTACT_A, POLAR_ELEMENTS, _structure

    pair = _structure(_slab_complex(tmp_path / "slab.pdb", 3.8))
    binder = pair.chain_id == "A"
    polar, atom = _contacts(pair, binder)

    bc = pair.coord[binder].astype(float)
    tc = pair.coord[~binder].astype(float)
    d = np.linalg.norm(bc[:, None, :] - tc[None, :, :], axis=-1)
    atom_ref = int(np.sum(d <= CONTACT_A))
    bpol = np.isin(pair.element[binder], np.array(sorted(POLAR_ELEMENTS)))
    tpol = np.isin(pair.element[~binder], np.array(sorted(POLAR_ELEMENTS)))
    polar_ref = int(np.sum((d <= POLAR_CONTACT_A) & bpol[:, None] & tpol[None, :]))
    assert (polar, atom) == (polar_ref, atom_ref)


# --- buried-unsatisfied donor/acceptor typing --------------------------------


def test_polar_typing_matches_known_chemistry() -> None:
    assert _polar_type("ALA", "N", "N") == "d"  # backbone amide donor
    assert _polar_type("ALA", "O", "O") == "a"  # backbone carbonyl acceptor
    assert _polar_type("ASP", "OD1", "O") == "a"  # carboxylate acceptor
    assert _polar_type("LYS", "NZ", "N") == "d"  # ammonium donor
    assert _polar_type("SER", "OG", "O") == "b"  # hydroxyl both
    assert _polar_type("ALA", "CB", "C") is None  # carbon is not polar


def test_angle_filter_rejects_a_backward_approach() -> None:
    centre = np.array([0.0, 0.0, 0.0])
    anc = np.array([0.0, 0.0, -1.0])
    behind = np.array([0.0, 0.0, -3.0])  # same side as the antecedent -> ~0 deg
    front = np.array([0.0, 0.0, 3.0])  # opposite the antecedent -> ~180 deg
    assert _angle_ok(centre, front, anc, None) is True
    assert _angle_ok(centre, behind, anc, None) is False


def test_unsatisfied_total_is_the_sum_of_donors_and_acceptors(tmp_path: Path) -> None:
    measured = measure_interface(_complex(tmp_path / "close.pdb", 4.5), "A")
    assert (
        measured.buried_unsatisfied_polar_atoms
        == measured.buried_unsatisfied_donors + measured.buried_unsatisfied_acceptors
    )


# --- thresholds and provenance ----------------------------------------------


def test_every_threshold_reported_names_where_it_came_from(tmp_path: Path) -> None:
    called = verdicts(measure_interface(_complex(tmp_path / "close.pdb", 4.5), "A"))
    assert set(called) == set(REFERENCE_THRESHOLDS)
    for name, line in called.items():
        assert line["threshold_source"], name
        assert line["state"] in {"clear", "hit", "no_cut", "untested"}
    assert called["dsasa_total_a2"]["state"] == "no_cut"
    assert called["shape_complementarity"]["state"] in {"no_cut", "untested"}


def test_the_bindcraft_bars_are_marked_as_that_projects_defaults() -> None:
    source = str(REFERENCE_THRESHOLDS["interface_residues_binder"]["source"])
    assert "BindCraft" in source
    assert "not a published value" in source
    assert "DEFINITION DIFFERS" in str(REFERENCE_THRESHOLDS["interface_polar_contacts"]["source"])


def test_shape_complementarity_threshold_records_the_scale_mismatch() -> None:
    source = str(REFERENCE_THRESHOLDS["shape_complementarity"]["source"])
    assert "SCALE DIFFERS" in source
    assert "Lawrence" in source
    assert REFERENCE_THRESHOLDS["shape_complementarity"]["min"] is None


def test_calibration_numbers_are_recorded_in_the_sources() -> None:
    assert "Worker A calibration" in str(REFERENCE_THRESHOLDS["buried_unsatisfied_polar_atoms"]["source"])
    assert "0.166" in str(REFERENCE_THRESHOLDS["shape_complementarity"]["source"])
