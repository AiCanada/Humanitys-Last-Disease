"""P2Rank's receptor, prediction table, and screen output stay linked."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from rfproteina.disease import p2rank, p2rank_runtime

_HEADER = (
    b"name     , rank,  score, probability, sas_points, surf_atoms, center_x, center_y, center_z, residue_ids, "
    b"surf_atom_ids\n"
)
_FIRST = b"pocket1  ,    1,  12.25,       1.000,         24,          3,    2.000,   -3.100,    4.250, A_1 A_2, 1 2 5\n"
_SECOND = b"pocket2  ,    2,   1.50,       0.000,         10,          2,   -5.000,    1.700,    0.000, A_3 A_4, 9 13\n"
_RESIDUES = frozenset(f"A_{index}" for index in range(1, 21))


def _pdb(*, residues=20, zero_b=False):
    """Small artificial coordinates with enough standard residues for the input contract."""
    atoms = []
    for index in range(1, residues + 1):
        for offset, (name, element) in enumerate((("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"))):
            serial = len(atoms) + 1
            x, y, z = 3.8 * (index - 1) + 0.7 * offset, float(offset % 2), float(index % 3)
            b = 0.0 if zero_b else 14.0 + 0.4 * index + offset
            atoms.append(
                f"ATOM  {serial:5d} {name:^4s} ALA A{index:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b:6.2f}          {element:>2s}\n"
            )
    return ("".join(atoms) + "TER\nEND\n").encode("ascii")


def _manifest(directory, **overrides):
    settings = {"version": 1, "receptor": "receptor.pdb", "structure_source": "xray", **overrides}
    (directory / p2rank.MANIFEST).write_text(json.dumps(settings), encoding="utf-8")


@pytest.fixture
def prepared(tmp_path):
    pytest.importorskip("gemmi")
    (tmp_path / "receptor.pdb").write_bytes(_pdb())
    _manifest(tmp_path)
    return tmp_path


def _forbid_provider(*_args):
    pytest.fail("invalid or absent receptor preparation reached P2Rank")


def test_release_csv_parses_padded_fields_and_boundary_probabilities():
    pockets = p2rank.parse_predictions(_HEADER + _FIRST + _SECOND, _RESIDUES)
    assert [pocket["rank"] for pocket in pockets] == [1, 2]
    assert [pocket["probability"] for pocket in pockets] == [1.0, 0.0]
    assert [pocket["score"] for pocket in pockets] == [12.25, 1.5]
    assert list(pockets[0]["center_angstrom"]) == [2.0, -3.1, 4.25]
    assert pockets[0]["sas_points"] == 24 and pockets[0]["surface_atoms"] == 3
    assert list(pockets[0]["residue_ids"]) == ["A_1", "A_2"]
    assert p2rank.parse_predictions(_HEADER, _RESIDUES) == []


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        _HEADER.replace(b"probability", b"score") + _FIRST,
        _HEADER.replace(b"rank,", b"rank_missing,") + _FIRST,
        _HEADER + _FIRST + _FIRST,
        _HEADER + _FIRST + _SECOND.replace(b"    2,", b"    4,", 1),
        _HEADER + _FIRST + _SECOND.replace(b"pocket2", b"pocket1"),
        _HEADER + _FIRST.replace(b"12.25", b"NaN"),
        _HEADER + _FIRST.replace(b"-3.100", b"1e1000"),
        _HEADER + _FIRST.replace(b"1.000", b"1.001"),
        _HEADER + _FIRST.replace(b"1.000", b"-0.01"),
        _HEADER + _FIRST.replace(b"A_1 A_2", b"A_1 B_2"),
        _HEADER + _FIRST.rsplit(b",", 1)[0] + b"\n",
        _HEADER + _FIRST.rstrip(b"\n") + b",surplus\n",
        _HEADER + _FIRST.replace(b"A_1 A_2", b"\xff"),
    ],
)
def test_malformed_or_unattributable_predictions_fail_as_one_table(payload):
    with pytest.raises(ValueError):
        p2rank.parse_predictions(payload, _RESIDUES)


@pytest.mark.parametrize(
    ("source", "profile"),
    [
        ("xray", "default"),
        ("alphafold", "alphafold"),
        ("predicted", "alphafold"),
        ("cryoem", "alphafold"),
        ("nmr", "alphafold"),
    ],
)
def test_prepare_links_exact_receptor_bytes_and_supported_profile(prepared, source, profile):
    _manifest(prepared, structure_source=source)
    path = prepared / "receptor.pdb"
    first = p2rank.prepare(prepared)
    assert first.profile == profile
    assert first.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    path.write_bytes(path.read_bytes().replace(b" 14.40", b" 15.40", 1))
    second = p2rank.prepare(prepared)
    assert second.profile == profile and second.sha256 != first.sha256


@pytest.mark.parametrize(
    "bad_manifest",
    [
        None,
        b"{}",
        b'{"version":true,"receptor":"receptor.pdb","structure_source":"xray"}',
        b'{"version":1,"receptor":"receptor.pdb","receptor":"other.pdb","structure_source":"xray"}',
        b'{"version":1,"receptor":"receptor.pdb","structure_source":"other"}',
        b'{"version":1,"receptor":"../receptor.pdb","structure_source":"xray"}',
        b'{"version":1,"receptor":"/receptor.pdb","structure_source":"xray"}',
        b'{"version":1,"receptor":"C:/temp/receptor.pdb","structure_source":"xray"}',
        b'{"version":1,"receptor":"missing.pdb","structure_source":"xray"}',
        b'{"version":1,"receptor":"receptor.pdb","structure_source":"xray","profile":"default"}',
    ],
)
def test_missing_or_invalid_manifest_never_launches_provider(prepared, monkeypatch, bad_manifest):
    manifest = prepared / p2rank.MANIFEST
    manifest.unlink()
    if bad_manifest is not None:
        manifest.write_bytes(bad_manifest)
    monkeypatch.setattr(p2rank, "_predict", _forbid_provider)
    result = p2rank.screen_run(prepared, candidate="named")
    assert result["status"] == "untested" and result["blocker"] and not result.get("pockets")


def test_short_or_symlinked_receptor_is_refused(prepared, monkeypatch):
    monkeypatch.setattr(p2rank, "_predict", _forbid_provider)
    path = prepared / "receptor.pdb"
    path.write_bytes(_pdb(residues=19))
    short = p2rank.screen_run(prepared)
    assert short["status"] == "untested" and short["blocker"]

    path.unlink()
    target = prepared / "other.pdb"
    target.write_bytes(_pdb())
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    alias = p2rank.screen_run(prepared)
    assert alias["status"] == "untested" and alias["blocker"]


def test_source_profile_requires_matching_structure_metadata_and_b_factors(prepared, monkeypatch):
    monkeypatch.setattr(p2rank, "_predict", _forbid_provider)
    path = prepared / "receptor.pdb"
    path.write_bytes(_pdb(zero_b=True))
    assert p2rank.screen_run(prepared)["blocker"] == "p2rank_xray_b_factors_missing"
    _manifest(prepared, structure_source="predicted")
    assert p2rank.prepare(prepared).profile == "alphafold"

    path.write_bytes(b"EXPDTA    ELECTRON MICROSCOPY\n" + _pdb(zero_b=True))
    assert p2rank.screen_run(prepared)["blocker"] == "p2rank_structure_source_mismatch"
    _manifest(prepared, structure_source="cryoem")
    assert p2rank.prepare(prepared).profile == "alphafold"


def test_receptor_result_keeps_empty_and_failed_predictions_distinct(prepared, monkeypatch):
    calls = []

    def predict(receptor):
        calls.append((receptor.sha256, receptor.profile))
        return _HEADER

    monkeypatch.setattr(p2rank, "_predict", predict)
    empty = p2rank.screen_run(prepared)
    assert empty["status"] == "complete" and empty["pockets"] == []
    assert empty.get("blocker") is None
    assert calls == [(hashlib.sha256((prepared / "receptor.pdb").read_bytes()).hexdigest(), "default")]

    monkeypatch.setattr(p2rank, "_predict", lambda _receptor: b"")
    failed = p2rank.screen_run(prepared)
    assert failed["status"] == "untested" and failed["blocker"] and not failed.get("pockets")


@pytest.mark.parametrize(
    ("output", "returncode", "supported"),
    [
        (b'openjdk version "17.0.12"', 0, True),
        (b'java version "21.0.5"', 0, True),
        (b'openjdk version "11.0.24"', 0, False),
        (b'openjdk version "25.0.1"', 0, False),
        (b'openjdk version "17.0.12"', 1, False),
        (b"Java support is unavailable", 0, False),
    ],
)
def test_java_probe_requires_supported_version_and_controlled_environment(
    tmp_path, monkeypatch, output, returncode, supported,
):
    exe = tmp_path / "java"

    def process(argv, **kwargs):
        assert argv == [str(exe), "-version"] and kwargs["timeout"] > 0
        assert {"JAVA_OPTS", "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS", "CLASSPATH"}.isdisjoint(
            kwargs["env"]
        )
        return SimpleNamespace(stdout=output, returncode=returncode)

    for variable in ("JAVA_OPTS", "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS", "CLASSPATH"):
        monkeypatch.setenv(variable, "ambient-test-value")
    monkeypatch.setattr(p2rank_runtime.subprocess, "run", process)
    assert p2rank_runtime._supported_java(exe, (1, 2, 3, 4, 5)) is supported


def test_wrong_java_keeps_receptor_provenance_and_named_blocker(prepared, monkeypatch):
    monkeypatch.setattr(p2rank_runtime, "_verified_release", lambda: prepared)
    monkeypatch.setattr(p2rank_runtime, "_supported_java", lambda *_args: False)
    assert not p2rank_runtime.installed()
    result = p2rank.screen_run(prepared)
    assert result["status"] == "untested" and result["blocker"] == "p2rank_runtime_unavailable"
    assert result["receptor_sha256"] == hashlib.sha256(_pdb()).hexdigest()
    assert result["pockets"] == []


def test_pockets_belong_to_receptor_context_regardless_of_ligand_selection(prepared, monkeypatch):
    (prepared / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\na\tCCO\nb\tCCC\n", encoding="utf-8")
    monkeypatch.setattr(p2rank, "_predict", lambda _receptor: _HEADER + _FIRST + _SECOND)
    first = p2rank.screen_run(prepared, candidate="a")
    second = p2rank.screen_run(prepared, candidate="b")
    assert first["status"] == second["status"] == "complete"
    assert first["pockets"] == second["pockets"] == p2rank.parse_predictions(_HEADER + _FIRST + _SECOND, _RESIDUES)
    assert all("candidate_id" not in pocket for pocket in first["pockets"])
    assert first["scope"] == second["scope"] == "receptor"
    assert first["receptor_sha256"] == second["receptor_sha256"] == hashlib.sha256(_pdb()).hexdigest()


def test_saved_report_rejects_altered_provenance_and_pocket_fields(prepared, monkeypatch):
    monkeypatch.setattr(p2rank, "_predict", lambda _receptor: _HEADER + _FIRST)
    result = p2rank.screen_run(prepared)
    assert p2rank.report_view(result) == result
    (pocket,) = result["pockets"]
    for damage in (
        {"profile": "alphafold"},
        {"receptor_sha256": "unverified"},
        {"archive_sha256": "0" * 64},
        {"pockets": [{**pocket, "center_angstrom": [True, 1.0, 0.0]}]},
        {"pockets": [{**pocket, "probability": float("nan")}]},
        {"pockets": [{**pocket, "residue_ids": ["A_1", "A_1"]}]},
    ):
        projected = p2rank.report_view({**result, **damage})
        assert projected["status"] == "untested" and projected["blocker"] == "p2rank_invalid_output"
        assert projected["pockets"] == []


def test_picker_checks_receptor_without_running_java_and_standalone_reports_it(prepared, monkeypatch, capsys):
    from rfproteina.disease import screen_selection
    from scripts import screen_extras

    available = {p2rank.METHOD_ID: True}
    monkeypatch.setattr(p2rank, "_predict", _forbid_provider)
    arguments = ["p2rank", str(prepared)]
    (prepared / p2rank.MANIFEST).unlink()
    (missing,) = screen_selection.plan(prepared, "ligand", available=available)
    assert missing.installed and not missing.applicable and not missing.recommended
    assert screen_extras.main(arguments) == 2
    assert json.loads(capsys.readouterr().out)["blocker"] == "p2rank_manifest_missing"

    _manifest(prepared)
    (ready,) = screen_selection.plan(prepared, "ligand", available=available)
    assert ready.installed and ready.applicable and ready.recommended

    monkeypatch.setattr(p2rank, "_predict", lambda _receptor: _HEADER + _FIRST)
    assert screen_extras.main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == p2rank.screen_run(prepared)
