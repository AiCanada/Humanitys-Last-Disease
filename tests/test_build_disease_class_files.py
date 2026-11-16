"""Behaviour of scripts/build_disease_class_files.py, run as the user would run it."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rfproteina.disease.classes import load_class_files
from rfproteina.disease.combinations import HIT_NM, parse_class_filename
from tests.helpers import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "build_disease_class_files.py"


def _load_script():
    """Import the script once. Spawning it per case would re-import torch every time."""
    spec = importlib.util.spec_from_file_location("build_disease_class_files", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_script()


def _structures(root: Path, *names: str) -> Path:
    source = root / "src"
    source.mkdir(parents=True, exist_ok=True)
    for name in names:
        (source / name).write_text("data_X\n#\n", encoding="utf-8")
    return source


def _manifest(root: Path, name: str, *lines: str) -> Path:
    path = root / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _Result:
    """The parts of a finished run the tests assert on."""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.fixture
def run(capsys, monkeypatch):
    """Call the script's main() in-process and capture what it printed."""

    def _call(*args: str) -> _Result:
        monkeypatch.setattr(sys, "argv", ["build_disease_class_files.py", *args])
        capsys.readouterr()
        code = BUILD.main()
        captured = capsys.readouterr()
        return _Result(code, captured.out, captured.err)

    return _call


def test_a_state_column_builds_a_directory_the_training_loader_reads(tmp_path: Path, run):
    _structures(tmp_path, "a.cif", "b.cif", "c.cif", "d.cif")
    manifest = _manifest(
        tmp_path,
        "states.csv",
        "path,kind,axis,state",
        "src/a.cif,ligand,herg,toxic",
        "src/b.cif,ligand,herg,clean",
        "src/c.cif,protein,mhc_i,clean",
        "src/d.cif,protein,mhc_i,hit",
    )
    out = tmp_path / "classes"
    result = run(str(manifest), "--out-dir", str(out), "--copy")
    assert result.returncode == 0, result.stderr

    names = sorted(path.name for path in out.glob("*.cif"))
    assert names == [
        "ligand_herg_clean_0001.cif",
        "ligand_herg_toxic_0001.cif",
        "protein_mhc_i_clean_0001.cif",
        "protein_mhc_i_toxic_0001.cif",
    ]
    # The point of the script: every name it writes is one the training loader can read.
    loaded = load_class_files(out)
    assert len(loaded) == 4
    assert all(item.binder_kind and item.axis and item.state and not item.coarse for item in loaded)


def test_a_nm_reading_is_judged_at_the_projects_own_threshold(tmp_path: Path, run):
    _structures(tmp_path, "a.cif", "b.cif")
    manifest = _manifest(
        tmp_path,
        "readings.csv",
        "file,nm",
        f"src/a.cif,{HIT_NM - 1}",
        f"src/b.cif,{HIT_NM + 1}",
    )
    out = tmp_path / "classes"
    assert run(str(manifest), "--out-dir", str(out), "--kind", "ligand", "--axis", "herg").returncode == 0
    names = sorted(path.name for path in out.glob("*.cif"))
    assert names == ["ligand_herg_clean_0001.cif", "ligand_herg_toxic_0001.cif"]
    # At or below the threshold is a hit, so the stronger binder is the toxic one.
    provenance = json.loads((out / "class_files_provenance.json").read_text(encoding="utf-8"))
    assert len(provenance["runs"]) == 1
    by_file = {row["class_file"]: row for row in provenance["runs"][0]["files"]}
    assert by_file["ligand_herg_toxic_0001.cif"]["source"].endswith("a.cif")
    assert by_file["ligand_herg_clean_0001.cif"]["source"].endswith("b.cif")


def test_a_coarse_bucket_needs_an_axis_to_become_a_contrast(tmp_path: Path, run):
    _structures(tmp_path, "a.cif", "b.cif")
    manifest = _manifest(
        tmp_path,
        "buckets.csv",
        "filename,bucket",
        "src/a.cif,high_affinity_clean",
        "src/b.cif,high_affinity_toxic",
    )
    out = tmp_path / "classes"
    # Without an axis the bucket says nothing a training run can use.
    missing = run(str(manifest), "--out-dir", str(out), "--kind", "protein")
    assert missing.returncode == 1
    assert "no liability axis" in missing.stderr
    assert not out.exists()

    named = run(str(manifest), "--out-dir", str(out), "--kind", "protein", "--bucket-axis", "mhc_i")
    assert named.returncode == 0, named.stderr
    assert sorted(path.name for path in out.glob("*.cif")) == [
        "protein_mhc_i_clean_0001.cif",
        "protein_mhc_i_toxic_0001.cif",
    ]


def test_an_invalid_manifest_writes_nothing_and_names_every_reason(tmp_path: Path, run):
    source = _structures(tmp_path, "a.cif", "b.cif", "c.cif")
    (source / "e.pdb").write_text("ATOM\n", encoding="utf-8")
    manifest = _manifest(
        tmp_path,
        "bad.csv",
        "path,kind,axis,state",
        "src/a.cif,ligand,herg,toxic",
        "src/e.pdb,ligand,herg,clean",
        "src/b.cif,ligand,immunogenicity,clean",
        "src/c.cif,ligand,herg,maybe",
        "src/missing.cif,ligand,herg,clean",
    )
    out = tmp_path / "classes"
    result = run(str(manifest), "--out-dir", str(out))
    assert result.returncode == 1
    assert not out.exists(), "a rejected manifest must not leave a half-built directory"
    # Each row is diagnosed on its own terms rather than by a single generic failure.
    assert "e.pdb is a PDB encoding" in result.stderr
    assert "immunogenicity is not a ligand axis" in result.stderr
    assert "'maybe' is not one of clean, clear, toxic, hit" in result.stderr
    assert "missing.cif is not a file" in result.stderr


def test_skip_invalid_builds_from_the_rest(tmp_path: Path, run):
    _structures(tmp_path, "a.cif", "b.cif")
    manifest = _manifest(
        tmp_path,
        "mixed.csv",
        "path,kind,axis,state",
        "src/a.cif,ligand,herg,toxic",
        "src/nope.cif,ligand,herg,clean",
        "src/b.cif,ligand,herg,clean",
    )
    out = tmp_path / "classes"
    result = run(str(manifest), "--out-dir", str(out), "--skip-invalid")
    assert result.returncode == 0, result.stderr
    assert len(list(out.glob("*.cif"))) == 2


def test_a_one_sided_axis_builds_but_warns(tmp_path: Path, run):
    _structures(tmp_path, "a.cif", "b.cif")
    manifest = _manifest(
        tmp_path,
        "onesided.csv",
        "path,kind,axis,state",
        "src/a.cif,ligand,herg,toxic",
        "src/b.cif,ligand,herg,hit",
    )
    out = tmp_path / "classes"
    result = run(str(manifest), "--out-dir", str(out))
    assert result.returncode == 0, result.stderr
    # A label the model never sees contrasted teaches it nothing, so say so.
    assert "only one side" in result.stdout
    assert "ligand/herg" in result.stdout


def test_an_existing_directory_is_not_overwritten(tmp_path: Path, run):
    _structures(tmp_path, "a.cif")
    manifest = _manifest(tmp_path, "one.csv", "path,kind,axis,state", "src/a.cif,ligand,herg,toxic")
    out = tmp_path / "classes"
    out.mkdir()
    (out / "keep.txt").write_text("existing", encoding="utf-8")

    refused = run(str(manifest), "--out-dir", str(out))
    assert refused.returncode == 2
    assert "already has files" in refused.stderr
    assert (out / "keep.txt").read_text(encoding="utf-8") == "existing"

    forced = run(str(manifest), "--out-dir", str(out), "--force")
    assert forced.returncode == 0, forced.stderr
    assert (out / "keep.txt").exists(), "--force adds to the directory, it does not clear it"

    # A second build must not quietly replace the first run's file with a differently labelled one.
    again = run(str(manifest), "--out-dir", str(out), "--force")
    assert again.returncode == 0, again.stderr
    assert sorted(path.name for path in out.glob("*.cif")) == [
        "ligand_herg_toxic_0001.cif",
        "ligand_herg_toxic_0002.cif",
    ]
    provenance = json.loads((out / "class_files_provenance.json").read_text(encoding="utf-8"))
    assert len(provenance["runs"]) == 2, "each run keeps its own provenance"


def test_a_dry_run_reports_without_writing(tmp_path: Path, run):
    _structures(tmp_path, "a.cif")
    manifest = _manifest(tmp_path, "one.csv", "path,kind,axis,state", "src/a.cif,ligand,herg,toxic")
    out = tmp_path / "classes"
    result = run(str(manifest), "--out-dir", str(out), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "nothing was written" in result.stdout
    assert not out.exists()


@pytest.mark.parametrize("suffix", [".cif", ".cif.zst", ".bcif.gz"])
def test_every_generated_name_reads_back_as_what_it_claims(tmp_path: Path, suffix: str, run):
    _structures(tmp_path, f"a{suffix}")
    manifest = _manifest(tmp_path, "one.csv", "path,kind,axis,state", f"src/a{suffix},protein,plasma_hub,clean")
    out = tmp_path / "classes"
    assert run(str(manifest), "--out-dir", str(out)).returncode == 0
    written = next(path for path in out.iterdir() if path.name != "class_files_provenance.json")
    assert written.name.endswith(suffix)
    parsed = parse_class_filename(written.name)
    assert (parsed.binder_kind, parsed.axis, parsed.state) == ("protein", "plasma_hub", "clear")


def test_it_runs_as_a_program_and_sets_an_exit_code(tmp_path: Path):
    """One end-to-end spawn, so the entry point and its exit codes are covered for real.

    The rest of the cases call main() in-process, because every spawn re-imports torch.
    """
    _structures(tmp_path, "a.cif")
    good = _manifest(tmp_path, "one.csv", "path,kind,axis,state", "src/a.cif,ligand,herg,toxic")
    out = tmp_path / "classes"
    finished = subprocess.run(
        [sys.executable, str(SCRIPT), str(good), "--out-dir", str(out)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert finished.returncode == 0, finished.stderr
    assert (out / "ligand_herg_toxic_0001.cif").exists()

    bad = _manifest(tmp_path, "bad.csv", "path,kind,axis,state", "src/gone.cif,ligand,herg,toxic")
    refused = subprocess.run(
        [sys.executable, str(SCRIPT), str(bad), "--out-dir", str(tmp_path / "none")],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert refused.returncode == 1
    assert "is not a file" in refused.stderr


def test_a_thousands_comma_is_refused_rather_than_truncated(tmp_path: Path, run):
    """`48,000` is two CSV fields. Reading it as 48 would invert the label silently."""
    _structures(tmp_path, "a.cif")
    manifest = _manifest(tmp_path, "m.csv", "path,nm", "src/a.cif,48,000")
    out = tmp_path / "classes"
    result = run(str(manifest), "--out-dir", str(out), "--kind", "ligand", "--axis", "herg")
    assert result.returncode == 1
    assert "more fields than the header" in result.stderr
    assert not out.exists()

    # The same reading, quoted, is the clean one it should always have been.
    quoted = _manifest(tmp_path, "q.csv", "path,nm", '"src/a.cif","48000"')
    assert run(str(quoted), "--out-dir", str(out), "--kind", "ligand", "--axis", "herg").returncode == 0
    assert [path.name for path in out.glob("*.cif")] == ["ligand_herg_clean_0001.cif"]


def test_a_short_row_is_refused(tmp_path: Path, run):
    _structures(tmp_path, "a.cif")
    manifest = _manifest(tmp_path, "m.csv", "path,kind,axis,state", "src/a.cif,ligand,herg")
    result = run(str(manifest), "--out-dir", str(tmp_path / "classes"))
    assert result.returncode == 1
    assert "fewer fields than the header" in result.stderr


def test_a_spreadsheet_byte_order_mark_does_not_hide_the_first_column(tmp_path: Path, run):
    """Excel writes UTF-8 with a BOM, which would otherwise make the first column invisible."""
    _structures(tmp_path, "a.cif")
    manifest = tmp_path / "bom.csv"
    manifest.write_text("state,path,kind,axis\nclean,src/a.cif,ligand,herg\n", encoding="utf-8-sig")
    out = tmp_path / "classes"
    result = run(str(manifest), "--out-dir", str(out))
    assert result.returncode == 0, result.stderr
    # Without the fix the state column vanished and the row fell through to another source.
    assert [path.name for path in out.glob("*.cif")] == ["ligand_herg_clean_0001.cif"]


def test_two_headers_meaning_the_same_field_are_refused(tmp_path: Path, run):
    _structures(tmp_path, "a.cif")
    manifest = _manifest(tmp_path, "m.csv", "path,kind,axis,state,State", "src/a.cif,ligand,herg,toxic,clean")
    result = run(str(manifest), "--out-dir", str(tmp_path / "classes"))
    assert result.returncode == 1
    assert "both read as 'state'" in result.stderr


def test_a_manifest_that_contradicts_itself_is_refused(tmp_path: Path, run):
    _structures(tmp_path, "a.cif")
    manifest = _manifest(
        tmp_path,
        "m.csv",
        "path,kind,axis,state",
        "src/a.cif,ligand,herg,clean",
        "src/a.cif,ligand,herg,toxic",
    )
    out = tmp_path / "classes"
    result = run(str(manifest), "--out-dir", str(out))
    assert result.returncode == 1
    assert "opposite outcomes" in result.stderr
    assert not out.exists(), "one structure cannot be both, so neither label is written"


def test_building_inside_the_structures_is_refused(tmp_path: Path, run):
    """An out-dir holding the sources could reach an original, so it is refused outright."""
    source = _structures(tmp_path, "a.cif")
    manifest = _manifest(tmp_path, "m.csv", "path,kind,axis,state", "src/a.cif,ligand,herg,toxic")
    result = run(str(manifest), "--out-dir", str(source), "--force", "--copy")
    assert result.returncode == 2
    assert "inside" in result.stderr
    assert (source / "a.cif").read_text(encoding="utf-8") == "data_X\n#\n", "the original is untouched"


def test_a_json_row_that_is_not_an_object_is_named(tmp_path: Path, run):
    _structures(tmp_path, "a.cif")
    manifest = tmp_path / "m.json"
    manifest.write_text("[1, 2]", encoding="utf-8")
    result = run(str(manifest), "--out-dir", str(tmp_path / "classes"))
    assert result.returncode == 2
    assert "row 1 is int, not an object" in result.stderr


def test_links_by_default_and_copies_on_request(tmp_path: Path, run):
    _structures(tmp_path, "a.cif")
    manifest = _manifest(tmp_path, "m.csv", "path,kind,axis,state", "src/a.cif,ligand,herg,toxic")

    linked = tmp_path / "linked"
    assert run(str(manifest), "--out-dir", str(linked)).returncode == 0
    link = linked / "ligand_herg_toxic_0001.cif"
    if link.is_symlink():
        assert link.resolve() == (tmp_path / "src" / "a.cif").resolve()
    else:
        pytest.skip("this filesystem does not allow symlinks, so the script copied instead")

    copied = tmp_path / "copied"
    assert run(str(manifest), "--out-dir", str(copied), "--copy").returncode == 0
    duplicate = copied / "ligand_herg_toxic_0001.cif"
    assert not duplicate.is_symlink()
    assert duplicate.read_text(encoding="utf-8") == (tmp_path / "src" / "a.cif").read_text(encoding="utf-8")
