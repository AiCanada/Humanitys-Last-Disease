"""Exercise PK-Sim release and model identity checks without touching an installed solver."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from rfproteina.disease import pksim_runtime as runtime
from scripts import install_pksim as installer

COMPOUND = "Ethanol"
KEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
OTHER_KEY = "KFZMGEQAYNKOFK-UHFFFAOYSA-N"
SIMULATION = "IV control, replicate 1"
OUTPUT = "Organism|Kidney|Ethanol|Tissue"
CSV = f'IndividualId,Time [min],"{OUTPUT} [µmol/l]"\n0,0,0\n0,1,1\n'.encode()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    # Independently configure the trusted cache regardless of the active checkout's environment.
    root = tmp_path / "trusted"
    root.mkdir()
    monkeypatch.setattr(runtime, "cache_path", lambda: root)
    monkeypatch.setattr(runtime, "_winpath", str)
    monkeypatch.setenv("WSL_INTEROP", "test")
    return root


@pytest.fixture
def snapshot(tmp_path):
    sim = {
        "Name": SIMULATION,
        "Individual": "Adult",
        "Compounds": [{"Name": COMPOUND, "Protocol": {"Name": "single"}}],
        "OutputSelections": [OUTPUT],
    }
    data = {
        "ApplicationName": "PK-Sim",
        "Name": "Prepared",
        "Individuals": [{"Name": "Adult", "OriginData": {"Species": "Human"}}],
        "Protocols": [{"Name": "single", "ApplicationType": "Intravenous"}],
        "Compounds": [{"Name": COMPOUND, "IsSmallMolecule": True}],
        "Simulations": [sim, {**sim, "Name": "Unselected case"}],
    }
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def release(tmp_path, monkeypatch):
    path = tmp_path / "portable.zip"
    files = {"PKSim.CLI.exe": b"stand-in CLI", "lib/numeric.dll": b"stand-in solver"}
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    monkeypatch.setattr(runtime, "ARCHIVE_SIZE", path.stat().st_size)
    monkeypatch.setattr(runtime, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(runtime, "FILES_SHA256", hashlib.sha256(runtime.canonical_json(manifest)).hexdigest())
    return path


@pytest.mark.parametrize("changed", ["archive length", "archive digest", "extracted manifest"])
def test_failed_verification_never_publishes_or_leaves_temporary_install(cache, release, monkeypatch, changed):
    if changed == "archive length":
        monkeypatch.setattr(runtime, "ARCHIVE_SIZE", release.stat().st_size + 1)
    elif changed == "archive digest":
        monkeypatch.setattr(runtime, "ARCHIVE_SHA256", "0" * 64)
    else:
        monkeypatch.setattr(runtime, "FILES_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="length mismatch|checksum mismatch|failed verification"):
        installer.install(release)
    assert not runtime.engine_path().exists()
    assert list(cache.iterdir()) == []


def test_installer_idempotence_and_tampered_dependency(cache, release):
    installed = installer.install(release)
    assert runtime.installed()
    assert installer.install(release) == installed
    # Keep the official manifest and replace one extracted solver component.
    component = installed / "lib" / "numeric.dll"
    component.write_bytes(b"modified solver library")
    assert not runtime.installed()
    with pytest.raises(ValueError, match="failed verification"):
        installer.install(release)
    assert component.read_bytes() == b"modified solver library"


def test_unexpected_library_beside_verified_solver_invalidates_installation(cache, release):
    root = installer.install(release)
    assert runtime.installed()
    (root / "extra.dll").write_bytes(b"outside the pinned release")
    assert not runtime.installed()


def test_native_windows_solver_must_be_able_to_open_the_cache(cache, release, monkeypatch):
    root = installer.install(release)

    def inaccessible(path):
        assert path == root
        raise runtime.PKSimError("pksim_runtime_unavailable")

    monkeypatch.setattr(runtime, "_winpath", inaccessible)
    assert not runtime.installed()


def test_registration_is_idempotent_and_cannot_rebind_an_existing_model(cache, snapshot):
    digest = installer.register(snapshot, COMPOUND, KEY)
    assert digest == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert installer.register(snapshot, COMPOUND, KEY) == digest
    assert runtime.load_model(digest, COMPOUND) == (snapshot.read_bytes(), KEY)
    with pytest.raises(ValueError, match="different chemical identity"):
        installer.register(snapshot, COMPOUND, OTHER_KEY)
    with pytest.raises(runtime.PKSimError, match="^pksim_compound_mismatch$"):
        runtime.load_model(digest, "Isopropanol")
    assert runtime.load_model(digest, COMPOUND) == (snapshot.read_bytes(), KEY)
    assert [child.name for child in (cache / "models").iterdir()] == [digest]


def test_candidate_local_model_and_identity_cannot_create_a_trusted_registration(cache, snapshot, tmp_path, monkeypatch):
    raw = snapshot.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    candidate = tmp_path / "candidate"
    pretend = candidate / ".cache" / "pksim" / "models" / digest
    pretend.mkdir(parents=True)
    (pretend / "snapshot.json").write_bytes(raw)
    (pretend / "identity.json").write_text(
        json.dumps({"model_sha256": digest, "compound": COMPOUND, "inchikey": KEY}), encoding="utf-8"
    )
    monkeypatch.chdir(candidate)
    with pytest.raises(runtime.PKSimError, match="^pksim_model_unregistered$"):
        runtime.load_model(digest, COMPOUND)
    with pytest.raises(runtime.PKSimError, match="^pksim_model_unregistered$"):
        runtime.run(raw, SIMULATION)
    assert not (cache / "models").exists()


@pytest.mark.parametrize("replace", ["snapshot", "receipt"])
def test_registered_artifact_substitution_fails_before_solver(cache, snapshot, monkeypatch, replace):
    raw = snapshot.read_bytes()
    digest = installer.register(snapshot, COMPOUND, KEY)
    registered = cache / "models" / digest
    if replace == "snapshot":
        (registered / "snapshot.json").write_bytes(raw.replace(b"Prepared", b"Unreviewed"))
    else:
        receipt = registered / "identity.json"
        data = json.loads(receipt.read_bytes())
        data["model_sha256"] = "0" * 64
        receipt.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: pytest.fail("ran unverified model"))
    with pytest.raises(runtime.PKSimError, match="^pksim_model_unregistered$"):
        runtime.run(raw, SIMULATION)


@pytest.fixture
def runnable(cache, snapshot, monkeypatch):
    installer.register(snapshot, COMPOUND, KEY)
    monkeypatch.setattr(runtime, "installed", lambda: True)
    return snapshot.read_bytes()


@pytest.mark.parametrize("case", ["wrong file", "wrong summary", "extra file", "no results", "empty file"])
def test_solver_exit_zero_requires_exact_case_attribution(cache, runnable, monkeypatch, case):
    def invoke(command, **kwargs):
        out = Path(command[command.index("--output") + 1])
        name = SIMULATION if case != "wrong file" else "Foreign"
        if case != "no results":
            (out / f"model-{name}-Results.csv").write_bytes(CSV if case != "empty file" else b"")
        select = "Different" if case == "wrong summary" else SIMULATION
        (out / "results.csv").write_text(f'"Sim_id"\n{select}\n', encoding="utf-8-sig")
        if case == "extra file":
            (out / "foreign.csv").write_bytes(CSV)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime.subprocess, "run", invoke)
    with pytest.raises(runtime.PKSimError, match="^pksim_invalid_export$"):
        runtime.run(runnable, SIMULATION)
    assert not list(cache.glob("run-*"))


def test_timeout_then_success_only_uses_fresh_scratch_and_keeps_other_runs(cache, runnable, monkeypatch):
    directories = []
    untouched = cache / "run-existing"
    untouched.mkdir()
    sentinel = untouched / "keep.csv"
    sentinel.write_bytes(b"existing output")

    def invoke(command, **kwargs):
        inp = Path(command[command.index("--input") + 1])
        out = Path(command[command.index("--output") + 1])
        assert out != untouched and out.parent == inp.parent == kwargs["cwd"]
        assert kwargs["timeout"] == runtime.TIMEOUT and kwargs["check"]
        assert list(out.iterdir()) == []
        model = json.loads((inp / "model.json").read_bytes())
        assert [s["Name"] for s in model["Simulations"]] == [SIMULATION]
        assert model["Simulations"][0]["OutputSelections"] == [OUTPUT]
        directories.append(out.parent)
        if len(directories) == 1:
            (out / "partial.csv").write_bytes(b"partial output")
            raise subprocess.TimeoutExpired(command, runtime.TIMEOUT)
        (out / "results.csv").write_text(f'"Sim_id"\n{SIMULATION}\n', encoding="utf-8-sig")
        (out / f"model-{SIMULATION}-Results.csv").write_bytes(CSV)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime.subprocess, "run", invoke)
    with pytest.raises(runtime.PKSimError, match="^pksim_execution_timeout$"):
        runtime.run(runnable, SIMULATION)
    assert not directories[0].exists()
    assert runtime.run(runnable, SIMULATION) == CSV
    assert len(set(directories)) == 2 and all(not d.exists() for d in directories)
    assert sentinel.read_bytes() == b"existing output"
