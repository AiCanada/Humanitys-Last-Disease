"""API keys and personal paths stay out of repository uploads."""

import json
from pathlib import Path

import pytest

from rfproteina.disease.key_store import save_api_keys
from rfproteina.disease.publish import RepositoryKeyError, repository_relative, scrub_for_upload


def test_upload_text_drops_keys_and_personal_paths():
    home = Path.home()
    raw = "\n".join(
        [
            "key sk-abcDEF1234567890abcd",
            "Authorization: Bearer secret-token",
            f"home {home}/notes",
            r"win C:\Users\someone\keys",
            "/mnt/c/Users/someone/keys",
            "/home/someone/notes",
            f"repo {Path('.project-root').resolve().parent}/disease_runs/gene_8TLD_2",
        ]
    )
    cleaned = scrub_for_upload(raw)
    assert "sk-abcDEF1234567890abcd" not in cleaned
    assert "secret-token" not in cleaned
    assert "[redacted]" in cleaned
    assert str(home) not in cleaned
    assert "someone" not in cleaned
    assert "/mnt/" not in cleaned
    assert "Git Hub" not in cleaned
    assert "<home>" in cleaned


def test_a_repository_path_drops_the_machine_prefix():
    root = Path(".project-root").resolve().parent
    assert repository_relative(root / "disease_runs" / "gene_8TLD_2.yaml") == "disease_runs/gene_8TLD_2.yaml"


def test_a_path_outside_the_repository_keeps_its_location(tmp_path: Path):
    target = tmp_path / "kept.yaml"
    assert repository_relative(target) == str(target.resolve())


def test_api_keys_are_refused_inside_the_repository(monkeypatch: pytest.MonkeyPatch):
    store = Path(".project-root").resolve().parent / "tests" / "_key_store_probe"
    monkeypatch.setenv("RFD4_KEY_DIR", str(store))
    with pytest.raises(RepositoryKeyError, match="repository"):
        save_api_keys({"grok": "sk-abcDEF1234567890abcd"})
    assert not store.exists()


def test_a_windows_path_is_scrubbed_from_a_json_document():
    """Scrubbing the finished document is not enough, and this is why.

    ``json.dumps`` writes one backslash as two, so ``C:\\Users\\name`` becomes
    ``C:\\\\Users\\\\name`` in the document and no longer matches a pattern
    written for a single separator. Every writer in this package scrubbed the
    encoded document, so on a native Windows run the user name and the
    checkout's absolute path -- the first two things AGENTS.md names -- went
    into ``screen.json``, ``follow_up.json``, ``generation.json``,
    ``target_research.json`` and ``returned_structures.json`` verbatim. A path
    written with forward slashes was scrubbed either way, which is what hid it
    on a WSL run.
    """
    from rfproteina.disease.publish import scrubbed_json

    payload = {
        "win": r"C:\Users\someone\Downloads\design.cif",
        "repo": str(Path(".project-root").resolve().parent / "disease_runs" / "gene_8TLD_2"),
        "posix": "/home/someone/notes",
        "wsl": "/mnt/c/Users/someone/notes",
        "key": "sk-abcDEF1234567890abcd",
        "commands": [r"rfd4-generate out_dir=C:\Users\someone\runs"],
        "count": 3,
        "flag": True,
        "nothing": None,
    }
    text = scrubbed_json(payload)
    assert "someone" not in text
    assert "Git Hub" not in text
    assert "sk-abcDEF1234567890abcd" not in text
    assert "<home>" in text
    assert text.endswith("\n")
    # Still JSON, and the values that are not paths are untouched.
    restored = json.loads(text)
    assert restored["count"] == 3
    assert restored["flag"] is True
    assert restored["nothing"] is None


def test_a_written_report_carries_no_user_name(tmp_path: Path):
    """The same thing through a real writer, which is where it was reproduced."""
    from rfproteina.disease.research import write_report

    path = write_report(
        tmp_path,
        {"win": r"C:\Users\someone\Downloads\design.cif", "note": "kept"},
    )
    text = path.read_text(encoding="utf-8")
    assert "someone" not in text
    assert "kept" in text
