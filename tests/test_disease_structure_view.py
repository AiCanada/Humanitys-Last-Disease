"""The finished page draws the disease structure, the complex, and the binder."""

import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from rfproteina.disease.structure_view import StructureTrio, render_structure_views, structure_trios
from rfproteina.disease.ui import BrowserSession, OptionPanel, RestartFreshError

_ATOM_HEADER = """\
data_fixture
#
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.type_symbol
"""
_SPEC = """\
binder:
  input: disease-mark.cif
  conditions:
    C_SEQ:
      true:
        - select: "sel('C')"
"""


def _atoms(rows: list[tuple[str, str, str]]) -> str:
    body = "".join(f"ATOM {atom} {label} {auth} C\n" for atom, label, auth in rows)
    return _ATOM_HEADER + body + "#\n"


def _polymer(label: str, auth: str, mark: str = "") -> list[tuple[str, str, str]]:
    atoms = ["N", "CA", "C", "O"]
    if mark:
        atoms.append(mark)
    return [(atom, label, auth) for atom in atoms]


def _ligand(label: str, auth: str) -> list[tuple[str, str, str]]:
    return [("C1", label, auth)]


def _prepare(root: Path, name: str) -> tuple[Path, Path]:
    output = root / name
    output.mkdir()
    disease = root / "disease-mark.cif"
    disease.write_text(
        _atoms([*_polymer("C", "D", "ZZMARK"), *_polymer("B", "C"), *_ligand("J", "J")]),
        encoding="utf-8",
    )
    (root / f"{name}.yaml").write_text(_SPEC, encoding="utf-8")
    generated = _atoms([*_polymer("A", "A", "BBMARK"), *_polymer("C", "C"), *_ligand("L", "L")])
    for sample in ("binder_0", "binder_1"):
        folder = output / "inference_samples" / sample
        folder.mkdir(parents=True)
        (folder / f"{sample}_generated.cif").write_text(generated, encoding="utf-8")
    return output, disease


def _catalog(page: str) -> list[dict[str, object]]:
    match = re.search(r'<script type="application/json" id="atom-catalog">(.*?)</script>', page)
    assert match is not None
    loaded = json.loads(match.group(1))
    assert isinstance(loaded, list)
    return loaded


def _expect_missing(url: str) -> None:
    try:
        urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
        assert exc.read() == b"Not found."
        return
    raise AssertionError(url)


def test_the_disease_view_uses_the_conditioned_label_chain(tmp_path: Path) -> None:
    output, disease = _prepare(tmp_path, "hidden-output")
    trios = structure_trios(output, entry_id="1EGF", chain="1")

    assert [trio.sample for trio in trios] == ["binder_0", "binder_1"]
    assert trios[0].title == "1EGF chain 1"
    assert trios[0].disease_file.resolve() == disease.resolve()
    assert trios[0].disease_chains == ("C",)
    assert trios[0].target_chains == ("C",)
    assert trios[0].binder_chains == ("A",)
    assert trios[1].disease_file == trios[0].disease_file
    assert trios[0].complex_file != trios[1].complex_file
    assert "L" not in trios[0].binder_chains
    assert "L" not in trios[0].target_chains

    session = BrowserSession(opener=lambda _url: None)
    session.note_structures(trios)
    page = session.done_page()
    assert "Return to the terminal." not in page
    assert ">Close</button>" in page
    assert ">Disease structure</figcaption>" in page
    assert ">Disease structure with binder</figcaption>" in page
    assert ">Binder</figcaption>" in page


def test_the_generated_target_is_used_when_the_input_lacks_that_chain(tmp_path: Path) -> None:
    output = tmp_path / "fallback-run"
    output.mkdir()
    (tmp_path / "disease-mark.cif").write_text(_atoms(_polymer("B", "C", "ZZMARK")), encoding="utf-8")
    (tmp_path / "fallback-run.yaml").write_text(_SPEC, encoding="utf-8")
    generated = output / "inference_samples" / "binder_0" / "binder_0_generated.cif"
    generated.parent.mkdir(parents=True)
    generated.write_text(_atoms([*_polymer("A", "A", "BBMARK"), *_polymer("C", "C")]), encoding="utf-8")

    trios = structure_trios(output, entry_id="1EGF", chain="1")
    assert len(trios) == 1
    assert trios[0].disease_file.resolve() == generated.resolve()
    assert trios[0].disease_chains == ("C",)
    assert trios[0].binder_chains == ("A",)


def test_metrics_keep_only_the_named_binder(tmp_path: Path) -> None:
    output = tmp_path / "metrics-run"
    output.mkdir()
    (tmp_path / "metrics-run.yaml").write_text(_SPEC, encoding="utf-8")
    folder = output / "inference_samples" / "binder_0"
    folder.mkdir(parents=True)
    (folder / "binder_0_generated.cif").write_text(
        _atoms([*_polymer("A", "A"), *_polymer("C", "C"), *_polymer("D", "D")]),
        encoding="utf-8",
    )
    metrics = output / "inference_metrics"
    metrics.mkdir()
    (metrics / "per_example_metrics.csv").write_text(
        "example_id,chain_of_interest\nbinder_0,A_1\n",
        encoding="utf-8",
    )

    trios = structure_trios(output, entry_id="1EGF", chain="1")
    assert trios[0].target_chains == ("C",)
    assert trios[0].binder_chains == ("A",)
    assert trios[0].disease_chains == ("C",)


def test_a_metrics_chain_is_the_binder_when_no_sequence_is_fixed(tmp_path: Path) -> None:
    output = tmp_path / "plain-run"
    output.mkdir()
    folder = output / "inference_samples" / "binder_0"
    folder.mkdir(parents=True)
    generated = folder / "binder_0_generated.cif"
    generated.write_text(_atoms([*_polymer("A", "A"), *_polymer("D", "D")]), encoding="utf-8")
    metrics = output / "inference_metrics"
    metrics.mkdir()
    (metrics / "per_example_metrics.csv").write_text(
        "example_id,chain_of_interest\nbinder_0,A_1\n",
        encoding="utf-8",
    )

    trios = structure_trios(output, entry_id="1EGF", chain="1")
    assert trios[0].binder_chains == ("A",)
    assert trios[0].disease_chains == ("D",)
    assert trios[0].disease_file.resolve() == generated.resolve()


def test_an_unreadable_sample_is_left_out(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    output, _disease = _prepare(tmp_path, "hidden-output")
    bad = output / "inference_samples" / "binder_2" / "binder_2_generated.cif"
    bad.parent.mkdir(parents=True)
    bad.write_text("not a structure\n", encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="Humanitys-Last-Disease")

    trios = structure_trios(output, entry_id="1EGF", chain="1")
    assert [trio.sample for trio in trios] == ["binder_0", "binder_1"]
    messages = [record.message for record in caplog.records]
    assert "a generated structure could not be drawn" in messages
    assert "hidden-output" not in " ".join(messages)


def test_one_design_hides_the_sample_menu() -> None:
    page = render_structure_views(
        [
            {
                "sample": "binder_0",
                "title": "1EGF chain 1",
                "disease_token": "0123456789abcdef",
                "disease_chains": ["C"],
                "complex_token": "fedcba9876543210",
                "target_chains": ["C"],
                "binder_chains": ["A"],
            }
        ]
    )
    assert 'id="atom-sample"' not in page
    assert ">Disease structure</figcaption>" in page
    assert ">Disease structure with binder</figcaption>" in page
    assert ">Binder</figcaption>" in page
    assert "0123456789abcdef" not in page.split('id="atom-catalog">', 1)[0]


def test_the_finished_page_serves_three_atomic_views(tmp_path: Path) -> None:
    output, disease = _prepare(tmp_path, "hidden-output")
    trios = structure_trios(output, entry_id="1EGF", chain="1")
    ghost = StructureTrio(
        sample="ghost",
        title="gone",
        disease_file=tmp_path / "no-such.cif",
        disease_chains=("C",),
        complex_file=trios[0].complex_file,
        target_chains=("C",),
        binder_chains=("A",),
    )
    session = BrowserSession(opener=lambda _url: None)
    session.note_structures([*trios, ghost])
    session.note_closing(
        ["Screen 1EGF chain 1: untested."],
        [{"entry_id": "1EGF", "chain": "1", "screen": {"call": "untested"}}],
    )
    page = session.done_page()
    section = page.split('<section class="atom-views">', 1)[1].split("</section>", 1)[0]
    catalog = _catalog(page)

    assert page.index("<h2>Structures</h2>") < page.index("<h2>Panel closed</h2>")
    assert ">Disease structure</figcaption>" in page
    assert ">Disease structure with binder</figcaption>" in page
    assert ">Binder</figcaption>" in page
    assert 'id="atom-sample"' in page
    assert "Every heavy atom is drawn." in section
    assert "Dark green carbon is the binder." in section
    assert ".lchain" in section
    assert "{chain:" not in section
    assert "grayCarbon" in section
    assert "darkgreenCarbon" in section
    assert '"greenCarbon"' not in section
    assert "orangeCarbon" in section
    assert 'altLoc: "*"' in section
    assert "selectedAtoms" in section
    assert "Point at an atom to name it." in section
    assert "getView" in section
    assert "setView" in section
    assert "zoom(0.85, 0)" in section
    assert "function blank" in section
    assert "viewers[name].clear()" in section
    assert "cartoon" not in section
    assert 'type="submit"' not in section
    assert 'type="button"' in section
    assert "atom-reset" in section
    assert "Reset view" in section
    assert "#a9a9a9" in section
    assert "#006400" in section
    assert "#ffa500" in section
    assert "This structure could not be loaded." in section
    assert "This view has no atoms to draw." in section
    assert "The 3D viewer did not load." in section
    assert "hidden-output" not in page
    assert "disease-mark" not in page
    assert "ZZMARK" not in page
    assert "ghost" not in page
    assert "gone" not in page
    assert 'class="axis-panel"' in page
    # The user asked cure wording out of this flow on 2026-10-31, and the closing
    # board is part of it. This pins the absence rather than the sentence that was here.
    assert "cure" not in page.casefold()
    assert page.rindex(">Close</button>") > page.rindex(">Restart Server</button>")
    assert len(catalog) == 2
    assert catalog[0]["disease_chains"] == ["C"]
    assert catalog[0]["target_chains"] == ["C"]
    assert catalog[0]["binder_chains"] == ["A"]
    assert catalog[0]["disease_token"] == catalog[1]["disease_token"]
    assert catalog[0]["complex_token"] != catalog[1]["complex_token"]
    for item in catalog:
        for value in item.values():
            if isinstance(value, str):
                assert "/" not in value
                assert "\\" not in value

    session._ensure()
    try:
        assert session._server is not None
        base = f"http://127.0.0.1:{session._server.server_address[1]}"
        disease_token = str(catalog[0]["disease_token"])
        complex_token = str(catalog[0]["complex_token"])
        with urllib.request.urlopen(f"{base}/structure/{disease_token}", timeout=5) as response:
            assert response.headers["Content-Type"] == "chemical/x-mmcif"
            disease_body = response.read()
        assert disease_body == disease.read_bytes()
        assert b"ZZMARK" in disease_body
        assert b"BBMARK" not in disease_body
        with urllib.request.urlopen(f"{base}/structure/{complex_token}", timeout=5) as response:
            complex_body = response.read()
        assert b"BBMARK" in complex_body
        assert b"ATOM C1 L L C" in complex_body
        with urllib.request.urlopen(f"{base}/viewer/3dmol.js", timeout=5) as response:
            assert response.headers["Content-Type"] == "application/javascript"
            script = response.read()
        if b"3Dmol" not in script:
            raise AssertionError("viewer script did not contain 3Dmol")
        assert session.structure_bytes("../secret") is None
        assert session.structure_bytes("A" * 16) is None
        _expect_missing(f"{base}/structure/not-a-token")
        _expect_missing(f"{base}/structure/abcdef0123456789")
        session.clear_closing()
        cleared = session.done_page()
        assert "Return to the terminal." in cleared
        assert "Disease structure" not in cleared
        assert session.structure_bytes(disease_token) is None
        _expect_missing(f"{base}/structure/{disease_token}")
    finally:
        session.mark_sent()
        session.close()


def test_structures_noted_before_the_session_clear_on_restart(tmp_path: Path) -> None:
    output, _disease = _prepare(tmp_path, "hidden-output")
    trios = structure_trios(output, entry_id="1EGF", chain="1")
    panel = OptionPanel(lambda _page: {"__nav": ["fresh"]})
    panel.note_structures(trios)
    assert len(panel._pending_structures) == 2
    with pytest.raises(RestartFreshError):
        panel.ask("<p>later</p>")
    panel.prepare_fresh_start()
    assert panel._pending_structures == []

    waiting = OptionPanel(lambda _page: {})
    waiting.note_structures(trios)
    session = BrowserSession(opener=lambda _url: None)
    waiting._session = session
    waiting._flush_structures()
    assert waiting._pending_structures == []
    assert ">Binder</figcaption>" in session.done_page()
    waiting._clear_closing()
    assert "Return to the terminal." in session.done_page()


def test_structures_are_kept_when_the_archive_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rfproteina.disease.cli import _save_finished_generations
    from rfproteina.disease.dispatch import GenerationRequest, RunResult
    from rfproteina.disease.lookup import Disease
    from rfproteina.disease.structures import StructureOption

    output, disease = _prepare(tmp_path, "gene_1EGF_A")
    calls = {"n": 0}

    def _boom(*_args: object, **_kwargs: object) -> None:
        calls["n"] += 1
        raise OSError("disk full")

    monkeypatch.setattr("rfproteina.disease.cli.screen_generation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("rfproteina.disease.cli.save_generation", _boom)
    request = GenerationRequest(
        host="local",
        argv=["rfd4-generate", f"out_dir={output}"],
        local_input=output.parent / "gene_1EGF_A.yaml",
        remote_input="gene_1EGF_A.yaml",
        remote_out_dir="gene_1EGF_A",
    )
    option = StructureOption(
        group="gene",
        accession="P00000",
        entry_id="1EGF",
        chain="A",
        title="fixture",
        predicted=False,
        source="fixture",
    )
    disease_row = Disease(name="Fixture", disease_id="MONDO_0000000")
    lines, screens, trios = _save_finished_generations(
        [RunResult(host="local", argv=list(request.argv), returncode=0)],
        [request],
        [option],
        tmp_path / "kept-out",
        disease_row,
        "protein",
    )
    assert lines == []
    assert screens == []
    assert calls["n"] == 1
    assert len(trios) == 2
    assert trios[0].disease_file.resolve() == disease.resolve()
    assert trios[0].disease_chains == ("C",)
    assert trios[0].binder_chains == ("A",)
    captured = capsys.readouterr()
    assert "Generation archive was not saved for 1EGF chain A: disk full" in captured.out
    assert "hidden-output" not in captured.out

    skipped, _skipped_screens, skipped_trios = _save_finished_generations(
        [RunResult(host="local", argv=list(request.argv), returncode=1)],
        [request],
        [option],
        tmp_path / "kept-out",
        disease_row,
        "protein",
    )
    assert skipped == []
    assert skipped_trios == []
    assert calls["n"] == 1
