"""A run written with ``compress_outputs=true`` is screened, not refused.

``_generated_cifs`` globbed plain ``*_generated.cif`` only, so every design in
such a run was invisible: the screen measured nothing and, until
``designs_compressed`` was added, said nothing either. Naming the reason was the
first half. Reading the file is this half.

``.cif.gz`` goes through the standard library and always works. ``.cif.zst`` is
what the model itself writes, and ``zstandard`` arrives with the model stack
rather than with this package, so where it is missing the design is still named
by ``designs_compressed`` rather than found and silently yielding nothing.

A binary CIF stays refused. It is a different format, not a compressed one, and
this module parses CIF text.
"""

import gzip
import io
from pathlib import Path

import pytest

from rfproteina.disease.screen import (
    _DESIGN_TEXT_LIMIT,
    _design_stem,
    _design_text,
    _designs_blocker,
    _generated_cifs,
    _zstandard,
)
from tests.helpers.disease import DEPOSIT_CIF, UBIQUITIN, condition_spec

_UBIQUITIN = UBIQUITIN

_CIF = """data_design
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM N MET A 1 1.000 2.000 3.000
ATOM CA MET A 1 2.000 3.000 4.000
ATOM C MET A 1 3.000 4.000 5.000
"""


def _samples(root: Path) -> Path:
    directory = root / "inference_samples" / "binder_0"
    directory.mkdir(parents=True)
    return directory


def test_a_gzipped_design_is_read(tmp_path: Path) -> None:
    """The standard-library half, which works on every machine."""
    directory = _samples(tmp_path)
    path = directory / "binder_0_generated.cif.gz"
    path.write_bytes(gzip.compress(_CIF.encode("utf-8")))
    assert _design_text(path) == _CIF
    # Found and readable, so the run has a design and ``_designs_blocker`` is
    # never reached. It answers only the question "nothing was measured -- why",
    # so asking it here would be asking about a state that does not occur.
    assert _generated_cifs(tmp_path) == [path]


@pytest.mark.skipif(_zstandard() is None, reason="zstandard is not installed")
def test_a_zstd_design_is_read(tmp_path: Path) -> None:
    """What ``compress_outputs=true`` actually writes."""
    module = _zstandard()
    directory = _samples(tmp_path)
    path = directory / "binder_0_generated.cif.zst"
    path.write_bytes(module.ZstdCompressor().compress(_CIF.encode("utf-8")))
    assert _design_text(path) == _CIF
    assert _generated_cifs(tmp_path) == [path]


def test_a_plain_design_is_still_read_the_same_way(tmp_path: Path) -> None:
    """The guard: the common case is untouched."""
    directory = _samples(tmp_path)
    path = directory / "binder_0_generated.cif"
    path.write_text(_CIF, encoding="utf-8")
    assert _design_text(path) == _CIF
    assert _generated_cifs(tmp_path) == [path]


def test_a_binary_cif_is_still_refused_and_named(tmp_path: Path) -> None:
    """A different format, not a compressed one, so there is no text in it."""
    directory = _samples(tmp_path)
    (directory / "binder_0_generated.bcif").write_bytes(b"\x00\x01\x02binary")
    assert _generated_cifs(tmp_path) == []
    assert _designs_blocker(tmp_path) == "designs_compressed"


def test_a_truncated_compressed_design_is_refused_not_raised(tmp_path: Path) -> None:
    """Half a file is not a design, and it must not reach a caller as one."""
    directory = _samples(tmp_path)
    whole = gzip.compress(_CIF.encode("utf-8"))
    path = directory / "binder_0_generated.cif.gz"
    path.write_bytes(whole[: len(whole) // 2])
    assert _design_text(path) is None


@pytest.mark.skipif(_zstandard() is None, reason="zstandard is not installed")
def test_a_truncated_zstd_design_is_refused_rather_than_read_short(tmp_path: Path) -> None:
    """The half the gzip case above does not cover, and the half that matters.

    ``gzip`` refuses a frame whose tail is missing on its own. ``zstandard``'s
    ``stream_reader`` hands back the bytes it managed to decompress and says
    nothing, so a design killed mid-write -- which is what an interrupted
    ``compress_outputs=true`` leaves, and ``.cif.zst`` is what that writes --
    came back as a shorter CIF and was measured as the whole design.

    The payload spans more than one decompression block, because that is the
    shape where a real prefix comes back rather than nothing: the atom table at
    the end of this file falls past the cut, so eight missing bytes turned a
    design with coordinates into one the screen measured as having none.
    """
    module = _zstandard()
    directory = _samples(tmp_path)
    padding = "".join(f"# filler {index:06d}\n" for index in range(160_000))
    whole = ("data_design\n#\n" + padding + _CIF[len("data_design\n") :]).encode("utf-8")
    assert len(whole) > 2 << 20, "the cut has to land past a block boundary"
    path = directory / "binder_0_generated.cif.zst"
    path.write_bytes(module.ZstdCompressor().compress(whole)[:-8])
    assert _design_text(path) is None
    # The probe only reads the first chunk, so the file is still listed: the
    # refusal has to come from the full read, which is what gives the design
    # ``structure_unreadable`` rather than an absent measurement.
    assert _generated_cifs(tmp_path) == [path]


@pytest.mark.skipif(_zstandard() is None, reason="zstandard is not installed")
def test_a_whole_zstd_design_is_not_refused_by_the_truncation_check(tmp_path: Path) -> None:
    """The guard on that refusal: a complete frame still reads, whole.

    Including one written by a streaming writer, which embeds no content size
    and is the frame shape the truncation check had to be able to tell from a
    cut-short one.
    """
    module = _zstandard()
    directory = _samples(tmp_path)
    body = _CIF + "#" + "x" * 300_000 + "\n"
    framed = directory / "binder_0_generated.cif.zst"
    framed.write_bytes(module.ZstdCompressor().compress(body.encode("utf-8")))
    assert _design_text(framed) == body
    buffer = io.BytesIO()
    with module.ZstdCompressor().stream_writer(buffer, closefd=False) as writer:
        writer.write(body.encode("utf-8"))
    streamed = directory / "binder_1_generated.cif.zst"
    streamed.write_bytes(buffer.getvalue())
    assert _design_text(streamed) == body


def test_a_file_that_is_not_compressed_at_all_is_refused(tmp_path: Path) -> None:
    """Plain text under a ``.gz`` name is the mislabelled case."""
    directory = _samples(tmp_path)
    path = directory / "binder_0_generated.cif.gz"
    path.write_text(_CIF, encoding="utf-8")
    assert _design_text(path) is None


def test_the_plain_design_wins_when_both_spellings_sit_together(tmp_path: Path) -> None:
    """A decompress-in-place leaves both behind; the design is one design."""
    directory = _samples(tmp_path)
    plain = directory / "binder_0_generated.cif"
    plain.write_text(_CIF, encoding="utf-8")
    (directory / "binder_0_generated.cif.gz").write_bytes(gzip.compress(_CIF.encode("utf-8")))
    assert _generated_cifs(tmp_path) == [plain]


def test_two_designs_in_different_folders_are_both_found(tmp_path: Path) -> None:
    """One compressed, one not, and the ids stay one per design."""
    first = _samples(tmp_path)
    second = tmp_path / "inference_samples" / "binder_1"
    second.mkdir(parents=True)
    plain = first / "binder_0_generated.cif"
    plain.write_text(_CIF, encoding="utf-8")
    packed = second / "binder_1_generated.cif.gz"
    packed.write_bytes(gzip.compress(_CIF.encode("utf-8")))
    assert _generated_cifs(tmp_path) == [plain, packed]


def test_a_stem_is_taken_against_the_whole_suffix() -> None:
    """The id and the metrics row are both found by the stem.

    ``removesuffix("_generated.cif")`` left ``binder_0_generated.cif.zst``
    whole, so a compressed design would have carried the compression into its
    own example id and matched no row in the metrics table.
    """
    assert _design_stem("binder_0_generated.cif") == "binder_0"
    assert _design_stem("binder_0_generated.cif.gz") == "binder_0"
    assert _design_stem("binder_0_generated.cif.zst") == "binder_0"
    assert _design_stem("something_else.cif") == "something_else.cif"


def test_a_decompression_bomb_is_refused(tmp_path: Path) -> None:
    """A small file on disk is not a small file in memory.

    The bound is taken after decompressing, because that is the only place the
    expanded size is known. It applies to a compressed design only: a plain one
    is read whole and unbounded, the way this always read it.
    """
    directory = _samples(tmp_path)
    path = directory / "binder_0_generated.cif.gz"
    path.write_bytes(gzip.compress(b"A" * (_DESIGN_TEXT_LIMIT + 1024)))
    assert path.stat().st_size < 1_000_000, "the point is that it is small on disk"
    assert _design_text(path) is None


def test_a_large_plain_design_is_still_read(tmp_path: Path) -> None:
    """The guard on the bound: it must not refuse what used to be measured."""
    directory = _samples(tmp_path)
    path = directory / "binder_0_generated.cif"
    padding = "\n#" + "x" * 4096
    path.write_text(_CIF + padding, encoding="utf-8")
    text = _design_text(path)
    assert text is not None
    assert text.endswith(padding)


def test_a_plain_cif_renamed_to_zst_is_not_counted_as_a_design(tmp_path: Path) -> None:
    """A careless copy renames rather than compresses, and that is not a design.

    Counting it would make a sample out of a file that yields no text. The run
    holds a design it could not read, which is what ``designs_compressed`` says
    and what this screen said for every compressed design before it read any.
    """
    directory = _samples(tmp_path)
    path = directory / "binder_0_generated.cif.zst"
    path.write_text(_CIF, encoding="utf-8")
    assert _generated_cifs(tmp_path) == []
    assert _designs_blocker(tmp_path) == "designs_compressed"
    # And the read refuses it rather than raising: ZstdError is not a ValueError.
    assert _design_text(path) is None


def test_a_plain_cif_renamed_to_gz_is_not_counted_either(tmp_path: Path) -> None:
    """The same for the gzip spelling."""
    directory = _samples(tmp_path)
    path = directory / "binder_0_generated.cif.gz"
    path.write_text(_CIF, encoding="utf-8")
    assert _generated_cifs(tmp_path) == []
    assert _designs_blocker(tmp_path) == "designs_compressed"


def test_a_readable_compressed_design_is_not_reported_as_compressed(tmp_path: Path) -> None:
    """The guard: the blocker is for designs that could not be read, not for compression."""
    directory = _samples(tmp_path)
    path = directory / "binder_0_generated.cif.gz"
    path.write_bytes(gzip.compress(_CIF.encode("utf-8")))
    assert _generated_cifs(tmp_path) == [path]


def _finished_run(root: Path, name: str, design_name: str, *, pack: str = "") -> Path:
    """A finished run with one design, written plain or compressed.

    Shaped like the fixture in ``test_disease_screen_only``: an ``_entity_poly``
    sequence and a metrics row naming the chain, which is what the sequence
    axes are measured from.
    """
    run = root / name
    sample = run / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    cif = (
        "data_design\n#\nloop_\n_entity_poly.entity_id\n"
        "_entity_poly.pdbx_strand_id\n_entity_poly.pdbx_seq_one_letter_code_can\n"
        f"1 A {_UBIQUITIN}\n#\n"
    )
    raw = cif.encode("utf-8")
    if pack == "gz":
        raw = gzip.compress(raw)
    elif pack == "zst":
        raw = _zstandard().ZstdCompressor().compress(raw)
    (sample / design_name).write_bytes(raw)
    metrics = run / "inference_metrics"
    metrics.mkdir(parents=True)
    (metrics / "per_example_metrics.csv").write_text(
        "example_id,chain_of_interest\nbinder_0,A\n",
        encoding="utf-8",
    )
    return run


def _measured(run: Path) -> dict:
    """The parts of a screen that come from reading the design."""
    from rfproteina.disease.screen import screen_generation

    screen = screen_generation(run, binder_kind="protein", target_id="P05113")
    return {
        "samples": screen["samples"],
        "sequence_developability": screen["sequence_developability"],
        "designed_interface": screen["designed_interface"],
    }


def test_a_gzipped_run_screens_exactly_like_a_plain_one(tmp_path: Path) -> None:
    """The whole point: the run is measured, not refused.

    Compared against the plain run rather than against fixed expectations, so
    this says the two are the same screen rather than asserting what that screen
    happens to be. The example id is included in ``samples``, which is what
    would move if the compression leaked into the stem.
    """
    plain = _finished_run(tmp_path, "plain", "binder_0_generated.cif")
    packed = _finished_run(tmp_path, "packed", "binder_0_generated.cif.gz", pack="gz")
    assert _measured(packed) == _measured(plain)
    # And it really did measure something, so the comparison is not two blanks.
    measured = _measured(plain)
    assert measured["samples"], "the plain run measured nothing, so this proves nothing"
    assert "sequence_blocker" not in measured["sequence_developability"]


@pytest.mark.skipif(_zstandard() is None, reason="zstandard is not installed")
def test_a_zstd_run_screens_exactly_like_a_plain_one(tmp_path: Path) -> None:
    """What ``compress_outputs=true`` writes, end to end through the screen."""
    plain = _finished_run(tmp_path, "plain", "binder_0_generated.cif")
    packed = _finished_run(tmp_path, "packed", "binder_0_generated.cif.zst", pack="zst")
    assert _measured(packed) == _measured(plain)


def test_a_compressed_condition_spec_input_still_names_the_target(tmp_path: Path) -> None:
    """``inputs=...cif.zst`` is the documented way to run this model.

    The condition spec's input is read to resolve the target's accession, and
    the program's own suffix list -- every encoding AtomWorks accepts -- holds
    ``.cif.zst``, ``.cif.gz`` and ``.cif.gzip``. Read as plain text that raised
    ``UnicodeError``, the reader returned no accession, and the target could not
    be named from the very spec that named it.
    """
    for suffix, pack in ((".cif", None), (".cif.gz", "gz"), (".cif.zst", "zst")):
        if pack == "zst" and _zstandard() is None:
            continue
        root = tmp_path / f"case{suffix.replace('.', '_')}"
        campaign = root / "gene"
        campaign.mkdir(parents=True)
        raw = DEPOSIT_CIF.encode("utf-8")
        if pack == "gz":
            raw = gzip.compress(raw)
        elif pack == "zst":
            raw = _zstandard().ZstdCompressor().compress(raw)
        (root / f"deposit{suffix}").write_bytes(raw)
        (root / "gene.yaml").write_text(condition_spec(f"deposit{suffix}", "C"), encoding="utf-8")

        from rfproteina.disease.screen import screen_generation

        record = screen_generation(campaign, binder_kind="protein", target_id="IL5")
        assert record["target_accession"] == "P05113", f"{suffix} resolved no accession"


def test_an_inverse_folding_spec_is_written_from_a_compressed_design(tmp_path: Path) -> None:
    """The follow-up plan reads a design the screen just learned to read.

    ``sample_cifs`` now returns compressed paths, so ``follow_up`` hands one to
    ``write_inverse_folding_spec``. Read as plain text it raised
    ``UnicodeDecodeError`` and the plan lost that sample in silence.
    """
    from rfproteina.disease.design_spec import backbone_spans, write_inverse_folding_spec

    atoms = ["data_design", "#", "loop_"] + [
        f"_atom_site.{key}" for key in ("group_PDB", "label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id")
    ]
    for chain in ("A", "B"):
        for seq in (1, 2):
            for atom in ("N", "CA", "C", "O"):
                atoms.append(f"ATOM {atom} GLY {chain} {seq}")
    atoms.append("#")
    raw = ("\n".join(atoms) + "\n").encode("utf-8")

    run = tmp_path / "run"
    run.mkdir()
    packed = run / "binder_0_generated.cif.gz"
    packed.write_bytes(gzip.compress(raw))
    plain = run / "plain_generated.cif"
    plain.write_bytes(raw)

    # The spans are what the spec is built from, and they must not depend on
    # how the file was stored.
    assert backbone_spans(packed) == backbone_spans(plain)
    spec = write_inverse_folding_spec(packed, tmp_path / "fresh", designed_chain="A")
    text = spec.read_text(encoding="utf-8")
    assert spec.name == "inverse_folding.yaml"
    assert "C_CRDGRP" in text
    # The spec points at the compressed file itself, which the model reads.
    assert "binder_0_generated.cif.gz" in text


def test_a_design_spec_drops_the_compression_from_its_own_name(tmp_path: Path) -> None:
    """``Path.stem`` takes one suffix off, so the spec was named for a .cif."""
    from rfproteina.disease.design_spec import structure_stem

    assert structure_stem(Path("binder_0_generated.cif")) == "binder_0_generated"
    assert structure_stem(Path("binder_0_generated.cif.gz")) == "binder_0_generated"
    assert structure_stem(Path("binder_0_generated.cif.zst")) == "binder_0_generated"
    assert structure_stem(Path("binder_0_generated.cif.gzip")) == "binder_0_generated"
    assert structure_stem(Path("8V4F.pdb")) == "8V4F"


def test_the_viewer_serves_cif_text_for_a_compressed_design(tmp_path: Path) -> None:
    """The browser parses CIF text, so the archive has to be opened here.

    ``sample_cifs`` returns compressed designs now, so one reaches the viewer.
    Served as its bytes on disk the browser got gzip or zstd and drew nothing,
    which looks exactly like a design that failed to render.
    """
    from rfproteina.disease.structure_view import StructureTrio
    from rfproteina.disease.ui import BrowserSession

    run = tmp_path / "run"
    run.mkdir()
    packed = run / "binder_0_generated.cif.gz"
    packed.write_bytes(gzip.compress(_CIF.encode("utf-8")))
    plain = run / "plain_generated.cif"
    plain.write_text(_CIF, encoding="utf-8")

    session = BrowserSession(opener=lambda _url: None)
    session.note_structures(
        [
            StructureTrio(
                sample="binder_0",
                title="compressed",
                disease_file=plain,
                disease_chains=("A",),
                complex_file=packed,
                target_chains=("A",),
                binder_chains=("A",),
            )
        ]
    )
    try:
        catalog = session._catalog
        assert catalog, "the trio was not registered"
        served = session.structure_bytes(str(catalog[0]["complex_token"]))
        assert served is not None, "the compressed design served nothing"
        # CIF text, not the archive's bytes.
        assert served == _CIF.encode("utf-8")
        assert not served.startswith(b"\x1f\x8b"), "the gzip header reached the browser"
        # A plain file is still served as its exact bytes.
        assert session.structure_bytes(str(catalog[0]["disease_token"])) == plain.read_bytes()
    finally:
        session.mark_sent()
        session.close()


def test_a_gzip_spelled_design_is_read_and_not_called_absent(tmp_path: Path) -> None:
    """``_design_text`` decodes ``.cif.gzip``, so the lists must agree with it.

    The spelling arrived for a user-supplied condition-spec input, where
    ``combinations._STRUCTURE_SUFFIXES`` accepts it. Left out of the design
    suffix lists, a design this reader decodes perfectly was reported as
    ``designs_absent`` -- no design at all, for a folder holding one. A
    half-supported spelling is a false claim in the suppressing direction.
    """
    directory = _samples(tmp_path)
    path = directory / "binder_0_generated.cif.gzip"
    path.write_bytes(gzip.compress(_CIF.encode("utf-8")))
    assert _design_text(path) == _CIF
    assert _generated_cifs(tmp_path) == [path]
    assert _design_stem(path.name) == "binder_0"


def test_the_gzip_spelling_still_loses_to_a_plain_design(tmp_path: Path) -> None:
    """Plain keeps the highest index, so adding a longer suffix cannot unseat it."""
    directory = _samples(tmp_path)
    plain = directory / "binder_0_generated.cif"
    plain.write_text(_CIF, encoding="utf-8")
    (directory / "binder_0_generated.cif.gzip").write_bytes(gzip.compress(_CIF.encode("utf-8")))
    assert _generated_cifs(tmp_path) == [plain]


def test_an_unreadable_gzip_design_is_named_compressed_not_absent(tmp_path: Path) -> None:
    """And when it cannot be read, the blocker names the right remedy."""
    directory = _samples(tmp_path)
    (directory / "binder_0_generated.cif.gzip").write_text(_CIF, encoding="utf-8")
    assert _generated_cifs(tmp_path) == []
    assert _designs_blocker(tmp_path) == "designs_compressed"
