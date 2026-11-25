"""MolProbity resolves an exact finished candidate and decodes only bounded stable bytes."""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import pytest

from rfproteina.disease import molprobity_inputs as inputs
from rfproteina.disease.validation_checks import ValidationCandidate

_CIF = b"# original file header\r\ndata_example\r\n_entry.id example\r\n"


def _write(directory: Path, name: str, raw: bytes = _CIF) -> Path:
    path = directory / "inference_samples" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def test_choose_authoritative_plain_bytes_without_reading_coordinates(tmp_path, monkeypatch):
    plain = _write(tmp_path, "binder_1_generated.cif")
    _write(tmp_path, "binder_1_generated.cif.gz", gzip.compress(_CIF))
    _write(tmp_path, "artifact.cif")

    def forbidden(*args, **kwargs):
        pytest.fail("availability check read coordinates")

    monkeypatch.setattr(inputs, "read_structure", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    candidates, blocker = inputs.resolve_inputs(tmp_path)
    assert blocker is None and candidates == [inputs.ModelInput("binder_1", plain)]
    assert inputs.resolve_inputs(tmp_path, "binder_1") == (candidates, None)
    assert inputs.resolve_inputs(tmp_path, "binder_9") == ([], "molprobity_candidate_not_found")


def test_missing_inputs_and_explicit_empty_scope_do_not_fall_back(tmp_path):
    assert inputs.resolve_inputs(tmp_path) == ([], "molprobity_models_missing")
    _write(tmp_path, "binder_1_generated.cif")
    assert inputs.resolve_inputs(tmp_path, "") == ([], "molprobity_candidate_not_found")
    assert inputs.resolve_inputs(tmp_path, "../binder_1") == ([], "molprobity_candidate_not_found")


def test_bare_duplicate_fails_closed_but_exact_nested_id_remains_available(tmp_path):
    _write(tmp_path, "binder_1_generated.cif")
    nested = _write(tmp_path, "other/binder_1_generated.cif")
    _write(tmp_path, "binder_2_generated.cif")
    rows, blocker = inputs.resolve_inputs(tmp_path)
    assert blocker is None
    assert rows[0] == inputs.ModelInput("", None, "molprobity_candidate_ambiguous")
    assert {row.candidate_id for row in rows if row.path} == {"binder_2", "other_binder_1"}
    assert inputs.resolve_inputs(tmp_path, "other_binder_1") == ([inputs.ModelInput("other_binder_1", nested)], None)
    assert inputs.resolve_inputs(tmp_path, "binder_1") == ([rows[0]], None)


def test_duplicate_example_id_in_supplied_inventory_does_not_select_first(tmp_path):
    first = _write(tmp_path, "binder_1_generated.cif")
    second = _write(tmp_path, "binder_2_generated.cif")
    choices = [ValidationCandidate(str(i), "Candidate", "same", (path,)) for i, path in enumerate((first, second))]
    assert inputs.resolve_inputs(tmp_path, "same", candidates=choices) == ([], "molprobity_candidate_ambiguous")
    rows, blocker = inputs.resolve_inputs(tmp_path, candidates=choices)
    assert blocker is None
    assert rows == [inputs.ModelInput("", None, "molprobity_candidate_ambiguous")] * 2


def test_whole_run_cap_preserves_one_explicit_finalist(tmp_path):
    for i in range(inputs.MAX_MODELS):
        _write(tmp_path, f"binder_{i}_generated.cif")
    rows, blocker = inputs.resolve_inputs(tmp_path)
    assert len(rows) == inputs.MAX_MODELS and blocker is None
    finalist = _write(tmp_path, f"binder_{inputs.MAX_MODELS}_generated.cif")
    assert inputs.resolve_inputs(tmp_path) == ([], "molprobity_too_many_models")
    rows, blocker = inputs.resolve_inputs(tmp_path, f"binder_{inputs.MAX_MODELS}")
    assert blocker is None and rows == [inputs.ModelInput(f"binder_{inputs.MAX_MODELS}", finalist)]


def test_authoritative_alias_is_not_replaced_by_another_format_or_candidate(tmp_path):
    other = _write(tmp_path, "binder_2_generated.cif")
    alias = other.with_name("binder_1_generated.cif")
    try:
        alias.symlink_to(other)
    except OSError:
        pytest.skip("symlinks unavailable")
    _write(tmp_path, "binder_1_generated.cif.gz", gzip.compress(_CIF))
    rows, blocker = inputs.resolve_inputs(tmp_path, "binder_1")
    assert blocker is None
    assert rows == [inputs.ModelInput("binder_1", None, "molprobity_structure_unreadable")]
    assert inputs.resolve_inputs(tmp_path, "binder_2") == ([inputs.ModelInput("binder_2", other)], None)
    with pytest.raises(inputs.MolProbityInputError, match="molprobity_structure_unreadable"):
        inputs.read_structure(alias)


def test_direct_snapshot_rejects_a_symlink_in_parent(tmp_path):
    actual = _write(tmp_path, "parent/binder_1_generated.cif")
    alias = actual.parent.with_name("alias")
    try:
        alias.symlink_to(actual.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(inputs.MolProbityInputError, match="molprobity_structure_unreadable"):
        inputs.read_structure(alias / actual.name)


def test_replaced_path_cannot_attest_bytes_from_old_descriptor(tmp_path, monkeypatch):
    source = _write(tmp_path, "binder_1_generated.cif")
    replacement = _write(tmp_path, "replacement.cif", _CIF.replace(b"example", b"changed"))
    open_original = inputs._open

    def swapping_open(path):
        descriptor = open_original(path)
        try:
            replacement.replace(path)
        except OSError:
            os.close(descriptor)
            pytest.skip("open-file path replacement unavailable")
        return descriptor

    monkeypatch.setattr(inputs, "_open", swapping_open)
    with pytest.raises(inputs.MolProbityInputError, match="molprobity_structure_unreadable"):
        inputs.read_structure(source)


@pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".cif.gzip"])
def test_byte_exact_plain_and_gzip_snapshots(tmp_path, suffix):
    raw = _CIF if suffix == ".cif" else gzip.compress(_CIF[:30]) + gzip.compress(_CIF[30:])
    source = _write(tmp_path, "binder_1_generated" + suffix, raw)
    assert inputs.read_structure(source) == _CIF


def test_multiframe_unknown_length_zstd_snapshot_and_truncated_footer(tmp_path):
    zstd = pytest.importorskip("zstandard")
    codec = zstd.ZstdCompressor(write_checksum=True, write_content_size=False)
    raw = codec.compress(_CIF[:30]) + codec.compress(_CIF[30:])
    source = _write(tmp_path, "binder_1_generated.cif.zst", raw)
    assert inputs.read_structure(source) == _CIF
    source.write_bytes(raw[:-1])
    with pytest.raises(inputs.MolProbityInputError, match="molprobity_structure_invalid"):
        inputs.read_structure(source)


def test_missing_zstd_support_is_a_distinct_metadata_refusal(tmp_path, monkeypatch):
    _write(tmp_path, "binder_1_generated.cif.zst")
    monkeypatch.setattr(inputs, "_has_zstd", lambda: False)
    assert inputs.resolve_inputs(tmp_path) == ([inputs.ModelInput("binder_1", None, "molprobity_zstd_unavailable")], None)


def test_compressed_and_expanded_limits_are_independent(tmp_path, monkeypatch):
    source = _write(tmp_path, "binder_1_generated.cif.gz", gzip.compress(_CIF + (b"# padding\n" * 500)))
    monkeypatch.setattr(inputs, "MAX_STRUCTURE_BYTES", 128)
    monkeypatch.setattr(inputs, "MAX_ENCODED_BYTES", 512)
    assert inputs.resolve_inputs(tmp_path) == ([inputs.ModelInput("binder_1", source)], None)
    with pytest.raises(inputs.MolProbityInputError, match="molprobity_input_over_limit"):
        inputs.read_structure(source)
    monkeypatch.setattr(inputs, "MAX_ENCODED_BYTES", 8)
    assert inputs.resolve_inputs(tmp_path) == ([inputs.ModelInput("binder_1", None, "molprobity_input_over_limit")], None)
    with pytest.raises(inputs.MolProbityInputError, match="molprobity_input_over_limit"):
        inputs.read_structure(source)


@pytest.mark.parametrize(("suffix", "raw"), [
    (".cif", b"ATOM      1\n"),
    (".cif", b"data_x\n\xff\n"),
    (".cif.gz", gzip.compress(_CIF)[:-1]),
    (".cif.gz", gzip.compress(_CIF) + b"foreign"),
])
def test_invalid_structure_and_gzip_frames_fail_closed(tmp_path, suffix, raw):
    source = _write(tmp_path, "binder_1_generated" + suffix, raw)
    with pytest.raises(inputs.MolProbityInputError, match="molprobity_structure_invalid"):
        inputs.read_structure(source)
