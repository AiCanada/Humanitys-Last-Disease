"""Tests for the paralog-fed off-target transform. No network, AFDB fetch injected."""

from __future__ import annotations

import json

import pytest

from rfproteina.metrics.oracles.paralog_off_target import (
    OffTargetUnavailableError,
    ParalogOffTargetSubstitutionTransform,
    read_paralog_identifiers,
)


def _patch_parent_setup(monkeypatch):
    """Neutralise the parent's structure-parsing setup so tests stay offline/light."""
    from rfproteina.metrics.oracles import pre_oracle_transform

    monkeypatch.setattr(pre_oracle_transform.OffTargetSubstitutionTransform, "setup", lambda self: None)


def _ok_fetch(acc, version, dest):
    dest.write_text(f"# AFDB {acc} v{version}\n")
    return True


def test_read_paralog_identifiers():
    record = {"paralogs": ["EGFR", "ERBB3", "EGFR"]}
    assert read_paralog_identifiers(record) == ["EGFR", "ERBB3"]


def test_read_paralog_identifiers_empty():
    assert read_paralog_identifiers({}) == []
    assert read_paralog_identifiers({"paralogs": None}) == []


def test_requires_a_source():
    with pytest.raises(ValueError, match="paralog_accessions or paralog_record_path"):
        ParalogOffTargetSubstitutionTransform()


def test_candidate_identifiers_from_list():
    t = ParalogOffTargetSubstitutionTransform(paralog_accessions=["P04626", "P00533"])
    assert t._candidate_identifiers() == {"P04626": "P04626", "P00533": "P00533"}


def test_candidate_identifiers_from_record_path(tmp_path):
    record = tmp_path / "paralog.json"
    record.write_text(json.dumps({"paralogs": ["EGFR", "ERBB3"]}))
    t = ParalogOffTargetSubstitutionTransform(paralog_record_path=record)
    assert t._candidate_identifiers() == {"EGFR": "EGFR", "ERBB3": "ERBB3"}


def test_max_targets_caps():
    t = ParalogOffTargetSubstitutionTransform(paralog_accessions=["P1", "P2", "P3"], max_targets=2)
    assert len(t._candidate_identifiers()) == 2


def test_resolve_accession_direct():
    t = ParalogOffTargetSubstitutionTransform(paralog_accessions=["P04626"])
    assert t._resolve_accession("P04626") == "P04626"


def test_resolve_symbol_via_injected_resolver():
    t = ParalogOffTargetSubstitutionTransform(
        paralog_accessions=["EGFR"], symbol_resolver=lambda s: "P00533" if s == "EGFR" else None
    )
    assert t._resolve_accession("EGFR") == "P00533"


def test_resolve_symbol_disabled():
    t = ParalogOffTargetSubstitutionTransform(paralog_accessions=["EGFR"], resolve_symbols=False)
    assert t._resolve_accession("EGFR") is None


def test_setup_success_fetches_and_sets_ids(tmp_path, monkeypatch):
    _patch_parent_setup(monkeypatch)
    t = ParalogOffTargetSubstitutionTransform(
        paralog_accessions=["P04626", "P00533"],
        cache_dir=tmp_path / "cache",
        fetch_fn=_ok_fetch,
    )
    t.setup()
    assert set(t.off_target_paths) == {"P04626", "P00533"}
    assert t.expected_save_ids() == ["P04626", "P00533"]
    for path in t.off_target_paths.values():
        assert path.exists()


def test_setup_uses_cache_without_refetch(tmp_path, monkeypatch):
    _patch_parent_setup(monkeypatch)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "AF-P04626-F1-model_v4.cif").write_text("# cached\n")
    calls = []

    def counting_fetch(acc, version, dest):
        calls.append(acc)
        return _ok_fetch(acc, version, dest)

    t = ParalogOffTargetSubstitutionTransform(
        paralog_accessions=["P04626"], cache_dir=cache, fetch_fn=counting_fetch
    )
    t.setup()
    assert calls == []  # already cached, no fetch


def test_setup_raises_when_nothing_resolves(tmp_path, monkeypatch):
    _patch_parent_setup(monkeypatch)
    t = ParalogOffTargetSubstitutionTransform(
        paralog_accessions=["NOTAGENE"],
        cache_dir=tmp_path / "cache",
        resolve_symbols=True,
        symbol_resolver=lambda s: None,
        fetch_fn=_ok_fetch,
    )
    with pytest.raises(OffTargetUnavailableError, match="None of the paralog"):
        t.setup()


def test_setup_raises_when_no_candidates(tmp_path, monkeypatch):
    _patch_parent_setup(monkeypatch)
    record = tmp_path / "paralog.json"
    record.write_text(json.dumps({"paralogs": []}))
    t = ParalogOffTargetSubstitutionTransform(paralog_record_path=record, cache_dir=tmp_path / "c")
    with pytest.raises(OffTargetUnavailableError, match="No paralog identifiers"):
        t.setup()


def test_require_all_raises_on_fetch_failure(tmp_path, monkeypatch):
    _patch_parent_setup(monkeypatch)
    t = ParalogOffTargetSubstitutionTransform(
        paralog_accessions=["P04626"],
        cache_dir=tmp_path / "cache",
        require_all=True,
        fetch_fn=lambda a, v, d: False,
    )
    with pytest.raises(OffTargetUnavailableError, match="could not fetch"):
        t.setup()


def test_expected_save_ids_before_setup():
    t = ParalogOffTargetSubstitutionTransform(paralog_accessions=["P04626", "P00533"])
    assert t.expected_save_ids() == ["P04626", "P00533"]
