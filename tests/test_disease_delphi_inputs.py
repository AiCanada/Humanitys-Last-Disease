"""Keep DELPHI nominations paired and require complete, supported IMGT domains."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rfproteina.disease import delphi_inputs as inputs

_HEADER = "candidate_id\tvh\tvl\n"


def _put(directory, contents):
    (directory / inputs.MANIFEST).write_text(contents, encoding="utf-8")


def test_explicit_pair_header_and_exact_candidate_scope(tmp_path):
    (tmp_path / "ligand_candidates.tsv").write_text("candidate_id\tsmiles\nfirst\tCCO\n", encoding="utf-8")
    assert inputs.read_pairs(tmp_path) == ([], "delphi_pairs_missing")
    _put(tmp_path, "BARCODE\tHSEQ\tLSEQ\nfirst\tAGH\tAIL\n")
    assert inputs.read_pairs(tmp_path) == ([], "delphi_pair_header_invalid")
    _put(tmp_path, _HEADER)
    assert inputs.read_pairs(tmp_path) == ([], "delphi_pairs_empty")
    _put(tmp_path, "\ufeff" + _HEADER + "first\tAGH\tAIL\nsecond\tANH\tAML\n")
    assert inputs.read_pairs(tmp_path) == ([inputs.Pair("first", "AGH", "AIL"), inputs.Pair("second", "ANH", "AML")], None)
    assert inputs.read_pairs(tmp_path, "second") == ([inputs.Pair("second", "ANH", "AML")], None)
    assert inputs.read_pairs(tmp_path, "SECOND") == ([], "delphi_candidate_not_nominated")
    assert inputs.read_pairs(tmp_path, "") == ([], "delphi_candidate_not_nominated")


def test_invalid_rows_retain_only_bounded_identifiers_and_blockers(tmp_path):
    too_long = "A" * (inputs.MAX_DOMAIN_LENGTH + 1)
    _put(
        tmp_path, _HEADER + f"duplicate\tAGH\tAIL\nduplicate\tGHI\tAML\nwide\tAC\tDE\textra\n"
        f"no_light\tAGH\t\ntoo_long\t{too_long}\tAML\nunsupported\tAZ\tAGH\n"
        "id with space\tAC\tDE\ncase\tAc\tDE\nshort\tAC\nexact\tAC\tDE\n",
    )
    rows, blocker = inputs.read_pairs(tmp_path)
    assert blocker is None
    assert [(row.candidate_id, row.blocker) for row in rows] == [
        ("duplicate", "delphi_candidate_id_duplicate"),
        ("duplicate", "delphi_candidate_id_duplicate"),
        ("wide", "delphi_pair_width_invalid"),
        ("no_light", "delphi_domains_required"),
        ("too_long", "delphi_domain_length_out_of_range"),
        ("unsupported", "delphi_domain_residues_unsupported"),
        ("", "delphi_candidate_id_invalid"),
        ("case", "delphi_domain_residues_unsupported"),
        ("short", "delphi_pair_width_invalid"),
        ("exact", None),
    ]
    assert all(not row.vh and not row.vl for row in rows[:-1])
    assert "AGH" not in repr(inputs.Pair("first", "AGH", "AIL"))
    assert inputs.read_pairs(tmp_path, "duplicate") == (rows[:2], None)


def test_an_oversized_domain_cell_does_not_discard_other_nominations(tmp_path):
    _put(tmp_path, _HEADER + "bad\t" + "A" * (inputs.MAX_BYTES // 2) + "\tAIL\n" + "first\tAGH\tAIL\n")
    assert inputs.read_pairs(tmp_path) == ([
        inputs.Pair("bad", blocker="delphi_domain_length_out_of_range"), inputs.Pair("first", "AGH", "AIL"),
    ], None)


@pytest.mark.parametrize("candidate", [None, "first"])
def test_scoping_does_not_bypass_the_byte_and_row_limits(tmp_path, candidate):
    _put(tmp_path, _HEADER + "first\tAGH\tAIL\n" + "\n" * inputs.MAX_BYTES)
    assert inputs.read_pairs(tmp_path, candidate) == ([], "delphi_pairs_over_limit")
    extra_rows = "".join(f"candidate{i}\tAGH\tAIL\n" for i in range(inputs.MAX_ROWS))
    _put(tmp_path, _HEADER + "first\tAGH\tAIL\n" + extra_rows)
    assert inputs.read_pairs(tmp_path, candidate) == ([], "delphi_pairs_over_limit")


def test_unreadable_nomination_and_no_symlink_reads(tmp_path):
    path = tmp_path / inputs.MANIFEST
    path.write_bytes(b"\xff")
    assert inputs.read_pairs(tmp_path) == ([], "delphi_pairs_unreadable")
    path.unlink()
    path.mkdir()
    assert inputs.read_manifest(tmp_path) == (None, "delphi_pairs_unreadable")
    path.rmdir()
    target = tmp_path / "elsewhere.tsv"
    target.write_text(_HEADER + "a\tAGH\tAIL\n", encoding="utf-8")
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlink not supported")
    assert inputs.read_manifest(tmp_path) == (None, "delphi_pairs_unreadable")


def test_read_rejects_a_changed_snapshot(tmp_path, monkeypatch):
    _put(tmp_path, _HEADER + "first\tAGH\tAIL\n")
    before = (tmp_path / inputs.MANIFEST).stat()
    changed = SimpleNamespace(st_size=before.st_size, st_mtime_ns=before.st_mtime_ns + 1, st_ctime_ns=before.st_ctime_ns)
    after = iter((before, changed))
    monkeypatch.setattr(inputs.os, "fstat", lambda _: next(after))
    assert inputs.read_manifest(tmp_path) == (None, "delphi_pairs_unreadable")


def _numbered(kind, cdr3=(((105, " "), "A"), ((111, " "), "R"), ((111, "A"), "Q"), ((117, " "), "D"))):
    # Similar to ANARCII's output with omitted IMGT positions represented as '-'.
    end = 128 if kind == "H" else 127
    anchors = {23: "C", 41: "W", 104: "C", **dict.fromkeys((31, 32, 33, 57, 58, 59, 60), "-")}
    residues = [((pos, " "), anchors.get(pos, "H" if kind == "H" else "L")) for pos in range(1, 105)]
    residues.extend(cdr3)
    residues.extend(((pos, " "), "T") for pos in range(118, end + 1))
    if end == 127:
        residues.append(((128, " "), "-"))
    seq = "".join(aa for label, aa in residues if aa != "-")
    return seq, {
        "scheme": "imgt", "chain_type": kind, "error": None,
        "numbering": residues, "query_start": 0, "query_end": len(seq) - 1,
    }


def test_numbering_checks_roles_keyed_by_identity_and_deduplicates_domains():
    vh, heavy = _numbered("H")
    vl, light = _numbered("K")

    def number(seqs):
        assert len(seqs) == 2
        assert set(seqs.values()) == {vh, vl}
        # Reordered results prove that pair attribution is not positional.
        return {key: heavy if seq == vh else light for key, seq in reversed(seqs.items())}

    reports = inputs.number_pairs([
        inputs.Pair("first", vh, vl), inputs.Pair("alias", vh, vl), inputs.Pair("swapped", vl, vh),
        inputs.Pair("single_domain", vh, ""), inputs.Pair("single_role", vh, vh),
    ], number=number)
    assert reports["first"].cdr3 == reports["alias"].cdr3 == "ARQD"
    assert reports["swapped"].blocker == reports["single_role"].blocker == "delphi_chain_roles_invalid"
    assert reports["single_domain"].blocker == "delphi_domains_required"
    assert "ARQD" not in repr(reports["first"])


@pytest.mark.parametrize("which", ["tail", "leading", "offset", "cysteine_gap", "internal_modified", "incomplete_light"])
def test_truncated_fused_shifted_and_inexact_domains_are_not_scored(which):
    vh, heavy = _numbered("H")
    vl, light = _numbered("L")
    if which == "tail":
        vh += "GGG"
    elif which == "leading":
        vh = "AGH" + vh
    elif which == "offset":
        heavy.update(query_start=1)
    elif which == "cysteine_gap":
        index = sum(aa != "-" for label, aa in heavy["numbering"] if label[0] < 104)
        heavy["numbering"][103] = ((104, " "), "-")
        vh = vh[:index] + vh[index + 1 :]
        heavy.update(query_end=len(vh) - 1)
    elif which == "internal_modified":
        vh = vh[:50] + "E" + vh[51:]
    else:
        # IMGT slot 127 still exists as an unoccupied gap, as in native ANARCII output.
        light["numbering"][-2] = ((127, " "), "-")
        vl = vl[:-1]
        light.update(query_end=len(vl) - 1)
    result = inputs.number_pairs([inputs.Pair("candidate", vh, vl)], number=lambda seqs: {
        key: heavy if seq == vh else light for key, seq in seqs.items()
    })
    assert result["candidate"].blocker == "delphi_domain_incomplete"
    assert result["candidate"].cdr3 is None


@pytest.mark.parametrize("cdr3", [(), (((105, " "), "C"),), tuple(((111, c), "A") for c in " ABCDEFGHIJKLMNOPQRSTUVWXY")])
def test_do_not_feed_unsupported_cdr3s_to_a_truncating_encoder(cdr3):
    vh, heavy = _numbered("H", cdr3)
    vl, light = _numbered("K")
    result = inputs.number_pairs([inputs.Pair("candidate", vh, vl)], number=lambda seqs: {
        key: heavy if seq == vh else light for key, seq in seqs.items()
    })
    assert result["candidate"].blocker == "delphi_cdr3_unsupported"


def test_numbering_backend_failure_is_named_without_reprinting_input():
    def bad_backend(sequences):
        raise RuntimeError(next(iter(sequences.values())))

    pair = inputs.Pair("first", "AGH", "AML")
    assert inputs.number_pairs([pair], number=bad_backend) == {
        "first": inputs.NumberedPair(blocker="delphi_domain_numbering_failed"),
    }
