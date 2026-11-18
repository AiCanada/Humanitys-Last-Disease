"""Numbering parse, IMGT regions, and the liability motif scan.

No model is loaded here: ANARCII takes about 28 seconds to start on CPU, so
every test feeds the parser ANARCII's own output shape instead. The real
end-to-end run is recorded in docs/screening-implementation-handoff.md.
"""

from __future__ import annotations

import pytest

from rfproteina.disease.developability import (
    IMGT_CANONICAL_CYSTEINES,
    LIABILITY_MOTIFS,
    NumberingUnavailableError,
    developability_report,
    parse_anarcii,
    read_deepsp_table,
    region_of,
    scan_chain,
)


def _anarcii(pairs: list[tuple[int, str]], *, chain_type: str = "H", scheme: str = "imgt", error=None) -> dict:
    """One ANARCII result, in the shape the real library returns it."""
    return {
        "chain_type": chain_type,
        "scheme": scheme,
        "score": 30.0,
        "error": error,
        "query_start": 0,
        "query_end": len(pairs),
        "numbering": [((position, " "), residue) for position, residue in pairs],
    }


def _chain(pairs: list[tuple[int, str]], **kwargs):
    return parse_anarcii({"H": _anarcii(pairs, **kwargs)})[0]


def test_imgt_regions_are_the_published_spans() -> None:
    assert region_of(23) == "fr1"
    assert region_of(27) == "cdr1"
    assert region_of(38) == "cdr1"
    assert region_of(39) == "fr2"
    assert region_of(56) == "cdr2"
    assert region_of(105) == "cdr3"
    assert region_of(117) == "cdr3"
    assert region_of(118) == "fr4"
    assert region_of(400) == "other"


def test_a_gap_is_not_a_residue() -> None:
    chain = _chain([(1, "E"), (2, "-"), (3, "Q")])
    assert [item.residue for item in chain.residues] == ["E", "Q"]


def test_a_numbering_in_another_scheme_is_refused_rather_than_read_as_imgt() -> None:
    with pytest.raises(NumberingUnavailableError, match="unsupported_scheme"):
        _chain([(1, "E")], scheme="kabat")


def test_an_anarcii_error_is_raised_not_swallowed() -> None:
    with pytest.raises(NumberingUnavailableError, match="anarcii_refused"):
        _chain([(1, "E")], error="Less than 50 non insertion residues numbered.")


def test_the_deamidation_and_isomerisation_motifs_are_found_in_their_regions() -> None:
    chain = _chain([(56, "N"), (57, "G"), (105, "D"), (106, "G")])
    found = {(item["liability"], item["motif"], item["region"]) for item in scan_chain(chain)}
    assert ("deamidation", "NG", "cdr2") in found
    assert ("isomerisation", "DG", "cdr3") in found


def test_a_sequon_needs_a_middle_residue_that_is_not_proline() -> None:
    flagged = _chain([(27, "N"), (28, "A"), (29, "T")])
    skipped = _chain([(27, "N"), (28, "P"), (29, "T")])
    assert any(item["liability"] == "n_glycosylation_sequon" for item in scan_chain(flagged))
    assert not any(item["liability"] == "n_glycosylation_sequon" for item in scan_chain(skipped))


def test_the_conserved_cysteines_are_not_reported_as_unpaired() -> None:
    conserved = _chain([(position, "C") for position in IMGT_CANONICAL_CYSTEINES])
    assert not any(item["liability"] == "unpaired_cysteine" for item in scan_chain(conserved))


def test_an_extra_cysteine_is_reported_with_the_reading_it_took() -> None:
    chain = _chain([(23, "C"), (104, "C"), (107, "C")])
    found = [item for item in scan_chain(chain) if item["liability"] == "unpaired_cysteine"]
    assert len(found) == 1
    assert found[0]["parity"] == "odd_count_at_least_one_is_unpaired"
    pair = _chain([(23, "C"), (104, "C"), (107, "C"), (110, "C")])
    both = [item for item in scan_chain(pair) if item["liability"] == "unpaired_cysteine"]
    assert {item["parity"] for item in both} == {"even_count_may_be_an_extra_disulfide"}


def test_cdr_only_drops_framework_findings() -> None:
    chain = _chain([(10, "M"), (105, "M")])
    assert len(scan_chain(chain)) == 2
    assert len(scan_chain(chain, cdr_only=True)) == 1


def test_every_motif_names_the_publication_it_came_from() -> None:
    for motif in LIABILITY_MOTIFS:
        assert motif.source.strip()
        assert motif.provenance == "cited"
        assert motif.rationale.strip()


def test_every_finding_carries_its_source_so_a_reader_cannot_lose_it() -> None:
    chain = _chain([(56, "N"), (57, "G")])
    for item in scan_chain(chain):
        assert item["source"]
        assert item["scheme"] == "imgt"


def test_the_report_admits_it_has_no_pass_cut() -> None:
    chain = _chain([(56, "N"), (57, "G"), (105, "W")])
    record = developability_report([chain])
    assert record["pass_cut"] is None
    assert record["pass_cut_reason"] == "no_published_motif_count_threshold"
    assert "exposure" in str(record["caveat"]).lower()
    assert record["counts"]["deamidation"] == 1
    assert record["chains"][0]["cdr_lengths"]["cdr2"] == 2


def test_an_empty_chain_list_is_a_blocker_not_an_empty_report() -> None:
    assert developability_report([]) == {"developability_blocker": "no_numbered_chains"}


def test_a_csv_that_is_not_deepsp_output_is_refused(tmp_path) -> None:
    path = tmp_path / "something.csv"
    path.write_text("name,score\nab,0.5\n", encoding="utf-8")
    record = read_deepsp_table(path)
    assert record["oracle_blocker"] == "oracle_columns_absent"


def test_a_deepsp_csv_is_read_with_its_provenance(tmp_path) -> None:
    """DeepSP suffixes every descriptor with a region, so the whole-Fv ones are read.

    This wrote ``SAP_pos,SCM_pos,SCM_neg`` until 2026-11-18. DeepSP emits no
    column by any of those names -- its own are ``SAP_pos_Fv`` and the rest --
    so the fixture matched a contract the tool never produced, and real output
    was refused as ``oracle_columns_absent``.
    """
    path = tmp_path / "deepsp.csv"
    path.write_text("Name,SAP_pos_Fv,SCM_pos_Fv,SCM_neg_Fv\nab1,120.5,-300.0,250.0\n", encoding="utf-8")
    record = read_deepsp_table(path)
    assert record["method"] == "deepsp"
    assert record["rows"][0]["scores"]["SAP_pos_Fv"] == 120.5
    assert record["rows"][0]["name"] == "ab1", "DeepSP writes a capital Name"
    assert record["pass_cut"] is None
    assert "MIT" in str(record["source"])


def test_an_unparsable_cell_is_refused_rather_than_dropped(tmp_path) -> None:
    path = tmp_path / "deepsp.csv"
    path.write_text("name,SAP_pos_Fv\nab1,high\n", encoding="utf-8")
    assert read_deepsp_table(path)["oracle_blocker"] == "cell_unparsed"
