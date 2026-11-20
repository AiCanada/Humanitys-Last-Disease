"""The H.U.M.A.N  Protocol: what the design must not bind, before the target.

No network, no MMseqs2 and no download. The alignment pass is injected -- the
same way ``offtarget`` and ``stability`` are already testable -- and the
proteome is a four-entry FASTA written into ``tmp_path``, so every reading here
is produced by this module's own arithmetic rather than by what happens to be
installed.

The structure route is not injected. It runs the real interface code on a real
two-chain PDB built below, because the thing worth testing there is that a
margin is taken between two measurements of the same kind, and a stubbed
measurement cannot show that.
"""

import gzip
import json
from pathlib import Path

import pytest

from rfproteina.disease import human_protocol as hp
from rfproteina.disease import results_actions, ui
from rfproteina.disease.offtarget import OffTargetHit
from rfproteina.disease.screening_tools import CLEAR, HIT, UNTESTED
from tests.helpers.disease import no_developer_proteome  # noqa: F401

#: A four-entry stand-in for the reference proteome, in UniProt's own header
#: shape. Two of the four are a family, which is what the proposal is for.
_FASTA = (
    ">sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo sapiens OX=9606 GN=HBA1 PE=1 SV=2\n"
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGK\n"
    "KVADALTNAVAHVDDMPNALSALSDLHAHKLR\n"
    ">sp|P68871|HBB_HUMAN Hemoglobin subunit beta OS=Homo sapiens OX=9606 GN=HBB PE=1 SV=2\n"
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKV\n"
    ">sp|P02768|ALBU_HUMAN Albumin OS=Homo sapiens OX=9606 GN=ALB PE=1 SV=2\n"
    "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFE\n"
    ">sp|P05113|IL5_HUMAN Interleukin-5 OS=Homo sapiens OX=9606 GN=IL5 PE=1 SV=1\n"
    "MRMLLHLSLLALGAAYVYAIPTEIPTSALVKETLALLSTHRTLLIANETLRIPVPVHKNHQ\n"
)


@pytest.fixture
def proteome(tmp_path: Path) -> Path:
    """The stand-in proteome as a plain FASTA."""
    path = tmp_path / "mini.fasta"
    path.write_text(_FASTA, encoding="utf-8")
    return path


def _hit(target: str, *, identity: float, qcov: float = 0.7, tcov: float = 0.6, bits: float = 100.0) -> OffTargetHit:
    """One alignment row, in the shape ``parse_search_table`` produces.

    MMseqs2 reports a UniProt-style header as the bare accession, which is why
    ``target`` is one here and not ``sp|P69905|HBA_HUMAN``.
    """
    return OffTargetHit("design_1", target, identity, 80, qcov, tcov, 1e-20, bits)


def _scan(*hits: OffTargetHit, design: str = "design_1") -> hp.Scan:
    """A pass that ran, holding these rows for one design."""
    return hp.Scan({design: tuple(hits)}, designs=(design,), evalue_max=1e-3)


_DESIGNS = [("design_1", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF")]


# --- the vocabulary ----------------------------------------------------------


def test_the_two_routes_are_a_closed_set_and_each_states_both_halves():
    """A route with no "does not answer" is the half of the record that matters."""
    assert set(hp.ROUTES) == {hp.SEQUENCE_RESEMBLANCE, hp.INTERFACE_CONTACTS}
    assert set(hp.ROUTE_ANSWERS) == set(hp.ROUTES)
    assert set(hp.ROUTE_DOES_NOT_ANSWER) == set(hp.ROUTES)
    for route in hp.ROUTES:
        assert hp.ROUTE_ANSWERS[route].endswith(".")
        assert hp.ROUTE_DOES_NOT_ANSWER[route].endswith(".")


def test_the_candidate_sources_are_a_closed_set():
    assert set(hp.CANDIDATE_SOURCES) == {hp.PROTEOME_ALIGNMENT, hp.NAMED_BY_OPERATOR}


def test_every_sentence_reads_as_one():
    """A blocker name is the whole explanation a reader gets, so it has to be prose."""
    for name, said in hp._BLOCKER_TEXT.items():
        assert said.endswith("."), name
        assert name not in said, f"{name} explains itself with its own slug"


def test_an_unknown_blocker_renders_as_its_spaced_name_rather_than_nothing():
    assert hp.sentence("some_future_name") == "some future name"
    assert hp.sentence("") == ""


def test_a_protocol_that_could_not_start_still_carries_a_record_a_page_can_draw(tmp_path: Path, proteome: Path):
    """Same keys as a full one, so nothing drawing it needs a special case.

    It exists at all because the results page reaches this state -- a catalog
    with no readable design -- and a blocker named from *there* would be
    checked against the results page's own vocabulary, which speaks in
    fragments for a different surface.
    """
    blocked = hp.blocked_record(hp.NO_DESIGNS)
    whole = hp.protocol_record(
        _DESIGNS, names=("ALB",), proteome=proteome, work_dir=tmp_path, search=lambda d, w: _scan()
    )
    assert set(blocked) == set(whole)
    assert blocked["blocker"] == hp.NO_DESIGNS
    assert blocked["reason"] == hp._BLOCKER_TEXT[hp.NO_DESIGNS]
    assert blocked["call"] == UNTESTED
    assert blocked["routes"] == []
    assert json.loads(json.dumps(blocked, allow_nan=False)) == blocked


def test_the_label_keeps_the_spacing_it_was_given():
    """The panel title spaces ``PROJECT H.U.M.A.N   V0.8888`` the same way on purpose."""
    assert hp.PROTOCOL_LABEL == "The H.U.M.A.N  Protocol"


def test_the_limit_sentence_says_which_question_is_unanswered():
    assert "docks or folds" in hp.PROTOCOL_LIMIT
    assert "not selectivity" in hp.PROTOCOL_LIMIT


# --- what the reader typed ---------------------------------------------------


def test_names_are_de_duplicated_with_the_reader_s_order_kept():
    """The order is their own ranking of what matters, and the readout is drawn in it."""
    assert hp.parse_names("P02768, hba1; ALB  P02768") == ("P02768", "HBA1", "ALB")
    assert hp.parse_names("") == ()
    assert hp.parse_names("   ") == ()


def test_a_pasted_column_is_bounded_before_it_is_split(tmp_path: Path):
    """The box is a browser field, so it is where untrusted text is bounded.

    The cut is on the whole box and comes before the split, which is what makes
    the two bounds below cheap: nothing downstream ever allocates the rest.
    """
    flood = " ".join(f"P{index:05d}" for index in range(100_000))
    found = hp.parse_names(flood)
    assert len(found) <= hp._NAME_FLOOD
    # Several times the selection cap, so the caller's own refusal is what the
    # reader sees rather than a silently shorter list.
    assert len(found) > hp.SELECTION_LIMIT


def test_several_boxes_are_bounded_before_they_are_joined():
    """The tab posts one value per tick box plus whatever was typed.

    Joining first and bounding after would allocate the whole of a hostile
    submission in order to throw it away, which is the bound not doing its job.
    """
    many = ["X" * 100_000] * 10_000
    found = hp.parse_many(many)
    assert len(found) <= hp._NAME_FLOOD
    assert all(len(name) <= hp._NAME_LIMIT for name in found)
    assert hp.parse_many(["P02768", "HBA1 ALB"]) == ("P02768", "HBA1", "ALB")
    assert hp.parse_many([]) == ()


def test_an_over_long_name_is_kept_short_rather_than_dropped():
    """Dropped, it would vanish between the box and the result with nothing said."""
    found = hp.parse_names("P02768 " + "X" * 5000)
    assert found[0] == "P02768"
    assert len(found) == 2
    assert len(found[1]) == hp._NAME_LIMIT


def test_a_selection_over_the_cap_never_opens_the_proteome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Walking a 7.7 MB file for a list that is about to be refused is work nobody asked for."""
    opened: list[object] = []
    monkeypatch.setattr(hp, "_proteome_entries", lambda path: opened.append(path) or iter(()))
    path = tmp_path / "mini.fasta"
    path.write_text(_FASTA, encoding="utf-8")
    record = hp.protocol_record(
        _DESIGNS,
        names=(" ".join(f"P{index:05d}" for index in range(hp.SELECTION_LIMIT + 1)),),
        proteome=path,
        work_dir=tmp_path,
        search=lambda designs, where: _scan(),
    )
    assert record["routes"][0]["blocker"] == "too_many_anti_targets"
    assert opened == []


def test_a_gene_symbol_is_not_an_accession():
    assert hp.is_accession("P69905")
    assert hp.is_accession("p69905-2")
    assert not hp.is_accession("HBA1")
    assert not hp.is_accession("")


def test_a_uniprot_header_yields_the_accession_entry_gene_and_description():
    found = hp._header_fields(">sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo sapiens OX=9606 GN=HBA1 PE=1 SV=2")
    assert found == ("P69905", "HBA_HUMAN", "HBA1", "Hemoglobin subunit alpha")


def test_a_header_in_another_shape_yields_what_it_can_and_guesses_nothing():
    assert hp._header_fields(">design_7 something") == ("DESIGN_7", "design_7", "", "something")
    assert hp._header_fields(">") == ("", "", "", "")


def test_a_long_description_is_cut_rather_than_stored_whole():
    long = "x" * 400
    _accession, _entry, _gene, described = hp._header_fields(f">sp|P1|E_HUMAN {long} GN=G")
    assert len(described) <= hp._TEXT_LIMIT
    assert described.endswith("...")


# --- the proteome ------------------------------------------------------------


def test_an_entry_is_found_by_accession_entry_name_or_gene_symbol(proteome: Path):
    found, missing = hp.proteome_records(("P69905", "HBB_HUMAN", "ALB"), proteome)
    assert missing == ()
    assert {key: entry.accession for key, entry in found.items()} == {
        "P69905": "P69905",
        "HBB_HUMAN": "P68871",
        "ALB": "P02768",
    }
    assert found["ALB"].description == "Albumin"


def test_a_name_the_proteome_does_not_hold_comes_back_as_missing(proteome: Path):
    found, missing = hp.proteome_records(("P69905", "NOTAPROTEIN"), proteome)
    assert set(found) == {"P69905"}
    assert missing == ("NOTAPROTEIN",)


def test_a_gzipped_proteome_is_read(tmp_path: Path):
    """UniProt ships it gzipped and the download registry fetches exactly that file."""
    path = tmp_path / "mini.fasta.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(_FASTA)
    found, missing = hp.proteome_records(("HBA1",), path)
    assert missing == ()
    assert found["HBA1"].accession == "P69905"


def test_no_proteome_names_every_choice_missing_rather_than_claiming_none_exists():
    found, missing = hp.proteome_records(("P69905",), None)
    assert found == {}
    assert missing == ("P69905",)


def test_which_of_the_two_proteome_faults_it_was(tmp_path: Path, proteome: Path):
    """Unset and unreadable send a reader to different places, so they are told apart."""
    assert hp.proteome_blocker(None) == "proteome_not_configured"
    assert hp.proteome_blocker(tmp_path / "absent.fasta") == "proteome_unreadable"
    assert hp.proteome_blocker(proteome) is None


# --- the alignment pass ------------------------------------------------------


def test_a_pass_with_no_design_sequence_blocks_rather_than_running(tmp_path: Path, proteome: Path):
    scan = hp.scan_designs([("design_1", "   ")], proteome, tmp_path)
    assert not scan.ran
    assert scan.blocker == "design_sequence_absent"
    assert scan.hits == {}


def test_a_pass_with_no_proteome_never_reaches_the_tool(tmp_path: Path):
    scan = hp.scan_designs(_DESIGNS, None, tmp_path)
    assert scan.blocker == "proteome_not_configured"
    scan = hp.scan_designs(_DESIGNS, tmp_path / "absent.fasta", tmp_path)
    assert scan.blocker == "proteome_unreadable"


def test_a_library_handed_in_as_the_designs_is_refused(tmp_path: Path, proteome: Path):
    """``--human-protocol`` takes a file a reader named, and a proteome is a file.

    Aligning one proteome against another is not this screen's question, and
    behind a button it is not work anybody asked for.
    """
    many = [(f"design_{index}", "MVLSPADKTNVKAAWGK") for index in range(hp.DESIGN_LIMIT + 1)]
    scan = hp.scan_designs(many, proteome, tmp_path)
    assert scan.blocker == "too_many_designs"
    assert not (tmp_path / "protocol_designs.fasta").exists()


def test_a_design_longer_than_one_chain_is_refused(tmp_path: Path, proteome: Path):
    long = [("design_1", "A" * (hp.DESIGN_RESIDUE_LIMIT + 1))]
    assert hp.scan_designs(long, proteome, tmp_path).blocker == "design_over_residue_limit"
    # The same bound the refold request keeps, so the two agree on what one
    # design's chain is rather than holding two opinions about it.
    from rfproteina.disease.refold_export import REQUEST_RESIDUE_LIMIT

    assert hp.DESIGN_RESIDUE_LIMIT == REQUEST_RESIDUE_LIMIT


def test_the_pass_is_not_given_an_hour_behind_a_button():
    """A measured 4.6 s pass does not need the batch screen's timeout.

    ``offtarget`` allows an hour because it screens a finished batch in the
    background. This one holds a panel request open while it runs.
    """
    assert hp.SCAN_TIMEOUT_S == 600.0


def test_a_pass_that_did_not_run_holds_no_hits_and_says_so():
    """An empty hit list from a pass that never happened must not read as "resembles nothing"."""
    blocked = hp.Scan({}, blocker="mmseqs_not_installed")
    assert not blocked.ran
    assert hp._best_per_accession(blocked) == {}


def test_the_strongest_alignment_across_designs_is_the_one_reported():
    """Two designs, one protein. The worse design is the one that matters."""
    scan = hp.Scan(
        {
            "design_1": (_hit("P69905", identity=0.3, bits=40.0),),
            "design_2": (_hit("P69905", identity=0.9, bits=200.0),),
        },
        designs=("design_1", "design_2"),
    )
    best = hp._best_per_accession(scan)
    assert set(best) == {"P69905"}
    design, hit = best["P69905"]
    assert design == "design_2"
    assert hit.fident == 0.9


# --- the resemblance route ---------------------------------------------------


def test_a_protein_the_proteome_does_not_hold_is_not_cleared(proteome: Path):
    """The load-bearing one.

    No row can exist for an entry the searched file does not contain, so its
    absence from the hits is evidence of nothing -- and that is exactly the
    case where a clear would be most confidently wrong.
    """
    found, _missing = hp.proteome_records(("NOTAPROTEIN",), proteome)
    chosen = hp.anti_targets_from(("NOTAPROTEIN",), found)
    report = hp.resemblance_report(_scan(), chosen, found)
    row = report["readings"][0]
    assert row["state"] == UNTESTED
    assert row["blocker"] == "anti_target_not_in_proteome"
    assert row["state"] != CLEAR


def test_no_alignment_over_the_cut_is_a_clear_for_this_route(proteome: Path):
    found, _missing = hp.proteome_records(("ALB",), proteome)
    report = hp.resemblance_report(_scan(), hp.anti_targets_from(("ALB",), found), found)
    row = report["readings"][0]
    assert row["state"] == CLEAR
    assert row["alignment"] is False
    assert row["identity"] is None


def test_a_resemblance_is_reported_and_not_judged_without_a_cut(proteome: Path):
    """Measuring and clearing are different acts, and this module will only do one."""
    found, _missing = hp.proteome_records(("HBA1",), proteome)
    chosen = hp.anti_targets_from(("HBA1",), found)
    report = hp.resemblance_report(_scan(_hit("P69905", identity=0.62)), chosen, found)
    row = report["readings"][0]
    assert row["identity"] == 0.62
    assert row["state"] == UNTESTED
    assert row["blocker"] == "resemblance_threshold_not_set"
    assert report["thresholds"]["resemblance_identity_max"] is None


def test_a_cut_turns_the_same_resemblance_into_a_hit(proteome: Path):
    found, _missing = hp.proteome_records(("HBA1",), proteome)
    chosen = hp.anti_targets_from(("HBA1",), found)
    scan = _scan(_hit("P69905", identity=0.62))
    limits = hp.ProtocolLimits(resemblance_identity_max=0.5)
    row = hp.resemblance_report(scan, chosen, found, limits=limits)["readings"][0]
    assert row["state"] == HIT
    assert row["blocker"] is None


def test_a_resemblance_under_the_coverage_floor_is_not_a_hit(proteome: Path):
    """A nine-residue perfect match in a long design is not resembling the protein."""
    found, _missing = hp.proteome_records(("HBA1",), proteome)
    chosen = hp.anti_targets_from(("HBA1",), found)
    scan = _scan(_hit("P69905", identity=1.0, qcov=0.05, tcov=0.04))
    limits = hp.ProtocolLimits(resemblance_identity_max=0.5, resemblance_coverage_min=0.5)
    row = hp.resemblance_report(scan, chosen, found, limits=limits)["readings"][0]
    assert row["identity"] == 1.0
    assert row["state"] == CLEAR


def test_nothing_chosen_declines_rather_than_clearing(proteome: Path):
    report = hp.resemblance_report(_scan(), (), {})
    assert report["blocker"] == "anti_targets_not_named"
    assert report["readings"] == []


def test_a_selection_over_the_cap_declines_rather_than_drawing_it(proteome: Path):
    chosen = tuple(hp.AntiTarget(f"P{index:05d}") for index in range(hp.SELECTION_LIMIT + 1))
    report = hp.resemblance_report(_scan(), chosen, {})
    assert report["blocker"] == "too_many_anti_targets"


def test_a_blocked_pass_blocks_the_whole_route_and_carries_the_tool_s_reason():
    blocked = hp.Scan({}, blocker="search_failed", tool_blocker="tool_error:mmseqs:1")
    report = hp.resemblance_report(blocked, (hp.AntiTarget("P69905"),), {})
    assert report["blocker"] == "search_failed"
    assert report["tool_blocker"] == "tool_error:mmseqs:1"
    assert report["readings"] == []


def test_the_tool_s_reason_carries_no_path():
    """``run_tool`` builds its blocker out of ``command[0]``, an absolute path.

    The record is held in a panel session and can be saved, and ``AGENTS.md``
    asks that a path identifying this machine not be among the things that
    leave it. The kind and the exit code are the whole of what a reader needs.
    """
    # A generic path, not this machine's. The whole point of the function is
    # that a checkout's absolute path does not leave it, and writing the real
    # one into the test that proves it would have committed the thing being
    # stripped -- which is what ``AGENTS.md`` names first among personal data.
    said = hp._tool_reason("tool_error:/checkout/screening subpackage/mmseqs/mmseqs:1")
    assert said == "tool_error:mmseqs:1"
    assert hp._tool_reason("tool_failed:C:\\Users\\someone\\mmseqs.exe") == "tool_failed:mmseqs.exe"
    assert hp._tool_reason("tool_timeout:mmseqs") == "tool_timeout:mmseqs"
    assert hp._tool_reason(None) == ""


def test_the_route_record_states_what_it_does_not_answer(proteome: Path):
    report = hp.resemblance_report(_scan(), (), {})
    assert report["does_not_answer"] == hp.RESEMBLANCE_CAVEAT
    assert "not a homolog" in report["does_not_answer"]


# --- the proposal ------------------------------------------------------------


def test_the_proposal_is_strongest_first():
    scan = _scan(
        _hit("P02768", identity=0.2, bits=30.0),
        _hit("P69905", identity=0.9, bits=200.0),
        _hit("P68871", identity=0.5, bits=90.0),
    )
    assert [item.accession for item in hp.propose_anti_targets(scan)] == ["P69905", "P68871", "P02768"]


def test_the_intended_target_is_kept_out_of_the_proposal():
    """A binder aimed at a human protein resembles that protein most of all."""
    scan = _scan(_hit("P69905", identity=1.0, bits=300.0), _hit("P68871", identity=0.4, bits=70.0))
    proposed = hp.propose_anti_targets(scan, exclude=("P69905",))
    assert [item.accession for item in proposed] == ["P68871"]


def test_the_proposal_is_capped():
    scan = _scan(*[_hit(f"P{index:05d}", identity=0.5, bits=float(index)) for index in range(40)])
    assert len(hp.propose_anti_targets(scan, limit=5)) == 5
    assert hp.propose_anti_targets(scan, limit=0) == ()


def test_the_proposal_names_no_entry_it_did_not_read():
    """MMseqs2 hands back the accession, so there is no entry name in the hit."""
    proposed = hp.propose_anti_targets(_scan(_hit("P69905", identity=0.9)))
    assert proposed[0].entry == ""
    assert proposed[0].gene == ""
    assert proposed[0].source == hp.PROTEOME_ALIGNMENT
    assert proposed[0].label == "P69905"


def test_enrich_fills_the_symbol_from_the_proteome(proteome: Path):
    """So the list a reader ticks says HBA1 rather than P69905."""
    filled = hp.enrich(hp.propose_anti_targets(_scan(_hit("P69905", identity=0.9))), proteome)
    assert filled[0].gene == "HBA1"
    assert filled[0].label == "HBA1"
    assert filled[0].description == "Hemoglobin subunit alpha"
    assert filled[0].identity == 0.9


def test_a_target_named_by_symbol_is_still_kept_out_of_the_proposal(proteome: Path):
    """Matched as typed, ``HBA1`` did not equal ``P69905``.

    The protein the design was aimed at then led the list of things to avoid,
    which is noise, and noise at the top.
    """
    assert hp.resolve("HBA1", proteome) == "P69905"
    assert hp.resolve("P69905", proteome) == "P69905"
    assert hp.resolve("NOTAPROTEIN", proteome) == "NOTAPROTEIN"
    assert hp.resolve("", proteome) == ""
    scan = _scan(_hit("P69905", identity=1.0, bits=300.0), _hit("P02768", identity=0.4, bits=70.0))
    proposed = hp.propose_anti_targets(scan, exclude=(hp.resolve("HBA1", proteome),))
    assert [item.accession for item in proposed] == ["P02768"]


def test_one_protein_named_twice_is_one_protein(proteome: Path):
    """``parse_names`` cannot see that ``ALB`` and ``P02768`` are the same entry.

    Ticking the box and typing the accession drew two identical rows, and the
    headline sentence counted "2 proteins chosen as off-limits" for one.
    """
    found, _missing = hp.proteome_records(("ALB", "P02768", "ALBU_HUMAN"), proteome)
    chosen = hp.anti_targets_from(("ALB", "P02768", "ALBU_HUMAN"), found)
    assert [item.accession for item in chosen] == ["P02768"]
    # Two names the proteome does not hold are still two names.
    assert len(hp.anti_targets_from(("NOPE", "ALSONOPE"), {})) == 2


def test_enrich_keeps_a_candidate_the_proteome_does_not_name(proteome: Path):
    """It was proposed because the alignment found it, and that stands either way."""
    filled = hp.enrich((hp.AntiTarget("Q00000", identity=0.4),), proteome)
    assert [item.accession for item in filled] == ["Q00000"]
    assert hp.enrich((), proteome) == ()


# --- the interface route -----------------------------------------------------


def _pdb(chains: dict[str, int], *, gap: float = 4.5) -> str:
    """A PDB of alanine chains, the second offset by ``gap`` so the two contact.

    Written by column rather than with f-string spacing, because PDB is a
    fixed-column format and ``results_compare`` reads it as one.
    """
    lines: list[str] = []
    serial = 1
    for index, (chain, residues) in enumerate(chains.items()):
        offset = index * gap
        for number in range(1, residues + 1):
            base = (number * 3.8, offset, 0.0)
            for name, shift in (("N", -1.2), ("CA", 0.0), ("C", 1.2), ("O", 1.8), ("CB", 0.6)):
                x, y, z = base[0] + shift, base[1], base[2] + (0.9 if name == "CB" else 0.0)
                element = name[0]
                lines.append(
                    "ATOM  "
                    + f"{serial:5d}"
                    + " "
                    + f" {name:<3}"
                    + " "
                    + "ALA"
                    + " "
                    + chain
                    + f"{number:4d}"
                    + "    "
                    + f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    + f"{1.00:6.2f}{0.00:6.2f}"
                    + " " * 10
                    + f"{element:>2}"
                )
                serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


_PAIR = _pdb({"A": 10, "B": 10})


def test_the_built_pdb_is_read_as_one(tmp_path: Path):
    """Guards every interface test below against passing on an unreadable file."""
    from rfproteina.disease.results_compare import chain_labels, looks_like_pdb

    assert looks_like_pdb(_PAIR)
    assert chain_labels(_PAIR) == ("A", "B")


def test_an_interface_is_measured_on_both_chains(tmp_path: Path):
    reading = hp.measure_one(_PAIR, "A", tmp_path, where="on_target")
    assert reading.blocker == ""
    assert reading.measured
    assert reading.binder_chain == "A"
    assert reading.partner_chains == ("B",)
    assert reading.dsasa_total_a2 is not None and reading.dsasa_total_a2 > 0
    assert reading.method


def test_an_empty_model_declines_with_no_numbers(tmp_path: Path):
    reading = hp.measure_one("", "A", tmp_path, where="P02768")
    assert reading.blocker == "complex_not_supplied"
    assert not reading.measured
    assert reading.dsasa_total_a2 is None


def test_a_structure_holding_only_the_design_has_no_interface(tmp_path: Path):
    """Named for the fault, because this runs on both sides of the comparison.

    ``anti_target_chain_absent`` read wrongly on the on-target side: the same
    measurement runs on this run's own complex, where "the supplied model" is
    not what was read.
    """
    reading = hp.measure_one(_pdb({"A": 10}), "A", tmp_path, where="P02768")
    assert reading.blocker == "no_partner_chain"
    on_target = hp.measure_one(_pdb({"A": 10}), "A", tmp_path, where="on_target")
    assert on_target.blocker == "no_partner_chain"
    assert "supplied model" not in hp.sentence("no_partner_chain")


def test_a_chain_the_model_does_not_hold_is_not_measured(tmp_path: Path):
    reading = hp.measure_one(_PAIR, "Z", tmp_path, where="P02768")
    assert reading.blocker == "design_chain_not_found"


def test_text_that_is_not_a_structure_is_refused(tmp_path: Path):
    assert hp.measure_one("not a structure", "A", tmp_path, where="x").blocker == "complex_unreadable"


def test_a_structure_over_the_byte_gate_is_not_parsed(tmp_path: Path):
    """The gate comes first because these terms cost per atom.

    A SASA and a dot surface over a pasted 60 MiB file would allocate gigabytes
    inside a panel request, so the size is checked before the parse rather than
    after it.
    """
    huge = "ATOM  " + "x" * hp.MODEL_BYTE_LIMIT
    reading = hp.measure_one(huge, "A", tmp_path, where="P02768")
    assert reading.blocker == "model_too_large"
    assert reading.dsasa_total_a2 is None
    assert list(tmp_path.iterdir()) == []


def test_a_structure_with_more_chains_than_one_complex_is_refused(tmp_path: Path):
    many = _pdb({chr(ord("A") + index): 4 for index in range(hp.MODEL_CHAIN_LIMIT + 1)})
    assert hp.measure_one(many, "A", tmp_path, where="P02768").blocker == "model_holds_too_many_chains"


def test_a_protein_name_cannot_become_a_path(tmp_path: Path):
    """``where`` arrives from a browser field and was interpolated into a path.

    The separators are replaced the way an uploaded file name's are, and the
    fixed prefix means even ``..`` cannot name the parent.
    """
    inside = tmp_path / "work"
    for hostile in ("../../escaped", "..", "/etc/passwd", "a\\b", ""):
        path = hp._work_file(inside, hostile, _PAIR)
        assert path.parent == inside, hostile
        assert ".." not in path.name, hostile
        assert "/" not in path.name and "\\" not in path.name, hostile
    reading = hp.measure_one(_PAIR, "A", inside, where="../../escaped")
    assert reading.measured
    assert not (tmp_path.parent / "escaped.pdb").exists()


def test_a_structure_that_defeats_the_measurement_does_not_end_the_run(tmp_path: Path, monkeypatch):
    """The run has already finished, and a button must not be able to end it.

    A MemoryError or a numpy failure from a file that cleared both gates is
    neither an OSError nor a ValueError, so it would pass ``_results_action``'s
    catch and take the panel down with it.
    """
    import rfproteina.disease.interface_quality as iq

    def explode(*args, **kwargs):
        raise MemoryError("not enough room for the dot surface")

    monkeypatch.setattr(iq, "measure_interface", explode)
    reading = hp.measure_one(_PAIR, "A", tmp_path, where="P02768")
    assert reading.blocker == "interface_unreadable"
    assert "MemoryError" in reading.detail
    assert not reading.measured


def test_a_failure_detail_carries_no_path_and_no_essay(tmp_path: Path):
    """It is stored in a record a reader can download, and drawn in a table cell.

    A path belongs in a log rather than in something that leaves the machine,
    and an arbitrary exception's text has no length -- a numpy failure can
    carry a whole array's repr.
    """
    said = hp._detail(OSError(28, "No space left on device", str(tmp_path / "secret.cif")))
    assert "secret.cif" not in said
    assert str(tmp_path) not in said
    assert said == "OSError: No space left on device"
    long = hp._detail(ValueError("x" * 500))
    assert len(long) <= hp._TEXT_LIMIT
    assert long.startswith("ValueError: ")
    assert hp._detail(RuntimeError("")) == "RuntimeError"


def test_the_design_chain_is_found_by_sequence_and_not_by_label():
    """A reader's model names its chains however its own tool did."""
    assert hp.design_chain_in(_PAIR, _PAIR, "A") == "A"
    assert hp.design_chain_in("", _PAIR, "A") == ""
    assert hp.design_chain_in(_PAIR, "", "A") == ""
    assert hp.design_chain_in(_PAIR, _PAIR, "") == ""


def test_a_margin_is_only_taken_where_both_sides_were_measured(tmp_path: Path):
    on_target = hp.measure_one(_PAIR, "A", tmp_path, where="on_target")
    blocked = hp.InterfaceReading("P02768", blocker="complex_not_supplied")
    report = hp.interface_report(on_target, [blocked])
    row = report["margins"][0]
    assert row["state"] == UNTESTED
    assert row["blocker"] == "complex_not_supplied"
    assert row["dsasa_total_a2"] is None


def test_a_margin_is_reported_and_not_judged_without_a_cut(tmp_path: Path):
    on_target = hp.measure_one(_PAIR, "A", tmp_path, where="on_target")
    off_target = hp.measure_one(_pdb({"A": 10, "B": 10}, gap=9.0), "A", tmp_path, where="P02768")
    report = hp.interface_report(on_target, [off_target])
    row = report["margins"][0]
    assert row["state"] == UNTESTED
    assert row["blocker"] == "margin_threshold_not_set"
    assert row["dsasa_total_a2"] is not None
    assert report["thresholds"]["margin_dsasa_min"] is None


def test_a_zero_lead_is_a_hit_against_any_cut(tmp_path: Path):
    """The same interface on both sides is no selectivity at all."""
    on_target = hp.measure_one(_PAIR, "A", tmp_path, where="on_target")
    off_target = hp.measure_one(_PAIR, "A", tmp_path, where="P02768")
    limits = hp.ProtocolLimits(margin_dsasa_min=50.0)
    row = hp.interface_report(on_target, [off_target], limits=limits)["margins"][0]
    assert row["dsasa_total_a2"] == 0.0
    assert row["state"] == HIT


def test_a_lead_over_the_cut_clears(tmp_path: Path):
    on_target = hp.measure_one(_PAIR, "A", tmp_path, where="on_target")
    off_target = hp.measure_one(_pdb({"A": 10, "B": 10}, gap=12.0), "A", tmp_path, where="P02768")
    limits = hp.ProtocolLimits(margin_dsasa_min=1.0)
    row = hp.interface_report(on_target, [off_target], limits=limits)["margins"][0]
    assert row["state"] == CLEAR


def test_a_contact_lead_stays_a_whole_number(tmp_path: Path):
    """A count measured to one decimal place reads as a number somebody computed."""
    on_target = hp.measure_one(_PAIR, "A", tmp_path, where="on_target")
    off_target = hp.measure_one(_PAIR, "A", tmp_path, where="P02768")
    row = hp.interface_report(on_target, [off_target])["margins"][0]
    assert isinstance(row["interface_atom_contacts"], int)


def test_the_interface_route_declines_when_no_model_was_supplied(tmp_path: Path):
    on_target = hp.measure_one(_PAIR, "A", tmp_path, where="on_target")
    report = hp.interface_report(on_target, [])
    assert report["blocker"] == "complex_not_supplied"
    assert report["margins"] == []


def test_the_on_target_side_being_absent_blocks_the_route(tmp_path: Path):
    """Half a comparison is not a comparison, so no margin goes out at all."""
    absent = hp.InterfaceReading("on_target", blocker="on_target_complex_absent")
    off_target = hp.measure_one(_PAIR, "A", tmp_path, where="P02768")
    report = hp.interface_report(absent, [off_target])
    assert report["blocker"] == "on_target_complex_absent"
    assert report["margins"] == []


def test_the_failing_side_is_the_one_the_reason_names(tmp_path: Path):
    """Checked in the order they fail.

    Reading the on-target side first reported ``on_target_complex_absent``
    whenever the reader's own model was what could not be measured -- a true
    sentence about the wrong half, which sends somebody to look at a file that
    is fine.
    """
    on_target = hp.measure_one(_PAIR, "A", tmp_path, where="on_target")
    theirs = hp.InterfaceReading("P02768", blocker="model_too_large")
    report = hp.interface_report(on_target, [theirs])
    assert report["blocker"] == "model_too_large"
    assert [row["blocker"] for row in report["margins"]] == ["model_too_large"]


def test_every_interface_term_is_measured_on_both_sides(tmp_path: Path):
    """A margin is like for like, so each term has to exist on the reading."""
    reading = hp.measure_one(_PAIR, "A", tmp_path, where="on_target")
    for term in hp.INTERFACE_TERMS:
        assert hasattr(reading, term), term
        assert term in reading.as_record()


# --- the whole record --------------------------------------------------------


def test_nothing_chosen_costs_no_alignment(tmp_path: Path, proteome: Path):
    """A whole-proteome pass behind a button, for a route about to decline.

    The resemblance route refuses an empty selection on its own, so the
    alignment is not run first. Measured by counting the calls: the injected
    pass stands in for MMseqs2 and is never asked for.
    """
    calls: list[object] = []

    def pass_fn(designs, where):
        calls.append(designs)
        return _scan()

    record = hp.protocol_record(_DESIGNS, names=(), proteome=proteome, work_dir=tmp_path, search=pass_fn)
    assert record["routes"][0]["blocker"] == "anti_targets_not_named"
    assert calls == []


def test_no_model_costs_no_on_target_measurement(tmp_path: Path, proteome: Path):
    """The other half of the same point.

    The on-target interface is a SASA and a dot surface over a whole complex.
    Measured on every press of a button that cannot answer, it is seconds spent
    on a number the route then declines to use.
    """
    record = hp.protocol_record(
        _DESIGNS,
        names=("ALB",),
        proteome=proteome,
        work_dir=tmp_path,
        complex_text=_PAIR,
        binder_chain="A",
        search=lambda designs, where: _scan(),
    )
    interface = record["routes"][1]
    assert interface["blocker"] == "complex_not_supplied"
    assert interface["on_target"]["dsasa_total_a2"] is None
    # Nothing was written, which is what says nothing was measured: the
    # measurement writes the structure into the work directory first.
    assert list(tmp_path.glob("protocol_on_target.*")) == []


def test_the_record_carries_both_routes_and_the_code_that_produced_it(tmp_path: Path, proteome: Path):
    record = hp.protocol_record(
        _DESIGNS,
        names=("HBA1",),
        target="P05113",
        proteome=proteome,
        work_dir=tmp_path,
        search=lambda designs, where: _scan(_hit("P69905", identity=0.62)),
    )
    assert record["protocol"] == hp.PROTOCOL_LABEL
    assert record["order"] == "anti_targets_before_target"
    assert [route["route"] for route in record["routes"]] == list(hp.ROUTES)
    assert set(record["code"]) == {"commit", "tree_state"}
    assert record["target"] == {"named": True, "accession": "P05113"}


def test_every_record_survives_a_json_round_trip(tmp_path: Path, proteome: Path):
    """A frozen dataclass leaking into a record is the likeliest first bug here."""
    record = hp.protocol_record(
        _DESIGNS,
        names=("HBA1", "ALB", "NOTAPROTEIN"),
        proteome=proteome,
        work_dir=tmp_path,
        model_text=_PAIR,
        model_of="ALB",
        complex_text=_PAIR,
        binder_chain="A",
        search=lambda designs, where: _scan(_hit("P69905", identity=0.62)),
    )
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_the_protocol_cannot_be_cleared_from_resemblance_alone(tmp_path: Path, proteome: Path):
    """The central claim, and the reason this screen exists.

    Every resemblance reading clears and the call is still undecided, because
    the route that answers the binding question was not given a model. A clear
    here would be the screen saying "selective" on the strength of an
    alignment, which is not what an alignment says.
    """
    record = hp.protocol_record(
        _DESIGNS,
        names=("ALB",),
        proteome=proteome,
        work_dir=tmp_path,
        limits=hp.ProtocolLimits(resemblance_identity_max=0.5),
        search=lambda designs, where: _scan(),
    )
    assert [row["state"] for row in record["routes"][0]["readings"]] == [CLEAR]
    assert record["call"] == UNTESTED
    assert record["blocker"] == "complex_not_supplied"


def test_a_hit_on_either_route_is_the_protocol_s_call(tmp_path: Path, proteome: Path):
    """One protein it touches answers the question the protocol asks."""
    record = hp.protocol_record(
        _DESIGNS,
        names=("HBA1",),
        proteome=proteome,
        work_dir=tmp_path,
        limits=hp.ProtocolLimits(resemblance_identity_max=0.5),
        search=lambda designs, where: _scan(_hit("P69905", identity=0.9)),
    )
    assert record["call"] == HIT


def test_a_clear_needs_every_reading_and_every_route(tmp_path: Path, proteome: Path):
    record = hp.protocol_record(
        _DESIGNS,
        names=("ALB",),
        proteome=proteome,
        work_dir=tmp_path,
        model_text=_pdb({"A": 10, "B": 10}, gap=12.0),
        model_of="ALB",
        complex_text=_PAIR,
        binder_chain="A",
        limits=hp.ProtocolLimits(resemblance_identity_max=0.5, margin_dsasa_min=1.0),
        search=lambda designs, where: _scan(),
    )
    assert record["call"] == CLEAR
    assert record["blocker"] is None


def test_an_injected_pass_does_not_override_a_missing_proteome(tmp_path: Path):
    """Without the FASTA the membership check cannot run, so no reading is founded."""
    record = hp.protocol_record(
        _DESIGNS,
        names=("HBA1",),
        proteome=None,
        work_dir=tmp_path,
        search=lambda designs, where: _scan(_hit("P69905", identity=0.9)),
    )
    assert record["routes"][0]["blocker"] == "proteome_not_configured"
    assert record["routes"][0]["readings"] == []
    assert record["call"] == UNTESTED


def test_a_model_with_no_pairing_named_is_not_labelled_from_the_selection(tmp_path: Path, proteome: Path):
    """Taking the first ticked protein would put a name on a file it is not in."""
    record = hp.protocol_record(
        _DESIGNS,
        names=("HBA1", "ALB"),
        proteome=proteome,
        work_dir=tmp_path,
        model_text=_PAIR,
        complex_text=_PAIR,
        binder_chain="A",
        search=lambda designs, where: _scan(),
    )
    row = record["routes"][1]["margins"][0]
    assert row["blocker"] == "model_pairing_not_named"
    assert row["dsasa_total_a2"] is None


def test_a_model_of_a_protein_nobody_chose_is_not_measured(tmp_path: Path, proteome: Path):
    """Otherwise the interface is measured against something the comparison is not about."""
    record = hp.protocol_record(
        _DESIGNS,
        names=("HBA1",),
        proteome=proteome,
        work_dir=tmp_path,
        model_text=_PAIR,
        model_of="ALB",
        complex_text=_PAIR,
        binder_chain="A",
        search=lambda designs, where: _scan(),
    )
    row = record["routes"][1]["margins"][0]
    assert row["where"] == "ALB"
    assert row["blocker"] == "model_protein_not_chosen"


def test_a_pairing_may_be_named_by_symbol_accession_or_entry_name(tmp_path: Path, proteome: Path):
    """The reader chose ``ALB``; the record calls it P02768. Either name is theirs."""
    for spelling in ("ALB", "P02768", "ALBU_HUMAN"):
        record = hp.protocol_record(
            _DESIGNS,
            names=("ALB",),
            proteome=proteome,
            work_dir=tmp_path,
            model_text=_PAIR,
            model_of=spelling,
            complex_text=_PAIR,
            binder_chain="A",
            search=lambda designs, where: _scan(),
        )
        row = record["routes"][1]["margins"][0]
        assert row["blocker"] != "model_protein_not_chosen", spelling


def test_a_name_the_proteome_does_not_hold_is_reported_on_the_record(tmp_path: Path, proteome: Path):
    record = hp.protocol_record(
        _DESIGNS,
        names=("HBA1", "NOTAPROTEIN"),
        proteome=proteome,
        work_dir=tmp_path,
        search=lambda designs, where: _scan(),
    )
    assert record["not_in_proteome"] == ["NOTAPROTEIN"]
    assert [item["accession"] for item in record["anti_targets"]] == ["P69905", "NOTAPROTEIN"]


def test_the_summary_counts_what_was_chosen(tmp_path: Path, proteome: Path):
    record = hp.protocol_record(
        _DESIGNS,
        names=("HBA1", "ALB"),
        proteome=proteome,
        work_dir=tmp_path,
        search=lambda designs, where: _scan(),
    )
    said = hp.summary_line(record)
    assert said.startswith(hp.PROTOCOL_LABEL)
    assert "2 proteins" in said
    one = hp.protocol_record(
        _DESIGNS, names=("ALB",), proteome=proteome, work_dir=tmp_path, search=lambda d, w: _scan()
    )
    assert "1 protein " in hp.summary_line(one)


def test_the_readout_gives_both_absolutes_and_the_lead(tmp_path: Path, proteome: Path):
    """A lead on its own is not readable, and the two routes' rows differ.

    Both routes carry a key called ``readings`` holding different shapes. Run
    through the resemblance formatter, an interface reading printed
    "None (complex_not_supplied)" -- a line assembled out of keys it does not
    have -- and nothing caught it until a model was actually supplied.
    """
    record = hp.protocol_record(
        _DESIGNS,
        names=("ALB",),
        proteome=proteome,
        work_dir=tmp_path,
        model_text=_pdb({"A": 10, "B": 10}, gap=9.0),
        model_of="ALB",
        complex_text=_PAIR,
        binder_chain="A",
        search=lambda designs, where: _scan(),
    )
    text = "\n".join(hp.readout_lines(record))
    assert "None (" not in text
    assert "on-target" in text and "off-target" in text and "lead" in text
    for term in hp.INTERFACE_TERMS:
        assert term in text, term


def test_a_withheld_verdict_still_reports_what_was_measured(tmp_path: Path, proteome: Path):
    """The distinction both renderings got wrong: a reason is not a missing number.

    ``margin_threshold_not_set`` measured both sides perfectly well and is
    withholding only the verdict. Skipping its numbers because the row carries
    a sentence hid the measurement behind "no cut was named" -- which, on a
    screen whose whole argument is that it reports what it measured and
    declines to judge it, is the one thing it must not do.
    """
    record = hp.protocol_record(
        _DESIGNS,
        names=("ALB",),
        proteome=proteome,
        work_dir=tmp_path,
        model_text=_pdb({"A": 10, "B": 10}, gap=9.0),
        model_of="ALB",
        complex_text=_PAIR,
        binder_chain="A",
        search=lambda designs, where: _scan(),
    )
    margin = record["routes"][1]["margins"][0]
    assert margin["blocker"] == "margin_threshold_not_set"
    assert margin["dsasa_total_a2"] is not None

    text = "\n".join(hp.readout_lines(record))
    assert str(margin["dsasa_total_a2"]) in text
    assert hp.sentence("margin_threshold_not_set") in text

    page = ui._results_tools("", record=record)
    assert "<th>On-target</th><th>Off-target</th><th>Lead</th>" in page
    assert hp.sentence("margin_threshold_not_set") in page


def test_the_readout_states_both_halves_of_every_route(tmp_path: Path, proteome: Path):
    record = hp.protocol_record(
        _DESIGNS,
        names=("HBA1",),
        proteome=proteome,
        work_dir=tmp_path,
        search=lambda designs, where: _scan(_hit("P69905", identity=0.62)),
    )
    lines = hp.readout_lines(record)
    text = "\n".join(lines)
    for route in hp.ROUTES:
        assert route in text
        assert hp.ROUTE_DOES_NOT_ANSWER[route] in text
    assert "does not answer" in text


def test_the_default_proteome_prefers_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The results page has no --proteome flag to read, so this is how it finds one."""
    named = tmp_path / "named.fasta"
    named.write_text(_FASTA, encoding="utf-8")
    monkeypatch.setenv("HUMAN_PROTEOME", str(named))
    assert hp.default_proteome() == named
    monkeypatch.delenv("HUMAN_PROTEOME")
    monkeypatch.setattr("rfproteina.disease.tables.downloaded_proteome", lambda *args: None)
    assert hp.default_proteome() is None


# --- the tab on the results page ---------------------------------------------


def test_the_tab_sits_beside_the_other_two():
    page = ui._results_tools("")
    assert 'id="results-tab-human" value="human"' in page
    assert hp.PROTOCOL_LABEL in page
    assert 'class="results-panel results-panel-human"' in page


def test_the_tab_the_reader_submitted_is_the_one_that_opens():
    """Three tabs, compared by name.

    While there were two this was "is it the combined one", so the protocol's
    tab posted its own name back and the page returned with Binder showing and
    the result on a tab nobody was looking at.
    """
    page = ui._results_tools("", active=ui.HUMAN_TAB)
    assert 'id="results-tab-human" value="human" checked' in page
    assert 'id="results-tab-binder" value="binder">' in page
    assert 'id="results-tab-combined" value="combined">' in page


def test_an_unknown_tab_value_opens_the_first_rather_than_none():
    page = ui._results_tools("", active="from-another-build")
    assert 'id="results-tab-binder" value="binder" checked' in page


def test_both_buttons_post_an_action_the_module_knows():
    page = ui._results_tools("")
    for action in (results_actions.HUMAN_PROPOSE, results_actions.HUMAN_COMPARE):
        assert f'value="{action}"' in page
        assert action in results_actions.ACTIONS


def test_the_limit_sentence_is_on_the_tab_before_any_result():
    """Above the buttons. A reader holding a number is past reading what it is not."""
    page = ui._results_tools("")
    assert hp.PROTOCOL_LIMIT in page


def test_the_reader_s_own_words_are_the_tab_s_steps():
    page = ui._results_tools("")
    assert "What not to bind to" in page
    assert "The target to bind to" in page
    assert "Compare the output against what not to bind to" in page


def test_before_a_proposal_the_list_says_so_rather_than_rendering_empty():
    """An empty list under "what not to bind to" reads as "nothing to avoid"."""
    page = ui._results_tools("")
    assert "No proposal yet" in page
    assert 'name="human_pick"' not in page


def test_the_tick_boxes_are_drawn_from_a_proposal():
    record = {
        "candidates": [
            {"accession": "P69905", "gene": "HBA1", "description": "Hemoglobin subunit alpha", "identity": 0.62},
            {"accession": "P02768", "gene": "ALB", "description": "Albumin", "identity": None},
        ]
    }
    page = ui._results_tools("", record=record)
    assert page.count('name="human_pick"') == 2
    assert 'value="P69905"' in page
    assert "HBA1 (P69905)" in page
    assert "identity 0.62" in page


def test_the_interface_table_draws_both_absolutes_beside_the_lead(tmp_path: Path, proteome: Path):
    """The same three numbers the text readout prints, from the same record."""
    record = hp.protocol_record(
        _DESIGNS,
        names=("ALB",),
        proteome=proteome,
        work_dir=tmp_path,
        model_text=_pdb({"A": 10, "B": 10}, gap=9.0),
        model_of="ALB",
        complex_text=_PAIR,
        binder_chain="A",
        limits=hp.ProtocolLimits(margin_dsasa_min=1.0),
        search=lambda designs, where: _scan(),
    )
    page = ui._results_tools("", record=record)
    assert "<th>On-target</th><th>Off-target</th><th>Lead</th>" in page
    assert "Buried area" in page
    assert "Shape complementarity" in page
    assert "Against ALB" in page
    # The on-target figure itself, so the table is drawn from the record rather
    # than from the margin alone.
    area = record["routes"][1]["on_target"]["dsasa_total_a2"]
    assert str(area) in page


def test_no_table_is_drawn_where_no_margin_was_taken(tmp_path: Path, proteome: Path):
    """Four rows of em-dashes under a heading is noise around the one sentence.

    The text readout skips the terms for a blocked row, and the two renderings
    have to agree about that.
    """
    record = hp.protocol_record(
        _DESIGNS,
        names=("ALB",),
        proteome=proteome,
        work_dir=tmp_path,
        model_text=_PAIR,
        model_of="HBA1",
        complex_text=_PAIR,
        binder_chain="A",
        search=lambda designs, where: _scan(),
    )
    page = ui._results_tools("", record=record)
    assert "Against HBA1" in page
    assert "<th>On-target</th>" not in page
    assert hp.sentence("model_protein_not_chosen") in page
    text = "\n".join(hp.readout_lines(record))
    assert hp.sentence("model_protein_not_chosen") in text
    for term in hp.INTERFACE_TERMS:
        assert term not in text, term


def test_the_readout_draws_a_row_per_protein_and_keeps_the_caveat(tmp_path: Path, proteome: Path):
    record = hp.protocol_record(
        _DESIGNS,
        names=("HBA1", "ALB"),
        proteome=proteome,
        work_dir=tmp_path,
        limits=hp.ProtocolLimits(resemblance_identity_max=0.5),
        search=lambda designs, where: _scan(_hit("P69905", identity=0.62)),
    )
    page = ui._results_tools("", record=record)
    assert page.count('class="protocol-readout"') == 1
    assert "HBA1" in page and "ALB" in page
    assert 'class="state-hit"' in page
    assert hp.ROUTE_DOES_NOT_ANSWER[hp.SEQUENCE_RESEMBLANCE] in page


def test_a_submission_the_panel_will_not_read_is_bounded_not_allocated():
    """Pre-existing, on the form this tab posts to.

    ``do_POST`` read whatever ``Content-Length`` claimed, so a client claiming
    four gigabytes had the panel try to allocate four gigabytes, and a header
    that was not a number raised out of the handler instead of answering.
    """
    assert ui._body_length("0") == 0
    assert ui._body_length(" 1024 ") == 1024
    assert ui._body_length(str(ui.POST_BYTE_LIMIT)) == ui.POST_BYTE_LIMIT
    assert ui._body_length(str(ui.POST_BYTE_LIMIT + 1)) is None
    assert ui._body_length("4000000000") is None
    for bad in ("", "   ", "-1", "abc", "12abc", "0x10", "1e9", "+5", "1.5"):
        assert ui._body_length(bad) is None, bad


def test_the_bound_still_admits_the_largest_legitimate_submission():
    """A structure upload is the big one, and the retriever accepts 64 MiB of model."""
    from rfproteina.disease.results_retrieve import TEXT_LIMIT

    assert ui.POST_BYTE_LIMIT > TEXT_LIMIT


def test_a_hostile_content_length_is_refused_over_the_wire():
    """The handler, over a real socket, rather than the helper on its own.

    This is the half that can regress silently: ``_body_length`` can be
    perfect while nothing calls it. If the bound were gone the server would try
    to read four gigabytes and this test would time out rather than fail, which
    is why the timeout is short and why only three bytes are actually sent.

    The session is checked afterwards as well. A body that was refused unread
    was never applied, so the panel has to be on the page it was on.
    """
    import http.client
    import threading
    import time

    session = ui.BrowserSession(opener=lambda _url: None)

    def serve() -> None:
        session.exchange("<p>one</p>", step=1)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        for _ in range(250):
            if session._server is not None and session.step() == 1:
                break
            time.sleep(0.02)
        assert session._server is not None, "the panel server never came up"
        port = session._server.server_address[1]

        for header, expected in (("4000000000", 413), ("not-a-number", 400), ("-1", 400)):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.putrequest("POST", "/", skip_accept_encoding=True)
                connection.putheader("Content-Type", "application/x-www-form-urlencoded")
                connection.putheader("Content-Length", header)
                connection.endheaders()
                connection.send(b"a=1")
                assert connection.getresponse().status == expected, header
            finally:
                connection.close()

        assert session.step() == 1
        assert session.page() == "<p>one</p>"
    finally:
        session._posted.set()
        session._next_ready.set()
        session._result.set()
        worker.join(timeout=2)


def test_a_missing_number_is_drawn_as_a_dash_not_a_blank_cell():
    """A blank cell and a measured zero look the same, and here they are not."""
    assert ui._human_cell(None) == "&mdash;"
    assert ui._human_cell(0) == "0"


def test_the_protocol_declines_when_no_design_sequence_could_be_read():
    """The results page's own entry point, with a catalog holding nothing usable."""
    done = results_actions.perform(results_actions.HUMAN_COMPARE, [{"sample": "d1"}], lambda token: None)
    assert done.record is not None
    assert done.record["blocker"] == "design_sequence_absent"
    assert done.sentence == hp.sentence("design_sequence_absent")
    assert done.payload == ""


def test_the_page_names_a_design_only_for_a_measurement_it_took(tmp_path: Path, proteome: Path, monkeypatch):
    """A sentence on a safety page that describes work that did not happen.

    The on-target interface is measured only where there is an off-target one
    to compare it with. This line was set on every comparison, so the page read
    "the on-target interface was measured on design_1" under a route that had
    just declined to measure anything.
    """
    monkeypatch.setattr(hp, "default_proteome", lambda: proteome)
    monkeypatch.setattr(hp, "scan_designs", lambda designs, where, work, **kwargs: _scan())
    monkeypatch.setattr(
        results_actions,
        "binder_sequences",
        lambda *args: [results_actions.BinderSequence("design_1", strand="A", sequence=_DESIGNS[0][1], fasta=">x\n")],
    )
    catalog = [{"sample": "design_1", "complex_token": "tok", "binder_chains": ["A"]}]
    read = lambda token: _PAIR.encode("utf-8")  # noqa: E731 - one line, used twice below

    without_a_model = results_actions.perform(
        results_actions.HUMAN_COMPARE, catalog, read, [f"{results_actions.HUMAN_PICK}=P02768"]
    )
    assert without_a_model.record is not None
    assert without_a_model.record["interface_design"] == ""
    assert "was measured on" not in ui._results_tools("", record=without_a_model.record)

    with_a_model = results_actions.perform(
        results_actions.HUMAN_COMPARE,
        catalog,
        read,
        [
            f"{results_actions.HUMAN_PICK}=P02768",
            f"{results_actions.HUMAN_MODEL_OF}=P02768",
            f"{results_actions.HUMAN_MODEL}={_pdb({'A': 10, 'B': 10}, gap=9.0)}",
        ],
    )
    assert with_a_model.record is not None
    assert with_a_model.record["interface_design"] == "design_1"
    assert hp.on_target_measured(with_a_model.record)
    assert "was measured on design_1" in ui._results_tools("", record=with_a_model.record)


def test_a_route_is_found_by_name_and_not_by_position(tmp_path: Path, proteome: Path):
    """The two routes are not interchangeable, so neither is their order."""
    record = hp.protocol_record(
        _DESIGNS, names=("ALB",), proteome=proteome, work_dir=tmp_path, search=lambda d, w: _scan()
    )
    assert hp.route_of(record, hp.SEQUENCE_RESEMBLANCE)["route"] == hp.SEQUENCE_RESEMBLANCE
    assert hp.route_of(record, hp.INTERFACE_CONTACTS)["route"] == hp.INTERFACE_CONTACTS
    assert hp.route_of(record, "no_such_route") == {}
    assert hp.route_of({}, hp.INTERFACE_CONTACTS) == {}
    assert not hp.on_target_measured(record)
    assert not hp.on_target_measured({})


def test_every_field_the_buttons_read_is_one_the_tab_posts():
    """One spelling each, shared, rather than two lists held against each other.

    A field the page renames and the reader does not is silent -- no error, no
    empty value, just an input that stops arriving. ``flat_fields`` carries the
    docstring of the last time that happened here, when every typed path,
    pasted model and accession went on the floor while the page reported that
    no download could be found.
    """
    record = {"candidates": [{"accession": "P02768", "gene": "ALB", "description": "Albumin"}]}
    page = ui._results_tools("", record=record)
    for name in results_actions.HUMAN_FIELDS:
        assert f'name="{name}"' in page, name


def test_the_tab_s_inputs_reach_the_comparison(tmp_path: Path, proteome: Path, monkeypatch):
    """The glue, end to end: a submission in the shape the browser posts.

    Everything between the form and the record is exercised here and nowhere
    else -- the catalog walk that finds the run's complex, the tick boxes read
    as several values under one name, the free-text box beside them, and the
    pairing. The alignment is the only thing stubbed, because MMseqs2 against
    the real proteome is not something a test should run.
    """
    monkeypatch.setattr(hp, "default_proteome", lambda: proteome)
    monkeypatch.setattr(hp, "scan_designs", lambda designs, where, work, **kwargs: _scan())

    catalog = [{"sample": "design_1", "complex_token": "tok", "binder_chains": ["A"]}]
    monkeypatch.setattr(
        results_actions,
        "binder_sequences",
        lambda *args: [results_actions.BinderSequence("design_1", strand="A", sequence=_DESIGNS[0][1], fasta=">x\n")],
    )
    fields = [
        f"{results_actions.HUMAN_PICK}=P02768",
        f"{results_actions.HUMAN_PICK}=HBA1",
        f"{results_actions.HUMAN_NAMES}=NOTAPROTEIN",
        f"{results_actions.HUMAN_TARGET}=P05113",
        f"{results_actions.HUMAN_MODEL_OF}=P02768",
        f"{results_actions.HUMAN_MODEL}={_PAIR}",
    ]
    done = results_actions.perform(results_actions.HUMAN_COMPARE, catalog, lambda token: _PAIR.encode("utf-8"), fields)
    assert done.record is not None
    record = done.record
    # Both tick boxes and the typed name, in the order they were posted.
    assert [item["accession"] for item in record["anti_targets"]] == ["P02768", "P69905", "NOTAPROTEIN"]
    assert record["target"] == {"named": True, "accession": "P05113"}
    assert record["not_in_proteome"] == ["NOTAPROTEIN"]
    # The run's own complex was found through the catalog, and the pairing was
    # honoured rather than guessed from the first ticked protein.
    assert record["interface_design"] == "design_1"
    assert [row["where"] for row in record["routes"][1]["margins"]] == ["P02768"]
    assert record["routes"][1]["margins"][0]["blocker"] == "margin_threshold_not_set"
    assert done.payload_name == "human_protocol.txt"
    assert hp.PROTOCOL_LABEL in done.sentence


def test_the_proposal_button_reaches_the_same_tab(tmp_path: Path, proteome: Path, monkeypatch):
    """The other button, through the same glue."""
    monkeypatch.setattr(hp, "default_proteome", lambda: proteome)
    monkeypatch.setattr(hp, "scan_designs", lambda designs, where, work, **kwargs: _scan(_hit("P69905", identity=0.9)))
    monkeypatch.setattr(
        results_actions,
        "binder_sequences",
        lambda *args: [results_actions.BinderSequence("design_1", strand="A", sequence=_DESIGNS[0][1], fasta=">x\n")],
    )
    done = results_actions.perform(
        results_actions.HUMAN_PROPOSE,
        [{"sample": "design_1", "complex_token": "tok", "binder_chains": ["A"]}],
        lambda token: _PAIR.encode("utf-8"),
        [f"{results_actions.HUMAN_TARGET}=P05113"],
    )
    assert done.record is not None
    assert [item["accession"] for item in done.record["candidates"]] == ["P69905"]
    assert done.record["candidates"][0]["gene"] == "HBA1"
    # And the page draws a tick box for it.
    assert f'name="{results_actions.HUMAN_PICK}" value="P69905"' in ui._results_tools("", record=done.record)


# --- the headless route ------------------------------------------------------
#
# These stop before the alignment: the mode is reachable, it reads designs out
# of the three shapes a reader has, and it refuses rather than picking a chain.
# The scan itself is covered above with an injected pass, and running MMseqs2
# against the real proteome is not something a test should do.


def _cif(chains: dict[str, str]) -> str:
    rows = "\n".join(f"{index + 1} {chain} {sequence}" for index, (chain, sequence) in enumerate(chains.items()))
    return (
        "data_design\n#\nloop_\n_entity_poly.entity_id\n_entity_poly.pdbx_strand_id\n"
        f"_entity_poly.pdbx_seq_one_letter_code_can\n{rows}\n#\n"
    )


def test_designs_are_read_from_a_fasta(tmp_path: Path):
    from rfproteina.disease.cli import _designs_for_protocol

    path = tmp_path / "designs.fasta"
    path.write_text(">design_1\nMVLSPADK\n>design_2\nMKWVTFIS\n", encoding="utf-8")
    designs, refusal = _designs_for_protocol(path, None)
    assert refusal == ""
    assert designs == [("design_1", "MVLSPADK"), ("design_2", "MKWVTFIS")]


def test_designs_are_read_from_a_one_chain_structure(tmp_path: Path):
    from rfproteina.disease.cli import _designs_for_protocol

    path = tmp_path / "design.cif"
    path.write_text(_cif({"A": "MVLSPADK"}), encoding="utf-8")
    designs, refusal = _designs_for_protocol(path, None)
    assert refusal == ""
    assert designs == [("design_A", "MVLSPADK")]


def test_a_structure_with_two_chains_is_refused_with_its_chains_named(tmp_path: Path):
    """Picking wrongly would compare the *target* against the proteins to avoid."""
    from rfproteina.disease.cli import _designs_for_protocol

    path = tmp_path / "complex.cif"
    path.write_text(_cif({"A": "MVLSPADK", "B": "MKWVTFIS"}), encoding="utf-8")
    designs, refusal = _designs_for_protocol(path, None)
    assert designs == []
    assert "A, B" in refusal
    assert "--human-chain" in refusal
    named, refusal = _designs_for_protocol(path, "B")
    assert refusal == ""
    assert named == [("complex_B", "MKWVTFIS")]


def test_a_chain_the_structure_does_not_hold_is_refused(tmp_path: Path):
    from rfproteina.disease.cli import _designs_for_protocol

    path = tmp_path / "complex.cif"
    path.write_text(_cif({"A": "MVLSPADK", "B": "MKWVTFIS"}), encoding="utf-8")
    designs, refusal = _designs_for_protocol(path, "Z")
    assert designs == []
    assert "no chain Z" in refusal


def test_a_file_too_large_to_be_a_run_s_designs_is_not_read(tmp_path: Path):
    """Stat before read.

    The design bounds inside the protocol are counted after a parse, so a
    reader who points this at a reference proteome would have it read into
    memory first.
    """
    from rfproteina.disease.cli import _designs_for_protocol

    path = tmp_path / "huge.fasta"
    path.write_text(">d\n" + "A" * hp.MODEL_BYTE_LIMIT, encoding="utf-8")
    designs, refusal = _designs_for_protocol(path, None)
    assert designs == []
    assert "larger than" in refusal


def test_a_file_that_is_neither_is_refused(tmp_path: Path):
    from rfproteina.disease.cli import _designs_for_protocol

    path = tmp_path / "notes.txt"
    path.write_text("nothing here\n", encoding="utf-8")
    designs, refusal = _designs_for_protocol(path, None)
    assert designs == []
    assert "neither FASTA records nor a polymer chain" in refusal


def test_the_mode_refuses_a_path_that_is_not_a_file(tmp_path: Path):
    from rfproteina.disease.cli import main

    assert main(["--human-protocol", str(tmp_path / "absent.fasta")]) == 2


def test_the_mode_cannot_be_asked_for_beside_a_screen(tmp_path: Path):
    """Both read a finished run and neither is a step of the other."""
    from rfproteina.disease.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--human-protocol", "x.fasta", "--screen", "."])
