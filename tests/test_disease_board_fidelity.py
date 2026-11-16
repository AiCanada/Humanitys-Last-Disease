"""What the board says, against what the record it was drawn from holds.

Every test here builds an axis record in the shape a screening module really
writes, renders the closed page, and reads the page back. The page is the only
thing most readers ever see, so a verdict rendered wrongly is a false verdict
however right the record is.
"""

import re

import pytest

from rfproteina.disease.combinations import PANEL_NM, PANEL_NM_SOURCE
from rfproteina.disease.oligo import innate_immune_record
from rfproteina.disease.screen_board import render_closing


def _page(screen: dict[str, object]) -> str:
    return render_closing([], [{"entry_id": "1ABC", "chain": "A", "screen": screen}])


def _panel(page: str, heading: str) -> str:
    for block in re.findall(r'<article class="axis-panel">.*?</article>', page, re.S):
        if f"<h3>{heading}</h3>" in block:
            return block
    raise AssertionError(f"no panel headed {heading}")


def _text(markup: str) -> str:
    """The panel as a reader sees it, with the tags taken out."""
    from html import unescape

    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", markup)))


def test_a_declared_clear_is_not_shown_as_a_measured_one() -> None:
    """``screened_clear`` over a person's call is not "every check is clear".

    ``screen._stamp_source`` lets ``axis_evidence.json`` override a measurement
    and keeps what the measurement said under ``measured_state``. The call, the
    status chip and the panel heading all read exactly as they do for a measured
    clear, so a bar cleared by a human's declaration over a measured hit read as
    a screen that had measured every axis clear. ``mcp_api._curated_axes``
    already names the declared axes beside the call for the MCP reader; this is
    the same disclosure for the person reading the page.
    """
    screen = {
        "call": "screened_clear",
        "states": {"ada": "clear", "paralog": "clear"},
        "sequence_liability": {
            "ada": {
                "state": "clear",
                "measurement": 5.0,
                "blocker": None,
                "measured_state": "hit",
                "measured_blocker": None,
                "source": "axis_evidence",
                "curated": {"reason": "tolerated in the clinic", "source": "Smith 2020"},
            }
        },
        "target_liability": {
            "paralog": {"state": "clear", "measurement": 0.0, "blocker": None, "source": "measured"},
        },
    }
    page = _page(screen)
    assert "Every check in this bar is clear." in page
    # The headline, before any tab is opened.
    assert "Declared by a person, not measured: ada, which measured hit." in page
    # The status grid, which is the other thing a reader sees unopened.
    assert "<strong>Clear (declared)</strong>" in page
    assert page.count("<strong>Clear (declared)</strong>") == 1
    # The panel's own heading.
    assert "Clear on this check. Declared by a person, not measured." in _text(_panel(page, "ADA"))
    # A measured clear beside it keeps its own words.
    paralog = _text(_panel(page, "Paralog"))
    assert "Clear on this check." in paralog
    assert "declared" not in paralog.casefold()


def test_a_measured_clear_says_nothing_about_a_declaration() -> None:
    """The guard above must not fire on a screen nobody declared anything on."""
    screen = {
        "call": "screened_clear",
        "states": {"paralog": "clear"},
        "target_liability": {
            "paralog": {"state": "clear", "measurement": 0.0, "blocker": None, "source": "measured"},
        },
    }
    page = _page(screen)
    assert "Every check in this bar is clear." in page
    assert "Declared by a person" not in page
    assert "(declared)" not in page


def test_a_partial_mhc_scan_draws_no_bar_to_compare_against() -> None:
    """A scan that covered part of the chain was compared with nothing.

    ``liability._density_record`` writes ``reference_density`` -- the bar the
    design was judged against -- only when the comparison was actually made,
    and leaves it off when the scan covered part of the chain, because "under
    the highest of three" was never established. The per-protein
    ``reference_densities`` ride along either way. The board took the highest of
    those as the cut and drew it as a dashed line under "The dashed line is the
    cut", putting the design on the clear side of a line nothing had measured it
    against.
    """
    record = {
        "state": "untested",
        "measurement": 1.32,
        "blocker": "scan_incomplete",
        "source": "untested",
        "method": "binding_peptides_per_100_windows",
        "binding_peptides": 2,
        "reference_densities": {"P01009": 1.64, "P0CG48": 0.0, "P0DOX5": 0.49},
    }
    page = _page({"states": {"mhc_i": "untested"}, "sequence_liability": {"mhc_i": record}})
    panel = _panel(page, "MHC class I")
    assert "stroke-dasharray" not in panel
    assert "The dashed line is the cut." not in panel
    assert "No cut is drawn. This design was not compared with them." in _text(panel)
    assert "Design compared with tolerated proteins" not in panel
    # The densities themselves were measured and stay on the page.
    assert "1.64" in panel
    assert "1.32" in panel

    # The same record with the bar the verdict used keeps the comparison.
    decided = dict(record, state="clear", blocker=None, source="measured", reference_density=1.64)
    compared = _panel(_page({"states": {"mhc_i": "clear"}, "sequence_liability": {"mhc_i": decided}}), "MHC class I")
    assert "stroke-dasharray" in compared
    assert "The dashed line is the cut." in compared
    assert "Design compared with tolerated proteins" in compared


def test_a_panel_axis_states_the_cut_its_own_record_was_judged_against(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guide read the live table; the verdict used the stored threshold.

    ``toxicology._record`` stores ``threshold`` and ``threshold_source`` per
    axis precisely so a stored screen keeps the cut it was judged against when
    the owner changes :data:`combinations.PANEL_NM`. ``_tox_guide`` read
    ``panel_cut`` instead, so after such a change the guide stated a bar the
    status and the chart both deny: a reading of 4000 is a hit at the stored
    10000 while the guide said anything above 3000 is clear.
    """
    monkeypatch.setitem(PANEL_NM, "herg", 3000.0)
    monkeypatch.setitem(PANEL_NM_SOURCE, "herg", "program_convention_3uM")
    record = {
        "state": "hit",
        "measurement": 4000.0,
        "blocker": None,
        "source": "measured",
        "threshold": 10000.0,
        "threshold_source": "program_convention_10uM",
    }
    panel = _text(_panel(_page({"states": {"herg": "hit"}, "toxicology": {"herg": record}}), "hERG"))
    assert "At or below 10000 nM is a hit on this panel protein." in panel
    assert "Above 10000 nM is clear on this panel protein." in panel
    assert "The cut is 10000 nM (10 uM), a program convention." in panel
    assert "3000" not in panel

    # A record from before the per-axis cut existed has no threshold of its
    # own, and the live table is then the only cut there is.
    older = {"state": "hit", "measurement": 1000.0, "blocker": None, "source": "measured"}
    fallback = _text(_panel(_page({"states": {"herg": "hit"}, "toxicology": {"herg": older}}), "hERG"))
    assert "At or below 3000 nM is a hit on this panel protein." in fallback


def test_a_panel_axis_names_a_sourced_cut_as_sourced(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "A program convention" is a claim about provenance, not a fixed phrase."""
    monkeypatch.setitem(PANEL_NM, "bsep", 25_000.0)
    monkeypatch.setitem(PANEL_NM_SOURCE, "bsep", "Morgan 2010 DILI review")
    record = {
        "state": "clear",
        "measurement": 40_000.0,
        "blocker": None,
        "source": "measured",
        "threshold": 25_000.0,
        "threshold_source": "Morgan 2010 DILI review",
    }
    panel = _text(_panel(_page({"states": {"bsep": "clear"}, "toxicology": {"bsep": record}}), "BSEP"))
    assert "The cut is 25000 nM (25 uM); its source is recorded as Morgan 2010 DILI review." in panel
    assert "a program convention" not in panel


def test_an_innate_immune_hit_names_the_rule_that_fired() -> None:
    """A GTCGTT hit is not an instance of the murine consensus.

    ``oligo.innate_immune_record`` reads two published rules -- the murine
    RRCGYY of Krieg 1995 and the human-optimal GTCGTT of Hartmann 2000 -- and
    stores the one that matched at each position for exactly this reason. The
    board described only the murine formula, so a hit driven by GTCGTT was
    presented to the reader as a purine-purine-CG-pyrimidine-pyrimidine
    hexamer, which GTCGTT is not, and nothing on the page named the real motif.
    """
    record = innate_immune_record("AAAAGTCGTTTTTT")
    assert record["state"] == "hit"
    assert [site["motif"] for site in record["motif_sites"]] == ["GTCGTT"]
    panel = _text(
        _panel(
            _page({"states": {"innate_immune": "hit"}, "nucleic_liability": {"innate_immune": record}}), "Innate immune"
        )
    )
    assert "Hit. This check found a problem." in panel
    assert "GTCGTT" in panel
    assert "Hartmann 2000" in panel
    assert "RRCGYY" in panel
    assert "Krieg 1995" in panel
    # The hexamer is findable in the chain, by the rule that read it.
    assert "base 5, GTCGTT" in panel
    assert "The design holds 1 CpG." in panel


def test_an_innate_immune_cpg_without_a_motif_is_not_told_no_cut_applies() -> None:
    """The to-do line contradicted the High line on its own panel.

    ``oligo.innate_immune_record`` reused ``no_pass_cut`` for a design that
    holds CpG with none of it in either hexamer. A cut does apply there -- one
    hexamer is a hit -- and the measured zero is refused because TLR9 is not the
    only innate sensor, which is the reason ``cpg_free_not_evidence`` gives for
    its own zero. The board told the reader no cut applied, three lines above a
    guide stating one, and offered nothing to do about it.

    Fixed at the cause rather than on the page: ``oligo`` now names this case
    ``cpg_not_consensus``, which is the fact being reported, so the sentence
    sits in ``_BLOCKERS`` with every other one and the board needs no per-axis
    override to avoid contradicting itself.
    """
    record = innate_immune_record("AAAACGAAAA")
    assert record["state"] == "untested"
    assert record["blocker"] == "cpg_not_consensus"
    assert record["measurement"] == 0
    panel = _text(
        _panel(
            _page({"states": {"innate_immune": "untested"}, "nucleic_liability": {"innate_immune": record}}),
            "Innate immune",
        )
    )
    assert "No pass cut applies to it" not in panel
    assert "none of it is one of the two consensus hexamers" in panel
    assert "Curate the call into axis_evidence.json." in panel
    assert "The design holds 1 CpG." in panel


def test_an_immunogenicity_no_pass_cut_keeps_its_own_sentence() -> None:
    """The axis override above must not reach the axis the sentence is true for."""
    record = {
        "state": "untested",
        "measurement": 0.51,
        "blocker": "no_pass_cut",
        "source": "untested",
        "positive_windows": 3,
        "presentation": "not_configured",
    }
    panel = _text(
        _panel(
            _page({"states": {"immunogenicity": "untested"}, "sequence_liability": {"immunogenicity": record}}),
            "Immunogenicity",
        )
    )
    assert "A number was measured. No pass cut applies to it, so the check stays undecided." in panel
    assert "Add mhc_i.tsv to the output folder, so the presented 9-mers are named." in panel


def test_a_seed_clear_names_the_file_it_is_a_clear_against() -> None:
    """The guide says the reference's size and digest are recorded with it.

    They were -- in ``screen.json``, where the page's reader never looks. The
    clear named no file at all, so "a clear against that file" could not be
    checked against anything.
    """
    record = {
        "state": "clear",
        "measurement": 0,
        "blocker": None,
        "source": "measured",
        "seed_width": 15,
        "seeds_searched": 24,
        "reference_bases": 4096,
        "reference_digest": "3f7a1c",
        "reference_file": "transcriptome.fa",
    }
    panel = _text(
        _panel(
            _page(
                {
                    "states": {"off_target_hybridization": "clear"},
                    "nucleic_liability": {"off_target_hybridization": record},
                }
            ),
            "Off-target hybridization",
        )
    )
    assert "Clear on this check." in panel
    assert "The reference is transcriptome.fa, 4096 bases, digest 3f7a1c." in panel
