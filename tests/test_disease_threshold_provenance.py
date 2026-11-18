"""Every cut a screen records has to say where it came from.

The point of the registry is not the strings it holds, it is that a cut cannot
be added without one. These tests fail when a new threshold reaches the stored
verdict unclassified, which is the only way the provenance stays true as the
screen grows.
"""

from __future__ import annotations

from rfproteina.disease.screen import _applied_cuts, _measured_line, _with_interface_thresholds
from rfproteina.disease.threshold_provenance import (
    ALL_PROVENANCE,
    BORROWED,
    CALLER_SUPPLIED,
    CONVENTION,
    INTERFACE_PROVENANCE,
    KINDS,
    PROVENANCE_KEY,
    PUBLISHED,
    REPORTED_ONLY,
    SCREEN_BAR_PROVENANCE,
    cut_names,
    describe_cuts,
    provenance_of,
    summary_line,
    unsourced,
)


def test_every_cut_a_screen_records_has_a_provenance() -> None:
    """The guard. A new threshold with no entry fails here, not silently in a verdict."""
    for kind in ("protein", "peptide", "ligand", "nucleic"):
        cuts = _applied_cuts(None, kind)
        missing = unsourced(cuts)
        assert missing == [], f"{kind} screen records cuts with no provenance: {missing}"


def test_the_screen_stores_the_provenance_beside_the_cuts() -> None:
    cuts = _applied_cuts(None, "protein")
    assert PROVENANCE_KEY in cuts
    described = cuts[PROVENANCE_KEY]
    assert isinstance(described, dict)
    # One record per cut, and the provenance block does not describe itself.
    assert set(described) == set(cut_names(cuts))
    assert PROVENANCE_KEY not in described


def test_every_kind_is_in_the_closed_vocabulary() -> None:
    for name, found in ALL_PROVENANCE.items():
        assert found.kind in KINDS, f"{name} has kind {found.kind!r}, which is not in KINDS"


def test_a_published_cut_carries_its_citation_and_a_convention_does_not() -> None:
    """The distinction the whole module exists for, on the two cuts that show it."""
    published = provenance_of("instability_max")
    assert published.kind == PUBLISHED
    assert "Guruprasad" in published.citation
    assert "1990" in published.citation

    convention = provenance_of("gravy_max")
    assert convention.kind == CONVENTION
    assert convention.citation == ""
    # A reader must be told it is this program's choice, not left to infer it.
    assert "this program" in convention.caveat


def test_only_published_cuts_carry_a_citation() -> None:
    for name, found in ALL_PROVENANCE.items():
        if found.kind != PUBLISHED:
            assert found.citation == "", f"{name} is {found.kind} but carries a citation"


def test_the_ada_cut_is_published_and_still_says_why_it_is_refused() -> None:
    """A published threshold is not a portable one, and the record has to say so."""
    found = provenance_of("ada_median_rank_max")
    assert found.kind == PUBLISHED
    assert "Paul" in found.citation
    assert "10.1016/j.jim.2015.03.022" in found.citation
    assert "consensus" in found.caveat
    assert "rank_scale_unknown" in found.caveat


def test_the_unset_target_cuts_say_the_caller_supplies_them() -> None:
    cuts = _applied_cuts(None, "protein")
    for name in ("plasma_abundance_max", "plasma_hub_max", "plasma_fold_max"):
        assert cuts[name] is None
        assert provenance_of(name).kind == CALLER_SUPPLIED


def test_an_unregistered_cut_is_reported_rather_than_left_bare() -> None:
    invented = {"a_cut_nobody_registered": 1.0}
    assert unsourced(invented) == ["a_cut_nobody_registered"]
    record = describe_cuts(invented)["a_cut_nobody_registered"]
    assert record["kind"] == "unrecorded"
    assert "no entry" in record["caveat"]


def test_the_existing_source_slugs_do_not_move() -> None:
    """Readers and two MCP tests already assert these exact strings."""
    assert provenance_of("gravy_max").source == "program_convention"
    assert provenance_of("instability_max").source == "expasy_protparam_guruprasad_1990"
    cuts = _applied_cuts(None, "protein")
    assert cuts["gravy_max_source"] == "program_convention"
    assert cuts["instability_max_source"] == "expasy_protparam_guruprasad_1990"


def test_the_panel_cut_source_agrees_with_the_registry() -> None:
    """combinations and the registry must not drift apart about the same cut."""
    from rfproteina.disease.combinations import LIGAND_AXES, panel_cut_source

    registered = provenance_of("ligand_panel_nm_max").source
    for axis in LIGAND_AXES:
        assert panel_cut_source(axis) == registered, (
            f"{axis}: combinations says {panel_cut_source(axis)!r}, registry says {registered!r}"
        )


def _screen_with(gravy: float, instability: float, passed: bool) -> dict[str, object]:
    return {
        "sequence_developability": {
            "gravy": gravy,
            "instability_index": instability,
            "sequence_pass": passed,
        },
        "cuts": _applied_cuts(None, "protein"),
    }


def test_a_failed_sequence_pass_names_the_bar_and_the_kind_of_bar() -> None:
    """The case that prompted this: a convention decided the verdict, silently.

    GRAVY 0.1304 is over the 0.1 convention; instability 30.0 clears the
    published bar of 40. The line has to say which one rejected the design and
    that it was this program's own number.
    """
    line = _measured_line("8TLD", "2", _screen_with(0.1304, 30.0034, False))
    assert line is not None
    assert "failed gravy_max 0.1, convention" in line
    # The bar that passed is not blamed.
    assert "instability_max" not in line


def test_a_failure_against_the_published_bar_says_published() -> None:
    line = _measured_line("X", "1", _screen_with(0.0, 55.0, False))
    assert line is not None
    assert "failed instability_max 40.0, published" in line
    assert "gravy_max" not in line


def test_both_bars_failing_names_both() -> None:
    line = _measured_line("X", "1", _screen_with(0.9, 55.0, False))
    assert line is not None
    assert "gravy_max" in line and "instability_max" in line
    assert "convention" in line and "published" in line


def test_a_pass_adds_no_explanation() -> None:
    line = _measured_line("X", "1", _screen_with(0.0, 10.0, True))
    assert line is not None
    assert "sequence_pass true" in line
    assert "failed" not in line


def test_a_screen_stored_before_cuts_carried_bars_still_formats() -> None:
    """Graceful on an old record: no bars to name, so no sentence is invented."""
    old = {
        "sequence_developability": {"gravy": 0.9, "instability_index": 55.0, "sequence_pass": False},
    }
    line = _measured_line("X", "1", old)
    assert line is not None
    assert "sequence_pass false" in line
    assert "failed" not in line


def test_every_interface_bar_has_a_provenance() -> None:
    """The same guard, for the bars that live in interface_quality."""
    from rfproteina.disease.interface_quality import REFERENCE_THRESHOLDS

    missing = sorted(set(REFERENCE_THRESHOLDS) - set(INTERFACE_PROVENANCE))
    assert missing == [], f"interface bars with no provenance: {missing}"
    stale = sorted(set(INTERFACE_PROVENANCE) - set(REFERENCE_THRESHOLDS))
    assert stale == [], f"provenance for bars that no longer exist: {stale}"


def test_a_bar_with_no_cut_is_reported_only_and_one_with_a_cut_is_not() -> None:
    """The classification has to track what the runtime actually adopts."""
    from rfproteina.disease.interface_quality import REFERENCE_THRESHOLDS

    for name, entry in REFERENCE_THRESHOLDS.items():
        adopted = entry.get("min") is not None or entry.get("max") is not None
        kind = provenance_of(name).kind
        if adopted:
            assert kind != REPORTED_ONLY, f"{name} adopts a bar but is classified reported_only"
        else:
            assert kind == REPORTED_ONLY, f"{name} adopts no bar but is classified {kind}"


def test_the_borrowed_bars_say_whose_default_and_that_they_are_slack() -> None:
    """A borrowed bar that is slack is a floor, and the record must not imply otherwise."""
    for name in ("interface_residues_binder", "interface_polar_contacts"):
        found = provenance_of(name)
        assert found.kind == BORROWED
        assert "BindCraft" in found.caveat
        assert "not a published value" in found.caveat or "DEFINITION DIFFERS" in found.caveat
        assert "floor" in found.caveat


def test_the_interface_block_carries_the_bars_and_their_kinds() -> None:
    measured = {"interface_polar_contacts": 9, "interface_residues_binder": 27}
    attached = _with_interface_thresholds(measured)
    assert isinstance(attached, dict)
    block = attached[PROVENANCE_KEY]
    # The measurements are untouched beside it.
    assert attached["interface_polar_contacts"] == 9
    assert block["interface_polar_contacts"]["kind"] == BORROWED
    assert block["interface_polar_contacts"]["bar"] == {"min": 3}
    assert block["buried_unsatisfied_polar_atoms"]["bar"] == {"max": 4}
    # No cut adopted reads as no bar, not as a missing one.
    assert block["shape_complementarity"]["bar"] is None
    assert block["shape_complementarity"]["kind"] == REPORTED_ONLY


def test_a_screen_with_no_interface_is_left_alone() -> None:
    assert _with_interface_thresholds(None) is None
    assert _with_interface_thresholds("absent") == "absent"


def test_every_bar_inside_a_screen_has_a_provenance() -> None:
    """The guard, extended to the thresholds that live inside individual screens.

    Each module owns its own threshold dict. This walks them and fails when one
    gains a bar the registry does not know, or keeps one the registry still
    describes, so the two cannot drift apart.
    """
    from rfproteina.disease.ligand_alerts import PROPERTY_BARS
    from rfproteina.disease.novelty import NOVELTY_THRESHOLDS
    from rfproteina.disease.offtarget import OFFTARGET_THRESHOLDS
    from rfproteina.disease.stability import STABILITY_BARS

    owned = {
        "novelty": NOVELTY_THRESHOLDS,
        "offtarget": OFFTARGET_THRESHOLDS,
        "stability": STABILITY_BARS,
        "ligand_alerts": PROPERTY_BARS,
    }
    expected = {f"{module}.{bar}" for module, bars in owned.items() for bar in bars}
    missing = sorted(expected - set(SCREEN_BAR_PROVENANCE))
    assert missing == [], f"bars with no provenance: {missing}"
    stale = sorted(set(SCREEN_BAR_PROVENANCE) - expected)
    assert stale == [], f"provenance for bars that no longer exist: {stale}"


def test_the_colliding_evalue_bars_are_kept_apart() -> None:
    """novelty and offtarget both have evalue_max. A flat key would merge them."""
    novelty = provenance_of("novelty.evalue_max")
    offtarget = provenance_of("offtarget.evalue_max")
    assert novelty.kind == CONVENTION and offtarget.kind == CONVENTION
    assert novelty.caveat != offtarget.caveat
    assert "Foldseek" in novelty.caveat
    # A bare lookup must not silently answer for either of them.
    assert provenance_of("evalue_max").kind == "unrecorded"


def test_the_novelty_cut_is_published_and_warns_about_its_polarity() -> None:
    found = provenance_of("novelty.same_fold_tmscore")
    assert found.kind == PUBLISHED
    assert "Xu and Zhang" in found.citation
    assert "NOT novel" in found.caveat


def test_the_lipinski_bars_are_published_and_qed_is_not() -> None:
    for bar in ("mol_weight", "clogp", "h_bond_donors", "h_bond_acceptors"):
        found = provenance_of(f"ligand_alerts.{bar}")
        assert found.kind == PUBLISHED, f"{bar} should be published"
        assert "Lipinski" in found.citation
    qed = provenance_of("ligand_alerts.qed")
    assert qed.kind == CONVENTION
    assert qed.citation == ""
    assert "publish no pass mark" in qed.caveat


def test_a_summary_line_names_the_kind_and_an_unset_cut_reads_as_unset() -> None:
    assert "published" in summary_line("instability_max", 40.0)
    assert "Guruprasad" in summary_line("instability_max", 40.0)
    assert "convention" in summary_line("gravy_max", 0.1)
    assert "unset" in summary_line("plasma_hub_max", None)
