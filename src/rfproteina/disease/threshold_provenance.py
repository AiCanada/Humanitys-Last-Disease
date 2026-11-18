"""Where each screening threshold came from, in a form a reader can act on.

Sibling to :mod:`rfproteina.disease.provenance`, which answers *which code*
produced a record. This one answers *where its numbers came from*. A verdict
is reproducible against its inputs, its cuts, its bar and its code; recording
the cuts makes it reproducible, and recording what kind of thing each cut is
makes it interpretable.

The gap this closes is small and specific. Twelve cuts are written into every
screen. Two of them named a source. From the values alone a reader cannot tell
that ``instability_max`` of 40 is a published prediction rule while
``gravy_max`` of 0.1 is a number this program chose, and a
``sequence_pass: false`` decided by the second reads exactly like one decided
by the first.

The kinds are a closed vocabulary, so a consumer can branch on them instead of
matching strings:

``published``
    A paper states this threshold for this purpose. It may still be the wrong
    threshold for your design, but someone defended it in print.
``recommended``
    An external body documents it as recommended practice without a paper
    behind the specific number. IEDB's class I selection rule is the example.
``convention``
    This program chose it. No publication and no external recommendation. A
    pass against one of these is a statement about this program's strictness,
    not about the literature.
``caller_supplied``
    The operator passes it, and it is ``None`` until they do. An axis with one
    of these unset reports ``threshold_not_set`` rather than guessing.
``measurement_parameter``
    Not a pass bar at all: a constant that shapes a measurement. Recorded so
    the number stays reproducible, and classified so nobody reads it as a cut.

Every classification is taken from what the defining constant already says
about itself in ``liability.py``, ``screen.py`` or ``combinations.py``. None is
inferred, and where this file and a constant disagree the constant is right.

A cut that reaches :func:`describe_cuts` with no entry is reported as
``unrecorded`` rather than left bare, and ``tests/test_disease_threshold_provenance.py``
fails on it, so a new threshold cannot be added without saying where it came from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

#: A paper states this threshold for this purpose.
PUBLISHED = "published"
#: An external body recommends it; no paper defends the specific number.
RECOMMENDED = "recommended"
#: This program's own choice. Nothing external behind it.
CONVENTION = "convention"
#: The operator supplies it; ``None`` until they do.
CALLER_SUPPLIED = "caller_supplied"
#: Another project's default, carried across. Not a published value, and the
#: definition or the scale it was fitted against may differ from the one it is
#: applied to here. Every such entry says which.
BORROWED = "borrowed"
#: Measured and recorded, with no cut adopted. Distinct from
#: ``measurement_parameter``: this is a quantity that could carry a bar and
#: deliberately does not, so a reader does not go looking for the pass.
REPORTED_ONLY = "reported_only"
#: Shapes a measurement rather than deciding a pass.
MEASUREMENT_PARAMETER = "measurement_parameter"
#: No entry in :data:`ALL_PROVENANCE`. A defect, not a kind.
UNRECORDED = "unrecorded"

#: The closed vocabulary. A kind outside this set is a bug in this file.
KINDS = frozenset(
    {
        PUBLISHED,
        RECOMMENDED,
        CONVENTION,
        CALLER_SUPPLIED,
        BORROWED,
        REPORTED_ONLY,
        MEASUREMENT_PARAMETER,
        UNRECORDED,
    }
)


@dataclass(frozen=True)
class Provenance:
    """What kind of threshold this is, and the shortest honest account of it.

    ``source`` is the stable slug already written beside two of the cuts, kept
    unchanged so existing readers and tests do not move. ``citation`` is empty
    for anything that is not ``published``: an empty citation is the point, not
    an omission to fill in later.

    ``caveat`` carries the condition under which the number stops applying, and
    is the field most worth reading. ``ada_median_rank_max`` is a genuinely
    published threshold that is still refused on most machines, because it is
    defined on a rank scale the installed predictor does not produce.
    """

    kind: str
    source: str
    citation: str = ""
    caveat: str = ""

    def as_record(self) -> dict[str, str]:
        """The flat form stored in a screen. Empty fields are dropped."""
        record = {"kind": self.kind, "source": self.source}
        if self.citation:
            record["citation"] = self.citation
        if self.caveat:
            record["caveat"] = self.caveat
        return record


#: Every cut ``screen._applied_cuts`` writes, and where it came from.
CUT_PROVENANCE: dict[str, Provenance] = {
    "gravy_max": Provenance(
        kind=CONVENTION,
        source="program_convention",
        caveat=(
            "No published bar says where a developable protein stops. The number asks that "
            "the chain not be net hydrophobic overall; it is this program's strictness, not "
            "a literature value."
        ),
    ),
    "instability_max": Provenance(
        kind=PUBLISHED,
        source="expasy_protparam_guruprasad_1990",
        citation=(
            "Guruprasad, Reddy and Pandit, Protein Engineering 4(2):155-161 (1990), as applied "
            "by ExPASy ProtParam: a protein whose instability index is below 40 is predicted "
            "as stable."
        ),
    ),
    "clash_distance_angstrom": Provenance(
        kind=MEASUREMENT_PARAMETER,
        source="non_bonded_clash_distance",
        caveat="Defines what counts as a clash for the reported fraction. No pass cut is adopted.",
    ),
    "mhc_i_rank_max": Provenance(
        kind=RECOMMENDED,
        source="iedb_class_i_selection",
        caveat="IEDB's recommended class I selection: percentile rank <= 1, or IC50 <= 500 nM.",
    ),
    "mhc_i_ic50_nm_max": Provenance(
        kind=RECOMMENDED,
        source="iedb_class_i_selection",
        caveat="The IC50 half of the same IEDB rule. Either condition selects a peptide.",
    ),
    "mhc_ii_rank_max": Provenance(
        kind=RECOMMENDED,
        source="iedb_class_ii_recommended_rank",
        caveat=(
            "IEDB's recommended class II percentile rank. It assumes a continuous rank: the "
            "installed TEPITOPE scorer emits only the bins 1, 10 and 100, so at a cut of 10 "
            "two of its three bins pass and the reference density is refused as implausible."
        ),
    ),
    "mhc_ii_ic50_nm_max": Provenance(
        kind=RECOMMENDED,
        source="iedb_class_ii_binding_cutoff",
        caveat="The IC50 binding cutoff for class II. Unused by a scorer that reports no IC50.",
    ),
    "ada_median_rank_max": Provenance(
        kind=PUBLISHED,
        source="paul_2015_dr_panel_breadth",
        citation=(
            "Paul et al., Journal of Immunological Methods 422:28-34 (2015), "
            "doi:10.1016/j.jim.2015.03.022: a blind-validated universal rule, a median "
            "percentile rank of 20 or better across the seven-allele DR panel."
        ),
        caveat=(
            "Defined on the IEDB consensus percentile rank. A table produced by any other "
            "method is on a different scale, and the axis reports rank_scale_unknown rather "
            "than applying the number anyway. A published threshold is not a portable one."
        ),
    ),
    "ligand_panel_nm_max": Provenance(
        kind=CONVENTION,
        source="program_convention_10uM",
        caveat=(
            "10 uM for every panel axis, this program's screening convention. No entry is from "
            "a published rule: hERG is normally judged as a margin over free Cmax rather than "
            "at a fixed concentration, and BSEP inhibition in DILI work is commonly discussed "
            "nearer 25-50 uM."
        ),
    ),
    "plasma_abundance_max": Provenance(
        kind=CALLER_SUPPLIED,
        source="caller_supplied",
        caveat="No published value to fall back on. Unset until the operator passes one.",
    ),
    "plasma_hub_max": Provenance(
        kind=CALLER_SUPPLIED,
        source="caller_supplied",
        caveat="No published value to fall back on. Unset until the operator passes one.",
    ),
    "plasma_fold_max": Provenance(
        kind=CALLER_SUPPLIED,
        source="caller_supplied",
        caveat="No published value to fall back on. Unset until the operator passes one.",
    ),
}

#: The interface-quality bars in ``interface_quality.REFERENCE_THRESHOLDS``.
#:
#: These are the bars most in need of a kind. Three are BindCraft's defaults
#: carried across to proxies BindCraft never fitted, and each is slack against
#: this program's own measured distribution by a wide margin: a reader seeing
#: ``interface_polar_contacts >= 3`` cleared has learned almost nothing, since
#: the measured median is 18. They are floors that catch a catastrophe, not
#: discriminators, and the caveat says so rather than leaving the bar to imply
#: otherwise. The two with no cut say that too, so nobody hunts for a pass that
#: was deliberately not adopted.
#:
#: Every figure below is quoted from ``REFERENCE_THRESHOLDS`` in
#: ``interface_quality.py``, which is the runtime source of truth.
INTERFACE_PROVENANCE: dict[str, Provenance] = {
    "interface_residues_binder": Provenance(
        kind=BORROWED,
        source="bindcraft_average_n_interface_residues",
        caveat=(
            "BindCraft settings_filters/default_filters.json, read 2026-11-16: that project's "
            "default, not a published value. Measured median here is 27 (range 16-47) over 23 "
            "designs, so every accepted design clears the bar of 7. A floor, not a discriminator."
        ),
    ),
    "interface_polar_contacts": Provenance(
        kind=BORROWED,
        source="bindcraft_average_n_interface_hbonds",
        caveat=(
            "BindCraft's default of 3, read 2026-11-16. DEFINITION DIFFERS: BindCraft counts "
            "Rosetta hydrogen bonds on a protonated model, this counts heavy-atom N/O pairs within "
            "3.5 A and overcounts against that. Measured median here is 18, so the bar of 3 is "
            "slack and kept only as a floor."
        ),
    ),
    "buried_unsatisfied_polar_atoms": Provenance(
        kind=BORROWED,
        source="bindcraft_average_n_interface_unsat_hbonds",
        caveat=(
            "BindCraft's default of 4, read 2026-11-16. DEFINITION DIFFERS: this is a "
            "chemistry-typed heavy-atom proxy without hydrogens. Measured median here is 2, under "
            "the bar, so <= 4 is adopted as a calibrated-compatible soft cut."
        ),
    ),
    "shape_complementarity": Provenance(
        kind=REPORTED_ONLY,
        source="lawrence_colman_1993_scale_differs",
        caveat=(
            "NO CUT adopted. Lawrence & Colman 1993 report Sc ~0.64-0.76 for natural interfaces "
            "and BindCraft's bar is 0.55, both on the Connolly surface with re-entrant patches. "
            "SCALE DIFFERS: this Sc is on a contact dot surface without them and runs "
            "systematically lower, measured median 0.166 (range 0.069-0.254) over 23 designs. "
            "0.55 would fail every accepted design. Triage on the measured distribution."
        ),
    ),
    "dsasa_total_a2": Provenance(
        kind=REPORTED_ONLY,
        source="no_cut_adopted",
        caveat=(
            "NO CUT adopted. BindCraft's Average_dSASA default of 1 is a fraction of the maximum, "
            "not an area, so it cannot be carried across to this absolute value. Measured median "
            "2393 A^2 (range 1353-4356) over 23 designs, recorded for reference only."
        ),
    ),
}

#: Every threshold this module can classify, from either family. The two names
#: do not overlap, and a lookup should not care which family a bar came from.
ALL_PROVENANCE: dict[str, Provenance] = {**CUT_PROVENANCE, **INTERFACE_PROVENANCE}

#: Where the per-cut records are stored inside ``cuts``.
PROVENANCE_KEY = "provenance"

#: What a cut with no entry gets, so the gap shows in the verdict rather than hiding.
UNRECORDED_PROVENANCE = Provenance(
    kind=UNRECORDED,
    source="unrecorded",
    caveat=(
        "This threshold has no entry in threshold_provenance.CUT_PROVENANCE. Treat the pass "
        "it decides as unsourced."
    ),
)


def provenance_of(cut: str) -> Provenance:
    """Where ``cut`` came from, from either family. Never raises.

    A verdict always says something: a threshold nobody registered comes back
    ``unrecorded`` with a caveat saying to treat the pass it decides as
    unsourced, which is the honest answer and a visible one.
    """
    return ALL_PROVENANCE.get(cut, UNRECORDED_PROVENANCE)


def cut_names(cuts: Mapping[str, object]) -> list[str]:
    """The cut names in ``cuts``: not the ``*_source`` strings, not the provenance block."""
    return sorted(
        name
        for name in cuts
        if not name.endswith("_source") and name != PROVENANCE_KEY
    )


def describe_cuts(cuts: Mapping[str, object]) -> dict[str, dict[str, str]]:
    """One provenance record per cut, keyed by cut name.

    Keyed rather than flattened into ``<cut>_kind`` and ``<cut>_citation``
    pairs so the verdict gains one key instead of three dozen, which matters to
    every consumer holding an allowlist of record keys.
    """
    return {name: provenance_of(name).as_record() for name in cut_names(cuts)}


def unsourced(cuts: Mapping[str, object]) -> list[str]:
    """Thresholds with no provenance entry, from either family. Empty is healthy."""
    return [name for name in cut_names(cuts) if name not in ALL_PROVENANCE]


def describe_interface_thresholds() -> dict[str, dict[str, str]]:
    """One provenance record per interface-quality bar, keyed by bar name.

    Read straight from :data:`INTERFACE_PROVENANCE` rather than from a measured
    record, because these bars are fixed for the program: unlike the target
    cuts, no caller supplies them per run.
    """
    return {name: found.as_record() for name, found in sorted(INTERFACE_PROVENANCE.items())}


def summary_line(cut: str, value: object) -> str:
    """One line a human can read, for a report or a console summary."""
    found = provenance_of(cut)
    shown = "unset" if value is None else value
    if found.kind == PUBLISHED:
        return f"{cut} = {shown} ({found.kind}: {found.citation})"
    if found.caveat:
        return f"{cut} = {shown} ({found.kind}: {found.caveat})"
    return f"{cut} = {shown} ({found.kind})"
