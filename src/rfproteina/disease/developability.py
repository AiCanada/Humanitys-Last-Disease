"""Antibody developability from sequence: numbering, CDRs, and chemical liability motifs.

Nothing in this module judges a design. It reports, the way
``rfproteina.disease.solubility`` reports: a record with findings, each finding
carrying the motif, the region it sits in, and the publication the motif comes
from. No pass cut is invented here, because the published work this rests on
reports frequencies of modification, not a number of motifs above which an
antibody fails. Lu et al. found that only 47 per cent of CDRH2 NG instances and
19 per cent of CDRH2 DG instances actually modified under the matching stress,
so a motif is a place to look, not a verdict.

Three layers, in order of how much they assume:

1. **Numbering.** ANARCII (BSD-3-Clause, University of Oxford, installed in
   this project's virtual environment as ``anarcii`` 2.0.8) assigns IMGT
   numbers to a variable domain and names the chain type. It is a torch model
   and takes roughly 28 seconds to load on CPU, so it is imported lazily and
   never at module scope, and the tests do not call it.
2. **Regions.** IMGT CDR and framework delimitations are fixed spans in the
   IMGT unique numbering, so once a chain is numbered its CDRs are a lookup,
   not a prediction. Only IMGT is implemented. Kabat and Chothia CDRs are
   different spans on different numbering and are not guessed at here.
3. **Motifs.** A scan over the numbered residues for the chemical liabilities
   that the therapeutic-antibody literature names, each with its source.

What this module does **not** do: it does not compute solvent exposure, so an
oxidation or deamidation motif flagged here is a sequence motif and not an
exposure-weighted risk. The reviews say plainly that exposure is the dominant
predictor for Met and Trp oxidation; a sequence scan over-calls. That is said
in the record itself, in ``caveat``, so a reader downstream cannot lose it.

Two external developability oracles are supported as *readers*, not as
installs: DeepSP and CrossAbSense. Neither is run here. CrossAbSense's
published weights sit behind a frozen ESM-Cambrian 6B encoder, which is not a
CPU screening step, and DeepSP ships Keras models this environment has no
TensorFlow for. Both write a CSV of per-sequence numbers, and the readers below
take that CSV and attach the method, the assay and the provenance to each
value, refusing any row that does not name them.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

#: IMGT unique numbering for the V-DOMAIN: CDR1 is positions 27-38, CDR2 56-65
#: and CDR3 105-117, with the two conserved cysteines at 23 and 104 and the
#: conserved tryptophan at 41. Source: Lefranc et al., "IMGT unique numbering
#: for immunoglobulin and T cell receptor variable domains and Ig superfamily
#: V-like domains", Dev Comp Immunol 27:55-77 (2003).
IMGT_CDR_SPANS: dict[str, tuple[int, int]] = {
    "cdr1": (27, 38),
    "cdr2": (56, 65),
    "cdr3": (105, 117),
}
IMGT_FRAMEWORK_SPANS: dict[str, tuple[int, int]] = {
    "fr1": (1, 26),
    "fr2": (39, 55),
    "fr3": (66, 104),
    "fr4": (118, 128),
}
#: The two cysteines every V domain carries, which pair with each other. A
#: cysteine anywhere else is the unpaired-cysteine liability below.
IMGT_CANONICAL_CYSTEINES = (23, 104)
#: The scheme this module understands. ANARCII numbers in IMGT by default and
#: reports the scheme it used; anything else is refused rather than read as
#: IMGT.
SCHEME = "imgt"
NUMBERING_METHOD = "anarcii"
NUMBERING_LICENCE = "BSD-3-Clause"

_STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
_GAP = "-"


class NumberingUnavailableError(Exception):
    """ANARCII is not installed, or it refused the sequence. No regions are guessed."""


@dataclass(frozen=True)
class LiabilityMotif:
    """One chemical liability, its motifs, and where the motif list comes from."""

    name: str
    kind: str
    motifs: tuple[str, ...]
    rationale: str
    source: str
    hierarchy: str | None = None
    provenance: str = "cited"


#: The motif sets, each with the publication that lists it. The two dipeptide
#: sets are quoted from Lu et al.'s "canonical motifs" paragraph, which names
#: them explicitly and gives the lability ordering that follows.
LIABILITY_MOTIFS: tuple[LiabilityMotif, ...] = (
    LiabilityMotif(
        name="deamidation",
        kind="dipeptide",
        motifs=("NG", "NS", "NN", "NT", "NH"),
        rationale=(
            "Asn deamidation runs through a succinimide intermediate whose formation depends on the n+1 residue. "
            "NG was the most frequently modified motif across 131 clinical-stage antibodies, with CDRH2 and CDRL1 "
            "the most affected regions."
        ),
        source=(
            "Lu et al., 'Deamidation and isomerization liability analysis of 131 clinical-stage antibodies', "
            "mAbs 11(1):45-57 (2019), doi:10.1080/19420862.2018.1548233"
        ),
        hierarchy="NG > NS > NT (decreasing lability, as stated by the same paper)",
    ),
    LiabilityMotif(
        name="isomerisation",
        kind="dipeptide",
        motifs=("DG", "DS", "DD", "DT", "DH"),
        rationale=(
            "Asp isomerisation to isoAsp runs through the same succinimide chemistry. DG dominated the observed "
            "isomerisation events, concentrated in CDRH3."
        ),
        source=(
            "Lu et al., 'Deamidation and isomerization liability analysis of 131 clinical-stage antibodies', "
            "mAbs 11(1):45-57 (2019), doi:10.1080/19420862.2018.1548233"
        ),
        hierarchy="DG > DS > DD or DT (decreasing lability, as stated by the same paper)",
    ),
    LiabilityMotif(
        name="oxidation",
        kind="residue",
        motifs=("M", "W"),
        rationale=(
            "Met and Trp are the residues oxidation reaches first, and CDR loops are where they are exposed. "
            "Sequence-based assessment flags every Met and Trp in the variable domains; the reviews are explicit "
            "that this over-calls, because solvent exposure is the dominant predictor and this module computes none."
        ),
        source=(
            "Xu et al., 'Structure, heterogeneity and developability assessment of therapeutic antibodies', "
            "mAbs 11(2):239-264 (2019), doi:10.1080/19420862.2018.1553476; and "
            "'In silico prediction of post-translational modifications in therapeutic antibodies', "
            "mAbs 14(1) (2022), PMC8791605"
        ),
    ),
    LiabilityMotif(
        name="unpaired_cysteine",
        kind="cysteine",
        motifs=("C",),
        rationale=(
            "A cysteine outside the conserved IMGT 23/104 pair has no partner in the domain. A free CDR cysteine is "
            "readily modified by free cysteine in culture medium and has been shown to reduce antigen binding; the "
            "review's advice is to eliminate such candidates."
        ),
        source=(
            "Xu et al., 'Structure, heterogeneity and developability assessment of therapeutic antibodies', "
            "mAbs 11(2):239-264 (2019), doi:10.1080/19420862.2018.1553476"
        ),
    ),
    LiabilityMotif(
        name="n_glycosylation_sequon",
        kind="sequon",
        motifs=("N-X-S", "N-X-T"),
        rationale=(
            "N-linked glycosylation is directed by the Asn-Xaa-Ser/Thr sequon, with Xaa anything but proline. A "
            "sequon in a CDR risks heterogeneous occupancy and a glycan over the binding site."
        ),
        source=(
            "Xu et al., 'Structure, heterogeneity and developability assessment of therapeutic antibodies', "
            "mAbs 11(2):239-264 (2019), doi:10.1080/19420862.2018.1553476, which describes N-linked glycosylation "
            "at the Asn-Xaa-Ser/Thr sequon in variable domains"
        ),
    ),
)

#: Said in the record so a downstream reader cannot drop it.
SEQUENCE_SCAN_CAVEAT = (
    "Motifs are sequence patterns, not exposure-weighted risks. No solvent accessibility is computed here, and "
    "the cited work reports that only a minority of canonical motifs actually modify under stress."
)


@dataclass(frozen=True)
class Residue:
    """One numbered residue of a variable domain."""

    chain: str
    chain_type: str
    position: int
    insertion: str
    residue: str

    @property
    def label(self) -> str:
        """IMGT label as a reader would write it: ``H105`` or ``H111A``."""
        return f"{self.chain_type}{self.position}{self.insertion.strip()}"


@dataclass(frozen=True)
class NumberedChain:
    """A numbered variable domain, with its regions already cut."""

    name: str
    chain_type: str
    scheme: str
    residues: tuple[Residue, ...]
    score: float | None = None
    regions: dict[str, str] = field(default_factory=dict)


def numbering_available() -> tuple[bool, str | None]:
    """Whether ANARCII can be imported. Checked, never assumed."""
    try:
        import anarcii  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment dependent
        return False, f"anarcii_import_failed:{error}"
    return True, None


def number_chains(sequences: Mapping[str, str], *, cpu: bool = True, ncpu: int = 1) -> list[NumberedChain]:
    """IMGT-number each named sequence with ANARCII.

    Raises :class:`NumberingUnavailableError` when ANARCII is absent, and skips
    nothing silently: a sequence ANARCII reports an error for is returned as a
    chain with ``chain_type`` ``"F"`` and no residues only if ANARCII itself
    said so, otherwise the error is raised with the sequence's name attached.
    """
    available, reason = numbering_available()
    if not available:
        raise NumberingUnavailableError(reason or "anarcii_unavailable")
    from anarcii import Anarcii

    names = list(sequences)
    if not names:
        return []
    model = Anarcii(seq_type="antibody", batch_size=1, cpu=cpu, ncpu=ncpu, verbose=False)
    numbered = model.number([sequences[name] for name in names])
    return parse_anarcii(dict(zip(names, list(numbered.values()), strict=True)))


def parse_anarcii(results: Mapping[str, Mapping[str, object]]) -> list[NumberedChain]:
    """ANARCII's own output dict -> :class:`NumberedChain` objects.

    Split out from :func:`number_chains` so the whole parse, region cut and
    motif scan can be tested without loading a torch model. ANARCII writes
    ``numbering`` as ``[((position, insertion), residue), ...]`` with ``-`` for
    an unoccupied position, and reports ``chain_type``, ``scheme``, ``score``
    and ``error``.
    """
    chains: list[NumberedChain] = []
    for name, result in results.items():
        error = result.get("error")
        if error:
            raise NumberingUnavailableError(f"anarcii_refused:{name}:{error}")
        scheme = str(result.get("scheme") or "").casefold()
        if scheme != SCHEME:
            # A Kabat or Chothia numbering carries different CDR spans, and
            # reading one with the IMGT spans would cut the wrong loops.
            raise NumberingUnavailableError(f"unsupported_scheme:{scheme or 'unset'}")
        chain_type = str(result.get("chain_type") or "?")
        raw = result.get("numbering") or ()
        residues: list[Residue] = []
        for item in raw:  # type: ignore[union-attr]
            (position, insertion), letter = item
            if letter == _GAP:
                continue
            residues.append(
                Residue(
                    chain=name,
                    chain_type=chain_type,
                    position=int(position),
                    insertion=str(insertion),
                    residue=str(letter).upper(),
                )
            )
        score = result.get("score")
        chains.append(
            NumberedChain(
                name=name,
                chain_type=chain_type,
                scheme=SCHEME,
                residues=tuple(residues),
                score=float(score) if isinstance(score, int | float) else None,
                regions=region_sequences(residues),
            )
        )
    return chains


def region_of(position: int) -> str:
    """Which IMGT region a position falls in. ``"other"`` past 128."""
    for region, (low, high) in IMGT_CDR_SPANS.items():
        if low <= position <= high:
            return region
    for region, (low, high) in IMGT_FRAMEWORK_SPANS.items():
        if low <= position <= high:
            return region
    return "other"


def region_sequences(residues: Sequence[Residue]) -> dict[str, str]:
    """The residues of each IMGT region, in order, as plain strings."""
    found: dict[str, list[str]] = {}
    for residue in residues:
        found.setdefault(region_of(residue.position), []).append(residue.residue)
    return {region: "".join(letters) for region, letters in found.items()}


def scan_chain(chain: NumberedChain, *, cdr_only: bool = False) -> list[dict[str, object]]:
    """Every liability motif in one numbered chain, newest rule first.

    ``cdr_only`` restricts the scan to CDR1-3. The default is the whole domain,
    because the framework liabilities are real even though the cited frequency
    work discusses CDRs; every finding carries its region so a caller can
    filter without losing the count.
    """
    residues = [item for item in chain.residues if item.residue in _STANDARD_RESIDUES]
    findings: list[dict[str, object]] = []
    for motif in LIABILITY_MOTIFS:
        if motif.kind == "dipeptide":
            findings.extend(_scan_dipeptides(chain, residues, motif))
        elif motif.kind == "residue":
            findings.extend(_scan_residues(chain, residues, motif))
        elif motif.kind == "cysteine":
            findings.extend(_scan_cysteines(chain, residues, motif))
        elif motif.kind == "sequon":
            findings.extend(_scan_sequons(chain, residues, motif))
    if cdr_only:
        findings = [item for item in findings if str(item["region"]).startswith("cdr")]
    return sorted(findings, key=lambda item: (str(item["chain"]), int(item["position"]), str(item["liability"])))


def _finding(chain: NumberedChain, residue: Residue, motif: LiabilityMotif, text: str, **extra: object) -> dict:
    record: dict[str, object] = {
        "chain": chain.name,
        "chain_type": chain.chain_type,
        "liability": motif.name,
        "motif": text,
        "label": residue.label,
        "position": residue.position,
        "region": region_of(residue.position),
        "scheme": SCHEME,
        "source": motif.source,
        "rationale": motif.rationale,
        "provenance": motif.provenance,
    }
    if motif.hierarchy:
        record["hierarchy"] = motif.hierarchy
    record.update(extra)
    return record


def _scan_dipeptides(chain: NumberedChain, residues: list[Residue], motif: LiabilityMotif) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for first, second in pairwise(residues):
        pair = f"{first.residue}{second.residue}"
        if pair in motif.motifs:
            found.append(_finding(chain, first, motif, pair, next_label=second.label))
    return found


def _scan_residues(chain: NumberedChain, residues: list[Residue], motif: LiabilityMotif) -> list[dict[str, object]]:
    return [_finding(chain, item, motif, item.residue) for item in residues if item.residue in motif.motifs]


def _scan_cysteines(chain: NumberedChain, residues: list[Residue], motif: LiabilityMotif) -> list[dict[str, object]]:
    extra = [item for item in residues if item.residue == "C" and item.position not in IMGT_CANONICAL_CYSTEINES]
    if not extra:
        return []
    # Two non-canonical cysteines can pair with each other, which the cited
    # review notes happens in some CDRH loops with no stability penalty. An
    # even count is therefore reported as possibly paired rather than as free,
    # and the record says which reading it took instead of choosing silently.
    parity = "odd_count_at_least_one_is_unpaired" if len(extra) % 2 else "even_count_may_be_an_extra_disulfide"
    return [_finding(chain, item, motif, "C", non_canonical_cysteines=len(extra), parity=parity) for item in extra]


def _scan_sequons(chain: NumberedChain, residues: list[Residue], motif: LiabilityMotif) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for index in range(len(residues) - 2):
        first, middle, last = residues[index], residues[index + 1], residues[index + 2]
        if first.residue != "N" or middle.residue == "P" or last.residue not in {"S", "T"}:
            continue
        text = f"N{middle.residue}{last.residue}"
        found.append(_finding(chain, first, motif, text, sequon_labels=[first.label, middle.label, last.label]))
    return found


def developability_report(chains: Sequence[NumberedChain], *, cdr_only: bool = False) -> dict[str, object]:
    """One record over every chain: regions, findings, counts, and the caveat.

    No state, no measurement, no pass cut -- this is a report in the sense
    ``solubility.py`` uses the word, and the screen's call does not move on it.
    """
    if not chains:
        return {"developability_blocker": "no_numbered_chains"}
    findings: list[dict[str, object]] = []
    described: list[dict[str, object]] = []
    for chain in chains:
        findings.extend(scan_chain(chain, cdr_only=cdr_only))
        described.append(
            {
                "chain": chain.name,
                "chain_type": chain.chain_type,
                "scheme": chain.scheme,
                "residues": len(chain.residues),
                "score": chain.score,
                "cdr1": chain.regions.get("cdr1", ""),
                "cdr2": chain.regions.get("cdr2", ""),
                "cdr3": chain.regions.get("cdr3", ""),
                "cdr_lengths": {
                    name: len(chain.regions.get(name, "")) for name in ("cdr1", "cdr2", "cdr3")
                },
            }
        )
    counts: dict[str, int] = {}
    cdr_counts: dict[str, int] = {}
    for item in findings:
        name = str(item["liability"])
        counts[name] = counts.get(name, 0) + 1
        if str(item["region"]).startswith("cdr"):
            cdr_counts[name] = cdr_counts.get(name, 0) + 1
    return {
        "method": f"{NUMBERING_METHOD}_imgt_motif_scan",
        "numbering": {"method": NUMBERING_METHOD, "licence": NUMBERING_LICENCE, "scheme": SCHEME},
        "chains": described,
        "findings": findings,
        "counts": counts,
        "cdr_counts": cdr_counts,
        "scanned": "cdr_only" if cdr_only else "whole_variable_domain",
        "caveat": SEQUENCE_SCAN_CAVEAT,
        "pass_cut": None,
        "pass_cut_reason": "no_published_motif_count_threshold",
    }


# --------------------------------------------------------------------------
# External oracles, read rather than run.
# --------------------------------------------------------------------------

#: DeepSP's three predicted structural-aggregation descriptors. DeepSP is a
#: sequence-based surrogate for the structure-derived spatial aggregation
#: propensity and spatial charge map, published MIT by Lai et al. The columns
#: are the ones its own predictor writes.
DEEPSP_COLUMNS = ("SAP_pos", "SCM_pos", "SCM_neg")
DEEPSP_SOURCE = "DeepSP (MIT), https://github.com/Lailabcode/DeepSP"
#: CrossAbSense's five published oracles, as named on its model card, with the
#: assay each one predicts. Apache-2.0 for both code and the v0.9 weights.
CROSSABSENSE_PROPERTIES = {
    "HIC": "hydrophobic interaction chromatography retention time",
    "PR_CHO": "polyreactivity against CHO cell lysate",
    "AC-SINS": "self-association by affinity-capture self-interaction nanoparticle spectroscopy",
    "Tm2": "thermostability, second melting transition",
    "Titer": "expression yield in HEK293",
}
CROSSABSENSE_SOURCE = "CrossAbSense v0.9 (Apache-2.0), https://huggingface.co/SimonCrouzet/CrossAbSense"
ORACLE_ROW_LIMIT = 64
ORACLE_FILE_LIMIT = 1_048_576


def read_oracle_table(path: Path, *, method: str, source: str, properties: Sequence[str]) -> dict[str, object]:
    """A developability oracle's own CSV -> a record, with provenance attached.

    The file is the oracle's output, produced by whoever ran it. Nothing is run
    here and nothing is rescaled. A column the oracle does not define is
    ignored rather than read as a score, and a file with no defined column at
    all is refused, because a record of zero scores reads as "the oracle found
    nothing" when it means "this is not that oracle's output".
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return {"oracle_blocker": "oracle_unreadable", "method": method}
    if size > ORACLE_FILE_LIMIT:
        return {"oracle_blocker": "oracle_too_large", "method": method}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return {"oracle_blocker": "oracle_unreadable", "method": method}
    reader = csv.DictReader(io.StringIO(text))
    names = [name.strip() for name in (reader.fieldnames or [])]
    wanted = [name for name in properties if name in names]
    if not wanted:
        return {"oracle_blocker": "oracle_columns_absent", "method": method, "expected": list(properties)}
    rows: list[dict[str, object]] = []
    unparsed: list[str] = []
    for row in reader:
        if len(rows) >= ORACLE_ROW_LIMIT:
            return {"oracle_blocker": "oracle_truncated", "method": method}
        clean = {key.strip(): (value or "").strip() for key, value in row.items() if key}
        scores: dict[str, float] = {}
        for name in wanted:
            cell = clean.get(name, "")
            if not cell:
                continue
            try:
                scores[name] = float(cell)
            except ValueError:
                unparsed.append(name)
        if not scores:
            continue
        rows.append({"name": clean.get("name") or clean.get("id") or f"row{len(rows) + 1}", "scores": scores})
    if unparsed:
        return {"oracle_blocker": "cell_unparsed", "method": method, "unparsed_columns": sorted(set(unparsed))}
    if not rows:
        return {"oracle_blocker": "oracle_not_reported", "method": method}
    return {
        "method": method,
        "source": source,
        "properties": list(wanted),
        "rows": rows,
        "pass_cut": None,
        "pass_cut_reason": "oracle_scales_are_assay_units_with_no_published_pass_mark",
    }


def read_deepsp_table(path: Path) -> dict[str, object]:
    """DeepSP's prediction CSV, as a record."""
    return read_oracle_table(path, method="deepsp", source=DEEPSP_SOURCE, properties=DEEPSP_COLUMNS)


def read_crossabsense_table(path: Path) -> dict[str, object]:
    """CrossAbSense's prediction CSV, as a record."""
    return read_oracle_table(
        path,
        method="crossabsense_v0.9",
        source=CROSSABSENSE_SOURCE,
        properties=tuple(CROSSABSENSE_PROPERTIES),
    )
