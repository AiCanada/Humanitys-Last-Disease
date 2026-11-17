"""Humanness and antibody-likeness, from the two genuinely open routes that exist.

Two different quantities are reported here, and the distinction is the whole
point of the module:

1. **OASis identity** -- the fraction of a variable domain's overlapping 9-mers
   that are seen in human repertoires. This is a humanness measurement: the
   reference set is human donors, so a low identity names peptides a human
   immune system has not seen. The peptide set used is the one published with
   BioPhi (Prihoda et al., "BioPhi: A platform for antibody design,
   humanization, and humanness evaluation based on natural antibody
   repertoires and deep learning", mAbs 14(1):2020203, 2022). The full OASis
   database is 6.5 GB compressed and roughly 21.5 GB on disk; this module uses
   the subset shipped inside the ``promb`` package (MIT, MSD), which is the
   OASis 9-mer set restricted to peptides found in more than 10 per cent of
   human subjects -- BioPhi's own default threshold. An identity is meaningless
   without its threshold, so the threshold is written into every record.

2. **AbLang-2 pseudo-log-likelihood** -- how expected the sequence is under an
   antibody language model. This is **not** humanness. AbLang-2 was trained on
   the Observed Antibody Space, which is roughly 88 per cent human and 11 per
   cent mouse (Olsen et al. 2022), so a murine sequence can score well. It is
   reported as ``antibody_likeness`` and never as a humanness score, and the
   record says so in ``not_humanness_reason``.

Neither number carries a pass cut here. BioPhi reports an OASis identity
alongside a percentile against clinical-stage therapeutics rather than a pass
mark, and no publication gives a pseudo-log-likelihood cutoff. Inventing one
would be inventing a threshold.

Licences, each read from the distributing project on 2026-11-16:

* ``promb`` 1.0.2 -- MIT (PyPI metadata), https://github.com/MSDLLCPapers/promb.
  It ships ``OASis_9mers_v1_10perc_subjects.txt.gz`` inside the wheel.
* OASis database v1 -- CC-BY-4.0, Zenodo record 5164685. The packaged subset
  derives from it, so attribution is owed to the OASis authors.
* BioPhi -- MIT (Merck Sharp & Dohme). Not installed here; it is conda-only and
  needs HMMER.
* AbLang-2 (``ablang2`` 0.2.1) -- BSD-3-Clause (PyPI metadata),
  https://github.com/oxpig/AbLang2. Weights are fetched by the package itself
  on first use; nothing is vendored here.

IgFold is deliberately absent: its licence is academic-use only, which fails
this program's free-and-open rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec

#: The packaged OASis subset is thresholded at "seen in more than 10 per cent
#: of human subjects". BioPhi names four conventional settings -- loose 1,
#: relaxed 10, medium 50, strict 90 -- and 10 is its default. Only this one is
#: available offline, so it is the only one this module can report.
OASIS_SUBJECT_THRESHOLD_PERCENT = 10
OASIS_PEPTIDE_LENGTH = 9
OASIS_METHOD = "oasis_identity_10pct_subjects"
OASIS_SOURCE = (
    "Prihoda et al., 'BioPhi: A platform for antibody design, humanization, and humanness evaluation based on "
    "natural antibody repertoires and deep learning', mAbs 14(1):2020203 (2022); peptide set from the promb "
    "package (MIT), derived from the OASis v1 database (CC-BY-4.0, Zenodo 5164685)"
)
ABLANG2_METHOD = "ablang2_pseudo_log_likelihood"
ABLANG2_SOURCE = (
    "Olsen, Moal and Deane, 'Addressing the antibody germline bias and its effect on language models for improved "
    "antibody design', Bioinformatics (2024); model ablang2 0.2.1, BSD-3-Clause"
)
ABLANG2_NOT_HUMANNESS = (
    "AbLang-2 is trained on the Observed Antibody Space, which is about 88 per cent human and 11 per cent mouse. "
    "A high pseudo-log-likelihood means the sequence looks like a natural antibody, not that it looks human."
)

_STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")


class HumannessUnavailableError(Exception):
    """A humanness backend is not installed. No score is estimated in its place."""


@dataclass(frozen=True)
class HumannessOption:
    """One way to score humanness, with the licence facts needed to decide on it."""

    name: str
    licence: str
    licence_verified: str
    weights: str
    installed_here: bool
    note: str


def humanness_options() -> list[HumannessOption]:
    """Every route considered, including the ones not wired in, and why."""
    oasis_ok, _ = oasis_available()
    ablang_ok, _ = ablang2_available()
    return [
        HumannessOption(
            name="OASis identity (promb packaged 10% subject peptide set)",
            licence="promb MIT; peptide set derived from OASis v1, CC-BY-4.0",
            licence_verified="PyPI metadata for promb 1.0.2 and Zenodo record 5164685, read 2026-11-16",
            weights="none; a 40 MB peptide table ships inside the wheel",
            installed_here=oasis_ok,
            note="Humanness in the strict sense: the reference is human donor repertoires.",
        ),
        HumannessOption(
            name="OASis full database via BioPhi",
            licence="BioPhi MIT; OASis database CC-BY-4.0",
            licence_verified="GitHub Merck/BioPhi LICENSE and Zenodo record 5164685, read 2026-11-16",
            weights="6.5 GB download, about 21.5 GB on disk; conda-only install, needs HMMER",
            installed_here=False,
            note="Only this route can move the subject threshold off 10 per cent.",
        ),
        HumannessOption(
            name="AbLang-2 pseudo-log-likelihood",
            licence="BSD-3-Clause",
            licence_verified="PyPI metadata for ablang2 0.2.1, read 2026-11-16",
            weights="fetched by the package on first use, tens of MB",
            installed_here=ablang_ok,
            note="Antibody-likeness, not humanness. See ABLANG2_NOT_HUMANNESS.",
        ),
        HumannessOption(
            name="IgFold",
            licence="academic use only",
            licence_verified="not wired in; excluded on licence grounds by this program's rule",
            weights="n/a",
            installed_here=False,
            note="Excluded. ImmuneBuilder (BSD-3) is the structure-prediction replacement, and is not humanness.",
        ),
    ]


# --------------------------------------------------------------------------
# OASis identity
# --------------------------------------------------------------------------


def oasis_available() -> tuple[bool, str | None]:
    """Whether the packaged OASis peptide set can be loaded.

    ``find_spec`` rather than an import, because importing ``promb`` pulls in
    pandas and the import itself is the slow part of asking the question.
    """
    if find_spec("promb") is None:
        return False, "promb_not_installed"
    return True, None


@lru_cache(maxsize=1)
def _oasis_db() -> object:
    """The packaged human-OAS peptide set, loaded once per process.

    Loading reads a 40 MB gzip into a frozenset, so it is cached. The cache is
    on this private function rather than on the public one so a caller cannot
    be handed a stale record.
    """
    available, reason = oasis_available()
    if not available:
        raise HumannessUnavailableError(reason or "promb_unavailable")
    from promb import init_db

    return init_db("human-oas", verbose=False)


def peptide_windows(sequence: str, width: int = OASIS_PEPTIDE_LENGTH) -> list[str]:
    """Overlapping windows of one domain, dropping any with a non-standard residue."""
    text = "".join((sequence or "").split()).upper()
    found = [text[index : index + width] for index in range(max(0, len(text) - width + 1))]
    return [piece for piece in found if set(piece) <= _STANDARD_RESIDUES]


def oasis_identity(sequences: Mapping[str, str]) -> dict[str, object]:
    """OASis identity per named variable domain, at the 10 per cent subject threshold.

    The identity is ``matched / scored``: the fraction of the domain's 9-mers
    that appear in the human peptide set. A domain shorter than nine residues,
    or one whose windows all carry a non-standard residue, is reported with
    ``identity`` of ``None`` and a reason rather than as zero -- an identity of
    zero is a claim about peptides that were looked up, and none were.
    """
    database = _oasis_db()
    chains: list[dict[str, object]] = []
    for name, sequence in sequences.items():
        pieces = peptide_windows(sequence)
        if not pieces:
            chains.append({"chain": name, "identity": None, "blocker": "no_scorable_peptides"})
            continue
        matched = [piece for piece in pieces if database.contains(piece)]  # type: ignore[attr-defined]
        unmatched = [piece for piece in pieces if piece not in set(matched)]
        chains.append(
            {
                "chain": name,
                "identity": round(len(matched) / len(pieces), 5),
                "peptides_scored": len(pieces),
                "peptides_matched": len(matched),
                "peptides_unmatched": len(unmatched),
                "unmatched_examples": unmatched[:10],
            }
        )
    return {
        "method": OASIS_METHOD,
        "source": OASIS_SOURCE,
        "subject_threshold_percent": OASIS_SUBJECT_THRESHOLD_PERCENT,
        "peptide_length": OASIS_PEPTIDE_LENGTH,
        "scale": "fraction on 0-1, higher is more human",
        "chains": chains,
        "pass_cut": None,
        "pass_cut_reason": "biophi_reports_a_percentile_against_therapeutics_not_a_pass_mark",
    }


# --------------------------------------------------------------------------
# AbLang-2 antibody-likeness
# --------------------------------------------------------------------------


def ablang2_available() -> tuple[bool, str | None]:
    """Whether AbLang-2 is installed. The weights download is separate.

    ``find_spec`` rather than an import: importing ``ablang2`` imports torch,
    which costs about twenty seconds and is not what the question asked.
    """
    if find_spec("ablang2") is None:
        return False, "ablang2_not_installed"
    return True, None


@lru_cache(maxsize=1)
def _ablang2_model(device: str = "cpu", ncpu: int = 1) -> object:  # pragma: no cover - loads weights
    available, reason = ablang2_available()
    if not available:
        raise HumannessUnavailableError(reason or "ablang2_unavailable")
    import ablang2

    model = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=ncpu, device=device)
    model.freeze()
    return model


def ablang2_naturalness(pairs: Sequence[tuple[str, str, str]], *, device: str = "cpu") -> dict[str, object]:
    """Pseudo-log-likelihood per paired VH/VL, under AbLang-2.

    ``pairs`` is ``(name, heavy, light)``. AbLang-2's paired model takes the two
    chains together, which is what it was fine-tuned on; a single chain can be
    passed with the other left empty, and the record says which was done so a
    reader does not compare a paired score against an unpaired one.
    """
    if not pairs:
        raise ValueError("no_pairs")
    model = _ablang2_model(device=device)
    scores = model([[heavy or "", light or ""] for _name, heavy, light in pairs], mode="pseudo_log_likelihood")
    chains: list[dict[str, object]] = []
    for (name, heavy, light), score in zip(pairs, list(scores), strict=True):
        chains.append(
            {
                "chain": name,
                "score": round(float(score), 5),
                "input": "paired" if (heavy and light) else "single_chain",
            }
        )
    return {
        "method": ABLANG2_METHOD,
        "source": ABLANG2_SOURCE,
        "scale": "mean natural-log likelihood per residue, at most 0, higher is more expected",
        "measures": "antibody_likeness",
        "not_humanness_reason": ABLANG2_NOT_HUMANNESS,
        "chains": chains,
        "pass_cut": None,
        "pass_cut_reason": "no_published_pseudo_log_likelihood_cutoff",
    }


def humanness_report(
    sequences: Mapping[str, str],
    *,
    pairs: Sequence[tuple[str, str, str]] | None = None,
) -> dict[str, object]:
    """Both routes over the same design, each with its own blocker when absent.

    A backend that is missing produces a blocker inside its own section. The
    report never falls back from one to the other: an OASis identity and a
    language-model likelihood are different quantities and substituting one
    for the other would be reporting a number under the wrong name.
    """
    record: dict[str, object] = {"humanness": {}, "antibody_likeness": {}}
    try:
        record["humanness"] = oasis_identity(sequences)
    except HumannessUnavailableError as error:
        record["humanness"] = {"blocker": str(error), "method": OASIS_METHOD}
    if pairs:
        try:
            record["antibody_likeness"] = ablang2_naturalness(pairs)
        except HumannessUnavailableError as error:
            record["antibody_likeness"] = {"blocker": str(error), "method": ABLANG2_METHOD}
    else:
        record["antibody_likeness"] = {"blocker": "no_paired_chains_supplied", "method": ABLANG2_METHOD}
    return record
