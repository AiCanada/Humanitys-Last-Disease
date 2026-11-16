"""Write a finished design as the input an external folder takes.

The other half of :py:mod:`rfproteina.metrics.oracles.imported_refolds`. That
module files a fold once you have one; this one produces the request that gets
you it, for any folder driven by sequence rather than by a local checkpoint --
including a web service that has no API, where the submission is a person
pasting text.

Two things it is careful about, because both are easy to get wrong by hand:

**Every chain, not just the binder.** Refolding a binder on its own answers
whether the binder folds. It cannot answer whether the *complex* forms, which
is what the interface criteria are about -- a predicted TM-score for the chain
of interest, an interface predicted aligned error, an aligned binder RMSD. A
request that omits the target silently changes the question, so a design whose
structure holds more than one polymer chain exports all of them and says which
one is the design.

**The names match on the way back.** Each request is named for its design, so a
structure downloaded under that name (or that name with a tool's own suffix) is
matched to the right design by ``import_refolds`` without anyone keeping track.

This writes designed sequences to a file meant to leave the machine. That is
the point of it: nothing here uploads anything, and a request sits on disk
until someone sends it.

**Deliberately service-agnostic.** A FASTA is what any sequence-driven folder
takes, so nothing here knows one service's request format from another's. For
the hosted AlphaFold Server specifically -- its JSON dialect, its allowlists,
its token budget and the archive it returns -- use
:py:mod:`rfproteina.af3server`, which is where that knowledge lives and the
only place it lives.

What this module does own, and what :py:mod:`rfproteina.af3server.bundle` reads
from it rather than working out again, is where a design's files belong:
:py:func:`design_bases` and :py:func:`refold_status`. That is a fact about this
repository's run layout, not about any folder.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rfproteina.disease.screen import (
    _cif_text,
    _generated_cifs,
    _polymer_sequences,
    _sample_ids,
    designed_chain,
)

#: Residues one request may total across every chain. A sequence-driven service
#: bounds the work it accepts, and a request past the bound is refused at the
#: far end after the effort of sending it. Named here so a caller can raise or
#: lower it to match whatever they are sending to, rather than discovering the
#: limit from a rejection.
REQUEST_RESIDUE_LIMIT = 5000
#: Chains one request may hold. A generated complex is a handful; a file with
#: more than this is not one design's complex.
REQUEST_CHAIN_LIMIT = 16
#: Suffix of a written request.
REQUEST_SUFFIX = ".fasta"

#: Removed from here: an AlphaFold Server JSON exporter that duplicated
#: :py:mod:`rfproteina.af3server.request`. Two implementations of one
#: published dialect is two opinions about it, and they had already
#: diverged -- this one mapped ``polypeptide(d)`` onto ``proteinChain``,
#: which the server would fold as the L enantiomer of a different
#: molecule; it had no ligand or ion entity, so a small-molecule design
#: exported its protein and silently dropped the cofactor; and it counted
#: the 5,000 cap in residues where the server counts tokens, one per
#: ligand atom and ion included. ``rfd4-af3server`` is the way in now.
#: What this module keeps is what is not specific to that one service:
#: the FASTA request any sequence-driven folder takes, and the round-trip
#: bookkeeping (:py:func:`design_bases`, :py:func:`refold_status`) that
#: :py:mod:`rfproteina.af3server.bundle` now reads rather than reinvents.


@dataclass(frozen=True)
class RefoldRequest:
    """One design's request, or why there is none."""

    example_id: str
    blocker: str | None = None
    path: Path | None = None
    chains: tuple[str, ...] = ()
    residues: int = 0
    chain_of_interest: str | None = None

    @property
    def written(self) -> bool:
        return self.blocker is None


@dataclass
class ExportReport:
    """Every design the export was asked about."""

    written: list[RefoldRequest] = field(default_factory=list)
    refused: list[RefoldRequest] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """A record shaped for a run log: counts, and every reason by name."""
        return {
            "requests": len(self.written),
            "residues": sum(item.residues for item in self.written),
            "refused": [{"example_id": item.example_id, "blocker": item.blocker} for item in self.refused],
        }


def design_fasta(
    text: str,
    *,
    example_id: str,
    chain_of_interest: str | None = None,
) -> tuple[str, tuple[str, ...], int] | str:
    """A FASTA of every polymer chain in one generated structure.

    Returns the text, the chain ids in file order, and the total residue count
    -- or a blocker name when no request can be made from this structure.

    The header carries the design and the chain, and marks the chain of
    interest, so a person reading the request in a web form can tell which
    sequence is the design they are checking.
    """
    # `_polymer_sequences` omits a chain whose `entity_poly` row carries no
    # sequence, or carries the CIF nulls `.` and `?`. So a file whose chains
    # are all blank arrives here as nothing at all, and there is no separate
    # "the chains are there but empty" case to report: a guard for one was
    # unreachable, and the test that claimed to cover it could not build an
    # input that reached it either.
    named = _polymer_sequences(text)
    if not named:
        return "no_polymer_sequence"
    if len(named) > REQUEST_CHAIN_LIMIT:
        return "too_many_chains"
    residues = sum(len(seq) for seq in named.values())
    if residues > REQUEST_RESIDUE_LIMIT:
        return "over_residue_limit"

    marked = _strand_for(chain_of_interest, tuple(named))
    lines: list[str] = []
    for chain, sequence in named.items():
        role = "design" if marked is not None and chain == marked else "context"
        lines.append(f">{example_id}|chain_{chain}|{role}")
        lines.extend(_wrapped(sequence))
    return "\n".join(lines) + "\n", tuple(named), residues


def binder_fasta(
    text: str,
    *,
    example_id: str,
    chain_of_interest: str | None = None,
) -> tuple[str, str, int] | str:
    """A FASTA of the **binder alone**, or a blocker name.

    Returns the text, the strand it was taken from, and its residue count.

    A different question from :func:`design_fasta`, and the difference is the
    whole reason this is a separate function rather than a flag on it. That one
    exports every chain because the interface criteria -- a predicted TM-score
    for the chain of interest, an interface predicted aligned error, an aligned
    binder RMSD -- are about whether the *complex* forms, and a request that
    omits the target silently asks something else instead. This one asks that
    something else on purpose: whether the binder's own sequence folds to the
    structure it was designed as. Monomer self-consistency, not an interface.

    So nothing downstream of this may be read as evidence about the complex,
    and a caller that wants the complex question wants :func:`design_fasta`.

    The chain is reconciled through :func:`_strand_for`, because the metrics
    table records a chain instance id -- ``A_1`` -- while ``entity_poly`` names
    the strand ``A``. Matching the two literally finds nothing on every real
    run, and picking the first chain instead would export the target as though
    it were the design on any structure whose target comes first.

    A structure that names no chain of interest returns a blocker rather than a
    guess: with no mark there is nothing to say which chain is the binder, and
    the wrong sequence sent for folding is worse than none.
    """
    named = _polymer_sequences(text)
    if not named:
        return "no_polymer_sequence"
    marked = _strand_for(chain_of_interest, tuple(named))
    if marked is None:
        return "no_chain_of_interest"
    sequence = named[marked]
    if not sequence:
        return "no_polymer_sequence"
    if len(sequence) > REQUEST_RESIDUE_LIMIT:
        return "over_residue_limit"
    lines = [f">{example_id}|chain_{marked}|binder"]
    lines.extend(_wrapped(sequence))
    return "\n".join(lines) + "\n", marked, len(sequence)


def write_refold_requests(out_dir: Path, destination: Path) -> ExportReport:
    """Write one request per generated design under ``out_dir``.

    The chain of interest comes from the same place the screen reads it, so the
    request marks the chain the verdict will be taken on rather than a guess.
    A run whose metrics table names no chain still exports -- every chain is in
    the request either way -- and simply marks none of them as the design.
    """
    report = ExportReport()
    interest = designed_chain(out_dir) or None
    paths = _generated_cifs(out_dir)
    ids = _sample_ids(out_dir, paths)
    for index, path in enumerate(paths):
        example_id = ids[index]
        text = _cif_text(path)
        if text is None:
            report.refused.append(RefoldRequest(example_id, "structure_unreadable"))
            continue
        made = design_fasta(text, example_id=example_id, chain_of_interest=interest)
        if isinstance(made, str):
            report.refused.append(RefoldRequest(example_id, made))
            continue
        body, chains, residues = made
        target = destination / f"{example_id}{REQUEST_SUFFIX}"
        try:
            destination.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        except OSError:
            report.refused.append(RefoldRequest(example_id, "write_failed"))
            continue
        report.written.append(
            RefoldRequest(
                example_id,
                None,
                path=target,
                chains=chains,
                residues=residues,
                chain_of_interest=_strand_for(interest, chains),
            )
        )
    if report.written:
        _write_manifest(destination, report)
    return report


class _RefoldOracle(Protocol):
    """The one thing a status check needs of an oracle.

    Stated as a protocol rather than taken as ``OracleMetric`` because that is
    genuinely all this reads, and naming the whole class would import torch
    into a module that does nothing but compare two lists of names.
    """

    def already_refolded(self, out_dir: Path, example_id: str, save_id: str | None = ...) -> bool: ...


@dataclass(frozen=True)
class RoundTrip:
    """What has come back of what was asked for.

    The leg between the two is manual and rate-limited -- the AlphaFold Server
    caps jobs per day -- so a batch of designs comes back over several sittings.
    Without this the only way to tell what is still outstanding is to compare a
    directory listing against a manifest by eye, and the failure that invites
    is resubmitting designs that already came back while the ones that did not
    stay missing.
    """

    requested: tuple[str, ...] = ()
    refolded: tuple[str, ...] = ()
    outstanding: tuple[str, ...] = ()
    unrequested: tuple[str, ...] = ()
    #: Requested designs whose refold has nowhere to go, because the id in the
    #: manifest does not name the design's own folder. Also counted in
    #: ``outstanding``: nothing can come back for them until the id and the
    #: directory agree, so a batch holding one is not finished.
    unplaceable: tuple[str, ...] = ()
    blocker: str | None = None

    @property
    def complete(self) -> bool:
        """Every requested design has a fold. False when nothing was requested."""
        return self.blocker is None and bool(self.requested) and not self.outstanding

    def as_dict(self) -> dict[str, object]:
        if self.blocker is not None:
            return {"blocker": self.blocker}
        record: dict[str, object] = {
            "requested": len(self.requested),
            "refolded": len(self.refolded),
            "outstanding": list(self.outstanding),
        }
        if self.unrequested:
            record["unrequested"] = list(self.unrequested)
        if self.unplaceable:
            record["unplaceable"] = list(self.unplaceable)
        return record


def refold_status(oracle: _RefoldOracle, out_dir: Path, destination: Path, *, save_id: str | None = None) -> RoundTrip:
    """Compare the requests written to ``destination`` against the folds in ``out_dir``.

    A design counts as back when the configured ``oracle`` says it is -- the
    same predicate the metrics use -- rather than when a file is sitting
    somewhere, so this agrees with what a screen would do next.

    ``unrequested`` holds designs that are refolded but absent from the
    manifest. That is not an error: a run may hold folds from an earlier export
    or from a local oracle. It is reported because the alternative is a count
    that quietly disagrees with the directory.

    ``unplaceable`` holds requested designs whose refold has nowhere to go --
    see :py:func:`design_bases` for how an id and a directory come to disagree.
    They are outstanding and will stay outstanding, which is worth saying
    outright: reported as merely not back yet, they would be resubmitted at
    every sitting and come back no closer each time.
    """
    manifest = destination / "refold_requests.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except OSError:
        return RoundTrip(blocker="no_manifest")
    except (UnicodeError, ValueError):
        return RoundTrip(blocker="manifest_unreadable")
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), list):
        return RoundTrip(blocker="manifest_not_a_manifest")

    requested: list[str] = []
    for item in payload["requests"]:
        if isinstance(item, dict) and isinstance(item.get("example_id"), str):
            requested.append(item["example_id"])

    bases = design_bases(out_dir)
    back: list[str] = []
    outstanding: list[str] = []
    unplaceable: list[str] = []
    for example_id in requested:
        base = bases.get(example_id)
        if base is None:
            # No directory this design's refold could be filed in. Named and
            # counted as outstanding rather than probed at the run root, where
            # the answer would be "not back" for a reason that is not the one
            # reported, every time it is asked.
            unplaceable.append(example_id)
            outstanding.append(example_id)
            continue
        (back if _is_refolded(oracle, base, example_id, save_id) else outstanding).append(example_id)

    asked = set(requested)
    unrequested = [
        name for name in sorted(bases) if name not in asked and _is_refolded(oracle, bases[name], name, save_id)
    ]
    return RoundTrip(
        requested=tuple(requested),
        refolded=tuple(back),
        outstanding=tuple(outstanding),
        unrequested=tuple(unrequested),
        unplaceable=tuple(unplaceable),
    )


def _is_refolded(oracle: _RefoldOracle, out_dir: Path, example_id: str, save_id: str | None) -> bool:
    """``oracle.already_refolded``, treating a refusal as "not back".

    An oracle asked about an id it considers unusable raises rather than
    answering. That is the same outcome for this question as a fold that has
    not arrived, and it must not take the whole status report down with it.
    """
    try:
        return bool(oracle.already_refolded(out_dir, example_id, save_id))
    except (OSError, ValueError, TypeError):
        return False


def design_bases(out_dir: Path) -> dict[str, Path]:
    """Each design's id, and the directory its refold belongs under.

    :py:func:`design_output_dir` puts a design's files in ``base / id`` and
    refuses an id that does not land on a direct child of ``base``. A run nests
    its designs below ``inference_samples/``, and may nest them a further level
    when one run holds several batches, so there is no single base for a whole
    run -- each design has its own, and it is the parent of its own folder.

    Read from the same place the export reads, so an id here is exactly the id
    written into the manifest. Deriving it any other way -- listing the
    directories under the run, say -- produced ``inference_samples`` as a design
    name and reported every real design as never having come back.

    A design is mapped only when ``base / id`` is the folder its own structure
    was read from, because the two can disagree. :py:func:`_sample_ids` falls
    back to a flattened path id when the binder token repeats across batches, so
    a design at ``inference_samples/run_a/binder_0/`` is called
    ``run_a_binder_0`` while the parent of its folder is
    ``inference_samples/run_a``. Pairing those filed the refold in
    ``inference_samples/run_a/run_a_binder_0/`` -- a directory holding no design
    -- and ``already_refolded`` then read that same wrong place, so the import
    and the status check agreed with each other while both were wrong, and the
    design's refold metrics were silently absent from the screen.

    There is no base for such a design: ``design_output_dir`` cannot join a
    flattened id back onto a nested folder. It is left out here so that
    :py:func:`refold_status` names it rather than misfiling it.
    """
    paths = _generated_cifs(out_dir)
    ids = _sample_ids(out_dir, paths)
    return {
        example_id: path.parent.parent
        for example_id, path in zip(ids, paths, strict=True)
        if path.parent.name == example_id
    }


def _write_manifest(destination: Path, report: ExportReport) -> None:
    """A manifest beside the requests, so the round trip can be checked.

    Written because the return leg is manual: a person downloads files and puts
    them in a directory, and without a record of what was asked for there is
    nothing to compare the arrivals against.
    """
    payload = {
        "requests": [
            {
                "example_id": item.example_id,
                "file": item.path.name if item.path is not None else None,
                "chains": list(item.chains),
                "residues": item.residues,
                "chain_of_interest": item.chain_of_interest,
            }
            for item in report.written
        ],
        "residue_limit": REQUEST_RESIDUE_LIMIT,
    }
    try:
        (destination / "refold_requests.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError:
        # The requests themselves are written and usable. A missing manifest
        # costs the round-trip check, not the requests.
        return


#: One trailing instance number on a chain id, as the metrics table writes it.
_CHAIN_INSTANCE = re.compile(r"_\d+$")


def _strand_for(interest: str | None, chains: tuple[str, ...]) -> str | None:
    """The ``entity_poly`` strand id for the table's chain of interest.

    The metrics table records a chain *instance* id -- ``A_1`` for the first
    instance of chain ``A`` -- because the screen matches it against a parsed
    structure's ``chain_iid`` annotation, where that is the right spelling.
    ``entity_poly`` names the strand ``A``, so the two have to be reconciled
    here or every real run exports with its design chain unmarked.

    An exact match wins, so a table recording a bare strand id still works.
    Failing that, one trailing instance number is dropped and that is matched.
    A value matching neither leaves the chain unmarked rather than guessed: a
    request that marked the wrong chain as the design would be worse than one
    marking none, because the mark is what a reader trusts.
    """
    if interest is None:
        return None
    if interest in chains:
        return interest
    base = _CHAIN_INSTANCE.sub("", interest)
    return base if base and base in chains else None


def _wrapped(sequence: str, width: int = 60) -> list[str]:
    """FASTA body lines. Sixty is the conventional width and every reader takes it."""
    return [sequence[start : start + width] for start in range(0, len(sequence), width)] or [""]
