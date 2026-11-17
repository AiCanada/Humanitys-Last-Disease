"""What the buttons on the results page do.

The results page runs after the generation call has returned, so there is no
``args``, no output directory and no disease to consult -- only the structures
behind the viewers and, for each, the chain the run called the binder. Every
action here is answerable from exactly that, and every one returns a sentence
rather than raising: the run has already finished, and a button must not be
able to end it.

Only the binder is sent anywhere, which is a different question from the one
``refold_export`` asks. A refold request carries every chain because the
interface criteria are about whether the *complex* forms; this asks whether the
binder's own sequence folds to the shape it was designed as.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.limits import USER_AGENT
from rfproteina.disease.refold_export import binder_fasta, design_fasta
from rfproteina.disease.research import read_public_json
from rfproteina.disease.results_compare import (
    apply_transform,
    best_match,
    chain_labels,
    compare_complex,
)
from rfproteina.disease.results_retrieve import (
    TEXT_LIMIT,
    Retrieved,
    from_accession,
    from_path,
    from_text,
    newest_download,
)
from rfproteina.disease.screen import _polymer_sequences

#: What the AlphaFold 2 overlay says when this binder is not in the database.
#: For a de novo design that is the expected answer, so it reads as a finding.
NO_AF2_ENTRY = "No Alpha Fold 2 Database Entry Found, Possible De-NoVo Compound"
#: The same, for the whole complex. A complex is not an entry the database
#: holds at all -- it stores one chain per accession -- so this is expected.
NO_AF2_ENTRY_COMBINED = (
    "No Alpha Fold 2 Database Entry Found, Possible De-NoVo Compound. The database stores "
    "one chain per accession and holds no complexes, so a combined structure is not "
    "something it can return."
)
#: Retrieval outcomes that mean the same thing for the binder.
_NO_AF2_BLOCKERS = frozenset({"afdb_no_entry", "afdb_no_model_url"})

#: How recently a download must have been written to be taken as the answer to
#: this click. A stale archive from last week overlaid as though it were the
#: fold just asked for is the wrong answer delivered confidently, so the window
#: is narrow and a miss says so rather than reaching further back.
AUTO_WINDOW = 3600.0

#: The ``__action`` values the results page sends.
OUTPUT_FASTA = "results-fasta"
RETRIEVE_AF2 = "results-af2"
PREPARE_AF3 = "results-af3"
RETRIEVE_AF3 = "results-af3-import"
#: The same three steps for the whole complex, on their own tab.
COMBINED_FASTA = "results-combined-fasta"
PREPARE_AF3C = "results-af3c-prepare"
RETRIEVE_AF2C = "results-af2c"
RETRIEVE_AF3C = "results-af3c"
ACTIONS = (
    OUTPUT_FASTA,
    RETRIEVE_AF2,
    PREPARE_AF3,
    RETRIEVE_AF3,
    COMBINED_FASTA,
    PREPARE_AF3C,
    RETRIEVE_AF2C,
    RETRIEVE_AF3C,
)

StructureBytes = Callable[[str], bytes | None]


@dataclass(frozen=True)
class BinderSequence:
    """One design's binder sequence, or why it could not be read."""

    sample: str
    strand: str = ""
    sequence: str = ""
    fasta: str = ""
    blocker: str = ""

    @property
    def found(self) -> bool:
        return bool(self.sequence) and not self.blocker


def binder_sequences(
    catalog: Sequence[Mapping[str, object]],
    structure_bytes: StructureBytes,
) -> list[BinderSequence]:
    """The binder sequence for every design the results page is showing.

    The binder is named by the catalog's ``binder_chains``, which
    ``structure_trios`` filled from the same place the screen read its chain of
    interest. Taking the first chain in the file instead would export the
    *target* as the design on any structure whose target comes first, which is
    most of them.
    """
    found: list[BinderSequence] = []
    for item in catalog:
        sample = str(item.get("sample", "")) or "design"
        token = str(item.get("complex_token", ""))
        chains = item.get("binder_chains")
        binder_chains = tuple(str(name) for name in chains) if isinstance(chains, list | tuple) else ()
        if not token:
            found.append(BinderSequence(sample, blocker="structure_not_registered"))
            continue
        raw = structure_bytes(token)
        if raw is None:
            found.append(BinderSequence(sample, blocker="structure_unreadable"))
            continue
        text = raw.decode("utf-8", errors="replace")
        if not binder_chains:
            found.append(BinderSequence(sample, blocker="no_chain_of_interest"))
            continue
        made = binder_fasta(text, example_id=sample, chain_of_interest=binder_chains[0])
        if isinstance(made, str):
            found.append(BinderSequence(sample, blocker=made))
            continue
        body, strand, _residues = made
        sequences = _polymer_sequences(text)
        found.append(
            BinderSequence(sample, strand=strand, sequence=sequences.get(strand, ""), fasta=body)
        )
    return found


@dataclass(frozen=True)
class Done:
    """What one action produced: a sentence, a payload, and any overlay.

    ``payload`` is the part that was missing and the reason this exists. The
    first version of these buttons reported "Prepared a binder-only request"
    and produced nothing at all -- no file, no text on screen, nothing anyone
    could upload. A sentence claiming a thing was prepared, with the thing
    itself nowhere, is the button overstating itself, which is the one move a
    page of safety verdicts must not make.

    So the text comes back and the page shows it: selectable for copying,
    pasting and saving, and offered as a download under ``payload_name``.
    """

    sentence: str
    #: ``(which, text, detail)`` per overlay touched. Empty text clears one.
    overlays: tuple[tuple[str, str, str], ...] = ()
    #: The text the action produced, for the page to show and offer.
    payload: str = ""
    #: A file name for that text, used by the download link.
    payload_name: str = ""


def perform(
    action: str,
    catalog: Sequence[Mapping[str, object]],
    structure_bytes: StructureBytes,
    fields: Sequence[str] = (),
    directories: Sequence[Path] | None = None,
) -> Done:
    """Run one action and return the sentence the page should show.

    ``directories`` bounds where a download may be looked for. Injected so a
    test never reads the machine's real downloads folder and picks up whatever
    happens to be sitting in it.
    """
    if action in {COMBINED_FASTA, PREPARE_AF3C}:
        made = combined_sequences(catalog, structure_bytes)
        body = "".join(item.fasta for item in made if item.found)
        name = "combined.fasta" if action == COMBINED_FASTA else "combined_for_alphafold3.fasta"
        said = _combined_sentence(made)
        if action == PREPARE_AF3C and body:
            said += (
                " Give it to the AlphaFold Server yourself; nothing was sent from here."
            )
        return Done(said, payload=body, payload_name=name if body else "")
    if action in {RETRIEVE_AF2C, RETRIEVE_AF3C}:
        return _retrieve(action, catalog, structure_bytes, fields, directories)
    if action in {OUTPUT_FASTA, PREPARE_AF3}:
        found = binder_sequences(catalog, structure_bytes)
        body = "".join(item.fasta for item in found if item.found)
        name = "binder.fasta" if action == OUTPUT_FASTA else "binder_for_alphafold3.fasta"
        said = _fasta_sentence(found) if action == OUTPUT_FASTA else _af3_sentence(found)
        return Done(said, payload=body, payload_name=name if body else "")
    if action in {RETRIEVE_AF2, RETRIEVE_AF3}:
        return _retrieve(action, catalog, structure_bytes, fields, directories)
    # Not an error the run should notice: an unknown action is a page out of
    # step with this build, and the page comes back either way.
    return Done("That button is not one this version knows.")


def _retrieve(
    action: str,
    catalog: Sequence[Mapping[str, object]],
    structure_bytes: StructureBytes,
    fields: Sequence[str],
    directories: Sequence[Path] | None = None,
) -> Done:
    """Retrieve a model by whichever route the page supplied, then compare it.

    The routes are tried in the order that asks least of the machine: pasted
    text first, then a path, then an accession. A person who pasted a model has
    already done the retrieving, and asking the network again would be both
    slower and a chance to disagree with what they pasted.
    """
    which = {
        RETRIEVE_AF2: "af2",
        RETRIEVE_AF3: "af3",
        RETRIEVE_AF2C: "af2c",
        RETRIEVE_AF3C: "af3c",
    }[action]
    service = "af2" if which.startswith("af2") else "af3"
    combined = which.endswith("c")
    pasted = _field(fields, f"{which}_paste")
    path_text = _field(fields, f"{which}_path")
    accession = _field(fields, f"{which}_accession")

    if pasted:
        got = from_text(pasted, label="Pasted model")
    elif path_text:
        got = from_path(Path(path_text))
    elif accession:
        if service == "af3":
            return Done(
                "The AlphaFold Server is not keyed by accession and holds nothing to look up. "
                "Paste the model from its archive, or choose the archive file."
            )
        got = from_accession(accession, read_json=read_public_json, read_bytes=_read_bytes)
    else:
        # Nothing typed, so look for it. This is what closes the loop without
        # being told: after the download click the file is sitting in a
        # downloads folder under the name its service gave it.
        #
        # Matched per service. The server leaves ``fold_<job>.zip`` and the
        # database leaves ``AF-<accession>-F1-model_v6.cif``; searching for
        # either would let the newest server archive answer a database lookup
        # and be captioned as a database model of the target.
        guess = newest_download(
            within_seconds=AUTO_WINDOW, directories=directories, service=service
        )
        if guess is None:
            if service == "af2":
                message = NO_AF2_ENTRY_COMBINED if combined else NO_AF2_ENTRY
                return Done(
                    f"{message} Paste a model or open a downloaded one.",
                    overlays=((which, "", message),),
                )
            return Done(
                "Nothing to retrieve from, and no recent AlphaFold download found. "
                "Paste a model or give a file to open."
            )
        got = from_path(guess)
        if got.found:
            got = Retrieved(
                text=got.text,
                source=f"Found {guess.name} in your downloads",
                route=got.route,
                chains=got.chains,
                verified=False,
            )

    if not catalog:
        return Done(f"Retrieved a model, but this run shows no structure to compare it against. {got.caption()}")
    shape = overlay_for_complex if combined else overlay_for
    made = shape(catalog[0], got, which=which, structure_bytes=structure_bytes)
    if not made.found:
        return Done(made.detail, overlays=((which, "", made.detail),))
    return Done(made.detail, overlays=((which, made.text, made.detail),))


def _field(fields: Sequence[str], name: str) -> str:
    """One submitted value by name, from the panel's flat ``name=value`` list."""
    prefix = f"{name}="
    for entry in fields:
        if entry.startswith(prefix):
            return entry[len(prefix) :].strip()
    return ""


def _read_bytes(url: str) -> bytes:
    """Fetch a model file named by the database's own record.

    Narrow on purpose: the URL is the one the prediction record gave, not one
    assembled here, and nothing about this machine goes with the request beyond
    the project's own user agent.
    """
    from urllib.request import Request, urlopen

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        return bytes(response.read(TEXT_LIMIT + 1))


def _fasta_sentence(found: Sequence[BinderSequence]) -> str:
    read = [item for item in found if item.found]
    if not read:
        return "No binder sequence could be read. " + _why(found)
    where = ", ".join(f"{item.sample} (chain {item.strand}, {len(item.sequence)} residues)" for item in read[:4])
    more = f" and {len(read) - 4} more" if len(read) > 4 else ""
    tail = "" if len(read) == len(found) else " " + _why([item for item in found if not item.found])
    return (
        f"Binder FASTA for {where}{more}, below -- copy it or download it. "
        f"This is the binder alone, not the complex.{tail}"
    )


def _af2_sentence(found: Sequence[BinderSequence]) -> str:
    read = [item for item in found if item.found]
    if not read:
        return "No binder sequence could be read, so nothing could be looked up. " + _why(found)
    return (
        "The AlphaFold database holds precomputed predictions and accepts no sequence for "
        "folding, so nothing was sent. A designed binder has no entry there by construction: "
        "the only sequence-keyed lookup is an exact checksum match, which a design misses, and "
        "so does a 99%-identical variant. Retrieve a model for an accession this run resolved, "
        "or fold the binder with AlphaFold 3 instead."
    )


def _af3_sentence(found: Sequence[BinderSequence]) -> str:
    read = [item for item in found if item.found]
    if not read:
        return "No binder sequence could be read, so no request was prepared. " + _why(found)
    return (
        f"Binder sequence for {len(read)} design(s), below. Copy it, or download it, and give it "
        "to the AlphaFold Server yourself: it has no submission API and issues no credential, so "
        "nothing was sent from here. Bring the archive it gives you back to the AlphaFold 3 box "
        "below, or paste the model straight in. A binder-only fold answers whether the binder "
        "folds as designed, not whether the complex forms."
    )


#: What a reader is told when a design could not be exported, keyed by the
#: blocker the sequence helpers record.
#:
#: These are fragments, not sentences: they are shown after a sample name, as
#: "(design_007: its structure file could not be read)", so each one has to read
#: on from the colon. The board's sentences stand alone instead, which is why
#: this is its own table rather than an import of ``screen_board._BLOCKERS``:
#: the results page is a separate surface with its own voice, and the one name
#: the two share -- ``structure_unreadable`` -- wants a different wording here.
#:
#: Three of these come from ``refold_export.design_fasta`` and ``binder_fasta``,
#: which hand a blocker back as a bare ``return "name"`` from a function named
#: for neither convention. The vocabulary scan in
#: ``test_disease_board_vocabulary`` cannot see that shape, so it cannot demand
#: a word for them; they are listed there as unseen so a fourth cannot arrive
#: quietly, and they are spelled out here because a reader hitting one still
#: deserves a sentence.
_BLOCKER_TEXT = {
    "structure_not_registered": "this run never registered a structure for it",
    "structure_unreadable": "its structure file could not be read",
    "no_chain_of_interest": "no chain of interest is recorded, so the binder cannot be told from the target",
    "no_polymer_sequence": "its structure holds no polymer sequence",
    "too_many_chains": "it has more chains than one request can carry",
    "over_residue_limit": "it is longer than the request's residue limit",
    "no_af2_entry": "the AlphaFold database has no entry for it",
    "not_superposed": "the two structures were never superposed, so there is nothing to draw",
}


def _said(blocker: str) -> str:
    """The reader's words for one blocker.

    An unknown name falls back to itself with its underscores removed, which is
    what the board does: a missing word should read as prose with a gap in it,
    not as ``no_polymer_sequence`` with the punctuation of source code.
    """
    if not blocker:
        return "unknown"
    return _BLOCKER_TEXT.get(blocker, blocker.replace("_", " "))


def _why(found: Sequence[BinderSequence]) -> str:
    """Name the blockers, so a refusal says which designs and what stopped them."""
    blocked = [item for item in found if not item.found]
    if not blocked:
        return ""
    named = ", ".join(f"{item.sample}: {_said(item.blocker)}" for item in blocked[:4])
    more = f" and {len(blocked) - 4} more" if len(blocked) > 4 else ""
    return f"({named}{more})"

# ----------------------------------------------------------- retrieved models


@dataclass(frozen=True)
class Overlay:
    """One retrieved model placed against this run's own structure."""

    which: str
    text: str = ""
    detail: str = ""
    blocker: str = ""

    @property
    def found(self) -> bool:
        return bool(self.text) and not self.blocker


def overlay_for(
    item: Mapping[str, object],
    retrieved: Retrieved,
    *,
    which: str,
    structure_bytes: StructureBytes,
) -> Overlay:
    """Compare a retrieved model against this design, and word the result.

    The comparison is against the **combined** structure, which is what the run
    produced, and the chain it is measured on is chosen by measurement rather
    than by position: a retrieved complex names its chains in an order nobody
    promised, so :func:`best_match` superposes each candidate and keeps the one
    that actually fits.

    Which of this run's chains to compare against depends on the model. An
    Both overlays are about the **binder**, so both compare against the binder
    chain. A retrieved model that turns out to be some other protein does not
    become a large deviation: it is reported as no entry for this binder,
    which for a designed sequence is the expected answer and a useful one.
    """
    if not retrieved.found:
        if which == "af2" and retrieved.blocker in _NO_AF2_BLOCKERS:
            return Overlay(which, blocker="no_af2_entry", detail=NO_AF2_ENTRY)
        return Overlay(which, blocker=retrieved.blocker, detail=retrieved.caption())
    token = str(item.get("complex_token", ""))
    raw = structure_bytes(token) if token else None
    if raw is None:
        return Overlay(which, blocker="structure_unreadable", detail="This run's structure could not be read.")
    own = raw.decode("utf-8", errors="replace")
    chains = item.get("binder_chains")
    mine = tuple(str(name) for name in chains) if isinstance(chains, list | tuple) else ()
    if not mine:
        return Overlay(
            which,
            text=retrieved.text,
            detail=f"{retrieved.caption()} This run names no chain to compare it against.",
        )
    found = best_match(own, mine[0], retrieved.text, retrieved.chains)
    if which == "af2" and found.blocker == "no_common_sequence":
        # A model came back and it is not this binder. For a designed sequence
        # that is the expected result, and it is worth stating as a finding
        # rather than as a failure to retrieve something.
        return Overlay(which, blocker="no_af2_entry", detail=NO_AF2_ENTRY)
    if not found.measured or found.transform is None:
        # Not drawn. Without a superposition the two structures sit in
        # unrelated frames -- a predictor's frame has nothing to do with a
        # deposit's -- so the picture would be two objects in one box that
        # reads as a comparison and is not one. The sentence says why.
        return Overlay(which, blocker=found.blocker or "not_superposed",
                       detail=f"{found.sentence()} {retrieved.caption()}")
    rotation, translation = found.transform
    moved = apply_transform(retrieved.text, rotation, translation)
    return Overlay(which, text=moved, detail=f"{found.sentence()} {retrieved.caption()}")


# ------------------------------------------------------- the combined structure


@dataclass(frozen=True)
class CombinedSequence:
    """One design's whole complex as FASTA, or why it could not be read."""

    sample: str
    chains: tuple[str, ...] = ()
    residues: int = 0
    fasta: str = ""
    blocker: str = ""

    @property
    def found(self) -> bool:
        return bool(self.fasta) and not self.blocker


def combined_sequences(
    catalog: Sequence[Mapping[str, object]],
    structure_bytes: StructureBytes,
) -> list[CombinedSequence]:
    """Every chain of every design, which is what a complex request needs.

    ``design_fasta`` rather than ``binder_fasta``: this asks whether the
    complex forms, and a request that leaves the target out asks whether the
    binder folds instead. It marks which chain is the design, so a reader of
    the request can tell them apart.
    """
    found: list[CombinedSequence] = []
    for item in catalog:
        sample = str(item.get("sample", "")) or "design"
        token = str(item.get("complex_token", ""))
        chains = item.get("binder_chains")
        binder_chains = tuple(str(name) for name in chains) if isinstance(chains, list | tuple) else ()
        if not token:
            found.append(CombinedSequence(sample, blocker="structure_not_registered"))
            continue
        raw = structure_bytes(token)
        if raw is None:
            found.append(CombinedSequence(sample, blocker="structure_unreadable"))
            continue
        text = raw.decode("utf-8", errors="replace")
        made = design_fasta(
            text,
            example_id=sample,
            chain_of_interest=binder_chains[0] if binder_chains else None,
        )
        if isinstance(made, str):
            found.append(CombinedSequence(sample, blocker=made))
            continue
        body, named, residues = made
        found.append(CombinedSequence(sample, chains=named, residues=residues, fasta=body))
    return found


def _combined_sentence(found: Sequence[CombinedSequence]) -> str:
    read = [item for item in found if item.found]
    if not read:
        return "No combined sequence could be read. " + _why_combined(found)
    where = ", ".join(
        f"{item.sample} ({len(item.chains)} chains, {item.residues} residues)" for item in read[:4]
    )
    more = f" and {len(read) - 4} more" if len(read) > 4 else ""
    return (
        f"Combined FASTA for {where}{more}, below -- copy it or download it. Every chain is in "
        "it and the design is marked, so a fold of this asks whether the complex forms."
    )


def _why_combined(found: Sequence[CombinedSequence]) -> str:
    blocked = [item for item in found if not item.found]
    if not blocked:
        return ""
    named = ", ".join(f"{item.sample}: {_said(item.blocker)}" for item in blocked[:4])
    return f"({named})"


def overlay_for_complex(
    item: Mapping[str, object],
    retrieved: Retrieved,
    *,
    which: str,
    structure_bytes: StructureBytes,
) -> Overlay:
    """Compare a retrieved complex against the one this run produced."""
    if not retrieved.found:
        if which == "af2c" and retrieved.blocker in _NO_AF2_BLOCKERS:
            return Overlay(which, blocker="no_af2_entry", detail=NO_AF2_ENTRY_COMBINED)
        return Overlay(which, blocker=retrieved.blocker, detail=retrieved.caption())
    token = str(item.get("complex_token", ""))
    raw = structure_bytes(token) if token else None
    if raw is None:
        return Overlay(which, blocker="structure_unreadable", detail="This run's structure could not be read.")
    own = raw.decode("utf-8", errors="replace")

    binder = item.get("binder_chains")
    targets = item.get("target_chains")
    binder_chains = tuple(str(name) for name in binder) if isinstance(binder, list | tuple) else ()
    target_chains = tuple(str(name) for name in targets) if isinstance(targets, list | tuple) else ()
    mine = tuple(dict.fromkeys(target_chains + binder_chains))
    if not mine:
        mine = chain_labels(own)

    found = compare_complex(
        own,
        retrieved.text,
        own_chains=mine,
        model_chains=retrieved.chains,
        binder_chain=binder_chains[0] if binder_chains else "",
    )
    if which == "af2c" and found.blocker == "nothing_paired":
        return Overlay(which, blocker="no_af2_entry", detail=NO_AF2_ENTRY_COMBINED)
    if not found.measured or found.transform is None:
        return Overlay(
            which,
            blocker=found.blocker or "not_superposed",
            detail=f"{found.sentence()} {retrieved.caption()}",
        )
    rotation, translation = found.transform
    moved = apply_transform(retrieved.text, rotation, translation)
    return Overlay(which, text=moved, detail=f"{found.sentence()} {retrieved.caption()}")
