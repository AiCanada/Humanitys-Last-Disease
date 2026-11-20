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
#: The H.U.M.A.N  Protocol tab, which runs in two steps because the first one
#: is what fills the list the second one reads: propose the human proteins this
#: design already resembles, then compare the design against the ones ticked.
HUMAN_PROPOSE = "results-human-propose"
HUMAN_COMPARE = "results-human-compare"

#: The fields the protocol's buttons read. One spelling each, shared with the
#: page that posts them: ``ui`` imports these for its ``name=`` attributes
#: rather than writing the names a second time.
#:
#: Not a style preference. A field the page renames and the reader does not is
#: silent -- no error, no empty value, just an input that stops arriving -- and
#: ``flat_fields`` carries the docstring of the last time that happened here,
#: when every typed path, pasted model and accession went on the floor while
#: the page reported that no download could be found. Two lists checked against
#: each other would catch it; one list cannot have it.
HUMAN_PICK = "human_pick"
HUMAN_NAMES = "human_names"
HUMAN_TARGET = "human_target"
HUMAN_MODEL = "human_model"
HUMAN_MODEL_OF = "human_model_of"
HUMAN_FIELDS = (HUMAN_PICK, HUMAN_NAMES, HUMAN_TARGET, HUMAN_MODEL, HUMAN_MODEL_OF)

#: The Conformational Dynamics Analyst tab, which runs after generation and
#: screening because it reads a finished design. Two buttons: one asks the
#: generator for an ensemble, one draws the ensemble it produced.
DYNAMICS_RUN = "results-dynamics-run"
DYNAMICS_DRAW = "results-dynamics-draw"
#: Its fields, shared with the page that posts them for the reason the
#: protocol's are: a renamed input that nothing reads is silent.
DYNAMICS_CHAIN = "dynamics_chain"
DYNAMICS_TASK = "dynamics_task"
DYNAMICS_FRAMES = "dynamics_frames"
DYNAMICS_REPLICATES = "dynamics_replicates"
DYNAMICS_RMSF_MAX = "dynamics_rmsf_max"
DYNAMICS_MOBILE_MAX = "dynamics_mobile_max"
DYNAMICS_DRIFT_MAX = "dynamics_drift_max"
#: Opt-in, because the page takes seconds and the movie takes minutes to an
#: hour. A tick box rather than a second button: it is the same drawing.
DYNAMICS_MOVIE = "dynamics_movie"
DYNAMICS_FIELDS = (
    DYNAMICS_CHAIN,
    DYNAMICS_TASK,
    DYNAMICS_FRAMES,
    DYNAMICS_REPLICATES,
    DYNAMICS_RMSF_MAX,
    DYNAMICS_MOBILE_MAX,
    DYNAMICS_DRIFT_MAX,
    DYNAMICS_MOVIE,
)
ACTIONS = (
    OUTPUT_FASTA,
    RETRIEVE_AF2,
    PREPARE_AF3,
    RETRIEVE_AF3,
    COMBINED_FASTA,
    PREPARE_AF3C,
    RETRIEVE_AF2C,
    RETRIEVE_AF3C,
    HUMAN_PROPOSE,
    HUMAN_COMPARE,
    DYNAMICS_RUN,
    DYNAMICS_DRAW,
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
        found.append(BinderSequence(sample, strand=strand, sequence=sequences.get(strand, ""), fasta=body))
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
    #: A machine-readable record for an action whose result is a table rather
    #: than a block of text. The page draws it; this module does not render
    #: HTML, and the sentence above cannot carry a dozen numbers.
    record: Mapping[str, object] | None = None


def perform(
    action: str,
    catalog: Sequence[Mapping[str, object]],
    structure_bytes: StructureBytes,
    fields: Sequence[str] = (),
    directories: Sequence[Path] | None = None,
    held: Mapping[str, object] | None = None,
) -> Done:
    """Run one action and return the sentence the page should show.

    ``directories`` bounds where a download may be looked for. Injected so a
    test never reads the machine's real downloads folder and picks up whatever
    happens to be sitting in it.

    ``held`` is what earlier actions on this page left behind. Only the
    dynamics tab reads it, and only because its two buttons are one job split
    in half: the first produces an ensemble and the second draws it, so the
    second has to be told where the first put it.
    """
    if action in {COMBINED_FASTA, PREPARE_AF3C}:
        made = combined_sequences(catalog, structure_bytes)
        body = "".join(item.fasta for item in made if item.found)
        name = "combined.fasta" if action == COMBINED_FASTA else "combined_for_alphafold3.fasta"
        said = _combined_sentence(made)
        if action == PREPARE_AF3C and body:
            said += " Give it to the AlphaFold Server yourself; nothing was sent from here."
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
    if action in {HUMAN_PROPOSE, HUMAN_COMPARE}:
        return _human_protocol(action, catalog, structure_bytes, fields)
    if action in {DYNAMICS_RUN, DYNAMICS_DRAW}:
        earlier = (held or {}).get("dynamics")
        return _dynamics(
            action,
            catalog,
            structure_bytes,
            fields,
            earlier if isinstance(earlier, Mapping) else None,
        )
    # Not an error the run should notice: an unknown action is a page out of
    # step with this build, and the page comes back either way.
    return Done("That button is not one this version knows.")


def _all_fields(fields: Sequence[str], name: str) -> list[str]:
    """Every submitted value under one name, not just the first.

    A tick-box list posts its name once per box ticked, and :func:`_field`
    returns the first match: reading the protocol's selection through that
    compared the design against one protein and silently dropped the rest.
    """
    prefix = f"{name}="
    return [entry[len(prefix) :].strip() for entry in fields if entry.startswith(prefix)]


def _human_protocol(
    action: str,
    catalog: Sequence[Mapping[str, object]],
    structure_bytes: StructureBytes,
    fields: Sequence[str],
) -> Done:
    """The H.U.M.A.N  Protocol tab's two buttons.

    Imported here rather than at the top of the module so the results page
    keeps costing nothing for a reader who never opens this tab: the protocol
    reaches a reference proteome, an alignment binary and -- on the interface
    route -- the structure terms, none of which the FASTA buttons or the
    viewers need.

    The work directory is temporary and removed on the way out. What the scan
    writes is a FASTA of the designs, an alignment table and MMseqs2's own
    scratch, and none of it is a result: the record is the result, and it comes
    back in memory.
    """
    import tempfile

    from rfproteina.disease import human_protocol as protocol

    designs = [(item.sample, item.sequence) for item in binder_sequences(catalog, structure_bytes) if item.found]
    if not designs:
        # The name and its sentence both come from the protocol module. Written
        # as a literal here it would be checked against this module's own
        # vocabulary, which speaks in fragments for a different surface.
        record = protocol.blocked_record(protocol.NO_DESIGNS)
        return Done(str(record["reason"]), record=record)
    proteome = protocol.default_proteome()
    with tempfile.TemporaryDirectory(prefix="human-protocol-") as temporary:
        work = Path(temporary)
        if action == HUMAN_PROPOSE:
            return _human_proposal(designs, proteome, work, fields)
        return _human_comparison(designs, proteome, work, fields, catalog, structure_bytes)


def _human_proposal(
    designs: Sequence[tuple[str, str]],
    proteome: Path | None,
    work: Path,
    fields: Sequence[str],
) -> Done:
    """Step one: the human proteins these designs already resemble.

    This is what fills the tick-box list. It is derived from the alignment
    rather than read off a curated panel, because a panel of proteins to avoid
    would be this program's guess at what matters for somebody else's
    therapeutic, and what the alignment can say is which human proteins *this*
    design already looks like.
    """
    from rfproteina.disease import human_protocol as protocol

    scan = protocol.scan_designs(designs, proteome, work)
    if not scan.ran:
        said = protocol.sentence(scan.blocker)
        return Done(said, record={"candidates": [], "blocker": scan.blocker, "reason": said})
    # Resolved before it is excluded, because a reader may have named the
    # target by gene symbol and the alignment reports accessions.
    target = protocol.resolve(_field(fields, HUMAN_TARGET), proteome)
    proposed = protocol.enrich(
        protocol.propose_anti_targets(scan, exclude=(target,) if target else ()),
        proteome,
    )
    if not proposed:
        return Done(
            "No human protein aligned to these designs over the E-value cut, so there is "
            "nothing to propose. Name the proteins to avoid yourself.",
            record={"candidates": [], "blocker": None, "reason": ""},
        )
    said = (
        f"{len(proposed)} human protein{'' if len(proposed) == 1 else 's'} these designs already "
        "resemble, strongest first. Tick the ones this design must not bind, then compare. "
        "Resemblance is not binding: see the note on the tab."
    )
    return Done(said, record={"candidates": [item.as_record() for item in proposed], "blocker": None, "reason": ""})


def _human_comparison(
    designs: Sequence[tuple[str, str]],
    proteome: Path | None,
    work: Path,
    fields: Sequence[str],
    catalog: Sequence[Mapping[str, object]],
    structure_bytes: StructureBytes,
) -> Done:
    """Step two: the designs against the proteins that were chosen.

    The on-target side of the interface route is this run's own complex, taken
    from the first design the page is showing that has one. Which design it was
    is recorded, because an interface number with no design beside it cannot be
    acted on.
    """
    from rfproteina.disease import human_protocol as protocol

    names = [*_all_fields(fields, HUMAN_PICK), *_all_fields(fields, HUMAN_NAMES)]
    complex_text, binder_chain, sample = _first_complex(catalog, structure_bytes)
    record = protocol.protocol_record(
        designs,
        names=names,
        target=_field(fields, HUMAN_TARGET),
        proteome=proteome,
        work_dir=work,
        model_text=_field(fields, HUMAN_MODEL),
        model_of=_field(fields, HUMAN_MODEL_OF),
        complex_text=complex_text,
        binder_chain=binder_chain,
    )
    drawn = dict(record)
    # Only where that measurement was actually taken. The on-target interface
    # is measured only when there is an off-target one to compare it with, and
    # this line was being set on every comparison -- so the page said "the
    # on-target interface was measured on design_1" under a route that had
    # declined to measure anything at all.
    drawn["interface_design"] = sample if protocol.on_target_measured(record) else ""
    return Done(
        protocol.summary_line(record),
        payload="\n".join(protocol.readout_lines(record)),
        payload_name="human_protocol.txt",
        record=drawn,
    )


def _dynamics(
    action: str,
    catalog: Sequence[Mapping[str, object]],
    structure_bytes: StructureBytes,
    fields: Sequence[str],
    held: Mapping[str, object] | None = None,
) -> Done:
    """The Conformational Dynamics Analyst tab's two buttons.

    Imported here rather than at the top of the module for the reason the
    protocol is: a reader who never opens this tab should not pay for numpy
    geometry, a vendored generator and a viewer they are not using.

    Run asks the generator for an ensemble of ONE chain and measures it. Draw
    turns the ensemble it produced into a page, and a movie when the writer for
    one is installed. Neither reaches the network: everything the generator
    would otherwise fetch is checked for first and declined by name.
    """
    from rfproteina.disease import dynamics as analyst

    design_text, chain_hint, sample = _first_design(catalog, structure_bytes)
    if not design_text:
        # The name and its sentence both come from the analyst's module.
        # Written as a literal here it would be checked against this module's
        # vocabulary, which speaks in fragments for a different surface.
        record = analyst.blocked_record(analyst.NO_DESIGN)
        return Done(str(record["reason"]), record={"dynamics": record})
    if action == DYNAMICS_DRAW:
        return _dynamics_page(fields, held)

    chain = _field(fields, DYNAMICS_CHAIN) or chain_hint
    limits = analyst.DynamicsLimits(
        rmsf_max_a=_number(_field(fields, DYNAMICS_RMSF_MAX)),
        mobile_fraction_max=_number(_field(fields, DYNAMICS_MOBILE_MAX)),
        drift_max_a=_number(_field(fields, DYNAMICS_DRIFT_MAX)),
    )
    task = _field(fields, DYNAMICS_TASK) or analyst.FORWARD
    frames = int(_number(_field(fields, DYNAMICS_FRAMES)) or 10)
    replicates = int(_number(_field(fields, DYNAMICS_REPLICATES)) or 1)

    # Kept rather than written into a temporary directory, because the run and
    # the draw are two button presses: an ensemble that vanished with the
    # process left the second button with nothing to draw, which reads as a
    # design that does not move.
    kept = analyst.ensembles_root()
    where = (kept / (sample or "design")) if kept is not None else None

    def produce(text: str, which: str, mode: str) -> object:
        return analyst.generate_ensemble(text, which, mode, frames=frames, replicates=replicates, work_dir=where)

    record = analyst.analyse(
        design_text,
        chain=chain,
        task_mode=task,
        limits=limits,
        generate=produce,  # type: ignore[arg-type]
    )
    drawn = dict(record)
    drawn["design_sample"] = sample
    if where is not None and record.get("measured") is not None:
        drawn["ensemble_dir"] = str(where / "ensemble")
    return Done(
        analyst.summary_line(record),
        payload="\n".join(analyst.readout_lines(record)),
        payload_name="conformational_dynamics.txt",
        record={"dynamics": drawn},
    )


def _dynamics_page(fields: Sequence[str], held: Mapping[str, object] | None) -> Done:
    """Draw the ensemble the last run produced, and a movie when one is asked for.

    The viewer is the vendored one, which is what reads this generator's
    output. The generation and screening pages keep the viewers they already
    have; this draws ensembles and nothing else.

    It runs under this project's own interpreter rather than the generator's,
    because the viewer needs numpy and nothing else -- so a page can be drawn
    on a machine where the generator is not installed at all, from an ensemble
    somebody else produced.
    """
    from rfproteina.disease import dynamics as analyst

    where = (held or {}).get("ensemble_dir")
    if not isinstance(where, str) or not where:
        # Nothing to draw until a run has produced one. Said plainly rather
        # than drawing an empty page, which would read as a design that does
        # not move.
        record = analyst.blocked_record(analyst.NO_ENSEMBLE)
        said = f"Run the generator first: there is no ensemble to draw yet. {analyst.OUT_OF_DISTRIBUTION}"
        return Done(said, record={"dynamics": {**record, "drawn": None}})

    out = Path(where).parent / "page"
    drawn = analyst.draw_ensemble(
        where,
        out,
        chain=_field(fields, DYNAMICS_CHAIN),
        title=str((held or {}).get("design_sample") or ""),
        movie=_field(fields, DYNAMICS_MOVIE).strip().lower() in {"1", "on", "true", "yes"},
    )
    page = drawn.get("page")
    if not page:
        return Done(str(drawn.get("reason") or ""), record={"dynamics": {**dict(held or {}), "drawn": drawn}})
    said = f"Drew {drawn['conformers_drawn']} conformers. {analyst.OUT_OF_DISTRIBUTION}"
    if drawn.get("movie"):
        said = f"Drew {drawn['conformers_drawn']} conformers, with a movie. {analyst.OUT_OF_DISTRIBUTION}"
    elif drawn.get("blocker"):
        said = f"{said} {drawn.get('reason')}"
    return Done(said, record={"dynamics": {**dict(held or {}), "drawn": drawn}})


def _number(text: str) -> float | None:
    """A finite number a reader typed, or None. A blank is not a cut."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value == value and value not in {float("inf"), float("-inf")} else None


def _first_design(
    catalog: Sequence[Mapping[str, object]],
    structure_bytes: StructureBytes,
) -> tuple[str, str, str]:
    """``(text, binder_chain, sample)`` for the first design with a readable structure.

    The binder chain is handed over as the default because the generator takes
    one chain and the run recorded which one is the design. Reading the first
    chain in the file instead would analyse the target on most of this
    program's outputs.
    """
    for item in catalog:
        token = str(item.get("complex_token", ""))
        chains = item.get("binder_chains")
        binder_chains = tuple(str(name) for name in chains) if isinstance(chains, list | tuple) else ()
        if not token:
            continue
        raw = structure_bytes(token)
        if raw is None:
            continue
        text = raw.decode("utf-8", errors="replace")
        return text, (binder_chains[0] if binder_chains else ""), str(item.get("sample", ""))
    return "", "", ""


def _first_complex(
    catalog: Sequence[Mapping[str, object]],
    structure_bytes: StructureBytes,
) -> tuple[str, str, str]:
    """``(text, binder_chain, sample)`` for the first design with a readable complex."""
    for item in catalog:
        token = str(item.get("complex_token", ""))
        chains = item.get("binder_chains")
        binder_chains = tuple(str(name) for name in chains) if isinstance(chains, list | tuple) else ()
        if not token or not binder_chains:
            continue
        raw = structure_bytes(token)
        if raw is None:
            continue
        return raw.decode("utf-8", errors="replace"), binder_chains[0], str(item.get("sample", ""))
    return "", "", ""


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
        guess = newest_download(within_seconds=AUTO_WINDOW, directories=directories, service=service)
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
        return Overlay(
            which, blocker=found.blocker or "not_superposed", detail=f"{found.sentence()} {retrieved.caption()}"
        )
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
    where = ", ".join(f"{item.sample} ({len(item.chains)} chains, {item.residues} residues)" for item in read[:4])
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
