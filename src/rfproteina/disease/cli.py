"""Command line for the disease-binder flow.

``human`` confirms a disease, then lists deposited 3D structures for the
disease proteins, their pathway members, and interacting proteins, including
DNA or RNA present in those deposits. The choose-structures page scores each
design class by how many of those structures were returned, and each row's
program score is how many public databases returned a record. The user confirms
which structures to download. That page can sort the rows by program score and
include or exclude experimental and predicted structures. The returned list is
saved for comparison with the LLM search. The same page lists the models at the
bottom. The two search buttons ask those models. A question box sends its text
to the checked models and keeps their replies in the box. Grok Bot needs no API
key: Submit question pastes that box into the open Chief of Staff chat and puts
the reply back in the box. Download does not: it fetches
only the ticked structures from the structure archive already used for those
deposits, and it does not call a model. ``--research`` still asks the models
before download when that flag is set. When none were confirmed, structures the
models named are the generation inputs. API keys typed on the landing
page are saved encrypted for the next run. They are not written into the
command or the report.
It writes a requested-clear annotation and, for a protein
or peptide binder against a parsed protein chain, a condition spec that fixes
that chain. It then prints ``rfd4-generate`` commands. ``--run`` starts those
commands in parallel on this machine and on every named remote machine. Without
``--run``, generation is not started.

Starting ``human`` with no arguments, or passing ``--ui``, opens a browser
panel. Dropdowns and tick boxes set every option. Flags without ``--ui`` skip it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from http.client import HTTPException
from pathlib import Path

from rfproteina.disease.archive import save_generation
from rfproteina.disease.dispatch import (
    GenerationRequest,
    Retrieve,
    Runner,
    RunResult,
    Transfer,
    execute_generation,
    generation_requests,
    is_local_host,
    scp_retrieve,
    scp_transfer,
    subprocess_runner,
)
from rfproteina.disease.focus import BiologicalFocus, search_additions, target_text
from rfproteina.disease.follow_up import follow_up_commands
from rfproteina.disease.grok_bot import GROK_BOT, AskDesktop
from rfproteina.disease.jobs import write_job
from rfproteina.disease.logs import log, start_log, stop_log
from rfproteina.disease.lookup import NOT_FOUND_MESSAGE, Disease, DiseaseNotFoundError, confirm_disease, match_diseases
from rfproteina.disease.publish import (
    repository_relative,
    scrub_path_prefix,
    scrubbed_json,
    write_text_atomically,
)
from rfproteina.disease.refold_export import (
    design_bases,
    refold_status,
    write_refold_requests,
)
from rfproteina.disease.research import (
    Complete,
    LookupMissError,
    ReadJson,
    SourceHit,
    ask_checked_models,
    collect_record_scores,
    compare_search_options,
    explicit_structure_rows,
    format_missing_report,
    format_option_comparison,
    format_report,
    format_structure_ranking,
    generation_rows,
    llm_results,
    merge_found_by,
    missing_structures,
    rank_structures,
    read_public_json,
    research_targets,
    search_plan,
    write_report,
)
from rfproteina.disease.screen import (
    ScreenWalkTruncatedError,
    base_name,
    designed_chain,
    format_screen_lines,
    format_screen_lines_for_directory,
    generation_directories,
    job_file_for_directory,
    label_for,
    read_screen,
    sample_cifs,
    screen_generation,
)
from rfproteina.disease.shards import model_checkpoint, shard_for_hosts
from rfproteina.disease.sources import (
    MAX_ASSOCIATIONS,
    GetJson,
    PostJson,
    ReadValue,
    collect_structure_options,
    fetch_structure_cif,
    focus_targets,
    get_json,
    get_json_value,
    partner_symbols,
    pathway_uniprot_members,
    post_json,
    query_parts,
    reactome_pathway_list,
    search_diseases,
)
from rfproteina.disease.structure_view import StructureTrio, structure_trios
from rfproteina.disease.structures import (
    GROUPS,
    RecordScore,
    StructureOption,
    campaign_counts,
    download_selected,
    group_options,
    options_from_records,
    returned_structure_records,
    save_returned_structures,
    select_options,
)
from rfproteina.disease.tables import downloaded_proteome, sample_line, stage_for_run
from rfproteina.disease.target_liability import TargetLimits
from rfproteina.disease.trace import (
    OPEN_TARGETS_FOCUS,
    REACTOME_PATHWAY_LIST,
    SearchTrace,
    restamp_screening,
    search_trace,
    write_research_traces,
)
from rfproteina.disease.ui import (
    BINDER_KINDS,
    MAX_PANEL_PASSES,
    RECORDS_ACTION,
    Exchange,
    OptionPanel,
    RestartFreshError,
    ReturnToStartError,
    StepBackError,
    WorkerChoice,
    display_directory,
    wants_option_ui,
)
from rfproteina.metrics.oracles.imported_refolds import PathOracle, import_refolds

ReadLine = Callable[[str], str]
Fetch = Callable[[StructureOption], bytes]


class ChoiceRequiredError(Exception):
    """Several names matched, or the given choice was out of range.

    Distinct from a disease that was not found: the disease exists, and the
    caller only has to say which one.
    """


def _load_catalog(path: Path) -> list[Disease]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Disease(name=row["name"], disease_id=row["disease_id"]) for row in payload]


def _flat_options(grouped: dict[str, list[StructureOption]]) -> list[StructureOption]:
    return [row for group in GROUPS for row in grouped.get(group, [])]


def _append_found(
    records: list[StructureOption],
    grouped: dict[str, list[StructureOption]],
    flat: list[StructureOption],
    added: list[StructureOption],
) -> None:
    """Put structures the first search found after the last listed number.

    Rebuilding the list by group would drop a new protein into the protein
    block. A list that already ends at 239 would then show those new rows as
    36-44 and move every later number, and a ranking of the old numbers would
    no longer name the same structures. Appending leaves both where they were.
    """
    records.extend(added)
    flat.extend(added)
    for row in added:
        grouped[row.group].append(row)


def _print_structures(grouped: dict[str, list[StructureOption]]) -> list[StructureOption]:
    flat = _flat_options(grouped)
    print("You can initiate a binding design campaign against the following targets:")
    for name, count in campaign_counts(flat):
        print(f"  {name}: {count} structures")
    number = 1
    for group in GROUPS:
        rows = grouped.get(group, [])
        print(f"{group}:")
        if not rows:
            print("  (none)")
            continue
        for row in rows:
            if row.predicted:
                mark = " predicted"
            elif not row.methodology_known:
                mark = " method unknown"
            else:
                mark = ""
            print(f"  {number}. {row.entry_id} chain {row.chain} {row.accession} {row.title}{mark}")
            number += 1
    return flat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="human")
    parser.add_argument("--catalog", type=Path, help="Local disease catalog JSON. Skips the disease search.")
    parser.add_argument("--disease", help="Disease name. Prompted when omitted.")
    parser.add_argument("--accept-only", action="store_true", help="Confirm the single matched disease.")
    parser.add_argument("--choice", type=int, help="1-based index when several similar names match.")
    parser.add_argument("--structures-json", type=Path, help="Local structure records. Skips the structure search.")
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Download every structure known to be experimental. Predicted rows, and rows whose "
            "determination method could not be read, stay unselected until named with --pick."
        ),
    )
    parser.add_argument("--pick", help="Comma-separated list numbers, including predicted rows.")
    parser.add_argument(
        "--binder-kind",
        default="protein",
        choices=BINDER_KINDS,
        help="Binder kind for the requested-clear annotation.",
    )
    parser.add_argument(
        "--confirm-inference",
        action="store_true",
        help="Write jobs and print commands without a prompt.",
    )
    parser.add_argument(
        "--hosts",
        help=(
            "Comma-separated machines. local runs on this machine. Any other name is a remote "
            "machine reached by ssh. There is no limit on how many names you pass."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the generation commands in parallel on this machine and on the named remote machines.",
    )
    parser.add_argument(
        "--local-workers",
        type=int,
        help="Parallel generation processes on this machine. Omit it to start every local request at once.",
    )
    parser.add_argument(
        "--remote-workers",
        type=int,
        help="Parallel generation requests across all remote machines at once. Omit it to start every "
        "remote request at once. A name repeated in --hosts is a second worker on that machine, which "
        "is how to say how many run on one of them.",
    )
    parser.add_argument(
        "--remote-dir",
        default="disease_runs",
        help="Directory on each remote machine for inputs and designs. No limit is applied to how many "
        "machines are named; --remote-workers bounds how many of their requests run at once.",
    )
    parser.add_argument(
        "--modified-ckpt",
        type=Path,
        help="Finished disease checkpoint. Until that file exists, commands use the original release model.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("disease_runs"))
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Where structure files are downloaded. Defaults to --out-dir.",
    )
    parser.add_argument(
        "--bar",
        help=(
            "Which named set of axes must all be clear. Defaults to the kind's "
            "own bar: sequence for a protein or peptide, coverage for a ligand, "
            "oligo for a nucleic binder. A protein or peptide also has plasma "
            "and all; a ligand also has strict, plasma and all; a nucleic "
            "binder also has all. An unknown bar is refused before anything is "
            "measured."
        ),
    )
    parser.add_argument(
        "--proteome",
        type=Path,
        help=(
            "Protein FASTA the off-target 9-mer screen reads. Defaults to "
            "$HUMAN_PROTEOME. Without either, that axis stays untested."
        ),
    )
    parser.add_argument(
        "--follow-up",
        action="store_true",
        help=(
            "After each screen, write and print the generation commands the "
            "screen selects for the next round. Nothing is started."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--screen",
        type=Path,
        metavar="DIR",
        help=(
            "Re-screen a finished generation directory, or every one beneath "
            "it, reading whatever liability tables are now present. Nothing is "
            "generated, downloaded or started."
        ),
    )
    parser.add_argument(
        "--stage-tables",
        action="store_true",
        help=(
            "Convert the downloaded reference tables into each run directory "
            "before screening it, narrowed to that run's own target and this "
            "disease. Without this the screen finds no fold table even on a "
            "machine holding every download, because the two live in different "
            "directories under different names. Also points the off-target axis "
            "at the downloaded human proteome unless --proteome names one. "
            "Reads only what is already downloaded; fetch with "
            "scripts/fetch_screening_tables.py."
        ),
    )
    parser.add_argument(
        "--lookup-targets",
        action="store_true",
        help=(
            "Look the target up in UniProt, the Human Protein Atlas and STRING "
            "to measure the paralog and plasma axes. Without this they report "
            "network_not_permitted. --plasma-abundance-max and --plasma-hub-max "
            "only do anything together with this."
        ),
    )
    parser.add_argument(
        "--plasma-abundance-max",
        type=float,
        metavar="PG_PER_L",
        help=(
            "Blood concentration at or above which a target counts as an "
            "antigen sink. No published number sets this, so without it the "
            "axis reports the concentration and stays untested."
        ),
    )
    parser.add_argument(
        "--plasma-hub-max",
        type=int,
        metavar="PARTNERS",
        help=(
            "High-confidence interaction partners above which a target counts "
            "as a hub. Unset leaves the degree reported and untested."
        ),
    )
    parser.add_argument(
        "--paralog-identity-max",
        type=float,
        metavar="PERCENT",
        help=(
            "Percent sequence identity at or above which a human paralog counts "
            "as a cross-reactivity liability. Measured by asking Ensembl Compara "
            "for the target's paralogs and keeping the closest one's identity, "
            "which is what UniRef50 membership cannot give: KRAS is a UniRef50 "
            "singleton while HRAS is 85.6 percent identical to it. Unset leaves "
            "the identity reported and untested, because no published "
            "cross-reactivity cut exists to judge it against. Needs "
            "--lookup-targets."
        ),
    )
    parser.add_argument(
        "--plasma-fold-max",
        type=float,
        metavar="FOLD",
        help=(
            "Disease-versus-healthy plasma fold change, by magnitude, at or "
            "above which the target counts as a liability. Needs a "
            "plasma_fold table in the output directory naming its study."
        ),
    )
    parser.add_argument(
        "--research",
        action="append",
        metavar="PROVIDER",
        help=(
            "After the program search options are confirmed, score each protein or gene by how many of "
            "UniProt, NCBI, STRING, IntAct, the Human Protein Atlas, PRIDE, and "
            "PeptideAtlas returned a record, cite the Therapeutic Target Database, "
            "ask each provider for new research, methods, and ideas, and compare the "
            "ratings they give each target. The model names deposited PDB "
            "structures from the list. Those names are compared with the program search. "
            "Confirmed program options are downloaded. When none were confirmed, the "
            "named structures are the generation inputs. "
            "Repeat the flag, pass a comma-separated list, or pass all for every key. "
            "Names are grok, claude, openai, nvidia, and openrouter. "
            "A key typed on the landing page is saved encrypted. Otherwise the "
            "provider environment variable is used. The landing page chooses the "
            "model for each key. Otherwise RFD4_GROK_MODEL, RFD4_CLAUDE_MODEL, "
            "RFD4_OPENAI_MODEL, RFD4_NVIDIA_MODEL, or RFD4_OPENROUTER_MODEL is used. "
            "Pass --models to ask every listed model or named ones."
        ),
    )
    parser.add_argument(
        "--models",
        action="append",
        metavar="MODEL",
        help=(
            "Which listed models --research asks. Pass all for every listed model "
            "of the selected providers, or provider|model-id for one model, for "
            "example nvidia|z-ai/glm-5.3-flash. Repeat the flag or pass a "
            "comma-separated list. Omit it to ask the one model chosen for each "
            "selected provider. With no --research, the named models choose the "
            "providers."
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        metavar="TARGET",
        help=(
            "Send every listed structure of these ranked targets to generation. Repeat "
            "the flag or pass a comma-separated list of accessions, gene symbols, or "
            "rank numbers from the search. Pass all for every ranked target. Needs "
            "--research, and adds to any --all or --pick selection."
        ),
    )
    parser.add_argument(
        "--target-structure",
        type=Path,
        help=(
            "A custom .pdb or .cif target to bind to, listed with the structures the "
            "disease search returns. The panel uploads one from its first page."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help=(
            "Where to append the run log. Defaults to Humanitys-Last-Disease.log beside the key "
            "store, or HUMANITYS_LAST_DISEASE_LOG when that is set. Every step and every error is "
            "recorded. API keys are not."
        ),
    )
    mode.add_argument(
        "--ui",
        action="store_true",
        help="Open the browser panel. Dropdowns and tick boxes set every option.",
    )
    mode.add_argument(
        "--refold",
        type=Path,
        metavar="DIR",
        help=(
            "Prepare a finished generation directory for folding somewhere "
            "else, and file the results when they come back. Writes one FASTA "
            "per design, then reports which designs are still outstanding. "
            "For the hosted AlphaFold Server specifically, rfd4-af3server "
            "prepare writes its own upload format. Nothing is generated, "
            "downloaded or uploaded."
        ),
    )
    parser.add_argument(
        "--refold-dir",
        type=Path,
        metavar="DIR",
        help=(
            "Where the refold requests are written and read back from. "
            "Defaults to refold_requests inside the --refold directory."
        ),
    )
    parser.add_argument(
        "--refold-import",
        type=Path,
        metavar="DIR",
        help=(
            "Instead of writing requests, file the folds in this directory "
            "against their designs. Subdirectories are searched, so an "
            "unpacked download can be passed as it is. Needs --refold."
        ),
    )
    parser.add_argument(
        "--refold-oracle",
        default="af3",
        metavar="NAME",
        help=(
            "The oracle name the refolds are filed under. It appears in every "
            "filename, so it has to match the oracle the metrics are "
            "configured with. Defaults to af3."
        ),
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    post: PostJson = post_json,
    get: GetJson = get_json,
    read_value: ReadValue | None = None,
    fetch: Fetch | None = None,
    read_line: ReadLine = input,
    runner: Runner | None = None,
    transfer: Transfer | None = None,
    retrieve: Retrieve | None = None,
    exchange: Exchange | None = None,
    read_json: ReadJson | None = None,
    complete: Complete | None = None,
    ask_desktop: AskDesktop | None = None,
) -> int:
    # The log opens before the panel, so a failure in the very first page is in it.
    written = start_log(_log_override(argv))
    if written is not None:
        print(f"Run log: {written}")
    log().info("human started")
    panel = OptionPanel(exchange)
    # Going back restarts the run, and the structure lookup is nearly all of what a
    # restart costs. The records it read outlive the restart, so Back no longer pays
    # for them twice. It is held here and not in a module global, so one run cannot
    # hand stale records to the next.
    entry_records: dict[str, dict] = {}
    try:
        # Going back cannot unwind a stack of blocking page calls, so the run restarts
        # and the panel replays the earlier answers until it reaches the page asked for.
        # Every page is shown before anything is downloaded, so a restart repeats the
        # lookups and nothing else.
        for _pass in range(MAX_PANEL_PASSES):
            try:
                code = _run(
                    argv,
                    panel=panel,
                    post=post,
                    get=get,
                    read_value=read_value,
                    fetch=fetch,
                    read_line=read_line,
                    runner=runner,
                    transfer=transfer,
                    retrieve=retrieve,
                    read_json=read_json,
                    complete=complete,
                    ask_desktop=ask_desktop,
                    entry_records=entry_records,
                )
            except StepBackError as back:
                log().info("panel went back to step %s", back.target)
                panel.rewind(back.target)
                continue
            except ReturnToStartError:
                log().info("panel returned to the start page")
                panel.prepare_saved_start()
                continue
            except RestartFreshError:
                log().info("panel restarted")
                entry_records.clear()
                panel.prepare_fresh_start()
                continue
            if panel.browser_open():
                choice = panel.finish_choice()
                if choice == "start":
                    log().info("panel returned to the start page")
                    panel.prepare_saved_start()
                    continue
                if choice == "fresh":
                    log().info("panel restarted")
                    entry_records.clear()
                    panel.prepare_fresh_start()
                    continue
            log().info("human finished with exit %s", code)
            return code
        _fail(f"The panel went back {MAX_PANEL_PASSES} times without finishing.")
        return 2
    except KeyboardInterrupt:
        log().warning("human was interrupted")
        print("Stopped.", file=sys.stderr)
        return 130
    except Exception:
        # Anything unforeseen is recorded with its traceback and named in one line,
        # rather than printed as a traceback the next run cannot look back at.
        log().exception("human stopped on an unexpected error")
        print("human stopped on an unexpected error.", file=sys.stderr)
        if written is not None:
            print(f"The traceback is in {written}.", file=sys.stderr)
        return 2
    finally:
        panel.close()
        stop_log()


def _log_override(argv: list[str] | None) -> Path | None:
    """Read ``--log-file`` before the real parse, so an early failure is still logged."""
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--log-file", type=Path)
    try:
        known, _rest = early.parse_known_args(list(argv) if argv is not None else sys.argv[1:])
    except SystemExit:
        return None
    return known.log_file


def _fail(message: str) -> None:
    """Report a failure on stderr and record the same line in the run log."""
    log().error(message)
    print(message, file=sys.stderr)


def _run(
    argv: list[str] | None,
    *,
    panel: OptionPanel,
    post: PostJson,
    get: GetJson,
    read_value: ReadValue | None,
    fetch: Fetch | None,
    read_line: ReadLine,
    runner: Runner | None,
    transfer: Transfer | None,
    retrieve: Retrieve | None,
    read_json: ReadJson | None,
    complete: Complete | None,
    ask_desktop: AskDesktop | None,
    entry_records: dict[str, dict] | None = None,
) -> int:
    incoming = list(sys.argv[1:] if argv is None else argv)
    if wants_option_ui(incoming):
        try:
            incoming = panel.collect(incoming, build_parser())
        except (OSError, ValueError) as exc:
            # OSError as well as ValueError, matching the probe below: drawing
            # the page stats the paths in the output field, and a pasted path
            # whose component is too long raises ENAMETOOLONG, which is neither
            # a ValueError nor something the panel can show as a notice.
            print(str(exc), file=sys.stderr)
            return 2
    selected = build_parser().parse_args(incoming)
    if selected.screen is not None:
        # Screening reads a finished run and writes its verdict. None of the
        # pipeline below applies: no disease to confirm, nothing to search for,
        # download, plan or dispatch.
        return _screen_only(selected, incoming, read_json=read_json)

    if selected.refold is not None:
        # Preparing a fold request and filing the answer both read a finished
        # run and write beside it. As with screening, none of the pipeline
        # below applies.
        return _refold_only(selected)

    chosen_instead: list[Disease] = []

    def confirm_one(disease: Disease) -> bool:
        if selected.accept_only:
            return True
        if selected.ui:
            # ``catalog`` is the list loaded for this query, just above the
            # call, and ``query`` is what the user typed. The menu filters the
            # catalog down to the names that match it, so an ontology search's
            # neighbours do not appear under a Matching names heading.
            chosen = panel.confirm_disease(disease, catalog, query)
            if chosen is None:
                return False
            if chosen.disease_id != disease.disease_id:
                chosen_instead.append(chosen)
            return True
        answer = read_line(f"Confirm {disease.name} ({disease.disease_id})? [y/N]: ")
        return answer.strip().casefold() in {"y", "yes"}

    def choose_one(hits: list[Disease]) -> int:
        if selected.ui and selected.choice is None:
            try:
                return panel.choose_disease(hits)
            except ValueError as exc:
                raise ChoiceRequiredError(str(exc)) from exc
        if selected.choice is None:
            for index, disease in enumerate(hits, start=1):
                print(f"{index}. {disease.name} ({disease.disease_id})")
            raise ChoiceRequiredError(f"{len(hits)} similar names matched. Re-run with --choice 1..{len(hits)}.")
        index = selected.choice - 1
        if not 0 <= index < len(hits):
            raise ChoiceRequiredError(f"--choice {selected.choice} is outside 1..{len(hits)}.")
        return index

    while True:
        query = selected.disease if selected.disease is not None else read_line("Disease name: ")
        if selected.catalog is None:
            try:
                catalog = search_diseases(query, post=post)
            except (OSError, ValueError) as exc:
                print(f"Disease lookup failed: {exc}", file=sys.stderr)
                return 2
        else:
            catalog = _load_catalog(selected.catalog)
        if not match_diseases(query, catalog, ranked=selected.catalog is None):
            print(NOT_FOUND_MESSAGE)
            if selected.ui:
                incoming = panel.replace_disease(selected, NOT_FOUND_MESSAGE)
                selected = build_parser().parse_args(incoming)
            else:
                selected.disease = None
            continue
        try:
            disease = confirm_disease(
                query,
                catalog,
                confirm_one=confirm_one,
                choose_one=choose_one,
                ranked=selected.catalog is None,
            )
        except ChoiceRequiredError as exc:
            _fail(str(exc))
            return 2
        except DiseaseNotFoundError as exc:
            print(str(exc))
            return 1
        break
    if chosen_instead:
        disease = chosen_instead[-1]
    args = selected

    print(f"Confirmed: {disease.name} ({disease.disease_id})")
    log().info("confirmed disease %s (%s)", disease.name, disease.disease_id)
    # ``--accept-only`` already skipped the disease page. It skips this page too.
    # A normal panel confirms the disease first, then chooses the biology.
    # Reactome and STRING return a JSON list. ``get`` accepts an object only.
    reader = read_value or get_json_value
    focus = (
        None
        if not args.ui or args.accept_only
        else _biological_focus(panel, disease, args, post=post, read_value=reader)
    )
    extra_accessions: list[str] = []
    extra_symbols: list[str] = []
    if focus is not None:
        extra_accessions, extra_symbols = search_additions(focus)
        pathway = focus.pathway_name or ("any" if focus.pathway_id in {"", "any"} else focus.pathway_id)
        target = target_text(focus) or "none"
        print(f"Biological focus: pathway {pathway}, mechanism {focus.mechanism}, target {target}.")
        log().info(
            "biological focus pathway %s mechanism %s target named %s",
            focus.pathway_id,
            focus.mechanism,
            bool(focus.target.strip()),
        )
    try:
        records = _load_records(
            args.structures_json,
            disease.disease_id,
            post=post,
            get=get,
            records=entry_records,
            extra_accessions=extra_accessions,
            extra_symbols=extra_symbols,
            read_value=reader,
        )
    except (OSError, ValueError) as exc:
        _fail(f"Structure lookup failed: {exc}")
        return 2
    if not records:
        print(f"No atomic structures found for {disease.name}.")
        return 0

    if args.target_structure is not None:
        try:
            uploaded, kept = _uploaded_target(args.target_structure, args.out_dir)
        except (OSError, ValueError) as exc:
            _fail(f"The uploaded target could not be read: {exc}")
            return 2
        records = [uploaded, *records]
        print(f"Custom target: {kept}")
        log().info("custom target %s", kept)
    grouped = group_options(records)
    flat = _print_structures(grouped)
    # Confirm the program's own search on the structure page. That page also
    # holds the model ticks. Flags that already name rows skip the page.
    collected: dict[str, dict[str, SourceHit]] = {}
    page_providers: list[str] = []
    score_map: dict[str, RecordScore] = {}

    def _on_lookup(done: int, total: int, label: str) -> None:
        # Asking the public databases is a read. Calling it a download on the page
        # before any structure is chosen said the opposite of what was happening.
        panel.note_progress(done, total, label, RECORDS_ACTION)

    def _keep_returned(rows: list[StructureOption], scores: dict[str, RecordScore] | None) -> None:
        path = save_returned_structures(args.out_dir, rows, scores)
        print(f"Saved {len(rows)} returned structure(s) to {path}.")
        log().info("saved %s returned structure(s) to %s", len(rows), path)

    ranking: list[dict[str, object]] | None = None
    # Which model named each structure, kept across passes. Keyed by entry id and
    # chain. New rows are appended, so an earlier number still names the same row.
    found_by: dict[str, list[list[str]]] = {}
    notice = ""
    # The question box is both the message and the reply history. Each ask appends.
    transcript = panel.posted_question()
    try:
        program_selected = _choose(grouped, flat, take_all=args.all, picks=args.pick)
        # The page stays up after a search or a question. The first search adds
        # structures, the second ranks them, and a question sends the box to the
        # checked models and keeps their replies in that box. Download leaves the
        # loop and does not call a model. Each pass re-reads the record scores
        # because the first search can lengthen the list.
        for _pass in range(MAX_PANEL_PASSES):
            if program_selected or not args.ui:
                break
            # Each row's program score is the public-database record count. That needs
            # no API key, and a later search reuses what is collected. Only rows that
            # have no score yet are looked up, so a pass after the first search costs
            # the structures it added and nothing more.
            pending = [row for row in flat if row.accession.strip().casefold() not in score_map]
            if pending:
                more, scored = collect_record_scores(pending, read_json, on_progress=_on_lookup)
                collected.update(more)
                score_map.update(scored)
                panel.clear_download()
                log().info("collected record scores for %s more target(s)", len(scored))
            _keep_returned(flat, score_map)
            results = llm_results(flat, found=found_by, ranking={"ranked": ranking or []})
            action, take_all, picks, page_providers = panel.choose_structures(
                flat,
                score_map,
                ranking=ranking,
                results=results,
                notice=notice,
                transcript=transcript,
                focus=focus,
            )
            notice = ""
            # A ticked earlier reply is part of the message. The typed question
            # stays at the end. Other buttons leave the box as it was posted.
            transcript = panel.question_for_models() if action == "ask" else panel.posted_question()
            if action == "ask":
                if not transcript.strip():
                    notice = "Type a question for the checked models."
                    continue
                if not page_providers:
                    notice = "Tick a model for the question."
                    continue
                try:
                    transcript = ask_checked_models(
                        transcript,
                        providers=page_providers,
                        complete=complete,
                        api_keys=panel.api_keys(),
                        chat_models=panel.models(),
                        search_models=panel.research_models(),
                        key_slots=panel.key_slots(),
                        ask_desktop=ask_desktop,
                    )
                except (OSError, ValueError) as exc:
                    notice = str(exc)
                    continue
                plan = panel.research_models()
                asked = sum(len(models) for models in plan.values()) if plan else len(page_providers)
                if plan and GROK_BOT in page_providers:
                    asked += 1
                log().info("asked %s checked model provider(s)", asked)
                continue
            if action in {"missing", "rank"} and not page_providers:
                notice = "Tick a model for the search."
                continue
            if action == "missing":

                def verify_named(url: str) -> object:
                    # An RCSB entry is a JSON object. An AlphaFold prediction record is a
                    # JSON list, which get_json does not accept.
                    if url.startswith("https://alphafold.ebi.ac.uk/"):
                        return read_public_json(url)
                    return get(url)

                added, found_report = missing_structures(
                    disease,
                    flat,
                    providers=page_providers,
                    verify=verify_named,
                    read_json=read_json,
                    complete=complete,
                    api_keys=panel.api_keys(),
                    chat_models=panel.models(),
                    search_models=panel.research_models(),
                    key_slots=panel.key_slots(),
                )
                print(format_missing_report(found_report))
                found_by = merge_found_by(found_by, found_report)
                log().info(
                    "first search added %s, refused %s, attributed to %s model(s)",
                    found_report.get("added"),
                    found_report.get("not_in_the_data_bank"),
                    len(found_by),
                )
                if added:
                    _append_found(records, grouped, flat, added)
                continue
            if action == "rank":
                order = rank_structures(
                    flat,
                    providers=page_providers,
                    complete=complete,
                    api_keys=panel.api_keys(),
                    chat_models=panel.models(),
                    search_models=panel.research_models(),
                    key_slots=panel.key_slots(),
                )
                ranking = [row for row in order["ranked"] if isinstance(row, dict)]
                print(format_structure_ranking(order))
                log().info("second search ranked %s structure(s)", len(ranking))
                continue
            program_selected = _choose(grouped, flat, take_all=take_all, picks=picks)
            if not program_selected and not ranking:
                # Download with nothing ticked. On the command line that is the
                # end of the run and the printed line tells the user which flag
                # to pass, but on the page there is no flag to pass and the
                # panel simply closed: the run ended on the structures page
                # having downloaded nothing, with 382 structures listed and the
                # only explanation on a terminal the user was not looking at.
                # The page is where the choice is made, so the page comes back.
                #
                # A ranking is the one case where an empty download still has
                # somewhere to go: the ranked targets are offered next, and the
                # user picks there instead. That path stays as it was.
                notice = "Tick a structure to download, or tick a design class to take all of its structures."
                continue
            break
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # Download does not ask a model. The search buttons already did, and only
    # when they were pressed. A model tick left on the page is not a second
    # request about every listed structure. ``--research`` and ``--models`` are
    # that request from the command line, so a flag run still asks before it
    # downloads.
    providers = _providers(args.research)
    search_models = None
    # ``--research none`` turns the search off even when ``--models`` is also set.
    research_off = bool(args.research) and not providers
    if args.models and not research_off:
        try:
            names, search_models = search_plan(providers, _model_tokens(args.models))
        except ValueError as exc:
            _fail(str(exc))
            return 2
        providers = list(names)
    report: dict[str, object] | None = None
    named: list[StructureOption] = []
    if providers:
        try:
            report = research_targets(
                disease,
                flat,
                providers=providers,
                read_json=read_json,
                complete=complete,
                api_keys=panel.api_keys(),
                chat_models=panel.models(),
                search_models=search_models,
                sources=collected,
                key_slots=panel.key_slots(),
            )
            named = generation_rows(report, flat)
            llm_options = explicit_structure_rows(report, flat)
            comparison = compare_search_options(flat, llm_options, confirmed=program_selected)
            report["option_comparison"] = comparison
            report["returned_structures"] = returned_structure_records(flat, score_map)
            # The choose-structures page already saved this list. A flag run has no
            # page, so the comparison report is the other place the list is kept.
            if not (args.out_dir / "returned_structures.json").is_file():
                _keep_returned(flat, score_map)
            path = write_report(args.out_dir, report)
        except (OSError, ValueError) as exc:
            _fail(f"Target research failed: {exc}")
            return 2
        print(format_report(report))
        print(format_option_comparison(comparison))
        print(f"Wrote {path}.")
        log().info(
            "searched with %s, ranking %s, unavailable %s",
            report.get("providers"),
            report.get("ranking"),
            sorted(report.get("providers_unavailable") or {}),
        )
        log().info(
            "option comparison confirmed %s, in both %s, only program %s, only llm %s",
            len(comparison["confirmed"]),
            len(comparison["in_both"]),
            len(comparison["only_program"]),
            len(comparison["only_llm"]),
        )
        # No page for the comparison. The structures page the user just left already
        # holds this: every row the program search returned, the rows a model named
        # among them, which model named each one, and the ranking. A page repeating
        # it stood between Download and the generation page saying nothing new. The
        # comparison is still printed and still written to the report.
    elif args.target:
        _fail("Pass --research so the targets are ranked before --target picks them.")
        return 2
    targets: list[str] = []
    if report is not None and not named:
        ranked = _ranked(report)
        try:
            targets = _targets(args.target, ranked)
            if not targets and not program_selected and not args.all and not args.pick and args.ui and ranked:
                targets = panel.choose_targets(ranked)
        except ValueError as exc:
            _fail(str(exc))
            return 2
    elif args.target:
        ranked = _ranked(report or {})
        try:
            targets = _targets(args.target, ranked)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    selected = list(program_selected)
    if targets:
        # Choosing targets is itself the selection, so it decides which rows go to
        # generation rather than adding to every row the providers named.
        selected = _add_targets(selected, flat, targets)
    elif named and not program_selected and not args.all and not args.pick:
        selected = list(named)
    if not selected:
        print("Pass --all, or choose structures with --pick before download.")
        return 0

    print(f"Selected {len(selected)} structure(s) for download.")
    log().info(
        "selected %s structure(s): %s",
        len(selected),
        [f"{row.entry_id}_{row.chain}" for row in selected],
    )
    # One file per ticked deposit, from the RCSB archive the structure search
    # already used. The public record databases were asked when the page was
    # built. Download does not ask them again and does not call a model.
    reader = fetch if fetch is not None else fetch_structure_cif

    def _on_download(done: int, total: int, label: str) -> None:
        panel.note_download(done, total, label)

    # Downloading is the first thing that cannot be undone, so navigation closes here.
    panel.seal()
    try:
        download_dir = args.out_dir if args.download_dir is None else args.download_dir
        written = _place_for_jobs(
            _download(selected, download_dir, reader, _on_download if args.ui else None),
            args.out_dir,
        )
    except (OSError, ValueError) as exc:
        _fail(f"Structure download failed: {exc}")
        return 2

    # The page offering the model has to know whether there is a checkpoint for the
    # modified one, so the file is looked for before the page rather than after it.
    modified = _modified_checkpoint(args.modified_ckpt)
    fell_back = args.modified_ckpt is not None and modified is None
    if fell_back:
        print(f"Modified checkpoint {args.modified_ckpt} was not found. Commands use the original release model.")
    # A panel run always shows this page, even with the first page's tick set: the
    # tick answers the inference question, it does not stand in for choosing the
    # model and starting generation, which is all this page is.
    if args.ui:
        try:
            confirmed, start_generation, model, workers = panel.confirm_inference(
                len(written),
                modified="" if modified is None else str(modified),
                confirmed=bool(args.confirm_inference),
                hosts=args.hosts,
                local_workers=args.local_workers,
                remote_workers=args.remote_workers,
            )
        except ValueError as exc:
            _fail(f"Generation dispatch failed: {exc}")
            return 2
    else:
        confirmed = _inference_confirmed(args.confirm_inference, len(written), read_line)
        start_generation = args.run
        # No page asked, so the model is the one a named checkpoint has always selected.
        model = "original" if modified is None else "modified"
        # ...and the flags are the only thing that said how wide to run.
        workers = WorkerChoice(local=args.local_workers, remote=args.remote_workers)
    if not confirmed:
        print("Inference was not confirmed. Generation was not started.")
        log().info("inference was not confirmed for %s structure(s)", len(written))
        return 0

    ready = _write_jobs(written, selected, args.out_dir, binder_kind=args.binder_kind)
    if not ready:
        print("No structure could be annotated. Generation was not started.")
        return 0
    hosts = _hosts(args.hosts)
    shards = shard_for_hosts(ready, hosts)
    try:
        checkpoint = model_checkpoint(model, modified)
    except ValueError as exc:
        _fail(f"Generation dispatch failed: {exc}")
        return 2
    # The fall-back line above already says the commands use the original model.
    if not fell_back:
        if checkpoint is None:
            print("Using the original release model.")
        else:
            print(f"Using modified checkpoint {checkpoint}.")
    try:
        requests = generation_requests(
            shards,
            str(args.out_dir),
            modified_ckpt=checkpoint,
            remote_dir=args.remote_dir,
            binder_kind=args.binder_kind,
        )
    except (OSError, ValueError) as exc:
        _fail(f"Generation dispatch failed: {exc}")
        return 2
    for request in requests:
        print(" ".join(shlex.quote(part) for part in request.argv))
    if not start_generation:
        print("Generation was not started.")
        return 0
    # Checked on what the start-generation page returned, not on the flags: the
    # page may have changed either one, and validating the flag would pass a
    # value the page chose straight through to the semaphore.
    if workers.local is not None and workers.local < 1:
        _fail(
            "Generation dispatch failed: Local workers must be at least 1. "
            "Omit the limit to start every local request."
        )
        return 2
    if workers.remote is not None and workers.remote < 1:
        _fail(
            "Generation dispatch failed: Remote workers must be at least 1. "
            "Omit the limit to start every remote request."
        )
        return 2
    try:
        _mark_generation_started(requests)
        local_count = sum(1 for request in requests if is_local_host(request.host))
        remote_count = len(requests) - local_count
        print(f"Starting {local_count} local and {remote_count} remote generation request(s).")
        results = execute_generation(
            requests,
            runner=runner if runner is not None else subprocess_runner,
            transfer=transfer if transfer is not None else scp_transfer,
            retrieve=retrieve if retrieve is not None else scp_retrieve,
            local_workers=workers.local,
            remote_workers=workers.remote,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        _fail(f"Generation dispatch failed: {exc}")
        return 2
    for result in results:
        quoted = " ".join(shlex.quote(part) for part in result.argv)
        print(f"{result.host} exit {result.returncode}: {quoted}")
    lines, screens, trios = _save_finished_generations(
        results,
        requests,
        ready,
        args.out_dir,
        disease,
        args.binder_kind,
        proteome=_resolve_proteome(args),
        read_json=read_json,
        limits=TargetLimits(
            abundance_max=args.plasma_abundance_max,
            hub_max=args.plasma_hub_max,
            fold_max=args.plasma_fold_max,
            paralog_identity_max=args.paralog_identity_max,
        ),
        follow_up=bool(args.follow_up),
        lookup_targets=bool(args.lookup_targets),
        stage_tables=bool(args.stage_tables),
        disease_query=query,
        focus=focus,
        search=search_trace(
            live_structures=args.structures_json is None,
            live_disease=args.catalog is None,
            focus=focus,
            collected=collected,
        ),
    )
    panel.note_closing(lines, screens)
    panel.note_structures(trios)
    if any(result.returncode != 0 for result in results):
        print("Generation failed.")
        return 1
    print("Generation finished.")
    return 0


#: Suffixes the custom target accepts. Compressed bytes are refused outright.
_TARGET_SUFFIXES = (".pdb", ".cif")
#: First bytes of the compressors a renamed archive would start with.
_COMPRESSED = (b"\x1f\x8b", b"\x28\xb5\x2f\xfd", b"BZh", b"PK\x03\x04", b"\xfd7zXZ")


def _uploaded_target(path: Path, out_dir: Path) -> tuple[StructureOption, Path]:
    """Copy a custom target beside the run and return the row that lists it.

    The file did not come from the program, so it is checked before it is kept: the
    suffix must be one the model reads, the bytes must not be a renamed archive, and
    it must decode as text. It is copied into its own directory rather than read in
    place, so a later step cannot be changed underneath it. Nothing in it is
    executed and nothing is repaired.
    """
    suffix = path.suffix.casefold()
    if suffix not in _TARGET_SUFFIXES:
        allowed = ", ".join(_TARGET_SUFFIXES)
        raise ValueError(f"{path.name} is not one of {allowed}.")
    raw = path.read_bytes()
    if not raw.strip():
        raise ValueError(f"{path.name} is empty.")
    if any(raw.startswith(magic) for magic in _COMPRESSED):
        raise ValueError(f"{path.name} holds compressed bytes under a {suffix} name.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name} is not text: {exc}") from None
    if not _looks_like_structure(text, suffix):
        raise ValueError(f"{path.name} does not read as a {suffix} structure.")
    directory = out_dir / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    kept = directory / path.name
    kept.write_bytes(raw)
    return (
        StructureOption(
            group="protein",
            accession=path.stem,
            entry_id=path.stem,
            chain="A",
            title=f"uploaded {path.name}",
            predicted=False,
            source="upload",
            methodology_known=True,
        ),
        kept,
    )


def _looks_like_structure(text: str, suffix: str) -> bool:
    """Whether the text reads as the kind of file its name claims."""
    if suffix == ".cif":
        return "data_" in text or "_atom_site." in text
    return any(line.startswith(("ATOM", "HETATM", "HEADER", "MODEL", "CRYST1")) for line in text.splitlines())


def _load_records(
    structures_json: Path | None,
    disease_id: str,
    *,
    post: PostJson,
    get: GetJson,
    records: dict[str, dict] | None = None,
    extra_accessions: list[str] | None = None,
    extra_symbols: list[str] | None = None,
    read_value: ReadValue | None = None,
) -> list[StructureOption]:
    if structures_json is None:
        return collect_structure_options(
            disease_id,
            post=post,
            get=get,
            include_related=True,
            records=records,
            extra_accessions=extra_accessions,
            extra_symbols=extra_symbols,
            read_value=read_value,
        )
    payload = json.loads(structures_json.read_text(encoding="utf-8"))
    return options_from_records(payload)


def _biological_focus(
    panel: OptionPanel,
    disease: Disease,
    args: argparse.Namespace,
    *,
    post: PostJson,
    read_value: ReadValue,
) -> BiologicalFocus:
    """Ask for the pathway, mechanism, and optional target, then fill the evidence.

    A local structure file is not a live search. The page still asks, and a
    typed target can rank those rows, but Reactome and Open Targets are not called.
    """
    live = args.structures_json is None
    pathways: list[tuple[str, str]] = []
    targets: list[tuple[str, str]] = []
    accessions: list[str] = []
    symbols: list[str] = []
    # A read that errored is not a read that found nothing, and both leave these
    # lists empty. The evidence chain was taking "live" as proof of a consult
    # and recording Reactome and Open Targets as consulted on a run whose fetch
    # had raised; this is the only place that knows the difference, so the
    # outcome is carried out on the focus and read back by ``_service_notes``.
    #
    # The two reads are caught separately because their outcomes differ: a
    # failed target list means the pathway list was never attempted, which is
    # ``not_performed`` and not a failure of Reactome's.
    failures: list[str] = []
    if live:
        try:
            accessions, symbols, targets = focus_targets(disease.disease_id, post=post)
        except (OSError, ValueError) as exc:
            log().warning("biological target list failed: %s", exc)
            failures.append(OPEN_TARGETS_FOCUS)
        else:
            try:
                pathways = reactome_pathway_list(read_value, accessions)
            except (OSError, ValueError) as exc:
                log().warning("reactome pathway list failed: %s", exc)
                failures.append(REACTOME_PATHWAY_LIST)
    posted = panel.choose_focus(disease, pathways, targets)
    pathway_members: list[str] = []
    partners: list[str] = []
    if live and posted.pathway_id not in {"", "any"}:
        pathway_members = pathway_uniprot_members(read_value, posted.pathway_id, MAX_ASSOCIATIONS)
    if live and posted.mechanism == "interaction":
        partners = partner_symbols(read_value, [*accessions, *symbols], MAX_ASSOCIATIONS)
    accession, symbol = query_parts(posted.target)
    return BiologicalFocus(
        pathway_id=posted.pathway_id,
        pathway_name=posted.pathway_name,
        mechanism=posted.mechanism,
        target=posted.target,
        target_accession=accession,
        target_symbol=symbol,
        disease_accessions=tuple(accessions),
        disease_symbols=tuple(symbols),
        pathway_accessions=tuple(pathway_members),
        interaction_symbols=tuple(partners),
        fetch_failures=tuple(failures),
    )


def _model_tokens(values: list[str] | None) -> list[str]:
    """Flatten repeated and comma-separated ``--models`` values."""
    if not values:
        return []
    tokens: list[str] = []
    for value in values:
        tokens.extend(part for part in value.replace(",", " ").split() if part)
    return tokens


def _providers(values: list[str] | None) -> list[str]:
    """Flatten repeated and comma-separated ``--research`` values. ``none`` turns it off."""
    if not values:
        return []
    names: list[str] = []
    for value in values:
        names.extend(part for part in value.replace(",", " ").split())
    if any(name.casefold() in {"none", "off"} for name in names):
        return []
    return names


def _ranked(report: dict[str, object]) -> list[dict[str, object]]:
    """The ranked view of one report, highest combined value first."""
    comparison = report.get("comparison")
    ranked = comparison.get("ranked") if isinstance(comparison, dict) else None
    if not isinstance(ranked, list):
        return []
    return [row for row in ranked if isinstance(row, dict)]


def _targets(values: list[str] | None, ranked: list[dict[str, object]]) -> list[str]:
    """Identifiers for the ``--target`` values. A number is a position on the ranking."""
    if not values:
        return []
    identifiers = [str(row.get("identifier") or "") for row in ranked]
    wanted: list[str] = []
    for value in values:
        for raw in value.replace(",", " ").split():
            token = raw.strip()
            if token.casefold() in {"all", "every"}:
                wanted.extend(identifiers)
                continue
            if token.isdigit():
                index = int(token)
                if not 1 <= index <= len(identifiers):
                    raise ValueError(f"Target {index} is outside 1..{len(identifiers)}.")
                wanted.append(identifiers[index - 1])
                continue
            match = [name for name in identifiers if name.casefold() == token.casefold()]
            if not match:
                listed = ", ".join(name for name in identifiers if name) or "none"
                raise ValueError(f"Target {token!r} is not on the ranked list. Ranked: {listed}.")
            wanted.append(match[0])
    chosen: list[str] = []
    for name in wanted:
        if name and name not in chosen:
            chosen.append(name)
    return chosen


def _add_targets(
    selected: list[StructureOption],
    flat: list[StructureOption],
    targets: list[str],
) -> list[StructureOption]:
    """Add every listed structure of each chosen target, keeping the list order."""
    wanted = {name.casefold() for name in targets}
    rows = list(selected)
    for row in flat:
        if row.accession.strip().casefold() in wanted and row not in rows:
            rows.append(row)
    held = {row.accession.strip().casefold() for row in rows}
    missing = [name for name in targets if name.casefold() not in held]
    if missing:
        print(f"No structure is listed for {', '.join(missing)}, so nothing was added for it.")
    return rows


def _choose(
    grouped: dict[str, list[StructureOption]],
    flat: list[StructureOption],
    *,
    take_all: bool,
    picks: str | None,
) -> list[StructureOption]:
    chosen: list[StructureOption] = []
    if take_all:
        chosen.extend(select_options(grouped, None))
    if not picks:
        return chosen
    for token in picks.split(","):
        text = token.strip()
        if not text:
            continue
        if not text.isdigit():
            raise ValueError(f"Structure choice {text!r} is not a number from the list.")
        index = int(text)
        if not 1 <= index <= len(flat):
            raise ValueError(f"Structure choice {index} is outside 1..{len(flat)}.")
        row = flat[index - 1]
        if row not in chosen:
            chosen.append(row)
    return chosen


def _download(
    selected: list[StructureOption],
    directory: Path,
    fetch: Fetch,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    return download_selected(selected, directory, fetch, progress=progress)


def _place_for_jobs(written: list[Path], out_dir: Path) -> list[Path]:
    """Return each structure file from inside the output directory.

    A separate download directory keeps the fetched file. The job writer and
    the generation spec read the copy in the output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    placed: list[Path] = []
    for source in written:
        dest = out_dir / source.name
        if source.resolve() != dest.resolve():
            shutil.copyfile(source, dest)
        placed.append(dest)
    return placed


def _inference_confirmed(flag: bool, count: int, read_line: ReadLine) -> bool:
    if flag:
        return True
    answer = read_line(f"Confirm inference for {count} downloaded structure(s)? [y/N]: ")
    return answer.strip().casefold() in {"y", "yes"}


def _write_jobs(
    written: list[Path],
    selected: list[StructureOption],
    directory: Path,
    *,
    binder_kind: str,
) -> list[StructureOption]:
    ready: list[StructureOption] = []
    for path, option in zip(written, selected, strict=True):
        try:
            spec = write_job(
                path,
                directory,
                binder_kind=binder_kind,
                target_id=option.accession,
                chain=option.chain,
            )
        except ValueError as exc:
            print(f"Skipped {option.entry_id}: {exc}")
            continue
        if spec["generation_started"]:
            raise RuntimeError(f"Job for {option.entry_id} marked generation as started.")
        ready.append(option)
    return ready


def _save_finished_generations(
    results: list[RunResult],
    requests: list[GenerationRequest],
    structures: list[StructureOption],
    out_dir: Path,
    disease: Disease,
    binder_kind: str,
    *,
    proteome: Path | None = None,
    read_json: ReadJson | None = None,
    limits: TargetLimits | None = None,
    follow_up: bool = False,
    lookup_targets: bool = False,
    stage_tables: bool = False,
    disease_query: str = "",
    focus: BiologicalFocus | None = None,
    search: SearchTrace | None = None,
) -> tuple[list[str], list[dict[str, object]], list[StructureTrio]]:
    """Copy each generation that exited 0 into the disease and target archive.

    A failed request is left out. The live output directory is not moved.
    The returned lines are the screen sentences for the copies that were saved.
    The records are the same screens, for the closed page's tabs.
    The structures are the three views of each generated sample. They are
    collected before the archive copy, so a failed copy still has them.
    A research trace is written into the live directory before that copy, so
    the archive keeps one record per generated binder.
    """
    lines: list[str] = []
    screens: list[dict[str, object]] = []
    trios: list[StructureTrio] = []
    if len(results) != len(requests):
        print("Generation archive skipped: the finished requests do not match the plan.")
        return lines, screens, trios
    # One memo for the whole archive loop. Two structures of one target ask the
    # same services the same questions, and the shipped fixture is exactly that.
    reader = _screen_reader(read_json, lookup_targets=lookup_targets)
    for result, request in zip(results, requests, strict=True):
        if result.returncode != 0:
            continue
        output_dir = _output_dir(request)
        option = _structure_for_archive(output_dir, structures)
        if output_dir is None or option is None:
            print("Generation archive skipped: a finished request has no target.")
            continue
        if stage_tables:
            # Before the screen, not after: the screen reads what is on disk
            # when it runs.
            _stage_screen_tables(output_dir, disease.name, option.accession)
        screen = _record_screen(
            output_dir,
            binder_kind,
            option,
            proteome=proteome,
            read_json=reader,
            limits=limits,
        )
        trios.extend(structure_trios(output_dir, entry_id=option.entry_id, chain=option.chain))
        write_research_traces(
            output_dir,
            disease_name=disease.name,
            disease_id=disease.disease_id,
            disease_query=disease_query,
            focus=focus,
            option=option,
            binder_kind=binder_kind,
            argv=request.argv,
            screen=screen,
            search=search,
        )
        try:
            path = save_generation(
                out_dir,
                output_dir,
                disease_name=disease.name,
                disease_id=disease.disease_id,
                target_id=option.accession,
                group=option.group,
                entry_id=option.entry_id,
                chain=option.chain,
                binder_kind=binder_kind,
            )
        except OSError as exc:
            print(f"Generation archive was not saved for {option.entry_id} chain {option.chain}: {exc}")
            continue
        shown = display_directory(str(path))
        print(f"Saved generation {shown}")
        log().info("saved generation %s", shown)
        if screen is not None:
            screens.append({"entry_id": option.entry_id, "chain": option.chain, "screen": screen})
            for line in format_screen_lines(option.entry_id, option.chain, screen):
                print(line)
                lines.append(line)
            if follow_up:
                lines.extend(_record_follow_up(output_dir, screen))
    return lines, screens, trios


def _condition_spec(output_dir: Path) -> Path:
    """The condition spec this run was generated from, live or archived.

    A live run's spec sits beside its output directory; the archive copies it
    inside, under the original stem. Looking only beside would hand a
    follow-up a path that does not exist for everything under ``generations/``.
    """
    inside = output_dir / f"{base_name(output_dir)}.yaml"
    if inside.is_file():
        return inside
    return output_dir.parent / f"{output_dir.name}.yaml"


def _refold_only(args: argparse.Namespace) -> int:
    """Export a finished run for an external folder, or import what came back.

    Both directions exist because the folder may not be something this machine
    can run: AF3's weights are licence-gated and the AlphaFold Server has no
    API, so the fold happens elsewhere and the only thing crossing the gap is
    files. The export writes them; the import files them where the refolding
    metrics already look.

    What is written here is a FASTA, which is what any sequence-driven folder
    takes. The hosted AlphaFold Server's own JSON dialect is not written here:
    :py:mod:`rfproteina.af3server.request` owns it, checked against that
    service's allowlists, token budget and daily quota. Two implementations of
    one dialect is one too many, and the other one is the complete one.

    Printed as JSON because the sequel to this command is usually another
    command, and every count here is something a script needs to branch on.
    """
    run = args.refold
    if not run.is_dir():
        _fail(f"--refold: not a directory: {run}")
        return 2
    requests = args.refold_dir if args.refold_dir is not None else run / "refold_requests"
    oracle = PathOracle(oracle_name=args.refold_oracle)
    record: dict[str, object] = {"run": str(run), "oracle": args.refold_oracle}

    if args.refold_import is not None:
        if not args.refold_import.is_dir():
            _fail(f"--refold-import: not a directory: {args.refold_import}")
            return 2
        bases = design_bases(run)
        imported = import_refolds(
            oracle,
            run,
            args.refold_import,
            example_ids=sorted(bases),
            bases=bases,
        )
        record["imported"] = imported.as_dict()
        log().info(f"refold import: {imported.as_dict()}")
    else:
        fasta = write_refold_requests(run, requests)
        record["requests"] = str(requests)
        record["fasta"] = fasta.as_dict()
        log().info(f"refold export: {fasta.as_dict()}")

    # Always last, in both directions: after an export it says what to send,
    # and after an import what has still not arrived.
    record["status"] = refold_status(oracle, run, requests).as_dict()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def _screen_only(
    args: argparse.Namespace,
    incoming: list[str],
    *,
    read_json: ReadJson | None,
) -> int:
    """Re-screen finished runs. Nothing is generated, downloaded or dispatched.

    This exists because the tables the screen reads are produced after a run
    finishes -- the user scores the designed chain with a local predictor and
    drops the result in. Until now the screen ran once, seconds after the
    generate subprocess returned, so anything supplied afterwards was never
    read by anything.

    One directory that is itself a run is screened alone; anything else is
    walked and every run beneath it is screened.
    """
    root = args.screen
    if not root.is_dir():
        _fail(f"{root} is not a directory.")
        return 2
    cut_short = False
    try:
        runs = generation_directories(root)
    except ScreenWalkTruncatedError as stopped:
        # Screen what was found, but never let a short list read as a complete
        # one: silence here meant a run under the named directory was skipped
        # while the output said none existed and the status said success.
        runs = stopped.found
        cut_short = True
        print(
            f"The search stopped after {stopped.visited} directories, "
            f"so runs beneath {display_directory(str(root))} may be unscreened."
        )
    if not runs:
        print(f"No finished generation was found in {display_directory(str(root))}.")
        return 1 if cut_short else 0
    reader = _screen_reader(read_json, lookup_targets=args.lookup_targets)
    limits = TargetLimits(
        abundance_max=args.plasma_abundance_max,
        hub_max=args.plasma_hub_max,
        fold_max=args.plasma_fold_max,
        paralog_identity_max=args.paralog_identity_max,
    )
    skipped = 0
    for run in runs:
        label = label_for(root, run)
        try:
            screened = _screen_one(run, args, incoming, reader=reader, limits=limits, label=label)
        except Exception as exc:
            # The isolation this function advertises has to hold for anything a
            # run's own files can raise, not only the causes already named. A
            # sidecar of deeply nested JSON raised RecursionError out of
            # json.loads, which no narrow clause caught: a two hundred run
            # archive wrote no screens at all, exit 2, and the message named no
            # directory, so nothing said which run was at fault.
            print(f"Screen was not written for {label or display_directory(str(run))}: {exc}")
            screened = False
        if not screened:
            skipped += 1
    if len(runs) > 1 or skipped:
        # A count, because at two hundred runs the per-directory lines scroll
        # past and the exit code alone does not say how much was skipped.
        print(f"Screened {len(runs) - skipped} of {len(runs)} run(s), skipped {skipped}.")
    return 1 if skipped or cut_short else 0


def _screen_one(
    run: Path,
    args: argparse.Namespace,
    incoming: list[str],
    *,
    reader: ReadJson | None,
    limits: TargetLimits,
    label: str,
) -> bool:
    """Screen one finished run, printing its lines. False when it was skipped."""
    shown = label or display_directory(str(run))
    try:
        # Asked for its refusal, not its contents: a screen already here that
        # cannot be read is the only evidence of whatever went wrong with it,
        # and this is about to overwrite the file.
        read_screen(run)
    except ValueError as exc:
        print(f"Screen was not replaced for {shown}: {exc}")
        return False
    kind, target = _screened_identity(run, incoming)
    if kind is None:
        print(f"Screen was not written for {shown}: nothing names the binder kind.")
        return False
    if not target:
        print(f"{shown} names no target, so the target axes stay untested.")
    try:
        screen = screen_generation(
            run,
            binder_kind=kind,
            target_id=target,
            bar=args.bar,
            proteome=_resolve_proteome(args),
            read_json=reader,
            limits=limits,
        )
    except (HTTPException, LookupMissError, OSError, ValueError) as exc:
        print(f"Screen was not written for {shown}: {exc}")
        return False
    # The screen file has just been replaced. Any trace beside it still holds
    # the previous verdict and names that file as where it came from, so it is
    # refreshed here rather than left to disagree with its own evidence.
    refreshed = restamp_screening(run, screen)
    if label:
        print(label)
    for line in format_screen_lines_for_directory(run, screen):
        print(line)
    if refreshed:
        print(f"Refreshed {len(refreshed)} research trace(s).")
    if args.follow_up:
        # Named from the run's position in the tree, not from base_name: that
        # strips the archive's -N collision suffix, so gene_8TLD_2 and
        # gene_8TLD_2-2 both resolved to one root and the second run's spec
        # overwrote the first's. Both are chain A_1 with one failed binder_0, so
        # the surviving plan re-folded the other folder's design. base_name is
        # for finding files copied inside an archived folder, never for naming
        # a new output directory, which has to be unique per run.
        _record_follow_up(
            run,
            screen,
            fresh_root=args.out_dir / "follow_up" / _follow_up_leaf(run, label),
        )
    return True


def _follow_up_leaf(run: Path, label: str) -> str:
    """Where one run's follow-up spec goes, unique per run.

    A tree's label is the run's path relative to the root, so it is already
    unique and keeps the archive's shape. Screening one directory at a time
    gives no label, and a folder's own name is not unique across trees: two
    runs called ``gene_8TLD_2`` in different campaigns shared a root, the
    second invocation overwrote the first's spec, and the surviving plan named
    a spec that folds the other run's design. The digest is of the resolved
    path, so the same run always lands in the same place.
    """
    if label:
        # Kept as the plain relative label, so the follow-up tree mirrors the
        # archive and a reader finds a run's plan where the run lives. That is
        # unique within one --screen root but not across two: two campaign trees
        # each holding asthma/IL13/gene_8TLD_2, screened separately into one
        # --out-dir, resolve to the same leaf. Each spec is still written to its
        # own run's follow_up.json, so no plan is lost; what is shared is the
        # generation directory they name. Screen one root per --out-dir, or pass
        # a different --out-dir per campaign.
        return label
    digest = hashlib.sha256(str(run.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{run.name}-{digest}"


def _screened_identity(run: Path, incoming: list[str]) -> tuple[str | None, str]:
    """The binder kind and target for a finished run, from the files beside it.

    ``generation.json`` is preferred over the job file: the archive wrote it
    from the same structure that drove the original screen, it is inside the
    folder so it cannot describe a different run, and it is the only one that
    also names the entry and chain.

    An explicitly typed ``--binder-kind`` wins over both, because it is the
    only way to screen a directory that was assembled by hand. A kind that
    nothing names is refused rather than guessed: the kind decides which axes
    exist at all, so guessing writes a verdict over axes that never applied.
    """
    kind = _typed_binder_kind(incoming)
    target = ""
    for source in (run / "generation.json", job_file_for_directory(run)):
        if source is None or not source.is_file():
            continue
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        # RecursionError is listed on purpose and is not covered by the others:
        # deeply nested JSON exhausts the decoder's stack and raises it rather
        # than JSONDecodeError, and it inherits from neither ValueError nor
        # OSError. Without it a single malformed file beside a run aborted the
        # whole screen, which is what the run log recorded on 2026-11-08.
        except (OSError, UnicodeError, RecursionError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if kind is None:
            found = payload.get("binder_kind")
            if isinstance(found, str) and found.strip():
                kind = found.strip()
        if not target:
            named = payload.get("target_id")
            if isinstance(named, str) and named.strip():
                target = named.strip()
        if kind is not None and target:
            break
    return kind, target


def _typed_binder_kind(incoming: list[str]) -> str | None:
    """``--binder-kind`` when it was actually typed, rather than its default.

    The same pre-parse ``--log-file`` already uses, over the argv this call was
    given rather than the process's own -- a caller passing argv as an argument
    is the normal case in tests and would otherwise be read from `sys.argv`.
    Reading the parsed namespace cannot tell the default apart from the same
    value typed on purpose, and that difference decides whether a file on disk
    is allowed to name the kind instead.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--binder-kind")
    known, _rest = parser.parse_known_args(incoming)
    value = known.binder_kind
    return value.strip() if isinstance(value, str) and value.strip() else None


def _screen_reader(read_json: ReadJson | None, *, lookup_targets: bool) -> ReadJson | None:
    """The reader the target axes use, remembered for the life of one invocation.

    Built once per invocation rather than once per screen: a campaign screens
    several structures of the same target, and a per-screen memo would ask the
    same services the same questions once per structure. An injected reader
    wins over the flag, so a test's stub is still a stub.
    """
    reader = read_json
    if reader is None and lookup_targets:
        reader = read_public_json
    return None if reader is None else _remembering_reader(reader)


def _remembering_reader(read: ReadJson) -> ReadJson:
    """``read`` with each URL asked at most once for the life of this run.

    An archive holds several runs for one target, so screening a tree asks the
    same services the same questions once per run folder. The memo sits above
    the reader, never inside it, so a real request still passes through the
    per-host allowance while a remembered answer -- which is not a request --
    does not consume one.

    A definite answer is remembered, including "no record": re-asking about a
    target a service does not hold is the same waste. A failure is not, because
    remembering one unavailable moment would report every run in the tree as
    unavailable.
    """
    answers: dict[str, object] = {}
    misses: set[str] = set()
    guard = threading.Lock()

    def remembering(url: str) -> object:
        with guard:
            if url in misses:
                raise LookupMissError(url)
            if url in answers:
                return answers[url]
        try:
            found = read(url)
        except LookupMissError:
            with guard:
                misses.add(url)
            raise
        with guard:
            answers[url] = found
        return found

    return remembering


def _portable_command(command: Sequence[str]) -> str:
    """One planned command line with this machine taken out of it, still runnable.

    ``scrubbed_json`` runs the free-text scrubber over every string it writes,
    and that scrubber replaces a personal prefix -- the checkout root among them
    -- wherever it appears, because in a sentence a home directory can sit
    anywhere. A command line is not a sentence. The replacement landed inside
    the token and left ``out_dir=<home>/disease_runs/gene_8TLD_2``, which is not
    a path and cannot be run, so the plan this module exists to write down was
    the one copy of it that could not be used. The same commands printed to the
    terminal are not scrubbed and were right all along, so the file and the
    terminal disagreed and the file held the broken copy.

    A path inside the checkout is written checkout-relative, which is what
    AGENTS.md asks of a path in a stored file and is runnable from the
    repository root these commands are meant to be run from anyway. A path
    outside it keeps every component after its personal prefix, which is what
    :func:`scrub_path_prefix` is for: a reader uses the rest to tell two runs
    apart.
    """
    return shlex.join([_portable_token(token) for token in command])


def _portable_token(token: str) -> str:
    """A token with an absolute path in it made portable; anything else untouched.

    Only an absolute path can name this machine, and only an absolute path is
    worth rewriting: a relative value is already portable, and putting one
    through :func:`repository_relative` would resolve it against the working
    directory and invent a path it never named. ``recycle_strategy=medium/flat``
    is the override that makes this concrete -- its value holds a separator and
    is not a path, and resolved from anywhere but the repository root it came
    back as ``/somewhere/else/medium/flat``.
    """
    key, separator, value = token.partition("=")
    if separator and _absolute_path(value):
        return f"{key}={_portable_path(value)}"
    if not separator and _absolute_path(token):
        return _portable_path(token)
    return token


def _absolute_path(value: str) -> bool:
    """Whether THIS platform can resolve the value as an absolute path.

    Deliberately the running platform's answer and not both platforms'. The
    token is rewritten here, at the moment the plan is written, so the only
    spelling that can occur is the one this machine produces: every path in a
    plan comes from ``Path.as_posix()`` of a real path on this machine.

    Accepting the other platform's spelling as well looked free and was not.
    ``_portable_path`` resolves what it is handed, and a value this platform
    cannot read as absolute -- ``C:\\Users\\name\\runs`` under WSL -- was joined
    onto the working directory instead, so the stored token came out as a
    different non-path from every directory the program might be run from.
    Left alone it stays exactly what the caller wrote, and the free-text
    scrubber still takes any account name out of it.
    """
    return Path(value).is_absolute()


def _portable_path(value: str) -> str:
    """One absolute path with this machine taken out of it.

    ``repository_relative`` returns a path outside the checkout unchanged, and
    ``scrub_path_prefix`` does nothing to a relative one, so a single pass
    through both is right in either case.
    """
    return scrub_path_prefix(repository_relative(Path(value)))


def _record_follow_up(
    output_dir: Path,
    screen: dict[str, object],
    *,
    fresh_root: Path | None = None,
) -> list[str]:
    """Write and print the next samples this screen selects. Nothing is started.

    These are argv lists for the user to run, kept in ``follow_up.json`` beside
    the designs they came from. Writing a plan down rather than launching it is
    the division every module here keeps: a planned shard does not start a
    generation either.
    """
    chain = designed_chain(output_dir)
    if not chain:
        print(f"Follow-up skipped for {output_dir.name}: no chain of interest in the metrics.")
        return []
    try:
        commands = follow_up_commands(
            screen,
            generation_dir=output_dir,
            generated_cifs=sample_cifs(output_dir),
            original_spec=_condition_spec(output_dir),
            fresh_root=fresh_root or output_dir.parent / f"{output_dir.name}_follow_up",
            designed_chain=chain,
        )
    except (OSError, ValueError) as exc:
        print(f"Follow-up was not written for {output_dir.name}: {exc}")
        return []
    if not commands:
        return []
    spelled = [shlex.join(command) for command in commands]
    payload = {
        "version": 1,
        "started": False,
        # Accurate as written: a path outside the checkout stays absolute, with
        # only its personal prefix taken off, because the rest is what tells two
        # runs apart.
        "paths": "paths inside the checkout are relative to the repository root",
        "commands": [_portable_command(command) for command in commands],
    }
    try:
        write_text_atomically(output_dir / "follow_up.json", scrubbed_json(payload))
    except OSError as exc:
        # A plan that could not be written is still worth printing, but it must
        # not take the archive loop down with it on a full or read-only disk.
        print(f"Follow-up was not written for {output_dir.name}: {exc}")
    lines = [f"Follow-up for {output_dir.name}: {len(commands)} command(s), not started."]
    lines.extend(f"  {command}" for command in spelled)
    for line in lines:
        print(line)
    return lines


def _resolve_proteome(args: argparse.Namespace) -> Path | None:
    """The FASTA the off-target axis should read, and why.

    ``--proteome`` always wins, because it is the explicit instruction. With
    ``--stage-tables`` and no ``--proteome``, the downloaded human reference
    proteome stands in: the registry fetches that exact file for this axis, and
    without this the axis reported ``proteome_not_configured`` on a machine
    holding it. Printed rather than silent -- which FASTA a clear was measured
    against is part of the measurement, and the axis records only the blocker.

    Returns None when neither applies, which leaves the existing
    ``HUMAN_PROTEOME`` fallback inside the reader exactly as it was.
    """
    if args.proteome is not None:
        return Path(args.proteome)
    if not getattr(args, "stage_tables", False):
        return None
    found = downloaded_proteome()
    if found is not None:
        print(f"Off-target axis will read the downloaded proteome: {found.name}")
    return found


def _stage_screen_tables(output_dir: Path, disease_name: str, target_id: str) -> None:
    """Convert the downloaded reference tables into this run's directory.

    The screen reads its tables out of the run directory, and the downloads sit
    somewhere else under the names their servers gave them. Without this step a
    machine holding every download screened exactly as a bare one did: the fold
    axis reported no table and the run looked equipped.

    Narrowed to this run's own target and this run's own disease, which is what
    makes the result readable -- the whole-disease conversion is past the fold
    reader's 1 MB bound and would be written and then refused.

    Never fatal. Every outcome here is a thinner screen, not a failed one, so
    each is printed and the run continues to the screen either way.
    """
    try:
        made = stage_for_run(output_dir, disease=disease_name, targets=[target_id])
    except OSError as exc:
        print(f"Screen tables were not staged: {exc}")
        return
    for stem, path in made.staged:
        print(f"Staged {stem} for the screen: {path.name}")
    for stem, reason in made.shortfalls:
        # Only the shortfalls. The three downloads that feed no axis are a
        # standing fact about those artifacts, not news about this run.
        print(f"Screen table not staged: {stem}: {reason}")
    fold = made.fold
    if fold is not None and fold.usable and not fold.answers_accessions:
        print(
            "Staged fold table carries no UniProt accession, and this screen looks its "
            "target up by one: the axis will read untested. Fetch the UniProt human "
            "reference proteome to close that."
        )
    if fold is not None and fold.blocker == "disease_not_in_table" and fold.available:
        print(f"The fold table names no {disease_name!r}. It does hold: {sample_line(fold.available)}")
    if fold is not None and fold.blocker == "targets_not_in_table":
        # The disease is there and this run's target is not, which is the
        # ordinary outcome for a target that resolved to no accession. Named
        # apart from the disease miss because the two call for different fixes.
        named = sample_line(fold.available_genes)
        print(f"{disease_name!r} reports no fold for {target_id}." + (f" It does report: {named}" if named else ""))


def _record_screen(
    output_dir: Path,
    binder_kind: str,
    option: StructureOption,
    *,
    proteome: Path | None = None,
    read_json: ReadJson | None = None,
    limits: TargetLimits | None = None,
    lookup_targets: bool = False,
) -> dict[str, object] | None:
    """Write the measured screen. A read or write error skips that screen only.

    ``proteome`` is the FASTA named by ``--proteome``. When it is None the axis
    falls back to ``HUMAN_PROTEOME``, and without either it stays untested.

    ``read_json`` is what the target axes look themselves up with. The caller
    resolves it, by ``_screen_reader``, so that one memo covers a whole archive
    loop instead of one per screen. Reaching three public services is not
    something a run should do because nobody said not to: without
    ``--lookup-targets`` those axes report ``network_not_permitted``, which is
    true and says what to pass. ``plasma_fold`` reads a local table and is
    measured either way. A ligand binder also reads ``toxicology.tsv`` from
    the same directory. A protein or peptide binder also reads
    ``paralog_coevolution.tsv`` from the same directory onto the paralog record.
    The same protein or peptide screen records chemical motifs, free cysteine
    when the CIF names disulfide partners, and the Bjellqvist 1994 sequence
    charge under ``chemical_liability``. A motif leaves the call as the named
    bar decided it. Ligand and nucleic screens leave that scan unread.
    ``modality_screen`` follows the binder kind: furin, Asp-Pro, N-terminal
    glutamine, nonstandard letters, the aliphatic index, the average
    molecular weight, the Pace 1995 extinction at 280 nm, the N-end rule
    half-life, the residue and atomic composition, a Kyte-Doolittle span, and
    an AGGRESCAN 2007 normalized sum on a protein or peptide, CpG on a nucleic
    binder, and one non-polymer
    formula weight on a ligand. A missing
    weight is calculated from the chemical formula when every element is known,
    and left out when the formula does not parse. Those counts leave the call
    as the named bar decided it.

    ``lookup_targets`` remains for a caller that passes neither, so this is
    still safe to call directly.
    """
    reader = read_json
    if reader is None and lookup_targets:
        reader = read_public_json
    try:
        return screen_generation(
            output_dir,
            binder_kind=binder_kind,
            target_id=option.accession,
            proteome=proteome,
            read_json=reader,
            limits=limits,
        )
    except (HTTPException, LookupMissError, OSError, ValueError) as exc:
        # A service answering "no record" is routine, and must not cost the
        # screen. Every axis already has a blocker for it; this is the backstop
        # for one that does not.
        print(f"Screen was not saved for {option.entry_id} chain {option.chain}: {exc}")
        return None


def _output_dir(request: GenerationRequest) -> Path | None:
    """The local directory named by this request's ``out_dir=`` argument."""
    for part in request.argv:
        if part.startswith("out_dir="):
            return Path(part.split("=", 1)[1])
    return None


def _structure_for_archive(output_dir: Path | None, structures: list[StructureOption]) -> StructureOption | None:
    if output_dir is None:
        return None
    for option in structures:
        if f"{option.group}_{option.entry_id}_{option.chain}" == output_dir.name:
            return option
    return None


def _mark_generation_started(requests: list[GenerationRequest]) -> None:
    """Record that generation is being launched. Planning leaves the flag false."""
    for request in requests:
        job = _job_path(request.local_input)
        spec = json.loads(job.read_text(encoding="utf-8"))
        spec["generation_started"] = True
        job.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def _job_path(generation_input: Path) -> Path:
    name = generation_input.name
    if name.endswith(".annotated.cif"):
        stem = name.removesuffix(".annotated.cif")
    elif name.endswith(".yaml"):
        stem = name.removesuffix(".yaml")
    else:
        raise ValueError(f"Generation input {name} is not an annotated CIF or a condition spec.")
    return generation_input.with_name(f"{stem}.job.json")


def _modified_checkpoint(path: Path | None) -> Path | None:
    if path is not None and path.is_file():
        return path
    return None


def _hosts(text: str | None) -> list[str]:
    if text is None:
        return ["local"]
    hosts = [host.strip() for host in text.split(",") if host.strip()]
    return hosts or ["local"]


if __name__ == "__main__":
    raise SystemExit(main())
