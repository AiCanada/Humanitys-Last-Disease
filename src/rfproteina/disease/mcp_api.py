"""In-process tools for the Humanitys-Last-Disease MCP server.

An agent calls these instead of starting ``human`` and reading its text.
``run`` uses the same ``main`` as the command. Screening and the verdict
return the screen record. This module does not start a shell, and it does
not read a prompt from stdin, because stdin belongs to the MCP client.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

# The one spelling of the program name. Repeated as a literal here and in
# mcp_layers, it had to be edited in both on the rename to
# Humanitys-Last-Disease, and the output schema *requires* "program": a miss
# would have answered every call with a name the server's own serverInfo
# contradicted. mcp_layers does not import this module at module scope, so
# this direction is the acyclic one.
from rfproteina.disease.mcp_layers import PROGRAM

# The closed vocabulary of threshold kinds, imported rather than copied. It is
# a frozenset in one module and it grew by two entries while this check was
# being written, which is exactly what a literal list here would have missed:
# a kind added there would read as malformed here and the entry would be
# dropped from every verdict until someone noticed. ``threshold_provenance``
# imports nothing but the standard library, so this costs nothing at import.
from rfproteina.disease.threshold_provenance import KINDS as _PROVENANCE_KINDS
from rfproteina.disease.threshold_provenance import PROVENANCE_KEY

#: A residue run long enough to be a sequence rather than a word. Upper case
#: from nine, which is the shortest window the screen measures and is the case
#: every table and CIF reader normalises a peptide to before recording it. Any
#: case from twenty, which no ordinary word reaches.
#:
#: Nine letters in any case is not a sequence test. The twenty amino-acid codes
#: omit only B, J, O, U, X and Z, so "difference", "clearance", "percentage"
#: and "reference_proteins_pooled" all read as sequences. Against the ten
#: stored screens that rule matched twenty-one runs and every one was a field
#: name or an English word, while this one matches none of them and still
#: redacts a nine-residue epitope.
_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{9,}|(?i:[acdefghiklmnpqrstvwy]{20,})")
#: The same test for text a person wrote: a structure title, a database detail.
#: Titles are often all upper case, and the nine-letter rule turned KALLIKREIN
#: and TRANSFERASE into omitted sequences, which took the identity of the
#: structure with them. Twenty residues in any case is still removed.
_PROSE_SEQUENCE = re.compile(r"(?i:[acdefghiklmnpqrstvwy]{20,})")
#: ``screen_version`` and ``code`` say which record this is and which code
#: applied the cuts. A stored verdict is of unknown age without them.
#:
#: ``bar_axes`` through ``design_diversity`` are keys the current writer records
#: and this view filtered out: five of them arrived with the screen's own version 2 and ``refold`` is
#: the whole of version 3, so the view returned a version-1 verdict out of a
#: version-3 record and said ``screen_version: 3`` while doing it.
#:
#: ``bar_axes`` is the list the bar was read from at the moment of the verdict:
#: ``bar`` is a name, and a name does not survive an edit to the axes it names,
#: which is the hazard the key was written for. ``cuts`` is every number the
#: verdict was taken against, including the three plasma cuts a caller passes to
#: ``screen_directory`` -- a client could set them and never see them applied.
#: ``metrics_file`` says which per-design table the numbers came from, and
#: ``metrics_blocker`` why there were none, without which a refused table and a
#: run nobody scored are one answer; ``chain_blocker`` is the same for a run
#: whose table names two chains. ``refold`` is the whole of version 3: every
#: declared pass criterion, its bound, the worst value reached and which bounds
#: were missed, where a bare false ``ppi_pass`` named none of them.
#: ``solubility`` is a required key, so its absence from the view was a required
#: part of every record going unreported. ``design_diversity`` is the one
#: measurement about the set rather than a design, and the only thing standing
#: between a reader and eight copies of one molecule read as eight shots.
#:
#: What stays out stays out on purpose, and
#: ``tests/test_disease_mcp_record_keys.py`` makes each exclusion a named
#: decision rather than an omission: ``samples``, ``sequence_developability``
#: and ``designed_interface`` are per-design measurements the layer tools
#: return, ``inputs`` is a per-file listing whose moved entries already arrive
#: under ``replaced``, and ``chemical_liability`` and ``modality_screen`` are
#: counts with no state that enter no call.
#:
#: ``designed_interface_quality``, ``ligand_alerts``, ``mhc_class_i`` and
#: ``synthesis`` arrived with version 4. Each is a report that enters no call,
#: which by the rule above would keep it out; they are returned by decision, so
#: a client reads them without asking for the whole record. Three keys inside
#: them stay in screen.json, see ``_HELD_BACK``.
_VERDICT_FIELDS = (
    "call",
    "bar",
    "states",
    "untested",
    "hits",
    "ppi_pass",
    "ppi_blocker",
    "pre_oracle_pass",
    "requested_class",
    "target_accession",
    "job_mismatch",
    "screen_version",
    "code",
    "bar_axes",
    "cuts",
    "metrics_file",
    "metrics_blocker",
    "chain_blocker",
    "refold",
    "solubility",
    "protein_sol",
    "design_diversity",
    "designed_interface_quality",
    "prodigy_affinity",
    "vina_pose",
    "tnp_profile",
    "ligand_alerts",
    "ligand_solubility",
    "qupkake",
    "sygma",
    "method_selection",
    "mhc_class_i",
    "synthesis",
    # The last three were excluded by name and the reasons did not survive review.
    #
    # ``chemical_liability`` and ``modality_screen`` were kept out as "no state,
    # enters no call". That is the rule the four reports above it already
    # override, and for the same reason: a report recorded beside a verdict is
    # read by a client that has no cause to ask for the whole record. Returning
    # six no-call blocks and withholding two more on the ground that they are
    # no-call was the list disagreeing with itself.
    #
    # ``inputs`` was kept out because ``replaced.changed_inputs`` already names
    # the supplied files that moved. It names them only on a re-screen whose
    # verdict changed: a first screen writes no ``replaced`` at all, so the
    # predictor tables, panels and curated evidence a verdict rested on could
    # not be read back by any client at any time. ``screen.py`` puts it plainly
    # where it builds the block -- a verdict that cannot name its inputs cannot
    # be checked against them.
    "inputs",
    "chemical_liability",
    "modality_screen",
)
#: Keys a returned block leaves in screen.json. ``repeat_sequence`` is a verbatim
#: slice of the designed construct, and ``_redact`` removes a DNA slice but not an
#: RNA one, whose U is outside its alphabet; ``longest_repeat`` beside it carries
#: the length a reader acts on. ``smiles`` is the screened molecule, and the same
#: sequence rule reads a long carbon chain as a peptide, so a fatty acid came back
#: as ``[sequence omitted](=O)O``.
#:
#: ``reference`` is the third. ``mhc_tables.py`` stores it as ``str(path)`` of
#: the ``mhc_i_reference.tsv`` it wrote inside the run directory, which makes it
#: an absolute path identifying this checkout, and because it is a ``str`` and
#: not a ``Path`` it goes through the free-text scrubber rather than ``_shown``
#: -- which, as ``_shown``'s own docstring says, collapses two different run
#: directories into one string naming neither. Its sibling ``metrics_file`` is
#: relativised before it is stored (``screen.py:_metrics_name``); this one never
#: was. Whether the reference table exists still travels, under ``scan``.
_HELD_BACK: dict[str, frozenset[str]] = {
    "synthesis": frozenset({"repeat_sequence"}),
    "ligand_alerts": frozenset({"smiles"}),
    "ligand_solubility": frozenset({"structures"}),
    "qupkake": frozenset({"structures"}),
    "sygma": frozenset({"structures"}),
    "mhc_class_i": frozenset({"reference"}),
}
#: The two blocks that carry a nested ``provenance`` record, keyed by the thing
#: the threshold applies to. ``cuts`` gained one on every cut, and
#: ``designed_interface_quality`` on every interface bar. Neither is a new
#: top-level key, so the pin in ``test_disease_mcp_record_keys.py`` -- which
#: compares top-level names -- stayed green while the view returned the
#: numbers and dropped the account of where they came from. That account is
#: the whole point of the block: a reader cannot otherwise tell a published
#: cut from a default this program borrowed from another project and never
#: fitted, and three of the five interface bars are exactly that.
_PROVENANCE_BLOCKS = ("cuts", "designed_interface_quality")
#: What one provenance entry may carry. ``citation`` is present only on a
#: ``published`` kind, and an empty one is the point rather than an omission;
#: ``caveat`` says what the number does not cover; ``bar`` is the threshold
#: itself and is written only on the interface record, where the measurement
#: and its bar would otherwise sit in two different files.
_PROVENANCE_FIELDS = ("kind", "source", "citation", "caveat", "bar")
#: ``source`` says whether a state was measured or declared by a person in
#: ``axis_evidence.json``. A declared state keeps what the measurement said
#: under ``measured_state`` and ``measured_blocker``, and what it was declared
#: on under ``curated``. Without these four a declared clear over a measured
#: hit reads as a measured clear, beside the number that made it a hit.
#: ``scan_limit`` says a verdict rests on a scan that did not cover the whole
#: chain, and ``threshold`` with ``threshold_source`` say what the axis was
#: judged against and where that cut came from. Each was added so a reader
#: could see what a number rests on, which it cannot do if the view drops them.
#:
#: By that same test the last eight belong here too. ``reference_density`` is the
#: bar a density verdict was taken against, without which a clear is a number
#: beside no comparison. ``units`` says what the measurement is in.
#: ``unscored_peptides`` counts windows the predictor declined, which a clear
#: rests on having none of. ``not_nm_columns`` names affinity columns whose
#: unit the table never stated, so they were not read. ``presented_complete``
#: says whether the presented set covered the design. ``fold_source`` is the
#: study a plasma-fold verdict cites, which that axis refuses to be measured
#: without -- so it is precisely what the number rests on.
#:
#: ``reference_rank_scale_ambiguous`` is the sharpest of them: it says the bar
#: this verdict was taken against may be a hundredfold out, because every
#: percentile rank in the reference reads the same whether that column is a
#: percentage or a fraction of one. The verdict still stands, because refusing
#: would refuse the ordinary binders-only export, and that is exactly why the
#: disclosure has to travel beside it rather than stay in the file.
#: ``reference_proteins_too_short`` names reference proteins that were never in
#: the running for the bar, which a reader cannot otherwise tell apart from one
#: that was measured and carried no binder.
#:
#: This stays an allowlist of what QUALIFIES a verdict, not a copy of the
#: record. Bulk evidence -- the peptide, allele and paralog listings -- stays in
#: screen.json, where a reader who wants it can go.
#: The last fourteen are the same species as the ones above them and were
#: missing, which is what a list that stopped being updated looks like: every
#: pair below was one decision, and only one half of it was here.
#:
#: ``threshold_source`` and ``fold_source`` were kept while ``method`` -- which
#: computation, or which supplied table, produced the measurement at all -- was
#: dropped, so a score a person left beside the run read as one this program
#: computed. ``scan_limit`` says a scan did not cover the whole chain while
#: ``covered_examples`` and ``uncovered_examples`` say which designs it covered
#: at all: one supplied table is one scan, and a run of eight could read
#: ``clear`` on a scan of one. ``unscored_peptides`` counts windows a predictor
#: declined while ``mask_unpublished_alleles`` names molecules nothing published
#: can score, and ``presentation`` says whether presentation was configured.
#: ``reference_density`` is the bar a density verdict was taken against while
#: ``reference_file``, ``reference_digest`` and ``reference_bases`` say which
#: file that bar came from and how large it was -- a clear against a stub and a
#: clear against a transcriptome read identically without them. ``units`` was
#: kept and ``assay`` dropped, so the view returned what a concentration is in
#: and not how it was obtained; ``network``, ``confidence`` and ``release`` are
#: the same for a partner count, which is a number about whatever the cut
#: admitted, from a database that re-derives the network each release.
#: ``displaced_method`` and ``displaced_table`` are why an axis reads untested
#: in a directory that holds the data for a verdict.
#:
#: The allowlist is still an allowlist. The bulk evidence stays in screen.json:
#: the peptide, allele, ligand and paralog listings, the per-window scores, the
#: cluster id, the mask vector.
#:
#: The last fifteen were added against one test, applied after the producers
#: moved again: a key qualifies when it is a scalar (or a list of column names,
#: as ``not_nm_columns`` is) AND it pairs with a name already here, written by
#: the same function, often the next line. ``_mhc_record`` writes five keys in
#: a row -- ``not_nm_columns``, ``rank``, ``ic50``, ``trigger``,
#: ``unscored_peptides`` -- and three of them were here; ``rank``, ``ic50`` and
#: ``trigger`` say whether ``measurement`` is a percentile rank or an affinity
#: in nM, and those axes write no ``units`` and no ``threshold`` at all, so
#: without them a clear is a bare number on an unstated scale. ``seed_width``
#: and ``seeds_searched`` sit in the dict literal that ``reference_file``,
#: ``reference_digest`` and ``reference_bases`` came from, and are the width
#: and the denominator a zero-match clear is a count over.
#: ``reference_method_undeclared`` is the twin of
#: ``reference_rank_scale_ambiguous``: the tool identity behind the bar was
#: taken on trust. ``proteome_kmers`` and ``proteome_digest`` are which
#: proteome file an absence claim was made against and how large it was, which
#: is the argument already accepted for the oligo reference.
#: ``unparsed_columns`` is the sibling of ``not_nm_columns``.
#: ``qualifier`` says a panel cell was a bound such as "> 10000" while
#: ``measurement`` stays the figure the table printed, and ``n_readings`` how
#: many cells the lowest was taken from. ``calis_mask_allele_specific`` is the
#: correction to ``method``, and ``binding_peptides`` the numerator of the
#: density whose bar ``reference_density`` already carries. ``compara_examined``
#: is the count the paralog axis's only clear rests on, and ``compara_blocker``
#: why that pass never measured anything.
#:
#: What that test excludes, and why they are not here: the protein, allele,
#: method and width listings (``reference_proteins``, ``reference_proteins_pooled``,
#: ``design_method``, ``reference_method``, ``design_alleles``, ``missing_alleles``),
#: and the per-item maps (``reference_densities``, ``stated_length``,
#: ``observed_extent``, ``motif_sites``, ``transmembrane_spans``). A listing is
#: what screen.json is for.
#:
#: The final sixteen complete the same sweep. Nine are counts that say how wide
#: a clear is: ``alleles`` and ``peptides_scored`` are the panel and the set the
#: median rule was applied over, so a clear over two peptides and one over
#: sixty-two stop reading alike; ``binding_alleles`` is how close the nearest
#: peptide came; ``presented_ligands`` is how many presented windows were
#: scored at all; ``positive_windows`` and ``promiscuous_peptides`` are the
#: numbers behind a ``no_pass_cut``; ``cpg_count`` is what an undecided innate
#: axis turns on; ``window_residues`` is the width a span count was taken at,
#: as ``seed_width`` is for the oligo scan. ``failed`` names the designs that
#: made a run-level axis a hit -- the hit-side twin of ``covered_examples``.
#:
#: ``positive_examples`` is the tenth, and the names behind ``positive_windows``
#: rather than another count: ``liability.py`` writes it on the line after that
#: count in every branch that sets one. A count says how many windows reached
#: the cut, which leaves a reader who wants to look at one with nothing to look
#: up -- the same gap ``covered_examples`` and ``failed`` were added to close
#: beside their own counts.
#:
#: The last six are one stored note that no client has ever seen: the
#: co-evolution note ``read_paralog_coevolution`` merges onto the paralog axis.
#: ``coevolution_signal`` is the key that selects the sentence a reader is meant
#: to get, ``coevolution_blocker`` is why there is no note, and ``same_shape``,
#: ``coevolution_level``, ``other_species_toxicity`` and ``coevolution_rows``
#: are what the note rests on. Its two listings -- ``coevolution_signals`` and
#: ``coevolution_species`` -- stay in screen.json with the rest of the bulk.
_AXIS_FIELDS = (
    "state",
    "measurement",
    "blocker",
    "source",
    "curated",
    "measured_state",
    "measured_blocker",
    "scan_limit",
    "threshold",
    "threshold_source",
    "reference_density",
    "units",
    "unscored_peptides",
    "not_nm_columns",
    "presented_complete",
    "fold_source",
    "fold_column",
    "linear_fold_magnitude",
    "reference_rank_scale_ambiguous",
    "reference_proteins_too_short",
    "method",
    "presentation",
    "mask_unpublished_alleles",
    "displaced_method",
    "displaced_table",
    "covered_examples",
    "uncovered_examples",
    "assay",
    "network",
    "confidence",
    "release",
    "reference_file",
    "reference_digest",
    "reference_bases",
    "trigger",
    "rank",
    "ic50",
    "binding_peptides",
    "reference_method_undeclared",
    # Which predictor produced a measured class I or immunogenicity verdict.
    # The undeclared and mismatched paths were disclosed here and the standing
    # verdict was not, so the only record carrying a measurement was the only
    # one that did not say what measured it.
    "design_method",
    "reference_method",
    "restricting_method",
    "proteome_kmers",
    "proteome_digest",
    "seed_width",
    "seeds_searched",
    "qualifier",
    "n_readings",
    "unparsed_columns",
    "calis_mask_allele_specific",
    "compara_examined",
    "compara_scored",
    "compara_blocker",
    "alleles",
    "peptides_scored",
    "binding_alleles",
    "presented_ligands",
    "positive_windows",
    "positive_examples",
    "promiscuous_peptides",
    "cpg_count",
    "window_residues",
    "failed",
    "same_shape",
    "coevolution_signal",
    "coevolution_level",
    "coevolution_rows",
    "other_species_toxicity",
    "human_toxicity_state",
    "coevolution_blocker",
)
#: The groups of a screen that hold one record per axis. ``chemical_liability``
#: is not one of them: it is the first design's motif counts and charge, it has
#: no state, and those numbers do not clear an axis.
_AXIS_GROUPS = ("sequence_liability", "target_liability", "nucleic_liability", "toxicology")
_CURATED_SOURCE = "axis_evidence"
#: Directories this server may use besides the checkout, separated the way
#: ``PATH`` is. The MCP entry point defaults it to the checkout; direct library
#: callers may leave it unset and retain the historical absolute-path behavior.
_ROOTS_ENV = "RFD4_MCP_ROOTS"
_BROWSER_ONLY_OPTIONS = {"ui", "validate_output"}
_VALIDATION_PAGE_SIZE = 100
_VALIDATION_BYTE_BUDGET = 256 * 1024 * 1024
_VALIDATION_REPORT_DIR = "validation_reports"
_VALIDATION_REPORT_NAME = re.compile(r"validation-[0-9a-f]{32}\.json")
_VALIDATION_REPORT_ENTRY_LIMIT = 512
_SECRET_WORD = re.compile(r"(?i)key|token|secret|password")
_OUTPUT_PAGE_SIZE = 100
_OUTPUT_ENTRY_LIMIT = 2_000
_OUTPUT_DIRECTORY_LIMIT = 512
_OUTPUT_READ_DEFAULT = 64 * 1024
_OUTPUT_READ_MAX = 1024 * 1024
_OUTPUT_PATH_MAX = 512
_OUTPUT_CLASSIFY_MAX = 256 * 1024 * 1024
_OUTPUT_SCAN_CHUNK = 64 * 1024
_OUTPUT_SCAN_OVERLAP = 4 * 1024
_SCREEN_JSON_MAX = 16 * 1024 * 1024
_CAPTURE_MAX_CHARS = 4 * 1024 * 1024
_CAPTURE_MAX_LINES = 4_096
_CAPTURE_LINE_MAX = 64 * 1024
_CAPTURE_MARKER = "[program output truncated by MCP]"
_MAX_REMOTE_HOSTS = 64
_REMOTE_HOST = re.compile(
    r"(?:(?:[A-Za-z0-9_][A-Za-z0-9_.-]{0,63})@)?"
    r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]{0,252}|\[[0-9A-Fa-f:.%]+\])"
)
_REMOTE_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9._-]{0,127}")
_SENSITIVE_ARTIFACT = re.compile(
    r"(?i)(?:^|[._-])(?:api[_-]?keys?|tokens?|secrets?|passwords?|credentials?|authorization)(?:[._-]|$)"
)
_PRIVATE_KEY_NAME = re.compile(r"(?i)^(?:id_(?:rsa|dsa|ecdsa|ed25519)|ssh_host_[A-Za-z0-9_]+_key)$")
_SECRET_BYTES = re.compile(
    rb"(?:sk-|xai-|nvapi-|hf_|ghp_|github_pat_|AKIA)[A-Za-z0-9_./+=-]{8,}"
    rb"|Authorization\s*:\s*(?:Bearer|Basic)\s+\S{4,}"
    rb"|(?:api[_-]?key|access[_-]?token|token|secret|password)[\"']?\s*[:=]\s*[\"']?[^\s\"']{4,}"
    rb"|-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
    re.IGNORECASE,
)
SCREEN_FIELDS = frozenset(
    {
        "directory",
        "screen_selection",
        "screen_methods",
        "binder_kind",
        "bar",
        "proteome",
        "lookup_targets",
        "plasma_abundance_max",
        "plasma_hub_max",
        "plasma_fold_max",
        "follow_up",
        "stage_tables",
        "out_dir",
        "log_file",
        "include_record",
        "paralog_identity_max",
    }
)
#: The options only the refold mode reads. ``cli._refold_only`` is the one
#: function in the program that looks at any of them, so passed without
#: ``refold`` they configure nothing at all -- and a client that set one would
#: read its own call as having asked for something.
_REFOLD_OPTIONS = frozenset({"refold", "refold_dir", "refold_import", "refold_oracle"})
_HEADLESS_MODES = frozenset(
    {"declare_corpora", "benchmark_answer", "dynamics", "human_protocol", "screen", "refold"}
)
_MODE_OPTIONS = {
    "benchmark_answer": frozenset({"benchmark_structure"}),
    "dynamics": frozenset(
        {"dynamics_chain", "dynamics_task", "dynamics_frames", "rmsf_max", "mobile_fraction_max", "drift_max"}
    ),
    "human_protocol": frozenset({"not_binding", "human_chain", "resemblance_identity_max"}),
    "declare_corpora": frozenset({"corpora_against"}),
    "refold": _REFOLD_OPTIONS - {"refold"},
}
#: ``refold_oracle`` is a name, not a path, so it does not go through
#: :func:`_path` -- and it is written into every filename the refold mode files
#: a fold under (``PathOracle.output_stem`` builds ``<example>_<oracle>`` and
#: joins it under the run). A separator or a ``..`` in it would put those files
#: outside the run directory, which is the one thing every path argument here
#: is checked for. Opening this mode to a client is what makes it reachable, so
#: it is checked here.
_ORACLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_LAYER_TOOLS = (
    "describe_architecture",
    "retrieve_databases",
    "retrieve_structures",
    "read_sequence_measures",
    "read_results",
    "backend_status",
    "declare_corpora",
)
TOOL_NAMES = (
    "list_options", "run", "screen_directory", "read_verdict", *_LAYER_TOOLS[:5],
    "list_output_files", "read_output_file", *_LAYER_TOOLS[5:],
    "plan_validation", "validate_output_files", "save_validation_report", "list_validation_reports",
    "plan_screening", "run_selected_screens",
)
#: Called with each finished line the program prints, as it is printed rather
#: than when the run ends. Raising from it abandons the run, which is how a
#: cancelled call stops.
ProgressLine = Callable[[str], None]
StructuredResult = Callable[[str, object], None]


class _Unanswered(BaseException):
    """The program asked for a disease name, and this server has no one to ask.

    ``cli.main`` ends with ``except Exception``. An ordinary exception raised
    from the prompt would be logged as an unexpected failure and the loop would
    keep asking. This passes through that handler.
    """


def call_tool(
    name: str,
    arguments: Mapping[str, object] | None,
    *,
    on_line: ProgressLine | None = None,
) -> dict[str, object]:
    """Run one tool. ``arguments`` is the MCP object, which may be empty.

    ``on_line`` is called with each line the program finishes printing, while it
    is still running. The caller turns those into progress notifications; the
    lines are also returned at the end, so nothing is only reported live.
    """
    supplied = {} if arguments is None else arguments
    if name == "list_options":
        _reject_unknown(supplied, set())
        return list_options()
    if name == "run":
        return run_program(supplied, on_line=on_line)
    if name == "screen_directory":
        return screen_directory(supplied, on_line=on_line)
    if name in {"plan_screening", "run_selected_screens"}:
        return _screening_panel(name, supplied, on_line=on_line)
    if name == "read_verdict":
        return read_verdict(supplied)
    if name == "list_output_files":
        return list_output_files(supplied)
    if name == "read_output_file":
        return read_output_file(supplied)
    if name == "plan_validation":
        return plan_validation(supplied)
    if name == "validate_output_files":
        return validate_output_files(supplied)
    if name == "save_validation_report":
        return save_validation_report(supplied)
    if name == "list_validation_reports":
        return list_validation_reports(supplied)
    if name in _LAYER_TOOLS:
        from rfproteina.disease.mcp_layers import call_layer

        return call_layer(name, supplied)
    raise ValueError("Unknown tool.")


def stop_work() -> int:
    """End any generation child the call in flight started, and say how many.

    Cancelling used to be noticed only when the program next printed a line, and
    generation prints nothing from the moment it starts its requests until every
    child has exited. So a cancellation during the one step that takes hours did
    nothing at all. Ending the child makes the request fail, which the program
    reports in the ordinary way and this server then declines to answer.
    """
    from rfproteina.disease.dispatch import stop_running

    return stop_running()


def list_options() -> dict[str, object]:
    """Every option the program accepts, under the name ``run`` expects."""
    from rfproteina.disease.cli import build_parser

    options: list[dict[str, object]] = []
    for action in build_parser()._actions:
        if not action.option_strings or action.dest in {"help", *_BROWSER_ONLY_OPTIONS}:
            continue
        entry: dict[str, object] = {
            "name": action.dest,
            "flag": _flag(action),
            "kind": _kind(action),
            "help": action.help or "",
        }
        if action.choices is not None:
            entry["choices"] = [str(choice) for choice in action.choices]
        default = _default(action)
        if default is not None:
            entry["default"] = default
        options.append(entry)
    return {
        "program": PROGRAM,
        "options": options,
        # The same options as a schema, so an agent can validate a run call
        # before making it. It rides here rather than in the tool definition
        # because building it imports the package, which costs about fifty
        # seconds, and tool discovery has to answer at once.
        "schema": options_schema(),
        "notes": [
            "Pass these names to run. A false flag is omitted. null omits the option.",
            "screen and the browser panel cannot be used together.",
            (
                "run takes one mode per call: the disease pipeline, benchmark answer, dynamics, "
                "H.U.M.A.N. protocol, screen, refold, or corpus declaration."
            ),
            (
                "refold prepares a finished run for folding elsewhere, or files the folds that came back, "
                "and returns that record as refold. refold_dir, refold_import and refold_oracle need it."
            ),
            "run starts generation only when run is true. confirm_inference writes jobs.",
            "This server does not prompt. Pass accept_only, or choice when several names match.",
            "Without confirm_inference, a finished download does not start generation.",
            "Screen a directory with screen_directory. Read a stored screen with read_verdict.",
            "Use plan_screening to inspect recommendations and run_selected_screens to execute a selection.",
            "Use plan_validation and validate_output_files for the output validation panel's local checks.",
            "Use save_validation_report to run selected checks and save a JSON report inside the run.",
            "Use list_validation_reports to find saved report paths for the run.",
            "A verdict call of screened_clear is every axis in that bar clear.",
            "An axis with source axis_evidence was declared by a person. It was not measured.",
            "A relative path is inside this checkout and cannot leave it.",
            (
                "The launched MCP server confines every path to this checkout by default. "
                f"An operator may explicitly widen that boundary with {_ROOTS_ENV}."
            ),
            "API keys are not an option.",
            "describe_architecture names the AI interface, the MCP tools, and the computational backend.",
        ],
    }


def options_schema() -> dict[str, object]:
    """JSON Schema for the ``run`` tool's ``options`` object.

    Built from the same parser the command builds, so an agent selects an
    option from the tool definition itself rather than calling
    ``list_options`` first and guessing the type, and a client can reject a
    bad option before the call instead of after it. ``list_options`` stays the
    readable form of the same facts. A name that reads as a secret is left out,
    so the schema can never advertise a place to put a key.
    """
    from rfproteina.disease.cli import build_parser

    properties: dict[str, object] = {}
    for action in build_parser()._actions:
        if not action.option_strings or action.dest in {"help", *_BROWSER_ONLY_OPTIONS}:
            continue
        if _SECRET_WORD.search(action.dest):
            continue
        properties[action.dest] = _option_schema(action)
    return {
        "type": "object",
        "description": (
            "Options by the name the command uses. Omit an option to leave it unset; "
            "false omits a flag. The browser panel and API keys are not options here."
        ),
        "properties": properties,
        "additionalProperties": False,
    }


def _option_schema(action: argparse.Action) -> dict[str, object]:
    """One option as a schema entry, typed the way :func:`_encode` accepts it."""
    kind = _kind(action)
    entry: dict[str, object] = {"type": kind}
    if kind == "array":
        entry["items"] = {"type": "string"}
    elif action.type is Path:
        entry["pattern"] = r".*\S.*"
    if action.choices is not None:
        entry["enum"] = [str(choice) for choice in action.choices]
    if action.help:
        entry["description"] = action.help
    return entry


def run_program(arguments: Mapping[str, object], *, on_line: ProgressLine | None = None) -> dict[str, object]:
    """Run the program in one of its modes and return what that mode produced.

    The disease pipeline needs a disease and, because this server has nobody to
    ask, ``accept_only`` or ``choice``. The CLI's other headless modes prompt for
    neither. Their option paths are the same ones the ordinary command uses.
    """
    from rfproteina.disease.ui import wants_option_ui

    if "options" not in arguments:
        raise ValueError("Pass options, the object list_options describes.")
    extra = set(arguments) - {"options"}
    if extra:
        raise ValueError("run takes options only.")
    options = arguments["options"]
    if not isinstance(options, Mapping):
        raise ValueError("options must be an object.")
    if _BROWSER_ONLY_OPTIONS.intersection(options):
        raise ValueError("This server does not open the browser panel.")
    _reject_secrets(options)
    # Encode first so the mode guards use values that have already passed the
    # published type, choice and path checks. False boolean flags encode to
    # nothing and therefore do not select a mode.
    argv = argv_from_options(options)
    _validate_remote_options(options)
    # An option whose value is null is not an option: :func:`argv_from_options`
    # leaves it out. Tested for presence instead, ``screen: null`` read as a
    # screen to both guards below and reached the program as neither -- so the
    # call went down the disease pipeline with no ``accept_only``, which is the
    # one thing the second guard exists to prevent, performed a live disease
    # search, and then met a confirmation prompt this server cannot answer.
    supplied = {name for name, value in options.items() if value is not None and value is not False}
    selected_modes = sorted(_HEADLESS_MODES & supplied)
    if len(selected_modes) > 1:
        raise ValueError(f"Choose one program mode, rather than: {', '.join(selected_modes)}.")
    mode = _run_mode(supplied)
    for owner, companions in _MODE_OPTIONS.items():
        stray = sorted(companions & supplied) if owner not in supplied else []
        if stray:
            raise ValueError(f"These options need {owner}, and nothing else reads them: {', '.join(stray)}.")
    oracle = options.get("refold_oracle")
    if isinstance(oracle, str) and _ORACLE_NAME.fullmatch(oracle) is None:
        raise ValueError(
            "refold_oracle is a plain name: a letter or digit, then letters, digits, dot, dash or underscore."
        )
    refold = options.get("refold")
    if mode == "refold" and isinstance(refold, str) and refold.strip():
        _require_finished_run(_path(refold))
    # Only the disease pipeline searches, confirms and can ask a question.
    pipeline = not _HEADLESS_MODES & supplied
    if pipeline and "disease" not in supplied:
        raise ValueError("Set disease or select one of the headless modes listed by list_options.")
    if pipeline and not options.get("accept_only") and "choice" not in supplied:
        raise ValueError("Pass accept_only or choice. This server does not prompt.")
    if wants_option_ui(argv):
        raise ValueError("This server does not open the browser panel. Set the options here.")
    structured: dict[str, object] = {}

    def record(name: str, value: object) -> None:
        if name in {"structure_options", "dynamics", "human_protocol"}:
            structured[name] = _redact(value)

    code, messages = _call_main(argv, on_line=on_line, on_record=record)
    payload: dict[str, object] = {"program": PROGRAM, "mode": mode, "exit_code": code, "messages": messages}
    payload.update(structured)
    if mode == "screen" and isinstance(options["screen"], str):
        payload["verdict"] = _screens(_path(options["screen"]), include_record=False)
    if mode == "refold":
        # The originals, not the redacted lines in the payload: see
        # :func:`_printed_json`. A stand-in for the program carries none, and
        # then no record is claimed.
        report = _printed_json(list(getattr(messages, "raw", ())))
        if report is not None:
            payload["refold"] = report
    if mode == "benchmark_answer":
        answers = _printed_json_array(list(getattr(messages, "raw", ())))
        if answers is not None:
            payload["benchmark_answers"] = answers
    if mode == "declare_corpora" and code == 0:
        from rfproteina.disease.mcp_layers import declare_corpora

        against = options.get("corpora_against")
        payload["corpora"] = declare_corpora({"against": against} if isinstance(against, str) else {})
    return payload


def _run_mode(supplied: set[str]) -> str:
    """The CLI branch that will consume this validated option set."""
    for name in ("declare_corpora", "benchmark_answer", "dynamics", "human_protocol", "screen", "refold"):
        if name in supplied:
            return name
    return "disease_pipeline"


def _require_finished_run(path: Path) -> None:
    """Refuse a refold directory that is not itself a finished run.

    ``_path`` normalises a relative path against the checkout, so
    ``{"refold": "."}`` named the checkout root: a directory, so every guard
    passed, and the program then made ``refold_requests`` beside the source
    tree and reported nothing in it. The export reads designs from
    ``<run>/inference_samples``, so a parent of runs exports nothing either and
    says so only as a count of zero.

    The test for a finished run belongs to the screen, which owns it, rather
    than being spelled a second time here: ``generation_directories`` answers
    with the directory itself the moment it holds designs or a metrics table,
    and walks below it only when it does not.
    """
    from rfproteina.disease.screen import ScreenWalkTruncatedError, generation_directories

    try:
        found = generation_directories(path)
    except ScreenWalkTruncatedError as stopped:
        found = stopped.found
    if path not in found:
        raise ValueError("refold takes one finished generation directory, the run its designs were written in.")


def _printed_json(lines: list[str]) -> dict[str, object] | None:
    """The JSON document the program printed, read back as an object.

    The refold mode's entire result is one printed document -- what was
    requested, what came back, what is still outstanding -- and the capture
    hands the lines over one at a time, so a client that could have read the
    record got its fragments instead. It is printed as JSON precisely because
    the sequel to that command is usually another command.

    ``lines`` are the raw lines, not the ones returned in ``messages``. Those
    have each been through the free-text scrubber, and reassembling a document
    from them handed back something that parsed and was not what was printed:
    ``scrub_for_upload`` replaces a home segment wherever it sits, so a run at
    a directory with an internal home segment came back as a redacted label, not a path
    that exists, and the same string for another person's identical run, which
    is the very fault ``scrub_path_prefix`` was written to fix. The
    nine-residue rule did the same to an example id: ``KALLIKREIN_A_0`` became
    ``[sequence omitted]_A_0``. The document is parsed first and redacted as a
    record by :func:`_refold_record`, where a path can be scrubbed as a path.

    The lines arrive stripped, which costs the indentation and nothing else: a
    document is bounded by its own first ``{`` and its last ``}``, and the
    nested closes in between are never the last line. None when nothing here
    parses, and the printed lines are still returned in ``messages`` either
    way, so a document this cannot read is not a document lost.
    """
    try:
        start = lines.index("{")
    except ValueError:
        return None
    end = max((index for index, line in enumerate(lines) if line == "}"), default=-1)
    if end <= start:
        return None
    try:
        found = json.loads("".join(lines[start : end + 1]))
    except ValueError:
        return None
    return _refold_record(found) if isinstance(found, dict) else None


def _printed_json_array(lines: list[str]) -> list[object] | None:
    """A JSON array printed by a headless mode, with returned prose scrubbed."""
    try:
        start = lines.index("[")
    except ValueError:
        return None
    end = max((index for index, line in enumerate(lines) if line == "]"), default=-1)
    if end <= start:
        return None
    try:
        found = json.loads("".join(lines[start : end + 1]))
    except ValueError:
        return None
    if not isinstance(found, list):
        return None
    redacted = _redact(found)
    return redacted if isinstance(redacted, list) else None


#: The keys of the refold record whose value is a path on this computer.
#: ``cli._refold_only`` stores them with ``str(path)``.
_REFOLD_PATHS = frozenset({"run", "requests"})


def _refold_record(record: dict[str, object]) -> dict[str, object]:
    """Redact the refold document as a record rather than as free text.

    A path goes through :func:`_shown`, which replaces a personal prefix and
    leaves the rest of the path alone, so what comes back still names the
    directory a client would pass to its next call. Everything else is held to
    the twenty-residue rule, because the strings here are example ids and
    filenames, and a design called ``KALLIKREIN_A_0`` is not a sequence.
    """
    found: dict[str, object] = {}
    for key, value in record.items():
        if key in _REFOLD_PATHS and isinstance(value, str) and value:
            found[str(key)] = _shown(Path(value))
            continue
        found[str(key)] = _redact(_as_prose(value))
    return found


def _as_prose(value: object) -> object:
    """The same value with every string marked as text a person wrote."""
    if isinstance(value, str):
        return _Prose(value)
    if isinstance(value, Mapping):
        return {key: _as_prose(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_as_prose(item) for item in value]
    return value


def screen_directory(arguments: Mapping[str, object], *, on_line: ProgressLine | None = None) -> dict[str, object]:
    """Screen one directory, or every finished run beneath it, and return the verdicts.

    Nothing is generated or downloaded. Target lookups run only when
    ``lookup_targets`` is true. ``follow_up`` writes the next commands and
    does not start them.
    """
    _reject_unknown(arguments, SCREEN_FIELDS)
    if "screen_methods" in arguments:
        from rfproteina.disease.screen_selection import METHOD_IDS

        methods = arguments["screen_methods"]
        if not isinstance(methods, list) or not all(isinstance(k, str) and k in METHOD_IDS for k in methods):
            raise ValueError("Choose a list of known screen method IDs.")
        if len(set(methods)) != len(methods) or arguments.get("screen_selection") not in (None, "explicit"):
            raise ValueError("Choose distinct individual screen methods or a screening preset.")
    if "screen_selection" in arguments and arguments["screen_selection"] not in (
        "explicit", "recommended", "all_installed",
    ):
        raise ValueError("Choose a named screening preset.")
    directory = arguments.get("directory")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("Name the directory to screen.")
    include_record = _optional_flag(arguments.get("include_record"))
    root = _path(directory)
    if not root.is_dir():
        raise ValueError("That directory is not on this computer.")
    options: dict[str, object] = {"screen": directory}
    if "screen_methods" in arguments:
        options["screen_selection"] = "explicit"
    for name in SCREEN_FIELDS - {"directory", "include_record"}:
        if name in arguments and arguments[name] is not None:
            options[name] = arguments[name]
    argv = argv_from_options(options)
    # Taken before the call, so a screen the call failed to replace is not
    # reported as the verdict this call wrote.
    files = _screen_files(root)
    before = {label: _file_state(path) for label, path in files.items()}
    code, messages = _call_main(argv, on_line=on_line)
    body = _screens(root, include_record=include_record)
    _mark_written(body, files, before)
    body["exit_code"] = code
    body["messages"] = messages
    return body


def read_verdict(arguments: Mapping[str, object]) -> dict[str, object]:
    """Return stored screens under a directory. Nothing is written or rescreened."""
    allowed = {"directory", "include_record"}
    _reject_unknown(arguments, allowed)
    directory = arguments.get("directory")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("Name the directory to read.")
    root = _path(directory)
    if not root.is_dir():
        raise ValueError("That directory is not on this computer.")
    return _screens(root, include_record=_optional_flag(arguments.get("include_record")))


def _output_scope(arguments: Mapping[str, object]) -> Path:
    """One concrete run under an explicitly approved output root."""
    path, _candidate = _validation_scope({"directory": arguments.get("directory")})
    return path


def _sensitive_artifact(relative: PurePosixPath) -> bool:
    """Whether a run-local file name looks like configuration or a credential."""
    for part in relative.parts:
        lowered = part.casefold()
        if part.startswith(".") or lowered in {"api_keys.json", "master.key"}:
            return True
        if (
            lowered.endswith((".key", ".pem", ".p12", ".pfx", ".ppk"))
            or _PRIVATE_KEY_NAME.fullmatch(part)
            or _SENSITIVE_ARTIFACT.search(part)
        ):
            return True
    return False


def _artifact_media_type(relative: PurePosixPath) -> str:
    name = relative.name.casefold()
    if name.endswith(".json"):
        return "application/json"
    if name.endswith((".csv", ".tsv", ".txt", ".log", ".fasta", ".fa", ".yaml", ".yml")):
        return "text/plain"
    if name.endswith((".cif", ".mmcif", ".pdb")):
        return "chemical/x-structure"
    if name.endswith(".gz"):
        return "application/gzip"
    if name.endswith(".zst"):
        return "application/zstd"
    return "application/octet-stream"


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    """Count entries as they are examined and return a complete, ordered inventory."""
    found: list[dict[str, object]] = []
    examined = 0
    directories_seen = 0
    changed = "The run changed during inventory. Request a fresh listing."

    def visit(descriptor: int, prefix: PurePosixPath) -> None:
        nonlocal examined, directories_seen
        directories_seen += 1
        if directories_seen > _OUTPUT_DIRECTORY_LIMIT:
            raise ValueError("The run has too many directories to inventory in one call.")
        before = os.fstat(descriptor)
        with os.scandir(descriptor) as entries:
            for entry in entries:
                examined += 1
                if examined > _OUTPUT_ENTRY_LIMIT + _OUTPUT_DIRECTORY_LIMIT:
                    raise ValueError("The run has too many directory entries to inventory in one call.")
                if _sensitive_artifact(PurePosixPath(entry.name)):
                    continue
                relative = prefix / entry.name
                details = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(details.st_mode):
                    child = _open_directory(entry.name, dir_fd=descriptor)
                    try:
                        if not os.path.samestat(details, os.fstat(child)):
                            raise ValueError(changed)
                        visit(child, relative)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(details.st_mode):
                    if len(found) >= _OUTPUT_ENTRY_LIMIT:
                        raise ValueError("The run has too many output files to inventory in one call.")
                    found.append(
                        {
                            "path": relative.as_posix(),
                            "bytes": details.st_size,
                            "media_type": _artifact_media_type(relative),
                        }
                    )
        after = os.fstat(descriptor)
        if (before.st_mtime_ns, before.st_ctime_ns) != (after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(changed)

    try:
        root_descriptor = _open_directory(root)
        try:
            visit(root_descriptor, PurePosixPath())
            if not os.path.samestat(os.fstat(root_descriptor), root.stat()) or _resolved(root) != root:
                raise ValueError(changed)
        finally:
            os.close(root_descriptor)
    except OSError:
        raise ValueError("The full run inventory could not be read. Request a fresh listing.") from None
    return sorted(found, key=lambda item: str(item["path"]))


def _page_value(arguments: Mapping[str, object], name: str, default: int, maximum: int | None = None) -> int:
    value = arguments.get(name, default)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        bound = f" no greater than {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a nonnegative integer{bound}.")
    return value


def _expected_sha256(arguments: Mapping[str, object]) -> str | None:
    if "expected_sha256" not in arguments:
        return None
    value = arguments["expected_sha256"]
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest from the previous response.")
    return value


def list_output_files(arguments: Mapping[str, object]) -> dict[str, object]:
    """List bounded, non-secret regular artifacts under one finished run."""
    _reject_unknown(arguments, {"directory", "offset", "limit", "expected_sha256"})
    root = _output_scope(arguments)
    offset = _page_value(arguments, "offset", 0)
    limit = _page_value(arguments, "limit", _OUTPUT_PAGE_SIZE, _OUTPUT_PAGE_SIZE)
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    expected = _expected_sha256(arguments)
    inventory = _artifact_inventory(root)
    encoded = json.dumps(inventory, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    inventory_digest = hashlib.sha256(encoded).hexdigest()
    if expected is not None and inventory_digest != expected:
        raise ValueError("The inventory has changed since the previous page. Restart at offset zero.")
    page = inventory[offset : offset + limit]
    next_offset = offset + len(page) if offset + len(page) < len(inventory) else None
    return {
        "program": PROGRAM,
        "directory": _shown(root),
        "files": page,
        "next_offset": next_offset,
        "total_files": len(inventory),
        "inventory_sha256": inventory_digest,
    }


def _relative_output_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > _OUTPUT_PATH_MAX:
        raise ValueError("Name a short output path returned by list_output_files.")
    _plain(value)
    if "\\" in value:
        raise ValueError("Use the POSIX-style path returned by list_output_files.")
    relative = PurePosixPath(value)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("The output path must stay inside the finished run.")
    if _sensitive_artifact(relative):
        raise ValueError("Credential and private-key files are not output artifacts.")
    return relative


def _scan_output_chunk(descriptor: int, size: int, offset: int, count: int) -> tuple[bytes, str]:
    """Hash and classify the whole file, retaining the requested slice from those same bytes."""
    if size > _OUTPUT_CLASSIFY_MAX:
        raise ValueError("The output is too large to classify safely for MCP reading.")
    digest = hashlib.sha256()
    data = bytearray()
    overlap = b""
    position = 0
    while position < size:
        chunk = os.read(descriptor, min(_OUTPUT_SCAN_CHUNK, size - position))
        if not chunk:
            raise ValueError("The output changed while it was being read.")
        digest.update(chunk)
        combined = overlap + chunk
        if _SECRET_BYTES.search(combined):
            raise ValueError("The output appears to contain a credential and was not returned.")
        overlap = combined[-_OUTPUT_SCAN_OVERLAP:]
        start = max(offset, position)
        stop = min(offset + count, position + len(chunk))
        if start < stop:
            data.extend(chunk[start - position : stop - position])
        position += len(chunk)
    return bytes(data), digest.hexdigest()


def read_output_file(arguments: Mapping[str, object]) -> dict[str, object]:
    """Read one bounded chunk from a regular run artifact without following links."""
    _reject_unknown(arguments, {"directory", "path", "offset", "max_bytes", "expected_sha256"})
    root = _output_scope(arguments)
    relative = _relative_output_path(arguments.get("path"))
    offset = _page_value(arguments, "offset", 0)
    max_bytes = _page_value(arguments, "max_bytes", _OUTPUT_READ_DEFAULT, _OUTPUT_READ_MAX)
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least 1.")
    expected = _expected_sha256(arguments)

    original_root = root.stat()
    with contextlib.ExitStack() as resources:
        root_descriptor = _open_directory(root)
        resources.callback(os.close, root_descriptor)
        if not os.path.samestat(original_root, os.fstat(root_descriptor)):
            raise ValueError("The run directory changed while opening the output.")
        parent_descriptor = root_descriptor
        for part in relative.parts[:-1]:
            opened = _open_directory(part, dir_fd=parent_descriptor)
            resources.callback(os.close, opened)
            parent_descriptor = opened
        try:
            descriptor = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_descriptor)
        except OSError:
            raise ValueError("That output is not a readable regular file.") from None
        resources.callback(os.close, descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("That output is not a readable regular file.")
        if offset > before.st_size:
            raise ValueError("offset is past the end of the output.")
        data, file_digest = _scan_output_chunk(descriptor, before.st_size, offset, max_bytes)
        after = os.fstat(descriptor)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ValueError("The output changed while it was being read.")
        if not os.path.samestat(original_root, root.stat()) or _resolved(root) != root:
            raise ValueError("The run directory changed while the output was being read.")
        if expected is not None and file_digest != expected:
            raise ValueError("The output has changed since the previous read. Restart from the first chunk.")

    media_type = _artifact_media_type(relative)
    content = None
    binary = media_type in {"application/gzip", "application/zstd"} or re.search(
        rb"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", data
    )
    if not binary:
        with contextlib.suppress(UnicodeDecodeError):
            content = data.decode("utf-8")
    if content is None:
        content = base64.b64encode(data).decode("ascii")
        encoding = "base64"
    else:
        encoding = "utf-8"
    end = offset + len(data)
    return {
        "program": PROGRAM,
        "directory": _shown(root),
        "path": relative.as_posix(),
        "media_type": media_type,
        "encoding": encoding,
        "content": content,
        "offset": offset,
        "bytes_returned": len(data),
        "total_bytes": before.st_size,
        "next_offset": end if end < before.st_size else None,
        "chunk_sha256": hashlib.sha256(data).hexdigest(),
        "file_sha256": file_digest,
    }


def _validation_scope(arguments: Mapping[str, object]) -> tuple[Path, str | None]:
    """Resolve a concrete run in an output root and accept only an opaque candidate ID."""
    from rfproteina.disease.screen import _is_generation_directory
    from rfproteina.disease.trace import VALIDATION_FILE
    from rfproteina.disease.validation_checks import synthesis_receipt_attached

    directory = arguments.get("directory")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("Name one finished run directory.")
    _plain(directory)
    candidate = arguments.get("candidate_id")
    if "candidate_id" in arguments and (
        not isinstance(candidate, str) or re.fullmatch("[0-9a-f]{24}", candidate) is None
    ):
        raise ValueError("Use a candidate ID returned by plan_validation.")

    path = _resolved(_path(directory))
    checkout = _resolved(_repo_root())
    roots = (checkout / "disease_runs", checkout / "runs", *(_allowed_roots() or ()))
    if not _inside(path, roots):
        raise ValueError("Choose a directory under runs, disease_runs, or RFD4_MCP_ROOTS.")
    record = path / VALIDATION_FILE
    if not path.is_dir() or not (
        _is_generation_directory(path, follow_symlinks=False)
        or (not record.is_symlink() and record.is_file())
        or synthesis_receipt_attached(path)
    ):
        raise ValueError("Choose a concrete finished run directory, rather than a parent directory.")
    return path, candidate if isinstance(candidate, str) else None


def plan_validation(arguments: Mapping[str, object]) -> dict[str, object]:
    """Show available and recommended local checks with an optional candidate scope."""
    from rfproteina.disease.validation_checks import candidate_validation_plan

    _reject_unknown(arguments, {"directory", "candidate_id", "candidate_offset"})
    offset = arguments.get("candidate_offset", 0)
    if isinstance(offset, float) and offset.is_integer():
        offset = int(offset)
    if type(offset) is not int or offset < 0:
        raise ValueError("candidate_offset must be a nonnegative integer.")
    path, candidate = _validation_scope(arguments)
    candidates, choices = candidate_validation_plan((path,), candidate)
    if not choices:
        raise ValueError("That candidate is unavailable in this run. Refresh the plan.")
    end = offset + _VALIDATION_PAGE_SIZE
    return {
        "program": PROGRAM,
        "candidate_id": candidate,
        "candidates": [{"id": item.key, "label": item.label} for item in candidates[offset:end]],
        "next_candidate_offset": end if end < len(candidates) else None,
        "checks": [
            {
                "id": item.key,
                "label": item.label,
                "installed": item.installed,
                "recommended": item.recommended,
                "reason": item.reason,
                "workload": item.workload,
            }
            for item in choices
        ],
    }


def _screening_panel(name: str, arguments: Mapping[str, object], *, on_line: ProgressLine | None) -> dict:
    from rfproteina.disease import screen_panel

    scope = {"directory", "candidate_id"}
    _reject_unknown(arguments, scope if name == "plan_screening" else scope | {"selection", "methods"})
    path, candidate = _validation_scope(arguments)
    result = {"program": PROGRAM, "candidate_id": candidate, "directory": _shown(path)}
    if name == "plan_screening":
        choices = screen_panel.choices_for(path, candidate)
        return {**result, "methods": [item.as_record() for item in choices]}
    if ("methods" in arguments) == ("selection" in arguments):
        raise ValueError("Pass a list of method IDs or one selection mode.")
    methods = arguments.get("methods", [])
    if not isinstance(methods, list) or any(not isinstance(key, str) for key in methods):
        raise ValueError("Methods must be a list of screen IDs.")
    mode = arguments.get("selection", "explicit")
    if "selection" in arguments and mode not in ("recommended", "all_installed"):
        raise ValueError("selection must be recommended or all_installed.")
    return {**result, "results": screen_panel.run(path, candidate, mode, methods, on_line=on_line)}


def _validation_request(arguments: Mapping[str, object]) -> tuple[Path, str | None, list[str]]:
    from rfproteina.disease.validation_checks import CHECKS, candidate_validation_plan

    _reject_unknown(arguments, {"directory", "candidate_id", "checks", "selection"})
    selection = arguments.get("selection")
    checks = arguments.get("checks")
    if "selection" in arguments and "checks" in arguments:
        raise ValueError("Pass explicit checks or one selection mode, not both.")
    if "selection" in arguments and selection not in ("recommended", "all_installed"):
        raise ValueError("selection must be recommended or all_installed.")
    if selection is None and (not isinstance(checks, list) or not 1 <= len(checks) <= len(CHECKS)):
        raise ValueError("Choose one or more check IDs from plan_validation.")
    allowed = {check.key for check in CHECKS}
    if isinstance(checks, list) and (
        any(not isinstance(key, str) or key not in allowed for key in checks) or len(set(checks)) != len(checks)
    ):
        raise ValueError("Choose distinct check IDs returned by plan_validation.")
    path, candidate = _validation_scope(arguments)
    if isinstance(selection, str):
        _candidates, choices = candidate_validation_plan((path,), candidate)
        if not choices:
            raise ValueError("That candidate is unavailable in this run. Refresh the plan.")
        checks = [
            item.key
            for item in choices
            if item.installed and (selection == "all_installed" or item.recommended)
        ]
        if not checks:
            raise ValueError(f"No {selection.replace('_', ' ')} checks are available for this scope.")
    assert isinstance(checks, list)
    return path, candidate, checks


def _validation_results(path: Path, candidate: str | None, checks: list[str]) -> dict[str, object]:
    from rfproteina.disease.validation_checks import run_validation_checks

    report = run_validation_checks(
        (path,), checks, candidate_key=candidate, name_files=False, read_budget=_VALIDATION_BYTE_BUDGET
    )
    return {
        "program": PROGRAM,
        "candidate_id": candidate,
        "results": [{"id": key, "result": result} for key, result in report.items()],
    }


def validate_output_files(arguments: Mapping[str, object]) -> dict[str, object]:
    """Run an explicit selection of read-only checks against a finished run."""
    return _validation_results(*_validation_request(arguments))


def _validation_report_folder(path: Path) -> Path:
    folder = path / _VALIDATION_REPORT_DIR
    if _resolved(folder) != folder:
        raise ValueError("The validation report directory must be inside the run.")
    return folder


def list_validation_reports(arguments: Mapping[str, object]) -> dict[str, object]:
    """List finished report filenames without opening them or running checks."""
    _reject_unknown(arguments, {"directory"})
    path, _candidate = _validation_scope(arguments)
    folder = _validation_report_folder(path)
    report_files = []
    try:
        original = folder.lstat()
    except FileNotFoundError:
        return {"program": PROGRAM, "report_files": []}
    except OSError:
        raise ValueError("The validation report directory could not be listed.") from None
    try:
        with contextlib.ExitStack() as resources:
            scan_target: Path | int = folder
            if os.scandir in os.supports_fd:
                descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                resources.callback(os.close, descriptor)
                if not os.path.samestat(original, os.fstat(descriptor)):
                    raise ValueError("The validation report directory changed while listing it.")
                scan_target = descriptor
            entries = resources.enter_context(os.scandir(scan_target))
            for count, entry in enumerate(entries, 1):
                if count > _VALIDATION_REPORT_ENTRY_LIMIT:
                    raise ValueError("Too many entries in the validation report directory to list.")
                if _VALIDATION_REPORT_NAME.fullmatch(entry.name) and entry.is_file(follow_symlinks=False):
                    report_files.append(f"{_VALIDATION_REPORT_DIR}/{entry.name}")
        if not os.path.samestat(original, folder.lstat()) or _resolved(folder) != folder:
            raise ValueError("The validation report directory changed while listing it.")
    except OSError:
        raise ValueError("The validation report directory could not be listed.") from None
    return {"program": PROGRAM, "report_files": sorted(report_files)}


def save_validation_report(arguments: Mapping[str, object]) -> dict[str, object]:
    """Run the chosen checks and publish a new JSON report inside the run."""
    from rfproteina.disease.publish import scrubbed_json

    path, candidate, checks = _validation_request(arguments)
    _validation_report_folder(path)
    report_file = ""
    published = ""
    try:
        original = path.stat()
        with contextlib.ExitStack() as resources:
            run_descriptor = _open_directory(path)
            resources.callback(os.close, run_descriptor)
            if not os.path.samestat(original, os.fstat(run_descriptor)):
                raise ValueError("The run directory changed during validation.")
            result = _validation_results(path, candidate, checks)
            if not os.path.samestat(original, path.stat()) or _resolved(path) != path:
                raise ValueError("The run directory changed during validation.")

            with contextlib.suppress(FileExistsError):
                os.mkdir(_VALIDATION_REPORT_DIR, dir_fd=run_descriptor)
            folder_descriptor = _open_directory(_VALIDATION_REPORT_DIR, dir_fd=run_descriptor)
            resources.callback(os.close, folder_descriptor)

            destination = f"validation-{uuid4().hex}.json"
            temporary = f".validation-{uuid4().hex}.tmp"
            report_file = f"{_VALIDATION_REPORT_DIR}/{destination}"
            document = {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "report_file": report_file,
                **result,
            }
            text = scrubbed_json(document)
            temporary_descriptor = None
            try:
                temporary_descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=folder_descriptor,
                )
                with os.fdopen(temporary_descriptor, "w", encoding="utf-8", closefd=False) as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(
                    temporary,
                    destination,
                    src_dir_fd=folder_descriptor,
                    dst_dir_fd=folder_descriptor,
                    follow_symlinks=False,
                )
                destination_identity = os.stat(destination, dir_fd=folder_descriptor, follow_symlinks=False)
                if not os.path.samestat(os.fstat(temporary_descriptor), destination_identity):
                    with contextlib.suppress(OSError):
                        os.unlink(destination, dir_fd=folder_descriptor)
                    raise ValueError("The staged validation report changed before publication.")
                published = destination
            finally:
                if temporary_descriptor is not None:
                    with contextlib.suppress(OSError):
                        os.close(temporary_descriptor)
                with contextlib.suppress(OSError):
                    os.unlink(temporary, dir_fd=folder_descriptor)

            if not os.path.samestat(original, path.stat()) or _resolved(path) != path:
                if published:
                    with contextlib.suppress(OSError):
                        os.unlink(published, dir_fd=folder_descriptor)
                raise ValueError("The run directory changed while saving the validation report.")
            current_folder = os.stat(_VALIDATION_REPORT_DIR, dir_fd=run_descriptor, follow_symlinks=False)
            if (
                not os.path.samestat(current_folder, os.fstat(folder_descriptor))
                or _resolved(path / _VALIDATION_REPORT_DIR) != path / _VALIDATION_REPORT_DIR
            ):
                if published:
                    with contextlib.suppress(OSError):
                        os.unlink(published, dir_fd=folder_descriptor)
                raise ValueError("The validation report directory changed while saving the report.")
    except OSError:
        raise ValueError("The validation report could not be saved inside this run.") from None
    return {**result, "report_file": report_file}


def _open_directory(path: str | Path, *, dir_fd: int | None = None) -> int:
    """Open a directory without following its final link, optionally below an approved directory."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise OSError("Directory descriptors are unavailable.")
    if any(operation not in os.supports_dir_fd for operation in (os.open, os.mkdir, os.link, os.unlink)):
        raise OSError("Relative directory descriptors are unavailable.")
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)


def argv_from_options(options: Mapping[str, object]) -> list[str]:
    """Flags for the same parser as the command. Only provided options are included."""
    from rfproteina.disease.cli import build_parser

    _reject_secrets(options)
    parser = build_parser()
    by_dest = {action.dest: action for action in parser._actions if action.option_strings}
    unknown = [str(name) for name in options if name not in by_dest or name in {"help", *_BROWSER_ONLY_OPTIONS}]
    if unknown:
        raise ValueError("Unknown option.")
    argv: list[str] = []
    for action in parser._actions:
        if action.dest not in options or options[action.dest] is None:
            continue
        argv.extend(_encode(action, options[action.dest]))
    return argv


def _call_main(
    argv: list[str],
    *,
    on_line: ProgressLine | None = None,
    on_record: StructuredResult | None = None,
) -> tuple[int, list[str]]:
    """Call the program. Its prints stay in the returned lines, not on stdout.

    Both streams are captured into one, so the lines keep the order the program
    produced them in rather than all of stdout followed by all of stderr. That
    order is what makes them usable as progress while the run is still going.
    """
    from rfproteina.disease.cli import main

    stream = _LineStream(on_line)
    code = 2
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try:
                code = main(argv, read_line=_decline, report=on_record)
            except _Unanswered:
                raise ValueError("That disease was not found. Pass another name.") from None
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 2
            except KeyboardInterrupt:
                raise ValueError("Stopped.") from None
    except (OSError, RuntimeError):
        raise ValueError("The program could not be run.") from None
    finally:
        stream.flush_partial()
    return code, _Lines(stream.lines, stream.raw)


class _Lines(list):
    """The program's lines, carrying the unredacted originals beside them.

    A list subclass rather than a third return value: ``_call_main``'s
    two-value shape is what the stand-ins for the program return in the tests,
    and only a printed record needs the originals. ``raw`` is an attribute, so
    it is not serialised with the list and never reaches a client; a stand-in
    that returns a plain list simply has none, and :func:`_printed_json` then
    reads nothing.
    """

    __slots__ = ("raw",)

    def __init__(self, lines: list[str], raw: list[str]) -> None:
        super().__init__(lines)
        self.raw = list(raw)


class _LineStream(io.TextIOBase):
    """Collects what the program prints and reports each line as it is finished.

    The program writes a line at a time but a single ``print`` reaches a stream
    as several writes, so a line is only reported once its newline arrives. A
    last line without one is reported when the run ends.
    """

    def __init__(self, on_line: ProgressLine | None) -> None:
        super().__init__()
        self._on_line = on_line
        self._partial = ""
        self._captured_chars = 0
        self._truncated = False
        self.lines: list[str] = []
        #: The same lines before redaction. Never returned to a client: they
        #: are here so a printed JSON document can be parsed as the record it
        #: is, rather than reassembled from strings the free-text scrubber has
        #: rewritten inside their quotes. See :func:`_printed_json`.
        self.raw: list[str] = []

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not isinstance(text, str) or not text:
            return 0
        if self._truncated:
            return len(text)
        remaining = max(_CAPTURE_MAX_CHARS - self._captured_chars, 0)
        kept = text[:remaining]
        self._captured_chars += len(kept)
        self._partial += kept
        while not self._truncated and "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._report(line)
        if len(kept) != len(text):
            self._truncate()
        return len(text)

    def flush_partial(self) -> None:
        """Report a final line the program left without a newline."""
        if self._truncated:
            self._partial = ""
            return
        remaining, self._partial = self._partial, ""
        if remaining:
            self._report(remaining)

    def _report(self, raw: str) -> None:
        """Keep a line worth returning, and hand it on. The log path is not one.

        Two lines name that path, not one: the program announces it at the start
        and names it again on an unexpected error. Scrubbing hides it only while
        it sits under the home directory, so a log moved elsewhere by
        ``RFD4_KEY_DIR`` or ``--log-file`` went out with the second line intact.
        """
        if self._truncated:
            return
        if len(self.raw) >= _CAPTURE_MAX_LINES:
            self._truncate()
            return
        oversized = len(raw) > _CAPTURE_LINE_MAX
        stripped = raw[:_CAPTURE_LINE_MAX].strip()
        if not stripped or stripped.startswith(("Run log:", "The traceback is in ")):
            if oversized:
                self._truncate()
            return
        self.raw.append(stripped)
        redacted = _redact(stripped)
        if not isinstance(redacted, str) or not redacted:
            return
        self.lines.append(redacted)
        if self._on_line is not None:
            self._on_line(redacted)
        if oversized:
            self._truncate()

    def _truncate(self) -> None:
        """Stop retaining output and make the loss explicit in the returned lines."""
        if self._truncated:
            return
        self._truncated = True
        self._partial = ""
        self.lines.append(_CAPTURE_MARKER)
        if self._on_line is not None:
            self._on_line(_CAPTURE_MARKER)


def _decline(prompt: str) -> str:
    """A confirmation this server was not given. Empty declines the question.

    The disease prompt is different. After an unmatched name the program clears
    the name and asks again, and an empty answer sends it round once more.
    """
    if prompt.startswith("Disease name:"):
        raise _Unanswered
    return ""


def _read_screen_no_follow(directory: Path) -> Mapping[str, object] | None:
    """Read one bounded screen record through its own regular-file descriptor."""
    path = directory / "screen.json"
    try:
        original = path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise ValueError("screen.json is not readable.") from None
    if not stat.S_ISREG(original.st_mode) or stat.S_ISLNK(original.st_mode):
        raise ValueError("screen.json must be a regular file inside the run.")
    if original.st_size > _SCREEN_JSON_MAX:
        raise ValueError("screen.json is too large to return.")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not os.path.samestat(original, opened):
                raise ValueError("screen.json changed while it was opened.")
            raw = handle.read(_SCREEN_JSON_MAX + 1)
            after = os.fstat(handle.fileno())
        if len(raw) > _SCREEN_JSON_MAX:
            raise ValueError("screen.json is too large to return.")
        if not os.path.samestat(opened, after) or (opened.st_size, opened.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("screen.json changed while it was read.")
        if not os.path.samestat(original, path.lstat()):
            raise ValueError("screen.json changed while it was read.")
        payload = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("screen.json is not a readable screen.") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("call"), str):
        raise ValueError("screen.json is not a screen.")
    return payload


def _screens(directory: Path, *, include_record: bool) -> dict[str, object]:
    """Stored screens under ``directory``, without walking into another screen's samples."""
    from rfproteina.disease.screen import ScreenWalkTruncatedError, label_for, walk_for_screens

    if not directory.is_dir():
        return {
            "program": PROGRAM,
            "directory": _shown(directory),
            "complete": False,
            "runs": [],
        }
    runs: list[dict[str, object]] = []
    complete = True
    own = _one_screen(directory, "", include_record=include_record, read_screen=_read_screen_no_follow)
    if own is not None:
        runs.append(own)
    try:
        for current, filenames in walk_for_screens(directory):
            if "screen.json" not in filenames or current == directory:
                continue
            found = _one_screen(
                current,
                label_for(directory, current),
                include_record=include_record,
                read_screen=_read_screen_no_follow,
            )
            if found is not None:
                runs.append(found)
    except ScreenWalkTruncatedError:
        complete = False
    return {
        "program": PROGRAM,
        "directory": _shown(directory),
        "complete": complete,
        "runs": runs,
    }


def _one_screen(
    directory: Path,
    label: str,
    *,
    include_record: bool,
    read_screen: Callable[[Path], Mapping[str, object] | None],
) -> dict[str, object] | None:
    """One stored screen, or an error entry when the file is present and unreadable.

    A file that parses can still fail afterwards. A screen whose
    ``sequence_developability`` carries ``sequence_pass`` without ``gravy``
    raises out of the line renderer, and a deeply nested one out of the walk
    that redacts it. Either used to end the whole call, so one such file
    anywhere in a tree hid every other run's verdict. It is this entry that
    says so now, and the others are returned.
    """
    from rfproteina.disease.screen import format_screen_lines_for_directory

    entry: dict[str, object] = {"directory": _shown(directory), "label": label}
    entry.update(_archive_copy(directory))
    try:
        payload = read_screen(directory)
    except ValueError as exc:
        entry["error"] = _redact(str(exc))
        return entry
    except RecursionError:
        entry["error"] = "screen.json is nested too deeply to read."
        return entry
    if payload is None:
        return None
    entry["lines"] = []
    try:
        entry["verdict"] = _verdict(payload, include_record=include_record)
        entry["lines"] = [_redact(line) for line in format_screen_lines_for_directory(directory, payload)]
    except Exception:
        entry["error"] = "screen.json was read, and part of it could not be shown."
    return entry


def _archive_copy(directory: Path) -> dict[str, object]:
    """Whether this run is a copy the generation archive made, and that copy's id.

    It matters because the walk returns both. Reading the verdicts under an
    output directory returns the live runs and every copy beneath the archive,
    with nothing saying which was which, so a client tallying runs counted each
    one twice -- on this checkout, thirteen runs for six.

    The test is where the run sits AND that it carries an archive record, not
    the record alone. ``generation.json`` looked like proof -- the archive
    writes one into every folder it copies a run into, and the program writes
    one nowhere else -- but it also READS one, and the documentation tells a
    person to create it: ``docs/reference/screen-inputs.md`` answers
    ``target_not_named`` with "supply ``generation.json`` or a job file that
    does", and ``cli._screened_identity`` and ``screen._names_for_directory``
    read it from any run being screened. A live run whose target was named that
    way carried the file, and a filename test called it a copy -- so a client
    doing what this flag's own description says would drop the only copy of
    that run from its tally.
    ``run_id`` alone is not the test either: the archive began recording it
    part-way through, and six of the seven copies in this checkout predate it.
    Marking only the seventh would leave the double count this flag exists to
    stop.
    So both: the archive's own root name on the path, which is how
    ``save_generation`` lays a copy out, and a record beside the run. A
    hand-written record is not under the archive root, and a folder a person
    happens to name ``generations`` holds no record.

    ``run_id`` still travels when the record has one, because it is what tells
    two copies of one target apart. A link is not rejected: the archive copies
    a run with ``symlinks=True``, so a record that was a link in the source
    stays one in the copy.
    """
    from rfproteina.disease.archive import GENERATION_ARCHIVE

    if GENERATION_ARCHIVE not in directory.parent.parts:
        return {}
    path = directory / "generation.json"
    try:
        if not path.is_file():
            return {}
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        # Under the archive root with a record that cannot be read: still a
        # copy, and saying so is what stops it being counted as a second run.
        return {"archived": True}
    found: dict[str, object] = {"archived": True}
    run_id = body.get("run_id") if isinstance(body, Mapping) else None
    if isinstance(run_id, str) and run_id.strip():
        found["archive_run_id"] = _redact(run_id)
    return found


def _without_held_back(payload: Mapping[str, object]) -> dict[str, object]:
    """``payload`` with every ``_HELD_BACK`` key dropped from its own block.

    The per-field loop in :func:`_verdict` applies ``_HELD_BACK`` on the way
    into the view, but ``include_record`` returns the stored screen whole and so
    went around it: a repeated subsequence, a ligand SMILES and the absolute
    path of a reference table all travelled inside ``record``, which is the one
    request that reaches a client without passing that loop. The allowlist's
    whole purpose is that those three stay in screen.json, so it has to hold on
    both paths.

    A block whose stored value is not a mapping is left exactly as found. That
    loop can rely on ``_VERDICT_TYPES`` having screened the value first; this
    runs on the raw record, where a hand-edited file could put anything.
    """
    record = dict(payload)
    for name, held in _HELD_BACK.items():
        block = record.get(name)
        if isinstance(block, Mapping):
            record[name] = {key: item for key, item in block.items() if key not in held}
    return record


def _affinity_view(report: Mapping[str, object]) -> dict[str, object]:
    """Put input-supplied sample and chain names in values handled by the text scrubber."""
    samples = report.get("samples")
    if not isinstance(samples, Mapping):
        return dict(report)
    rows = []
    for name, sample in samples.items():
        row = dict(sample) if isinstance(sample, Mapping) else {"malformed": True}
        counts = row.get("residues_by_chain")
        if isinstance(counts, Mapping):
            row["residues_by_chain"] = [{"chain": chain, "residues": count} for chain, count in counts.items()]
        rows.append({**row, "example_id": name})
    return {**report, "samples": rows}


def _sygma_view(block: Mapping) -> dict:
    """Retain sequence-like InChIKeys only when the local structure verifies them."""
    from rfproteina.disease.sygma_engine import structure_id

    view = dict(block)
    structures, parents = block.get("structures"), block.get("parents")
    if not isinstance(structures, Mapping) or not isinstance(parents, list):
        return view
    try:
        from rdkit import Chem
    except (ImportError, OSError):
        return view

    def product_view(product: object) -> object:
        if not isinstance(product, Mapping):
            return product
        key, identifier = product.get("inchikey"), product.get("product_id")
        if not isinstance(key, str) or len(key) != 27 or not _SEQUENCE.search(key) or not isinstance(identifier, str):
            return product
        smiles = structures.get(identifier)
        if not isinstance(smiles, str) or not smiles.isascii() or structure_id(smiles) != identifier:
            return product
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and Chem.MolToInchiKey(mol) == key:
                return {**product, "inchikey": _Prose(key)}
        except (ValueError, RuntimeError, OverflowError):
            pass
        return product

    view["parents"] = [
        {**p, "products": [product_view(row) for row in p["products"]]}
        if isinstance(p, Mapping) and isinstance(p.get("products"), list)
        else p
        for p in parents
    ]
    return view


def _verdict(payload: Mapping[str, object], *, include_record: bool) -> dict[str, object]:
    """The decision fields. The full record is included only when asked for.

    A field whose stored type is not the one the tool's schema declares is left
    out and named under ``malformed``. Only ``call`` is checked when a screen is
    read, so a hand-edited file could otherwise make this server return a
    result its own schema rejects, and a client that validates would then
    discard every run in the answer over one field of one of them.
    """
    affinity = payload.get("prodigy_affinity")
    if isinstance(affinity, Mapping):
        payload = {**payload, "prodigy_affinity": _affinity_view(affinity)}
    metabolites = payload.get("sygma")
    if isinstance(metabolites, Mapping):
        payload = {**payload, "sygma": _sygma_view(metabolites)}
    view: dict[str, object] = {}
    malformed: list[str] = []
    for name in _VERDICT_FIELDS:
        if name not in payload:
            continue
        expected = _VERDICT_TYPES.get(name)
        if expected is not None and not expected(payload[name]):
            malformed.append(name)
            continue
        value = payload[name]
        held = _HELD_BACK.get(name)
        if held:
            value = {key: item for key, item in value.items() if key not in held}
        if name in _PROVENANCE_BLOCKS:
            value = _provenance(value, name, malformed)
        view[name] = _redact(value)
    replaced = _replaced(payload.get("replaced"), malformed)
    if replaced is not None:
        view["replaced"] = replaced
    for name in _AXIS_GROUPS:
        if name not in payload:
            continue
        summary = _axis_summary(payload[name], name, malformed)
        if summary is not None:
            view[name] = summary
    curated = _curated_axes(payload)
    if curated is not None:
        view["curated_axes"] = curated
    if malformed:
        view["malformed"] = malformed
    if include_record:
        view["record"] = _redact(_without_held_back(payload))
    return view


def _axis_summary(
    value: object,
    group: str = "",
    malformed: list[str] | None = None,
) -> dict[str, object] | None:
    """What each axis record says about its state and where the state came from.

    Peptides and sequences stay out. ``state`` and ``source`` are text in the
    schema, so one that is not is left out and named in ``malformed``.
    """
    noted = malformed if malformed is not None else []
    if not isinstance(value, Mapping):
        noted.append(group)
        return None
    summary: dict[str, object] = {}
    for axis, record in value.items():
        if not isinstance(record, Mapping):
            noted.append(f"{group}.{axis}")
            continue
        kept: dict[str, object] = {}
        for field in _AXIS_FIELDS:
            if field not in record:
                continue
            expected = _AXIS_TYPES.get(field)
            if expected is not None and not expected(record[field]):
                noted.append(f"{group}.{axis}.{field}")
                continue
            kept[field] = _redact(record[field])
        if kept:
            summary[str(axis)] = kept
    return summary or None


def _curated_axes(payload: Mapping[str, object]) -> list[str] | None:
    """The axes whose state a person declared in ``axis_evidence.json``.

    Named beside the call, because the call and the flat ``states`` map show a
    declared clear and a measured one the same way. None when no axis record
    says where its state came from: that is a screen written before the source
    was recorded, and nothing can be said about it either way.
    """
    stamped = False
    found: set[str] = set()
    for name in _AXIS_GROUPS:
        group = payload.get(name)
        if not isinstance(group, Mapping):
            continue
        for axis, record in group.items():
            if not isinstance(record, Mapping):
                continue
            if "source" in record:
                stamped = True
            if record.get("source") == _CURATED_SOURCE or "curated" in record:
                stamped = True
                found.add(str(axis))
    return sorted(found) if stamped else None


def _replaced(value: object, malformed: list[str]) -> dict[str, object] | None:
    """What a rescreen overwrote: the previous call, and the supplied files that moved.

    The record keeps this only when the verdict changed. No ``changed_inputs``
    means the verdict changed with no supplied file added, removed or edited.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        malformed.append("replaced")
        return None
    view: dict[str, object] = {}
    call = value.get("call")
    if isinstance(call, str):
        view["call"] = _redact(call)
    elif "call" in value:
        malformed.append("replaced.call")
    changed = value.get("changed_inputs")
    if isinstance(changed, list):
        moved: list[dict[str, object]] = []
        for item in changed:
            if not isinstance(item, Mapping) or not isinstance(item.get("file"), str):
                malformed.append("replaced.changed_inputs")
                continue
            entry: dict[str, object] = {"file": _redact(item["file"])}
            if isinstance(item.get("change"), str):
                entry["change"] = _redact(item["change"])
            moved.append(entry)
        view["changed_inputs"] = moved
    elif "changed_inputs" in value:
        malformed.append("replaced.changed_inputs")
    return view


def _provenance(block: object, label: str, malformed: list[str]) -> object:
    """A block's nested provenance record, each entry checked field by field.

    Checked the way :func:`_axis_summary` checks an axis: a field whose stored
    type is wrong is left out and named, and the rest of the entry still
    travels. Dropping the whole block over one bad field would take the twelve
    sound accounts with it, and dropping the whole entry would take the
    ``source`` slug -- which is the stable name two tests already pin -- over a
    caveat somebody hand-edited.

    ``kind`` is checked against the vocabulary imported from the module that
    declares it, never a copy. That set is explicitly closed there: a kind
    outside it is called a bug in that file, which is precisely the thing worth
    reporting to a client rather than passing through as though it were a kind.

    A block with no provenance is returned unchanged, so this is safe to run
    over a screen written before the record existed.
    """
    if not isinstance(block, Mapping) or PROVENANCE_KEY not in block:
        return block
    entries = block[PROVENANCE_KEY]
    record = dict(block)
    if not isinstance(entries, Mapping):
        malformed.append(f"{label}.{PROVENANCE_KEY}")
        del record[PROVENANCE_KEY]
        return record
    kept: dict[str, object] = {}
    for name, entry in entries.items():
        if not isinstance(entry, Mapping):
            malformed.append(f"{label}.{PROVENANCE_KEY}.{name}")
            continue
        fields: dict[str, object] = {}
        for field in _PROVENANCE_FIELDS:
            if field not in entry:
                continue
            expected = _PROVENANCE_TYPES.get(field)
            if expected is not None and not expected(entry[field]):
                malformed.append(f"{label}.{PROVENANCE_KEY}.{name}.{field}")
                continue
            fields[field] = entry[field]
        if fields:
            kept[str(name)] = fields
    record[PROVENANCE_KEY] = kept
    return record


def _is_text(value: object) -> bool:
    return isinstance(value, str)


def _is_names(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_records(value: object) -> bool:
    """A list of objects, which is what ``inputs`` is.

    Its two sentinel forms are lists of one object as well -- the folder that
    could not be listed and the folder with more files than the writer will
    record both arrive as ``[{...}]`` -- so this admits them without a special
    case, and a reader tells them apart by the ``file`` value rather than by
    the shape.
    """
    return isinstance(value, list) and all(isinstance(item, Mapping) for item in value)


def _is_states(value: object) -> bool:
    return isinstance(value, Mapping) and all(isinstance(item, str) for item in value.values())


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_object(value: object) -> bool:
    return isinstance(value, Mapping)


def _is_optional_number(value: object) -> bool:
    if value is None or type(value) is int:
        return True
    return isinstance(value, float) and math.isfinite(value)


#: The type the tool's schema declares for a verdict field. A field that is not
#: listed has no declared type and is returned as stored.
def _is_flag(value: object) -> bool:
    return isinstance(value, bool)


def _is_kind(value: object) -> bool:
    """A threshold kind this program declares. Imported, so it cannot drift."""
    return isinstance(value, str) and value in _PROVENANCE_KINDS


def _is_bar(value: object) -> bool:
    """The threshold itself: a ``min`` or ``max`` object, or null where no cut is adopted.

    Null is a real answer here and not a missing one. Two of the five interface
    bars deliberately adopt no cut, and the record says so rather than leaving
    a reader to hunt for a pass that was never defined.
    """
    return value is None or isinstance(value, Mapping)


#: The type each provenance field is checked against. ``kind`` is the one that
#: matters: the rest are free text a person wrote, and a wrong type in them
#: costs the reader that sentence, while a kind outside the closed set is the
#: record claiming a category this program does not define.
_PROVENANCE_TYPES: dict[str, Callable[[object], bool]] = {
    "kind": _is_kind,
    "source": _is_text,
    "citation": _is_text,
    "caveat": _is_text,
    "bar": _is_bar,
}
#: The type the axis schema declares for each axis field, for the same reason
#: :data:`_VERDICT_TYPES` exists and with the same consequence when it is
#: missing: a stored field whose type the schema forbids makes a validating
#: client throw away every run in the answer over one field of one of them.
#:
#: ``_axis_summary`` checked ``state`` and ``source`` and returned the other
#: thirty as stored, while the schema declared a type for most of them -- so
#: one hand-edited ``unscored_peptides: "none"`` was enough. Only the fields
#: the schema leaves untyped are absent here: a measurement is a number on some
#: axes and a name on others, a blocker likewise, and ``threshold``,
#: ``reference_density``, ``confidence``, ``rank``, ``ic50``, ``trigger``,
#: ``curated``, ``measured_state`` and ``measured_blocker`` are the same story
#: or carry a null the axis means.
_AXIS_TYPES: dict[str, Callable[[object], bool]] = {
    "state": _is_text,
    "source": _is_text,
    "scan_limit": _is_text,
    "threshold_source": _is_text,
    "units": _is_text,
    "fold_source": _is_text,
    "fold_column": _is_text,
    "linear_fold_magnitude": _is_optional_number,
    "method": _is_text,
    "presentation": _is_text,
    "displaced_method": _is_text,
    "displaced_table": _is_text,
    "assay": _is_text,
    "network": _is_text,
    "release": _is_text,
    "reference_file": _is_text,
    "reference_digest": _is_text,
    "proteome_digest": _is_text,
    "qualifier": _is_text,
    "compara_blocker": _is_text,
    "unscored_peptides": _is_count,
    "reference_bases": _is_count,
    "binding_peptides": _is_count,
    "proteome_kmers": _is_count,
    "seed_width": _is_count,
    "seeds_searched": _is_count,
    "n_readings": _is_count,
    "compara_examined": _is_count,
    "compara_scored": _is_count,
    "presented_complete": _is_flag,
    "reference_rank_scale_ambiguous": _is_flag,
    "reference_method_undeclared": _is_flag,
    "calis_mask_allele_specific": _is_flag,
    "not_nm_columns": _is_names,
    "reference_proteins_too_short": _is_names,
    "mask_unpublished_alleles": _is_names,
    "covered_examples": _is_names,
    "uncovered_examples": _is_names,
    "unparsed_columns": _is_names,
    "alleles": _is_count,
    "peptides_scored": _is_count,
    "binding_alleles": _is_count,
    "presented_ligands": _is_count,
    "positive_windows": _is_count,
    "positive_examples": _is_names,
    "promiscuous_peptides": _is_count,
    "cpg_count": _is_count,
    "window_residues": _is_count,
    "coevolution_rows": _is_count,
    "failed": _is_names,
    "same_shape": _is_flag,
    "coevolution_signal": _is_text,
    "coevolution_level": _is_text,
    "other_species_toxicity": _is_text,
    "human_toxicity_state": _is_text,
    "coevolution_blocker": _is_text,
}
#: ``bar_axes`` and ``metrics_file`` are deliberately absent: the first is a
#: list of axes or the text ``absent``, the second a path or null, and a type
#: claimed for either would drop a valid field into ``malformed``.
_VERDICT_TYPES: dict[str, Callable[[object], bool]] = {
    "call": _is_text,
    "bar": _is_text,
    "states": _is_states,
    "untested": _is_names,
    "hits": _is_names,
    "screen_version": _is_count,
    "code": _is_object,
    "cuts": _is_object,
    "metrics_blocker": _is_text,
    "chain_blocker": _is_text,
    "refold": _is_object,
    "solubility": _is_object,
    "design_diversity": _is_object,
    "designed_interface_quality": _is_object,
    "prodigy_affinity": _is_object,
    "vina_pose": _is_object,
    "tnp_profile": _is_object,
    "ligand_alerts": _is_object,
    "ligand_solubility": _is_object,
    "qupkake": _is_object,
    "sygma": _is_object,
    "method_selection": _is_object,
    "protein_sol": _is_object,
    "mhc_class_i": _is_object,
    "synthesis": _is_object,
    "inputs": _is_records,
    "chemical_liability": _is_object,
    "modality_screen": _is_object,
}


class _Prose(str):
    """Text a person wrote, which :func:`_redact` holds to the twenty-residue rule.

    A subclass rather than an argument, because a layer result is redacted once
    more as a whole before it is returned, and that pass has to know the same
    thing about the same string.
    """

    __slots__ = ()


def _redact(value: object) -> object:
    """Drop sequences, secrets, and personal paths from a returned value.

    A phrase match used to replace the whole string with ``[omitted]``, which
    threw away whatever was measured in the same line. It matched wording rather
    than meaning, so most of what it removed was a line stating that no such
    outcome was established. The screen records measurements and named blockers,
    and returning those readable is the point of returning them at all.
    """
    from rfproteina.disease.publish import scrub_for_upload

    if isinstance(value, str):
        rule = _PROSE_SEQUENCE if isinstance(value, _Prose) else _SEQUENCE
        cleaned = rule.sub("[sequence omitted]", value)
        scrubbed = scrub_for_upload(cleaned)
        return _Prose(scrubbed) if isinstance(value, _Prose) else scrubbed
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            # Structure selectors are SHA-256 hex, not biological sequence.
            # Keeping only this fixed alphabet and length exception preserves
            # the opaque selection handle without weakening sequence removal.
            if (
                name == "selector"
                and isinstance(item, str)
                and len(item) == 64
                and all(character in "0123456789abcdef" for character in item)
            ):
                redacted[name] = item
            else:
                redacted[name] = _redact(item)
        return redacted
    if isinstance(value, list | tuple):
        return [_redact(item) for item in value]
    if isinstance(value, Path):
        return _shown(value)
    return value


def _path(value: str) -> Path:
    """Where a path argument points, or a refusal.

    A relative path is inside this checkout, and one that climbs out of it
    with ``..`` is refused: joined as written, ``../../x`` named any file on
    the machine while reading as a path under the checkout. An absolute path
    is used as given unless ``RFD4_MCP_ROOTS`` names the directories this
    server may use. Then every path has to resolve inside the checkout or one
    of them, and links are followed before the comparison, so a link inside an
    allowed directory cannot lead out of it.

    Without that variable links are not followed. A folder inside the checkout
    that is a link to another volume is an ordinary place to keep outputs, and
    refusing its relative name protected nothing while the absolute name of
    the same place was accepted.

    Every path argument of every tool comes through here, because the program
    appends to ``--log-file`` and writes under ``--out-dir``, and a client
    could otherwise name any file for either.
    """
    path = Path(value)
    checkout = _repo_root()
    roots = _allowed_roots()
    if not path.is_absolute():
        if roots is None:
            plain = Path(os.path.normpath(checkout / path))
            if not _inside(plain, (checkout,)):
                raise ValueError("A relative path cannot leave this checkout.")
            return plain
        resolved = _resolved(checkout / path)
        if not _inside(resolved, (checkout, *roots)):
            raise ValueError("A relative path cannot leave this checkout.")
        return resolved
    if roots is None:
        return path
    resolved = _resolved(path)
    if not _inside(resolved, (checkout, *roots)):
        raise ValueError("That path is outside the directories this server may use.")
    return resolved


def _allowed_roots() -> tuple[Path, ...] | None:
    """The directories ``RFD4_MCP_ROOTS`` names, or None when it names none.

    Read on each call, so a test or a restart with another value is seen. An
    entry that is not absolute is ignored: it would depend on where the server
    was started. A value with no usable entry leaves the checkout alone
    allowed, which errs towards refusing.
    """
    raw = os.environ.get(_ROOTS_ENV, "")
    if not raw.strip():
        return None
    roots: list[Path] = []
    for item in raw.split(os.pathsep):
        candidate = Path(item.strip())
        if not item.strip() or not candidate.is_absolute():
            continue
        try:
            roots.append(_resolved(candidate))
        except ValueError:
            continue
    return tuple(roots)


def _resolved(path: Path) -> Path:
    """``path`` with links followed and ``..`` removed. It need not exist yet."""
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        raise ValueError("That path cannot be used.") from None


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _shown(path: Path) -> str:
    """A checkout-relative path when the file lives here, otherwise the path as it is.

    Only a personal prefix at its start is replaced. The whole path used to go
    through the free-text scrubber, which removes a home-directory segment
    wherever it sits, so two different run directories could be returned as
    one string that named neither.
    """
    from rfproteina.disease.publish import repository_relative, scrub_path_prefix

    return scrub_path_prefix(repository_relative(path))


def _screen_files(directory: Path) -> dict[str, Path]:
    """``screen.json`` under ``directory``, keyed the same way a verdict labels a run."""
    from rfproteina.disease.screen import ScreenWalkTruncatedError, label_for, walk_for_screens

    files = {"": directory / "screen.json"}
    try:
        for current, _filenames in walk_for_screens(directory):
            files[label_for(directory, current)] = current / "screen.json"
    except ScreenWalkTruncatedError:
        return files
    return files


#: What identifies one write of a file, and a digest of its bytes.
_FileState = tuple[tuple[int, int, int], str]


def _file_state(path: Path) -> _FileState | None:
    """Modification time, inode and size, with a digest of the bytes. None when unreadable.

    The time alone misses a write. The screen is replaced atomically, and on a
    filesystem with coarse timestamps two writes share one. A replace puts a
    new file under the name, so the inode moves even when the time and the
    size do not.
    """
    try:
        found = path.lstat()
        if not stat.S_ISREG(found.st_mode) or stat.S_ISLNK(found.st_mode):
            return None
        if found.st_size > _SCREEN_JSON_MAX:
            digest = f"oversized:{found.st_size}"
        else:
            hashed = hashlib.sha256()
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if not os.path.samestat(found, opened):
                    return None
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hashed.update(chunk)
                after = os.fstat(handle.fileno())
            if not os.path.samestat(opened, after) or (opened.st_size, opened.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                return None
            digest = hashed.hexdigest()
    except OSError:
        return None
    return (found.st_mtime_ns, found.st_ino, found.st_size), digest


def _mark_written(
    body: dict[str, object],
    files: Mapping[str, Path],
    before: Mapping[str, _FileState | None],
) -> None:
    """Say, for each stored screen, whether this call wrote it and whether it changed.

    The two are different questions. A rescreen that reaches the same verdict
    writes the file and changes nothing in it. A screen this call did not
    write is the stored one, of whatever age: the program screens a bounded
    number of runs, and a run past that bound is returned without being
    visited.
    """
    runs = body.get("runs")
    if not isinstance(runs, list):
        return
    for run in runs:
        if not isinstance(run, dict):
            continue
        label = run.get("label")
        key = label if isinstance(label, str) else ""
        path = files.get(key)
        after = _file_state(path) if path is not None else None
        earlier = before.get(key)
        # Bytes that differ were written, whatever the time and the inode say.
        run["written"] = after is not None and after != earlier
        run["changed"] = after is not None and (earlier is None or after[1] != earlier[1])


def _repo_root() -> Path:
    """The checkout that holds this package."""
    return Path(__file__).resolve().parents[3]


def _reject_unknown(arguments: Mapping[str, object], allowed: set[str]) -> None:
    if any(name not in allowed for name in arguments):
        raise ValueError("Unknown option.")


def _reject_secrets(options: Mapping[str, object]) -> None:
    for name in options:
        if _SECRET_WORD.search(str(name)):
            raise ValueError("API keys are not an option.")


def _validate_remote_options(options: Mapping[str, object]) -> None:
    """Keep SSH destinations and remote paths from becoming command options.

    The generation dispatcher passes a host directly to ``ssh`` and ``scp``.
    A destination beginning with ``-`` is therefore an SSH option rather than
    a host, and legacy scp parses its remote path through a shell. MCP accepts
    ordinary aliases, DNS names, IPv4, bracketed IPv6 and ``user@host`` while
    refusing syntax that has another meaning to those programs.
    """
    raw_hosts = options.get("hosts")
    if raw_hosts is not None:
        if not isinstance(raw_hosts, str):
            raise ValueError("hosts is a comma-separated string.")
        _plain(raw_hosts)
        hosts = [item.strip() for item in raw_hosts.split(",") if item.strip()]
        if len(hosts) > _MAX_REMOTE_HOSTS:
            raise ValueError(f"At most {_MAX_REMOTE_HOSTS} generation hosts may be selected in one call.")
        if any(_REMOTE_HOST.fullmatch(host) is None for host in hosts):
            raise ValueError(
                "Each generation host must be a host alias, address, or user@host name, not an SSH option."
            )

    raw_directory = options.get("remote_dir")
    if raw_directory is None:
        return
    if not isinstance(raw_directory, str):
        raise ValueError("remote_dir is a POSIX path string.")
    _plain(raw_directory)
    if not raw_directory or len(raw_directory) > _OUTPUT_PATH_MAX or "\\" in raw_directory:
        raise ValueError("remote_dir must be a short POSIX directory path.")
    path = PurePosixPath(raw_directory)
    parts = [part for part in path.parts if part != "/"]
    if not parts or any(part in {"", ".", ".."} or _REMOTE_PATH_SEGMENT.fullmatch(part) is None for part in parts):
        raise ValueError("remote_dir may contain only ordinary POSIX path segments and cannot traverse upward.")


def _flag(action: argparse.Action) -> str:
    return max(action.option_strings, key=len)


def _kind(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction):
        return "boolean"
    if isinstance(action, argparse._AppendAction):
        return "array"
    if action.type is int:
        return "integer"
    if action.type is float:
        return "number"
    return "string"


def _default(action: argparse.Action) -> object:
    if isinstance(action, argparse._StoreTrueAction):
        return False
    if isinstance(action, argparse._AppendAction) or action.default in {None, argparse.SUPPRESS}:
        return None
    if isinstance(action.default, Path):
        return action.default.as_posix()
    if isinstance(action.default, str | int | float | bool):
        return action.default
    return str(action.default)


def _encode(action: argparse.Action, value: object) -> list[str]:
    """One flag, or nothing when a boolean is false. The value is never a shell string."""
    flag = _flag(action)
    if isinstance(action, argparse._StoreTrueAction):
        if value is True:
            return [flag]
        if value is False:
            return []
        raise ValueError(f"{action.dest} is true or false.")
    if isinstance(action, argparse._AppendAction):
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f"{action.dest} is a string or a list of strings.")
        argv: list[str] = []
        for item in items:
            _plain(item)
            if action.type is Path and not item.strip():
                raise ValueError(f"{action.dest} must be a non-empty path.")
            argv.extend([flag, str(_path(item)) if action.type is Path else item])
        return argv
    text = _scalar(action, value)
    if action.choices is not None and text not in {str(choice) for choice in action.choices}:
        raise ValueError(f"{action.dest} is not one of the listed choices.")
    return [flag, text]


def _scalar(action: argparse.Action, value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(f"{action.dest} has the wrong type.")
    if action.type is int:
        # An integral float is accepted because this server publishes these
        # options as JSON Schema "integer", and that keyword matches any number
        # with no fractional part. A client that validated its call against
        # the schema list_options returns, or against screen_directory's own
        # input schema, could be refused by the very server that published it:
        # JSON has one number type, so a language without an integer type sends
        # 1 as 1.0.
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{action.dest} is an integer.")
        return str(value)
    if action.type is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{action.dest} is a number.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{action.dest} is a number.")
        return format(number, "g")
    if not isinstance(value, str):
        raise ValueError(f"{action.dest} is a string.")
    _plain(value)
    if action.type is Path:
        # Every path option, not only the directory: the same refusal applies
        # to where the program writes as to where it reads.
        if not value.strip():
            raise ValueError(f"{action.dest} must be a non-empty path.")
        return str(_path(value))
    return value


def _optional_flag(value: object) -> bool:
    """A boolean tool argument. Absent means false."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise ValueError("That option is true or false.")


def _plain(text: str) -> None:
    if "\x00" in text or "\n" in text or "\r" in text:
        raise ValueError("An option cannot contain a newline.")
