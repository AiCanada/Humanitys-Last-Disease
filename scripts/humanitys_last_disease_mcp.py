#!/usr/bin/env python3
"""Humanitys-Last-Disease as an MCP server.

Speak MCP on stdin and stdout. Diagnostics stay on stderr. On Windows this
hands the same stdio to the checkout's Linux environment, where the tools run
in-process. Tool text is not a shell command, and the program is not parsed
from its own printed lines.

Both protocol eras are answered. A request that carries its protocol version
in ``_meta`` is served statelessly under revision 2026-07-28, which removed
the ``initialize`` handshake and requires ``server/discover``. A request that
opens with ``initialize`` is served under the handshake revisions. A client
that speaks only one of the two still works, which a server implementing only
the handshake does not: the 2026-07-28 compatibility matrix scores a modern
client against a legacy server as a failure.

The client should start this file with the repository as the working directory:
``python -u scripts/humanitys_last_disease_mcp.py``.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

SERVER_NAME = "Humanitys-Last-Disease"
#: Revisions that carry the protocol version, capabilities and identity in each
#: request's ``_meta`` and have no handshake. Newest first.
MODERN_VERSIONS = ("2026-07-28",)
#: Handshake revisions, newest first for version negotiation.
LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SUPPORTED_VERSIONS = MODERN_VERSIONS + LEGACY_VERSIONS
_SERVE_ENV = "HUMANITYS_LAST_DISEASE_MCP_SERVE"
_ROOTS_ENV = "RFD4_MCP_ROOTS"
_META = "_meta"
_VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
_SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"
#: JSON-RPC codes, and the MCP code for a version this server does not speak.
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
#: A request for a tool this server does not have, or one that is not shaped as
#: a tool call. The specification answers both as a protocol error, and its own
#: example for an unknown tool carries this code.
_INVALID_PARAMS = -32602
#: A failure in this server while answering, as opposed to a tool that failed.
_INTERNAL_ERROR = -32603
_SERVER_BUSY = -32000
_UNSUPPORTED_VERSION = -32022
#: How long a client may cache a list result, and who may cache it. The tool
#: list does not change while the process lives. The scope is private because
#: a result names directories on this computer.
_LIST_TTL_MS = 3_600_000
_CACHE_SCOPE = "private"
#: A frame that was not JSON, kept apart from a frame that was JSON but not a
#: request: the two are different JSON-RPC errors.
_UNREADABLE = object()
#: Shortest gap between two progress notifications for one call. The program
#: prints in bursts and the spec asks both ends to rate limit, so a burst is
#: reported once; every line is still returned when the call ends.
_PROGRESS_EVERY_S = 0.25
#: How long a line may be in a progress message before it is cut.
_PROGRESS_LINE_MAX = 300
#: Maximum body size for framed messages and line size for newline messages.
_MAX_FRAME = 8 * 1024 * 1024
#: Cumulative byte limit for the headers and their trailing blank line.
_MAX_FRAME_HEADERS = 8 * 1024
#: A batch is parsed as one frame and keeps every immediate reply until its
#: queued calls finish. Bound the item count separately from the frame size so
#: a tiny-invalid-item batch cannot turn one frame into hundreds of thousands
#: of Python objects and a response larger than the request.
_MAX_BATCH_ITEMS = 256
#: How often a call that has gone quiet says so again.
_HEARTBEAT_S = 20.0
#: When the client closes stdin, how long a call already running may take to
#: finish on its own, and then how long it gets after being told to stop. The
#: binding asks a server to exit promptly once its input is closed.
_SHUTDOWN_GRACE_S = 5.0
_SHUTDOWN_CANCEL_S = 5.0
#: One call may run while the rest wait. Bound all outstanding calls so a
#: client cannot fill memory while a long generation occupies the worker.
_MAX_OUTSTANDING_CALLS = 32

#: Tool refusals are readable by design, unlike unexpected exceptions. They
#: still cross the process boundary, so remove credentials and local paths
#: before returning their text. Kept here rather than importing the program's
#: publication helper: the protocol tests and the initial handshake deliberately
#: load this file without importing the program or its environment.
_ERROR_SECRET = re.compile(
    r"(?i)(?:"
    r"sk-[A-Za-z0-9_\-]{16,}|"
    r"xai-[A-Za-z0-9_\-]{16,}|"
    r"nvapi-[A-Za-z0-9_\-]{16,}|"
    r"hf_[A-Za-z0-9]{16,}|"
    r"ghp_[A-Za-z0-9]{16,}|"
    r"github_pat_[A-Za-z0-9_]{16,}|"
    r"AKIA[0-9A-Z]{16}"
    r")"
)
_ERROR_BEARER = re.compile(r"(?i)(authorization:\s*bearer\s+)\S+")
_ERROR_NAMED_SECRET = re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*([:=])\s*\S+")
_ERROR_WIN_PATH = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[/\\][^\s\"']+")
_ERROR_UNC_PATH = re.compile(r"(?<![\\])\\\\[^\\\s\"']+(?:\\[^\\\s\"']+)+")
_ERROR_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_:])/(?:[^/\s\"']+/)*[^/\s\"']+")
_ERROR_REDACTED = "[redacted]"
_ERROR_PATH = "<path>"
_INSTRUCTIONS = (
    "Humanitys-Last-Disease is this server. Use list_options, run, screen_directory, read_verdict, "
    "describe_architecture, retrieve_databases, retrieve_structures, read_sequence_measures, "
    "read_results, list_output_files, read_output_file, backend_status, declare_corpora, plan_validation, "
    "validate_output_files, save_validation_report, and list_validation_reports. "
    "describe_architecture names the AI interface, the MCP tools, and the computational backend. "
    "run takes one mode per call: the disease pipeline, benchmark answer, dynamics, H.U.M.A.N. protocol, "
    "screen, refold, or corpus declaration. "
    "Do not shell out to the program or parse its printed lines. "
    "Use plan_screening to inspect recommendations and run_selected_screens to run the chosen methods. "
    "Run returns structured structure_options, dynamics, and human_protocol records when those stages produce them. "
    "To select a returned structure reliably, pass its selector back through options.select_structure; repeat it "
    "for multiple rows. A stale selector is refused before download or generation. "
    "Use list_output_files and read_output_file for bounded access to regular artifacts in an approved finished run. "
    "Use plan_validation for the candidate-specific output validation checklist; pass explicit check IDs or select "
    "recommended or all_installed checks in one validation call. "
    "Use save_validation_report to run the selected checks and create a local JSON report in that run. "
    "Use list_validation_reports to find saved report filenames for the run. "
    "A run marked archived is a copy the generation archive made of another run's files. The "
    "copy is screened in place like any other run, so the verdict beside it is the copy's own "
    "and can differ from the verdict beside the run it was taken from. "
    "In a verdict, an axis whose source is axis_evidence was declared by a person and not measured; "
    "curated_axes names those axes beside the call. "
    "A relative path is inside this checkout and cannot leave it. "
    "The launched server confines every path to this checkout by default. An operator may explicitly "
    "widen that boundary with RFD4_MCP_ROOTS."
)
Handler = Callable[[str, Mapping[str, object] | None], Mapping[str, object]]
#: Where the worker leaves the progress callback for the call it is running.
#: It is held here rather than passed through ``Handler`` so that a handler
#: supplied to :func:`serve` stays a two-argument seam and needs to know nothing
#: about progress. One worker thread runs one call, so there is one slot.
_CURRENT = threading.local()

#: One axis as a schema entry: its state, what was measured where anything was,
#: the blocker that keeps it untested, and where the state came from. These
#: are ``mcp_api._AXIS_FIELDS``. No type is claimed for a measurement or a
#: blocker, because a measurement is a number on some axes and a name on
#: others, and a client that validated a guess here would reject a whole
#: structured result over a field it does not read.
_AXIS_SCHEMA: dict[str, object] = {
    "type": "object",
    "description": (
        "One entry per axis. An axis is clear, a hit, or untested with a named blocker. "
        "Read source before state: a state a person declared is not a measurement."
    ),
    "additionalProperties": {
        "type": "object",
        "properties": {
            "state": {"type": "string"},
            "measurement": {
                "description": (
                    "What was measured. On an axis whose source is axis_evidence this is the "
                    "measurement the declared state overrode, not the basis of that state."
                ),
            },
            "blocker": {},
            "source": {
                "type": "string",
                "description": (
                    "measured: the state is the measurement's. untested: nothing decided it. "
                    "axis_evidence: a person declared the state in axis_evidence.json, over whatever was measured."
                ),
            },
            "curated": {
                "description": (
                    "Only on a declared state. What the person recorded behind it, or the text "
                    "basis_not_stated when nothing was recorded."
                ),
            },
            "measured_state": {"description": "Only on a declared state. What the measurement itself said."},
            "measured_blocker": {
                "description": "Only on a declared state. Why the measurement could not decide, when it could not.",
            },
            "scan_limit": {
                "type": "string",
                "description": (
                    "Present when the table this verdict rests on did not cover the whole chain. "
                    "A hit still stands -- a wider scan can only add binders -- but the number is "
                    "a count over the part that was read. An absence is never reported this way: "
                    "a clear requires a scan that looked everywhere."
                ),
            },
            "threshold": {"description": "The cut this axis was judged against, in the axis's own units."},
            "threshold_source": {
                "type": "string",
                "description": (
                    "Where that cut came from. program_convention_10uM is this program's own "
                    "panel-wide convention and not a published bar for any one protein."
                ),
            },
            "reference_density": {
                "description": (
                    "The bar a density verdict was taken against, in the same units as the "
                    "measurement. Without it a clear is a number beside no comparison."
                ),
            },
            "units": {"type": "string", "description": "What the measurement is in."},
            "unscored_peptides": {
                "type": "integer",
                "description": (
                    "Windows the predictor declined to score. They are not non-binders, so a "
                    "clear rests on this being zero."
                ),
            },
            "not_nm_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Affinity columns whose unit the table never stated, so they were not read. "
                    "A hit on the columns that were read still stands; a clear cannot speak for "
                    "the rows these cover."
                ),
            },
            "presented_complete": {
                "type": "boolean",
                "description": "Whether the presented peptide set covered the whole design.",
            },
            "fold_source": {
                "type": "string",
                "description": (
                    "The study a plasma-fold verdict cites. The axis refuses to be measured "
                    "without one, so this is what the number rests on."
                ),
            },
            "fold_column": {"type": "string", "description": "Source column for the supplied fold measurement."},
            "linear_fold_magnitude": {
                "type": ["number", "null"],
                "description": "Linear fold magnitude compared with the configured threshold.",
            },
            "human_toxicity_state": {
                "type": "string",
                "description": "Human evidence status for a cross-species note.",
            },
            "reference_rank_scale_ambiguous": {
                "type": "boolean",
                "description": (
                    "The bar this verdict was taken against may be a hundredfold out: every "
                    "percentile rank in the reference reads the same whether that column is a "
                    "percentage or a fraction of one, and nothing published tells them apart. "
                    "The verdict stands, because refusing would refuse the ordinary "
                    "binders-only export. Read a clear against this before trusting it."
                ),
            },
            "reference_proteins_too_short": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Reference proteins shorter than every peptide width scored, so they hold "
                    "no window of that width and never competed to set the bar. Distinct from "
                    "a protein that was measured and carried no binder."
                ),
            },
            "method": {
                "type": "string",
                "description": (
                    "Which computation, or which supplied table, produced the measurement. A "
                    "score a person dropped beside the run and one this program computed are "
                    "both measurements, and without this they read alike."
                ),
            },
            "presentation": {
                "type": "string",
                "description": "Whether the scored windows were restricted to presented ligands, or not configured.",
            },
            "mask_unpublished_alleles": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Molecules whose anchor positions the scoring paper publishes no mask for, "
                    "so no score for them can be computed from anything published. The axis is "
                    "untested on these rather than clear."
                ),
            },
            "displaced_method": {
                "type": "string",
                "description": (
                    "The computation a supplied score displaced. An axis that reads untested on "
                    "a directory that holds the data for a verdict can only be read as such "
                    "when the displacement is named."
                ),
            },
            "displaced_table": {
                "type": "string",
                "description": "The table whose supplied score displaced that computation.",
            },
            "covered_examples": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The designs of this run that the table behind the state is a scan of. A run "
                    "holds many designs and one supplied table is one scan: a clear covers these "
                    "designs and says nothing about the others. The design-level counterpart of "
                    "scan_limit, which is about one chain."
                ),
            },
            "uncovered_examples": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Designs the table is not a scan of. No predictor saw these, and the state "
                    "does not speak for them."
                ),
            },
            "assay": {
                "type": "string",
                "description": (
                    "Which assay the measurement was read from. units says what the number is "
                    "in; this says how it was obtained."
                ),
            },
            "network": {
                "type": "string",
                "description": "Which interaction network a partner count was taken over.",
            },
            "confidence": {
                "description": (
                    "The score a partner had to reach to be counted, in the source's own scale. "
                    "A degree is a number about whatever this cut admitted."
                ),
            },
            "release": {
                "type": "string",
                "description": (
                    "The version of the source database the measurement came from, when the "
                    "response names one. A source that re-derives its network every release "
                    "cannot be queried for the same number later without it."
                ),
            },
            "reference_file": {
                "type": "string",
                "description": (
                    "The reference the search ran against, by name. A clear here is an absence "
                    "claim about this file and no other."
                ),
            },
            "reference_digest": {
                "type": "string",
                "description": (
                    "A digest of that reference, so a swapped or rebuilt file is visible between "
                    "two runs that both read one named the same."
                ),
            },
            "reference_bases": {
                "type": "integer",
                "description": (
                    "How large that reference is. A clear against a stub is not a clear against "
                    "a transcriptome, and the two read identically without this."
                ),
            },
            "trigger": {
                "description": (
                    "Which number decided this axis, rank or ic50. These axes carry no units and "
                    "no threshold, so without it the measurement is a bare number that is a "
                    "percentile rank on one table and an affinity in nM on another."
                ),
            },
            "rank": {
                "description": "The best percentile rank in the table, when it held ranks.",
            },
            "ic50": {
                "description": "The best affinity in nM, when the table held affinities.",
            },
            "binding_peptides": {
                "type": "integer",
                "description": (
                    "How many peptides of the table were counted as binding: the numerator of the "
                    "density that reference_density is the bar for."
                ),
            },
            "reference_method_undeclared": {
                "type": "boolean",
                "description": (
                    "The bar this verdict was taken against came from a tool neither table named, "
                    "so the two were compared on the assumption that one method produced both."
                ),
            },
            "design_method": {"type": "string", "description": "Method recorded for the design results."},
            "reference_method": {"type": "string", "description": "Method recorded for the reference results."},
            "restricting_method": {"type": "string", "description": "Method recorded for the restricting evidence."},
            "proteome_kmers": {
                "type": "integer",
                "description": (
                    "How many distinct windows the proteome file held. A clear here is an absence "
                    "claim about that file, and a clear against a small one reads the same without this."
                ),
            },
            "proteome_digest": {
                "type": "string",
                "description": "A digest of that proteome, so a swapped or rebuilt file is visible.",
            },
            "seed_width": {
                "type": "integer",
                "description": "The window width the hybridization search ran at.",
            },
            "seeds_searched": {
                "type": "integer",
                "description": "How many distinct seeds were searched: the denominator of a clear at zero matches.",
            },
            "qualifier": {
                "type": "string",
                "description": (
                    "The bound the panel cell was written with, such as a greater-than. The "
                    "measurement stays the figure the table printed, so without this a verdict "
                    "read off a bound looks like one read off a number."
                ),
            },
            "n_readings": {
                "type": "integer",
                "description": "How many cells named this axis. The measurement is the lowest of them.",
            },
            "unparsed_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Columns whose cells could not be read, which is why this axis is untested. "
                    "The sibling of not_nm_columns."
                ),
            },
            "calis_mask_allele_specific": {
                "type": "boolean",
                "description": (
                    "Whether the position mask was applied per molecule. False says one published "
                    "mask was applied to every one of them, which the method name alone implies otherwise."
                ),
            },
            "compara_examined": {
                "type": "integer",
                "description": "Number of homology records returned by Ensembl Compara.",
            },
            "compara_scored": {
                "type": "integer",
                "description": "Number of those records with a usable percent identity.",
            },
            "compara_blocker": {
                "type": "string",
                "description": (
                    "Why that pass measured nothing, beside a record whose own blocker still "
                    "speaks for the cluster result."
                ),
            },
            "alleles": {
                "type": "integer",
                "description": "How wide the allele panel the median was taken over was.",
            },
            "peptides_scored": {
                "type": "integer",
                "description": (
                    "How many peptides the rule was applied to. A clear over two peptides and a "
                    "clear over sixty-two are not the same claim."
                ),
            },
            "binding_alleles": {
                "type": "integer",
                "description": (
                    "The most panel alleles any one peptide bound. A clear says no peptide's "
                    "median reached the cut; this says how close the nearest came."
                ),
            },
            "presented_ligands": {
                "type": "integer",
                "description": "How many presented windows were scored at all: the width of this clear.",
            },
            "positive_windows": {
                "type": "integer",
                "description": "Windows scoring above the cut. The number behind a blocker of no_pass_cut.",
            },
            "positive_examples": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The designs those windows were found in. positive_windows says how many "
                    "reached the cut; this says which design to go and read. Empty when the "
                    "count is zero, because then there is nothing to point at."
                ),
            },
            "promiscuous_peptides": {
                "type": "integer",
                "description": "Peptides binding more than one allele of the panel.",
            },
            "cpg_count": {
                "type": "integer",
                "description": (
                    "How many motif occurrences the design holds. This axis reaches no clear, so "
                    "it is the number an undecided verdict turns on."
                ),
            },
            "window_residues": {
                "type": "integer",
                "description": "The window width a span count was taken at, as seed_width is for a hybridization scan.",
            },
            "failed": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The designs that made this run-level axis a hit: the hit-side counterpart of "
                    "covered_examples. Empty on a clear, because a clear is about all of them."
                ),
            },
            "same_shape": {
                "type": "boolean",
                "description": "Whether a co-evolution table reported the target keeping its shape in other species.",
            },
            "coevolution_signal": {
                "type": "string",
                "description": (
                    "The co-evolution note the rows agreed on. It is the key that selects the "
                    "sentence a reader is meant to get, and no client saw it until now."
                ),
            },
            "coevolution_level": {
                "type": "string",
                "description": "How conserved the target is across the species the table covered.",
            },
            "other_species_toxicity": {
                "type": "string",
                "description": "What the table reported about harm in those other species.",
            },
            "coevolution_rows": {
                "type": "integer",
                "description": "How many rows agreed on that note.",
            },
            "coevolution_blocker": {
                "type": "string",
                "description": (
                    "Why there is no co-evolution note: not supplied, not classified, unparsed, or "
                    "rows that disagreed with each other."
                ),
            },
        },
    },
}
#: The account of each threshold a block was read against, nested inside that
#: block. Two blocks carry one: ``cuts`` and ``designed_interface_quality``.
#:
#: Nothing is required. ``mcp_api._provenance`` checks each field and drops a
#: malformed one while returning the rest of the entry, the way an axis record
#: is checked, so an entry can legitimately arrive missing any field -- and a
#: schema that demanded one would make a validating client throw away every run
#: in the answer over one hand-edited caveat.
#:
#: The kind vocabulary is described here and declared in
#: ``rfproteina.disease.threshold_provenance.KINDS``. It is not listed as an
#: enum, because this file must not import the package to answer discovery, and
#: a copied list would be the drift this record exists to prevent: that set grew
#: by two kinds while this schema was being written. The check that matters runs
#: where the import is already paid for, and names an unknown kind under
#: ``malformed`` rather than passing it on as though it were one.
_PROVENANCE_SCHEMA: dict[str, object] = {
    "type": "object",
    "description": (
        "One entry per threshold in this block, keyed by the thing it applies to: what kind of "
        "number the bar is and where it came from. Read it before reading a pass. A cut this "
        "program adopted as its own convention, or borrowed from another project's defaults, is "
        "not a published one, and a clear against a borrowed default whose quantity that project "
        "counts differently says less than it appears to."
    ),
    "additionalProperties": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "description": (
                    "published: a cut from a cited paper, or from the documented application of "
                    "one. recommended: a tool's own recommended selection. convention: this "
                    "program's strictness, with no literature value behind it. borrowed: another "
                    "project's default, applied here to a quantity that project may count "
                    "differently. reported_only: measured, with no cut adopted on purpose. "
                    "caller_supplied: no default exists and the axis stays unset until an operator "
                    "passes one. measurement_parameter: shapes what is measured rather than "
                    "deciding a pass. unrecorded: no entry was found, which is a defect rather "
                    "than a kind."
                ),
            },
            "source": {
                "type": "string",
                "description": (
                    "The stable slug naming this threshold's origin. Two of them are also written "
                    "beside the cut itself, as gravy_max_source and instability_max_source."
                ),
            },
            "citation": {
                "type": "string",
                "description": (
                    "Present only on a published kind. Its absence on the others is the point "
                    "rather than an omission waiting to be filled in."
                ),
            },
            "caveat": {
                "type": "string",
                "description": (
                    "What the number does not cover: a scale it does not transfer to, a definition "
                    "that differs from the one it was fitted on, or a pass deliberately not adopted."
                ),
            },
            "bar": {
                "description": (
                    "The threshold itself, as a min or a max object. Written on the interface "
                    "record, where the measurement and the bar it is read against would otherwise "
                    "sit in two different files. Null where no cut is adopted, which is an answer "
                    "and not a gap."
                ),
            },
        },
    },
}
#: The decision fields of one screen. These are ``mcp_api._VERDICT_FIELDS``
#: plus the axis groups, which are ``mcp_api._AXIS_GROUPS``, the fields
#: ``_verdict`` derives, and the optional full record. A test holds the lists
#: together. The wire shape lives here rather than beside those constants
#: because declaring it must not import the package: that import takes about
#: fifty seconds, and tool discovery has to answer at once.
_VERDICT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "call": {
            "type": "string",
            "description": "The screen's call. screened_clear is every axis in the named bar clear.",
        },
        "bar": {
            "type": "string",
            "description": (
                "The named set of axes this verdict was taken against. The axes themselves are "
                "bar_axes: a name does not survive an edit to the bar it names."
            ),
        },
        "bar_axes": {
            "description": (
                "The axes that bar required, read from the bar table at the moment of the "
                "verdict, so a stored screened_clear keeps meaning what it meant when it was "
                "written. The text absent on a panel target, which names no axes, and on a "
                "binder kind that has no bars at all."
            ),
        },
        "states": {
            "type": "object",
            "description": "One state per axis: hit, clear, or untested.",
            "additionalProperties": {"type": "string"},
        },
        "untested": {"type": "array", "items": {"type": "string"}},
        "hits": {"type": "array", "items": {"type": "string"}},
        "ppi_pass": {},
        "ppi_blocker": {},
        "pre_oracle_pass": {},
        "requested_class": {},
        "target_accession": {},
        "job_mismatch": {},
        "screen_version": {
            "type": "integer",
            "description": "The record shape this screen was written in. Absent on a screen older than the number.",
        },
        "code": {
            "type": "object",
            "description": "Which code applied the cuts, as the screen recorded it.",
        },
        "cuts": {
            "type": "object",
            "description": (
                "Every number this verdict was taken against, as the screen recorded them, each "
                "cut that is this program's own convention carrying a source that says so. A "
                "verdict is reproducible only against its own cuts: edit one later and every "
                "stored screen silently changes meaning. provenance says, for each cut, what "
                "kind of number it is. read_results drops provenance for size and returns the "
                "cuts themselves; this tool returns both."
            ),
            "properties": {"provenance": _PROVENANCE_SCHEMA},
        },
        "metrics_file": {
            "description": (
                "The per-design metrics table this screen read, relative to the run. The primary "
                "file wins and the reeval file stands in when it is absent, so two verdicts on "
                "one run can rest on different measurements. Null when no table was read."
            ),
        },
        "metrics_blocker": {
            "type": "string",
            "description": (
                "Why the per-design table yielded no rows: refused for a repeated column, "
                "unreadable, absent. Absent from the verdict when it yielded rows. A verdict "
                "carrying this is not a run nobody scored."
            ),
        },
        "chain_blocker": {
            "type": "string",
            "description": (
                "Present when the table names more than one chain of interest. The top-level "
                "sequence and interface blocks repeat the first design, so this says why the "
                "later designs' measurements are empty."
            ),
        },
        "refold": {
            "type": "object",
            "description": (
                "Every pass criterion this repository declares, with each bound, the worst value "
                "the run reached against it, and which bounds were missed, beside the criteria "
                "file they were read from. Recorded without moving the call, as ppi_pass is: a "
                "false ppi_pass names none of the bounds behind it and this does. "
                "refold_blocker says why nothing was evaluated, which on a run with no oracle "
                "configured is every criterion at once."
            ),
        },
        "solubility": {
            "type": "object",
            "description": (
                "Solubility rows supplied beside the run, each with the method and the study "
                "behind it, or solubility_blocker saying why there are none. The scales are not "
                "comparable and no published threshold spans them, so this judges nothing and "
                "clears no axis."
            ),
        },
        "protein_sol": {
            "type": "object",
            "description": "Native Protein-Sol scaled solubility predictions and coverage blockers for each sample.",
        },
        "tnp_profile": {
            "type": "object",
            "description": "TNP developability descriptors, clinical reference flags, and per-design coverage.",
        },
        "vina_pose": {
            "type": "object",
            "description": "Per-design fixed-pose Vina scores in kcal/mol, checked input identities and blockers.",
        },
        "prodigy_affinity": {
            "type": "object",
            "description": (
                "Post-generation PRODIGY protein-pair estimates. Each sample row carries its example_id, "
                "method, predicted free energy in kcal/mol and derived Kd in mol/L, or a named blocker. "
                "The report gives the number of predictions and all attempted structures."
            ),
        },
        "design_diversity": {
            "type": "object",
            "description": (
                "The spread across this run's designed chains: how many were readable, how many "
                "are distinct, and the identity and k-mer comparisons behind that. The one "
                "measurement about the set rather than about a design -- designs 8 with distinct "
                "1 is one molecule clearing every axis eight times."
            ),
        },
        "designed_interface_quality": {
            "type": "object",
            "description": (
                "The first design's buried area, interface residues, polar satisfaction and "
                "Lawrence-Colman Sc, with the reference_thresholds they are read against, or "
                "interface_quality_blocker saying why they are missing. A report recorded beside the "
                "verdict: it does not move the call. provenance says what kind of bar each "
                "threshold is, and carries the bar itself: three of the five are another "
                "project's defaults for quantities it counts differently, and two adopt no cut "
                "at all, so a measurement here is not read against a published standard."
            ),
            "properties": {"provenance": _PROVENANCE_SCHEMA},
        },
        "ligand_alerts": {
            "type": "object",
            "description": (
                "RDKit structure alerts for a ligand SMILES supplied beside the run, or "
                "ligand_alerts_blocker. Recorded, not judged: a catalogue hit is a flag for review, "
                "and the states under verdicts are this report's own, not axis states, and are not "
                "counted in hits. The SMILES itself stays in screen.json."
            ),
        },
        "method_selection": {"type": "object", "description": "Chosen, attempted and inapplicable methods."},
        "sygma": {
            "type": "object",
            "description": (
                "Possible human metabolites ranked separately for each nominated parent. Each product carries its "
                "InChIKey, score, pathway and unresolved stereochemistry count. The local screen.json contains "
                "the parent, intermediate and product structures as isomeric SMILES."
            ),
        },
        "qupkake": {
            "type": "object",
            "description": (
                "QupKake site-specific aqueous microscopic pKa. Each measurement identifies its direction, "
                "mapped heavy atom, and conjugate acid/base state IDs. Includes per-nomination coverage and blockers. "
                "The atom-mapped structures are available in local screen.json."
            ),
        },
        "ligand_solubility": {
            "type": "object",
            "description": (
                "ADMET-AI AqSolDB aqueous log10 molar solubility for each distinct nominated compound. "
                "Includes signed predictions, coverage, blockers and corpus provenance. "
                "Compound structures stay in the local screen file."
            ),
        },
        "mhc_class_i": {
            "type": "object",
            "description": (
                "The first design's class I peptide-window count for a protein or peptide run, "
                "and whether the scan that writes a table can run: whether MHCflurry, its "
                "presentation weights and an allele list are each present, and whether an "
                "mhc_i.tsv is already beside the run. This report can decide a verdict. When "
                "all three are present and no table is there, the screen writes mhc_i.tsv "
                "itself and the mhc_i axis reads it, so what scan says the write did, and "
                "scan_blocker says about a predictor that failed, is what that axis rests on. "
                "reference_reused true means the tolerated-protein reference table was already "
                "beside the run and was not rescanned, which is the ordinary case after the "
                "first screen of a directory. The reference table's path stays in screen.json, "
                "so whether it exists travels under scan and reference_reused and not as a path."
            ),
        },
        "synthesis": {
            "type": "object",
            "description": (
                "Construct-synthesis feasibility for the first design of a nucleic run, or "
                "synthesis_blocker. Recorded, not judged: every synthesis bar is a suggested default, "
                "and the states under verdicts are not axis states and are not counted in hits. The "
                "repeated subsequence itself stays in screen.json; longest_repeat gives its length."
            ),
        },
        "replaced": {
            "type": "object",
            "description": (
                "Present when this screen overwrote one with a different verdict. "
                "changed_inputs is absent when no supplied file was added, removed or edited."
            ),
            "properties": {
                "call": {"type": "string", "description": "The call of the screen that was overwritten."},
                "changed_inputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"file": {"type": "string"}, "change": {"type": "string"}},
                        "required": ["file"],
                    },
                },
            },
        },
        "inputs": {
            "type": "array",
            "description": (
                "Every file supplied beside the run, read or refused, each with its size in bytes "
                "and a digest, and link true when the name is a symlink. A verdict rests on these: "
                "the predictor tables, the toxicology panel, the co-evolution notes, the solubility "
                "scores, the curated evidence. A verdict that cannot name its inputs cannot be "
                "checked against them, and replaced.changed_inputs names only the ones that MOVED, "
                "on a re-screen whose verdict changed -- a first screen has no replaced at all. "
                "Two sentinel forms, both a list of one entry: file inputs_not_listed means the "
                "folder refused to be listed, and file inputs_truncated with files set means there "
                "were more files than the writer will record. Both are withheld whole rather than "
                "cut short, because a truncated list reads as a complete one and every file past "
                "the cut would look absent. Only the run folder's own files; the designs and the "
                "metrics table are named elsewhere in this record. A symlink's target is not "
                "written down -- it is a path outside the run and often a personal one, and the "
                "digest already says whether the bytes moved."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "The file's own name, never a path."},
                    "bytes": {"type": "integer"},
                    "digest": {
                        "type": "string",
                        "description": (
                            "file_too_large when the file was past the digest limit, so a swapped "
                            "table is still visible by size."
                        ),
                    },
                    "link": {"type": "boolean"},
                    "files": {
                        "type": "integer",
                        "description": "Only on the inputs_truncated sentinel: how many files were found.",
                    },
                },
            },
        },
        "chemical_liability": {
            "type": "object",
            "description": (
                "The first design's motif counts, free cysteine, isoelectric point and net charge "
                "at pH 7.4, with pka_set naming the scale the charge was computed on. Counts come "
                "with the residue positions behind them: deamidation, isomerization, glycosylation "
                "sequons, methionines and tryptophans. A report recorded beside the verdict -- it "
                "carries no state and no axis clears on it, and a chain with a non-standard residue "
                "has none of it. isoelectric_point_blocker stands in where the charge never crosses "
                "zero between pH 0 and 14. read_results omits this block for size."
            ),
        },
        "modality_screen": {
            "type": "object",
            "description": (
                "The drug-type report for this binder kind, and the largest block here that moves "
                "no call. A protein or peptide carries the furin and Asp-Pro motif counts, "
                "N-terminal glutamine, non-standard letters, the aliphatic index, the molecular "
                "weight, the Pace 1995 extinction and absorbance at 280 nm for the reduced and "
                "cystine forms, the N-end rule half-life in three organisms, the residue and atom "
                "composition, sequence entropy, a Kyte-Doolittle transmembrane span count and an "
                "AGGRESCAN 2007 normalized sum. A nucleic binder carries GC fraction, CpG count "
                "and positions, and the longest G run. A ligand carries one formula weight and the "
                "component composition. Each measurement names its own source where it has one "
                "(extinction_source, half_life_source, molecular_weight_source, "
                "transmembrane_method). A weight is left out rather than guessed when a letter has "
                "no residue mass. read_results omits this block for size."
            ),
        },
        "sequence_liability": _AXIS_SCHEMA,
        "target_liability": _AXIS_SCHEMA,
        "nucleic_liability": _AXIS_SCHEMA,
        "toxicology": _AXIS_SCHEMA,
        "curated_axes": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Axes whose state a person declared in axis_evidence.json. A call that rests on one "
                "was not measured on that axis. Absent when the screen does not record where states came from."
            ),
        },
        "malformed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Stored fields left out because their type is not the one declared here.",
        },
        "record": {
            "type": "object",
            "description": "The stored screen, sequences and personal paths removed. Only with include_record.",
        },
    },
}


def _directory_schema(*, required: bool, climbs: bool = False) -> dict[str, object]:
    """One directory argument. ``backend_status`` may omit it.

    ``climbs`` adds the opt-in for a file the program writes once per output
    directory rather than once per run.
    """
    properties: dict[str, object] = {"directory": {"type": "string"}}
    if climbs:
        properties["from_output_dir"] = {
            "type": "boolean",
            "description": (
                "Answer from the nearest output directory above this one when this directory holds "
                "no file of its own, and name it under read_from. Off by default: a lookup covering "
                "an output directory is not a measurement of one run inside it."
            ),
        }
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = ["directory"]
    return schema


def _stored_schema(*, directory: bool | None, tree: bool = False, climbs: bool = False) -> dict[str, object]:
    """program and status. Extra stored fields stay allowed.

    ``tree`` adds what a tool that walks every folder under the directory
    reports about the walk itself. ``climbs`` adds where a tool found a file the
    program writes once per output directory rather than once per run.
    """
    properties: dict[str, object] = {
        "program": {"type": "string"},
        "status": {"type": "string"},
    }
    required = ["program", "status"]
    if directory is not None:
        properties["directory"] = {"type": "string"}
        if directory:
            required.append("directory")
    if climbs:
        properties["read_from"] = {
            "type": "string",
            "description": (
                "Present only when from_output_dir was passed and the file behind this answer was "
                "found in an ancestor rather than the named directory. The answer then covers every "
                "run under that output directory and is not a measurement of this one."
            ),
        }
    if tree:
        properties["complete"] = {
            "type": "boolean",
            "description": "False when the walk stopped before every folder was read.",
        }
        properties["unreadable"] = {
            "type": "array",
            "description": "Files the walk could not read. Everything else under the directory is still returned.",
            "items": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string"},
                    "file": {"type": "string"},
                    "error": {"type": "string"},
                },
                "required": ["folder", "file", "error"],
            },
        }
    return {"type": "object", "properties": properties, "required": required}


def _screens_schema(*, ran: bool) -> dict[str, object]:
    """The verdicts under a directory.

    ``ran`` adds the fields a call that also invoked the program returns, so
    reading a stored screen and screening afresh do not claim one shape.
    """
    entry: dict[str, object] = {
        "type": "object",
        "properties": {
            "directory": {"type": "string"},
            "label": {
                "type": "string",
                "description": "Path relative to the directory asked for. Empty for that directory itself.",
            },
            "lines": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The same sentences the command prints for this screen.",
            },
            "verdict": _VERDICT_SCHEMA,
            "error": {
                "type": "string",
                "description": (
                    "Present when a screen file is there and cannot be read, or was read and could not "
                    "be shown in full. The other runs are still returned."
                ),
            },
            "written": {
                "type": "boolean",
                "description": (
                    "Set by screen_directory. True when the screen file was written while this call ran: "
                    "its modification time, inode or size is not what it was before. False means the "
                    "verdict shown is the stored one, of whatever age. It does not mean the verdict is current."
                ),
            },
            "changed": {
                "type": "boolean",
                "description": (
                    "Set by screen_directory. True when the file's bytes differ from before the call. "
                    "A rescreen that reaches the same verdict is written and not changed."
                ),
            },
            "archived": {
                "type": "boolean",
                "description": (
                    "True when this run is a copy the generation archive made under the output "
                    "directory, not the run it was taken from. A walk returns both, so counting "
                    "an archived copy as a second run counts one run's designs twice. Only the "
                    "files are a copy: the copy is screened in place like any other run, so the "
                    "verdict here is the copy's own and can differ from the one beside the run it "
                    "was taken from. Dropping every archived row can therefore drop the only "
                    "measured verdict for that target."
                ),
            },
            "archive_run_id": {
                "type": "string",
                "description": (
                    "That copy's own id, from the record the archive wrote beside it. It is what "
                    "tells two copies of one target apart."
                ),
            },
        },
        "required": ["directory", "label"],
    }
    properties: dict[str, object] = {
        "program": {"type": "string"},
        "directory": {"type": "string"},
        "complete": {
            "type": "boolean",
            "description": "False when the walk stopped before every folder was read, so runs may be missing.",
        },
        "runs": {"type": "array", "items": entry},
    }
    required = ["program", "directory", "complete", "runs"]
    if ran:
        properties["exit_code"] = {"type": "integer"}
        properties["messages"] = {"type": "array", "items": {"type": "string"}}
        required = [*required, "exit_code", "messages"]
    return {"type": "object", "properties": properties, "required": required}


_VALIDATION_SCOPE = {
    "directory": {"type": "string", "description": "One finished run under runs, disease_runs, or RFD4_MCP_ROOTS."},
    "candidate_id": {
        "type": "string",
        "pattern": "^[0-9a-f]{24}$",
        "description": "Optional candidate ID returned for this run by plan_validation.",
    },
}
_VALIDATION_CHECKS_INPUT = {
    "type": "object",
    "properties": {
        **_VALIDATION_SCOPE,
        "checks": {
            "type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True,
            "description": "Explicit installed check IDs from plan_validation.",
        },
        "selection": {
            "type": "string",
            "enum": ["recommended", "all_installed"],
            "description": "Resolve the current installed recommendations inside this call instead of passing checks.",
        },
    },
    "required": ["directory"],
    "oneOf": [
        {"required": ["checks"], "not": {"required": ["selection"]}},
        {"required": ["selection"], "not": {"required": ["checks"]}},
    ],
    "additionalProperties": False,
}
_VALIDATION_RESULT_PROPERTIES = {
    "program": {"type": "string"},
    "candidate_id": {"type": ["string", "null"]},
    "results": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "result": {"type": "string"}},
            "required": ["id", "result"],
        },
    },
}

_TOOLS: tuple[dict[str, object], ...] = (
    {
        "name": "list_options",
        "description": (
            "List every Humanitys-Last-Disease option by the name run accepts. "
            "Does not start the program or open the browser."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "outputSchema": {
            "type": "object",
            "properties": {
                "program": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "flag": {"type": "string"},
                            "kind": {"type": "string"},
                            "help": {"type": "string"},
                            "choices": {"type": "array", "items": {"type": "string"}},
                            "default": {},
                        },
                        "required": ["name", "flag", "kind", "help"],
                    },
                },
                "schema": {
                    "type": "object",
                    "description": "The same options as a JSON Schema for the run tool's options object.",
                },
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["program", "options", "schema", "notes"],
        },
    },
    {
        "name": "run",
        "description": (
            "Run Humanitys-Last-Disease with the selected options. Does not open the browser and does not prompt. "
            "One mode per call: the disease pipeline, benchmark answer, dynamics, H.U.M.A.N. protocol, "
            "screening, refolding, or corpus declaration. "
            "Only the pipeline would ask a question, so only it needs accept_only, or choice when several "
            "disease names match. "
            "Generation starts only when options.run is true. "
            "Returns the exit code and the program's own lines, with the refold mode's own record as refold. "
            "Screen a directory and read its verdict with screen_directory instead of parsing those lines."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "description": (
                        "Option names from list_options, which also returns them as a schema. "
                        "false omits a flag. Do not pass API keys."
                    ),
                }
            },
            "required": ["options"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "program": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": [
                        "disease_pipeline", "declare_corpora", "benchmark_answer", "dynamics",
                        "human_protocol", "screen", "refold",
                    ],
                },
                "exit_code": {"type": "integer", "description": "0 succeeded. 2 is a usage failure."},
                "messages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The program's own lines. Read a verdict from verdict, not from these.",
                },
                "verdict": _screens_schema(ran=False),
                "refold": {
                    "type": "object",
                    "description": (
                        "The refold mode's own record, which the program prints as one JSON document: the "
                        "run and the oracle name, then the requests directory with the export counts and "
                        "every design refused by name, or the import counts, and last the round trip -- how "
                        "many designs were requested, how many came back, and which are still outstanding. "
                        "Present when options.refold was set and that document could be read; the lines it "
                        "was read from are in messages either way."
                    ),
                },
                "benchmark_answers": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Structured answers printed by options.benchmark_answer.",
                },
                "corpora": {
                    "type": "object",
                    "description": "Structured corpus declarations when options.declare_corpora is true.",
                },
                "structure_options": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Numbered structure choices from the disease pipeline. Each has an opaque selector that can "
                        "be passed through options.select_structure without depending on live-list ordering."
                    ),
                },
                "dynamics": {
                    "type": "object",
                    "description": "The structured conformational-dynamics record produced by dynamics mode.",
                },
                "human_protocol": {
                    "type": "object",
                    "description": "The structured H.U.M.A.N. protocol record or anti-target proposal.",
                },
            },
            "required": ["program", "mode", "exit_code", "messages"],
        },
    },
    {
        "name": "screen_directory",
        "description": (
            "Screen a finished generation directory, or every finished run beneath it, and return each "
            "verdict. Does not generate, download, or start a shell. "
            "lookup_targets contacts UniProt, the Human Protein Atlas, and STRING only when true. "
            "follow_up writes the next commands and does not start them. "
            "screened_clear means every axis in the named bar is clear. "
            "An axis whose source is axis_evidence was declared by a person, not measured. "
            "out_dir and log_file are written to: a relative path cannot leave this checkout, and when "
            "RFD4_MCP_ROOTS is set every path must be inside this checkout or a directory it names."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "A run directory, or a parent of run directories."},
                "screen_selection": {"type": "string", "enum": ["recommended", "all_installed", "explicit"]},
                "screen_methods": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                "binder_kind": {
                    "type": "string",
                    "enum": ["protein", "ligand", "peptide", "nucleic"],
                    "description": "Set only when the run's own files do not name the kind.",
                },
                "bar": {
                    "type": "string",
                    "description": "Which named set of axes must all be clear. Defaults to the kind's own bar.",
                },
                "proteome": {"type": "string"},
                "lookup_targets": {"type": "boolean"},
                "plasma_abundance_max": {"type": "number"},
                "plasma_hub_max": {"type": "integer"},
                "plasma_fold_max": {"type": "number"},
                "paralog_identity_max": {"type": "number"},
                "follow_up": {"type": "boolean"},
                "stage_tables": {"type": "boolean"},
                "out_dir": {"type": "string"},
                "log_file": {"type": "string"},
                "include_record": {
                    "type": "boolean",
                    "description": "Include the stored screen with sequences removed. The verdict is always included.",
                },
            },
            "required": ["directory"],
            "additionalProperties": False,
        },
        "outputSchema": _screens_schema(ran=True),
    },
    {
        "name": "read_verdict",
        "description": (
            "Read stored screen verdicts under a directory. Does not rescreen, write, generate, or start a shell. "
            "screened_clear means every axis in the named bar is clear. "
            "An axis whose source is axis_evidence was declared by a person, not measured. "
            "A run marked archived is a copy the generation archive made of another run's files in the "
            "same tree, so a tally that counts its designs separately counts that run's designs twice. "
            "Its verdict is the one last screened inside the copy, which can differ from the verdict "
            "beside the run it was taken from, so dropping every archived run can drop the only "
            "measured verdict for that target."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string"},
                "include_record": {"type": "boolean"},
            },
            "required": ["directory"],
            "additionalProperties": False,
        },
        "outputSchema": _screens_schema(ran=False),
    },
    {
        "name": "describe_architecture",
        "description": (
            "Describe the three layers: the AI interface, the MCP tools, and the computational backend. "
            "Does not call a model, start generation, or use the GPU."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "outputSchema": _stored_schema(directory=None),
    },
    {
        "name": "retrieve_databases",
        "description": (
            "Read stored database statuses for one directory. Does not contact a database. "
            "A directory with no stored lookup is not_recorded. The lookup is written once per "
            "output directory, not once per run, so a run directory has none of its own: pass "
            "from_output_dir to be answered from the output directory above it instead, which "
            "read_from then names."
        ),
        "inputSchema": _directory_schema(required=True, climbs=True),
        "outputSchema": _stored_schema(directory=True, climbs=True),
    },
    {
        "name": "retrieve_structures",
        "description": (
            "Read stored structure rows for one directory. Does not download a structure or return its sequence. "
            "The structure list is written once per output directory, not once per run, so pass from_output_dir "
            "to be answered from the output directory above a run, which read_from then names."
        ),
        "inputSchema": _directory_schema(required=True, climbs=True),
        "outputSchema": _stored_schema(directory=True, climbs=True),
    },
    {
        "name": "read_sequence_measures",
        "description": (
            "Read stored sequence measurements for one directory. "
            "The sequence text is not returned. Does not rescreen."
        ),
        "inputSchema": _directory_schema(required=True),
        "outputSchema": _stored_schema(directory=True),
    },
    {
        "name": "read_results",
        "description": (
            "Read stored research traces, generation records, and screen verdicts. Does not rescreen or generate."
        ),
        "inputSchema": _directory_schema(required=True),
        "outputSchema": _stored_schema(directory=True, tree=True),
    },
    {
        "name": "list_output_files",
        "description": (
            "List regular output artifacts in one approved finished run. Symbolic links, hidden files, "
            "and credential filenames are excluded. Pass inventory_sha256 as expected_sha256 on later pages "
            "to require the same listing of paths, sizes and media types."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": _VALIDATION_SCOPE["directory"],
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "expected_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
            "required": ["directory"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "program": {"type": "string"},
                "directory": {"type": "string"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "bytes": {"type": "integer", "minimum": 0},
                            "media_type": {"type": "string"},
                        },
                        "required": ["path", "bytes", "media_type"],
                    },
                },
                "next_offset": {"type": ["integer", "null"]},
                "total_files": {"type": "integer", "minimum": 0},
                "inventory_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
            "required": ["program", "directory", "files", "next_offset", "total_files", "inventory_sha256"],
        },
    },
    {
        "name": "read_output_file",
        "description": (
            "Read a chunk from a regular artifact in the approved run. Each call checks the complete file for "
            "credential-like content and returns its SHA-256. Pass that digest as expected_sha256 on subsequent "
            "pages. The encoding applies to this chunk. The file limit is 256 MiB."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": _VALIDATION_SCOPE["directory"],
                "path": {"type": "string", "maxLength": 512},
                "offset": {"type": "integer", "minimum": 0},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": 1048576},
                "expected_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
            "required": ["directory", "path"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "program": {"type": "string"},
                "directory": {"type": "string"},
                "path": {"type": "string"},
                "media_type": {"type": "string"},
                "encoding": {"type": "string", "enum": ["utf-8", "base64"]},
                "content": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "bytes_returned": {"type": "integer", "minimum": 0},
                "total_bytes": {"type": "integer", "minimum": 0},
                "next_offset": {"type": ["integer", "null"]},
                "chunk_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "file_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
            "required": [
                "program", "directory", "path", "media_type", "encoding", "content", "offset",
                "bytes_returned", "total_bytes", "next_offset", "chunk_sha256", "file_sha256",
            ],
        },
    },
    {
        "name": "backend_status",
        "description": (
            "Report the computational backend for one directory, or the backend with nothing recorded. "
            "Does not start GPU work, docking, or a prediction."
        ),
        "inputSchema": _directory_schema(required=False),
        "outputSchema": _stored_schema(directory=False, tree=True),
    },
    {
        "name": "declare_corpora",
        "description": (
            "Declare what every embedded model was trained on, and this program's coverage of the "
            "DDD Benchmark. Pass 'against' to name a dataset or assay and get each model's overlap "
            "with it; overlap is a property of the (model, dataset) pair, not of the model. "
            "Reads nothing from disk and contacts nothing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "against": {
                    "type": "string",
                    "description": (
                        "A dataset or assay name to report overlap against, such as 'HIC'. "
                        "Omit it to get the declarations without a pairing."
                    ),
                }
            },
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "models": {"type": "array", "items": {"type": "object"}},
                "corpora": {"type": "object"},
                "benchmark_coverage": {"type": "object"},
            },
            "required": ["models", "corpora", "benchmark_coverage"],
            "additionalProperties": False,
        },
    },
    {
        "name": "plan_validation",
        "description": (
            "List candidate IDs and installed or recommended read-only output checks for one finished run. "
            "Use candidate_id to focus recommendations on one candidate's files and bench record."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {**_VALIDATION_SCOPE, "candidate_offset": {"type": "integer", "minimum": 0}},
            "required": ["directory"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "program": {"type": "string"},
                "candidate_id": {"type": ["string", "null"]},
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}, "label": {"type": "string"}},
                        "required": ["id", "label"],
                    },
                },
                "next_candidate_offset": {"type": ["integer", "null"]},
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                            "installed": {"type": "boolean"},
                            "recommended": {"type": "boolean"},
                            "reason": {"type": "string"},
                            "workload": {"type": "string"},
                        },
                        "required": ["id", "label", "installed", "recommended", "reason", "workload"],
                    },
                },
            },
            "required": ["program", "candidate_id", "candidates", "next_candidate_offset", "checks"],
        },
    },
    {
        "name": "validate_output_files",
        "description": (
            "Run explicit checks from plan_validation, or the current recommended/all-installed selection. "
            "Returns aggregate results for one finished run or candidate."
        ),
        "inputSchema": _VALIDATION_CHECKS_INPUT,
        "outputSchema": {
            "type": "object",
            "properties": _VALIDATION_RESULT_PROPERTIES,
            "required": ["program", "candidate_id", "results"],
        },
    },
    {
        "name": "save_validation_report",
        "description": (
            "Run explicit, recommended, or all-installed output checks and create a new JSON report under "
            "validation_reports in the finished run."
        ),
        "inputSchema": _VALIDATION_CHECKS_INPUT,
        "outputSchema": {
            "type": "object",
            "properties": {
                **_VALIDATION_RESULT_PROPERTIES,
                "report_file": {"type": "string", "description": "The new report path relative to the run."},
            },
            "required": ["program", "candidate_id", "results", "report_file"],
        },
    },
    {
        "name": "list_validation_reports",
        "description": "List saved validation report filenames for one finished run.",
        "inputSchema": {
            "type": "object",
            "properties": {"directory": _VALIDATION_SCOPE["directory"]},
            "required": ["directory"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "program": {"type": "string"},
                "report_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 512,
                    "description": "Saved report filenames relative to the run.",
                },
            },
            "required": ["program", "report_files"],
        },
    },
    {
        "name": "plan_screening",
        "description": "List installed post-generation methods, input requirements, compute and recommendations. "
        "Pass a candidate ID from plan_validation to inspect one candidate.",
        "inputSchema": {"type": "object", "properties": _VALIDATION_SCOPE,
                        "required": ["directory"], "additionalProperties": False},
        "outputSchema": {
            "type": "object", "properties": {
                "program": {"type": "string"}, "directory": {"type": "string"},
                "candidate_id": {"type": ["string", "null"]},
                "methods": {"type": "array", "items": {"type": "object", "properties": {
                    **{k: {"type": "string"} for k in ("id", "label", "purpose", "reason", "workload")},
                    "operation": {"type": "string", "enum": [
                        "compute", "structure_measurement", "native_result_reader", "recorded_evidence",
                    ]},
                    **{k: {"type": "boolean"} for k in ("installed", "applicable", "recommended")}},
                    "required": ["id", "label", "purpose", "reason", "workload",
                                 "installed", "applicable", "recommended", "operation"]}},
            }, "required": ["program", "directory", "candidate_id", "methods"],
        },
    },
    {
        "name": "run_selected_screens",
        "description": "Run selected supplemental screening methods serially on existing candidate inputs. "
        "Returns individual results and reasons for skipped methods.",
        "inputSchema": {"type": "object", "properties": {
            **_VALIDATION_SCOPE, "selection": {"type": "string", "enum": ["recommended", "all_installed"]},
            "methods": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}},
            "oneOf": [{"required": ["selection"]}, {"required": ["methods"]}],
            "required": ["directory"], "additionalProperties": False},
        "outputSchema": {"type": "object", "properties": {
            "program": {"type": "string"}, "directory": {"type": "string"},
            "candidate_id": {"type": ["string", "null"]},
            "results": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "status": {"type": "string"}, "reason": {"type": "string"},
                "measurement": {"type": "object"}}, "required": ["id", "status"]}}},
            "required": ["program", "directory", "candidate_id", "results"]},
    },
)

#: What each tool does to the world, in the terms a client shows a person
#: before it lets a tool run. They are hints and a client must not rely on
#: them for safety, but each tool advertises its actual side effects.
#: ``screen_directory`` replaces a stored
#: verdict and can ask public databases; screening twice writes the same
#: bytes. ``run`` can start hours of generation here and on other machines.
_READS_ONLY = {"readOnlyHint": True, "openWorldHint": False}
_ANNOTATIONS: dict[str, dict[str, object]] = {
    "list_options": {"title": "List the program's options", **_READS_ONLY},
    "run": {
        "title": "Run Humanitys-Last-Disease",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
    "screen_directory": {
        "title": "Screen finished runs",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
    "read_verdict": {"title": "Read stored screen verdicts", **_READS_ONLY},
    "describe_architecture": {"title": "Describe the three layers", **_READS_ONLY},
    "retrieve_databases": {"title": "Read stored database lookups", **_READS_ONLY},
    "retrieve_structures": {"title": "Read stored structure rows", **_READS_ONLY},
    "read_sequence_measures": {"title": "Read stored sequence measurements", **_READS_ONLY},
    "read_results": {"title": "Read stored results", **_READS_ONLY},
    "list_output_files": {"title": "List finished-run output files", **_READS_ONLY},
    "read_output_file": {"title": "Read a finished-run output file", **_READS_ONLY},
    "backend_status": {"title": "Report the computational backend", **_READS_ONLY},
    "declare_corpora": {"title": "Declare what the models were trained on", **_READS_ONLY},
    "plan_validation": {"title": "Plan candidate output validation", **_READS_ONLY},
    "validate_output_files": {"title": "Check selected output files", **_READS_ONLY},
    "save_validation_report": {
        "title": "Save an output validation report",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
    "list_validation_reports": {"title": "List saved validation reports", **_READS_ONLY},
    "plan_screening": {"title": "Plan post-generation screens", **_READS_ONLY},
    "run_selected_screens": {"title": "Run selected post-generation screens", **_READS_ONLY},
}
for _tool in _TOOLS:
    _tool["annotations"] = dict(_ANNOTATIONS[str(_tool["name"])])
_TOOL_NAMES = frozenset(str(_tool["name"]) for _tool in _TOOLS)


def main() -> int:
    """Start the server in this checkout's Linux environment."""
    if sys.platform == "win32":
        try:
            command = wsl_command(_repo())
        except (OSError, ValueError, subprocess.SubprocessError):
            sys.stderr.write("Could not translate the configured MCP run roots into WSL paths. Check RFD4_MCP_ROOTS.\n")
            return 1
        return _relay(command)
    # An operator may widen this list explicitly. With no setting, the MCP
    # process is confined to this checkout rather than inheriting access to
    # every absolute path the OS account can read or write.
    if not os.environ.get(_ROOTS_ENV, "").strip():
        checkout = _repo().resolve()
        os.environ[_ROOTS_ENV] = os.pathsep.join(str(checkout / name) for name in ("runs", "disease_runs"))
    if os.environ.get(_SERVE_ENV) == "1":
        return serve()
    python = venv_python(_repo())
    if python is not None:
        os.environ[_SERVE_ENV] = "1"
        os.execv(str(python), [str(python), "-u", str(Path(__file__).resolve())])
    return serve()


def wsl_command(repo: Path) -> list[str]:
    """The Windows command that runs this server in the checkout's Linux environment.

    The program text is not an argument. stdio is the MCP stream.
    """
    root = os.environ.get("SYSTEMROOT", r"C:\Windows")
    wsl = str(Path(root) / "System32" / "wsl.exe")
    command = [wsl, "--cd", str(repo), "--exec", "env", f"{_SERVE_ENV}=1"]
    roots = os.environ.get(_ROOTS_ENV, "").strip()
    if roots:
        # Windows uses semicolons between paths. A POSIX root list can pass through
        # unchanged. wslpath resolves the actual mount instead of assuming /mnt/<drive>.
        translated = []
        for entry in roots.split(";"):
            entry = entry.strip()
            if PureWindowsPath(entry).is_absolute():
                entry = subprocess.check_output(
                    [wsl, "--exec", "/usr/bin/wslpath", "-u", entry],
                    stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, encoding="utf-8", timeout=10,
                ).rstrip("\r\n")
                if not entry.startswith("/") or any(c in entry for c in ":\r\n"):
                    raise ValueError("The translated run root cannot be represented as a WSL path.")
            translated.append(entry)
        command.append(f"{_ROOTS_ENV}={':'.join(translated)}")
    return [*command, ".venv/bin/python", "-u", "scripts/humanitys_last_disease_mcp.py"]


def venv_python(repo: Path) -> Path | None:
    """The checkout virtualenv, when this process is not already running it."""
    if sys.platform == "win32":
        return None
    environment = repo / ".venv"
    candidate = environment / "bin" / "python"
    if not candidate.is_file():
        return None
    try:
        # A virtualenv executable often links to the base interpreter. Only the prefix
        # tells us whether this process can actually import the checkout's dependencies.
        if environment.samefile(sys.prefix):
            return None
    except OSError:
        return None
    return candidate


def serve(
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
    handler: Handler | None = None,
) -> int:
    """Answer MCP on ``stdin`` until the client closes it. Only protocol bytes are written.

    The client's stdout is taken once, here, and nothing else is allowed to
    reach it: importing the program's package prints a GPU line, and a line
    that is not a valid MCP message is what the stdio binding forbids a server
    to write. ``sys.stdout`` is pointed at stderr for the lifetime of the loop,
    where the binding allows anything. That alone was not enough: it rebinds a
    name, while a child process inherits descriptors. See :func:`_take_stdout`
    and :func:`_take_stdin`, which move both ends of the protocol off
    descriptors 1 and 0 before any call can start one.
    """
    incoming = _take_stdin() if stdin is None else stdin
    writer = _Writer(_take_stdout() if stdout is None else stdout)
    calls = _Calls(writer, _dispatch if handler is None else handler)
    with contextlib.redirect_stdout(sys.stderr):
        if handler is None:
            _warm()
        try:
            for message in _messages(incoming):
                if isinstance(message, list):
                    _handle_batch(message, calls, writer)
                    continue
                for reply in _handle(message, calls):
                    writer.send(reply)
        except OSError:
            # The client closed its end while a reply was being written, or the
            # stream itself failed. Nobody is left to answer, and a traceback on
            # the way out would read as a crash of a server that did its job.
            pass
        finally:
            calls.shutdown()
    return 0


def _take_stdout() -> BinaryIO:
    """Move the client's stream off file descriptor 1 and return it.

    ``redirect_stdout`` rebinds a Python name. A child process inherits file
    descriptors, so it is not covered: for a real generation the program starts
    ``rfd4-generate`` without redirecting it, and that child would write its
    Hydra log and its resolved config -- absolute checkpoint, data and output
    paths, none of them scrubbed -- onto descriptor 1, which is the client's
    pipe. Not between frames either: a write that does not end in a newline
    lands inside one, so the frame cannot even be skipped.

    Duplicating the descriptor and pointing 1 at stderr makes that output a
    diagnostic, which the binding allows, and leaves the returned stream as the
    only way to reach the client. It also means the ``redirect_stdout`` above is
    no longer load-bearing: should the loop ever unwind while a call is still
    running, what it restores is a descriptor that now goes to stderr.
    """
    saved = os.dup(1)
    os.dup2(2, 1)
    # Buffered, as ``sys.stdout.buffer`` was. A raw stream writes in one syscall
    # and reports a short write rather than finishing it; :class:`_Writer`
    # flushes every message, so buffering costs nothing here.
    return os.fdopen(saved, "wb")


def _take_stdin() -> BinaryIO:
    """Move the client's requests off file descriptor 0 and return them.

    The same inheritance, the other way round. ``ssh <host> <command>``, which
    the program runs for a remote generation, forwards its own standard input to
    the remote process and reads it as it arrives -- and descriptor 0 here is the
    stream the read loop is parsing. Two readers on one pipe means frames the
    client sends during a run go to whichever gets them first. Pointing
    descriptor 0 at an empty file leaves a child reading end-of-file.
    """
    saved = os.dup(0)
    with open(os.devnull, "rb") as empty:
        os.dup2(empty.fileno(), 0)
    # Buffered, as ``sys.stdin.buffer`` was. A raw stream's ``read`` returns
    # what has arrived rather than what was asked for, which truncated any frame
    # a pipe delivered in more than one piece, and read a line a byte at a time.
    return os.fdopen(saved, "rb")


class _Writer:
    """The one thing that writes to the client.

    A tool call answers from another thread while the read loop keeps going, so
    two messages can be ready at once. The lock is what stops them interleaving
    mid-line. ``ensure_ascii`` also escapes any newline inside a string, which
    the stdio binding requires of a message.
    """

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._pending_ids: set[str] = set()

    def reserve_id(self, request_id: object) -> bool:
        key = _key(request_id)
        with self._lock:
            if key in self._pending_ids:
                return False
            self._pending_ids.add(key)
            return True

    def forget_id(self, request_id: object) -> None:
        with self._lock:
            self._pending_ids.discard(_key(request_id))

    def send(self, message: object) -> None:
        text = json.dumps(message, separators=(",", ":"), ensure_ascii=True)
        with self._lock:
            self._stream.write(text.encode("ascii"))
            self._stream.write(b"\n")
            self._stream.flush()
            # A batch holds all its IDs until the whole response reaches the client.
            replies = message if isinstance(message, list) else (message,)
            for reply in replies:
                if isinstance(reply, dict) and _valid_request_id(reply.get("id")):
                    self._pending_ids.discard(_key(reply["id"]))


class _Cancelled(BaseException):
    """Raised inside the progress callback to abandon a call the client cancelled.

    It derives from ``BaseException`` deliberately. ``cli.main`` ends with
    ``except Exception``, which logs a traceback and returns exit code 2, so an
    ordinary exception would be swallowed there and a cancelled call would look
    like a failed run instead of no run. This passes through that handler while
    still letting its ``finally`` close the panel and the run log.
    """


class _Batch:
    """Collect one batch's immediate and worker replies into one response frame."""

    def __init__(self, writer: _Writer) -> None:
        self._writer = writer
        self._replies: list[dict[str, object]] = []
        self._pending = 0
        self._closed = False
        self._sent = False
        self._lock = threading.Lock()

    def expect(self) -> None:
        with self._lock:
            self._pending += 1

    def immediate(self, reply: dict[str, object]) -> None:
        with self._lock:
            self._replies.append(reply)

    def complete(self, reply: dict[str, object] | None) -> None:
        send = None
        with self._lock:
            if reply is not None:
                self._replies.append(reply)
            self._pending -= 1
            send = self._ready_locked()
        if send is not None:
            self._writer.send(send)

    def finish(self) -> None:
        send = None
        with self._lock:
            self._closed = True
            send = self._ready_locked()
        if send is not None:
            self._writer.send(send)

    def _ready_locked(self) -> list[dict[str, object]] | None:
        if self._sent or not self._closed or self._pending:
            return None
        self._sent = True
        return list(self._replies) if self._replies else None


class _Calls:
    """Runs tool calls off the read loop, one at a time.

    The read loop has to keep reading. On stdio a cancellation is a message
    rather than a closed stream, so a server that blocks inside a tool cannot be
    told to stop, and cannot answer a ping either -- and generation is measured
    in hours. Calls are taken one at a time rather than together because the
    program expects one run per process: it opens a single run log, and the
    capture of its printed output replaces ``sys.stdout`` for the whole process.
    """

    def __init__(self, writer: _Writer, handler: Handler) -> None:
        self._writer = writer
        self._handler = handler
        self._queue: queue.Queue[
            tuple[Mapping[str, object], bool, threading.Event, str, _Batch | None] | None
        ] = queue.Queue()
        self._events: dict[str, threading.Event] = {}
        #: The call the worker is inside, as opposed to the ones still queued.
        self._running: threading.Event | None = None
        self._lock = threading.Lock()
        self._drained = threading.Event()
        self._drained.set()
        self._worker = threading.Thread(target=self._work, name="Humanitys-Last-Disease-tool", daemon=True)
        self._worker.start()

    def reserve_id(self, request_id: object) -> bool:
        return self._writer.reserve_id(request_id)

    def submit(self, message: Mapping[str, object], *, modern: bool, batch: _Batch | None = None) -> bool:
        """Queue a call whose response ID has already been reserved."""
        key = _key(message.get("id"))
        with self._lock:
            if len(self._events) >= _MAX_OUTSTANDING_CALLS and key not in self._events:
                return False
            event = threading.Event()
            self._events[key] = event
            self._drained.clear()
            if batch is not None:
                batch.expect()
        self._queue.put((message, modern, event, key, batch))
        return True

    def cancel(self, request_id: object) -> None:
        """Mark a call cancelled. An unknown or finished id is ignored, as the spec allows."""
        if not _valid_request_id(request_id):
            return
        key = _key(request_id)
        with self._lock:
            event = self._events.get(key)
            if event is None:
                return
            event.set()
            # Hold the worker lock through the signal so the current call
            # owns any running children until the stop request is issued.
            if self._running is event:
                _stop_work()

    def shutdown(self) -> None:
        """Give a call in flight a moment to finish, then tell it to stop."""
        if self._drained.wait(_SHUTDOWN_GRACE_S):
            self._queue.put(None)
            return
        with self._lock:
            events = list(self._events.values())
            running = self._running is not None
        for event in events:
            event.set()
        # The flag alone is seen only when the program next prints, and
        # generation prints nothing until its children exit. ``cancel`` ends
        # the child for that reason, and a closed stream has to as well: the
        # process is about to exit, and a generation left behind would hold a
        # GPU for hours with nothing left to collect what it wrote.
        if running:
            _stop_work()
        self._drained.wait(_SHUTDOWN_CANCEL_S)
        self._queue.put(None)

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            message, modern, event, key, batch = job
            with self._lock:
                self._running = event
            try:
                # Withdrawn while it waited behind another call. Starting it
                # anyway would open the run log and begin the work for a request
                # the client has already let go of.
                if event.is_set() or not self._answer(message, modern, event, batch):
                    self._writer.forget_id(message["id"])
                    if batch is not None:
                        batch.complete(None)
            except Exception as exc:
                # A tool that fails is answered inside ``_call``. What reaches
                # here failed in this server while answering, and the worker
                # has to outlive it: there is one worker, so a dead one leaves
                # every later call waiting for an answer that never comes.
                sys.stderr.write(f"Humanitys-Last-Disease: a call could not be answered: {type(exc).__name__}\n")
                with contextlib.suppress(Exception):
                    reply = _error(message.get("id"), _INTERNAL_ERROR, "Internal error.")
                    if batch is None:
                        self._writer.send(reply)
                    else:
                        batch.complete(reply)
            finally:
                with self._lock:
                    self._running = None
                    if self._events.get(key) is event:
                        self._events.pop(key)
                    if not self._events:
                        self._drained.set()

    def _answer(
        self, message: Mapping[str, object], modern: bool, event: threading.Event, batch: _Batch | None
    ) -> bool:
        params = message.get("params")
        progress = _Progress(self._writer, _progress_token(params), event)
        _CURRENT.report = progress.line
        finished = threading.Event()
        beat = threading.Thread(
            target=_heartbeat,
            args=(progress, finished),
            name="Humanitys-Last-Disease-beat",
            daemon=True,
        )
        beat.start()
        try:
            result = _call(params, self._handler)
        except _Cancelled:
            # The spec asks a server to stop, free what it held, and send no
            # response at all for a cancelled request.
            return False
        finally:
            _CURRENT.report = None
            finished.set()
            progress.close()
        # A tool with nothing to print never reaches the check inside the
        # callback, so a cancellation for one is only visible here. Answering it
        # would be a response for a request the client has released.
        if event.is_set():
            return False
        reply = _result(message, _decorate(result, modern=modern))
        if batch is None:
            self._writer.send(reply)
        else:
            batch.complete(reply)
        return True


def _heartbeat(progress: _Progress, finished: threading.Event) -> None:
    """Repeat the last thing the program said while it is still working.

    Generation is the long operation and it is silent for all of it: the program
    prints that it is starting the requests, then prints nothing until every
    child has exited. Without this a client sees one notification and then hours
    of nothing, and a client is allowed to reset its timeout when progress
    arrives -- which is the difference between a long call and a dead one.
    """
    while not finished.wait(_HEARTBEAT_S):
        progress.heartbeat()


class _Progress:
    """Progress for one call: one count that only rises, and one token.

    The program's lines and the heartbeat both report through here so they
    cannot hand out the same number or keep reporting once the call is answered.
    """

    def __init__(self, writer: _Writer, token: object, event: threading.Event) -> None:
        self._writer = writer
        self._token = token
        self._event = event
        self._lock = threading.Lock()
        self._sent = 0
        self._at = 0.0
        self._last = ""
        self._closed = False

    def line(self, text: str) -> None:
        """Called with each line the program finishes printing."""
        # Checked on every line, not only the ones that get reported, so a
        # cancellation takes effect at the next thing the program says.
        if self._event.is_set():
            raise _Cancelled
        with self._lock:
            self._last = text
        self._emit(text, gap=_PROGRESS_EVERY_S)

    def heartbeat(self) -> None:
        """Say again what the program last said, if it has said anything."""
        with self._lock:
            text = self._last
        if text:
            self._emit(text, gap=_HEARTBEAT_S)

    def close(self) -> None:
        """No notification after this. They have to stop once the call is answered."""
        with self._lock:
            self._closed = True

    def _emit(self, text: str, *, gap: float) -> None:
        if self._token is None:
            return
        with self._lock:
            if self._closed:
                return
            now = time.monotonic()
            if self._sent and now - self._at < gap:
                return
            self._at = now
            self._sent += 1
            count = self._sent
        # A courtesy, sent from inside the program's own print. A client that has
        # stopped reading must not surface there as the program failing.
        with contextlib.suppress(OSError):
            self._writer.send(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": self._token,
                        # A count of what has been reported, because the number of
                        # steps is not known in advance and the value has to rise
                        # every time. The line itself carries the meaning.
                        "progress": count,
                        "message": text[:_PROGRESS_LINE_MAX],
                    },
                }
            )


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _warm() -> None:
    """Import the program in the background, so the first tool call is not what waits for it.

    Importing ``rfproteina`` imports torch, about fifty seconds on this
    checkout, and the handshake has to be answered long before that. Python's
    per-module import lock makes a later import from the request path wait for
    this one rather than race it, and the caller has already pointed
    ``sys.stdout`` away from the protocol stream.
    """

    def load() -> None:
        with contextlib.suppress(Exception):
            _api()

    threading.Thread(target=load, name="Humanitys-Last-Disease-warm", daemon=True).start()


def _api() -> object:
    """The in-process tools. Imported late: this is what costs the fifty seconds."""
    entry = str(_repo() / "src")
    if entry not in sys.path:
        sys.path.insert(0, entry)
    from rfproteina.disease import mcp_api

    return mcp_api


def _dispatch(name: str, arguments: Mapping[str, object] | None) -> Mapping[str, object]:
    try:
        api = _api()
    except ImportError:
        raise ValueError("The Humanitys-Last-Disease environment is not installed in this interpreter.") from None
    return api.call_tool(name, arguments, on_line=getattr(_CURRENT, "report", None))


def _stop_work() -> None:
    """Ask the program to end a child the running call started.

    Called from the read loop, so it must not raise and must not wait: a server
    that died trying to cancel is worse than one that could not. A child can
    only exist once the program's dispatch module is loaded, so when it is not
    there is nothing to stop, and loading the package to find that out would
    hold the read loop for most of a minute.
    """
    if "rfproteina.disease.dispatch" not in sys.modules:
        return
    with contextlib.suppress(Exception):
        _api().stop_work()


def _key(request_id: object) -> str:
    """A dictionary key for a request id.

    JSON-RPC allows a string or a number, and a client may send the same value
    as either, so the type is part of the key. An id that is neither still gets
    a key rather than raising, because it has already been answered by then.
    """
    if isinstance(request_id, float) and request_id.is_integer():
        request_id = int(request_id)
    return f"{type(request_id).__name__}:{request_id!r}"


def _progress_token(params: object) -> object:
    """The token a client includes when it wants progress for this call.

    It is a bare ``progressToken`` in ``_meta``, not one of the prefixed keys,
    and the same spelling in every revision that has progress at all.
    """
    if not isinstance(params, dict):
        return None
    meta = params.get(_META)
    if not isinstance(meta, dict):
        return None
    token = meta.get("progressToken")
    # A string or an integer, and nothing else. A float was being echoed back
    # verbatim, which made this server the one sending a token the schema does
    # not allow. A bool is an int in Python and is not one here.
    return token if isinstance(token, str | int) and not isinstance(token, bool) else None


def _handle(message: object, calls: _Calls) -> list[dict[str, object]]:
    """The immediate replies one non-batch frame earns."""
    reply = _reply_to(message, calls)
    return [] if reply is None else [reply]


def _handle_batch(message: list[object], calls: _Calls, writer: _Writer) -> None:
    """Answer a JSON-RPC batch once, after every requested tool call finishes."""
    if not message:
        writer.send(_error(None, _INVALID_REQUEST, "Invalid request."))
        return
    if len(message) > _MAX_BATCH_ITEMS:
        writer.send(_error(None, _INVALID_REQUEST, "Batch request has too many items."))
        return
    batch = _Batch(writer)
    for item in message:
        reply = _reply_to(item, calls, batch=batch)
        if reply is not None:
            batch.immediate(reply)
    batch.finish()


def _reply_to(message: object, calls: _Calls | None, *, batch: _Batch | None = None) -> dict[str, object] | None:
    if message is _UNREADABLE:
        return _error(None, _PARSE_ERROR, "Parse error.")
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, _INVALID_REQUEST, "Invalid request.")
    # A message with no id is a notification, whatever it asks for, and a
    # notification is never answered. Replying to one is what made this server
    # send a result for notifications/initialized's siblings.
    if "id" not in message:
        if message.get("method") == "notifications/cancelled":
            params = message.get("params")
            if isinstance(params, dict) and calls is not None:
                calls.cancel(params.get("requestId"))
        return None
    request_id = message.get("id")
    if not _valid_request_id(request_id):
        return _error(None, _INVALID_REQUEST, "Invalid request.")
    if calls is not None and not calls.reserve_id(request_id):
        return _error(None, _INVALID_REQUEST, "Request ID is already in use.")
    method = message.get("method")
    if not isinstance(method, str):
        return _error(request_id, _INVALID_REQUEST, "Invalid request.")
    version = _declared_version(message)
    if version is not None and version not in SUPPORTED_VERSIONS:
        return _error(
            request_id,
            _UNSUPPORTED_VERSION,
            "Unsupported protocol version",
            data={"supported": list(SUPPORTED_VERSIONS), "requested": version},
        )
    modern = version in MODERN_VERSIONS
    if method == "server/discover":
        return _result(message, _discover())
    if method == "initialize":
        # An initialize request selects the handshake era even when it also
        # carries modern metadata, because that is what the client can read.
        return _result(message, _initialize(message.get("params")))
    if method == "ping":
        return _result(message, _decorate({}, modern=modern))
    if method == "tools/list":
        return _result(message, _decorate({"tools": list(_TOOLS)}, modern=modern, cacheable=True))
    if method == "tools/call":
        # A call no tool can be given is refused here, at once, as a protocol
        # error. It is the request that is wrong, not a tool that failed, and
        # it should not wait its turn behind a call that is still running.
        params = message.get("params")
        if isinstance(params, dict) and "task" in params:
            return _error(request_id, _METHOD_NOT_FOUND, "Task-augmented tool calls are unavailable.")
        problem = _call_problem(params)
        if problem is not None:
            return _error(request_id, _INVALID_PARAMS, problem)
        # Answered by the worker, not here: this loop has to stay free to read
        # the cancellation that stops it.
        if calls is None:
            return _error(request_id, _INTERNAL_ERROR, "Internal error.")
        if not calls.submit(message, modern=modern, batch=batch):
            return _error(request_id, _SERVER_BUSY, "Server is busy.")
        return None
    return _error(request_id, _METHOD_NOT_FOUND, "Method not found.")


def _declared_version(message: Mapping[str, object]) -> str | None:
    """The protocol version a modern request carries in ``params._meta``."""
    params = message.get("params")
    if not isinstance(params, dict):
        return None
    meta = params.get(_META)
    if not isinstance(meta, dict):
        return None
    version = meta.get(_VERSION_KEY)
    return version if isinstance(version, str) else None


def _valid_request_id(request_id: object) -> bool:
    """MCP request IDs must be strings or integers, including integer-valued JSON numbers."""
    return type(request_id) in (str, int) or (isinstance(request_id, float) and request_id.is_integer())


def _decorate(result: Mapping[str, object], *, modern: bool, cacheable: bool = False) -> dict[str, object]:
    """A modern result carries its type, this server's identity, and a cache hint.

    A handshake-era result carries none of them: the field names postdate those
    revisions, and a client reading one has no use for them.
    """
    if not modern:
        return dict(result)
    decorated = dict(result)
    decorated["resultType"] = "complete"
    meta = dict(decorated.get(_META) or {})
    meta[_SERVER_INFO_KEY] = {"name": SERVER_NAME, "version": _version()}
    decorated[_META] = meta
    if cacheable:
        decorated["ttlMs"] = _LIST_TTL_MS
        decorated["cacheScope"] = _CACHE_SCOPE
    return decorated


def _discover() -> dict[str, object]:
    """Answer the mandatory modern discovery call.

    It is answered whatever era the request declares: a client probing with it
    is deciding which era to use, so refusing the probe is how a dual-era
    server gets mistaken for a handshake-only one.
    """
    return _decorate(
        {
            "supportedVersions": list(SUPPORTED_VERSIONS),
            "capabilities": {"tools": {"listChanged": False}},
            "instructions": _INSTRUCTIONS,
        },
        modern=True,
        cacheable=True,
    )


def _initialize(params: object) -> dict[str, object]:
    """Answer the handshake. An unsupported request gets the newest era this server speaks.

    The rule is to answer with the same version when it is supported and
    otherwise with the latest the server supports. The latest overall is a
    modern revision with no handshake at all, which a client that sent
    ``initialize`` cannot read, so the answer is the newest handshake revision.
    Offering the oldest instead, as this did, negotiated every current client
    down to 2024-11-05, where the structured tool results this server returns
    did not yet exist.
    """
    version = LEGACY_VERSIONS[0]
    if isinstance(params, dict):
        requested = params.get("protocolVersion")
        if isinstance(requested, str) and requested in LEGACY_VERSIONS:
            version = requested
    return {
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "version": _version()},
        "instructions": _INSTRUCTIONS,
    }


def _call_problem(params: object) -> str | None:
    """Why a ``tools/call`` request cannot be given to a tool at all, or None when it can.

    These are faults of the request itself: a name no tool here has, or
    parameters that are not the shape a call takes. The specification answers
    them as a protocol error, and has from its first revision; an unknown tool
    was being returned as a tool result with ``isError`` instead, which is how
    a tool that ran and failed is reported, and which the conformance checks
    count as a failure.

    What a tool makes of its arguments is a different matter and stays a
    result with ``isError``: a model can read that and correct the call, which
    it cannot do with a name that does not exist. The name is echoed as one
    short run of text, because it is the client's own and may be anything.
    """
    if not isinstance(params, dict):
        return "Invalid params: tools/call takes an object."
    name = params.get("name")
    if not isinstance(name, str):
        return "Invalid params: name must be a string."
    if name not in _TOOL_NAMES:
        return "Unknown tool: " + _scrub_error_text(name)[:80]
    arguments = params.get("arguments", {})
    if arguments is not None and not isinstance(arguments, dict):
        return "Invalid params: arguments must be an object."
    return None


def _call(params: object, handler: Handler) -> dict[str, object]:
    # The read loop has already refused a request that names no tool or is not
    # shaped as a call. These guards stay for a caller that comes here directly.
    if not isinstance(params, dict):
        return _tool_error("Tool arguments must be an object.")
    name = params.get("name")
    if not isinstance(name, str) or name not in _TOOL_NAMES:
        return _tool_error("Unknown tool.")
    arguments = params.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _tool_error("Tool arguments must be an object.")
    try:
        payload = handler(name, arguments)
    except ValueError as exc:
        return _tool_error(str(exc).strip() or "The tool could not be run.")
    except Exception as exc:
        # A ValueError carries a message written to be read. Anything else
        # carries whatever the failure happened to say, which can be a path on
        # this computer, so only the class name is passed on. Saying nothing at
        # all leaves an agent with no way to tell a missing file from a denied
        # one, and nothing to report to the person who could fix it.
        return _tool_error(f"The tool could not be run: {type(exc).__name__}.")
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError):
        return _tool_error("The tool result could not be encoded.")
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload,
        "isError": False,
    }


def _tool_error(message: str) -> dict[str, object]:
    return {"content": [{"type": "text", "text": _scrub_error_text(message)}], "isError": True}


def _scrub_error_text(message: str) -> str:
    """One readable refusal without a credential, machine path, or control line."""
    cleaned = _ERROR_SECRET.sub(_ERROR_REDACTED, message)
    cleaned = _ERROR_BEARER.sub(r"\1" + _ERROR_REDACTED, cleaned)
    cleaned = _ERROR_NAMED_SECRET.sub(r"\1\2" + _ERROR_REDACTED, cleaned)
    cleaned = _ERROR_WIN_PATH.sub(_ERROR_PATH, cleaned)
    cleaned = _ERROR_UNC_PATH.sub(_ERROR_PATH, cleaned)
    cleaned = _ERROR_POSIX_PATH.sub(_ERROR_PATH, cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned or "The tool could not be run."


def _result(message: Mapping[str, object], result: object) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": _request_id(message), "result": result}


def _error(request_id: object, code: int, message: str, data: object = None) -> dict[str, object]:
    error: dict[str, object] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _request_id(message: Mapping[str, object]) -> object:
    return message.get("id")


def _version() -> str:
    try:
        text = (_repo() / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "0.0.3"
    for line in text.splitlines():
        if line.startswith('version = "'):
            return line.split('"', 2)[1]
    return "0.0.3"


def _messages(stream: BinaryIO) -> Iterator[object]:
    """Yield one JSON value per MCP frame. Newline frames and Content-Length frames both count."""
    read_line = _line_reader(stream)
    while True:
        line = read_line(_MAX_FRAME + 1)
        if not line:
            return
        if len(line) > _MAX_FRAME and not line.endswith(b"\n"):
            # A line with no end in sight. It is one unreadable frame, and the
            # rest of it is dropped up to the next newline, where frames line
            # up again. Reading it whole first is how a client that sends no
            # newline takes all the memory this process can get.
            while True:
                rest = read_line(_MAX_FRAME + 1)
                if not rest or rest.endswith(b"\n"):
                    break
            yield _UNREADABLE
            continue
        if line.lower().startswith(b"content-length:"):
            try:
                length = int(line.split(b":", 1)[1].strip())
            except ValueError:
                yield _UNREADABLE
                continue
            if length < 0 or length > _MAX_FRAME:
                yield _UNREADABLE
                return
            if not _read_frame_headers(read_line, len(line)):
                # Without the separator the start of the body is uncertain.
                yield _UNREADABLE
                return
            payload = _read_exactly(stream, length)
            if payload is None:
                yield _UNREADABLE
                return
            yield _decode(payload)
            continue
        if not line.strip():
            continue
        yield _decode(line)


def _line_reader(stream: BinaryIO) -> Callable[[int], bytes]:
    """Read one line subject to the parser's current byte limit.

    The bound is passed only to a real stream. A stand-in that takes no size,
    as a test's does, is read as it always was.
    """
    if isinstance(stream, io.IOBase):
        return stream.readline
    return lambda _limit: stream.readline()


def _read_frame_headers(read_line: Callable[[int], bytes], first_size: int) -> bool:
    """Read through the required blank line within the remaining header budget."""
    remaining = _MAX_FRAME_HEADERS - first_size
    while remaining > 0:
        header = read_line(remaining + 1)
        if len(header) > remaining or not header.endswith(b"\n"):
            return False
        if header in {b"\n", b"\r\n"}:
            return True
        remaining -= len(header)
    return False


def _read_exactly(stream: BinaryIO, length: int) -> bytes | None:
    """Exactly ``length`` bytes, or ``None`` if the stream ended first.

    One ``read`` is not enough on its own. A buffered stream loops internally,
    but this reader does not get to choose what it is handed, and on a raw one a
    single ``read`` returns only what has arrived -- so a frame a pipe delivered
    in pieces came back truncated and the message was lost rather than reported.
    """
    pieces: list[bytes] = []
    remaining = length
    while remaining > 0:
        piece = stream.read(remaining)
        if not piece:
            return None
        pieces.append(piece)
        remaining -= len(piece)
    return b"".join(pieces)


def _decode(payload: bytes) -> object:
    try:
        return json.loads(payload.decode("utf-8"), parse_float=_finite_json_float, parse_constant=_reject_json_constant)
    except (UnicodeError, ValueError, RecursionError):
        return _UNREADABLE


def _reject_json_constant(_value: str) -> None:
    """JSON has no NaN or infinity values."""
    raise ValueError("Non-finite JSON number.")


def _finite_json_float(token: str) -> float:
    number = float(token)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number.")
    return number


def _relay(command: list[str]) -> int:
    """Copy this process's stdio to the Linux server. The command is not a shell line."""
    if not Path(command[0]).is_file():
        sys.stderr.write("Humanitys-Last-Disease runs in its Linux environment, and WSL is not on this computer.\n")
        return 1
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return 1
        return 1
    copying = (
        threading.Thread(
            target=_copy,
            args=(sys.stdin.buffer, process.stdin),
            kwargs={"close_destination": True},
            daemon=True,
        ),
        threading.Thread(
            target=_copy,
            args=(process.stdout, sys.stdout.buffer),
            kwargs={"close_destination": False},
            daemon=True,
        ),
    )
    for thread in copying:
        thread.start()
    code = process.wait()
    copying[1].join(timeout=5)
    return code if isinstance(code, int) else 1


def _copy(source: BinaryIO, destination: BinaryIO, *, close_destination: bool) -> None:
    # ``read`` on a buffered stream waits until it has the whole block or the
    # stream ends. A client sends one short message and then waits for the
    # answer, so nothing was passed on at all: started on Windows, the server
    # never saw ``initialize`` and every session hung at its first request. It
    # only looked right when the input was a file, which ends. ``read1`` hands
    # over what has arrived. A raw stream has no ``read1`` and needs none.
    read = getattr(source, "read1", None) or source.read
    try:
        while True:
            block = read(65536)
            if not block:
                break
            destination.write(block)
            destination.flush()
    except OSError:
        return
    finally:
        if close_destination:
            try:
                destination.close()
            except OSError:
                return


if __name__ == "__main__":
    raise SystemExit(main())
