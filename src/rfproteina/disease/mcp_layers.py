"""Three layers of the Humanitys-Last-Disease MCP server.

The AI interface lists the models the disease panel can ask. The MCP layer
reads stored lookups, structures, sequence measurements, and results, and it
points generative design and screening at the tools that already run them.
The computational backend reports what a directory already recorded. None of
these calls starts a model, a GPU job, docking, or a prediction.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path

PROGRAM = "Humanitys-Last-Disease"
NOT_RECORDED = "not_recorded"
NOT_PERFORMED = "not_performed"
#: The modified-weights model, under the name the project carries. Named once
#: because two literals for one model drift: ``describe_architecture`` lists it
#: and ``backend_status`` reports it, and a reader matching the two by eye
#: cannot tell a rename from a second component.
MODIFIED_MODEL = "ModRFD4"
#: A component that exists as a plan and not yet as a file on this machine.
#: Distinct from :data:`NOT_RECORDED`, which means this call was not told, and
#: from :data:`NOT_PERFORMED`, which means it was told and the answer was no.
NOT_INSTALLED = "not_installed"
_MAX_JSON_BYTES = 2_000_000
_DIRECTORY_ENTRY_LIMIT = 2_048
#: How far up a tool looks for an artifact the program writes beside the output
#: directory rather than inside a run. Four is the archive's own depth --
#: ``generations/<disease>/<target>/<run>`` -- which is the longest climb there
#: is from a directory holding a screen to the directory holding the lookup.
_ARTIFACT_CLIMB = 4
#: The argument a caller passes to be answered from the output directory above
#: a run. Off by default, because a lookup covering an output directory is not a
#: measurement of one run in it.
CLIMB_OPTION = "from_output_dir"
_TEXT_CAP = 300
#: What a capped string ends in, so a caller is never handed the first 300
#: characters of a title as if they were the whole of it.
_CUT_MARK = " [cut]"
#: Confidence metrics where a higher number is the more confident one: pLDDT,
#: pTM and ipTM are all scores, and so is anything whose name says confidence.
#:
#: That last alternative was missing. The trace writer stores a metric whose
#: name matches its own rule (``refold.trace_metric_kind``) and this one read the same
#: rule back minus ``confidence``, so a column the writer kept on purpose was
#: filtered out again here -- and a trace carrying only such columns reached
#: ``_ranking`` with no shared metric at all, which falls back to ordering by
#: filename under a component called Candidate ranking. Two spellings of one
#: rule; ``tests/test_disease_mcp_record_keys.py`` now pins them together.
_HIGHER_IS_BETTER = r"plddt|iptm|(?<![A-Za-z])ptm(?![A-Za-z])|confidence"
#: Confidence metrics that are a predicted aligned error, where a lower number
#: is the more confident one. This repository's own
#: ``configs/metrics/pass_criteria/all_pass_criteria.yaml`` is the authority
#: either way: it bounds ``oracle_ppi.min_ipae_coi`` with ``max`` and
#: ``oracle_ppi.ptm_coi`` with ``min``.
_LOWER_IS_BETTER = r"ipae|(?<![A-Za-z])pae(?![A-Za-z])"
#: Both families. Written as one alternation of the two above so the ranking
#: cannot come to know a metric this filter does not, or the reverse.
_CONFIDENCE = re.compile(f"{_HIGHER_IS_BETTER}|{_LOWER_IS_BETTER}", re.IGNORECASE)
_ERROR_METRIC = re.compile(_LOWER_IS_BETTER, re.IGNORECASE)
_DOCKING_NOTE = "Docking pose search is unavailable. AutoDock Vina can score supplied poses during screening."
#: ``record_version`` and ``run_id`` are this record's shape and its identity,
#: and ``scrubbed_files`` how many of the copied files were rewritten before
#: the copy could be read. All three were being dropped, so two archived runs
#: of one target saved the same day came back indistinguishable, from a record
#: whose version nothing stated -- the two things the verdict view keeps
#: ``screen_version`` and ``code`` for. Pinned against the writer in
#: ``tests/test_disease_mcp_record_keys.py``.
_GENERATION_FIELDS = (
    "disease_name",
    "disease_id",
    "target_id",
    "group",
    "entry_id",
    "chain",
    "binder_kind",
    "source",
    "saved_at",
    "record_version",
    "run_id",
    "scrubbed_files",
)
_JOB_FIELDS = ("class_name", "binder_kind", "target_id", "generation_started")
#: ``read_results`` returned five fields out of a twenty-one-field verdict:
#: a call with no axes behind it, no bar axes, no cuts, no code and no record
#: version -- a verdict a reader could neither check nor date. It returns the
#: verdict view now, derived from ``mcp_api._VERDICT_FIELDS`` rather than listed
#: again here, because a third hand-written list over one record is how the
#: first two came to disagree.
#:
#: ``refold`` is held back, and only from this tool: it is 4 KB of the 5.4 KB a
#: version-3 verdict costs on a real screen, and this tool walks a whole tree. A
#: summary of it travels instead -- how many criteria were declared, how many
#: were evaluated, and the blocker when none were -- and ``read_verdict``
#: returns the criteria table in full for one run.
#:
#: Three of version 4's reports are held back for the same reason, with no
#: summary: ``designed_interface_quality`` is about 3 KB on every screen, three
#: quarters of it the same reference table, and ``synthesis`` and
#: ``ligand_alerts`` are about 2 KB each. ``read_verdict`` returns all three.
#: ``mhc_class_i`` travels here too, and stays out of the set. Measured, it is
#: about 350 bytes on a run that scanned cleanly -- a third of a kilobyte, not
#: the quarter this comment claimed before anyone counted, because the ``note``
#: alone is 196 characters -- and about 600 with a failed scan, where
#: ``scan_blocker`` carries up to 200 characters of the predictor's own stderr.
#: Against 2 KB each for ``synthesis`` and ``ligand_alerts``, 3 KB for
#: ``designed_interface_quality`` and 4 KB for ``refold``, that is well inside
#: the budget, and the state of the scan is what a reader of a tree most needs.
#:
#: ``scan_blocker`` is the one part of it that is not the program's own words:
#: ``mhc_tables.py`` puts a subprocess's stderr there, and a stderr can name a
#: path that ``scrub_for_upload`` does not cover, which only strips the home
#: directory, ``USERPROFILE`` and the checkout root. It is bounded and it is
#: scrubbed, so it travels, but it is a publication decision rather than a size
#: one and it is recorded here as such.
#: ``modality_screen`` and ``chemical_liability`` join them on the same ground.
#: Measured: the drug-type report is 1,403 bytes on a protein design and 1,316
#: on a peptide, of which the residue composition, the atom counts and the
#: AGGRESCAN sub-record are two thirds; the motif block is 376 bytes, or 433
#: when the CIF named bonds and the free-cysteine listing appears. Together
#: that is about the size of ``synthesis`` and ``ligand_alerts``, which are
#: already out. Neither moves a call, and a reader walking a tree is looking
#: for which runs to open rather than for one design's atom counts.
#: ``read_verdict`` returns both whole, with nothing held back: there is no
#: sequence, no molecule and no path in either -- a residue composition is a
#: count per letter and a site list is residue numbers.
_SCREEN_OMITTED = frozenset(
    {
        "refold",
        "designed_interface_quality",
        "prodigy_affinity",
        "protein_sol",
        "vina_pose",
        "tnp_profile",
        "synthesis",
        "ligand_alerts",
        "ligand_ames",
        "ligand_dili",
        "delphi_psr",
        "peptide_rbc_evidence",
        "molprobity_geometry",
        "ligand_microsomal_clearance",
        "ligand_caco2",
        "ligand_solubility",
        "qupkake",
        "sygma",
        "modality_screen",
        "chemical_liability",
    }
)
#: The same judgement at key granularity, for a block that has to travel whole
#: except for one part of it. ``cuts`` cannot be omitted: the numbers a verdict
#: was taken against are what the verdict means, and they differ between runs.
#: ``cuts["provenance"]`` does not differ between runs -- it is the same twelve
#: accounts of where this program's thresholds come from, measured at 4,602
#: bytes on a protein screen. That is larger than every block already dropped
#: here: ``synthesis`` and ``ligand_alerts`` at about 2 KB each,
#: ``designed_interface_quality`` at 3 KB, ``refold`` at 4 KB. It was 2,908
#: bytes when this rule was written and grew when four of the caveats were
#: filled in with what their numbers actually rest on, which is the direction
#: this block will keep moving: a caveat earns its place by being long enough
#: to be useful. Repeating it once per run would make the prose the larger half
#: of a tree-wide answer while saying nothing about any particular run.
#: ``read_verdict`` returns it in full for one run, which is where a reader
#: goes when a number is in question.
#:
#: ``designed_interface_quality`` carries a provenance record too, and needs no
#: entry here because the whole block is already omitted above.
#:
#: ``inputs`` is deliberately in neither set, although it is a list and could
#: grow. It is the one new field whose content differs from run to run and is
#: about that run: which predictor tables, panels and curated evidence it was
#: handed, each with a size and a digest. Comparing two runs in a tree is
#: exactly the question it answers, so dropping it here would remove it from
#: the tool it is most useful in. The writer caps it at 64 files and withholds
#: the whole list past that, so the worst case is bounded rather than open.
_SCREEN_TRIMMED: dict[str, frozenset[str]] = {"cuts": frozenset({"provenance"})}
#: The fields of that summary, in the order ``refold_report`` writes them.
_REFOLD_SUMMARY = ("declared", "evaluated", "criteria_file", "threshold_source", "refold_blocker")
#: One design's measurements, and the three reasons a measurement is missing.
#: Without those a null arrives with no cause: a run whose designs were written
#: compressed, one whose chain was never named, and one whose metrics row was
#: ambiguous all returned gravy null and contact_ms null and said nothing, while
#: the screen beside them named which it was. ``row_blocker`` is about the
#: metrics row, ``sequence_blocker`` about the chain, ``interface_blocker``
#: about the contact surface, and the screen writes each only when it applies.
_SAMPLE_FIELDS = (
    "example_id",
    "contact_ms",
    "clash_fraction",
    "gravy",
    "instability_index",
    "sequence_pass",
    "sequence_identity",
    "ca_rmsd",
    "row_blocker",
    "sequence_blocker",
    "interface_blocker",
)
#: The first design's sequence block. ``sequence_blocker`` is why the three
#: numbers are null when they are, and :func:`read_sequence_measures` reports
#: the status over the measurements alone, so a block carrying nothing but a
#: reason is not announced as a recorded measurement.
_DEVELOPABILITY_FIELDS = ("gravy", "instability_index", "sequence_pass", "sequence_blocker")
#: The names in that tuple that are reasons rather than measurements.
_DEVELOPABILITY_REASONS = frozenset({"sequence_blocker"})
_STRUCTURE_TEXT = ("entry_id", "chain", "accession", "group", "source", "kind")
#: The spellings a stored field uses to say it holds no value: the two this
#: view substitutes for a key it did not find, and the ``absent`` the screen
#: writes for a verdict it could not reach (``screen._ABSENT``). Named once
#: because it was written out three times in this module -- for a PPI flag, a
#: checkpoint name and a measurement -- which is the drift this file complains
#: about elsewhere, committed here.
_NO_VALUE = frozenset({NOT_RECORDED, NOT_PERFORMED, "absent"})


def call_layer(name: str, arguments: Mapping[str, object]) -> dict[str, object]:
    """Run one layer tool. An unknown name is the same error as the other tools.

    The names come from ``mcp_api._LAYER_TOOLS``, the same tuple that routes a
    call here in the first place, rather than from a second set spelled out in
    this module. Imported inside the call because ``mcp_api`` imports this
    module at module scope and that is the acyclic direction.
    """
    from rfproteina.disease.mcp_api import _LAYER_TOOLS

    if name not in _LAYER_TOOLS:
        raise ValueError("Unknown tool.")
    if name == "describe_architecture":
        _reject(arguments, set())
        return describe_architecture()
    if name == "retrieve_databases":
        return retrieve_databases(arguments)
    if name == "retrieve_structures":
        return retrieve_structures(arguments)
    if name == "read_sequence_measures":
        return read_sequence_measures(arguments)
    if name == "read_results":
        return read_results(arguments)
    if name == "declare_corpora":
        return declare_corpora(arguments)
    return backend_status(arguments)


def declare_corpora(arguments: Mapping[str, object]) -> dict[str, object]:
    """What every embedded model was trained on, and the benchmark coverage that follows.

    Reads nothing from disk and contacts nothing. The answer is the registry in
    :mod:`rfproteina.disease.corpus_provenance` plus, when ``against`` names a
    dataset, the overlap each model has with it -- which is a property of the
    pair and so cannot be precomputed into the registry. Published reference
    studies keep cohort metrics and their sources separate from model declarations.
    """
    from rfproteina.disease import corpus_provenance, ddd_benchmark, immuscope_study

    _reject(arguments, {"against"})
    against = arguments.get("against")
    if against is not None and not isinstance(against, str):
        raise ValueError("against must be a string.")
    models = sorted(corpus_provenance.TRAINED_ON)
    return {
        "models": [corpus_provenance.declare(model, against=against) for model in models],
        "corpora": corpus_provenance.describe_corpora(),
        "benchmark_coverage": ddd_benchmark.coverage(),
        "reference_studies": {"immuscope": immuscope_study.published_study()},
    }


def describe_architecture() -> dict[str, object]:
    """The three layers. This call does not ask a model or read a directory."""
    from rfproteina.disease.research import MODEL_CHOICES

    other_models: list[dict[str, object]] = []
    for provider in ("nvidia", "openrouter"):
        for model_id, label in MODEL_CHOICES[provider]:
            other_models.append({"id": model_id, "label": label, "provider": provider})
    payload: dict[str, object] = {
        "status": "listed",
        "layers": [
            {
                "id": "ai_interface",
                "name": "AI Interface",
                "status": "listed",
                "providers": [
                    _provider("OpenAI", "openai", MODEL_CHOICES),
                    _provider("Claude", "claude", MODEL_CHOICES),
                    _provider("xAI", "grok", MODEL_CHOICES),
                    {
                        "name": "Other compatible models",
                        "providers": ["nvidia", "openrouter"],
                        "models": other_models,
                        "status": "listed",
                    },
                ],
                "notes": [
                    "API keys are not accepted.",
                    "A model reply is not a measurement.",
                ],
            },
            {
                "id": "mcp_integration",
                "name": "MCP Integration Layer",
                "status": "listed",
                "capabilities": [
                    # Named here because a client that calls this to learn the
                    # surface was told about neither, and read_verdict is the
                    # one tool that returns a verdict with its per-axis
                    # evidence without rescreening anything.
                    {"name": "Program options", "tool": "list_options"},
                    {"name": "Database retrieval", "tool": "retrieve_databases"},
                    {"name": "Protein structure retrieval", "tool": "retrieve_structures"},
                    {"name": "Sequence processing", "tool": "read_sequence_measures"},
                    {
                        "name": "Generative design execution",
                        "tool": "run",
                        "note": (
                            "Generation starts only when options.run is true. "
                            "confirm_inference writes jobs and does not start them."
                        ),
                    },
                    {
                        "name": "Benchmark answers",
                        "tool": "run",
                        "note": "options.benchmark_answer returns structured benchmark_answers.",
                    },
                    {
                        "name": "Conformational dynamics analysis",
                        "tool": "run",
                        "note": "options.dynamics runs the installed headless dynamics workflow.",
                    },
                    {
                        "name": "H.U.M.A.N. protocol analysis",
                        "tool": "run",
                        "note": "options.human_protocol runs the installed headless protocol workflow.",
                    },
                    {
                        "name": "Refolding workflow",
                        "tool": "run",
                        "note": (
                            "options.refold prepares fold requests or imports returned folds; "
                            "the response carries the structured refold record."
                        ),
                    },
                    {
                        "name": "Screening pipeline execution",
                        "tool": "screen_directory",
                        "note": "lookup_targets contacts public databases only when true.",
                    },
                    {
                        "name": "Verdict retrieval",
                        "tool": "read_verdict",
                        "note": (
                            "Stored verdicts with their per-axis evidence. Nothing is rescreened, "
                            "and a run marked archived is a copy the generation archive made."
                        ),
                    },
                    {"name": "Results retrieval", "tool": "read_results"},
                    {"name": "Output artifact inventory", "tool": "list_output_files"},
                    {"name": "Output artifact reading", "tool": "read_output_file"},
                    {"name": "Candidate validation recommendations", "tool": "plan_validation"},
                    {"name": "Read-only output validation", "tool": "validate_output_files"},
                    {"name": "Save output validation report", "tool": "save_validation_report"},
                    {"name": "List saved validation reports", "tool": "list_validation_reports"},
                    {"name": "Installed scientific screen recommendations", "tool": "plan_screening"},
                    {"name": "Selected scientific screens", "tool": "run_selected_screens"},
                    {
                        "name": "Training-corpus declaration",
                        "tool": "declare_corpora",
                        "note": (
                            "What each embedded model was fitted on. Overlap is reported for a "
                            "(model, dataset) pair, because contamination is a property of the "
                            "pair and not of the model."
                        ),
                    },
                ],
            },
            {
                "id": "computational_backend",
                "name": "Computational Backend",
                "status": "listed",
                "tool": "backend_status",
                "components": [
                    {
                        "id": "modified_rfd4",
                        "name": MODIFIED_MODEL,
                        "status": NOT_INSTALLED,
                        "note": (
                            f"{MODIFIED_MODEL} is modified weights only. No checkpoint file ships with this "
                            "checkout and none is installed, so every command currently generates with the "
                            "original release model. A run names its own file with --modified-ckpt; "
                            "backend_status reports which model a directory actually recorded."
                        ),
                    },
                    {
                        "id": "gpu",
                        "name": "GPU processing",
                        "status": NOT_PERFORMED,
                        "note": "This call does not start GPU work.",
                    },
                    {
                        "id": "structural_prediction",
                        "name": "Structural prediction",
                        "status": NOT_PERFORMED,
                        "note": "Structural prediction is not launched here.",
                    },
                    {
                        "id": "docking",
                        "name": "Docking",
                        "status": NOT_PERFORMED,
                        "note": _DOCKING_NOTE,
                    },
                    {
                        "id": "binding_assessment",
                        "name": "Binding assessment",
                        "status": "selected_by_run",
                        "note": (
                            "backend_status reads stored contact and PPI measurements. "
                            "A clear screen is a computational result. It is not experimental validation."
                        ),
                    },
                    {
                        "id": "candidate_ranking",
                        "name": "Candidate ranking",
                        "status": "selected_by_run",
                        "note": "backend_status orders stored numeric confidence. It does not invent a score.",
                    },
                ],
            },
        ],
    }
    return _finish(payload)


def retrieve_databases(arguments: Mapping[str, object]) -> dict[str, object]:
    """Stored database statuses for one directory. No database is contacted."""
    from rfproteina.disease.research import SOURCE_LABELS, SOURCE_ORDER

    root = _require_directory(arguments, climbs=True)
    climb = _climbing(arguments)
    payload: dict[str, object] = {
        "status": NOT_RECORDED,
        "directory": _directory_field(root),
        "subjects": [],
        "databases": [{"key": name, "name": SOURCE_LABELS[name], "status": NOT_RECORDED} for name in SOURCE_ORDER],
        "note": (
            "These names are the databases the program can ask. "
            "This directory has no stored lookup, so none of them was consulted."
        ),
    }
    research, read_from = _artifact(root, "target_research.json", climb=climb)
    if read_from is not None:
        payload["read_from"] = _directory_field(read_from)
        # The default note says this directory holds no lookup, which stops
        # being true the moment one is read from an ancestor. The field above
        # is the part a caller reads; this is the part a person does.
        payload["note"] = (
            "This directory holds no lookup of its own. The one under read_from covers the output "
            "directory these runs sit in, so it is not a measurement of this run alone."
        )
    if _is_real_file(research):
        body = _read_json(research)
        subjects: list[dict[str, object]] = []
        rows = body.get("subjects") if isinstance(body, dict) else None
        if isinstance(rows, list):
            subjects.extend(_subject(row, SOURCE_ORDER) for row in rows if isinstance(row, dict))
        payload["subjects"] = subjects
        # The status follows what the file holds, not whether it exists. An
        # empty or non-object file was reported as a recorded lookup, and it
        # took the database list and the note saying none of them was consulted
        # out of the answer in the same breath, so nothing was left to explain
        # the empty list. One subject that found nothing is still a lookup that
        # was performed and keeps its "recorded".
        if subjects:
            payload["status"] = "recorded"
            payload.pop("databases", None)
            payload.pop("note", None)
    counts = _database_counts(root, climb=climb)
    if counts is not None:
        # Lifted, so one answer has one provenance. The structure list climbs
        # on its own, and a run holding its own lookup but no structure list
        # reported no read_from at all while the counts beside it were the
        # output directory's -- and read_from is the only field the schema
        # declares for this, so that is the field a client reads.
        where = counts.pop("read_from", None)
        payload["structure_database_counts"] = counts
        if where is not None and "read_from" not in payload:
            payload["read_from"] = where
    return _finish(payload)


def retrieve_structures(arguments: Mapping[str, object]) -> dict[str, object]:
    """Stored structure rows and generation identity. No file is downloaded."""
    root = _require_directory(arguments, climbs=True)
    structures: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    stored, read_from = _artifact(root, "returned_structures.json", climb=_climbing(arguments))
    if _is_real_file(stored):
        body = _read_json(stored)
        rows = body.get("structures") if isinstance(body, dict) else None
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = _structure_row(row)
                key = (str(item["entry_id"]), str(item["chain"]), str(item["group"]))
                if key in seen:
                    continue
                seen.add(key)
                structures.append(item)
    generations: list[dict[str, object]] = []
    identity = root / "generation.json"
    if _is_real_file(identity):
        body = _read_json(identity)
        if isinstance(body, dict):
            generations.append(_allow(body, _GENERATION_FIELDS))
    status = "recorded" if structures or generations else NOT_RECORDED
    payload: dict[str, object] = {
        "status": status,
        "directory": _directory_field(root),
        "structures": structures,
        "generations": generations,
    }
    if read_from is not None:
        # The structure list is written once per output directory. Saying where
        # it was read keeps a parent's list from reading as this run's own.
        payload["read_from"] = _directory_field(read_from)
    return _finish(payload)


def read_sequence_measures(arguments: Mapping[str, object]) -> dict[str, object]:
    """Stored sequence measurements. The sequence text is left in the file."""
    root = _require_directory(arguments)
    payload: dict[str, object] = {
        "status": NOT_RECORDED,
        "directory": _directory_field(root),
        "note": "The sequence text is not returned.",
    }
    path = root / "screen.json"
    if not _is_real_file(path):
        return _finish(payload)
    body = _read_json(path)
    if not isinstance(body, dict):
        return _finish(payload)
    developability = _allow(body.get("sequence_developability"), _DEVELOPABILITY_FIELDS)
    samples = _samples(body.get("samples"))
    payload["sequence_developability"] = developability
    payload["samples"] = samples
    # Recorded when a measurement is here, not when the file is. A verdict
    # written before the sequence was read came back as recorded with gravy,
    # the instability index, the sequence verdict and the sample list all
    # empty, which is the same answer a directory with no screen gives.
    #
    # Taken over the measurements alone, and over the ones that are really
    # there. Three things mean "no measurement here" and only one of them used
    # to count: the string not_recorded this view substitutes for a missing
    # key, a stored null, and the text absent the screen writes for a verdict
    # it could not reach. So a run whose designs were written compressed came
    # back as a recorded sequence measurement whose every number was null,
    # beside a sequence_blocker naming exactly why there was none.
    measured = [
        value
        for name, value in developability.items()
        if name not in _DEVELOPABILITY_REASONS and not _unmeasured(value)
    ]
    if samples or measured:
        payload["status"] = "recorded"
    return _finish(payload)


def read_results(arguments: Mapping[str, object]) -> dict[str, object]:
    """Stored traces, generation records, jobs, and screen verdicts."""
    root = _require_directory(arguments)
    stored = _load(root)
    traces = [_public_trace(folder, filename, payload) for folder, filename, payload in stored.traces]
    screens = [_public_screen(folder, payload) for folder, payload in stored.screens]
    found = bool(traces or stored.generations or stored.jobs or screens)
    return _finish(
        {
            "status": "recorded" if found else NOT_RECORDED,
            "directory": _directory_field(root),
            "complete": stored.complete,
            "traces": traces,
            "generations": stored.generations,
            "jobs": stored.jobs,
            "screens": screens,
            "unreadable": stored.unreadable,
        }
    )


def backend_status(arguments: Mapping[str, object]) -> dict[str, object]:
    """The computational backend for one directory, or with nothing recorded.

    Docking and GPU work are reported as not performed. This call does not
    start either, and it does not start a prediction.
    """
    root = _optional_directory(arguments)
    stored = _load(root) if root is not None else _empty()
    components = [
        _modified(stored.traces),
        _idle("gpu", "GPU processing", "This call does not start GPU work."),
        _structural(stored.traces, stored.screens),
        _idle("docking", "Docking", _DOCKING_NOTE),
        _binding(stored.screens),
        _ranking(stored.traces, stored.screens),
    ]
    recorded = any(component.get("status") == "recorded" for component in components)
    payload: dict[str, object] = {
        "status": "recorded" if recorded else NOT_RECORDED,
        "components": components,
    }
    if root is not None:
        payload["directory"] = _directory_field(root)
        payload["complete"] = stored.complete
        payload["unreadable"] = stored.unreadable
    return _finish(payload)


def _provider(name: str, provider: str, choices: Mapping[str, tuple[tuple[str, str], ...]]) -> dict[str, object]:
    models = [{"id": model_id, "label": label} for model_id, label in choices[provider]]
    return {"name": name, "provider": provider, "models": models, "status": "listed"}


def _subject(row: Mapping[str, object], order: tuple[str, ...]) -> dict[str, object]:
    held = row.get("sources")
    sources_in = held if isinstance(held, dict) else {}
    sources: dict[str, object] = {}
    for name in order:
        entry = sources_in.get(name)
        status = NOT_RECORDED
        detail: object = NOT_RECORDED
        if isinstance(entry, dict):
            raw_status = entry.get("status")
            if isinstance(raw_status, str) and raw_status.strip():
                status = raw_status
            if "detail" in entry:
                detail = _capped(entry.get("detail"))
        sources[name] = {"status": status, "detail": detail}
    identifier = row.get("identifier")
    subject: dict[str, object] = {
        "identifier": identifier if isinstance(identifier, str) and identifier.strip() else NOT_RECORDED,
        "sources": sources,
    }
    count = row.get("sources_with_record")
    subject["sources_with_record"] = count if isinstance(count, int) and not isinstance(count, bool) else NOT_RECORDED
    return subject


def _artifact(root: Path, name: str, *, climb: bool) -> tuple[Path, Path | None]:
    """The named artifact for this directory, and the ancestor it was found in.

    The program writes ``target_research.json`` and ``returned_structures.json``
    once per output directory, not once per run: the lookup and the structure
    list are about the search that led to the runs. Every other tool here takes
    a run directory, so pointed at one -- the directory ``read_verdict``,
    ``screen_directory``, ``read_results`` and ``backend_status`` all take --
    these two answer ``not_recorded`` for the very runs the lookup was
    performed for.

    That is the default, and it is deliberate: a lookup covering an output
    directory is not a measurement of one run inside it, and
    ``tests/test_disease_mcp_layers.py::test_a_missing_lookup_is_not_recorded``
    writes a lookup in a parent to pin exactly that. ``climb`` is the caller
    asking for the output directory's answer anyway, and then the second return
    value says which directory it came from, so the answer can never pass for
    this run's own.

    The search starts where it was told to either way, so a directory holding
    its own file is answered from it and nothing above is read.

    The climb stops at :func:`_climb_bound`, because ``mcp_api._path`` is the
    one place this server's containment is enforced and a lexical walk out of
    the directory it approved would answer from a file it would have refused.
    """
    here = root / name
    if _is_real_file(here) or not climb:
        return here, None
    bound = _climb_bound(root)
    current = root
    for _step in range(_ARTIFACT_CLIMB):
        parent = current.parent
        if parent == current or (bound is not None and not _inside_bound(parent, bound)):
            break
        candidate = parent / name
        if _is_real_file(candidate):
            return candidate, parent
        current = parent
    return here, None


def _climb_bound(root: Path) -> tuple[Path, ...] | None:
    """How far up a climb may go, or None when only the step count bounds it.

    ``mcp_api._path`` enforces two promises about where this server may read,
    and a four-step lexical walk kept neither. With ``RFD4_MCP_ROOTS`` set --
    the mode whose whole purpose is containment -- it refuses any path outside
    the checkout or the roots that variable names, so the climb is held to
    exactly those: without this, ``{"directory": "/srv/runs"}`` under
    ``RFD4_MCP_ROOTS=/srv/runs`` reached ``/srv`` and ``/``.

    With the variable unset, ``_path`` promises only that a RELATIVE path
    cannot leave the checkout, and allows any absolute one. So a directory
    inside the checkout is held to the checkout, which keeps that promise for
    every relative argument -- ``{"directory": "."}`` climbed to the filesystem
    root -- while an absolute directory elsewhere is left to the step count,
    since a climb out of it reaches nothing the caller could not have named
    outright, and an output directory on another volume is an ordinary place
    to keep runs.

    ``Path.parent`` is lexical and ``_path`` follows links only when the
    variable is set, so with it unset a climb out of a linked directory walks
    the names it was given rather than the directory's real ancestors. That is
    the same view of the tree the rest of this module takes.
    """
    from rfproteina.disease.mcp_api import _allowed_roots, _repo_root

    checkout = _repo_root()
    roots = _allowed_roots()
    if roots is not None:
        return (checkout, *roots)
    if _inside_bound(root, (checkout,)):
        return (checkout,)
    return None


def _inside_bound(path: Path, bound: tuple[Path, ...]) -> bool:
    from rfproteina.disease.mcp_api import _inside

    return _inside(path, bound)


def _database_counts(root: Path, *, climb: bool = False) -> dict[str, object] | None:
    """Integer database counts from the structure list. The names are not in that file."""
    path, read_from = _artifact(root, "returned_structures.json", climb=climb)
    if not _is_real_file(path):
        return None
    body = _read_json(path)
    rows = body.get("structures") if isinstance(body, dict) else None
    counts: list[int] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            count = row.get("databases")
            if isinstance(count, int) and not isinstance(count, bool):
                counts.append(count)
    result: dict[str, object] = {
        "status": "recorded" if counts else NOT_RECORDED,
        "note": "The structure list counts databases and does not name them.",
    }
    if counts:
        result["counts"] = counts
    if read_from is not None:
        result["read_from"] = _directory_field(read_from)
    return result


def _structure_row(row: Mapping[str, object]) -> dict[str, object]:
    item = _allow({key: row.get(key) for key in _STRUCTURE_TEXT if key in row}, _STRUCTURE_TEXT)
    predicted = row.get("predicted")
    item["predicted"] = predicted if isinstance(predicted, bool) else NOT_RECORDED
    item["title"] = _capped(row.get("title")) if "title" in row else NOT_RECORDED
    return item


def _samples(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [_allow(item, _SAMPLE_FIELDS) for item in value if isinstance(item, dict)]


def _unmeasured(value: object) -> bool:
    """Whether a stored field holds no measurement.

    Three spellings mean it, and a status taken over the first alone reported a
    measurement that was not there: :data:`NOT_RECORDED` is what this view puts
    where a key was missing, ``None`` is what the screen stores when it read
    the design and had no number to write, and ``absent`` is what it writes for
    a verdict it could not reach (``screen._ABSENT``).
    """
    return value is None or value in _NO_VALUE


def _allow(
    value: object,
    fields: tuple[str, ...],
    *,
    optional: frozenset[str] | tuple[str, ...] = (),
) -> dict[str, object]:
    """Copy the named fields. A missing field stays not_recorded. A sequence key is not a field.

    ``optional`` names fields the record declares as absent-or-present rather
    than always-present. A missing one of those is left out instead of masked,
    because the writer means something by the absence: screen.py keeps its
    optional keys "absent rather than null, so a reader can tell 'this run has
    no ligand panel' from 'this run has an empty one'", and filling them with
    not_recorded here would throw that distinction away for every screen.
    """
    source = value if isinstance(value, dict) else {}
    kept: dict[str, object] = {}
    for name in fields:
        if name not in source or name == "sequence":
            if name in optional:
                continue
            kept[name] = NOT_RECORDED
            continue
        kept[name] = source[name]
    return kept


def _public_trace(folder: str, filename: str, payload: Mapping[str, object]) -> dict[str, object]:
    dropped = _drop_sequence(dict(payload))
    if not isinstance(dropped, dict):
        dropped = {}
    dropped["folder"] = folder
    dropped["file"] = filename
    return dropped


def screen_fields() -> tuple[str, ...]:
    """The verdict fields this tool returns: the verdict view less what it holds back.

    Derived at call time from the one list, so this view cannot fall behind the
    record the way a copy would.
    """
    from rfproteina.disease.mcp_api import _VERDICT_FIELDS

    return tuple(name for name in _VERDICT_FIELDS if name not in _SCREEN_OMITTED)


def _refold_summary(value: object) -> dict[str, object] | str:
    """How much of the refold evaluation happened, without the criteria table.

    ``evaluated`` is the number that matters here: zero means no declared
    criterion was scored, which is the opposite of a structural result being on
    record, and a reader of a tree should see that without carrying eleven
    criteria per run.
    """
    if not isinstance(value, Mapping):
        return NOT_RECORDED
    summary = {name: value[name] for name in _REFOLD_SUMMARY if name in value}
    criteria = value.get("criteria")
    if isinstance(criteria, list):
        summary["criteria"] = len(criteria)
    return summary or NOT_RECORDED


def _trimmed(payload: Mapping[str, object]) -> dict[str, object]:
    """``payload`` with every :data:`_SCREEN_TRIMMED` key dropped from its block.

    A block whose stored value is not a mapping is left as found, for the same
    reason :func:`_without_held_back` leaves one: this runs on the raw record,
    where a hand-edited file can hold anything, and a size rule is not the
    place to decide what a malformed block is.
    """
    record = dict(payload)
    for name, trimmed in _SCREEN_TRIMMED.items():
        block = record.get(name)
        if isinstance(block, Mapping):
            record[name] = {key: item for key, item in block.items() if key not in trimmed}
    return record


def _public_screen(folder: str, payload: Mapping[str, object]) -> dict[str, object]:
    from rfproteina.disease.mcp_api import _curated_axes, _without_held_back
    from rfproteina.disease.screen import SCREEN_OPTIONAL_KEYS

    # ``_HELD_BACK`` has to be applied here as well as in ``_verdict``. ``_allow``
    # copies a named block whole, so these tools walked straight past it: a tree
    # of runs returned the repeated subsequence, the ligand SMILES and the
    # reference table's absolute path that the single-run view strips, and the
    # comment on ``_HELD_BACK`` promised a reader none of the three ever leave
    # screen.json. One of them is held back precisely because it names this
    # machine, and this is the tool that returns every run at once.
    payload = _without_held_back(payload)
    payload = _trimmed(payload)
    # The program's own declaration of which keys are absent-or-present, so a
    # run that legitimately has no ligand panel is not reported as one whose
    # panel went unrecorded.
    view = _allow(payload, screen_fields(), optional=frozenset(SCREEN_OPTIONAL_KEYS))
    view["refold"] = _refold_summary(payload.get("refold"))
    view["folder"] = folder
    # The call is shown here without its axis records, so the axes a person
    # declared are named beside it. A call resting on them is not a measured one.
    curated = _curated_axes(payload)
    view["curated_axes"] = curated if curated is not None else NOT_RECORDED
    view["samples"] = _samples(payload.get("samples"))
    return view


def _load(root: Path) -> _Tree:
    folders, complete = _folders(root)
    tree = _Tree(complete=complete)
    for folder, directory in folders:
        _take_traces(tree, folder, directory)
        _take_named(tree, folder, directory)
    return tree


def _empty() -> _Tree:
    return _Tree(complete=True)


def _folders(root: Path) -> tuple[list[tuple[str, Path]], bool]:
    from rfproteina.disease.screen import ScreenWalkTruncatedError, label_for, walk_for_screens

    found: list[tuple[str, Path]] = [("", root)]
    complete = True
    try:
        for current, _filenames in walk_for_screens(root):
            found.append((label_for(root, current), current))
    except ScreenWalkTruncatedError:
        complete = False
    return found, complete


def _take_traces(tree: _Tree, folder: str, directory: Path) -> None:
    from rfproteina.disease.trace import TRACE_DIR

    trace_dir = directory / TRACE_DIR
    if not _is_real_dir(trace_dir):
        return
    for path in _listed(tree, folder, trace_dir):
        if path.suffix != ".json" or not _is_real_file(path):
            continue
        dropped = _stored(tree, folder, path)
        if isinstance(dropped, dict):
            tree.traces.append((folder, path.name, dropped))


def _take_named(tree: _Tree, folder: str, directory: Path) -> None:
    generation = directory / "generation.json"
    if _is_real_file(generation):
        body = _stored(tree, folder, generation)
        if isinstance(body, dict):
            record = _allow(body, _GENERATION_FIELDS)
            record["folder"] = folder
            tree.generations.append(record)
    for path in _listed(tree, folder, directory):
        if not path.name.endswith(".job.json") or not _is_real_file(path):
            continue
        body = _stored(tree, folder, path)
        if not isinstance(body, dict):
            continue
        record = _allow(body, _JOB_FIELDS)
        record["folder"] = folder
        record["file"] = path.name
        tree.jobs.append(record)
    screen = directory / "screen.json"
    if _is_real_file(screen):
        dropped = _stored(tree, folder, screen)
        if isinstance(dropped, dict):
            tree.screens.append((folder, dropped))


def _stored(tree: _Tree, folder: str, path: Path) -> object:
    """One file of a tree, without its sequence. An unreadable one is noted, not raised.

    A tool that reads one directory names the file it could not read and
    stops, because that file is the answer. A walk is different: one job file
    cut short by an interrupted write used to end the whole call, and every
    readable run under the same directory went unreported with it.
    """
    try:
        return _drop_sequence(_read_json(path))
    except (ValueError, RecursionError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else f"{path.name} is not readable."
        tree.unreadable.append({"folder": folder, "file": path.name, "error": reason})
        return None


def _listed(tree: _Tree, folder: str, directory: Path) -> list[Path]:
    """The entries of one folder of a tree. A folder that cannot be listed is noted."""
    try:
        return _children(directory)
    except ValueError as exc:
        tree.unreadable.append({"folder": folder, "file": directory.name, "error": str(exc)})
        return []


def _modified(traces: list[tuple[str, str, dict[str, object]]]) -> dict[str, object]:
    from rfproteina.disease.shards import MODIFIED_EXPERIMENT, ORIGINAL_EXPERIMENT

    chosen: Mapping[str, object] | None = None
    status = NOT_RECORDED
    for _folder, _filename, payload in traces:
        command = _command(payload)
        experiment = command.get("experiment")
        if experiment == MODIFIED_EXPERIMENT:
            chosen = command
            status = "recorded"
            break
        if experiment == ORIGINAL_EXPERIMENT and chosen is None:
            chosen = command
            status = NOT_PERFORMED
    component: dict[str, object] = {"id": "modified_rfd4", "name": MODIFIED_MODEL, "status": status}
    if chosen is None:
        return component
    experiment = chosen.get("experiment")
    checkpoint = chosen.get("checkpoint_name")
    component["experiment"] = experiment if isinstance(experiment, str) and experiment.strip() else NOT_RECORDED
    component["checkpoint_name"] = checkpoint if isinstance(checkpoint, str) and checkpoint.strip() else NOT_RECORDED
    if component["checkpoint_name"] in _NO_VALUE:
        component["checkpoint_name"] = NOT_RECORDED
    return component


def _structural(
    traces: list[tuple[str, str, dict[str, object]]],
    screens: list[tuple[str, dict[str, object]]],
) -> dict[str, object]:
    component: dict[str, object] = {
        "id": "structural_prediction",
        "name": "Structural prediction",
        "status": NOT_PERFORMED,
        "note": "Structural prediction is not launched here.",
    }
    measured = False
    passes: list[dict[str, object]] = []
    for folder, screen in screens:
        entry, flag = _ppi_entry(folder, screen)
        measured = measured or flag
        if len(entry) > 1:
            passes.append(entry)
    confidence: list[dict[str, object]] = []
    for _folder, filename, payload in traces:
        metrics = _confidence(payload)
        if metrics:
            confidence.append({"example_id": _example_id(payload, filename), "metrics": metrics})
            measured = True
    refolds: list[dict[str, object]] = []
    for folder, block in _per_folder(screens, "refold"):
        outcome, evaluated = _refold_outcome(folder, block)
        measured = measured or evaluated
        refolds.append(outcome)
    # A reason is not a measurement. backend_status reads this status as
    # "there is a record here" and any one component saying so makes the whole
    # call say it, so neither a ppi_blocker nor a refold_blocker may set it:
    # a run with no oracle configured has eleven criteria and no evaluation,
    # which is the opposite of a structural prediction being on record.
    if measured:
        component["status"] = "recorded"
        component.pop("note", None)
    if passes:
        component["ppi_pass"] = passes
    if confidence:
        component["confidence"] = confidence
    if refolds:
        component["refold"] = refolds
    return component


def _refold_outcome(folder: str, block: Mapping[str, object]) -> tuple[dict[str, object], bool]:
    """One screen's refold evaluation, and whether any criterion was scored.

    What a reader of this component wants to know is which criteria were
    evaluated and which bounds were missed. Both are here; the threshold
    records they were computed from are not. Eleven criteria times
    ``THRESHOLD_LIMIT`` is up to 176 five-key bound records per folder -- 4 KB
    on a real screen, times every folder under the directory -- and the part
    that answers the question is ``failed_thresholds``, which
    ``refold._criterion`` already built for exactly this purpose. The cut, the
    worst value reached and whether it held are returned per run by
    ``read_verdict``, which is the same division of labour ``_SCREEN_OMITTED``
    already draws for this block.

    The counts come from :func:`_refold_summary` rather than being built again,
    so this module has one place that reads them.
    """
    from rfproteina.disease.refold import CRITERIA_LIMIT

    summary = _refold_summary(block)
    outcome: dict[str, object] = {"folder": folder}
    if isinstance(summary, dict):
        outcome.update(summary)
    passed: list[str] = []
    failed: list[dict[str, object]] = []
    incomplete: list[str] = []
    unresolved: list[str] = []
    criteria = block.get("criteria")
    listed = criteria if isinstance(criteria, list) else []
    for item in listed[:CRITERIA_LIMIT]:
        if not isinstance(item, Mapping):
            continue
        name = _blocker(item.get("name"))
        if name is None:
            continue
        if _blocker(item.get("thresholds_blocker")) is not None:
            unresolved.append(name)
        state = item.get("state")
        if state == "incomplete":
            incomplete.append(name)
            continue
        if state == "pass":
            passed.append(name)
            continue
        if state != "fail":
            continue
        entry: dict[str, object] = {"name": name}
        missed = item.get("failed_thresholds")
        if isinstance(missed, list):
            named = [column for column in missed if isinstance(column, str) and column.strip()]
            if named:
                entry["failed_thresholds"] = named
        failed.append(entry)
    if len(listed) > CRITERIA_LIMIT:
        outcome["criteria_not_listed"] = len(listed) - CRITERIA_LIMIT
    if passed:
        outcome["passed"] = passed
    if failed:
        outcome["failed"] = failed
    if incomplete:
        outcome["incomplete"] = incomplete
    if unresolved:
        outcome["thresholds_not_resolved"] = unresolved
    # Counted from what was classified here, not from the stored ``evaluated``:
    # a hand-edited number beside an empty criteria list must not record.
    return outcome, bool(passed or failed)


def _binding(screens: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
    measured = False
    measurements: list[dict[str, object]] = []
    for folder, screen in screens:
        item, flag = _ppi_entry(folder, screen)
        measured = measured or flag
        samples: list[dict[str, object]] = []
        raw_samples = screen.get("samples")
        listed = raw_samples if isinstance(raw_samples, list) else []
        for sample in listed:
            if not isinstance(sample, dict):
                continue
            row = _contact_row(sample)
            if row is None:
                continue
            measured = measured or "contact_ms" in row or "clash_fraction" in row
            samples.append(row)
        if not samples:
            interface = screen.get("designed_interface")
            if isinstance(interface, Mapping):
                row = _contact_row(interface)
                # The run-level reason rides on the folder below, not here: it
                # is about the run, and a synthetic sample row would say it
                # twice.
                if row is not None and ("contact_ms" in row or "clash_fraction" in row):
                    measured = True
                    row.pop("interface_blocker", None)
                    samples.append(row)
        if samples:
            item["samples"] = samples
        # Why there is no contact surface, where the nulls are. Six ways reach
        # a null one and they used to be one silence, which this component then
        # reported as nothing being here at all.
        interface = screen.get("designed_interface")
        reason = _blocker(interface.get("interface_blocker")) if isinstance(interface, Mapping) else None
        if reason is not None:
            item["interface_blocker"] = reason
        chain = _blocker(screen.get("chain_blocker"))
        if chain is not None:
            item["chain_blocker"] = chain
        if len(item) > 1:
            measurements.append(item)
    component: dict[str, object] = {"id": "binding_assessment", "name": "Binding assessment"}
    if measured:
        component["status"] = "recorded"
    else:
        # Still true, and now it travels beside what the record says about it:
        # a directory whose designs were written compressed names that, and
        # this component used to answer as though the directory were empty.
        component["status"] = NOT_PERFORMED
        component["note"] = "No stored contact or PPI measurement is in this directory."
    if measurements:
        component["measurements"] = measurements
    return component


def _contact_row(stored: Mapping[str, object]) -> dict[str, object] | None:
    """One design's contact surface: the numbers, and the reason there are none.

    ``clash_fraction`` was read past twice and dropped both times. A design
    measured for clash and not for contact was left out of this component
    entirely, as was every design carrying only a reason.
    """
    row: dict[str, object] = {}
    contact = _number(stored.get("contact_ms"))
    if contact is not None:
        row["contact_ms"] = contact
    clash = _number(stored.get("clash_fraction"))
    if clash is not None:
        row["clash_fraction"] = clash
    reason = _blocker(stored.get("interface_blocker"))
    if reason is not None:
        row["interface_blocker"] = reason
    if not row:
        return None
    example = _blocker(stored.get("example_id"))
    if example is not None:
        row["example_id"] = example
    return row


def _ranking(
    traces: list[tuple[str, str, dict[str, object]]],
    screens: list[tuple[str, dict[str, object]]],
) -> dict[str, object]:
    """Order binder traces by one shared confidence metric. No metric means no score.

    The order is one row per stored binder trace. The screen beside them may
    record that those designs are not distinct molecules -- ``designs 8,
    distinct 1`` is one molecule under eight names -- and this component listed
    eight separately ranked candidates with nothing saying so. That spread is
    reported here beside the order.
    """
    rows: list[tuple[str, str, dict[str, int | float]]] = []
    for _folder, filename, payload in traces:
        if payload.get("kind") != "binder":
            continue
        rows.append((filename, _example_id(payload, filename), _confidence(payload)))
    component: dict[str, object] = {
        "id": "candidate_ranking",
        "name": "Candidate ranking",
        "status": NOT_PERFORMED if not rows else "recorded",
    }
    spread = _spread(screens)
    if not rows:
        component["note"] = "No binder trace is stored."
        # Reported even with nothing to rank: a measured spread is not a
        # candidate record, so it must not move the status, and dropping it
        # would lose the one measurement about the set.
        if spread:
            component["design_diversity"] = spread
        return component
    if spread:
        component["design_diversity"] = spread
        if any(_repeats(row) for row in spread):
            component["order_caveat"] = _ORDER_CAVEAT
    shared = set(rows[0][2])
    for _filename, _example, metrics in rows[1:]:
        shared &= set(metrics)
    common = sorted(shared)
    if not common:
        ordered = sorted(rows, key=lambda row: (row[0], row[1]))
        component["metric"] = NOT_RECORDED
        component["order"] = [{"example_id": example, "file": filename} for filename, example, _metrics in ordered]
        return component
    # The alphabetically first metric the traces share, which a test pins. Worth
    # knowing rather than fixing: a metric whose name begins before "ipae" would
    # become the ranking metric on every PPI trace, and the rule this module now
    # shares with the trace writer admits any name containing "confidence". No
    # such column exists -- the oracle writes plddt, pae, ptm, iptm and their
    # coi spellings -- so nothing selects differently today.
    metric = common[0]
    # Best first, which depends on what the metric is. A predicted aligned
    # error is better when it is smaller, and the shared metric of any PPI
    # trace is one: sorting every metric descending listed the worst candidate
    # first under a component called Candidate ranking. The direction is
    # returned beside the name so a caller does not have to know the families.
    lower_is_better = _ERROR_METRIC.search(metric) is not None
    sign = 1.0 if lower_is_better else -1.0
    ordered = sorted(rows, key=lambda row: (sign * row[2][metric], row[1], row[0]))
    component["metric"] = metric
    # The same answer the sort above was taken on, not a constant. Hard-coded,
    # it told a caller "higher is better" about the very metric it had just
    # ordered the other way, so a reader trusting the label would read the list
    # backwards and take the worst candidate for the best.
    component["metric_direction"] = "lower_is_better" if lower_is_better else "higher_is_better"
    component["order"] = [
        {"example_id": example, "file": filename, "value": metrics[metric]} for filename, example, metrics in ordered
    ]
    return component


#: Said once, beside the order rather than in place of it. Three facts, because
#: ``design_diversity`` counts chains and names none of them: it cannot say
#: which rows are copies, and its count need not equal the number of rows,
#: which are stored trace files.
_ORDER_CAVEAT = (
    "A screen in this directory records fewer distinct designs than designs, so the order below "
    "lists one molecule more than once. design_diversity counts readable designed chains and names "
    "no design, so it cannot say which rows are copies, and its count need not equal the number of "
    "rows: those are stored binder traces."
)


def _spread(screens: list[tuple[str, dict[str, object]]]) -> list[dict[str, object]]:
    """What each screen recorded about how different its designs are from each other.

    The phrase is the producer's own (``diversity.diversity_phrase``), so this
    module does not get a second wording of a measurement it did not take.
    """
    from rfproteina.disease.diversity import diversity_phrase

    found: list[dict[str, object]] = []
    for folder, block in _per_folder(screens, "design_diversity"):
        row: dict[str, object] = {"folder": folder}
        for name in ("designs", "distinct"):
            count = _whole(block.get(name))
            if count is not None:
                row[name] = count
        pairwise = _blocker(block.get("pairwise"))
        if pairwise is not None:
            row["pairwise"] = pairwise
        phrase = diversity_phrase(block)
        if isinstance(phrase, str) and phrase.strip():
            row["phrase"] = phrase
        found.append(row)
    return found


def _repeats(row: Mapping[str, object]) -> bool:
    """Whether a stored spread says the designs are not all distinct molecules."""
    designs = _whole(row.get("designs"))
    distinct = _whole(row.get("distinct"))
    return designs is not None and distinct is not None and distinct < designs


def _confidence(payload: Mapping[str, object]) -> dict[str, int | float]:
    model = payload.get("model_confidence")
    metrics = model.get("metrics") if isinstance(model, dict) else None
    found: dict[str, int | float] = {}
    if not isinstance(metrics, dict):
        return found
    for key, raw in metrics.items():
        if not isinstance(key, str) or _CONFIDENCE.search(key) is None:
            continue
        number = _number(raw)
        if number is not None:
            found[key] = number
    return found


def _command(payload: Mapping[str, object]) -> Mapping[str, object]:
    model = payload.get("generative_model")
    command = model.get("command") if isinstance(model, dict) else None
    if isinstance(command, dict):
        return command
    return {}


def _example_id(payload: Mapping[str, object], filename: str) -> str:
    raw = payload.get("example_id")
    if isinstance(raw, str) and raw.strip() and raw not in {NOT_RECORDED, NOT_PERFORMED}:
        return raw
    stem = Path(filename).stem
    return stem or NOT_RECORDED


def _ppi_real(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        text = value.strip()
        return bool(text) and text not in _NO_VALUE
    return False


def _idle(component_id: str, name: str, note: str) -> dict[str, object]:
    return {"id": component_id, "name": name, "status": NOT_PERFORMED, "note": note}


def _blocker(value: object) -> str | None:
    """A stored reason, or None. Named once because five places were testing for it."""
    return value if isinstance(value, str) and value.strip() else None


def _whole(value: object) -> int | None:
    """A count as stored. A bool is not one, and neither is a negative."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _ppi_entry(folder: str, screen: Mapping[str, object]) -> tuple[dict[str, object], bool]:
    """One screen's PPI flag and the reason it has none, and whether a flag was measured.

    Both components that report the flag used to ask ``_ppi_real`` for
    themselves, and this module already carries a comment about two spellings
    of one rule drifting apart. One producer means the two cannot come to
    disagree about one screen.
    """
    entry: dict[str, object] = {"folder": folder}
    flag = screen.get("ppi_pass")
    measured = _ppi_real(flag)
    if measured:
        entry["ppi_pass"] = flag
    # Carried whatever the flag says, and carried verbatim including
    # ``recorded``: the program prints that value too, so filtering it here
    # would be a second opinion about the same field. A false and a
    # never-asked were one answer without it.
    reason = _blocker(screen.get("ppi_blocker"))
    if reason is not None:
        entry["ppi_blocker"] = reason
    return entry, measured


def _per_folder(screens: list[tuple[str, dict[str, object]]], key: str) -> list[tuple[str, Mapping[str, object]]]:
    """Each folder whose screen carries a mapping at ``key``. A stored non-mapping is skipped."""
    found: list[tuple[str, Mapping[str, object]]] = []
    for folder, screen in screens:
        block = screen.get(key)
        if isinstance(block, Mapping):
            found.append((folder, block))
    return found


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _capped(value: object) -> str:
    """A title or a database detail, redacted as prose and cut to a readable length.

    Marked as prose so that an upper-case word is not taken for a peptide, here
    and again when the whole result is redacted before it is returned.

    A string past the cap ends in :data:`_CUT_MARK`, because the first 300
    characters of a title returned as the title is a truncation presented as
    the record: a caller comparing it against a deposit, or quoting it, had
    nothing saying anything was missing. The mark is inside the cap, so the
    returned string is still no longer than the cap allows.
    """
    from rfproteina.disease.mcp_api import _Prose, _redact

    if not isinstance(value, str) or not value.strip():
        return NOT_RECORDED
    redacted = _redact(_Prose(value))
    if not isinstance(redacted, str) or not redacted.strip():
        return NOT_RECORDED
    if len(redacted) <= _TEXT_CAP:
        return _Prose(redacted)
    return _Prose(redacted[: _TEXT_CAP - len(_CUT_MARK)] + _CUT_MARK)


def _drop_sequence(value: object) -> object:
    """Remove a key named exactly ``sequence``. Measurements that only contain that word stay."""
    if isinstance(value, dict):
        return {str(key): _drop_sequence(item) for key, item in value.items() if key != "sequence"}
    if isinstance(value, list):
        return [_drop_sequence(item) for item in value]
    return value


def _read_json(path: Path) -> object:
    try:
        size = path.stat().st_size
    except OSError:
        raise ValueError(f"{path.name} is not readable.") from None
    if size > _MAX_JSON_BYTES:
        raise ValueError(f"{path.name} is too large to return.")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ValueError(f"{path.name} is not readable.") from None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        raise ValueError(f"{path.name} is not readable.") from None


def _require_directory(arguments: Mapping[str, object], *, climbs: bool = False) -> Path:
    """One directory argument. ``climbs`` also allows the opt-in for an ancestor's file."""
    _reject(arguments, {"directory", CLIMB_OPTION} if climbs else {"directory"})
    return _directory_path(arguments.get("directory"))


def _climbing(arguments: Mapping[str, object]) -> bool:
    """Whether the caller asked for the output directory's file. Absent means no."""
    from rfproteina.disease.mcp_api import _optional_flag

    return _optional_flag(arguments.get(CLIMB_OPTION))


def _optional_directory(arguments: Mapping[str, object]) -> Path | None:
    _reject(arguments, {"directory"})
    if "directory" not in arguments or arguments.get("directory") is None:
        return None
    return _directory_path(arguments.get("directory"))


def _directory_path(raw: object) -> Path:
    from rfproteina.disease.mcp_api import _path

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Name the directory to read.")
    path = _path(raw)
    try:
        ready = path.is_dir()
    except OSError:
        ready = False
    if not ready:
        raise ValueError("That directory is not on this computer.")
    return path


def _directory_field(path: Path) -> str:
    from rfproteina.disease.mcp_api import _shown

    return str(_shown(path))


def _reject(arguments: Mapping[str, object], allowed: set[str]) -> None:
    from rfproteina.disease.mcp_api import _reject_unknown

    _reject_unknown(arguments, allowed)


def _children(path: Path) -> list[Path]:
    try:
        found = []
        with os.scandir(path) as entries:
            for count, entry in enumerate(entries, 1):
                if count > _DIRECTORY_ENTRY_LIMIT:
                    raise ValueError(f"{path.name} has too many entries to inspect in one call.")
                found.append(Path(entry.path))
    except OSError:
        raise ValueError(f"{path.name} is not readable.") from None
    return sorted(found, key=lambda item: item.name)


def _is_real_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _is_real_dir(path: Path) -> bool:
    try:
        return path.is_dir() and not path.is_symlink()
    except OSError:
        return False


def _finish(payload: dict[str, object]) -> dict[str, object]:
    from rfproteina.disease.mcp_api import _redact

    redacted = _redact(payload)
    if not isinstance(redacted, dict):
        raise ValueError("The result could not be returned.")
    # The directory is already a shown path. Read again as free text, a
    # home-like folder in the middle of it is cut out, and two directories
    # come back as one.
    if isinstance(payload.get("directory"), str):
        redacted["directory"] = payload["directory"]
    redacted["program"] = PROGRAM
    return redacted


class _Tree:
    """Files read under one directory. The walk says whether it saw every folder."""

    def __init__(self, *, complete: bool) -> None:
        self.traces: list[tuple[str, str, dict[str, object]]] = []
        self.screens: list[tuple[str, dict[str, object]]] = []
        self.generations: list[dict[str, object]] = []
        self.jobs: list[dict[str, object]] = []
        #: Files and folders the walk could not read, each with its folder and why.
        self.unreadable: list[dict[str, object]] = []
        self.complete = complete
