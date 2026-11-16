"""Public-database research for targets on a confirmed disease list.

The record score is how many of the asked public databases returned a record
for each protein or gene, and every selected provider rates the target for
binder design so the ratings can be compared.
The model then proposes untested research, methods, and ideas grounded in those
records, and names deposited atomic structures (PDB id and chain) from the
displayed list. That step runs after the program search options are confirmed.
The program's structure list and the structures the models named are compared.
When no program option was confirmed, the named structures are the generation
inputs. The model does not invent coordinates. API keys come from the encrypted
store or the environment.
The landing page chooses the model sent with each key.
This module does not score sequences, list motifs, or describe how to make a
substance.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from rfproteina.disease.grok_bot import GROK_BOT, GROK_BOT_LABEL, AskDesktop, ask_grok_bot
from rfproteina.disease.key_store import KeySlot, load_model_choices
from rfproteina.disease.limits import slot
from rfproteina.disease.lookup import Disease
from rfproteina.disease.publish import scrubbed_json, write_text_atomically
from rfproteina.disease.sources import (
    alphafold_api_url,
    alphafold_entries,
    alphafold_match,
    alphafold_uniprot,
    is_alphafold_id,
)
from rfproteina.disease.structures import RecordScore, StructureOption, structure_kind

PROVIDERS = ("grok", "claude", "openai", "nvidia", "openrouter")
#: How each provider is written for a person. A report key carries the
#: lower-case provider name, which is an id and not a name to show.
PROVIDER_LABELS: tuple[tuple[str, str], ...] = (
    ("grok", "Grok"),
    ("claude", "Claude"),
    ("openai", "OpenAI"),
    ("nvidia", "NVIDIA"),
    ("openrouter", "OpenRouter"),
)
#: Environment names, in preference order, and the message when none is set.
_KEY_SOURCES: dict[str, tuple[tuple[str, ...], str]] = {
    "grok": (
        ("XAI_API_KEY", "GROK_API_KEY"),
        "Enter the Grok key on the first page, or set XAI_API_KEY or GROK_API_KEY.",
    ),
    "claude": (("ANTHROPIC_API_KEY",), "Enter the Claude key on the first page, or set ANTHROPIC_API_KEY."),
    "openai": (("OPENAI_API_KEY",), "Enter the OpenAI key on the first page, or set OPENAI_API_KEY."),
    "nvidia": (
        ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"),
        "Enter the NVIDIA key on the first page, or set NVIDIA_API_KEY or NVIDIA_NIM_API_KEY.",
    ),
    "openrouter": (("OPENROUTER_API_KEY",), "Enter the OpenRouter key on the first page, or set OPENROUTER_API_KEY."),
}
#: OpenAI-compatible chat endpoints: URL, model environment variable, default model.
_CHAT_ENDPOINTS: dict[str, tuple[str, str, str]] = {
    "grok": ("https://api.x.ai/v1/chat/completions", "RFD4_GROK_MODEL", "grok-4"),
    "openai": ("https://api.openai.com/v1/chat/completions", "RFD4_OPENAI_MODEL", "gpt-4.1-mini"),
    "nvidia": (
        "https://integrate.api.nvidia.com/v1/chat/completions",
        "RFD4_NVIDIA_MODEL",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1/chat/completions",
        "RFD4_OPENROUTER_MODEL",
        "openai/gpt-4.1-mini",
    ),
}
#: Landing-page model lists. The label is shown; the id is sent.
#: OpenRouter's Nemotron 3.5 id is ``nvidia/nemotron-3.5-super-vl``.
#: ``nvidia/nemotron-3.5-super-vl-120b-a12b`` is the NVIDIA NIM id.
MODEL_CHOICES: dict[str, tuple[tuple[str, str], ...]] = {
    "grok": (
        ("grok-4.7", "Grok 4.7"),
        ("grok-4.6", "Grok 4.6"),
        ("grok-4.5", "Grok 4.5"),
    ),
    "claude": (
        ("claude-fable-5-1", "Fable 5.1"),
        ("claude-opus-5-5", "Opus 5.5"),
        ("claude-opus-5", "Opus 5"),
    ),
    "openai": (
        ("gpt-6.1-terra", "Terra"),
        ("gpt-6-luna", "Luna"),
        ("gpt-6.1-sol", "Sol"),
    ),
    "nvidia": (
        ("nvidia/nemotron-3.5-super-vl-120b-a12b", "Nemotron 3.5 Super VL 120B"),
        ("nvidia/nemotron-3-ultra-550b-a55b", "Nemotron 3 Ultra 550B"),
        ("deepseek-ai/deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
        ("z-ai/glm-5.3-flash", "GLM 5.3 Flash"),
    ),
    "openrouter": (
        ("nvidia/nemotron-3.5-super-vl", "Nemotron 3.5 Super VL"),
        ("nvidia/nemotron-3-ultra-550b-a55b", "Nemotron 3 Ultra 550B"),
    ),
}
RESEARCH_LIMIT = 8
USER_AGENT = "Humanitys-Last-Disease (target research)"
#: How many targets are asked about at once in the record-score phase. The pace
#: each database is actually asked at is its own allowance in
#: :py:mod:`rfproteina.disease.limits`, not this number; this only bounds how many
#: targets are in progress, so a slow database cannot stall the ones behind it.
RECORD_WORKERS = 8
#: Seconds to wait for a provider reply, overridable with RFD4_CHAT_TIMEOUT.
CHAT_TIMEOUT = 180.0
#: Partners asked of STRING. The count feeds the network weight; five names are shown.
STRING_PARTNER_LIMIT = 50
TTD_URL = "https://ttd.idrblab.cn/"
#: RCSB entry records, used to check a structure a model names actually exists.
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry"
#: TTD's whole target list, names with the gene symbol in parentheses. The path
#: argument is ignored by the service; it is required by the route.
TTD_TARGETS_URL = "https://ttd.idrblab.cn/api/ttd/getSearchTargetByTargetNameList/all"
#: A rating span at or below this width counts as provider agreement.
AGREE_WITHIN = 15
#: A rating span at or above this width counts as a disputed target.
DISPUTE_FROM = 30
#: How much the provider consensus moves the combined rank against the record score.
MODEL_WEIGHT = 0.4
#: What the numbers in a report mean. The report states no clinical outcome either way.
REPORT_NOTE = (
    "The record score is how many of the asked public databases returned a record for that "
    "protein or gene. The generation inputs are deposited atomic structures named by PDB id "
    "and chain, not coordinates invented by a provider."
)
SYSTEM_PROMPT = (
    "Using the public database records in the user message, propose unique new research: "
    "distinct questions, methods, and ideas about cellular process, disease mechanism, "
    "and therapeutic targets. Ground each idea in a record that was returned, and name "
    "that database. Mark every proposal as an untested idea. "
    "Also give a rating, an integer from 0 to 100, for how valuable this target is for "
    "binder design, using the evidence that was returned, the disease link, the interaction "
    "network, and whether the protein class is one binders are normally made against. "
    "From the deposited structures listed in the user message, choose the atomic structures "
    "to use as generation inputs. Reply with a single JSON object and no other text: "
    '{"rating": <integer 0-100>, "reason": "<one sentence>", "ideas": "<the research>", '
    '"structures": [{"pdb_id": "<PDB id from the list>", "chain": "<chain>"}]}. '
    "Name only structures from that list. Do not invent a PDB id, an EMDB id, or coordinates. "
    "Do not give a probability of clinical success, a binding score, "
    "an amino-acid sequence, a sequence motif, a synthesis step, or a change that "
    "increases toxicity. If a database returned no record, do not invent a result for it."
)
MISSING_PROMPT = (
    "The user message lists a disease, its targets, and the deposited structures a "
    "database search already found. Find deposited structures for that Disease that "
    "the list does not already contain. Reply with a single JSON object and no other "
    'text: {"structures": [{"PDB_id": "<4-character PDB id or empty>", '
    '"uniprot_id": "<UniProt accession, or empty>", '
    '"alphafold_id": "<AF-<UniProt>-F<fragment> or empty>", '
    '"chain": "<chain>", "reason": "<Three sentence or less>"}]}. '
    "When the PDB id is missing, give the UniProt accession, or the AlphaFold id. "
    "https://alphafoldserver.com/ is the AlphaFold 3 server for input sequence or FASTA "
    "to then output 3D structure. "
    "Reply with an empty list when you know of none. Do not guess."
)
RANK_PROMPT = (
    "The user message lists deposited structures for one disease. Rate each for how "
    "valuable it is to design a new binder against, and reply with a single JSON object "
    'and no other text: {"ranked": [{"pdb_id": "<4-character PDB id or empty>", '
    '"uniprot_id": "<UniProt accession, or empty>", '
    '"alphafold_id": "<AF-<UniProt>-F<fragment> or empty>", '
    '"chain": "<chain>", "rating": <integer 0-100>, '
    '"reason": "<Three or less sentence>"}]}. '
    "When the PDB id is missing, give the uniprot_id or the AlphaFold id. "
    "Rate only structures from the list."
)
#: How much of the question box is sent. The box itself keeps the whole text.
_ASK_LIMIT = 12_000
_NO_REPLY = "No reply was kept."
ASK_PROMPT = (
    "Answer the user's message. Do not invent a database result, a PDB id, an amino-acid "
    "sequence, a synthesis step, or a probability of cure. If you do not know, say so."
)
SOURCE_ORDER = (
    "uniprot",
    "ncbi",
    "string",
    "intact",
    "human_protein_atlas",
    "pride",
    "peptideatlas",
    "ttd",
)
SOURCE_LABELS = {
    "uniprot": "UniProt",
    "ncbi": "NCBI",
    "string": "STRING",
    "intact": "IntAct",
    "human_protein_atlas": "Human Protein Atlas",
    "pride": "PRIDE",
    "peptideatlas": "PeptideAtlas",
    "ttd": "Therapeutic Target Database",
}
_TTD_ENTRY = re.compile(r"^(?P<name>.+?)\s*\((?P<symbol>[^()]+)\)\s*$")
_UNIPROT = re.compile(
    r"^(?:[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){2}"
    r"|[OPQ][0-9][A-Z0-9]{3}[0-9]"
    r"|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])"
    r"(?:-\d+)?$"
)
_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{30,}", re.IGNORECASE)
_DRUGGABLE_CLASSES = (
    "enzyme",
    "kinase",
    "protease",
    "receptor",
    "g-protein coupled",
    "transporter",
    "ion channel",
    "cd marker",
    "drug target",
)

ReadJson = Callable[[str], object]
Complete = Callable[[str, str, str], str]


class LookupMissError(Exception):
    """The database answered, and this target had no record."""


@dataclass(frozen=True)
class Subject:
    """One protein accession or gene symbol from the displayed structure list."""

    kind: str
    identifier: str
    label: str


@dataclass(frozen=True)
class SourceHit:
    """One database's answer for a subject, with the value signals it carries."""

    status: str
    detail: str
    #: Interaction partners this database reported, for the network weight.
    partners: int = 0
    #: This database tied the target to a disease.
    disease_linked: bool = False
    #: This database put the target in a class binders are normally made against.
    druggable: bool = False
    #: Entries this database returned. A database holding three counts three, so the
    #: instance total can exceed the number of databases asked.
    records: int = 0
    #: Gene symbol this database reported, which is how TTD is keyed.
    gene: str = ""


def research_subjects(rows: list[StructureOption], *, limit: int = RESEARCH_LIMIT) -> list[Subject]:
    """Unique UniProt accessions first, then gene symbols, capped at ``limit``."""
    proteins: list[Subject] = []
    genes: list[Subject] = []
    seen: set[str] = set()
    for row in rows:
        identifier = _usable_identifier(row.accession)
        if not identifier:
            continue
        if row.group == "protein" and _UNIPROT.fullmatch(identifier):
            key = identifier.split("-", 1)[0]
            if key not in seen:
                seen.add(key)
                proteins.append(Subject("protein", identifier, row.title))
        elif row.group == "gene" and _UNIPROT.fullmatch(identifier) is None:
            key = f"gene:{identifier.casefold()}"
            if key not in seen:
                seen.add(key)
                genes.append(Subject("gene", identifier, row.title))
    return (proteins + genes)[:limit]


def research_targets(
    disease: Disease,
    rows: list[StructureOption],
    *,
    providers: Sequence[str] | str | None = None,
    provider: str | None = None,
    read_json: ReadJson | None = None,
    complete: Complete | None = None,
    api_keys: dict[str, str] | None = None,
    chat_models: dict[str, str] | None = None,
    search_models: dict[str, Sequence[str]] | None = None,
    key_slots: dict[str, Sequence[KeySlot]] | None = None,
    sources: dict[str, dict[str, SourceHit]] | None = None,
) -> dict[str, object]:
    """Rank the displayed targets for binder generation with every selected provider.

    ``providers`` takes one name, several names, or ``"all"`` for every provider
    this build knows; ``provider`` is the same thing for one name. Each provider
    reads the same records, so the public databases are asked once per target
    however many providers or models were selected. A provider with no key is
    reported as unavailable and the others still run; no usable provider at all
    raises. A landing-page key is used before the environment. ``search_models``
    lists the model ids to ask for the providers that were narrowed. When it is
    omitted, each provider is asked once, using a landing-page model, then the
    provider's model environment variable, then a saved choice, then the built-in
    default.
    """
    chosen = chosen_providers(providers, provider)
    if search_models:
        unknown = sorted(name for name in search_models if name not in PROVIDERS)
        if unknown:
            known = ", ".join(PROVIDERS)
            raise ValueError(f"Unknown research provider {unknown[0]!r}. Expected one of {known}.")
        chosen = tuple(name for name in PROVIDERS if name in chosen or name in search_models)
    usable, unavailable = (chosen, {}) if complete is not None else usable_providers(chosen, api_keys)
    plans = {name: _model_plan(name, search_models) for name in usable}
    subjects = research_subjects(rows)
    report: dict[str, object] = {
        "disease": disease.name,
        "disease_id": disease.disease_id,
        "provider": usable[0] if len(usable) == 1 else "",
        "providers": list(usable),
        "providers_unavailable": unavailable,
        "model_weight": MODEL_WEIGHT,
        "subjects": [],
        "ranking": [],
        "comparison": {},
        "generation_inputs": [],
    }
    if not subjects:
        report["note"] = "No UniProt accession or gene symbol was in the displayed list."
        return report
    if complete is None:
        report["chat_models"] = {
            key: forced or resolve_chat_model(name, chat_models) for name in usable for key, forced in plans[name]
        }

    def speak(provider_name: str, system: str, prompt: str, forced: str = "") -> str:
        models = {provider_name: forced} if forced else chat_models
        return complete_chat(provider_name, system, prompt, api_keys, models, key_slots=key_slots)

    reader = read_public_json if read_json is None else read_json
    counts = _structure_counts(rows)
    shared: dict[str, object] = {}
    written: list[dict[str, object]] = []
    for subject in subjects:
        # Reuse whatever the structures page already asked for, so one run asks once.
        hits = (sources or {}).get(subject.identifier) or _lookup_sources(subject, reader, shared)
        notes = _source_notes(hits)
        listed = _rows_for_subject(subject, rows)
        structures, experimental = counts.get(subject.identifier, (0, 0))
        value = target_value(hits)
        models: dict[str, object] = {}
        for name in usable:
            for key, forced in plans[name]:
                models[key] = _ask_provider(
                    name,
                    _speaker(complete, speak, forced),
                    disease,
                    subject,
                    notes,
                    listed,
                )
        generation_inputs = _generation_choice(models, listed)
        row: dict[str, object] = {
            "kind": subject.kind,
            "identifier": subject.identifier,
            "label": subject.label,
            "sources_asked": value["asked"],
            "sources_with_record": value["evidence"],
            "score": value["evidence"],
            "structures": structures,
            "structures_total": len(rows),
            "experimental_structures": experimental,
            "program_score": value["program_score"],
            "record_share": value["record_share"],
            "partners": value["partners"],
            "disease_linked": value["disease_linked"],
            "druggable": value["druggable"],
            "sources": notes,
            "models": models,
            "narrative": _subject_narrative(models),
            "generation_inputs": generation_inputs,
        }
        row.update(_consensus(models))
        row["combined"] = _combined(row)
        written.append(row)
    report["subjects"] = written
    comparison = compare_targets(written)
    report["comparison"] = comparison
    report["ranking"] = [str(entry["identifier"]) for entry in comparison["ranked"]]
    report["generation_inputs"] = _unique_inputs(written)
    return report


def target_value(hits: dict[str, SourceHit]) -> dict[str, object]:
    """The program score for one target, with the signals its records carried.

    The score is how many of the asked public databases returned a record for
    this protein or gene, and nothing else. The deposit method of its structures
    does not enter it: an experimental deposit and a predicted one count the
    same here, and the structure list carries no score of its own. The
    interaction count, the disease link and the
    protein class are reported as facts the providers read, and none of them
    moves the score.
    """
    evidence = sum(1 for hit in hits.values() if hit.status == "record")
    asked = len(SOURCE_ORDER)
    return {
        "evidence": evidence,
        "asked": asked,
        "partners": max((hit.partners for hit in hits.values()), default=0),
        "disease_linked": any(hit.disease_linked for hit in hits.values()),
        "druggable": any(hit.druggable for hit in hits.values()),
        "program_score": evidence,
        "record_share": record_share(evidence, asked),
    }


def collect_record_scores(
    rows: list[StructureOption],
    read_json: ReadJson | None = None,
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, dict[str, SourceHit]], dict[str, RecordScore]]:
    """Ask the public databases about every listed protein and gene, once each.

    No API key is needed and no provider is called, so this can run before the
    choose-structures page and give every row its program score. The hits are
    returned as well, so a later search reuses them instead of asking again.

    The score for one accession is how many of the asked databases returned a
    record for it, out of how many were asked. A database that is down counts
    the same as one with no record; the per-source status says which it was.

    Every listed protein and gene is scored. :py:data:`RESEARCH_LIMIT` bounds how
    many targets a *provider* is asked about, because that costs tokens, but the
    public databases are free and a row with no score looks the same as a row
    that scored nothing. ``on_progress`` is called with the number of targets
    finished, the total, and the current identifier, because a long list is many
    requests and the page cannot be served until they finish.

    Each target is asked about on its own thread, because eight databases
    answering in turn for one target, for every target, was most of the wait
    before the structures page. How fast any one database is actually asked is its
    own allowance in :py:mod:`rfproteina.disease.limits`, not the number of
    threads, so none is asked faster than it permits. Every database is still
    asked about every target, nothing is skipped and no timeout is shortened: the
    answers are assembled in listed order once they are all in, so what comes back
    is what came back before. The progress count still rises one at a time, but
    which target finishes first no longer follows the listed order, because they
    no longer finish in it.
    """
    reader = read_public_json if read_json is None else read_json
    subjects = research_subjects(rows, limit=len(rows))
    shared: dict[str, object] = {}
    if subjects:
        # The one whole-list fetch of the run, taken before the threads start
        # rather than raced for by them. With no subjects it is not fetched.
        _ttd_targets(reader, shared)
    finished = {"count": 0}
    counter = threading.Lock()

    def look(subject: Subject) -> dict[str, SourceHit]:
        found = _lookup_sources(subject, reader, shared)
        if on_progress is not None:
            with counter:
                finished["count"] += 1
                on_progress(finished["count"], len(subjects), subject.identifier)
        return found

    if len(subjects) < 2:
        answers = [look(subject) for subject in subjects]
    else:
        with ThreadPoolExecutor(max_workers=min(RECORD_WORKERS, len(subjects))) as pool:
            answers = list(pool.map(look, subjects))
    hits: dict[str, dict[str, SourceHit]] = {}
    scores: dict[str, RecordScore] = {}
    for subject, found in zip(subjects, answers, strict=True):
        hits[subject.identifier] = found
        scores[subject.identifier.casefold()] = RecordScore(
            instances=sum(hit.records for hit in found.values()),
            databases=sum(1 for hit in found.values() if hit.status == "record"),
            asked=len(SOURCE_ORDER),
        )
    return hits, scores


def record_share(found: int, asked: int) -> int:
    """The program score as a percentage, so it ranks beside 0-100 provider ratings."""
    if asked <= 0:
        return 0
    return int(round(100 * found / asked))


def compare_targets(subjects: list[dict[str, object]]) -> dict[str, object]:
    """Rank the targets and name where the providers agreed and where they split."""
    ranked = sorted(
        subjects,
        key=lambda row: (-_as_int(row.get("combined")), str(row.get("identifier"))),
    )
    agreed = [str(row["identifier"]) for row in ranked if row.get("agreement") == "agreement"]
    disputed: list[dict[str, object]] = []
    for row in ranked:
        ratings = row.get("ratings")
        if row.get("agreement") != "disputed" or not isinstance(ratings, dict) or not ratings:
            continue
        high = max(ratings, key=lambda name: ratings[name])
        low = min(ratings, key=lambda name: ratings[name])
        disputed.append(
            {
                "identifier": str(row["identifier"]),
                "spread": row.get("rating_spread"),
                "highest": high,
                "highest_rating": ratings[high],
                "lowest": low,
                "lowest_rating": ratings[low],
            }
        )
    best: dict[str, tuple[int, str]] = {}
    for row in subjects:
        ratings = row.get("ratings")
        if not isinstance(ratings, dict):
            continue
        for name, rating in ratings.items():
            current = best.get(name)
            if current is None or int(rating) > current[0]:
                best[name] = (int(rating), str(row["identifier"]))
    return {
        "ranked": [_ranked_entry(row) for row in ranked],
        "agreed": agreed,
        "disputed": disputed,
        "top_by_provider": {name: identifier for name, (_, identifier) in best.items()},
    }


def format_report(report: dict[str, object]) -> str:
    """Terminal text: record-count score, provider comparison, ideas, and PDB inputs."""
    providers = [str(name) for name in _as_list(report.get("providers"))]
    heading = f"Target research for {report['disease']} ({report['disease_id']})"
    if providers:
        heading += " with " + ", ".join(providers)
    lines = [heading + ".", REPORT_NOTE]
    chat_models = report.get("chat_models")
    if isinstance(chat_models, dict) and chat_models:
        shown = []
        for name, model_id in chat_models.items():
            label = str(name)
            if " " not in label:
                label = f"{label} {model_id}"
            shown.append(label)
        if shown:
            lines.append("Models: " + ", ".join(shown) + ".")
    unavailable = report.get("providers_unavailable")
    if isinstance(unavailable, dict):
        for name in sorted(unavailable):
            lines.append(f"{name} was not used: {unavailable[name]}")
    note = report.get("note")
    if note:
        lines.append(str(note))
    comparison = report.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    ranked = _as_list(comparison.get("ranked"))
    if ranked:
        lines.append("Ranked for binder generation, highest value first:")
        for position, entry in enumerate(ranked, start=1):
            if isinstance(entry, dict):
                lines.append("  " + _ranked_line(position, entry))
    disputed = _as_list(comparison.get("disputed"))
    if disputed:
        lines.append("The providers did not agree on:")
        for entry in disputed:
            if not isinstance(entry, dict):
                continue
            lines.append(
                f"  {entry.get('identifier')}: {entry.get('highest')} rated it "
                f"{entry.get('highest_rating')}, {entry.get('lowest')} rated it "
                f"{entry.get('lowest_rating')}."
            )
    for subject in _as_list(report.get("subjects")):
        if not isinstance(subject, dict):
            continue
        lines.append(f"{subject.get('identifier')}: score {subject.get('score')} of {subject.get('sources_asked')}.")
        lines.append(f"Structures {subject.get('structures')} of {subject.get('structures_total')} found.")
        records = _record_labels(subject.get("sources"))
        if records:
            lines.append("Records from " + ", ".join(records) + ".")
        lines.append("Research ideas:")
        narrative = str(subject.get("narrative") or "").strip()
        if narrative:
            lines.append(narrative)
        inputs = subject.get("generation_inputs")
        if isinstance(inputs, list) and inputs:
            lines.append("Generation inputs:")
            for item in inputs:
                if isinstance(item, dict):
                    lines.append("  " + _input_line(item))
        models = subject.get("models")
        if isinstance(models, dict):
            for name in sorted(models):
                row = models[name]
                if not isinstance(row, dict):
                    continue
                dropped = row.get("structures_not_listed")
                if isinstance(dropped, int) and not isinstance(dropped, bool) and dropped:
                    lines.append(f"  {dropped} named structure(s) were not on the list.")
                lines.extend(f"  {line}" for line in _model_lines(name, row))
    return "\n".join(lines)


def missing_structures(
    disease: Disease,
    rows: list[StructureOption],
    *,
    providers: Sequence[str] | str | None = None,
    verify: Callable[[str], object] | None = None,
    read_json: ReadJson | None = None,
    complete: Complete | None = None,
    api_keys: dict[str, str] | None = None,
    chat_models: dict[str, str] | None = None,
    search_models: dict[str, Sequence[str]] | None = None,
    key_slots: dict[str, Sequence[KeySlot]] | None = None,
) -> tuple[list[StructureOption], dict[str, object]]:
    """Structures the models name that the program search did not find.

    A 4-character PDB id is kept when the Protein Data Bank has that deposit.
    When the PDB id is missing or not in the bank, an AlphaFold Database id
    ``AF-<UniProt>-F<fragment>`` is kept when that published model exists. A
    UniProt accession alone is read as ``AF-<accession>-F1``. The sequence in
    the AlphaFold record is not kept. The report records what each model named,
    what was kept, and what was dropped.
    """
    chosen = chosen_providers(providers)
    usable, unavailable = (chosen, {}) if complete is not None else usable_providers(chosen, api_keys)
    plans = {name: _model_plan(name, search_models) for name in usable}
    listed = {_row_key(row) for row in rows}
    checked = read_public_json if verify is None else verify

    def speak(provider_name: str, system: str, prompt: str, forced: str = "") -> str:
        models = {provider_name: forced} if forced else chat_models
        return complete_chat(provider_name, system, prompt, api_keys, models, key_slots)

    prompt = _missing_prompt(disease, rows)
    named: dict[str, list[dict[str, str]]] = {}
    no_answer: dict[str, str] = {}
    kept: list[StructureOption] = []
    seen: set[tuple[str, str, str]] = set()
    dropped: list[str] = []
    # Which model named each structure that is on the list once this search ends,
    # whether this search added it or the program search already had it. Both count
    # as that model naming it, and the page filters on either.
    found_by: dict[str, list[list[str]]] = {}
    for name in usable:
        for key, forced in plans[name]:
            speaker = _speaker(complete, speak, forced)
            try:
                answer = speaker(name, MISSING_PROMPT, prompt)
            except (OSError, ValueError) as exc:
                named[key] = [{"pdb_id": "", "chain": "", "reason": f"no answer: {exc}"}]
                no_answer[key] = str(exc)
                continue
            proposals = _named_structures(answer)
            named[key] = proposals
            for item in proposals:
                chain = _blank_id(item.get("chain")) or "A"
                pdb_id = item.get("pdb_id", "").strip()
                uniprot_id = item.get("uniprot_id", "").strip()
                alphafold_id = item.get("alphafold_id", "").strip()
                if not pdb_id and not uniprot_id and not alphafold_id:
                    continue
                # A deposit that is already on the list is not fetched again. The PDB
                # id is that deposit when it is present, including when the reply also
                # names an AlphaFold model of the same protein.
                if _is_pdb_id(pdb_id):
                    row_key = (pdb_id.casefold(), chain.casefold(), "protein")
                    if row_key in listed or row_key in seen:
                        _note_found(found_by, key, pdb_id, chain)
                        continue
                    found_row = _verified_record(pdb_id, checked)
                    if found_row is not None:
                        _store_found(kept, seen, found_by, key, pdb_id, chain, found_row)
                        continue
                model = _accepted_model(uniprot_id, alphafold_id, checked)
                if model is not None:
                    entry_id, title, accession = model
                    row_key = (entry_id.casefold(), chain.casefold(), "protein")
                    if row_key in listed or row_key in seen:
                        _note_found(found_by, key, entry_id, chain)
                        continue
                    _store_found(kept, seen, found_by, key, entry_id, chain, (title, accession, True))
                    continue
                if pdb_id:
                    dropped.append(pdb_id)
                refused = alphafold_id or (_alphafold_fallback(uniprot_id, "") if uniprot_id else "")
                if refused and refused.casefold() != pdb_id.casefold():
                    dropped.append(refused)
                elif uniprot_id and not refused and not pdb_id:
                    dropped.append(uniprot_id)
    report: dict[str, object] = {
        "providers": list(usable),
        "providers_unavailable": unavailable,
        "named": named,
        "added": [f"{row.entry_id} chain {row.chain}" for row in kept],
        "found_by": {model: sorted(entries) for model, entries in found_by.items()},
        "not_in_the_data_bank": sorted(set(dropped)),
        "no_answer": no_answer,
    }
    return kept, report


def ask_checked_models(
    question: str,
    *,
    providers: Sequence[str] | str | None = None,
    complete: Complete | None = None,
    api_keys: dict[str, str] | None = None,
    chat_models: dict[str, str] | None = None,
    search_models: dict[str, Sequence[str]] | None = None,
    key_slots: dict[str, Sequence[KeySlot]] | None = None,
    ask_desktop: AskDesktop | None = None,
) -> str:
    """Send the box to each checked model and append the replies.

    The returned text is the posted box, then one block per model. A later submit
    sends that whole text again, so earlier replies stay in the box. The outgoing
    message is only the last ``_ASK_LIMIT`` characters. A reply is passed through
    :func:`public_narrative`. One model that fails is recorded under its own name
    and the others still answer. Grok Bot has no API key. Its block is the reply
    read from the Grok Bot window after the box is pasted to Chief of Staff.
    Nothing here is written to the run log.
    """
    if not question.strip():
        return question
    named = _provider_tokens(providers)
    desktop = GROK_BOT in named
    chat = [name for name in named if name != GROK_BOT]
    if not chat and not desktop:
        chosen_providers(providers)
    blocks: list[str] = []
    if chat:
        blocks.extend(
            _chat_replies(
                question,
                chat,
                complete=complete,
                api_keys=api_keys,
                chat_models=chat_models,
                search_models=search_models,
                key_slots=key_slots,
                keep_going=desktop,
            )
        )
    if desktop:
        blocks.append(_grok_bot_block(question, ask_desktop))
    if not blocks:
        return question
    return question.rstrip() + "\n\n" + "\n\n".join(blocks)


def _provider_tokens(providers: Sequence[str] | str | None) -> list[str]:
    """Provider ids in the selection, before an unknown name is refused."""
    if providers is None:
        return []
    items = [providers] if isinstance(providers, str) else list(providers)
    names: list[str] = []
    for item in items:
        names.extend(part.strip().casefold() for part in str(item).replace(",", " ").split() if part.strip())
    return names


def _chat_replies(
    question: str,
    chat: Sequence[str],
    *,
    complete: Complete | None,
    api_keys: dict[str, str] | None,
    chat_models: dict[str, str] | None,
    search_models: dict[str, Sequence[str]] | None,
    key_slots: dict[str, Sequence[KeySlot]] | None,
    keep_going: bool,
) -> list[str]:
    """One block per keyed model. A missing key still leaves room for Grok Bot."""
    chosen = chosen_providers(chat)
    try:
        usable, unavailable = (chosen, {}) if complete is not None else usable_providers(chosen, api_keys)
    except ValueError as exc:
        if keep_going:
            return [str(exc)]
        raise
    plans = {name: _model_plan(name, search_models) for name in usable}

    def speak(provider_name: str, system: str, prompt: str, forced: str = "") -> str:
        models = {provider_name: forced} if forced else chat_models
        return complete_chat(provider_name, system, prompt, api_keys, models, key_slots)

    outgoing = question[-_ASK_LIMIT:]
    blocks: list[str] = []
    for name in chosen:
        if name in unavailable:
            blocks.append(f"{model_display(name)}:\n{unavailable[name]}")
            continue
        for key, forced in plans[name]:
            speaker = _speaker(complete, speak, forced)
            try:
                answer = speaker(name, ASK_PROMPT, outgoing)
            except (OSError, ValueError) as exc:
                blocks.append(f"{model_display(key)}:\n{exc}")
                continue
            reply = public_narrative(answer)
            blocks.append(f"{model_display(key)}:\n{reply or _NO_REPLY}")
    return blocks


def _grok_bot_block(question: str, ask_desktop: AskDesktop | None) -> str:
    """The Chief of Staff reply, or the reason it did not come back."""
    speaker = ask_grok_bot if ask_desktop is None else ask_desktop
    try:
        answer = speaker(question[-_ASK_LIMIT:])
    except (OSError, ValueError) as exc:
        return f"{GROK_BOT_LABEL}:\n{exc}"
    reply = public_narrative(answer)
    return f"{GROK_BOT_LABEL}:\n{reply or _NO_REPLY}"


def _note_found(found_by: dict[str, list[list[str]]], model: str, entry_id: str, chain: str) -> None:
    """Record that ``model`` named this structure. A repeat is recorded once."""
    entry = [entry_id, chain]
    entries = found_by.setdefault(model, [])
    if entry not in entries:
        entries.append(entry)


def _store_found(
    kept: list[StructureOption],
    seen: set[tuple[str, str, str]],
    found_by: dict[str, list[list[str]]],
    model: str,
    entry_id: str,
    chain: str,
    found_row: tuple[str, str, bool],
) -> None:
    """Keep one structure a model named after its archive record checked out."""
    title, accession, predicted = found_row
    seen.add((entry_id.casefold(), chain.casefold(), "protein"))
    _note_found(found_by, model, entry_id, chain)
    kept.append(
        StructureOption(
            group="protein",
            accession=accession,
            entry_id=entry_id,
            chain=chain,
            title=title,
            predicted=predicted,
            source="alphafold" if predicted else "llm",
            methodology_known=True,
        )
    )


def _is_pdb_id(entry_id: str) -> bool:
    """True for a 4-character Protein Data Bank id."""
    return re.fullmatch(r"[0-9A-Za-z]{4}", entry_id) is not None


def _alphafold_fallback(uniprot_id: str, alphafold_id: str) -> str:
    """AF id used when the PDB id is missing or not in the Protein Data Bank.

    An explicit ``AF-<UniProt>-F<fragment>`` is kept, including F2 and later
    segments. A UniProt accession by itself becomes ``AF-<accession>-F1``.
    """
    if is_alphafold_id(alphafold_id):
        return alphafold_id.upper()
    if _UNIPROT.fullmatch(uniprot_id):
        return f"AF-{uniprot_id}-F1"
    return ""


def _uniprot_key(accession: str) -> str:
    """Accession without an isoform suffix, for comparing P00533 with P00533-4."""
    return re.sub(r"-\d+$", "", accession.strip().upper())


def _same_uniprot(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return _uniprot_key(left) == _uniprot_key(right)


def _row_uniprot(row: StructureOption) -> str:
    """UniProt accession shown for this row on the LLM list."""
    carried = alphafold_uniprot(row.entry_id)
    if carried:
        return carried
    accession = (row.accession or "").strip().upper()
    if _UNIPROT.fullmatch(accession):
        return accession
    return ""


def _model_for_uniprot(uniprot_id: str, verify: Callable[[str], object]) -> tuple[str, str, str] | None:
    """Published model for a UniProt accession. F1, or the only other fragment.

    The sequence in the AlphaFold record is not kept.
    """
    requested = f"AF-{uniprot_id}-F1"
    url = alphafold_api_url(requested)
    if url is None:
        return None
    try:
        payload = verify(url)
    except (OSError, ValueError, LookupMissError):
        return None
    chosen = alphafold_match(payload, requested)
    if chosen is None:
        rows = [
            row
            for row in alphafold_entries(payload)
            if _same_uniprot(row.get("accession", ""), uniprot_id)
            or _same_uniprot(alphafold_uniprot(row.get("entry_id", "")), uniprot_id)
        ]
        if len(rows) == 1:
            chosen = rows[0]
    if chosen is None:
        return None
    return chosen["entry_id"], chosen["title"], chosen["accession"]


def _accepted_model(uniprot_id: str, alphafold_id: str, verify: Callable[[str], object]) -> tuple[str, str, str] | None:
    """Model to keep when the PDB id is missing or not in the Protein Data Bank.

    Returns the published entry id, title, and accession. An AlphaFold id selects
    that fragment. A UniProt accession selects F1, or the only published fragment
    when F1 is absent. The sequence is not kept.
    """
    if is_alphafold_id(alphafold_id):
        url = alphafold_api_url(alphafold_id)
        payload: object = None
        if url is not None:
            try:
                payload = verify(url)
            except (OSError, ValueError, LookupMissError):
                payload = None
        exact = alphafold_match(payload, alphafold_id) if payload is not None else None
        if exact is not None:
            return exact["entry_id"], exact["title"], exact["accession"]
        built = f"AF-{uniprot_id}-F1" if _UNIPROT.fullmatch(uniprot_id) else ""
        if not built or built.casefold() == alphafold_id.casefold():
            return None
        return _model_for_uniprot(uniprot_id, verify)
    if _UNIPROT.fullmatch(uniprot_id):
        return _model_for_uniprot(uniprot_id, verify)
    return None


def _narrow_rows(rows: list[StructureOption], chain: str) -> StructureOption | None:
    """One listed structure. A chain the model named picks among several."""
    if chain:
        named = [row for row in rows if row.chain.casefold() == chain.casefold()]
        if len(named) == 1:
            return named[0]
        if len(named) > 1:
            return None
    if len(rows) == 1:
        return rows[0]
    return None


def _listed_row(item: dict, rows: list[StructureOption]) -> StructureOption | None:
    """The listed structure this reply object names.

    A listed PDB id is used when the reply gives one. Otherwise the AlphaFold id
    is used, and otherwise the UniProt accession shown on the list. The second
    search rates only structures from that list.
    """
    fields = _structure_ids(item)
    chain = _blank_id(item.get("chain"))
    matches: list[StructureOption] = []
    if _is_pdb_id(fields["pdb_id"]):
        matches = [row for row in rows if row.entry_id.casefold() == fields["pdb_id"].casefold()]
    if not matches and is_alphafold_id(fields["alphafold_id"]):
        wanted = fields["alphafold_id"].casefold()
        matches = [row for row in rows if row.entry_id.casefold() == wanted]
    if not matches and fields["uniprot_id"]:
        matches = [row for row in rows if _same_uniprot(_row_uniprot(row), fields["uniprot_id"])]
    if not matches and fields["uniprot_id"]:
        built = _alphafold_fallback(fields["uniprot_id"], "")
        if built:
            matches = [row for row in rows if row.entry_id.casefold() == built.casefold()]
    return _narrow_rows(matches, chain)


def _blank_id(value: object) -> str:
    """An id field a model left empty. A dash is how the list prints a blank id.

    The prompt's own placeholders, such as ``<4-character PDB id or empty>``, are
    empty too. A model that copies one of those and fills another field still names
    the field it filled.
    """
    text = str(value or "").strip()
    if text.casefold() in {"-", "empty", "none", "null", "n/a", "or empty"}:
        return ""
    if text.startswith("<") and text.endswith(">"):
        return ""
    return text


def _reply_field(item: dict, *names: str) -> str:
    """The first filled field, matching the JSON key without regard to case."""
    wanted = [name.casefold() for name in names]
    found: dict[str, object] = {}
    for key, value in item.items():
        folded = str(key).casefold()
        if folded in wanted and folded not in found:
            found[folded] = value
    for name in wanted:
        if name not in found:
            continue
        text = _blank_id(found[name])
        if text:
            return text
    return ""


def _structure_ids(item: dict) -> dict[str, str]:
    """PDB, UniProt, and AlphaFold ids from one model object.

    The first search asks for ``PDB_id``. The second asks for ``pdb_id``. Either
    key is accepted, and so are ``uniprot_id`` and ``alphafold_id``. Legacy
    ``id``, ``af3_id``, and ``af_id`` are still read.
    """
    pdb = _reply_field(item, "PDB_id", "pdb_id", "id", "entry_id", "pdb")
    uniprot = _reply_field(item, "uniprot_id", "uniprot").upper()
    alphafold = _reply_field(item, "alphafold_id", "af_id", "af3_id")
    if is_alphafold_id(pdb):
        if not is_alphafold_id(alphafold):
            alphafold = pdb
        pdb = ""
    elif _UNIPROT.fullmatch(pdb.upper()) and not _UNIPROT.fullmatch(uniprot):
        uniprot = pdb.upper()
        pdb = ""
    if is_alphafold_id(uniprot):
        if not is_alphafold_id(alphafold):
            alphafold = uniprot
        uniprot = ""
    elif uniprot and _UNIPROT.fullmatch(uniprot) is None:
        uniprot = ""
    if is_alphafold_id(alphafold):
        alphafold = alphafold.upper()
    elif _UNIPROT.fullmatch(alphafold.upper()) and not uniprot:
        uniprot = alphafold.upper()
        alphafold = ""
    else:
        alphafold = ""
    return {"pdb_id": pdb.upper(), "uniprot_id": uniprot, "alphafold_id": alphafold}


def merge_found_by(
    collected: dict[str, list[list[str]]],
    report: dict[str, object] | None,
) -> dict[str, list[list[str]]]:
    """Add one first-search report's attribution to what earlier passes found.

    The page can be searched more than once, and a later pass must not erase what
    an earlier one attributed. Structures are keyed by entry id and chain. New
    rows are appended, so an earlier listed number still names the same structure.
    """
    merged = {model: [list(entry) for entry in entries] for model, entries in collected.items()}
    raw = (report or {}).get("found_by")
    if isinstance(raw, dict):
        for model, entries in raw.items():
            for entry in _as_list(entries):
                key = _found_pair(entry)
                if key is None:
                    continue
                _note_found(merged, str(model), key[0], key[1])
    return {model: sorted(entries) for model, entries in merged.items()}


def _found_pair(entry: object) -> tuple[str, str] | None:
    """One ``[entry_id, chain]`` record, or None when it is not one."""
    if isinstance(entry, list | tuple) and len(entry) == 2:
        entry_id = str(entry[0]).strip()
        chain = str(entry[1]).strip()
        if entry_id:
            return (entry_id, chain)
    return None


def model_display(key: str) -> str:
    """A readable name for a report key.

    ``_model_plan`` keys a chosen model by ``provider model_id`` and a provider
    left to its own default by its name alone. The id is swapped for the label the
    panel lists it under and the provider for the name the panel writes it as, so
    neither reaches a page as an id.

    The provider is left off when the model's own label already opens with it:
    that is what made ``grok grok-4.7`` read "grok Grok 4.7" rather than
    "Grok 4.7". It is kept otherwise, because NVIDIA and OpenRouter both offer
    Nemotron and a result has to say which of them answered.
    """
    text = str(key).strip()
    provider, _space, model_id = text.partition(" ")
    shown = dict(PROVIDER_LABELS).get(provider, provider)
    if not model_id:
        return shown
    for listed_id, label in MODEL_CHOICES.get(provider, ()):
        if listed_id == model_id or base_model(model_id) == listed_id:
            if label.casefold().startswith(shown.casefold()):
                return label
            return f"{shown} {label}"
    return text


def llm_results(
    rows: list[StructureOption],
    *,
    found: dict[str, list[list[str]]] | None = None,
    ranking: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Every listed structure an LLM search named or rated, and the models that did.

    One entry per structure in listed order, carrying the listed number so the page
    shows the same ID# its tick uses. A model that named a structure in the first
    search and a model that rated it in the second both appear against it, because
    the page's model filter accepts either.
    """
    claims: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for model, entries in (found or {}).items():
        for entry in _as_list(entries):
            pair = _found_pair(entry)
            if pair is None:
                continue
            key = (pair[0].casefold(), pair[1].casefold())
            claim = claims.setdefault(key, {}).setdefault(
                str(model), {"model": str(model), "found": False, "rating": None}
            )
            claim["found"] = True
    values: dict[tuple[str, str], object] = {}
    for item in _as_list((ranking or {}).get("ranked")):
        if not isinstance(item, dict):
            continue
        key = (str(item.get("entry_id", "")).casefold(), str(item.get("chain", "")).casefold())
        values[key] = item.get("value")
        marks = item.get("ratings")
        if not isinstance(marks, dict):
            continue
        for model, rating in marks.items():
            claim = claims.setdefault(key, {}).setdefault(
                str(model), {"model": str(model), "found": False, "rating": None}
            )
            claim["rating"] = rating
    listed: list[dict[str, object]] = []
    for number, row in enumerate(rows, start=1):
        key = (row.entry_id.strip().casefold(), row.chain.strip().casefold())
        marks = claims.get(key)
        if not marks:
            continue
        models = [
            {
                "model": name,
                "label": model_display(name),
                "found": bool(marks[name]["found"]),
                "rating": marks[name]["rating"],
            }
            for name in sorted(marks)
        ]
        listed.append(
            {
                "number": number,
                "entry_id": row.entry_id,
                "chain": row.chain,
                "accession": row.accession,
                "title": row.title,
                "source": row.source,
                "value": values.get(key),
                "models": models,
            }
        )
    return listed


def result_models(results: list[dict[str, object]] | None) -> list[str]:
    """Every model named in the results, in display order, for the model filter."""
    seen: dict[str, str] = {}
    for item in results or []:
        if not isinstance(item, dict):
            continue
        for mark in _as_list(item.get("models")):
            if isinstance(mark, dict) and mark.get("model"):
                seen.setdefault(str(mark["model"]), str(mark.get("label") or mark["model"]))
    return sorted(seen)


def _missing_prompt(disease: Disease, rows: list[StructureOption]) -> str:
    lines = [
        f"Disease: {disease.name} ({disease.disease_id})",
        "Targets: " + (", ".join(sorted({row.accession for row in rows if row.accession})) or "(none)"),
        "Structures already found:",
    ]
    # The ids and chains are what the model has to avoid repeating. Titles would make
    # the prompt many times longer on a list of a hundred structures, and a long prompt
    # is what times the call out.
    for row in rows:
        lines.append(_listed_structure_line(row, title=False))
    lines.append(
        "Find and Name deposited structures for this disease that are not on that list. "
        "Do not Search any of the 8 databases (UniProt, NCBI Gene, STRING, IntAct, "
        "Human Protein Atlas, PRIDE, PeptideAtlas, Therapeutic Target Database) "
        "Instead Search Recent, within past 2 years, medical and or scientific journals. "
        "Newly found structure and ID# would possibly be in PDB, or listed "
        "databases for ID or Structure extraction."
    )
    lines.append(
        "When the PDB id is not available, give the UniProt accession, or the AlphaFold id. "
        "https://alphafoldserver.com/ is the AlphaFold 3 server for a input sequence or FASTA "
        "to then output 3D structure."
    )
    lines.append("Reply with the JSON object only.")
    return "\n".join(lines)


def _named_structures(answer: str) -> list[dict[str, str]]:
    """PDB, UniProt, and AlphaFold ids the model named, with its reason."""
    payload = _json_object(answer)
    if payload is None:
        return []
    stated = payload.get("structures")
    if not isinstance(stated, list):
        return []
    found: list[dict[str, str]] = []
    for item in stated:
        if isinstance(item, str):
            fields = _structure_ids({"id": item})
            chain = ""
            reason = ""
        elif isinstance(item, dict):
            fields = _structure_ids(item)
            chain = str(item.get("chain") or "").strip()
            reason = public_narrative(str(item.get("reason") or ""))
        else:
            continue
        found.append({**fields, "chain": chain, "reason": reason})
    return found


def _verified_title(entry_id: str, verify: Callable[[str], object]) -> str | None:
    """The deposit's title, or None when the Protein Data Bank does not have it."""
    if not re.fullmatch(r"[0-9A-Za-z]{4}", entry_id):
        return None
    try:
        entry = verify(f"{RCSB_ENTRY_URL}/{entry_id}")
    except (OSError, ValueError, LookupMissError):
        return None
    if not isinstance(entry, dict) or not entry:
        return None
    title = str((entry.get("struct") or {}).get("title") or "").strip()
    return title or f"PDB {entry_id}"


def _verified_record(entry_id: str, verify: Callable[[str], object]) -> tuple[str, str, bool] | None:
    """Title, accession, and whether the model is a published AlphaFold prediction.

    A 4-character id is checked in the Protein Data Bank. An AF id is checked on
    https://alphafold.ebi.ac.uk/. The sequence in that record is not kept.
    """
    if is_alphafold_id(entry_id):
        url = alphafold_api_url(entry_id)
        if url is None:
            return None
        try:
            payload = verify(url)
        except (OSError, ValueError, LookupMissError):
            return None
        record = alphafold_match(payload, entry_id)
        if record is None:
            return None
        return record["title"], record["accession"], True
    title = _verified_title(entry_id, verify)
    if title is None:
        return None
    return title, entry_id, False


def rank_structures(
    rows: list[StructureOption],
    *,
    providers: Sequence[str] | str | None = None,
    complete: Complete | None = None,
    api_keys: dict[str, str] | None = None,
    chat_models: dict[str, str] | None = None,
    search_models: dict[str, Sequence[str]] | None = None,
    key_slots: dict[str, Sequence[KeySlot]] | None = None,
) -> dict[str, object]:
    """Rate every listed structure for binder design, highest value first.

    Each model rates the structures it was shown. A structure's value is the mean of
    the ratings it was given, and a structure no model rated keeps no rating at all
    rather than a zero it did not earn.
    """
    chosen = chosen_providers(providers)
    usable, unavailable = (chosen, {}) if complete is not None else usable_providers(chosen, api_keys)
    plans = {name: _model_plan(name, search_models) for name in usable}
    reason_by_model: dict[str, dict[tuple[str, str], str]] = {}

    def speak(provider_name: str, system: str, prompt: str, forced: str = "") -> str:
        models = {provider_name: forced} if forced else chat_models
        return complete_chat(provider_name, system, prompt, api_keys, models, key_slots)

    prompt = _rank_prompt(rows)
    by_model: dict[str, dict[tuple[str, str], int]] = {}
    no_answer: dict[str, str] = {}
    for name in usable:
        for key, forced in plans[name]:
            speaker = _speaker(complete, speak, forced)
            try:
                answer = speaker(name, RANK_PROMPT, prompt)
            except (OSError, ValueError) as exc:
                by_model[key] = {}
                reason_by_model[key] = {}
                no_answer[key] = str(exc)
                continue
            ratings, reasons = _rated_structures(answer, rows)
            by_model[key] = ratings
            reason_by_model[key] = reasons
    ranked: list[dict[str, object]] = []
    for number, row in enumerate(rows, start=1):
        key = (row.entry_id.casefold(), row.chain.casefold())
        ratings = {model: marks[key] for model, marks in by_model.items() if key in marks}
        reasons = {model: notes[key] for model, notes in reason_by_model.items() if key in notes}
        # The mean is the models' ratings only. Database instance counts are not an input.
        mean = round(sum(ratings.values()) / len(ratings), 1) if ratings else None
        ranked.append(
            {
                "number": number,
                "entry_id": row.entry_id,
                "chain": row.chain,
                "accession": row.accession,
                "title": row.title,
                "source": row.source,
                "ratings": ratings,
                "reasons": reasons,
                "value": mean,
            }
        )
    ranked.sort(key=lambda item: (-(item["value"] if item["value"] is not None else -1.0), item["number"]))
    return {
        "providers": list(usable),
        "providers_unavailable": unavailable,
        "ranked": ranked,
        "no_answer": no_answer,
        "order": [f"{item['entry_id']}_{item['chain']}" for item in ranked],
    }


def _listed_structure_line(row: StructureOption, *, title: bool) -> str:
    """One listed deposit: PDB id, UniProt accession, and AlphaFold Database id."""
    if is_alphafold_id(row.entry_id):
        pdb = "-"
        alphafold = row.entry_id.upper()
        uniprot = alphafold_uniprot(row.entry_id) or "-"
    else:
        pdb = row.entry_id
        alphafold = "-"
        accession = (row.accession or "").upper()
        uniprot = accession if _UNIPROT.fullmatch(accession) else "-"
    line = f"- PDB {pdb} UniProt {uniprot} AlphaFold {alphafold} chain {row.chain} {row.accession}"
    if title:
        line += f" {row.title}"
    return line


def _proposal_id(item: dict) -> str:
    """One id for a listed structure. A PDB id is used when that field is a PDB id."""
    fields = _structure_ids(item)
    if _is_pdb_id(fields["pdb_id"]):
        return fields["pdb_id"]
    af_id = _alphafold_fallback(fields["uniprot_id"], fields["alphafold_id"])
    if af_id:
        return af_id
    return fields["pdb_id"]


def _rank_prompt(rows: list[StructureOption]) -> str:
    lines = ["Structures to rate:"]
    for row in rows:
        lines.append(_listed_structure_line(row, title=True))
    lines.append(
        "Rate each for designing a binder against it. Focus on research and reasoning with, "
        "Mechanism of Therapeutic Action: Allosteric Binders, Occupancy-Driven Inhibitors, "
        "Molecular Glues & PROTACs and also Immunogenicity and Biochemical Target Dynamics: "
        "Selectivity, Binding Affinity (\\(K_{d}\\) or IC₅₀), Ect. "
        "Reply with the JSON object only."
    )
    return "\n".join(lines)


def _rated_structures(
    answer: str, rows: list[StructureOption]
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], str]]:
    """Ratings the model gave, and the sentence it gave for a person to review.

    The sentence is not a score. Database instance counts are computed elsewhere
    and are not read here.
    """
    payload = _json_object(answer)
    listed = {(row.entry_id.casefold(), row.chain.casefold()) for row in rows}
    marks: dict[tuple[str, str], int] = {}
    reasons: dict[tuple[str, str], str] = {}
    rated = payload.get("ranked") if isinstance(payload, dict) else None
    if not isinstance(rated, list):
        return marks, reasons
    for item in rated:
        if not isinstance(item, dict):
            continue
        row = _listed_row(item, rows)
        rating = _rating(item.get("rating"))
        if row is None or rating is None:
            continue
        reason = public_narrative(str(item.get("reason") or ""))
        key = (row.entry_id.casefold(), row.chain.casefold())
        if key not in listed:
            continue
        marks[key] = rating
        if reason:
            reasons[key] = reason
    return marks, reasons


def structures_by_database(
    rows: list[StructureOption],
    sources: dict[str, dict[str, SourceHit]] | None = None,
) -> list[dict[str, object]]:
    """Every structure with what each asked database returned for its target.

    The databases answer about a protein or gene, so every structure of one target
    carries that target's answers. A target that was never asked about shows every
    database as not asked, which is not the same as a database saying no.
    """
    held = sources or {}
    by_identifier = {key.strip().casefold(): value for key, value in held.items()}
    listed: list[dict[str, object]] = []
    for number, row in enumerate(rows, start=1):
        hits = by_identifier.get(row.accession.strip().casefold())
        databases: dict[str, str] = {}
        for name in SOURCE_ORDER:
            hit = (hits or {}).get(name)
            databases[name] = hit.status if hit is not None else "not asked"
        found = sum(1 for status in databases.values() if status == "record")
        listed.append(
            {
                "number": number,
                "entry_id": row.entry_id,
                "chain": row.chain,
                "accession": row.accession,
                "title": row.title,
                "group": row.group,
                "source": row.source,
                "kind": structure_kind(row),
                "databases": databases,
                "found": found,
                "asked": len(SOURCE_ORDER),
            }
        )
    return listed


def structures_by_model(ranking: dict[str, object] | None) -> dict[str, list[dict[str, object]]]:
    """Each model's structures, highest value to generate against first.

    A structure the model did not rate is left out of that model's list rather than
    ranked at the bottom on a rating it never gave.
    """
    by_model: dict[str, list[dict[str, object]]] = {}
    for item in _as_list((ranking or {}).get("ranked")):
        if not isinstance(item, dict):
            continue
        marks = item.get("ratings")
        if not isinstance(marks, dict):
            continue
        for model, rating in marks.items():
            by_model.setdefault(str(model), []).append(
                {
                    "entry_id": item.get("entry_id"),
                    "chain": item.get("chain"),
                    "accession": item.get("accession"),
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "rating": rating,
                }
            )
    for rows in by_model.values():
        rows.sort(key=lambda row: (-_as_int(row.get("rating")), str(row.get("entry_id"))))
    return {model: by_model[model] for model in sorted(by_model)}


def format_structures_by_database(listed: list[dict[str, object]]) -> str:
    """Terminal text for every structure against the eight databases."""
    lines = ["Structures by database:"]
    for item in listed:
        databases = item.get("databases")
        answered = (
            ", ".join(SOURCE_LABELS[name] for name in SOURCE_ORDER if (databases or {}).get(name) == "record")
            if isinstance(databases, dict)
            else ""
        )
        lines.append(
            f"  {item.get('number')}. PDB {item.get('entry_id')} chain {item.get('chain')} "
            f"{item.get('accession')}: {item.get('found')} of {item.get('asked')}"
            + (f" from {answered}" if answered else "")
            + "."
        )
    return "\n".join(lines)


def format_structures_by_model(by_model: dict[str, list[dict[str, object]]]) -> str:
    """Terminal text for each model's structures, highest value first."""
    if not by_model:
        return "Structures by model: (no model rated a structure)."
    lines = ["Structures by model, highest value to generate against first:"]
    for model, rows in by_model.items():
        lines.append(f"  {model}:")
        for position, row in enumerate(rows, start=1):
            lines.append(
                f"    {position}. PDB {row.get('entry_id')} chain {row.get('chain')} "
                f"{row.get('accession')} rating {row.get('rating')}, from the {row.get('source')} search."
            )
    return "\n".join(lines)


def _no_answer_lines(report: dict[str, object]) -> list[str]:
    """Models that never replied. Silence is not the same as finding nothing."""
    failed = report.get("no_answer")
    if not isinstance(failed, dict) or not failed:
        return []
    return [f"  {name} gave no answer: {failed[name]}" for name in sorted(failed)]


def format_missing_report(report: dict[str, object]) -> str:
    """Terminal text for the first search: who named what, what was added, what was refused."""
    providers = ", ".join(str(name) for name in _as_list(report.get("providers"))) or "(none)"
    lines = [f"First search with {providers}."]
    added = [str(item) for item in _as_list(report.get("added"))]
    lines.append("  Added: " + (", ".join(added) if added else "(none)") + ".")
    found_by = report.get("found_by")
    if isinstance(found_by, dict):
        for model in sorted(found_by):
            entries = [pair for pair in (_found_pair(item) for item in _as_list(found_by[model])) if pair]
            if entries:
                named = ", ".join(f"{entry_id} chain {chain}" for entry_id, chain in entries)
                lines.append(f"  {model_display(model)} named: {named}.")
    refused = [str(item) for item in _as_list(report.get("not_in_the_data_bank"))]
    if refused:
        lines.append("  Not in the Protein Data Bank or AlphaFold, so not added: " + ", ".join(refused) + ".")
    lines.extend(_no_answer_lines(report))
    unavailable = report.get("providers_unavailable")
    if isinstance(unavailable, dict):
        for name in sorted(unavailable):
            lines.append(f"  {name} was not used: {unavailable[name]}")
    return "\n".join(lines)


def format_structure_ranking(report: dict[str, object]) -> str:
    """Terminal text for the second search, highest LLM ranking first.

    The lines are for a person to review. They do not change the database
    instance counts or how many databases returned a record.
    """
    providers = ", ".join(str(name) for name in _as_list(report.get("providers"))) or "(none)"
    lines = [
        f"Second search with {providers}. Highest value to generate a binder against first:",
        "LLM ranking is ready for manual review. These rankings do not change the database "
        "instance counts or how many databases returned a record.",
    ]
    lines.extend(_no_answer_lines(report))
    for position, item in enumerate(_as_list(report.get("ranked")), start=1):
        if not isinstance(item, dict):
            continue
        marks = item.get("ratings") if isinstance(item.get("ratings"), dict) else {}
        reasons = item.get("reasons") if isinstance(item.get("reasons"), dict) else {}
        if marks:
            parts = []
            for name in sorted(marks):
                phrase = f"LLM {model_display(name)} Ranking {marks[name]}"
                reason = str(reasons.get(name) or "").strip()
                if reason:
                    phrase += f" ({reason})"
                parts.append(phrase)
            stated = "; ".join(parts)
        else:
            stated = "no LLM ranking"
        lines.append(
            f"  {position}. PDB {item.get('entry_id')} chain {item.get('chain')} "
            f"{item.get('accession')} {stated}, from the {item.get('source')} search."
        )
    return "\n".join(lines)


def compare_search_options(
    program_rows: list[StructureOption],
    llm_rows: list[StructureOption],
    confirmed: list[StructureOption] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Split the program structure search from the structures the models named.

    ``confirmed`` is the subset the program options already accepted for download.
    A model can only add a structure that is already on ``program_rows``; anything
    else stays in ``only_llm`` so the comparison can show it.
    """
    program_keys = {_row_key(row) for row in program_rows}
    llm_keys = {_row_key(row) for row in llm_rows}
    return {
        "confirmed": [_option_record(row) for row in (confirmed or [])],
        "in_both": [_option_record(row) for row in program_rows if _row_key(row) in llm_keys],
        "only_program": [_option_record(row) for row in program_rows if _row_key(row) not in llm_keys],
        "only_llm": [_option_record(row) for row in llm_rows if _row_key(row) not in program_keys],
    }


def format_option_comparison(comparison: dict[str, list[dict[str, str]]]) -> str:
    """Terminal text for the program search beside the structures the models named."""
    lines = [
        "Compared with the program search:",
        "  Confirmed program options are downloaded. When none were confirmed, the download is "
        "the structures the models named, or each target's experimental rows when they named none.",
    ]
    confirmed = _option_labels(comparison.get("confirmed"))
    lines.append("  Confirmed program options: " + (", ".join(confirmed) if confirmed else "(none)") + ".")
    for heading, key in (
        ("In both", "in_both"),
        ("Only in the program search", "only_program"),
        ("Only in the LLM options", "only_llm"),
    ):
        labels = _option_labels(comparison.get(key))
        lines.append(f"  {heading}: " + (", ".join(labels) if labels else "(none)") + ".")
    return "\n".join(lines)


def _row_key(row: StructureOption) -> tuple[str, str, str]:
    return (row.entry_id.casefold(), row.chain.casefold(), row.group.casefold())


def _option_record(row: StructureOption) -> dict[str, str]:
    return {
        "pdb_id": row.entry_id,
        "chain": row.chain,
        "accession": row.accession,
        "title": row.title,
        "group": row.group,
        "label": f"PDB {row.entry_id} chain {row.chain}",
    }


def _option_labels(rows: object) -> list[str]:
    if not isinstance(rows, list):
        return []
    labels: list[str] = []
    for item in rows:
        if isinstance(item, dict) and item.get("label"):
            labels.append(str(item["label"]))
    return labels


def explicit_structure_rows(report: dict[str, object], rows: list[StructureOption]) -> list[StructureOption]:
    """Structures the models named. The experimental fallback is not one of those options."""
    stated: list[dict[str, str]] = []
    for subject in _as_list(report.get("subjects")):
        if not isinstance(subject, dict):
            continue
        models = subject.get("models")
        if not isinstance(models, dict):
            continue
        for row in models.values():
            if not isinstance(row, dict):
                continue
            chosen = row.get("chosen_structures")
            if not isinstance(chosen, list):
                continue
            for item in chosen:
                if isinstance(item, dict):
                    stated.append({"pdb_id": str(item.get("pdb_id") or ""), "chain": str(item.get("chain") or "")})
    matched = _match_structures(stated, rows)
    chosen_rows: list[StructureOption] = []
    for item in matched:
        pdb_id = str(item.get("pdb_id") or "").casefold()
        chain = str(item.get("chain") or "").casefold()
        group = str(item.get("group") or "").casefold()
        for row in rows:
            if (
                row.entry_id.casefold() == pdb_id
                and row.chain.casefold() == chain
                and row.group.casefold() == group
                and row not in chosen_rows
            ):
                chosen_rows.append(row)
                break
    return chosen_rows


def generation_rows(report: dict[str, object], rows: list[StructureOption]) -> list[StructureOption]:
    """Real listed structures the report named as generation inputs.

    An id that is not on ``rows`` cannot match, so a model-invented PDB id is
    left out.
    """
    raw = report.get("generation_inputs")
    if not isinstance(raw, list):
        return []
    chosen: list[StructureOption] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pdb_id = str(item.get("pdb_id") or "").casefold()
        chain = str(item.get("chain") or "").casefold()
        group = str(item.get("group") or "").casefold()
        for row in rows:
            if row.entry_id.casefold() != pdb_id or row.chain.casefold() != chain:
                continue
            if group and row.group.casefold() != group:
                continue
            if row not in chosen:
                chosen.append(row)
            break
    return chosen


def _input_line(item: dict[str, object]) -> str:
    mark = " predicted" if item.get("predicted") else ""
    return f"PDB {item.get('pdb_id')} chain {item.get('chain')} " f"{item.get('accession')} {item.get('title')}{mark}"


def _model_lines(name: str, row: dict[str, object]) -> list[str]:
    error = str(row.get("error") or "").strip()
    if error:
        return [f"{name}: no answer. {error}"]
    rating = row.get("rating")
    stated = "no rating" if rating is None else f"rating {rating}"
    reason = str(row.get("reason") or "").strip()
    head = f"{name}: {stated}." + (f" {reason}" if reason else "")
    dropped = row.get("structures_not_listed")
    if isinstance(dropped, int) and dropped > 0:
        head += f" {dropped} named structure(s) were not on the list and were dropped."
    lines = [head]
    narrative = str(row.get("narrative") or "").strip()
    if narrative:
        lines.append(f"  {narrative}")
    return lines


def _ranked_entry(row: dict[str, object]) -> dict[str, object]:
    """The compact ranked view the panel and the report both read."""
    return {
        "identifier": str(row.get("identifier") or ""),
        "label": str(row.get("label") or ""),
        "kind": str(row.get("kind") or ""),
        "combined": _as_int(row.get("combined")),
        "record_share": _as_int(row.get("record_share")),
        "model_rating": row.get("model_rating"),
        "agreement": str(row.get("agreement") or ""),
        "structures": _as_int(row.get("structures")),
        "structures_total": _as_int(row.get("structures_total")),
        "program_score": _as_int(row.get("program_score")),
        "experimental_structures": _as_int(row.get("experimental_structures")),
        "sources_with_record": _as_int(row.get("sources_with_record")),
        "sources_asked": _as_int(row.get("sources_asked")),
        "ratings": dict(row.get("ratings") or {}),
    }


def _ranked_line(position: int, entry: dict[str, object]) -> str:
    rating = entry.get("model_rating")
    stated = "no provider rating" if rating is None else f"providers {rating}"
    ratings = entry.get("ratings")
    if isinstance(ratings, dict) and ratings:
        stated += " (" + ", ".join(f"{name} {ratings[name]}" for name in sorted(ratings)) + ")"
    text = f"{position}. {entry.get('identifier')}"
    label = str(entry.get("label") or "").strip()
    if label:
        text += f" {label}"
    return (
        f"{text}: combined {entry.get('combined')}, program score "
        f"{entry.get('program_score')} of {entry.get('sources_asked')}, structures "
        f"{entry.get('structures')} of {entry.get('structures_total')}, {stated}, "
        f"{entry.get('agreement')}."
    )


def search_plan(
    providers: Sequence[str] | None = None,
    model_tokens: Sequence[str] | None = None,
    *,
    all_models: bool = False,
    saved_providers: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]] | None]:
    """Providers to ask, and the listed models for the ones that were narrowed.

    The model map is None when every selected provider keeps its one landing-page,
    environment, or default model. ``all`` asks every listed model of the selected
    providers. ``saved_providers`` asks every listed model of those providers, which
    are the ones with a saved API key. A token is ``provider|model-id``. With no
    provider and a model token, the providers are the ones those models belong to.
    """
    tokens = [token.strip() for token in (model_tokens or []) if token and token.strip()]
    if any(token.casefold() in {"all", "every"} for token in tokens):
        all_models = True
        tokens = [token for token in tokens if token.casefold() not in {"all", "every"}]
    selected = tuple(chosen_providers(providers)) if providers else ()
    explicit = _explicit_models(tokens)
    if all_models:
        names = list(selected or PROVIDERS)
        for name in explicit:
            if name not in names:
                names.append(name)
        ordered = tuple(name for name in PROVIDERS if name in names)
        return ordered, {name: _catalog(name) for name in ordered}
    if saved_providers is not None:
        allowed = {name.casefold() for name in saved_providers}
        names = [name for name in PROVIDERS if name in allowed]
        for name in explicit:
            if name not in names:
                names.append(name)
        ordered = tuple(name for name in PROVIDERS if name in names)
        if not ordered:
            return (), None
        plan = {
            name: _catalog(name) if name in allowed else explicit[name]
            for name in ordered
            if name in allowed or name in explicit
        }
        return ordered, plan or None
    if not explicit:
        return selected, None
    names = list(selected)
    for name in explicit:
        if name not in names:
            names.append(name)
    ordered = tuple(name for name in PROVIDERS if name in names)
    plan = {name: explicit[name] for name in ordered if name in explicit}
    return ordered, plan or None


def _catalog(provider: str) -> tuple[str, ...]:
    return tuple(model_id for model_id, _label in model_choices(provider))


def _explicit_models(tokens: Sequence[str]) -> dict[str, tuple[str, ...]]:
    found: dict[str, list[str]] = {}
    for token in tokens:
        if "|" not in token:
            raise ValueError(f"Unknown model {token!r}. Expected all, or provider|model-id.")
        provider, model_id = (part.strip() for part in token.split("|", 1))
        provider = provider.casefold()
        catalog = _catalog(provider)
        if model_id not in catalog:
            known = ", ".join(catalog)
            raise ValueError(f"Unknown {provider} model {model_id!r}. Expected one of {known}.")
        bucket = found.setdefault(provider, [])
        if model_id not in bucket:
            bucket.append(model_id)
    return {name: tuple(model_id for model_id in _catalog(name) if model_id in ids) for name, ids in found.items()}


def _model_plan(provider: str, search_models: dict[str, Sequence[str]] | None) -> list[tuple[str, str]]:
    """Report key and forced model id for one provider.

    A chosen model is named in the key, whether one was chosen or several. Keying
    a single choice by the provider alone lost which model answered: the page then
    read "Grok" for a run that asked Grok 4.7, and the saved report could not say
    either. Only a provider left to its own default is keyed by name alone,
    because then no model was forced and naming one would be a guess.
    """
    raw = None if not search_models else search_models.get(provider)
    if raw is None:
        return [(provider, "")]
    catalog = _catalog(provider)
    ids: list[str] = []
    for item in raw:
        model_id = str(item).strip()
        if not model_id:
            continue
        if model_id not in catalog:
            known = ", ".join(catalog)
            raise ValueError(f"Unknown {provider} model {model_id!r}. Expected one of {known}.")
        if model_id not in ids:
            ids.append(model_id)
    ids = [model_id for model_id in catalog if model_id in ids]
    if not ids:
        return [(provider, "")]
    return [(f"{provider} {model_id}", model_id) for model_id in ids]


def chosen_providers(
    providers: Sequence[str] | str | None,
    provider: str | None = None,
) -> tuple[str, ...]:
    """Normalise the selection. ``all`` means every provider this build knows."""
    names: list[str] = []
    for value in (providers, provider):
        if value is None:
            continue
        items = [value] if isinstance(value, str) else list(value)
        for item in items:
            names.extend(part for part in str(item).replace(",", " ").split())
    if not names:
        known = ", ".join(("all", *PROVIDERS))
        raise ValueError(f"Choose at least one research provider. Expected any of {known}.")
    chosen: list[str] = []
    for raw in names:
        name = raw.strip().casefold()
        if name in {"all", "every"}:
            chosen.extend(PROVIDERS)
            continue
        if name not in PROVIDERS:
            known = ", ".join(("all", *PROVIDERS))
            raise ValueError(f"Unknown research provider {name!r}. Expected one of {known}.")
        chosen.append(name)
    return tuple(name for name in PROVIDERS if name in chosen)


def providers_with_keys(api_keys: dict[str, str] | None = None) -> tuple[str, ...]:
    """Providers that have a saved key or an environment key. The secret is not returned."""
    found: list[str] = []
    for name in PROVIDERS:
        try:
            api_key(name, api_keys)
        except ValueError:
            continue
        found.append(name)
    return tuple(found)


def usable_providers(
    chosen: Sequence[str],
    api_keys: dict[str, str] | None = None,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Split the selection into the providers that have a key and those that do not.

    This runs before any database is asked. When none of the selected providers
    has a key the raised message is every reason, naming each missing variable.
    """
    usable: list[str] = []
    unavailable: dict[str, str] = {}
    for name in chosen:
        try:
            api_key(name, api_keys)
        except ValueError as exc:
            unavailable[name] = str(exc)
            continue
        usable.append(name)
    if not usable:
        raise ValueError(" ".join(unavailable[name] for name in chosen if name in unavailable))
    return tuple(usable), unavailable


def _speaker(
    complete: Complete | None,
    speak: Callable[[str, str, str, str], str],
    forced: str,
) -> Complete:
    """The injected completion, or one call that sends ``forced`` when it is set."""
    if complete is not None:
        return complete

    def speak_one(provider_name: str, system: str, prompt: str) -> str:
        return speak(provider_name, system, prompt, forced)

    return speak_one


def _ask_provider(
    provider: str,
    speaker: Complete,
    disease: Disease,
    subject: Subject,
    notes: dict[str, dict[str, str]],
    listed: list[StructureOption],
) -> dict[str, object]:
    """One provider's research ideas and chosen deposited structures.

    A provider that fails is recorded against its own name and the others still
    answer, so one unreachable endpoint does not discard the whole comparison.
    """
    try:
        answer = speaker(provider, SYSTEM_PROMPT, _prompt(disease, subject, notes, listed))
    except (OSError, ValueError) as exc:
        return {"rating": None, "reason": "", "narrative": "", "chosen_structures": None, "error": str(exc)}
    return _listed_choice_only(_parse_answer(answer), listed)


def _listed_choice_only(answer: dict[str, object], listed: list[StructureOption]) -> dict[str, object]:
    """Drop named structures that are not on the displayed list.

    A provider can name a deposit that does not exist. Generation already
    refuses one, but an invented id must not reach the written report either,
    because anything reading that file would treat it as a real deposit. The
    kept rows are the matched ones and the rest become a count.
    """
    stated = answer.get("chosen_structures")
    if not isinstance(stated, list):
        return answer
    matched = _match_structures([item for item in stated if isinstance(item, dict)], listed)
    kept = [{"pdb_id": str(item.get("pdb_id") or ""), "chain": str(item.get("chain") or "")} for item in matched]
    return {
        **answer,
        "chosen_structures": kept,
        "structures_not_listed": max(0, len(stated) - len(kept)),
    }


def _parse_answer(text: str) -> dict[str, object]:
    """Read the JSON object the provider was asked for, or keep the text as ideas."""
    payload = _json_object(text)
    if payload is None:
        return {
            "rating": None,
            "reason": "",
            "narrative": public_narrative(text),
            "chosen_structures": None,
            "error": "",
        }
    ideas = payload.get("ideas")
    if isinstance(ideas, list):
        ideas = " ".join(str(item).strip() for item in ideas if str(item).strip())
    elif not isinstance(ideas, str):
        ideas = "" if ideas is None else str(ideas)
    return {
        "rating": _rating(payload.get("rating")),
        "reason": public_narrative(str(payload.get("reason") or "")),
        "narrative": public_narrative(ideas),
        "chosen_structures": _stated_structures(payload.get("structures")),
        "error": "",
    }


def _json_object(text: str) -> dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _rating(value: object) -> int | None:
    """A 0..100 rating, or None when the provider did not state a usable number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        cleaned = value.strip().rstrip("%").strip()
        try:
            value = float(cleaned)
        except ValueError:
            return None
    if not isinstance(value, int | float) or math.isnan(value) or math.isinf(value):
        return None
    return max(0, min(100, int(round(float(value)))))


def _consensus(models: dict[str, object]) -> dict[str, object]:
    """Average the ratings that were stated, and say how far apart they were."""
    ratings: dict[str, int] = {}
    for name, row in models.items():
        if not isinstance(row, dict):
            continue
        rating = row.get("rating")
        if isinstance(rating, int) and not isinstance(rating, bool):
            ratings[name] = rating
    if not ratings:
        return {"model_rating": None, "rating_spread": None, "agreement": "no rating", "ratings": {}}
    values = sorted(ratings.values())
    spread = values[-1] - values[0]
    if len(values) == 1:
        agreement = "one provider"
    elif spread <= AGREE_WITHIN:
        agreement = "agreement"
    elif spread >= DISPUTE_FROM:
        agreement = "disputed"
    else:
        agreement = "partial"
    return {
        "model_rating": round(sum(values) / len(values), 1),
        "rating_spread": spread,
        "agreement": agreement,
        "ratings": ratings,
    }


def _combined(row: dict[str, object]) -> int:
    """The program score, as a percentage, moved toward the provider consensus."""
    share = float(_as_int(row.get("record_share")))
    rating = row.get("model_rating")
    if isinstance(rating, bool) or not isinstance(rating, int | float):
        return int(round(share))
    return int(round((1.0 - MODEL_WEIGHT) * share + MODEL_WEIGHT * float(rating)))


def _rows_for_subject(subject: Subject, rows: list[StructureOption]) -> list[StructureOption]:
    """Displayed rows whose accession is this protein or gene."""
    ident = subject.identifier.casefold()
    return [row for row in rows if row.accession.strip().casefold() == ident]


def _subject_narrative(models: dict[str, object]) -> str:
    """Research ideas from the providers that answered, in provider order."""
    parts: list[str] = []
    for name in models:
        row = models[name]
        if not isinstance(row, dict):
            continue
        text = str(row.get("narrative") or "").strip()
        if text and text not in parts:
            parts.append(text)
    return " ".join(parts)


def _generation_choice(models: dict[str, object], listed: list[StructureOption]) -> list[dict[str, object]]:
    """Deposited structures to send to generation.

    A provider may name only rows from ``listed``. An invented id is dropped.
    When no named row matches, the experimental rows for this target are used.
    """
    stated: list[dict[str, str]] | None = None
    for row in models.values():
        if not isinstance(row, dict):
            continue
        chosen = row.get("chosen_structures")
        if not isinstance(chosen, list):
            continue
        if stated is None:
            stated = []
        for item in chosen:
            if isinstance(item, dict):
                stated.append({"pdb_id": str(item.get("pdb_id") or ""), "chain": str(item.get("chain") or "")})
    matched = _match_structures(stated or [], listed) if stated else []
    if matched:
        return matched
    experimental = [row for row in listed if not row.predicted and row.methodology_known]
    return [_structure_payload(row) for row in (experimental or listed)]


def _stated_structures(value: object) -> list[dict[str, str]] | None:
    """PDB ids the provider named, or None when the answer did not name any."""
    if not isinstance(value, list):
        return None
    stated: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            stated.append({"pdb_id": item, "chain": ""})
        elif isinstance(item, dict):
            stated.append({"pdb_id": _proposal_id(item), "chain": str(item.get("chain") or "")})
    return stated


def _match_structures(named: list[dict[str, str]], listed: list[StructureOption]) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in named:
        pdb_key = item.get("pdb_id", "").strip().casefold()
        chain_key = item.get("chain", "").strip().casefold()
        if not pdb_key:
            continue
        for row in listed:
            if row.entry_id.casefold() != pdb_key:
                continue
            if chain_key and row.chain.casefold() != chain_key:
                continue
            key = (row.entry_id, row.chain, row.group)
            if key not in seen:
                seen.add(key)
                found.append(_structure_payload(row))
            break
    return found


def _structure_payload(row: StructureOption) -> dict[str, object]:
    experimental = not row.predicted and row.methodology_known
    return {
        "pdb_id": row.entry_id,
        "chain": row.chain,
        "accession": row.accession,
        "title": row.title,
        "group": row.group,
        "predicted": row.predicted or not row.methodology_known,
        "experimental": experimental,
    }


def _unique_inputs(subjects: list[dict[str, object]]) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for subject in subjects:
        inputs = subject.get("generation_inputs")
        if not isinstance(inputs, list):
            continue
        for item in inputs:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("pdb_id")), str(item.get("chain")), str(item.get("group")))
            if key not in seen:
                seen.add(key)
                found.append(item)
    return found


def _structure_counts(rows: list[StructureOption]) -> dict[str, tuple[int, int]]:
    """Per accession, how many rows were listed and how many are experimental."""
    counts: dict[str, tuple[int, int]] = {}
    for row in rows:
        identifier = _usable_identifier(row.accession)
        if not identifier:
            continue
        listed, experimental = counts.get(identifier, (0, 0))
        usable = 1 if (not row.predicted and row.methodology_known) else 0
        counts[identifier] = (listed + 1, experimental + usable)
    return counts


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def write_report(directory: Path, report: dict[str, object]) -> Path:
    """Write ``target_research.json``. Keys and personal paths are removed."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "target_research.json"
    write_text_atomically(path, scrubbed_json(report))
    return path


def key_for_model(
    provider: str,
    model: str,
    api_keys: dict[str, str] | None = None,
    key_slots: dict[str, Sequence[KeySlot]] | None = None,
) -> str:
    """The key tied to ``model``, or the provider key when none names that model."""
    rows = list((key_slots or {}).get(provider, ()))
    wanted = model.strip()
    for row in rows:
        if wanted and row.model == wanted and row.secret.strip():
            return row.secret.strip()
    for row in rows:
        if not row.model and row.secret.strip():
            return row.secret.strip()
    if rows and rows[0].secret.strip():
        return rows[0].secret.strip()
    return api_key(provider, api_keys)


def checked_key(provider: str, key: str) -> str:
    """The key, or an error naming the fault without ever echoing the value.

    No provider key carries a space or a quote mark. When something else lands in
    the field — a line of a shell command, a sentence — the service answers that
    the authentication header is missing, which reads like a rejected key rather
    than a mis-paste, and the search then reports finding nothing.
    """
    for char, fault in ((" ", "whitespace"), ('"', "a quote mark"), ("'", "a quote mark")):
        if char in key or (fault == "whitespace" and any(item.isspace() for item in key)):
            raise ValueError(
                f"The saved {provider} key contains {fault}, so it is not an API key. "
                f"Re-enter the {provider} key on the first page."
            )
    return key


def api_key(provider: str, api_keys: dict[str, str] | None = None) -> str:
    """Return the provider key. The error names the variable and not the secret."""
    supplied = (api_keys or {}).get(provider, "").strip()
    if supplied:
        return supplied
    source = _KEY_SOURCES.get(provider)
    if source is None:
        known = ", ".join(PROVIDERS)
        raise ValueError(f"Unknown research provider {provider!r}. Expected one of {known}.")
    names, message = source
    for name in names:
        key = (os.environ.get(name) or "").strip()
        if key:
            return key
    raise ValueError(message)


def complete_chat(
    provider: str,
    system: str,
    user: str,
    api_keys: dict[str, str] | None = None,
    chat_models: dict[str, str] | None = None,
    key_slots: dict[str, Sequence[KeySlot]] | None = None,
) -> str:
    """Ask one provider. A failure names the provider and redacts the key.

    When several keys are saved, the one tied to the model being sent is used.
    """
    model = resolve_chat_model(provider, chat_models)
    key = checked_key(provider, key_for_model(provider, model, api_keys, key_slots))
    try:
        return _request_completion(provider, key, system, user, chat_models)
    except ValueError as exc:
        raise ValueError(_redact(str(exc), key)) from None


def read_public_json(url: str) -> object:
    """GET a public JSON value. HTTP 404 and an empty body mean no record.

    A failed read raises :class:`ValueError`, which every caller already
    handles. ``http.client.HTTPException`` is included in that: a truncated
    body raises ``IncompleteRead``, which is neither an ``OSError`` nor a
    ``ValueError``, so left uncaught it ends the whole run with a traceback
    instead of one source reporting itself down.
    """
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    try:
        with slot(url), urlopen(request, timeout=45) as response:
            raw = response.read()
            status = getattr(response, "status", 200)
    except HTTPError as exc:
        if exc.code == 404:
            exc.read()
            raise LookupMissError(url) from None
        body = exc.read().decode("utf-8", errors="replace")[:200]
        raise ValueError(f"Could not read JSON from {url}: HTTP {exc.code} {body}") from None
    except (HTTPException, OSError) as exc:
        raise ValueError(f"Could not read JSON from {url}: {exc}") from None
    if status == 204 or not raw:
        raise LookupMissError(url)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON from {url}: {exc}") from None


def public_narrative(text: str) -> str:
    """Drop long amino-acid strings from provider text and keep what it said.

    Whole sentences used to be dropped on a phrase match. It matched wording
    rather than meaning, so most of what went was a sentence stating that no
    such outcome was established, or one reporting a trial result, while a
    sentence overclaiming in other words was kept. What a provider should not
    give it is told in its own prompt, which is where that belongs; the answer
    it returns is recorded as given, minus sequences.
    """
    cleaned = _SEQUENCE.sub("[sequence omitted]", text)
    kept: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned):
        piece = part.strip()
        if not piece or piece == "[sequence omitted]":
            continue
        kept.append(piece)
    return " ".join(kept).strip()


def _lookup_sources(
    subject: Subject,
    read: ReadJson,
    cache: dict[str, object] | None = None,
) -> dict[str, SourceHit]:
    """Ask every source. TTD goes last, because it is keyed by gene symbol.

    ``cache`` holds the whole-list answers that cost one request per run rather
    than one per target. A caller that omits it pays that request per target.
    """
    shared = {} if cache is None else cache
    found: dict[str, SourceHit] = {}
    for name in SOURCE_ORDER:
        if name == "ttd":
            continue
        found[name] = _lookup_one(name, subject, read)
    uniprot = found.get("uniprot")
    symbol = uniprot.gene if uniprot is not None else ""
    found["ttd"] = _lookup_ttd(subject, read, symbol, shared)
    return {name: found[name] for name in SOURCE_ORDER}


def _source_notes(hits: dict[str, SourceHit]) -> dict[str, dict[str, str]]:
    """The serialisable view of the lookups. The signals stay on the hits."""
    return {name: {"status": hit.status, "detail": hit.detail} for name, hit in hits.items()}


def _lookup_one(name: str, subject: Subject, read: ReadJson) -> SourceHit:
    try:
        return _FINDERS[name](subject, read)
    except LookupMissError:
        return _hit_none()
    except (OSError, ValueError) as exc:
        return _hit_down(str(exc))


def _uniprot(subject: Subject, read: ReadJson) -> SourceHit:
    payload = read(_uniprot_url(subject))
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return _hit_none()
    entry = results[0]
    parts: list[str] = []
    disease_linked = False
    name = _dig(entry, ("proteinDescription", "recommendedName", "fullName", "value"))
    if name:
        parts.append(name)
    for comment in entry.get("comments") or []:
        if not isinstance(comment, dict):
            continue
        kind = str(comment.get("commentType") or "")
        if kind not in {"FUNCTION", "DISEASE"}:
            continue
        if kind == "DISEASE":
            disease_linked = True
        for text in comment.get("texts") or []:
            if isinstance(text, dict) and text.get("value"):
                parts.append(str(text["value"]))
    if not parts:
        parts.append(f"UniProt entry {entry.get('primaryAccession') or subject.identifier}")
    return _hit_record(
        " ".join(parts),
        disease_linked=disease_linked,
        records=len(results),
        gene=_uniprot_gene(entry),
    )


def _ncbi(subject: Subject, read: ReadJson) -> SourceHit:
    try:
        search = read(_ncbi_search_url(subject))
    except LookupMissError:
        return _hit_none()
    ids = _ncbi_ids(search)
    if not ids:
        return _hit_none()
    try:
        summary = read(_ncbi_summary_url(ids[0]))
    except LookupMissError:
        return _hit_record(f"NCBI gene {ids[0]}", records=len(ids))
    text = _ncbi_summary_text(summary, ids[0])
    return _hit_record(text or f"NCBI gene {ids[0]}", records=len(ids))


def _string(subject: Subject, read: ReadJson) -> SourceHit:
    payload = read(_string_url(subject))
    rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
    if not rows:
        return _hit_none()
    names: list[str] = []
    for row in rows:
        partner = str(row.get("preferredName_B") or row.get("preferredName_A") or "").strip()
        if partner and partner not in names:
            names.append(partner)
    listed = ", ".join(names[:5])
    detail = f"STRING lists {len(rows)} interaction partners"
    if listed:
        detail += f", including {listed}"
    return _hit_record(detail + ".", partners=len(rows), records=len(rows))


def _intact(subject: Subject, read: ReadJson) -> SourceHit:
    count = _interaction_count(read(_intact_url(subject)))
    if count <= 0:
        return _hit_none()
    return _hit_record(f"IntAct lists {count} interactions.", partners=count, records=count)


def _hpa(subject: Subject, read: ReadJson) -> SourceHit:
    rows = _object_rows(read(_hpa_url(subject)))
    matched = [row for row in rows if _hpa_match(row, subject)]
    if not matched:
        return _hit_none()
    row = matched[0]
    description = _hpa_field(row, "gd", "Gene description")
    disease = _hpa_field(row, "di", "Disease involvement")
    protein_class = _hpa_field(row, "pc", "Protein class")
    parts = [part for part in (description, protein_class, disease) if part]
    if not parts:
        parts.append(f"Human Protein Atlas entry for {subject.identifier}")
    return _hit_record(
        " ".join(parts),
        disease_linked=bool(disease.strip()),
        druggable=_druggable(f"{protein_class} {description}"),
        records=len(matched),
    )


def _pride(subject: Subject, read: ReadJson) -> SourceHit:
    titles = _pride_titles(read(_pride_url(subject)))
    if not titles:
        return _hit_none()
    return _hit_record("PRIDE projects: " + "; ".join(titles) + ".", records=len(titles))


def _peptideatlas(subject: Subject, read: ReadJson) -> SourceHit:
    """Protein-level PeptideAtlas evidence. Do not call the peptide-sequence mapper."""
    if _UNIPROT.fullmatch(subject.identifier) is None:
        # Keyed by accession, and ``research_subjects`` produces gene-symbol
        # subjects on purpose, so this is a question that cannot be put rather
        # than one that came back empty.
        return _hit_not_asked("PeptideAtlas is keyed by UniProt accession, and this subject is a gene symbol.")
    payload = read(_peptideatlas_url(subject))
    if not _mentions_peptideatlas(payload):
        return _hit_none()
    return _hit_record("PeptideAtlas lists proteomics evidence for this protein.")


def _uniprot_gene(entry: dict[str, object]) -> str:
    """The first gene symbol on a UniProt entry, or empty when it names none."""
    genes = entry.get("genes")
    if not isinstance(genes, list):
        return ""
    for row in genes:
        if not isinstance(row, dict):
            continue
        name = row.get("geneName")
        value = name.get("value") if isinstance(name, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def ttd_target_names(payload: object) -> dict[str, str]:
    """Gene symbol to target name, from TTD's target list.

    An entry reads ``Epidermal growth factor receptor (EGFR)``. A few list more
    than one symbol, so each is indexed separately.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return {}
    found: dict[str, str] = {}
    for row in results:
        match = _TTD_ENTRY.match(str(row))
        if match is None:
            continue
        name = match.group("name").strip()
        for part in match.group("symbol").replace("/", ",").split(","):
            symbol = part.strip().casefold()
            if symbol and symbol not in found:
                found[symbol] = name
    return found


def _ttd_targets(read: ReadJson, cache: dict[str, object]) -> dict[str, str] | None:
    """TTD's target list, fetched once per run. None means it could not be read."""
    if "ttd" not in cache:
        try:
            cache["ttd"] = ttd_target_names(read(TTD_TARGETS_URL))
        except (OSError, ValueError, LookupMissError):
            cache["ttd"] = None
    listed = cache["ttd"]
    return listed if isinstance(listed, dict) else None


def _lookup_ttd(subject: Subject, read: ReadJson, symbol: str, cache: dict[str, object]) -> SourceHit:
    """Whether TTD lists this target, by gene symbol."""
    # _ttd_targets already absorbs a failed read. Nothing broader is caught here, so a
    # test stub that refuses an unexpected URL still reports it rather than being hidden.
    listed = _ttd_targets(read, cache)
    if listed is None:
        return _hit_down(f"Therapeutic Target Database could not be read. See {TTD_URL}.")
    for candidate in (symbol, subject.identifier):
        name = listed.get(str(candidate).strip().casefold())
        if name:
            return _hit_record(f"Therapeutic Target Database lists {name} ({candidate}).")
    return _hit_none()


_FINDERS = {
    "uniprot": _uniprot,
    "ncbi": _ncbi,
    "string": _string,
    "intact": _intact,
    "human_protein_atlas": _hpa,
    "pride": _pride,
    "peptideatlas": _peptideatlas,
    # "ttd" is not here: it is keyed by gene symbol, so _lookup_sources asks it last.
}


def _prompt(
    disease: Disease,
    subject: Subject,
    sources: dict[str, dict[str, str]],
    listed: list[StructureOption],
) -> str:
    lines = [
        f"Disease: {disease.name} ({disease.disease_id})",
        f"Target: {subject.kind} {subject.identifier} {subject.label}",
        "Database records:",
    ]
    for name in SOURCE_ORDER:
        note = sources[name]
        detail = note["detail"]
        if detail:
            lines.append(f"- {SOURCE_LABELS[name]}: {note['status']}. {detail}")
        else:
            lines.append(f"- {SOURCE_LABELS[name]}: {note['status']}.")
    lines.append("Deposited atomic structures:")
    if not listed:
        lines.append("- none")
    for row in listed:
        kind = "predicted" if row.predicted or not row.methodology_known else "experimental"
        lines.append(f"- PDB {row.entry_id} chain {row.chain} {row.accession} {row.title} {kind}")
    lines.append(
        "Propose unique new research, methods, and ideas that these records support. "
        "Name the database behind each idea. "
        "Choose the deposited structures above that should be the generation inputs. "
        "Reply with the JSON object only."
    )
    return "\n".join(lines)


#: Sampling NVIDIA publishes for its reasoning ids: its own example for
#: ``deepseek-ai/deepseek-v4.1-flash`` sends temperature 1 with top_p 0.95.
#: Sampling belongs to the model rather than to the host, so the same id proxied
#: through OpenRouter is sent the same settings.
REASONING_SAMPLING: dict[str, float] = {"temperature": 1.0, "top_p": 0.95}
#: Model families that answer only after a reasoning trace.
REASONING_FAMILIES = ("deepseek", "nemotron", "reason", "think")
#: Output allowance for a reasoning id, the figure NVIDIA's Nemotron example
#: sends. The trace is paid for out of the same allowance as the answer, so the
#: budget that suits a chat model is spent thinking and the answer comes back
#: empty.
REASONING_MAX_OUTPUT = 16384
#: Output allowance for a plain chat id.
CHAT_MAX_OUTPUT = 1200
#: NVIDIA ids documented with a chat-template reasoning switch. Its Nemotron
#: example passes ``chat_template_kwargs`` through the OpenAI client's
#: ``extra_body``, which puts it at the top level of the JSON. Its DeepSeek
#: example sends no switch at all, so the field is kept to the ids that document
#: it: an unknown field is how a working call breaks.
THINKING_FAMILIES = ("nemotron",)
#: Whether that switch is on by default. These prompts ask for one small JSON
#: object, so a reasoning trace is cost with little to show for it, and spending
#: the allowance on a trace is how an answer fails to arrive at all.
#: ``RFD4_CHAT_THINKING=on`` turns it back on.
THINKING_DEFAULT = False
#: A fixed seed, as NVIDIA's example sends, so a repeated search sends the same
#: request and a recorded run can be reproduced. ``RFD4_CHAT_SEED`` overrides it.
CHAT_SEED = 42


def is_reasoning_model(model: str) -> bool:
    """Whether this id answers only after a reasoning trace."""
    lowered = base_model(model).lower()
    return any(family in lowered for family in REASONING_FAMILIES)


def reasoning_switch(provider: str, model: str) -> dict[str, object]:
    """The request field that turns this id's reasoning trace on or off.

    Each host spells it differently. NVIDIA takes a chat-template switch for its
    Nemotron ids. OpenRouter takes its own ``reasoning`` field, and the build it
    serves of the same model is a separate id, so the two never substitute for
    each other. A host with nothing documented is sent nothing.
    """
    wanted = thinking_enabled()
    if provider == "nvidia" and any(family in base_model(model).lower() for family in THINKING_FAMILIES):
        return {"chat_template_kwargs": {"enable_thinking": wanted}}
    if provider == "openrouter":
        return {"reasoning": {"enabled": wanted}}
    return {}


def thinking_enabled() -> bool:
    """Whether to ask for a reasoning trace. ``RFD4_CHAT_THINKING`` decides."""
    raw = os.environ.get("RFD4_CHAT_THINKING", "").strip().casefold()
    if raw in {"on", "1", "true", "yes"}:
        return True
    if raw in {"off", "0", "false", "no"}:
        return False
    return THINKING_DEFAULT


def chat_seed() -> int:
    """The seed to send. ``RFD4_CHAT_SEED`` overrides the default."""
    raw = os.environ.get("RFD4_CHAT_SEED", "").strip()
    try:
        return int(raw)
    except ValueError:
        return CHAT_SEED


def _request_completion(
    provider: str,
    key: str,
    system: str,
    user: str,
    chat_models: dict[str, str] | None = None,
) -> str:
    model = resolve_chat_model(provider, chat_models)
    if provider == "claude":
        payload = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model": model,
                "max_tokens": CHAT_MAX_OUTPUT,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            key,
        )
        return _chat_text(payload, provider)
    endpoint = _CHAT_ENDPOINTS.get(provider)
    if endpoint is None:
        known = ", ".join(PROVIDERS)
        raise ValueError(f"Unknown research provider {provider!r}. Expected one of {known}.")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {key}",
    }
    if provider == "openrouter":
        headers["X-Title"] = "Humanitys-Last-Disease"
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if is_reasoning_model(model):
        body["max_tokens"] = REASONING_MAX_OUTPUT
        body.update(REASONING_SAMPLING)
        body["seed"] = chat_seed()
        body.update(reasoning_switch(provider, model))
    else:
        body["max_tokens"] = CHAT_MAX_OUTPUT
    return _post_chat(endpoint[0], body, headers, key, provider)


def chat_timeout() -> float:
    """Seconds to wait for a provider. ``RFD4_CHAT_TIMEOUT`` overrides the default.

    Sixty seconds is not enough for a reasoning model: an NVIDIA call timed out at
    that limit on a three-structure prompt, which the run then reported as the model
    finding nothing.
    """
    raw = os.environ.get("RFD4_CHAT_TIMEOUT", "").strip()
    try:
        chosen = float(raw)
    except ValueError:
        return CHAT_TIMEOUT
    return chosen if chosen > 0 else CHAT_TIMEOUT


def _post_json(url: str, payload: dict[str, object], headers: dict[str, str], secret: str) -> dict[str, object]:
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=chat_timeout()) as response:
            raw = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"HTTP {exc.code} from {url}: {_redact(body, secret)}") from None
    except (HTTPException, OSError) as exc:
        raise ValueError(f"Could not reach {url}: {_redact(str(exc), secret)}") from None
    return _decode_json(raw, url)


def _decode_json(raw: bytes, url: str) -> dict[str, object]:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON from {url}: {exc}") from None
    if not isinstance(body, dict):
        raise ValueError(f"Expected a JSON object from {url}.")
    return body


def _response_lines(response: object) -> Iterator[bytes]:
    """The response body line by line, read as the lines arrive where that is possible."""
    readline = getattr(response, "readline", None)
    if callable(readline):
        while True:
            line = readline()
            if not line:
                return
            yield line
    raw = response.read()
    if isinstance(raw, bytes):
        yield from raw.splitlines()


def _stream_piece(chunk: dict[str, object]) -> tuple[str, str]:
    """One delta's answer text and finish reason.

    Only ``content`` is taken. A reasoning model sends its trace separately as
    ``reasoning_content``, and folding that into the answer would break the JSON
    the caller then parses.
    """
    choices = chunk.get("choices")
    if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
        return "", ""
    first = choices[0]
    delta = first.get("delta")
    piece = ""
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            piece = content
    reason = first.get("finish_reason")
    return piece, reason if isinstance(reason, str) else ""


def _post_chat(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    secret: str,
    provider: str,
) -> str:
    """Send one chat request and return the answer text.

    The request asks for a stream. A reasoning model can think for minutes
    before its first word, and waiting out that whole answer in a single read is
    what timed an NVIDIA call out at both 60 and 180 seconds; a stream keeps
    bytes arriving, so each read starts its own timeout rather than one read
    covering the entire answer. A service that ignores ``stream`` and replies
    with a single JSON object is still read correctly.
    """
    sent = dict(payload)
    sent["stream"] = True
    request = Request(url, data=json.dumps(sent).encode("utf-8"), headers=headers, method="POST")
    pieces: list[str] = []
    plain = bytearray()
    streamed = False
    stopped = ""
    try:
        with urlopen(request, timeout=chat_timeout()) as response:
            for raw_line in _response_lines(response):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    if line:
                        plain.extend(raw_line)
                    continue
                streamed = True
                event = line[len("data:") :].strip()
                if not event or event == "[DONE]":
                    continue
                try:
                    chunk = json.loads(event)
                except json.JSONDecodeError:
                    continue
                piece, reason = _stream_piece(chunk)
                if piece:
                    pieces.append(piece)
                if reason:
                    stopped = reason
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"HTTP {exc.code} from {url}: {_redact(body, secret)}") from None
    except (HTTPException, OSError) as exc:
        raise ValueError(f"Could not reach {url}: {_redact(str(exc), secret)}") from None
    if not streamed:
        return _chat_text(_decode_json(bytes(plain), url), provider)
    text = "".join(pieces).strip()
    if text:
        return text
    if stopped == "length":
        raise ValueError(_spent_budget(provider, payload))
    raise ValueError(f"{provider} returned no text.")


def _spent_budget(provider: str, payload: dict[str, object]) -> str:
    allowance = payload.get("max_tokens")
    return (
        f"{provider} spent its whole {allowance}-token output allowance before answering. "
        "Raise RFD4_CHAT_TIMEOUT and the allowance, or choose a model that does not reason first."
    )


def _chat_text(payload: dict[str, object], provider: str) -> str:
    if provider == "claude":
        blocks = payload.get("content")
        if isinstance(blocks, list):
            parts = [str(block.get("text") or "") for block in blocks if isinstance(block, dict)]
            text = "\n".join(part for part in parts if part).strip()
            if text:
                return text
        raise ValueError("Claude returned no text.")
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first = choices[0]
        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.strip():
            return content.strip()
        if first.get("finish_reason") == "length":
            raise ValueError(
                f"{provider} spent its whole output allowance before answering. "
                "Choose a model that does not reason first, or raise the allowance."
            )
    raise ValueError(f"{provider} returned no text.")


def _uniprot_url(subject: Subject) -> str:
    if subject.kind == "protein":
        query = f"accession:{subject.identifier}"
    else:
        query = f"gene_exact:{subject.identifier} AND organism_id:9606"
    fields = "accession,protein_name,gene_names,cc_function,cc_disease,keyword"
    return "https://rest.uniprot.org/uniprotkb/search?query=" + quote(query) + "&format=json&size=1&fields=" + fields


def _ncbi_search_url(subject: Subject) -> str:
    if subject.kind == "protein":
        term = f"{subject.identifier}[Accession] AND Homo sapiens[Organism]"
    else:
        term = f"{subject.identifier}[sym] AND Homo sapiens[Organism]"
    return (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        "?db=gene&retmode=json&retmax=1&tool=Humanitys-Last-Disease&term=" + quote(term)
    )


def _ncbi_summary_url(gene_id: str) -> str:
    return (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=gene&retmode=json&tool=Humanitys-Last-Disease&id={gene_id}"
    )


def _string_url(subject: Subject) -> str:
    return (
        "https://string-db.org/api/json/interaction_partners?species=9606"
        f"&limit={STRING_PARTNER_LIMIT}&identifiers=" + quote(subject.identifier)
    )


def _intact_url(subject: Subject) -> str:
    return (
        "https://www.ebi.ac.uk/intact/ws/interaction/findInteractions/"
        + quote(subject.identifier, safe="")
        + "?page=0&pageSize=1"
    )


def _hpa_url(subject: Subject) -> str:
    return (
        "https://www.proteinatlas.org/api/search_download.php?format=json&compress=no"
        "&columns=g,gs,gd,up,pc,di&search=" + quote(subject.identifier)
    )


def _pride_url(subject: Subject) -> str:
    return "https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects?pageSize=3&page=0&keyword=" + quote(
        subject.identifier
    )


def _peptideatlas_url(subject: Subject) -> str:
    return "https://www.ebi.ac.uk/proteins/api/proteomics/" + quote(subject.identifier, safe="")


def _ncbi_ids(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    result = payload.get("esearchresult")
    if not isinstance(result, dict):
        return []
    idlist = result.get("idlist")
    if not isinstance(idlist, list):
        return []
    return [str(item) for item in idlist if str(item).isdigit()]


def _ncbi_summary_text(payload: object, gene_id: str) -> str:
    if not isinstance(payload, dict):
        return ""
    result = payload.get("result")
    if not isinstance(result, dict):
        return ""
    row = result.get(gene_id)
    if not isinstance(row, dict):
        return ""
    parts = [str(row.get("name") or ""), str(row.get("description") or ""), str(row.get("summary") or "")]
    return " ".join(part for part in parts if part)


def _interaction_count(payload: object) -> int:
    if isinstance(payload, bool):
        return 0
    if isinstance(payload, int):
        return payload
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        total = payload.get("totalElements")
        if isinstance(total, int) and not isinstance(total, bool):
            return total
        content = payload.get("content")
        if isinstance(content, list):
            return len(content)
    return 0


def _object_rows(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("results", "hits", "content"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _hpa_match(row: dict[str, object], subject: Subject) -> bool:
    symbol = _hpa_field(row, "gs", "Gene", "Gene synonym").casefold()
    uniprot = _hpa_field(row, "up", "Uniprot", "UniProt")
    if subject.kind == "protein":
        tokens = uniprot.replace(",", " ").replace(";", " ").split()
        return subject.identifier.split("-", 1)[0] in tokens
    return symbol == subject.identifier.casefold()


def _hpa_field(row: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _pride_titles(payload: object) -> list[str]:
    projects: list[object] = []
    if isinstance(payload, dict):
        embedded = payload.get("_embedded")
        if isinstance(embedded, dict) and isinstance(embedded.get("projects"), list):
            projects = list(embedded["projects"])
        elif isinstance(payload.get("compactprojects"), list):
            projects = list(payload["compactprojects"])
        elif isinstance(payload.get("projects"), list):
            projects = list(payload["projects"])
    elif isinstance(payload, list):
        projects = list(payload)
    titles: list[str] = []
    for row in projects:
        if not isinstance(row, dict):
            continue
        title = row.get("title") or row.get("accession")
        if title:
            titles.append(str(title))
    return titles[:3]


def _mentions_peptideatlas(payload: object) -> bool:
    if isinstance(payload, dict):
        return any(_mentions_peptideatlas(key) or _mentions_peptideatlas(value) for key, value in payload.items())
    if isinstance(payload, list):
        return any(_mentions_peptideatlas(item) for item in payload)
    if isinstance(payload, str) and _SEQUENCE.search(payload) is None:
        return "peptideatlas" in payload.casefold()
    return False


def _dig(payload: dict[str, object], path: tuple[str, ...]) -> str:
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return ""
        current = current[key]
    return str(current or "")


def _record_labels(sources: object) -> list[str]:
    if not isinstance(sources, dict):
        return []
    labels: list[str] = []
    for name in SOURCE_ORDER:
        note = sources.get(name)
        if isinstance(note, dict) and note.get("status") == "record":
            labels.append(SOURCE_LABELS[name])
    return labels


def _hit_record(
    detail: str,
    *,
    partners: int = 0,
    disease_linked: bool = False,
    druggable: bool = False,
    records: int = 1,
    gene: str = "",
) -> SourceHit:
    return SourceHit(
        "record",
        _public_text(detail),
        partners=max(0, partners),
        disease_linked=disease_linked,
        druggable=druggable,
        records=max(1, records),
        gene=gene.strip(),
    )


def _druggable(protein_class: str) -> bool:
    """True when the protein class is one binders are normally made against."""
    text = protein_class.casefold()
    return any(family in text for family in _DRUGGABLE_CLASSES)


def _hit_none() -> SourceHit:
    return SourceHit("no record", "")


def _hit_down(detail: str) -> SourceHit:
    return SourceHit("unavailable", _public_text(detail))


def _hit_not_asked(detail: str) -> SourceHit:
    """A database this subject cannot be looked up in, so it was never asked.

    Distinct from ``no record``, which says the database WAS asked and holds
    nothing about the target, and from ``unavailable``, which says the read
    failed. Reporting "no record" for a question that was never put is the same
    unearned claim the evidence chain removes elsewhere, and it is worse here
    than it looks: ``cli`` deliberately remembers a definite answer -- "no
    record" among them -- so the claim stood for the rest of the run folder and
    re-asking could not correct it.

    Nothing counts this as evidence: every consumer scores and reports on
    ``status == "record"``, so this is a fourth word for the reader and not a
    new branch for the callers.
    """
    return SourceHit("not asked", _public_text(detail))


def _public_text(text: str, limit: int = 400) -> str:
    cleaned = " ".join(_SEQUENCE.sub("[sequence omitted]", text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _usable_identifier(text: str) -> str:
    cleaned = text.strip()
    if not cleaned or len(cleaned) > 40 or any(char.isspace() for char in cleaned):
        return ""
    return cleaned


def model_choices(provider: str) -> tuple[tuple[str, str], ...]:
    """The landing-page model list for one provider. Each item is an id and a label."""
    choices = MODEL_CHOICES.get(provider)
    if choices is None:
        known = ", ".join(PROVIDERS)
        raise ValueError(f"Unknown research provider {provider!r}. Expected one of {known}.")
    return choices


def panel_model(provider: str, stored: dict[str, str] | None = None) -> str:
    """The id the landing-page list should show for one provider."""
    allowed = [model_id for model_id, _label in model_choices(provider)]
    saved = (stored or {}).get(provider, "").strip()
    if saved in allowed:
        return saved
    env_name, _default = _model_spec(provider)
    env = os.environ.get(env_name, "").strip()
    if env in allowed:
        return env
    return allowed[0]


def resolve_chat_model(provider: str, chat_models: dict[str, str] | None = None) -> str:
    """The model id to send.

    A landing-page choice wins when it is one of the listed ids, with or without a
    route suffix. Otherwise the provider's environment variable is used, including
    a custom id, then a saved choice, then the first model the provider lists. The
    built-in default is a last resort for a provider that lists nothing: sending a
    model the panel does not offer is how a fresh install ends up asking for a
    retired id.
    """
    allowed = {model_id for model_id, _label in model_choices(provider)}
    explicit = (chat_models or {}).get(provider, "").strip()
    if explicit:
        if not is_offered_model(provider, explicit):
            known = ", ".join(model_id for model_id, _label in model_choices(provider))
            raise ValueError(f"Unknown {provider} model {explicit!r}. Expected one of {known}.")
        return explicit
    env_name, default = _model_spec(provider)
    env = os.environ.get(env_name, "").strip()
    if env:
        return env
    saved = load_model_choices().get(provider, "").strip()
    if saved in allowed:
        return saved
    listed = [model_id for model_id, _label in model_choices(provider)]
    return listed[0] if listed else default


#: OpenRouter publishes a routing tier as a suffix on the model id, as in
#: ``nvidia/nemotron-3-ultra-550b-a55b:free``. The suffix picks how the request
#: is routed, not a different model, so a listed id carrying one is still that
#: model and needs no entry of its own in the offered list.
ROUTE_SUFFIXES = ("free", "nitro", "floor", "online", "extended", "beta", "thinking")


def base_model(model: str) -> str:
    """The model id without its routing suffix."""
    head, _found, tail = model.partition(":")
    return head if tail and tail.casefold() in ROUTE_SUFFIXES else model


def is_offered_model(provider: str, model: str) -> bool:
    """Whether this id is one the panel lists, with or without a routing suffix."""
    allowed = {model_id for model_id, _label in model_choices(provider)}
    return model in allowed or base_model(model) in allowed


def _model_spec(provider: str) -> tuple[str, str]:
    """Environment variable and built-in default for one provider."""
    if provider == "claude":
        return ("RFD4_CLAUDE_MODEL", "claude-sonnet-4-5")
    endpoint = _CHAT_ENDPOINTS.get(provider)
    if endpoint is None:
        known = ", ".join(PROVIDERS)
        raise ValueError(f"Unknown research provider {provider!r}. Expected one of {known}.")
    return endpoint[1], endpoint[2]


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text
