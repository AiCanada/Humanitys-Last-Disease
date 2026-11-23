"""Browser panel for every human option.

Starting ``human`` with no arguments opens this panel. ``--ui`` opens it
with any flags you already passed filled in. Dropdowns set the fixed choices.
Tick boxes set the yes/no choices and the structures to download. Flags
without ``--ui`` skip the panel.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from email import message_from_bytes
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from rfproteina.disease.archive import generation_archive_root
from rfproteina.disease.design_spec import cif_text
from rfproteina.disease.dispatch import is_local_host
from rfproteina.disease.dynamics import ANALYST_LABEL
from rfproteina.disease.dynamics import TASK_MODES as DYNAMICS_TASK_MODES
from rfproteina.disease.dynamics import TASK_PURPOSE as DYNAMICS_TASK_PURPOSE
from rfproteina.disease.focus import (
    DISTINCTION,
    MECHANISMS,
    BiologicalFocus,
    display_rank,
    focus_from_fields,
    focus_is_active,
    focus_summary,
    rank_score,
    row_role,
)
from rfproteina.disease.grok_bot import GROK_BOT, GROK_BOT_LABEL, GROK_BOT_MODEL
from rfproteina.disease.human_protocol import PROTOCOL_LABEL, PROTOCOL_LIMIT
from rfproteina.disease.human_protocol import SEQUENCE_RESEMBLANCE as HUMAN_SEQUENCE_ROUTE
from rfproteina.disease.key_store import (
    KeySlot,
    load_key_slots,
    load_model_choices,
    save_key_slots,
    save_model_choices,
)
from rfproteina.disease.logs import log, redact_fields
from rfproteina.disease.lookup import Disease, related_names
from rfproteina.disease.machines import LOCAL_NAMES, discover_remote_machines, pick_remote_machine
from rfproteina.disease.research import (
    PROVIDER_LABELS,
    PROVIDERS,
    SOURCE_LABELS,
    SOURCE_ORDER,
    RecordScore,
    is_offered_model,
    model_choices,
    model_display,
    panel_model,
    providers_with_keys,
    result_models,
    search_plan,
)
from rfproteina.disease.results_actions import COMBINED_FASTA as RESULTS_COMBINED_FASTA
from rfproteina.disease.results_actions import (
    DYNAMICS_CHAIN,
    DYNAMICS_DRIFT_MAX,
    DYNAMICS_FRAMES,
    DYNAMICS_MOBILE_MAX,
    DYNAMICS_MOVIE,
    DYNAMICS_REPLICATES,
    DYNAMICS_RMSF_MAX,
    DYNAMICS_TASK,
    HUMAN_MODEL,
    HUMAN_MODEL_OF,
    HUMAN_NAMES,
    HUMAN_PICK,
    HUMAN_TARGET,
)
from rfproteina.disease.results_actions import DYNAMICS_DRAW as RESULTS_DYNAMICS_DRAW
from rfproteina.disease.results_actions import DYNAMICS_RUN as RESULTS_DYNAMICS_RUN
from rfproteina.disease.results_actions import HUMAN_COMPARE as RESULTS_HUMAN_COMPARE
from rfproteina.disease.results_actions import HUMAN_PROPOSE as RESULTS_HUMAN_PROPOSE
from rfproteina.disease.results_actions import OUTPUT_FASTA as RESULTS_FASTA
from rfproteina.disease.results_actions import PREPARE_AF3 as RESULTS_AF3
from rfproteina.disease.results_actions import PREPARE_AF3C as RESULTS_AF3C_PREPARE
from rfproteina.disease.results_actions import RETRIEVE_AF2 as RESULTS_AF2
from rfproteina.disease.results_actions import RETRIEVE_AF2C as RESULTS_AF2C
from rfproteina.disease.results_actions import RETRIEVE_AF3 as RESULTS_AF3_IMPORT
from rfproteina.disease.results_actions import RETRIEVE_AF3C as RESULTS_AF3C
from rfproteina.disease.screen import SCREEN_LISTING_PREFIX, screen_reports_for_directory
from rfproteina.disease.screen_board import render_closing
from rfproteina.disease.screen_selection import Choice as ScreenChoice
from rfproteina.disease.shards import GENERATION_MODELS
from rfproteina.disease.structure_view import (
    StructureTrio,
    catalog_entry,
    render_structure_views,
    viewer_javascript,
)
from rfproteina.disease.structures import (
    StructureOption,
    apply_structure_view,
    campaign_counts,
    campaign_pick_numbers,
    score_parts,
    structure_campaigns,
    structure_kind,
)
from rfproteina.disease.validation_checks import (
    ValidationCandidate,
    ValidationChoice,
    candidate_validation_plan,
    run_validation_checks,
)

Exchange = Callable[[str], dict[str, list[str]]]
#: Where :py:func:`with_nav` puts the Back and Forward buttons. The renderers do not
#: know which moves are available, so the panel fills this in when it shows the page.
NAV_SLOT = "<!--rfd4-nav-->"
#: Suffixes the custom target upload accepts, lower-cased.
UPLOAD_SUFFIXES = (".pdb", ".cif")
#: What the choose-structures page was asked to do.
STRUCTURE_ACTIONS = ("download", "missing", "rank", "ask")
#: The ``results-tab`` value for the protocol's tab. The other two are their
#: own names, ``binder`` and ``combined``, and this one is read the same way.
HUMAN_TAB = "human"
#: The Conformational Dynamics Analyst's tab. It sits last because it reads a
#: finished design: generation, then screening, then this.
DYNAMICS_TAB = "dynamics"
#: Which tabs the results page has, so a stale or hand-typed value falls back
#: to the first rather than opening nothing at all.
RESULTS_TABS = ("binder", "combined", HUMAN_TAB, DYNAMICS_TAB)
VALIDATION_ACTION = "results-validate-output"
VALIDATION_FIELD = "validation_check"
VALIDATION_SCOPE_ACTION = "results-validation-scope"
VALIDATION_SCOPE_FIELD = "validation_scope"
VALIDATION_CANDIDATE_FIELD = "validation_candidate"
SCREENING_ACTION = "results-screen-methods"
SCREENING_FIELD = "screen_method"
#: The sentence above the analyst's buttons. The model takes one polypeptide
#: chain -- its own stated limitation -- so the thing a reader is most likely
#: to want from it is the thing it cannot do, and that is said before the run
#: rather than reported after it.
DYNAMICS_SINGLE_CHAIN = (
    "One chain at a time. The generator takes a single polypeptide chain, so this says how "
    "the design itself moves, not whether an interface holds -- a design-plus-target complex "
    "is not an input it has. It is also trained on natural proteins, which a de novo design "
    "is not, so treat the numbers as the model's opinion rather than a measurement."
)
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
#: A structure address is 16 hex characters from token_hex(8). Anything else is refused.
_STRUCTURE_TOKEN = re.compile(r"^[0-9a-f]{16}$")
#: Panel passes allowed before the run gives up. Each one needs a real submission from
#: the browser, so this only bounds a client that goes back without ever going forward.
MAX_PANEL_PASSES = 500
#: Bytes of one submission the panel will read. Generous, because the largest
#: legitimate one is a structure upload and ``results_retrieve.TEXT_LIMIT``
#: already accepts a 64 MiB model, and a bound all the same: the handler read
#: whatever ``Content-Length`` claimed, so a client claiming four gigabytes had
#: the panel try to allocate four gigabytes. The server listens on loopback,
#: which makes this a local process or a cross-origin form post rather than the
#: open internet -- a smaller exposure, not a different one.
POST_BYTE_LIMIT = 128 * 1024 * 1024


def _body_length(header: str) -> int | None:
    """The submission's length, or None when it is not one this will read.

    None covers both faults a ``Content-Length`` can have, because the caller
    does the same thing with each: a value that is not a non-negative integer,
    and one past :data:`POST_BYTE_LIMIT`.
    """
    text = (header or "").strip()
    if not text.isdigit():
        return None
    length = int(text)
    return length if length <= POST_BYTE_LIMIT else None


class StepBackError(Exception):
    """The panel was asked to go back.

    The run restarts and replays the answers before ``target``, so the user lands
    on that page again rather than at the beginning.
    """

    def __init__(self, target: int) -> None:
        super().__init__(f"Go back to panel step {target}.")
        self.target = target


class ReturnToStartError(Exception):
    """The panel was asked to show the first page again, keeping this run's answers."""


class RestartFreshError(Exception):
    """The panel was asked to drop this run and show the first page empty."""


BINDER_KINDS = ("protein", "ligand", "peptide", "nucleic")
#: Shown at the top of every page. The three spaces are intentional.
PANEL_TITLE = "PROJECT H.U.M.A.N   V0.8888\nHigh Utility Molecular Affinity Nexus"
#: Shown centred on the title row of every page, beside PANEL_TITLE. Display
#: text, so it keeps its apostrophe -- unlike the machine-facing name, which
#: drops it because an apostrophe breaks a single-quoted shell path.
PROJECT_NAME = "Humanity's Last Disease"
_WORKER_PRESETS = ("1", "2", "4", "8")
#: What a saved key looks like on the page. The field never receives the secret.
_KEY_MASK = "*****"
_PAGE_STYLE = """
/* The browser hides [hidden] through its own stylesheet, which any author rule beats.
   The structure rows are display:flex and the LLM rows display:grid, so without this
   the Include/Exclude filters set the attribute and nothing disappears. */
[hidden] { display: none !important; }
/* Each result already carries the ID# its own tick uses. An ordered-list marker
   beside it is a second number that counts something else. */
ol.llm-results { list-style: none; margin: 6px 0 0; padding: 0; }
ol.llm-results > li { padding: 5px 0; border-bottom: 1px solid #d3dde6; }
ol.llm-results > li:last-child { border-bottom: 0; }
body { margin: 0; background: #e7edf2; color: #17202a;
  font: 15px/1.35 "Segoe UI", system-ui, sans-serif; }
main { max-width: 1140px; margin: 18px auto; padding: 0 14px 28px; }
h1 { margin: 0 0 12px; font-size: 14px; }
.notice { margin: 0 0 12px; padding: 8px 10px; border-radius: 8px;
  background: #fdecea; color: #8a1f11; font-weight: 700; }
/* Not .notice: the disease page scrolls to that class and focuses the disease box. */
/* Results tabs. Radios and sibling selectors, the way the screen board does
   it, so the page keeps working with no JavaScript. The inputs post a
   `results-tab` field, which is how the active tab survives a click. */
.results-tabbed > input[type=radio] { position: absolute; opacity: 0; pointer-events: none; }
.results-tab { display: inline-block; padding: 7px 14px; margin: 0 4px 10px 0; cursor: pointer;
  border: 1px solid #c8ccd4; border-radius: 8px 8px 0 0; background: #f2f4f8; font-weight: 600; }
.results-tabbed > input:checked + label.results-tab { background: #1f4fd8; color: #fff;
  border-color: #1f4fd8; }
.results-panel { display: none; border-top: 2px solid #1f4fd8; padding-top: 12px; }
#results-tab-binder:checked ~ .results-panel-binder { display: block; }
#results-tab-combined:checked ~ .results-panel-combined { display: block; }
#results-tab-human:checked ~ .results-panel-human { display: block; }
#results-tab-dynamics:checked ~ .results-panel-dynamics { display: block; }
/* The protocol's two lists. Each candidate is one row: a tick box, what the
   protein is, and the resemblance that put it on the list. */
ul.protocol-picks { list-style: none; margin: 6px 0 10px; padding: 0; }
ul.protocol-picks > li { padding: 4px 0; border-bottom: 1px solid #d3dde6; }
ul.protocol-picks > li:last-child { border-bottom: 0; }
.protocol-why { color: #41525f; font-size: 0.86rem; }
table.protocol-readout { border-collapse: collapse; width: 100%; margin: 8px 0; }
table.protocol-readout th, table.protocol-readout td { border-bottom: 1px solid #d6dae2;
  padding: 4px 6px; text-align: left; font-size: 0.9rem; vertical-align: top; }
table.protocol-readout td.state-hit { color: #8f2318; font-weight: 700; }
table.protocol-readout td.state-clear { color: #0b5c40; font-weight: 700; }
table.protocol-readout td.state-untested { color: #3d4a57; }
table.protocol-readout caption, .protocol-against { text-align: left; font-weight: 600;
  margin: 6px 0 0; padding: 6px 0 2px; }
.protocol-against .state-hit { color: #8f2318; }
.protocol-against .state-clear { color: #0b5c40; }
.protocol-against .state-untested { color: #3d4a57; }
table.protocol-readout caption .state-hit { color: #8f2318; }
table.protocol-readout caption .state-clear { color: #0b5c40; }
table.protocol-readout caption .state-untested { color: #3d4a57; }
/* Not .hint: this one must not be skimmable. It is the sentence that stops a
   resemblance number being read as a binding result. */
.protocol-limit { margin: 6px 0 10px; padding: 8px 10px; border-radius: 8px;
  background: #fdf3e3; color: #6b4a10; }
.results-retrieve { border: 1px solid #d6dae2; border-radius: 8px; padding: 10px 12px;
  margin: 10px 0; }
.results-retrieve legend { font-weight: 600; padding: 0 6px; }
.results-payload textarea { width: 100%; font-family: ui-monospace, Menlo, Consolas, monospace;
  font-size: 0.86rem; }
.results-save { display: inline-block; margin-top: 6px; }
.screen-readout { margin: 0 0 10px; padding: 8px 10px; border-radius: 8px;
  background: #eef5ea; color: #1d3b24; }
.screen-readout p { margin: 0 0 4px; }
.screen-readout p:last-child { margin-bottom: 0; }
.screen-readout .screen-where { font-weight: 700; }
/* One column when narrow. At 700px the two stacks sit side by side. Each stack
   sizes itself, so a tall Structures Directory cannot push Binder and model
   or Machines below it. Both stay directly under Disease. */
.grid { display: grid; gap: 12px; align-items: start; grid-template-columns: 1fr; }
.stack { display: grid; gap: 12px; align-items: start; align-content: start; }
.stack > section { margin: 0; }
.wide { grid-column: 1 / -1; }
@media (min-width: 700px) {
  .grid { grid-template-columns: 1fr 1fr; }
}
section { background: #fff; border: 1px solid #d5dee6; border-radius: 10px;
  padding: 12px 14px 4px; margin: 0 0 12px; }
h2 { margin: 0 0 8px; font-size: 12px; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; color: #3d4a57; }
/* Short fields pair up inside a panel, and drop to one per row below the breakpoint. */
.row { display: grid; gap: 0 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
/* Local workers and Remote workers each pair a dropdown with its own custom
   counter. The captions are different lengths -- "Local workers (structures to
   bind, locally)" wraps to a second line in half a panel and "Custom local
   workers" does not -- and with each label a plain block, the control under the
   taller caption sat lower than the control beside it. So the pair shares three
   row tracks, caption / control / note, the same way .llm below shares its rows,
   and the two controls stay level whatever the captions and notes do.
   Where subgrid is unsupported the track rule is dropped and `span 3` leaves
   each label one block spanning the three rows, which is the old behaviour. */
.row.workers { grid-template-rows: auto auto auto; align-items: start; }
.row.workers > label.field { display: grid; grid-row: span 3;
  grid-template-rows: subgrid; row-gap: 0; margin: 0 0 9px; }
/* The caption is a span now so it can be placed, and this restores the label
   look that the shared `label.field span` rule would otherwise give it. */
.row.workers label.field > span.caption { align-self: start; margin-top: 0;
  font-weight: 600; font-size: 13px; line-height: 1.25; color: inherit; }
.row.workers label.field > select,
.row.workers label.field > input { align-self: start; }
.llm-list { display: grid; }
/* Captions share row 1. The password, model list, and buttons share row 2.
   A saved-key note stays on row 3 under the password, so it cannot drop the model. */
.llm { display: grid; column-gap: 12px; row-gap: 2px; align-items: center;
  grid-template-columns: 1fr 1fr auto; }
.llm-caption { margin: 0; font-weight: 600; font-size: 13px; }
.llm-key-caption { grid-column: 1; grid-row: 1; }
.llm-model-caption { grid-column: 2; grid-row: 1; }
.llm > input[type=password] { grid-column: 1; grid-row: 2; margin-top: 0; }
.llm > select { grid-column: 2; grid-row: 2; margin-top: 0; }
.llm-actions { grid-column: 3; grid-row: 2; display: flex; gap: 6px;
  align-items: center; margin: 0; }
.llm-saved { grid-column: 1; grid-row: 3; margin: 0; }
label.field { display: block; margin: 0 0 9px; font-weight: 600; font-size: 13px; }
label.field span, .hint, .llm-saved { display: block; font-weight: 400; color: #52616e;
  font-size: 12px; line-height: 1.3; margin-top: 2px; }
.hint { margin: 0 0 9px; }
input[type=text], input[type=number], input[type=password], select, textarea {
  width: 100%; box-sizing: border-box; margin-top: 3px; padding: 6px 9px;
  border: 1px solid #b7c3ce; border-radius: 7px; background: #fff;
  font: inherit; color: inherit; }
input[readonly] { background: #f4f7f9; }
select { min-height: 32px; }
textarea { min-height: 46px; resize: vertical; }
/* The question box uses the native corner grip. The plainer textarea rule above
   locks resize to vertical, so this more specific rule is what frees both axes. */
.llm-ask { display: flex; flex-direction: column; align-items: stretch; gap: 6px;
  margin: 0 0 12px; }
.llm-ask textarea.llm-question { width: 100%; max-width: 100%; min-width: 0; flex: none;
  min-height: 2.75rem; height: 7.5rem; resize: both; overflow: auto; }
.llm-ask-actions { display: flex; flex-direction: row; flex-wrap: wrap; align-items: center;
  gap: 6px; }
/* The two LLM search buttons. Their labels are a sentence each, so at the
   full button size they outweighed everything else on the page. Same size as
   the Add/Submit pair directly above them. */
.llm-searches { margin: 0 0 9px; }
.llm-searches button { margin: 0 8px 4px 0; padding: 3px 8px; font-size: 12px;
  font-weight: 600; white-space: normal; text-align: left; }
.llm-ask-actions button { margin: 0; padding: 3px 8px; font-size: 12px; font-weight: 600;
  white-space: nowrap; }
.llm-ask-actions button.llm-add-structures { background: #e3e9ef; color: #26323d; }
label.prior-reply { display: flex; gap: 8px; align-items: flex-start; font-weight: 600; }
label.prior-reply input { width: 1rem; height: 1rem; margin-top: 2px; accent-color: #0b6e4f; }
.ticks { display: grid; gap: 6px; margin: 0 0 9px;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
.ticks label { display: flex; gap: 8px; align-items: flex-start;
  padding: 7px 9px; border: 1px solid #d5dee6; border-radius: 7px;
  background: #f8fafb; font-weight: 600; font-size: 13px; }
.ticks input { width: 1rem; height: 1rem; margin-top: 1px; accent-color: #0b6e4f; }
.structure-view { display: grid; gap: 8px; margin: 0 0 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
.structure-view label { font-weight: 600; font-size: 13px; }
.structure-view label span { display: block; font-weight: 400; color: #52616e;
  font-size: 12px; line-height: 1.3; margin-top: 2px; }
.machine-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  margin: 0 0 9px; }
/* Search machines and Windows File Explorer. Neither is the step's action --
   they only narrow the list below -- and at the full button size the two of them
   read as the thing to press. Same size as the LLM search pair above, so the
   page has one smaller-button size rather than a new one per section.
   Selected as button.nav, not button: both carry that class, and the plainer
   selector ties with `button.nav` below on specificity and loses on order, so
   its margin-right would survive and sit on top of this flex row's own gap. */
.machine-actions button.nav { margin: 0; padding: 3px 8px; font-size: 12px;
  font-weight: 600; white-space: nowrap; }
/* A worker count is a number to step, not a string to type, so it is sized to
   the number instead of stretching to the column. The native spinner is the
   counter: nothing here sets appearance: none, so stepping already works
   without script, and a +/- pair of our own would only be another thing that
   can disagree with the field it is meant to be driving. */
label.field.counter input[type=number] { width: 7rem; }
ul.campaign { list-style: none; padding: 0; margin: 0 0 12px; display: grid; gap: 6px; }
ul.campaign label { display: flex; gap: 8px; align-items: flex-start; cursor: pointer;
  padding: 7px 9px; border: 1px solid #d5dee6; border-radius: 7px; background: #f8fafb;
  font-weight: 600; font-size: 13px; }
ul.campaign input { width: 1rem; height: 1rem; margin-top: 1px; accent-color: #0b6e4f; }
.badge { font-weight: 400; color: #8a5a00; }
button { margin: 4px 0 0; padding: 10px 18px; border: 0; border-radius: 8px;
  background: #0b6e4f; color: #fff; font: inherit; font-weight: 700; cursor: pointer; }
button[disabled], button[aria-busy="true"] { background: #5c756c; cursor: progress; }
/* Back and Forward stay visibly secondary to the step's own action. */
button.nav { margin-right: 8px; background: #e3e9ef; color: #26323d; }
/* Every page. The action stays last in the form, and the row places Back,
   Forward, and that action beside the title. */
form.landing { display: grid; grid-template-columns: 1fr auto; align-items: center;
  column-gap: 12px; }
form.landing > h1 { grid-column: 1; grid-row: 1; margin: 0 0 12px; }
form.landing > .page-actions-top { grid-column: 2; grid-row: 1; display: flex;
  justify-content: flex-end; align-items: center; gap: 8px; margin: 0 0 12px;
  flex-wrap: wrap; }
form.landing > h1 { grid-column: 1 / -1; }
form.landing > .page-actions-top { grid-column: 1 / -1; grid-row: 2; }
form.landing > :not(h1):not(.page-actions-top) { grid-column: 1 / -1; }
/* Title row. The outer rule for a non-h1 child already spans every column,
   so this only has to lay the row out. Three columns with equal outer
   tracks put the centre item at the middle of the page rather than at the
   middle of whatever is left over beside the title. */
.title-row { display: grid; grid-template-columns: 1fr auto 1fr;
  align-items: center; column-gap: 12px; margin: 0 0 12px; }
.title-row > h1 { grid-column: 1; margin: 0; }
.title-row > .project-name { grid-column: 2; margin: 0; text-align: center;
  font-size: 20px; font-weight: 700; }
.step-title { margin: 0 0 12px; font-size: 20px; font-weight: 700; }
.page-actions-top button { margin: 0; padding: 6px 10px; font-size: 13px; white-space: nowrap; }
.file-picks { display: grid; gap: 6px; margin: 0 0 9px; }
.file-picks button.file-pick { display: block; width: 100%; margin: 0; text-align: left;
  font-size: 13px; font-weight: 600; padding: 7px 9px; white-space: normal;
  word-break: break-all; }
.file-picks button.file-pick[aria-pressed="true"] { outline: 2px solid #0b6e4f; }
button.llm-add, button.llm-remove { margin: 0; padding: 3px 8px; border-radius: 6px;
  font-size: 11px; font-weight: 600; line-height: 1.2; white-space: nowrap; }
#run-status { margin: 8px 0 12px; }
#run-status p { margin: 0 0 6px; font-weight: 700; }
progress { width: 100%; height: 16px; accent-color: #0b6e4f; }
"""
_IDLE_PROGRESS = {"active": False, "done": 0, "total": 0, "label": "", "action": ""}
#: What the bar says while structures are being written.
DOWNLOAD_ACTION = "Downloading"
#: What it says while the public databases are being asked for the program score.
RECORDS_ACTION = "Reading database records"
_PAGE_SCRIPT = """
(function () {
  var form = document.querySelector("form");
  if (!form) return;
  var validation = form.querySelector("[data-validation-checks]");
  if (validation) {
    var refreshChecks = validation.querySelector("[data-validation-refresh]");
    if (refreshChecks) {
      validation.querySelectorAll("[data-validation-scope]").forEach(function (selector) {
        selector.addEventListener("change", function () { refreshChecks.click(); });
      });
    }
  }
  form.querySelectorAll("[data-validation-checks], [data-screening-methods]").forEach(function (panel) {
    var count = panel.querySelector("[data-screening-count]");
    function updateCount() {
      if (count) count.textContent = panel.querySelectorAll("input[type=checkbox]:enabled:checked").length;
    }
    if (count) panel.addEventListener("change", updateCount);
    panel.addEventListener("click", function (event) {
      var choice = event.target.closest("button[data-validation-select]");
      if (!choice) return;
      var mode = choice.getAttribute("data-validation-select");
      panel.querySelectorAll("input[type=checkbox]:enabled").forEach(function (box) {
        box.checked = mode === "all" || (mode === "recommended" && box.hasAttribute("data-recommended"));
      });
      updateCount();
    });
  });
  // The confirm page uses the named disease while its tick is set. While the
  // tick is clear, the matching-names menu is the choice. The inference page
  // has the same tick name and no menu, so both have to be present.
  var acceptDisease = form.querySelector('input[name="accept"]');
  var diseaseChoice = form.querySelector(".disease-choice");
  function syncDiseaseChoice() {
    if (!acceptDisease || !diseaseChoice) return;
    var useNamed = acceptDisease.checked;
    diseaseChoice.hidden = useNamed;
    var menu = diseaseChoice.querySelector("select");
    if (menu) menu.disabled = useNamed;
  }
  if (acceptDisease && diseaseChoice) {
    acceptDisease.addEventListener("change", syncDiseaseChoice);
    syncDiseaseChoice();
  }
  var notice = document.querySelector(".notice");
  var diseaseBox = form.querySelector('input[name="disease"]');
  // Only the disease page scrolls to its notice. Other pages keep their own
  // notice lower down, and jumping there hides the new step.
  if (notice && diseaseBox) {
    if (notice.scrollIntoView) notice.scrollIntoView({block: "center"});
    diseaseBox.focus();
  }
  var submitting = false;
  var leaving = false;
  var shown = form.getAttribute("data-step");
  function digits(value) {
    if (!value) return false;
    for (var i = 0; i < value.length; i++) {
      var code = value.charCodeAt(i);
      if (code < 48 || code > 57) return false;
    }
    return true;
  }
  // A tab left on an earlier page follows the panel. A click follows too, once
  // the server step changes, so a missed redirect does not leave the tab behind.
  // Keep asking until this document unloads. One replace that does not complete
  // must not strand the tab on the page it already showed.
  var results = null;
  function follow(data) {
    if (leaving || !data) return;
    if (data.closed) {
      if (form.getAttribute("data-closed") === "true") return;
      leaving = true;
      window.location.replace("/closed");
      return;
    }
    if (data.results !== undefined && data.results !== null) {
      // An overlay landed while this page was open -- a download finished and
      // was imported without anyone clicking. The page holds the old view, so
      // it reloads itself rather than showing a stale "no model retrieved yet".
      if (results === null) {
        results = String(data.results);
      } else if (String(data.results) !== results) {
        leaving = true;
        window.location.reload();
        return;
      }
    }
    if (shown === null || shown === "") return;
    if (data.step === undefined || data.step === null) return;
    var next = String(data.step);
    if (!digits(next) || next === shown) return;
    window.location.replace("/step/" + next);
  }
  function paint(data) {
    var status = document.getElementById("run-status");
    var bar = document.getElementById("run-progress");
    var caption = document.getElementById("run-status-label");
    if (!status || !bar || !caption || !data || !data.active || !data.total) {
      if (status) status.hidden = true;
      return;
    }
    status.hidden = false;
    bar.max = data.total;
    bar.value = data.done;
    var text = (data.action || "Working") + " " + data.done + " of " + data.total;
    if (data.label) text += ": " + data.label;
    caption.textContent = text;
  }
  function watch() {
    fetch("/progress", {cache: "no-store"}).then(function (response) {
      return response.json();
    }).then(function (data) {
      follow(data);
      if (submitting) paint(data);
    }).catch(function () {});
  }
  var timer = window.setInterval(watch, 300);
  watch();
  // pagehide means this document is actually going away. The submit itself
  // does not: the old page stays up until the next step loads.
  window.addEventListener("pagehide", function () {
    leaving = true;
    window.clearInterval(timer);
  });
  form.addEventListener("submit", function (event) {
    if (submitting) {
      event.preventDefault();
      return;
    }
    submitting = true;
    var all = form.querySelectorAll("button[type=submit]");
    var pressed = event.submitter;
    if (!pressed || pressed.tagName !== "BUTTON") {
      pressed = all.length ? all[all.length - 1] : null;
    } else if (pressed.getAttribute("name") === "__nav" && document.activeElement !== pressed) {
      // Enter in a field activates the first submit. Return, Restart, Back, and
      // Forward are those buttons. The page action is the last one.
      pressed = all.length ? all[all.length - 1] : pressed;
    }
    for (var i = 0; i < all.length; i++) {
      // The pressed button keeps its enabled state on purpose: a disabled control is
      // left out of the form data, so disabling it would drop __nav and a Back press
      // would arrive as an ordinary submission.
      if (all[i] !== pressed) all[i].disabled = true;
    }
    if (pressed) {
      pressed.textContent = "Processing";
      pressed.setAttribute("aria-busy", "true");
    }
  });
  function placeAdd(rows) {
    var provider = rows && rows.closest(".llm-provider");
    if (!provider) return;
    var adds = provider.querySelectorAll(".llm-add");
    var add = null;
    for (var i = 0; i < adds.length; i++) {
      if (!add) add = adds[i];
      else adds[i].remove();
    }
    if (!add) return;
    var visible = rows.querySelectorAll(".llm:not([hidden])");
    var last = visible.length ? visible[visible.length - 1] : null;
    var actions = last && last.querySelector(".llm-actions");
    if (actions) actions.appendChild(add);
    else provider.appendChild(add);
  }
  document.addEventListener("click", function (event) {
    var add = event.target.closest && event.target.closest(".llm-add");
    if (!add) return;
    var provider = add.closest(".llm-provider");
    var rows = provider && provider.querySelector(".llm-rows");
    var last = rows && rows.querySelector(".llm:last-of-type");
    if (!rows || !last) return;
    var copy = last.cloneNode(true);
    copy.hidden = false;
    copy.querySelectorAll("input, select, button").forEach(function (el) { el.disabled = false; });
    var slot = copy.querySelector('input[type="hidden"]');
    if (slot) slot.value = "new";
    var password = copy.querySelector('input[type="password"]');
    if (password) password.value = "";
    var note = copy.querySelector(".llm-saved");
    if (note) note.remove();
    var copiedAdd = copy.querySelector(".llm-add");
    if (copiedAdd) copiedAdd.remove();
    rows.appendChild(copy);
    placeAdd(rows);
  });
  document.addEventListener("click", function (event) {
    var button = event.target.closest && event.target.closest(".llm-remove");
    if (!button) return;
    var row = button.closest(".llm");
    var rows = row && row.parentElement;
    if (!row || !rows) return;
    var kept = rows.querySelectorAll(".llm:not([hidden])");
    if (kept.length <= 1) {
      var slot = row.querySelector('input[type="hidden"]');
      if (!slot || slot.value === "new") return;
      var staying = row.querySelector(".llm-add");
      if (staying) rows.parentElement.appendChild(staying);
      slot.disabled = true;
      row.querySelectorAll("input, select").forEach(function (el) { el.disabled = true; });
      row.hidden = true;
      return;
    }
    var parked = row.querySelector(".llm-add");
    if (parked) rows.appendChild(parked);
    row.remove();
    placeAdd(rows);
  });
  // Model ticks live in .llm-search. Structure-row ticks stay outside it.
  var view = document.querySelector(".structure-view");
  var structureRows = document.querySelector(".structure-rows");
  if (view && structureRows) {
    // Each sort mode names the metric it reads. A row missing that metric sorts
    // last in both directions rather than counting as a zero it never earned.
    var SORT_KEYS = {
      high: "data-score",
      low: "data-score",
      "same-high": "data-instances",
      "same-low": "data-instances",
      biological: "data-rank"
    };
    function metricOf(label, attribute) {
      var raw = label.getAttribute(attribute);
      if (raw === null || raw === "") return null;
      var value = Number(raw);
      return isNaN(value) ? null : value;
    }
    // One model's number from data-rankings. The mean in data-value is not a
    // sort key: higher value, lower value, Has, and No follow the model or
    // Grok Bot chosen beside the sort.
    function rankingOf(node, model) {
      if (!model) return null;
      var raw = node.getAttribute("data-rankings") || "";
      var parts = raw.split("|");
      for (var i = 0; i < parts.length; i++) {
        var cut = parts[i].indexOf("=");
        if (cut < 0) continue;
        if (parts[i].substring(0, cut) === model) {
          var text = parts[i].substring(cut + 1);
          if (text === "") return null;
          var value = Number(text);
          return isNaN(value) ? null : value;
        }
      }
      return null;
    }
    function chosenRankModel() {
      var chosen = view.querySelector('select[name="rank_model"]');
      return chosen ? chosen.value : "";
    }
    function namedBy(label, model) {
      var raw = label.getAttribute("data-models");
      if (!raw) return false;
      var names = raw.split("|");
      for (var i = 0; i < names.length; i++) {
        if (names[i] === model) return true;
      }
      return false;
    }
    // The ranking filter reads the model's number, never the database counts.
    function hasLlmRanking(node, mode) {
      var model = chosenRankModel();
      if (mode === "ranked") {
        if (model) return rankingOf(node, model) !== null;
        return metricOf(node, "data-value") !== null;
      }
      if (mode === "unranked") {
        if (model) return rankingOf(node, model) === null;
        return metricOf(node, "data-value") === null;
      }
      if (mode.indexOf("model:") === 0) {
        var name = mode.substring(6);
        var raw = node.getAttribute("data-rankings") || "";
        var parts = raw.split("|");
        for (var i = 0; i < parts.length; i++) {
          var cut = parts[i].indexOf("=");
          if (cut < 0) continue;
          if (parts[i].substring(0, cut) === name) return parts[i].substring(cut + 1) !== "";
        }
        return false;
      }
      return false;
    }
    function applyStructureView() {
      var sort = view.querySelector('select[name="sort"]');
      var experimental = view.querySelector('select[name="experimental"]');
      var predicted = view.querySelector('select[name="predicted"]');
      var llmModel = view.querySelector('select[name="llm_model"]');
      var llmState = view.querySelector('select[name="llm_state"]');
      var llmRank = view.querySelector('select[name="llm_rank"]');
      var llmRankState = view.querySelector('select[name="llm_rank_state"]');
      var mode = sort ? sort.value : "listed";
      var showExperimental = !experimental || experimental.value !== "exclude";
      var showPredicted = !predicted || predicted.value !== "exclude";
      // "Any model" filters nothing, so the include/exclude beside it has no
      // subject and must not blank the list. LLM ranking "Any" is the same for
      // hiding rows. Has, No, and the higher and lower value sorts read the
      // model or Grok Bot in this block, never the mean and never the database
      // instance counts.
      var wantedModel = llmModel && llmModel.value !== "any" ? llmModel.value : "";
      var keepNamed = !llmState || llmState.value !== "exclude";
      var rankMode = llmRank && llmRank.value !== "any" ? llmRank.value : "";
      var keepRanked = !llmRankState || llmRankState.value !== "exclude";
      var allBox = everyRowBox();
      // All Structure is a server-side "every experimental row" flag, so it cannot
      // stand while the rows on show are narrower than that. A ticked class always
      // contradicted it; a model or ranking filter did too, and left it able to
      // fetch rows this page was hiding.
      var narrowed = checkedCampaigns().length > 0 || wantedModel !== "" || rankMode !== "";
      if (allBox) {
        allBox.disabled = !showExperimental || narrowed;
        if (allBox.disabled) allBox.checked = false;
      }
      var labels = Array.prototype.slice.call(structureRows.querySelectorAll("label[data-kind]"));
      var sortKey = SORT_KEYS[mode];
      var byValue = mode === "value-high" || mode === "value-low";
      var rankModel = chosenRankModel();
      var descending = mode.indexOf("-low") === -1 && mode !== "low";
      labels.sort(function (left, right) {
        var leftNumber = Number(left.getAttribute("data-number"));
        var rightNumber = Number(right.getAttribute("data-number"));
        var leftScore = null;
        var rightScore = null;
        if (byValue) {
          leftScore = rankingOf(left, rankModel);
          rightScore = rankingOf(right, rankModel);
        } else if (sortKey) {
          leftScore = metricOf(left, sortKey);
          rightScore = metricOf(right, sortKey);
        }
        if (byValue || sortKey) {
          if (leftScore === null && rightScore === null) return leftNumber - rightNumber;
          if (leftScore === null) return 1;
          if (rightScore === null) return -1;
          if (leftScore !== rightScore) return descending ? rightScore - leftScore : leftScore - rightScore;
        }
        return leftNumber - rightNumber;
      });
      var wanted = checkedCampaigns();
      labels.forEach(function (label) {
        var kind = label.getAttribute("data-kind");
        var kindShown = kind === "predicted" ? showPredicted : showExperimental;
        var inClass = wanted.length === 0;
        for (var w = 0; w < wanted.length && !inClass; w++) {
          if (hasCampaign(label, wanted[w])) inClass = true;
        }
        var byModel = true;
        if (wantedModel) {
          byModel = namedBy(label, wantedModel) === keepNamed;
        }
        var byRank = true;
        if (rankMode) {
          byRank = hasLlmRanking(label, rankMode) === keepRanked;
        }
        var show = kindShown && inClass && byModel && byRank;
        label.hidden = !show;
        var box = label.querySelector("input");
        if (box && box.name === "pick") {
          box.disabled = !show;
          if (!show) box.checked = false;
        }
        structureRows.appendChild(label);
      });
      var resultItems = document.querySelectorAll("ol.llm-results > li");
      for (var r = 0; r < resultItems.length; r++) {
        var shown = true;
        if (wantedModel) {
          shown = namedBy(resultItems[r], wantedModel) === keepNamed;
        }
        if (shown && rankMode) {
          shown = hasLlmRanking(resultItems[r], rankMode) === keepRanked;
        }
        resultItems[r].hidden = !shown;
        // The reply tick goes with its row, exactly as the pick box does above.
        // `hidden` does not keep an enabled named control out of the submission,
        // so a ticked reply on a hidden row was still prepended to the message
        // sent to the models while the page showed neither the row nor the tick.
        // Unquoted, like the pick lookup above. This script is embedded in every
        // page, so a quoted attribute value written here reads as a rendered tick
        // to anything scanning the page text for one.
        var replyBox = resultItems[r].querySelector("input[name=prior_reply]");
        if (replyBox) {
          replyBox.disabled = !shown;
          if (!shown) replyBox.checked = false;
        }
      }
      var chosen = campaignBoxes();
      for (var c = 0; c < chosen.length; c++) {
        if (chosen[c].checked) applyCampaign(chosen[c]);
      }
      syncAllBox();
    }
    function campaignNames(label) {
      var raw = label.getAttribute("data-campaigns") || "";
      if (!raw) return [];
      return raw.split("|");
    }
    function hasCampaign(label, name) {
      var names = campaignNames(label);
      for (var i = 0; i < names.length; i++) {
        if (names[i] === name) return true;
      }
      return false;
    }
    function structureLabels() {
      return structureRows.querySelectorAll("label[data-kind]");
    }
    function pickBox(label) {
      var box = label.querySelector("input");
      if (!box || box.name !== "pick") return null;
      return box;
    }
    function everyRowBox() {
      var inputs = structureRows.querySelectorAll("input");
      for (var i = 0; i < inputs.length; i++) {
        if (inputs[i].name === "all") return inputs[i];
      }
      return null;
    }
    // The rows this page is offering: on show, with a tick that can still be sent.
    function selectableBoxes() {
      var labels = structureLabels();
      var boxes = [];
      for (var i = 0; i < labels.length; i++) {
        if (labels[i].hidden) continue;
        var pick = pickBox(labels[i]);
        if (pick && !pick.disabled) boxes.push(pick);
      }
      return boxes;
    }
    // All Structure stands for every row on show, so unticking one row clears it
    // and the download is what the ticks say it is.
    function syncAllBox() {
      var box = everyRowBox();
      if (!box || box.disabled) return;
      var boxes = selectableBoxes();
      var on = boxes.length > 0;
      for (var i = 0; i < boxes.length; i++) {
        if (!boxes[i].checked) on = false;
      }
      box.checked = on;
    }
    // A class box does two things: it ticks that class's rows, and it narrows the
    // view to them. Only the second is something the user asks for by clicking it,
    // so only a box clicked here narrows anything. syncCampaignBoxes ticks a box to
    // mirror its rows, and a mirror is not a filter: read as one, it stood All
    // Structure down the moment All Structure had ticked every row.
    function chosenCampaign(box) {
      return box.getAttribute("data-chosen") === "yes";
    }
    function checkedCampaigns() {
      var boxes = campaignBoxes();
      var names = [];
      for (var i = 0; i < boxes.length; i++) {
        if (boxes[i].checked && chosenCampaign(boxes[i])) names.push(boxes[i].value);
      }
      return names;
    }
    function campaignBoxes() {
      var list = document.querySelector("ul.campaign");
      if (!list) return [];
      return list.querySelectorAll("input.campaign-select");
    }
    function applyCampaign(box) {
      var labels = structureLabels();
      var others = campaignBoxes();
      for (var i = 0; i < labels.length; i++) {
        var label = labels[i];
        if (label.hidden || !hasCampaign(label, box.value)) continue;
        var pick = pickBox(label);
        if (!pick || pick.disabled) continue;
        if (box.checked) {
          pick.checked = true;
          continue;
        }
        var covered = false;
        for (var j = 0; j < others.length; j++) {
          if (others[j] === box || !others[j].checked) continue;
          if (hasCampaign(label, others[j].value)) covered = true;
        }
        if (!covered) pick.checked = false;
      }
    }
    function syncCampaignBoxes() {
      var boxes = campaignBoxes();
      var labels = structureLabels();
      for (var i = 0; i < boxes.length; i++) {
        var count = 0;
        var marked = 0;
        for (var j = 0; j < labels.length; j++) {
          if (labels[j].hidden || !hasCampaign(labels[j], boxes[i].value)) continue;
          var pick = pickBox(labels[j]);
          if (!pick || pick.disabled) continue;
          count += 1;
          if (pick.checked) marked += 1;
        }
        boxes[i].checked = count > 0 && marked === count;
      }
    }
    view.addEventListener("change", applyStructureView);
    var campaignList = document.querySelector("ul.campaign");
    if (campaignList) {
      campaignList.addEventListener("change", function (event) {
        var box = event.target;
        if (!box || !box.classList || !box.classList.contains("campaign-select")) return;
        box.setAttribute("data-chosen", box.checked ? "yes" : "no");
        // Unticking has to run while its rows are still on show, so it is first.
        if (!box.checked) applyCampaign(box);
        applyStructureView();
      });
    }
    structureRows.addEventListener("change", function (event) {
      var box = event.target;
      if (!box) return;
      if (box.name === "all") {
        // The reported bug: the box said All Structure and ticked nothing, because
        // it was only ever read on the server. It now ticks every row the page is
        // offering, and clears every one of them when it is cleared.
        var boxes = selectableBoxes();
        for (var i = 0; i < boxes.length; i++) boxes[i].checked = box.checked;
        syncCampaignBoxes();
        return;
      }
      if (box.name !== "pick") return;
      syncCampaignBoxes();
      syncAllBox();
    });
    applyStructureView();
  }
  var search = document.querySelector(".llm-search");
  if (search) {
    var everyModel = search.querySelector('input[name="all_models"]');
    function listedModels() {
      return search.querySelectorAll('input[name="model"]');
    }
    if (everyModel) {
      everyModel.addEventListener("change", function () {
        var boxes = listedModels();
        for (var i = 0; i < boxes.length; i++) boxes[i].checked = everyModel.checked;
      });
      search.addEventListener("change", function (event) {
        var box = event.target;
        if (!box || box.name !== "model") return;
        var boxes = listedModels();
        var on = boxes.length > 0;
        for (var i = 0; i < boxes.length; i++) {
          if (!boxes[i].checked) on = false;
        }
        everyModel.checked = on;
      });
    }
  }
  // Add selected structures writes the ticked rows into the question box. It is
  // not a submit, so the models are asked only when Submit question is pressed.
  var questionBox = form.querySelector("textarea.llm-question");
  var addStructures = form.querySelector(".llm-add-structures");
  var askNote = document.getElementById("llm-ask-note");
  function askNoteText(text) {
    if (!askNote) return;
    askNote.hidden = !text;
    askNote.textContent = text || "";
  }
  if (questionBox && addStructures) {
    addStructures.addEventListener("click", function () {
      // Unquoted so this selector is not counted as another structure tick.
      var picks = form.querySelectorAll(".structure-rows input[name=pick]");
      var present = {};
      var existing = questionBox.value.split("\\n");
      for (var i = 0; i < existing.length; i++) present[existing[i].trim()] = true;
      var lines = [];
      var selected = 0;
      for (var p = 0; p < picks.length; p++) {
        var box = picks[p];
        if (!box.checked || box.disabled) continue;
        selected += 1;
        var label = box.closest("label");
        var text = label ? label.textContent.replace(/\\s+/g, " ").trim() : "";
        if (!text || present[text]) continue;
        lines.push(text);
        present[text] = true;
      }
      if (!selected) {
        askNoteText("Tick a structure to add it.");
        return;
      }
      if (!lines.length) {
        askNoteText("Selected structures are already in the message.");
        return;
      }
      var current = questionBox.value.replace(/\\s+$/, "");
      questionBox.value = current ? current + "\\n\\n" + lines.join("\\n") : lines.join("\\n");
      askNoteText("");
      questionBox.focus();
      var end = questionBox.value.length;
      questionBox.selectionStart = end;
      questionBox.selectionEnd = end;
      questionBox.scrollTop = questionBox.scrollHeight;
    });
  }
})();
"""


@dataclass(frozen=True)
class WorkerChoice:
    """How wide to run, as the start-generation page left it.

    Returned beside the model rather than folded into :class:`OptionForm`,
    because it answers a narrower question: the form is every setting a run was
    started with, and this is the two that the last page before generation can
    still change. ``None`` on either side means every request of that kind at
    once, which is the default and what happened before the page offered them.
    """

    local: int | None = None
    remote: int | None = None


@dataclass
class OptionForm:
    """The options the panel edits. Empty optional fields stay unset."""

    disease: str | None = None
    catalog: str | None = None
    target_structure: str | None = None
    accept_only: bool = False
    choice: int | None = None
    structures_json: str | None = None
    take_all: bool = False
    pick: str | None = None
    binder_kind: str = "protein"
    out_dir: str = "disease_runs"
    #: None uses ``out_dir``. A different path is where the structure files are fetched.
    download_dir: str | None = None
    hosts: str | None = None
    remote_dir: str = "disease_runs"
    local_workers: int | None = None
    #: Requests in flight on remote machines at once, across all of them.
    #: ``None`` starts every remote request at once, which is what happened
    #: before this existed, so leaving it unset changes nothing.
    remote_workers: int | None = None
    modified_ckpt: str | None = None
    confirm_inference: bool = False
    run: bool = False
    #: Screening options. The panel shows no control for these; they ride along
    #: hidden so a flag the user typed is not dropped when the panel rebuilds
    #: the command. They decide which liability axes can be measured at all, so
    #: losing one silently records the defaults as the request.
    proteome: str | None = None
    lookup_targets: bool = False
    plasma_abundance_max: float | None = None
    plasma_hub_max: int | None = None
    plasma_fold_max: float | None = None
    follow_up: bool = False
    #: Research options, carried for the same reason as the screening block
    #: above and with the same absence of a control. A ``--ui`` run dropped all
    #: three: the research pass the user asked for never ran, no
    #: ``target_research.json`` was written, and because the report stayed empty
    #: the ranked-target page could not be reached at all -- while ``--target``
    #: alone, which the command line refuses outright, was silently ignored.
    #: Each is an ``append`` flag, so each is a tuple of its posted values.
    research: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    target: tuple[str, ...] = ()


def wants_option_ui(incoming: list[str]) -> bool:
    """True when the user asked for the panel, or passed no arguments at all.

    Screen-only and validation-only runs bypass the landing form. Their paths
    belong to the CLI rather than the form's rebuilt command.
    """
    if "--help" in incoming or "-h" in incoming:
        return False
    if _names_finished_run(incoming):
        return False
    return not incoming or "--ui" in incoming


def _names_finished_run(incoming: list[str]) -> bool:
    """Whether argv asks to screen or validate a finished run, including abbreviations.

    Asked of a parser rather than of the strings. argparse accepts
    ``--screen=DIR`` and every unambiguous abbreviation as well -- ``--sc``,
    ``--scr``, ``--scree`` -- and a comparison against the two literal
    spellings let the rest through: ``--ui --scr DIR``, which spelled in full
    the parser refuses on its mutually exclusive group, instead opened the
    panel, which rebuilds the command from a form that has no field for the
    flag. The flag was dropped and a read-only re-screen became a full
    download-and-generate run against that directory.

    A spelling the real parser considers ambiguous is matched here too. That is
    the safe direction: the panel stays shut and the real parser then reports
    the ambiguity, rather than the panel opening and swallowing the request.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--screen")
    parser.add_argument("--validate-output")
    try:
        # Stderr is swallowed because this is a question, not the parse. A
        # dangling `--screen` makes argparse print a usage block naming only
        # this throwaway parser's one option, so the user saw that and then the
        # real parser's own usage, two of them, disagreeing about what the
        # program accepts.
        with contextlib.redirect_stderr(io.StringIO()):
            known, _rest = parser.parse_known_args(incoming)
    except SystemExit:
        # A finished-run flag without a value. Left to the real parser, whose
        # error names the flag; answering True here would refuse the panel over
        # an argv that is not going to run either way.
        return False
    return known.screen is not None or known.validate_output is not None


class OptionPanel:
    """Shows each step in the browser, or through an injected exchange in tests."""

    #: Refused submissions accepted in a row before the panel gives up. A person in a
    #: browser reads the notice and corrects the field, so this is never reached; a
    #: client that keeps sending the same refused value would otherwise loop forever.
    MAX_REFUSED = 20

    def __init__(self, exchange: Exchange | None = None) -> None:
        self._exchange = exchange
        self._session: BrowserSession | None = None
        self._key_store_error = ""
        #: Answers given in this pass, one per page shown.
        self._answers: list[dict[str, list[str]]] = []
        #: Answers handed straight back without showing the page, after a Back.
        self._replay: list[dict[str, list[str]]] = []
        #: Answers from the pass that was interrupted, which Forward restores.
        self._ahead: list[dict[str, list[str]]] = []
        #: True once the run has written files, which closes Back and Forward.
        self._sealed = False
        #: The first page to draw after Return to start. None uses the launch flags.
        self._landing_override: OptionForm | None = None
        #: Model ids chosen on the landing page during this run. Empty keeps the env or saved choice.
        self._chat_models: dict[str, str] = {}
        #: Listed models chosen on the research page. None keeps one model per provider.
        self._search_models: dict[str, tuple[str, ...]] | None = None
        #: The question box as last posted. Replies are written back into this text.
        self._posted_question = ""
        #: Ticked earlier search or research replies, in page order.
        self._posted_replies: tuple[str, ...] = ()
        #: Structures noted before the browser session exists. Shown on the finished page.
        self._pending_structures: list[StructureTrio] = []
        #: Every saved key, with the model it is sent with. The first secret is ``_api_keys``.
        self._key_slots: dict[str, list[KeySlot]] = {}
        try:
            loaded = load_key_slots()
        except ValueError as exc:
            self._api_keys = {}
            self._key_store_error = str(exc)
        else:
            self._key_slots = {name: list(rows) for name, rows in loaded.items()}
            self._api_keys = {name: rows[0].secret for name, rows in self._key_slots.items() if rows}

    def collect(self, incoming: list[str], parser: argparse.ArgumentParser) -> list[str]:
        """Show the option panel and return the argv the rest of the program parses."""
        cleaned = [item for item in incoming if item != "--ui"]
        if self._landing_override is not None:
            current = self._landing_override
            self._landing_override = None
            return self._until_named(current, "")
        return self._until_named(_form_from_namespace(parser.parse_args(cleaned)), "")

    def replace_disease(self, args: argparse.Namespace, notice: str) -> list[str]:
        """Show the same options again after a disease name was not found."""
        return self._until_named(_form_from_namespace(args), notice)

    def _until_named(self, current: OptionForm, notice: str) -> list[str]:
        where = ""
        hosts_text: str | None = None
        machine_search = ""
        out_file_search = ""
        selected_out_file = ""
        file_steps = 0
        refused = 0
        while True:
            if not notice and self._key_store_error:
                notice = self._key_store_error
            fields = self.ask(
                render_options(
                    current,
                    notice,
                    saved=tuple(self._api_keys),
                    where=where,
                    hosts_text=hosts_text,
                    slots=self._key_slots,
                    machines=tuple(discover_remote_machines(machine_search)),
                    machine_search=machine_search,
                    out_file_search=out_file_search,
                    selected_out_file=selected_out_file,
                )
            )
            log().info("option panel submitted: %s", redact_fields(fields))
            action = _first(fields, "__action")
            picked = _first(fields, "out_pick")
            if action in {"search", "search-files", "browse-machine"} or picked:
                # Searching is not an answer, so the page comes back narrowed. Ticked
                # machines are folded into the text box so a second search keeps them.
                # A clicked output file or folder does the same for that directory.
                # Windows File Explorer adds the computer the person selected there.
                browsing = action == "browse-machine" and not picked
                if action == "search" and not picked:
                    refused += 1
                    if refused >= self.MAX_REFUSED:
                        raise ValueError("The machine search was repeated too many times.")
                elif browsing:
                    refused += 1
                    if refused >= self.MAX_REFUSED:
                        raise ValueError("The machine browser was opened too many times.")
                else:
                    file_steps += 1
                    if file_steps >= _MAX_FILE_BROWSE:
                        raise ValueError("The output directory was browsed too many times.")
                machine_search = _first(fields, "machine_search")
                out_file_search = _first(fields, "out_file_search")
                current = draft_from_fields(fields)
                where = _first(fields, "where")
                names = remote_names_from_fields(fields)
                notice = ""
                if browsing:
                    try:
                        chosen = pick_remote_machine()
                    except ValueError as exc:
                        notice = str(exc)
                        chosen = ""
                    if chosen and chosen not in names:
                        names.append(chosen)
                    elif not chosen and not notice:
                        notice = "No remote machine was selected."
                hosts_text = ",".join(names)
                if picked:
                    current, selected_out_file = choose_output_entry(current, picked)
                    if not selected_out_file:
                        out_file_search = ""
                continue
            try:
                self._remember_keys(fields)
                self._remember_models(fields)
                form = form_from_fields(fields)
            except ValueError as exc:
                # A field the reader refuses is the user's to correct. The page comes
                # back with what was typed and the reason, because stopping the run
                # here would close the browser and lose the whole form.
                log().warning("option panel rejected a field: %s", exc)
                refused += 1
                if refused >= self.MAX_REFUSED:
                    log().error("option panel gave up after %s refused submissions", refused)
                    raise
                notice = str(exc)
                current = draft_from_fields(fields)
                where = _first(fields, "where")
                hosts_text = _first(fields, "hosts")
                continue
            refused = 0
            where = ""
            hosts_text = None
            if form.disease or form.catalog:
                return to_argv(form)
            notice = "Enter the scientific or medical name of the disease."
            current = form

    def confirm_disease(
        self,
        disease: Disease,
        alternatives: Sequence[Disease] = (),
        query: str = "",
    ) -> Disease | None:
        """The named disease when its tick is set, otherwise the matching-names menu.

        A ticked box wins even if a menu value was posted with it. A clear box
        uses the menu. An empty or invalid menu declines.

        The posted index is resolved against the list the page rendered, from
        the same three arguments. They have to agree: a menu built from one
        option set and read against another returns a disease the user did
        not pick, and nothing downstream could tell.
        """
        options = _confirm_options(disease, alternatives, query)
        fields = self.ask(render_confirm(disease, alternatives, query))
        if _checked(fields, "accept"):
            return disease
        raw = _first(fields, "choice")
        if not raw.isdigit():
            return None
        index = int(raw) - 1
        if not 0 <= index < len(options):
            return None
        return options[index]

    def choose_focus(
        self,
        disease: Disease,
        pathways: Sequence[tuple[str, str]] = (),
        targets: Sequence[tuple[str, str]] = (),
    ) -> BiologicalFocus:
        """Pathway, mechanism, and optional target. A blank post keeps every pathway."""
        fields = self.ask(render_focus(disease, pathways, targets))
        return focus_from_fields(fields, pathways)

    def choose_disease(self, hits: list[Disease]) -> int:
        """Dropdown of similar names. Returns the chosen index."""
        raw = _first(self.ask(render_disease_choice(hits)), "choice")
        if not raw.isdigit():
            raise ValueError(f"{len(hits)} similar names matched. Choose one in the dropdown.")
        index = int(raw) - 1
        if not 0 <= index < len(hits):
            raise ValueError(f"Choice {raw} is outside 1..{len(hits)}.")
        return index

    def choose_structures(
        self,
        rows: list[StructureOption],
        scores: dict[str, RecordScore] | None = None,
        *,
        ranking: list[dict[str, object]] | None = None,
        results: list[dict[str, object]] | None = None,
        notice: str = "",
        transcript: str = "",
        focus: BiologicalFocus | None = None,
    ) -> tuple[str, bool, str | None, list[str]]:
        """Structure ticks, plus the model ticks and the question box.

        ``all`` and the ticked numbers are a union. A ticked design class adds every
        listed structure in that class. Download returns no providers, so only the
        ticked structures continue. The two searches and Submit question read the
        model ticks. ``transcript`` is the question box, including earlier replies.
        ``scores`` gives each row its program score, which is how many public
        databases returned a record.
        """
        keyed = providers_with_keys(self._api_keys)
        fields = self.ask(
            render_structures(
                rows,
                saved=keyed,
                scores=scores,
                ranking=ranking,
                results=results,
                notice=notice,
                transcript=transcript,
                focus=focus,
            )
        )
        action = _first(fields, "next") or "download"
        if action not in STRUCTURE_ACTIONS:
            known = ", ".join(STRUCTURE_ACTIONS)
            raise ValueError(f"Unknown structure action {action!r}. Expected one of {known}.")
        posted = fields.get("llm_question") or []
        self._posted_question = posted[0].rstrip() if posted else ""
        chosen_replies = {item.strip() for item in fields.get("prior_reply", []) if item.strip()}
        self._posted_replies = tuple(text for key, text in prior_reply_lines(ranking, results) if key in chosen_replies)
        numbers = [item.strip() for item in fields.get("pick", []) if item.strip()]
        seen: set[str] = set()
        merged: list[str] = []
        for item in campaign_pick_numbers(rows, fields.get("campaign", [])) + numbers:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
        numbers = merged
        experimental = _first(fields, "experimental")
        predicted = _first(fields, "predicted")
        take_all, numbers = apply_structure_view(
            rows,
            take_all=_checked(fields, "all"),
            picks=numbers,
            experimental=experimental or "include",
            predicted=predicted or "include",
        )
        numbers = _filter_picks_by_llm_ranking(
            numbers,
            ranking,
            _first(fields, "llm_rank") or "any",
            _first(fields, "llm_rank_state") or "include",
            _first(fields, "rank_model"),
        )
        numbers = _filter_picks_by_llm_model(
            numbers,
            results,
            _first(fields, "llm_model") or "any",
            _first(fields, "llm_state") or "include",
        )
        picks = ",".join(numbers) if numbers else None
        # Download fetches the ticked structures and leaves the models off.
        # The two searches and Submit question are what read the model ticks.
        if action in {"missing", "rank", "ask"}:
            providers, models = research_from_fields(fields, saved=keyed, include_grok_bot=action == "ask")
        else:
            providers, models = [], None
        self._search_models = models
        return action, take_all, picks, providers

    def posted_question(self) -> str:
        """The question box as posted, without a trailing blank line."""
        return self._posted_question

    def question_for_models(self) -> str:
        """The box plus ticked earlier replies that are not already written in it.

        The replies come first and the typed question stays at the end, so a
        further question is sent with the search or research text it refers to.
        """
        return _question_with_replies(self._posted_question, self._posted_replies)

    def show_option_comparison(
        self,
        comparison: dict[str, list[dict[str, str]]],
        *,
        by_database: list[dict[str, object]] | None = None,
        by_model: dict[str, list[dict[str, object]]] | None = None,
    ) -> None:
        """Show the confirmed program search beside the structures the models named."""
        self.ask(render_option_comparison(comparison, by_database=by_database, by_model=by_model))

    def research_models(self) -> dict[str, tuple[str, ...]] | None:
        """Listed models chosen on the research page. None keeps one model per provider."""
        return self._search_models

    def choose_targets(self, ranked: list[dict[str, object]]) -> list[str]:
        """Tick boxes for the ranked targets. Returns the chosen identifiers."""
        fields = self.ask(render_targets(ranked))
        identifiers = [str(entry.get("identifier") or "") for entry in ranked]
        if _checked(fields, "all_targets"):
            return [name for name in identifiers if name]
        chosen: list[str] = []
        for value in fields.get("target", []):
            name = value.strip()
            if not name:
                continue
            if name not in identifiers:
                raise ValueError(f"Target {name!r} is not on the ranked list.")
            if name not in chosen:
                chosen.append(name)
        return chosen

    def api_keys(self) -> dict[str, str]:
        """The first saved key of each provider. Blank fields keep that key."""
        return dict(self._api_keys)

    def key_slots(self) -> dict[str, tuple[KeySlot, ...]]:
        """Every saved key and the model it is sent with."""
        return {name: tuple(rows) for name, rows in self._key_slots.items()}

    def models(self) -> dict[str, str]:
        """Model ids chosen on the landing page during this run."""
        return dict(self._chat_models)

    def _remember_models(self, fields: dict[str, list[str]]) -> None:
        labels = dict(_PROVIDER_LABELS)
        chosen: dict[str, str] = {}
        for provider, name in _MODEL_FIELDS:
            value = _first(fields, name)
            if not value:
                continue
            allowed = tuple(model_id for model_id, _label in model_choices(provider))
            if value not in allowed:
                known = ", ".join(allowed)
                raise ValueError(f"Unknown {labels[provider]} model {value!r}. Expected one of {known}.")
            chosen[provider] = value
        if not chosen:
            return
        self._chat_models.update(chosen)
        if save_model_choices(chosen):
            print("Saved model choices for the next run.")

    def _remember_keys(self, fields: dict[str, list[str]]) -> None:
        """Store typed keys. ``*****`` or a blank field keeps that slot's secret.

        A provider left out of the form keeps its saved keys. The listing marker
        is present when this page drew that provider, so removing every row drops
        those keys. A form that never drew the rows does not.
        """
        changed = False
        updated = {name: list(rows) for name, rows in self._key_slots.items()}
        for provider, name in _API_KEY_FIELDS:
            listed = f"{provider}_listed" in fields
            posted = name in fields or f"{provider}_slot" in fields
            if not posted and not listed:
                continue
            current = self._key_slots.get(provider, [])
            built = [] if not posted else _slots_from_fields(provider, fields, current)
            if built != current:
                if built:
                    updated[provider] = built
                else:
                    updated.pop(provider, None)
                changed = True
        if not changed:
            return
        self._key_slots = updated
        self._api_keys = {name: rows[0].secret for name, rows in updated.items() if rows}
        save_key_slots(updated)
        self._key_store_error = ""
        print("Saved encrypted API keys for the next run.")

    def ask(self, html: str) -> dict[str, list[str]]:
        """Return the submitted form, replaying an earlier answer after a Back.

        Tests pass an exchange and never open a browser. A page the user has
        already answered in an interrupted pass offers Forward, which hands that
        answer back instead of asking again.
        """
        index = len(self._answers)
        if index < len(self._replay):
            answer = self._replay[index]
            self._answers.append(answer)
            return answer
        ahead = index < len(self._ahead) and not self._sealed
        remembered = self._ahead[index] if ahead else None
        fields = self._show(
            with_nav(html, back=index > 0 and not self._sealed, forward=remembered is not None),
            index,
        )
        move = _first(fields, "__nav")
        if move == "start":
            # The posted fields are this page. Keeping them is what puts the same
            # disease, directories, and ticks back on the first page.
            self._keep_posted(fields)
            self._answers = [*self._answers[:index], fields]
            raise ReturnToStartError()
        if move == "fresh":
            raise RestartFreshError()
        if self._sealed:
            move = ""
        if move == "back" and index > 0:
            raise StepBackError(back_target(fields, index))
        if move == "forward" and remembered is not None:
            fields = remembered
        elif remembered is not None and fields != remembered:
            # The answer changed, so the answers that followed it no longer apply and
            # Forward stops being offered past this page.
            del self._ahead[index + 1 :]
        self._answers.append(fields)
        return fields

    def browser_open(self) -> bool:
        """True when a browser is waiting, so the finished page can offer a choice."""
        return self._session is not None

    def choose_screens(self, directory: Path, kind: str) -> tuple[str, ...]:
        """Offer the actual finished run's choices before the first optional calculation."""
        from rfproteina.disease import screen_selection

        choices = screen_selection.plan(directory, kind)
        body = _scientific_screen_panel(choices, None, [], submit_button=False)
        page = _page(f"Choose screens for {directory.name}", body, "Continue with selected screens", restart=False)
        fields = self._show(page, len(self._answers))
        return screen_selection.select(choices, "explicit", fields.get(SCREENING_FIELD, ()))

    def finish_choice(self) -> str:
        """The finished page. Close ends the run. The other two start it again.

        ``close`` is the Close button. ``start`` keeps this run's answers.
        ``fresh`` drops them. Tests that never opened a browser get ``close``
        without a page, because nothing is waiting to be shown one.
        """
        if self._session is None:
            return "close"
        actions = 0
        while True:
            fields = self._show(with_nav(self._session.done_page(), back=False, forward=False), len(self._answers))
            move = _first(fields, "__nav")
            if move == "start":
                return "start"
            if move == "fresh":
                return "fresh"
            action = _first(fields, "__action")
            if not action:
                return "close"
            # An action is not an answer, so the page comes back with its
            # result instead of the run ending. Without this loop every button
            # on this page was a Close button: ``finish_choice`` read only
            # ``__nav``, so any other submission fell through to "close", the
            # panel shut down, and the POST that pressed the button never got
            # a page at all.
            actions += 1
            if actions >= self.MAX_RESULTS_ACTIONS:
                raise ValueError("The results page was asked for too many things.")
            self._session.note_result(self._results_action(action, flat_fields(fields)))

    #: Bounded like every other repeated non-answer on these pages, and for the
    #: same reason: a test whose injected exchange returns the same fields on
    #: every call would otherwise spin here forever.
    MAX_RESULTS_ACTIONS = 40

    def _results_action(self, action: str, fields: list[str]) -> str:
        """Do what a results-page button asked, and say what happened.

        Returns one sentence for the page. Never raises for a failure of the
        thing asked: a run that has already finished must not be ended by a
        button, so every outcome is a sentence and the page comes back.
        """
        try:
            return self._session.run_results_action(action, fields)
        except (OSError, ValueError, LookupError) as exc:
            log().info("results action %s failed: %s", action, exc)
            return f"That did not work: {exc}"

    def prepare_saved_start(self) -> None:
        """Show the first page from the answers already given, and allow Back again."""
        self._unseal()
        self._clear_closing()
        landing = self._latest_landing()
        self._landing_override = draft_from_fields(landing) if landing is not None else None
        self._ahead = list(self._answers)
        self._replay = []
        self._answers = []

    def prepare_fresh_start(self) -> None:
        """Drop this run's answers and show the first page as a new launch would."""
        self._answers = []
        self._replay = []
        self._ahead = []
        self._landing_override = None
        self._posted_question = ""
        self._posted_replies = ()
        self._search_models = None
        self._chat_models = {}
        self._reload_keys()
        self._unseal()
        self._clear_closing()

    def _latest_landing(self) -> dict[str, list[str]] | None:
        """The last first-page submission. Later pages do not carry its fields."""
        for fields in reversed(self._answers):
            if "binder_kind" in fields or "disease" in fields:
                return fields
        return None

    def _keep_posted(self, fields: dict[str, list[str]]) -> None:
        """Save keys and model choices from the page being left, when they are valid."""
        try:
            self._remember_keys(fields)
            self._remember_models(fields)
        except (OSError, ValueError) as exc:
            log().warning("panel kept the page without saving a field: %s", exc)

    def _reload_keys(self) -> None:
        """Read the saved keys again. A fresh start keeps what was already stored."""
        try:
            loaded = load_key_slots()
        except ValueError as exc:
            self._api_keys = {}
            self._key_store_error = str(exc)
            return
        self._key_slots = {name: list(rows) for name, rows in loaded.items()}
        self._api_keys = {name: rows[0].secret for name, rows in self._key_slots.items() if rows}
        self._key_store_error = ""

    def _unseal(self) -> None:
        self._sealed = False
        if self._session is not None:
            self._session.unseal()

    def _clear_closing(self) -> None:
        self._pending_structures.clear()
        if self._session is not None:
            self._session.clear_closing()

    def seal(self) -> None:
        """Stop offering Back, because the run has started writing files.

        Going back replays the earlier answers by restarting the run, which would
        download every selected structure again. Navigation therefore closes as
        soon as the first file is written, and a browser Back lands on the page
        being shown instead of rewinding.
        """
        self._sealed = True
        if self._session is not None:
            self._session.seal()

    def confirm_inference(
        self,
        count: int,
        *,
        modified: str = "",
        confirmed: bool = False,
        hosts: str | None = None,
        local_workers: int | None = None,
        remote_workers: int | None = None,
    ) -> tuple[bool, bool, str, WorkerChoice]:
        """Inference, starting generation, and which of the two models generates.

        One page, so every answer comes back together. ``modified`` is the
        modified checkpoint this run found; without one the page offers the
        original model alone. ``confirmed`` is the first page's tick, which
        answers the inference question here instead of skipping this page.

        Continue starts generation. A second tick box used to ask that, sitting
        directly above the button that already did it, so the two answers are
        one answer: clearing the inference box stops here, and continuing runs.
        """
        fields = self.ask(
            render_inference(
                count,
                modified=modified,
                confirmed=confirmed,
                hosts=hosts,
                local_workers=local_workers,
                remote_workers=remote_workers,
            )
        )
        accepted = _checked(fields, "accept")
        # Read strictly, the way the first page reads them: a number that is not
        # one refuses the page rather than falling back to "every request at
        # once", which would start an uncapped run nobody asked for on the one
        # page where the next click launches it.
        return (
            accepted,
            accepted,
            generation_model_from_fields(fields, modified=modified),
            WorkerChoice(
                local=_workers_from_fields(fields),
                remote=_workers_from_fields(fields, field="remote_workers", label="Remote workers"),
            ),
        )

    def rewind(self, target: int) -> None:
        """Prepare the next pass to replay up to ``target`` and ask there."""
        self._ahead = list(self._answers)
        self._replay = self._answers[:target]
        self._answers = []

    def _show(self, html: str, index: int) -> dict[str, list[str]]:
        if self._exchange is not None:
            return self._exchange(html)
        if self._session is None:
            self._session = BrowserSession()
            self._flush_structures()
        return self._session.exchange(html, step=index)

    def close(self) -> None:
        """Release the browser step that is waiting for the next page."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def note_progress(self, done: int, total: int, label: str, action: str) -> None:
        """Publish progress for the open panel, named by what is being done."""
        if self._session is not None:
            self._session.note_progress(done, total, label, action)

    def note_download(self, done: int, total: int, label: str) -> None:
        """Publish structure-download progress for the open panel."""
        self.note_progress(done, total, label, DOWNLOAD_ACTION)

    def note_closing(
        self,
        lines: list[str],
        screens: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        """Sentences, and the screens behind them, for the page shown when the panel closes."""
        if self._session is not None:
            self._session.note_closing(lines, screens)

    def note_structures(self, trios: Sequence[StructureTrio]) -> None:
        """Keep each generated design for the three views on the finished page."""
        if self._session is None:
            self._pending_structures.extend(trios)
            return
        self._session.note_structures(trios)

    def note_validation_runs(self, directories: Sequence[Path], *, open_panel: bool = False) -> None:
        """Attach finished run directories to the results page."""
        if open_panel and self._session is None:
            self._session = BrowserSession()
        if self._session is not None:
            self._session.note_validation_runs(directories, standalone=open_panel)

    def _flush_structures(self) -> None:
        """Hand structures noted before the session existed to that session."""
        if self._session is None or not self._pending_structures:
            return
        self._session.note_structures(self._pending_structures)
        self._pending_structures.clear()

    def clear_download(self) -> None:
        """Clear the download bar. The pressed button stays on Processing until the next page."""
        if self._session is not None:
            self._session.clear_download()


class BrowserSession:
    """One localhost page. Each submit waits until the next page, or until close.

    A finished submit redirects, so the browser's current address is a GET.
    Refreshing loads that page again. With no page stored, the GET is the main page.
    """

    def __init__(self, opener: Callable[[str], None] | None = None) -> None:
        self._opener = opener or _open_browser
        self._lock = threading.Lock()
        self._html = ""
        self._posted = threading.Event()
        self._fields: dict[str, list[str]] = {}
        self._next_ready = threading.Event()
        self._next_html: str | None = None
        self._opened = False
        self._sent = threading.Event()
        self._inflight = False
        self._result = threading.Event()
        self._progress = dict(_IDLE_PROGRESS)
        self._step = 0
        self._closed = False
        self._sealed = False
        self._closing: list[str] = []
        #: The last results-page action's sentence. Replaced, never appended.
        #: NOT ``_result``, which is this session's threading.Event: naming it
        #: that shadowed the Event and broke ``submit`` on ``_result.clear()``.
        self._action_result = ""
        #: Retrieved models held in memory, served under the same token shape
        #: as a file. A pasted model was never a file and is not made one.
        self._text_bytes: dict[str, bytes] = {}
        #: The text the last results-page action produced, shown on the page so
        #: it can be copied, and served under a token so it can be saved.
        self._payload = ""
        self._payload_name = ""
        self._payload_token = ""
        #: The machine-readable half of the last results-page action, for a tab
        #: whose result is a table of numbers rather than a sentence. Merged
        #: across actions, so the protocol's proposal survives its comparison.
        self._action_record: dict[str, object] = {}
        #: Bumped whenever a results-page overlay changes, so an open tab can
        #: tell that the page it is looking at is out of date.
        self._results_version = 0
        #: Which results tab was open when the page was last submitted.
        self._results_tab = "binder"
        #: Whether to watch the download folders at all.
        self._auto_watch = True
        self._auto_checked = 0.0
        self._auto_seen: set[tuple[str, float]] = set()
        self._screens: list[Mapping[str, object]] = []
        #: Token to the CIF it may serve. The page never receives the path.
        self._files: dict[str, Path] = {}
        self._file_bytes: dict[str, bytes] = {}
        self._path_tokens: dict[Path, str] = {}
        self._catalog: list[dict[str, object]] = []
        self._validation_runs: tuple[Path, ...] = ()
        self._validation_only = False
        self._validation_scope = "all"
        self._validation_candidate: str | None = None
        self._validation_selection: tuple[str, ...] | None = None
        self._validation_report: dict[str, str] = {}
        self._screen_method_selection: tuple[str, ...] | None = None
        self._screen_method_results: list[dict] = []
        self._shutdown = False
        #: Set when this session opened the panel window, so Close can end that window.
        self._window_port: str | None = None
        self._uploads: Path | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def exchange(self, html: str, step: int = 0) -> dict[str, list[str]]:
        """Serve ``html`` as panel step ``step`` and block until it is submitted."""
        self._ensure()
        with self._lock:
            self._html = html
            self._next_html = html
            self._step = step
            self._progress = dict(_IDLE_PROGRESS)
        self._posted.clear()
        self._next_ready.set()
        if not self._opened:
            self._opened = True
            assert self._server is not None
            port = self._server.server_address[1]
            url = f"http://127.0.0.1:{port}/"
            if self._opener is _open_browser:
                self._window_port = str(port)
            print(f"Option panel: {url}")
            self._opener(url)
        self._posted.wait()
        return dict(self._fields)

    def submit(self, fields: dict[str, list[str]]) -> str | None:
        """Record one POST and wait for the next page. ``None`` means the panel is done.

        A browser retry while that POST is still waiting must not replace it.
        Both answers use the page prepared for the first submission.
        """
        with self._lock:
            if self._inflight:
                follower = True
            else:
                self._inflight = True
                follower = False
                self._fields = fields
                self._next_html = None
                self._result.clear()
        if follower:
            self._result.wait()
            with self._lock:
                return self._next_html
        self._sent.clear()
        self._next_ready.clear()
        self._posted.set()
        self._next_ready.wait()
        with self._lock:
            page = self._next_html
            self._inflight = False
        self._result.set()
        return page

    def upload_dir(self) -> Path:
        """A new, empty directory for uploads from this panel session."""
        with self._lock:
            if self._uploads is None:
                self._uploads = Path(tempfile.mkdtemp(prefix="Humanitys-Last-Disease-upload-"))
            return self._uploads

    def step(self) -> int:
        """Which panel step is being shown."""
        with self._lock:
            return self._step

    def step_url(self) -> str:
        """The address of the step being shown."""
        return f"/step/{self.step()}"

    def seal(self) -> None:
        """Refuse to rewind from here on, because files have been written."""
        with self._lock:
            self._sealed = True

    def unseal(self) -> None:
        """Allow Back again. Return to start and Restart Server both begin a new pass."""
        with self._lock:
            self._sealed = False

    def clear_result(self) -> None:
        """Drop the results-page sentence, payload and record when this run starts over.

        The record goes with the rest. A protocol comparison belongs to the
        designs it was taken over, and carrying one into the next run would
        draw the last run's off-target table under this run's structures.
        """
        with self._lock:
            self._action_result = ""
            self._payload = ""
            self._payload_name = ""
            self._payload_token = ""
            self._action_record = {}

    def clear_closing(self) -> None:
        """Drop the previous run's screen and structures, so the next finish shows only its own."""
        with self._lock:
            self._closing.clear()
            self._screens.clear()
            self._files.clear()
            self._file_bytes.clear()
            self._path_tokens.clear()
            self._catalog.clear()
            self._validation_runs = ()
            self._validation_only = False
            self._validation_scope = "all"
            self._validation_candidate = None
            self._validation_selection = None
            self._validation_report.clear()
            self._screen_method_selection = None
            self._screen_method_results.clear()

    def sealed(self) -> bool:
        with self._lock:
            return self._sealed

    def go_back(self, target: int) -> str | None:
        """Ask the waiting flow to go back to ``target`` and return the page it shows."""
        return self.submit({"__nav": ["back"], "__to": [str(target)]})

    def mark_sent(self) -> None:
        """The POST response has been written, so close may stop the server."""
        self._sent.set()

    def note_progress(self, done: int, total: int, label: str, action: str) -> None:
        """Record progress through a step, named by the work it is doing."""
        with self._lock:
            self._progress = {
                "active": True,
                "done": done,
                "total": total,
                "label": label,
                "action": action,
            }

    def note_download(self, done: int, total: int, label: str) -> None:
        """Record how many selected structures have been written."""
        self.note_progress(done, total, label, DOWNLOAD_ACTION)

    def clear_download(self) -> None:
        """Return the progress request to idle."""
        with self._lock:
            self._progress = dict(_IDLE_PROGRESS)

    def progress(self) -> dict[str, object]:
        """Download progress plus the step the panel is showing.

        An open tab compares ``step`` with the page it loaded. When they differ,
        the tab opens that step instead of staying on the main page.
        """
        self._auto_import()
        with self._lock:
            reported = dict(self._progress)
            reported["step"] = self._step
            reported["closed"] = self._closed
            reported["results"] = self._results_version
            return reported

    def _auto_import(self) -> None:
        """Pick up an AlphaFold archive the moment it finishes downloading.

        Hung off the poll the page already makes rather than a thread of its
        own, and throttled: the page asks every 300 ms, and walking the
        download folders that often would cost more than the feature is worth.

        Only while the results page is the one showing, and only once per
        file -- the archive is remembered by path and modification time, so a
        second poll does not re-import and re-overlay the same fold.
        """
        from rfproteina.disease.results_retrieve import newest_download

        with self._lock:
            if not self._catalog or self._closed or not self._auto_watch:
                return
            now = time.monotonic()
            if now - self._auto_checked < self.AUTO_POLL_SECONDS:
                return
            self._auto_checked = now
            seen = self._auto_seen
        try:
            found = newest_download(within_seconds=self.AUTO_WINDOW_SECONDS)
        except OSError:
            return
        if found is None:
            return
        try:
            mark = (str(found), found.stat().st_mtime)
        except OSError:
            return
        if mark in seen:
            return
        with self._lock:
            self._auto_seen.add(mark)
        self.run_results_action("results-af3-import", [f"af3_path={found}"])
        with self._lock:
            self._results_version += 1

    #: How often the download folders are walked, however often the page asks.
    AUTO_POLL_SECONDS = 2.0
    #: How recently a download must have been written to be this fold.
    AUTO_WINDOW_SECONDS = 3600.0

    def note_closing(
        self,
        lines: list[str],
        screens: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        """Keep the screen sentences, and the records behind them, for the closed page."""
        if not lines and not screens:
            return
        with self._lock:
            self._closing.extend(lines)
            if screens:
                self._screens.extend(screens)

    def note_result(self, sentence: str) -> None:
        """The outcome of the last results-page button, replacing the one before.

        Its own field rather than ``note_closing``, which only ever extends
        ``_closing``: routed through there, every click left another paragraph
        on the page and the tenth told the reader nine stale things. It lives
        on the session rather than being an argument to :meth:`done_page`
        because the request handler calls that method bare -- for ``GET
        /closed`` and for the response after close -- where an argument would
        simply be lost.
        """
        with self._lock:
            self._action_result = sentence.strip()

    def run_results_action(self, action: str, fields: list[str] | None = None) -> str:
        """Carry out one results-page action and return a sentence about it.

        Answerable from session state alone, which is the constraint this page
        works under: ``finish_choice`` runs after the generation call returned,
        so there is no ``args``, no ``out_dir`` and no disease to consult. What
        it does have is the structures behind the viewers and the chain each
        one calls the binder.
        """
        for entry in fields or ():
            if entry.startswith("results-tab="):
                asked = entry.removeprefix("results-tab=").strip()
                with self._lock:
                    self._results_tab = asked if asked in RESULTS_TABS else "binder"

        if action in {VALIDATION_ACTION, VALIDATION_SCOPE_ACTION, SCREENING_ACTION}:
            return self._run_validation_action(action, fields or ())

        from rfproteina.disease.results_actions import perform

        with self._lock:
            catalog = [dict(item) for item in self._catalog]
            earlier = dict(self._action_record)
        done = perform(action, catalog, self.structure_bytes, fields or [], held=earlier)
        token = self.token_for_text(done.payload) if done.payload else ""
        with self._lock:
            self._payload = done.payload
            self._payload_name = done.payload_name
            self._payload_token = token
            if done.record is not None:
                # Kept across the post, and merged rather than replaced: the
                # protocol's two buttons each fill a different half of the tab,
                # and a comparison that dropped the proposal cleared the tick
                # boxes the reader had just used.
                self._action_record = {**self._action_record, **dict(done.record)}
        for which, text, detail in done.overlays:
            token = self.token_for_text(text) if text else ""
            with self._lock:
                for item in self._catalog:
                    item[f"{which}_token"] = token
                    item[f"{which}_detail"] = detail
        return done.sentence

    def _run_validation_action(self, action: str, fields: Sequence[str]) -> str:
        scope_field = f"{VALIDATION_SCOPE_FIELD}="
        scopes = [entry.removeprefix(scope_field) for entry in fields if entry.startswith(scope_field)]
        with self._lock:
            directories = self._validation_runs
            if not directories:
                raise ValueError("There is no finished run to validate.")
            chosen = "all"
            if len(directories) > 1:
                if len(scopes) != 1 or scopes[0] not in dict(_validation_scope_options(directories)):
                    raise ValueError("Choose a finished run from the list.")
                chosen = scopes[0]
            if chosen != self._validation_scope:
                self._validation_scope = chosen
                self._validation_candidate = None
                self._validation_selection = None
                self._validation_report.clear()
                self._screen_method_selection = None
                self._screen_method_results.clear()
                return "Recommendations refreshed for the selected run scope."
            if chosen != "all":
                directories = (directories[int(chosen)],)
            candidate_key = self._validation_candidate

        candidate_field = f"{VALIDATION_CANDIDATE_FIELD}="
        posted = [entry.removeprefix(candidate_field) for entry in fields if entry.startswith(candidate_field)]
        if len(posted) > 1 or (posted and len(directories) != 1):
            raise ValueError("Choose an available candidate from the list.")
        chosen_candidate = candidate_key
        if posted:
            chosen_candidate = None if posted[0] == "all" else posted[0]
        if chosen_candidate != candidate_key or action == VALIDATION_SCOPE_ACTION:
            if chosen_candidate is not None:
                _, choices = candidate_validation_plan(directories, chosen_candidate)
                if not choices:
                    raise ValueError("Choose an available candidate from the list.")
            with self._lock:
                self._validation_candidate = chosen_candidate
                self._validation_selection = None
                self._validation_report.clear()
                self._screen_method_selection = None
                self._screen_method_results.clear()
            return "Recommendations refreshed. Review the selected checks before running them."

        if action == SCREENING_ACTION:
            from rfproteina.disease import screen_panel

            if len(directories) != 1:
                raise ValueError("Choose one finished run for scientific screening.")
            prefix = f"{SCREENING_FIELD}="
            methods = tuple(entry.removeprefix(prefix) for entry in fields if entry.startswith(prefix))
            completed = 0

            def progress(label: str) -> None:
                nonlocal completed
                if not label.startswith("Running "):
                    completed += 1
                self.note_progress(completed, len(methods), label, SCREENING_ACTION)

            results = screen_panel.run(directories[0], candidate_key, "explicit", methods, on_line=progress)
            with self._lock:
                self._screen_method_selection = methods
                self._screen_method_results = results
            if not results:
                return "No scientific screens selected."
            return f"Returned {len(results)} screening result{'s' if len(results) != 1 else ''}."

        field_prefix = f"{VALIDATION_FIELD}="
        selected = tuple(entry.removeprefix(field_prefix) for entry in fields if entry.startswith(field_prefix))
        report = run_validation_checks(directories, selected, candidate_key=candidate_key)
        with self._lock:
            self._validation_selection = selected
            self._validation_report = report
        if not report:
            return "Select at least one installed validation check."
        return f"Completed {len(report)} validation {'checks' if len(report) > 1 else 'check'}."

    def note_validation_runs(self, directories: Sequence[Path], *, standalone: bool = False) -> None:
        """Register local directories from the completed requests."""
        with self._lock:
            self._validation_runs = tuple(dict.fromkeys(directories))
            self._validation_only = standalone
            self._validation_scope = "0" if len(self._validation_runs) > 1 else "all"
            self._validation_candidate = None
            self._validation_selection = None
            self._validation_report.clear()
            self._screen_method_selection = None
            self._screen_method_results.clear()

    def note_structures(self, trios: Sequence[StructureTrio]) -> None:
        """Register the files behind the three views. A missing file is left out."""
        with self._lock:
            for trio in trios:
                disease = self._token_for(trio.disease_file)
                complex_token = self._token_for(trio.complex_file)
                if disease is None or complex_token is None:
                    continue
                self._catalog.append(catalog_entry(trio, disease, complex_token))

    def structure_bytes(self, token: str) -> bytes | None:
        """The CIF registered under ``token``. Any other token is refused."""
        if _STRUCTURE_TOKEN.fullmatch(token) is None:
            return None
        with self._lock:
            held = self._text_bytes.get(token)
            if held is not None:
                return held
            path = self._files.get(token)
            if path is None:
                return None
            cached = self._file_bytes.get(token)
        if cached is not None:
            return cached
        try:
            if path.name.endswith((".zst", ".gz", ".gzip")):
                # The browser's viewer parses CIF text, so a compressed design
                # has to be decompressed here or it renders nothing. A plain
                # file is still served as its exact bytes, with no round trip
                # through a decode.
                text = cif_text(path)
                data = None if text is None else text.encode("utf-8")
            else:
                data = path.read_bytes()
        except OSError:
            return None
        if data is None:
            return None
        with self._lock:
            if token not in self._files:
                return None
            self._file_bytes[token] = data
        return data

    def token_for_text(self, text: str) -> str | None:
        """A token serving ``text`` straight from memory.

        A retrieved model may never have been a file: the paste route exists
        because it asks nothing of the machine, and the viewer fetches by token
        rather than by path. So the bytes are held here under a token of the
        same shape, and ``structure_bytes`` finds them before it looks for a
        path. Nothing is written to disk, which is also the right answer for a
        model a person pasted in: it was not ours to keep.
        """
        body = text.encode("utf-8")
        if not body:
            return None
        with self._lock:
            token = secrets.token_hex(8)
            while token in self._files or token in self._text_bytes:
                token = secrets.token_hex(8)
            self._text_bytes[token] = body
            return token

    def _token_for(self, path: Path) -> str | None:
        """A token for ``path``. The caller holds the session lock."""
        try:
            resolved = path.resolve()
        except OSError:
            return None
        if not resolved.is_file():
            return None
        token = self._path_tokens.get(resolved)
        if token is not None:
            return token
        token = secrets.token_hex(8)
        while token in self._files:
            token = secrets.token_hex(8)
        self._files[token] = resolved
        self._path_tokens[resolved] = token
        return token

    def done_page(self) -> str:
        """The last page. A recorded screen is shown under the closed heading.

        Close asks for the short page that shuts the panel window. Until that
        flag is set, this is the finished run. Generated structures are drawn
        above that screen.
        """
        with self._lock:
            if self._shutdown:
                return _shutdown_page()
            lines = list(self._closing)
            screens = list(self._screens)
            catalog = list(self._catalog)
            result = self._action_result
            payload = self._payload
            payload_name = self._payload_name
            payload_token = self._payload_token
            active = self._results_tab
            action_record = dict(self._action_record)
            directories = self._validation_runs
            validation_only = self._validation_only
            scope = self._validation_scope
            candidate_key = self._validation_candidate
            selected = self._validation_selection
            validation_report = dict(self._validation_report)
            method_selection = self._screen_method_selection
            method_results = list(self._screen_method_results)
        structures = render_structure_views(catalog)
        validation = ""
        if directories:
            included = directories if scope == "all" else (directories[int(scope)],)
            candidates, choices = candidate_validation_plan(included, candidate_key)
            validation = _validation_panel(
                choices, selected, validation_report, directories, scope, candidates, candidate_key
            )
            if len(included) == 1 and choices:
                from rfproteina.disease import screen_panel

                methods = screen_panel.choices_for(included[0], candidate_key)
                validation += _scientific_screen_panel(methods, method_selection, method_results)
        if not lines and not screens and not structures and not validation:
            return _done_page()
        board = render_closing(lines, screens) if lines or screens else ""
        # In the body, never the actions row: ``_page`` keeps Close as the last
        # submit control, and both the page script's Enter handling and a test
        # depend on that ordering.
        tools = _results_tools(result, payload, payload_name, payload_token, active, action_record) if catalog else ""
        notice = f'<p class="screen-readout">{_esc(result)}</p>' if validation and result and not catalog else ""
        body = structures + tools + notice + validation + board
        return _page(PANEL_TITLE, body, "Close", restart=not validation_only)

    def close(self) -> None:
        """Unblock the waiting page, close the panel window, and stop the server."""
        if self._server is None:
            return
        with self._lock:
            self._shutdown = True
            self._next_html = None
            self._closed = True
        self._next_ready.set()
        self._result.set()
        self._sent.wait(timeout=5)
        self._close_window()
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None

    def _close_window(self) -> None:
        """End the window this session opened. Other browser windows stay up."""
        port = self._window_port
        if not port or not (_running_in_wsl() or sys.platform.startswith("win")):
            return
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", _close_script(port)],
            cwd=_windows_cwd(),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            print("The panel window was left open.")

    def page(self) -> str:
        with self._lock:
            return self._html

    def remember(self, html: str) -> None:
        """Keep the page a later refresh should show."""
        if not html:
            return
        with self._lock:
            self._html = html

    def page_for_get(self) -> str:
        """The stored page, or the main page when nothing is stored."""
        with self._lock:
            html = self._html
        return html or _main_page()

    def _ensure(self) -> None:
        if self._server is not None:
            return
        session = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path != "/" and path.endswith("/"):
                    path = path.rstrip("/")
                if path == "/viewer/3dmol.js":
                    script = viewer_javascript()
                    if script is None:
                        self._missing()
                        return
                    self._send(script, "application/javascript")
                    return
                if path.startswith("/structure/"):
                    payload = session.structure_bytes(path.removeprefix("/structure/"))
                    if payload is None:
                        self._missing()
                        return
                    self._send(payload, "chemical/x-mmcif")
                    return
                if path == "/progress":
                    body = json.dumps(session.progress()).encode("utf-8")
                    self._send(body, "application/json")
                    return
                if path == "/closed":
                    self._send_page(session.done_page(), closed=True)
                    return
                asked = _step_in(path)
                current = session.step()
                # A mirror in front of this panel (the Cell relay) reads the step and
                # then asks for that page. The panel can move on in between, and a
                # request that arrives late must not be taken for this browser's own
                # Back. The mirror says so with a header, and gets the current page.
                rewindable = (
                    asked is not None
                    and asked < current
                    and not session.sealed()
                    and "X-Cell-No-Rewind" not in self.headers
                )
                if not rewindable and asked != current:
                    # No step to go to, or the run has started writing: show where it is.
                    self._redirect(session.step_url())
                    return
                if asked == current:
                    self._send_page(session.page_for_get())
                    return
                # The browser's own Back button reached an earlier step. The flow is
                # rewound and whatever page it comes back with is served here, so the
                # address the browser is already showing is the page it gets.
                try:
                    page = session.go_back(asked)
                    if page is None:
                        self._send_page(session.done_page(), closed=True)
                        return
                    session.remember(page)
                    self._send_page(page)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                finally:
                    # This GET answers a submission, so close may stop the server.
                    session.mark_sent()

            def do_POST(self) -> None:
                try:
                    length = _body_length(self.headers.get("Content-Length", "0"))
                    if length is None:
                        # A header that is not a length, or a length past the
                        # bound. Either way nothing is read: the old line did
                        # `self.rfile.read(int(header))`, so a client claiming
                        # four gigabytes had the panel try to allocate four
                        # gigabytes, and a header that was not a number raised
                        # out of this handler instead of answering.
                        self._refuse(413 if self.headers.get("Content-Length", "").strip().isdigit() else 400)
                        return
                    raw = self.rfile.read(length)
                    fields = parse_submission(
                        raw,
                        self.headers.get("Content-Type", ""),
                        session.upload_dir(),
                    )
                    page = session.submit(fields)
                    if page:
                        # The browser follows this to a GET of the page it should
                        # keep. A refresh then reloads that page instead of the form.
                        session.remember(page)
                        self._redirect(session.step_url())
                    else:
                        self._send_page(session.done_page(), closed=True)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                finally:
                    session.mark_sent()

            def log_message(self, format: str, *args: object) -> None:
                return

            def _send_page(self, html: str, *, closed: bool = False) -> None:
                """Send one panel page stamped with the step it belongs to."""
                self._send(_with_step(html, session.step(), closed=closed).encode("utf-8"))

            def _refuse(self, status: int) -> None:
                """Answer a submission this handler will not read, and say nothing else.

                No page, because a body that was refused unread leaves the
                session exactly as it was: redrawing the panel here would claim
                the submission had been applied.

                The body is left on the socket rather than drained, which is
                safe only because this server does not set ``protocol_version``
                and so answers HTTP/1.0: the connection closes after the
                response, and there is no later request on it to misparse. A
                keep-alive server would have to read and discard the body here,
                which is the whole cost the bound exists to avoid.
                """
                try:
                    self.send_response(status)
                    self.send_header("Content-Length", "0")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

            def _redirect(self, location: str) -> None:
                try:
                    self.send_response(303)
                    self.send_header("Location", location)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

            def _missing(self) -> None:
                """A short refusal. The body does not repeat the address or a path."""
                body = b"Not found."
                try:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

            def _send(self, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
                # The browser often closes the POST while the terminal is still
                # writing commands. That disconnect is not a failed run.
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

        def _step_in(path: str) -> int | None:
            """The step a path names, or None when it names no step."""
            prefix = "/step/"
            if not path.startswith(prefix):
                return None
            tail = path[len(prefix) :]
            return int(tail) if tail.isdigit() else None

        class PanelServer(ThreadingHTTPServer):
            def handle_error(self, request: object, client_address: object) -> None:
                error = sys.exception()
                if isinstance(error, BrokenPipeError | ConnectionResetError | ConnectionAbortedError):
                    return
                super().handle_error(request, client_address)

        self._server = PanelServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()


def render_options(
    form: OptionForm,
    notice: str = "",
    saved: tuple[str, ...] = (),
    *,
    where: str = "",
    hosts_text: str | None = None,
    slots: dict[str, Sequence[KeySlot]] | None = None,
    machines: tuple[str, ...] = (),
    machine_search: str = "",
    out_file_search: str = "",
    selected_out_file: str = "",
) -> str:
    """The control panel: dropdowns for fixed choices and tick boxes for the switches.

    ``where`` and ``hosts_text`` refill the machine controls from a submission the
    reader refused. Choosing remote machines and naming none is such a case, and
    it cannot be expressed as ``form.hosts``, so it would otherwise snap back to
    this machine while the notice asked for a remote name. ``slots`` are the saved
    keys. Each one is shown as ***** beside the model it is sent with.
    """
    uploaded = ""
    if form.target_structure:
        # A file input cannot be pre-filled, so the chosen path rides along hidden. A
        # newly chosen file is sent before it and therefore wins.
        uploaded = (
            f'<input type="hidden" name="target_structure" value="{_esc(form.target_structure)}">\n'
            f"<span>{_esc(form.target_structure)}</span>"
        )
    derived, derived_names = _where(form.hosts)
    where = where if where in dict(_WHERE) else derived
    remote_names = derived_names if hosts_text is None else hosts_text
    worker, custom = _worker_choice(form.local_workers)
    remote_worker, remote_custom = _worker_choice(form.remote_workers)
    choice = "" if form.choice is None else str(form.choice)
    # Flags can still set these. The main page no longer shows their labels.
    carried = ""
    if form.accept_only:
        carried += '<input type="hidden" name="accept_only" value="yes">\n'
    if choice:
        carried += f'<input type="hidden" name="choice" value="{_esc(choice)}">\n'
    if form.run:
        carried += '<input type="hidden" name="run" value="yes">\n'
    carried += _carried_screening(form)
    carried += _carried_research(form)
    kept_structures = ""
    if form.structures_json:
        kept_structures += f'<input type="hidden" name="structures_json" value="{_esc(form.structures_json)}">\n'
    if form.take_all:
        kept_structures += '<input type="hidden" name="take_all" value="yes">\n'
    if form.pick:
        kept_structures += f'<input type="hidden" name="pick" value="{_esc(form.pick)}">\n'
    shown_out = display_directory(form.out_dir)
    shown_download = display_directory(form.out_dir if form.download_dir is None else form.download_dir)
    # Named, not created. Drawing this page must not write into the folder on
    # screen: the file list offers a `parent` row, so looking around with `..`
    # would otherwise leave a `generations` directory in every folder visited.
    # `save_generation` creates the archive when a generation is actually saved.
    shown_archive = display_directory(str(generation_archive_root(form.out_dir)))
    stored_models = load_model_choices()
    labels = dict(_PROVIDER_LABELS)
    key_rows = slots or {}
    key_html = "\n".join(
        _api_key_block(provider, labels[provider], key_name, saved, stored_models, key_rows.get(provider, ()))
        for provider, key_name in _API_KEY_FIELDS
    )
    body = f"""
<div class="grid">
<div class="stack">
<section class="panel-disease">
<h2>Disease</h2>
<div class="row">
<label class="field">Disease or Biological Condition
<input type="text" name="disease" value="{_esc(form.disease or "")}" required>
<span>Scientific or medical name.</span>
</label>
<label class="field">Catalog JSON
<input type="text" name="catalog" value="{_esc(form.catalog or "")}">
<span>Optional local file. Leave blank to search the disease name.</span>
</label>
</div>
<div class="row">
<label class="field">Upload Individual Custom Disease to Investigate - Atomic level Shape .PDB .CIF files
<input type="file" name="target_structure" accept=".pdb,.cif,.PDB,.CIF">
{uploaded}</label>
<div class="ticks">
<label><input type="checkbox" name="confirm_inference" value="yes"{_on(form.confirm_inference)}>
Confirm inference without another prompt (useful for uploaded targets)</label>
</div>
</div>
{carried}</section>
<section class="panel-binder">
<h2>Binder and model</h2>
<div class="row">
<label class="field">Binder kind
<select name="binder_kind">
{_select_options(((kind, kind) for kind in BINDER_KINDS), form.binder_kind)}
</select>
<span>Which liability axes the request marks clear.</span>
</label>
<label class="field">Modified checkpoint
<input type="text" name="modified_ckpt" value="{_esc(form.modified_ckpt or "")}">
<span>Leave blank to use the original release model.</span>
</label>
</div>
</section>
<section class="panel-api">
<h2>API keys</h2>
<p class="hint">Optional. Each key is tied to the model beside it.
A saved key is shown as ***** and is stored encrypted on this computer for the next run.
It is not written into the command, the job file, or the research report.
Add another key for the same provider when a different model should use a different key.</p>
<div class="llm-list">
{key_html}
</div>
</section>
<section class="panel-machines">
<h2>Machines</h2>
<div class="row">
<label class="field">Where generation runs
<select name="where">
{_select_options(_WHERE, where)}
</select>
<span>There is no limit on how many remote machines you name.</span>
</label>
<label class="field">Remote directory
<input type="text" name="remote_dir" value="{_esc(form.remote_dir)}" required>
<span>Directory on each remote machine for inputs and designs.</span>
</label>
</div>
<label class="field">Remote machines
<textarea name="hosts">{_esc(remote_names)}</textarea>
<span>Comma-separated names. local is this machine and is added from the dropdown.</span>
</label>
<div class="row">
<label class="field">Search for remote machines
<input type="text" name="machine_search" value="{_esc(machine_search)}">
<span>Part of a name narrows the list below. Nothing is probed over the network.</span>
</label>
</div>
<div class="machine-actions">
<button type="submit" class="nav" name="__action" value="search" formnovalidate>Search machines</button>
<button type="submit" class="nav" name="__action" value="browse-machine" formnovalidate>Windows File Explorer</button>
</div>
<p class="hint">Select a remote machine in Windows File Explorer. Its name is added to the box above.</p>
<p class="hint">{_esc(_machine_hint(machines, machine_search))}</p>
<div class="ticks">
{_machine_ticks(machines)}
</div>
<div class="row workers">
<label class="field">
<span class="caption">Local workers (structures to bind, locally)</span>
<select name="local_workers">
{_select_options(_WORKERS, worker)}
</select>
<span>The limit applies only to this machine.</span>
</label>
<label class="field counter">
<span class="caption">Custom local workers</span>
<input type="number" name="local_workers_custom" min="1" step="1" value="{_esc(custom)}">
<span>Used when the dropdown says Custom.</span>
</label>
</div>
<div class="row workers">
<label class="field">
<span class="caption">Remote workers (structures to bind, remotely)</span>
<select name="remote_workers">
{_select_options(_REMOTE_WORKERS, remote_worker)}
</select>
<span>Used when generation runs on remote machines. A name repeated in the box
above is a second worker on that machine, which is how to say how many run on
one of them.</span>
</label>
<label class="field counter">
<span class="caption">Custom remote workers</span>
<input type="number" name="remote_workers_custom" min="1" step="1" value="{_esc(remote_custom)}">
<span>Used when the dropdown says Custom.</span>
</label>
</div>
</section>
</div>
<div class="stack">
<section class="panel-structures">
<h2>Structures Directory</h2>
<div class="row">
<label class="field">Output directory
<input type="text" name="out_dir" value="{_esc(shown_out)}" required>
<span>Where job files and designs are written.</span>
</label>
<label class="field">Download directory
<input type="text" name="download_dir" value="{_esc(shown_download)}">
<span>Where structure files are downloaded.</span>
</label>
</div>
<label class="field">Generation archive
<input type="text" name="generation_archive" value="{_esc(shown_archive)}" readonly>
<span>Finished generations are saved here by disease name, then target, for later search or training.</span>
</label>
{_output_file_browser(form.out_dir, out_file_search, selected_out_file)}
{kept_structures}</section>
</div>
</div>
"""
    alert = f'<p class="notice">{_esc(notice)}</p>\n' if notice else ""
    return _page(PANEL_TITLE, alert + body, "Continue")


def _machine_ticks(machines: tuple[str, ...]) -> str:
    """One tick box per discovered machine. A tick joins the names in the text box."""
    if not machines:
        return ""
    return "\n".join(
        f'<label><input type="checkbox" name="remote_pick" value="{_esc(name)}">\n{_esc(name)}</label>'
        for name in machines
    )


def _machine_hint(machines: tuple[str, ...], query: str) -> str:
    """What the list is, and why it may be empty."""
    if machines:
        return (
            f"{len(machines)} machine(s) from your ssh config and known_hosts. " "Ticking one adds it to the box above."
        )
    if query:
        return f"No known machine matches {query}. Clear the search to see them all."
    return "No machines found in your ssh config or known_hosts. Type names in the box above."


def remote_names_from_fields(fields: dict[str, list[str]]) -> list[str]:
    """Remote names from the text box and the ticked list, in that order.

    This machine is never one of them: the dropdown is what adds it.
    """
    names: list[str] = []
    typed = _split_hosts(_first(fields, "hosts"))
    picked = [value.strip() for value in fields.get("remote_pick", []) if value.strip()]
    for name in [*typed, *picked]:
        if name.casefold() in LOCAL_NAMES:
            continue
        if name not in names:
            names.append(name)
    return names


def _confirm_options(
    disease: Disease,
    alternatives: Sequence[Disease] = (),
    query: str = "",
) -> list[Disease]:
    """The matched disease first, then each name matching the query once.

    Filtered, not the whole catalog. An ontology search answers with the
    query's semantic neighbourhood, so a Huntington query also returns
    familial Alzheimer disease and mild Canavan disease -- true neighbours,
    and noise under a heading that reads Matching names. The filter is
    ``related_names``, the same substring test ``match_diseases`` applies
    once an exact name has not won.

    ``query`` falls back to the disease's own name, so a caller with no query
    still gets the siblings sharing the confirmed name rather than the whole
    catalog.
    """
    options = [disease]
    seen = {disease.disease_id}
    for other in related_names(query or disease.name, alternatives):
        if other.disease_id in seen:
            continue
        seen.add(other.disease_id)
        options.append(other)
    return options


def render_confirm(
    disease: Disease,
    alternatives: Sequence[Disease] = (),
    query: str = "",
) -> str:
    """Tick box for the single hit, plus the menu the no-tick page uses.

    The menu is shown while Use this disease is clear. The matched disease is
    the first option, and each other name matching ``query`` is listed once
    per disease id.
    """
    options = [
        (str(index), f"{item.name} ({item.disease_id})")
        for index, item in enumerate(_confirm_options(disease, alternatives, query), start=1)
    ]
    body = f"""
<section>
<h2>Confirm the disease</h2>
<p>{_esc(disease.name)} ({_esc(disease.disease_id)})</p>
<div class="ticks">
<label><input type="checkbox" name="accept" value="yes">
Use this disease</label>
</div>
<label class="field disease-choice">Matching names
<select name="choice">
{_select_options(options, "1")}
</select>
<span>This list is used to help confirm Disease</span>
</label>
</section>
"""
    return _page("Confirm disease", body, "Continue")


def render_disease_choice(hits: list[Disease]) -> str:
    """Dropdown of the similar names."""
    options = [(str(index), f"{disease.name} ({disease.disease_id})") for index, disease in enumerate(hits, start=1)]
    body = f"""
<section>
<h2>Choose the disease</h2>
<label class="field">Matching names
<select name="choice">
{_select_options(options, "1")}
</select>
</label>
</section>
"""
    return _page("Choose disease", body, "Use this disease")


def render_focus(
    disease: Disease,
    pathways: Sequence[tuple[str, str]] = (),
    targets: Sequence[tuple[str, str]] = (),
) -> str:
    """Pathway, mechanism, and an optional protein or gene for the structure search.

    The disease is already confirmed. Any pathway keeps the disease-associated
    proteins. A typed target is used instead of the menu when both are posted.
    """
    pathway_options = [("any", "Any pathway"), *pathways]
    target_options = [("", "No named target")]
    for symbol, accession in targets:
        label = symbol or accession
        if symbol and accession:
            label = f"{symbol} ({accession})"
        value = f"{symbol}|{accession}" if accession or symbol else ""
        if not value:
            continue
        target_options.append((value, label))
    body = f"""
<section class="panel-focus">
<h2>Relevant biology</h2>
<p>{_esc(disease.name)} ({_esc(disease.disease_id)})</p>
<p class="hint">{_esc(DISTINCTION)}</p>
<label class="field">Relevant biological pathway
<select name="pathway">
{_select_options(pathway_options, "any")}
</select>
<span>One disease can use more than one pathway. Any pathway keeps the disease-associated proteins.</span>
</label>
<label class="field">Potential therapeutic mechanism
<select name="mechanism">
{_select_options(MECHANISMS, "any")}
</select>
<span>These roles are different. The choice ranks the structure list.
It does not decide which protein starts the disease.</span>
</label>
<label class="field">Optional target protein or gene
<select name="target_menu">
{_select_options(target_options, "")}
</select>
<span>From the proteins associated with this disease. Leave this on No named target to skip it.</span>
</label>
<label class="field">Or type a protein or gene
<input type="text" name="target_name" value="">
<span>A typed name is used instead of the list above. Leave both blank to search without a named target.</span>
</label>
</section>
"""
    return _page("Choose a biological target", body, "Search targets")


def _research_grouped_html(saved: tuple[str, ...] = ()) -> str:
    """One tick per listed model whose provider has an API key.

    ``saved`` is that provider set, from a saved key or an environment variable.
    A provider with no key is left off the page and is not searched. Each tick is
    named the way every other part of the panel names that model, which is what
    the provider id in front of the label was not: "grok Grok 4.7".
    """
    ticks = [
        """<label><input type="checkbox" name="all_models" value="yes">
Search every listed model (With saved API Key)</label>"""
    ]
    allowed = {name.casefold() for name in saved}
    for provider, _label in _PROVIDER_LABELS:
        if provider not in allowed:
            continue
        for model_id, _model_label in model_choices(provider):
            value = f"{provider}|{model_id}"
            shown = model_display(f"{provider} {model_id}")
            ticks.append(
                "<label>"
                f'<input type="checkbox" name="model" value="{_esc(value)}" data-saved="yes">\n'
                f"{_esc(shown)}</label>"
            )
    listed = "\n".join(ticks)
    return f'<div class="ticks llm-search">\n{listed}\n</div>\n{_grok_bot_html()}'


def _grok_bot_html() -> str:
    """Grok Bot, listed with no API key and outside the saved-key select-all.

    The select-all box only ticks models inside ``.llm-search``. This one stays
    beside that list so "every listed model" still means every model with a key.
    Submit question is what pastes the box to Chief of Staff.
    """
    value = f"{GROK_BOT}|{GROK_BOT_MODEL}"
    return (
        '<div class="ticks llm-grok-bot">\n'
        "<label>"
        f'<input type="checkbox" name="model" value="{_esc(value)}" data-saved="no">\n'
        "Grok Bot</label>\n"
        '<p class="hint">No API key. Submit question pastes this box into the open '
        "Chief of Staff chat in Grok Bot and returns the reply here.</p>\n"
        "</div>"
    )


def _structure_view_controls(
    extra: str = "",
    models: Sequence[tuple[str, str]] | None = None,
    *,
    sort: str = "listed",
    biological: bool = False,
) -> str:
    """Sort and include/exclude, the same controls the structures page carries.

    "Score" is the different-databases count the hint above the rows describes,
    and is the wording this page has always used for it. The added orders name
    their own metric in the words the rows already use: the same-databases
    instance count, and the value one model or Grok Bot gave a target. That
    choice is the model list beside the sort. ``extra`` is appended inside the
    block, which is how the ranking filter joins the kind filters without a
    second block the page script would not find.
    """
    listed = _rank_subject_options() if not models else list(models)
    options = "\n".join(
        f'<option value="{_esc(value)}"{"" if index else " selected"}>{_esc(label)}</option>'
        for index, (value, label) in enumerate(listed)
    )
    sort_rows = [("listed", "Listed order")]
    if biological:
        sort_rows.append(("biological", "Biological rank"))
    sort_rows.extend(
        (
            ("high", "Score highest"),
            ("low", "Score lowest"),
            ("same-high", "Instances in same databases highest"),
            ("same-low", "Instances in same databases lowest"),
            ("value-high", "LLM higher value"),
            ("value-low", "LLM lower value"),
        )
    )
    sort_html = "\n".join(
        f'<option value="{_esc(value)}"{(" selected" if value == sort else "")}>{_esc(label)}</option>'
        for value, label in sort_rows
    )
    return f"""<div class="structure-view">
<label>Sort
<select name="sort">
{sort_html}
</select>
</label>
<label>LLM model or Grok Bot
<select name="rank_model">
{options}
</select>
<span>LLM higher value, LLM lower value, and LLM ranking use this model or Grok Bot.</span>
</label>
<label>Experimental structures
<select name="experimental">
<option value="include" selected>Include</option>
<option value="exclude">Exclude</option>
</select>
</label>
<label>Predicted structures
<select name="predicted">
<option value="include" selected>Include</option>
<option value="exclude">Exclude</option>
</select>
</label>
{extra}
</div>"""


def _database_rows(listed: list[dict[str, object]]) -> str:
    """One row per structure, naming every database that answered for its target."""
    if not listed:
        return '<div class="ticks structure-rows"></div>'
    rows = []
    for item in listed:
        databases = item.get("databases")
        answered = (
            ", ".join(f"{SOURCE_LABELS[name]} {(databases or {}).get(name)}" for name in SOURCE_ORDER)
            if isinstance(databases, dict)
            else ""
        )
        label = (
            f"{item.get('number')}. PDB {item.get('entry_id')} chain {item.get('chain')} "
            f"{item.get('accession')} {item.get('title')} | {item.get('found')} of "
            f"{item.get('asked')} | {answered}"
        )
        rows.append(
            f'<label data-kind="{_esc(str(item.get("kind") or ""))}" '
            f'data-score="{_esc(str(item.get("found") or 0))}" '
            f'data-number="{_esc(str(item.get("number") or 0))}">{_esc(label)}</label>'
        )
    listed_html = "\n".join(rows)
    return f'<div class="ticks structure-rows">\n{listed_html}\n</div>'


def _model_rows(by_model: dict[str, list[dict[str, object]]]) -> str:
    """Each model's structures, highest value to generate against first."""
    if not by_model:
        return "<p>(no model rated a structure)</p>"
    blocks = []
    for model, rows in by_model.items():
        items = "\n".join(
            f"<li>{_esc(str(position))}. PDB {_esc(str(row.get('entry_id')))} chain "
            f"{_esc(str(row.get('chain')))} {_esc(str(row.get('accession')))} rating "
            f"{_esc(str(row.get('rating')))}, from the {_esc(str(row.get('source')))} search</li>"
            for position, row in enumerate(rows, start=1)
        )
        blocks.append(f"<h3>{_esc(model)}</h3>\n<ol>\n{items}\n</ol>")
    return "\n".join(blocks)


def render_option_comparison(
    comparison: dict[str, list[dict[str, str]]],
    *,
    by_database: list[dict[str, object]] | None = None,
    by_model: dict[str, list[dict[str, object]]] | None = None,
) -> str:
    """The program structure search beside the structures the selected models named.

    ``by_database`` lists every structure against the asked databases, and
    ``by_model`` lists every structure under each model that rated it, highest
    value first. The sort and include/exclude controls are the structures page's,
    and the same page script drives them.
    """

    def block(title: str, key: str) -> str:
        rows = comparison.get(key) or []
        if not rows:
            items = "<li>(none)</li>"
        else:
            items = "\n".join(f"<li>{_esc(str(item.get('label') or ''))}</li>" for item in rows)
        return f"<h3>{_esc(title)}</h3>\n<ul>\n{items}\n</ul>"

    confirmed = block("Confirmed program options", "confirmed")
    body = f"""
<section>
<h2>Program search and LLM options</h2>
<p class="hint">Confirmed program options are downloaded. When none were confirmed,
the download is the structures the models named, or each target's experimental
rows when they named none. The program side is the returned structure list
saved for this comparison.</p>
{confirmed}
{block("In both", "in_both")}
{block("Only in the program search", "only_program")}
{block("Only in the LLM options", "only_llm")}
</section>
<section>
<h2>Structures by database</h2>
{_structure_view_controls(models=_rank_subject_options(extra=tuple(by_model or {})))}
{_database_rows(by_database or [])}
</section>
<section>
<h2>Structures by model</h2>
{_model_rows(by_model or {})}
</section>
"""
    return _page("Compare options", body, "Continue")


def render_targets(ranked: list[dict[str, object]]) -> str:
    """One tick box per ranked target. The ticked targets go to generation."""
    ticks = [
        """<label><input type="checkbox" name="all_targets" value="yes">
Every ranked target</label>"""
    ]
    for position, entry in enumerate(ranked, start=1):
        identifier = str(entry.get("identifier") or "")
        ticks.append(
            f'<label><input type="checkbox" name="target" value="{_esc(identifier)}">\n'
            f"{_esc(_target_label(position, entry))}{_target_marks(entry)}</label>"
        )
    joined = "\n".join(ticks)
    body = f"""
<section>
<h2>Targets for binder generation</h2>
<p class="hint">Highest combined value first. Combined is the public-record value moved
toward what the providers rated the target. Ticking a target selects every structure the
list holds for it. Leave every box clear to pick structures by number instead.</p>
<div class="ticks">
{joined}
</div>
</section>
"""
    return _page("Choose targets", body, "Use these targets")


def _target_label(position: int, entry: dict[str, object]) -> str:
    rating = entry.get("model_rating")
    stated = "no provider rating" if rating is None else f"providers {rating}"
    ratings = entry.get("ratings")
    if isinstance(ratings, dict) and ratings:
        stated += " (" + ", ".join(f"{name} {ratings[name]}" for name in sorted(ratings)) + ")"
    label = str(entry.get("label") or "").strip()
    head = f"{position}. {entry.get('identifier')}"
    if label:
        head += f" {label}"
    return (
        f"{head} | combined {entry.get('combined')} | records "
        f"{entry.get('program_score')} of {entry.get('sources_asked')} | {stated} | "
        f"{entry.get('structures')} of {entry.get('structures_total')} structure(s)"
    )


def _target_marks(entry: dict[str, object]) -> str:
    marks = []
    if not entry.get("structures"):
        marks.append("no structure listed")
    elif not entry.get("experimental_structures"):
        marks.append("predicted only")
    if entry.get("agreement") == "disputed":
        marks.append("providers disagreed")
    if not marks:
        return ""
    return f' <span class="badge">{_esc(", ".join(marks))}</span>'


def research_from_fields(
    fields: dict[str, list[str]],
    saved: Sequence[str] = (),
    *,
    include_grok_bot: bool = False,
) -> tuple[list[str], dict[str, tuple[str, ...]] | None]:
    """Providers and listed models from the structures page.

    Only providers in ``saved`` are searched. Those are the ones with an API key.
    The every-model tick asks every listed model of that set. A ticked model whose
    provider has no key is ignored. An empty provider list leaves the search off.
    Grok Bot is the exception when ``include_grok_bot`` is set: it has no key, and
    only Submit question asks for it. A different model id on that tick is ignored.
    """
    allowed = tuple(name for name in PROVIDERS if name in {item.casefold() for item in saved})
    tokens = []
    desktop = False
    for token in fields.get("model", []):
        text = token.strip()
        if not text or "|" not in text:
            if text:
                tokens.append(text)
            continue
        provider, model_id = (part.strip() for part in text.split("|", 1))
        provider = provider.casefold()
        if provider == GROK_BOT:
            if model_id == GROK_BOT_MODEL:
                desktop = True
            continue
        if provider in allowed:
            tokens.append(text)
    every = _checked(fields, "all_models") or _checked(fields, "saved_models")
    if every:
        # The tick already means every keyed model. A stray "all" token must not
        # widen that to providers that have no key.
        names, plan = search_plan((), [item for item in tokens if "|" in item], saved_providers=allowed)
    else:
        names, plan = search_plan((), tokens)
    names = [name for name in names if name in allowed]
    if include_grok_bot and desktop and GROK_BOT not in names:
        names.append(GROK_BOT)
    if plan is not None:
        plan = {name: models for name, models in plan.items() if name in allowed}
    return names, plan or None


def render_inference(
    count: int,
    *,
    modified: str = "",
    confirmed: bool = False,
    hosts: str | None = None,
    local_workers: int | None = None,
    remote_workers: int | None = None,
) -> str:
    """The last decisions before any generation command exists.

    This was a terminal prompt, which a panel run could not see and could not
    answer, so it is a page like every other step.

    ``modified`` is the modified checkpoint this run found, and is empty when it
    found none. Without a checkpoint there is nothing for that model to load, so
    the page holds that choice back and says why rather than offering one it
    cannot carry out. With one, the modified model is the selected choice, because
    that is what naming a checkpoint has always meant.

    ``hosts``, ``local_workers`` and ``remote_workers`` pre-fill the Machines
    block. The two worker counts are offered here as well as on the first page
    because this is the last page before anything runs and they are the only
    machine settings that can still honestly be changed: they are read when
    ``execute_generation`` is called, after the job files and the per-host
    commands have been written. Which machines run cannot be changed here for
    the same reason -- those names are already baked into the written jobs -- so
    it is shown and not offered.
    """
    ready = bool(modified)
    held = "" if ready else " disabled"
    local_choice, local_custom = _worker_choice(local_workers)
    remote_choice, remote_custom = _worker_choice(remote_workers)
    # ``confirmed`` is the first page's tick. It answers the inference question
    # rather than skipping this page, because the model is chosen here and
    # generation is started here: skipping it did both silently.
    if confirmed:
        opening = (
            f"Inference confirmed for {count} downloaded structure(s). Continue writes the job "
            "files, prints the generation commands, and starts them, on this machine and on "
            "every machine named on the first page."
        )
        asked = '<input type="hidden" name="accept" value="yes">'
        closing = "The downloads are kept whether or not generation starts."
    else:
        opening = (
            f"{count} structure(s) were downloaded. Keep the box ticked and Continue writes the "
            "job files, prints the generation commands, and starts them, on this machine and on "
            "every machine named on the first page."
        )
        asked = (
            '<label><input type="checkbox" name="accept" value="yes" checked>\n'
            f"Run inference on {count} downloaded structure(s)</label>"
        )
        closing = "Clear the box to stop here. The downloads are kept either way."
    note = (
        f"Modified Low Toxic loads {_esc(modified)}."
        if ready
        else "Modified Low Toxic needs a checkpoint file, and none was named on the first page, "
        "so the original is the one model this run can generate with."
    )
    body = f"""
<section>
<h2>Inference</h2>
<p class="hint">{opening}</p>
<div class="ticks">
{asked}
</div>
<h3>Generate with</h3>
<div class="ticks">
<label><input type="radio" name="generate_with" value="original"{_on(not ready)}>
Original RFD4</label>
<label><input type="radio" name="generate_with" value="modified"{held}{_on(ready)}>
Modified Low Toxic</label>
</div>
<p class="hint">{note} One model generates: every structure goes through the model
picked here, and the designs say which one in the report beside them.</p>
<p class="hint">{closing}</p>
</section>
<section class="panel-machines">
<h2>Machines</h2>
<p class="hint">{_esc(_where_summary(hosts))} Continue starts generation, so this is
the last page these two can be changed on.</p>
<div class="row workers">
<label class="field">
<span class="caption">Local workers (structures to bind, locally)</span>
<select name="local_workers">
{_select_options(_WORKERS, local_choice)}
</select>
<span>The limit applies only to this machine.</span>
</label>
<label class="field counter">
<span class="caption">Custom local workers</span>
<input type="number" name="local_workers_custom" min="1" step="1" value="{_esc(local_custom)}">
<span>Used when the dropdown says Custom.</span>
</label>
</div>
<div class="row workers">
<label class="field">
<span class="caption">Remote workers (structures to bind, remotely)</span>
<select name="remote_workers">
{_select_options(_REMOTE_WORKERS, remote_choice)}
</select>
<span>Used when generation runs on remote machines.</span>
</label>
<label class="field counter">
<span class="caption">Custom remote workers</span>
<input type="number" name="remote_workers_custom" min="1" step="1" value="{_esc(remote_custom)}">
<span>Used when the dropdown says Custom.</span>
</label>
</div>
<p class="hint">Which machines run, and the directory on each, are set on the first
page: the jobs are written against those names before this page is reached, so
changing them here would not change where the designs go.</p>
</section>
"""
    return _page("Confirm inference", body, "Continue")


def _where_summary(hosts: str | None) -> str:
    """One sentence naming where generation is about to run.

    Shown because the worker counts below it are meaningless without it: a
    remote cap on a local-only run bounds nothing, and a reader who cannot see
    which it is has no way to tell that from a cap that is working.
    """
    names = [name.strip() for name in (hosts or "").split(",") if name.strip()]
    remote = [name for name in names if not is_local_host(name)]
    local = any(is_local_host(name) for name in names) or not names
    if local and remote:
        return f"Generation runs on this machine and on {len(remote)} remote machine(s): {', '.join(remote)}."
    if remote:
        return f"Generation runs on {len(remote)} remote machine(s): {', '.join(remote)}."
    return "Generation runs on this machine only, so the remote limit bounds nothing."


def generation_model_from_fields(fields: dict[str, list[str]], *, modified: str = "") -> str:
    """Which of the two models the start-generation page asked to generate with.

    One radio, so one value arrives. A page that was never asked -- a flag run,
    or a client that posted only the tick boxes -- leaves the field out, and then
    the model is the one naming a checkpoint has always selected.

    A request for the modified model without a checkpoint is refused rather than
    answered with the release model, because the two are not interchangeable:
    whichever one ran is the one the designs came from.
    """
    default = "modified" if modified else "original"
    picked = (_first(fields, "generate_with") or default).strip().casefold()
    if picked not in GENERATION_MODELS:
        listed = ", ".join(GENERATION_MODELS)
        raise ValueError(f"Unknown generation model {picked!r}. Expected one of {listed}.")
    if picked == "modified" and not modified:
        raise ValueError("This run has no modified checkpoint, so only the original model can generate.")
    return picked


_CAMPAIGN_NOTES = {
    "Proteins": "General protein-to-protein interfaces, such as oncological targets like PD-L1.",
    "DNA & RNA": "Nucleic acid binding complexes.",
    "Small Molecules": "Binding pockets around therapeutic drug targets or chemical substrates.",
    "Sugars (Glycans)": "Complex carbohydrates.",
    "Antibodies & Nanobodies": "Structural motifs that act as immune-system binders.",
}


def render_structures(
    rows: list[StructureOption],
    saved: tuple[str, ...] = (),
    scores: dict[str, RecordScore] | None = None,
    *,
    ranking: list[dict[str, object]] | None = None,
    results: list[dict[str, object]] | None = None,
    notice: str = "",
    transcript: str = "",
    focus: BiologicalFocus | None = None,
) -> str:
    """Structure ticks, campaign counts, and the model ticks at the top.

    A question box is the first thing in the body. Submit question sends that
    text to the ticked models, and each reply is written back into the same box
    so earlier replies stay. Add selected structures writes the ticked rows into
    that box before the message is sent. A ticked search or research reply is
    sent again with whatever further question is in the box. ``transcript`` is
    that text.

    Both searches are live from the start. The second ranks every structure found,
    and the program search has already found the rows on this page, so it has
    something to rank whether or not the first search has added anything.
    ``ranking`` is the second search's result, which puts the rows in value order
    and shows each row's rating. ``results`` is every structure either search
    named or rated; it is listed under the search with the ID# its own tick
    carries, and it drives the model filter above the rows.

    A structure a model named joins the program's own target list, so there is one
    list of targets and not two. What each search reported is posted below the LLM
    search, under the buttons that ran it: the named and rated list, then the
    ranking's written review. When neither search returned anything, that spot
    says ``no results LLM search not yet requested.``.

    The search and those readouts come first on the page, above the rows. Running
    a search re-shows this page, and with the readouts under the list the answer
    arrived hundreds of rows below the button that asked for it.

    The count beside each design class is how many of these rows can start that
    kind of design. A row carries two counts: every entry the databases hold for
    that structure, and how many distinct databases returned anything. The model
    list is the LLM search for this page.
    """
    gate = f'<p class="notice">{_esc(notice)}</p>' if notice else ""
    values: dict[int, dict[str, object]] = {}
    for item in ranking or []:
        if not isinstance(item, dict):
            continue
        number = item.get("number")
        if isinstance(number, int) and not isinstance(number, bool):
            values[number] = item
    # A ranked page lists rows in value order while every tick keeps its own number,
    # so the numbers a later --pick names do not move under the user.
    numbered = list(enumerate(rows, start=1))
    order = [number for number in (values or {}) if 1 <= number <= len(rows)]
    if order:
        # A model rating still decides the order. The biological role stays on the label.
        order.sort(key=lambda number: list(values).index(number))
        placed = set(order)
        numbered = [(number, rows[number - 1]) for number in order]
        numbered += [(number, row) for number, row in enumerate(rows, start=1) if number not in placed]
    elif focus is not None:
        numbered = display_rank(rows, focus)
    ticks = [
        """<label><input type="checkbox" name="all" value="yes">
All Structure</label>"""
    ]
    # Which models named or rated each row, keyed by listed number, so the model
    # filter has something to read. A metric that reaches the label text only
    # cannot be sorted on, so each one is emitted as a data attribute as well.
    row_models = _models_by_number(results)
    # Models that returned a ranking. Every target shows each of them, including a
    # target that model did not rank. The database counts above are not reused here.
    asked_models = _ranking_models(ranking)
    for number, row in numbered:
        mark = ' <span class="badge">predicted</span>' if row.predicted else ""
        label = f"{number}. {row.entry_id} chain {row.chain} {row.accession} {row.title}"
        role = row_role(row, focus) if focus is not None else ""
        if role:
            label += f" | {role}"
        rank_attr = f' data-rank="{rank_score(row, focus)}"' if focus is not None else ""
        score = (scores or {}).get(row.accession.strip().casefold())
        score_attr = ""
        instance_attr = ""
        score_text = ""
        if score is not None:
            instances, databases, asked = score_parts(score)
            score_text = (
                f" | Instances of Structure in same databases {instances}"
                f" | Instances of Structure in different databases {databases}/{asked}"
            )
            score_attr = str(databases)
            instance_attr = str(instances)
        rated = values.get(number)
        value_attr = ""
        rankings_attr = ""
        rank_text = ""
        if rated is not None and rated.get("value") is not None:
            value_attr = str(rated.get("value"))
        if asked_models:
            marks = rated.get("ratings") if isinstance(rated, dict) and isinstance(rated.get("ratings"), dict) else {}
            phrases: list[str] = []
            pairs: list[str] = []
            for model in asked_models:
                rating = marks.get(model) if model in marks else None
                phrases.append(_llm_ranking_phrase(model, rating))
                pairs.append(f"{model}=" + ("" if rating is None else str(rating)))
            # The rank sits on the structure itself. The database counts follow it,
            # so a long count does not push the ranking off the row the user reads.
            rank_text = " | " + " | ".join(phrases)
            rankings_attr = "|".join(pairs)
        label += rank_text + score_text
        classes = "|".join(structure_campaigns(row))
        ticks.append(
            f'<label data-kind="{structure_kind(row)}" data-score="{score_attr}" '
            f'data-instances="{instance_attr}" data-value="{value_attr}"{rank_attr} '
            f'data-rankings="{_esc(rankings_attr)}" '
            f'data-models="{_esc("|".join(row_models.get(number, ())))}" '
            f'data-number="{number}" data-campaigns="{_esc(classes)}">'
            f'<input type="checkbox" name="pick" value="{number}">\n{_esc(label)}{mark}</label>'
        )
    joined = "\n".join(ticks)
    campaign = "\n".join(
        "<li><label>"
        f'<input type="checkbox" class="campaign-select" name="campaign" value="{_esc(name)}">\n'
        f"{_esc(name)}: {count} structures. {_esc(_CAMPAIGN_NOTES[name])}</label></li>"
        for name, count in campaign_counts(rows)
    )
    search = _research_grouped_html(saved)
    rank_label = "Research: All found structures by highest value targets for generation"
    biological_sort = focus is not None and focus_is_active(focus) and not order
    controls = _structure_view_controls(
        _llm_rank_filter_html(ranking) + _model_filter_html(results),
        _rank_subject_options(saved, ranking),
        sort="biological" if biological_sort else "listed",
        biological=focus is not None,
    )
    review = _ranking_review_html(ranking)
    found = _llm_results_html(results)
    # A search that named nothing and rated nothing still has to say so. The
    # ranking list holds every row even when no model returned a rating, so an
    # empty review is not a result.
    if not found and not review and not _ranking_has_ratings(ranking):
        found = '<h3>LLM search results</h3>\n<p class="notice">no results LLM search not yet requested.</p>'
    reply_note = ""
    if prior_reply_lines(ranking, results):
        reply_note = (
            '<p class="hint">Tick a search or research reply to include it. '
            "Type any further question in the box, then press Submit question.</p>\n"
        )
    question = _esc(transcript)
    body = f"""
<section>
<div class="llm-ask">
<textarea class="llm-question" name="llm_question" rows="6"
 aria-label="Question for the checked models">{question}</textarea>
<div class="llm-ask-actions">
<button type="button" class="llm-add-structures">Add selected structures</button>
<button type="submit" name="next" value="ask" formnovalidate>Submit question</button>
</div>
</div>
<p class="hint" id="llm-ask-note" hidden></p>
<h2>LLM search</h2>
{search}
<p class="llm-searches">
<button type="submit" name="next" value="missing" formnovalidate>First search:
find structures the program search missed</button>
<button type="submit" name="next" value="rank" formnovalidate>{rank_label}</button>
</p>
{gate}
{reply_note}{found}
{review}
<h2>Structures</h2>
{('<p class="hint">' + _esc(focus_summary(focus)) + "</p>") if focus is not None else ""}
<p class="hint">You can initiate a binding design campaign against the following targets:
Click a class to select its structures.</p>
<ul class="campaign">
{campaign}
</ul>
<p class="hint">The score is how many of UniProt, NCBI, STRING, IntAct, the Human
Protein Atlas, PRIDE, and PeptideAtlas returned a record for same structure.
Returned structures are saved for comparison with the LLM search.</p>
{controls}
<div class="ticks structure-rows">
{joined}
</div>
</section>
"""
    return _page("Choose structures", body, "Download Structures")


def _models_by_number(results: list[dict[str, object]] | None) -> dict[int, tuple[str, ...]]:
    """The models against each listed number, for the row's ``data-models``."""
    by_number: dict[int, tuple[str, ...]] = {}
    for item in results or []:
        if not isinstance(item, dict):
            continue
        number = item.get("number")
        if not isinstance(number, int) or isinstance(number, bool):
            continue
        names: list[str] = []
        for mark in item.get("models") or ():
            if isinstance(mark, dict) and mark.get("model"):
                name = str(mark["model"])
                if name not in names:
                    names.append(name)
        if names:
            by_number[number] = tuple(names)
    return by_number


def _shown_model(mark: dict[str, object]) -> str:
    """The name to show for one model in a result.

    ``llm_results`` has already put the key through ``model_display``, which names
    the provider and the model. A mark carrying no label is read through it here
    rather than showing the key, which is an id.
    """
    key = str(mark.get("model") or "").strip()
    shown = str(mark.get("label") or "").strip()
    return shown or (model_display(key) if key else "")


def _result_labels(results: list[dict[str, object]] | None) -> dict[str, str]:
    """Each model's report key mapped to the name shown for it."""
    labels: dict[str, str] = {}
    for item in results or []:
        if not isinstance(item, dict):
            continue
        for mark in item.get("models") or ():
            if isinstance(mark, dict) and mark.get("model"):
                labels[str(mark["model"])] = _shown_model(mark)
    return labels


def _ranking_name(model_key: str) -> str:
    """The model name used in ``LLM {name} Ranking``."""
    return model_display(model_key)


def _llm_ranking_phrase(model_key: str, rating: object) -> str:
    """One model's ranking, or ``none`` when that model did not rank this target."""
    stated = "none" if rating is None else str(rating)
    return f"LLM {_ranking_name(model_key)} Ranking {stated}"


def _ranking_models(ranking: list[dict[str, object]] | None) -> list[str]:
    """Models that returned at least one ranking, in a stable order."""
    seen: list[str] = []
    for item in ranking or []:
        if not isinstance(item, dict):
            continue
        marks = item.get("ratings")
        if not isinstance(marks, dict):
            continue
        for name in marks:
            text = str(name)
            if text not in seen:
                seen.append(text)
    return sorted(seen)


def _rank_subject_options(
    saved: Sequence[str] = (),
    ranking: list[dict[str, object]] | None = None,
    extra: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Models the value sort and the LLM ranking read, then Grok Bot.

    A key already on the ranking comes first, because that is the name written
    on the row. Listed models of providers with a key follow. Grok Bot is always
    last when it has not already ranked, and it needs no API key.
    """
    options: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(key: str) -> None:
        text = key.strip()
        if not text or text in seen:
            return
        seen.add(text)
        label = GROK_BOT_LABEL if text == GROK_BOT else model_display(text)
        options.append((text, label))

    for key in [*_ranking_models(ranking), *extra]:
        add(key)
    allowed = {name.casefold() for name in saved}
    for provider, _label in _PROVIDER_LABELS:
        if provider not in allowed:
            continue
        for model_id, _model_label in model_choices(provider):
            add(f"{provider} {model_id}")
    add(GROK_BOT)
    return options


def _llm_rank_filter_html(ranking: list[dict[str, object]] | None) -> str:
    """Include or exclude by the LLM ranking. Database counts are not a choice here.

    ``Any`` filters nothing. Has and No follow the model or Grok Bot chosen
    beside the sort, not the mean of every model. A named model option is that
    model's own ranking.
    """
    options = [
        '<option value="any" selected>Any</option>',
        '<option value="ranked">Has an LLM ranking</option>',
        '<option value="unranked">No LLM ranking</option>',
    ]
    for model in _ranking_models(ranking):
        label = f"LLM {_ranking_name(model)} Ranking"
        options.append(f'<option value="model:{_esc(model)}">{_esc(label)}</option>')
    joined = "\n".join(options)
    return f"""<label>LLM ranking
<select name="llm_rank">
{joined}
</select>
</label>
<label>Structures with that ranking
<select name="llm_rank_state">
<option value="include" selected>Include</option>
<option value="exclude">Exclude</option>
</select>
</label>"""


def _filter_picks_by_llm_ranking(
    picks: list[str],
    ranking: list[dict[str, object]] | None,
    metric: str,
    state: str,
    model: str = "",
) -> list[str]:
    """Drop ticked rows the LLM-ranking filter excludes.

    ``Any`` keeps every tick. Has and No follow ``model`` when one is chosen,
    which is the model or Grok Bot beside the sort. A row the ranking does not
    mention has no LLM ranking. This does not read or change the database
    instance counts.
    """
    if not metric or metric == "any":
        return picks
    by_number: dict[int, dict[str, object]] = {}
    for item in ranking or []:
        if not isinstance(item, dict):
            continue
        number = item.get("number")
        if isinstance(number, int) and not isinstance(number, bool):
            by_number[number] = item
    keep_match = state != "exclude"
    kept: list[str] = []
    for item in picks:
        if not item.isdigit():
            kept.append(item)
            continue
        if _row_matches_llm_ranking(by_number.get(int(item)), metric, model) == keep_match:
            kept.append(item)
    return kept


def _filter_picks_by_llm_model(
    picks: list[str],
    results: list[dict[str, object]] | None,
    model: str,
    state: str,
) -> list[str]:
    """Drop ticked rows the LLM-model filter excludes.

    ``Any model`` keeps every tick, matching the control beside it.

    This is the counterpart of :func:`_filter_picks_by_llm_ranking`, which the
    model filter was missing. The gap was reachable through a ticked design
    class: :func:`campaign_pick_numbers` expands a class to every row in it
    server-side, the ranking filter then narrowed that and the model filter did
    not, so a row this page had hidden -- and shown no tick for -- was still
    downloaded and carried into generation, with exit 0 and no notice. The
    class box itself stays ticked while a filter hides some of its rows, so
    there was nothing on the page to say so either.

    Read through :func:`_models_by_number`, which is the same function that
    fills each row's ``data-models`` attribute, so this cannot answer
    differently from the ``namedBy`` check the page applies to those rows.
    """
    if not model or model == "any":
        return picks
    named = _models_by_number(results)
    keep_match = state != "exclude"
    kept: list[str] = []
    for item in picks:
        if not item.isdigit():
            kept.append(item)
            continue
        if (model in named.get(int(item), ())) == keep_match:
            kept.append(item)
    return kept


def _row_matches_llm_ranking(item: dict[str, object] | None, metric: str, model: str = "") -> bool:
    """Whether one ranked row is the selected LLM-ranking metric.

    ``model`` is the model or Grok Bot beside the sort. Has and No read that
    model's number. A rating of zero still counts. The named ``model:`` choice
    names its own model and does not use ``model``.
    """
    ratings = item.get("ratings") if isinstance(item, dict) else None
    marks = ratings if isinstance(ratings, dict) else {}
    subject = model.strip()
    if metric == "ranked":
        if subject:
            return subject in marks and marks.get(subject) is not None
        return bool(marks)
    if metric == "unranked":
        if subject:
            return not (subject in marks and marks.get(subject) is not None)
        return not marks
    if metric.startswith("model:"):
        name = metric.removeprefix("model:")
        return name in marks and marks.get(name) is not None
    return False


def _ranking_has_ratings(ranking: list[dict[str, object]] | None) -> bool:
    """Whether any listed row carries a rating a model actually returned."""
    for item in ranking or []:
        if not isinstance(item, dict):
            continue
        marks = item.get("ratings")
        if isinstance(marks, dict) and any(rating is not None for rating in marks.values()):
            return True
    return False


def _review_reply_text(item: dict[str, object]) -> str:
    """One research reply: the structure, the ranking, and the sentence."""
    reasons = item.get("reasons")
    if not isinstance(reasons, dict):
        return ""
    bits: list[str] = []
    marks = item.get("ratings") if isinstance(item.get("ratings"), dict) else {}
    for model in sorted(reasons):
        text = str(reasons[model]).strip()
        if not text:
            continue
        rating = marks.get(model) if isinstance(marks, dict) else None
        bits.append(f"{_llm_ranking_phrase(str(model), rating)}: {text}")
    if not bits:
        return ""
    head = f"#{item.get('number')} {item.get('entry_id')} chain {item.get('chain')}"
    return f"{head} — {' '.join(bits)}"


def _result_reply_text(item: dict[str, object]) -> str:
    """One search reply: the structure and what each model said about it."""
    head = f"#{item.get('number')} {item.get('entry_id')} chain {item.get('chain')} {item.get('accession')}"
    title = str(item.get("title") or "").strip()
    if title:
        head += f" {title}"
    said: list[str] = []
    for mark in item.get("models") or ():
        if not isinstance(mark, dict):
            continue
        name = _shown_model(mark)
        if not name:
            continue
        rating = mark.get("rating")
        phrase = f"LLM {name} Ranking {rating if rating is not None else 'none'}"
        if mark.get("found"):
            phrase += "; named it"
        said.append(phrase)
    if said:
        return f"{head} {'; '.join(said)}"
    return head


def prior_reply_lines(
    ranking: list[dict[str, object]] | None = None,
    results: list[dict[str, object]] | None = None,
) -> list[tuple[str, str]]:
    """Checkbox value and plain text for each earlier search or research reply.

    Search rows come first, in the order the results list shows them. Research
    rows follow, and only a ranking that has a sentence is a reply.
    """
    lines: list[tuple[str, str]] = []
    index = 0
    for item in results or []:
        if not isinstance(item, dict):
            continue
        lines.append((f"result:{index}", _result_reply_text(item)))
        index += 1
    index = 0
    for item in ranking or []:
        if not isinstance(item, dict):
            continue
        text = _review_reply_text(item)
        if not text:
            continue
        lines.append((f"review:{index}", text))
        index += 1
    return lines


def _question_with_replies(question: str, replies: Sequence[str]) -> str:
    """Put ticked replies before the typed question, without repeating a line."""
    present = {line.strip() for line in question.splitlines() if line.strip()}
    fresh = [text for text in replies if text.strip() and text.strip() not in present]
    if not fresh:
        return question
    block = "\n".join(fresh)
    body = question.rstrip()
    if not body:
        return block
    return f"{block}\n\n{body}"


def _ranking_review_html(ranking: list[dict[str, object]] | None) -> str:
    """The completed ranking, for a person to read before anything else changes.

    The message is the model's ranking and the sentence it gave. It is not
    applied to the database instance counts. A ranking list with no sentence
    is not a review: the caller says there are no results. Each sentence can
    be ticked and sent again with a further question.
    """
    lines: list[str] = []
    for key, text in prior_reply_lines(ranking, None):
        lines.append(
            '<li><label class="prior-reply">'
            f'<input type="checkbox" name="prior_reply" value="{_esc(key)}">\n'
            f"{_esc(text)}</label></li>"
        )
    if not lines:
        return ""
    listed = f'<ul class="llm-review">\n{chr(10).join(lines)}\n</ul>'
    return (
        '<p class="notice">LLM ranking is ready for manual review. These rankings do not change '
        "the database instance counts or how many databases returned a record.</p>\n" + listed
    )


def _model_filter_html(results: list[dict[str, object]] | None) -> str:
    """Include or exclude the structures one model named.

    Nothing is rendered until a search has returned something. A control that can
    only ever filter an empty list is a dead control, and this page has already
    had to answer for one of those.
    """
    models = result_models(results)
    if not models:
        return ""
    labels = _result_labels(results)
    options = ['<option value="any" selected>Any model</option>']
    options += [f'<option value="{_esc(name)}">{_esc(labels.get(name, name))}</option>' for name in models]
    joined = "\n".join(options)
    return f"""<label>LLM model
<select name="llm_model">
{joined}
</select>
</label>
<label>Structures that model named
<select name="llm_state">
<option value="include" selected>Include</option>
<option value="exclude">Exclude</option>
</select>
</label>"""


def _llm_results_html(results: list[dict[str, object]] | None) -> str:
    """Every structure the searches named, with its ID# and the models that named it.

    The ID# is the listed number the structure's own tick carries, so a result can
    be found among the rows above without counting down the list.
    """
    if not results:
        return ""
    items: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        number = item.get("number")
        names: list[str] = []
        pairs: list[str] = []
        for mark in item.get("models") or ():
            if not isinstance(mark, dict):
                continue
            name = _shown_model(mark)
            if not name:
                continue
            model_key = str(mark.get("model"))
            names.append(model_key)
            rating = mark.get("rating")
            pairs.append(f"{model_key}=" + ("" if rating is None else str(rating)))
        value = item.get("value")
        value_attr = "" if value is None else str(value)
        reply = _result_reply_text(item)
        items.append(
            f'<li data-number="{_esc(str(number))}" data-value="{_esc(value_attr)}" '
            f'data-rankings="{_esc("|".join(pairs))}" data-models="{_esc("|".join(names))}">'
            '<label class="prior-reply">'
            f'<input type="checkbox" name="prior_reply" value="result:{len(items)}">\n'
            f"{_esc(reply)}</label></li>"
        )
    joined = "\n".join(items)
    return f"""<h3>LLM search results</h3>
<p class="hint">Every structure either search named or rated, with the ID# its tick
carries above and the model that named it.</p>
<ol class="llm-results">
{joined}
</ol>"""


def _carried_screening(form: OptionForm) -> str:
    """Hidden inputs for the screening flags the page shows no control for.

    Without these the panel rebuilds the command without them, and the screen
    runs with the defaults while the user believes they asked for more. They
    are hidden rather than shown because the panel's wording is not mine to
    invent; a flag typed on the command line still reaches the run.
    """
    values: list[tuple[str, str]] = []
    if form.proteome:
        values.append(("proteome", form.proteome))
    if form.lookup_targets:
        values.append(("lookup_targets", "yes"))
    if form.plasma_abundance_max is not None:
        values.append(("plasma_abundance_max", repr(form.plasma_abundance_max)))
    if form.plasma_hub_max is not None:
        values.append(("plasma_hub_max", str(form.plasma_hub_max)))
    if form.plasma_fold_max is not None:
        values.append(("plasma_fold_max", repr(form.plasma_fold_max)))
    if form.follow_up:
        values.append(("follow_up", "yes"))
    return "".join(f'<input type="hidden" name="{name}" value="{_esc(value)}">\n' for name, value in values)


#: Hidden-input name for each carried ``append`` research flag. Prefixed rather
#: than named after the flag because ``render_targets`` already posts ``target``
#: for its own tick boxes, and a reader that saw both could not tell a carried
#: flag from a ticked target.
_CARRIED_RESEARCH = (("research", "carried_research"), ("models", "carried_models"), ("target", "carried_target"))


def _carried_research(form: OptionForm) -> str:
    """Hidden inputs for the research flags the page shows no control for.

    The same reason as :func:`_carried_screening`: the panel rebuilds the
    command from this form, so a flag with no field here is a flag the user
    typed and the run never saw. ``--research``, ``--models`` and ``--target``
    were all three of those, and losing them is not a smaller default but a
    whole stage of the run that silently did not happen.

    Each flag is an ``append``, so each value rides as its own hidden input and
    comes back as its own entry.
    """
    values = [(field, item) for attribute, field in _CARRIED_RESEARCH for item in getattr(form, attribute) if item]
    return "".join(f'<input type="hidden" name="{name}" value="{_esc(value)}">\n' for name, value in values)


def _research_from_fields(fields: dict[str, list[str]]) -> dict[str, object]:
    """The carried research flags, as the ``append`` lists they came from.

    Nothing here raises: every value was written by :func:`_carried_research`
    from an already-parsed argv, and the parser is what refuses an unknown
    provider or model.
    """
    return {
        attribute: tuple(item.strip() for item in fields.get(field, []) if item.strip())
        for attribute, field in _CARRIED_RESEARCH
    }


def _screening_from_fields(fields: dict[str, list[str]], *, strict: bool) -> dict[str, object]:
    """The carried screening flags. ``strict`` refuses a number that is not one."""
    if strict:
        abundance = _optional_float(_first(fields, "plasma_abundance_max"), "Plasma abundance maximum")
        hub = _optional_int(_first(fields, "plasma_hub_max"), "Plasma hub maximum")
        fold = _optional_float(_first(fields, "plasma_fold_max"), "Plasma fold maximum")
    else:
        abundance = _number_or_none(_first(fields, "plasma_abundance_max"))
        hub = _whole_or_none(_first(fields, "plasma_hub_max"))
        fold = _number_or_none(_first(fields, "plasma_fold_max"))
    return {
        "proteome": _blank(_first(fields, "proteome")),
        "lookup_targets": _checked(fields, "lookup_targets"),
        "plasma_abundance_max": abundance,
        "plasma_hub_max": hub,
        "plasma_fold_max": fold,
        "follow_up": _checked(fields, "follow_up"),
    }


def form_from_fields(fields: dict[str, list[str]]) -> OptionForm:
    """Read one submitted option panel."""
    kind = _first(fields, "binder_kind") or "protein"
    if kind not in BINDER_KINDS:
        known = ", ".join(BINDER_KINDS)
        raise ValueError(f"Unknown binder kind {kind!r}. Expected one of {known}.")
    choice = _optional_int(_first(fields, "choice"), "Similar-name choice")
    out_dir = _process_path(_first(fields, "out_dir") or "disease_runs")
    remote_dir = _first(fields, "remote_dir") or "disease_runs"
    return OptionForm(
        disease=_blank(_first(fields, "disease")),
        catalog=_blank(_first(fields, "catalog")),
        target_structure=_uploaded_structure(fields),
        accept_only=_checked(fields, "accept_only"),
        choice=choice,
        structures_json=_blank(_first(fields, "structures_json")),
        take_all=_checked(fields, "take_all"),
        pick=_blank(_first(fields, "pick")),
        binder_kind=kind,
        out_dir=out_dir,
        download_dir=_download_dir(fields, out_dir),
        hosts=_hosts_from_fields(fields),
        remote_dir=remote_dir,
        local_workers=_workers_from_fields(fields),
        remote_workers=_workers_from_fields(fields, field="remote_workers", label="Remote workers"),
        modified_ckpt=_blank(_first(fields, "modified_ckpt")),
        confirm_inference=_checked(fields, "confirm_inference"),
        run=_checked(fields, "run"),
        **_screening_from_fields(fields, strict=True),
        **_research_from_fields(fields),
    )


def _uploaded_structure(fields: dict[str, list[str]]) -> str | None:
    """The uploaded target path, refusing a suffix the model cannot read."""
    value = _blank(_first(fields, "target_structure"))
    if value is None:
        return None
    if Path(value).suffix.casefold() not in UPLOAD_SUFFIXES:
        allowed = ", ".join(UPLOAD_SUFFIXES)
        raise ValueError(f"Upload a {allowed} file. {Path(value).name} is neither.")
    return value


def draft_from_fields(fields: dict[str, list[str]]) -> OptionForm:
    """Everything readable from a refused submission, for putting the page back.

    Nothing here raises. A value the strict reader rejected falls back to the
    default so the rest of what the user typed still survives the round trip.
    """
    kind = _first(fields, "binder_kind")
    return OptionForm(
        disease=_blank(_first(fields, "disease")),
        catalog=_blank(_first(fields, "catalog")),
        target_structure=_blank(_first(fields, "target_structure")),
        accept_only=_checked(fields, "accept_only"),
        choice=_whole_or_none(_first(fields, "choice")),
        structures_json=_blank(_first(fields, "structures_json")),
        take_all=_checked(fields, "take_all"),
        pick=_blank(_first(fields, "pick")),
        binder_kind=kind if kind in BINDER_KINDS else "protein",
        out_dir=_process_path(_first(fields, "out_dir") or "disease_runs"),
        download_dir=_download_dir(fields, _process_path(_first(fields, "out_dir") or "disease_runs")),
        hosts=_blank(",".join(_split_hosts(_first(fields, "hosts")))),
        remote_dir=_first(fields, "remote_dir") or "disease_runs",
        local_workers=_draft_workers(fields),
        remote_workers=_draft_workers(fields, "remote_workers"),
        modified_ckpt=_blank(_first(fields, "modified_ckpt")),
        confirm_inference=_checked(fields, "confirm_inference"),
        run=_checked(fields, "run"),
        **_screening_from_fields(fields, strict=False),
        **_research_from_fields(fields),
    )


def _draft_workers(fields: dict[str, list[str]], field: str = "local_workers") -> int | None:
    """The worker cap a refused submission chose, read as the strict reader reads it.

    The select carries every preset; only ``custom`` puts the number in the text
    box beside it. Reading the box alone turned a chosen 8 back into "every
    local request at once" on any redraw -- a refusal about another field, or a
    click on Search files -- and the user was then one Continue away from an
    uncapped run they had not asked for.

    ``field`` names the pair, so the remote cap is redrawn by the same rule. A
    cap that survived a redraw on one of the two and not the other would be the
    same defect over again, on whichever one was forgotten.
    """
    selected = _first(fields, field) or "every"
    if selected.lstrip("-").isdigit():
        return int(selected)
    return _whole_or_none(_first(fields, f"{field}_custom"))


def _whole_or_none(text: str) -> int | None:
    """The number, or None when it is not one. The strict reader is what refuses."""
    return int(text) if text.lstrip("-").isdigit() else None


def _number_or_none(text: str) -> float | None:
    """The number, or None when it is not one. The strict reader is what refuses."""
    try:
        return float(text)
    except ValueError:
        return None


def _download_dir(fields: dict[str, list[str]], out_dir: str) -> str | None:
    """A download directory that differs from the output directory, or None."""
    raw = _process_path(_first(fields, "download_dir"))
    if not raw or _same_path(raw, out_dir):
        return None
    return raw


def _download_dir_arg(download_dir: Path | None, out_dir: Path) -> str | None:
    if download_dir is None or _same_path(download_dir, out_dir):
        return None
    return str(download_dir)


def _same_path(left: Path | str, right: Path | str) -> bool:
    return Path(_process_path(str(left))).resolve() == Path(_process_path(str(right))).resolve()


_WIN_DRIVE = re.compile(r"^([A-Za-z]):[/\\](.*)$")
_WSL_MOUNT = re.compile(r"/mnt/([A-Za-z])(?:/(.*))?$")


def display_directory(text: str) -> str:
    """Complete path for the Structures Directory fields, using forward slashes.

    A Windows drive path is shown as ``A:/...``. A WSL mount ``/mnt/a/...`` is
    shown the same way. A network share is shown exactly as it was typed. Any
    other path is the absolute path.
    """
    raw = text.strip()
    if not raw:
        return raw
    if raw.startswith(("\\\\", "//")):
        # A UNC share has no local parent to resolve against. Resolving it would
        # relocate it under the working directory, and because this value is
        # what the page posts back, the relocation would then be adopted as the
        # user's choice. Shown unchanged, it round-trips.
        return raw
    windows = _WIN_DRIVE.match(raw)
    if windows:
        return _drive_text(windows.group(1), windows.group(2).replace("\\", "/"))
    try:
        resolved = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        # RuntimeError and ValueError as well as OSError, because neither of the
        # two calls on this line raises only OSError. On POSIX
        # ``expanduser()`` raises RuntimeError for ``~name`` when there is no
        # such account, and ``resolve()`` raises ValueError for a path pathlib
        # cannot process. So ``~someon`` typed for ``~someone`` was caught neither
        # here nor by the ``(OSError, ValueError)`` around the panel: it closed
        # the run at exit 2 and took every other field on the page with it,
        # naming nothing. A path that cannot be resolved is shown exactly as
        # typed, which is what the field posts back, so the value round-trips.
        resolved = Path(raw)
    posix = resolved.as_posix()
    mount = _WSL_MOUNT.fullmatch(posix)
    if mount:
        return _drive_text(mount.group(1), mount.group(2) or "")
    if resolved.drive:
        return _drive_text(resolved.drive[:1], posix.split(":", 1)[-1].lstrip("/"))
    return posix


def _drive_text(letter: str, rest: str) -> str:
    """``A:/`` plus the remainder, with no doubled slash after the drive."""
    tail = rest.lstrip("/")
    return f"{letter.upper()}:/{tail}" if tail else f"{letter.upper()}:/"


def _process_path(text: str) -> str:
    """The path this process can open.

    A posted ``A:/...`` value is the Windows form of a WSL mount. Relative paths
    and paths that already work here are kept as posted.
    """
    raw = text.strip()
    windows = _WIN_DRIVE.match(raw)
    if not windows or sys.platform.startswith("win"):
        return raw
    letter = windows.group(1).lower()
    if not Path(f"/mnt/{letter}").is_dir():
        return raw
    rest = windows.group(2).replace("\\", "/").lstrip("/")
    return f"/mnt/{letter}/{rest}" if rest else f"/mnt/{letter}"


#: How many output files and folders one page lists.
_LIST_LIMIT = 40
#: Names read from one directory before the page stops and sorts.
_SCAN_CAP = 2000
#: Directories a name search will open before it stops.
_WALK_CAP = 800
#: File-list clicks accepted in one panel pass. Browsing is not a refused field.
_MAX_FILE_BROWSE = 200
_SKIP_WALK = frozenset({".git", ".venv", "__pycache__", "node_modules", ".grok"})


def list_output_entries(directory: str, query: str = "", *, limit: int = _LIST_LIMIT) -> tuple[tuple[str, str], ...]:
    """Files and folders under the output directory, for the mouse list.

    Each row is ``(kind, path)``. ``kind`` is ``parent``, ``dir``, or ``file``.
    ``parent`` is the folder above, so a click can move up. An empty query lists
    the directory itself. A query matches names inside it.
    """
    root = _safe_resolve(Path(_process_path(directory or "disease_runs")))
    rows: list[tuple[str, str]] = []
    parent = root.parent
    if parent != root:
        rows.append(("parent", str(parent)))
    # `is_dir()` is a stat, and a path component over the filesystem's limit
    # raises ENAMETOOLONG rather than answering False. Pathlib ignores only
    # some errno values, so this is guarded like the resolve and the listing
    # around it: an unusable path offers no rows, it does not end the run.
    try:
        if not root.is_dir():
            return tuple(rows)
    except OSError:
        return tuple(rows)
    needle = query.strip().casefold()
    matches = _search_output(root, needle, limit) if needle else _list_children(root, limit)
    rows.extend((kind, str(path)) for kind, path in matches)
    return tuple(rows)


def choose_output_entry(form: OptionForm, picked: str) -> tuple[OptionForm, str]:
    """Move the output directory to a clicked folder, or to the folder of a clicked file.

    A file stays selected. A folder becomes the output directory and clears that
    selection. A path that is neither is ignored.
    """
    raw = picked.strip()
    if not raw:
        return form, ""
    path = _safe_resolve(Path(_process_path(raw)))
    if _is_dir(path):
        return replace(form, out_dir=str(path)), ""
    if _is_file(path):
        return replace(form, out_dir=str(path.parent)), str(path)
    return form, ""


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        # RuntimeError for an unknown ``~name`` on POSIX, as in
        # :func:`display_directory`: neither an OSError nor anything a page can
        # show, so it closed the panel instead of redrawing it. ValueError
        # because ``resolve()`` raises it for a path pathlib cannot process --
        # an embedded NUL among them -- which ``is_dir()`` quietly answers False
        # for while this would have ended the run.
        return path


def _is_dir(path: Path) -> bool:
    """Whether ``path`` is a directory, with a stat that cannot end the run.

    ``Path.is_dir()`` answers False for the errno values pathlib chooses to
    ignore and raises for the rest, so a child inside a tree this process
    cannot traverse raises ``PermissionError`` and a component past the
    filesystem's length limit raises ``ENAMETOOLONG``. Either one reached
    :func:`list_output_entries` from *past* its own guard -- the stats below
    run after ``iterdir()`` has already succeeded -- and closed the panel with
    a bare errno line, where the useful answer is that this row is not a folder
    the browser can open.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def _is_file(path: Path) -> bool:
    """Whether ``path`` is a file. Guarded like :func:`_is_dir`."""
    try:
        return path.is_file()
    except OSError:
        return False


def _list_children(root: Path, limit: int) -> list[tuple[str, Path]]:
    found: list[Path] = []
    try:
        for child in root.iterdir():
            found.append(child)
            if len(found) >= _SCAN_CAP:
                break
    except OSError:
        return []
    classified = [(child, _entry_kind(child)) for child in found]
    # An entry the filesystem will not classify is left out rather than guessed.
    # Calling an unstattable directory a file put a row on the page that
    # announced itself as clickable and then did nothing: `choose_output_entry`
    # tests the same stat, fails it the same way, and hands back the form
    # unchanged, so the hint counted a folder as a file and the click was inert
    # with nothing saying why. A row that cannot be acted on is not offered.
    usable = [(child, kind) for child, kind in classified if kind is not None]
    usable.sort(key=lambda item: (item[1] != "dir", item[0].name.casefold()))
    return [(kind, _safe_resolve(child)) for child, kind in usable[:limit]]


def _entry_kind(path: Path) -> str | None:
    """``"dir"``, ``"file"``, or None when the filesystem will not answer."""
    try:
        return "dir" if path.is_dir() else "file"
    except OSError:
        return None


def _ignore_walk_error(_error: OSError) -> None:
    """Skip a directory the search cannot open."""
    return


def _search_output(root: Path, needle: str, limit: int) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    seen = 0
    for dirpath, dirnames, filenames in root.walk(top_down=True, on_error=_ignore_walk_error, follow_symlinks=False):
        seen += 1
        if seen > _WALK_CAP:
            break
        dirnames[:] = [name for name in dirnames if name not in _SKIP_WALK and not name.startswith(".")]
        current = Path(dirpath)
        ranked = [("dir", name) for name in dirnames] + [("file", name) for name in filenames]
        ranked.sort(key=lambda item: (item[0] != "dir", item[1].casefold()))
        for kind, name in ranked:
            if needle not in name.casefold():
                continue
            found.append((kind, _safe_resolve(current / name)))
            if len(found) >= limit:
                return found
    return found


def _screen_readout(directory: str) -> str:
    """Stored screens for this output folder and the generation folders inside it.

    The class is not ``notice``. That class scrolls the disease page to the
    disease box, and this readout sits on the same page.
    """
    try:
        reports = screen_reports_for_directory(Path(_process_path(directory or "disease_runs")))
    except Exception:
        # A backstop, not the fix. An unreadable file is skipped per folder, by
        # `screen_lines_for_directory`, so one bad `screen.json` costs its own
        # report and no other -- caught here instead, it discarded every valid
        # report gathered before it and the page then said a folder holding
        # eight good screens had none.
        #
        # What remains here is the guarantee that drawing a page cannot end the
        # run, which is worth keeping broad: this reads a tree of files the code
        # did not write and cannot constrain, the readout is an extra on the
        # page, and showing nothing is a complete answer where exiting is not.
        # Before the per-folder skip existed, a deeply nested `screen.json`
        # (RecursionError out of `json.loads`) or one carrying `sequence_pass`
        # without `gravy` (KeyError out of `_measured_line`) made the panel's
        # *first* page unrenderable, so the user could not even edit the output
        # field to get away from the file.
        return ""
    blocks: list[str] = []
    listing: list[str] = []
    for label, lines in reports:
        # The listing's own line -- what it truncated or collapsed -- rides on
        # the last report. Left there it renders as one of that run's verdict
        # sentences, under that run's label, and which run collects it is
        # whichever happens to be last. It is about the listing, so it is shown
        # after the runs, in a block of its own, with no label.
        own = [line for line in lines if not line.startswith(SCREEN_LISTING_PREFIX)]
        listing.extend(line for line in lines if line.startswith(SCREEN_LISTING_PREFIX))
        if not own:
            continue
        where = f'<p class="screen-where">{_esc(label)}</p>' if label else ""
        body = where + "".join(f"<p>{_esc(line)}</p>" for line in own)
        blocks.append(f'<div class="screen-readout">{body}</div>')
    for line in listing:
        blocks.append(f'<div class="screen-readout screen-listing"><p>{_esc(line)}</p></div>')
    if not blocks:
        return ""
    return "\n".join(blocks) + "\n"


def _output_file_browser(directory: str, query: str, selected: str) -> str:
    """Search box and clickable rows for the output directory."""
    rows = list_output_entries(directory, query)
    hint = _output_files_hint(directory, rows, query, selected)
    buttons = _output_file_buttons(rows, selected)
    return f"""{_screen_readout(directory)}<label class="field">Search output files
<input type="text" name="out_file_search" value="{_esc(query)}">
<span>Part of a file or folder name in the output directory. Click a row to select it.</span>
</label>
<button type="submit" class="nav" name="__action" value="search-files" formnovalidate>Search files</button>
<p class="hint">{_esc(hint)}</p>
<div class="file-picks">
{buttons}
</div>
"""


def _output_file_buttons(rows: tuple[tuple[str, str], ...], selected: str) -> str:
    selected_key = str(_safe_resolve(Path(_process_path(selected)))) if selected else ""
    parts: list[str] = []
    for kind, path in rows:
        shown = display_directory(path)
        label = f"..  {shown}" if kind == "parent" else shown
        pressed = ' aria-pressed="true"' if kind == "file" and path == selected_key else ""
        parts.append(
            f'<button type="submit" class="nav file-pick" name="out_pick" value="{_esc(path)}"'
            f" formnovalidate{pressed}>{_esc(label)}</button>"
        )
    return "\n".join(parts)


def _output_files_hint(directory: str, rows: tuple[tuple[str, str], ...], query: str, selected: str) -> str:
    root = _safe_resolve(Path(_process_path(directory or "disease_runs")))
    files = sum(kind == "file" for kind, _path in rows)
    folders = sum(kind == "dir" for kind, _path in rows)
    try:
        exists, is_dir = root.exists(), root.is_dir()
    except OSError:
        # A path the filesystem will not even stat -- a component past its
        # length limit -- is reported as not existing rather than ending the
        # run from inside a page the user is still filling in.
        exists, is_dir = False, False
    if not exists:
        text = "This output directory does not exist yet. Click .. to choose a folder that does."
    elif not is_dir:
        text = "That output path is a file. Click .. to choose the folder around it."
    elif query.strip() and files + folders == 0:
        text = f"No file or folder matches {query.strip()}."
    elif files + folders == 0:
        text = "No files or folders in this directory. Click .. to move up."
    else:
        text = f"{files} file(s) and {folders} folder(s). Click one to select it."
        if files + folders >= _LIST_LIMIT:
            text += f" Showing the first {files + folders}."
    if selected:
        text += f" Selected {display_directory(selected)}."
    return text


def to_argv(form: OptionForm) -> list[str]:
    """Argv for the same parser the flags use. ``--ui`` keeps the later structure page."""
    argv = ["--ui"]
    if form.disease:
        argv.extend(["--disease", form.disease])
    if form.catalog:
        argv.extend(["--catalog", form.catalog])
    if form.target_structure:
        argv.extend(["--target-structure", form.target_structure])
    if form.accept_only:
        argv.append("--accept-only")
    if form.choice is not None:
        argv.extend(["--choice", str(form.choice)])
    if form.structures_json:
        argv.extend(["--structures-json", form.structures_json])
    if form.take_all:
        argv.append("--all")
    if form.pick:
        argv.extend(["--pick", form.pick])
    argv.extend(["--binder-kind", form.binder_kind, "--out-dir", form.out_dir])
    if form.download_dir:
        argv.extend(["--download-dir", form.download_dir])
    if form.hosts:
        argv.extend(["--hosts", form.hosts])
    argv.extend(["--remote-dir", form.remote_dir])
    if form.local_workers is not None:
        argv.extend(["--local-workers", str(form.local_workers)])
    if form.remote_workers is not None:
        argv.extend(["--remote-workers", str(form.remote_workers)])
    if form.modified_ckpt:
        argv.extend(["--modified-ckpt", form.modified_ckpt])
    if form.confirm_inference:
        argv.append("--confirm-inference")
    if form.run:
        argv.append("--run")
    if form.proteome:
        argv.extend(["--proteome", form.proteome])
    if form.lookup_targets:
        argv.append("--lookup-targets")
    if form.plasma_abundance_max is not None:
        argv.extend(["--plasma-abundance-max", repr(form.plasma_abundance_max)])
    if form.plasma_hub_max is not None:
        argv.extend(["--plasma-hub-max", str(form.plasma_hub_max)])
    if form.plasma_fold_max is not None:
        argv.extend(["--plasma-fold-max", repr(form.plasma_fold_max)])
    if form.follow_up:
        argv.append("--follow-up")
    # One flag per value, because each is an `append` and a comma-joined value
    # would re-enter the parser as a single provider name.
    for value in form.research:
        argv.extend(["--research", value])
    for value in form.models:
        argv.extend(["--models", value])
    for value in form.target:
        argv.extend(["--target", value])
    return argv


def _form_from_namespace(args: argparse.Namespace) -> OptionForm:
    return OptionForm(
        disease=args.disease,
        catalog=None if args.catalog is None else str(args.catalog),
        target_structure=None if args.target_structure is None else str(args.target_structure),
        accept_only=bool(args.accept_only),
        choice=args.choice,
        structures_json=None if args.structures_json is None else str(args.structures_json),
        take_all=bool(args.all),
        pick=args.pick,
        binder_kind=args.binder_kind,
        out_dir=str(args.out_dir),
        download_dir=_download_dir_arg(args.download_dir, args.out_dir),
        hosts=args.hosts,
        remote_dir=args.remote_dir,
        local_workers=args.local_workers,
        remote_workers=args.remote_workers,
        modified_ckpt=None if args.modified_ckpt is None else str(args.modified_ckpt),
        confirm_inference=bool(args.confirm_inference),
        run=bool(args.run),
        proteome=None if args.proteome is None else str(args.proteome),
        lookup_targets=bool(args.lookup_targets),
        plasma_abundance_max=args.plasma_abundance_max,
        plasma_hub_max=args.plasma_hub_max,
        plasma_fold_max=args.plasma_fold_max,
        follow_up=bool(args.follow_up),
        research=tuple(args.research or ()),
        models=tuple(args.models or ()),
        target=tuple(args.target or ()),
    )


_API_KEY_FIELDS = (
    ("grok", "grok_key"),
    ("claude", "claude_key"),
    ("openai", "openai_key"),
    ("nvidia", "nvidia_key"),
    ("openrouter", "openrouter_key"),
)
_MODEL_FIELDS = (
    ("grok", "grok_model"),
    ("claude", "claude_model"),
    ("openai", "openai_model"),
    ("nvidia", "nvidia_model"),
    ("openrouter", "openrouter_model"),
)
#: The research module owns these, so a report key and a page agree on the name.
_PROVIDER_LABELS = PROVIDER_LABELS
_WHERE = (
    ("local", "This machine only"),
    ("local-and-remote", "This machine and remote machines"),
    ("remote", "Remote machines only"),
)
#: The remote dropdown. Same shape and same values as _WORKERS so one parser
#: reads both; the wording differs because the thing being counted does --
#: these are requests spread over the named machines, not processes here.
_REMOTE_WORKERS = (
    ("every", "Every remote request at once"),
    ("1", "1 request across all remote machines"),
    ("2", "2 requests across all remote machines"),
    ("4", "4 requests across all remote machines"),
    ("8", "8 requests across all remote machines"),
    ("custom", "Custom"),
)
_WORKERS = (
    ("every", "Every local request at once"),
    ("1", "1 process on this machine"),
    ("2", "2 processes on this machine"),
    ("4", "4 processes on this machine"),
    ("8", "8 processes on this machine"),
    ("custom", "Custom"),
)


def _api_key_block(
    provider: str,
    label: str,
    key_name: str,
    saved: tuple[str, ...],
    stored: dict[str, str],
    slots: Sequence[KeySlot],
) -> str:
    """One provider: a row per saved key, or one empty row, and a button that adds another."""
    saved_rows = list(slots)
    if saved_rows:
        last = len(saved_rows) - 1
        body = "\n".join(
            _api_key_row(provider, label, key_name, saved, stored, slot, str(index), index == last)
            for index, slot in enumerate(saved_rows)
        )
    else:
        body = _api_key_row(provider, label, key_name, saved, stored, None, "new", True)
    return f"""<div class="llm-provider">
<input type="hidden" name="{provider}_listed" value="yes">
<div class="llm-rows">
{body}
</div>
</div>"""


def _api_key_row(
    provider: str,
    label: str,
    key_name: str,
    saved: tuple[str, ...],
    stored: dict[str, str],
    slot: KeySlot | None,
    slot_id: str,
    include_add: bool,
) -> str:
    """One key, the model it is sent with, and the small buttons at the end of the row.

    Remove sits beside Add API key on the last row. A saved secret is not put in
    the page. The password shows ``*****`` instead.
    """
    allowed = {model_id for model_id, _label in model_choices(provider)}
    if slot is None:
        selected = panel_model(provider, stored)
        typed = ""
        note = ""
    else:
        selected = slot.model if slot.model in allowed else panel_model(provider, stored)
        typed = f' value="{_esc(_KEY_MASK)}"'
        owners = saved if provider in saved else (*saved, provider)
        note = _key_hint(provider, owners)
    hint = f'<span class="llm-saved">{_esc(note)}</span>' if note else ""
    add = '<button type="button" class="nav llm-add">Add API key</button>' if include_add else ""
    return f"""<div class="llm">
<input type="hidden" name="{provider}_slot" value="{_esc(slot_id)}">
<span class="llm-caption llm-key-caption">{label} API Key</span>
<span class="llm-caption llm-model-caption">{label} model</span>
<input type="password" name="{key_name}"{typed} autocomplete="off" spellcheck="false"
aria-label="{_esc(label)} API Key">
<select name="{provider}_model" aria-label="{_esc(label)} model">
{_select_options(model_choices(provider), selected)}
</select>
<div class="llm-actions">
<button type="button" class="nav llm-remove">Remove</button>
{add}
</div>
{hint}
</div>"""


def _accepted_model(provider: str, raw: str, previous: str) -> str:
    """A posted model id, or ``previous`` when the field was left blank."""
    value = raw.strip()
    if not value:
        return previous
    if not is_offered_model(provider, value):
        allowed = tuple(model_id for model_id, _label in model_choices(provider))
        label = dict(_PROVIDER_LABELS)[provider]
        known = ", ".join(allowed)
        raise ValueError(f"Unknown {label} model {value!r}. Expected one of {known}.")
    return value


def _field_at(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""


def _slots_from_fields(provider: str, fields: dict[str, list[str]], existing: Sequence[KeySlot]) -> list[KeySlot]:
    """Pair posted keys with the models beside them.

    A slot id of ``0``, ``1``, ... keeps that saved secret when the field is blank
    or ``*****``. ``new`` appends a typed key. The mask itself is never stored.
    With no slot ids, one typed key replaces that provider and a blank field keeps
    every saved key.
    """
    posted_slots = fields.get(f"{provider}_slot", [])
    posted_keys = fields.get(f"{provider}_key", [])
    posted_models = fields.get(f"{provider}_model", [])
    if not posted_slots:
        secret = posted_keys[0].strip() if posted_keys else ""
        previous = existing[0].model if existing else ""
        model = _accepted_model(provider, posted_models[0] if posted_models else "", previous)
        for extra in posted_models[1:]:
            _accepted_model(provider, extra, "")
        if not secret or secret == _KEY_MASK:
            return list(existing)
        return [KeySlot(secret, model)]

    built: list[KeySlot] = []
    used: set[int] = set()
    width = max(len(posted_slots), len(posted_keys), len(posted_models))
    for index in range(width):
        slot_id = _field_at(posted_slots, index).strip()
        secret = _field_at(posted_keys, index).strip()
        raw_model = _field_at(posted_models, index)
        if slot_id.isdigit():
            position = int(slot_id)
            previous_slot = existing[position] if 0 <= position < len(existing) else None
            if previous_slot is None or position in used:
                model = _accepted_model(provider, raw_model, "")
                if secret and secret != _KEY_MASK:
                    built.append(KeySlot(secret, model))
                continue
            used.add(position)
            model = _accepted_model(provider, raw_model, previous_slot.model)
            kept = previous_slot.secret if secret in {"", _KEY_MASK} else secret
            if kept.strip():
                built.append(KeySlot(kept.strip(), model))
            continue
        model = _accepted_model(provider, raw_model, "")
        if secret and secret != _KEY_MASK:
            built.append(KeySlot(secret, model))
    return built


def _key_hint(provider: str, saved: tuple[str, ...]) -> str:
    """The note under a saved key. A blank field still keeps that key."""
    if provider in saved:
        return "A key is already saved encrypted on this computer."
    return ""


def parse_submission(raw: bytes, content_type: str, uploads: Path | None = None) -> dict[str, list[str]]:
    """Fields from one POST, whether urlencoded or multipart.

    A file part is written under ``uploads`` and its saved path becomes the field
    value, so every reader downstream sees an ordinary string and the form still
    round-trips through argv. A part with no filename or no bytes is skipped.
    """
    if "multipart/form-data" not in content_type.casefold():
        return parse_qs(raw.decode("utf-8", errors="replace"), keep_blank_values=True)
    head = b"MIME-Version: 1.0\r\nContent-Type: " + content_type.encode("utf-8") + b"\r\n\r\n"
    message = message_from_bytes(head + raw)
    fields: dict[str, list[str]] = {}
    for part in message.walk():
        if part.get_content_maintype() == "multipart":
            continue
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            if not payload or uploads is None:
                continue
            fields.setdefault(name, []).append(str(save_upload(uploads, filename, payload)))
            continue
        fields.setdefault(name, []).append(payload.decode("utf-8", errors="replace"))
    return fields


def save_upload(directory: Path, filename: str, payload: bytes) -> Path:
    """Write one uploaded file into its own directory and return the path.

    The name is reduced to its last component with anything unusual replaced, so a
    browser cannot steer the write outside ``directory``. There is no size limit. The
    suffix is not checked here either: a wrong one is reported by the reader, which can
    put the page back.
    """
    safe = _UNSAFE_NAME.sub("_", Path(filename.replace("\\", "/")).name).strip("._") or "upload"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / safe
    path.write_bytes(payload)
    return path


def with_nav(html: str, *, back: bool, forward: bool) -> str:
    """Fill the nav slot with the moves that are available on this page."""
    buttons = ""
    if back:
        buttons += '<button type="submit" class="nav" name="__nav" value="back" formnovalidate>Back</button>\n'
    if forward:
        buttons += '<button type="submit" class="nav" name="__nav" value="forward" formnovalidate>Forward</button>\n'
    return html.replace(NAV_SLOT, buttons, 1)


def back_target(fields: dict[str, list[str]], index: int) -> int:
    """Which step a Back asks for. The browser's own button names it; a button does not."""
    asked = _first(fields, "__to")
    if asked.isdigit():
        return max(0, min(int(asked), index - 1))
    return index - 1


def _with_step(html: str, step: int, *, closed: bool = False) -> str:
    """Stamp the form with the step this response is showing.

    The page is rendered before the session step exists, so the handler adds
    the stamp when it sends the response. The open tab compares it with
    ``/progress`` and opens the panel's current page when they differ.
    """
    opening = html.find("<form ")
    if opening == -1:
        return html
    closing = html.find(">", opening)
    if closing != -1 and "data-step=" in html[opening:closing]:
        # Already stamped. Tested on the form's own opening tag, not on the
        # whole document: `_esc` escapes the angle brackets and quotes but not
        # `=`, so the literal text `data-step=` typed into any field the page
        # echoes back -- the disease box, the file search -- satisfied a
        # document-wide test and the page then went out carrying no stamp at
        # all. `follow()` reads a null `data-step` and returns, so that tab
        # quietly stopped following the panel with nothing to say it had.
        return html
    marker = f'<form data-step="{step}" '
    if closed:
        marker = f'<form data-step="{step}" data-closed="true" '
    return html.replace("<form ", marker, 1)


def _results_payload(payload: str, name: str, token: str) -> str:
    """The text an action produced, where it can be copied or saved.

    Shown rather than merely described. A sentence reporting that a sequence
    was produced, with the sequence itself nowhere on the page, leaves the
    reader to take the button's word for it -- and there is nothing to paste
    into the service the sentence just told them to use.

    A textarea rather than a ``<pre>`` because a textarea's contents select
    with one click and survive a copy without picking up the page's styling,
    and ``readonly`` because editing it here would change nothing.
    """
    if not payload:
        return ""
    save = ""
    if token and name:
        # Served from the same token route the viewers use, so saving costs no
        # new endpoint and the bytes are the ones the page is showing.
        save = (
            f'<a class="nav results-save" href="/structure/{_esc(token)}"'
            f' download="{_esc(name)}">Download {_esc(name)}</a>'
        )
    lines = payload.count(chr(10))
    rows = min(max(lines + 1, 4), 14)
    return f"""<div class="results-payload">
<label class="field">Produced ({lines} line{"" if lines == 1 else "s"}) — select to copy
<textarea readonly rows="{rows}" spellcheck="false">{_esc(payload)}</textarea></label>
<p class="results-row">{save}</p>
</div>
"""


def _validation_scope_options(directories: Sequence[Path]) -> list[tuple[str, str]]:
    options = [("all", f"All {len(directories)} finished runs")]
    options.extend((str(i), f"{i + 1}. {path.name or 'Run'}") for i, path in enumerate(directories))
    return options


def _scientific_screen_panel(
    choices: Sequence[ScreenChoice], selected: tuple[str, ...] | None, results: list[dict],
    *, submit_button: bool = True,
) -> str:
    outcomes = {row["id"]: row for row in results}
    groups = {"Calculations": [], "Local result readers": [], "Previously recorded observations": []}
    installed = sum(c.installed for c in choices)
    checked_count = 0
    for choice in choices:
        key = choice.method.key
        checked = choice.installed and (choice.recommended if selected is None else key in selected)
        checked_count += checked
        attributes = (" data-recommended" if choice.recommended else "") + ("" if choice.installed else " disabled")
        outcome = outcomes.get(key, {})
        detail = ""
        if outcome:
            body = (json.dumps(outcome["measurement"], ensure_ascii=False, indent=2)
                    if "measurement" in outcome else outcome["reason"])
            detail = f'<details><summary>{_esc(outcome["status"])}</summary><pre>{_esc(body)}</pre></details>'
        group = {"recorded_evidence": "Previously recorded observations",
                 "native_result_reader": "Local result readers"}.get(choice.method.operation, "Calculations")
        groups[group].append(
            f'<div><label><input type="checkbox" name="{SCREENING_FIELD}" '
            f'value="{_esc(key)}"{attributes}{_on(checked)}> '
            f'<strong>{_esc(choice.method.label)}</strong></label> '
            f'<small>{_esc(choice.method.workload)}</small><p class="hint">{_esc(choice.reason)}</p>{detail}</div>'
        )
    return (
        '<section class="results-tools" data-screening-methods><h2>Post-generation screening</h2>'
        '<p class="hint">Recommendations use the candidate type, supplied inputs and installed software. '
        'Selected methods run one at a time.</p><p class="results-row">'
        '<button type="button" class="nav" data-validation-select="all">Select all installed screens</button>'
        '<button type="button" class="nav" data-validation-select="recommended">Select recommended screens</button>'
        '<button type="button" class="nav" data-validation-select="none">Clear screen selection</button></p>'
        f'<p aria-live="polite"><span data-screening-count>{checked_count}</span>'
        f' of {installed} installed screens selected'
        f' · {len(choices)} listed</p>'
        + "".join(f'<fieldset><legend>{heading}</legend>{"".join(rows)}</fieldset>'
                  for heading, rows in groups.items() if rows)
        + (f'<button type="submit" class="nav" name="__action" value="{SCREENING_ACTION}" formnovalidate>'
           'Run selected screens</button>' if submit_button else '')
        + '</section>'
    )


def _validation_panel(
    choices: Sequence[ValidationChoice],
    selected: tuple[str, ...] | None,
    report: Mapping[str, str],
    directories: Sequence[Path],
    scope: str,
    candidates: Sequence[ValidationCandidate],
    candidate_key: str | None,
) -> str:
    scope_control = ""
    if len(directories) > 1:
        options = _select_options(_validation_scope_options(directories), scope)
        scope_control = (
            f'<label>Validation scope <select name="{VALIDATION_SCOPE_FIELD}" data-validation-scope>'
            f'{options}</select></label>'
        )
    candidate_control = ""
    if candidates or candidate_key is not None:
        options = [("all", "All candidates"), *((candidate.key, candidate.label) for candidate in candidates)]
        if candidate_key is not None and candidate_key not in dict(options):
            options.append((candidate_key, "Candidate unavailable"))
        candidate_options = _select_options(options, "all" if candidate_key is None else candidate_key)
        candidate_control = (
            f'<label>Candidate <select name="{VALIDATION_CANDIDATE_FIELD}" data-validation-scope>'
            f'{candidate_options}</select></label>'
        )
    guidance = "Recommendations are selected from installed checks with matching files or bench records."
    if candidate_key is not None and not choices:
        guidance = "The selected candidate is unavailable. Choose another candidate to refresh the checks."
    rows = []
    available = any(choice.installed for choice in choices)
    for choice in choices:
        checked = choice.installed and (choice.recommended if selected is None else choice.key in selected)
        recommended = " data-recommended" if choice.recommended else ""
        recommendation = " <small>Recommended</small>" if choice.recommended else ""
        disabled = " disabled" if not choice.installed else ""
        outcome = (
            f'<p class="hint" style="white-space:pre-wrap;overflow-wrap:anywhere">{_esc(report[choice.key])}</p>'
            if choice.key in report else ""
        )
        rows.append(
            f'<div><label><input type="checkbox" name="{VALIDATION_FIELD}" value="{_esc(choice.key)}"'
            f'{recommended}{disabled}{_on(checked)}> <strong>{_esc(choice.label)}</strong>{recommendation}</label>'
            f'<p class="hint">{_esc(choice.description)} {_esc(choice.reason)} {_esc(choice.workload)}</p>'
            f'{outcome}</div>'
        )
    run_disabled = " disabled" if not available else ""
    return f'''<section class="results-tools" data-validation-checks>
<h2>Validate output files</h2>
<p class="hint">{guidance}</p>
<p class="results-row">{scope_control}{candidate_control}
<button type="submit" class="nav" name="__action" value="{VALIDATION_SCOPE_ACTION}"
data-validation-refresh formnovalidate>Refresh recommendations</button></p>
<p class="results-row">
<button type="button" class="nav" data-validation-select="all"{run_disabled}>Select all installed</button>
<button type="button" class="nav" data-validation-select="recommended"{run_disabled}>Select recommended</button>
<button type="button" class="nav" data-validation-select="none"{run_disabled}>Clear selection</button>
</p>
{"".join(rows)}
<p class="results-row"><button type="submit" class="nav" name="__action" value="{VALIDATION_ACTION}"
 formnovalidate{run_disabled}>Run selected checks</button></p>
</section>'''


def _results_tools(
    result: str,
    payload: str = "",
    name: str = "",
    token: str = "",
    active: str = "binder",
    record: Mapping[str, object] | None = None,
) -> str:
    """The results-page buttons, and the last one's outcome.

    Only shown where there is a generated structure to act on. The sentence
    uses ``screen-readout`` rather than ``notice``: a notice is the red one the
    disease page scrolls to, and none of these outcomes is an error.

    It sits above both panels rather than inside one. Written into the binder
    panel it was hidden whenever the combined tab was the open one, so every
    combined button reported its outcome onto the tab the reader was not
    looking at and the click appeared to do nothing at all. The same goes for
    the handover block, which is how a combined FASTA is copied or saved.

    The labels say what each one does rather than what was asked for. Nothing
    is sent to the AlphaFold database -- it serves precomputed predictions and
    its API is GET-only -- and nothing is sent to the AlphaFold Server either,
    which has no submission API and issues no credential. Calling either a
    "send" would be the button overstating itself, which on a page of safety
    verdicts is the one thing it must not do.
    """
    readout = f'<p class="screen-readout">{_esc(result)}</p>' if result else ""
    handover = _results_payload(payload, name, token)
    # Which tab is open, kept across the post. Without this every click on a
    # combined button answered on the combined tab and then showed the binder
    # one, so the reader had to find their way back to their own result.
    #
    # Compared by name and not by "is it the combined one", which is how it
    # read while there were two: a third tab then opened, posted its own name
    # back, and the page returned with Binder showing and the protocol's result
    # on a tab nobody was looking at -- the exact bug the comment above
    # records, one tab along.
    binder_on = " checked" if active == "binder" else ""
    combined_on = " checked" if active == "combined" else ""
    human_on = " checked" if active == HUMAN_TAB else ""
    dynamics_on = " checked" if active == DYNAMICS_TAB else ""
    if not (binder_on or combined_on or human_on or dynamics_on):
        binder_on = " checked"
    return f"""<section class="results-tools results-tabbed">
<input type="radio" name="results-tab" id="results-tab-binder" value="binder"{binder_on}>
<label for="results-tab-binder" class="results-tab">Binder only</label>
<input type="radio" name="results-tab" id="results-tab-combined" value="combined"{combined_on}>
<label for="results-tab-combined" class="results-tab">Combined structure</label>
<input type="radio" name="results-tab" id="results-tab-human" value="{HUMAN_TAB}"{human_on}>
<label for="results-tab-human" class="results-tab">{_esc(PROTOCOL_LABEL)}</label>
<input type="radio" name="results-tab" id="results-tab-dynamics" value="{DYNAMICS_TAB}"{dynamics_on}>
<label for="results-tab-dynamics" class="results-tab">{_esc(ANALYST_LABEL)}</label>
{readout}
{handover}
<div class="results-panel results-panel-binder">
<h2>Binder sequence and folding</h2>
<p class="hint">These act on the <strong>binder alone</strong>. A binder folded by itself
answers whether it folds as designed; it says nothing about whether the complex forms,
which is what the refold requests under the run directory are for.</p>
<p class="results-row">
<button type="submit" class="nav" name="__action"
 value="{RESULTS_FASTA}" formnovalidate>Binder only to FASTA Proteina</button>
</p>
<p class="results-row">
<button type="submit" class="nav" name="__action"
 value="{RESULTS_AF3}" formnovalidate>Prepare binder for AlphaFold 3 Server</button>
</p>
{_retrieve_row("af2", "AlphaFold 2 model", RESULTS_AF2, accession=True)}
{_retrieve_row("af3", "AlphaFold 3 model", RESULTS_AF3_IMPORT, accession=False)}
</div>
<div class="results-panel results-panel-combined">
<h2>Combined structure</h2>
<p class="hint">The other question: can the whole complex be reproduced, rather than the
binder on its own. The request carries every chain and marks which one is the design, so a
fold of it answers whether the complex forms.</p>
<p class="results-row">
<button type="submit" class="nav" name="__action"
 value="{RESULTS_COMBINED_FASTA}" formnovalidate>replicate the combined Structure ?</button>
</p>
<p class="results-row">
<button type="submit" class="nav" name="__action"
 value="{RESULTS_AF3C_PREPARE}" formnovalidate>Replicate A De-NoVo Combined Sequence to AF 3 Server</button>
</p>
{_retrieve_row("af2c", "AlphaFold 2 model of the complex", RESULTS_AF2C, accession=True,
               label_override="Replicate A Combined Known Sequence to AF 2 Database")}
{_retrieve_row("af3c", "AlphaFold 3 model of the complex", RESULTS_AF3C, accession=False)}
</div>
{_human_panel(record)}
{_dynamics_panel(record)}
</section>
"""


def _human_panel(record: Mapping[str, object] | None) -> str:
    """The H.U.M.A.N  Protocol tab: choose what not to bind, then the target, then compare.

    Two steps in the reader's own order. Propose fills the list from what these
    designs already resemble; the tick boxes and the free-text box are the
    selection; the target box is next; compare is last. The limit notice sits
    above the buttons rather than below the result, because a reader who has
    just been handed a number is past the point of reading what it does not
    mean.
    """
    held = record if isinstance(record, Mapping) else {}
    return f"""<div class="results-panel results-panel-human">
<h2>{_esc(PROTOCOL_LABEL)}</h2>
<p class="hint">What this design must <strong>not</strong> bind, chosen before the target
rather than after it. Propose lists the human proteins these designs already resemble,
strongest first; tick the ones to avoid, or name your own.</p>
<p class="protocol-limit">{_esc(PROTOCOL_LIMIT)}</p>
<fieldset class="results-retrieve">
<legend>1. What not to bind to</legend>
<p class="results-row"><button type="submit" class="nav" name="__action"
 value="{RESULTS_HUMAN_PROPOSE}" formnovalidate>Propose what not to bind to</button></p>
{_human_candidates(held)}
<label class="field">Or name them yourself — accessions, entry names or gene symbols
<input type="text" name="{HUMAN_NAMES}" size="48" autocomplete="off"
 placeholder="P02768 HBA1 FCGRT"></label>
</fieldset>
<fieldset class="results-retrieve">
<legend>2. The target to bind to</legend>
<label class="field">UniProt accession of the intended target
<input type="text" name="{HUMAN_TARGET}" size="16" autocomplete="off" placeholder="P05113"></label>
<p class="hint">Named so it is kept out of the proposal. A binder aimed at a human protein
resembles that protein's family most of all, and the intended target leading the list of
things to avoid is noise, not a finding.</p>
</fieldset>
<fieldset class="results-retrieve">
<legend>3. A model of the design with one of them, if you have one</legend>
<p class="hint">This is the only route here that answers whether the design <em>binds</em>
what it must not. Nothing in this program docks or folds a pair: fold it elsewhere, paste
the model, and the interface is measured against this run's own complex by the same code.</p>
<label class="field">Which of them this model is of — required, and it has to be one you chose
<input type="text" name="{HUMAN_MODEL_OF}" size="16" autocomplete="off" placeholder="P02768"></label>
<label class="field">Paste the mmCIF or PDB of the design with that protein
<textarea name="{HUMAN_MODEL}" rows="3" spellcheck="false"
 placeholder="data_... or ATOM ..."></textarea></label>
</fieldset>
<p class="results-row"><button type="submit" class="nav" name="__action"
 value="{RESULTS_HUMAN_COMPARE}" formnovalidate>Compare the output against what not to bind to</button></p>
{_human_readout(held)}
</div>
"""


def _dynamics_panel(record: Mapping[str, object] | None) -> str:
    """The Conformational Dynamics Analyst tab: ask what moves, then draw it.

    Last of the four, because it reads a finished design: generation, then
    screening, then this. The chain is asked for rather than assumed, which is
    the whole shape of this tab -- the generator takes one polypeptide chain,
    so a design-plus-target complex is not an input it has, and the notice
    above the buttons says so before anything is run rather than after.
    """
    held = record.get("dynamics") if isinstance(record, Mapping) else None
    held = held if isinstance(held, Mapping) else {}
    return f"""<div class="results-panel results-panel-dynamics">
<h2>{_esc(ANALYST_LABEL)}</h2>
<p class="hint">How far this design moves, and which parts of it move first. Every other
structural number in this program is taken on one conformer; this one is taken over an
ensemble.</p>
<p class="protocol-limit">{_esc(DYNAMICS_SINGLE_CHAIN)}</p>
<fieldset class="results-retrieve">
<legend>1. The chain to analyse</legend>
<label class="field">Chain — the design's own, not the target's
<input type="text" name="{DYNAMICS_CHAIN}" size="6" autocomplete="off" placeholder="A"></label>
</fieldset>
<fieldset class="results-retrieve">
<legend>2. What to ask for</legend>
{_dynamics_tasks()}
<label class="field">Conformers (frames)
<input type="text" name="{DYNAMICS_FRAMES}" size="6" autocomplete="off" placeholder="10"></label>
<label class="field">Replicates
<input type="text" name="{DYNAMICS_REPLICATES}" size="6" autocomplete="off" placeholder="1"></label>
</fieldset>
<fieldset class="results-retrieve">
<legend>3. Cuts, if you have them</legend>
<p class="hint">Left blank, the fluctuation is measured and reported and no verdict is
reached. Nothing published says how mobile a design may be, so this program will not
pick a number for you.</p>
<label class="field">Residue fluctuation that counts as mobile (&Aring;)
<input type="text" name="{DYNAMICS_RMSF_MAX}" size="8" autocomplete="off" placeholder="2.0"></label>
<label class="field">Share of the chain allowed to be mobile — needed with the cut above,
because which residues move cannot decide anything without how much of the chain they are
<input type="text" name="{DYNAMICS_MOBILE_MAX}" size="8" autocomplete="off" placeholder="0.15"></label>
<label class="field">Drift from the design the ensemble may sit at (&Aring;)
<input type="text" name="{DYNAMICS_DRIFT_MAX}" size="8" autocomplete="off" placeholder="3.0"></label>
</fieldset>
<p class="results-row"><button type="submit" class="nav" name="__action"
 value="{RESULTS_DYNAMICS_RUN}" formnovalidate>Generate an ensemble and measure it</button></p>
<fieldset class="results-retrieve">
<legend>4. Draw it</legend>
<p class="hint">The page holds the whole ensemble and takes seconds. The movie takes
minutes to an hour depending on how many conformers there are, which is why it is a
separate tick rather than something that happens by default.</p>
<label><input type="checkbox" name="{DYNAMICS_MOVIE}" value="on"> Render an mp4 as well</label>
<p class="results-row"><button type="submit" class="nav" name="__action"
 value="{RESULTS_DYNAMICS_DRAW}" formnovalidate>Draw the ensemble</button></p>
</fieldset>
{_dynamics_readout(held)}
</div>
"""


def _dynamics_tasks() -> str:
    """The three things the generator can be asked for, each with its purpose."""
    rows: list[str] = []
    for index, mode in enumerate(DYNAMICS_TASK_MODES):
        checked = " checked" if index == 0 else ""
        rows.append(
            f'<li><label><input type="radio" name="{DYNAMICS_TASK}" value="{_esc(mode)}"{checked}> '
            f"<strong>{_esc(mode)}</strong></label> "
            f'<span class="protocol-why">{_esc(DYNAMICS_TASK_PURPOSE.get(mode, ""))}</span></li>'
        )
    return f'<ul class="protocol-picks">{"".join(rows)}</ul>'


def _dynamics_readout(record: Mapping[str, object]) -> str:
    """What the run came to: the numbers, the call, and both caveats.

    The caveats are drawn with the numbers rather than under them. A reader who
    has just been handed a fluctuation curve is past the point of reading what
    produced it.
    """
    if not record:
        return ""
    parts: list[str] = []
    state = str(record.get("state", "untested"))
    blocker = record.get("blocker")
    if blocker:
        parts.append(f'<p class="hint">{_esc(str(record.get("reason") or blocker))}</p>')
    measured = record.get("measured")
    if isinstance(measured, Mapping):
        rows = [
            ("Conformers", measured.get("frames")),
            ("Residues", measured.get("residues")),
            ("Chain", measured.get("chain")),
            ("Fluctuation, mean (&Aring;)", measured.get("rmsf_mean_a")),
            ("Fluctuation, highest (&Aring;)", measured.get("rmsf_max_a")),
            ("Drift from the design (&Aring;)", measured.get("drift_a")),
            ("Ensemble spread (&Aring;)", measured.get("spread_a")),
            ("Radius of gyration (&Aring;)", measured.get("radius_of_gyration_a")),
        ]
        cells = "".join(f"<tr><td>{label}</td><td>{_human_cell(value)}</td></tr>" for label, value in rows)
        parts.append(
            f'<table class="protocol-readout"><caption>Measured &mdash; '
            f'<span class="state-{_esc(state)}">{_esc(state)}</span></caption>'
            f"<tr><th>Term</th><th>Value</th></tr>{cells}</table>"
        )
    mobile = record.get("mobile_residues")
    if isinstance(mobile, list) and mobile:
        shown = ", ".join(_esc(str(position)) for position in mobile[:24])
        more = "" if len(mobile) <= 24 else f" and {len(mobile) - 24} more"
        parts.append(f'<p class="hint">Mobile residues: {shown}{more}.</p>')
    for key in ("out_of_distribution", "timescale"):
        said = record.get(key)
        if isinstance(said, str) and said:
            parts.append(f'<p class="protocol-limit">{_esc(said)}</p>')
    sample = record.get("design_sample")
    if isinstance(sample, str) and sample:
        parts.append(f'<p class="hint">Measured on {_esc(sample)}.</p>')
    parts.append(_dynamics_drawn(record.get("drawn")))
    return "".join(parts)


def _dynamics_drawn(drawn: object) -> str:
    """Where the viewer put the page and the movie.

    The paths are named rather than linked. The page is a self-contained file
    on disk, outside anything this server offers over a token, and inventing a
    route to it would be this panel serving arbitrary files to say something it
    can say in a sentence.
    """
    if not isinstance(drawn, Mapping):
        return ""
    parts: list[str] = []
    page = drawn.get("page")
    if isinstance(page, str) and page:
        count = drawn.get("conformers_drawn")
        parts.append(f'<p class="hint">Drew {_esc(count)} conformers. Page: <code>{_esc(page)}</code></p>')
    movie = drawn.get("movie")
    if isinstance(movie, str) and movie:
        parts.append(f'<p class="hint">Movie: <code>{_esc(movie)}</code></p>')
    said = drawn.get("reason")
    if isinstance(said, str) and said:
        parts.append(f'<p class="hint">{_esc(said)}</p>')
    return "".join(parts)


def _human_candidates(record: Mapping[str, object]) -> str:
    """The tick boxes, when a proposal has filled them.

    Before one has run there are no boxes and the list says so rather than
    rendering an empty one: an empty list under "what not to bind to" reads as
    "nothing to avoid", which is the opposite of what it means.
    """
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return '<p class="hint">No proposal yet. Propose, or name proteins below.</p>'
    rows: list[str] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        accession = str(item.get("accession", ""))
        if not accession:
            continue
        gene = str(item.get("gene") or "")
        described = str(item.get("description") or "")
        identity = item.get("identity")
        why = f"identity {identity}" if identity is not None else "proposed"
        label = f"{gene} ({accession})" if gene else accession
        rows.append(
            f'<li><label><input type="checkbox" name="{HUMAN_PICK}" value="{_esc(accession)}"> '
            f"<strong>{_esc(label)}</strong> {_esc(described)}</label>"
            f' <span class="protocol-why">{_esc(why)}</span></li>'
        )
    return f'<ul class="protocol-picks">{"".join(rows)}</ul>' if rows else ""


def _human_readout(record: Mapping[str, object]) -> str:
    """The comparison, route by route. Each route states what it does not answer.

    Drawn from the record the action returned rather than from a sentence, so
    the page shows the same numbers a headless run prints and neither has to be
    kept in step with the other by hand.
    """
    routes = record.get("routes")
    if not isinstance(routes, list) or not routes:
        return ""
    blocks: list[str] = []
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        blocks.append(_human_route(route))
    design = record.get("interface_design")
    if isinstance(design, str) and design:
        blocks.append(f'<p class="hint">The on-target interface was measured on {_esc(design)}.</p>')
    missing = record.get("not_in_proteome")
    if isinstance(missing, list) and missing:
        named = ", ".join(_esc(str(name)) for name in missing)
        blocks.append(f'<p class="hint">Not in the reference proteome, so nothing was searched for them: {named}.</p>')
    return "".join(blocks)


def _human_route(route: Mapping[str, object]) -> str:
    """One route's rows, under what it answers and what it does not."""
    parts = [
        f"<h3>{_esc(str(route.get('route', '')).replace('_', ' '))}</h3>",
        f"<p>{_esc(str(route.get('answers', '')))}</p>",
        f'<p class="protocol-limit">Does not answer: {_esc(str(route.get("does_not_answer", "")))}</p>',
    ]
    blocker = route.get("blocker")
    if blocker:
        parts.append(f'<p class="hint">{_esc(str(route.get("reason") or blocker))}</p>')
    rows = [row for row in (route.get("readings") or []) if isinstance(row, Mapping)]
    if rows and str(route.get("route")) == HUMAN_SEQUENCE_ROUTE:
        parts.append(_human_resemblance_rows(rows))
    margins = [row for row in (route.get("margins") or []) if isinstance(row, Mapping)]
    if margins:
        parts.append(_human_interface_tables(route, margins))
    return "".join(parts)


def _human_cell(value: object) -> str:
    """One number in the readout, or an em-dash where there is none.

    A blank cell and a measured zero look the same in a table, and on this page
    the difference is "not measured" against "no interface at all".
    """
    return "&mdash;" if value is None else _esc(value)


def _human_resemblance_rows(rows: Sequence[Mapping[str, object]]) -> str:
    cells: list[str] = []
    for row in rows:
        state = str(row.get("state", "untested"))
        said = str(row.get("reason") or "")
        if not said and row.get("alignment") is False:
            said = "No alignment over the E-value cut."
        cells.append(
            f"<tr><td>{_esc(str(row.get('label') or row.get('accession') or ''))}</td>"
            f'<td class="state-{_esc(state)}">{_esc(state)}</td>'
            f"<td>{_human_cell(row.get('identity'))}</td>"
            f"<td>{_human_cell(row.get('coverage'))}</td>"
            f"<td>{_esc(said)}</td></tr>"
        )
    return (
        '<table class="protocol-readout"><tr><th>Protein</th><th>State</th>'
        f"<th>Identity</th><th>Coverage</th><th>Note</th></tr>{''.join(cells)}</table>"
    )


#: What each interface term is called on the page. The record keys are the
#: measuring module's own names; these are the words for a reader.
_HUMAN_TERM_LABELS = {
    "dsasa_total_a2": "Buried area (A²)",
    "interface_atom_contacts": "Atom contacts",
    "interface_polar_contacts": "Polar contacts",
    "shape_complementarity": "Shape complementarity",
}


def _human_interface_tables(route: Mapping[str, object], margins: Sequence[Mapping[str, object]]) -> str:
    """One table per protein: every term on-target, off-target, and the lead.

    A lead on its own is not readable. 213 square angstrom more interface
    on-target than off-target is a different finding over a 300 square angstrom
    interface than over a 3000 one, and the lead alone cannot tell the two
    apart. Both absolutes are already on the route record -- ``on_target`` and
    ``readings`` -- so this draws them rather than storing a second copy of a
    number that is free to drift from the first.
    """
    on_target = route.get("on_target")
    on_target = on_target if isinstance(on_target, Mapping) else {}
    readings = {str(row.get("where")): row for row in (route.get("readings") or []) if isinstance(row, Mapping)}
    terms = [str(name) for name in (route.get("terms") or [])]
    blocks: list[str] = []
    for margin in margins:
        where = str(margin.get("where") or "")
        state = str(margin.get("state", "untested"))
        said = str(margin.get("reason") or "")
        note = f'<p class="hint">{_esc(said)}</p>' if said else ""
        caption = f'Against {_esc(where)} &mdash; <span class="state-{_esc(state)}">{_esc(state)}</span>'
        if not any(margin.get(term) is not None for term in terms):
            # No table where no margin was taken: every cell would be an
            # em-dash, and four rows of them under a heading is noise around
            # the one sentence that says why.
            #
            # Tested on the numbers and not on the presence of a reason, which
            # is a different question: ``margin_threshold_not_set`` measured
            # both sides perfectly well and is withholding only the verdict,
            # and drawing no table for it hid the measurement behind "no cut
            # was named". The text readout makes the same test, and the two
            # renderings have to agree.
            blocks.append(f'<p class="protocol-against">{caption}</p>{note}')
            continue
        theirs = readings.get(where, {})
        rows = "".join(
            f"<tr><td>{_esc(_HUMAN_TERM_LABELS.get(term, term.replace('_', ' ')))}</td>"
            f"<td>{_human_cell(on_target.get(term))}</td>"
            f"<td>{_human_cell(theirs.get(term))}</td>"
            f"<td>{_human_cell(margin.get(term))}</td></tr>"
            for term in terms
        )
        blocks.append(
            f'<table class="protocol-readout"><caption>{caption}</caption>'
            "<tr><th>Term</th><th>On-target</th><th>Off-target</th><th>Lead</th></tr>"
            f"{rows}</table>{note}"
        )
    return "".join(blocks)


def _retrieve_row(which: str, label: str, action: str, *, accession: bool, label_override: str = "") -> str:
    """One model's retrieval routes: paste, open a file, and for AF2 an accession.

    Three routes rather than one because the step already taken differs.
    Neither service can be driven end to end -- the server has no submission
    API and the database accepts no sequence -- so a person arrives here having
    downloaded a file, or with the text on their clipboard, or knowing only an
    accession. Pasting asks least of the machine and is offered first.
    """
    extra = ""
    if accession:
        extra = (
            '<label class="field">UniProt accession'
            f' <input type="text" name="{which}_accession" size="12"'
            ' placeholder="P05113" autocomplete="off"></label>'
        )
    return f"""<fieldset class="results-retrieve">
<legend>{_esc(label)}</legend>
<label class="field">Paste the mmCIF or PDB
<textarea name="{which}_paste" rows="3" spellcheck="false"
 placeholder="data_... or ATOM ..."></textarea></label>
<label class="field">Or open a downloaded file
<input type="text" name="{which}_path" size="40" autocomplete="off"
 placeholder="a path to a .cif, .pdb or .zip"></label>
{extra}
<p class="results-row"><button type="submit" class="nav" name="__action"
 value="{action}" formnovalidate>{_esc(label_override or "Retrieve and compare")}</button></p>
</fieldset>
"""


def _page(title: str, body: str, submit: str, *, restart: bool = True) -> str:
    """One panel page. The project name is the top heading on every page.

    Back, Forward, and the action sit beside that heading. The buttons stay
    after the fields in the markup, so the action remains the last submit
    control. A step name is the line under the project name.
    """
    actions = (
        '<div class="page-actions page-actions-top">'
        f"{_session_buttons() if restart else ''}"
        f"{NAV_SLOT}"
        f'<button type="submit">{_esc(submit)}</button></div>'
    )
    heading = (
        '<div class="title-row">'
        f"{_heading(PANEL_TITLE)}"
        f'<p class="project-name">{_esc(PROJECT_NAME)}</p>'
        "</div>\n"
    )
    step = "" if title == PANEL_TITLE else f'<p class="step-title">{_esc(title)}</p>\n'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(" ".join(PANEL_TITLE.splitlines()))}</title>
<style>{_PAGE_STYLE}</style>
</head>
<body>
<main>
<form class="landing" method="post" action="/" enctype="multipart/form-data">
{heading}{step}{body}
{actions}
<div id="run-status" hidden>
<p id="run-status-label"></p>
<progress id="run-progress" value="0" max="1"></progress>
</div>
</form>
</main>
<script>{_PAGE_SCRIPT}</script>
</body>
</html>
"""


def _done_page() -> str:
    return _page(PANEL_TITLE, "<section><h2>Panel closed</h2><p>Return to the terminal.</p></section>", "Close")


def _session_buttons() -> str:
    """Return to start and Restart Server. Both skip validation, and both sit before the action."""
    return (
        '<button type="submit" class="nav" name="__nav" value="start" formnovalidate>'
        "Return to start page</button>\n"
        '<button type="submit" class="nav" name="__nav" value="fresh" formnovalidate>'
        "Restart Server</button>\n"
    )


def _shutdown_page() -> str:
    """The response to Close. The script closes the tab, and the server closes the window."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Closing</title>
</head>
<body>
<p>Closing the panel.</p>
<script>
window.close();
setTimeout(function () { window.close(); }, 200);
</script>
</body>
</html>
"""


def _heading(title: str) -> str:
    """The page title. A second line stays on its own row, and repeated spaces stay."""
    lines = [_heading_line(line) for line in title.splitlines() if line]
    return "<h1>" + "<br>".join(lines) + "</h1>\n"


def _heading_line(line: str) -> str:
    """Escape one title line and keep a run of spaces from collapsing."""
    return re.sub(r" {2,}", lambda match: "&nbsp;" * len(match.group(0)), _esc(line))


def _main_page() -> str:
    """Landing page shown when a refresh has no stored step."""
    return render_options(OptionForm())


def _select_options(options: Iterable[tuple[str, str]], selected: str) -> str:
    lines: list[str] = []
    for value, label in options:
        mark = " selected" if value == selected else ""
        lines.append(f'<option value="{_esc(value)}"{mark}>{_esc(label)}</option>')
    return "\n".join(lines)


def _where(hosts: str | None) -> tuple[str, str]:
    names = _split_hosts(hosts or "")
    remote = [name for name in names if name.casefold() not in {"local", "localhost"}]
    has_local = any(name.casefold() in {"local", "localhost"} for name in names)
    if remote and has_local:
        return "local-and-remote", ", ".join(remote)
    if remote:
        return "remote", ", ".join(remote)
    return "local", ""


def _worker_choice(value: int | None) -> tuple[str, str]:
    if value is None:
        return "every", ""
    text = str(value)
    if text in _WORKER_PRESETS:
        return text, ""
    return "custom", text


def _hosts_from_fields(fields: dict[str, list[str]]) -> str | None:
    where = _first(fields, "where") or "local"
    remote = remote_names_from_fields(fields)
    if where == "local":
        if remote:
            raise ValueError(
                "Remote machines are named but generation is set to this machine only. "
                "Choose where generation runs in the dropdown, or clear the machines."
            )
        return None
    if where == "remote":
        if not remote:
            raise ValueError("Name at least one remote machine, or choose this machine in the dropdown.")
        return ",".join(remote)
    if where == "local-and-remote":
        return ",".join(["local", *remote])
    raise ValueError(f"Unknown machine choice {where!r}.")


def _workers_from_fields(
    fields: dict[str, list[str]],
    *,
    field: str = "local_workers",
    label: str = "Local workers",
) -> int | None:
    """One worker count, from its dropdown and its custom box.

    ``field`` names the pair -- ``local_workers`` with ``local_workers_custom``,
    or ``remote_workers`` with ``remote_workers_custom``. Parameterised rather
    than copied: the two differ only in which field they read and what the error
    calls it, and a second copy of this would be a second chance for the two to
    disagree about what ``every`` or a blank custom box means.
    """
    selected = _first(fields, field) or "every"
    if selected == "every":
        return None
    if selected == "custom":
        chosen = _optional_int(_first(fields, f"{field}_custom"), label)
    elif selected.lstrip("-").isdigit():
        chosen = int(selected)
    else:
        raise ValueError(f"{label} must be a whole number.")
    # One floor for both branches, because ``lstrip("-").isdigit()`` accepts a
    # minus sign and ``_optional_int`` is shared with fields where a negative is
    # not this question's problem. ``min="1"`` on the box is only a hint the
    # browser may enforce; a posted form is not obliged to have come from it.
    #
    # Refused here rather than left to the CLI and the dispatcher, which both
    # already refuse it: by then the designs have been chosen and the jobs are
    # about to be written, and the person is told on a different screen from the
    # one holding the field they have to correct. Nothing is lost by refusing
    # early -- a worker count below one asks for no workers, and "no limit" is
    # what the dropdown's own first option already says.
    if chosen is not None and chosen < 1:
        raise ValueError(f"{label} must be at least 1, or leave the dropdown on its 'every ... at once' option.")
    return chosen


def _optional_int(text: str, label: str) -> int | None:
    if not text:
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    raise ValueError(f"{label} must be a whole number.")


def _optional_float(text: str, label: str) -> float | None:
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{label} must be a number.") from None


def _split_hosts(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]


def _checked(fields: dict[str, list[str]], name: str) -> bool:
    return any(value.strip() for value in fields.get(name, []))


def flat_fields(fields: Mapping[str, Sequence[str]]) -> list[str]:
    """The panel's submission as the flat ``name=value`` list actions read.

    A form arrives as a mapping of name to values, and every results action
    reads one flat list. Handing the mapping straight over iterated its keys,
    so ``af3c_path=`` never matched a bare ``af3c_path`` and every typed path,
    pasted model and accession was dropped on the floor. What survived was the
    no-input fallback, so the page answered that no download could be found
    while the reader was looking at the path they had just typed into it.
    """
    return [f"{name}={value}" for name, values in fields.items() for value in values]


def _first(fields: dict[str, list[str]], name: str) -> str:
    values = fields.get(name) or []
    return values[0].strip() if values else ""


def _blank(text: str) -> str | None:
    return text or None


def _on(checked: bool) -> str:
    return " checked" if checked else ""


def _esc(value: object) -> str:
    return escape(str(value), quote=True)


def _running_in_wsl() -> bool:
    version = Path("/proc/version")
    return version.is_file() and "microsoft" in version.read_text(encoding="utf-8", errors="ignore").casefold()


def _windows_cwd() -> str | None:
    """A Windows directory cmd and PowerShell can start from.

    ``start`` inherits the repository directory. From ``A:\\`` that launch is
    denied, so the browser is started from ``C:\\Windows`` instead.
    """
    candidate = "C:\\Windows" if sys.platform.startswith("win") else "/mnt/c/Windows"
    return candidate if Path(candidate).is_dir() else None


def _panel_port(url: str) -> str:
    """The port in an address such as ``http://127.0.0.1:41947/``."""
    tail = url.rsplit(":", 1)[-1]
    digits: list[str] = []
    for char in tail:
        if not char.isdigit():
            break
        digits.append(char)
    return "".join(digits) or "0"


def _ps_single(value: str) -> str:
    """One PowerShell single-quoted literal."""
    return "'" + value.replace("'", "''") + "'"


def _launch_script(url: str) -> str:
    """Open the panel in its own window, so Close does not close any other window."""
    statements = [
        "$ErrorActionPreference='Stop'",
        f"$url={_ps_single(url)}",
        f"$port={_ps_single(_panel_port(url))}",
        "$dir=Join-Path ([System.IO.Path]::GetTempPath()) ('rfd4-panel-' + $port)",
        "New-Item -ItemType Directory -Force -Path $dir | Out-Null",
        '$edge="$env:ProgramFiles\\Microsoft\\Edge\\Application\\msedge.exe"',
        '$edge86="${env:ProgramFiles(x86)}\\Microsoft\\Edge\\Application\\msedge.exe"',
        "$edgeLocal=Join-Path $env:LOCALAPPDATA 'Microsoft\\Edge\\Application\\msedge.exe'",
        '$chrome="$env:ProgramFiles\\Google\\Chrome\\Application\\chrome.exe"',
        '$chrome86="${env:ProgramFiles(x86)}\\Google\\Chrome\\Application\\chrome.exe"',
        "$chromeLocal=Join-Path $env:LOCALAPPDATA 'Google\\Chrome\\Application\\chrome.exe'",
        "$browser=$null",
        "foreach($candidate in @($edge,$edge86,$edgeLocal,$chrome,$chrome86,$chromeLocal)){",
        "if(Test-Path -LiteralPath $candidate){$browser=$candidate; break}}",
        (
            "if($browser){Start-Process -FilePath $browser -ArgumentList @("
            "'--app=' + $url,'--user-data-dir=' + $dir,'--no-first-run','--disable-sync'"
            ")} else {Start-Process $url}"
        ),
    ]
    return "; ".join(statements)


def _close_script(port: str) -> str:
    """End processes whose command line names this panel's window, and no others."""
    statements = [
        f"$port={_ps_single(port)}",
        "$dir=Join-Path ([System.IO.Path]::GetTempPath()) ('rfd4-panel-' + $port)",
        "$names=@('msedge.exe','chrome.exe')",
        "Get-CimInstance Win32_Process | Where-Object {",
        "$names -contains $_.Name -and $_.CommandLine -and $_.CommandLine.Contains($dir)",
        "} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
    ]
    return "; ".join(statements)


def _open_browser(url: str) -> None:
    """Open the panel. From WSL this asks Windows to open its own window."""
    if not (_running_in_wsl() or sys.platform.startswith("win")):
        webbrowser.open(url)
        return
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", _launch_script(url)],
        cwd=_windows_cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return
    detail = (completed.stderr or completed.stdout).strip()
    print(f"The browser did not open. Paste the panel address into it. {detail}".rstrip())
