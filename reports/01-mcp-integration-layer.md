# Production report 1 — MCP Integration Layer

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** the Model Context Protocol server through which an assistant drives the program.
**Status:** implemented and tested. Not medically verified. Nothing in this report is a
clinical claim.

## Where the code lives

Every path in this report is relative to the repository root of the RFD4-Proteina fork
that carries the disease flow, which is this repository. The code is present here, so every
statement below can be checked against the source it names.

| Component | Path |
|---|---|
| Server (JSON-RPC over stdio) | `scripts/humanitys_last_disease_mcp.py` |
| Tool implementations | `src/rfproteina/disease/mcp_api.py` |
| Three-layer self-description and stored-state readers | `src/rfproteina/disease/mcp_layers.py` |
| Subprocess execution and cancellation | `src/rfproteina/disease/dispatch.py` |
| Public-service rate policy | `src/rfproteina/disease/limits.py` |
| Server registration | `.mcp.json` |
| Server registration, per-tool timeouts | `.grok/config.toml` |

## What this layer is

A hand-written JSON-RPC stdio server. It does not use an MCP SDK. It is registered as a
single server:

```json
{"mcpServers": {"Humanitys-Last-Disease": {"command": "python", "args": ["-u", "scripts/humanitys_last_disease_mcp.py"]}}}
```

Tools are dispatched **in process** by `call_tool()` in `mcp_api.py`. The server does not
open a shell, does not read from standard input for a prompt, and does not open a browser.

### Protocol

Two protocol eras are answered from one server:

- `2026-07-28` — stateless, carried in `_meta` (`MODERN_VERSIONS`).
- `2025-11-25`, `2025-06-18`, `2025-03-26`, and `2024-11-05` are the
  `initialize` handshake revisions (`LEGACY_VERSIONS`). [MCP version negotiation](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)

`SUPPORTED_VERSIONS` is the concatenation, so a client of either era connects without a
shim.

## MCP tools

Descriptions are the tools' own, as declared in `scripts/humanitys_last_disease_mcp.py`.

| Tool | Declared behaviour |
|---|---|
| `list_options` | Lists every option by the name `run` accepts. Does not start the program or open the browser. |
| `run` | Runs any headless program mode with the selected options. This includes the disease pipeline, benchmark answers, dynamics, H.U.M.A.N. protocol analysis, screening, refolding and corpus declaration. Generation starts **only** when `options.run` is true. |
| `screen_directory` | Screens a finished generation directory, or every finished run beneath it, and returns each verdict. It accepts the screening-only table-staging and paralog-identity controls. |
| `read_verdict` | Reads stored verdicts under a directory. Does not rescreen, write, generate, or start a shell. |
| `describe_architecture` | Describes the three layers. Does not call a model, start generation, or use the GPU. |
| `retrieve_databases` | Reads stored database statuses for one directory. **Does not contact a database.** |
| `retrieve_structures` | Reads stored structure rows for one directory. **Does not download a structure** or return its sequence. |
| `read_sequence_measures` | Reads stored sequence measurements. **The sequence text is not returned.** |
| `read_results` | Reads stored research traces, generation records and screen verdicts. |
| `list_output_files` | Lists regular artifacts under one approved finished run, with pagination, size and media type. Links, hidden files and credential/private-key names are omitted. |
| `read_output_file` | Returns a chunk, its encoding and digest, the whole-file digest, and the next byte offset. |
| `backend_status` | Reports the computational backend for one directory. Does not start GPU work, docking, or a prediction. |
| `declare_corpora` | Returns declared training corpora, benchmark coverage and available overlap information. |
| `plan_validation` | Lists candidate IDs and recommended local checks for a finished run. |
| `validate_output_files` | Runs the requested read-only file and evidence checks. |
| `save_validation_report` | Runs the requested validation checks and saves a new JSON report inside the run. |
| `list_validation_reports` | Lists saved validation report filenames for a finished run. |
| `plan_screening` | Lists scientific screening choices for this run or candidate with installation, applicability, workload, operation and recommendation. |
| `run_selected_screens` | Runs the selected calculations or readers in order and returns a result for each selected method. |

The local write tool `save_validation_report` advertises `readOnlyHint: false` and
`openWorldHint: false`. `run` and `screen_directory` advertise writes and potential network
access. Inspection tools advertise read-only access.

`list_options` is generated from the CLI parser and returns the JSON Schema accepted by `run`.
Every CLI option is available there except the browser launcher and browser-only validation option.
The validation workflow is exposed through `plan_validation`, `validate_output_files`,
`save_validation_report`, and `list_validation_reports`.

`run` also returns structured `structure_options`, `dynamics`, and `human_protocol` records when
those stages produce them. A client does not need to parse progress prose to obtain those values.
Each structure row has an opaque, order-independent `selector`. Pass one or more back through
`options.select_structure`; the selecting call resolves them against its own fresh lookup and refuses
a stale selector before any structure download, job write, or generation.

For candidate validation, call `plan_validation` with `directory` set to one finished run
under `runs`, `disease_runs`, or a root configured through `RFD4_MCP_ROOTS`. Pass a returned
`candidate_id` for candidate-specific recommendations. Use `next_candidate_offset` as
`candidate_offset` to see more candidates. Call `validate_output_files` with the same
scope and an explicit list of installed check IDs. Execution checks a workload of up to
512 files and 256 MiB on disk per request. The review-receipt check reports whether a regular
`synthesis_review.json` is present in the run. It reads no receipt contents and skips symbolic links.
Instead of copying IDs from a prior plan, a validation or report call may set `selection` to
`recommended` or `all_installed`; the server resolves the current plan inside that same call.

Call `save_validation_report` with the same scope and check IDs to run the checks against the
current inputs and create a report under `validation_reports/`. The response includes
`report_file`, relative to the run. Each successful call creates a new file.

Call `list_validation_reports` with that run's `directory` to inventory saved reports.
It returns report filenames relative to the run in `report_files`. Listing stops with
an error if `validation_reports/` contains more than 512 entries.

For scientific screening, call `plan_screening` with the same directory and optional
candidate ID. Pass `selection: "recommended"`, `selection: "all_installed"` or an explicit
`methods` array to `run_selected_screens`. An empty array runs zero methods. A selected
method can calculate a measurement, read a native result or return previously recorded
observations. The plan identifies that operation before execution. Use `screen_directory`
with `screen_selection` or `screen_methods` to save a run-level verdict with the selections.

The browser and MCP share this screening planner. A method can be installed and manually
selectable even when the candidate lacks the inputs needed for a measurement; execution records
that outcome and its reason. [Screen selection guide](../docs/reference/screen-inputs.md#choose-scientific-screens-for-a-finished-candidate).

Call `list_output_files` to inventory eligible non-hidden regular artifacts under one approved concrete
run, including generated structures, metrics, logs, records and validation reports. Pass its
`inventory_sha256` as `expected_sha256` on later pages to require the same paths, sizes and media types. Use
`read_output_file` with the returned relative path. Each call returns up to 1 MiB. Pass the first
response's `file_sha256` as `expected_sha256` alongside each subsequent `offset`. If the artifact
changes, restart at offset zero. Decode each chunk using its own `encoding`, especially when a
page boundary splits a UTF-8 character. Each read scans the complete file for credential signatures
and checks a file size limit of 256 MiB. Listings stop if the service cannot read a complete inventory.

`read_verdict` includes native Protein-Sol scores and per-sample coverage when present. AutoDock
Vina reports an empirical score for a supplied pose. The backend lists docking pose search separately.

## Declared capabilities

`describe_architecture()` in `mcp_layers.py` declares the capability-to-tool mapping. The
third column is the part a reader needs and the tool names do not convey.
The `tools/list` response also includes input schemas, output schemas and annotations for each tool.

| Capability | Tool | What it does |
|---|---|---|
| Program options | `list_options` | Lists the option names `run` accepts. Starts nothing. |
| Verdict retrieval | `read_verdict` | **Reads stored verdicts with their per-axis evidence. Rescreens nothing.** |
| Database retrieval | `retrieve_databases` | **Reads stored JSON. Contacts no database.** |
| Protein structure retrieval | `retrieve_structures` | **Reads stored rows. Downloads nothing.** |
| Sequence processing | `read_sequence_measures` | **Reads stored measurements.** Returns no sequence text. |
| Generative design execution | `run` | Live, and only when `options.run` is true. |
| Benchmark answers | `run` | Runs `options.benchmark_answer` and returns `benchmark_answers` as structured records. |
| Conformational dynamics | `run` | Runs the installed headless dynamics workflow without requiring a disease selection. |
| H.U.M.A.N. protocol | `run` | Runs the installed headless protocol workflow without requiring a disease selection. |
| Refolding workflow | `run` | Prepares fold requests or imports returned folds and returns the structured refold record. |
| Screening pipeline execution | `screen_directory` | Live. Contacts public databases only when `lookup_targets` is true. |
| Results retrieval | `read_results` | Reads stored records. |
| Output artifact inventory | `list_output_files` | Lists bounded regular-file metadata under one approved finished run. |
| Output artifact reading | `read_output_file` | Reads bounded chunks without following symbolic links. |
| Candidate validation recommendations | `plan_validation` | Lists installed checks, applicability and workload for a finished run or candidate. |
| Read-only output validation | `validate_output_files` | Runs only the requested file and evidence checks. |
| Save output validation report | `save_validation_report` | Creates a local report for the requested checks. |
| List saved validation reports | `list_validation_reports` | Lists the filenames of saved reports in one finished run. |
| Training-corpus declaration | `declare_corpora` | Returns the corpus and benchmark coverage the project declares. |
| Computational backend status | `backend_status` | Reports installed backend components for a directory. |

The retrieval tools read stored files. Live database lookups happen inside `run` or
`screen_directory` with `lookup_targets` enabled. A directory with no stored lookup reports
`not_recorded`.

## Safety properties

**Path confinement.** General tool paths resolve inside the checkout or an operator-configured
root. File and validation calls require a concrete run under `runs`, `disease_runs`, or a root
explicitly listed in `RFD4_MCP_ROOTS`. On Windows the launcher converts native semicolon-separated
roots with `wslpath` and passes the resulting paths directly to the Linux server.

**Structured-result redaction.** Verdict, result and sequence-measure tools continue to drop
designed sequence text by key name and residue-run pattern. `read_output_file` is the explicit raw
artifact boundary: it can return generated structure or FASTA bytes selected by the caller, while
still refusing credentials, links, traversal and unbounded reads.

**Secrets are refused at the schema boundary.** A field whose name looks like a secret is
rejected rather than forwarded, so an assistant cannot hand the program an API key through
a tool call.

**The server never prompts.** The program it drives is interactive; when that program asks
a question with nobody to answer, the server raises `_Unanswered`, which subclasses
`BaseException` rather than `Exception`. This is precise and load-bearing: the program's
entry point ends in `except Exception`, so an ordinary exception would be logged as an
unexpected failure and the prompt loop would ask again forever. Deriving from
`BaseException` passes straight through that handler.

**Remote generation arguments.** MCP accepts local aliases, host names, addresses and
`user@host`, and constrains remote output paths to ordinary POSIX segments. SSH options, shell
metacharacters, upward traversal and more than 64 hosts are refused before the program starts.

**Protocol and output bounds.** A JSON-RPC batch holds at most 256 items, no more than 32 tool
calls may be outstanding, program text capture is bounded, directory inventories are bounded, and
tool errors are scrubbed before they enter the protocol response.

**Cancellation.** `stop_work()` is not a tool; it is a hook into `dispatch.stop_running()`,
which terminates registered child processes. Generation children are registered, so it can
terminate them. It is best effort by the code's own account: a remote job receives the
signal only at the local `ssh`, a file copy already in flight is not held at all, and the
program's short-lived helpers (`scp`, `reg.exe`, PowerShell, the browser opener) are not
registered.

**Rate policy.** Outbound requests are paced per host by `limits.py` wherever the caller
takes a `limits.slot()` — the JSON readers in `sources.py` and `research.py`. One service is
paced to its published rate; the rest are given a deliberately conservative concurrency cap.
The structure mmCIF download, the screening-table HEAD and body fetches, and the
predicted-model fetch take no slot and are unpaced; the structure download also sends only
an `Accept` header, so it goes out under the bare interpreter user agent rather than this
program's.

## Interface boundaries

- Inspection tools do not start model or GPU work.
- Structure fetching and live database queries run through `run` or `screen_directory`.
- API keys are read from the program's configured environment and are not accepted in MCP arguments.
- Structured summaries remove designed sequence text; exact run artifacts require an explicit `read_output_file` call.

## Verification

Covered by the MCP protocol, modes, evidence, naming, layer and validation suites under `tests/`,
including `test_disease_mcp.py`, `test_disease_mcp_protocol.py`, `test_disease_mcp_modes.py`,
`test_disease_mcp_evidence.py`, `test_disease_mcp_naming.py`, `test_disease_mcp_layers.py`, and
`test_disease_mcp_validation.py`, plus the artifact boundary in `test_disease_mcp_outputs.py`.
The screening-selection tests exercise installation checks, candidate matching, recommended
selections and manual overrides.
