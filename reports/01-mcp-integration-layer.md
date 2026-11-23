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
- `2025-11-25`, `2025-06-18`, `2025-03-26`, `2024-11-05` — the handshake revisions (`LEGACY_VERSIONS`).

`SUPPORTED_VERSIONS` is the concatenation, so a client of either era connects without a
shim.

## Available tools

Descriptions are the tools' own, as declared in `scripts/humanitys_last_disease_mcp.py`.

| Tool | Declared behaviour |
|---|---|
| `list_options` | Lists every option by the name `run` accepts. Does not start the program or open the browser. |
| `run` | Runs the program with the selected options. Does not open the browser and does not prompt. Generation starts **only** when `options.run` is true. |
| `screen_directory` | Screens a finished generation directory, or every finished run beneath it, and returns each verdict. |
| `read_verdict` | Reads stored verdicts under a directory. Does not rescreen, write, generate, or start a shell. |
| `describe_architecture` | Describes the three layers. Does not call a model, start generation, or use the GPU. |
| `retrieve_databases` | Reads stored database statuses for one directory. **Does not contact a database.** |
| `retrieve_structures` | Reads stored structure rows for one directory. **Does not download a structure** or return its sequence. |
| `read_sequence_measures` | Reads stored sequence measurements. **The sequence text is not returned.** |
| `read_results` | Reads stored research traces, generation records and screen verdicts. |
| `backend_status` | Reports the computational backend for one directory. Does not start GPU work, docking, or a prediction. |
| `declare_corpora` | Returns declared training corpora and available overlap information. |
| `plan_screening` | Lists installed and unavailable methods, their input requirements and candidate-specific recommendations. |
| `run_selected_screens` | Runs the recommended set, all installed methods or explicit method IDs and reports each outcome. |
| `plan_validation` | Lists local output checks for the selected run or candidate. |
| `validate_output_files` | Executes the selected output checks and returns their findings. |
| `save_validation_report` | Executes the checks and saves a JSON report in the finished run. |
| `list_validation_reports` | Lists saved report filenames for the run. |
| `list_output_files` | Lists eligible regular artifacts with pagination. |
| `read_output_file` | Returns a bounded chunk of a selected artifact and its SHA-256 digest. |

The browser and MCP share the screening planner. A method can be installed and manually selectable even when the candidate lacks the inputs needed for a measurement. Execution records that outcome and its reason. [Screen selection guide](../docs/reference/screen-inputs.md#choose-scientific-screens-for-a-finished-candidate).

## Capability discovery

`describe_architecture` lists the capability-to-tool mapping. The `tools/list` response includes input schemas, output schemas and annotations for each tool. The retrieval tools read stored records. A directory with no stored lookup reports `not_recorded`.

## Safety properties

**Path confinement.** The launched server defaults to the checkout. Operators can add absolute directories through `RFD4_MCP_ROOTS`. Artifact tools require a finished run under an approved root.

**Sequence and artifact access.** Stored measurement summaries redact designed sequence text. Explicit reads of an eligible output file can return its original content, including a generated structure or sequence.

**Secrets are refused at the schema boundary.** A field whose name looks like a secret is
rejected rather than forwarded, so an assistant cannot hand the program an API key through
a tool call.

**The server never prompts.** The program it drives is interactive; when that program asks
a question with nobody to answer, the server raises `_Unanswered`, which subclasses
`BaseException` rather than `Exception`. This is precise and load-bearing: the program's
entry point ends in `except Exception`, so an ordinary exception would be logged as an
unexpected failure and the prompt loop would ask again forever. Deriving from
`BaseException` passes straight through that handler.

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

## Verification

The MCP regression suite covers discovery, tool schemas, protocol frames, output access and validation reports. The screening selection tests exercise installation checks, candidate matching, recommended selections and manual overrides.
