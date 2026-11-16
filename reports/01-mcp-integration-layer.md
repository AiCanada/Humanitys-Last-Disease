# Production report 1 — MCP Integration Layer

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** the Model Context Protocol server through which an assistant drives the program.
**Status:** implemented and tested. Not medically verified. Nothing in this report is a
clinical claim.

## Where the code lives

Every path in this report is relative to the repository root of the RFD4-Proteina fork
that carries the disease flow. **That code is not in this repository yet** — this repository
currently holds the upstream tree. A reader looking for `src/rfproteina/disease/` here will
not find it; the paths are given so each statement can be checked against the source when it
is published.

| Component | Path |
|---|---|
| Server (JSON-RPC over stdio) | `scripts/humanitys_last_disease_mcp.py` |
| Tool implementations | `src/rfproteina/disease/mcp_api.py` |
| Three-layer self-description and stored-state readers | `src/rfproteina/disease/mcp_layers.py` |
| Subprocess execution and cancellation | `src/rfproteina/disease/dispatch.py` |
| Public-service rate policy | `src/rfproteina/disease/limits.py` |
| Server registration | `.mcp.json` |

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
- `2025-06-18`, `2025-03-26`, `2024-11-05` — the handshake revisions (`LEGACY_VERSIONS`).

`SUPPORTED_VERSIONS` is the concatenation, so a client of either era connects without a
shim.

## The ten tools

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

Each tool carries MCP annotations. Only `run` and `screen_directory` declare
`readOnlyHint: false`, `destructiveHint: true`, `openWorldHint: true`. The other eight
declare `readOnlyHint: true`, `openWorldHint: false`.

## The eight declared capabilities, and what each one actually does

`describe_architecture()` in `mcp_layers.py` declares the capability-to-tool mapping. The
third column is the part a reader needs and the tool names do not convey.

| Capability | Tool | What it does |
|---|---|---|
| Program options | `list_options` | Lists the option names `run` accepts. Starts nothing. |
| Verdict retrieval | `read_verdict` | **Reads stored verdicts with their per-axis evidence. Rescreens nothing.** |
| Database retrieval | `retrieve_databases` | **Reads stored JSON. Contacts no database.** |
| Protein structure retrieval | `retrieve_structures` | **Reads stored rows. Downloads nothing.** |
| Sequence processing | `read_sequence_measures` | **Reads stored measurements.** Returns no sequence text. |
| Generative design execution | `run` | Live, and only when `options.run` is true. |
| Screening pipeline execution | `screen_directory` | Live. Contacts public databases only when `lookup_targets` is true. |
| Results retrieval | `read_results` | Reads stored records. |

**Six of the eight are read-only views of files on disk, not the retrieval their names
suggest.** The real database traffic and the real structure downloads happen inside `run`,
or inside `screen_directory` with `lookup_targets` enabled. This is deliberate and the
module says so in its own docstring: *"None of these calls starts a model, a GPU job,
docking, or a prediction."* A directory with no stored lookup reads `not_recorded` rather
than an empty result, so absence of a record is distinguishable from a record of absence.

## Safety properties

**Path confinement.** A requested directory is used as given unless `RFD4_MCP_ROOTS` names
directories the server may use, in which case every path must resolve inside this checkout
**or** one of those directories, and a path outside both is refused.

**No sequence leaves the server.** Designed sequence text is dropped on the way out, by
key name and by residue-run pattern. `read_sequence_measures` returns the measurements of a
chain without returning the chain.

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

## What this layer does not do

- It does not start a model, a GPU job, docking, or a structural prediction.
- It does not fetch a structure or query a database except through `run` /
  `screen_directory`.
- It does not accept API keys.
- It does not return designed sequences.

## Verification

Covered by the project's own test suite (`tests/test_disease_mcp.py`,
`tests/test_disease_mcp_layers.py`), which runs green as part of the disease suite.
