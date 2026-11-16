# Production report 2 — AI Interface

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** how the program asks a language model, and how credentials are held.
**Status:** implemented. Not medically verified. A model reply is not a measurement.

## Where the code lives

Paths are relative to the repository root of the RFD4-Proteina fork carrying the disease
flow, which is not in this repository yet.

| Component | Path |
|---|---|
| Provider client and model tables | `src/rfproteina/disease/research.py` |
| Desktop-app path | `src/rfproteina/disease/grok_bot.py` |
| Credential store | `src/rfproteina/disease/key_store.py` |
| Redaction of records | `src/rfproteina/disease/logs.py` |

## The shape of it, stated plainly

There is **no provider SDK and no class hierarchy**. The entire HTTP client is one
function, `_request_completion()` in `research.py`, built on `urllib.request` and raw JSON.
No `openai` or `anthropic` package is a dependency.

Five providers are reachable:

| Provider | Endpoint |
|---|---|
| xAI (Grok) | `https://api.x.ai/v1/chat/completions` |
| OpenAI | `https://api.openai.com/v1/chat/completions` |
| NVIDIA NIM | `https://integrate.api.nvidia.com/v1/chat/completions` |
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` |
| Anthropic (Claude) | `https://api.anthropic.com/v1/messages` |

Four of the five share one code path through an endpoint table, because they are all
OpenAI-compatible `/chat/completions`. **Claude is a special-cased branch**, with its own
body shape (`system` as a top-level field, `max_tokens` required) and its own headers
(`x-api-key`, `anthropic-version`).

Per-provider divergence is handled by small switch functions rather than polymorphism: a
reasoning switch emits `chat_template_kwargs.enable_thinking` for one NVIDIA model family
and `reasoning.enabled` for OpenRouter; an `X-Title` header is added only for OpenRouter.

**This is an endpoint table plus one hardcoded branch, not a provider-agnostic
abstraction.** Adding a sixth provider whose request shape is not OpenAI-compatible means
editing `_request_completion` itself. Stating this matters because "supports four
providers" invites the assumption that a fifth is configuration.

Model identifiers are a **hardcoded display table** (`MODEL_CHOICES`), not fetched from any
provider. The list can therefore go stale relative to what a provider offers.

## A sixth path that is not an API client

`grok_bot.py` drives a desktop application rather than an endpoint. It launches the app by
URI or shortcut, pipes the prompt to it over standard input through a PowerShell helper,
and scrapes the reply from the window with a second helper. Request and response are
correlated by a random marker, and the reply is handed back base64-encoded on one line of
standard output.

It is serialised by a turn lock (one paste at a time) and rate-claims the app's opening.
The reply budget is ten minutes.

This is **user-interface automation, not an integration**. It depends on a window being
present and on the application's layout, and it is the right thing to call out because a
reader scanning a provider list would otherwise count it as a fifth or sixth API.

## Credentials

Two files in a configuration directory outside any repository, overridable by an
environment variable:

- `api_keys.json` — a versioned envelope holding one Fernet token. Decrypted, it is a
  per-provider map of slots, each carrying a secret and a chosen model.
- `master.key` — 32 random bytes, generated on first write.

Both are written mode `0600` into a `0700` directory, by atomic temporary file and rename.
Those modes are POSIX-only: on Windows `os.chmod` sets just the read-only attribute, and the
files inherit the parent directory's ACL.
`refuse_repository_key_store()` runs before **every** write, so the store can never land
inside a checkout.

Chosen model identifiers are stored in plaintext in a third file beside the ciphertext.
That is deliberate: a model identifier is not a secret.

Provider slots: `grok`, `claude`, `openai`, `nvidia`, `openrouter`. There is **no slot for
a code-hosting or model-hub token**; those come from the environment if they are needed at
all.

Loading precedence is saved key first, then the provider's environment variable.

### The encryption, and one honest limitation

The Fernet key is derived as `HMAC-SHA256(machine_tag, master.key)`, where `machine_tag` is
read from the system machine identifier, falling back to the node name, falling back to a
fixed constant.

On Linux the identifier is present, so the ciphertext is bound to the machine as well as to
`master.key`. **On Windows neither the identifier file nor the node-name call exists, so the
tag resolves to the fixed constant and the derivation reduces to `master.key` alone.** The
binding is therefore platform-dependent, and this is a Windows-primary project.

This is not a flaw in the atomic writes, and it is the ordinary threat model for a local
credential store: the protection that matters is the store's location outside the
repository — enforced under POSIX by the `0600`/`0700` modes, and under Windows by the
user-profile ACL rather than by those mode bits, since `os.chmod` does not apply them there.
It is recorded here because "encrypted and machine-bound" would be an overstatement on the
platform the program mostly runs on, and so would leaning on those mode bits there.

### Keeping keys out of records

Three independent mechanisms, because one is not enough:

1. Named secret fields are redacted when a record is written.
2. HTTP error bodies are passed through a redactor before being surfaced, since a provider
   can echo a key back in an error message.
3. The MCP server rejects a secret-shaped field at the schema boundary, so a key cannot be
   introduced through a tool call.

## What this layer does not do

- It does not accept API keys through the MCP server.
- It does not treat a model reply as evidence. A model's rating of a target is combined
  with record-derived evidence under a fixed weight; it never becomes a measurement.
- It does not fetch model lists, so the offered identifiers are only as current as the
  table.
