"""The wire layer for a DDD Benchmark evaluation, with the unknown part named.

The benchmark evaluates a system by querying an endpoint the operator provides:
"Your model stays in your environment. We query only the endpoint you provide."
**The request and response format for that path is not published.** The public
leaderboard scores chat-completions models; the specialist path is arranged by
email, and specialist systems are ranked alongside generalists.

So this module does the one thing that is safe to do before the format is
confirmed: it keeps the guess in a single place, names it, and makes a wrong
guess announce itself.

* :data:`WIRE_ASSUMPTION` rides in every response. A mismatch then shows up in
  the first reply, rather than as a column of quietly mis-scored answers.
* :func:`decode_request` declines everything on a payload it does not
  recognise, with ``request_shape_unconfirmed``. It does not raise and it does
  not infer a schema -- the same discipline ``developability.read_oracle_table``
  uses when a CSV is not the oracle output it claims to be.

There is no server here, on purpose. ``scripts/humanitys_last_disease_mcp.py``
is already this program's remote surface; if the harness can speak MCP, nothing
further is needed. If it needs HTTPS, that shim is small and belongs outside the
science layer, written once the format is known.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rfproteina.disease import ddd_benchmark
from rfproteina.disease.ddd_benchmark import BenchmarkItem

#: The format this module assumes, until Insilico confirms the specialist one.
#: Dated, because an assumption with no date reads as a fact.
WIRE_ASSUMPTION = "openai_chat_completions_unconfirmed_2026_11_18"

#: What a payload must carry to be recognised: a benchmark name this program
#: knows, under one of these keys. Anything else is refused rather than guessed.
_ITEM_KEYS = ("benchmark", "item", "task", "name")

#: This module's one blocker, with its sentence. Kept here rather than in
#: ``ddd_benchmark`` because it describes the wire, not the science: nothing was
#: wrong with the item, the request could not be read.
_BLOCKER_TEXT: dict[str, str] = {
    "request_shape_unconfirmed": (
        "This endpoint did not recognise the shape of the request, so it answered nothing "
        "rather than guess a schema. The format it assumes is named in the response."
    ),
}


def _named_item(entry: Mapping[str, Any]) -> BenchmarkItem | None:
    for key in _ITEM_KEYS:
        name = entry.get(key)
        if isinstance(name, str) and name in ddd_benchmark.ITEMS_BY_NAME:
            return ddd_benchmark.ITEMS_BY_NAME[name]
    return None


def _entries(payload: Any) -> list[Mapping[str, Any]]:
    """The per-item dicts in a payload, under any of the shapes seen in the wild."""
    if isinstance(payload, Mapping):
        for key in ("items", "tasks", "benchmarks", "requests"):
            found = payload.get(key)
            if isinstance(found, Sequence) and not isinstance(found, str | bytes):
                return [entry for entry in found if isinstance(entry, Mapping)]
        return [payload]
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        return [entry for entry in payload if isinstance(entry, Mapping)]
    return []


def decode_request(payload: Any) -> list[tuple[BenchmarkItem, dict[str, Any]]]:
    """The items a request asks for, with their inputs. Empty when unrecognised.

    An empty list is the signal to decline everything: see
    :func:`unrecognised_response`. Returning nothing is deliberate, because the
    alternative -- inventing a mapping from an unknown schema to this program's
    items -- would produce answers to questions nobody asked.
    """
    decoded: list[tuple[BenchmarkItem, dict[str, Any]]] = []
    for entry in _entries(payload):
        item = _named_item(entry)
        if item is None:
            continue
        decoded.append(
            (
                item,
                {
                    "structure": entry.get("structure"),
                    "disease": entry.get("disease"),
                    "sequence": entry.get("sequence"),
                },
            )
        )
    return decoded


def encode_response(answers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The reply body, carrying the assumption it was built under.

    ``valid_fraction`` is computed here as well as scored there, so the two can
    be compared: if the harness reports a different number, the disagreement is
    about what counts as an answer, which is worth finding out early.
    """
    return {
        "wire_assumption": WIRE_ASSUMPTION,
        "answers": list(answers),
        "valid_fraction": ddd_benchmark.valid_fraction(answers),
        "answered": sum(1 for record in answers if record.get("answered")),
        "items": len(answers),
    }


def unrecognised_response() -> dict[str, Any]:
    """Every item declined, for a request this module could not read.

    Every item rather than none, so the reply still says what this program
    covers and why each item did or did not produce a number. A bare error
    would tell the reader only that something went wrong here.
    """
    answers = [
        {
            **ddd_benchmark.answer(item),
            "blocker": "request_shape_unconfirmed",
            "answered": False,
            "value": None,
            "reason": f"{_BLOCKER_TEXT['request_shape_unconfirmed']} It assumes {WIRE_ASSUMPTION}.",
        }
        for item in ddd_benchmark.BENCHMARK_ITEMS
    ]
    return encode_response(answers)


def handle(payload: Any, **kwargs: Any) -> dict[str, Any]:
    """A whole request, decoded, answered and encoded. The only entry point a shim needs."""
    decoded = decode_request(payload)
    if not decoded:
        return unrecognised_response()
    answers = [
        ddd_benchmark.answer(
            item,
            structure=inputs.get("structure"),
            disease=inputs.get("disease"),
            rank_fn=kwargs.get("rank_fn"),
        )
        for item, inputs in decoded
    ]
    return encode_response(answers)
