"""The wire layer carries its own assumption and refuses to guess a schema.

Read-only: no network, no server, no sockets. This module is two functions and
a named constant precisely so it can be tested before the format it assumes has
been confirmed.
"""

from __future__ import annotations

import json

from rfproteina.disease import ddd_benchmark as ddd
from rfproteina.disease import ddd_endpoint as wire


def _rank(item, inputs):
    return ["A1", "A2"]


def test_the_assumption_is_named_and_dated():
    """An assumption with no date reads as a fact."""
    assert "unconfirmed" in wire.WIRE_ASSUMPTION
    assert "2026_11_18" in wire.WIRE_ASSUMPTION


def test_a_response_names_the_assumption_it_was_encoded_under():
    """So a format mismatch shows up in the first reply, not as a mis-scored column."""
    body = wire.encode_response(ddd.answer_all(rank_fn=_rank))
    assert body["wire_assumption"] == wire.WIRE_ASSUMPTION
    assert body["items"] == 20
    assert body["answered"] == 10
    assert body["valid_fraction"] == 0.5


def test_a_named_benchmark_is_decoded_under_any_of_the_accepted_keys():
    for key in ("benchmark", "item", "task", "name"):
        decoded = wire.decode_request({"items": [{key: "Mutations"}]})
        assert [item.benchmark for item, _ in decoded] == ["Mutations"]


def test_a_bare_list_and_a_bare_object_both_decode():
    assert len(wire.decode_request([{"benchmark": "Mutations"}, {"benchmark": "SMAC"}])) == 2
    assert len(wire.decode_request({"benchmark": "Mutations"})) == 1


def test_inputs_travel_with_the_item():
    decoded = wire.decode_request(
        {"items": [{"benchmark": "Mutations", "structure": "design.pdb", "disease": "cystic fibrosis"}]}
    )
    _, inputs = decoded[0]
    assert inputs["structure"] == "design.pdb"
    assert inputs["disease"] == "cystic fibrosis"


def test_an_unrecognised_payload_decodes_to_nothing_rather_than_a_guess():
    for payload in (None, "a string", 7, {}, {"messages": [{"role": "user", "content": "hi"}]}):
        assert wire.decode_request(payload) == []


def test_an_unknown_benchmark_name_is_skipped_not_invented():
    decoded = wire.decode_request({"items": [{"benchmark": "Retrosynthesis Planning"}]})
    assert decoded == []


def test_an_unrecognised_request_declines_everything_and_says_why():
    body = wire.handle({"messages": [{"role": "user", "content": "hi"}]})
    assert body["valid_fraction"] == 0.0
    assert body["items"] == 20
    assert {record["blocker"] for record in body["answers"]} == {"request_shape_unconfirmed"}
    assert all(record["value"] is None for record in body["answers"])
    assert wire.WIRE_ASSUMPTION in body["answers"][0]["reason"]


def test_a_recognised_request_is_answered_through_the_science_layer():
    body = wire.handle({"items": [{"benchmark": "Mutations"}, {"benchmark": "SMAC"}]}, rank_fn=_rank)
    assert body["items"] == 2
    assert body["answered"] == 1
    assert body["valid_fraction"] == 0.5
    answered = [record for record in body["answers"] if record["answered"]]
    assert answered[0]["item"] == "Mutations"


def test_every_declined_answer_still_carries_its_corpus_declaration():
    body = wire.handle({"messages": []})
    assert all(record["corpus"]["caveat"] for record in body["answers"])


def test_a_whole_response_survives_a_json_round_trip():
    body = wire.handle({"items": [{"benchmark": "Mutations"}]}, rank_fn=_rank)
    assert json.loads(json.dumps(body, allow_nan=False)) == body
