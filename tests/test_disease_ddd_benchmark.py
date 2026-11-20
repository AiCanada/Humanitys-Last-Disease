"""The DDD benchmark adapter answers what it can and declines the rest by name.

Read-only: no network, no weights, no subprocess. The ranking producers are
injected, which is how ``stability.run_site_saturation`` and
``screening_tools.run_tool`` are already testable.
"""

from __future__ import annotations

import json

import pytest

from rfproteina.disease import corpus_provenance as cp
from rfproteina.disease import ddd_benchmark as ddd

#: The twenty benchmark names, read off the benchmark's own Explorer on
#: 2026-11-18. Pinned so the scope cannot quietly grow a twenty-first.
_SCOPE = {
    "HAC Retention Time",
    "HIC Retention Time",
    "CHO Polyreactivity",
    "Ovalbumin Polyreactivity",
    "SMAC",
    "Tm",
    "Evolutionary pressure",
    "Mutations",
    "Target Prediction for Cancer Diseases",
    "Target Prediction for Cardiovascular Diseases",
    "Target Prediction for Fibrotic Diseases",
    "Target Prediction for Inflammatory & Immunological Diseases",
    "Target Prediction for Mental & Behavioral Disorders",
    "Target Prediction for Metabolic & Endocrine Diseases",
    "Target Prediction for Neurological Diseases",
    "Target Prediction for Ophthalmic Diseases",
    "Target Prediction for Reproductive Health Conditions",
    "iPSC, genome-wide (Pan-Genome)",
    "iPSC, strong perturbations",
    "K562, genome-scale (Replogle)",
}


def _rank(item, inputs):
    return ["A1", "A2", "A3"]


def test_the_registry_is_the_twenty_benchmarks_in_scope():
    assert {item.benchmark for item in ddd.BENCHMARK_ITEMS} == _SCOPE
    assert len(ddd.BENCHMARK_ITEMS) == 20


def test_ten_of_the_twenty_are_answerable_today():
    """The measured number, recorded so it moves only on purpose.

    Answerable: ResidueBench Mutations, and the nine target-prediction groups.
    Declined: the six developability assays (no installed oracle), evolutionary
    pressure (no conservation model), and the three perturbation sets.
    """
    assert len(ddd.answerable_items()) == 10
    assert ddd.coverage() == {
        "items": 20,
        "answerable": 10,
        "declined": 10,
        "domains": ["Structural Biology", "Target Biology", "Therapeutic Biomolecules"],
        "blockers": sorted({item.blocker for item in ddd.BENCHMARK_ITEMS if item.blocker}),
    }


def test_mutations_is_the_only_answerable_structure_item():
    structural = [i for i in ddd.BENCHMARK_ITEMS if i.domain == ddd.STRUCTURAL_BIOLOGY]
    assert {i.benchmark for i in structural if i.answerable} == {"Mutations"}
    mutations = ddd.ITEMS_BY_NAME["Mutations"]
    assert mutations.answers_from == "thermompnn_point_stability_v1"
    assert mutations.answer_kind == ddd.RANKING


def test_no_developability_assay_is_answerable_today():
    """CrossAbSense is read rather than run here, and its weights are not installed."""
    assays = [i for i in ddd.BENCHMARK_ITEMS if i.domain == ddd.THERAPEUTIC_BIOMOLECULES]
    assert len(assays) == 6
    assert [i for i in assays if i.answerable] == []


def test_the_two_developability_blockers_are_different_facts():
    """A size cap and a missing model have different fixes, so they get different names."""
    by_name = ddd.ITEMS_BY_NAME
    assert by_name["HIC Retention Time"].blocker == "oracle_weights_over_size_cap:HIC"
    assert by_name["CHO Polyreactivity"].blocker == "oracle_weights_not_installed:PR_CHO"
    assert by_name["SMAC"].blocker == "no_model_for_assay:SMAC"
    assert by_name["HAC Retention Time"].blocker == "no_model_for_assay:HAC"


def test_every_blocker_the_registry_names_has_a_sentence_in_this_module():
    """Set equality both directions: an unused sentence is as wrong as a missing one.

    Keyed by the whole name, suffix included, which is what the module emits and
    what the board-vocabulary scan looks up.
    """
    used = {item.blocker for item in ddd.BENCHMARK_ITEMS if item.blocker}
    # These two are emitted at call time rather than named by the registry: one
    # when nothing was wired to produce the answer, one when the producer threw.
    assert used | {"runner_not_supplied", "producer_failed"} == set(ddd._BLOCKER_TEXT)


@pytest.mark.parametrize("item", ddd.BENCHMARK_ITEMS, ids=lambda i: i.benchmark)
def test_an_item_is_either_answerable_or_names_a_blocker_never_both(item):
    assert item.answerable != bool(item.blocker)


@pytest.mark.parametrize("item", ddd.BENCHMARK_ITEMS, ids=lambda i: i.benchmark)
def test_every_answer_carries_a_corpus_block_in_both_states(item):
    """A declined item still declares its corpus, or the contaminated ones hide."""
    record = ddd.answer(item, rank_fn=_rank)
    assert "corpus" in record
    assert record["corpus"]["overlap"] in cp.OVERLAP_KINDS
    assert record["corpus"]["caveat"]


@pytest.mark.parametrize("item", ddd.BENCHMARK_ITEMS, ids=lambda i: i.benchmark)
def test_answered_is_explicit_and_agrees_with_the_value(item):
    """The scorer counts one field. ``value: null`` alone would be ambiguous."""
    record = ddd.answer(item, rank_fn=_rank)
    assert isinstance(record["answered"], bool)
    assert record["answered"] is (record["value"] is not None)


@pytest.mark.parametrize("item", [i for i in ddd.BENCHMARK_ITEMS if not i.answerable], ids=lambda i: i.benchmark)
def test_a_declined_answer_carries_no_value_and_a_readable_reason(item):
    record = ddd.answer(item, rank_fn=_rank)
    assert record["answered"] is False
    assert record["value"] is None
    assert record["blocker"] == item.blocker
    assert record["reason"].endswith(".")


def test_a_declined_item_stays_declined_even_when_inputs_are_supplied():
    """An input cannot make a missing model appear."""
    record = ddd.answer(ddd.ITEMS_BY_NAME["Evolutionary pressure"], structure="design.pdb", rank_fn=_rank)
    assert record["answered"] is False
    assert record["blocker"] == "no_conservation_model"


def test_an_answerable_item_with_no_producer_declines_rather_than_returning_zero():
    record = ddd.answer(ddd.ITEMS_BY_NAME["Mutations"])
    assert record["answered"] is False
    assert record["blocker"] == "runner_not_supplied"
    assert record["value"] is None


def test_the_crossabsense_declines_still_declare_the_gdpa1_overlap():
    """The point of the pairing: the one domain it could score is the contaminated one."""
    record = ddd.answer(ddd.ITEMS_BY_NAME["HIC Retention Time"])
    assert record["answered"] is False
    assert record["corpus"]["overlap"] == cp.BENCHMARK_LABEL_CORPUS
    assert record["corpus"]["overlap_against"] == "HIC"
    assert any(entry["name"] == "GDPa1" for entry in record["corpus"]["corpora"])


def test_the_target_items_declare_the_open_targets_lookup():
    record = ddd.answer(ddd.ITEMS_BY_NAME["Target Prediction for Cancer Diseases"], rank_fn=_rank)
    assert record["answered"] is True
    assert record["corpus"]["model"] == "opentargets_associations"
    assert record["corpus"]["overlap"] == cp.BENCHMARK_LABEL_CORPUS


def test_the_valid_fraction_of_a_whole_sweep_is_ten_of_twenty():
    answers = ddd.answer_all(rank_fn=_rank)
    assert len(answers) == 20
    assert ddd.valid_fraction(answers) == 0.5


def test_the_valid_fraction_with_nothing_wired_is_zero():
    """Measured: with no producer supplied, every item declines."""
    assert ddd.valid_fraction(ddd.answer_all()) == 0.0
    assert ddd.valid_fraction([]) == 0.0


def test_every_answer_survives_a_json_round_trip():
    """A frozen dataclass leaking into a record is the likeliest first bug here."""
    answers = ddd.answer_all(rank_fn=_rank)
    assert json.loads(json.dumps(answers, allow_nan=False)) == answers


def test_a_producer_failure_becomes_a_decline_carrying_the_reason():
    """A scorer needs "no answer" and "wrong answer" told apart, not kinds of exception."""

    def explode(item, inputs):
        raise RuntimeError("the oracle was unreachable")

    record = ddd.answer_or_decline(ddd.ITEMS_BY_NAME["Mutations"], rank_fn=explode)
    assert record["answered"] is False
    assert record["value"] is None
    assert record["blocker"] == "producer_failed"
    assert "the oracle was unreachable" in record["reason"]
    assert record["corpus"]["caveat"]


def test_the_mutation_ranking_puts_the_most_stabilising_position_first(tmp_path):
    """The scan's convention is that a more negative ddG is more stabilising, and
    Precision@K reads the head of the list, so the order is the answer."""
    structure = tmp_path / "design.pdb"
    structure.write_text("ATOM\n")

    def fake_scan(path, chain):
        return [
            {"position": 1, "wildtype": "A", "mutation": "V", "ddG_pred": -0.2},
            {"position": 2, "wildtype": "L", "mutation": "I", "ddG_pred": -2.5},
            {"position": 2, "wildtype": "L", "mutation": "P", "ddG_pred": 3.0},
            {"position": 3, "wildtype": "K", "mutation": "R", "ddG_pred": -1.1},
        ]

    ranked = ddd.rank_from_installed_tools(
        ddd.ITEMS_BY_NAME["Mutations"], {"structure": structure}, predict_fn=fake_scan
    )
    assert ranked == [2, 3, 1]


def test_the_mutation_producer_refuses_without_a_structure():
    with pytest.raises(ValueError, match="no structure"):
        ddd.rank_from_installed_tools(ddd.ITEMS_BY_NAME["Mutations"], {})


def test_the_target_producer_refuses_without_a_disease():
    item = ddd.ITEMS_BY_NAME["Target Prediction for Cancer Diseases"]
    with pytest.raises(ValueError, match="no disease"):
        ddd.rank_from_installed_tools(item, {})


def test_a_declined_item_has_no_producer_at_all():
    with pytest.raises(ValueError, match="no producer"):
        ddd.rank_from_installed_tools(ddd.ITEMS_BY_NAME["SMAC"], {})


def test_our_one_answerable_structure_item_is_itself_a_label_corpus():
    """Worth stating plainly: ThermoMPNN is fitted on Megascale, which measures ddG.

    The ResidueBench Mutations answer is the program's best coverage and is
    still drawn from a model trained on the quantity being scored. The
    declaration says so on every answer rather than leaving a reader to find it.
    """
    record = ddd.answer(ddd.ITEMS_BY_NAME["Mutations"], rank_fn=_rank)
    assert record["answered"] is True
    assert record["corpus"]["overlap"] == cp.BENCHMARK_LABEL_CORPUS
    assert record["corpus"]["overlap_against"] == "ddG"


def test_every_answer_records_the_code_that_produced_it():
    record = ddd.answer(ddd.ITEMS_BY_NAME["Mutations"], rank_fn=_rank)
    assert set(record["code"]) == {"commit", "tree_state"}
