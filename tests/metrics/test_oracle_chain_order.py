"""Which chain order the ``chain_*`` arrays and ``chain_pair_*`` matrices use.

``extract_confidence_metrics`` is given a chain order and indexes every per-chain
and interface value by it. The order used to be read straight out of the
summary's ``chain_ids``, defaulting to ``[]``.

``chain_ids`` is not always written. It is a local-only key of AlphaFold 3's
summary: the hosted AlphaFold Server's summary has exactly ten keys and that is
not one of them, and an older local AlphaFold 3 wrote a nine-key summary without
it either. With the order empty, ``extract_confidence_metrics`` iterates
``combinations(range(0))`` and emits no chain-pair metric, then returns at
``chain_of_interest not in []`` before every per-chain value. Nothing raises.
``overall_ptm`` and ``overall_iptm`` are still there, so a refold looks like it
worked while ``ptm_coi``, ``iptm_coi`` and ``min_ipae_coi`` -- the numbers a
binder refold is run for -- are simply gone.

These tests pin the derivation and, just as importantly, pin that the derived
order is the *file's* order rather than a sorted or invented one. A wrong order
is worse than no order: it attributes one chain's number to another.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
from rfproteina.metrics.oracles.oracle_metric import chain_order_from_confidences, extract_confidence_metrics


def summary(chain_ids: list[str] | None = None) -> dict:
    """A two-chain AF3-shaped summary. ``chain_ids`` only when asked for."""
    body: dict = {
        "ptm": 0.61,
        "iptm": 0.51,
        "chain_ptm": [0.68, 0.64],
        "chain_iptm": [0.51, 0.51],
        "chain_pair_iptm": [[0.68, 0.51], [0.51, 0.64]],
        "chain_pair_pae_min": [[0.76, 2.21], [2.01, 0.76]],
        "ranking_score": 0.66,
    }
    if chain_ids is not None:
        body["chain_ids"] = chain_ids
    return body


def full(tokens: list[str]) -> dict:
    return {"token_chain_ids": tokens, "atom_plddts": [76.63] * len(tokens)}


class TestDerivation:
    def test_a_declared_chain_ids_is_used_as_given(self) -> None:
        assert chain_order_from_confidences(summary(["A", "B"]), None) == ["A", "B"]

    def test_a_declared_chain_ids_wins_over_the_derivation(self) -> None:
        """The summary is the authority when it speaks. Nothing silently overrides it."""
        assert chain_order_from_confidences(summary(["X", "Y"]), full(["A", "A", "B"])) == ["X", "Y"]

    def test_the_order_is_derived_when_chain_ids_is_absent(self) -> None:
        """A hosted AlphaFold Server summary, which never carries chain_ids."""
        assert chain_order_from_confidences(summary(), full(["A", "A", "A", "B", "B"])) == ["A", "B"]

    def test_the_derived_order_is_first_appearance_not_sorted(self) -> None:
        """It indexes the matrices, so only the file's own order is right."""
        assert chain_order_from_confidences(summary(), full(["B", "B", "A"])) == ["B", "A"]

    def test_an_interleaved_token_list_still_yields_each_chain_once(self) -> None:
        assert chain_order_from_confidences(summary(), full(["B", "A", "B", "A", "C"])) == ["B", "A", "C"]

    @pytest.mark.parametrize(
        "declared",
        [None, [], "AB", 0],
        ids=["absent", "empty", "a string not a list", "a number"],
    )
    def test_an_unusable_chain_ids_falls_through_to_the_derivation(self, declared: object) -> None:
        """An empty list is as uninformative as an absent key, and read the same way.

        ``"AB"`` is the trap: a bare string is iterable, so taking it as a list
        of chains would silently produce ``["A", "B"]`` from a value that was
        never a chain list.
        """
        body = summary()
        if declared is not None:
            body["chain_ids"] = declared
        assert chain_order_from_confidences(body, full(["A", "B"])) == ["A", "B"]

    def test_neither_source_leaves_no_order_rather_than_a_guess(self) -> None:
        assert chain_order_from_confidences(summary(), None) == []
        assert chain_order_from_confidences(summary(), {}) == []

    def test_a_token_list_that_is_not_a_list_leaves_no_order(self) -> None:
        """A bare string is iterable, so it would otherwise read as its letters."""
        assert chain_order_from_confidences(summary(), {"token_chain_ids": "AB"}) == []
        assert chain_order_from_confidences(summary(), {"token_chain_ids": []}) == []
        assert chain_order_from_confidences(summary(), {"token_chain_ids": None}) == []

    def test_numeric_chain_labels_become_strings(self) -> None:
        """A label read as an int would never match a chain id spelled as text."""
        assert chain_order_from_confidences(summary(), full([1, 1, 2])) == ["1", "2"]


class TestWhatTheOrderUnlocks:
    def test_the_derived_order_restores_every_per_chain_and_interface_metric(self) -> None:
        """One call with the order, one without, and the difference named."""
        order = chain_order_from_confidences(summary(), full(["A", "B"]))
        with_order = extract_confidence_metrics(summary(), "A", order)
        without = extract_confidence_metrics(summary(), "A", chain_order_from_confidences(summary(), None))
        restored = sorted(set(with_order) - set(without))
        assert restored == ["iptm_A_B", "iptm_coi", "min_ipae_A_B", "min_ipae_coi", "ptm_coi"]
        # And the globals were never the problem, which is why this was invisible.
        assert {"overall_ptm", "overall_iptm", "ranking_score"} <= set(without)

    def test_the_restored_values_are_the_chains_own_numbers(self) -> None:
        """Not merely present: indexed at the right chain.

        ``chain_ptm`` is [0.68, 0.64] and the chain of interest is the second
        one, so a per-chain value of 68 would mean the order was wrong.
        """
        order = chain_order_from_confidences(summary(), full(["A", "B"]))
        metrics = extract_confidence_metrics(summary(), "B", order)
        assert metrics["ptm_coi"] == pytest.approx(64.0)
        # chain_pair_pae_min is not symmetric: [B][A] is 2.01, [A][B] is 2.21,
        # and the COI reduction averages the ordered pair.
        assert metrics["min_ipae_coi"] == pytest.approx(2.11)


class TestTheFullConfidencesFile:
    """Where the local oracle gets the order from, and what it does with a bad file.

    The derivation above needs ``token_chain_ids``, which lives in AF3's *full*
    confidences file rather than the summary beside it.
    :py:meth:`AF3Oracle._parse_af3_full_confidences` finds and reads that file.

    Its contract is that an unusable file costs the chain order and nothing
    else: a refold that cannot derive its order still produces the global
    metrics, exactly as it did before the derivation existed. A read that raised
    instead would turn a missing number into a failed refold -- worse than the
    defect being fixed, because that one at least produced structures.
    """

    def _written(self, tmp_path: Path, name: str, payload: bytes) -> AF3Oracle:
        (tmp_path / name).write_bytes(payload)
        return AF3Oracle(oracle_name="af3")

    def test_the_file_is_read_and_its_order_derived(self, tmp_path: Path) -> None:
        """The whole point: a summary with no chain_ids still gets its order."""
        oracle = self._written(tmp_path, "job_confidences.json", json.dumps(full(["A", "A", "B"])).encode("utf-8"))
        loaded = oracle._parse_af3_full_confidences(tmp_path, "job")
        assert loaded is not None
        assert chain_order_from_confidences(summary(), loaded) == ["A", "B"]

    def test_a_compressed_file_is_read_too(self, tmp_path: Path) -> None:
        """``--compress_large_output_files`` zstd-compresses this file, not the summary."""
        zstandard = pytest.importorskip("zstandard")
        payload = zstandard.ZstdCompressor().compress(json.dumps(full(["B", "A"])).encode("utf-8"))
        oracle = self._written(tmp_path, "job_confidences.json.zst", payload)
        loaded = oracle._parse_af3_full_confidences(tmp_path, "job")
        assert loaded is not None
        assert chain_order_from_confidences(summary(), loaded) == ["B", "A"]

    def test_an_absent_file_is_not_an_error(self, tmp_path: Path) -> None:
        """An older local AlphaFold 3 wrote no such file. That is a None, not a raise."""
        assert AF3Oracle(oracle_name="af3")._parse_af3_full_confidences(tmp_path, "job") is None

    def test_the_plain_spelling_is_preferred_over_the_compressed_one(self, tmp_path: Path) -> None:
        """Both can sit side by side after a decompress in place; reading costs less."""
        zstandard = pytest.importorskip("zstandard")
        (tmp_path / "job_confidences.json").write_text(json.dumps(full(["A", "B"])), encoding="utf-8")
        (tmp_path / "job_confidences.json.zst").write_bytes(
            zstandard.ZstdCompressor().compress(json.dumps(full(["Z", "Y"])).encode("utf-8"))
        )
        loaded = AF3Oracle(oracle_name="af3")._parse_af3_full_confidences(tmp_path, "job")
        assert chain_order_from_confidences(summary(), loaded) == ["A", "B"]

    def test_a_file_that_is_not_json_costs_the_order_and_nothing_more(self, tmp_path: Path) -> None:
        oracle = self._written(tmp_path, "job_confidences.json", b"{not json")
        assert oracle._parse_af3_full_confidences(tmp_path, "job") is None

    def test_a_json_array_is_refused_rather_than_indexed(self, tmp_path: Path) -> None:
        """A list has no ``token_chain_ids``, and reading it as a mapping would raise later."""
        oracle = self._written(tmp_path, "job_confidences.json", b"[1, 2, 3]")
        assert oracle._parse_af3_full_confidences(tmp_path, "job") is None

    def test_a_mislabelled_compressed_file_does_not_take_the_refold_down(self, tmp_path: Path) -> None:
        """A plain JSON renamed ``.zst``, which is what a careless copy produces.

        ``zstandard.ZstdError`` subclasses ``Exception`` and not ``ValueError``,
        so the usual read errors did not catch it and it escaped the read. The
        bound on decompression raises the same error, so the guard that exists
        to contain a huge file would have crashed on one too.
        """
        pytest.importorskip("zstandard")
        oracle = self._written(tmp_path, "job_confidences.json.zst", json.dumps(full(["A", "B"])).encode("utf-8"))
        assert oracle._parse_af3_full_confidences(tmp_path, "job") is None

    def test_a_truncated_compressed_file_is_refused_the_same_way(self, tmp_path: Path) -> None:
        zstandard = pytest.importorskip("zstandard")
        whole = zstandard.ZstdCompressor().compress(json.dumps(full(["A", "B"])).encode("utf-8"))
        oracle = self._written(tmp_path, "job_confidences.json.zst", whole[: len(whole) // 2])
        assert oracle._parse_af3_full_confidences(tmp_path, "job") is None

    def test_the_bound_is_large_enough_for_a_real_file(self) -> None:
        """Lowered bounds in a test must not be mistaken for the shipped one."""
        from rfproteina.metrics.oracles.af3_oracle import _FULL_CONFIDENCE_LIMIT

        assert _FULL_CONFIDENCE_LIMIT >= 1024**3, "a 5,000-token PAE matrix is tens of millions of numbers"
