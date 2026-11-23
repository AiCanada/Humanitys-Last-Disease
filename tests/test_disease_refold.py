"""The pass criteria this repository declares, read off the metrics table.

`configs/metrics/pass_criteria/all_pass_criteria.yaml` declares eleven named
criteria and `rfd4-evaluate` writes a `<name>.pass` column for each into
`per_example_metrics.csv`. The disease screen read two of them -- `ppi.pass`
and `pre_oracle_pass` -- and the other nine were evaluated, written into the
file the screen opens, and discarded.

A bare `ppi_pass: false` also said nothing about *which* of its three bounds was
missed, and `ptm_coi`, `min_ipae_coi` and an aligned binder RMSD ask for very
different work.

The criteria and their cuts are read from that config, never copied into the
code, so these tests assert against the repository's own bar.
"""

from pathlib import Path

import pytest

from rfproteina.disease.refold import (
    CRITERIA_FILE,
    applicable_tags,
    read_criteria,
    refold_report,
)
from rfproteina.disease.screen import screen_generation


@pytest.fixture(scope="module")
def declared() -> dict:
    found = read_criteria()
    assert found is not None, "the repository's pass-criteria config should be readable from the checkout"
    return found


class TestTheConfigIsTheSourceOfTruth:
    def test_the_criteria_are_the_ones_the_repository_declares(self, declared: dict) -> None:
        assert {"ppi", "developability", "unconditional", "motif_scaffold"} <= set(declared)

    def test_ppi_carries_the_three_bounds_from_the_config(self, declared: dict) -> None:
        assert sorted(declared["ppi"]) == [
            "oracle_ppi.min_ipae_coi",
            "oracle_ppi.ptm_coi",
            "refolded_target_backbone_aligned_binder_rmsd.rmsd",
        ]

    def test_a_same_file_interpolation_is_followed_once(self, declared: dict) -> None:
        """`internal_coords` borrows `motif_scaffold`'s thresholds by reference."""
        assert declared["internal_coords"] == declared["motif_scaffold"]

    def test_a_checkout_without_the_config_answers_nothing(self, tmp_path: Path) -> None:
        assert read_criteria(tmp_path / "nowhere") is None

    def test_the_report_names_the_file_it_read(self) -> None:
        report = refold_report([{"example_id": "binder_0"}])
        assert report["criteria_file"] == CRITERIA_FILE.as_posix()


class TestNotEvaluated:
    def test_a_run_with_no_oracle_gives_one_reason_not_eleven(self) -> None:
        report = refold_report([{"example_id": "binder_0", "chain_of_interest": "A"}])
        assert report["evaluated"] == 0
        assert report["refold_blocker"] == "not_evaluated"
        assert all(item["state"] == "not_evaluated" for item in report["criteria"])

    def test_an_unevaluated_criterion_still_carries_its_bounds(self) -> None:
        """So a reader can see what would have to hold, before paying for an oracle."""
        report = refold_report([{"example_id": "binder_0"}])
        ppi = next(item for item in report["criteria"] if item["name"] == "ppi")
        bounds = {item["metric"]: item for item in ppi["thresholds"]}
        assert bounds["oracle_ppi.ptm_coi"]["cut"] == 80.0
        assert bounds["oracle_ppi.ptm_coi"]["bound"] == "min"
        assert bounds["oracle_ppi.ptm_coi"]["measured"] is None
        assert bounds["oracle_ppi.ptm_coi"]["met"] is None

    def test_a_missing_config_is_a_different_silence(self, tmp_path: Path) -> None:
        assert refold_report([], root=tmp_path / "nowhere")["refold_blocker"] == "criteria_not_found"


class TestWhichBoundWasMissed:
    def _rows(self, ipae: str) -> list[dict[str, str]]:
        return [
            {
                "example_id": "binder_0",
                "ppi.pass": "0" if float(ipae) > 1.5 else "1",
                "oracle_ppi.ptm_coi": "86.2",
                "oracle_ppi.min_ipae_coi": ipae,
                "refolded_target_backbone_aligned_binder_rmsd.rmsd": "1.1",
                "metrics_tags": "['ppi_refolding']",
            }
        ]

    def test_a_failing_criterion_names_the_bound(self) -> None:
        ppi = next(item for item in refold_report(self._rows("2.9"))["criteria"] if item["name"] == "ppi")
        assert ppi["state"] == "fail"
        assert ppi["failed_thresholds"] == ["oracle_ppi.min_ipae_coi"]

    def test_the_bounds_that_held_are_marked_as_held(self) -> None:
        ppi = next(item for item in refold_report(self._rows("2.9"))["criteria"] if item["name"] == "ppi")
        bounds = {item["metric"]: item for item in ppi["thresholds"]}
        assert bounds["oracle_ppi.ptm_coi"]["met"] is True
        assert bounds["oracle_ppi.min_ipae_coi"]["met"] is False
        assert bounds["oracle_ppi.min_ipae_coi"]["measured"] == 2.9

    def test_a_passing_criterion_names_no_bound(self) -> None:
        ppi = next(item for item in refold_report(self._rows("1.1"))["criteria"] if item["name"] == "ppi")
        assert ppi["state"] == "pass"
        assert "failed_thresholds" not in ppi

    def test_the_run_level_count_follows(self) -> None:
        report = refold_report(self._rows("2.9"))
        assert report["evaluated"] == 1
        assert "refold_blocker" not in report


class TestTheWorstDesignDecides:
    def test_a_ceiling_reports_the_highest_value_any_design_reached(self) -> None:
        rows = [
            {"ppi.pass": "1", "oracle_ppi.min_ipae_coi": "0.9"},
            {"ppi.pass": "1", "oracle_ppi.min_ipae_coi": "1.4"},
        ]
        ppi = next(item for item in refold_report(rows)["criteria"] if item["name"] == "ppi")
        bounds = {item["metric"]: item for item in ppi["thresholds"]}
        assert bounds["oracle_ppi.min_ipae_coi"]["measured"] == 1.4
        assert bounds["oracle_ppi.min_ipae_coi"]["met"] is True

    def test_a_floor_reports_the_lowest(self) -> None:
        rows = [{"ppi.pass": "1", "oracle_ppi.ptm_coi": "91.0"}, {"ppi.pass": "1", "oracle_ppi.ptm_coi": "72.0"}]
        ppi = next(item for item in refold_report(rows)["criteria"] if item["name"] == "ppi")
        bounds = {item["metric"]: item for item in ppi["thresholds"]}
        assert bounds["oracle_ppi.ptm_coi"]["measured"] == 72.0
        assert bounds["oracle_ppi.ptm_coi"]["met"] is False

    def test_one_failing_design_fails_the_run(self) -> None:
        """A run is not a passer because one of its designs was."""
        rows = [{"ppi.pass": "1"}, {"ppi.pass": "0"}]
        ppi = next(item for item in refold_report(rows)["criteria"] if item["name"] == "ppi")
        assert ppi["state"] == "fail"


class TestDevelopability:
    """The criterion the screen was recomputing half of while discarding whole."""

    _ROWS = [
        {
            "developability.pass": "0",
            "rosetta_developability.sap_score": "61.0",
            "rosetta_developability.hydrophobic_exposure_sasa": "900.0",
            "sequence_developability.gravy": "-0.2",
            "sequence_developability.instability_index": "33.0",
        }
    ]

    def test_it_bounds_four_metrics_not_the_two_the_screen_recomputes(self) -> None:
        dev = next(item for item in refold_report(self._ROWS)["criteria"] if item["name"] == "developability")
        assert len(dev["thresholds"]) == 4
        assert {item["metric"] for item in dev["thresholds"]} >= {
            "rosetta_developability.sap_score",
            "rosetta_developability.hydrophobic_exposure_sasa",
        }

    def test_the_rosetta_half_can_fail_on_its_own(self) -> None:
        dev = next(item for item in refold_report(self._ROWS)["criteria"] if item["name"] == "developability")
        assert dev["state"] == "fail"
        assert dev["failed_thresholds"] == ["rosetta_developability.sap_score"]


class TestOnTheScreen:
    def _run(self, tmp_path: Path, metrics: str) -> dict:
        output = tmp_path / "run"
        folder = output / "inference_samples" / "binder_0"
        folder.mkdir(parents=True)
        (folder / "binder_0_generated.cif").write_text("data_b\n#\n", encoding="utf-8")
        table = output / "inference_metrics" / "per_example_metrics.csv"
        table.parent.mkdir(parents=True)
        table.write_text(metrics, encoding="utf-8")
        return screen_generation(output, binder_kind="protein", target_id="P05113")

    def test_the_screen_carries_the_report(self, tmp_path: Path) -> None:
        record = self._run(tmp_path, "example_id,chain_of_interest\nbinder_0,A\n")
        assert record["refold"]["declared"] == 11

    def test_the_missed_bound_reaches_the_record(self, tmp_path: Path) -> None:
        record = self._run(
            tmp_path,
            "example_id,chain_of_interest,ppi.pass,oracle_ppi.ptm_coi,oracle_ppi.min_ipae_coi\n"
            "binder_0,A,0,86.2,2.9\n",
        )
        ppi = next(item for item in record["refold"]["criteria"] if item["name"] == "ppi")
        assert ppi["failed_thresholds"] == ["oracle_ppi.min_ipae_coi"]

    def test_none_of_it_moves_the_call(self, tmp_path: Path) -> None:
        """Like ppi_pass: the evaluation's verdict, not this screen's.

        A screen that silently adopted nine new ways to fail would change the
        meaning of every stored call that had none of them.
        """
        record = self._run(
            tmp_path,
            "example_id,chain_of_interest,ppi.pass,developability.pass\nbinder_0,A,0,0\n",
        )
        assert record["call"] == "untested"
        assert "refold" not in record["states"]


def test_the_metrics_tags_are_readable() -> None:
    """They are what decides which criteria applied, when a run records them."""
    assert applicable_tags([{"metrics_tags": "['ppi_refolding', 'developability']"}]) == [
        "ppi_refolding",
        "developability",
    ]
    assert applicable_tags([{"metrics_tags": ""}]) == []


@pytest.mark.parametrize("raw", ["2", "-1", "0.5", "NaN", "inf", "failed"])
def test_nonbinary_refold_flags_never_pass(raw: str) -> None:
    item = next(c for c in refold_report([{"ppi.pass": raw}])["criteria"] if c["name"] == "ppi")
    assert item["state"] not in {"pass", "fail"}


def test_incomplete_refold_is_distinct_from_a_measured_failure(tmp_path: Path) -> None:
    metrics = tmp_path / "inference_metrics"
    metrics.mkdir()
    (metrics / "per_example_metrics.csv").write_text(
        "example_id,ppi.pass,ppi.all_metrics_present\none,1,1\ntwo,0,0\n",
        encoding="utf-8",
    )
    screen = screen_generation(tmp_path, binder_kind="other", target_id="")
    ppi = next(c for c in screen["refold"]["criteria"] if c["name"] == "ppi")
    assert ppi["state"] == "incomplete"
    assert ppi["rows_measured"] == 1
    assert ppi["rows_expected"] == 2
    assert screen["ppi_pass"] == "absent"
    assert screen["ppi_blocker"] == "metrics_incomplete"


def test_a_shared_oracle_column_does_not_score_an_unrelated_criterion() -> None:
    rows = [
        {"metrics_tags": "ppi_refolding", "ppi.pass": "1", "oracle_ppi.min_ipae_coi": "1.3"},
        {"metrics_tags": "antibody", "antibody.pass": "0", "oracle_ppi.min_ipae_coi": "3.5"},
    ]
    report = refold_report(rows)
    ppi = next(c for c in report["criteria"] if c["name"] == "ppi")
    assert ppi["state"] == "pass"
    assert "failed_thresholds" not in ppi
    bound = next(b for b in ppi["thresholds"] if b["metric"] == "oracle_ppi.min_ipae_coi")
    assert bound["measured"] == 1.3


def test_refold_partial_coverage_cannot_confirm_a_bound() -> None:
    rows = [{"ppi.pass": "1", "oracle_ppi.min_ipae_coi": "0.9"}, {"ppi.pass": "", "metrics_tags": "ppi_refolding"}]
    ppi = next(c for c in refold_report(rows)["criteria"] if c["name"] == "ppi")
    assert ppi["state"] == "incomplete"
    bound = next(b for b in ppi["thresholds"] if b["metric"] == "oracle_ppi.min_ipae_coi")
    assert bound["measured"] == 0.9
    assert bound["met"] is None


def test_reductions_and_two_sided_bounds_use_each_designs_own_inputs(tmp_path: Path) -> None:
    config = tmp_path / CRITERIA_FILE
    config.parent.mkdir(parents=True)
    config.write_text(
        "sample:\n  metric_thresholds:\n    pair: {mean_of: [a, b], min: 2, max_exclusive: 5}\n"
        "    malformed: {min: 1, typo: 2}\n",
        encoding="utf-8",
    )
    rows = [
        {"sample.pass": "1", "a": "2", "b": "4", "pair": "99"},
        {"sample.pass": "0", "a": "4", "b": "6"},
        {"sample.pass": "", "a": "4", "b": "inf"},
    ]
    sample = refold_report(rows, root=tmp_path)["criteria"][0]
    floor, ceiling, invalid = sample["thresholds"]
    assert (floor["measured"], floor["met"]) == (3, None)
    assert (ceiling["measured"], ceiling["met"]) == (5, False)
    assert ceiling["input_metrics"] == ["a", "b"]
    assert ceiling["reduction"] == "mean_of"
    assert sample["failed_thresholds"] == ["pair"]
    assert invalid == {"metric": "malformed", "bound": "unsupported"}
