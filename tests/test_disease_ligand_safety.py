"""Tests for rfproteina.disease.ligand_safety (CPU-only, fast, network mocked)."""

from __future__ import annotations

import json

import pytest

from rfproteina.disease import ligand_safety as ls

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class _Resp:
    def __init__(self, data):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._data


class _FakeOpener:
    """A stand-in for urlopen that answers from canned routes by URL substring."""

    def __init__(self, routes):
        self.routes = routes
        self.seen = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.seen.append(url)
        for substring, payload in self.routes:
            if substring in url:
                return _Resp(json.dumps(payload).encode("utf-8"))
        return _Resp(json.dumps({}).encode("utf-8"))


def _herg_activities():
    return {
        "activities": [
            {"standard_type": "IC50", "standard_relation": "=", "standard_value": "50", "standard_units": "nM", "activity_id": 1},
            {"standard_type": "Ki", "standard_relation": None, "standard_value": "150", "standard_units": "nM", "activity_id": 2},
            {"standard_type": "IC50", "standard_relation": ">", "standard_value": "10000", "standard_units": "nM", "activity_id": 3},
            {"standard_type": "IC50", "standard_relation": "=", "standard_value": "5", "standard_units": "uM", "activity_id": 4},
        ]
    }


def test_fetch_measured_nm_aggregates_median_in_nm():
    opener = _FakeOpener(
        [
            ("molecule?molecule_structures__standard_inchi_key", {"molecules": [{"molecule_chembl_id": "CHEMBL25"}]}),
            ("target_chembl_id=CHEMBL240", _herg_activities()),
        ]
    )
    result = ls.fetch_measured_nm(ASPIRIN, opener=opener)
    herg = result.axes["herg"]
    assert herg["value_nm"] == 100.0
    assert herg["n_values"] == 2
    assert herg["unit"] == "nM"
    assert "CHEMBL240" in herg["source"]
    assert "bounded" in herg["note"]
    assert result.molecule_chembl_id == "CHEMBL25"


def test_axes_without_activity_stay_untested():
    opener = _FakeOpener(
        [("molecule?molecule_structures__standard_inchi_key", {"molecules": [{"molecule_chembl_id": "CHEMBL25"}]})]
    )
    result = ls.fetch_measured_nm(ASPIRIN, opener=opener)
    assert result.axes["cyp3a4"]["value_nm"] is None
    assert result.axes["cyp3a4"]["n_values"] == 0


def test_molecule_unknown_to_chembl_leaves_all_axes_untested():
    opener = _FakeOpener([("molecule?molecule_structures__standard_inchi_key", {"molecules": []})])
    result = ls.fetch_measured_nm(ASPIRIN, opener=opener)
    assert result.molecule_chembl_id is None
    assert all(reading["value_nm"] is None for reading in result.axes.values())
    assert result.axes["herg"]["note"] == "molecule_not_in_chembl"


def test_to_toxicology_rows_only_emits_determinate_values():
    opener = _FakeOpener(
        [
            ("molecule?molecule_structures__standard_inchi_key", {"molecules": [{"molecule_chembl_id": "CHEMBL25"}]}),
            ("target_chembl_id=CHEMBL240", _herg_activities()),
        ]
    )
    rows = ls.to_toxicology_rows(ls.fetch_measured_nm(ASPIRIN, opener=opener))
    assert len(rows) == 1
    assert rows[0]["axis"] == "herg"
    assert rows[0]["unit"] == "nM"
    assert rows[0]["nm"] == "100.0"


def test_measured_record_never_trips_the_probability_guard():
    opener = _FakeOpener(
        [
            ("molecule?molecule_structures__standard_inchi_key", {"molecules": [{"molecule_chembl_id": "CHEMBL25"}]}),
            ("target_chembl_id=CHEMBL240", _herg_activities()),
        ]
    )
    ls.assert_no_probability_in_nm(ls.fetch_measured_nm(ASPIRIN, opener=opener).as_record())


def test_guard_catches_a_probability_in_an_nm_column():
    with pytest.raises(ValueError, match="probability stored under nM key"):
        ls.assert_no_probability_in_nm({"herg_nm": 0.9, "unit": "probability"})


def test_guard_catches_a_bad_probability_value():
    with pytest.raises(ValueError, match="_prob"):
        ls.assert_no_probability_in_nm({"herg_prob": 7.0})


def test_probabilities_record_passes_the_guard():
    record = ls.PanelProbabilities(smiles="CCO", model="BayeshERG", probabilities={"herg_prob": 0.9}).as_record()
    ls.assert_no_probability_in_nm(record)
    assert record["unit"] == "probability"


def test_bayesherg_and_admet_refuse_loudly_when_absent():
    with pytest.raises(ls.SafetyBackendUnavailableError):
        ls.bayesherg_probability("CCO")
    with pytest.raises(ls.SafetyBackendUnavailableError):
        ls.admet_ai_predictions("CCO")


def test_chembl_targets_cover_every_panel_axis():
    from rfproteina.disease.combinations import LIGAND_AXES

    assert set(ls.CHEMBL_TARGETS) == set(LIGAND_AXES)
    for target in ls.CHEMBL_TARGETS.values():
        assert target["chembl_id"].startswith("CHEMBL")


def test_inchikey_refuses_empty_and_unparsable():
    with pytest.raises(ls.LigandUnreadableError):
        ls._inchikey("")
    with pytest.raises(ls.LigandUnreadableError):
        ls._inchikey("this is not smiles !!!")
