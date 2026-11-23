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


def _herg_activities(second_endpoint="IC50"):
    return {
        "activities": [
            {
                "standard_type": "IC50",
                "standard_relation": "=",
                "standard_value": "50",
                "assay_chembl_id": "ASSAY1" if second_endpoint == "IC50" else "",
                "standard_units": "nM",
                "activity_id": 1,
            },
            {
                "standard_type": second_endpoint,
                "standard_relation": "=",
                "standard_value": "150",
                "assay_chembl_id": "ASSAY1" if second_endpoint == "IC50" else "",
                "standard_units": "nM",
                "activity_id": 2,
            },
            {
                "standard_type": "IC50",
                "standard_relation": ">",
                "standard_value": "10000",
                "standard_units": "nM",
                "activity_id": 3,
            },
            {
                "standard_type": "IC50",
                "standard_relation": "=",
                "standard_value": "5",
                "standard_units": "uM",
                "activity_id": 4,
            },
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


def test_chembl_keeps_unidentified_assays_separate_and_requires_an_exact_relation():
    rows = _herg_activities()["activities"][:2]
    for row in rows:
        row.pop("assay_chembl_id")
    opener = _FakeOpener([("molecule?", {"molecules": [{"molecule_chembl_id": "CHEMBL25"}]}),
                          ("target_chembl_id=CHEMBL240", {"activities": rows})])
    result = ls.fetch_measured_nm(ASPIRIN, opener=opener).axes["herg"]
    assert result["value_nm"] is None and len(result["opinions"]) == 2
    rows[1]["standard_relation"] = None
    assert ls._nanomolar(rows)[0] == [50.0]
    rows[1]["activity_id"] = rows[0]["activity_id"]
    with pytest.raises(ls.SafetyBackendUnavailableError, match="duplicate_activity_id"):
        ls._axis_activities("CHEMBL25", "CHEMBL240", opener, 100)


def test_every_advertised_chembl_page_contributes_to_the_assay():
    pages = _herg_activities()["activities"]
    opener = _FakeOpener([
        ("offset=0", {"activities": pages[:1], "page_meta": {"next": "?offset=1", "total_count": 2}}),
        ("offset=1", {"activities": pages[1:2], "page_meta": {"next": None, "total_count": 2}}),
    ])
    result = ls._axis_activities("CHEMBL25", "CHEMBL240", opener, 1)
    assert [activity["activity_id"] for activity in result] == [1, 2]
    assert len(opener.seen) == 2


def test_truncated_chembl_result_cannot_clear_an_axis():
    opener = _FakeOpener([("offset=0", {"activities": [], "page_meta": {"next": None, "total_count": 3}})])
    with pytest.raises(ls.SafetyBackendUnavailableError, match="chembl_activities_incomplete"):
        ls._axis_activities("CHEMBL25", "CHEMBL240", opener, 1)


def test_different_chembl_endpoints_and_assays_survive_export_at_their_own_values():
    activities = _herg_activities(second_endpoint="Ki")
    activities["activities"].append({**activities["activities"][0], "assay_chembl_id": "second",
                                     "standard_value": "20000", "activity_id": 5})
    opener = _FakeOpener([
        ("molecule?molecule_structures__standard_inchi_key", {"molecules": [{"molecule_chembl_id": "CHEMBL25"}]}),
        ("target_chembl_id=CHEMBL240", activities),
    ])
    panel = ls.fetch_measured_nm(ASPIRIN, opener=opener)
    assert panel.axes["herg"]["value_nm"] is None
    assert panel.axes["herg"]["n_values"] == 3
    rows = ls.to_toxicology_rows(panel)
    assert {(r["standard_type"], r["assay_id"], float(r["nm"])) for r in rows} == {
        ("IC50", "", 50), ("Ki", "", 150), ("IC50", "second", 20000),
    }


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


def test_bayesherg_and_admet_refuse_loudly_when_absent(monkeypatch):
    """Absence is reported, never faked -- whatever happens to be installed.

    Forced rather than assumed. ADMET-AI is installable here through the
    ``ligand-safety`` extra, so a test that relied on it being missing would pass
    by accident when it is absent and load a real model ensemble when it is not.
    """
    import sys

    monkeypatch.delenv("BAYESHERG_ROOT", raising=False)
    monkeypatch.delenv("BAYESHERG_PYTHON", raising=False)
    monkeypatch.setitem(sys.modules, "admet_ai", None)
    ls.reset_admet_model()
    with pytest.raises(ls.SafetyBackendUnavailableError, match="BAYESHERG_ROOT and BAYESHERG_PYTHON"):
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


# --------------------------------------------------------------------------- ADMET-AI
#
# ADMET-AI itself is optional and heavy, so these drive a stand-in that returns the
# shape the real one returns: a frame indexed by SMILES whose columns are endpoint
# ids. The decoy columns matter as much as the real ones -- they are the values that
# sit inside 0-1 without being probabilities, and the point of the allowlist is that
# they never reach a ``*_prob`` key.

#: The seven allowlisted endpoints plus decoys that a range filter would have kept:
#: ``QED`` is a 0-1 regression, ``LD50_Zhu`` and ``Caco2_Wang`` are logs,
#: ``PAINS_alert`` and ``Lipinski`` are counts, the substrate classifier answers a
#: different question from CYP3A4 inhibition, and the percentile runs 0-100.
_ADMET_ROW = {
    "hERG": 0.91,
    "CYP1A2_Veith": 0.12,
    "CYP2C19_Veith": 0.22,
    "CYP2C9_Veith": 0.32,
    "CYP2D6_Veith": 0.42,
    "CYP3A4_Veith": 0.52,
    "Pgp_Broccatelli": 0.62,
    "QED": 0.55,
    "LD50_Zhu": 0.40,
    "Caco2_Wang": 0.80,
    "PAINS_alert": 0.0,
    "Lipinski": 1.0,
    "CYP3A4_Substrate_CarbonMangels": 0.77,
    "hERG_drugbank_approved_percentile": 0.95,
}

_PANEL_PROB_KEYS = {
    "herg_prob",
    "cyp1a2_prob",
    "cyp2c19_prob",
    "cyp2c9_prob",
    "cyp2d6_prob",
    "cyp3a4_prob",
    "pgp_prob",
}

#: RDKit cannot read this, so ADMET-AI would drop it. Real ADMET-AI raises inside
#: torch when a batch has nothing readable left, so the reader filters first.
_UNPARSABLE = "ThisIsNotSmiles((("


@pytest.fixture(autouse=True)
def _clear_admet_model_cache():
    """No test may inherit another's model.

    The loader caches the model for the life of the process, so a stand-in left
    behind would answer a later test -- including the one that checks the real
    package's absence is reported.
    """
    ls.reset_admet_model()
    yield
    ls.reset_admet_model()


class _FakeFrame:
    """What ADMET-AI hands back for a list of SMILES, reduced to the part read here."""

    def __init__(self, rows):
        self._rows = rows

    def to_dict(self, orient="dict"):
        assert orient == "index", f"the reader must ask for orient='index', not {orient!r}"
        return {key: dict(value) for key, value in self._rows.items()}


class _AdmetSpy:
    """What the stand-in saw: the kwargs it was built with, and each predict call."""

    def __init__(self):
        self.built: list[dict] = []
        self.calls: list[list[str]] = []


def _install_fake_admet(monkeypatch, *, row=None, drop=()):
    """Put a stand-in ``admet_ai`` on sys.modules. Returns an :class:`_AdmetSpy`."""
    import sys
    import types

    spy = _AdmetSpy()
    template = _ADMET_ROW if row is None else row

    class _Model:
        def __init__(self, **kwargs):
            spy.built.append(kwargs)

        def predict(self, smiles):
            assert isinstance(smiles, list), "the reader must always pass a list, never a bare str"
            assert smiles, "an empty batch must never reach the model: real ADMET-AI raises inside torch"
            spy.calls.append(list(smiles))
            return _FakeFrame({item: dict(template) for item in smiles if item not in drop})

    module = types.ModuleType("admet_ai")
    module.ADMETModel = _Model
    monkeypatch.setitem(sys.modules, "admet_ai", module)
    ls.reset_admet_model()
    return spy


def test_admet_allowlist_is_seven_inhibition_classifications():
    from rfproteina.disease.combinations import LIGAND_AXES

    assert set(ls.ADMET_AI_ENDPOINTS) <= set(LIGAND_AXES)
    assert len(ls.ADMET_AI_ENDPOINTS) == 7
    assert ls.ADMET_AI_UNCOVERED_AXES == ("nav1_5", "bcrp", "bsep")
    assert set(ls.ADMET_AI_ENDPOINTS) | set(ls.ADMET_AI_UNCOVERED_AXES) == set(LIGAND_AXES)
    # A substrate classifier answers "is it metabolised by" and must never stand in
    # for an inhibition axis.
    assert not [name for name in ls.ADMET_AI_ENDPOINTS.values() if "Substrate" in name]


def test_admet_reads_only_the_allowlisted_endpoints(monkeypatch):
    _install_fake_admet(monkeypatch)
    reading = ls.admet_ai_predictions(ASPIRIN)
    assert set(reading.probabilities) == _PANEL_PROB_KEYS
    assert reading.probabilities["herg_prob"] == 0.91
    # 0.52 is CYP3A4_Veith (inhibition), not 0.77 from the substrate model.
    assert reading.probabilities["cyp3a4_prob"] == 0.52
    joined = " ".join(reading.probabilities).lower()
    for leaked in ("qed", "ld50", "caco2", "alert", "lipinski", "percentile", "substrate"):
        assert leaked not in joined


def test_admet_record_passes_the_units_guard(monkeypatch):
    _install_fake_admet(monkeypatch)
    record = ls.admet_ai_predictions(ASPIRIN).as_record()
    ls.assert_no_probability_in_nm(record)
    assert record["unit"] == "probability"
    assert record["method"] == ls.ADMET_AI_METHOD
    assert record["endpoints"]["cyp3a4"] == "CYP3A4_Veith"
    assert record["uncovered_axes"] == ["nav1_5", "bcrp", "bsep"]


def test_admet_model_is_built_without_physchem_or_drugbank(monkeypatch):
    spy = _install_fake_admet(monkeypatch)
    ls.admet_ai_predictions(ASPIRIN)
    assert spy.built == [{"include_physchem": False, "drugbank_path": None}]


def test_admet_model_is_loaded_once_across_calls(monkeypatch):
    spy = _install_fake_admet(monkeypatch)
    ls.admet_ai_predictions(ASPIRIN)
    ls.admet_ai_predictions("CCO")
    ls.admet_ai_predictions_many(["CCO", "CCC"])
    assert len(spy.built) == 1, "the ensembles must be loaded once per process, not per molecule"
    ls.reset_admet_model()
    ls.admet_ai_predictions(ASPIRIN)
    assert len(spy.built) == 2, "reset_admet_model must force a rebuild"


def test_admet_refuses_a_value_off_the_unit_interval(monkeypatch):
    _install_fake_admet(monkeypatch, row=dict(_ADMET_ROW, hERG=1.3))
    with pytest.raises(ValueError, match="not a probability on 0-1"):
        ls.admet_ai_predictions(ASPIRIN)


def test_admet_refuses_a_non_finite_value(monkeypatch):
    _install_fake_admet(monkeypatch, row=dict(_ADMET_ROW, hERG=float("nan")))
    with pytest.raises(ValueError, match="not a probability on 0-1"):
        ls.admet_ai_predictions(ASPIRIN)


def test_admet_refuses_when_no_panel_endpoint_is_present(monkeypatch):
    """An empty models directory predicts nothing; that must not read as a clean result."""
    _install_fake_admet(monkeypatch, row={"QED": 0.55, "LD50_Zhu": 0.4})
    with pytest.raises(ls.SafetyBackendUnavailableError, match="no_panel_endpoint"):
        ls.admet_ai_predictions(ASPIRIN)


def test_admet_records_an_endpoint_missing_from_the_build(monkeypatch):
    row = {key: value for key, value in _ADMET_ROW.items() if key != "Pgp_Broccatelli"}
    _install_fake_admet(monkeypatch, row=row)
    reading = ls.admet_ai_predictions(ASPIRIN)
    assert "pgp_prob" not in reading.probabilities
    assert set(reading.probabilities) == _PANEL_PROB_KEYS - {"pgp_prob"}
    assert any("Pgp_Broccatelli" in note for note in reading.notes)


def test_admet_batch_omits_a_smiles_the_backend_dropped(monkeypatch):
    """A molecule RDKit reads but ADMET-AI still drops must not borrow another's row."""
    _install_fake_admet(monkeypatch, drop={"CCO"})
    readings = ls.admet_ai_predictions_many([ASPIRIN, "CCO", "CCC"])
    assert set(readings) == {ASPIRIN, "CCC"}
    assert readings[ASPIRIN].probabilities["herg_prob"] == 0.91


def test_admet_single_smiles_the_backend_drops_is_unreadable(monkeypatch):
    _install_fake_admet(monkeypatch, drop={"CCO"})
    with pytest.raises(ls.LigandUnreadableError, match="unparsable_smiles"):
        ls.admet_ai_predictions("CCO")


def test_admet_filters_unreadable_smiles_before_the_model_sees_them(monkeypatch):
    """Real ADMET-AI raises inside torch on a batch with nothing parsable left.

    Its dataloader has zero length, ``trainer.predict`` returns ``None`` and
    ``torch.cat`` is handed that ``None``. Observed against admet-ai 2.0.1. So the
    unreadable ones must be filtered out before the model is called.
    """
    spy = _install_fake_admet(monkeypatch)
    readings = ls.admet_ai_predictions_many([ASPIRIN, _UNPARSABLE, "CCC"])
    assert set(readings) == {ASPIRIN, "CCC"}
    assert spy.calls == [[ASPIRIN, "CCC"]], "the unreadable SMILES must never reach the model"


def test_admet_all_unreadable_batch_never_calls_the_model(monkeypatch):
    spy = _install_fake_admet(monkeypatch)
    assert ls.admet_ai_predictions_many([_UNPARSABLE, "also bad ((("]) == {}
    assert spy.calls == [], "an all-unreadable batch must not be handed to the model at all"


def test_admet_single_unreadable_smiles_is_refused(monkeypatch):
    _install_fake_admet(monkeypatch)
    with pytest.raises(ls.LigandUnreadableError, match="unparsable_smiles"):
        ls.admet_ai_predictions(_UNPARSABLE)


def test_admet_batch_collapses_duplicates_and_refuses_an_empty_smiles(monkeypatch):
    _install_fake_admet(monkeypatch)
    readings = ls.admet_ai_predictions_many([ASPIRIN, ASPIRIN, " " + ASPIRIN + " "])
    assert set(readings) == {ASPIRIN}
    assert ls.admet_ai_predictions_many([]) == {}
    with pytest.raises(ls.LigandUnreadableError, match="empty_smiles"):
        ls.admet_ai_predictions_many([ASPIRIN, "   "])


def test_admet_still_refuses_loudly_when_the_package_is_absent(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "admet_ai", None)
    ls.reset_admet_model()
    with pytest.raises(ls.SafetyBackendUnavailableError, match="admet_ai_unavailable"):
        ls.admet_ai_predictions(ASPIRIN)


# ------------------------------------------- BindingDB, the offline nM lookup
#
# The point of this route is that it answers for the ligand in hand and for no
# other. A subset holding 1.44M other people's compounds is a reference; only
# the InChIKey match turns one of its rows into a reading about this design.

_BDB_SUBSET_HEADER = (
    "inchikey",
    "inchikey_upstream",
    "smiles",
    "panel_axis",
    "accession",
    "affinity_nanomolar",
    "relation",
    "assay",
    "chains",
    "curation",
    "source",
)
#: Two real structures that are genuinely alike, and one that is not, so a
#: similarity ranking has something to rank.
_SALICYLIC = "OC(=O)c1ccccc1O"
_ETHANOL = "CCO"
_BDB_CURATED = "Curated from the literature by BindingDB"


def _subset(path, rows):
    """A subset file shaped exactly as tables.bindingdb_panel_subset writes one."""
    import csv as _csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_BDB_SUBSET_HEADER)
        writer.writerows(rows)
    return path


def _bdb_reading(
    key,
    axis="herg",
    nm="100",
    relation="=",
    assay="IC50",
    chains="1",
    curation=_BDB_CURATED,
    smiles=ASPIRIN,
):
    return (
        key,
        "STOREDKEY-UHFFFAOYSA-N",
        smiles,
        axis,
        ls.BINDINGDB_PANEL_ACCESSIONS[axis],
        nm,
        relation,
        assay,
        chains,
        curation,
        "BindingDB",
    )


def test_bindingdb_reader_columns_match_what_the_writer_writes():
    """One spelling of the subset's columns, checked across the two modules."""
    from rfproteina.disease import tables

    assert set(ls._BINDINGDB_NEEDED) <= set(tables.PANEL_SUBSET_COLUMNS)
    assert set(_BDB_SUBSET_HEADER) == set(tables.PANEL_SUBSET_COLUMNS)


def test_bindingdb_lookup_takes_the_median_of_this_ligands_readings(tmp_path):
    key = ls._inchikey(ASPIRIN)
    path = _subset(
        tmp_path / "s.tsv",
        [
            _bdb_reading(key, nm="100"),
            _bdb_reading(key, nm="300"),
            _bdb_reading(key, nm="200"),
            # another compound entirely, at the same target
            _bdb_reading("XXXXXXXXXXXXXX-XXXXXXXXXX-X", nm="1"),
        ],
    )
    panel = ls.measured_nm_from_bindingdb(ASPIRIN, path)
    assert panel.method == ls.BINDINGDB_METHOD
    assert panel.inchikey == key
    herg = panel.axes["herg"]
    assert herg["value_nm"] == 200.0, "the median of this ligand's three readings, not the other compound's 1 nM"
    assert herg["n_values"] == 3
    assert herg["unit"] == "nM"
    assert herg["target"] == "Q12809"
    assert herg["chembl_target"] == "", "there is no ChEMBL id on this route and none is invented"
    assert "median_of_3_nM_readings" in herg["note"] and "IC50" in herg["note"]


def test_bindingdb_lookup_leaves_an_unmeasured_ligand_untested(tmp_path):
    """A novel design is not in BindingDB, and nothing is the right answer."""
    path = _subset(tmp_path / "s.tsv", [_bdb_reading("SOMEOTHERKEY-AAAAAAAAAA-N")])
    panel = ls.measured_nm_from_bindingdb(ASPIRIN, path)
    assert all(record["value_nm"] is None for record in panel.axes.values())
    assert all(record["n_values"] == 0 for record in panel.axes.values())
    assert panel.axes["herg"]["note"] == "ligand_not_measured_against_this_target_in_bindingdb"
    assert any("no reading matched" in note for note in panel.notes)
    assert ls.to_toxicology_rows(panel) == [], "nothing measured writes no row, rather than a zero"


def test_bindingdb_keeps_endpoint_specific_medians_in_the_export(tmp_path):
    key = ls._inchikey(ASPIRIN)
    path = _subset(tmp_path / "s.tsv", [_bdb_reading(key, nm="100", assay="Ki"),
                                        _bdb_reading(key, nm="20000", assay="IC50")])
    panel = ls.measured_nm_from_bindingdb(ASPIRIN, path)
    assert panel.axes["herg"]["value_nm"] is None
    assert {(r["standard_type"], float(r["nm"])) for r in ls.to_toxicology_rows(panel)} == {
        ("Ki", 100), ("IC50", 20000),
    }


def test_bindingdb_lookup_excludes_a_bound_and_counts_it(tmp_path):
    key = ls._inchikey(ASPIRIN)
    path = _subset(
        tmp_path / "s.tsv",
        [
            _bdb_reading(key, nm="50"),
            _bdb_reading(key, nm="10000", relation=">"),
            _bdb_reading(key, nm="1", relation="<"),
        ],
    )
    herg = ls.measured_nm_from_bindingdb(ASPIRIN, path).axes["herg"]
    assert herg["value_nm"] == 50.0
    assert herg["n_values"] == 1
    assert "2_bounded_readings_excluded" in herg["note"]


def test_bindingdb_lookup_includes_a_complex_unless_told_otherwise(tmp_path):
    """Dropping multi-chain rows looks like the safe choice and discards the file.

    hERG and the CYPs are assayed with an accessory subunit or a redox partner,
    so 99.0% of the real subset's hERG rows and 97.8% of its CYP3A4 rows name
    more than one chain while still measuring the panel protein -- the rows even
    say ``Target Name: Cytochrome P450 3A4``. Excluding them answered
    "unmeasured" for ketoconazole on CYP3A4, where keeping them gives 48.0 nM
    over 72 readings. So the default keeps them, and strictness has to be asked
    for and says what it cost.
    """
    key = ls._inchikey(ASPIRIN)
    path = _subset(
        tmp_path / "s.tsv",
        [_bdb_reading(key, nm="50"), _bdb_reading(key, nm="500", chains="3")],
    )
    default = ls.measured_nm_from_bindingdb(ASPIRIN, path).axes["herg"]
    assert default["n_values"] == 2, "a complex row still measures the panel protein"
    assert default["value_nm"] == 275.0
    strict = ls.measured_nm_from_bindingdb(ASPIRIN, path, single_chain_only=True).axes["herg"]
    assert strict["value_nm"] == 50.0
    assert strict["n_values"] == 1
    assert "1_multi_chain_readings_excluded" in strict["note"]


def test_bindingdb_lookup_names_the_row_licences_it_used(tmp_path):
    key = ls._inchikey(ASPIRIN)
    path = _subset(
        tmp_path / "s.tsv",
        [_bdb_reading(key, nm="50"), _bdb_reading(key, nm="60", curation="ChEMBL")],
    )
    panel = ls.measured_nm_from_bindingdb(ASPIRIN, path)
    licences = next(note for note in panel.notes if note.startswith("row licences present"))
    assert "ChEMBL" in licences and _BDB_CURATED in licences


def test_bindingdb_lookup_fills_the_protein_column_with_the_accession(tmp_path):
    key = ls._inchikey(ASPIRIN)
    path = _subset(tmp_path / "s.tsv", [_bdb_reading(key, axis="nav1_5", nm="75")])
    rows = ls.to_toxicology_rows(ls.measured_nm_from_bindingdb(ASPIRIN, path))
    assert len(rows) == 1
    assert rows[0]["axis"] == "nav1_5"
    assert rows[0]["protein"] == "Q14524", "the UniProt accession, not a blank cell"
    assert rows[0]["unit"] == "nM"


def test_bindingdb_lookup_record_passes_the_units_guard(tmp_path):
    key = ls._inchikey(ASPIRIN)
    path = _subset(tmp_path / "s.tsv", [_bdb_reading(key, nm="0.5")])
    record = ls.measured_nm_from_bindingdb(ASPIRIN, path).as_record()
    # 0.5 nM is a real concentration that also looks like a probability. The
    # guard must not confuse the two, and the record must declare nM.
    ls.assert_no_probability_in_nm(record)
    assert record["unit"] == "nM"


def test_bindingdb_lookup_refuses_a_missing_or_malformed_subset(tmp_path):
    with pytest.raises(ls.SafetyBackendUnavailableError, match="bindingdb_subset_not_found"):
        ls.measured_nm_from_bindingdb(ASPIRIN, tmp_path / "absent.tsv")
    bare = tmp_path / "bare.tsv"
    bare.write_text("inchikey\tpanel_axis\n", encoding="utf-8")
    with pytest.raises(ls.SafetyBackendUnavailableError, match="bindingdb_subset_column_absent"):
        ls.measured_nm_from_bindingdb(ASPIRIN, bare)
    with pytest.raises(ls.LigandUnreadableError):
        ls.measured_nm_from_bindingdb("", tmp_path / "absent.tsv")


def test_bindingdb_lookup_reads_what_the_reduction_wrote(tmp_path):
    """The two halves, joined: reduce a BindingDB-shaped file, then look a ligand up in it."""
    from rfproteina.disease.tables import bindingdb_panel_subset

    header = (
        "Ligand SMILES",
        "Ligand InChI Key",
        "Target Name",
        "Target Source Organism According to Curator or DataSource",
        "Ki (nM)",
        "IC50 (nM)",
        "Kd (nM)",
        "EC50 (nM)",
        "Curation/DataSource",
        "Article DOI",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (SwissProt) Primary ID of Target Chain 2",
    )
    key = ls._inchikey(ASPIRIN)
    download = tmp_path / "bdb.tsv"
    with download.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        handle.write(
            "\t".join([ASPIRIN, key, "hERG", "Homo sapiens", "", "250", "", "", _BDB_CURATED, "10.1/x", "Q12809", ""])
            + "\n"
        )
        handle.write(
            "\t".join([ASPIRIN, key, "hERG", "Homo sapiens", "", "350", "", "", _BDB_CURATED, "10.1/y", "Q12809", ""])
            + "\n"
        )
    subset = bindingdb_panel_subset(download, tmp_path / "out")
    assert subset.rows == 2
    panel = ls.measured_nm_from_bindingdb(ASPIRIN, subset.path)
    assert panel.axes["herg"]["value_nm"] is None
    assert panel.axes["herg"]["n_values"] == 2
    measurements = {float(row["nm"]): row["source"] for row in ls.to_toxicology_rows(panel)}
    assert set(measurements) == {250, 350}
    assert "10.1/x" in measurements[250]
    assert "10.1/y" in measurements[350]
    assert panel.axes["bsep"]["value_nm"] is None


# ------------------------------------------- BindingDB, the read-across route
#
# The exact-key lookup answers only for a molecule already measured, which a
# designed ligand never is. This is the part that makes the file useful to one
# that has not been: the nearest measured analogue, its own number, and the
# similarity that qualifies it. Every test here also guards the one way it could
# do harm -- a neighbour's number being read as the design's.


@pytest.fixture(autouse=True)
def _clear_bindingdb_index():
    ls.reset_bindingdb_index()
    yield
    ls.reset_bindingdb_index()


def _neighbour_subset(tmp_path):
    """Aspirin and salicylic acid are alike; ethanol is not. hERG for all three."""
    key = ls._inchikey
    return _subset(
        tmp_path / "s.tsv",
        [
            _bdb_reading(key(_SALICYLIC), nm="40", smiles=_SALICYLIC),
            _bdb_reading(key(_ETHANOL), nm="7", smiles=_ETHANOL),
            _bdb_reading(key(ASPIRIN), nm="900", smiles=ASPIRIN, axis="cyp3a4"),
        ],
    )


def test_read_across_finds_the_closest_measured_compound(tmp_path):
    path = _neighbour_subset(tmp_path)
    out = ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.0)
    herg = out.neighbours["herg"]
    assert herg, "aspirin has measured analogues on hERG in this subset"
    assert herg[0]["smiles"] == _SALICYLIC, "salicylic acid is nearer to aspirin than ethanol is"
    assert herg[0]["neighbour_nanomolar"] == 40.0
    assert 0.0 < herg[0]["similarity"] <= 1.0
    assert herg[0]["similarity"] > herg[-1]["similarity"], "ranked by similarity, closest first"
    assert out.method == ls.READ_ACROSS_METHOD
    assert out.fingerprint == "morgan_r2_2048_tanimoto"


def test_read_across_is_never_a_reading_of_this_design(tmp_path):
    """The record must not be mistakable for a measurement, by construction."""
    path = _neighbour_subset(tmp_path)
    record = ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.0).as_record()
    flat = json.dumps(record)
    assert '"neighbour_nanomolar"' in flat
    # Nothing a panel reader would take as a concentration for this design.
    for entry in record["neighbours"]["herg"]:
        assert not [key for key in entry if key.endswith("_nm")]
        assert "similarity" in entry, "a neighbour's number never travels without its similarity"
        assert entry["inchikey"], "nor without naming the compound it belongs to"
    assert any("DIFFERENT molecule" in note for note in record["notes"])
    assert any("*_nm" in note for note in record["notes"])
    # And the probability guard still passes over it: no probability, no nM key.
    ls.assert_no_probability_in_nm(record)


def test_read_across_honours_the_similarity_floor(tmp_path):
    path = _neighbour_subset(tmp_path)
    loose = ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.0).neighbours["herg"]
    strict = ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.99).neighbours["herg"]
    assert len(loose) == 2
    assert strict == [], "nothing in this subset is a 0.99 analogue of aspirin"


def test_read_across_caps_the_number_of_neighbours(tmp_path):
    path = _neighbour_subset(tmp_path)
    out = ls.nearest_measured_neighbours(ASPIRIN, path, neighbours=1, min_similarity=0.0)
    assert len(out.neighbours["herg"]) == 1


def test_read_across_scores_the_same_molecule_as_identical(tmp_path):
    path = _subset(tmp_path / "s.tsv", [_bdb_reading(ls._inchikey(ASPIRIN), nm="12", smiles=ASPIRIN)])
    out = ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.0)
    assert out.neighbours["herg"][0]["similarity"] == 1.0
    assert out.closest["herg"] == 1.0


def test_read_across_keeps_the_most_potent_reading_per_compound(tmp_path):
    """Understating a liability is the dangerous direction, so the lowest wins."""
    key = ls._inchikey(ASPIRIN)
    path = _subset(
        tmp_path / "s.tsv",
        [
            _bdb_reading(key, nm="900", smiles=ASPIRIN),
            _bdb_reading(key, nm="30", smiles=ASPIRIN),
            _bdb_reading(key, nm="400", smiles=ASPIRIN),
        ],
    )
    (entry,) = ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.0).neighbours["herg"]
    assert entry["neighbour_nanomolar"] == 30.0
    assert entry["n_readings"] == 3


def test_read_across_excludes_a_bounded_reading(tmp_path):
    path = _subset(tmp_path / "s.tsv", [_bdb_reading(ls._inchikey(ASPIRIN), nm="10000", relation=">", smiles=ASPIRIN)])
    assert ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.0).neighbours["herg"] == []


def test_read_across_reports_an_axis_with_nothing_measured(tmp_path):
    path = _neighbour_subset(tmp_path)
    out = ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.0)
    assert out.neighbours["bsep"] == [], "an axis with no measured compound says so, rather than borrowing one"
    assert out.closest["bsep"] == 0.0


def test_read_across_names_the_row_licences_of_the_neighbours_it_used(tmp_path):
    path = _subset(
        tmp_path / "s.tsv",
        [_bdb_reading(ls._inchikey(ASPIRIN), smiles=ASPIRIN, curation="ChEMBL")],
    )
    out = ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.0)
    assert any("ChEMBL" in note for note in out.notes)


def test_read_across_builds_its_index_once_per_file(tmp_path):
    path = _neighbour_subset(tmp_path)
    ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.0)
    assert len(ls._BDB_INDEX) == 1
    ls.nearest_measured_neighbours(_SALICYLIC, path, min_similarity=0.0)
    assert len(ls._BDB_INDEX) == 1, "a second query must reuse the parsed structures"
    ls.reset_bindingdb_index()
    assert ls._BDB_INDEX == {}


def test_read_across_refuses_bad_input(tmp_path):
    path = _neighbour_subset(tmp_path)
    with pytest.raises(ls.LigandUnreadableError):
        ls.nearest_measured_neighbours("ThisIsNotSmiles(((", path)
    with pytest.raises(ls.LigandUnreadableError):
        ls.nearest_measured_neighbours("", path)
    with pytest.raises(ls.SafetyBackendUnavailableError, match="bindingdb_subset_not_found"):
        ls.nearest_measured_neighbours(ASPIRIN, tmp_path / "absent.tsv")
    with pytest.raises(ValueError, match="at least 1"):
        ls.nearest_measured_neighbours(ASPIRIN, path, neighbours=0)
    with pytest.raises(ValueError, match="must be on 0-1"):
        ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=1.5)


def test_read_across_refuses_a_subset_without_structures(tmp_path):
    """The exact-key lookup does not need smiles; this cannot work without it."""
    path = tmp_path / "s.tsv"
    # Everything the shared parse needs, so smiles is the only thing absent.
    path.write_text("inchikey\tpanel_axis\taffinity_nanomolar\trelation\tassay\tchains\tcuration\n", encoding="utf-8")
    with pytest.raises(ls.SafetyBackendUnavailableError, match="column_absent:smiles"):
        ls.nearest_measured_neighbours(ASPIRIN, path)


# --------------------------------------- BindingDB, matching at two resolutions
#
# BindingDB spells one drug several ways. An exact key answers for exactly the
# stereoisomer and salt in hand, which is right and also why ketoconazole
# resolved on six axes and not on CYP3A4 -- its CYP3A4 rows carry a different
# form. Connectivity finds those, at the cost of pooling forms that are not the
# same molecule, so it has to say so.


def test_bindingdb_lookup_connectivity_finds_another_form_of_the_same_drug(tmp_path):
    key = ls._inchikey(ASPIRIN)
    skeleton = key.split("-")[0]
    other_form = f"{skeleton}-ZZZZZZZZZZ-N"
    path = _subset(
        tmp_path / "s.tsv",
        [
            _bdb_reading(key, axis="herg", nm="100", smiles=ASPIRIN),
            _bdb_reading(other_form, axis="cyp3a4", nm="7", smiles=ASPIRIN),
        ],
    )
    exact = ls.measured_nm_from_bindingdb(ASPIRIN, path)
    assert exact.axes["herg"]["value_nm"] == 100.0
    assert exact.axes["cyp3a4"]["value_nm"] is None, "a different salt is a different key"

    wide = ls.measured_nm_from_bindingdb(ASPIRIN, path, match="connectivity")
    assert wide.axes["herg"]["value_nm"] == 100.0
    assert wide.axes["cyp3a4"]["value_nm"] == 7.0, "same connectivity, so the other form is found"


def test_bindingdb_connectivity_match_labels_itself(tmp_path):
    key = ls._inchikey(ASPIRIN)
    other_form = f"{key.split('-')[0]}-ZZZZZZZZZZ-N"
    path = _subset(
        tmp_path / "s.tsv",
        [_bdb_reading(key, nm="100", smiles=ASPIRIN), _bdb_reading(other_form, nm="300", smiles=ASPIRIN)],
    )
    wide = ls.measured_nm_from_bindingdb(ASPIRIN, path, match="connectivity")
    assert wide.method == ls.BINDINGDB_CONNECTIVITY_METHOD
    assert wide.method != ls.BINDINGDB_METHOD, "a pooled reading must not wear the exact method's name"
    assert any("skeleton" in note and "may differ" in note for note in wide.notes)
    herg = wide.axes["herg"]
    assert herg["n_values"] == 2
    assert herg["value_nm"] == 200.0, "the median across both forms"
    assert "2_distinct_structures" in herg["note"], "the pooling is counted, not hidden"


def test_bindingdb_exact_match_stays_the_default_and_says_so(tmp_path):
    path = _subset(tmp_path / "s.tsv", [_bdb_reading(ls._inchikey(ASPIRIN), smiles=ASPIRIN)])
    out = ls.measured_nm_from_bindingdb(ASPIRIN, path)
    assert out.method == ls.BINDINGDB_METHOD
    assert any("full InChIKey" in note for note in out.notes)
    assert not any("distinct_structures" in record["note"] for record in out.axes.values())


def test_bindingdb_lookup_refuses_an_unknown_match_mode(tmp_path):
    path = _subset(tmp_path / "s.tsv", [_bdb_reading(ls._inchikey(ASPIRIN), smiles=ASPIRIN)])
    with pytest.raises(ValueError, match="match must be one of"):
        ls.measured_nm_from_bindingdb(ASPIRIN, path, match="fuzzy")


def test_bindingdb_subset_is_parsed_once_for_both_readers(tmp_path):
    """One parse of 60,095 rows, two readers. The fingerprints are the extra half."""
    path = _neighbour_subset(tmp_path)
    ls.reset_bindingdb_index()
    assert ls._BDB_ROWS == {} and ls._BDB_INDEX == {}
    ls.measured_nm_from_bindingdb(ASPIRIN, path)
    assert len(ls._BDB_ROWS) == 1, "the exact lookup parsed it"
    assert ls._BDB_INDEX == {}, "and did not pay for fingerprints it does not use"
    ls.nearest_measured_neighbours(ASPIRIN, path, min_similarity=0.0)
    assert len(ls._BDB_ROWS) == 1, "the read-across reused that parse rather than repeating it"
    assert len(ls._BDB_INDEX) == 1
    ls.reset_bindingdb_index()
    assert ls._BDB_ROWS == {} and ls._BDB_INDEX == {}


def test_bindingdb_cache_notices_a_rebuilt_subset(tmp_path):
    """A subset rewritten in place must not be answered out of the old parse."""
    import os
    import time

    path = _subset(tmp_path / "s.tsv", [_bdb_reading(ls._inchikey(ASPIRIN), nm="100", smiles=ASPIRIN)])
    assert ls.measured_nm_from_bindingdb(ASPIRIN, path).axes["herg"]["value_nm"] == 100.0
    time.sleep(0.01)
    _subset(path, [_bdb_reading(ls._inchikey(ASPIRIN), nm="4000", smiles=ASPIRIN)])
    os.utime(path, (time.time() + 1, time.time() + 1))
    assert ls.measured_nm_from_bindingdb(ASPIRIN, path).axes["herg"]["value_nm"] == 4000.0


# ----------------------------------- GtoPdb, secondary pharmacology by analogy
#
# What the ten-axis panel cannot ask: not "does this hit hERG" but "what else
# does this hit, and did it end a drug". Validated live against the real
# download on terfenadine -- withdrawn for hERG-mediated torsades -- where the
# screen returned H1 as its primary target, Kv11.1 at 56 nM marked off-target,
# and the withdrawal flagged first. These hold the contract that made that
# readable.

_OT_COLUMNS = (
    "inchikey",
    "inchikey_upstream",
    "smiles",
    "ligand_name",
    "ligand_id",
    "target_id",
    "target_name",
    "gene_symbol",
    "family_class",
    "family_name",
    "affinity_nanomolar",
    "relation",
    "assay",
    "action",
    "selectivity",
    "primary_target",
    "approved",
    "withdrawn",
    "source",
)


@pytest.fixture(autouse=True)
def _clear_offtarget_index():
    ls.reset_offtarget_index()
    yield
    ls.reset_offtarget_index()


def _ot_row(
    smiles,
    *,
    name="compound",
    target="K<SUB>v</SUB>11.1",
    symbol="KCNH2",
    family="vgic",
    family_name="Voltage-gated potassium channels",
    nm="56.0",
    action="Channel blocker",
    primary="false",
    approved="",
    withdrawn="",
):
    key = ls._inchikey(smiles)
    return (
        key,
        "PUBLISHED-N",
        smiles,
        name,
        "1",
        "572",
        target,
        symbol,
        family,
        family_name,
        nm,
        "=",
        "IC50",
        action,
        "Not Determined",
        primary,
        approved,
        withdrawn,
        "GtoPdb 2026.3.1; ODbL",
    )


def _ot_subset(path, rows):
    import csv as _csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_OT_COLUMNS)
        writer.writerows(rows)
    return path


def test_offtarget_columns_match_what_the_writer_writes():
    from rfproteina.disease import tables

    assert set(ls._OFFTARGET_NEEDED) <= set(tables.GTOPDB_OFFTARGET_COLUMNS)
    assert set(_OT_COLUMNS) == set(tables.GTOPDB_OFFTARGET_COLUMNS)


def test_offtarget_profile_groups_hits_by_family(tmp_path):
    path = _ot_subset(
        tmp_path / "ot.tsv",
        [
            _ot_row(ASPIRIN, nm="56.0", family="vgic", target="Kv11.1"),
            _ot_row(ASPIRIN, nm="27.0", family="gpcr", target="H1 receptor", action="Antagonist", primary="true"),
            _ot_row(ASPIRIN, nm="8100.0", family="enzyme", target="CYP2J2", action="Inhibitor"),
        ],
    )
    profile = ls.offtarget_profile(ASPIRIN, path, min_similarity=0.0)
    assert set(profile.families) == {"vgic", "gpcr", "enzyme"}
    assert profile.method == ls.OFFTARGET_METHOD
    assert profile.strongest == {"gpcr": 27.0, "vgic": 56.0, "enzyme": 8100.0}
    assert profile.families["gpcr"][0]["primary_target"] is True
    assert profile.families["vgic"][0]["primary_target"] is False, "an off-target hit is marked as one"
    assert profile.families["enzyme"][0]["action"] == "Inhibitor"
    (analogue,) = profile.analogues
    assert analogue["measured_targets"] == 3


def test_offtarget_profile_puts_a_withdrawn_analogue_first(tmp_path):
    """The loudest thing the file can say, so it leads the record."""
    path = _ot_subset(tmp_path / "ot.tsv", [_ot_row(ASPIRIN, name="terfenadine-like", withdrawn="yes", approved="yes")])
    profile = ls.offtarget_profile(ASPIRIN, path, min_similarity=0.0)
    assert profile.withdrawn_analogues == ("terfenadine-like",)
    assert profile.approved_analogues == ("terfenadine-like",), "approved then withdrawn is how GtoPdb marks one"
    assert profile.notes[0].startswith("a close analogue is a WITHDRAWN drug")
    assert profile.analogues[0]["withdrawn"] is True


def test_offtarget_profile_says_nothing_when_nothing_is_close(tmp_path):
    path = _ot_subset(tmp_path / "ot.tsv", [_ot_row(ASPIRIN)])
    profile = ls.offtarget_profile("CCCCCCCCCCCCCCCCCCNCCO", path, min_similarity=0.95)
    assert profile.analogues == ()
    assert profile.families == {}
    assert profile.withdrawn_analogues == ()
    assert any("no measured compound reached" in note for note in profile.notes)


def test_offtarget_profile_is_never_a_reading_of_this_design(tmp_path):
    path = _ot_subset(tmp_path / "ot.tsv", [_ot_row(ASPIRIN)])
    record = ls.offtarget_profile(ASPIRIN, path, min_similarity=0.0).as_record()
    flat = json.dumps(record)
    assert '"neighbour_nanomolar"' in flat
    for hits in record["families"].values():
        for hit in hits:
            assert not [key for key in hit if key.endswith("_nm")]
            assert hit["similarity"] and hit["neighbour_inchikey"] and hit["neighbour_name"]
    assert any("DIFFERENT molecule" in note for note in record["notes"])
    assert any("ODbL" in note for note in record["notes"]), "the licence travels with the analogy"
    ls.assert_no_probability_in_nm(record)


def test_offtarget_profile_caps_what_it_reports(tmp_path):
    rows = [_ot_row(ASPIRIN, nm=f"{index + 1}.0", target=f"target{index}") for index in range(9)]
    path = _ot_subset(tmp_path / "ot.tsv", rows)
    profile = ls.offtarget_profile(ASPIRIN, path, min_similarity=0.0, per_family=3)
    assert len(profile.families["vgic"]) == 3
    assert [hit["neighbour_nanomolar"] for hit in profile.families["vgic"]] == [1.0, 2.0, 3.0], "most potent first"


def test_offtarget_profile_caps_the_analogues_it_uses(tmp_path):
    path = _ot_subset(
        tmp_path / "ot.tsv",
        [_ot_row(ASPIRIN, name="a"), _ot_row(_SALICYLIC, name="b"), _ot_row(_ETHANOL, name="c")],
    )
    profile = ls.offtarget_profile(ASPIRIN, path, analogues=2, min_similarity=0.0)
    assert len(profile.analogues) == 2
    assert profile.analogues[0]["similarity"] >= profile.analogues[1]["similarity"]


def test_offtarget_index_is_built_once_and_resettable(tmp_path):
    path = _ot_subset(tmp_path / "ot.tsv", [_ot_row(ASPIRIN)])
    ls.offtarget_profile(ASPIRIN, path, min_similarity=0.0)
    assert len(ls._OFFTARGET_INDEX) == 1
    ls.offtarget_profile(_SALICYLIC, path, min_similarity=0.0)
    assert len(ls._OFFTARGET_INDEX) == 1, "a second query reuses the parsed structures"
    ls.reset_offtarget_index()
    assert ls._OFFTARGET_INDEX == {}


def test_offtarget_profile_refuses_bad_input(tmp_path):
    path = _ot_subset(tmp_path / "ot.tsv", [_ot_row(ASPIRIN)])
    with pytest.raises(ls.LigandUnreadableError):
        ls.offtarget_profile("ThisIsNotSmiles(((", path)
    with pytest.raises(ls.SafetyBackendUnavailableError, match="subset_not_found"):
        ls.offtarget_profile(ASPIRIN, tmp_path / "absent.tsv")
    with pytest.raises(ValueError, match="at least 1"):
        ls.offtarget_profile(ASPIRIN, path, analogues=0)
    with pytest.raises(ValueError, match="must be on 0-1"):
        ls.offtarget_profile(ASPIRIN, path, min_similarity=2.0)
    bare = tmp_path / "bare.tsv"
    bare.write_text("inchikey\tsmiles\n", encoding="utf-8")
    with pytest.raises(ls.SafetyBackendUnavailableError, match="column_absent"):
        ls.offtarget_profile(ASPIRIN, bare)
