"""RDKit structure alerts on a small molecule, and the numbers kept out of them.

The safety panel in ``toxicology.py`` reads concentrations. A classifier
probability is not a concentration, and this module is the place that could
quietly blur the two, so one test exists purely to hold that line.
"""

from __future__ import annotations

import pytest

from rfproteina.disease.ligand_alerts import (
    PROPERTY_BARS,
    LigandUnreadableError,
    screen_smiles,
    verdicts,
)

_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
_CATECHOL = "Oc1ccccc1O"


def test_a_plain_molecule_is_measured_on_every_property() -> None:
    screened = screen_smiles(_ASPIRIN)
    assert screened.mol_weight == pytest.approx(180.16, abs=0.1)
    assert screened.heavy_atoms == 13
    assert 0.0 <= screened.qed <= 1.0
    assert screened.h_bond_donors == 1


def test_an_unparsable_smiles_is_refused(tmp_path) -> None:
    with pytest.raises(LigandUnreadableError, match="smiles_unparsable"):
        screen_smiles("this is not a molecule")
    with pytest.raises(LigandUnreadableError, match="empty_smiles"):
        screen_smiles("   ")


def test_alerts_are_reported_with_the_catalogue_that_raised_them() -> None:
    screened = screen_smiles(_CATECHOL)
    for alert in screened.alerts:
        assert alert["name"]
        assert alert["catalogue"]
    assert screened.as_record()["alert_count"] == len(screened.alerts)


def test_a_clean_molecule_clears_the_alert_line() -> None:
    called = verdicts(screen_smiles("CCO"))
    assert called["structure_alerts"]["state"] == "clear"
    assert called["structure_alerts"]["threshold_source"].startswith("program convention")


def test_every_property_bar_names_its_source_or_says_it_is_a_guess() -> None:
    for name, rule in PROPERTY_BARS.items():
        source = str(rule["source"])
        assert source, name
        assert ("Lipinski" in source) or ("suggested default pending calibration" in source)


def test_a_missing_sa_score_says_why_rather_than_reporting_a_number() -> None:
    screened = screen_smiles(_ASPIRIN)
    assert (screened.sa_score is None) != (screened.sa_score_reason is None)


def test_no_probability_is_dressed_up_as_a_concentration() -> None:
    record = screen_smiles(_ASPIRIN).as_record()
    assert not [key for key in record if key.endswith("_nm")]
    assert "no predicted probability is converted into a concentration" in " ".join(record["notes"])
