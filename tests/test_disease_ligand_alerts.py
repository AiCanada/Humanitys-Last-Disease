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
    assert screened.alerts
    for alert in screened.alerts:
        assert alert["name"]
        assert alert["catalogue"]
    assert screened.as_record()["alert_count"] == len(screened.alerts)


def test_partial_catalogue_cannot_return_an_exhaustive_zero(monkeypatch):
    from rfproteina.disease import ligand_alerts

    monkeypatch.setattr(ligand_alerts, "CATALOGUES", (*ligand_alerts.CATALOGUES, "ABSENT"))
    clear = verdicts(screen_smiles("CCO"))["structure_alerts"]
    assert clear["state"] == "untested" and clear["value"] is None
    assert clear["missing_catalogues"] == ["ABSENT"]
    assert verdicts(screen_smiles(_CATECHOL))["structure_alerts"]["state"] == "hit"


def test_descriptor_precision_nonfinite_and_unspecified_atoms(monkeypatch):
    from dataclasses import replace

    from rdkit.Chem import Crippen

    monkeypatch.setattr(Crippen, "MolLogP", lambda _: 5.0001)
    screened = screen_smiles("CCO")
    assert verdicts(screened)["clogp"]["state"] == "hit"
    assert verdicts(replace(screened, qed=float("nan")))["qed"]["state"] == "untested"
    with pytest.raises(LigandUnreadableError, match="incomplete_molecule"):
        screen_smiles("C*")


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


# ------------- what the screening review of 2026-11-20 found in the rule of five


def test_the_rule_of_five_is_read_on_the_scale_lipinski_stated_it_on() -> None:
    """The published 5 and 10 are atom counts, not a SMARTS perception.

    Lipinski states donors as "the sum of OHs and NHs" and acceptors as "the sum
    of Ns and Os". RDKit's NumHDonors/NumHAcceptors are a different scale, and
    reading the cited cuts against them let ritonavir clear the acceptor bar its
    own count fails: 11 Ns and Os, which RDKit perceives as 9 acceptors.

    RDKit ships both scales, so this pins the cited cut against the cited scale
    rather than inventing a threshold.
    """
    ritonavir = "CC(C)c1nc(CN(C)C(=O)NC(C(=O)NC(Cc2ccccc2)CC(O)C(Cc2ccccc2)" "NC(=O)OCc2cncs2)C(C)C)cs1"
    screened = screen_smiles(ritonavir)
    assert screened.h_bond_acceptors == 11, "the sum of Ns and Os, not the SMARTS count of 9"
    assert screened.h_bond_donors == 4

    calls = verdicts(screened)
    assert calls["h_bond_acceptors"]["state"] == "hit", "11 is above the published 10"
    assert calls["h_bond_donors"]["state"] == "clear"


def test_both_rule_of_five_cuts_name_the_scale_they_are_counted_on() -> None:
    """A cut that does not name its scale is how the wrong scale got used."""
    for field, descriptor in (("h_bond_donors", "NHOHCount"), ("h_bond_acceptors", "NOCount")):
        source = PROPERTY_BARS[field]["source"]
        assert descriptor in source, f"{field} does not say it is counted as {descriptor}"
        assert "Lipinski" in source


def test_the_rule_of_five_is_blind_to_sulfur_and_says_so() -> None:
    """Reading the cited cut on the cited scale is not a tightening.

    RDKit's donor and acceptor SMARTS include sulfur; Lipinski's counts see only
    N and O. The two scales therefore cross, and a sulfur-rich ligand that the
    SMARTS scale flagged now clears these two bars. That is a limit of the rule
    of five, not a cut of ours to move, so it is pinned here rather than fixed:
    a reader must not take a clear on these bars as a hydrogen-bonding check.
    """
    polythiophene = "c1ccc(s1)" + "c1ccc(s1)" * 10
    screened = screen_smiles(polythiophene)
    assert screened.h_bond_acceptors == 0, "N+O sees no sulfur; the SMARTS scale counted 11"
    assert verdicts(screened)["h_bond_acceptors"]["state"] == "clear"

    hexathiophenol = "Sc1c(S)c(S)c(S)c(S)c1S"
    thiols = screen_smiles(hexathiophenol)
    assert thiols.h_bond_donors == 0, "six thiols, and the rule of five counts none of them"
    assert verdicts(thiols)["h_bond_donors"]["state"] == "clear"

    # The bars must admit this in their own text, or a reader takes the clear
    # above for a hydrogen-bonding verdict it is not.
    joined = " ".join(str(PROPERTY_BARS[f]["source"]) for f in ("h_bond_donors", "h_bond_acceptors"))
    assert "NHOHCount" in joined and "NOCount" in joined
