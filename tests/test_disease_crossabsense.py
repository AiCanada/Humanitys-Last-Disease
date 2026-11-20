"""The CrossAbSense download stays behind the 25 GB cap."""

import pytest

from rfproteina.disease.crossabsense import (
    CAP_BYTES,
    ESMC_6B_BYTES,
    ORACLE_ENCODERS,
    ORACLE_FINAL_BYTES,
    download_permitted,
    oracle_bytes,
    property_bytes,
    property_permitted,
    required_bytes,
)


def test_the_five_axes_name_their_published_encoders():
    assert set(ORACLE_ENCODERS) == set(ORACLE_FINAL_BYTES)
    assert ORACLE_ENCODERS["HIC"] == ("esmc_6b",)
    assert ORACLE_ENCODERS["PR_CHO"] == ("esmc_300m",)
    assert "prott5" in ORACLE_ENCODERS["AC-SINS_pH7.4"]
    assert "prott5" in ORACLE_ENCODERS["Tm2"]
    assert "esmc_6b" in ORACLE_ENCODERS["Titer"]


def test_the_encoder_alone_reaches_the_cap():
    assert ESMC_6B_BYTES > CAP_BYTES


def test_the_full_download_is_refused():
    assert oracle_bytes() < CAP_BYTES
    assert required_bytes() > CAP_BYTES
    assert download_permitted() is False


def test_the_gate_is_all_or_nothing_only_because_it_is_asked_that_way():
    """One head fits; the refusal of all five hides which.

    ``PR_CHO`` loads ESM-C 300M alone and needs 1,543,649,952 bytes, comfortably
    under the 25 GB cap. Every other head pulls ESM-C 6B and does not. Measured
    from the constants in this module, so the numbers move only when they do.

    Fitting is not a reason to fetch it: PR_CHO is a primary column of GDPa1,
    the corpus CrossAbSense was fitted on, so a number from that head is the
    most contaminated this program could produce. See
    ``corpus_provenance.overlap_with``.
    """
    assert property_bytes("PR_CHO") == 1_543_649_952
    assert property_permitted("PR_CHO") is True
    for name in ("HIC", "Titer", "AC-SINS_pH7.4", "Tm2"):
        assert property_permitted(name) is False, name
    assert property_bytes("Tm2") == 37_047_251_260
    assert download_permitted() is False


def test_an_unknown_property_raises_rather_than_reporting_nothing_to_fetch():
    with pytest.raises(KeyError):
        property_bytes("not_a_property")
