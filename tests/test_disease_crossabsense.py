"""The CrossAbSense download stays behind the 25 GB cap."""

from rfproteina.disease.crossabsense import (
    CAP_BYTES,
    ESMC_6B_BYTES,
    ORACLE_ENCODERS,
    ORACLE_FINAL_BYTES,
    download_permitted,
    oracle_bytes,
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
