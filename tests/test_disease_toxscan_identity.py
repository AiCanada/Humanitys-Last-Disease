"""Structure matching and original CSV representation for saved ToxScan classifications."""

import hashlib

import pytest

from rfproteina.disease import toxscan_csv as export
from rfproteina.disease import toxscan_inputs as inputs


def portal_bytes(rows):
    # The author's Papa Parse exporter quotes strings, leaves nulls bare, and joins records with CRLF.
    header = ",".join(f'"{field}"' for field in export.HEADERS)
    encoded = [
        ",".join("" if row.get(field) is None else f'"{row[field]}"' for field in export.HEADERS) for row in rows
    ]
    return "\r\n".join([header, *encoded]).encode("ascii")


@pytest.mark.parametrize("smiles", [
    "ClC(Br)=[C@AL1]=C(F)I", "ClC(Br)=[C@AL2]=C(F)I", "C~C", "C1/C=C/CCCC1",
    "C[C@H]([CH3:1])F", "F/C=CCl", "Cl/C=C/F.C1/C=C/CCCC1",
    "C[C@H](O)F |&1:1|", "C[C@H](O)F |o1:1|",
])
def test_unrepresentable_stereo_and_query_graphs_cannot_claim_a_matching_result(smiles):
    with pytest.raises(inputs.ToxScanError, match="toxscan_invalid_structure"):
        inputs.canonical(smiles)


def test_supported_geometry_and_explicit_hydrogen_survive_normalization():
    assert inputs.canonical("Cl/C=C/F") == inputs.canonical("F/C=C/Cl")
    assert inputs.canonical("Cl/C=C/F") != inputs.canonical(r"Cl/C=C\F")
    assert inputs.canonical("F/C=C/C=C/F") != inputs.canonical(r"F/C=C/C=C\F")
    assert inputs.canonical("C[Pt@SP1](Cl)(N)F") != inputs.canonical("C[Pt@SP2](Cl)(N)F")
    assert inputs.canonical("[H]/C(F)=C(Cl)/Br") == inputs.canonical("F/C=C(/Cl)Br")
    assert inputs.canonical("C[C@@H:3](F)O") == inputs.canonical("C[C@H](O)F")


def test_unmarked_nomination_cannot_inherit_an_axially_labeled_export():
    raw = portal_bytes([{"SMILES": "ClC(Br)=[C@AL1]=C(F)I", "Ames Mutagenicity": "YES"}])
    with pytest.raises(inputs.ToxScanError, match="toxscan_invalid_structure"):
        export.read_export(raw, inputs.canonical("FC(I)=C=C(Cl)Br"))


def test_byte_exact_original_serializer_controls_and_blanks():
    # Offline runs of the author's exporter. The first pair uses transcribed Figure S4 calls;
    # the complete second pair is a format control with synthetic numbers and flags.
    study = [{"SMILES": s, "Ames Mutagenicity": call} for s, call in (("NCCNCCN", "NO"), ("NCCNCCNCCNCCN", "YES"))]
    raw = portal_bytes(study)
    assert hashlib.sha256(raw).hexdigest() == "df0b08c60f4dc1ffd0b9cd250a4ab2905a3ebf9bf0cbd3dbf4735e0cfaeeeff2"
    for row in study:
        observation = export.read_export(raw, inputs.canonical(row["SMILES"]))
        assert observation["applicability_domain"] == "unreported" and observation["similarity_score"] is None
        calls = observation["results"]["classifications"]
        assert {key: value for key, value in calls.items() if value is not None} == {
            "Ames Mutagenicity": row["Ames Mutagenicity"],
        }
        assert all(value is None for value in observation["results"]["source_displays"].values())
    all_fields = [
        {
            "SMILES": s, export.AD_HEADER: "In AD", export.SIMILARITY_HEADER: score,
            **{name: f"{10 * (n + 1) + j + 1}mg/kg" for j, name in enumerate(export.REGRESSIONS)},
            **{name: ("NO" if (n + j) % 2 else "YES") for j, name in enumerate(export.CLASSIFICATIONS)},
        } for n, (s, score) in enumerate((("CCO", "0.75"), ("CC(=O)O", "0.85")))
    ]
    raw = portal_bytes(all_fields)
    assert hashlib.sha256(raw).hexdigest() == "e454c6f248707e34692341a6c1feb00fb504f8a04626aa100759fa56df824c0a"
    result = export.read_export(raw, inputs.canonical("C(C)O"))
    assert result["results"]["source_displays"]["Rat oral LD50"] == 15
    assert result["results"]["regression_scale"] == "unverified_source_display"


@pytest.mark.parametrize(("key", "text"), [
    (export.SIMILARITY_HEADER, "1.00000000000000001"), (export.SIMILARITY_HEADER, "-1e-500"),
    ("Rat oral LD50", "1e-1000mg/kg"), ("Rat oral LD50", "1e9999999999999999999999999mg/kg"),
])
def test_out_of_range_or_unrepresentable_source_numbers_do_not_become_valid_floats(key, text):
    raw = portal_bytes([{"SMILES": "CCO", "Ames Mutagenicity": "YES", key: text}])
    with pytest.raises(inputs.ToxScanError, match="toxscan_invalid_export_value"):
        export.read_export(raw, inputs.canonical("CCO"))
