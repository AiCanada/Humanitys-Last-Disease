"""The sequence physicochemistry against its published sources.

Every constant here is copied from the primary source named beside it, not
from the module under test, so a retyped table or a reworded formula fails
instead of agreeing with itself.
"""

import math

import pytest

from rfproteina.disease.chemical import (
    _CTERM_SPECIFIC,
    _NEGATIVE,
    _NTERM_SPECIFIC,
    _POSITIVE,
    PI_BLOCKER,
    chemical_record,
    isoelectric_point,
    net_charge,
)
from rfproteina.disease.modality import (
    _A3V,
    _ALIPHATIC_ILE_LEU,
    _ALIPHATIC_VAL,
    _HALF_LIFE,
    _HST,
    _aggrescan,
    _aggrescan_window,
    _aliphatic_index,
    _atom_counts,
    _half_life,
    _molecular_weight,
)
from rfproteina.metrics.physics.developability_metrics import (
    _DIWV,
    _KYTE_DOOLITTLE,
    compute_gravy,
    compute_instability_index,
)
from tests.helpers.disease import UBIQUITIN

#: Human ubiquitin, the sequence the rest of this suite screens with.
#: Bovine pancreatic trypsin inhibitor: His, Cys and Tyr all present.
BPTI = "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA"
#: The insulin B chain, short enough that the pKa terminals dominate.
INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"

# Kyte, J. and Doolittle, R.F., J. Mol. Biol. 1982, 157:105-132, Table 2.
KYTE_DOOLITTLE_1982 = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}  # fmt: skip

# Sanchez de Groot et al., BMC Struct. Biol. 2005, 5:18, as tabulated by
# AGGRESCAN (Conchillo-Sole et al., BMC Bioinformatics 2007, 8:65).
A3V_2007 = {
    "I": "1.822", "F": "1.754", "V": "1.594", "L": "1.380", "Y": "1.159",
    "W": "1.037", "M": "0.910", "C": "0.604", "A": "-0.036", "T": "-0.159",
    "S": "-0.294", "P": "-0.334", "G": "-0.535", "K": "-0.931", "H": "-1.033",
    "Q": "-1.231", "R": "-1.240", "N": "-1.302", "E": "-1.412", "D": "-1.836",
}  # fmt: skip

# Bjellqvist et al., Electrophoresis 1994, 15:529-539, the set ExPASy Compute
# pI/Mw and Biopython's IsoelectricPoint both encode.
BJELLQVIST_POSITIVE = {"K": 10.0, "R": 12.0, "H": 5.98}
BJELLQVIST_NEGATIVE = {"D": 4.05, "E": 4.45, "C": 9.0, "Y": 10.0}
BJELLQVIST_NTERM = {"A": 7.59, "M": 7.00, "S": 6.93, "P": 8.36, "T": 6.82, "V": 7.44, "E": 7.70}
BJELLQVIST_CTERM = {"D": 4.55, "E": 4.75}

# ExPASy ProtParam, Gasteiger et al. 2005: the N-end rule words per terminus,
# as (mammalian reticulocytes, yeast, E. coli). Proline has no E. coli figure.
N_END_RULE_2005 = {
    "A": ("4.4 hour", ">20 hour", ">10 hour"),
    "R": ("1 hour", "2 min", "2 min"),
    "N": ("1.4 hour", "3 min", ">10 hour"),
    "D": ("1.1 hour", "3 min", ">10 hour"),
    "C": ("1.2 hour", ">20 hour", ">10 hour"),
    "Q": ("0.8 hour", "10 min", ">10 hour"),
    "E": ("1 hour", "30 min", ">10 hour"),
    "G": ("30 hour", ">20 hour", ">10 hour"),
    "H": ("3.5 hour", "10 min", ">10 hour"),
    "I": ("20 hour", "30 min", ">10 hour"),
    "L": ("5.5 hour", "3 min", "2 min"),
    "K": ("1.3 hour", "3 min", "2 min"),
    "M": ("30 hour", ">20 hour", ">10 hour"),
    "F": ("1.1 hour", "3 min", "2 min"),
    "P": (">20 hour", ">20 hour", None),
    "S": ("1.9 hour", ">20 hour", ">10 hour"),
    "T": ("7.2 hour", ">20 hour", ">10 hour"),
    "W": ("2.8 hour", "3 min", "2 min"),
    "Y": ("2.8 hour", "10 min", "2 min"),
    "V": ("100 hour", ">20 hour", ">10 hour"),
}

ALL_TWENTY = "ACDEFGHIKLMNPQRSTVWY"


def test_the_kyte_doolittle_scale_is_the_1982_table():
    assert _KYTE_DOOLITTLE == KYTE_DOOLITTLE_1982


def test_gravy_is_the_mean_over_the_whole_chain():
    """ExPASy: the sum of the hydropathy values divided by the residue count."""
    expected = sum(KYTE_DOOLITTLE_1982[letter] for letter in UBIQUITIN) / len(UBIQUITIN)
    assert compute_gravy(UBIQUITIN) == pytest.approx(expected, abs=1e-12)
    assert compute_gravy(UBIQUITIN) == pytest.approx(-0.4895, abs=5e-5)
    # Lower case is the same chain. An empty chain has no mean. A letter the
    # scale does not cover is refused rather than skipped, which would divide
    # a short sum by the full length.
    assert compute_gravy(UBIQUITIN.lower()) == compute_gravy(UBIQUITIN)
    assert math.isnan(compute_gravy(""))
    with pytest.raises(ValueError, match="non-standard"):
        compute_gravy("MQIFX")


def test_every_one_of_the_four_hundred_diwv_cells_is_present():
    assert set(_DIWV) == set(ALL_TWENTY)
    for first in ALL_TWENTY:
        assert set(_DIWV[first]) == set(ALL_TWENTY), first


def test_the_diwv_table_keeps_the_published_irregular_cells():
    """Guruprasad 1990 as ExPASy and Biopython encode it.

    These five are the ones a retyped table rounds off: the stray third
    decimal on Phe-Tyr, the 24.64 that is not the 24.68 everywhere else, the
    lone 18.38, the lone -15.91, and the only 58.28 pair outside Arg-Arg.
    """
    assert _DIWV["F"]["Y"] == 33.601
    assert _DIWV["K"]["Q"] == 24.64
    assert _DIWV["P"]["E"] == 18.38
    assert _DIWV["Y"]["R"] == -15.91
    assert _DIWV["M"]["H"] == 58.28
    assert _DIWV["R"]["R"] == 58.28


def test_the_instability_index_is_ten_over_length_times_the_dipeptide_sum():
    """ExPASy: II = (10/L) * sum of DIWV over the L-1 dipeptides."""
    pairs = sum(_DIWV[UBIQUITIN[i]][UBIQUITIN[i + 1]] for i in range(len(UBIQUITIN) - 1))
    assert compute_instability_index(UBIQUITIN) == pytest.approx((10.0 / len(UBIQUITIN)) * pairs, abs=1e-12)
    assert compute_instability_index(UBIQUITIN) == pytest.approx(36.0553, abs=5e-5)
    # The divisor is the full length, so the last residue counts even though
    # no dipeptide starts on it: a 76-mer divides by 76, not by 75.
    assert compute_instability_index("MQ") == pytest.approx(5.0 * _DIWV["M"]["Q"], abs=1e-12)
    assert math.isnan(compute_instability_index("M"))
    assert math.isnan(compute_instability_index(""))


def test_the_pka_tables_are_the_bjellqvist_set():
    assert _POSITIVE == BJELLQVIST_POSITIVE
    assert _NEGATIVE == BJELLQVIST_NEGATIVE
    assert _NTERM_SPECIFIC == BJELLQVIST_NTERM
    assert _CTERM_SPECIFIC == BJELLQVIST_CTERM


def test_the_net_charge_is_henderson_hasselbalch_on_every_group():
    def published(sequence: str, ph: float) -> float:
        total = 0.0
        for table, residue, sign in (
            (BJELLQVIST_NTERM, sequence[0], 1.0),
            (BJELLQVIST_CTERM, sequence[-1], -1.0),
        ):
            pka = table.get(residue, 7.50 if sign > 0 else 3.55)
            ratio = 10 ** ((pka - ph) if sign > 0 else (ph - pka))
            total += sign * ratio / (ratio + 1.0)
        for letter in sequence:
            if letter in BJELLQVIST_POSITIVE:
                ratio = 10 ** (BJELLQVIST_POSITIVE[letter] - ph)
                total += ratio / (ratio + 1.0)
            elif letter in BJELLQVIST_NEGATIVE:
                ratio = 10 ** (ph - BJELLQVIST_NEGATIVE[letter])
                total -= ratio / (ratio + 1.0)
        return total

    for sequence in (UBIQUITIN, BPTI, INSULIN_B):
        for ph in (2.0, 7.0, 7.4, 11.0):
            assert net_charge(sequence, ph) == pytest.approx(published(sequence, ph), abs=1e-9)


def test_a_terminal_pka_applies_only_at_its_own_terminus():
    """Glu is 7.70 as an N-terminus and 4.75 as a C-terminus, never mid-chain."""
    # N-terminal Glu at its own pKa is exactly half protonated; the Gly
    # C-terminus keeps the 3.55 default and the Glu side chain keeps 4.45.
    assert net_charge("EG", 7.70) == pytest.approx(
        0.5 - (1.0 / (10 ** (3.55 - 7.70) + 1.0)) - (1.0 / (10 ** (4.45 - 7.70) + 1.0)),
        abs=1e-9,
    )
    # The same Glu as a C-terminus takes 4.75 instead, so the charge differs.
    assert net_charge("GE", 7.70) != pytest.approx(net_charge("EG", 7.70), abs=1e-6)
    # An internal Glu keeps the 4.45 side-chain value and the default termini.
    internal = net_charge("GEG", 7.0)
    default = (1.0 / (10 ** (7.0 - 7.50) + 1.0)) - (1.0 / (10 ** (3.55 - 7.0) + 1.0))
    assert internal == pytest.approx(default - (1.0 / (10 ** (4.45 - 7.0) + 1.0)), abs=1e-9)


def test_the_pi_is_the_zero_crossing_and_a_chain_with_no_side_chain_still_has_one():
    assert isoelectric_point(UBIQUITIN) == pytest.approx(6.5616, abs=5e-5)
    assert isoelectric_point(BPTI) == pytest.approx(9.2426, abs=5e-5)
    for sequence in (UBIQUITIN, BPTI, INSULIN_B, "GGGGG", "AAAAA"):
        assert abs(net_charge(sequence, isoelectric_point(sequence))) < 1e-7
    # Glycine alone: only the two termini ionise, so the crossing is their mean.
    assert isoelectric_point("G") == pytest.approx((7.50 + 3.55) / 2, abs=5e-5)


def test_a_chain_with_no_zero_crossing_is_refused_rather_than_pinned_to_the_bound():
    """A hundred arginines are still net positive at pH 14.

    Halving a bracket that holds no crossing walks to the bound, so the old
    solve returned a flat 14.0000 and the record stored it beside the pKa-set
    name as though it had been solved for.
    """
    basic = "R" * 101
    assert net_charge(basic, 14.0) > 0.0
    with pytest.raises(ValueError, match=PI_BLOCKER):
        isoelectric_point(basic)
    record = chemical_record(basic, partners=None)
    assert record is not None
    assert "isoelectric_point" not in record
    assert record["isoelectric_point_blocker"] == PI_BLOCKER
    # The charge was measured and the motif counts were measured, so they stay.
    assert record["net_charge_ph74"] == pytest.approx(100.5549, abs=5e-5)
    assert record["pka_set"] == "bjellqvist_1994"
    assert record["methionine_count"] == 0


def test_a_solvable_chain_still_stores_its_pi():
    record = chemical_record(UBIQUITIN, partners=None)
    assert record is not None
    assert record["isoelectric_point"] == pytest.approx(6.5616, abs=5e-5)
    assert "isoelectric_point_blocker" not in record


def test_the_aliphatic_index_is_ikai_1980():
    """ExPASy: X(Ala) + a*X(Val) + b*(X(Ile)+X(Leu)) with a=2.9 and b=3.9."""
    assert float(_ALIPHATIC_VAL) == 2.9
    assert float(_ALIPHATIC_ILE_LEU) == 3.9
    for sequence in (UBIQUITIN, BPTI, INSULIN_B):
        length = len(sequence)
        expected = (
            sequence.count("A") / length * 100
            + 2.9 * (sequence.count("V") / length * 100)
            + 3.9 * ((sequence.count("I") + sequence.count("L")) / length * 100)
        )
        assert _aliphatic_index(sequence) == pytest.approx(expected, abs=5e-5)
    assert _aliphatic_index(UBIQUITIN) == pytest.approx(100.0, abs=5e-5)
    # X is a mole percent, so doubling the chain does not change the index.
    assert _aliphatic_index(UBIQUITIN * 2) == pytest.approx(_aliphatic_index(UBIQUITIN), abs=5e-5)


def test_the_a3v_scale_and_window_rule_are_aggrescan_2007():
    assert {letter: str(value) for letter, value in _A3V.items()} == A3V_2007
    assert float(_HST) == -0.02
    # "a window size of 5 for <= 75 residues, 7 for <= 175, 9 for <= 300 and
    # 11 for > 300" -- Conchillo-Sole et al. 2007.
    assert [_aggrescan_window(n) for n in (1, 75, 76, 175, 176, 300, 301, 5000)] == [
        5, 5, 7, 7, 9, 9, 11, 11,
    ]  # fmt: skip


def test_the_virtual_residues_are_basic_at_the_n_terminus_and_acidic_at_the_c_terminus():
    """Residue 0 is the mean a3v of Lys and Arg, residue n+1 the mean of Asp and Glu.

    The published order is the chemically sensible one: a free alpha-amino
    group reads as basic and a free alpha-carboxyl as acidic. The AGGRESCAN
    web server behaves as though the two were swapped, and at least one
    reimplementation copies the server, so this pins the paper's order. The
    only residues that can tell them apart are the first and last windows.
    """
    basic = (float(_A3V["K"]) + float(_A3V["R"])) / 2
    acidic = (float(_A3V["D"]) + float(_A3V["E"])) / 2
    chain = "A" * 9
    found = _aggrescan(chain)
    assert found is not None
    window = found["window"]
    assert window == 5
    plain = float(_A3V["A"])
    # a4vSS over nine alanines: seven interior windows of pure Ala, plus the
    # two end windows that each carry one virtual residue, and the two
    # off-centre residues that copy them.
    head = (basic + 4 * plain) / 5
    tail = (acidic + 4 * plain) / 5
    expected = 2 * head + 2 * tail + 5 * plain
    assert found["a4vss"] == pytest.approx(expected, abs=5e-5)
    assert head > tail  # fails if the two virtual residues are swapped


def test_the_stored_aggrescan_numbers_are_the_quantities_they_are_named_for():
    """Na4vSS is a4vSS normalised to 100 residues; a3vSA is the raw a3v mean."""
    found = _aggrescan(UBIQUITIN)
    assert found is not None
    assert found["method"] == "aggrescan_2007"
    assert found["na4vss"] == pytest.approx(found["a4vss"] / len(UBIQUITIN) * 100, abs=5e-4)
    raw = sum(float(_A3V[letter]) for letter in UBIQUITIN) / len(UBIQUITIN)
    assert found["a3vsa"] == pytest.approx(raw, abs=5e-5)
    assert found["hot_spot_threshold"] == pytest.approx(-0.02, abs=1e-12)
    # A hot spot is five or more consecutive residues over the threshold with
    # no proline among them.
    for span in found["hotspots"]:
        start, end = span["residue_start"], span["residue_end"]
        assert end - start + 1 >= 5
        assert "P" not in UBIQUITIN[start - 1 : end]


def test_the_molecular_weight_adds_one_water_to_the_whole_chain():
    """Ubiquitin is C378 H629 N105 O118 S1 and 8564.7366 Da on the IUPAC table."""
    assert _atom_counts(UBIQUITIN) == {"C": 378, "H": 629, "N": 105, "O": 118, "S": 1}
    assert _molecular_weight(UBIQUITIN) == pytest.approx(8564.7366, abs=5e-4)
    # One water for the chain, not one per residue: two glycines weigh two
    # free glycines less one water.
    glycine = _molecular_weight("G")
    assert glycine is not None
    water = 2 * 1.00794 + 15.9994
    assert _molecular_weight("GG") == pytest.approx(2 * glycine - water, abs=5e-4)


def test_the_n_end_rule_words_are_the_protparam_table():
    assert _HALF_LIFE == N_END_RULE_2005
    for letter, (mammalian, yeast, ecoli) in N_END_RULE_2005.items():
        found = _half_life(letter)
        assert found is not None
        assert found["half_life_mammalian"] == mammalian
        assert found["half_life_yeast"] == yeast
        # Proline has no published E. coli estimate, so no key is stored.
        assert found.get("half_life_ecoli") == ecoli
