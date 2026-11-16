"""The atomic-weight table is the vintage the deposited files use, not today's.

Seven entries differ from IUPAC's current table, and that is deliberate: this
module computes a formula weight so it can be compared with the one deposited
beside it, and the deposited arithmetic is the older one. A deposited ZINC ION
states 65.409, which is the 2005 value; IUPAC has said 65.38 since 2007.

Both halves are pinned here, because the obvious tidying -- bringing seven
stale-looking constants up to date -- is what breaks it, and nothing else in
the suite would notice.
"""

from decimal import Decimal

import pytest

from rfproteina.disease.modality import _ATOMIC_WEIGHT, _mass_from_formula

#: The seven entries where this table and IUPAC's current one disagree, as
#: ``symbol: (what this table holds, IUPAC 2024, ciaaw.org/atomic-weights.htm)``.
#: Zinc changed in 2007, germanium in 2009, mercury in 2011, and cadmium,
#: molybdenum, selenium and ytterbium in 2013.
VINTAGE = {
    "Zn": ("65.409", "65.38"),
    "Ge": ("72.64", "72.630"),
    "Se": ("78.96", "78.971"),
    "Mo": ("95.94", "95.95"),
    "Cd": ("112.411", "112.414"),
    "Hg": ("200.59", "200.592"),
    "Yb": ("173.04", "173.045"),
}
#: Components as deposited, with the formula weight that sits beside them:
#: ``(formula, deposited weight)``. The first five are from the files in this
#: checkout; the rest are the Chemical Component Dictionary entries for the
#: elements whose weight changed, which is where a mistaken update would show.
DEPOSITED = {
    "ALA": ("C3 H7 N O2", "89.093"),
    "ARG": ("C6 H15 N4 O2 1", "175.209"),
    "ASN": ("C4 H8 N2 O3", "132.118"),
    "ATP": ("C10 H16 N5 O13 P3", "507.181"),
    "CA": ("Ca 2", "40.078"),
    "HOH": ("H2 O", "18.015"),
    "ZN": ("Zn 2", "65.409"),
    "SE": ("Se", "78.96"),
    "HG": ("Hg", "200.59"),
    "MO": ("Mo", "95.94"),
    "CD": ("Cd", "112.411"),
}


@pytest.mark.parametrize(("component", "formula", "deposited"), [(k, v[0], v[1]) for k, v in DEPOSITED.items()])
def test_the_computed_weight_equals_the_deposited_one(component: str, formula: str, deposited: str) -> None:
    """Comparing a computed weight with a deposited one only means something if they agree."""
    computed = _mass_from_formula(formula)
    assert computed is not None, component
    assert Decimal(str(computed)) == Decimal(deposited), component


def test_the_seven_vintage_weights_are_left_as_the_deposited_files_have_them() -> None:
    """Updating these to IUPAC's current values is the tidying this table must not get.

    Each one is the weight the Chemical Component Dictionary used, so changing
    it makes every computed weight for that element disagree with the file it
    is being checked against. The current value is named here so that the next
    reader can see the difference is known rather than overlooked.
    """
    for symbol, (held, current) in VINTAGE.items():
        assert _ATOMIC_WEIGHT[symbol] == held, (
            f"{symbol} is {_ATOMIC_WEIGHT[symbol]} and this table wants {held}, the weight the deposited "
            f"files were computed with. IUPAC's current value is {current}; using it here would make a "
            f"computed weight disagree with the deposited one."
        )
        assert held != current


def test_a_protein_weight_still_rests_on_the_light_elements() -> None:
    """None of the seven is C, H, N, O, P or S, so a chain's weight is untouched by them."""
    assert not {"C", "H", "N", "O", "P", "S"} & set(VINTAGE)
