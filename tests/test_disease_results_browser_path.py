"""The results page as a browser actually drives it.

Every test here failed before the fixes they pin, and the whole suite passed
anyway, because each one is about the join between the form and the action
rather than about either side on its own. Driving the real page in a browser
found all three in one click:

* the typed path never reached the action, so paste, file and accession were
  all silently ignored and only the no-input fallback ran;
* the tab the reader had open was not kept, so a combined click answered and
  then showed the binder tab;
* the outcome sentence was written inside the binder panel, so a combined
  click reported onto the tab the reader was not looking at.

Together they meant a person on the combined tab could type a path, press the
button, and see nothing happen at all.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rfproteina.disease.results_actions import RETRIEVE_AF3C
from rfproteina.disease.ui import BrowserSession, OptionPanel, _results_tools, flat_fields

_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}
TARGET = "ACDEFGHIKLMNPQRSTVWY"
BINDER = "WYVTSRQPNMLKIHGFEDCA"


def cif(chains: dict[str, tuple[str, tuple[float, float, float]]]) -> str:
    head = (
        "data_x\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
    )
    rows: list[str] = []
    serial = 0
    for chain, (sequence, offset) in chains.items():
        for index, letter in enumerate(sequence):
            serial += 1
            x = 2.3 * math.cos(index * 1.7) + offset[0]
            y = 2.3 * math.sin(index * 1.7) + offset[1]
            z = 1.5 * index + offset[2]
            rows.append(
                f"ATOM {serial} CA {_THREE[letter]} {chain} {index + 1} "
                f"{x:.3f} {y:.3f} {z:.3f}"
            )
    return head + "\n".join(rows) + "\n"


def complex_cif(binder_shift: float = 0.0) -> str:
    return cif({"C": (TARGET, (0, 0, 0)), "A": (BINDER, (14 + binder_shift, 0, 0))})


def _session_with_a_design() -> BrowserSession:
    session = BrowserSession(opener=lambda _url: None)
    token = session.token_for_text(complex_cif())
    assert token
    session._catalog = [
        {
            "sample": "binder_0",
            "title": "binder_0",
            "disease_token": token,
            "disease_chains": ["C"],
            "complex_token": token,
            "target_chains": ["C"],
            "binder_chains": ["A"],
            "af2_token": "", "af2_detail": "",
            "af3_token": "", "af3_detail": "",
            "af2c_token": "", "af2c_detail": "",
            "af3c_token": "", "af3c_detail": "",
        }
    ]
    return session


# ------------------------------------------------- the form reaching the action


def test_the_submission_becomes_the_flat_list_actions_read() -> None:
    """A mapping iterates its keys, and a bare key matches no ``name=`` prefix."""
    made = flat_fields({"af3c_path": ["/tmp/a.zip"], "__action": ["results-af3c"]})
    assert "af3c_path=/tmp/a.zip" in made
    assert "__action=results-af3c" in made


def test_a_field_with_several_values_keeps_all_of_them() -> None:
    made = flat_fields({"pick": ["one", "two"]})
    assert made == ["pick=one", "pick=two"]


def test_a_pasted_model_survives_its_own_newlines() -> None:
    """The paste box holds a whole mmCIF, which is the one value that is not a word."""
    made = flat_fields({"af3c_paste": [complex_cif()]})
    assert len(made) == 1
    assert made[0].startswith("af3c_paste=data_x\n")


def test_a_typed_path_reaches_the_action_and_is_compared(tmp_path: Path) -> None:
    """The defect a browser found: the path was read off the page and dropped.

    What reached ``perform`` was the field *names*, so no route matched and the
    page fell through to scanning the downloads folder -- then told the reader
    no download could be found, while they were looking at the path they had
    just typed into the box.
    """
    model = tmp_path / "fold_combined.cif"
    model.write_text(complex_cif(binder_shift=9.0), encoding="utf-8")

    session = _session_with_a_design()
    given = [
        {
            "__action": [RETRIEVE_AF3C],
            "af3c_path": [str(model)],
            "results-tab": ["combined"],
        },
        {},
    ]
    panel = OptionPanel(lambda _page: given.pop(0))
    panel._session = session

    assert panel.finish_choice() == "close"

    said = session._action_result
    assert "no recent AlphaFold download found" not in said, "the typed path was ignored"
    assert "Whole complex" in said
    assert "9.00 A off" in said, "and the comparison is of the file that was named"
    assert session._catalog[0]["af3c_token"], "so the overlay has a model to draw"


def test_the_tab_the_reader_had_open_is_the_one_that_comes_back(tmp_path: Path) -> None:
    model = tmp_path / "fold_combined.cif"
    model.write_text(complex_cif(), encoding="utf-8")
    session = _session_with_a_design()
    given = [
        {"__action": [RETRIEVE_AF3C], "af3c_path": [str(model)], "results-tab": ["combined"]},
        {},
    ]
    panel = OptionPanel(lambda _page: given.pop(0))
    panel._session = session
    panel.finish_choice()

    assert session._results_tab == "combined"
    page = session.done_page()
    assert 'id="results-tab-combined" value="combined" checked' in page
    assert 'id="results-tab-binder" value="binder">' in page


# ------------------------------------------------------ the outcome being seen


def test_the_outcome_is_not_written_inside_one_tab() -> None:
    """Shown inside the binder panel it was hidden whenever combined was open.

    The CSS shows one panel at a time, so a combined button reported its
    result onto the tab the reader was not looking at. From their side the
    click did nothing: no sentence, no overlay, no error.
    """
    page = _results_tools("Whole complex C-alpha RMSD 1.00 A.", active="combined")
    readout = page.index("screen-readout")
    assert readout < page.index('class="results-panel results-panel-binder"')
    assert readout < page.index('class="results-panel results-panel-combined"')


def test_the_combined_tab_is_the_one_marked_open_when_it_is_active() -> None:
    page = _results_tools("", active="combined")
    assert 'id="results-tab-combined" value="combined" checked' in page
    assert 'id="results-tab-binder" value="binder">' in page


def test_the_binder_tab_is_the_default() -> None:
    page = _results_tools("")
    assert 'id="results-tab-binder" value="binder" checked' in page
    assert 'id="results-tab-combined" value="combined">' in page


def test_every_results_button_is_on_the_page_once() -> None:
    """All eight, so neither tab loses a control when the markup is rearranged."""
    page = _results_tools("")
    for action in (
        "results-fasta",
        "results-af3",
        "results-af2",
        "results-af3-import",
        "results-combined-fasta",
        "results-af3c-prepare",
        "results-af2c",
        "results-af3c",
    ):
        assert page.count(f'value="{action}"') == 1, action


@pytest.mark.parametrize("active", ["binder", "combined"])
def test_the_handover_block_is_reachable_from_either_tab(active: str) -> None:
    """A combined FASTA is copied or saved from the same block as a binder one."""
    page = _results_tools("", payload=">x\nACDEF\n", name="combined.fasta", token="tok", active=active)
    where = page.index("combined.fasta")
    assert where < page.index('class="results-panel results-panel-binder"')
