"""The buttons on the results page, and what they are allowed to claim.

Three things these pin, and all three are about the page not overstating itself.

**Only the binder goes out.** That is a different question from the one the
refold requests ask, and deliberately: a refold carries every chain because the
interface criteria are about whether the *complex* forms, and a request that
omits the target silently asks something else. These buttons ask that something
else on purpose, and the sentences have to say which question was answered.

**Two labels the user asked for cannot be honoured as written.** Nothing can be
sent to the AlphaFold database -- its API is GET-only and keyed by accession,
and its only sequence-keyed route is an exact checksum match that a designed
binder misses by construction. Nothing can be sent to the AlphaFold Server
either, which has no submission API and issues no credential. A button that
said "send" would be lying about what it did.

**A retrieved model is compared, not assumed.** A model of the wrong protein
must produce a refusal rather than a number, and a model of the right protein
must be measured against the chain it actually corresponds to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfproteina.disease.results_actions import (
    NO_AF2_ENTRY,
    OUTPUT_FASTA,
    PREPARE_AF3,
    RETRIEVE_AF2,
    RETRIEVE_AF3,
    binder_sequences,
    overlay_for,
    perform,
)
from rfproteina.disease.results_retrieve import from_accession, from_text

_BINDER = "ACDEFGHIKLMNPQRSTVWY"
_TARGET = "WYVTSRQPNMLKIHGFEDCA"


def cif(chains: dict[str, str]) -> str:
    three = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
        "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
        "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
        "W": "TRP", "Y": "TYR",
    }
    head = (
        "data_x\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
    )
    rows: list[str] = []
    serial = 0
    for chain, sequence in chains.items():
        for index, letter in enumerate(sequence):
            serial += 1
            rows.append(
                f"ATOM {serial} CA {three[letter]} {chain} {index + 1} "
                f"{index * 3.8:.3f} {0.0:.3f} {0.0:.3f}"
            )
    body = head + "\n".join(rows) + "\n"
    # A proper loop_, because a FLAT _entity_poly holds exactly one row: two
    # flat blocks silently dropped the first chain, so this fixture had a
    # target and no binder, and every binder assertion failed for the wrong
    # reason. The real multi-chain files use a loop.
    poly = "loop_\n_entity_poly.pdbx_strand_id\n_entity_poly.pdbx_seq_one_letter_code\n"
    poly += "".join(f"{chain} {seq}\n" for chain, seq in chains.items())
    return body + poly


def one_design() -> tuple[list[dict[str, object]], dict[str, bytes]]:
    text = cif({"A": _BINDER, "C": _TARGET})
    store = {"c0": text.encode("utf-8")}
    item: dict[str, object] = {
        "sample": "binder_0",
        "complex_token": "c0",
        "binder_chains": ["A"],
        "target_chains": ["C"],
        "af2_token": "",
        "af2_detail": "",
        "af3_token": "",
        "af3_detail": "",
    }
    return [item], store


# ----------------------------------------------------------------- the binder


def test_the_binder_chain_is_read_and_not_the_first_chain() -> None:
    """Taking the first chain in the file would export the TARGET as the design
    on any structure whose target comes first, which is most of them."""
    catalog, store = one_design()
    found = binder_sequences(catalog, store.get)
    assert len(found) == 1
    assert found[0].strand == "A"
    assert found[0].sequence == _BINDER
    assert _TARGET not in found[0].fasta


def test_the_fasta_sentence_says_which_question_it_answered() -> None:
    catalog, store = one_design()
    said = perform(OUTPUT_FASTA, catalog, store.get).sentence
    assert "binder alone" in said
    assert "20 residues" in said


def test_a_design_with_no_chain_of_interest_is_named_not_guessed() -> None:
    catalog, store = one_design()
    catalog[0]["binder_chains"] = []
    found = binder_sequences(catalog, store.get)
    assert found[0].blocker == "no_chain_of_interest"
    assert "no_chain_of_interest" in perform(OUTPUT_FASTA, catalog, store.get).sentence


def test_an_unreadable_structure_does_not_end_the_run() -> None:
    catalog, _store = one_design()
    said = perform(OUTPUT_FASTA, catalog, lambda token: None).sentence
    assert "structure_unreadable" in said


# ------------------------------------------- the two labels that cannot stand


def test_nothing_is_sent_to_the_alphafold_database() -> None:
    """Its whole API is GET-only and keyed by accession.

    With nothing supplied and nothing found, the AlphaFold 2 pane reports the
    absence as the finding it is rather than as a retrieval that failed.
    """
    catalog, store = one_design()
    done = perform(RETRIEVE_AF2, catalog, store.get, [], [])
    assert NO_AF2_ENTRY in done.sentence
    assert done.overlays == (("af2", "", NO_AF2_ENTRY),)


def test_nothing_is_sent_to_the_alphafold_server() -> None:
    """No submission API, no credential."""
    catalog, store = one_design()
    said = perform(PREPARE_AF3, catalog, store.get).sentence
    assert "no submission API" in said
    assert "nothing was sent" in said
    assert "not whether the complex forms" in said


def test_an_accession_is_refused_for_the_server_which_is_not_keyed_by_one() -> None:
    catalog, store = one_design()
    said = perform(RETRIEVE_AF3, catalog, store.get, ["af3_accession=P05113"]).sentence
    assert "not keyed by accession" in said
    assert "archive" in said, "it has to say what to supply instead"


def test_an_unknown_action_is_not_an_error_the_run_notices() -> None:
    catalog, store = one_design()
    assert "not one this version knows" in perform("results-nope", catalog, store.get).sentence


# ------------------------------------------------------ retrieve and compare


def test_a_pasted_model_of_the_binder_compares_against_the_binder() -> None:
    catalog, store = one_design()
    done = perform(RETRIEVE_AF3, catalog, store.get, [f"af3_paste={cif({'A': _BINDER})}"])
    assert "RMSD 0.00" in done.sentence
    assert "20 of 20 residues" in done.sentence
    assert done.overlays and done.overlays[0][0] == "af3"
    assert done.overlays[0][1], "the overlay carries the model text"


def test_the_af2_overlay_is_about_the_binder_not_the_target() -> None:
    """Both overlays answer the same question, about the binder.

    An earlier version compared the database's model of the TARGET against the
    target chains and reported a confident RMSD. The number was real -- the
    database's model does sit that far from the deposited target -- but it was
    not an answer to the question the pane asks, which is about the binder. A
    true measurement under the wrong caption is still the page saying something
    it did not establish.
    """
    catalog, store = one_design()
    binder_model = overlay_for(
        catalog[0], from_text(cif({"A": _BINDER})), which="af2", structure_bytes=store.get
    )
    assert binder_model.found
    assert "RMSD 0.00" in binder_model.detail
    assert "20 of 20 residues" in binder_model.detail

    # The target's model is not this binder, however good a structure it is.
    target_model = overlay_for(
        catalog[0], from_text(cif({"A": _TARGET})), which="af2", structure_bytes=store.get
    )
    assert not target_model.found, "a model of the target must not fill the binder's pane"
    assert target_model.detail == NO_AF2_ENTRY


def test_a_designed_binder_reports_no_database_entry() -> None:
    """The expected answer for a de novo design, and a useful one.

    The database is keyed by accession and its only sequence-keyed route is an
    exact checksum match, so a designed sequence has no entry. Saying so is
    weak evidence the sequence is novel; an empty pane says nothing.
    """
    catalog, store = one_design()
    for retrieved in (
        from_text(cif({"A": _TARGET})),
        from_accession("DENOVO9", read_json=lambda url: [], read_bytes=lambda url: b""),
    ):
        made = overlay_for(catalog[0], retrieved, which="af2", structure_bytes=store.get)
        assert made.detail == NO_AF2_ENTRY
        assert not made.text, "nothing unrelated is drawn in the binder's pane"


def test_a_model_of_a_different_protein_refuses_rather_than_reporting_a_number() -> None:
    catalog, store = one_design()
    done = perform(RETRIEVE_AF3, catalog, store.get, ["af3_paste=" + cif({"A": "WWWWWWWWWWWWWWWW"})])
    assert "nothing to superpose" in done.sentence
    assert "different protein" in done.sentence
    assert done.overlays and done.overlays[0][1] == "", "no model is drawn for a refusal"


def test_the_caption_carries_the_provenance_with_the_number() -> None:
    """An RMSD without its source is a number nobody can check."""
    catalog, store = one_design()
    done = perform(RETRIEVE_AF3, catalog, store.get, [f"af3_paste={cif({'A': _BINDER})}"])
    assert "Not verified" in done.sentence
    assert "Pasted model" in done.sentence


def test_pasting_wins_over_a_path_because_the_person_already_retrieved_it() -> None:
    catalog, store = one_design()
    done = perform(
        RETRIEVE_AF3,
        catalog,
        store.get,
        [f"af3_paste={cif({'A': _BINDER})}", "af3_path=/nonexistent/model.cif"],
    )
    assert "RMSD 0.00" in done.sentence, "the paste was used, not the missing path"


def test_a_file_route_reports_a_missing_file_by_name(tmp_path: Path) -> None:
    catalog, store = one_design()
    done = perform(RETRIEVE_AF3, catalog, store.get, [f"af3_path={tmp_path / 'absent.cif'}"])
    assert "No file at that path" in done.sentence


def test_retrieving_with_no_structure_to_compare_says_so() -> None:
    done = perform(RETRIEVE_AF3, [], lambda token: None, [f"af3_paste={cif({'A': _BINDER})}"])
    assert "no structure to compare it against" in done.sentence


def test_a_run_naming_no_binder_still_shows_the_model() -> None:
    """The overlay is still worth drawing; only the number is unavailable."""
    catalog, store = one_design()
    catalog[0]["binder_chains"] = []
    made = overlay_for(
        catalog[0], from_text(cif({"A": _BINDER})), which="af3", structure_bytes=store.get
    )
    assert made.found, "the model is still drawn"
    assert "no chain to compare it against" in made.detail


@pytest.mark.parametrize("action", [OUTPUT_FASTA, PREPARE_AF3, RETRIEVE_AF2, RETRIEVE_AF3])
def test_every_action_returns_a_sentence_and_never_raises(action: str) -> None:
    """A run that has already finished must not be ended by a button."""
    catalog, store = one_design()
    done = perform(action, catalog, store.get)
    assert isinstance(done.sentence, str) and done.sentence


# ----------------------------------- the download fallback belongs to AF3 only


def test_an_empty_af2_retrieval_never_searches_the_downloads_folder(tmp_path: Path) -> None:
    """The AlphaFold 2 button has no download to fall back to.

    Its models arrive from the database by accession, never as a file that
    lands in a downloads folder. Searching there anyway handed the newest
    AlphaFold 3 archive to the AlphaFold 2 overlay, which would have captioned
    a fold of the binder as a database model of the target -- a wrong label on
    a real structure, which is worse than an empty pane.
    """
    catalog, store = one_design()
    archive = tmp_path / "fold_something.zip"
    import zipfile

    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fold_model_0.cif", cif({"A": _BINDER}))

    done = perform(RETRIEVE_AF2, catalog, store.get, [], [tmp_path])
    assert NO_AF2_ENTRY in done.sentence
    assert done.overlays and done.overlays[0][1] == "", "no model is drawn"


def test_an_empty_af3_retrieval_picks_up_the_archive_that_landed(tmp_path: Path) -> None:
    """The step that closes the loop: after the download click, no click.

    The two steps a person cannot avoid are the upload and the download; this
    is what happens after the second one, with nothing typed and no path
    given.
    """
    import zipfile

    catalog, store = one_design()
    archive = tmp_path / "fold_binder_0.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("terms_of_use.md", "terms")
        bundle.writestr("fold_model_0.cif", cif({"A": _BINDER}))

    done = perform(RETRIEVE_AF3, catalog, store.get, [], [tmp_path])
    assert "RMSD 0.00" in done.sentence
    assert "fold_binder_0.zip" in done.sentence, "the sentence names the file it found"
    assert done.overlays and done.overlays[0][0] == "af3"


def test_an_empty_af3_retrieval_with_no_download_says_so(tmp_path: Path) -> None:
    catalog, store = one_design()
    done = perform(RETRIEVE_AF3, catalog, store.get, [], [tmp_path])
    assert "no recent AlphaFold download found" in done.sentence
    assert not done.overlays
