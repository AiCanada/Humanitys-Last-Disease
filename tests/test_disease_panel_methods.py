"""Published-method checks on the sequence and safety-panel halves of the screen.

Each test here pins a rule to the source that states it. GRAVY and the
instability index are ExPASy ProtParam definitions over the twenty amino
acids, so a DNA chain spelled A, C, G and T has neither; the stability cut is
"smaller than 40", so exactly 40 is not stable; and a panel column that names
no unit is not a nanomolar reading just because the cut is in nM.

Nothing here reaches the network: every input is a table or a CIF written into
``tmp_path``.
"""

from pathlib import Path

from rfproteina.disease.combinations import LIGAND_AXES, ligand_states

# PANEL_NM, PANEL_NM_SOURCE and panel_cut are imported inside the tests that
# need them, not here. They were added by the fix these tests pin, so a
# module-scope import makes the whole file fail to collect against a tree
# that predates it -- and a collection error cannot tell a behaviour that
# changed from a name that did not exist yet, which is the one thing the
# fails-before check has to establish.
from rfproteina.disease.screen import screen_generation
from rfproteina.disease.toxicology import measure_toxicology
from tests.helpers.disease import (
    UBIQUITIN,
    no_developer_proteome,  # noqa: F401
    screen_page,
)

_UBIQUITIN = UBIQUITIN


def _run(root: Path, sequence: str, *, kind: str = "protein") -> dict[str, object]:
    """A finished one-design run, screened as ``kind``."""
    run = root / "gene_1ABC_1"
    sample = run / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text(
        "data_design\n#\nloop_\n_entity_poly.entity_id\n"
        "_entity_poly.pdbx_strand_id\n_entity_poly.pdbx_seq_one_letter_code_can\n"
        f"1 A {sequence}\n#\n",
        encoding="utf-8",
    )
    metrics = run / "inference_metrics"
    metrics.mkdir(parents=True)
    (metrics / "per_example_metrics.csv").write_text(
        "example_id,chain_of_interest\nbinder_0,A\n",
        encoding="utf-8",
    )
    return screen_generation(run, binder_kind=kind, target_id="P05113")


def _sequence_of(record: dict[str, object]) -> dict[str, object]:
    block = record["sequence_developability"]
    assert isinstance(block, dict)
    return block


def _table(folder: Path, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "toxicology.tsv").write_text(text, encoding="utf-8")
    return folder


def test_a_dna_design_gets_no_gravy_or_instability(tmp_path: Path) -> None:
    """ExPASy ProtParam defines both metrics over amino acids, and DNA is not one.

    Read as a peptide, ACGT has Kyte-Doolittle values 1.8, 2.5, -0.4 and -0.7,
    so this 32-nucleotide oligo used to publish GRAVY (1.8+2.5-0.4-0.7)/4 = 0.8
    and II = (10/32)*314.6 = 98.3125, with a sequence_pass verdict and two
    board gauges drawn from them. An RNA chain carrying a U raised instead, so
    the same molecule class got a number or no number depending on one letter.
    """
    record = _run(tmp_path, "ACGT" * 8, kind="nucleic")
    sequence = _sequence_of(record)
    assert sequence["gravy"] is None
    assert sequence["instability_index"] is None
    assert sequence["sequence_pass"] == "absent"
    assert sequence["sequence_blocker"] == "not_a_polypeptide"
    # The oligonucleotide report is the measurement that does apply, and it stays.
    modality = record["modality_screen"]
    assert isinstance(modality, dict)
    assert modality["gc_fraction"] == 0.5
    page = screen_page(record)
    # No gauge for a measurement that no longer exists, and the page says why.
    assert "<strong>GRAVY</strong>" not in page
    assert "<strong>Instability</strong>" not in page
    assert "Design checks" not in page
    assert "defined over amino acids" in page
    assert "Oligonucleotide triage" in page


def test_a_protein_design_still_gets_both_metrics(tmp_path: Path) -> None:
    """The gate is on the kind, not on the letters: a protein is unaffected.

    Ubiquitin's published values, which the audit reproduced against
    Biopython: GRAVY -0.489474 and instability index 36.055263.
    """
    record = _run(tmp_path, _UBIQUITIN, kind="protein")
    sequence = _sequence_of(record)
    assert sequence["gravy"] == -0.4895
    assert sequence["instability_index"] == 36.0553
    assert sequence["sequence_pass"] is True
    assert "sequence_blocker" not in sequence
    page = screen_page(record)
    assert "<strong>GRAVY</strong>" in page
    assert "<strong>Instability</strong>" in page


def test_an_instability_index_of_exactly_forty_does_not_pass(tmp_path: Path, monkeypatch) -> None:
    """ExPASy ProtParam: a protein whose instability index is "smaller than 40" is stable.

    40.0 itself is not, and ``sequence_pass`` used to call it stable. The
    value is forced here because the index is 10/L times a sum of table
    values, and landing on 40.0000 with a real chain is a coincidence.
    """
    import rfproteina.metrics.physics.developability_metrics as dm

    monkeypatch.setattr(dm, "compute_gravy", lambda sequence: 0.0)
    monkeypatch.setattr(dm, "compute_instability_index", lambda sequence: 40.0)
    record = _run(tmp_path, _UBIQUITIN, kind="protein")
    sequence = _sequence_of(record)
    assert sequence["gravy"] == 0.0
    assert sequence["instability_index"] == 40.0
    assert sequence["sequence_pass"] is False
    page = screen_page(record)
    assert "instability 40 is not below 40" in page
    assert "Below 40 passes this half" in page
    assert "At or below 40 passes this half" not in page


def test_an_instability_index_just_under_forty_still_passes(tmp_path: Path, monkeypatch) -> None:
    """Only the equality moved. Everything below 40 is still stable."""
    import rfproteina.metrics.physics.developability_metrics as dm

    monkeypatch.setattr(dm, "compute_gravy", lambda sequence: 0.0)
    monkeypatch.setattr(dm, "compute_instability_index", lambda sequence: 39.9999)
    record = _run(tmp_path, _UBIQUITIN, kind="protein")
    assert _sequence_of(record)["sequence_pass"] is True
    assert "instability is below 40." in screen_page(record)


def test_a_one_residue_chain_keeps_its_gravy_and_names_the_real_reason(tmp_path: Path) -> None:
    """Kyte-Doolittle Met = 1.9, so GRAVY("M") = 1.9 and nothing failed.

    The instability index sums over consecutive dipeptides and has none on a
    chain of one, which ``compute_instability_index`` reports as ``nan`` by
    contract. The whole record used to be discarded as ``metrics_failed``,
    the word for an import that raised or a nonstandard letter, which sent the
    reader to debug a metric that had not failed.
    """
    sequence = _sequence_of(_run(tmp_path, "M", kind="protein"))
    assert sequence["gravy"] == 1.9
    assert sequence["instability_index"] is None
    assert sequence["sequence_pass"] == "absent"
    assert sequence["sequence_blocker"] == "chain_shorter_than_dipeptide"


def test_a_nonstandard_letter_is_still_metrics_failed(tmp_path: Path) -> None:
    """The old blocker keeps its own meaning: a metric that really did refuse."""
    sequence = _sequence_of(_run(tmp_path, "MQIFVKTLTX", kind="protein"))
    assert sequence["gravy"] is None
    assert sequence["instability_index"] is None
    assert sequence["sequence_blocker"] == "metrics_failed"


def test_a_bare_panel_column_without_a_unit_is_not_assumed_nanomolar(tmp_path: Path) -> None:
    """50 uM hERG is a clean compound: 50000 nM, well above the 10 uM cut.

    ``herg`` names the protein and no unit, so nothing in the table says its
    numbers are nM. Read as nM it became a hit of 50.0 and the page told the
    reader a clean compound was a hERG liability. The module's own rule is
    that a reading whose unit is not nM is never rescaled onto the cut.
    """
    herg = measure_toxicology(_table(tmp_path / "bare", "herg\n50\n"))["herg"]
    assert herg["state"] == "untested"
    assert herg["blocker"] == "reading_not_nm"


def test_a_bare_panel_column_with_a_nanomolar_unit_is_read(tmp_path: Path) -> None:
    """A unit cell saying nM is all the column was missing."""
    herg = measure_toxicology(_table(tmp_path / "declared", "herg\tunit\n50\tnM\n"))["herg"]
    assert herg == {
        "state": "hit",
        "measurement": 50.0,
        "blocker": None,
        "method": "ligand_panel_nm",
        "units": "nM",
        "threshold": 10_000.0,
        "threshold_source": "program_convention_10uM",
        "n_readings": 1,
    }


def test_a_column_that_names_nanomolar_itself_needs_no_unit_cell(tmp_path: Path) -> None:
    """``herg_ic50_nm`` states its unit in its own name, so it is read as before."""
    found = measure_toxicology(_table(tmp_path / "suffix", "herg_ic50_nm\tbcrp_nm\n50000\t50\n"))
    assert found["herg"]["state"] == "clear"
    assert found["herg"]["measurement"] == 50_000.0
    assert found["bcrp"]["state"] == "hit"
    assert found["bcrp"]["measurement"] == 50.0


def test_every_panel_axis_records_the_cut_it_was_judged_on(tmp_path: Path) -> None:
    """One number for ten proteins said nothing about which axis it applied to.

    Every entry of ``PANEL_NM`` is still 10 uM, so no stored verdict moves;
    what the record now carries is the cut that was applied to this axis and
    where that cut came from.
    """
    from rfproteina.disease.combinations import PANEL_NM, PANEL_NM_SOURCE

    rows = "axis\tnm\n" + "".join(f"{axis}\t15000\n" for axis in LIGAND_AXES)
    found = measure_toxicology(_table(tmp_path / "panel", rows))
    assert set(found) == set(LIGAND_AXES)
    for axis, record in found.items():
        assert record["threshold"] == PANEL_NM[axis]
        assert record["threshold_source"] == PANEL_NM_SOURCE[axis]
        assert record["state"] == ("clear" if PANEL_NM[axis] < 15000.0 else "hit")


def test_the_panel_cut_table_is_a_convention_and_says_so() -> None:
    """No axis claims a publication it does not have, and no number has moved."""
    from rfproteina.disease.combinations import PANEL_NM, PANEL_NM_SOURCE, panel_cut

    assert set(PANEL_NM) == set(LIGAND_AXES)
    assert set(PANEL_NM_SOURCE) == set(LIGAND_AXES)
    for axis in LIGAND_AXES:
        assert PANEL_NM[axis] == 10_000.0
        assert panel_cut(axis) == 10_000.0
        assert PANEL_NM_SOURCE[axis] == "program_convention_10uM"


def test_the_judge_uses_each_axis_own_cut(tmp_path: Path, monkeypatch) -> None:
    """``ligand_states`` reads the per-axis table, not one panel-wide number."""
    from rfproteina.disease.combinations import PANEL_NM

    monkeypatch.setitem(PANEL_NM, "bsep", 25_000.0)
    assert ligand_states({"bsep": 15_000.0, "herg": 15_000.0}) == {"bsep": "hit", "herg": "clear"}
    bsep = measure_toxicology(_table(tmp_path / "bsep", "axis\tnm\nbsep\t15000\n"))["bsep"]
    assert bsep["state"] == "hit"
    assert bsep["threshold"] == 25_000.0
    # A bound is compared with the same cut, not with the panel-wide fallback.
    bound = measure_toxicology(_table(tmp_path / "bound", "axis\tnm\nbsep\t>15000\n"))["bsep"]
    assert bound["state"] == "untested"
    assert bound["blocker"] == "reading_not_decisive"


def test_the_screen_stores_the_whole_panel_cut_table(tmp_path: Path) -> None:
    """``cuts`` is what the verdict was taken against, per axis."""
    from rfproteina.disease.combinations import PANEL_NM

    output = _table(tmp_path / "ligand", "axis\tnm\nherg\t20000\npgp\t20000\nbcrp\t20000\n")
    record = screen_generation(output, binder_kind="ligand", target_id="P05113")
    cuts = record["cuts"]
    assert isinstance(cuts, dict)
    assert cuts["ligand_panel_nm_max"] == dict(PANEL_NM)
    toxicology = record["toxicology"]
    assert isinstance(toxicology, dict)
    assert toxicology["herg"]["threshold_source"] == "program_convention_10uM"
    page = screen_page(record)
    # The page reads the cut from the same table the verdict used, and says
    # what that number is rather than presenting it as a published bar.
    assert "At or below 10000 nM is a hit on this panel protein." in page
    assert "The cut is 10000 nM (10 uM), a program convention." in page
