"""The three axes that were declared in a bar and measured by nothing.

``AXES_BY_KIND`` has given a peptide ``membrane`` and a nucleic binder
``off_target_hybridization`` and ``innate_immune`` since the classes were
defined, and ``BARS`` requires all three. None of them was ever measured, so a
peptide could not clear its own ``sequence`` bar however hydrophilic it was, and
a nucleic design came back ``untested`` with no blocker and no file anyone could
supply to change it.

The membrane scan was already computed for the drug-type report and thrown
away. The two nucleic axes are new: exact 15-base seeds against a supplied
reference, and the TLR9 CpG consensus. A CpG-free oligo is deliberately not a
clear -- it cannot present the motif, but TLR9 is not the only innate sensor an
oligo meets, so that clear is curated, exactly as paralog's is.
"""

import random
from pathlib import Path

import pytest

from rfproteina.disease.combinations import BARS, PEPTIDE_AXES
from rfproteina.disease.modality import membrane_axis
from rfproteina.disease.oligo import (
    SEED_WIDTH,
    TRANSCRIPTOME_MIN_BASES,
    innate_immune_record,
)
from rfproteina.disease.screen import screen_generation
from tests.helpers.disease import UBIQUITIN, binder_cif, closing_page, no_developer_files  # noqa: F401

_HYDROPHILIC = "DEKRDEKRDEKRDEKRDEKR"
_HYDROPHOBIC = "IVLFIVLFIVLFIVLFIVLF"
_UBIQUITIN = UBIQUITIN
_OLIGO = "ATGCATGCATGCATGCATGCATGCATGCATGC"


@pytest.fixture(scope="module")
def reference() -> str:
    """A nucleotide FASTA just large enough to support a clear."""
    random.seed(11)
    letters = "".join(random.choice("ACGT") for _ in range(TRANSCRIPTOME_MIN_BASES))
    wrapped = "\n".join(letters[index : index + 60] for index in range(0, len(letters), 60))
    return f">filler\n{wrapped}\n"


def _run(
    tmp_path: Path,
    name: str,
    kind: str,
    designs: dict[str, str],
    *,
    transcriptome: str | None = None,
    evidence: str | None = None,
) -> dict:
    output = tmp_path / name
    for design, sequence in designs.items():
        folder = output / "inference_samples" / design
        folder.mkdir(parents=True)
        (folder / f"{design}_generated.cif").write_text(binder_cif(sequence), encoding="utf-8")
    table = output / "inference_metrics" / "per_example_metrics.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{design},A\n" for design in designs)
    table.write_text("example_id,chain_of_interest\n" + rows, encoding="utf-8")
    if transcriptome is not None:
        (output / "transcriptome.fa").write_text(transcriptome, encoding="utf-8")
    if evidence is not None:
        (output / "axis_evidence.json").write_text(evidence, encoding="utf-8")
    return screen_generation(output, binder_kind=kind, target_id="P05113")


def test_the_bars_these_axes_belong_to_have_always_required_them() -> None:
    """Which is why a screen that never measured them could not reach either bar."""
    assert "membrane" in BARS["peptide"]["sequence"]
    assert BARS["peptide"]["sequence"] == PEPTIDE_AXES
    assert BARS["nucleic"]["oligo"] == ("off_target_hybridization", "innate_immune")


class TestMembrane:
    def test_a_hydrophilic_peptide_clears_from_a_measurement(self, tmp_path: Path) -> None:
        record = _run(tmp_path, "clear", "peptide", {"binder_0": _HYDROPHILIC})
        membrane = record["sequence_liability"]["membrane"]
        assert membrane["state"] == "clear"
        assert membrane["source"] == "measured"
        assert "membrane" not in record["untested"]

    def test_a_span_is_a_hit_and_the_run_is_axis_hit(self, tmp_path: Path) -> None:
        record = _run(tmp_path, "hit", "peptide", {"binder_0": _HYDROPHOBIC})
        assert record["sequence_liability"]["membrane"]["state"] == "hit"
        assert record["sequence_liability"]["membrane"]["measurement"] == 1
        assert record["call"] == "axis_hit"
        assert "membrane" in record["hits"]

    def test_a_chain_too_short_to_hold_a_window_is_refused_not_cleared(self, tmp_path: Path) -> None:
        """A 19-residue criterion says nothing about an 8-mer, which is not the same as clearing it."""
        record = _run(tmp_path, "short", "peptide", {"binder_0": "DEKRDEKR"})
        membrane = record["sequence_liability"]["membrane"]
        assert membrane["state"] == "untested"
        assert membrane["blocker"] == "chain_shorter_than_window"

    def test_one_design_with_a_span_makes_the_run_a_hit_and_is_named(self, tmp_path: Path) -> None:
        record = _run(tmp_path, "mixed", "peptide", {"binder_0": _HYDROPHILIC, "binder_1": _HYDROPHOBIC})
        membrane = record["sequence_liability"]["membrane"]
        assert membrane["state"] == "hit"
        assert membrane["failed"] == ["binder_1"]

    def test_a_protein_binder_has_no_membrane_axis(self, tmp_path: Path) -> None:
        """The judge refuses an axis foreign to the kind, and the bar does not name it."""
        record = _run(tmp_path, "protein", "protein", {"binder_0": _UBIQUITIN})
        assert "membrane" not in record["sequence_liability"]
        assert "membrane" not in record["states"]

    def test_a_letter_with_no_hydropathy_says_so(self) -> None:
        assert membrane_axis("DEKRDEKRDEKRDEKRDEKX")["blocker"] == "hydropathy_unknown"

    def test_the_closed_page_gives_the_axis_a_tab(self, tmp_path: Path) -> None:
        page = closing_page(_run(tmp_path, "hit", "peptide", {"binder_0": _HYDROPHOBIC}))
        assert ">Membrane<" in page
        assert "Kyte-Doolittle 1982 mean hydropathy over a 19-residue window." in page


class TestInnateImmune:
    def test_the_consensus_hexamer_is_a_hit(self, tmp_path: Path) -> None:
        """AACGTT is purine-purine-CG-pyrimidine-pyrimidine."""
        record = _run(tmp_path, "hit", "nucleic", {"binder_0": "TTTTAACGTTTTTT"})
        innate = record["nucleic_liability"]["innate_immune"]
        assert innate["state"] == "hit"
        assert innate["measurement"] == 1
        # The one-based start of the hexamer, not of the CG inside it.
        assert innate["motif_positions"] == [5]
        assert record["call"] == "axis_hit"

    def test_a_cpg_outside_the_consensus_keeps_its_count_and_decides_nothing(self, tmp_path: Path) -> None:
        """TCCGAA is pyrimidine-pyrimidine-CG-purine-purine, the consensus reversed."""
        record = _run(tmp_path, "offmotif", "nucleic", {"binder_0": "TTTTTCCGAATTTT"})
        innate = record["nucleic_liability"]["innate_immune"]
        assert innate["state"] == "untested"
        # Not ``no_pass_cut``: a cut does apply on this axis -- one consensus
        # hexamer is a hit -- so saying none applies denied what the panel's own
        # High line asserts. The measured zero is refused for the reason
        # ``cpg_free_not_evidence`` gives for its own zero, and says so.
        assert innate["blocker"] == "cpg_not_consensus"
        assert innate["cpg_count"] == 1

    def test_a_cpg_free_design_is_refused_rather_than_cleared(self, tmp_path: Path) -> None:
        """It cannot present the motif. TLR9 is not the only innate sensor."""
        record = _run(tmp_path, "free", "nucleic", {"binder_0": "ATATATATATATATAT"})
        innate = record["nucleic_liability"]["innate_immune"]
        assert innate["state"] == "untested"
        assert innate["blocker"] == "cpg_free_not_evidence"
        assert innate["cpg_count"] == 0

    def test_rna_is_read_as_its_dna_spelling(self) -> None:
        assert innate_immune_record("UUUUAACGUUUUUU")["state"] == "hit"

    def test_a_chain_that_is_not_an_oligo_says_so(self) -> None:
        assert innate_immune_record(_UBIQUITIN)["blocker"] == "sequence_not_nucleic"
        assert innate_immune_record(None)["blocker"] == "sequence_absent"

    def test_the_method_travels_with_the_record(self) -> None:
        assert "Krieg 1995" in innate_immune_record("ATATATAT")["method"]


class TestOffTargetHybridization:
    def test_a_run_with_no_reference_says_what_is_missing(self, tmp_path: Path) -> None:
        record = _run(tmp_path, "none", "nucleic", {"binder_0": _OLIGO})
        found = record["nucleic_liability"]["off_target_hybridization"]
        assert found["state"] == "untested"
        assert found["blocker"] == "transcriptome_not_configured"

    def test_a_design_absent_from_the_reference_clears_against_that_file(self, tmp_path: Path, reference: str) -> None:
        record = _run(tmp_path, "clear", "nucleic", {"binder_0": _OLIGO}, transcriptome=reference)
        found = record["nucleic_liability"]["off_target_hybridization"]
        assert found["state"] == "clear"
        assert found["measurement"] == 0
        # A clear is a statement about one file, so the file travels with it.
        assert found["seed_width"] == SEED_WIDTH
        assert found["reference_bases"] == TRANSCRIPTOME_MIN_BASES
        assert len(found["reference_digest"]) == 16
        assert found["reference_file"] == "transcriptome.fa"

    def test_a_planted_match_is_a_hit(self, tmp_path: Path, reference: str) -> None:
        text = reference + f">planted\n{_OLIGO}\n"
        record = _run(tmp_path, "hit", "nucleic", {"binder_0": _OLIGO}, transcriptome=text)
        found = record["nucleic_liability"]["off_target_hybridization"]
        assert found["state"] == "hit"
        assert found["measurement"] > 0

    def test_a_match_on_the_other_strand_is_a_hit_too(self, tmp_path: Path, reference: str) -> None:
        """An antisense oligo hybridizes where the transcript carries its reverse complement."""
        revcomp = "".join({"A": "T", "C": "G", "G": "C", "T": "A"}[letter] for letter in reversed(_OLIGO))
        text = reference + f">antisense\n{revcomp}\n"
        record = _run(tmp_path, "anti", "nucleic", {"binder_0": _OLIGO}, transcriptome=text)
        assert record["nucleic_liability"]["off_target_hybridization"]["state"] == "hit"

    def test_a_match_straddling_a_read_block_is_still_found(self, tmp_path: Path, reference: str) -> None:
        """The scan is blocked, with a seed's worth of overlap carried between blocks."""
        text = reference + ">planted\n" + _OLIGO[:20] + "\n" + _OLIGO[20:] + "\n"
        record = _run(tmp_path, "straddle", "nucleic", {"binder_0": _OLIGO}, transcriptome=text)
        assert record["nucleic_liability"]["off_target_hybridization"]["state"] == "hit"

    def test_a_reference_too_small_cannot_support_a_clear(self, tmp_path: Path) -> None:
        record = _run(tmp_path, "tiny", "nucleic", {"binder_0": _OLIGO}, transcriptome=">one\nACGTACGTACGTACGTACGT\n")
        found = record["nucleic_liability"]["off_target_hybridization"]
        assert found["state"] == "untested"
        assert found["blocker"] == "transcriptome_too_small"

    def test_a_design_shorter_than_one_seed_is_not_screened(self, tmp_path: Path, reference: str) -> None:
        record = _run(tmp_path, "stub", "nucleic", {"binder_0": "ACGTACGT"}, transcriptome=reference)
        found = record["nucleic_liability"]["off_target_hybridization"]
        assert found["blocker"] == "chain_shorter_than_seed"

    def test_the_closed_page_gives_both_oligo_axes_a_tab(self, tmp_path: Path) -> None:
        page = closing_page(_run(tmp_path, "none", "nucleic", {"binder_0": _OLIGO}))
        assert ">Off-target hybridization<" in page
        assert ">Innate immune<" in page
        assert "Add transcriptome.fa to the output folder, or set HUMAN_TRANSCRIPTOME." in page


def test_the_oligo_bar_can_now_be_reached(tmp_path: Path, reference: str) -> None:
    """One axis clears from a measurement; the other is curated, as paralog's is."""
    record = _run(
        tmp_path,
        "curated",
        "nucleic",
        {"binder_0": _OLIGO},
        transcriptome=reference,
        evidence='{"innate_immune": "clear"}',
    )
    assert record["bar"] == "oligo"
    assert record["nucleic_liability"]["off_target_hybridization"]["source"] == "measured"
    assert record["nucleic_liability"]["innate_immune"]["source"] == "axis_evidence"
    assert record["call"] == "screened_clear"


def test_a_nucleic_design_without_the_file_still_cannot_clear(tmp_path: Path) -> None:
    """Which is the point: the bar is reachable, not given away."""
    record = _run(tmp_path, "unreachable", "nucleic", {"binder_0": _OLIGO}, evidence='{"innate_immune": "clear"}')
    assert record["call"] == "untested"
    assert "off_target_hybridization" in record["untested"]
