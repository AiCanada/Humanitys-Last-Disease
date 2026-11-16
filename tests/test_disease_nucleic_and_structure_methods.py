"""The nucleic, ligand and structure-derived computations, against their sources.

Three claims this file pins, each of which the code or its own comments got
wrong before it existed:

* The optimal *human* hexamer GTCGTT (Hartmann et al., J Immunol 2000,
  164:1617) is not an instance of Krieg's RRCGYY -- its second base is a
  pyrimidine -- and the module docstring used to say it was. It is read now as
  a second published rule of its own, so ODN 2006 and the rest of the CpG-B
  class are hits by Hartmann rather than by Krieg; which rule fired is stored
  beside each position.
* A 15-base seed is not rare in a transcriptome. 4**15 is 1.07e9 against about
  3e8 bases, so one seed in four occurs by chance and a design offers dozens.
  The docstring used to call the width one at which "a perfect match stops
  being an accident".
* ``_interface_numbers`` returns two numbers with two different scopes:
  ``contact_ms`` is the target-side surface for the designed chain, and
  ``clash_fraction`` is a whole-complex fraction. It carried no docstring at
  all, and both are stored side by side under ``designed_interface``.

The rest is the ordinary agreement check: motif counting, unit handling, and
the sequence the screen reads out of a multi-chain CIF.
"""

import itertools
import re
from pathlib import Path

import pytest

from rfproteina.disease import oligo, screen
from rfproteina.disease.chemical import chemical_record
from rfproteina.disease.combinations import HIT_NM
from rfproteina.disease.oligo import SEED_WIDTH, innate_immune_record
from rfproteina.disease.toxicology import measure_toxicology
from tests.helpers.disease import UBIQUITIN

_UBIQUITIN = UBIQUITIN
#: Hartmann et al., J Immunol 2000, 164:1617: the prototype human CpG-B ODN.
_ODN_2006 = "tcgtcgttttgtcgttttgtcgtt"
#: Krieg et al., Nature 1995, 374:546, as read by the murine-optimal hexamer.
_ODN_1826 = "tccatgacgttcctgacgtt"
_HEXAMER = re.compile(r"\b([ACGT]{6})\b")


def _flanked(hexamer: str) -> str:
    return f"AAAA{hexamer}TTTT"


def test_every_hexamer_the_docstring_calls_an_instance_is_really_a_hit():
    """A named example that the consensus does not match is a false claim.

    ``GTCGTT`` was named as "one instance of" purine-purine-CG-pyrimidine-
    pyrimidine. It is not one, and the code never called it a hit, so the
    comment promised coverage the measurement does not have.
    """
    doc = (oligo.__doc__ or "").replace("\n", " ")
    assert _HEXAMER.search(doc), "the docstring names no hexamer at all"
    claimed = 0
    for match in _HEXAMER.finditer(doc):
        if "instance of it" not in doc[match.end() : match.end() + 90]:
            continue
        claimed += 1
        assert innate_immune_record(_flanked(match.group(1)))["state"] == "hit", match.group(1)
    assert claimed == 1, "exactly one hexamer is offered as an instance of the consensus"


def test_the_human_optimal_hexamer_is_outside_this_consensus():
    """Outside Krieg's RRCGYY, and read by Hartmann's rule instead.

    The two assertions this test used to make -- that ``GTCGTT`` and ODN 2006
    come back ``untested``/``no_pass_cut`` -- pinned the gap the module has
    since closed, and Hartmann et al., J Immunol 2000, 164:1617 proves that
    silence wrong: GTCGTT is the optimal human hexamer, and ODN 2006 built on
    it is the canonical human TLR9 agonist. The murine consensus still cannot
    match GTCGTT, which is why the module reads two rules and names the one
    that fired rather than merging them.
    """
    assert innate_immune_record(_flanked("GACGTT"))["state"] == "hit"
    assert re.fullmatch("[AG][AG]CG[CT][CT]", "GTCGTT") is None
    record = innate_immune_record(_flanked("GTCGTT"))
    assert record["state"] == "hit"
    assert record["motif_sites"] == [{"position": 5, "motif": "GTCGTT"}]
    assert record["cpg_count"] == 1
    odn = innate_immune_record(_ODN_2006)
    assert odn["state"] == "hit"
    assert odn["measurement"] == 3
    assert [site["motif"] for site in odn["motif_sites"]] == ["GTCGTT"] * 3
    assert odn["cpg_count"] == 4


def test_the_consensus_counts_and_places_every_hexamer():
    """RRCGYY, overlapping, one-based at the start of the hexamer."""
    record = innate_immune_record(_ODN_1826)
    assert record["state"] == "hit"
    assert record["measurement"] == 2
    assert record["motif_positions"] == [6, 15]
    chain = _ODN_1826.upper()
    assert [chain[start - 1 : start + 5] for start in record["motif_positions"]] == ["GACGTT", "GACGTT"]
    assert record["cpg_count"] == 2


def test_u_is_read_as_t_and_case_does_not_matter():
    assert innate_immune_record("aacgtt")["state"] == "hit"
    assert innate_immune_record("AACGUU")["state"] == "hit"
    assert innate_immune_record("aacguu")["motif_positions"] == [1]
    assert innate_immune_record(_UBIQUITIN)["blocker"] == "sequence_not_nucleic"


def test_a_fifteen_base_seed_is_not_rare_in_a_transcriptome():
    """The arithmetic the docstring is allowed to draw a conclusion from.

    4**15 is only about three and a half times a human transcriptome, so a
    quarter of all seeds occur in one by chance, and a 60-nt design presents
    ninety-two of them.
    """
    space = 4**SEED_WIDTH
    assert space == 1_073_741_824
    assert 3e8 / space > 0.25
    seeds_of_a_60mer = 2 * (60 - SEED_WIDTH + 1)
    assert seeds_of_a_60mer * 3e8 / space > 10
    assert "stops being an accident" not in (oligo.__doc__ or "")


def test_the_seed_packing_is_collision_free_at_the_stated_width():
    """Two bits a base, so a 15-base seed is 30 bits and fits a uint32."""
    assert SEED_WIDTH * 2 == 30
    assert 4**SEED_WIDTH - 1 < 2**32
    packed = {oligo._pack("".join(window)) for window in itertools.product("ACGT", repeat=8)}
    assert len(packed) == 4**8
    assert oligo._pack("A" * SEED_WIDTH) == 0
    assert oligo._pack("T" * SEED_WIDTH) == 4**SEED_WIDTH - 1
    assert oligo._pack("ACGT") == 0b00_01_10_11
    assert oligo._pack("U" * SEED_WIDTH) == oligo._pack("T" * SEED_WIDTH)
    assert oligo._pack("ACGTN" + "A" * 10) is None


def test_both_orientations_of_the_design_are_searched():
    design = "ACGTTGCAACCTAGGTTACG"
    assert oligo._reverse_complement(design) == "CGTAACCTAGGTTGCAACGT"
    seeds = oligo._seeds(design) | oligo._seeds(oligo._reverse_complement(design))
    assert len(seeds) == 2 * (len(design) - SEED_WIDTH + 1)
    assert oligo._pack(design[:SEED_WIDTH]) in seeds
    assert oligo._pack(oligo._reverse_complement(design)[:SEED_WIDTH]) in seeds


def _tox(tmp_path: Path, name: str, table: str) -> dict[str, object]:
    folder = tmp_path / name
    folder.mkdir()
    (folder / "toxicology.tsv").write_text(table, encoding="utf-8")
    record = measure_toxicology(folder)["herg"]
    assert isinstance(record, dict)
    return record


def test_a_micromolar_cell_is_refused_and_never_rescaled(tmp_path: Path):
    """The thousand-fold error: 5 uM read as 5 nM turns a clear into a hit."""
    assert HIT_NM == 10_000.0
    for index, spelling in enumerate(("uM", "µM", "μM", "umol/L", "M", "%")):
        record = _tox(tmp_path, f"unit{index}", f"protein\tnm\tunit\nherg\t5\t{spelling}\n")
        assert record["state"] == "untested", spelling
        assert record["blocker"] == "reading_not_nm", spelling
    assert _tox(tmp_path, "nm", "protein\tnm\tunit\nherg\t5\tnM\n")["state"] == "hit"
    assert _tox(tmp_path, "blank", "protein\tnm\tunit\nherg\t5\t\n")["state"] == "hit"


def test_the_cut_is_at_or_below_and_the_units_travel_with_it(tmp_path: Path):
    at_cut = _tox(tmp_path, "at", "protein\tnm\nherg\t10000\n")
    assert at_cut["state"] == "hit"
    assert at_cut["measurement"] == 10_000.0
    assert at_cut["units"] == "nM"
    assert at_cut["threshold"] == HIT_NM
    assert _tox(tmp_path, "above", "protein\tnm\nherg\t10000.0001\n")["state"] == "clear"


def test_a_bound_decides_only_when_every_value_it_admits_does(tmp_path: Path):
    assert _tox(tmp_path, "gt", "protein\tnm\nherg\t>10000\n")["state"] == "clear"
    assert _tox(tmp_path, "le", "protein\tnm\nherg\t<=10000\n")["state"] == "hit"
    assert _tox(tmp_path, "le_u", "protein\tnm\nherg\t≤10000\n")["state"] == "hit"
    undecided = _tox(tmp_path, "ge", "protein\tnm\nherg\t>=10000\n")
    assert undecided["state"] == "untested"
    assert undecided["blocker"] == "reading_not_decisive"
    assert _tox(tmp_path, "ge_u", "protein\tnm\nherg\t≥10000\n")["blocker"] == "reading_not_decisive"


def test_an_unreadable_cell_is_not_a_reading(tmp_path: Path):
    assert _tox(tmp_path, "word", "protein\tnm\nherg\tweak\n")["blocker"] == "cell_unparsed"
    assert _tox(tmp_path, "empty", "protein\tnm\nherg\t\n")["blocker"] == "toxicology_not_reported"
    assert _tox(tmp_path, "incon", "protein\tnm\nherg\tinconclusive\n")["blocker"] == "toxicology_not_reported"
    assert _tox(tmp_path, "neg", "protein\tnm\nherg\t-5\n")["blocker"] == "reading_not_nm"
    spoiled = _tox(tmp_path, "spoiled", "protein\tnm\tunit\nherg\t50000\tnM\nherg\t20\tuM\n")
    assert spoiled["state"] == "untested"
    survives = _tox(tmp_path, "survives", "protein\tnm\tunit\nherg\t5\tnM\nherg\t20\tuM\n")
    assert survives["state"] == "hit"


def test_the_chemical_motifs_count_every_overlapping_occurrence():
    record = chemical_record("NNSSNGSTDGDGNPSNAS", partners=None)
    assert record["deamidation_sites"] == [
        {"residue": 2, "motif": "NS"},
        {"residue": 5, "motif": "NG"},
    ]
    assert record["isomerization_sites"] == [
        {"residue": 9, "motif": "DG"},
        {"residue": 11, "motif": "DG"},
    ]
    assert [site["residue"] for site in record["glycosylation_sites"]] == [1, 2, 5, 16]


def test_the_sequon_skips_proline_and_names_the_third_residue():
    """Asn-Xaa-Ser/Thr with Xaa not Pro (Gavel & von Heijne, 1990)."""
    assert chemical_record("NPSNPT", partners=None)["glycosylation_count"] == 0
    assert chemical_record("NAS", partners=None)["glycosylation_sites"] == [{"residue": 1, "motif": "NXS"}]
    assert chemical_record("NAT", partners=None)["glycosylation_sites"] == [{"residue": 1, "motif": "NXT"}]
    assert chemical_record("NAC", partners=None)["glycosylation_count"] == 0
    assert chemical_record("ANA", partners=None)["glycosylation_count"] == 0


def test_methionine_tryptophan_and_free_cysteine_are_one_based():
    record = chemical_record("MWCAC", partners=None)
    assert record["methionine_residues"] == [1]
    assert record["tryptophan_residues"] == [2]
    assert record["free_cysteine"] == "untested"
    assert "free_cysteine_count" not in record
    named = chemical_record("MWCAC", partners=[3])
    assert named["free_cysteine"] == "reported"
    assert named["free_cysteine_residues"] == [5]
    none_paired = chemical_record("MWCAC", partners=[])
    assert none_paired["free_cysteine_residues"] == [3, 5]


def test_ubiquitin_carries_one_isomerization_site_and_no_sequon():
    record = chemical_record(_UBIQUITIN, partners=None)
    assert record["isomerization_sites"] == [{"residue": 52, "motif": "DG"}]
    assert record["deamidation_count"] == 0
    assert record["glycosylation_count"] == 0
    assert record["methionine_residues"] == [1]
    assert record["tryptophan_count"] == 0


def test_the_two_interface_numbers_say_which_atoms_they_cover():
    """``clash_fraction`` is the complex's, not the designed chain's."""
    doc = screen._interface_numbers.__doc__ or ""
    assert "target-side" in doc
    assert "whole" in doc and "complex" in doc


def _two_chain_cif() -> Path:
    path = Path(__file__).parent / "data" / "sam_binder_generated_minimized.cif"
    if not path.is_file():
        pytest.skip("the two-chain fixture is not in this checkout")
    return path


def test_the_stored_interface_numbers_are_the_metrics_own_numbers():
    from rfproteina.metrics.design_metrics import compute_clash_stats
    from rfproteina.metrics.physics.contact_ms.metric import ContactMolecularSurfaceMetric

    loaded = screen._load_generated_structure(_two_chain_cif())
    assert loaded is not None
    labels = sorted({str(label) for label in loaded.chain_id})
    if len(labels) < 2:
        pytest.skip("the fixture holds one chain, so there is no interface to delegate")
    stored = screen._interface_numbers(loaded, labels[0])
    surface = ContactMolecularSurfaceMetric(device="cpu", compute_binder_side=False).compute(
        predicted_atom_array=loaded, chain_of_interest=labels[0]
    )
    clashes = compute_clash_stats(loaded, threshold=1.5)
    assert stored["contact_ms"] == float(f"{surface['contact_ms']:.4f}")
    assert stored["clash_fraction"] == float(f"{clashes['frac_atoms_clashing_1p5A']:.4f}")
    assert 0.0 <= stored["clash_fraction"] <= 1.0


def test_the_screened_sequence_is_the_designed_chain_not_the_target():
    text = (
        "data_design\n#\nloop_\n_entity_poly.entity_id\n_entity_poly.pdbx_strand_id\n"
        "_entity_poly.pdbx_seq_one_letter_code_can\n"
        "1 A DESIGNEDCHAIN\n"
        "2 E TARGETCHAIN\n#\n"
    )
    sequences = screen._polymer_sequences(text)
    assert sequences == {"A": "DESIGNEDCHAIN", "E": "TARGETCHAIN"}
    assert screen._sequence_for_chain(sequences, "A") == "DESIGNEDCHAIN"
    # The metrics table names the chain instance, and the CIF names the strand.
    assert screen._sequence_for_chain(sequences, "A_1") == "DESIGNEDCHAIN"
    assert screen._sequence_for_chain(sequences, "E_1") == "TARGETCHAIN"
    assert screen._sequence_for_chain(sequences, "B") is None


def _flat_entity_poly(
    sequence: str = "DESIGNEDCHAIN",
    strand: str = "A",
    *,
    canonical: bool = True,
    raw: str | None = None,
) -> str:
    """One polymer entity in the non-loop spelling this repository's writer uses.

    The sequence sits in a semicolon-delimited block split across lines, which
    is exactly how ``monomer-short_generated.cif`` is written on disk.
    """
    half = len(sequence) // 2
    block = f";{sequence[:half]}\n{sequence[half:]}\n;\n"
    lines = [
        "data_design",
        "#",
        "_entity_poly.entity_id                      0",
        "_entity_poly.type                           polypeptide(L)",
    ]
    if raw is not None:
        lines.append(f"_entity_poly.pdbx_seq_one_letter_code       {raw}")
    if canonical:
        lines.append("_entity_poly.pdbx_seq_one_letter_code_can   ")
        lines.append(block.rstrip("\n"))
    lines.append(f"_entity_poly.pdbx_strand_id                 {strand}")
    lines.append("#")
    return "\n".join(lines) + "\n"


class TestEntityPolyIsReadInBothSpellings:
    """``entity_poly`` written flat, which is how every single-entity design is.

    mmCIF writes a category with several rows as a ``loop_`` and the same
    category with one row as plain ``_name.item value`` lines. This repository's
    own writer emits both. So a structure with exactly one polymer entity -- a
    monomer, an inverse-folding output, an enzyme design, a motif scaffold, a
    small-molecule binder -- has no ``entity_poly`` loop at all, and a loop-only
    reader found nothing in every one of them.

    The sequence then read as *absent*, which is a different fact from a design
    that has none: every sequence-liability axis reported ``sequence_absent``
    and went untested, for a sequence sitting in the file in plain sight. It
    erred toward untested rather than toward a clear, so it was never a false
    clear -- it was an unmeasurable one.
    """

    def test_a_single_entity_written_flat_is_read(self) -> None:
        """The defect, at its smallest."""
        assert screen._polymer_sequences(_flat_entity_poly()) == {"A": "DESIGNEDCHAIN"}

    def test_a_semicolon_block_is_one_sequence_not_two_lines(self) -> None:
        """Split across lines in the file, joined without the break."""
        sequences = screen._polymer_sequences(_flat_entity_poly("ABCDEFGHIJKL"))
        assert sequences == {"A": "ABCDEFGHIJKL"}

    def test_a_flat_row_may_name_several_strands(self) -> None:
        """One entity instantiated twice is one row naming both."""
        assert screen._polymer_sequences(_flat_entity_poly(strand="A,B")) == {
            "A": "DESIGNEDCHAIN",
            "B": "DESIGNEDCHAIN",
        }

    def test_the_canonical_column_is_preferred_over_the_raw_one(self) -> None:
        """The raw column spells a modified residue as ``(ALY)``, which is not residues."""
        text = _flat_entity_poly("ABCDEF", raw="AB(ALY)DEF")
        assert screen._polymer_sequences(text) == {"A": "ABCDEF"}

    def test_the_raw_column_is_used_when_there_is_no_canonical_one(self) -> None:
        text = _flat_entity_poly(canonical=False, raw="ABCDEF")
        assert screen._polymer_sequences(text) == {"A": "ABCDEF"}

    @pytest.mark.parametrize("absent", [".", "?"])
    def test_a_flat_row_with_no_sequence_contributes_nothing(self, absent: str) -> None:
        text = _flat_entity_poly(canonical=False, raw=absent)
        assert screen._polymer_sequences(text) == {}

    def test_a_loop_is_read_exactly_as_before(self) -> None:
        """The regression guard: the loop path must not change at all."""
        text = (
            "data_design\n#\nloop_\n_entity_poly.entity_id\n_entity_poly.pdbx_strand_id\n"
            "_entity_poly.pdbx_seq_one_letter_code_can\n"
            "1 A DESIGNEDCHAIN\n"
            "2 E TARGETCHAIN\n#\n"
        )
        assert screen._polymer_sequences(text) == {"A": "DESIGNEDCHAIN", "E": "TARGETCHAIN"}

    def test_a_loop_row_is_not_also_read_as_a_flat_category(self) -> None:
        """Reading a row twice would be harmless here and a sign the skip broke.

        ``_nonloop_category`` skips loop bodies. If it stopped, the loop's own
        values would be scanned again as item/value lines, and a one-row loop
        would start contributing a second, differently-parsed row.
        """
        text = (
            "data_design\n#\nloop_\n_entity_poly.pdbx_strand_id\n"
            "_entity_poly.pdbx_seq_one_letter_code_can\n"
            "A DESIGNEDCHAIN\n#\n"
        )
        assert screen._nonloop_category(text, "_entity_poly.") == []
        assert screen._polymer_sequences(text) == {"A": "DESIGNEDCHAIN"}

    def test_a_loop_wins_over_a_flat_category_for_the_same_strand(self) -> None:
        """Both spellings in one file is not a real file, and the loop is read first.

        Pinned so the order is a decision rather than an accident: a change that
        let the trailing flat category overwrite a loop row would silently
        re-answer every verdict taken on such a file.
        """
        text = (
            "data_design\n#\nloop_\n_entity_poly.pdbx_strand_id\n"
            "_entity_poly.pdbx_seq_one_letter_code_can\n"
            "A FROMTHELOOP\n#\n"
            "_entity_poly.pdbx_strand_id                 A\n"
            "_entity_poly.pdbx_seq_one_letter_code_can   FROMTHEFLAT\n#\n"
        )
        assert screen._polymer_sequences(text) == {"A": "FROMTHELOOP"}

    def test_a_ragged_loop_is_still_reported_as_unparsed(self) -> None:
        """A file that did not parse is not a design with no sequence.

        The flat reader must not turn a ragged loop into a quiet empty result:
        the two have different remedies -- the file, and the run.
        """
        ragged: list[list[str]] = []
        text = (
            "data_design\n#\nloop_\n_entity_poly.pdbx_strand_id\n"
            "_entity_poly.pdbx_seq_one_letter_code_can\n"
            "A\n#\n"
        )
        assert screen._polymer_sequences(text, ragged) == {}
        assert ragged == [["_entity_poly.pdbx_strand_id", "_entity_poly.pdbx_seq_one_letter_code_can"]]

    def test_every_generated_design_in_the_repository_yields_its_sequence(self) -> None:
        """The measurement behind the fix, taken on the real files.

        Skipped where the checkout carries no runs; where it does, a design
        whose structure holds a polymer entity and reads as having no sequence
        is the defect, whatever spelling that file used.
        """
        paths = sorted(Path("runs").rglob("*_generated.cif")) + sorted(Path("disease_runs").rglob("*_generated.cif"))
        readable = [path for path in paths if "_entity_poly." in path.read_text(errors="replace")]
        if not readable:
            pytest.skip("this checkout has no generated designs to read")
        empty = [str(path) for path in readable if not screen._polymer_sequences(path.read_text(errors="replace"))]
        assert empty == [], f"{len(empty)} design(s) hold an entity_poly and still read as having no sequence"
