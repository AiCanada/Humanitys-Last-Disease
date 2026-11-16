"""Building the JSON the AlphaFold Server takes, and refusing what it will not.

Two classes of defect are pinned here, and they are opposites.

**Dropping an entity.** A request that leaves a chain out does not fail. It
succeeds at a different question: refolding a binder alone says whether the
binder folds, never whether the complex forms, which is what an interface
predicted aligned error is about. So a structure holding something the dialect
cannot carry is refused by name, and never emitted with that thing missing.
The hardest case is a non-polymer -- a cofactor has no row in ``_entity_poly``
at all, so a builder reading only the polymer table would have dropped it
silently and folded the protein on its own.

**Reading nothing and calling it empty.** An mmCIF category with one row is
written flat, as ``_name.item value`` lines; with two rows it is written as a
``loop_``. This repository's own writer emits both, so every single-polymer
design -- a monomer, an inverse-folding output, an enzyme design, a motif
scaffold, a small-molecule binder -- writes ``_entity_poly`` flat, and a
loop-only reader sees nothing in all of them. ``screen._polymer_sequences`` was
such a reader and has since been fixed; the repository now has two polymer
readers, and the test below exists to keep them from drifting apart.

The fixtures below are written in the exact two spellings the real files use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfproteina.af3server.request import (
    CHAIN_RESIDUE_FLOOR,
    DIALECT,
    IONS,
    JOBS_PER_FILE,
    LIGAND_COPY_LIMIT,
    LIGANDS,
    TOKEN_LIMIT,
    VERSION,
    FoldJob,
    dna_sequence,
    ion,
    job_from_cif,
    ligand,
    protein_chain,
    reverse_complement,
    rna_sequence,
    server_chain_ids,
    token_count,
    validate_payload,
    write_batches,
)
from rfproteina.af3server.structure import read_categories, read_entities

SEQUENCE_A = "SILDSLLEKYQFPEDLTLYIKSVFPATDSFSRFKDMVLYYVESKYSSS"
SEQUENCE_B = "SPITEYLASLSTYNDQSITFALEDESYEIYVEDLKKDEKKDKVLLSYY"


def flat_cif(sequence: str = SEQUENCE_A, poly_type: str = "polypeptide(L)", strand: str = "A") -> str:
    """One polymer entity, written the way the repository's writer writes one.

    Single-row categories are plain item/value lines, and the sequence sits in a
    semicolon-delimited block split across lines -- which is exactly how
    ``monomer-short_generated.cif`` is written.
    """
    half = len(sequence) // 2
    return (
        "data_design\n"
        "#\n"
        "_entry.id       design.cif\n"
        "#\n"
        "_entity.id                 0\n"
        "_entity.type               polymer\n"
        "_entity.pdbx_description   .\n"
        "#\n"
        "_entity_poly.entity_id                      0\n"
        f"_entity_poly.type                           {poly_type}\n"
        "_entity_poly.pdbx_seq_one_letter_code       \n"
        f";{sequence[:half]}\n{sequence[half:]}\n;\n"
        "_entity_poly.pdbx_seq_one_letter_code_can   \n"
        f";{sequence[:half]}\n{sequence[half:]}\n;\n"
        f"_entity_poly.pdbx_strand_id                 {strand}\n"
        "_entity_poly.pdbx_target_identifier         ?\n"
        "#\n"
    )


def loop_cif(first: str = SEQUENCE_A, second: str = SEQUENCE_B, strands: tuple[str, str] = ("A", "E")) -> str:
    """Two polymer entities, written as loops -- the binder-complex spelling."""
    return (
        "data_design\n"
        "#\n"
        "loop_\n"
        "_entity.id \n"
        "_entity.type \n"
        "_entity.pdbx_description \n"
        "0 polymer .\n"
        "1 polymer .\n"
        "#\n"
        "loop_\n"
        "_entity_poly.entity_id \n"
        "_entity_poly.type \n"
        "_entity_poly.pdbx_seq_one_letter_code \n"
        "_entity_poly.pdbx_seq_one_letter_code_can \n"
        "_entity_poly.pdbx_strand_id \n"
        "_entity_poly.pdbx_target_identifier \n"
        f"0 polypeptide(L) {first} {first} {strands[0]} ?\n"
        f"1 polypeptide(L) {second} {second} {strands[1]} ?\n"
        "#\n"
    )


def ligand_cif(code: str = "SAM", atoms: int = 27, sequence: str = SEQUENCE_A) -> str:
    """A protein plus a non-polymer entity, whose atoms are only in _atom_site."""
    rows = "".join(f"HETATM {index} C C{index} {code} B 0 ? 1.0 2.0 3.0 1.00 50.00\n" for index in range(1, atoms + 1))
    return (
        "data_design\n"
        "#\n"
        "loop_\n"
        "_entity.id \n"
        "_entity.type \n"
        "_entity.pdbx_description \n"
        "0 non-polymer .\n"
        "1 polymer .\n"
        "#\n"
        "_entity_poly.entity_id                      1\n"
        "_entity_poly.type                           polypeptide(L)\n"
        f"_entity_poly.pdbx_seq_one_letter_code_can   {sequence}\n"
        "_entity_poly.pdbx_strand_id                 A\n"
        "#\n"
        "loop_\n"
        "_atom_site.group_PDB \n"
        "_atom_site.id \n"
        "_atom_site.type_symbol \n"
        "_atom_site.label_atom_id \n"
        "_atom_site.label_comp_id \n"
        "_atom_site.label_asym_id \n"
        "_atom_site.label_entity_id \n"
        "_atom_site.pdbx_PDB_ins_code \n"
        "_atom_site.Cartn_x \n"
        "_atom_site.Cartn_y \n"
        "_atom_site.Cartn_z \n"
        "_atom_site.occupancy \n"
        "_atom_site.B_iso_or_equiv \n"
        f"{rows}"
        "#\n"
    )


class TestFlatAndLoopSpellings:
    def test_a_single_entity_design_written_flat_is_read(self) -> None:
        """The defect a loop-only reader has: this file is most of the repository."""
        entities = read_entities(flat_cif())
        assert not isinstance(entities, str), entities
        assert len(entities) == 1
        assert entities[0].sequence == SEQUENCE_A
        assert entities[0].strands == ("A",)
        assert entities[0].polymer_type == "polypeptide(l)"

    def test_the_two_readers_agree_about_every_sequence(self) -> None:
        """The repository has two polymer readers, and they must not drift apart.

        ``screen._polymer_sequences`` reads the same two spellings now, so on
        these fixtures the two agree exactly. A divergence would mean the screen
        measured one molecule while the refold request asked about another --
        the sequence-liability axes and the fold would be about different
        things, and nothing would say so.

        Not asserted as equality in general: this module additionally **requires**
        an ``_entity`` table, because without it there is no way to know the
        polymer rows are the whole structure and a dropped cofactor would go
        unnoticed. The screen has no such need. So this reader is the stricter
        of the two by design, and the invariant is agreement wherever it
        succeeds, with neither finding a chain the other missed.
        """
        from rfproteina.disease.screen import _polymer_sequences

        for text, expected in (
            (flat_cif(), {"A": SEQUENCE_A}),
            (loop_cif(), {"A": SEQUENCE_A, "E": SEQUENCE_B}),
        ):
            entities = read_entities(text)
            assert not isinstance(entities, str), entities
            mine = {strand: item.sequence for item in entities for strand in item.strands}
            theirs = _polymer_sequences(text)
            assert mine == expected, f"this reader: {mine}"
            assert theirs == expected, f"the screen's reader: {theirs}"

    def test_a_two_entity_design_written_as_loops_is_read(self) -> None:
        entities = read_entities(loop_cif())
        assert not isinstance(entities, str), entities
        assert [item.sequence for item in entities] == [SEQUENCE_A, SEQUENCE_B]
        assert [item.strands for item in entities] == [("A",), ("E",)]

    def test_a_semicolon_block_is_joined_without_its_line_breaks(self) -> None:
        """A sequence split across lines is one sequence, not two with a newline."""
        entities = read_entities(flat_cif())
        assert not isinstance(entities, str)
        assert "\n" not in entities[0].sequence

    def test_a_category_is_matched_exactly_not_by_prefix(self) -> None:
        """``_entity`` must not collect ``_entity_poly`` or ``_entity_poly_seq``."""
        found = read_categories(loop_cif(), ("_entity",))
        assert set(found) == {"_entity"}
        assert [row["_entity.type"] for row in found["_entity"].rows] == ["polymer", "polymer"]

    def test_an_item_with_no_value_does_not_consume_the_next_category(self) -> None:
        """A dangling item name must not swallow the line after it."""
        text = (
            "data_d\n#\n_entity.id                 0\n_entity.type\n#\n"
            "_entity_poly.entity_id   0\n_entity_poly.type   polypeptide(L)\n"
            f"_entity_poly.pdbx_seq_one_letter_code_can   {SEQUENCE_A}\n"
            "_entity_poly.pdbx_strand_id   A\n#\n"
        )
        found = read_categories(text, ("_entity", "_entity_poly"))
        assert found["_entity_poly"].rows[0]["_entity_poly.type"] == "polypeptide(L)"


class TestNonPolymers:
    def test_a_cofactor_is_seen_and_its_atoms_counted(self) -> None:
        """Only ``_atom_site`` knows a non-polymer exists, so it is read."""
        entities = read_entities(ligand_cif())
        assert not isinstance(entities, str), entities
        cofactor = next(item for item in entities if not item.is_polymer)
        assert cofactor.kind == "non-polymer"
        assert cofactor.components == ("SAM",)
        assert cofactor.atoms == 27

    def test_a_design_around_an_unaccepted_ligand_is_refused_by_name(self) -> None:
        """The dialect's ligand list is closed, and SAM is not on it.

        Refused rather than sent as the protein alone. This repository's own
        small-molecule benchmark is built around SAM, so this is the realistic
        case, not a contrived one.
        """
        assert job_from_cif(ligand_cif("SAM"), name="probe") == "ligand_not_accepted:SAM"

    def test_a_design_around_an_accepted_ligand_is_built_with_it(self) -> None:
        made = job_from_cif(ligand_cif("ATP", atoms=31), name="probe")
        assert not isinstance(made, str), made
        keys = [next(iter(entity)) for entity in made.sequences]
        assert keys == ["ligand", "proteinChain"]
        body = made.sequences[0]["ligand"]
        assert body["ligand"] == "CCD_ATP"
        # 31 ligand atoms are 31 tokens, read off the structure rather than guessed.
        assert made.tokens().tokens == 31 + len(SEQUENCE_A)
        assert made.tokens().exact

    def test_an_ion_is_built_as_an_ion_with_no_ccd_prefix(self) -> None:
        made = job_from_cif(ligand_cif("MG", atoms=1), name="probe")
        assert not isinstance(made, str), made
        assert made.sequences[0] == {"ion": {"ion": "MG", "count": 1}}

    def test_the_private_atom_count_is_not_written_into_the_payload(self) -> None:
        """``_atoms`` is this module's bookkeeping. The dialect has no such field."""
        made = job_from_cif(ligand_cif("ATP", atoms=31), name="probe")
        assert not isinstance(made, str)
        written = json.dumps(made.payload())
        assert "_atoms" not in written
        assert validate_payload([made.payload()]) == []

    def test_a_structure_with_no_entity_table_is_refused(self) -> None:
        """Without it there is no way to know the polymers are the whole structure."""
        text = (
            "data_d\n#\n_entity_poly.entity_id   0\n_entity_poly.type   polypeptide(L)\n"
            f"_entity_poly.pdbx_seq_one_letter_code_can   {SEQUENCE_A}\n"
            "_entity_poly.pdbx_strand_id   A\n#\n"
        )
        assert read_entities(text) == "entity_table_absent"
        assert job_from_cif(text, name="probe") == "entity_table_absent"

    def test_water_is_not_sent_and_does_not_refuse_the_job(self) -> None:
        """The server adds no water and takes none. It is not a dropped entity."""
        text = loop_cif().replace("1 polymer .\n", "1 polymer .\n2 water .\n")
        made = job_from_cif(text, name="probe")
        assert not isinstance(made, str), made
        assert len(made.sequences) == 2


class TestWhichEntityKeyAChainGetsWrittenUnder:
    """The file's stated ``_entity_poly.type`` decides, and nothing else does.

    An earlier exporter in this repository guessed the kind from the sequence
    alphabet when the file stated none, with a 12-residue floor to keep ``ACGT``
    from being read as DNA. The floor is a sound guard and the guess is still a
    guess: ``ACGTACGTACGT`` is twelve perfectly ordinary residues, and folding a
    peptide as a DNA strand answers a question nobody asked. This refuses
    instead, and the two tests at the end of this class are the places the two
    implementations disagreed on purpose.
    """

    @staticmethod
    def typed(declared: str, sequence: str = SEQUENCE_A) -> str:
        return flat_cif(sequence, poly_type=declared)

    @pytest.mark.parametrize(
        ("declared", "key"),
        [
            ("polypeptide(L)", "proteinChain"),
            ("polydeoxyribonucleotide", "dnaSequence"),
            ("polyribonucleotide", "rnaSequence"),
        ],
    )
    def test_the_stated_type_chooses_the_entity_key(self, declared: str, key: str) -> None:
        sequence = {"proteinChain": SEQUENCE_A, "dnaSequence": "ATGCATGCATGC", "rnaSequence": "AUGCAUGCAUGC"}[key]
        made = job_from_cif(self.typed(declared, sequence), name="probe")
        assert not isinstance(made, str), made
        assert next(iter(made.sequences[0])) == key

    @pytest.mark.parametrize(
        "declared",
        [
            "polydeoxyribonucleotide/polyribonucleotide hybrid",
            "polysaccharide(D)",
            "peptide nucleic acid",
            "other",
        ],
    )
    def test_a_type_the_dialect_cannot_express_refuses_the_job(self, declared: str) -> None:
        """Refused rather than folded as a protein, which is what guessing would do."""
        made = job_from_cif(self.typed(declared), name="probe")
        assert isinstance(made, str) and made.startswith("polymer_type_not_supported:"), made

    def test_a_d_polypeptide_is_refused_rather_than_sent_as_a_protein(self) -> None:
        """Where this differs from the exporter it replaced, and why.

        That one mapped ``polypeptide(D)`` onto ``proteinChain``. The server
        takes the 20 standard **L** amino acids and has no way to say D, so the
        job would have been folded as the mirror-image molecule and come back
        with confident numbers about something that was never designed. A
        refusal is the only honest answer the dialect allows.
        """
        made = job_from_cif(self.typed("polypeptide(D)"), name="probe")
        assert made == "polymer_type_not_supported:polypeptide(d)"

    def test_an_untyped_polymer_is_refused_rather_than_inferred(self) -> None:
        """The alphabet is not a type. ``ACGTACGTACGT`` is twelve residues too."""
        text = (
            "data_design\n#\n_entity.id                 0\n_entity.type               polymer\n#\n"
            "_entity_poly.entity_id                      0\n"
            "_entity_poly.pdbx_seq_one_letter_code_can   ACGTACGTACGT\n"
            "_entity_poly.pdbx_strand_id                 A\n#\n"
        )
        assert job_from_cif(text, name="probe") == "polymer_type_absent"


class TestServerChainLabels:
    """The labels the server gives the chains, which it assigns by position.

    The dialect carries no chain id, so nothing the design calls a chain
    survives the trip. Reading a result under the design's own strand name
    therefore finds no such chain -- and finds it silently, which is why this is
    computed rather than assumed.
    """

    def test_the_labels_run_in_reverse_spreadsheet_order(self) -> None:
        """``AA`` follows ``Z``, then ``BA`` -- least significant letter first.

        A spreadsheet would say ``AB`` next. Both agree for the first 26 chains
        and diverge after, so a wrong rule here mislabels only large complexes.
        """
        labels = server_chain_ids(30)
        assert labels[:3] == ("A", "B", "C")
        assert labels[25:28] == ("Z", "AA", "BA")
        assert len(set(labels)) == 30

    def test_no_chains_is_no_labels(self) -> None:
        assert server_chain_ids(0) == ()

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError):
            server_chain_ids(-1)

    def test_a_designs_strands_map_to_the_labels_in_emission_order(self) -> None:
        """The real case: a binder on strand A and a target on strand E."""
        made = job_from_cif(loop_cif(strands=("A", "E")), name="binder_0", chain_of_interest="A")
        assert not isinstance(made, str), made
        assert made.chain_labels() == {"A": "A", "E": "B"}
        assert made.server_chain("E") == "B"
        assert made.server_chain("Z") is None


class TestEntityBuilders:
    def test_a_non_standard_residue_refuses_the_chain(self) -> None:
        """The server takes the 20 standard amino acids. Substituting would fold another molecule."""
        assert protein_chain(SEQUENCE_A[:-1] + "X") == "sequence_not_standard_amino_acids"

    def test_a_chain_under_the_residue_floor_is_refused(self) -> None:
        """Four residues is the server's own floor, so three is a rejected upload."""
        assert protein_chain("AAA") == "chain_below_residue_floor"
        assert protein_chain("AAAA") != "chain_below_residue_floor"
        assert CHAIN_RESIDUE_FLOOR == 4

    def test_dna_and_rna_alphabets_are_not_interchangeable(self) -> None:
        assert dna_sequence("ATGCATGC")["sequence"] == "ATGCATGC"
        assert dna_sequence("AUGCAUGC") == "sequence_not_dna"
        assert rna_sequence("AUGCAUGC")["sequence"] == "AUGCAUGC"
        assert rna_sequence("ATGCATGC") == "sequence_not_rna"

    def test_a_template_is_only_turned_off_by_saying_so(self) -> None:
        """Omitting the key leaves the server's default, which is on."""
        assert "useStructureTemplate" not in protein_chain(SEQUENCE_A)
        assert protein_chain(SEQUENCE_A, use_structure_template=False)["useStructureTemplate"] is False

    def test_the_designed_chain_gets_no_template_and_the_context_keeps_one(self) -> None:
        """The design is the thing under test; a near neighbour must not answer for it."""
        made = job_from_cif(loop_cif(), name="probe", chain_of_interest="A")
        assert not isinstance(made, str), made
        designed, context = made.sequences[0]["proteinChain"], made.sequences[1]["proteinChain"]
        assert designed["useStructureTemplate"] is False
        assert "useStructureTemplate" not in context

    def test_a_modification_position_is_one_based(self) -> None:
        built = protein_chain(SEQUENCE_A, modifications=[("CCD_HY3", 1)])
        assert built["modifications"] == [{"ptmType": "CCD_HY3", "ptmPosition": 1}]
        assert protein_chain(SEQUENCE_A, modifications=[("CCD_HY3", 0)]) == ("modification_position_outside_sequence")

    def test_a_modification_outside_the_allowlist_is_refused(self) -> None:
        assert protein_chain(SEQUENCE_A, modifications=[("CCD_NOPE", 1)]) == "modification_not_accepted"

    def test_a_ligand_code_may_be_given_with_or_without_the_prefix(self) -> None:
        assert ligand("ATP")["ligand"]["ligand"] == "CCD_ATP"
        assert ligand("CCD_ATP")["ligand"]["ligand"] == "CCD_ATP"

    def test_an_ion_code_is_stripped_of_a_prefix_it_should_not_have(self) -> None:
        assert ion("CCD_MG")["ion"]["ion"] == "MG"
        assert ion("MG")["ion"]["ion"] == "MG"

    def test_too_many_ligand_copies_is_refused(self) -> None:
        assert ligand("ATP", count=LIGAND_COPY_LIMIT) != "ligand_copies_over_limit"
        assert ligand("ATP", count=LIGAND_COPY_LIMIT + 1) == "ligand_copies_over_limit"

    def test_the_allowlists_are_the_published_sizes(self) -> None:
        """Counted, so a code lost to an editing slip is caught rather than silently missing."""
        assert len(LIGANDS) == 19
        assert len(IONS) == 10
        assert all(code.startswith("CCD_") for code in LIGANDS)
        assert not any(code.startswith("CCD_") for code in IONS)

    def test_a_reverse_complement_is_offered_because_the_dialect_has_no_duplex(self) -> None:
        assert reverse_complement("ATGC") == "GCAT"
        assert reverse_complement("AUGC") is None


class TestTokenBudget:
    def test_copies_count_toward_the_budget(self) -> None:
        entity = protein_chain(SEQUENCE_A, count=3)
        assert token_count([{"proteinChain": entity}]).tokens == 3 * len(SEQUENCE_A)

    def test_an_ion_costs_one_token_per_copy_and_that_is_exact(self) -> None:
        """All ten accepted ions are monatomic, so nothing is being assumed."""
        sized = token_count([ion("MG", count=4)])
        assert sized.tokens == 4
        assert sized.exact

    def test_an_unmeasured_ligand_makes_the_total_a_floor_rather_than_a_guess(self) -> None:
        sized = token_count([ligand("ATP")])
        assert sized.tokens == 0
        assert not sized.exact
        assert sized.uncounted == ("CCD_ATP",)

    def test_a_floor_over_the_limit_still_refuses(self) -> None:
        """A floor answers "certainly too large" and never "certainly fits"."""
        big = {"proteinChain": {"sequence": "A" * (TOKEN_LIMIT + 1), "count": 1}}
        sized = token_count([big, ligand("ATP")])
        assert not sized.exact
        assert sized.over

    def test_a_design_over_the_limit_is_refused_before_upload(self) -> None:
        made = job_from_cif(flat_cif("A" * (TOKEN_LIMIT + 1)), name="probe")
        assert made == "over_token_limit"


class TestPayloadShape:
    def test_the_payload_matches_the_dialect(self) -> None:
        made = job_from_cif(loop_cif(), name="binder_0", chain_of_interest="A")
        assert not isinstance(made, str), made
        payload = made.payload()
        assert list(payload) == ["name", "modelSeeds", "sequences", "dialect", "version"]
        assert payload["dialect"] == DIALECT
        assert payload["version"] == VERSION and isinstance(payload["version"], int)
        assert payload["modelSeeds"] == []
        assert validate_payload([payload]) == []

    def test_a_seed_is_written_as_a_string(self) -> None:
        job = FoldJob("probe", ({"proteinChain": {"sequence": SEQUENCE_A, "count": 1}},), seeds=(1593933729,))
        assert job.payload()["modelSeeds"] == ["1593933729"]

    def test_a_root_that_is_not_a_list_is_the_first_thing_refused(self) -> None:
        """A list root is how the upstream loader tells this dialect from the native one."""
        assert validate_payload({"name": "x"}) == ["the root must be a list of jobs, even for a single job"]

    def test_the_version_must_be_the_integer_not_the_string(self) -> None:
        made = job_from_cif(flat_cif(), name="probe")
        assert not isinstance(made, str)
        payload = made.payload()
        payload["version"] = "1"
        assert any("not the integer 1" in problem for problem in validate_payload([payload]))

    def test_dialect_and_version_must_travel_together(self) -> None:
        made = job_from_cif(flat_cif(), name="probe")
        assert not isinstance(made, str)
        payload = made.payload()
        del payload["version"]
        assert any("both dialect and version" in problem for problem in validate_payload([payload]))

    def test_an_unexpected_job_key_is_refused_as_the_parser_would(self) -> None:
        made = job_from_cif(flat_cif(), name="probe")
        assert not isinstance(made, str)
        payload = made.payload()
        payload["notAKey"] = 1
        assert any("unexpected keys" in problem for problem in validate_payload([payload]))

    def test_an_entity_with_two_keys_is_refused(self) -> None:
        payload = [
            {
                "name": "p",
                "modelSeeds": [],
                "sequences": [{"proteinChain": {"sequence": SEQUENCE_A}, "ion": {"ion": "MG"}}],
                "dialect": DIALECT,
                "version": VERSION,
            }
        ]
        assert any("exactly one of" in problem for problem in validate_payload(payload))

    def test_every_problem_is_reported_not_just_the_first(self) -> None:
        """A validator that stops at the first turns one fix into one round trip each."""
        payload = [{"name": "has space", "modelSeeds": "nope", "sequences": [], "dialect": "wrong", "version": 9}]
        assert len(validate_payload(payload)) >= 4

    def test_two_jobs_sharing_a_name_is_refused(self) -> None:
        """Results come back named for the job, so two of a name spoils both."""
        made = job_from_cif(flat_cif(), name="probe")
        assert not isinstance(made, str)
        problems = validate_payload([made.payload(), made.payload()])
        assert any("is used by 2 jobs" in problem for problem in problems)

    def test_a_name_that_would_not_survive_as_a_filename_is_refused(self) -> None:
        assert job_from_cif(flat_cif(), name="has space") == "name_not_a_filename"
        assert job_from_cif(flat_cif(), name="../escape") == "name_not_a_filename"


class TestBatchFiles:
    def test_a_batch_is_written_and_reads_back_as_valid(self, tmp_path: Path) -> None:
        made = job_from_cif(loop_cif(), name="binder_0", chain_of_interest="A")
        assert not isinstance(made, str)
        written = write_batches([made], tmp_path / "jobs")
        assert [path.name for path in written] == ["af3server_jobs.json"]
        assert validate_payload(json.loads(written[0].read_text(encoding="utf-8"))) == []

    def test_batches_are_split_at_the_servers_own_file_limit(self, tmp_path: Path) -> None:
        jobs = []
        for index in range(5):
            made = job_from_cif(flat_cif(), name=f"binder_{index}")
            assert not isinstance(made, str)
            jobs.append(made)
        written = write_batches(jobs, tmp_path / "jobs", per_file=2)
        assert [path.name for path in written] == [
            "af3server_jobs_01.json",
            "af3server_jobs_02.json",
            "af3server_jobs_03.json",
        ]
        assert len(json.loads(written[-1].read_text(encoding="utf-8"))) == 1

    def test_nothing_is_written_when_any_batch_would_be_rejected(self, tmp_path: Path) -> None:
        """A half-written batch directory is worse than none."""
        good = job_from_cif(flat_cif(), name="binder_0")
        assert not isinstance(good, str)
        clash = FoldJob("binder_0", good.sequences)
        destination = tmp_path / "jobs"
        with pytest.raises(ValueError, match="would reject"):
            write_batches([good, clash], destination, per_file=JOBS_PER_FILE)
        assert not destination.exists()
