"""A design as the JSON the hosted AlphaFold Server takes.

The ``alphafoldserver`` dialect is exactly specified, and it is specified in two
places that disagree. The prose in ``google-deepmind/alphafold/server/README.md``
says a protein modification carries ``position``; the README's own worked
example, the shipped ``server/example.json``, and the parser in
``google-deepmind/alphafold3`` at ``src/alphafold3/common/folding_input.py`` all
say ``ptmPosition``. The same README calls template indices "O-based" twice,
meaning 0-based. Everything below follows the parser and the example file, never
the prose, and the constants are written out rather than derived so a reader can
check each one against those sources.

What the dialect is, in one place:

* The root is a JSON **list** of jobs. A single job is a one-element list --
  that is how the upstream loader tells this dialect from the native one.
* A job holds exactly ``name``, ``modelSeeds``, ``sequences``, ``dialect``,
  ``version``. ``dialect`` is the string ``alphafoldserver`` and ``version`` the
  integer ``1``; both are present or both absent, never one.
* Each element of ``sequences`` is an object with exactly one key, naming the
  entity: ``proteinChain``, ``dnaSequence``, ``rnaSequence``, ``ligand`` or
  ``ion``. The last two repeat their own key inside:
  ``{"ligand": {"ligand": "CCD_ATP", "count": 1}}``.
* ``ptmPosition``, ``basePosition`` and a glycan's ``position`` are 1-based.
* Modifications, ligands and ions come from closed allowlists, not from the open
  CCD. Ligand codes carry a ``CCD_`` prefix and ion codes do not.

**The rule this module exists to enforce.** A refold request that quietly leaves
a chain out does not fail -- it succeeds, at a different question. Refolding a
binder alone answers whether the binder folds; it cannot answer whether the
complex forms, which is what an interface predicted aligned error or an aligned
binder RMSD is about. So a structure holding an entity this dialect cannot carry
is **refused by name**, and never emitted with that entity dropped. Every
refusal below is one of those, and each one names what could not be carried.

Nothing in this module touches the network.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rfproteina.af3server.structure import Entity, read_entities
from rfproteina.disease.screen import _cif_text, _generated_cifs, _sample_ids, designed_chain

#: The dialect name, verbatim. ``folding_input.ALPHAFOLDSERVER_JSON_DIALECT``.
DIALECT = "alphafoldserver"
#: The dialect version, an integer and not the string ``"1"``.
#: ``folding_input.ALPHAFOLDSERVER_JSON_VERSION``.
VERSION = 1

#: Keys a job may hold, and the only ones. The upstream parser rejects any other
#: with ``Unexpected JSON keys in: ...``; emitting one is a refused upload.
JOB_KEYS = ("name", "modelSeeds", "sequences", "dialect", "version")
#: Keys that may name an entity, in the order the upstream parser tests them.
ENTITY_KEYS = ("proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion")

#: The 20 standard amino acids in IUPAC one-letter form. The server supports
#: these and nothing else -- no ``X``, ``B``, ``Z``, ``J``, ``U`` or ``O``. A
#: design carrying an ambiguity code cannot be folded by this service, and
#: silently substituting one would fold a different molecule.
PROTEIN_LETTERS = frozenset("ACDEFGHIKLMNPQRSTVWY")
#: DNA is spelled with these four letters only.
DNA_LETTERS = frozenset("ATGC")
#: RNA is spelled with these four letters only. ``U``, not ``T``.
RNA_LETTERS = frozenset("AUGC")

#: The 23 protein modifications the server accepts, each ``CCD_``-prefixed.
PROTEIN_MODIFICATIONS = frozenset(
    {
        "CCD_SEP", "CCD_TPO", "CCD_PTR", "CCD_NEP", "CCD_HIP", "CCD_ALY", "CCD_MLY", "CCD_M3L",
        "CCD_MLZ", "CCD_2MR", "CCD_AGM", "CCD_MCS", "CCD_HYP", "CCD_HY3", "CCD_LYZ", "CCD_AHB",
        "CCD_P1L", "CCD_SNN", "CCD_SNC", "CCD_TRF", "CCD_KCR", "CCD_CIR", "CCD_YHA",
    }
)  # fmt: skip
#: The 9 DNA base modifications the server accepts.
DNA_MODIFICATIONS = frozenset(
    {"CCD_5CM", "CCD_C34", "CCD_5HC", "CCD_6OG", "CCD_6MA", "CCD_1CC", "CCD_8OG", "CCD_5FC", "CCD_3DR"}
)
#: The 14 RNA base modifications the server accepts.
RNA_MODIFICATIONS = frozenset(
    {
        "CCD_PSU", "CCD_5MC", "CCD_OMC", "CCD_4OC", "CCD_5MU", "CCD_OMU", "CCD_UR3",
        "CCD_A2M", "CCD_MA6", "CCD_6MZ", "CCD_2MG", "CCD_OMG", "CCD_7MG", "CCD_RSQ",
    }
)  # fmt: skip
#: The 19 ligands the server accepts. A closed list: this dialect takes neither
#: SMILES nor an arbitrary CCD code, so a design built around any other cofactor
#: -- S-adenosylmethionine, for one, which this repository's own small-molecule
#: benchmark uses -- cannot be sent here as a complex at all.
LIGANDS = frozenset(
    {
        "CCD_ADP", "CCD_ATP", "CCD_AMP", "CCD_GTP", "CCD_GDP", "CCD_FAD", "CCD_NAD",
        "CCD_NAP", "CCD_NDP", "CCD_HEM", "CCD_HEC", "CCD_PLM", "CCD_OLA", "CCD_MYR",
        "CCD_CIT", "CCD_CLA", "CCD_CHL", "CCD_BCL", "CCD_BCB",
    }
)  # fmt: skip
#: The 10 ions the server accepts. **Bare**, with no ``CCD_`` prefix -- the
#: upstream parser strips the prefix from a ligand code and does not from an
#: ion, so the asymmetry is real and not a documentation slip. All ten are
#: monatomic, which is why an ion's token cost is exactly its copy count.
IONS = frozenset({"MG", "ZN", "CL", "CA", "NA", "MN", "K", "FE", "CU", "CO"})

#: Tokens one job may total. The server's own FAQ gives the counting rule: one
#: per amino-acid residue, one per nucleotide base, one per ligand atom, one per
#: ion, one per atom of a modified residue, and for a glycan one per glycan atom
#: plus one for the residue it attaches to.
TOKEN_LIMIT = 5000
#: Residues the shortest acceptable polymer chain may have. The FAQ states a
#: four-residue floor for every protein and nucleotide chain. A three-residue
#: context peptide is therefore a refused upload, not a small one.
CHAIN_RESIDUE_FLOOR = 4
#: Copies of one ligand a job may name.
LIGAND_COPY_LIMIT = 50
#: Jobs one uploaded file may carry, imported as drafts.
JOBS_PER_FILE = 100
#: Jobs the account may run per day. Not enforced here -- it is an account fact,
#: not a request fact -- but a caller sizing a batch needs it, and
#: :py:mod:`rfproteina.af3server.session` keeps the ledger against it.
JOBS_PER_DAY = 30

#: Characters a job name may hold. The name is what a downloaded file is called
#: on the way back, and
#: :py:func:`~rfproteina.metrics.oracles.imported_refolds.import_refolds` matches
#: that filename to a design by stem. A name carrying a separator or a space
#: would come back as something that matches nothing.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
#: Seeds one job may name. The server runs one seed per job and returns five
#: diffusion samples from it, so a list longer than one is already unusual.
_SEED_LIMIT = 16
#: An unsigned 32-bit seed, which is what the dialect documents.
_SEED_CEILING = 2**32

#: mmCIF ``_entity_poly.type`` values this dialect can carry, and the entity key
#: each becomes. Everything else -- ``polypeptide(d)``, a DNA/RNA hybrid strand,
#: a polysaccharide, a peptide nucleic acid -- is refused, because the server has
#: no way to express it and a dropped strand is a changed question.
POLYMER_KINDS = {
    "polypeptide(l)": "proteinChain",
    "polydeoxyribonucleotide": "dnaSequence",
    "polyribonucleotide": "rnaSequence",
}
#: Which alphabet each entity key is spelled in.
_ALPHABETS = {
    "proteinChain": PROTEIN_LETTERS,
    "dnaSequence": DNA_LETTERS,
    "rnaSequence": RNA_LETTERS,
}

#: A CIF past this is not one design's structure.
CIF_BYTE_LIMIT = 64 * 1024 * 1024
#: Entities one design may hold. A generated complex is a handful.
ENTITY_LIMIT = 32


@dataclass(frozen=True)
class TokenCount:
    """A job's size against :data:`TOKEN_LIMIT`, and whether it is the whole size.

    ``exact`` is false when the job holds a ligand whose atom count was not
    supplied, which makes ``tokens`` a **floor**. A floor is enough to refuse a
    job that already overflows and is not enough to promise one fits, so
    :attr:`over` answers only the first question and :attr:`uncounted` names
    what is missing from the sum.
    """

    tokens: int
    exact: bool = True
    uncounted: tuple[str, ...] = ()

    @property
    def over(self) -> bool:
        """Whether this job is certainly too large. A floor over the limit is."""
        return self.tokens > TOKEN_LIMIT

    def as_dict(self) -> dict[str, object]:
        return {
            "tokens": self.tokens,
            "exact": self.exact,
            "limit": TOKEN_LIMIT,
            "uncounted": list(self.uncounted),
        }


@dataclass(frozen=True)
class FoldJob:
    """One job, ready to be written into an upload file."""

    name: str
    sequences: tuple[dict[str, object], ...]
    seeds: tuple[int, ...] = ()
    #: The chain whose sequence is the design under test, as the structure
    #: spelled it. Carried for the record and for whoever reads the request; the
    #: dialect itself has no field for it.
    chain_of_interest: str | None = None
    #: Strand ids in the order their entities were emitted, so a reader can map
    #: a returned chain back to the structure that was sent.
    chains: tuple[str, ...] = ()

    def payload(self) -> dict[str, object]:
        """The job as the dialect spells it, keys in the documented order."""
        return {
            "name": self.name,
            "modelSeeds": [str(seed) for seed in self.seeds],
            "sequences": [_without_private(entity) for entity in self.sequences],
            "dialect": DIALECT,
            "version": VERSION,
        }

    def tokens(self) -> TokenCount:
        """How large this job is, by the server's own counting rule."""
        return token_count(self.sequences)

    def chain_labels(self) -> dict[str, str]:
        """This design's own strand to the label the server will give it.

        The dialect carries no chain ids, so the server assigns its own by
        position in ``sequences``. A design whose binder is strand ``E`` comes
        back as chain ``B``, and reading its interface metrics under ``E`` finds
        nothing -- silently, because a chain order that names no chain of
        interest simply returns early.
        """
        return dict(zip(self.chains, server_chain_ids(len(self.sequences)), strict=True))

    def server_chain(self, strand: str) -> str | None:
        """The server's label for one of this design's strands, or None."""
        return self.chain_labels().get(strand)


@dataclass
class RequestReport:
    """Every design the build was asked about, and what became of it."""

    jobs: list[FoldJob] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """A record shaped for a run log: counts, and every reason by name."""
        return {
            "jobs": len(self.jobs),
            "tokens": sum(job.tokens().tokens for job in self.jobs),
            "files": [path.name for path in self.files],
            "refused": [{"example_id": name, "blocker": blocker} for name, blocker in self.refused],
            "limits": {
                "tokens_per_job": TOKEN_LIMIT,
                "jobs_per_file": JOBS_PER_FILE,
                "jobs_per_day": JOBS_PER_DAY,
            },
        }


def protein_chain(
    sequence: str,
    *,
    count: int = 1,
    modifications: Sequence[tuple[str, int]] = (),
    use_structure_template: bool | None = None,
) -> dict[str, object] | str:
    """One ``proteinChain`` entity, or a blocker name.

    ``use_structure_template`` left as ``None`` omits the key, which the server
    reads as true -- it will search the PDB for a template. ``False`` is emitted
    explicitly, because omitting it does not turn it off.

    Only the four fields the upstream converter also accepts are emitted:
    ``sequence``, ``count``, ``modifications``, ``useStructureTemplate``. The
    server itself additionally allows ``glycans``, ``maxTemplateDate``,
    ``unpairedMsa`` and ``templates``, and the converter rejects all four -- two
    as unexpected keys and two with an explicit raise, so that the official
    ``server/example.json`` fails it three separate ways. Staying inside the
    intersection means one file can be uploaded *and* replayed through a local
    AlphaFold 3, which is the point of checking a refold twice.
    """
    letters = _letters(sequence)
    if not letters:
        return "sequence_empty"
    stray = sorted(set(letters) - PROTEIN_LETTERS)
    if stray:
        return "sequence_not_standard_amino_acids"
    if len(letters) < CHAIN_RESIDUE_FLOOR:
        return "chain_below_residue_floor"
    if count < 1:
        return "count_below_one"
    entity: dict[str, object] = {"sequence": letters, "count": count}
    marked = _modifications(modifications, len(letters), PROTEIN_MODIFICATIONS, "ptmType", "ptmPosition")
    if isinstance(marked, str):
        return marked
    if marked:
        entity["modifications"] = marked
    if use_structure_template is not None:
        entity["useStructureTemplate"] = use_structure_template
    return entity


def dna_sequence(
    sequence: str,
    *,
    count: int = 1,
    modifications: Sequence[tuple[str, int]] = (),
) -> dict[str, object] | str:
    """One ``dnaSequence`` entity, or a blocker name.

    Single-stranded. A duplex is two entities, the second carrying the reverse
    complement -- :func:`reverse_complement` builds it. The dialect has no
    concept of a duplex, so sending one strand of an intended duplex folds a
    different molecule rather than failing.
    """
    return _nucleic(sequence, "dnaSequence", DNA_MODIFICATIONS, count, modifications)


def rna_sequence(
    sequence: str,
    *,
    count: int = 1,
    modifications: Sequence[tuple[str, int]] = (),
) -> dict[str, object] | str:
    """One ``rnaSequence`` entity, or a blocker name."""
    return _nucleic(sequence, "rnaSequence", RNA_MODIFICATIONS, count, modifications)


def ligand(code: str, *, count: int = 1, atoms: int = 0) -> dict[str, object] | str:
    """One ``ligand`` entity, or a blocker name. The key repeats inside.

    ``atoms`` is the ligand's heavy-atom count, which the dialect has no field
    for and the server charges a token each for. Supplying it makes
    :func:`token_count` exact; it is carried under a leading underscore and
    stripped by :meth:`FoldJob.payload` before anything is written.
    """
    wanted = code.strip().upper()
    if not wanted.startswith("CCD_"):
        wanted = f"CCD_{wanted}"
    if wanted not in LIGANDS:
        return f"ligand_not_accepted:{wanted.removeprefix('CCD_')}"
    if count < 1:
        return "count_below_one"
    if count > LIGAND_COPY_LIMIT:
        return "ligand_copies_over_limit"
    body: dict[str, object] = {"ligand": wanted, "count": count}
    if atoms > 0:
        body["_atoms"] = atoms
    return {"ligand": body}


def ion(code: str, *, count: int = 1) -> dict[str, object] | str:
    """One ``ion`` entity, or a blocker name. Bare code, no ``CCD_`` prefix."""
    wanted = code.strip().upper().removeprefix("CCD_")
    if wanted not in IONS:
        return f"ion_not_accepted:{wanted}"
    if count < 1:
        return "count_below_one"
    return {"ion": {"ion": wanted, "count": count}}


def server_chain_ids(count: int) -> tuple[str, ...]:
    """The chain labels the server assigns, in ``sequences`` order.

    The dialect has no field for a chain id, so position is the only thing that
    says which chain is which. The labels run in what AlphaFold 3's own docs
    call reverse spreadsheet order -- ``A``..``Z``, then ``AA``, ``BA``, ``CA``,
    ..., ``ZA``, ``AB``, ``BB`` -- which is the least significant letter first,
    the opposite way round from a spreadsheet column. Guessing ``AA`` follows
    ``Z`` by the spreadsheet rule gives the right answer for the first 26 chains
    and the wrong one after, which is a mislabelling that only shows up on a
    large complex.

    Mirrors ``mmcif_lib.int_id_to_str_id``, which is what the upstream converter
    calls to name the chains it expands a ``count`` into.
    """
    if count < 0:
        raise ValueError("A chain count cannot be negative.")
    return tuple(_int_id_to_str_id(index) for index in range(1, count + 1))


def _int_id_to_str_id(number: int) -> str:
    """One 1-based position as its label. ``1`` is ``A``, ``27`` is ``AA``."""
    letters: list[str] = []
    value = number - 1
    while value >= 0:
        letters.append(chr(value % 26 + ord("A")))
        value = value // 26 - 1
    return "".join(letters)


def chain_labels_for_run(
    out_dir: Path,
    example_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Per design, its own strand to the label the server gives that chain.

    Rebuilt from the designs themselves rather than read from a file written at
    submission time. The mapping is a function of the structure -- entity order
    decides it -- so recomputing cannot go stale, and there is no sidecar to
    lose between a submission and a download that may be days apart.

    A design that cannot be expressed as a job contributes nothing, which is the
    honest answer: if no request could be built for it, no mapping for it was
    ever sent.
    """
    report = jobs_from_run(out_dir, example_ids=example_ids)
    return {job.name: job.chain_labels() for job in report.jobs}


def reverse_complement(sequence: str) -> str | None:
    """The other strand of a DNA duplex, or None when this is not plain DNA."""
    letters = _letters(sequence)
    if not letters or set(letters) - DNA_LETTERS:
        return None
    return letters.translate(str.maketrans("ATGC", "TACG"))[::-1]


def token_count(sequences: Iterable[Mapping[str, object]]) -> TokenCount:
    """How many tokens a job's entities total, by the server's counting rule.

    One token per amino acid, one per nucleotide base, one per ligand atom, one
    per ion -- and copies count, so ``count: 3`` of a 100-mer is 300.

    A ligand's atom count is not a field of the dialect, so unless the caller
    attached one it cannot be added. Rather than invent a number, the ligand is
    named in :attr:`TokenCount.uncounted` and the total becomes a floor. A floor
    refuses an overflowing job correctly and never claims a job fits.
    """
    total = 0
    uncounted: list[str] = []
    for entity in sequences:
        for key in ("proteinChain", "dnaSequence", "rnaSequence"):
            body = entity.get(key)
            if isinstance(body, Mapping):
                sequence = body.get("sequence")
                copies = _copies(body)
                if isinstance(sequence, str):
                    total += len(sequence) * copies
        body = entity.get("ion")
        if isinstance(body, Mapping):
            # Every one of the ten accepted ions is monatomic, so this is exact.
            total += _copies(body)
        body = entity.get("ligand")
        if isinstance(body, Mapping):
            atoms = body.get("_atoms")
            code = body.get("ligand")
            if isinstance(atoms, int) and atoms > 0:
                total += atoms * _copies(body)
            else:
                uncounted.append(str(code) if isinstance(code, str) else "ligand")
    return TokenCount(total, not uncounted, tuple(uncounted))


def job_from_cif(
    text: str,
    *,
    name: str,
    chain_of_interest: str | None = None,
    design_templates: bool | None = False,
    context_templates: bool | None = None,
) -> FoldJob | str:
    """One job from one design's mmCIF, or a blocker name.

    Every entity in the structure is sent, not just the designed chain, for the
    reason :py:mod:`rfproteina.disease.refold_export` gives: a request missing
    the target silently asks whether the binder folds instead of whether the
    complex forms.

    ``design_templates`` defaults to ``False`` and is emitted explicitly. The
    design is the thing under test, and letting the server retrieve a PDB
    template for it invites a near neighbour of the design to supply the answer
    the refold is supposed to produce from sequence alone. Context chains are
    left at the server default, because the target's experimental structure is a
    real and intended input to the complex question; pass ``context_templates``
    to say otherwise.

    An entity this dialect cannot carry refuses the whole job, and the blocker
    names which one it was.
    """
    if len(text.encode("utf-8", "replace")) > CIF_BYTE_LIMIT:
        return "structure_too_large"
    if not _SAFE_NAME.fullmatch(name):
        return "name_not_a_filename"

    entities = read_entities(text)
    if isinstance(entities, str):
        return entities
    if not entities:
        return "no_entities"
    if len(entities) > ENTITY_LIMIT:
        return "too_many_entities"

    sequences: list[dict[str, object]] = []
    chains: list[str] = []
    for entity in entities:
        built = _entity_payloads(entity, chain_of_interest, design_templates, context_templates)
        if isinstance(built, str):
            return built
        for strand, payload in built:
            sequences.append(payload)
            chains.append(strand)

    if not sequences:
        return "no_polymer_sequence"
    sized = token_count(sequences)
    if sized.over:
        return "over_token_limit"
    return FoldJob(
        name=name,
        sequences=tuple(sequences),
        chain_of_interest=chain_of_interest if chain_of_interest in chains else None,
        chains=tuple(chains),
    )


def _entity_payloads(
    entity: Entity,
    chain_of_interest: str | None,
    design_templates: bool | None,
    context_templates: bool | None,
) -> list[tuple[str, dict[str, object]]] | str:
    """One entity as its ``sequences`` elements, or a blocker name.

    A polymer instantiated as several strands becomes one element per strand
    rather than one element with ``count``, because the strands are what the
    returned chain labels line up against and a ``count`` loses their names.
    """
    if entity.is_polymer:
        key = POLYMER_KINDS.get(entity.polymer_type)
        if key is None:
            return f"polymer_type_not_supported:{entity.polymer_type}" if entity.polymer_type else "polymer_type_absent"
        if not entity.strands:
            return "chain_not_named"
        built: list[tuple[str, dict[str, object]]] = []
        for strand in entity.strands:
            if key == "proteinChain":
                designed = chain_of_interest is not None and strand == chain_of_interest
                made = protein_chain(
                    entity.sequence,
                    use_structure_template=design_templates if designed else context_templates,
                )
            elif key == "dnaSequence":
                made = dna_sequence(entity.sequence)
            else:
                made = rna_sequence(entity.sequence)
            if isinstance(made, str):
                return made
            built.append((strand, {key: made}))
        return built

    # Not a polymer. The only non-polymers this dialect can carry are a ligand
    # from its 19-code list and an ion from its 10. Anything else refuses the
    # job: sending the protein without its cofactor is a different question,
    # and the ligand is often the entire point of the design.
    if len(entity.components) != 1:
        return "non_polymer_components_not_single" if entity.components else "non_polymer_components_absent"
    code = entity.components[0]
    strands = entity.strands or (code,)
    # One element carries every copy, so the copy count is the number of strands
    # the entity was instantiated as -- a ligand's label is not something the
    # server returns a chain for. The count goes through the builder rather than
    # being written onto the finished entity, so that a structure holding more
    # copies than the dialect allows is refused here and not at the upload.
    made = ion(code, count=len(strands)) if code in IONS else ligand(code, count=len(strands), atoms=entity.atoms)
    if isinstance(made, str):
        return made
    return [(strands[0], made)]


def jobs_from_run(
    out_dir: Path,
    *,
    example_ids: Sequence[str] | None = None,
    design_templates: bool | None = False,
    context_templates: bool | None = None,
) -> RequestReport:
    """One job per generated design under ``out_dir``.

    The chain of interest comes from the same place the screen reads it, so the
    request marks the chain the verdict will be taken on rather than a guess.
    A design that cannot be expressed is named in the report, never skipped
    quietly: a batch that silently folded four of five designs is the failure
    this is most likely to have.
    """
    report = RequestReport()
    interest = designed_chain(out_dir) or None
    wanted = set(example_ids) if example_ids is not None else None
    paths = _generated_cifs(out_dir)
    ids = _sample_ids(out_dir, paths)
    for index, path in enumerate(paths):
        example_id = ids[index]
        if wanted is not None and example_id not in wanted:
            continue
        text = _cif_text(path)
        if text is None:
            report.refused.append((example_id, "structure_unreadable"))
            continue
        made = job_from_cif(
            text,
            name=example_id,
            chain_of_interest=interest,
            design_templates=design_templates,
            context_templates=context_templates,
        )
        if isinstance(made, str):
            report.refused.append((example_id, made))
            continue
        report.jobs.append(made)
    if wanted is not None:
        seen = {job.name for job in report.jobs} | {name for name, _ in report.refused}
        report.refused.extend((name, "design_not_in_run") for name in sorted(wanted - seen))
    return report


def write_batches(
    jobs: Sequence[FoldJob],
    destination: Path,
    *,
    per_file: int = JOBS_PER_FILE,
    stem: str = "af3server_jobs",
) -> list[Path]:
    """Write ``jobs`` as upload files, each holding at most ``per_file`` of them.

    One file per batch rather than one per job: the server imports a whole file
    as draft jobs, so a handful of designs is one upload. Files are numbered
    from 1 and zero-padded, so a directory listing is in submission order.

    Every payload is validated before anything is written. A file the server
    would reject is worth catching here rather than after the upload, and a
    half-written batch directory is worse than none -- so nothing is written
    until every job in every batch has passed.
    """
    if per_file < 1:
        raise ValueError("per_file must be at least 1.")
    batches = [list(jobs[start : start + per_file]) for start in range(0, len(jobs), per_file)]
    documents: list[tuple[Path, str]] = []
    width = max(2, len(str(len(batches))))
    for index, batch in enumerate(batches, start=1):
        payload = [job.payload() for job in batch]
        problems = validate_payload(payload)
        if problems:
            raise ValueError(f"Refusing to write a batch the server would reject: {'; '.join(problems)}")
        name = f"{stem}_{index:0{width}d}.json" if len(batches) > 1 else f"{stem}.json"
        documents.append((destination / name, json.dumps(payload, indent=2) + "\n"))
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path, document in documents:
        path.write_text(document, encoding="utf-8")
        written.append(path)
    return written


def validate_payload(payload: object) -> list[str]:
    """Every way ``payload`` departs from the dialect, in one list.

    All of them, not the first: a caller fixing an upload file wants the whole
    set, and a validator that stops at the first problem turns one fix into one
    round trip each. An empty list means the upstream parser would accept it.
    """
    problems: list[str] = []
    if not isinstance(payload, list):
        return ["the root must be a list of jobs, even for a single job"]
    if not payload:
        problems.append("the root list is empty, so no job would be submitted")
    if len(payload) > JOBS_PER_FILE:
        problems.append(f"{len(payload)} jobs exceeds the {JOBS_PER_FILE} one file may carry")
    names: dict[str, int] = {}
    for index, job in enumerate(payload):
        where = f"job {index}"
        if not isinstance(job, dict):
            problems.append(f"{where} is not an object")
            continue
        name = job.get("name")
        if isinstance(name, str):
            where = f"job {index} ({name})"
            names[name] = names.get(name, 0) + 1
        problems.extend(_job_problems(job, where))
    for name, seen in sorted(names.items()):
        if seen > 1:
            # Results come back named for the job, and a design is matched to a
            # downloaded file by that name. Two jobs sharing one spoils both.
            problems.append(f"the name {name!r} is used by {seen} jobs, so their results cannot be told apart")
    return problems


def _job_problems(job: Mapping[str, object], where: str) -> list[str]:
    problems: list[str] = []
    stray = sorted(set(job) - set(JOB_KEYS))
    if stray:
        problems.append(f"{where} holds unexpected keys {stray}, which the parser refuses")
    for required in ("name", "modelSeeds", "sequences"):
        if required not in job:
            problems.append(f"{where} has no {required}")

    name = job.get("name")
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        problems.append(f"{where} has a name that will not survive the round trip as a filename")

    has_dialect, has_version = "dialect" in job, "version" in job
    if has_dialect != has_version:
        problems.append(f"{where} must carry both dialect and version, or neither")
    if has_dialect and job.get("dialect") != DIALECT:
        problems.append(f"{where} has dialect {job.get('dialect')!r}, not {DIALECT!r}")
    if has_version and job.get("version") != VERSION:
        # `1` and `"1"` are not the same value here: the parser compares to an int.
        problems.append(f"{where} has version {job.get('version')!r}, not the integer {VERSION}")

    problems.extend(_seed_problems(job.get("modelSeeds"), where))

    sequences = job.get("sequences")
    if not isinstance(sequences, list):
        problems.append(f"{where} has sequences that is not a list")
        return problems
    if not sequences:
        problems.append(f"{where} has no entities, so there is nothing to fold")
    for position, entity in enumerate(sequences):
        problems.extend(_entity_problems(entity, f"{where} entity {position}"))

    sized = token_count(item for item in sequences if isinstance(item, Mapping))
    if sized.over:
        problems.append(f"{where} is {sized.tokens} tokens, past the {TOKEN_LIMIT} limit")
    return problems


def _seed_problems(seeds: object, where: str) -> list[str]:
    if seeds is None:
        return []
    if not isinstance(seeds, list):
        return [f"{where} has modelSeeds that is not a list"]
    problems: list[str] = []
    if len(seeds) > _SEED_LIMIT:
        problems.append(f"{where} names {len(seeds)} seeds, past the {_SEED_LIMIT} this writes")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, str | int):
            problems.append(f"{where} has a seed that is neither a string nor an integer")
            continue
        try:
            value = int(seed)
        except ValueError:
            problems.append(f"{where} has a seed {seed!r} that is not a number")
            continue
        if not 0 <= value < _SEED_CEILING:
            problems.append(f"{where} has a seed outside the unsigned 32-bit range")
    return problems


def _entity_problems(entity: object, where: str) -> list[str]:
    if not isinstance(entity, dict):
        return [f"{where} is not an object"]
    keys = sorted(entity)
    named = [key for key in keys if key in ENTITY_KEYS]
    if len(keys) != 1 or len(named) != 1:
        return [f"{where} must hold exactly one of {list(ENTITY_KEYS)}, and holds {keys}"]
    key = named[0]
    body = entity[key]
    if not isinstance(body, dict):
        return [f"{where} has a {key} that is not an object"]

    problems: list[str] = []
    count = body.get("count", 1)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        problems.append(f"{where} has count {count!r}, which must be an integer of at least 1")
    elif key == "ligand" and count > LIGAND_COPY_LIMIT:
        problems.append(f"{where} names {count} copies, past the {LIGAND_COPY_LIMIT} one ligand may have")

    if key in _ALPHABETS:
        allowed = {"sequence", "count", "modifications"}
        if key == "proteinChain":
            allowed |= {"useStructureTemplate"}
        stray = sorted(set(body) - allowed)
        if stray:
            problems.append(f"{where} holds {stray}, which the local converter refuses")
        sequence = body.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            problems.append(f"{where} has no sequence")
        else:
            outside = sorted(set(sequence) - _ALPHABETS[key])
            if outside:
                problems.append(f"{where} has a sequence holding {outside}, outside this entity's alphabet")
            elif len(sequence) < CHAIN_RESIDUE_FLOOR:
                problems.append(f"{where} has {len(sequence)} residues, under the {CHAIN_RESIDUE_FLOOR} floor")
        problems.extend(_modification_problems(body, key, where))
        template = body.get("useStructureTemplate")
        if template is not None and not isinstance(template, bool):
            problems.append(f"{where} has useStructureTemplate that is not a boolean")
    else:
        stray = sorted(set(body) - {key, "count"})
        if stray:
            problems.append(f"{where} holds {stray}, which this entity has no field for")
        code = body.get(key)
        if not isinstance(code, str):
            problems.append(f"{where} does not repeat its own {key} key with a code")
        elif key == "ligand" and code not in LIGANDS:
            problems.append(f"{where} names ligand {code!r}, which is not one of the {len(LIGANDS)} accepted")
        elif key == "ion" and code not in IONS:
            problems.append(f"{where} names ion {code!r}, which is not one of the {len(IONS)} accepted")
    return problems


def _modification_problems(body: Mapping[str, object], key: str, where: str) -> list[str]:
    marks = body.get("modifications")
    if marks is None:
        return []
    if not isinstance(marks, list):
        return [f"{where} has modifications that is not a list"]
    type_key, position_key = (
        ("ptmType", "ptmPosition") if key == "proteinChain" else ("modificationType", "basePosition")
    )
    allowed = {
        "proteinChain": PROTEIN_MODIFICATIONS,
        "dnaSequence": DNA_MODIFICATIONS,
        "rnaSequence": RNA_MODIFICATIONS,
    }[key]
    sequence = body.get("sequence")
    length = len(sequence) if isinstance(sequence, str) else 0
    problems: list[str] = []
    for mark in marks:
        if not isinstance(mark, dict):
            problems.append(f"{where} has a modification that is not an object")
            continue
        stray = sorted(set(mark) - {type_key, position_key})
        if stray:
            problems.append(f"{where} has a modification holding {stray}; this entity uses {type_key}/{position_key}")
        code = mark.get(type_key)
        if not isinstance(code, str) or code not in allowed:
            problems.append(f"{where} has modification {code!r}, which this entity does not accept")
        position = mark.get(position_key)
        if isinstance(position, bool) or not isinstance(position, int):
            problems.append(f"{where} has a {position_key} that is not an integer")
        elif length and not 1 <= position <= length:
            # 1-based. A 0 here is the most likely silent error in the format.
            problems.append(f"{where} has {position_key} {position}, outside 1..{length}")
    return problems


def _nucleic(
    sequence: str,
    key: str,
    allowed: frozenset[str],
    count: int,
    modifications: Sequence[tuple[str, int]],
) -> dict[str, object] | str:
    letters = _letters(sequence)
    if not letters:
        return "sequence_empty"
    if set(letters) - _ALPHABETS[key]:
        return "sequence_not_dna" if key == "dnaSequence" else "sequence_not_rna"
    if len(letters) < CHAIN_RESIDUE_FLOOR:
        return "chain_below_residue_floor"
    if count < 1:
        return "count_below_one"
    entity: dict[str, object] = {"sequence": letters, "count": count}
    marked = _modifications(modifications, len(letters), allowed, "modificationType", "basePosition")
    if isinstance(marked, str):
        return marked
    if marked:
        entity["modifications"] = marked
    return entity


def _modifications(
    marks: Sequence[tuple[str, int]],
    length: int,
    allowed: frozenset[str],
    type_key: str,
    position_key: str,
) -> list[dict[str, object]] | str:
    """The modification list, or a blocker name. Positions are 1-based."""
    built: list[dict[str, object]] = []
    for code, position in marks:
        wanted = code.strip().upper()
        if not wanted.startswith("CCD_"):
            wanted = f"CCD_{wanted}"
        if wanted not in allowed:
            return "modification_not_accepted"
        if not 1 <= position <= length:
            return "modification_position_outside_sequence"
        built.append({type_key: wanted, position_key: position})
    return built


def _copies(body: Mapping[str, object]) -> int:
    count = body.get("count", 1)
    return max(count, 0) if isinstance(count, int) and not isinstance(count, bool) else 0


def _without_private(entity: Mapping[str, object]) -> dict[str, object]:
    """The entity as it is written out, with this module's own fields removed.

    ``_atoms`` is carried on a ligand so the token count can be exact, and the
    dialect has no such field -- writing it would be an unexpected key and a
    refused upload.
    """
    cleaned: dict[str, object] = {}
    for key, body in entity.items():
        cleaned[key] = (
            {name: value for name, value in body.items() if not name.startswith("_")}
            if isinstance(body, Mapping)
            else body
        )
    return cleaned


def _letters(sequence: str) -> str:
    """The sequence with whitespace removed and upper-cased."""
    return "".join(sequence.split()).upper()
