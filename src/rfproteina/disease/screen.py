"""Write the measured screen for one finished generation.

The job file stays ``requested_clear``. Immunogenicity is the Calis 2013 class
I 9-mer score. A positive window among the ligands the class I table says
are presented is a hit; an unrestricted scan of every window is what stays
untested. ADA is
Paul 2015 DR-panel breadth from an MHC class II table. MHC class I and II are
read from predictor tables. Proteome offtarget is exact 9-mer identity against
a FASTA. A missing MHC table or proteome stays untested. A supplied
immunogenicity or ADA table wins over the sequence formula. A gene symbol is
looked up by the UniProt accession of the sequence-fixed chain when the
campaign structure names exactly one. That read is ``target_accession``. The
job's target id is not rewritten, and a read accession that is a safety-panel
protein makes the call ``target_is_panel`` whatever symbol the caller named.
``axis_evidence.json`` wins when it already names an axis. A ligand binder
reads nM concentrations from ``toxicology.tsv`` in the output directory. The
panel cut is 10_000 nM. A missing table stays untested, and a supplied state
wins on that axis too. Protein and peptide binders do not read that table.
A protein or peptide binder reads ``paralog_coevolution.tsv`` onto the paralog
record. The paralog state stays the UniRef50 result.
Sequence developability is computed for a protein or peptide binder only,
because GRAVY and the instability index are defined over amino acids; the
designed interface is computed from the designed chain whatever the kind, when
a generated CIF is present. A protein or peptide binder
also stores ``chemical_liability`` for that chain: motif counts, free cysteine
when the CIF names disulfide partners, and the Bjellqvist 1994 isoelectric
point and net charge at pH 7.4. Those numbers do not clear an axis. Each
generated CIF is stored in ``samples``. The top-level sequence, chemical, and
interface fields are the first CIF, not a summary that can hide a later
failure. This module does not start a refolding oracle, a network predictor, or
a cure claim.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import re
import warnings
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from rfproteina.disease.chemical import chemical_phrase, chemical_record
from rfproteina.disease.coevolution import coevolution_phrase
from rfproteina.disease.combinations import (
    AXES_BY_KIND,
    HIT_NM,
    LIGAND_AXES,
    PANEL_NM,
    bar_axes,
    judge_combination,
    target_is_panel_protein,
)
from rfproteina.disease.diversity import design_diversity, diversity_phrase
from rfproteina.disease.liability import (
    _DR_MEDIAN_MAX,
    _MHC_I_IC50,
    _MHC_I_RANK,
    _MHC_II_IC50,
    _MHC_II_RANK,
    SEQUENCE_LIABILITY_AXES,
    SUPPLIED_METHOD,
    measure_sequence_liability,
)
from rfproteina.disease.modality import modality_phrase, modality_record
from rfproteina.disease.oligo import NUCLEIC_LIABILITY_AXES, measure_nucleic_liability
from rfproteina.disease.prodigy import RunAffinity
from rfproteina.disease.provenance import code_version
from rfproteina.disease.publish import scrub_for_upload, scrubbed_json, write_text_atomically
from rfproteina.disease.refold import read_criteria, refold_report, row_pass_flag, trace_metric_kind
from rfproteina.disease.solubility import read_protein_sol, read_solubility, solubility_phrase
from rfproteina.disease.target_liability import (
    TARGET_LIABILITY_AXES,
    TargetLimits,
    measure_target_liability,
    uniprot_accession,
)
from rfproteina.disease.threshold_provenance import (
    PROVENANCE_KEY,
    describe_cuts,
    describe_interface_thresholds,
    provenance_of,
)
from rfproteina.disease.tnp import RunProfiles
from rfproteina.disease.toxicology import measure_toxicology
from rfproteina.disease.vina_pose import RunPoses
from rfproteina.utils.metric_thresholds import MetricThreshold
from rfproteina.utils.metrics_files import PENDING, PRIMARY, MetricsBlocker, job_tables, verified_sources, wide_csv_rows

#: A reader for a public JSON value. Injected so a screen reaches the network
#: only when its caller hands it one.
ReadJson = Callable[[str], object]
_EVIDENCE = "axis_evidence.json"
_SCREEN = "screen.json"
#: Heavy trees under a generation folder. The screen file is never inside them.
_SKIP_SCREEN_WALK = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        ".grok",
        "inference_samples",
        "inference_metrics",
        "evaluation_samples",
        "evaluation_metrics",
    }
)
_SCREEN_REPORT_LIMIT = 12
#: First word of the one line a listing adds about itself -- what it truncated
#: or collapsed. The line rides on the last report's lines because the reports
#: are per-run and this is not about any run, so a reader showing them must
#: separate it out again rather than printing it among that run's sentences.
SCREEN_LISTING_PREFIX = "Listing:"
#: How many runs one screen-only invocation will write. Separate from the
#: report limit, which is how many a page shows: showing twelve is a reading
#: choice, and screening only twelve of a larger archive would leave the rest
#: silently unscreened while the output looked complete.
SCREEN_RUN_LIMIT = 200
_SCREEN_WALK_CAP = 400
#: generations/<disease>/<target>/<folder>/screen.json is depth 4, and the walk
#: reads a folder at exactly this depth, so 4 reaches the archive and stops there.
_SCREEN_DEPTH = 4
#: Folders ``_holds_generation`` opens past the depth limit before it gives up
#: and answers "there may be a run down there". Only reached where the limit
#: prunes, and it decides a warning rather than a result.
_SCREEN_LOOKAHEAD = 60
#: How many pruned entries one walk asks ``_find_generation`` about. A dot name
#: and a symlinked directory are pruned at every depth, so asking without a
#: budget would turn a wide archive into a second walk of itself. Past the
#: budget nothing more is asked and nothing is claimed: a sweep is reported
#: partial only for a run that was actually seen behind a pruned entry.
_SCREEN_PRUNE_CHECKS = 40
_PPI_COLUMN = "ppi.pass"
_PRE_ORACLE_COLUMN = "pre_oracle_pass"
_CHAIN_COLUMN = "chain_of_interest"
_PRIMARY_METRICS = Path("inference_metrics") / "per_example_metrics.csv"
_REEVAL_METRICS = Path("inference_metrics") / "per_example_metrics_reeval.csv"
_EVALUATION_METRICS = Path("evaluation_metrics") / "per_example_metrics.csv"
_STATES = frozenset({"hit", "clear"})
_ABSENT = "absent"
_CANONICAL = "_entity_poly.pdbx_seq_one_letter_code_can"
_ONE_LETTER = "_entity_poly.pdbx_seq_one_letter_code"
_STRAND = "_entity_poly.pdbx_strand_id"

#: Sequence half of configs/metrics/pass_criteria/all_pass_criteria.yaml.
#: Rosetta SAP is not run here, so this is not developability.pass.
#: Shape of ``screen.json``. Raised when a key is added or changes meaning, so
#: a reader can tell which record it is holding rather than inferring it from
#: which keys happen to be present -- the shape has already changed once, and
#: the screens written before that have no way to say so.
#:
#: 3 adds ``refold``: every pass criterion
#: ``configs/metrics/pass_criteria/all_pass_criteria.yaml`` declares, with
#: each bound, the worst value the run reached against it, and which bounds
#: were missed. Nine of the eleven were evaluated into the metrics table and
#: discarded, and a false ``ppi_pass`` named none of the three bounds behind
#: it.
#:
#: 2 is the shape that can account for its own verdict. It adds, at the top
#: level, ``cuts`` (every number the verdict was taken against), ``bar_axes``
#: (which axes that bar required), ``metrics_file`` and ``metrics_blocker``
#: (which per-design table, or why there were no rows), ``chain_blocker``,
#: ``inputs`` (every supplied file with its digest), ``code`` (the commit that
#: applied all of it), ``replaced`` (the verdict a re-screen overwrote, and
#: which inputs moved) and ``nucleic_liability``. Nested shapes gained fields
#: with it: a sample carries ``row_blocker`` and a named ``interface_blocker``
#: and ``sequence_blocker``, an axis carries ``curated`` when a curated state
#: decided it, and a table-backed axis carries ``covered_examples``. A screen
#: written at 1 has none of these, and nothing in it says which of the two it
#: is -- which is the whole reason this number exists.
#:
#: Raising it is not optional bookkeeping. ``tests/test_disease_screen_shape.py``
#: pins both key sets beside this number, so a key cannot be added without the
#: version moving with it.
# Version 5 distinguishes incomplete refolds, assay-specific opinions and
# interface measurements without a calibrated decision threshold.
# Version 6 can include explicitly selected, per-design PRODIGY affinities.
# Version 7 also carries optional fixed-pose Vina scores on finished small-molecule complexes.
# Version 8 includes the optional post-generation VHH developability profile.
# Version 9 adds the compound-level ADMET-AI aqueous-solubility regression.
# Version 10 adds site-resolved aqueous microscopic pKa for nominated compounds.
# Version 11 adds ranked, per-parent SyGMa metabolite hypotheses.
SCREEN_VERSION = 12
#: Keys every screen carries, whatever the kind and whatever was supplied.
SCREEN_REQUIRED_KEYS = frozenset(
    {
        "bar",
        "bar_axes",
        "call",
        "code",
        "cuts",
        "designed_interface",
        "designed_interface_quality",
        "hits",
        "inputs",
        "metrics_file",
        "refold",
        "ppi_blocker",
        "ppi_pass",
        "pre_oracle_pass",
        "requested_class",
        "samples",
        "screen_version",
        "sequence_developability",
        "solubility",
        "states",
        "untested",
    }
)
#: Keys a screen carries only when there is something to say. Each is absent
#: rather than null, so a reader can tell "this run has no ligand panel" from
#: "this run has an empty one".
SCREEN_OPTIONAL_KEYS = frozenset(
    {
        "chain_blocker",
        "chemical_liability",
        "design_diversity",
        "job_mismatch",
        "metrics_blocker",
        "modality_screen",
        "nucleic_liability",
        "replaced",
        "sequence_liability",
        "target_accession",
        "target_liability",
        "ligand_alerts",
        "ligand_solubility",
        "method_selection",
        "qupkake",
        "sygma",
        "mhc_class_i",
        "prodigy_affinity",
        "protein_sol",
        "vina_pose",
        "tnp_profile",
        "synthesis",
        "toxicology",
    }
)
#: The GRAVY a design has to be at or under to pass the sequence half. This is
#: this program's own triage bar and NOT a published cut: Kyte and Doolittle
#: define the scale, and nothing in that paper or in the ExPASy documentation
#: says where a developable protein stops. A value near zero asks that the
#: chain not be net hydrophobic overall, which is the property the bar is
#: reaching for. It is recorded beside every verdict as ``gravy_max`` with
#: ``gravy_max_source`` naming it a convention, so a reader never takes
#: ``sequence_pass`` for a published criterion. Changing it is an owner's
#: decision about how strict this program is, not a correction.
_GRAVY_MAX = 0.1
_GRAVY_MAX_SOURCE = "program_convention"
#: The instability index at or above which ExPASy calls a protein unstable.
#: This one IS published: "A protein whose instability index is smaller than 40
#: is predicted as stable" (ExPASy ProtParam, on Guruprasad et al. 1990), which
#: is why the comparison against it is strict.
_INSTABILITY_MAX = 40.0
_INSTABILITY_MAX_SOURCE = "expasy_protparam_guruprasad_1990"
#: Non-bonded clash distance used by ``compute_clash_stats``. There is no pass cut.
_CLASH_THRESHOLD = 1.5
_CLASH_FRACTION = "frac_atoms_clashing_1p5A"
#: Where ``_sequence_block`` hands the chain it read back to its caller. It is
#: popped before the record is stored, so no sequence reaches screen.json.
_CHAIN_KEY = "_chain"
_IDENTITY_COLUMN = "sequence_similarity_to_input.sequence_identity_to_input"
_CA_RMSD_COLUMN = "ca_rmsd_to_input.rmsd"
#: Every metrics column this screen reads. A header that repeats one of them
#: leaves the verdict resting on whichever copy ``csv`` kept, which is the
#: last: a False in the first is invisible. A column nothing here reads may
#: repeat as often as the exporter likes.
_METRIC_COLUMNS = frozenset(
    {
        _PPI_COLUMN,
        _PRE_ORACLE_COLUMN,
        _CHAIN_COLUMN,
        _IDENTITY_COLUMN,
        _CA_RMSD_COLUMN,
        "example_id",
        "designed_structure_path",
    }
)
# These fields locate an observation and can change when a report is relocated.
_ROW_LOCATORS = {"example_id", "designed_structure_path", "designed_chain_path", "output_base_dir"}
#: A metrics file past this is not read. A run writes one row per design and
#: ``SCREEN_RUN_LIMIT`` caps a run at 200, so this is orders above any table
#: a run produces and only catches a file that is not one.
_METRICS_FILE_LIMIT = 16 * 1024 * 1024
#: Cells that name no chain. "?" and "." are the mmCIF nulls.
_CHAIN_ABSENT = frozenset({"nan", "none", "null", ".", "?"})
#: Fields an axis_evidence.json entry may carry beside its state.
_EVIDENCE_KEYS = frozenset({"state", "reason", "source"})
#: A basis is a sentence and a citation, not a document.
_BASIS_LIMIT = 300
#: What a curated state records when nothing was stated for it.
_NO_BASIS = "basis_not_stated"
#: Supplied files one run may carry. A folder past this is not a run
#: folder, and the screen records what it can rather than walking it.
_INPUT_LIMIT = 64
#: A supplied file past this is recorded by name and size, not digested.
_INPUT_BYTES = 256 * 1024 * 1024
#: Stands in for the whole list when a folder holds more files than one
#: run supplies, and makes the two screens incomparable on purpose.
_TRUNCATED = "inputs_truncated"
#: Stands in for the whole list when the folder could not be listed at all.
#: An empty list is an answer -- "this run supplied nothing" -- and a folder
#: that refused to be read is not that answer.
_UNLISTED = "inputs_not_listed"
#: Every stand-in above. A list carrying one of these is not a list of files,
#: so nothing may be concluded from a name's absence from it.
_WITHHELD_INPUTS = frozenset({_TRUNCATED, _UNLISTED})
#: A structure entry id as the run folders carry it: four characters with a
#: digit, which is every PDB code in the archive (8TLD, 9GVN, 9WWH, 8K4Q) and
#: not the kind prefix ("gene", "protein") or a suffix ("inverse").
_ENTRY_TOKEN = re.compile(r"(?=.*\d)[A-Za-z0-9]{4}")
#: A chain as a folder names one: a digit or two, or a letter with an optional
#: digit. "inverse" is not one, which is the point.
_CHAIN_TOKEN = re.compile(r"\d{1,2}|[A-Za-z]\d?")
#: An example id is used as a path component, so it has to begin with a letter
#: or digit. The older pattern admitted "..", which walked a sample's follow-up
#: spec out of its own directory into the shared root.
_SAFE_EXAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_BINDER_TOKEN = re.compile(r"binder_\d+")
_AF3_ENV = ("AF3_IMAGE", "AF3_MODEL_DIR", "AF3_JAX_CACHE_DIR")
_REFUSED_METRICS = frozenset({
    MetricsBlocker.PUBLICATION_PENDING, MetricsBlocker.AGGREGATION_REQUIRED, MetricsBlocker.PROVENANCE_MISMATCH,
    MetricsBlocker.PROVENANCE_UNREADABLE, "metrics_unreadable", "metrics_too_large", "metrics_columns_repeated",
    "metrics_row_too_long", "metrics_row_too_short",
})
_PPI_BLOCKERS = _REFUSED_METRICS | {"recorded", "oracle_not_configured", "not_requested", "metrics_incomplete"}

#: Same aliases as the combination judge. The bar is chosen before that call.
_KIND_ALIAS = {
    "small_molecule": "ligand",
    "non_polymer": "ligand",
    "dna": "nucleic",
    "rna": "nucleic",
    "nucleic_acid": "nucleic",
    "polypeptide": "protein",
}
_DEFAULT_BAR = {
    "protein": "sequence",
    "peptide": "sequence",
    "ligand": "coverage",
    "nucleic": "oligo",
}
_LIABILITY_KINDS = frozenset({"protein", "peptide"})
#: How many condition specs one screen will open. A generated structure points
#: back at the campaign deposit; this stops that hop from walking the disk.
_SPEC_LIMIT = 8
_AMBIGUOUS_ACCESSION = "target_accession_ambiguous"
_SELECT_CHAIN = re.compile(r"""sel\(\s*['"]([A-Za-z0-9]+)(?:/|['"])""")
_ACCESSION_PREFIXES = (
    "_struct_ref.",
    "_struct_ref_seq.",
    "_struct_asym.",
    "_entity_poly.",
)
#: Every axis block a screen can carry, the order its axes are printed in, and
#: whether the printed line reports its measurements as well as its curated
#: states. A curated state is printed from every block, whichever block the
#: axis sits in: the sequence axes were the only ones rendered, so a curated
#: clear over a measured hit on paralog, a plasma axis or either nucleic axis
#: left a screened_clear with nothing saying a human had declared it. Driven
#: from one list so a block added later cannot be left out of the sentence.
#: Toxicology is not here because ``_toxicology_clause`` reports it in nM.
_LIABILITY_BLOCKS = (
    ("sequence_liability", SEQUENCE_LIABILITY_AXES, True),
    ("target_liability", TARGET_LIABILITY_AXES, False),
    ("nucleic_liability", NUCLEIC_LIABILITY_AXES, False),
)
#: Generated designs this screen reads, longest suffix first so a stem is taken
#: against the whole of it, and so plain ``.cif`` keeps the highest index, which
#: is what makes plain beat compressed in ``_generated_cifs``.
#:
#: ``.cif.gzip`` is here because ``_design_text`` decodes it: the spelling
#: arrived for a user-supplied condition-spec input, where
#: ``combinations._STRUCTURE_SUFFIXES`` accepts it, and leaving it out of this
#: list meant a design the reader decodes perfectly was reported as
#: ``designs_absent`` -- no design at all, for a folder holding one. Nothing
#: here writes that spelling for a design, but a half-supported spelling is a
#: false claim in the suppressing direction, which is the worse one.
#: ``configs/inference.yaml`` writes plain ``.cif``;
#: ``compress_outputs=true`` writes ``.cif.zst``, and an archive or copy step can
#: leave a ``.cif.gz`` behind. Both were found and refused rather than read, so
#: a run written with compression measured nothing.
_DESIGN_SUFFIXES = (
    "_generated.cif.gzip",
    "_generated.cif.zst",
    "_generated.cif.gz",
    "_generated.cif",
)
#: Named once, because the suffix that needs a third-party module is read in
#: two places and the two must agree on which one it is.
_ZSTD_SUFFIX = "_generated.cif.zst"
#: Designs that are not a plain CIF. Consulted only once no readable design was
#: found, so by then each of these is one that could not be read: a binary CIF,
#: which is a different format rather than a compressed one and holds no text to
#: decode; an archive that does not decompress; or a ``.cif.zst`` on a machine
#: without ``zstandard``, which is not a declared dependency of this package.
_PACKED_DESIGNS = (
    "*_generated.cif.zst",
    "*_generated.cif.gz",
    "*_generated.cif.gzip",
    "*_generated.bcif",
)
#: Enough of an archive to tell one from plain bytes wearing an archive's name.
_PROBE_BYTES = 4096
#: Why a run measured nothing at all. Named on the two top-level blocks whose
#: nulls they explain, so the record and the printed lines both say it.
#: What one design's CIF text may expand to, for a *compressed* design only. A
#: compressed file's size on disk does not bound what it decompresses into, so
#: the bound is taken after. A plain file is read unbounded as it always was.
#: A generated design is one structure, which is well inside this.
_DESIGN_TEXT_LIMIT = 64 * 1024 * 1024
_DESIGNS_ABSENT = "designs_absent"
_DESIGNS_COMPRESSED = "designs_compressed"
_DESIGN_BLOCKERS = frozenset({_DESIGNS_ABSENT, _DESIGNS_COMPRESSED, MetricsBlocker.TABLE_ONLY})


def screen_generation(
    output_dir: Path,
    *,
    binder_kind: str,
    target_id: str,
    bar: str | None = None,
    proteome: Path | None = None,
    transcriptome: Path | None = None,
    read_json: ReadJson | None = None,
    limits: TargetLimits | None = None,
    screen_selection: str | None = None,
    screen_methods: Sequence[str] = (),
) -> dict[str, object]:
    """Judge one finished run and write ``screen.json`` beside its designs.

    Missing axes are left out of the evidence, so they stay untested. A
    measured proteome or MHC hit or clear is evidence for a protein or peptide
    binder. A Calis score is clear only when every 9-mer is at most 0. A
    positive window among the presented ligands is a hit; an unrestricted scan
    of every window is what stays untested. ADA is read from the class II table.
    ``axis_evidence.json`` overrides a measurement on that axis. A ``ppi.pass``
    cell is recorded and does not change the call. The primary metrics file
    wins; the reeval file is used only when the primary file is absent. An
    ``refold`` carries every pass criterion the repository declares, each
    bound, the worst value the run reached against it, and which bounds were
    missed. Those criteria are read from the config rather than copied here,
    and like ``ppi_pass`` they are recorded without moving the call.
    unreadable metrics file stores ``absent`` and still writes the screen,
    and ``metrics_blocker`` names why it yielded no rows. A table that repeats
    a column this screen reads, or that holds a row wider or narrower than its
    own header, is refused whole rather than read under the wrong names.
    Two metrics rows naming one design are that design's row while they agree;
    when they disagree the design records ``row_blocker`` and is measured
    against neither. A table naming more than one chain of interest names
    none, so a design whose own row is silent is left unmeasured rather than
    measured against a neighbour's chain.
    Every generated CIF is one ``samples`` entry. The top-level sequence and
    interface fields repeat the first sorted CIF.

    ``proteome`` is the FASTA the off-target axis reads, and it wins over
    ``HUMAN_PROTEOME``. The job file beside ``output_dir`` recorded the kind and
    the target when the run was planned; a disagreement with the arguments here
    is recorded as ``job_mismatch`` rather than passed over. A ligand binder
    also reads ``toxicology.tsv``. When that file is absent and ``ligand_smiles.txt``
    names a compound in the installed Octant CYP3A4 subset, the screen writes
    the converted nanomolar reading first. An existing table is not replaced.
    A hit or clear from that table is evidence
    unless ``axis_evidence.json`` already names the axis. A protein or peptide
    binder reads ``paralog_coevolution.tsv`` onto the paralog record. The
    paralog state stays the UniRef50 result. A peptide also measures
    ``membrane``: a 19-residue Kyte-Doolittle window above 1.6 is a span, one
    span is a hit, and a chain too short to hold a window stays untested. A
    nucleic binder measures ``off_target_hybridization`` from exact 15-base
    seeds of the design and its reverse complement against ``transcriptome.fa``
    beside the run or ``HUMAN_TRANSCRIPTOME``, and ``innate_immune`` from the
    TLR9 CpG consensus. A CpG-free oligo is not a clear, for the reason
    paralog's singleton cluster is not one. A protein or peptide binder also
    records ``chemical_liability`` from the designed chain. A motif count does
    not enter the call. Ligand and nucleic screens leave that scan unread.
    ``modality_screen`` is the drug-type report: furin, Asp-Pro, N-terminal
    glutamine, nonstandard letters, the aliphatic index, the average
    molecular weight, the Pace 1995 extinction at 280 nm, the N-end rule
    half-life, the residue and atomic composition, a Kyte-Doolittle span, and
    an AGGRESCAN 2007 normalized sum for a protein or peptide, CpG for a
    nucleic binder, and one non-polymer
    formula weight for a ligand. A missing
    weight is calculated from the chemical formula when every element is known.
    Otherwise the weight is left out. A quoted chem_comp value keeps its
    internal spaces, so a name or formula does not move the weight. An unclosed
    quote is not a cell. Those counts do not enter the call.
    """
    chosen = bar if bar is not None else _default_bar(binder_kind)
    if bar is not None:
        # Refused here rather than after the measurements: nine of the
        # thirteen declared bars had no way to be selected at all, and a
        # misspelling of one now in reach should not cost a scan first.
        bar_axes(binder_kind, bar)
    from rfproteina.disease import screen_selection as selection

    kind = _canonical_kind(binder_kind)
    active = set(selection.NATIVE_IDS)
    selection_record = None
    if screen_selection is not None or screen_methods:
        mode = screen_selection or "explicit"
        choices, selected = selection.execution_plan(output_dir, kind, mode, screen_methods)
        active = {c.method.key for c in choices if c.applicable and c.method.key in selected}
        selection_record = {
            "mode": mode,
            "selected": list(selected),
            "attempted": [key for key in selected if key in active],
            "skipped": {c.method.key: c.reason for c in choices if c.method.key in selected and not c.applicable},
        }
    metrics_path = _metrics_path(output_dir)
    rows, metrics_blocker = _metrics_table(output_dir, metrics_path)
    evidence = _read_evidence(output_dir)
    affinity = RunAffinity.configured(output_dir) if "prodigy_affinity" in active else None
    vina = RunPoses.configured(output_dir) if "vina_pose" in active else None
    tnp = RunProfiles.configured(output_dir) if "tnp_profile" in active else None
    if tnp is not None and kind != "protein" and tnp.blocker is None:
        tnp.blocker = "protein_binder_kind_required"
    if vina is not None and kind != "ligand" and vina.blocker is None:
        vina.blocker = "ligand_binder_kind_required"
    if affinity is not None and kind not in _LIABILITY_KINDS and affinity.blocker is None:
        affinity.blocker = "protein_binder_kind_required"
    samples, chains, coverage_rows = _sample_measurements(
        output_dir,
        rows,
        chemical=kind in _LIABILITY_KINDS,
        kind=kind,
        affinity=affinity,
        vina=vina,
        tnp=tnp,
    )
    liability, class_i_scan = _apply_sequence_liability(
        output_dir,
        rows,
        samples,
        binder_kind,
        evidence,
        proteome=proteome,
        sequences=chains,
    )
    target, read_accession = _apply_target_liability(
        output_dir,
        binder_kind,
        target_id,
        evidence,
        read_json=read_json,
        limits=limits,
    )
    toxicology = _apply_toxicology(output_dir, binder_kind, evidence)
    nucleic = _apply_nucleic_liability(
        output_dir,
        binder_kind,
        evidence,
        sequences=chains,
        samples=samples,
        transcriptome=transcriptome,
    )
    combination = judge_combination(
        binder_kind,
        evidence.states,
        bar=chosen,
        target_id=target_id,
        # The accession this screen read off the deposit's sequence-fixed
        # chain, when it read one. The judge tests the caller's ``target_id``
        # against the panel, and that is a gene symbol whenever an accession
        # had to be resolved at all -- so a panel protein named by any symbol
        # the table does not list was not on the panel as far as the call was
        # concerned. The same hERG deposit screened as ``target_is_panel``
        # under ``KCNH2`` and as ``screened_clear`` under ``LQT2``, with
        # ``target_accession: Q12809`` written in the same record. This module
        # already treats that accession as the target's identity: every
        # target-level axis is looked up by it. Measuring the panel against it
        # too is what stops the record from contradicting its own verdict.
        target_is_panel=read_accession is not None and target_is_panel_protein(read_accession),
    )
    refold = refold_report(coverage_rows)
    ppi = next((item for item in refold.get("criteria", []) if item["name"] == "ppi"), {})
    ppi_state = ppi.get("state")
    ppi_pass = (
        {"pass": True, "fail": False}.get(ppi_state, _ABSENT) if ppi else _flag_column(coverage_rows, _PPI_COLUMN)
    )
    ppi_blocker = "metrics_incomplete" if ppi_state == "incomplete" else _ppi_blocker(ppi_pass)
    if metrics_blocker in _REFUSED_METRICS:
        ppi_blocker = metrics_blocker
        if refold.get("refold_blocker") == "not_evaluated":
            refold["refold_blocker"] = metrics_blocker
    sequence, interface = _first_sample_fields(samples)
    interface_quality = _first_interface_quality(samples)
    if not samples:
        # The one no-measurement path that named no reason. A run whose
        # designs were written compressed, or never written at all, recorded
        # the same empty blocks as a run whose chain was simply never named,
        # so a reader had nothing to act on. Carried on the blocks the nulls
        # sit in rather than on a new key, which the printed lines and the
        # closed page already read.
        reason = _designs_blocker(output_dir)
        if (reason == _DESIGNS_ABSENT and metrics_path == output_dir / _EVALUATION_METRICS
                and any(row.get("designed_structure_path") for row in rows)):
            reason = MetricsBlocker.TABLE_ONLY.value
        sequence = _with_reason(sequence, reason, "sequence_blocker")
        interface = _with_reason(interface, reason, "interface_blocker")
        interface_quality = _with_reason(interface_quality, reason, "interface_quality_blocker")
    chemical = _first_chemical(samples)
    modality = _first_modality(samples)
    record: dict[str, object] = {
        "screen_version": SCREEN_VERSION,
        "cuts": _applied_cuts(limits, kind),
        # Which table the per-design numbers came from. The primary file wins
        # and the reeval file stands in when it is absent, so two verdicts on
        # one run can rest on different measurements with nothing saying so.
        "metrics_file": metrics_path.relative_to(output_dir).as_posix() if metrics_path is not None else None,
        # Every file supplied beside the run, read or refused. A verdict
        # that cannot name its inputs cannot be checked against them.
        "inputs": _supplied_inputs(output_dir),
        # Which code applied the cuts and read the tables. A re-screen
        # that changes the verdict while naming no changed input is
        # pointing here, and the record could not say where.
        "code": code_version(),
        # Every pass criterion this repository declares, with the numbers
        # behind it. Nine of the eleven were evaluated, written into this
        # same table, and discarded; a bare ppi_pass of false named none
        # of the three bounds it missed. Recorded, like ppi_pass, without
        # moving the call: they are the evaluation's verdict, not this
        # screen's.
        "refold": refold,
        "call": combination.call,
        "bar": combination.bar,
        # Which axes that bar required. The name alone does not survive an
        # edit to BARS, and a stored screened_clear has to keep meaning
        # what it meant when it was written.
        "bar_axes": _bar_requirement(binder_kind, combination.bar),
        "states": dict(combination.states),
        "untested": list(combination.untested),
        "hits": list(combination.hits),
        "ppi_pass": ppi_pass,
        "ppi_blocker": ppi_blocker,
        "pre_oracle_pass": _flag_column(coverage_rows, _PRE_ORACLE_COLUMN),
        "sequence_developability": sequence,
        "designed_interface": interface,
        "designed_interface_quality": _with_interface_thresholds(interface_quality),
        "samples": samples,
        "requested_class": "requested_clear",
    }
    if selection_record is not None:
        record["method_selection"] = selection_record
    if affinity is not None:
        record["prodigy_affinity"] = affinity.as_record()
    if vina is not None:
        record["vina_pose"] = vina.as_record()
    if tnp is not None:
        record["tnp_profile"] = tnp.as_record()
    if metrics_blocker is not None:
        # Why the per-design table yielded nothing. An empty read used to be
        # the same record as a run that was never scored, so a table refused
        # for a repeated column looked like one nobody had asked for.
        record["metrics_blocker"] = metrics_blocker
    if len(_named_chains(rows)) > 1:
        # A property of the table rather than of a design, and recorded
        # where one: the top-level blocks repeat the first design, so a
        # run whose later designs lost their chain said nothing at all
        # about why their interfaces came back empty.
        record["chain_blocker"] = "chain_disagreement"
    if chemical is not None:
        record["chemical_liability"] = chemical
    if modality is not None:
        record["modality_screen"] = modality
    # The one measurement about the set rather than about a design. Every
    # other screen reads one chain, so a batch whose replicates came back as
    # the same molecule passes all of them once per copy and reads as that
    # many separate shots.
    spread = design_diversity(chains)
    if spread is not None:
        record["design_diversity"] = spread
    # A supplied table is reported with its method and its study, and judges
    # nothing: the scales are not comparable and no published threshold spans
    # them. Native Protein-Sol reports carry per-sequence predictions separately.
    # The sequence AGGRESCAN is on the modality report.
    record["solubility"] = read_solubility(output_dir) if "solubility" in active else {
        "solubility_blocker": "not_selected_or_inapplicable"
    }
    protein_sol = read_protein_sol(output_dir, chains) if "protein_sol" in active else None
    if protein_sol is not None:
        record["protein_sol"] = protein_sol
    synthesis = _synthesis_block(binder_kind, chains)
    if synthesis is not None:
        record["synthesis"] = synthesis
    ligand_alerts = None
    if "ligand_alerts" in active:
        if selection_record is not None and (output_dir / "ligand_candidates.tsv").is_file():
            from rfproteina.disease import screen_assays

            ligand_alerts = screen_assays.safe_measure(output_dir, "ligand_alerts", None)
        else:
            ligand_alerts = _ligand_alerts_block(output_dir)
    if ligand_alerts is not None:
        record["ligand_alerts"] = ligand_alerts
    if kind == "ligand" and "ligand_solubility" in active:
        from rfproteina.disease.ligand_solubility import screen_run

        ligand_solubility = screen_run(output_dir)
        if ligand_solubility is not None:
            record["ligand_solubility"] = ligand_solubility
    from rfproteina.disease import ligand_metabolites, qupkake

    pka = (qupkake.screen_run(output_dir, kind=kind, use_defaults=selection_record is not None)
           if "qupkake" in active else None)
    if pka is not None:
        record["qupkake"] = pka
    metabolites = ligand_metabolites.screen_run(
        output_dir, kind=kind, use_defaults=selection_record is not None
    ) if "sygma" in active else None
    if metabolites is not None:
        record["sygma"] = metabolites
    mhc_class_i = _mhc_class_i_block(binder_kind, chains, output_dir, class_i_scan)
    if mhc_class_i is not None:
        record["mhc_class_i"] = mhc_class_i
    if liability is not None:
        record["sequence_liability"] = liability
    if nucleic is not None:
        record["nucleic_liability"] = nucleic
    if target is not None:
        record["target_liability"] = target
    if toxicology is not None:
        record["toxicology"] = toxicology
    if read_accession:
        record["target_accession"] = read_accession
    mismatch = _job_mismatch(output_dir, binder_kind, target_id)
    if mismatch is not None:
        record["job_mismatch"] = mismatch
    if selection_record is not None:
        from rfproteina.disease import screen_assays, screen_inventory, screen_physical

        evidence = screen_inventory.Evidence(output_dir, None, record)
        selection_record["recorded_evidence"] = {
            c.method.key: evidence.coverage(screen_inventory.BY_KEY[c.method.key], include_values=True)
            for c in choices if c.method.key in active and c.method.operation == "recorded_evidence"
        }
        selection_record["prepared_measurements"] = {
            c.method.key: screen_assays.safe_measure(output_dir, c.method.key, None)
            for c in choices if c.method.key in active and c.method.key in screen_assays.FILES
            and c.method.operation != "recorded_evidence"
        }
        structures = screen_physical.Structures(output_dir)
        selection_record["structure_measurements"] = {
            key: structures.measure(key) for key in selected if key in active and key in screen_physical.METHODS
        }
    text = scrubbed_json(record)
    # A re-screen overwrites the file outright, so the verdict it replaced left
    # no trace: a table dropped in afterwards could turn axis_hit into
    # screened_clear and nothing on disk remembered the hit. Recorded only when
    # the verdict actually changed, which is also what keeps screening twice
    # byte-identical.
    previous, prior_change = _previous_verdict(output_dir)
    if previous is not None and previous["digest"] != _text_digest(text):
        changed = _changed_inputs(previous.pop("inputs", None), record["inputs"])
        if changed:
            # Which of them moved. A verdict that changed because a
            # predictor table was swapped is a different event from one
            # that changed with nothing on disk touched.
            previous["changed_inputs"] = changed
        record["replaced"] = previous
        text = scrubbed_json(record)
    elif prior_change is not None:
        record["replaced"] = prior_change
        text = scrubbed_json(record)
    write_text_atomically(output_dir / _SCREEN, text)
    return record


@dataclass(frozen=True)
class _Evidence:
    """Curated axis states, and the basis each one was curated on.

    One object rather than two dictionaries threaded through four callers:
    the state and the reason for it are the same fact, and nothing here
    should be able to carry one without the other.
    """

    states: dict[str, str]
    basis: dict[str, dict[str, str]]


def _bar_requirement(binder_kind: str, bar: str | None) -> object:
    """The axes this verdict had to clear, read from ``BARS`` itself.

    ``cuts`` records every number the verdict was taken against, because
    editing one of those constants later changes the meaning of every stored
    screen. A bar is the same hazard and was not recorded: add an axis to
    ``BARS['protein']['sequence']`` and every stored ``screened_clear``
    silently starts claiming more than it was ever checked for, while
    removing one makes it claim less. The list is read at the moment of the
    verdict, so what is written cannot differ from what was applied.

    ``none`` is the bar of a panel target, which names no axes, and a binder
    of kind other has no bars at all.
    """
    if bar is None or bar == "none":
        return _ABSENT
    try:
        return list(bar_axes(binder_kind, bar))
    except (ValueError, KeyError):
        return _ABSENT


def _supplied_inputs(output_dir: Path) -> list[dict[str, object]]:
    """Every file supplied beside the run, with its size and a digest.

    A verdict rests on these: the predictor tables, the toxicology panel, the
    co-evolution notes, the solubility scores, the curated evidence. Two
    inputs already carried a digest -- the proteome and the off-target
    reference -- and the reason given there holds for every one of them: a
    swapped or rebuilt table is otherwise invisible between two runs, and the
    screen it changed can say only that the verdict changed.

    Recorded whether or not an axis read the file, because a table that was
    present and refused is as much a part of the record as one that decided
    an axis. Only the run folder's own files: the designs and the metrics
    table are named elsewhere in this record, and digesting a directory tree
    on every screen would cost more than it tells.

    A supplied file that is a symlink is digested like any other and marked
    ``link``. It was skipped, while every reader here follows a link, so the
    file decided the verdict and the record denied it existed -- and a
    ``replaced`` then said the verdict had moved with nothing on disk moving.
    The link's target is not written down: it is a path outside the run and
    often a personal one, and the digest already says whether the bytes moved.
    """
    try:
        found = sorted(path for path in output_dir.iterdir() if path.is_file())
    except OSError:
        # Withheld whole, for the same reason the limit below withholds: an
        # empty list is the answer "this run supplied nothing", so every table
        # that did decide the verdict read as absent and the next re-screen
        # reported all of them as removed -- a change that never happened. A
        # run folder can be writable and readable by name and still refuse to
        # be listed: a directory carrying execute without read opens every
        # file this screen reads and raises here, and the screen then measures
        # the whole run while claiming it was handed nothing.
        return [{"file": _UNLISTED}]
    found = [path for path in found if path.name != _SCREEN]
    if len(found) > _INPUT_LIMIT:
        # Withheld whole rather than cut at the limit. A list truncated
        # in silence reads as a complete one, so every file past the cut
        # would look absent and the next re-screen would report them all
        # as removed -- a change that never happened.
        return [{"file": _TRUNCATED, "files": len(found)}]
    inputs: list[dict[str, object]] = []
    for path in found:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        item: dict[str, object] = {"file": path.name, "bytes": size}
        item["digest"] = _file_digest(path) if size <= _INPUT_BYTES else "file_too_large"
        if path.is_symlink():
            # Said rather than passed over: the bytes behind this name live
            # somewhere else, so editing them changes the verdict without
            # anything in the run folder being touched.
            item["link"] = True
        inputs.append(item)
    return inputs


def _samples_root(output_dir: Path) -> Path:
    inference = output_dir / "inference_samples"
    return inference if inference.is_dir() else output_dir / "evaluation_samples"


def _designs_blocker(output_dir: Path) -> str:
    """Why this run has no design to measure.

    Asked only once ``_generated_cifs`` has come back empty, which is what makes
    the answer sound: by then any packed design in the folder is one that could
    not be read, so naming it is naming the reason. Asked of a run that does
    hold a readable design, it would report the compressed sibling beside it,
    which is why nothing asks it then.

    A design this screen cannot decode is named rather than passed off as an
    absent one: the two have different remedies, and a run written with
    ``compress_outputs=true`` used to measure nothing and say nothing.
    """
    root = _samples_root(output_dir)
    if not root.is_dir():
        return _DESIGNS_ABSENT
    for pattern in _PACKED_DESIGNS:
        try:
            if next(root.rglob(pattern), None) is not None:
                return _DESIGNS_COMPRESSED
        except OSError:
            return _DESIGNS_ABSENT
    return _DESIGNS_ABSENT


@cache
def _read_errors() -> tuple[type[BaseException], ...]:
    """Every error a design read can raise, the zstd module's own included.

    ``zstandard.ZstdError`` subclasses ``Exception`` and not ``ValueError``, so
    catching the usual read errors let a mislabelled ``.cif.zst`` -- a plain CIF
    renamed, which is what a careless copy produces -- escape the read and take
    the whole screen with it.
    """
    errors: tuple[type[BaseException], ...] = (OSError, EOFError, ValueError, MemoryError, zlib.error)
    found = getattr(_zstandard(), "ZstdError", None)
    if isinstance(found, type) and issubclass(found, BaseException):
        return (*errors, found)
    return errors


def _design_opens(path: Path) -> bool:
    """Whether a compressed design decompresses at all.

    ``_generated_cifs`` lists the designs this screen can read, and plain bytes
    under an archive's name are not one. Counting it would make a sample out of
    a file that yields no text, where the run should say instead that it holds a
    design it could not read -- which is what ``designs_compressed`` says, and
    what this screen said for every compressed design before it read any.

    Only the first chunk is decompressed. An archive whose header is good and
    whose tail is truncated passes here and is refused by the full read, which
    is right: that one really is a damaged compressed design rather than a file
    that was never compressed.
    """
    try:
        if path.name.endswith(".zst"):
            module = _zstandard()
            if module is None:
                return False
            with (
                path.open("rb") as handle,
                module.ZstdDecompressor().stream_reader(handle, read_across_frames=True) as reader,
            ):
                reader.read(_PROBE_BYTES)
        elif path.name.endswith((".gz", ".gzip")):
            with gzip.open(path, "rb") as handle:
                handle.read(_PROBE_BYTES)
    except _read_errors():
        return False
    return True


@cache
def _zstandard() -> object | None:
    """The ``zstandard`` module, or None where it is not installed.

    It arrives with the model stack rather than with this package, and the
    screen is meant to run on a machine that only reads a finished run. Where it
    is missing a ``.cif.zst`` design is named by ``designs_compressed`` -- which
    is what the screen did for every compressed design before -- instead of
    being found and then silently yielding nothing.
    """
    try:
        import zstandard
    except ImportError:
        return None
    return zstandard


def _complete_zstd_frames(module: object, handle: io.BufferedReader, expected: bytes) -> bool:
    """Verify the end of every frame and the bytes returned by the bounded read."""
    handle.seek(0)
    decoder = module.ZstdDecompressor()
    frame = decoder.decompressobj()
    offset = 0
    for block in iter(lambda: handle.read(4096), b""):
        while block:
            if frame.eof:
                frame = decoder.decompressobj()
            decoded = frame.decompress(block)
            if not expected.startswith(decoded, offset):
                return False
            offset += len(decoded)
            block = frame.unused_data
    return frame.eof and offset == len(expected)


class DesignTextBudgetError(ValueError):
    """Interactive inspection reached its byte budget before the end of a structure."""


def _design_text(path: Path, *, max_bytes: int | None = None) -> str | None:
    """CIF text, decompressed by the suffix the file carries.

    Used for a generated design and for a structure the user supplied, because
    both arrive compressed. ``combinations._STRUCTURE_SUFFIXES`` -- every
    encoding AtomWorks accepts -- lists ``.gzip`` beside ``.gz``, and the
    documented way to run this model is ``inputs=...monomer-short.cif.zst``, so
    a plain text read of a condition spec's input resolved no accession at all.

    None when the file will not open, will not decompress, or will not decode,
    which each caller reports in its own vocabulary rather than passing off as
    an absent measurement.
    """
    name = path.name
    compressed = name.endswith((".zst", ".gz", ".gzip"))
    limit = _DESIGN_TEXT_LIMIT if max_bytes is None else min(max_bytes, _DESIGN_TEXT_LIMIT)
    try:
        if name.endswith(".zst"):
            module = _zstandard()
            if module is None:
                return None
            with path.open("rb") as handle:
                # ``stream_reader`` rather than ``decompress``, which demands an
                # embedded content size that these writers do not always set.
                with module.ZstdDecompressor().stream_reader(handle, read_across_frames=True, closefd=False) as reader:
                    raw = reader.read(limit + 1)
                if len(raw) <= limit and not _complete_zstd_frames(module, handle, raw):
                    return None
        elif name.endswith((".gz", ".gzip")):
            with gzip.open(path, "rb") as handle:
                raw = handle.read(limit + 1)
        elif max_bytes is not None:
            with path.open("rb") as handle:
                raw = handle.read(max_bytes + 1)
        else:
            # Read whole and unbounded, exactly as this did before compression
            # was understood at all. A plain file's size on disk is its size,
            # so there is nothing here to guard against and a bound would only
            # refuse a large design that used to be measured.
            raw = path.read_bytes()
    except _read_errors():
        return None
    if max_bytes is not None and len(raw) > max_bytes:
        raise DesignTextBudgetError
    if compressed and len(raw) > _DESIGN_TEXT_LIMIT:
        # A compressed file's size on disk does not bound what it expands to, so
        # the bound is taken after. A design is one structure's coordinates, so
        # anything past this is a different file and is refused like any other
        # unreadable one.
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeError:
        return None


def _design_stem(name: str) -> str:
    """A design file's name with its generated suffix and compression removed."""
    for suffix in _DESIGN_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _file_digest(path: Path) -> str:
    """A short digest of one supplied file, read in blocks."""
    found = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                found.update(block)
    except OSError:
        return "unreadable"
    return found.hexdigest()[:16]


def _changed_inputs(before: object, after: object) -> list[dict[str, object]]:
    """Supplied files that differ between the replaced screen and this one.

    ``replaced`` said that the verdict changed. It could not say what changed,
    and that is the part a reader can act on: a verdict that moved because a
    predictor table was swapped is a different event from one that moved
    because a table was added, or removed, or because nothing on disk moved
    at all and the code did.
    """
    if _inputs_withheld(before) or _inputs_withheld(after):
        # One side is not a list of files, so nothing can be said about
        # what moved. Saying nothing is the only honest answer; saying
        # everything moved would be the alternative.
        return []
    old = _inputs_by_name(before)
    new = _inputs_by_name(after)
    found: list[dict[str, object]] = []
    for name in sorted(set(old) | set(new)):
        was = old.get(name)
        now = new.get(name)
        if was == now:
            continue
        change = "added" if was is None else "removed" if now is None else "edited"
        found.append({"file": name, "change": change})
    return found


def _inputs_withheld(value: object) -> bool:
    """Whether this screen withheld its input list rather than listing it.

    Either stand-in counts, and they are read from one set rather than named
    here: a list past the file limit, and a folder this screen could not list.
    The second was a bare empty list, which is indistinguishable from a run
    that supplied nothing, so the comparison below read it as every file
    having been removed.
    """
    if not isinstance(value, list):
        return False
    return any(isinstance(item, Mapping) and item.get("file") in _WITHHELD_INPUTS for item in value)


def _inputs_by_name(value: object) -> dict[str, str]:
    """One digest per supplied file, from either screen's own record."""
    found: dict[str, str] = {}
    if not isinstance(value, list):
        return found
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = item.get("file")
        digest = item.get("digest")
        if isinstance(name, str) and name and isinstance(digest, str):
            found[name] = digest
    return found


def _text_digest(text: str) -> str:
    """A short digest of a screen document, for comparing two of them."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _previous_verdict(output_dir: Path) -> tuple[dict[str, object] | None, dict | None]:
    """The current verdict and its last recorded change, if available.

    Digested without its own ``replaced`` entry, so the comparison is between
    the two verdicts rather than between their chains. Without that, each
    re-screen would differ from the last purely by carrying one more link.
    """
    try:
        payload = json.loads((output_dir / _SCREEN).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    stripped = {key: value for key, value in payload.items() if key != "replaced"}
    call = payload.get("call")
    return {
        "digest": _text_digest(scrubbed_json(stripped)),
        "call": call if isinstance(call, str) else _ABSENT,
        # Popped by the caller before this is stored: it is here to be
        # compared against the new inputs, not to be written twice.
        "inputs": payload.get("inputs"),
    }, payload.get("replaced") if isinstance(payload.get("replaced"), dict) else None


def _applied_cuts(limits: TargetLimits | None, kind: str | None = None) -> dict[str, object]:
    """Every number this verdict was taken against.

    A verdict is only reproducible against the cuts that produced it. Edit
    ``_GRAVY_MAX`` or a rank threshold later and every stored screen silently
    changes meaning, with nothing on disk saying which revision applied.

    The named constants are imported rather than retyped, so what is recorded
    cannot drift from what was used -- which is the whole point of recording
    it. The three target cuts are whatever the caller passed, ``null`` when it
    passed none, because those have no published value to fall back on.

    ``ligand_panel_nm_max`` is the whole per-axis panel table on a ligand
    screen, because one number said nothing about which of ten proteins it had
    been applied to. Any other kind has no panel, and its record must not name
    panel proteins at all, so it keeps the panel-wide fallback.
    """
    bar = limits if limits is not None else TargetLimits()
    cuts: dict[str, object] = {
        "gravy_max": _GRAVY_MAX,
        # Which of these two cuts is published and which is this program's own
        # is not something a reader can tell from the numbers, and the pass
        # they decide is read as though both were published.
        "gravy_max_source": _GRAVY_MAX_SOURCE,
        "instability_max": _INSTABILITY_MAX,
        "instability_max_source": _INSTABILITY_MAX_SOURCE,
        "clash_distance_angstrom": _CLASH_THRESHOLD,
        "mhc_i_rank_max": _MHC_I_RANK,
        "mhc_i_ic50_nm_max": _MHC_I_IC50,
        "mhc_ii_rank_max": _MHC_II_RANK,
        "mhc_ii_ic50_nm_max": _MHC_II_IC50,
        "ada_median_rank_max": _DR_MEDIAN_MAX,
        # One number for ten proteins said nothing about which axis it applied
        # to, so a ligand screen stores the whole per-axis table and each
        # ``toxicology`` record carries its own cut and that cut's source.
        "ligand_panel_nm_max": dict(PANEL_NM) if kind == "ligand" else HIT_NM,
        "plasma_abundance_max": bar.abundance_max,
        "plasma_hub_max": bar.hub_max,
        "plasma_fold_max": bar.fold_max,
    }
    # Two of these twelve named a source; the other ten were bare numbers, and
    # a reader cannot tell a published rule from a number this program chose by
    # looking at 40 and 0.1. ``ada_median_rank_max`` is the case that matters:
    # it IS the Paul 2015 value, and it is still refused on most machines
    # because it is defined on a rank scale the installed predictor does not
    # produce. Stored under one key rather than three dozen flattened ones, so
    # a consumer holding an allowlist of record keys gains a single entry.
    cuts[PROVENANCE_KEY] = describe_cuts(cuts)
    return cuts


def measured_contact_ms(screen: Mapping[str, object]) -> float | None:
    """Target-side contact molecular surface rounded to four decimals.

    A value above 0 is an interface. Zero is no interface. ``None`` was not measured.
    """
    interface = screen.get("designed_interface")
    if not isinstance(interface, dict):
        return None
    return _round4(interface.get("contact_ms"))


def format_screen_lines(entry_id: str, chain: str, screen: Mapping[str, object]) -> list[str]:
    """Terminal lines for one screen. An empty axis list omits that sentence."""
    lines = [_call_line(f"Screen {entry_id} chain {chain}", screen)]
    measured = _measured_line(entry_id, chain, screen)
    if measured is not None:
        lines.append(measured)
    return lines


def screen_lines_for_directory(directory: Path) -> list[str]:
    """Lines for ``screen.json`` in ``directory``. A missing or broken file is empty.

    The entry and chain come from ``generation.json`` when it names them, and
    otherwise from a ``group_entry_chain`` folder name. Personal paths in the
    stored text are removed before the lines are returned.
    """
    try:
        payload = read_screen(directory)
        if payload is None:
            return []
        return format_screen_lines_for_directory(directory, payload)
    except Exception:
        # Every exception, and around the formatting as well as the read, which
        # is what makes "a broken file is empty" true for every way a file can
        # be broken. `read_screen` names only OSError, UnicodeError and
        # JSONDecodeError, and a file it accepts can still fail afterwards: a
        # deeply nested document raises RecursionError out of `json.loads`, and
        # a screen whose `sequence_developability` carries `sequence_pass`
        # without `gravy` raises KeyError out of `_measured_line`.
        #
        # Isolating it per folder is the whole point. Caught any further out --
        # around the listing, or in the page that draws it -- one unreadable
        # file discards every valid report gathered before it, so a folder
        # holding eight good screens and one bad one reported none at all.
        return []


def read_screen(directory: Path) -> Mapping[str, object] | None:
    """The stored screen, or None when this directory has none.

    Raises ``ValueError`` when the file is there and is not a screen. The two
    are different answers and a caller about to overwrite the file needs to
    tell them apart: a screen that cannot be read is evidence of something, and
    replacing it destroys that. A caller only rendering what it finds can treat
    both as nothing to show.
    """
    path = directory / _SCREEN
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not a readable screen: {exc}") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("call"), str):
        raise ValueError(f"{path.name} is not a screen.")
    return payload


def format_screen_lines_for_directory(directory: Path, screen: Mapping[str, object]) -> list[str]:
    """Scrubbed lines for a screen, named the way this directory names itself.

    One implementation, so a screen written now and the same screen read back
    later produce the same sentences, and no caller has to repeat the choice
    between a named entry and an unnamed one.
    """
    entry, chain = _names_for_directory(directory)
    lines = format_screen_lines(entry, chain, screen) if entry and chain else _unnamed_lines(screen)
    return [scrub_for_upload(line) for line in lines]


def screen_reports_for_directory(
    directory: Path,
    *,
    limit: int = _SCREEN_REPORT_LIMIT,
) -> list[tuple[str, list[str]]]:
    """Screens to show for this folder.

    The folder's own ``screen.json`` is reported first with an empty label, and
    every generation folder underneath contributes one report after it,
    shallower paths first, labelled by its path relative to ``directory``. A
    broken file is skipped, and a truncated listing says so rather than reading
    as the whole set.
    """
    # A folder's own screen comes first and unlabelled, which is what opening
    # one folder should show. It no longer ENDS the search: a campaign root that
    # happened to hold a screen.json -- copied there, or left by an earlier
    # layout -- returned that one file as the only report and hid every run
    # beneath it, so a hit two levels down was invisible on the page.
    found: list[tuple[str, list[str]]] = []
    own = screen_lines_for_directory(directory)
    if own:
        found.append(("", own))
    if limit < 1:
        return found
    # Keyed on the sentences alone, deliberately: the archive stores the same
    # run under a campaign path as well as its own folder, and the two read
    # identically. Collapsing that is the point. What was missing is saying so,
    # below, rather than changing which reports survive.
    seen: set[tuple[str, ...]] = set()
    collapsed: list[str] = []
    over_limit = False
    gave_up = False
    try:
        for current, filenames in walk_for_screens(directory):
            if _SCREEN not in filenames or current == directory:
                # The folder's own screen is already the first report; the walk
                # yields it too, and reporting it twice is not two runs.
                continue
            lines = screen_lines_for_directory(current)
            if not lines:
                continue
            label = label_for(directory, current)
            key = tuple(lines)
            if key in seen:
                collapsed.append(label)
                continue
            # The cap is tested here, not at the top of the loop. Up there it
            # fired for whatever directory came next whether or not it held a
            # screen, so one screen-less folder visited after the last screen --
            # generations/, a notes folder -- turned a complete listing into a
            # claim that screens had been hidden.
            if len(found) >= limit:
                # Not a `break`. The walk reports a pruned subtree by raising
                # once it has finished, and abandoning the generator here threw
                # that signal away: an over-limit listing then said only that
                # there were more screens than shown, and never that whole
                # folders had gone unread. The panel's own limit is small, so
                # that was the usual case rather than the rare one. Draining the
                # rest walks what it was already going to walk; no further
                # report is added.
                over_limit = True
                continue
            seen.add(key)
            found.append((label, lines))
    except ScreenWalkTruncatedError:
        # A different fact from the report limit: the walk gave up part way
        # through the folders, so the reports shown may be fewer than the limit
        # and still not be all of them.
        gave_up = True
    if over_limit or gave_up or collapsed:
        # Silence read as "this is all of them", so a hit past the cap was
        # dropped without a word. This is a fact about the listing, not about
        # the run it is filed under, so it is prefixed to say so, and it names
        # the folded paths -- a count alone left no way to check them.
        notice = [SCREEN_LISTING_PREFIX]
        if over_limit:
            notice.append(f"more screens than the {limit} shown.")
        if gave_up:
            notice.append("stopped before every folder was read.")
        if collapsed:
            folded = ", ".join(sorted(collapsed))
            notice.append(f"collapsed as reading identically: {folded}.")
        line = " ".join(notice)
        if found:
            found[-1] = (found[-1][0], [*found[-1][1], line])
        else:
            # Nothing to ride on. Reporting nothing here is the worst case of
            # all: the search stopped before it found a single screen, and the
            # page would then be silent about having given up looking.
            found.append(("", [line]))
    return found


def walk_for_screens(directory: Path) -> Iterator[tuple[Path, list[str]]]:
    """Every directory worth looking in beneath ``directory``, and its filenames.

    One traversal for the two things that need it: listing the screens already
    stored, and finding the runs that need screening. They differ only in what
    they look for in each directory, so the depth limit, the visit cap, the
    skipped heavy trees and the sorted order live here once.

    ``directory`` itself is never yielded; a caller handles that case first,
    because "this folder is the answer" and "the answers are underneath" are
    different questions.
    """
    if not directory.is_dir():
        return
    try:
        walker = directory.walk(top_down=True, on_error=_ignore_screen_walk, follow_symlinks=False)
    except OSError:
        return
    try:
        root = directory.resolve()
    except (OSError, RuntimeError):
        root = directory
    visited = 0
    abandoned = False
    asked = 0
    for dirpath, dirnames, filenames in walker:
        visited += 1
        if visited > _SCREEN_WALK_CAP:
            raise ScreenWalkTruncatedError([], visited - 1)
        current = Path(dirpath)
        try:
            depth = len(current.relative_to(directory).parts)
        except ValueError:
            dirnames.clear()
            continue
        kept = sorted(name for name in dirnames if name not in _SKIP_SCREEN_WALK and not name.startswith("."))
        # Two prunings that pruned in silence, which is the one thing this walk
        # is not allowed to do. A name beginning with "." is dropped here, and a
        # symlinked directory is listed but never descended into because the
        # walk does not follow links -- so an archive kept on another volume and
        # linked in, or filed under `.archive`, read as a tree with nothing in
        # it: "No finished generation was found", exit 0, `complete: true`. The
        # depth cap already asks whether a pruned subtree hides a run; these ask
        # the same question, against the same small budget, and the answer rides
        # on the existing truncation signal rather than on a new one.
        unread = [current / name for name in dirnames if name not in _SKIP_SCREEN_WALK and name not in kept]
        # A walk that does not follow links counts a symlink to a directory as a
        # file, so it arrives among the filenames and never in `dirnames` at
        # all: that is the entry an archive on another volume is linked in as.
        # A link to a folder this walk reads anyway, under its own name, hides
        # nothing: `current -> gene_1ABC_1` beside the run it points at is an
        # ordinary layout, and asking about it called every such sweep partial.
        unread.extend(
            current / name
            for name in filenames
            if name not in _SKIP_SCREEN_WALK
            and _is_linked_dir(current / name)
            and not _read_under_its_own_name(root, current / name)
        )
        dirnames[:] = kept
        # Nothing is asked about what sits inside a run. The run is already the
        # answer, and a per-run `.hydra` in every run of an archive would spend
        # the whole budget on folders that cannot hold another run to screen.
        if unread and not _is_generation_directory(current):
            for entry in unread:
                if abandoned or asked >= _SCREEN_PRUNE_CHECKS:
                    break
                asked += 1
                # Only a run actually seen counts. A look that ran out of budget
                # is not one: a tool cache or a linked dataset holds hundreds of
                # folders and no run, and treating "did not finish looking" as
                # "may hide a run" made every sweep of such a tree partial, with
                # no way to clear it. That is the false gap described below.
                abandoned = _find_generation(entry) is True
        if depth >= _SCREEN_DEPTH:
            # The depth limit was the one cap that pruned in silence, so a sweep
            # that never looked at a run still reported itself complete: the
            # caller said "Screened N of N run(s), skipped 0" and exited 0. The
            # tool does this to itself -- `--follow-up` plans the next round into
            # `<out_dir>/follow_up/...`, which puts those runs a level past the
            # limit -- so after running a printed follow-up command, screening
            # the same root again silently did not see the new round.
            #
            # What counts as abandoned is a run actually going unseen, not
            # merely a folder going unopened. Two things made the cruder test
            # wrong in both directions.
            #
            # A run's own `inference_samples` and `inference_metrics` are pruned
            # here too, and losing those loses nothing: the run is already the
            # answer and the own-first rule means nothing looks inside one. The
            # archive keeps runs at exactly this depth, so counting their
            # contents reported every complete sweep as partial.
            #
            # And a folder having children says nothing about runs. This tool
            # builds such a tree itself: `--screen --follow-up` writes specs
            # into `<out_dir>/follow_up/generations/<disease>/<target>/...`,
            # which puts a spec-only folder at exactly this depth, so every
            # later sweep of the same root claimed truncation and exited 1
            # while there was nothing under there to screen. Announcing a false
            # gap on every run is worse than the silence this replaced, because
            # it cannot be cleared and it teaches the warning to be ignored.
            if dirnames and not _is_generation_directory(current):
                abandoned = abandoned or any(_holds_generation(current / name) for name in dirnames)
            dirnames.clear()
        if depth > _SCREEN_DEPTH or current == directory:
            continue
        yield current, list(filenames)
    if abandoned:
        # Raised after everything that was found has been yielded, so a caller
        # screens what the walk did reach and still learns the sweep was partial.
        raise ScreenWalkTruncatedError([], visited)


def _is_linked_dir(path: Path) -> bool:
    """Whether this entry is a symlink pointing at a directory.

    A link whose target cannot be reached, because a folder on the way to it
    may not be entered, is not one: ``is_dir`` raises for that, and one such
    link on a shared volume must not stop every run in the tree from being
    screened. A folder that cannot be opened is skipped the same way.
    """
    try:
        return path.is_symlink() and path.is_dir()
    except OSError:
        return False


def _read_under_its_own_name(root: Path, link: Path) -> bool:
    """Whether a linked folder is one the walk reads anyway, so the link hides nothing.

    True for a target inside the walked tree on a path the walk does not prune,
    and for the tree itself or a folder above it, which is a loop and not an
    archive. A target beneath a pruned folder is not read under any name, and
    one outside the tree is not read at all: both are asked about.
    """
    try:
        target = link.resolve()
    except (OSError, RuntimeError):
        return True
    if target == root or target in root.parents:
        return True
    try:
        parts = target.relative_to(root).parts
    except ValueError:
        return False
    return not any(part in _SKIP_SCREEN_WALK or part.startswith(".") for part in parts)


def _find_generation(directory: Path) -> bool | None:
    """Look for a finished run at or beneath ``directory``, within a small budget.

    True when one was seen, False when the whole subtree was read and held none,
    and None when the budget ran out first. The callers differ in what they make
    of None, which is why it is not folded into either answer here.
    """
    try:
        walker = directory.walk(top_down=True, on_error=_ignore_screen_walk, follow_symlinks=False)
    except OSError:
        return False
    for seen, (dirpath, dirnames, _filenames) in enumerate(walker, start=1):
        if seen > _SCREEN_LOOKAHEAD:
            return None
        dirnames[:] = [name for name in dirnames if name not in _SKIP_SCREEN_WALK and not name.startswith(".")]
        if _is_generation_directory(Path(dirpath)):
            return True
    return False


def _holds_generation(directory: Path) -> bool:
    """Whether a finished run sits at or beneath ``directory``.

    Asked only where the depth limit is about to prune, to tell a subtree that
    really does hide a run from one that merely has folders in it. The limit
    itself is unchanged: this looks, it does not yield, so which directories
    the walk reports is exactly what it was.

    Bounded by its own small budget, and an exhausted budget answers True.
    That is the safe direction: the claim then made is that the sweep may be
    partial, which is the honest answer when the question ran out of looking.
    """
    return _find_generation(directory) is not False


class ScreenWalkTruncatedError(Exception):
    """The search stopped at a cap, so some runs were never looked at.

    Raised rather than returned because the alternative was silence: the walk
    stopped at its visit cap, ``generation_directories`` returned a short list
    or none at all, and the caller said "no finished generation was found" and
    exited successfully. A run under the named directory would simply not be
    screened, and nothing said so.
    """

    def __init__(self, found: list[Path], visited: int) -> None:
        super().__init__(f"stopped after {visited} directories")
        self.found = found
        self.visited = visited


def generation_directories(directory: Path, *, limit: int | None = None) -> list[Path]:
    """Finished runs at or beneath ``directory``, shallower paths first.

    A directory holding designs or metrics is a run, and when the given one is
    itself a run it is the only answer -- the same own-first rule the stored
    screens follow. The test is for what a run leaves behind rather than for a
    ``screen.json``, because a run being screened for the first time has none.
    """
    if _is_generation_directory(directory):
        return [directory]
    # Resolved here rather than as a default, so the cap is the one in force at
    # the time of the call.
    cap = SCREEN_RUN_LIMIT if limit is None else limit
    found: list[Path] = []
    try:
        for current, _filenames in walk_for_screens(directory):
            if len(found) >= cap:
                raise ScreenWalkTruncatedError(found, cap)
            if _is_generation_directory(current):
                found.append(current)
    except ScreenWalkTruncatedError as stopped:
        # Carry what was found so far, so a caller can screen it and still say
        # that the search was cut short.
        raise ScreenWalkTruncatedError(found, stopped.visited) from None
    return found


def _is_generation_directory(directory: Path, *, follow_symlinks: bool = True) -> bool:
    """Recognize a finished run by its sample directory or published metrics table."""
    if follow_symlinks:
        return _samples_root(directory).is_dir() or _metrics_path(directory) is not None
    for name in ("inference_samples", "evaluation_samples"):
        samples = directory / name
        if not samples.is_symlink() and samples.is_dir():
            return True
    for relative in (_PRIMARY_METRICS, _REEVAL_METRICS, _EVALUATION_METRICS):
        table = directory / relative
        if not table.parent.is_symlink() and not table.is_symlink() and table.is_file():
            return True
    return False


def label_for(directory: Path, current: Path) -> str:
    """How a directory beneath another is named in output. Empty for itself."""
    if current == directory:
        return ""
    try:
        return scrub_for_upload(current.relative_to(directory).as_posix())
    except ValueError:
        return scrub_for_upload(current.name)


def _ignore_screen_walk(_error: OSError) -> None:
    """Skip a directory the screen list cannot open."""
    return


def _unnamed_lines(screen: Mapping[str, object]) -> list[str]:
    """The same sentences when the folder does not name an entry and chain."""
    lines = [_call_line("Screen", screen)]
    measured = _measured_line("", "", screen)
    if measured is not None:
        lines.append(measured)
    return lines


def _call_line(opening: str, screen: Mapping[str, object]) -> str:
    """The one sentence a reader takes the verdict from.

    ``opening`` is "Screen <entry> chain <chain>", or "Screen" where the folder
    names neither. One implementation for both, because the two had the same
    sentence written twice and a clause added to one would not reach the other.

    Both lists the judge produced are named, not one. ``untested`` was printed
    and ``hits`` was not, so an ``axis_hit`` named the axes that were not the
    problem and never the one that was. For seven of the twenty-two axes --
    a peptide's ``membrane``, ``paralog``, the three plasma axes and both
    nucleic axes -- that was the only chance: ``_liability_clause`` renders a
    measurement only for the axes its own block table declares, so a measured
    hit on any of them reached no printed line at all. A nucleic design with a
    TLR9 CpG motif printed "axis_hit. Untested: off_target_hybridization." and
    named ``innate_immune`` nowhere, pointing the reader at the one axis that
    was not the finding. The closing page names every axis; the line a
    ``--screen`` run prints named none of them.

    Read from the record's own two lists rather than from a table of blocks,
    so an axis added later, in a block added later, or in no block at all,
    cannot be left out of the sentence. Each name goes through ``one_token``:
    the record is read back off disk, and a name holding a line break would
    start a line of its own that could read as a second verdict.
    """
    line = f"{opening}: {screen['call']}."
    for label, key in (("Hit", "hits"), ("Untested", "untested")):
        named = screen.get(key)
        if isinstance(named, list) and named:
            line = f"{line} {label}: {', '.join(one_token(axis) for axis in named)}."
    return " ".join([line, *_verdict_notes(screen)])


def _verdict_notes(screen: Mapping[str, object]) -> list[str]:
    """Sentences about the verdict rather than about a measurement.

    Both are properties of the run and not of a number, so they ride on the
    call line beside ``Untested:`` rather than among the measurements.

    ``job_mismatch`` was recorded and shown to nobody: a run screened under a
    kind or a target the job file disagrees with printed a bare verdict, and the
    two records describe different runs. A run with no design to measure said
    nothing either, so it read like a run whose chain was never named.
    """
    notes: list[str] = []
    mismatch = screen.get("job_mismatch")
    if isinstance(mismatch, Mapping) and mismatch:
        named = ", ".join(
            f"{one_token(key)} {one_token(value)}" for key, value in sorted(mismatch.items()) if isinstance(key, str)
        )
        if named:
            notes.append(f"Job file disagrees: {named}.")
    sequence = screen.get("sequence_developability")
    blocker = sequence.get("sequence_blocker") if isinstance(sequence, Mapping) else None
    if blocker in _DESIGN_BLOCKERS:
        # About the designs only. A run with no design can still carry a metrics
        # table and curated axes, and "nothing measured" would deny both.
        notes.append(f"No design measured: {blocker}.")
    return notes


def one_token(value: object) -> str:
    """A value from a job file as one short run of text on a printed line.

    The job file is written by another step and is read back as it stands. A
    value holding a line break would otherwise start a line of its own in the
    printed verdict, and that line could read as a second verdict.
    """
    return " ".join(str(value).split())[:80]


def _names_for_directory(directory: Path) -> tuple[str, str]:
    """Entry and chain for a saved generation folder."""
    meta = directory / "generation.json"
    if meta.is_file():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            entry = payload.get("entry_id")
            chain = payload.get("chain")
            if isinstance(entry, str) and isinstance(chain, str) and entry.strip() and chain.strip():
                return entry.strip(), chain.strip()
    # Named from the folder, which is <kind>_<entry>_<chain> and may carry a
    # suffix: gene_8TLD_2_inverse is 8TLD chain 2, inverse-folded. Taking the
    # last two parts read that as "entry 2, chain inverse" -- every number in
    # the line was right and both names were wrong, on the one archived run a
    # reader would go looking for by name. So the entry/chain pair is found by
    # shape rather than by position.
    parts = [part for part in base_name(directory).split("_") if part]
    for first, second in itertools.pairwise(parts):
        if _ENTRY_TOKEN.fullmatch(first) and _CHAIN_TOKEN.fullmatch(second):
            return first, second
    if len(parts) == 3:
        # The canonical shape, correct by construction even when the entry does
        # not look like a PDB id.
        return parts[-2], parts[-1]
    # Anything else is not named rather than named wrongly; _unnamed_lines
    # reports the measurements without claiming an entry or a chain.
    return "", ""


def base_name(directory: Path) -> str:
    """The folder's name without the suffix the archive adds for a collision.

    ``save_generation`` appends ``-2``, ``-3`` and so on rather than overwrite a
    folder, and the files copied inside keep the original stem. So anything
    looking for one of those files has to undo the suffix first, and there is
    one implementation of that here rather than one per caller.
    """
    name = directory.name
    numbered = name.rsplit("-", 1)
    if len(numbered) == 2 and numbered[1].isdigit():
        return numbered[0]
    return name


def job_file_for_directory(output_dir: Path) -> Path | None:
    """The job file this generation was planned by, or None.

    A live run's job file sits beside its output directory; an archived run's
    was copied inside it, under the original stem. Looking only beside meant
    this never found an archived one, so the kind and target it records were
    never compared against what a screen was told -- the check silently did
    nothing for everything under ``generations/``.

    Inside is preferred over beside, because a file inside the folder cannot be
    about a different run while a sibling of a directory can be stale.
    """
    inside = output_dir / f"{base_name(output_dir)}.job.json"
    if inside.is_file():
        return inside
    # The archive substitutes illegal characters and truncates a long stem, so
    # the copied filename need not match the folder name. One candidate is
    # unambiguous; several are not, and guessing between them is worse than None.
    try:
        found = sorted(output_dir.glob("*.job.json"))
    except OSError:
        found = []
    if len(found) == 1:
        return found[0]
    beside = output_dir.parent / f"{output_dir.name}.job.json"
    return beside if beside.is_file() else None


def _default_bar(binder_kind: str) -> str | None:
    """Sequence, coverage, or oligo. A binder of kind other has no bar."""
    return _DEFAULT_BAR.get(_canonical_kind(binder_kind))


def _canonical_kind(binder_kind: str) -> str:
    key = binder_kind.strip().casefold().replace("-", "_").replace(" ", "_")
    return _KIND_ALIAS.get(key, key)


def _job_mismatch(output_dir: Path, binder_kind: str, target_id: str) -> dict[str, object] | None:
    """What the job file recorded, when it disagrees with what was asked here.

    The caller tells this module the kind and the target; the job file beside the
    output directory recorded both when the run was planned. A disagreement means
    the two describe different runs, and until now nothing looked. The caller's
    values stay authoritative, because they are what the measurement used, so
    this only records what the file said. A missing or unreadable job file is not
    a disagreement.
    """
    path = job_file_for_directory(output_dir)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    found: dict[str, object] = {}
    for key, asked in (("binder_kind", binder_kind), ("target_id", target_id)):
        recorded = payload.get(key)
        if isinstance(recorded, str) and recorded and recorded != asked:
            found[key] = recorded
    return found or None


def _apply_sequence_liability(
    output_dir: Path,
    rows: list[dict[str, str]],
    samples: list[dict[str, object]],
    binder_kind: str,
    evidence: _Evidence,
    *,
    proteome: Path | None = None,
    sequences: list[tuple[str, str | None]] | None = None,
) -> dict[str, dict[str, object]] | None:
    """Five axis records for a protein or peptide. Other kinds omit the key.

    A hit or clear from a real input is added to ``evidence`` unless that axis
    is already in ``axis_evidence.json``. The stored state is the one the judge
    sees. The designed sequence is not copied onto the record.

    A supplied state overrides a measurement, and then the measurement's own
    state and blocker are kept under their own names. A blocker says why a
    measurement could not decide, so leaving it beside a state that has been
    decided made the record contradict itself.
    """
    if _canonical_kind(binder_kind) not in _LIABILITY_KINDS:
        return None, None
    # Read once by the caller when it measured them. _designed_sequences is
    # the fallback for a caller that has none.
    pairs = sequences if sequences is not None else _designed_sequences(output_dir, rows)
    # A class I table is written here, before the axis is read, and only when
    # the operator named alleles and the weights are on disk. An existing
    # table is not replaced. A missing predictor leaves the axis untested.
    from rfproteina.disease.mhc_tables import prepare_class_i_table

    try:
        class_i_scan = prepare_class_i_table(output_dir, pairs, binder_kind=_canonical_kind(binder_kind))
    except (OSError, ValueError) as error:
        class_i_scan = {"state": "predictor_failed", "blocker": str(error)}
    from rfproteina.disease.tepitope import prepare_class_ii_table

    with contextlib.suppress(OSError, ValueError):
        prepare_class_ii_table(output_dir, pairs, binder_kind=_canonical_kind(binder_kind))
        # TEPITOPE has matrices for four of the panel's seven alleles, so the
        # table it writes carries no row at all for DRB3*01:01, DRB3*02:02 and
        # DRB4*01:01. The PSSM fitted here from IEDB's CC BY measurements covers
        # all seven and fills the gap at no download. What that buys is
        # coverage, not a verdict: on the protein path the mhc_ii axis is
        # refused by the reference-density guard before allele breadth is
        # reached, so it reads untested either way.
        #
        # The fill refuses any table it did not write, which is what keeps the
        # ADA axis out of this. _declares_consensus is a table-level any(), so
        # one operator row saying "consensus" would hand these rows a scale
        # they were not computed on -- and they are what completes the seven
        # alleles Paul et al.'s median needs, so they would be the reason it
        # started being computed. Hence foreign_table rather than append.
        from rfproteina.disease.class_ii import fill_missing_alleles

        fill_missing_alleles(output_dir, pairs)
    axes, by_example = measure_sequence_liability(
        pairs,
        directory=output_dir,
        proteome=proteome,
        binder_kind=_canonical_kind(binder_kind),
    )
    for sample in samples:
        example_id = sample.get("example_id")
        if isinstance(example_id, str) and example_id in by_example:
            sample["proteome_offtarget"] = dict(by_example[example_id])
    stamped: dict[str, dict[str, object]] = {}
    # The measured axes rather than the fixed five: a peptide also gets
    # membrane, and naming the tuple here would have dropped it.
    for axis in axes:
        record = axes.get(axis)
        if not isinstance(record, dict):
            continue
        stamped[axis] = _stamp_source(axis, record, evidence)
    return stamped, class_i_scan


def _stamp_source(axis: str, record: Mapping[str, object], evidence: _Evidence) -> dict[str, object]:
    """Say where this axis's state came from, and feed a measurement to the judge.

    A supplied state wins and keeps the measurement's own state and blocker
    under their own names, because a blocker explains why a measurement could
    not decide and would contradict a state that has been decided. Only a
    measured hit or clear enters the evidence; an untested measurement adds
    nothing, which is what keeps an unmeasured axis out of a verdict.

    A curated state also records what it was curated on, and says so either
    way: a curated call with nothing behind it is a fact about the record, not
    an absence to pass over.
    """
    item = dict(record)
    if axis in evidence.states:
        item["measured_state"] = item.get("state")
        item["measured_blocker"] = item.get("blocker")
        item["state"] = evidence.states[axis]
        item["blocker"] = None
        item["source"] = "axis_evidence"
        item["curated"] = dict(evidence.basis[axis]) if axis in evidence.basis else _NO_BASIS
    elif item.get("state") in _STATES:
        item["source"] = "measured"
        evidence.states[axis] = str(item["state"])
    else:
        item["source"] = "untested"
    return item


@dataclass(frozen=True)
class _TargetResolution:
    """What the target lookup is allowed to ask, and what the screen may store."""

    lookup_id: str
    read_accession: str | None
    unresolved: str | None


@dataclass(frozen=True)
class _SpecRead:
    """One condition spec's accession, or why it did not yield one."""

    status: str
    accession: str | None = None
    follow: Path | None = None


def _apply_target_liability(
    output_dir: Path,
    binder_kind: str,
    target_id: str,
    evidence: _Evidence,
    *,
    read_json: ReadJson | None,
    limits: TargetLimits | None,
) -> tuple[dict[str, dict[str, object]] | None, str | None]:
    """Paralog and the plasma axes, which are properties of the target.

    These run outside the sequence gate, because they say nothing about the
    designed chain and they apply to kinds that gate excludes. Only an axis this
    kind actually has is written: the judge refuses one foreign to the kind, and
    ``AXES_BY_KIND`` is the authority on which those are.

    The judge and the job comparison keep the caller's ``target_id``. The lookup
    uses a structure accession only when this read one. Without a reader the
    axes that need one say ``network_not_permitted``: a screen does not reach
    the network unless its caller passed a reader. The plasma fold axis reads a
    local table and is measured either way.

    The second value is the accession read from the structure, or None when the
    caller already named one or none could be read.
    """
    kind = _canonical_kind(binder_kind)
    applicable = set(AXES_BY_KIND.get(kind, ()))
    wanted = [axis for axis in TARGET_LIABILITY_AXES if axis in applicable]
    if not wanted:
        return None, None
    resolution = _resolve_target_accession(output_dir, target_id)
    found = measure_target_liability(
        resolution.lookup_id,
        directory=output_dir,
        read_json=read_json,
        limits=limits,
        unresolved=resolution.unresolved,
    )
    stamped = {axis: _stamp_source(axis, found[axis], evidence) for axis in wanted}
    return stamped, resolution.read_accession


def _apply_nucleic_liability(
    output_dir: Path,
    binder_kind: str,
    evidence: _Evidence,
    *,
    sequences: list[tuple[str, str | None]],
    samples: list[dict[str, object]],
    transcriptome: Path | None = None,
) -> dict[str, dict[str, object]] | None:
    """Both nucleic axes for a nucleic binder. Every other kind omits the key.

    ``AXES_BY_KIND`` has given a nucleic binder ``off_target_hybridization``
    and ``innate_immune`` since the classes were defined, and ``BARS`` makes
    both of them the ``oligo`` bar, but nothing measured either: a nucleic
    design came back ``untested`` with no blocker and no file anyone could
    supply to change it. The chains were already read for the drug-type
    report, so this costs no further pass over the designs.

    A measured hit or clear enters ``evidence`` on the same terms as every
    other axis, and ``axis_evidence.json`` still wins.
    """
    if _canonical_kind(binder_kind) != "nucleic":
        return None
    axes, by_example = measure_nucleic_liability(
        sequences,
        directory=output_dir,
        transcriptome=transcriptome,
    )
    for sample in samples:
        example_id = sample.get("example_id")
        if isinstance(example_id, str) and example_id in by_example:
            sample["off_target_hybridization"] = dict(by_example[example_id])
    return {axis: _stamp_source(axis, record, evidence) for axis, record in axes.items()}


def _apply_toxicology(
    output_dir: Path,
    binder_kind: str,
    evidence: _Evidence,
) -> dict[str, dict[str, object]] | None:
    """Panel readings for a ligand. Other kinds omit the key.

    A hit or clear is added to ``evidence`` unless ``axis_evidence.json``
    already names that axis. An untested reading adds nothing, which keeps a
    missing cell from becoming a clear. The stored state is the one the judge
    sees. Protein and peptide evidence never receives a panel axis.
    """
    if _canonical_kind(binder_kind) != "ligand":
        return None
    smiles = _read_ligand_smiles(output_dir)
    if smiles:
        # The Octant subset is already nanomolar. A missing subset, an unknown
        # ligand, or a table the operator already wrote leaves the file alone.
        from rfproteina.disease.octant import prepare_toxicology_table

        with contextlib.suppress(OSError):
            prepare_toxicology_table(output_dir, smiles)
    found = measure_toxicology(output_dir, ligand_smiles=smiles or "")
    return {axis: _stamp_source(axis, found[axis], evidence) for axis in LIGAND_AXES}


def _resolve_target_accession(output_dir: Path, target_id: str) -> _TargetResolution:
    """The accession the target lookup should use.

    An accession that was already passed in is used as given and is not recorded
    as a structure read. A gene symbol is read from the condition spec beside
    the run: the chain whose sequence is fixed, then the UNP accession of that
    chain. A generated structure drops those rows, so a miss there follows the
    structure path back to the campaign deposit. Two accessions are not guessed.
    """
    known = uniprot_accession(target_id)
    if known is not None:
        return _TargetResolution(lookup_id=known, read_accession=None, unresolved=None)
    found: set[str] = set()
    seen: set[Path] = set()
    queue = _spec_paths(output_dir)
    while queue and len(seen) < _SPEC_LIMIT:
        spec = queue.pop(0)
        try:
            key = spec.resolve()
        except OSError:
            continue
        if key in seen or not spec.is_file():
            continue
        seen.add(key)
        read = _accession_from_spec(spec)
        if read.status == "ambiguous":
            return _TargetResolution(target_id, None, _AMBIGUOUS_ACCESSION)
        if read.status == "one" and read.accession:
            found.add(read.accession)
        elif read.status == "none" and read.follow is not None:
            queue.extend(_campaign_specs(read.follow))
    if len(found) > 1:
        return _TargetResolution(target_id, None, _AMBIGUOUS_ACCESSION)
    if len(found) == 1:
        accession = next(iter(found))
        return _TargetResolution(accession, accession, None)
    return _TargetResolution(target_id, None, None)


def _spec_paths(directory: Path) -> list[Path]:
    """Condition specs for one run directory, not the trees underneath it."""
    found: list[Path] = []
    if directory.is_dir():
        found.extend(sorted(directory.glob("*.yaml")))
        found.extend(sorted(directory.glob("*.yml")))
    parent = directory.parent
    name = directory.name
    for suffix in (".yaml", ".yml"):
        sibling = parent / f"{name}{suffix}"
        if sibling.is_file():
            found.append(sibling)
    spec = _job_design_spec(parent / f"{name}.job.json", parent)
    if spec is not None:
        found.append(spec)
    return found


def _campaign_specs(structure: Path) -> list[Path]:
    """The campaign specs for a generated structure, which has no UNP rows."""
    current = structure
    for _ in range(_SPEC_LIMIT):
        if current.name == "inference_samples":
            return _spec_paths(current.parent)
        if current.parent == current:
            break
        current = current.parent
    return []


def _job_design_spec(job: Path, start: Path) -> Path | None:
    if not job.is_file():
        return None
    try:
        payload = json.loads(job.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    named = payload.get("design_spec")
    if not isinstance(named, str) or not named.strip():
        return None
    return _existing_file(start, named.strip())


def _existing_file(start: Path, relative: str) -> Path | None:
    candidate = Path(relative)
    if candidate.is_file():
        return candidate
    current = start
    for _ in range(_SPEC_LIMIT):
        found = current / relative
        if found.is_file():
            return found
        if current.parent == current:
            return None
        current = current.parent
    return None


def _accession_from_spec(spec: Path) -> _SpecRead:
    payload = _read_yaml(spec)
    chains = _fixed_chains(payload)
    raw_input = _spec_input(payload)
    if not chains or raw_input is None:
        return _SpecRead("skip")
    structure = _structure_path(spec.parent, raw_input)
    accessions = _accessions_in_structure(structure, chains)
    if accessions is None:
        return _SpecRead("ambiguous")
    if len(accessions) == 1:
        return _SpecRead("one", next(iter(accessions)))
    return _SpecRead("none", follow=structure)


def _read_yaml(path: Path) -> object | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


def _spec_input(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    binder = payload.get("binder")
    if not isinstance(binder, dict):
        return None
    raw = binder.get("input")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _fixed_chains(payload: object) -> list[str] | None:
    """Chains whose sequence is fixed. Backbone conditioning is not a chain."""
    if not isinstance(payload, dict):
        return None
    binder = payload.get("binder")
    if not isinstance(binder, dict):
        return None
    conditions = binder.get("conditions")
    if not isinstance(conditions, dict):
        return None
    sequenced = conditions.get("C_SEQ")
    if not isinstance(sequenced, dict):
        return None
    rows: object = None
    for key in (True, "true", "True"):
        if key in sequenced:
            rows = sequenced[key]
            break
    if not isinstance(rows, list):
        return None
    chains: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        select = row.get("select")
        if not isinstance(select, str):
            continue
        match = _SELECT_CHAIN.search(select)
        if match is None:
            continue
        chain = match.group(1)
        if chain not in chains:
            chains.append(chain)
    return chains or None


def _structure_path(start: Path, raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = start / candidate
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def _accessions_in_structure(path: Path, chains: list[str]) -> set[str] | None:
    """UNP accessions of ``chains``. None means the chain names more than one.

    The path comes from a condition spec's ``input``, which the program accepts
    compressed -- ``inputs=...monomer-short.cif.zst`` is the documented form --
    so this goes through the same decompressing read a design does. Read as
    plain text it raised ``UnicodeError`` and returned no accession, and the
    target then could not be named from the spec that named it.
    """
    text = _design_text(path)
    if text is None:
        return set()
    return _unp_for_chains(text, chains)


def _unp_for_chains(text: str, chains: list[str]) -> set[str] | None:
    """One accession per fixed chain, or None when those chains disagree.

    The label chain is ``struct_asym`` joined to a UNP ``struct_ref`` row. The
    author strand is ``struct_ref_seq``. Both are kept: the condition spec
    addresses the label chain, and the strand row is the same protein when the
    deposit used one letter for both. A chain with no row contributes nothing.

    All four categories are read in **both** spellings mmCIF allows, for the
    same reason ``_polymer_sequences`` is. A deposit with one UniProt reference
    -- a monomeric target, which is the common case -- writes ``_struct_ref`` as
    plain item/value lines rather than a ``loop_``, and reading loops alone
    found no accession in it. Of the target structures in this checkout, 117 of
    261 are written that way: ``gene_4ZW9_1.cif`` names ``P11169`` in a flat
    ``_struct_ref`` and this returned an empty set for it. The target then could
    not be named from the file that named it, so the target-liability axes had
    no accession to measure against.

    A real file uses one spelling per category, so the merge order within a pass
    is not a question these rows can raise. Loop rows are read first either way.
    """
    loops = _cif_loops(text, prefixes=_ACCESSION_PREFIXES)
    unp_by_ref: dict[str, set[str]] = {}
    unp_by_entity: dict[str, set[str]] = {}
    for row in _accession_rows(text, loops, "_struct_ref.", ("_struct_ref.db_name",)):
        if _bare(row.get("_struct_ref.db_name", "")).upper() != "UNP":
            continue
        accession = uniprot_accession(_bare(row.get("_struct_ref.pdbx_db_accession", "")))
        if accession is None:
            continue
        ref_id = _bare(row.get("_struct_ref.id", ""))
        if ref_id:
            unp_by_ref.setdefault(ref_id, set()).add(accession)
        entity = _bare(row.get("_struct_ref.entity_id", ""))
        if entity and entity not in {".", "?"}:
            unp_by_entity.setdefault(entity, set()).add(accession)
    label: dict[str, set[str]] = {}
    for row in _accession_rows(text, loops, "_struct_asym.", ("_struct_asym.id", "_struct_asym.entity_id")):
        chain = _bare(row.get("_struct_asym.id", ""))
        entity = _bare(row.get("_struct_asym.entity_id", ""))
        if not chain or chain in {".", "?"}:
            continue
        label[chain] = set(unp_by_entity.get(entity, ()))
    author: dict[str, set[str]] = {}
    for row in _accession_rows(text, loops, "_struct_ref_seq.", ("_struct_ref_seq.pdbx_strand_id",)):
        strand = _bare(row.get("_struct_ref_seq.pdbx_strand_id", ""))
        ref_id = _bare(row.get("_struct_ref_seq.ref_id", ""))
        if not strand or strand in {".", "?"} or ref_id not in unp_by_ref:
            continue
        chosen = set(unp_by_ref[ref_id])
        named = uniprot_accession(_bare(row.get("_struct_ref_seq.pdbx_db_accession", "")))
        if named is not None:
            chosen.add(named)
        author.setdefault(strand, set()).update(chosen)
    poly: dict[str, set[str]] = {}
    for row in _accession_rows(text, loops, "_entity_poly.", ("_entity_poly.entity_id", "_entity_poly.pdbx_strand_id")):
        entity = _bare(row.get("_entity_poly.entity_id", ""))
        accessions = unp_by_entity.get(entity, set())
        for strand in _strands(row.get("_entity_poly.pdbx_strand_id", "")):
            poly.setdefault(strand, set()).update(accessions)
    per_chain: list[set[str]] = []
    for chain in chains:
        parts: list[set[str]] = []
        if chain in label:
            parts.append(label[chain])
        if chain in author:
            parts.append(author[chain])
        chosen = set().union(*parts) if parts else set(poly.get(chain, ()))
        if len(chosen) > 1:
            return None
        per_chain.append(chosen)
    if any(item for item in per_chain) and any(not item for item in per_chain):
        return None
    if any(not item for item in per_chain):
        return set()
    union = set().union(*per_chain)
    if len(union) != 1:
        return None
    return union


def _accession_rows(
    text: str,
    loops: Sequence[tuple[list[str], list[dict[str, str]]]],
    prefix: str,
    required: tuple[str, ...],
) -> list[dict[str, str]]:
    """Rows of one category from both spellings, loop rows first.

    ``required`` names the items a usable row must carry, which is the same test
    the loop path applied to its columns. Applied to a non-loop row as well, so a
    category written flat but missing the item the join needs is passed over
    rather than contributing a row with a hole in it.

    One pass per category rather than one pass for all four: ``_nonloop_category``
    accumulates consecutive matching tags into a single row, so a prefix covering
    several categories would merge two adjacent ones into a row carrying both
    their items -- and that row would then satisfy two different joins at once.
    """
    rows: list[dict[str, str]] = []
    for columns, loop_rows in loops:
        if all(name in columns for name in required):
            rows.extend(loop_rows)
    rows.extend(row for row in _nonloop_category(text, prefix) if all(name in row for name in required))
    return rows


def _bare(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _designed_sequences(output_dir: Path, rows: list[dict[str, str]]) -> list[tuple[str, str | None]]:
    """Example id and designed sequence, in the same order as ``samples``."""
    pairs: list[tuple[str, str | None]] = []
    paths = _generated_cifs(output_dir)
    ids = _sample_ids(output_dir, paths)
    bindings, _ = _design_rows(rows, paths)
    for index, path in enumerate(paths):
        row, _ = bindings[path]
        interest, _chain_reason = _interest_for_row(row, rows)
        pairs.append((ids[index], _designed_sequence(path, interest)))
    return pairs


def _designed_sequence(path: Path, interest: str | None) -> str | None:
    if interest is None:
        return None
    text = _design_text(path)
    if text is None:
        return None
    sequence = _sequence_for_chain(_polymer_sequences(text), interest)
    if not sequence:
        return None
    return sequence


def _read_evidence(output_dir: Path) -> _Evidence:
    """Axis states the run already wrote, and the basis for each.

    Read as ``utf-8-sig`` like every other file here: a byte-order mark decodes
    cleanly but then fails the JSON parse, so an evidence file saved by an
    editor that writes one refused the whole run rather than being read.

    An axis may map to the state alone, which is how this file has always been
    written, or to an object carrying that state with a ``reason`` and a
    ``source``. Some axes have no measured clear at all -- paralog's singleton
    cluster, a CpG-free oligo -- so a curated clear is the only way they ever
    clear, and a verdict resting on one with nothing recorded behind it cannot
    be checked by anyone who was not in the room. An unknown key is refused
    rather than passed over, because a misspelled ``reason`` that is silently
    dropped leaves exactly the record it was meant to prevent.
    """
    path = output_dir / _EVIDENCE
    if not path.exists() and not path.is_symlink():
        return _Evidence({}, {})
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_one_entry_per_axis)
    except UnicodeError as exc:
        raise ValueError("axis_evidence.json must be UTF-8 text.") from exc
    except OSError as exc:
        # Every other malformed evidence file is a named refusal that the CLI
        # and the MCP layer render for that one directory. An OSError escaped
        # that: a folder named axis_evidence.json, a dangling link, or an
        # unreadable file aborted the whole sweep on the first bad run.
        raise ValueError("axis_evidence.json could not be read.") from exc
    if not isinstance(payload, dict):
        raise ValueError("axis_evidence.json must map axis names to hit or clear.")
    states: dict[str, str] = {}
    basis: dict[str, dict[str, str]] = {}
    for axis, value in payload.items():
        if not isinstance(axis, str):
            raise ValueError("axis_evidence.json must map axis names to hit or clear.")
        state, stated = _evidence_entry(axis, value)
        states[axis] = state
        if stated:
            basis[axis] = stated
    return _Evidence(states, basis)


def _one_entry_per_axis(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    """Refuse a repeated key rather than keep whichever copy came last.

    ``json.loads`` keeps the last value for a repeated key and says nothing, so
    a file naming one axis twice resolved to whichever entry came last: append a
    clear after a curated hit and the hit was gone, with the file's own bytes
    digested in ``inputs`` as though they said one thing. This is the refusal
    the metrics reader already makes for a repeated column, for the same reason,
    and this is the one file that can set a state outright.
    """
    found: dict[str, object] = {}
    for key, value in pairs:
        if key in found:
            # Said of a name, not of an axis: the same check runs inside one
            # axis's own entry, where the repeated name is a field such as reason.
            raise ValueError(f"axis_evidence.json names {key} twice. Each name is given once.")
        found[key] = value
    return found


def _evidence_entry(axis: str, value: object) -> tuple[str, dict[str, str]]:
    """One axis's curated state, and whatever basis was stated with it."""
    if isinstance(value, str):
        return _evidence_state(axis, value), {}
    if not isinstance(value, dict):
        raise ValueError(f"{axis} must be hit or clear, or an object naming one.")
    unknown = sorted(key for key in value if key not in _EVIDENCE_KEYS)
    if unknown:
        raise ValueError(f"{axis} has no {unknown[0]!r} field. Use state, reason or source.")
    raw = value.get("state")
    if not isinstance(raw, str):
        raise ValueError(f"{axis} must name a state of hit or clear.")
    stated: dict[str, str] = {}
    for key in ("reason", "source"):
        found = value.get(key)
        if found is None:
            continue
        if not isinstance(found, str) or not found.strip():
            raise ValueError(f"{axis} {key} must be text.")
        cleaned = " ".join(found.split())
        if len(cleaned) > _BASIS_LIMIT:
            raise ValueError(f"{axis} {key} is longer than {_BASIS_LIMIT} characters.")
        stated[key] = cleaned
    return _evidence_state(axis, raw), stated


def _evidence_state(axis: str, raw: str) -> str:
    token = raw.strip().casefold()
    if token not in _STATES:
        raise ValueError(f"{axis} must be hit or clear.")
    return token


def _metrics_path(output_dir: Path) -> Path | None:
    """Prefer the generation table, then its re-evaluation, then the offline evaluator's output."""
    for relative in (_PRIMARY_METRICS, _REEVAL_METRICS, _EVALUATION_METRICS):
        candidate = output_dir / relative
        if candidate.exists():
            return candidate
    return None


def _metric_rows(output_dir: Path) -> list[dict[str, str]]:
    """Rows of the chosen metrics file. A missing or refused file is empty."""
    return _metrics_table(output_dir, _metrics_path(output_dir))[0]


def _metrics_table(output_dir: Path, path: Path | None) -> tuple[list[dict[str, str]], str | None]:
    """Rows of the chosen metrics file, and the reason when there are none.

    This is the one table that moves the verdict, and it was the least
    guarded reader here: every failure returned an empty list, which reads
    downstream exactly like a run that was never scored. The refusals are the
    ones that change an answer rather than withhold it. A repeated column
    resolves to whichever copy ``csv`` kept last, so a False in the first is
    invisible. A row carrying a surplus field is misaligned from its first
    extra separator on, and one that stops short of the header never reached
    its later columns; either way a flag is read under another column's name.
    A row short of the header is also what an unterminated quote leaves
    behind, having swallowed the rows after it, and a failing design that
    disappears leaves the passes behind it looking like the whole run.
    """
    try:
        if path is None:
            for directory in (output_dir / "inference_metrics", output_dir / "evaluation_metrics"):
                if (directory / PENDING).exists():
                    return [], MetricsBlocker.PUBLICATION_PENDING.value
                if job_tables(directory):
                    return [], MetricsBlocker.AGGREGATION_REQUIRED.value
            return [], "metrics_not_configured"
        if path.parent.name == "evaluation_metrics":
            inference = output_dir / "inference_metrics"
            if (inference / PENDING).exists():
                return [], MetricsBlocker.PUBLICATION_PENDING.value
            if job_tables(inference):
                return [], MetricsBlocker.AGGREGATION_REQUIRED.value
        if (path.parent / PENDING).exists():
            return [], MetricsBlocker.PUBLICATION_PENDING.value
        with path.open("rb") as raw:
            if os.fstat(raw.fileno()).st_size > _METRICS_FILE_LIMIT:
                return [], "metrics_too_large"
            payload = raw.read(_METRICS_FILE_LIMIT + 1)
        if len(payload) > _METRICS_FILE_LIMIT:
            return [], "metrics_too_large"
        if b"\x00" in payload:
            return [], "metrics_unreadable"
        if path.name == PRIMARY:
            _, blocker = verified_sources(path.parent, hashlib.sha256(payload).hexdigest())
            if blocker is not None:
                return [], blocker
        elif any(shard != path for shard in job_tables(path.parent)):
            return [], MetricsBlocker.AGGREGATION_REQUIRED.value
        with io.StringIO(payload.decode("utf-8-sig"), newline="") as handle:
            try:
                return _screen_csv_records(csv.reader(handle, strict=True))
            except csv.Error as error:
                if "field larger than field limit" not in str(error):
                    raise
        return _screen_csv_records(iter(wide_csv_rows(payload)))
    except csv.Error as error:
        return [], "metrics_row_too_short" if str(error) == "unexpected end of data" else "metrics_unreadable"
    except (OSError, UnicodeError, MemoryError):
        return [], "metrics_unreadable"


def _screen_csv_records(records: Iterator[list[str]]) -> tuple[list[dict[str, str]], str | None]:
    names = next((row for row in records if row), None)
    if names is None:
        return [], "metrics_not_reported"
    if _repeated_metric_columns(names):
        return [], "metrics_columns_repeated"
    found = []
    for row in records:
        if not row:
            continue
        if len(row) != len(names):
            return [], "metrics_row_too_long" if len(row) > len(names) else "metrics_row_too_short"
        found.append(dict(zip(names, row, strict=True)))
    return (found, None) if found else ([], "metrics_not_reported")


def _repeated_metric_columns(names: Sequence[str | None]) -> bool:
    """Whether the header uses a column this screen reads more than once."""
    seen: set[str] = set()
    repeated: set[str] = set()
    for raw in names:
        if isinstance(raw, str):
            name = raw.strip()
            if name in seen:
                repeated.add(name)
            seen.add(name)
    if not repeated:
        return False
    if repeated & (_METRIC_COLUMNS | {"metrics_tags"}) or any(trace_metric_kind(name) for name in repeated):
        return True
    for name, thresholds in (read_criteria() or {}).items():
        if repeated & {f"{name}.pass", f"{name}.all_metrics_present"}:
            return True
        if isinstance(thresholds, dict):
            for label, spec in thresholds.items():
                try:
                    keys = MetricThreshold.parse(label, spec).keys
                except ValueError:
                    continue
                if repeated.intersection(keys):
                    return True
    return False


def _flag_column(rows: list[dict[str, str]], column: str) -> bool | str:
    """One pass/fail column. A false row wins. No measured cell stores absent."""
    flags = [row_pass_flag(row, column) for row in rows]
    if False in flags:
        return False
    return True if flags and all(flags) else _ABSENT


def _sample_measurements(
    output_dir: Path,
    rows: list[dict[str, str]],
    *,
    chemical: bool = False,
    kind: str | None = None,
    affinity: RunAffinity | None = None,
    vina: RunPoses | None = None,
    tnp: RunProfiles | None = None,
) -> tuple[list[dict[str, object]], list[tuple[str, str | None]], list[dict[str, str]]]:
    """One record per generated CIF, in filename order, and each design's chain.

    The chains come back beside the records rather than being read again. Three
    passes used to open and parse every CIF of a run -- these measurements, the
    liability axes, and the spread across designs -- so an eight-design run
    parsed twenty-four files to measure eight. They are returned here, not
    stored: no sequence reaches the record.

    A miss stays null or absent, and ``sequence_blocker`` names which miss it was.
    """
    samples: list[dict[str, object]] = []
    chains: list[tuple[str, str | None]] = []
    paths = _generated_cifs(output_dir)
    ids = _sample_ids(output_dir, paths)
    # Names for the designs that could not be decoded, taken from the same
    # function over both lists so a refused design cannot be given the name of
    # one that was read. The readable ids stay exactly what ``sample_cifs``
    # gives them, because everything joining onto this screen uses those.
    all_paths = [*paths, *_refused_designs(output_dir, paths)]
    refused = _sample_ids(output_dir, all_paths, preserved=dict(zip(paths, ids, strict=True)))[len(paths) :]
    bindings, coverage = _design_rows(rows, all_paths)
    for index, path in enumerate(paths):
        row, row_reason = bindings[path]
        interest, chain_reason = _interest_for_row(row, rows)
        text = _cif_text(path)
        if affinity is not None:
            affinity.observe(ids[index], text)
        if vina is not None:
            vina.observe(ids[index], text)
        if tnp is not None:
            tnp.observe(ids[index], text)
        sequence = _sequence_block(
            path,
            interest,
            text=text,
            chemical=chemical,
            kind=kind,
            chain_reason=chain_reason,
        )
        chain = sequence.pop(_CHAIN_KEY, None)
        chains.append((ids[index], chain if isinstance(chain, str) and chain else None))
        interface = _interface_block(path, interest, text=text, chain_reason=chain_reason)
        interface_quality = _interface_quality_block(path, interest, text=text, chain_reason=chain_reason)
        item: dict[str, object] = {
            "example_id": ids[index],
            "contact_ms": interface["contact_ms"],
            "clash_fraction": interface["clash_fraction"],
            "gravy": sequence["gravy"],
            "instability_index": sequence["instability_index"],
            "sequence_pass": sequence["sequence_pass"],
            "sequence_identity": _optional_metric(row, _IDENTITY_COLUMN),
            "ca_rmsd": _optional_metric(row, _CA_RMSD_COLUMN),
        }
        found = sequence.get("chemical_liability")
        if isinstance(found, dict):
            item["chemical_liability"] = found
        modality = sequence.get("modality_screen")
        if isinstance(modality, dict):
            item["modality_screen"] = modality
        blocker = sequence.get("sequence_blocker")
        if isinstance(blocker, str):
            item["sequence_blocker"] = blocker
        reason = interface.get("interface_blocker")
        if isinstance(reason, str):
            item["interface_blocker"] = reason
        if row_reason is not None:
            # Which metrics row this design was measured against, when there
            # was not exactly one. A null identity is otherwise the same
            # record as a table that does not carry that column at all.
            item["row_blocker"] = row_reason
        item["interface_quality"] = interface_quality
        samples.append(item)
    if affinity is not None:
        for example_id in refused:
            affinity.observe(example_id, None, blocker="structure_unreadable")
    if vina is not None:
        for example_id in refused:
            vina.observe(example_id, None)
    if tnp is not None:
        for example_id in refused:
            tnp.observe(example_id, None)
    if paths:
        for example_id in refused:
            # A design in the folder that this screen could not decode. It
            # reaches the axis roll-ups with no chain, so an axis stays
            # untested rather than reporting the designs that happened to be
            # readable as the run. Guarded on `paths`: with nothing readable
            # at all the `if not samples` branch below already names
            # designs_compressed, and a phantom chain would shift the
            # blockers the protein path reports.
            chains.append((example_id, None))
    return samples, chains, coverage


def _with_reason(found: dict[str, object], blocker: object, key: str) -> dict[str, object]:
    """Attach the named reason to a block of nulls, when there is one.

    The closed page reads these top-level blocks, so a reason left on the
    sample alone never reached the reader looking at the empty gauge.
    """
    if isinstance(blocker, str) and blocker:
        return {**found, key: blocker}
    return found


def _first_sample_fields(samples: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    """Top-level sequence and interface. They repeat the first CIF and hide nothing."""
    sequence: dict[str, object] = {
        "gravy": None,
        "instability_index": None,
        "sequence_pass": _ABSENT,
    }
    interface: dict[str, object] = {"contact_ms": None, "clash_fraction": None}
    if not samples:
        return sequence, interface
    first = samples[0]
    # Carried up with the nulls they explain: the closed page reads these
    # blocks, so a reason left on the sample alone showed an empty gauge and
    # said nothing about why.
    found = _with_reason(
        {
            "gravy": first["gravy"],
            "instability_index": first["instability_index"],
            "sequence_pass": first["sequence_pass"],
        },
        first.get("sequence_blocker"),
        "sequence_blocker",
    )
    return (
        found,
        _with_reason(
            {"contact_ms": first["contact_ms"], "clash_fraction": first["clash_fraction"]},
            first.get("interface_blocker"),
            "interface_blocker",
        ),
    )


def _first_modality(samples: list[dict[str, object]]) -> dict[str, object] | None:
    """The first CIF's drug-type report. A kind with nothing to report has none."""
    if not samples:
        return None
    found = samples[0].get("modality_screen")
    if isinstance(found, dict):
        return found
    return None


def _first_chemical(samples: list[dict[str, object]]) -> dict[str, object] | None:
    """The first CIF's chemical report. A ligand or nucleic sample has none."""
    if not samples:
        return None
    found = samples[0].get("chemical_liability")
    if isinstance(found, dict):
        return found
    return None


def designed_chain(output_dir: Path) -> str:
    """The chain the metrics table calls the chain of interest, or empty.

    Anything re-folding a design has to name the same chain the screen measured,
    so it comes from the same place the screen read it.
    """
    return _chain_of_interest(_metric_rows(output_dir)) or ""


def sample_cifs(output_dir: Path) -> dict[str, Path]:
    """Each generated CIF, by the example id the screen gave it.

    A screen and anything reading that screen have to agree about which file is
    which sample, so both take the ids from here rather than deriving them
    twice and disagreeing over an unusual filename.
    """
    paths = _generated_cifs(output_dir)
    return dict(zip(_sample_ids(output_dir, paths), paths, strict=True))


def _generated_cifs(output_dir: Path) -> list[Path]:
    """Every generated design this screen can read, in one stable order.

    Sorted on the stem rather than the filename so a run holding the same design
    twice -- plain beside compressed, which a decompress-in-place leaves behind
    -- keeps them adjacent, and so the order does not change when a run is
    written with compression. Within one stem the plain file is preferred,
    because reading it costs nothing.
    """
    root = _samples_root(output_dir)
    if not root.is_dir():
        return []
    readable = tuple(suffix for suffix in _DESIGN_SUFFIXES if suffix != _ZSTD_SUFFIX or _zstandard() is not None)
    found: dict[tuple[str, str], Path] = {}
    for path in root.rglob("*_generated.cif*"):
        if not path.is_file():
            continue
        suffix = next((candidate for candidate in readable if path.name.endswith(candidate)), None)
        if suffix is None:
            continue
        if suffix != "_generated.cif" and not _design_opens(path):
            continue
        key = (str(path.parent), _design_stem(path.name))
        kept = found.get(key)
        # ``_DESIGN_SUFFIXES`` is longest-first, so a higher index is the plainer
        # spelling. Plain beats compressed: reading it costs nothing, and a
        # decompress-in-place leaves both spellings of one design side by side,
        # which would otherwise be measured as two designs.
        if kept is None or _DESIGN_SUFFIXES.index(suffix) > _DESIGN_SUFFIXES.index(_design_suffix(kept)):
            found[key] = path
    return [found[key] for key in sorted(found)]


def _refused_designs(output_dir: Path, measured: list[Path]) -> list[Path]:
    """Designs in this folder that could not be decoded, so nothing measured them.

    ``_generated_cifs`` lists the designs this screen can read and drops the
    rest, and every axis then rolls up over the designs it did read. A run whose
    second design is a ``.bcif``, a ``.cif.zst`` on a machine without
    ``zstandard``, or an archive that does not open therefore reported the
    readable design's state as the run's, and named no reason either:
    ``_designs_blocker`` is asked only when nothing at all was read, so the one
    run that says "no design measured" is the run that measured none of them.
    A clear over the survivors is not a clear for the run -- the rule
    ``rollup_axis`` states and ``liability`` pins, that a design which could not
    be scored is not evidence it would have passed.

    A second spelling of a design that *was* read is not refused: that is one
    molecule written twice, which ``_generated_cifs`` already settles by stem.
    The spellings looked for are ``_PACKED_DESIGNS``, the same set the blocker
    names.
    """
    root = _samples_root(output_dir)
    if not root.is_dir():
        return []
    read = {(str(path.parent), path.name.split("_generated.")[0]) for path in measured}
    found: dict[tuple[str, str], Path] = {}
    for pattern in _PACKED_DESIGNS:
        try:
            packed = sorted(root.rglob(pattern))
        except OSError:
            return []
        for path in packed:
            key = (str(path.parent), path.name.split("_generated.")[0])
            if key in read or key in found or not path.is_file():
                continue
            found[key] = path
    return [found[key] for key in sorted(found)]


def _design_suffix(path: Path) -> str:
    """The generated suffix a design path carries, longest first."""
    return next((suffix for suffix in _DESIGN_SUFFIXES if path.name.endswith(suffix)), "_generated.cif")


def _sample_ids(output_dir: Path, paths: list[Path], *, preserved: Mapping[Path, str] | None = None) -> list[str]:
    """One id per generated CIF, unique within the run.

    The binder token is the best name but it is not unique. A run holding two
    designs under ``inference_samples/run_a/binder_0/`` and
    ``inference_samples/run_b/binder_0/`` gave both the id ``binder_0``, so one
    axis record stood for two molecules: the sample whose 9-mers matched the
    proteome was overwritten by the one whose did not, ``failed`` came back
    empty, and an ``axis_hit`` became ``screened_clear``. Adding a design
    removed a hit.

    A repeated token falls back to the path below ``inference_samples``, which
    is unique because the filesystem makes it so, and then to the index.
    """
    root = _samples_root(output_dir)
    plain = [_sample_id(path) for path in paths]
    counts: dict[str | None, int] = {}
    for name in plain:
        counts[name] = counts.get(name, 0) + 1
    ids: list[str] = []
    taken: set[str] = set(preserved.values()) if preserved else set()
    for path, name in zip(paths, plain, strict=True):
        if preserved and path in preserved:
            ids.append(preserved[path])
            continue
        candidate = name if name is not None and counts[name] == 1 else _path_id(root, path)
        base = candidate
        suffix = 0
        while candidate in taken:
            suffix += 1
            candidate = f"{base}_{suffix}"
        taken.add(candidate)
        ids.append(candidate)
    return ids


def _path_id(path_root: Path, path: Path) -> str:
    """The CIF's path below ``inference_samples``, as one safe token."""
    try:
        relative = path.relative_to(path_root)
    except ValueError:
        # Not under inference_samples at all. The absolute path still identifies
        # it, and a digest of that does not depend on how many designs there are.
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
        return f"sample_{digest}"
    stem = _design_stem(relative.name)
    # The usual layout repeats the token as its own folder name, giving
    # "binder_0_binder_0". The id is read by people and stored in screen.json,
    # so a parent that only restates the stem is left out.
    parents = [part for part in relative.parts[:-1] if part != stem]
    token = "_".join([*parents, stem])
    if _SAFE_EXAMPLE_ID.fullmatch(token):
        return token
    # Too long for an id, but falling back to the position made the id depend on
    # how many designs the run holds: inserting an earlier-sorting design
    # re-pointed sample_0 at a different molecule, and an archived screen then
    # named a file it was not measured from. A digest of the path keeps the id
    # identifying and stable.
    digest = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()[:8]
    short = f"{token[:48]}_{digest}"
    return short if _SAFE_EXAMPLE_ID.fullmatch(short) else f"sample_{digest}"


def _sample_id(path: Path) -> str | None:
    """A short filename token. A path from the metrics table is never used."""
    match = _BINDER_TOKEN.search(path.name)
    if match is not None:
        return match.group(0)
    stem = _design_stem(path.name)
    if _SAFE_EXAMPLE_ID.fullmatch(stem):
        return stem
    # No stable token from this filename. Returning a position here made the id
    # unique by construction, so _sample_ids never asked _path_id for the stable
    # one and an id still moved when another design was added.
    return None


def _row_for_cif(rows: list[dict[str, str]], path: Path) -> dict[str, str]:
    """The one metrics row naming this design, or empty when there is not one."""
    return _agreed_row(_candidate_rows(rows, path))


def _design_rows(
    rows: list[dict[str, str]], paths: list[Path],
) -> tuple[dict[Path, tuple[dict[str, str], str | None]], list[dict[str, str]]]:
    """Count each discovered design once and retain table-only observations."""
    matches = {path: _candidate_rows(rows, path) for path in paths}
    claimed = defaultdict(list)
    for path, candidates in matches.items():
        for candidate in candidates:
            claimed[id(candidate)].append(path)
    coverage = [row for row in rows if id(row) not in claimed]
    bindings = {}
    for path in paths:
        if path in bindings:
            continue
        # A basename may link several files and rows. Consume the whole group
        # once so the result is independent of which file was visited first.
        pending = [path]
        group = set()
        evidence = {}
        while pending:
            member = pending.pop()
            if member in group:
                continue
            group.add(member)
            for row in matches[member]:
                evidence[id(row)] = row
                pending.extend(claimed.pop(id(row), ()))
        candidates = list(evidence.values())
        agreed = _agreed_row(candidates)
        reason = None
        if len(group) > 1:
            reason = MetricsBlocker.ROW_SHARED.value
        elif candidates and not agreed:
            reason = "metrics_row_ambiguous"
        elif rows and not candidates:
            reason = "metrics_row_absent"
        for member in group:
            bindings[member] = ({} if reason else agreed, reason)
        coverage.append(agreed or _unresolved_flags(candidates))
        coverage.extend({} for _ in range(len(group) - 1))
    return bindings, coverage


def _unresolved_flags(candidates: list[dict[str, str]]) -> dict[str, str]:
    """Keep confirmed failures and mark disputed coverage without copying scores."""
    flags = {}
    for row in candidates:
        for key, cell in row.items():
            if not cell.strip():
                continue
            if key.endswith(".all_metrics_present"):
                flags.setdefault(key, "0")
            elif key.endswith(".pass"):
                presence = key.removesuffix(".pass") + ".all_metrics_present"
                if row_pass_flag(row, key) is False:
                    flags[key], flags[presence] = "0", "1"
                else:
                    flags.setdefault(presence, "0")
            elif key == _PRE_ORACLE_COLUMN and row_pass_flag(row, key) is False:
                flags[key] = "0"
    return flags


def _samples_location(value: str) -> str:
    """The run-relative identity that survives relocating a recorded structure path."""
    parts = value.replace("\\", "/").split("/")
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] in {"inference_samples", "evaluation_samples"}:
            return "/".join(parts[index:])
    return ""


def _candidate_rows(rows: list[dict[str, str]], path: Path) -> list[dict[str, str]]:
    """Prefer exact paths, relocated paths, evaluator IDs, then historical bare names."""
    match = _BINDER_TOKEN.search(path.name)
    token = match.group(0) if match is not None else None
    stem = _design_stem(path.name)
    aliases = {path.as_posix(), path.absolute().as_posix()}
    if path.is_symlink():
        with contextlib.suppress(OSError, RuntimeError):
            aliases.add(path.resolve().as_posix())
    locations = {_samples_location(value) for value in aliases} - {""}
    aliases.update(location.split("/", 1)[1] for location in locations if "/" in location)
    exact = []
    relocated = []
    legacy = []
    for row in rows:
        recorded = (row.get("designed_structure_path") or "").replace("\\", "/")
        location = _samples_location(recorded)
        if recorded in aliases:
            exact.append(row)
        elif location in locations:
            relocated.append(row)
        elif not location:
            legacy.append(row)
    found = exact or relocated
    if not found and path.parent.parent.name == "evaluation_samples":
        found = [row for row in rows if row.get("example_id") == path.parent.name]
    if not found:
        found = [
            row for row in legacy
            if _row_names_sample(row, token=token, stem=stem, filename=path.name)
        ]
    return found


def _agreed_row(candidates: list[dict[str, str]]) -> dict[str, str]:
    """Return one observation when all candidates agree on its recorded values."""
    if not candidates:
        return {}
    first = candidates[0]
    for other in candidates[1:]:
        columns = (first.keys() | other.keys()) - _ROW_LOCATORS
        if any(first.get(key) != other.get(key) for key in columns):
            return {}
    return first


def _row_names_sample(row: dict[str, str], *, token: str | None, stem: str, filename: str) -> bool:
    example = row.get("example_id", "")
    designed = row.get("designed_structure_path", "")
    if not isinstance(example, str):
        example = ""
    if not isinstance(designed, str):
        designed = ""
    example = example.strip()
    if token is not None and example == token:
        return True
    if example in {stem, filename}:
        return True
    for raw in (example, designed):
        if not raw:
            continue
        normalized = raw.replace("\\", "/")
        name = normalized.rstrip("/").split("/")[-1]
        if name in {stem, filename, token}:
            return True
        if token is not None and f"/{token}/" in f"/{normalized}/":
            return True
    return False


def _interest_for_row(row: dict[str, str], rows: list[dict[str, str]]) -> tuple[str | None, str | None]:
    """One design's chain of interest, and why there is none.

    The design's own row wins. The fallback across the table stands in only
    while the table is of one mind: when it names several chains, a design
    whose own row named none is left unmeasured rather than measured against
    a neighbour's chain.
    """
    if row:
        interest = _chain_of_interest([row])
        if interest is not None:
            return interest, None
    named = _named_chains(rows)
    if len(named) == 1:
        return named[0], None
    if len(named) > 1:
        return None, "chain_disagreement"
    return None, None


def _sequence_block(
    path: Path,
    interest: str | None,
    *,
    text: str | None,
    chemical: bool = False,
    kind: str | None = None,
    chain_reason: str | None = None,
) -> dict[str, object]:
    """GRAVY and instability for one CIF. A miss stays absent. The sequence is not stored.

    Both metrics are defined over amino acids, so they are measured for a
    protein or peptide binder only. Any other ``kind`` leaves them absent with
    ``not_a_polypeptide``; ``kind`` ``None`` is a caller that named no kind and
    is measured as before.

    A chemical or drug-type read that fails leaves those sequence metrics and omits that record.
    A nucleic chain that is not a protein sequence still keeps its oligonucleotide report.
    """
    absent: dict[str, object] = {
        "gravy": None,
        "instability_index": None,
        "sequence_pass": _ABSENT,
    }
    if interest is None:
        absent["sequence_blocker"] = chain_reason or "chain_not_named"
        if kind == "ligand" and text is not None:
            # The component dictionary belongs to the structure and needs no polymer chain.
            _attach_modality(absent, kind, "", text)
        return absent
    if text is None:
        return {**absent, "sequence_blocker": "structure_unreadable"}
    ragged: list[list[str]] = []
    sequence = _sequence_for_chain(_polymer_sequences(text, ragged=ragged), interest) or ""
    record = dict(absent)
    # Kept out of the record and handed back to the caller instead, so one read
    # of this file serves the metrics, the liability axes and the spread across
    # designs. The sequence is still never stored.
    record[_CHAIN_KEY] = sequence
    if not sequence:
        # A loop this reader could not divide into rows is not a chain with no
        # sequence in it. The remedy for the first is the file, for the second
        # the run, and both used to read as the second.
        record["sequence_blocker"] = "structure_unparsed" if ragged else "sequence_absent"
    if sequence and kind is not None and kind not in _LIABILITY_KINDS:
        # ExPASy ProtParam defines GRAVY as "the sum of hydropathy values
        # [Kyte & Doolittle 1982] of all the amino acids, divided by the number
        # of residues" and the instability index as (10/L) * sum of Guruprasad
        # 1990 DIWV over the consecutive dipeptides. Both scales cover the
        # twenty amino acids and nothing else. A DNA chain is spelled A, C, G
        # and T, which are four standard one-letter residue codes, so both
        # metrics returned a number for it -- GRAVY 0.8 and instability 98.3125
        # for an ACGT oligo -- while the same molecule class carrying a U
        # raised instead. Neither is a measurement of a nucleic acid, and the
        # board drew a gauge for the first one.
        record["sequence_blocker"] = "not_a_polypeptide"
    elif sequence:
        try:
            from rfproteina.metrics.physics.developability_metrics import compute_gravy, compute_instability_index

            gravy = compute_gravy(sequence)
            instability = compute_instability_index(sequence)
        except (OSError, ValueError, ImportError):
            gravy = None
            instability = None
        gravy_ok = gravy is not None and math.isfinite(gravy)
        instability_ok = instability is not None and math.isfinite(instability)
        if gravy_ok and instability_ok:
            record = {
                "gravy": float(f"{gravy:.4f}"),
                "instability_index": float(f"{instability:.4f}"),
                # ExPASy ProtParam: "A protein whose instability index is
                # smaller than 40 is predicted as stable, a value above 40
                # predicts that the protein may be unstable." The published
                # rule is strict, so exactly 40.0 is not stable and used to
                # pass. The GRAVY half keeps ``<=``: 0.1 is this project's own
                # triage bar and carries no published equality rule.
                "sequence_pass": gravy <= _GRAVY_MAX and instability < _INSTABILITY_MAX,
                _CHAIN_KEY: sequence,
            }
            if chemical:
                try:
                    found = chemical_record(sequence, partners=_disulfide_partners(text, interest))
                except (ValueError, OverflowError, TypeError):
                    found = None
                if found is not None:
                    record["chemical_liability"] = found
        elif gravy_ok and len(sequence) < 2:
            # The dipeptide sum has no term on a chain of one, which is not a
            # metric that failed: ``compute_instability_index`` returns nan by
            # contract below two residues. The mean hydropathy is still
            # defined, and discarding it named the wrong thing to go and fix.
            record["gravy"] = float(f"{gravy:.4f}")
            record["sequence_blocker"] = "chain_shorter_than_dipeptide"
        else:
            record["sequence_blocker"] = "metrics_failed"
    _attach_modality(record, kind, sequence, text)
    return record


def _attach_modality(record: dict[str, object], kind: str | None, sequence: str, text: str) -> None:
    """Store the drug-type report. A failure leaves the sequence metrics in place."""
    if kind not in {"protein", "peptide", "nucleic", "ligand"}:
        return
    try:
        components = _chem_components(text) if kind == "ligand" else None
        found = modality_record(kind, sequence, components=components)
    except (ValueError, OverflowError, TypeError):
        return
    if isinstance(found, dict):
        record["modality_screen"] = found


def _interface_block(
    path: Path,
    interest: str | None,
    *,
    text: str | None,
    chain_reason: str | None = None,
) -> dict[str, object]:
    """Contact surface and clash fraction for one CIF. A sequence-only file stays null.

    ``text`` is the file the caller already read. Opening it again here only to
    look for ``_atom_site.`` doubled the reads of every design in a run.
    """
    absent: dict[str, object] = {"contact_ms": None, "clash_fraction": None}
    # Six ways to reach a null contact surface, and they used to be one silence.
    # A sequence-only file is an ordinary outcome; a structure that would not
    # load is not, and a reader could not tell them apart.
    if interest is None:
        return {**absent, "interface_blocker": chain_reason or "chain_not_named"}
    if text is None:
        return {**absent, "interface_blocker": "structure_unreadable"}
    if "_atom_site." not in text:
        return {**absent, "interface_blocker": "coordinates_absent"}
    try:
        loaded = _load_generated_structure(path)
        if loaded is None:
            return {**absent, "interface_blocker": "coordinates_unusable"}
        label = _binder_label(loaded, interest)
        if label is None:
            return {**absent, "interface_blocker": "binder_chain_absent"}
        return _interface_numbers(loaded, label)
    except (OSError, ValueError, ImportError, RuntimeError, KeyError, TypeError, IndexError):
        return {**absent, "interface_blocker": "interface_failed"}


#: Compact interface-quality fields carried per sample and at the top level. The
#: reference thresholds are attached once at the top level, not repeated per
#: sample, so a run with many designs does not store the same table many times.
_IQ_FIELDS = (
    "binder_chain",
    "target_chains",
    "dsasa_total_a2",
    "dsasa_binder_a2",
    "dsasa_target_a2",
    "interface_residues_binder",
    "interface_residues_target",
    "interface_polar_contacts",
    "interface_atom_contacts",
    "buried_unsatisfied_polar_atoms",
    "buried_unsatisfied_donors",
    "buried_unsatisfied_acceptors",
    "shape_complementarity",
    "shape_complementarity_reason",
    "shape_complementarity_binder_median",
    "shape_complementarity_target_median",
    "interface_dots_binder",
    "interface_dots_target",
    "method",
    "sc_method",
)
_IQ_ABSENT: dict[str, object] = dict.fromkeys(_IQ_FIELDS, None)


def _interface_quality_labels(interest: str) -> list[str]:
    """The chain labels to try for the binder, in order: the interest, then its letter."""
    labels = [interest]
    letter = interest.split("_", 1)[0]
    if letter != interest:
        labels.append(letter)
    return labels


def _interface_quality_block(
    path: Path,
    interest: str | None,
    *,
    text: str | None,
    chain_reason: str | None = None,
) -> dict[str, object]:
    """Buried area, interface residues, polar contacts and contact-patch shape score for one CIF.

    This is a report, like ``designed_interface``: it is recorded beside the
    verdict and does not move the call. The terms, their thresholds and their
    provenance live in ``rfproteina.disease.interface_quality``. A sequence-only
    file, a missing chain or an unreadable structure leaves the block null with
    ``interface_quality_blocker`` naming which, exactly as ``_interface_block``
    does for the contact surface.
    """
    if interest is None:
        return {**_IQ_ABSENT, "interface_quality_blocker": chain_reason or "chain_not_named"}
    if text is None:
        return {**_IQ_ABSENT, "interface_quality_blocker": "structure_unreadable"}
    if "_atom_site." not in text:
        return {**_IQ_ABSENT, "interface_quality_blocker": "coordinates_absent"}
    try:
        from rfproteina.disease.interface_quality import InterfaceUnreadableError, measure_interface
    except ImportError:
        return {**_IQ_ABSENT, "interface_quality_blocker": "module_unavailable"}
    last_reason = f"binder_chain_absent:{interest}"
    for label in _interface_quality_labels(interest):
        try:
            measured = measure_interface(path, label, cif_text=text)
        except InterfaceUnreadableError as error:
            reason = str(error)
            if reason.startswith("binder_chain_absent"):
                last_reason = reason
                continue
            return {**_IQ_ABSENT, "interface_quality_blocker": reason}
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError):
            return {**_IQ_ABSENT, "interface_quality_blocker": "interface_quality_failed"}
        record = {field: getattr(measured, field) for field in _IQ_FIELDS}
        record["target_chains"] = list(measured.target_chains)
        return record
    return {**_IQ_ABSENT, "interface_quality_blocker": last_reason}


def _first_interface_quality(samples: list[dict[str, object]]) -> dict[str, object]:
    """Top-level interface quality: the first CIF's block, with the thresholds attached once."""
    block = dict(_IQ_ABSENT)
    if samples:
        found = samples[0].get("interface_quality")
        if isinstance(found, dict):
            block = dict(found)
    try:
        from rfproteina.disease.interface_quality import REFERENCE_THRESHOLDS

        block["reference_thresholds"] = REFERENCE_THRESHOLDS
    except ImportError:  # pragma: no cover - module is in-tree
        block["reference_thresholds"] = None
    return block


def _first_designed_sequence(sequences: Sequence[tuple[str, str | None]]) -> str | None:
    for _example_id, sequence in sequences:
        if isinstance(sequence, str) and sequence:
            return sequence
    return None


def _synthesis_block(binder_kind: str, sequences: Sequence[tuple[str, str | None]]) -> dict[str, object] | None:
    """Construct-synthesis feasibility for a nucleic design, or None for any other kind.

    Recorded, not judged: every synthesis bar is a suggested default (see
    ``rfproteina.disease.synthesis``), so this reports and does not move the
    call. Only a nucleic binder has a construct to score.
    """
    if _canonical_kind(binder_kind) != "nucleic":
        return None
    sequence = _first_designed_sequence(sequences)
    if not sequence:
        return None
    try:
        from rfproteina.disease.synthesis import SequenceUnreadableError, screen_sequence
        from rfproteina.disease.synthesis import verdicts as synthesis_verdicts
    except ImportError:  # pragma: no cover - module is in-tree
        return {"synthesis_blocker": "module_unavailable"}
    try:
        report = screen_sequence(sequence)
    except SequenceUnreadableError as error:
        return {"synthesis_blocker": str(error)}
    except (ValueError, TypeError, OverflowError):
        return {"synthesis_blocker": "synthesis_failed"}
    record = report.as_record()
    record["verdicts"] = synthesis_verdicts(report)
    return record


def _read_ligand_smiles(output_dir: Path) -> str | None:
    """The first token of ``ligand_smiles.txt``, or None when the run names none.

    The screen carries no SMILES of its own. The same file feeds the structure
    alerts and the Octant CYP3A4 lookup.
    """
    path = output_dir / "ligand_smiles.txt"
    try:
        if not path.is_file():
            return None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return line.strip().split()[0]
    except (OSError, UnicodeError):
        return None
    return None


def _ligand_alerts_block(output_dir: Path) -> dict[str, object] | None:
    """RDKit structure alerts for a supplied ligand SMILES, or None when none was supplied.

    Recorded, not judged: a catalogue hit is a flag for review, and no
    probability is turned into a concentration (see
    ``rfproteina.disease.ligand_alerts``).
    """
    from rfproteina.disease.ligand_inputs import read_run_smiles

    path = output_dir / "ligand_smiles.txt"
    try:
        smiles = read_run_smiles(path)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        return {"ligand_alerts_blocker": "ligand_input_unreadable"}
    except ValueError as error:
        return {"ligand_alerts_blocker": str(error)}
    try:
        from rfproteina.disease.ligand_alerts import LigandUnreadableError, screen_smiles
        from rfproteina.disease.ligand_alerts import verdicts as ligand_verdicts
    except ImportError:  # pragma: no cover - module is in-tree
        return {"ligand_alerts_blocker": "module_unavailable"}
    try:
        screened = screen_smiles(smiles, nominated=True)
    except LigandUnreadableError as error:
        return {"ligand_alerts_blocker": str(error)}
    record = screened.as_record()
    record["verdicts"] = ligand_verdicts(screened)
    return record


def _mhc_class_i_block(
    binder_kind: str,
    sequences: Sequence[tuple[str, str | None]],
    directory: Path | None = None,
    scan: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Class I window count and whether the MHCflurry scan can run.

    The table itself is written by :func:`_apply_sequence_liability` before the
    axis is read, and only when alleles, the predictor and the presentation
    weights are all present. This block reports that state. It does not start
    a second prediction.
    """
    if _canonical_kind(binder_kind) not in _LIABILITY_KINDS:
        return None
    sequence = _first_designed_sequence(sequences)
    if not sequence:
        return None
    try:
        from rfproteina.disease.mhc_tables import class_i_readiness, windows
    except ImportError:  # pragma: no cover - module is in-tree
        return None
    try:
        window_list = windows(sequence)
    except (ValueError, TypeError):
        return None
    readiness = class_i_readiness(directory)
    record: dict[str, object] = {
        "n_windows": len(window_list),
        "predictor_available": bool(readiness["predictor_available"]),
        "weights_present": bool(readiness["weights_present"]),
        "alleles_configured": bool(readiness["alleles_configured"]),
        "table_present": bool(readiness["table_present"]),
        "note": (
            "Class I windows from the designed chain. An mhc_i.tsv already beside the run is what the "
            "axis reads. Otherwise one is written when alleles, MHCflurry, and the presentation weights "
            "are all present."
        ),
    }
    blocker = readiness.get("allele_blocker")
    if blocker:
        record["allele_blocker"] = blocker
    if scan:
        record["scan"] = scan.get("state")
        if scan.get("blocker"):
            record["scan_blocker"] = scan["blocker"]
        if scan.get("reference"):
            record["reference"] = scan["reference"]
    return record


def _optional_metric(row: dict[str, str], column: str) -> float | None:
    """A finite metrics cell rounded to four decimals. A missing column is null, not a pass."""
    if column not in row:
        return None
    raw = row.get(column)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.casefold() in {"", "nan", "none", "null", "na", "."}:
        return None
    try:
        return _round4(float(text))
    except ValueError:
        return None


def _chain_of_interest(rows: list[dict[str, str]]) -> str | None:
    """The one chain the table names. Two different chains name none.

    The first non-empty cell used to win. A table holding designs against two
    chains -- two targets concatenated, or a re-run against a different chain
    -- then handed every design the first row's chain, and the interface was
    measured against a chain that design was never built for.
    """
    named = _named_chains(rows)
    return named[0] if len(named) == 1 else None


def _named_chains(rows: list[dict[str, str]]) -> list[str]:
    """Every distinct chain of interest the table names, in first-seen order."""
    found: list[str] = []
    for row in rows:
        raw = row.get(_CHAIN_COLUMN)
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if text and text.casefold() not in _CHAIN_ABSENT and text not in found:
            found.append(text)
    return found


def _ppi_blocker(ppi_pass: bool | str) -> str:
    """Why ``ppi.pass`` is present or missing. This does not start the oracle."""
    if ppi_pass is True or ppi_pass is False:
        return "recorded"
    if any(not os.environ.get(name, "").strip() for name in _AF3_ENV):
        return "oracle_not_configured"
    return "not_requested"


def _cif_text(path: Path) -> str | None:
    """One read of a generated CIF, shared by every measurement taken from it.

    None when the file will not open, decompress or decode, which each caller
    reports in its own vocabulary rather than passing off as an absent
    measurement.
    """
    return _design_text(path)


def _load_generated_structure(path: Path, *, cif_text: str | None = None) -> object | None:
    from io import StringIO

    import numpy as np
    from atomworks.io.utils.io_utils import load_any
    from biotite import DeserializationError, InvalidFileError

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            loaded = load_any(path) if cif_text is None else load_any(StringIO(cif_text), file_type="cif")
    # A file that cannot be parsed is an unreadable structure, which is what
    # this function already reports as None a few lines below for coordinates
    # it cannot use. Letting the exception escape instead aborted the entire
    # screen on one bad file: the run log recorded exactly that on 2026-11-08,
    # as "Failed to deserialize category 'atom_site'" out of a CIF whose
    # columns had different lengths. DeserializationError is named explicitly
    # because it derives straight from Exception, so none of the others cover
    # it.
    except (DeserializationError, InvalidFileError, OSError, ValueError, KeyError, IndexError):
        return None
    coord = getattr(loaded, "coord", None)
    if coord is not None and getattr(coord, "ndim", 0) == 3:
        loaded = loaded[0]
    coord = getattr(loaded, "coord", None)
    if coord is None or getattr(coord, "ndim", 0) != 2 or coord.shape[-1] != 3 or len(coord) == 0:
        return None
    if not bool(np.isfinite(coord).any()):
        return None
    return loaded


def _binder_label(atom_array: object, interest: str) -> str | None:
    """The designed chain's label, or None when that chain is not in the structure."""
    labels = _chain_labels(atom_array)
    if interest in labels:
        return interest
    letter = interest.split("_", 1)[0]
    if letter != interest and letter in labels:
        return letter
    return None


def _chain_labels(atom_array: object) -> set[str]:
    labels = {str(label) for label in atom_array.chain_id}
    categories = atom_array.get_annotation_categories()
    if "chain_iid" in categories:
        labels.update(str(label) for label in atom_array.chain_iid)
    return labels


def _interface_numbers(atom_array: object, label: str) -> dict[str, object]:
    """The two interface numbers, each as the metric that produced it defines it.

    ``contact_ms`` is ``ContactMolecularSurfaceMetric``'s own ``contact_ms``:
    the **target-side** contact molecular surface in square angstroms, the
    surface of every chain other than ``label`` that the designed chain packs
    against. Asking for the target chain instead returns the binder-side
    figure, so the number means nothing apart from the chain it was taken for.

    ``clash_fraction`` is ``frac_atoms_clashing_1p5A`` over the **whole
    complex**, not over the designed chain: the fraction of all atoms in the
    file, target atoms included, that have a non-bonded partner within 1.5 A,
    with pairs of two fixed-input atoms excluded. A binder-only fraction would
    be the larger number, by roughly the ratio of complex atoms to binder
    atoms.

    Either metric raising leaves its own number null, so a failure is absent
    rather than a figure the metric never produced.
    """
    from rfproteina.metrics.design_metrics import compute_clash_stats
    from rfproteina.metrics.physics.contact_ms.metric import ContactMolecularSurfaceMetric

    contact: float | None = None
    clash: float | None = None
    try:
        metric = ContactMolecularSurfaceMetric(device="cpu", compute_binder_side=False)
        surface = metric.compute(predicted_atom_array=atom_array, chain_of_interest=label)
        contact = _round4(surface.get("contact_ms"))
    except (OSError, ValueError, ImportError, RuntimeError, KeyError, TypeError, IndexError):
        contact = None
    try:
        clashes = compute_clash_stats(atom_array, threshold=_CLASH_THRESHOLD)
        clash = _round4(clashes.get(_CLASH_FRACTION))
    except (OSError, ValueError, ImportError, RuntimeError, KeyError, TypeError, IndexError):
        clash = None
    return {"contact_ms": contact, "clash_fraction": clash}


def _round4(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return float(f"{number:.4f}")


def _sequence_for_chain(sequences: dict[str, str], interest: str) -> str | None:
    if interest in sequences:
        return sequences[interest]
    letter = interest.split("_", 1)[0]
    if letter != interest and letter in sequences:
        return sequences[letter]
    return None


def _polymer_sequences(text: str, ragged: list[list[str]] | None = None) -> dict[str, str]:
    """Strand id to canonical one-letter sequence, from ``entity_poly`` only.

    Read from **both** spellings mmCIF allows. A category with several rows is a
    ``loop_``; the same category with one row is written as plain
    ``_name.item value`` lines, and this repository's own writer emits both. So
    a structure with exactly one polymer entity -- a monomer, an inverse-folding
    output, an enzyme design, a motif scaffold, a small-molecule binder -- has
    no ``entity_poly`` loop at all, and reading loops alone found nothing in
    every one of them. The sequence then read as absent, which is a different
    fact from a design that has none: every sequence-liability axis reported
    ``sequence_absent`` and went untested, for a sequence the file was holding
    in plain sight.

    ``_nonloop_category`` is the same reader ``_struct_conn_rows`` and
    ``_chem_components`` already use for their own single-row categories; it
    skips loop bodies, so a loop row is never read twice. Loop rows are
    collected first and ``setdefault`` keeps them, so a file with a loop is
    read exactly as before.

    ``ragged`` collects the columns of an ``entity_poly`` loop whose values did
    not fill its rows. A caller that passes one can say the file did not parse
    rather than that the chain held no sequence, which is a different fact and a
    different thing to go and fix. A non-loop category cannot be ragged -- it
    has no rows to under-fill -- so it adds nothing there.
    """
    dropped: list[list[str]] = []
    found: dict[str, str] = {}
    for columns, rows in _cif_loops(text, ragged=dropped):
        if _STRAND not in columns:
            continue
        sequence_column = _CANONICAL if _CANONICAL in columns else _ONE_LETTER
        if sequence_column not in columns:
            continue
        _collect_sequences(found, rows, sequence_column)
    _collect_sequences(found, _nonloop_category(text, "_entity_poly."), None)
    if ragged is not None:
        ragged.extend(columns for columns in dropped if _STRAND in columns)
    return found


def _collect_sequences(
    found: dict[str, str],
    rows: Sequence[Mapping[str, str]],
    sequence_column: str | None,
) -> None:
    """Add each row's sequence under every strand it names. First one wins.

    ``sequence_column`` is fixed for a loop, where the column set is the loop's
    and the same for every row. A non-loop category is one row, so it is chosen
    per row -- by which column is *present*, not by which holds something.
    Preferring presence is what the loop path does, and it is the right way
    round: the canonical column spells a modified residue as its parent amino
    acid, while the raw column spells it as a parenthesised component code that
    would be read here as residues named ``(``, ``A``, ``L``, ``Y``, ``)``.
    """
    for row in rows:
        column = sequence_column or (_CANONICAL if _CANONICAL in row else _ONE_LETTER)
        sequence = "".join(row.get(column, "").split()).upper()
        if not sequence or sequence in {".", "?"}:
            continue
        for strand in _strands(row.get(_STRAND, "")):
            found.setdefault(strand, sequence)


def _strands(raw: str) -> list[str]:
    if raw.strip() in {"", ".", "?"}:
        return []
    return [part.strip() for part in raw.split(",") if part.strip() not in {"", ".", "?"}]


def _disulfide_partners(text: str, chain: str) -> set[int] | None:
    """Sequence positions on ``chain`` that a disulfide names.

    None means the CIF names no disulfide at all, so free cysteine stays
    untested. A loop whose values do not fill a row is not a bond row. One
    non-loop ``_struct_conn`` category is one bond row. An empty set means a
    disulfide was named somewhere and this chain has no partner in it.

    A bond of another kind is not an answer to this question. Counting any
    ``_struct_conn`` row as one made a metal-coordination row stand for
    "disulfides were recorded and this chain has none", and every cysteine was
    then reported free as a hard count. That is reachable through this
    repository's own parse preset, which keeps ``covale`` and ``metalc`` while
    dropping ``disulf``: a design against a metalloprotein carries coordination
    rows and no disulfide rows, and the count was taken from a file that had
    said nothing about disulfides.
    """
    partners: set[int] = set()
    named = False
    for row in _struct_conn_rows(text):
        if not _row_is_disulfide(row):
            continue
        named = True
        for which in ("ptnr1", "ptnr2"):
            residue = _partner_residue(row, which, chain)
            if residue is not None:
                partners.add(residue)
    if not named:
        return None
    return partners


def _struct_conn_rows(text: str) -> list[dict[str, str]]:
    """Bond rows from ``_struct_conn`` loops and from a non-loop category.

    A loop whose values do not fill a row contributes nothing. A row that does
    not name ``conn_type_id`` is omitted. Loop rows are not read again as a
    non-loop category.
    """
    rows: list[dict[str, str]] = []
    for columns, loop_rows in _cif_loops(text, prefixes=("_struct_conn.",)):
        if "_struct_conn.conn_type_id" not in columns:
            continue
        rows.extend(loop_rows)
    rows.extend(_nonloop_struct_conn(text))
    return rows


def _nonloop_struct_conn(text: str) -> list[dict[str, str]]:
    """One row for each ``_struct_conn`` category written outside ``loop_``."""
    return [row for row in _nonloop_category(text, "_struct_conn.") if "_struct_conn.conn_type_id" in row]


def _chem_components(text: str) -> list[dict[str, str]]:
    """``_chem_comp`` rows from loops and from a single non-loop category."""
    rows: list[dict[str, str]] = []
    for columns, loop_rows in _cif_loops(text, prefixes=("_chem_comp.",)):
        if "_chem_comp.type" not in columns:
            continue
        rows.extend(loop_rows)
    rows.extend(row for row in _nonloop_category(text, "_chem_comp.") if "_chem_comp.type" in row)
    return rows


def _nonloop_category(text: str, prefix: str) -> list[dict[str, str]]:
    """One row per category written outside ``loop_`` for tags that start with ``prefix``.

    A value may sit on the tag line or on the next non-blank line. A following
    tag, comment, loop, or data block stays on its own line.
    """
    lines = text.splitlines()
    found: list[dict[str, str]] = []
    current: dict[str, str] = {}
    index = 0
    count = len(lines)
    while index < count:
        stripped = lines[index].strip()
        if stripped == "loop_":
            current = _store_category_row(found, current)
            index = _skip_one_loop(lines, index)
            continue
        if stripped.startswith(prefix):
            tag, value, index = _nonloop_item(lines, index)
            current[tag] = value
            continue
        if current and (stripped == "#" or stripped.startswith(("data_", "_"))):
            current = _store_category_row(found, current)
        index += 1
    _store_category_row(found, current)
    return found


def _store_category_row(found: list[dict[str, str]], row: dict[str, str]) -> dict[str, str]:
    if row:
        found.append(row)
    return {}


def _nonloop_item(lines: list[str], index: int) -> tuple[str, str, int]:
    """Tag, value, and the index of the next line still to be read."""
    parts = lines[index].strip().split(None, 1)
    tag = parts[0]
    if len(parts) == 2:
        return tag, _bare(parts[1]), index + 1
    nxt = index + 1
    count = len(lines)
    while nxt < count and not lines[nxt].strip():
        nxt += 1
    if nxt < count and lines[nxt].startswith(";"):
        text, nxt = _semicolon_text(lines, nxt)
        return tag, text, nxt
    if nxt < count and not _category_boundary(lines[nxt].strip()):
        return tag, _bare(lines[nxt].strip()), nxt + 1
    return tag, "", nxt


def _category_boundary(stripped: str) -> bool:
    """True when this line starts another CIF category, comment, loop, or data block."""
    return stripped == "loop_" or stripped.startswith(("data_", "_", "#"))


def _semicolon_text(lines: list[str], index: int) -> tuple[str, int]:
    count = len(lines)
    parts: list[str] = []
    first = lines[index][1:]
    if first:
        parts.append(first)
    index += 1
    while index < count and lines[index].strip() != ";":
        parts.append(lines[index])
        index += 1
    if index < count and lines[index].strip() == ";":
        index += 1
    return "".join(parts).strip(), index


def _skip_one_loop(lines: list[str], index: int) -> int:
    """The index of the first line after the loop that begins at ``index``."""
    count = len(lines)
    if index < count and lines[index].strip() == "loop_":
        index += 1
    while index < count and lines[index].lstrip().startswith("_"):
        index += 1
    return _skip_loop_body(lines, index)


def _row_is_disulfide(row: Mapping[str, str]) -> bool:
    kind = _token(row.get("_struct_conn.conn_type_id", "")).casefold()
    if kind in {"disulf", "disulfide"}:
        return True
    if kind != "covale":
        return False
    comps = (
        _token(row.get("_struct_conn.ptnr1_label_comp_id", "")).upper(),
        _token(row.get("_struct_conn.ptnr2_label_comp_id", "")).upper(),
    )
    atoms = (
        _token(row.get("_struct_conn.ptnr1_label_atom_id", "")).upper(),
        _token(row.get("_struct_conn.ptnr2_label_atom_id", "")).upper(),
    )
    return comps == ("CYS", "CYS") and atoms == ("SG", "SG")


def _partner_residue(row: Mapping[str, str], which: str, chain: str) -> int | None:
    label_chain = _token(row.get(f"_struct_conn.{which}_label_asym_id", ""))
    auth_chain = _token(row.get(f"_struct_conn.{which}_auth_asym_id", ""))
    label_seq = _positive_int(row.get(f"_struct_conn.{which}_label_seq_id", ""))
    auth_seq = _positive_int(row.get(f"_struct_conn.{which}_auth_seq_id", ""))
    if _same_chain(label_chain, chain) and label_seq is not None:
        return label_seq
    if _same_chain(auth_chain, chain) and auth_seq is not None:
        return auth_seq
    return None


def _same_chain(named: str, interest: str) -> bool:
    if not named:
        return False
    if named == interest:
        return True
    letter = interest.split("_", 1)[0]
    return letter != interest and named == letter


def _token(raw: str) -> str:
    text = _bare(raw).strip()
    if text in {".", "?"}:
        return ""
    return text


def _positive_int(raw: str) -> int | None:
    text = _token(raw)
    if not text.isdigit():
        return None
    number = int(text)
    if number < 1:
        return None
    return number


def _cif_tokens(line: str) -> list[str]:
    """Values on one loop line. A quoted value keeps the spaces inside it.

    The closing quote is the matching quote followed by whitespace or the end
    of the line. Quote marks are not part of the value. An unclosed quote is
    dropped, so the raw remainder is not stored as a cell.

    An unquoted ``#`` starting a token ends the line, which is the CIF comment
    rule. Only a line that was exactly ``#`` was recognised, so an ordinary
    trailing comment contributed its words as cells, the loop no longer divided
    by its own width, and the whole loop was dropped -- reported afterwards as a
    chain with no sequence in it.
    """
    tokens: list[str] = []
    index = 0
    length = len(line)
    while index < length:
        while index < length and line[index].isspace():
            index += 1
        if index >= length:
            break
        mark = line[index]
        if mark == "#":
            break
        if mark in {"'", '"'}:
            end = index + 1
            while end < length and not (line[end] == mark and (end + 1 == length or line[end + 1].isspace())):
                end += 1
            if end >= length:
                break
            tokens.append(_bare(line[index : end + 1]))
            index = end + 1
            continue
        end = index + 1
        while end < length and not line[end].isspace():
            end += 1
        tokens.append(line[index:end])
        index = end
    return tokens


def _cif_loops(
    text: str,
    prefixes: tuple[str, ...] = (),
    ragged: list[list[str]] | None = None,
) -> list[tuple[list[str], list[dict[str, str]]]]:
    """Every mmCIF loop as column names and row dictionaries.

    ``prefixes`` keeps only loops whose columns start with one of them. Other
    loops are skipped, so an accession read does not build the atom site. A
    quoted value is one cell, including when it contains spaces. An unclosed
    quote is not a cell.

    A loop whose values do not fill its rows yields no rows, as before. When
    ``ragged`` is given, that loop's columns are collected into it, so a caller
    can say the file did not parse instead of reporting an absent sequence.
    """
    lines = text.splitlines()
    loops: list[tuple[list[str], list[dict[str, str]]]] = []
    index = 0
    count = len(lines)
    while index < count:
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        columns: list[str] = []
        while index < count and lines[index].lstrip().startswith("_"):
            columns.append(lines[index].strip())
            index += 1
        if prefixes and not any(column.startswith(prefixes) for column in columns):
            index = _skip_loop_body(lines, index)
            continue
        values: list[str] = []
        while index < count:
            stripped = lines[index].strip()
            if stripped in {"loop_", "#"} or stripped.startswith(("data_", "_")):
                break
            if not stripped:
                index += 1
                continue
            if lines[index].startswith(";"):
                parts: list[str] = []
                first = lines[index][1:]
                if first:
                    parts.append(first)
                index += 1
                while index < count and lines[index].strip() != ";":
                    parts.append(lines[index])
                    index += 1
                if index < count and lines[index].strip() == ";":
                    index += 1
                values.append("".join(parts))
                continue
            values.extend(_cif_tokens(stripped))
            index += 1
        width = len(columns)
        rows: list[dict[str, str]] = []
        if width and len(values) % width == 0:
            for start in range(0, len(values), width):
                rows.append(dict(zip(columns, values[start : start + width], strict=True)))
        elif columns and values and ragged is not None:
            ragged.append(list(columns))
        if columns:
            loops.append((columns, rows))
    return loops


def _skip_loop_body(lines: list[str], index: int) -> int:
    """The index of the line that ends a loop, without reading its values."""
    count = len(lines)
    while index < count:
        stripped = lines[index].strip()
        if stripped in {"loop_", "#"} or stripped.startswith(("data_", "_")):
            break
        if lines[index].startswith(";"):
            index += 1
            while index < count and lines[index].strip() != ";":
                index += 1
            if index < count and lines[index].strip() == ";":
                index += 1
            continue
        index += 1
    return index


def _samples_clause(screen: Mapping[str, object]) -> str | None:
    """Count for more than one CIF. An absent sample is not counted as true."""
    samples = screen.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        return None
    true_count = 0
    failed: list[str] = []
    for item in samples:
        if not isinstance(item, dict):
            continue
        if item.get("sequence_pass") is True:
            true_count += 1
        elif item.get("sequence_pass") is False:
            name = item.get("example_id")
            if isinstance(name, str) and name:
                failed.append(name)
    clause = f"Samples {len(samples)}, sequence_pass {true_count} of {len(samples)}"
    if failed:
        clause = f"{clause}. Failed sequence_pass: {', '.join(failed)}"
    return clause


def _with_interface_thresholds(quality: object) -> object:
    """Attach reference values and identify them as unapplied on the local scale."""
    if not isinstance(quality, Mapping):
        return quality
    attached = dict(quality)
    attached[PROVENANCE_KEY] = {
        name: {**record, "bar": _interface_bar(name), "applied": False}
        for name, record in describe_interface_thresholds().items()
    }
    return attached


def _interface_bar(name: str) -> object:
    """The historical reference value, or ``None`` for a quantity without one.

    Imported inside the function because ``interface_quality`` pulls numpy and
    biotite, and every other use of it in this module is lazy for that reason.
    A machine without them still gets the provenance; it just cannot quote the
    number beside it.
    """
    try:
        from rfproteina.disease.interface_quality import REFERENCE_THRESHOLDS
    except ImportError:
        return None
    entry = REFERENCE_THRESHOLDS.get(name, {})
    minimum, maximum = entry.get("min"), entry.get("max")
    if minimum is not None:
        return {"min": minimum}
    if maximum is not None:
        return {"max": maximum}
    return None


#: The two sequence bars, each with the measurement it judges.
_SEQUENCE_BARS = (("gravy", "gravy_max"), ("instability_index", "instability_max"))


def _failed_sequence_bars(sequence: Mapping[str, object], cuts: object) -> str:
    """Which bars a failed ``sequence_pass`` failed, and what kind of bar each is.

    ``sequence_pass false`` on its own does not say which of the two numbers
    was out, and still less whether the bar that rejected the design is a
    published rule or this program's own convention. On the screen that
    prompted this, GRAVY 0.1304 exceeded a convention of 0.1 while the
    instability index of 30.0 cleared the published bar of 40 comfortably: the
    verdict was decided entirely by a number this program chose, and the line
    read exactly like one decided by Guruprasad 1990.

    Returns an empty string when the design passed, when the bars are not on
    the record -- a screen stored before ``cuts`` carried them -- or when
    nothing can be compared, because a sentence that cannot be supported is
    worse than none.
    """
    if sequence.get("sequence_pass") is not False or not isinstance(cuts, Mapping):
        return ""
    failed: list[str] = []
    for measured_key, cut_key in _SEQUENCE_BARS:
        measured = sequence.get(measured_key)
        bar = cuts.get(cut_key)
        if not _is_number(measured) or not _is_number(bar):
            continue
        if float(measured) > float(bar):
            failed.append(f"failed {cut_key} {bar}, {provenance_of(cut_key).kind}")
    return "; ".join(failed)


def _measured_line(entry_id: str, chain: str, screen: Mapping[str, object]) -> str | None:
    """The computed numbers. A screen with nothing measured keeps the single call line."""
    sequence = screen.get("sequence_developability")
    sequence_pass = sequence.get("sequence_pass") if isinstance(sequence, dict) else _ABSENT
    pre_oracle = screen.get("pre_oracle_pass", _ABSENT)
    ppi = screen.get("ppi_pass", _ABSENT)
    interface = screen.get("designed_interface")
    contact = interface.get("contact_ms") if isinstance(interface, dict) else None
    clash = interface.get("clash_fraction") if isinstance(interface, dict) else None
    blocker = screen.get("ppi_blocker")
    has_blocker = isinstance(blocker, str) and blocker in _PPI_BLOCKERS
    samples_clause = _samples_clause(screen)
    liability_clause = _liability_clause(screen)
    toxicology_clause = _toxicology_clause(screen)
    evolution_clause = _coevolution_clause(screen)
    chemical_clause = _chemical_clause(screen)
    modality_clause = _modality_clause(screen)
    spread_clause = _diversity_clause(screen)
    solubility_clause = _solubility_clause(screen)
    ligand_solubility = screen.get("ligand_solubility")
    ligand_solubility_clause = None
    if isinstance(ligand_solubility, dict):
        predicted, total = (ligand_solubility.get(k) for k in ("unique_predicted", "unique_compounds"))
        if ligand_solubility.get("status") == "untested" and isinstance(ligand_solubility.get("blocker"), str):
            ligand_solubility_clause = f"ADMET-AI aqueous solubility untested ({ligand_solubility['blocker']})"
        elif type(predicted) is int and type(total) is int and 0 <= predicted <= total:
            ligand_solubility_clause = (
                f"ADMET-AI aqueous solubility predicted for {predicted}/{total} distinct compounds"
            )
            if ligand_solubility.get("status") == "partial":
                ligand_solubility_clause += ", partial input coverage"
    affinity = screen.get("prodigy_affinity")
    pka = screen.get("qupkake")
    metabolites = screen.get("sygma")
    metabolite_clause = None
    if isinstance(metabolites, dict):
        if metabolites.get("status") == "untested" and isinstance(metabolites.get("blocker"), str):
            metabolite_clause = f"SyGMa untested ({metabolites['blocker']})"
        elif all(type(metabolites.get(k)) is int for k in ("ranked_products", "screened_parents", "attempted")):
            metabolite_clause = (
                f"SyGMa ranked {metabolites['ranked_products']} metabolite hypotheses across "
                f"{metabolites['screened_parents']} parents from {metabolites['attempted']} nominations "
                f"({metabolites.get('status')})"
            )
    pka_clause = None
    if isinstance(pka, dict):
        if pka.get("status") == "untested" and isinstance(pka.get("blocker"), str):
            pka_clause = f"QupKake untested ({pka['blocker']})"
        elif all(type(pka.get(k)) is int for k in ("predicted_sites", "compounds_with_predictions", "attempted")):
            pka_clause = (
                f"QupKake predicted {pka['predicted_sites']} microscopic pKa values across "
                f"{pka['compounds_with_predictions']} compounds from {pka['attempted']} nominations "
                f"({pka.get('status')})"
            )
    vina = screen.get("vina_pose")
    tnp = screen.get("tnp_profile")
    tnp_clause = None
    if isinstance(tnp, dict):
        complete, partial, attempted = (tnp.get(k) for k in ("profiled", "partial", "attempted"))
        if all(type(n) is int and n >= 0 for n in (complete, partial, attempted)) and complete + partial <= attempted:
            tnp_clause = f"TNP full profiles {complete}/{attempted}, partial profiles {partial}"
    vina_clause = None
    if isinstance(vina, dict):
        scored, total = vina.get("scored"), vina.get("attempted")
        if type(scored) is int and type(total) is int and 0 <= scored <= total:
            vina_clause = f"Vina fixed-pose score reported for {scored}/{total} designs"
    affinity_clause = None
    if isinstance(affinity, dict):
        scored, total = affinity.get("predicted"), affinity.get("attempted")
        if type(scored) is int and type(total) is int and 0 <= scored <= total:
            affinity_clause = f"PRODIGY affinity reported for {scored}/{total} designs"
    if (
        sequence_pass == _ABSENT
        and pre_oracle == _ABSENT
        and ppi == _ABSENT
        and not _is_number(contact)
        and not _is_number(clash)
        and not has_blocker
        and samples_clause is None
        and liability_clause is None
        and toxicology_clause is None
        and evolution_clause is None
        and chemical_clause is None
        and modality_clause is None
        and spread_clause is None
        and solubility_clause is None
        and ligand_solubility_clause is None
        and pka_clause is None
        and metabolite_clause is None
        and affinity_clause is None
        and vina_clause is None
        and tnp_clause is None
    ):
        return None
    parts: list[str] = []
    if isinstance(sequence, dict) and sequence_pass != _ABSENT:
        clause = (
            f"gravy {sequence['gravy']}, instability {sequence['instability_index']}, "
            f"sequence_pass {_word(sequence_pass)}"
        )
        failed = _failed_sequence_bars(sequence, screen.get("cuts"))
        if failed:
            clause = f"{clause} ({failed})"
        parts.append(clause)
    if chemical_clause is not None:
        parts.append(chemical_clause)
    if modality_clause is not None:
        parts.append(modality_clause)
    parts.append(f"pre_oracle_pass {_word(pre_oracle)}")
    parts.append(f"ppi.pass {_word(ppi)}")
    if _is_number(contact) and _is_number(clash):
        parts.append(f"contact_ms {contact}, clash_fraction {clash}")
    elif _is_number(contact):
        parts.append(f"contact_ms {contact}")
    elif _is_number(clash):
        parts.append(f"clash_fraction {clash}")
    if has_blocker:
        parts.append(f"ppi_blocker {blocker}")
    if samples_clause is not None:
        parts.append(samples_clause)
    if spread_clause is not None:
        parts.append(spread_clause)
    if solubility_clause is not None:
        parts.append(solubility_clause)
    if ligand_solubility_clause is not None:
        parts.append(ligand_solubility_clause)
    if pka_clause is not None:
        parts.append(pka_clause)
    if metabolite_clause is not None:
        parts.append(metabolite_clause)
    if affinity_clause is not None:
        parts.append(affinity_clause)
    if vina_clause is not None:
        parts.append(vina_clause)
    if tnp_clause is not None:
        parts.append(tnp_clause)
    if liability_clause is not None:
        parts.append(liability_clause)
    if toxicology_clause is not None:
        parts.append(toxicology_clause)
    if evolution_clause is not None:
        parts.append(evolution_clause)
    body = ". ".join(parts)
    if not entry_id and not chain:
        return f"Measured: {body}."
    return f"Measured {entry_id} chain {chain}: {body}."


def _modality_clause(screen: Mapping[str, object]) -> str | None:
    """A drug-type sentence. A count of zero stays off the measured line."""
    modality = screen.get("modality_screen")
    if not isinstance(modality, dict):
        return None
    return modality_phrase(modality)


def _chemical_clause(screen: Mapping[str, object]) -> str | None:
    """A hotspot sentence. A count of zero stays off the measured line."""
    chemical = screen.get("chemical_liability")
    if not isinstance(chemical, dict):
        return None
    return chemical_phrase(chemical)


def _liability_clause(screen: Mapping[str, object]) -> str | None:
    """Hit, clear, or a stored score, from every axis block the record carries.

    One loop over ``_LIABILITY_BLOCKS`` rather than a clause per block: the
    sequence axes were the only block rendered, so a curated clear over a
    measured hit on paralog, a plasma axis or either nucleic axis printed
    nothing at all and the line read as a plain screened_clear. A curated state
    is printed wherever it sits, including on an axis no tuple here declares --
    a peptide's ``membrane`` is one -- because a human's call on an axis is the
    part of a verdict a reader cannot reconstruct. The measurements of the other
    blocks keep their own clauses, so the sentence of an uncurated run is
    unchanged.
    """
    parts: list[str] = []
    for block, declared, measured in _LIABILITY_BLOCKS:
        records = screen.get(block)
        if not isinstance(records, dict):
            continue
        for axis in _block_axes(records, declared):
            record = records.get(axis)
            if not isinstance(record, dict):
                continue
            curated = record.get("source") == "axis_evidence"
            if not curated and not (measured and axis in declared):
                continue
            piece = _liability_piece(axis, record)
            if piece is not None:
                parts.append(piece)
    if not parts:
        return None
    return ". ".join(parts)


def _block_axes(records: Mapping[str, object], declared: Sequence[str]) -> list[str]:
    """The block's declared axes in order, then anything else it carries."""
    return [*[axis for axis in declared if axis in records], *[axis for axis in records if axis not in declared]]


def _solubility_clause(screen: Mapping[str, object]) -> str | None:
    """Supplied solubility scores. A run with no table has none."""
    supplied = screen.get("solubility")
    if not isinstance(supplied, dict):
        return None
    return solubility_phrase(supplied)


def _diversity_clause(screen: Mapping[str, object]) -> str | None:
    """The spread across this run's designs. A run with one design has none."""
    spread = screen.get("design_diversity")
    if not isinstance(spread, dict):
        return None
    return diversity_phrase(spread)


def _coevolution_clause(screen: Mapping[str, object]) -> str | None:
    """The co-evolution note, when one was stored. A missing table stays out."""
    target = screen.get("target_liability")
    if not isinstance(target, dict):
        return None
    paralog = target.get("paralog")
    if not isinstance(paralog, dict):
        return None
    return coevolution_phrase(paralog)


def _toxicology_clause(screen: Mapping[str, object]) -> str | None:
    """Hit, clear, or a stored panel reading. An axis with no number stays out."""
    toxicology = screen.get("toxicology")
    if not isinstance(toxicology, dict):
        return None
    parts: list[str] = []
    for axis in LIGAND_AXES:
        record = toxicology.get(axis)
        if not isinstance(record, dict):
            continue
        piece = _toxicology_piece(axis, record)
        if piece is not None:
            parts.append(piece)
    if not parts:
        return None
    return ". ".join(parts)


def _toxicology_piece(axis: str, record: Mapping[str, object]) -> str | None:
    source = record.get("source")
    state = record.get("state")
    measurement = record.get("measurement")
    if source == "measured" and state in _STATES:
        if _is_number(measurement):
            return f"{axis} {state} {measurement}, nM panel reading"
        return f"{axis} {state}"
    if source == "axis_evidence" and state in _STATES:
        return _liability_piece(axis, record)
    if source == "untested" and state == "untested" and _is_number(measurement):
        return f"{axis} untested {measurement}, nM panel reading"
    return None


def _immunogenicity_phrase(state: object, record: Mapping[str, object]) -> str:
    """The qualifier after a measured Calis number.

    ``every 9-mer <= 0`` is only an unrestricted clear. A restricted clear
    scored the presented ligands, and a hit has positive windows, so that
    sentence would report a clear the measurement does not support.
    """
    windows = record.get("positive_windows")
    count = windows if isinstance(windows, int) and not isinstance(windows, bool) else None
    if (
        state == "clear"
        and record.get("presentation") == "restricted"
        and count == 0
        and record.get("presented_complete") is True
    ):
        return ", every presented 9-mer <= 0"
    if state == "clear" and count == 0:
        return ", every 9-mer <= 0"
    if count is not None and count > 0:
        return f", positive 9-mers {count}"
    return ""


def _liability_piece(axis: str, record: Mapping[str, object]) -> str | None:
    source = record.get("source")
    state = record.get("state")
    measurement = record.get("measurement")
    if source == "measured" and state in _STATES:
        if axis == "proteome_offtarget" and _is_number(measurement):
            piece = f"{axis} {state} {measurement}, exact 9-mer identity"
        elif axis == "immunogenicity" and _is_number(measurement):
            piece = f"{axis} {state} {measurement}, Calis 2013 class I 9-mer" f"{_immunogenicity_phrase(state, record)}"
        elif axis == "ada" and _is_number(measurement):
            piece = f"{axis} {state} {measurement}, median DR panel rank, Paul 2015 seven-allele"
        elif _is_number(measurement):
            piece = f"{axis} {state} {measurement}"
        else:
            piece = f"{axis} {state}"
        if axis == "proteome_offtarget" and state == "hit":
            names = _failed_names(record.get("failed"))
            if names:
                piece = f"{piece}. Failed proteome_offtarget: {', '.join(names)}"
        return piece
    if source == "axis_evidence" and state in _STATES:
        # A declared state printed nothing at all, because only "measured" was
        # rendered. An axis_evidence clear over a measured hit therefore left a
        # screened_clear with no axis named and no sign that a human had
        # declared it -- the record kept measured_state, the only output a
        # reader sees did not. The wording here is deliberately the record's own
        # key names rather than new prose.
        piece = f"{axis} {state}, source axis_evidence"
        measured_state = record.get("measured_state")
        if isinstance(measured_state, str) and measured_state and measured_state != state:
            piece = f"{piece}, measured {measured_state}"
            if _is_number(measurement):
                piece = f"{piece} {measurement}"
        measured_blocker = record.get("measured_blocker")
        if isinstance(measured_blocker, str) and measured_blocker:
            piece = f"{piece}, measured_blocker {measured_blocker}"
        return piece
    if source != "untested" or state != "untested" or not _is_number(measurement):
        return None
    method = record.get("method")
    if axis == "immunogenicity" and method == "calis_2013_class_i_9mer":
        windows = record.get("positive_windows")
        piece = f"immunogenicity untested {measurement}, Calis 2013 class I 9-mer"
        if isinstance(windows, int) and not isinstance(windows, bool):
            piece = f"{piece}, positive 9-mers {windows}"
        return piece
    if method == SUPPLIED_METHOD:
        # A number the user's own table carried. It is reported rather than
        # dropped, and it is not named after a formula this module implements.
        return f"{axis} untested {measurement}, supplied table"
    return None


def _failed_names(failed: object) -> list[str]:
    if not isinstance(failed, list):
        return []
    return [str(name) for name in failed if isinstance(name, str) and name]


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)


def _word(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return _ABSENT
