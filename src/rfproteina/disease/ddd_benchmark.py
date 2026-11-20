"""Answer the DDD Benchmark items this program covers, and decline the rest by name.

The Insilico Drug Discovery and Development Benchmark scores a system across
seven domains. Three of them are this program's own subject matter; the rest are
not, and no attempt is made on them. Within the three, coverage is partial, and
the partiality is the point of this module: an item this program cannot answer
comes back as a **named decline**, never as a guess.

Scope, verified from the benchmark's Explorer on 2026-11-18 (7 domains, 308
benchmarks, Core 61):

* **Therapeutic Biomolecules (6)** -- "Antibody Developability": HAC Retention
  Time, HIC Retention Time, CHO Polyreactivity, Ovalbumin Polyreactivity, SMAC, Tm.
* **Structural Biology (2)** -- "ResidueBench": Evolutionary pressure, Mutations.
* **Target Biology (12)** -- nine "Target Prediction for *X* Diseases" plus three
  "Single-Cell Perturbation" sets.

Ten of the twenty are answerable here. That number is in a test, so it moves
only on purpose.

Why this is reported-only
-------------------------
These answers never reach a verdict. They are built in the shape
``stability.verdicts`` already uses for measured-and-reported quantities, and
deliberately not as an axis: :data:`combinations.CLASS_CONDITION_DIM` is
``2 * len(AXES)`` and is written into every annotated CIF, so adding an axis
would invalidate fine-tuned checkpoints and stored CIFs. An axis also moves a
verdict through ``rollup_axis``, and an external scorer's question must never
change whether a design reads ``screened_clear``. A benchmark measures this
program; a verdict describes a design.

What a decline is for
---------------------
The benchmark reports a **valid fraction** beside accuracy, so a decline is a
first-class outcome rather than a failure to produce one. Every decline here
carries a blocker name and the sentence that goes with it, which is the one
thing this program has that a frontier model does not: 164 blocker names, each
with a tested sentence. A scorer asking "why no answer" gets prose.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease import corpus_provenance
from rfproteina.disease.provenance import code_version

#: The three domains attempted. Named so a reader can see what is left out:
#: Medicinal Chemistry, Chemical Synthesis, Clinical Trials and Longevity are
#: out of scope, the first because ADMET-AI is fitted on the corpus that column
#: is built from and the rest because this program has no module for them.
THERAPEUTIC_BIOMOLECULES = "Therapeutic Biomolecules"
STRUCTURAL_BIOLOGY = "Structural Biology"
TARGET_BIOLOGY = "Target Biology"

#: A ranked list of items, scored by Precision@K / AP@K.
RANKING = "ranking"
#: A single number, scored by mean absolute error.
SCALAR = "scalar"


@dataclass(frozen=True)
class BenchmarkItem:
    """One benchmark, and how this program does or does not answer it.

    ``answers_from`` is the method slug of whatever would produce the number,
    and stays set even when the item is declined -- that is what lets the
    corpus declaration ride on a decline, so the contaminated items cannot hide
    behind having produced nothing.

    ``reference_dataset`` is the corpus label the benchmark corresponds to, used
    to derive the overlap. ``None`` where this program has no basis to claim a
    correspondence; guessing one would manufacture a finding.

    ``blocker`` is ``None`` for an answerable item and a blocker name otherwise.
    The two are mutually exclusive and a test enforces it. The field is called
    ``blocker`` rather than ``declined_blocker`` so that every construction below
    is a ``blocker=`` keyword carrying a literal, which is the shape
    ``tests/test_disease_board_vocabulary.py`` can see. A name the scan cannot
    find is a name nothing guarantees a sentence for.
    """

    benchmark: str
    domain: str
    group: str
    answer_kind: str
    answers_from: str | None
    reference_dataset: str | None
    blocker: str | None

    @property
    def answerable(self) -> bool:
        """Whether this program has a path to a number today."""
        return self.blocker is None


#: One sentence per blocker this module emits. The vocabulary test routes this
#: module to this table rather than to ``screen_board._BLOCKERS``, because the
#: board never renders a benchmark answer and duplicating the sentences into it
#: would be the mistake that test's own docstring records costing seventeen
#: duplicated lines when the results modules arrived.
_BLOCKER_TEXT: dict[str, str] = {
    "oracle_weights_over_size_cap:HIC": (
        "CrossAbSense predicts HIC from ESM-C 6B, which is 25.5 GB and over this program's "
        "download cap, so the weights are not installed and no number can be produced."
    ),
    "oracle_weights_over_size_cap:Tm2": (
        "CrossAbSense predicts Tm2 from ESM-C 6B and ProtT5 together, which is 37.0 GB and over "
        "this program's download cap, so the weights are not installed. The benchmark also does "
        "not say which melting readout it means, and GDPa1 has three."
    ),
    "oracle_weights_not_installed:PR_CHO": (
        "CrossAbSense predicts CHO polyreactivity from ESM-C 300M alone, which fits under the "
        "download cap, but the weights are left uninstalled on purpose: PR_CHO is a primary "
        "column of GDPa1, the dataset this oracle was fitted on, so a number from it would be "
        "recalled as much as predicted."
    ),
    "no_model_for_assay:HAC": (
        "GDPa1 measured HAC retention time, but CrossAbSense has no head for it, so this "
        "program has nothing that predicts it."
    ),
    "no_model_for_assay:PR_OVA": (
        "No model in this program predicts polyreactivity against ovalbumin. The CHO lysate "
        "head measures a different antigen, and reporting it here would be a wrong answer "
        "stated confidently."
    ),
    "no_model_for_assay:SMAC": (
        "No model in this program predicts SMAC retention. AC-SINS is also a self-interaction "
        "proxy but is measured on a different scale, so substituting it would be a guess."
    ),
    "no_conservation_model": (
        "Nothing in this program computes per-residue evolutionary pressure. The co-evolution "
        "module reads a caller-supplied paralog table and infers no conservation of its own."
    ),
    "no_perturbation_model": (
        "This program does not predict transcriptional response to genetic perturbation. It has "
        "no single-cell module at all."
    ),
    "producer_failed": (
        "This item is answerable, but the tool that answers it did not finish. The reason is "
        "appended to this sentence. A half-finished ranking is not reported, because a scorer "
        "would read it as an answer."
    ),
    "runner_not_supplied": (
        "This item is answerable, but the call supplied nothing to produce the answer with. "
        "Pass rank_fn, or run it through the CLI, which wires the real producer."
    ),
}

#: ``reference_dataset`` for the thermostability benchmark. GDPa1 measures three
#: melting readouts and the benchmark says only "Tm"; Tm2 is the column
#: CrossAbSense has a head for, so it is the one whose overlap is real. Which
#: column the benchmark means is on the list of questions for Insilico.
_TM_COLUMN = "Tm2"

#: The nine disease groups of TargetBench, exactly as the Explorer names them.
_TARGET_DISEASE_GROUPS = (
    "Target Prediction for Cancer Diseases",
    "Target Prediction for Cardiovascular Diseases",
    "Target Prediction for Fibrotic Diseases",
    "Target Prediction for Inflammatory & Immunological Diseases",
    "Target Prediction for Mental & Behavioral Disorders",
    "Target Prediction for Metabolic & Endocrine Diseases",
    "Target Prediction for Neurological Diseases",
    "Target Prediction for Ophthalmic Diseases",
    "Target Prediction for Reproductive Health Conditions",
)

#: The three Single-Cell Perturbation sets, as the Explorer names them.
_PERTURBATION_SETS = (
    "iPSC, genome-wide (Pan-Genome)",
    "iPSC, strong perturbations",
    "K562, genome-scale (Replogle)",
)


#: Shared by all six developability items, so only what differs is repeated.
_ASSAY = {
    "domain": THERAPEUTIC_BIOMOLECULES,
    "group": "Antibody Developability",
    "answer_kind": SCALAR,
    "answers_from": "crossabsense_v0.9",
}

# The six assays, written out one by one rather than built from a table,
# because each blocker has to appear as a literal `blocker=` argument for the
# vocabulary scan to find it. A helper taking the name as a parameter would
# hide every one of them.
#
# Two kinds of decline, kept apart because the fixes differ: a head that exists
# but whose encoder is over the download cap, and an assay with no head at all.
# Ovalbumin polyreactivity and SMAC have neither a head nor a GDPa1 column, so
# their reference dataset is None -- claiming AC-SINS for SMAC would manufacture
# an overlap finding out of a name resemblance.
_DEVELOPABILITY_ITEMS: tuple[BenchmarkItem, ...] = (
    BenchmarkItem(
        benchmark="HIC Retention Time",
        reference_dataset="HIC",
        blocker="oracle_weights_over_size_cap:HIC",
        **_ASSAY,
    ),
    BenchmarkItem(
        benchmark="Tm",
        reference_dataset=_TM_COLUMN,
        blocker="oracle_weights_over_size_cap:Tm2",
        **_ASSAY,
    ),
    BenchmarkItem(
        benchmark="CHO Polyreactivity",
        reference_dataset="PR_CHO",
        blocker="oracle_weights_not_installed:PR_CHO",
        **_ASSAY,
    ),
    BenchmarkItem(
        benchmark="HAC Retention Time",
        reference_dataset="HAC",
        blocker="no_model_for_assay:HAC",
        **_ASSAY,
    ),
    BenchmarkItem(
        benchmark="Ovalbumin Polyreactivity",
        reference_dataset=None,
        blocker="no_model_for_assay:PR_OVA",
        **_ASSAY,
    ),
    BenchmarkItem(
        benchmark="SMAC",
        reference_dataset=None,
        blocker="no_model_for_assay:SMAC",
        **_ASSAY,
    ),
)


BENCHMARK_ITEMS: tuple[BenchmarkItem, ...] = (
    *_DEVELOPABILITY_ITEMS,
    BenchmarkItem(
        benchmark="Mutations",
        domain=STRUCTURAL_BIOLOGY,
        group="ResidueBench",
        answer_kind=RANKING,
        answers_from="thermompnn_point_stability_v1",
        reference_dataset="ddG",
        blocker=None,
    ),
    BenchmarkItem(
        benchmark="Evolutionary pressure",
        domain=STRUCTURAL_BIOLOGY,
        group="ResidueBench",
        answer_kind=RANKING,
        answers_from=None,
        reference_dataset=None,
        blocker="no_conservation_model",
    ),
    *(
        BenchmarkItem(
            benchmark=name,
            domain=TARGET_BIOLOGY,
            group="TargetBench by Disease Group",
            answer_kind=RANKING,
            answers_from="opentargets_associations",
            reference_dataset="disease_target_association",
            blocker=None,
        )
        for name in _TARGET_DISEASE_GROUPS
    ),
    *(
        BenchmarkItem(
            benchmark=name,
            domain=TARGET_BIOLOGY,
            group="Single-Cell Perturbation",
            answer_kind=RANKING,
            answers_from=None,
            reference_dataset=None,
            blocker="no_perturbation_model",
        )
        for name in _PERTURBATION_SETS
    ),
)

#: By name, for a caller holding a benchmark name and nothing else.
ITEMS_BY_NAME: dict[str, BenchmarkItem] = {item.benchmark: item for item in BENCHMARK_ITEMS}


def blocker_sentence(blocker: str) -> str:
    """The sentence for a blocker name.

    Keyed by the whole name, suffix included, rather than by the part before
    the colon. The suffixed names are what the code emits and what
    ``tests/test_disease_board_vocabulary.py`` looks up, and keying on the stem
    would have let five assays share one sentence that could not say which
    encoder was missing or which of them was withheld on purpose.
    """
    return _BLOCKER_TEXT.get(blocker, "")


def _declined(item: BenchmarkItem) -> dict[str, object]:
    blocker = item.blocker or ""
    return {
        "item": item.benchmark,
        "domain": item.domain,
        "answered": False,
        "value": None,
        "blocker": blocker,
        "reason": blocker_sentence(blocker),
        "method": item.answers_from,
        "corpus": corpus_provenance.declare(item.answers_from, against=item.reference_dataset),
        "code": code_version(),
    }


def _answered(item: BenchmarkItem, value: object) -> dict[str, object]:
    return {
        "item": item.benchmark,
        "domain": item.domain,
        "answered": True,
        "value": value,
        "blocker": None,
        "reason": "",
        "method": item.answers_from,
        "corpus": corpus_provenance.declare(item.answers_from, against=item.reference_dataset),
        "code": code_version(),
    }


def answer(
    item: BenchmarkItem,
    *,
    structure: str | Path | None = None,
    disease: str | None = None,
    rank_fn: Callable[[BenchmarkItem, Mapping[str, object]], Sequence[object]] | None = None,
) -> dict[str, object]:
    """One benchmark item, answered or declined.

    Args:
        item: The benchmark to answer.
        structure: A PDB or mmCIF path, for the structure-level items.
        disease: A disease name, for the target-prediction items.
        rank_fn: Injection seam for the ranking producers, called with
            ``(item, inputs)``. Supplied by the caller so this module can be
            tested without a network call or a loaded model, which is how
            ``stability.run_site_saturation`` and ``screening_tools.run_tool``
            are already testable.

    A declined item is declined whatever is passed: an input cannot make a
    missing model appear, and answering one because the caller supplied
    something would be the guess this module exists to avoid.
    """
    if not item.answerable:
        return _declined(item)
    inputs: dict[str, object] = {"structure": structure, "disease": disease}
    if rank_fn is None:
        return _declined_for_missing_runner(item)
    return _answered(item, list(rank_fn(item, inputs)))


def _declined_for_missing_runner(item: BenchmarkItem) -> dict[str, object]:
    """An answerable item with nothing wired to run it is still a decline, not a zero."""
    record = _declined(item)
    record["blocker"] = "runner_not_supplied"
    record["reason"] = blocker_sentence("runner_not_supplied")
    return record


def answer_all(
    items: Iterable[BenchmarkItem] | None = None,
    *,
    structure: str | Path | None = None,
    disease: str | None = None,
    rank_fn: Callable[[BenchmarkItem, Mapping[str, object]], Sequence[object]] | None = None,
) -> list[dict[str, object]]:
    """Every item in scope, answered or declined, in registry order."""
    chosen = tuple(items) if items is not None else BENCHMARK_ITEMS
    return [answer(item, structure=structure, disease=disease, rank_fn=rank_fn) for item in chosen]


def valid_fraction(answers: Sequence[Mapping[str, object]]) -> float:
    """The share that carried a value, which is what the benchmark reports beside accuracy.

    Zero for an empty sequence rather than a division error: no answers is a
    valid fraction of none, not an undefined quantity.
    """
    if not answers:
        return 0.0
    return sum(1 for record in answers if record.get("answered")) / len(answers)


def rank_from_installed_tools(
    item: BenchmarkItem,
    inputs: Mapping[str, object],
    *,
    post: object | None = None,
    predict_fn: object | None = None,
) -> list[object]:
    """The real producers, chosen by what the item says answers it.

    Separate from :func:`answer` so the adapter stays testable without a model
    or a network call, which is the same seam ``stability.run_site_saturation``
    offers through ``predict_fn``. Raises rather than returning a partial list,
    because the caller turns an exception into a decline and a half-ranking
    would be scored as an answer.
    """
    if item.answers_from == "thermompnn_point_stability_v1":
        from rfproteina.disease import stability

        structure = inputs.get("structure")
        if not structure:
            raise ValueError("no structure supplied")
        rows = stability.run_site_saturation(str(structure), predict_fn=predict_fn)
        # Ranked by the most stabilising mutation available at each position:
        # the benchmark scores a ranking, and the scan's own convention is that
        # a more negative ddG is more stabilising.
        best: dict[int, float] = {}
        for row in rows:
            if row.position not in best or row.ddg_pred < best[row.position]:
                best[row.position] = row.ddg_pred
        return [position for position, _ in sorted(best.items(), key=lambda pair: pair[1])]

    if item.answers_from == "opentargets_associations":
        from rfproteina.disease import sources

        disease = inputs.get("disease")
        if not disease:
            raise ValueError("no disease supplied")
        kwargs = {"post": post} if post is not None else {}
        found = sources.search_diseases(str(disease), **kwargs)  # type: ignore[arg-type]
        if not found:
            raise ValueError("disease not found")
        _, symbols, _ = sources.focus_targets(found[0].disease_id, **kwargs)  # type: ignore[arg-type]
        return list(symbols)

    raise ValueError(f"no producer for {item.answers_from}")


def answer_or_decline(
    item: BenchmarkItem,
    *,
    structure: str | Path | None = None,
    disease: str | None = None,
    rank_fn: Callable[[BenchmarkItem, Mapping[str, object]], Sequence[object]] | None = None,
) -> dict[str, object]:
    """:func:`answer`, with any producer failure turned into a decline that says why.

    The distinction a scorer needs is between "no answer" and "a wrong answer",
    not between kinds of exception. A missing input, an unreachable database
    and a crashed model are all the first.
    """
    try:
        return answer(item, structure=structure, disease=disease, rank_fn=rank_fn)
    except Exception as error:
        record = _declined(item)
        record["blocker"] = "producer_failed"
        record["reason"] = f"{blocker_sentence('producer_failed')} The reason given was: {error}"
        return record


def answerable_items() -> tuple[BenchmarkItem, ...]:
    """The items with a path to a number today."""
    return tuple(item for item in BENCHMARK_ITEMS if item.answerable)


def coverage() -> dict[str, object]:
    """What this program claims against the benchmark, for a reader or a report."""
    answerable = answerable_items()
    return {
        "items": len(BENCHMARK_ITEMS),
        "answerable": len(answerable),
        "declined": len(BENCHMARK_ITEMS) - len(answerable),
        "domains": sorted({item.domain for item in BENCHMARK_ITEMS}),
        "blockers": sorted({item.blocker for item in BENCHMARK_ITEMS if item.blocker}),
    }
