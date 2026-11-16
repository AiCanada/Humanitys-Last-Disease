"""A downloaded AlphaFold Server archive, filed where the oracle would have.

The server hands back a zip. Inside it, for a job it named ``4GNK``:

.. code-block:: text

    fold_4gnk_job_request.json
    fold_4gnk_model_0.cif          ... _model_4.cif
    fold_4gnk_summary_confidences_0.json   ... _4.json
    fold_4gnk_full_data_0.json             ... _4.json
    terms_of_use.md

Five models per job, one seed, ordered by **descending** ``ranking_score`` --
model 0 is the best one. The ``fold_`` prefix carries a lowercased, sanitised
form of the job name, which is why nothing here reconstructs a filename from a
design id; the archive's own names are matched back to designs instead.

Three differences from a local AlphaFold 3 run matter, and all three are quiet:

1. **The summary has no ``chain_ids``.** It has exactly ten keys and that is not
   one of them. Read as an empty order it removes every per-chain and interface
   metric without raising, so the order is derived from ``token_chain_ids`` in
   the full-data JSON -- see
   :py:func:`~rfproteina.metrics.oracles.oracle_metric.chain_order_from_confidences`.
2. **The full-data JSON is schema-identical to the local one.** Same six keys in
   the same order, so ``token_chain_ids`` means the same thing on both paths and
   one derivation serves both.
3. **Numbers may be ``null``.** ``iptm`` is null for a single-chain fold, and a
   ``chain_pair_*`` entry is null for a chain with an ill-defined frame, because
   the writer substitutes null for NaN. Every read here tolerates it;
   ``extract_confidence_metrics`` already skips a ``None``.

What lands on disk is what the local oracle writes: the flat metric keys at
:py:meth:`~rfproteina.metrics.oracles.oracle_metric.OracleMetric.confidence_path`
and the structure at ``refolded_cif_path``, through
:py:func:`~rfproteina.metrics.oracles.imported_refolds.place_refold`. Every
refolding metric, pass criterion and screen report then runs unchanged.

Two things a server refold cannot give you, said here rather than discovered as
empty columns:

* **Three flat metrics are unreachable.** ``plddt_coi`` needs ``chain_plddt``,
  ``pae_coi`` needs ``chain_pae``, and ``ipae_coi`` needs ``chain_pair_pae`` --
  none of which is among the summary's ten keys. ``min_ipae_coi`` is, so the
  interface error is still measured; it is the mean-over-pairs variant that is
  not. Per-atom pLDDT is unaffected: it is in the model CIF's ``B_iso_or_equiv``
  column, which is where the pipeline reads it on the local path too.
* **The structure is in the server's own frame and atom order.** The local
  oracle remaps its prediction onto the input's atom indexing and aligns it to
  the input frame before writing; a file that arrived in a browser has had
  neither done to it. Confidence metrics are indifferent to that. Anything that
  compares coordinates to the design is not, and depends on doing its own
  superposition.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from rfproteina.af3server.request import chain_labels_for_run
from rfproteina.disease.refold_export import design_bases
from rfproteina.disease.screen import sample_cifs
from rfproteina.metrics.oracles.imported_refolds import ImportedRefold, ImportReport, place_refold
from rfproteina.metrics.oracles.oracle_metric import (
    OracleMetric,
    chain_order_from_confidences,
    extract_confidence_metrics,
)

#: Members one archive may hold. A job contributes about sixteen files and the
#: server caps an upload at 100 jobs, so this is generous for any real bundle
#: and still a bound.
MEMBER_LIMIT = 8192
#: Bytes one archive may expand to. The full-data arrays are O(tokens²), so a
#: single 5,000-token job's PAE matrix is already large; a bound is still needed,
#: because a zip's declared size is not its real one.
EXPANDED_LIMIT = 8 * 1024 * 1024 * 1024
#: Bytes one confidence JSON may expand to.
CONFIDENCE_LIMIT = 2 * 1024 * 1024 * 1024
#: Bytes one model CIF may expand to.
STRUCTURE_LIMIT = 256 * 1024 * 1024

#: ``fold_<job>_model_<n>.cif``, the structure of one model.
_MODEL = re.compile(r"^fold_(?P<job>.+)_model_(?P<index>\d+)\.cif$", re.IGNORECASE)
#: ``fold_<job>_summary_confidences_<n>.json``.
_SUMMARY = re.compile(r"^fold_(?P<job>.+)_summary_confidences_(?P<index>\d+)\.json$", re.IGNORECASE)
#: ``fold_<job>_full_data_<n>.json``.
_FULL = re.compile(r"^fold_(?P<job>.+)_full_data_(?P<index>\d+)\.json$", re.IGNORECASE)


@dataclass(frozen=True)
class FoldResult:
    """One model of one job, as it came out of the archive."""

    job: str
    model: int
    structure: bytes
    summary: dict[str, object] = field(default_factory=dict)
    full: dict[str, object] | None = None

    @property
    def ranking_score(self) -> float | None:
        """The server's own ranking of this model, or None when absent."""
        value = self.summary.get("ranking_score")
        return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None

    def chain_order(self) -> list[str]:
        """The chain order this model's ``chain_*`` values are indexed by."""
        return chain_order_from_confidences(self.summary, self.full)

    def metrics(self, chain_of_interest: str | None) -> dict[str, float]:
        """The flat metric keys the refolding pipeline reads.

        The same translation the local oracle applies to its own summary, so a
        server refold and a local refold produce the same columns and are
        comparable without anyone remembering which one this was.
        """
        return extract_confidence_metrics(
            summary_confidences=dict(self.summary),
            chain_of_interest=chain_of_interest,
            chains_in_order=self.chain_order(),
        )


@dataclass
class ArchiveReport:
    """What one archive held, and anything in it that was not usable."""

    results: list[FoldResult] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def jobs(self) -> dict[str, list[FoldResult]]:
        """Models by job name, best-ranked first within each job.

        Sorted on the server's own ``ranking_score`` rather than on the file
        index. The two agree in every bundle seen, but the score is the thing
        that means "best" and the index only reflects it.
        """
        grouped: dict[str, list[FoldResult]] = {}
        for result in self.results:
            grouped.setdefault(result.job, []).append(result)
        for models in grouped.values():
            models.sort(key=lambda item: (-(item.ranking_score or float("-inf")), item.model))
        return grouped

    def as_dict(self) -> dict[str, object]:
        return {
            "jobs": {name: [item.model for item in models] for name, models in self.jobs().items()},
            "models": len(self.results),
            "skipped": [{"member": name, "reason": reason} for name, reason in self.skipped],
        }


def read_archive(archive: Path) -> ArchiveReport | str:
    """Every model in a downloaded result archive, or a blocker name.

    A member that cannot be used is named in the report rather than dropped: an
    archive that silently yielded four of five designs is the failure this is
    most likely to have, and the server's own ``terms_of_use.md`` and
    ``job_request.json`` are expected members, not surprises.
    """
    if not archive.is_file():
        return "archive_not_found"
    try:
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            if len(infos) > MEMBER_LIMIT:
                return "archive_too_many_members"
            if sum(info.file_size for info in infos) > EXPANDED_LIMIT:
                return "archive_too_large"
            return _read_members(bundle, infos)
    except (OSError, zipfile.BadZipFile):
        return "archive_unreadable"


def _read_members(bundle: zipfile.ZipFile, infos: list[zipfile.ZipInfo]) -> ArchiveReport:
    report = ArchiveReport()
    structures: dict[tuple[str, int], bytes] = {}
    summaries: dict[tuple[str, int], dict[str, object]] = {}
    fulls: dict[tuple[str, int], dict[str, object]] = {}
    #: The member name each structure actually arrived under. Kept so a refusal
    #: names a file that exists in the archive: the key is lower-cased for
    #: matching, and reconstructing a name from it would print one that does not.
    filenames: dict[tuple[str, int], str] = {}

    for info in infos:
        if info.is_dir():
            continue
        name = info.filename
        # A member naming an absolute path or walking out of the archive is not
        # extracted anywhere, but it is also not a file this reads, and saying so
        # is better than passing over it.
        leaf = _member_leaf(name)
        if leaf is None:
            report.skipped.append((name, "member_path_unsafe"))
            continue

        for pattern, store, limit, decode in (
            (_MODEL, structures, STRUCTURE_LIMIT, False),
            (_SUMMARY, summaries, CONFIDENCE_LIMIT, True),
            (_FULL, fulls, CONFIDENCE_LIMIT, True),
        ):
            found = pattern.match(leaf)
            if found is None:
                continue
            if info.file_size > limit:
                report.skipped.append((name, "member_too_large"))
                break
            key = (found.group("job").lower(), int(found.group("index")))
            try:
                raw = bundle.read(info)
            except (OSError, zipfile.BadZipFile, RuntimeError):
                report.skipped.append((name, "member_unreadable"))
                break
            if not decode:
                store[key] = raw
                filenames[key] = name
                break
            try:
                loaded = json.loads(raw.decode("utf-8"))
            except (UnicodeError, ValueError, RecursionError):
                report.skipped.append((name, "member_not_json"))
                break
            if not isinstance(loaded, dict):
                report.skipped.append((name, "member_not_an_object"))
                break
            store[key] = loaded
            break

    for key in sorted(structures):
        job, model = key
        if key not in summaries:
            # Without the summary there are no confidence numbers, and a
            # structure filed with none would mark the design refolded while
            # measuring nothing.
            report.skipped.append((filenames[key], "summary_confidences_absent"))
            continue
        report.results.append(FoldResult(job, model, structures[key], summaries[key], fulls.get(key)))
    return report


def import_archive(
    oracle: OracleMetric,
    out_dir: Path,
    archive: Path,
    *,
    example_ids: list[str] | None = None,
    chain_of_interest: str | None = None,
    save_id: str | None = None,
    every_model: bool = False,
) -> ImportReport:
    """File every fold in ``archive`` against the designs under ``out_dir``.

    ``out_dir`` is the **run** directory. Each design's refold goes beside the
    design itself, which is not a direct child of the run: a run nests its
    designs under ``inference_samples/``, and a run holding several batches
    nests them one level further. So the base is resolved per design by
    :py:func:`~rfproteina.disease.refold_export.design_bases`, which reads the
    ids from the same place the screen does. Passing the run itself to
    ``place_refold`` wrote the pair into ``<run>/<id>/`` -- a fresh directory
    that no design lives in and no oracle looks in -- and then reported the
    design as filed. ``already_refolded`` stayed false at the real location, so
    every refolding metric was silently absent while the import said it worked.

    A design with no resolvable base is **named**, never guessed at. One exists:
    ``_sample_ids`` flattens a repeated binder token to a path id, so a design
    at ``inference_samples/run_a/binder_0/`` is called ``run_a_binder_0`` and no
    base can be joined back onto it.

    ``chain_of_interest`` is the **design's own** strand, as its structure spells
    it. The server does not keep those names: it labels chains by position in
    ``sequences``, so a design whose binder is strand ``E`` comes back as chain
    ``B``. The name is translated per design before the metrics are taken --
    naming a chain the result does not have silently drops every per-chain and
    interface value, which is the same failure
    :py:func:`~rfproteina.metrics.oracles.oracle_metric.chain_order_from_confidences`
    exists to prevent.

    By default only each job's best-ranked model is filed, because that is the
    one a refolding criterion is meant to be read on and five models under one
    design id would overwrite each other. ``every_model`` files all five under
    per-model save ids instead, for a caller who wants the spread.

    A job matching no design is **named** in the report, not skipped: the server
    lowercases and sanitises a job name on the way out, so a mismatch is the
    most likely way this quietly imports nothing.
    """
    report = ImportReport()
    read = read_archive(archive)
    if isinstance(read, str):
        report.unmatched.append(f"{archive.name}: {read}")
        return report

    # Candidates come from every design the run holds; bases from the subset
    # whose id can be joined back onto its own folder. Taking candidates from
    # the bases instead would leave a flattened id matching nothing at all, and
    # it would be reported as "no such design" -- which is both untrue and a
    # different thing to go and fix than "this design's refold has nowhere to go".
    bases = design_bases(out_dir)
    wanted = list(example_ids) if example_ids is not None else sorted(sample_cifs(out_dir))
    by_normal: dict[str, list[str]] = {}
    for name in wanted:
        by_normal.setdefault(_normalised(name), []).append(name)
    labels = chain_labels_for_run(out_dir, wanted) if chain_of_interest else {}

    with TemporaryDirectory(prefix="af3server-confidence-") as scratch:
        workspace = Path(scratch)
        for job, models in sorted(read.jobs().items()):
            candidates = by_normal.get(_normalised(job), [])
            if not candidates:
                report.unmatched.append(f"fold_{job}")
                continue
            if len(candidates) > 1:
                # Two designs whose ids differ only by the characters the server
                # flattens. Filing against either one would be a coin toss.
                report.refused.append(ImportedRefold(job, "design_id_ambiguous", save_id=save_id))
                continue
            example_id = candidates[0]
            base = bases.get(example_id)
            if base is None:
                report.refused.append(ImportedRefold(example_id, "design_base_not_resolved", save_id=save_id))
                continue
            interest = _server_chain(chain_of_interest, labels.get(example_id))
            chosen = models if every_model else models[:1]
            for result in chosen:
                target = save_id
                if every_model:
                    target = f"{save_id}.model_{result.model}" if save_id else f"model_{result.model}"
                report_entry = _place_one(oracle, base, example_id, result, workspace, interest, target)
                (report.placed if report_entry.placed else report.refused).append(report_entry)
    return report


def _server_chain(chain_of_interest: str | None, labels: dict[str, str] | None) -> str | None:
    """The label the server gave the design's chain of interest.

    Passed through unchanged when there is nothing to translate with. A caller
    who already knows the server's label, or whose design's strand happens to
    be the label the server assigned it, is then no worse off than before -- and
    a design whose request could not be rebuilt is not silently matched against
    a chain that is not there.
    """
    if chain_of_interest is None or not labels:
        return chain_of_interest
    return labels.get(chain_of_interest, chain_of_interest)


def _place_one(
    oracle: OracleMetric,
    out_dir: Path,
    example_id: str,
    result: FoldResult,
    workspace: Path,
    chain_of_interest: str | None,
    save_id: str | None,
) -> ImportedRefold:
    """Write one model's two files through ``place_refold``."""
    metrics: dict[str, object] = dict(result.metrics(chain_of_interest))
    # Recorded even when None, exactly as the local oracle records it: a reader
    # must be able to tell "no chain of interest was named" from "the COI
    # metrics are missing for another reason".
    metrics["chain_of_interest"] = chain_of_interest
    metrics["af3server_model"] = result.model
    score = result.ranking_score
    if score is not None:
        metrics["af3server_ranking_score"] = score
    if not result.chain_order():
        # Said rather than left to be found as empty interface columns. Without
        # the full-data JSON beside the summary there is no chain order at all,
        # so no per-chain or interface metric could be computed.
        metrics["af3server_chain_order"] = "not_recorded"

    stem = f"{example_id}_model_{result.model}"
    structure = workspace / f"{stem}.cif"
    confidence = workspace / f"{stem}_confidences.json"
    try:
        structure.write_bytes(result.structure)
        confidence.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        return ImportedRefold(example_id, "scratch_write_failed", save_id=save_id)
    return place_refold(
        oracle,
        out_dir,
        example_id,
        structure,
        save_id=save_id,
        confidence=confidence,
    )


def _member_leaf(member: str) -> str | None:
    """The leaf name of an archive member, or None when the path is unsafe.

    A zip may name ``../`` or an absolute path. Nothing here extracts to a path
    taken from the archive -- members are read into memory and written under a
    name this module builds -- so such a member cannot escape. It is still
    refused by name, because a bundle containing one is not a bundle this
    understands and passing over it silently is how the next reader inherits the
    assumption that it was safe.
    """
    if member.startswith("/") or member.startswith("\\") or ":" in member:
        return None
    parts = member.replace("\\", "/").split("/")
    if any(part in {"..", "."} for part in parts):
        return None
    leaf = parts[-1]
    return leaf or None


def _normalised(name: str) -> str:
    """A design id and a server job name reduced to one comparable spelling.

    The server lowercases the name it was given and replaces what it does not
    keep, so ``M0664_2dhn`` comes back inside ``fold_m0664_2dhn_model_0.cif``.
    Both sides are flattened the same way rather than guessing the exact
    sanitiser, and an id that collides with another under this flattening is
    refused rather than matched to the first one tried.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# There was a `_designs_under` here that listed the directories directly under
# the run. It is gone: on a real run it returns `inference_samples` and
# `inference_metrics` as design names and no actual design, so every job matched
# nothing. `design_bases` reads the ids from the same place the screen does.
