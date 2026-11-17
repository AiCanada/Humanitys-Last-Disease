"""Feed the off-target specificity oracle with the paralogs the paralog axis finds.

The ``oracle_off_target`` config re-folds a design against off-target structures
to ask whether a binder is specific. The base
:class:`~rfproteina.metrics.oracles.pre_oracle_transform.OffTargetSubstitutionTransform`
needs those structures as files on disk. The previous screening pass skipped
wiring it because "the paralog structure paths don't exist on disk". This module
solves that honestly: it derives the off-target set from the disease screen's
paralog axis and fetches each one's structure from the AlphaFold Protein
Structure Database (AFDB) on first use, caching it. No path is ever invented, and
when nothing can be resolved the transform raises a clear, actionable error
instead of substituting a fabricated structure.

Where the off-targets come from
-------------------------------
The paralog axis (``rfproteina.disease.target_liability``) records, for a target,
the other human proteins in its UniRef50 cluster (a ``hit``) or its closest
Compara paralog. Its persisted ``paralogs`` field is a list of **gene symbols**
(falling back to accessions). AFDB is keyed by **UniProt accession**, so a gene
symbol is first resolved to a reviewed human accession through UniProt's public
REST API (no key required), then AFDB is fetched as
``https://alphafold.ebi.ac.uk/files/AF-<accession>-F1-model_v<version>.cif``.

AFDB model files are small (typically ~0.1-5 MB of mmCIF each); the number fetched
is bounded by ``max_targets``. The fetch happens lazily in :meth:`setup`, is
cached under ``cache_dir`` so a second run does no network, and is fully
injectable (``fetch_fn``, ``symbol_resolver``) for tests.

Licence / data provenance
-------------------------
AFDB structures are distributed under CC-BY-4.0 by EMBL-EBI/DeepMind. UniProt data
is CC-BY-4.0. Both are free and openly licensed; neither needs an account or key.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from rfproteina.metrics.oracles.pre_oracle_transform import (
    OFF_TARGET_PARSE_CONFIG,
    OffTargetSubstitutionTransform,
)
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

#: UniProt accession pattern (same shape the paralog axis uses in target_liability).
_ACCESSION = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")

#: AFDB file URL template. Version 4 is the current public release family.
AFDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v{version}.cif"

#: UniProt REST search for a reviewed human accession given a gene symbol.
UNIPROT_SEARCH = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query=gene_exact:{symbol}+AND+organism_id:9606+AND+reviewed:true"
    "&format=tsv&fields=accession&size=1"
)


class OffTargetUnavailableError(RuntimeError):
    """Raised when no off-target structure could be resolved or fetched."""


def _save_id_for(accession: str) -> str:
    """A filesystem-safe save_id for an accession (free of path separators)."""
    return accession.replace("/", "_").replace("\\", "_")


def _as_plain(obj: object) -> object:
    """Coerce an OmegaConf ListConfig/DictConfig (what Hydra passes) to a plain
    list/dict, leaving plain containers and ``None`` untouched.

    Hydra instantiation hands config values as OmegaConf nodes, which are not
    ``list``/``dict`` subclasses, so an ``isinstance`` check against them would
    silently miss the configured accessions. This normalises at construction.
    """
    if obj is None or isinstance(obj, list | dict):
        return obj
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    try:
        return dict(obj)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return list(obj)  # type: ignore[call-overload]


def read_paralog_identifiers(record: dict) -> list[str]:
    """Pull candidate off-target identifiers from a paralog axis record.

    Reads the ``paralogs`` list (gene symbols, or accessions where the axis had
    no symbol). Returns them de-duplicated in order. An empty or malformed record
    yields an empty list rather than raising, so a caller can decide what "no
    paralogs" means.
    """
    paralogs = record.get("paralogs") if isinstance(record, dict) else None
    if not isinstance(paralogs, Iterable) or isinstance(paralogs, str | bytes):
        return []
    seen: dict[str, None] = {}
    for item in paralogs:
        text = str(item).strip()
        if text:
            seen.setdefault(text, None)
    return list(seen)


def _default_symbol_resolver(symbol: str) -> str | None:
    """Resolve a human gene symbol to a reviewed UniProt accession via REST.

    Returns ``None`` (not raises) when nothing is found or the service is
    unreachable; the caller treats that as "skip this one".
    """
    import urllib.parse
    import urllib.request

    url = UNIPROT_SEARCH.format(symbol=urllib.parse.quote(symbol))
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # https only; a scheme check is overkill here
            text = response.read().decode("utf-8")
    except Exception as error:  # network failure is a skip, not fatal
        logger.warning(f"UniProt lookup for gene {symbol!r} failed: {error}")
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # First line is the header ("Entry"); the second, if present, is the accession.
    if len(lines) >= 2 and _ACCESSION.fullmatch(lines[1]):
        return lines[1]
    return None


def _default_fetch_fn(accession: str, version: int, destination: Path) -> bool:
    """Download one AFDB mmCIF to ``destination``. Returns True on success."""
    import urllib.request

    url = AFDB_URL.format(accession=accession, version=version)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # https only; a scheme check is overkill here
            data = response.read()
    except Exception as error:  # missing/unreachable is a skip, not fatal
        logger.warning(f"AFDB fetch for {accession!r} failed: {error}")
        return False
    if not data:
        return False
    destination.write_bytes(data)
    return True



class ParalogOffTargetSubstitutionTransform(OffTargetSubstitutionTransform):
    """Off-target substitution fed by paralog accessions, fetching AFDB on first use.

    Args:
        paralog_accessions: Explicit off-targets, as ``{save_id: accession}`` or a
            bare list of accessions/gene symbols. Takes precedence over
            ``paralog_record_path``.
        paralog_record_path: Path to a JSON paralog axis record (the disease
            screen's target-liability output) whose ``paralogs`` field supplies
            the identifiers.
        cache_dir: Directory the fetched AFDB files are cached in.
        afdb_version: AFDB model version (default 4).
        max_targets: Cap on how many off-targets to fetch (``None`` = no cap).
        require_all: If True, raise when any identifier fails to resolve/fetch;
            if False (default), skip failures and proceed with what resolved.
        resolve_symbols: If True (default), resolve gene symbols to accessions via
            UniProt; if False, non-accession identifiers are skipped.
        symbol_resolver: Injection seam; ``(symbol) -> accession | None``.
        fetch_fn: Injection seam; ``(accession, version, destination) -> bool``.

    Raises:
        ValueError: If neither ``paralog_accessions`` nor ``paralog_record_path``
            is given.
    """

    def __init__(
        self,
        *,
        paralog_accessions: dict[str, str] | list[str] | None = None,
        paralog_record_path: str | Path | None = None,
        cache_dir: str | Path = ".afdb_cache",
        afdb_version: int = 4,
        max_targets: int | None = 8,
        require_all: bool = False,
        resolve_symbols: bool = True,
        symbol_resolver: Callable[[str], str | None] | None = None,
        fetch_fn: Callable[[str, int, Path], bool] | None = None,
    ) -> None:
        if paralog_accessions is None and paralog_record_path is None:
            raise ValueError("provide paralog_accessions or paralog_record_path")
        # Deliberately do NOT call super().__init__: the parent requires a
        # non-empty off_target_paths up front, and ours is resolved lazily in
        # setup() once the fetch has happened.
        self.parse_config = OFF_TARGET_PARSE_CONFIG
        self.off_target_paths: dict[str, Path] = {}
        self._off_targets: dict = {}
        self._paralog_accessions = _as_plain(paralog_accessions)
        self._paralog_record_path = Path(paralog_record_path) if paralog_record_path else None
        self.cache_dir = Path(cache_dir)
        self.afdb_version = afdb_version
        self.max_targets = max_targets
        self.require_all = require_all
        self.resolve_symbols = resolve_symbols
        self._symbol_resolver = symbol_resolver or _default_symbol_resolver
        self._fetch_fn = fetch_fn or _default_fetch_fn
        self._resolved_save_ids: list[str] | None = None

    def _candidate_identifiers(self) -> dict[str, str]:
        """``{save_id: identifier}`` from the configured source, before resolution."""
        if isinstance(self._paralog_accessions, dict):
            items = {str(k): str(v) for k, v in self._paralog_accessions.items()}
        elif isinstance(self._paralog_accessions, list):
            items = {_save_id_for(str(x)): str(x) for x in self._paralog_accessions}
        elif self._paralog_record_path is not None:
            import json

            record = json.loads(self._paralog_record_path.read_text())
            identifiers = read_paralog_identifiers(record)
            items = {_save_id_for(x): x for x in identifiers}
        else:
            items = {}
        if self.max_targets is not None:
            items = dict(list(items.items())[: self.max_targets])
        return items

    def _resolve_accession(self, identifier: str) -> str | None:
        """An identifier that is already an accession, else a symbol resolved to one."""
        if _ACCESSION.fullmatch(identifier):
            return identifier
        if not self.resolve_symbols:
            logger.warning(f"{identifier!r} is not an accession and resolve_symbols is off; skipping")
            return None
        accession = self._symbol_resolver(identifier)
        if accession is None:
            logger.warning(f"could not resolve gene symbol {identifier!r} to a UniProt accession")
        return accession

    def setup(self) -> None:
        """Resolve identifiers to accessions, fetch/caches AFDB, parse structures.

        Raises:
            OffTargetUnavailableError: If nothing could be resolved and fetched
                (or, when ``require_all``, if any single identifier failed).
        """
        candidates = self._candidate_identifiers()
        if not candidates:
            raise OffTargetUnavailableError(
                "No paralog identifiers were supplied or found in the record. The off-target oracle "
                "cannot run without at least one off-target. Point paralog_record_path at a paralog "
                "axis record with a non-empty 'paralogs' field, or set paralog_accessions. No structure was invented."
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        resolved: dict[str, Path] = {}
        for identifier in candidates.values():
            accession = self._resolve_accession(identifier)
            if accession is None:
                if self.require_all:
                    raise OffTargetUnavailableError(f"could not resolve {identifier!r} and require_all is set")
                continue
            cached = self.cache_dir / f"AF-{accession}-F1-model_v{self.afdb_version}.cif"
            if not cached.exists() and not self._fetch_fn(accession, self.afdb_version, cached):
                if self.require_all:
                    raise OffTargetUnavailableError(
                        f"could not fetch AFDB structure for {accession!r} and require_all is set"
                    )
                continue
            resolved[_save_id_for(accession)] = cached
        if not resolved:
            raise OffTargetUnavailableError(
                "None of the paralog identifiers could be resolved to a fetchable AFDB structure. "
                "Check network access to rest.uniprot.org and alphafold.ebi.ac.uk, or supply "
                "paralog_accessions directly. No structure was invented."
            )
        self.off_target_paths = resolved
        self._resolved_save_ids = list(resolved)
        # Parse the cached files exactly as the parent does.
        super().setup()

    def expected_save_ids(self) -> list[str]:
        """Resolved save_ids after :meth:`setup`, else the pre-resolution candidates.

        Once setup has run, this reflects exactly what :meth:`forward` will emit
        (the successfully fetched accessions), so the oracle's cache lookup and
        the forward output stay in agreement.
        """
        if self._resolved_save_ids is not None:
            return list(self._resolved_save_ids)
        return list(self._candidate_identifiers())
