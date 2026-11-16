"""Open Targets disease search and RCSB structure rows.

Parsers are pure. ``post_json`` does the HTTP call, and tests pass a fake.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from urllib.request import Request, urlopen

from rfproteina.disease.limits import USER_AGENT, slot
from rfproteina.disease.lookup import Disease
from rfproteina.disease.structures import StructureOption

OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_CIF_URL = "https://files.rcsb.org/download"
UNIPROT_ATTRIBUTE = "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession"
GENE_ATTRIBUTE = "rcsb_entity_source_organism.rcsb_gene_name.value"
MAX_ASSOCIATIONS = 8
MAX_STRUCTURE_ROWS = 20
#: How many deposit records are read at once. One search returns at most
#: ``MAX_STRUCTURE_ROWS`` rows, and reading their records one at a time was most of
#: the wait before the structure list appeared. Kept small: these are somebody
#: else's public servers, and the searches themselves stay strictly serial.
MAX_ENTRY_WORKERS = 8
STRING_PARTNERS_URL = "https://string-db.org/api/json/interaction_partners"
REACTOME_PATHWAYS_URL = "https://reactome.org/ContentService/data/mapping/UniProt"
REACTOME_PARTICIPANTS_URL = "https://reactome.org/ContentService/data/participants"
# Ten-character accessions (A0AB40D056) are tried before the six-character forms so a
# search inside a FASTA header does not stop on the first six characters.
_UNIPROT_ACCESSION = (
    r"(?:[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){2}" r"|[OPQ][0-9][A-Z0-9]{3}[0-9]" r"|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])"
)
_UNIPROT_ID = re.compile(rf"^{_UNIPROT_ACCESSION}$")
_GENE_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9-]{1,14}$")

DISEASE_SEARCH = """
query DiseaseSearch($q: String!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 15}) {
    hits { id name entity }
  }
}
"""

ASSOCIATED_TARGETS = """
query DiseaseTargets($id: String!) {
  disease(efoId: $id) {
    associatedTargets(page: {index: 0, size: 15}) {
      rows {
        target {
          id
          approvedSymbol
          proteinIds { id source }
        }
      }
    }
  }
}
"""

PostJson = Callable[[str, dict], dict]
GetJson = Callable[[str], dict]
GetBytes = Callable[[str], bytes]
ReadValue = Callable[[str], object]

CONTEXT_TARGETS = """
query DiseaseContext($id: String!) {
  disease(efoId: $id) {
    associatedTargets(page: {index: 0, size: 8}) {
      rows {
        target {
          id
          approvedSymbol
          bioType
          proteinIds { id source }
        }
      }
    }
  }
}
"""


def post_json(url: str, payload: dict) -> dict:
    """POST JSON and read a JSON object back.

    RCSB search returns HTTP 204 when a query matches nothing. That is an empty
    result, including an obsolete accession, and the lookup continues. A blank
    body with any other status is retried.
    """
    data = json.dumps(payload).encode("utf-8")

    def attempt() -> dict:
        request = Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        with slot(url), urlopen(request, timeout=45) as response:
            if getattr(response, "status", None) == 204:
                response.read()
                return {"result_set": [], "total_count": 0}
            return _read_json(url, response)

    return _retry(url, attempt)


def diseases_from_search(payload: dict) -> list[Disease]:
    """Read disease hits from an Open Targets search payload."""
    hits = (payload.get("data") or {}).get("search", {}).get("hits") or []
    diseases: list[Disease] = []
    for hit in hits:
        if hit.get("entity") not in (None, "disease"):
            continue
        name = str(hit.get("name") or "").strip()
        disease_id = str(hit.get("id") or "").strip()
        if name and disease_id:
            diseases.append(Disease(name=name, disease_id=disease_id))
    return diseases


def _uniprot_ids(target: dict) -> list[str]:
    accessions: list[str] = []
    for row in target.get("proteinIds") or []:
        source = str(row.get("source") or "").casefold()
        accession = str(row.get("id") or "").strip()
        if accession and "uniprot" in source:
            accessions.append(accession)
    return accessions


def accessions_from_targets(payload: dict) -> list[tuple[str, str]]:
    """Return ``(symbol, uniprot)`` pairs for genes associated with a disease.

    A target with no UniProt protein, such as a long non-coding RNA, still
    contributes its gene symbol with an empty accession. Pairing the symbol only
    with an accession would drop those targets entirely, and the gene group
    could then never list a structure that no protein accession reaches.
    """
    rows = (payload.get("data") or {}).get("disease", {}).get("associatedTargets", {}).get("rows") or []
    pairs: list[tuple[str, str]] = []
    for row in rows:
        target = row.get("target") or {}
        symbol = str(target.get("approvedSymbol") or "").strip() or "gene"
        accessions = _uniprot_ids(target)
        if not accessions:
            pairs.append((symbol, ""))
            continue
        for accession in accessions:
            pairs.append((symbol, accession))
    return pairs


def options_from_rcsb(payload: dict, *, accession: str, group: str) -> list[StructureOption]:
    """Read polymer-entity descriptions produced for one accession."""
    rows = payload.get("result_set") or []
    options: list[StructureOption] = []
    for row in rows:
        identifier = str(row.get("identifier") or "")
        split = _split_identifier(identifier)
        if split is None:
            continue
        entry_id, chain = split
        polymer = str(row.get("polymer") or group).casefold()
        if polymer not in {"protein", "dna", "rna", "gene"}:
            polymer = group
        options.append(
            StructureOption(
                group=polymer,
                accession=accession,
                entry_id=entry_id,
                chain=chain or "A",
                title=str(row.get("title") or identifier),
                predicted=bool(row.get("predicted")),
                source="rcsb",
            )
        )
    return options


def search_diseases(query: str, post: PostJson = post_json) -> list[Disease]:
    """Search Open Targets for a disease name."""
    payload = post(OPENTARGETS_URL, {"query": DISEASE_SEARCH, "variables": {"q": query}})
    return diseases_from_search(payload)


def disease_accessions(disease_id: str, post: PostJson = post_json) -> list[tuple[str, str]]:
    """Gene symbols and UniProt accessions associated with ``disease_id``."""
    payload = post(OPENTARGETS_URL, {"query": ASSOCIATED_TARGETS, "variables": {"id": disease_id}})
    return accessions_from_targets(payload)


def focus_targets(
    disease_id: str,
    *,
    post: PostJson = post_json,
    limit: int = MAX_ASSOCIATIONS,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Associated accessions, symbols, and the menu pairs, each capped at ``limit``."""
    pairs = disease_accessions(disease_id, post=post)
    accessions, symbols = _associations(pairs, limit)
    targets: list[tuple[str, str]] = []
    seen: set[str] = set()
    for symbol, accession in pairs:
        key = f"{symbol}|{accession}".casefold()
        if key in seen:
            continue
        seen.add(key)
        targets.append((symbol, accession))
        if len(targets) >= limit:
            break
    return accessions, symbols, targets


def get_json(url: str) -> dict:
    """GET a URL and read a JSON object back.

    A blank body is retried. One empty deposit record would otherwise be
    treated as a structure whose experimental method is unknown.
    """

    def attempt() -> dict:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
        with slot(url), urlopen(request, timeout=45) as response:
            return _read_json(url, response)

    return _retry(url, attempt)


def _read_json(url: str, response: object) -> dict:
    """Read one JSON object, including the HTTP status when the body is blank."""
    raw = response.read()
    status = getattr(response, "status", "unknown")
    if not raw:
        raise ValueError(f"Could not read JSON from {url}: HTTP {status} returned an empty body")
    return _json_object(url, raw)


def _json_object(url: str, raw: bytes) -> dict:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON from {url}: {exc}; body starts {raw[:80]!r}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"Expected a JSON object from {url}.")
    return body


def _retry(url: str, attempt: Callable[[], dict], *, attempts: int = 5) -> dict:
    """Call ``attempt`` until it returns JSON or the blank-body retries are spent.

    Live search calls sometimes return HTTP 200 with an empty body. Those are
    retried with a growing pause. A JSON value that is not an object is a real
    response and is not retried.
    """
    last = "empty response"
    for index in range(attempts):
        try:
            return attempt()
        except OSError as exc:
            last = str(exc)
        except ValueError as exc:
            message = str(exc)
            if not message.startswith("Could not read JSON"):
                raise
            last = message
        if index + 1 < attempts:
            time.sleep(2**index)
    raise ValueError(f"Could not read JSON from {url}: {last}")


def get_bytes(url: str) -> bytes:
    """GET a structure file."""
    request = Request(url, headers={"Accept": "chemical/x-mmcif, */*"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def entry_url(entry_id: str) -> str:
    """RCSB data-api URL for one deposited entry."""
    return f"{RCSB_ENTRY_URL}/{entry_id}"


def cif_url(entry_id: str) -> str:
    """Public mmCIF download URL. The archive name is uppercase."""
    return f"{RCSB_CIF_URL}/{entry_id.upper()}.cif"


ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
ALPHAFOLD_ENTRY = "https://alphafold.ebi.ac.uk"
#: AF-<UniProt>-F<fragment>. F1 is the usual segment. A long protein also has F2, F3, ....
#: An isoform is AF-P00533-4-F1. The id does not contain an AF3 token.
_AF_ID = re.compile(
    rf"^AF-(?P<body>{_UNIPROT_ACCESSION}(?:-\d+)?)-F\d+$",
    re.IGNORECASE,
)
_FASTA_ID = re.compile(
    rf"(AF-{_UNIPROT_ACCESSION}(?:-\d+)?-F\d+|{_UNIPROT_ACCESSION})",
    re.IGNORECASE,
)


def is_alphafold_id(entry_id: str) -> bool:
    """True when ``entry_id`` is an AlphaFold Database id such as ``AF-P00533-F1``."""
    return _AF_ID.fullmatch(entry_id.strip()) is not None


def alphafold_uniprot(entry_id: str) -> str:
    """UniProt accession carried by an AF id, including an isoform such as ``P00533-4``."""
    match = _AF_ID.fullmatch(entry_id.strip())
    if match is None:
        return ""
    return match.group("body").upper()


def alphafold_api_url(entry_id: str) -> str | None:
    """EBI prediction URL for one AF id. The sequence is not part of the URL."""
    match = _AF_ID.fullmatch(entry_id.strip())
    if match is None:
        return None
    parent = re.sub(r"-\d+$", "", match.group("body"))
    return f"{ALPHAFOLD_API}/{quote(parent)}"


def alphafold_entries(payload: object) -> list[dict[str, str]]:
    """Published models in one AlphaFold response. Sequence fields are not kept."""
    rows = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
    found: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        entry = str(row.get("entryId") or row.get("modelEntityId") or "").strip()
        cif_url_value = str(row.get("cifUrl") or "")
        if not entry or not cif_url_value.startswith(f"{ALPHAFOLD_ENTRY}/"):
            continue
        accession = str(row.get("uniprotAccession") or "").strip()
        title = str(row.get("uniprotDescription") or "").strip() or f"AlphaFold {entry}"
        found.append(
            {
                "entry_id": entry,
                "title": title,
                "accession": accession or entry,
                "chain": str(row.get("chainId") or "A").strip() or "A",
                "cif_url": cif_url_value,
            }
        )
    return found


def alphafold_match(payload: object, entry_id: str) -> dict[str, str] | None:
    """The published model for ``entry_id``. Sequence fields are not kept."""
    wanted = entry_id.strip().casefold()
    for record in alphafold_entries(payload):
        if record["entry_id"].casefold() == wanted:
            return record
    return None


def alphafold_cif_url(entry_id: str, read_json: Callable[[str], object]) -> str:
    """mmCIF URL on alphafold.ebi.ac.uk for one published AF id."""
    url = alphafold_api_url(entry_id)
    if url is None:
        raise ValueError(f"{entry_id} is not an AlphaFold id.")
    record = alphafold_match(read_json(url), entry_id)
    if record is None:
        raise ValueError(f"AlphaFold has no published model for {entry_id}.")
    return record["cif_url"]


def accession_from_fasta(text: str) -> str:
    """AF id or UniProt accession named by a FASTA header.

    The sequence is not returned. A header with neither id cannot be fetched:
    https://alphafold.ebi.ac.uk/ serves a published AF id, and a new prediction
    is pasted as sequence or FASTA at https://alphafoldserver.com/ while signed
    in. That site does not return a file to this program.
    """
    for line in text.splitlines():
        if not line.startswith(">"):
            continue
        match = _FASTA_ID.search(line)
        if match:
            return match.group(1).upper()
    raise ValueError(
        "The FASTA header needs an AF id or a UniProt accession. "
        "https://alphafold.ebi.ac.uk/ serves a published AF id. "
        "A new prediction is pasted as sequence or FASTA at https://alphafoldserver.com/ "
        "while signed in, and that site does not return a file to this program."
    )


def rcsb_accession_query(accession: str, *, rows: int = MAX_STRUCTURE_ROWS) -> dict:
    """Search polymer entities whose reference sequence is this UniProt accession."""
    return _rcsb_query(UNIPROT_ATTRIBUTE, accession, rows=rows)


def rcsb_gene_query(symbol: str, *, rows: int = MAX_STRUCTURE_ROWS) -> dict:
    """Search polymer entities annotated with this gene symbol."""
    return _rcsb_query(GENE_ATTRIBUTE, symbol, rows=rows)


def collect_structure_options(
    disease_id: str,
    *,
    post: PostJson = post_json,
    get: GetJson = get_json,
    row_limit: int = MAX_STRUCTURE_ROWS,
    association_limit: int = MAX_ASSOCIATIONS,
    include_related: bool = False,
    read_value: ReadValue | None = None,
    records: dict[str, dict] | None = None,
    extra_accessions: Sequence[str] | None = None,
    extra_symbols: Sequence[str] | None = None,
) -> list[StructureOption]:
    """List protein, DNA, RNA, and gene structures for one confirmed disease.

    Protein rows come from UniProt accessions. DNA and RNA rows are the same
    entries when the deposit also contains that polymer. Gene rows come from
    the associated gene symbols. Predicted models stay marked predicted.

    ``include_related`` also asks for pathway members, interacting proteins, and
    RNA-biotype genes, then looks up their deposited structures. A failed extra
    lookup leaves the disease-protein structures in place. Sequence motifs are
    not requested and are not returned.

    ``records`` is a deposit-record store the caller keeps. Passing one makes a
    second call reuse what the first read, which is what a panel that restarts the
    flow to go back a page needs: the lookup is the whole cost of that restart.
    Only records that were read are given back. An empty record means the lookup
    failed, and keeping that would leave the row marked not known to be
    experimental for as long as the caller holds the store, with no way to retry.

    ``extra_accessions`` and ``extra_symbols`` are a chosen pathway's proteins, a
    named target, or interaction partners. They are added after the association
    cap, and each list is itself capped, so naming a target does not drop the
    disease proteins and does not make the search unbounded. Omitting them leaves
    the search unchanged.
    """
    accessions, symbols = _associations(disease_accessions(disease_id, post=post), association_limit)
    if include_related:
        related_accessions, related_symbols = _related_identifiers(
            disease_id,
            accessions,
            symbols,
            post=post,
            read_value=read_value or get_json_value,
            limit=association_limit,
        )
        accessions = _extend_unique(accessions, related_accessions)
        symbols = _extend_unique(symbols, related_symbols)
    if extra_accessions:
        accessions = _extend_unique(accessions, [item for item in extra_accessions if item][:association_limit])
    if extra_symbols:
        symbols = _extend_unique(symbols, [item for item in extra_symbols if item][:association_limit])
    found: list[StructureOption] = []
    seen: set[tuple[str, str, str]] = set()
    # One deposit answers many rows and many accessions. Share the record cache across
    # every query of this call, so each identifier costs one request at most. What the
    # caller already holds starts it off, so a restarted flow reads none of it again.
    cache: dict[str, dict] = dict(records) if records is not None else {}
    for accession in accessions:
        payload = post(RCSB_SEARCH_URL, rcsb_accession_query(accession, rows=row_limit))
        _add_hits(found, seen, payload, label=accession, group="protein", get=get, row_limit=row_limit, cache=cache)
    for symbol in symbols:
        payload = post(RCSB_SEARCH_URL, rcsb_gene_query(symbol, rows=row_limit))
        _add_hits(found, seen, payload, label=symbol, group="gene", get=get, row_limit=row_limit, cache=cache)
    if records is not None:
        records.update({key: entry for key, entry in cache.items() if entry})
    return found


def fetch_structure_cif(
    option: StructureOption,
    *,
    read: GetBytes = get_bytes,
    read_json: Callable[[str], object] | None = None,
) -> bytes:
    """Download one entry mmCIF. The caller chooses which options are fetched.

    A PDB id comes from the RCSB archive. An AF id comes from the published
    model on https://alphafold.ebi.ac.uk/. A new AlphaFold 3 prediction is a
    signed-in paste of sequence or FASTA at https://alphafoldserver.com/ and is
    not requested here.
    """
    if is_alphafold_id(option.entry_id):
        reader = get_json_value if read_json is None else read_json
        url = alphafold_cif_url(option.entry_id, reader)
        payload = read(url)
        if not payload:
            raise ValueError(f"AlphaFold returned an empty structure for {option.entry_id}.")
        return payload
    payload = read(cif_url(option.entry_id))
    if not payload:
        raise ValueError(f"RCSB returned an empty structure for {option.entry_id}.")
    return payload


def _split_identifier(identifier: str) -> tuple[str, str] | None:
    """Split ``PDBID_ENTITY`` or ``AF_..._ENTITY`` on the last underscore."""
    text = identifier.strip()
    if not text:
        return None
    entry_id, separator, chain = text.rpartition("_")
    if not separator:
        return text, "A"
    if not entry_id:
        return None
    return entry_id, chain or "A"


def _rcsb_query(attribute: str, value: str, *, rows: int) -> dict:
    return {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {"attribute": attribute, "operator": "exact_match", "value": value},
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_content_type": ["experimental", "computational"],
        },
    }


def _associations(pairs: list[tuple[str, str]], limit: int) -> tuple[list[str], list[str]]:
    accessions: list[str] = []
    symbols: list[str] = []
    for symbol, accession in pairs:
        if accession and accession not in accessions:
            accessions.append(accession)
        if symbol and symbol.casefold() != "gene" and symbol not in symbols:
            symbols.append(symbol)
    return accessions[:limit], symbols[:limit]


def _add_hits(
    found: list[StructureOption],
    seen: set[tuple[str, str, str]],
    payload: dict,
    *,
    label: str,
    group: str,
    get: GetJson,
    row_limit: int,
    cache: dict[str, dict],
) -> None:
    rows = (payload.get("result_set") or [])[:row_limit]
    _prefetch_entries(get, rows, cache)
    for row in rows:
        identifier = str(row.get("identifier") or "")
        split = _split_identifier(identifier)
        if split is None:
            continue
        entry_id, chain = split
        entry = _load_entry(get, entry_id, cache)
        title = str((entry.get("struct") or {}).get("title") or identifier)
        method = str((entry.get("rcsb_entry_info") or {}).get("structure_determination_methodology") or "")
        # An absent method means the lookup failed or the deposit does not say. Either way the
        # row is not known to be experimental, and it must not be selected as though it were.
        methodology_known = bool(method)
        predicted = method.casefold() == "computational"
        groups = [group, *[item for item in _companion_groups(entry) if item != group]]
        for item_group in groups:
            key = (item_group, entry_id.upper(), chain)
            if key in seen:
                continue
            seen.add(key)
            found.append(
                StructureOption(
                    group=item_group,
                    accession=label,
                    entry_id=entry_id,
                    chain=chain,
                    title=title,
                    predicted=predicted,
                    source="rcsb",
                    methodology_known=methodology_known,
                    campaigns=_campaigns(item_group, title, entry),
                )
            )


def _prefetch_entries(get: GetJson, rows: Iterable[dict], cache: dict[str, dict]) -> None:
    """Read the deposit records this result set needs, several at a time.

    The caller then assembles its rows with its own serial loop and finds every
    record already read, so the listed order of the rows is untouched. That order
    is load-bearing: a row's listed position is the number ``--pick`` names, the
    index a design-class tick selects by, and the key the saved report writes it
    under, so it must not vary between runs of the same search.

    Deduplication happens before anything is dispatched, so one deposit answering
    many rows still costs one request. A single record is left to the caller,
    because a thread pool for one request is only overhead.

    Reading these concurrently means the order the records are *requested* in is no
    longer fixed. Nothing reads that order; the rows are ordered by the search
    result, not by which record came back first.
    """
    pending: list[str] = []
    asked: set[str] = set()
    for row in rows:
        split = _split_identifier(str(row.get("identifier") or ""))
        if split is None:
            continue
        entry_id = split[0]
        key = entry_id.strip().upper()
        if key in cache or key in asked:
            continue
        asked.add(key)
        pending.append(entry_id)
    if len(pending) < 2:
        return
    with ThreadPoolExecutor(max_workers=min(MAX_ENTRY_WORKERS, len(pending))) as pool:
        # Each record lands in the cache under its own key, which is what the loop
        # below reads. Anything _load_entry does not catch is raised here, as it
        # would have been when the record was read in the row loop.
        list(pool.map(lambda entry_id: _load_entry(get, entry_id, cache), pending))


def _load_entry(get: GetJson, entry_id: str, cache: dict[str, dict] | None = None) -> dict:
    """One deposit record. A repeated identifier is fetched once.

    A failed lookup returns an empty record and is cached as such, so a listing
    of many rows from one entry does not retry it per row.
    """
    key = entry_id.strip().upper()
    if cache is not None and key in cache:
        return cache[key]
    try:
        entry = get(entry_url(entry_id))
    except (OSError, ValueError):
        entry = {}
    if cache is not None:
        cache[key] = entry
    return entry


def _companion_groups(entry: dict) -> tuple[str, ...]:
    info = entry.get("rcsb_entry_info") or {}
    groups: list[str] = []
    hybrid = _count(info, "polymer_entity_count_nucleic_acid_hybrid")
    if _count(info, "polymer_entity_count_DNA") or hybrid:
        groups.append("dna")
    if _count(info, "polymer_entity_count_RNA") or hybrid:
        groups.append("rna")
    return tuple(groups)


def get_json_value(url: str) -> object:
    """GET a JSON object or list. Pathway and partner services return both.

    An empty body is retried. A JSON value that is neither an object nor a list
    is returned as a failure immediately.
    """

    def attempt() -> object:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
        with slot(url), urlopen(request, timeout=45) as response:
            raw = response.read()
            status = getattr(response, "status", "unknown")
            if not raw:
                raise ValueError(f"Could not read JSON from {url}: HTTP {status} returned an empty body")
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read JSON from {url}: {exc}") from exc
            if isinstance(body, dict | list):
                return body
            raise ValueError(f"Expected a JSON object or list from {url}.")

    last = "empty response"
    for index in range(5):
        try:
            return attempt()
        except OSError as exc:
            last = str(exc)
        except ValueError as exc:
            message = str(exc)
            if not message.startswith("Could not read JSON"):
                raise
            last = message
        if index + 1 < 5:
            time.sleep(2**index)
    raise ValueError(f"Could not read JSON from {url}: {last}")


def _related_identifiers(
    disease_id: str,
    accessions: list[str],
    symbols: list[str],
    *,
    post: PostJson,
    read_value: ReadValue,
    limit: int,
) -> tuple[list[str], list[str]]:
    """Pathway members, interacting proteins, and RNA-biotype genes.

    One hop only: partners of partners are not followed. Each lookup that fails
    is skipped. The result is capped so the structure search stays bounded.

    Reactome answers a pathway query in about 19 seconds, and it is asked once per
    UniProt seed, which made it the largest single cost of the whole lookup. Those
    queries are therefore run before the loop, several at a time, and the loop then
    reads what they returned. The loop itself is unchanged: it still walks the
    seeds in order and still stops at the cap, so which extras are kept is decided
    exactly as before.
    """
    extra_accessions: list[str] = []
    extra_symbols: list[str] = []
    known_accessions = {item.casefold() for item in accessions}
    known_symbols = {item.casefold() for item in symbols}
    for row in _context_rows(post, disease_id):
        target = row.get("target") or {}
        symbol = str(target.get("approvedSymbol") or "").strip()
        biotype = str(target.get("bioType") or "").casefold()
        if "rna" in biotype:
            _remember_symbol(extra_symbols, known_symbols, symbol)
    seeds = list(accessions[:limit]) + [item for item in symbols[:limit] if item not in accessions]
    members = _reactome_for_seeds(read_value, seeds, limit)
    for seed in seeds:
        if len(extra_accessions) >= limit and len(extra_symbols) >= limit:
            break
        for partner in _string_symbols(_safe_value(read_value, _string_partners_url(seed))):
            _remember_symbol(extra_symbols, known_symbols, partner)
        if not _UNIPROT_ID.fullmatch(seed.upper()):
            continue
        for member in members.get(seed, ()):
            token = member.upper()
            if token.casefold() in known_accessions or token in extra_accessions:
                continue
            extra_accessions.append(token)
            if len(extra_accessions) >= limit:
                break
    return extra_accessions[:limit], extra_symbols[:limit]


def _context_rows(post: PostJson, disease_id: str) -> list[dict]:
    try:
        payload = post(OPENTARGETS_URL, {"query": CONTEXT_TARGETS, "variables": {"id": disease_id}})
    except (OSError, ValueError, AssertionError):
        return []
    if not isinstance(payload, dict) or payload.get("errors"):
        return []
    disease = (payload.get("data") or {}).get("disease") or {}
    rows = (disease.get("associatedTargets") or {}).get("rows") or []
    return [row for row in rows if isinstance(row, dict)]


def _string_partners_url(identifier: str) -> str:
    return f"{STRING_PARTNERS_URL}?species=9606&limit=4&identifiers={quote(identifier)}"


def _string_symbols(payload: object) -> list[str]:
    rows = payload if isinstance(payload, list) else []
    symbols: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("preferredName_B") or "").strip()
        if _GENE_SYMBOL.fullmatch(name) and name.casefold() != "gene":
            symbols.append(name)
    return symbols


def _reactome_for_seeds(read_value: ReadValue, seeds: list[str], limit: int) -> dict[str, list[str]]:
    """Every seed's pathway members, read several at a time.

    Each seed is asked for once, in the order the seeds are listed, and the answers
    are keyed by seed, so the loop that consumes them is unaffected by which came
    back first. Reactome's own allowance paces the requests.

    This asks for every UniProt seed, where the loop would have stopped asking once
    its caps were full: at most one or two requests more than before, against four
    minutes less waiting.
    """
    wanted = [seed for seed in seeds if _UNIPROT_ID.fullmatch(seed.upper())]
    if not wanted:
        return {}
    if len(wanted) == 1:
        return {wanted[0]: _reactome_members(read_value, wanted[0], limit)}
    with ThreadPoolExecutor(max_workers=min(MAX_ENTRY_WORKERS, len(wanted))) as pool:
        found = list(pool.map(lambda seed: _reactome_members(read_value, seed, limit), wanted))
    return dict(zip(wanted, found, strict=True))


def query_parts(text: str) -> tuple[str, str]:
    """UniProt accession and gene symbol named by one target field.

    A menu value is ``SYMBOL|ACCESSION``. A typed accession or symbol is used
    on its own. Anything else, including a PDB id, is not a structure-search
    query. The caller can still match that text against a listed entry id.
    """
    raw = text.strip()
    if not raw:
        return "", ""
    accession = ""
    symbol = ""
    pieces = [part.strip() for part in raw.split("|") if part.strip()]
    for piece in pieces:
        token = piece.split()[0] if piece.split() else ""
        if not token:
            continue
        if _UNIPROT_ID.fullmatch(token.upper()) and not accession:
            accession = token.upper()
        elif _GENE_SYMBOL.fullmatch(token) and not symbol:
            symbol = token
    return accession, symbol


def reactome_pathway_list(
    read_value: ReadValue,
    accessions: Sequence[str],
    *,
    limit: int = 24,
) -> list[tuple[str, str]]:
    """Reactome pathway id and name for these accessions, in seed order.

    Every human pathway is kept, not only the first one a seed maps to. The
    structure search that follows a disease with no chosen pathway still uses
    that first pathway. This list is what the pathway menu offers instead.
    """
    wanted = [item for item in accessions if _UNIPROT_ID.fullmatch(item.upper())]
    if not wanted:
        return []

    def one(accession: str) -> list[tuple[str, str]]:
        payload = _safe_value(read_value, f"{REACTOME_PATHWAYS_URL}/{quote(accession)}/pathways")
        found: list[tuple[str, str]] = []
        if not isinstance(payload, list):
            return found
        for row in payload:
            if not isinstance(row, dict):
                continue
            pathway_id = str(row.get("stId") or "")
            if not pathway_id.startswith("R-HSA-"):
                continue
            name = str(row.get("displayName") or pathway_id).strip() or pathway_id
            found.append((pathway_id, name))
        return found

    if len(wanted) == 1:
        groups = [one(wanted[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(MAX_ENTRY_WORKERS, len(wanted))) as pool:
            groups = list(pool.map(one, wanted))
    listed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for group in groups:
        for pathway_id, name in group:
            if pathway_id in seen:
                continue
            seen.add(pathway_id)
            listed.append((pathway_id, name))
            if len(listed) >= limit:
                return listed
    return listed


def pathway_uniprot_members(read_value: ReadValue, pathway_id: str, limit: int) -> list[str]:
    """UniProt participants of one chosen pathway. An id that is not human is empty."""
    if not str(pathway_id).startswith("R-HSA-"):
        return []
    participants = _safe_value(read_value, f"{REACTOME_PARTICIPANTS_URL}/{quote(pathway_id)}")
    found: list[str] = []
    _collect_uniprot(participants, found, limit)
    return found


def partner_symbols(read_value: ReadValue, seeds: Sequence[str], limit: int) -> list[str]:
    """Gene symbols that interact with the seeds. Partners of partners are not followed."""
    found: list[str] = []
    known: set[str] = set()
    for seed in seeds:
        if len(found) >= limit:
            break
        for name in _string_symbols(_safe_value(read_value, _string_partners_url(seed))):
            _remember_symbol(found, known, name)
            if len(found) >= limit:
                break
    return found


def _reactome_members(read_value: ReadValue, accession: str, limit: int) -> list[str]:
    pathways = _safe_value(read_value, f"{REACTOME_PATHWAYS_URL}/{quote(accession)}/pathways")
    pathway_id = ""
    if isinstance(pathways, list):
        for row in pathways:
            if isinstance(row, dict) and str(row.get("stId") or "").startswith("R-HSA-"):
                pathway_id = str(row["stId"])
                break
    if not pathway_id:
        return []
    participants = _safe_value(read_value, f"{REACTOME_PARTICIPANTS_URL}/{quote(pathway_id)}")
    found: list[str] = []
    _collect_uniprot(participants, found, limit)
    return found


def _collect_uniprot(payload: object, found: list[str], limit: int) -> None:
    if len(found) >= limit:
        return
    if isinstance(payload, list):
        for item in payload:
            _collect_uniprot(item, found, limit)
            if len(found) >= limit:
                return
        return
    if not isinstance(payload, dict):
        return
    database = str(payload.get("databaseName") or payload.get("database") or "").casefold()
    schema = str(payload.get("schemaClass") or "").casefold()
    identifier = str(payload.get("identifier") or payload.get("accession") or "").split("-", 1)[0].strip().upper()
    if (
        _UNIPROT_ID.fullmatch(identifier)
        and ("uniprot" in database or "geneproduct" in schema)
        and identifier not in found
    ):
        found.append(identifier)
    for value in payload.values():
        if isinstance(value, dict | list):
            _collect_uniprot(value, found, limit)
            if len(found) >= limit:
                return


def _safe_value(read_value: ReadValue, url: str) -> object:
    try:
        return read_value(url)
    except (OSError, ValueError, AssertionError):
        return None


def _remember_symbol(found: list[str], known: set[str], symbol: str) -> None:
    if not _GENE_SYMBOL.fullmatch(symbol) or symbol.casefold() == "gene":
        return
    key = symbol.casefold()
    if key in known or key in {item.casefold() for item in found}:
        return
    found.append(symbol)


def _extend_unique(base: list[str], extra: list[str]) -> list[str]:
    found = list(base)
    seen = {item.casefold() for item in found}
    for item in extra:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(item)
    return found


def _campaigns(group: str, title: str, entry: dict) -> tuple[str, ...]:
    """Design classes for one row of a deposit. Nucleic rows stay nucleic-only."""
    if group in {"dna", "rna"}:
        return ("DNA & RNA",)
    if group not in {"protein", "gene"}:
        return ()
    found = ["Proteins"]
    info = entry.get("rcsb_entry_info") or {}
    if _count(info, "nonpolymer_entity_count"):
        found.append("Small Molecules")
    if _count(info, "branched_entity_count"):
        found.append("Sugars (Glycans)")
    if _mentions_antibody(title, entry):
        found.append("Antibodies & Nanobodies")
    return tuple(found)


def _mentions_antibody(title: str, entry: dict) -> bool:
    parts = [title]
    keywords = entry.get("struct_keywords")
    if isinstance(keywords, str):
        parts.append(keywords)
    elif isinstance(keywords, dict):
        parts.extend(str(value) for value in keywords.values())
    text = " ".join(parts).casefold()
    return any(word in text for word in ("antibody", "nanobody", "immunoglobulin", "vhh"))


def _count(info: dict, key: str) -> int:
    try:
        return int(info.get(key) or 0)
    except (TypeError, ValueError):
        return 0
