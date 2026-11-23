"""Measurements for the target, not for the design.

Paralog and the three plasma axes are properties of the protein a binder is
aimed at, so each is looked up by accession or gene symbol. No designed sequence
is involved anywhere in this module, which is why it is separate from
:py:mod:`rfproteina.disease.liability`: that module promises nothing it holds
sends a sequence anywhere, and keeping these lookups out of it keeps that
promise literally true.

Every read goes through the per-host allowance in
:py:mod:`rfproteina.disease.limits`, so a target lookup cannot outpace what a
public service permits.

Two of the three plasma axes have no published number that makes a measurement a
liability. Those ship measuring and reporting, with ``clear`` held behind
``threshold_not_set`` until a threshold is named, because inventing a cut and
calling the result clear is the one outcome worse than no measurement.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.parse import quote

from rfproteina.disease.coevolution import read_paralog_coevolution
from rfproteina.disease.liability import _column_key, _table_lines
from rfproteina.disease.research import LookupMissError

#: What a public read fails with when the service did not answer usefully.
#: ``HTTPException`` is in here because a truncated response -- a body shorter
#: than its own Content-Length -- raises it from ``http.client``, and it is
#: neither an OSError nor a ValueError, so it escaped every catch between the
#: reader and the screen and would have aborted a whole multi-target run.
_UNAVAILABLE = (HTTPException, OSError, ValueError)

TARGET_LIABILITY_AXES = ("paralog", "plasma_abundance", "plasma_hub", "plasma_fold")

ReadJson = Callable[[str], object]


@dataclass(frozen=True)
class TargetLimits:
    """Thresholds for the axes that have no published cut.

    Each is None until a caller names it, and an axis with no threshold reports
    its measurement and stays untested. That is deliberate: these three axes
    have real measurements and no published number that turns one into a
    liability, and a cut invented here would clear designs on our authority.
    """

    abundance_max: float | None = None
    hub_max: int | None = None
    fold_max: float | None = None
    #: Percent sequence identity at or above which a human paralog counts as a
    #: cross-reactivity liability. None until a caller names it, for the reason
    #: above and for one more: the identity that matters is over the contacted
    #: residues, and this is identity over the whole alignment, so a cut chosen
    #: here would be answering a slightly different question than it appears to.
    paralog_identity_max: float | None = None


#: A UniProt accession, which is what a paralog lookup needs. A gene symbol row
#: is not one, and is reported as such rather than guessed at.
_ACCESSION = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
_HUMAN = "9606"
#: UniRef50 is UniProt's own 50-per-cent-identity clustering, so the boundary is
#: theirs and not ours. Suzek et al., Bioinformatics 31:926-932 (2015).
_UNIREF_IDENTITY = "0.5"
_CLUSTER_FIELD = "uniref_cluster_50"
#: What the UniRef50 side of this axis measures by, and what its measurement
#: counts. This axis has two methods on two scales -- a count of human cluster
#: members here, a percent identity on the Compara path -- and only the second
#: named itself, so a stored ``measurement`` of 2 could not be told from an
#: identity of 2 per cent, which is the clear side of the cut the other method
#: is judged against. Which of the two answers is decided by the UniProt
#: release the query reached, so the record has to say.
_UNIREF_METHOD = "uniref50_cluster_human_members"
_UNIREF_UNITS = "human_cluster_members"
_COMPARA_UNITS = "percent_identity"
_MEMBER_LIMIT = 25
#: STRING's published high-confidence band. Below it a partner count is mostly
#: text-mining and the number means much less.
_STRING_CONFIDENCE = 700
_STRING_LIMIT = 2000
#: Physical rather than the default functional network: ``required_score``
#: filters the combined score without restricting the evidence channel, so the
#: functional network's degree is largely a count of co-mentions in text.
_STRING_NETWORK = "physical"
#: Ensembl Compara, asked over REST for one gene rather than downloaded whole:
#: the all-species homology dump is 11.23 GB and this question is about one
#: protein at a time. ``target_taxon`` keeps it to human paralogs.
_COMPARA = (
    "https://rest.ensembl.org/homology/symbol/homo_sapiens/"
    "{gene}?type=paralogues;content-type=application/json;target_taxon=" + _HUMAN
)
#: Compara returns every paralog its gene trees hold, at any identity, and most
#: of them are not a cross-reactivity question: ERBB2 has 53 paralogs whose
#: median identity is 17.4 per cent. A count is therefore an evolutionary
#: measurement and not this axis's, which is why what is recorded is the
#: identity of the closest one and not how many there are.
_COMPARA_LIMIT = 500
#: Named on every request, as STRING's own guidance asks and as the NCBI reads
#: in this package already do with ``tool=``.
_CALLER = "Humanitys-Last-Disease"
#: Human Protein Atlas blood-concentration columns, in picograms per litre. The
#: immunoassay value is preferred over the mass-spectrometry one where both are
#: reported, because it is the directly calibrated measurement.
_HPA_COLUMNS = "g,gs,up,blconcms,blconcia"
_HPA_IMMUNOASSAY = "Blood concentration - Conc. blood IM [pg/L]"
_HPA_MASS_SPEC = "Blood concentration - Conc. blood MS [pg/L]"
_HPA_UNIPROT = "Uniprot"
_FOLD_TABLE = "plasma_fold"
#: The bounds the sibling readers already keep, in the same numbers. The fold
#: reader read the whole file into memory with no bound at all, which is the one
#: supplied table that could take the screen down rather than refuse: a fold
#: table is a handful of rows per target, and a file past either bound is a
#: wrong file -- a proteome, a metrics dump, an unterminated export.
FOLD_FILE_LIMIT = 1_048_576
FOLD_ROW_LIMIT = 256
#: How much of a text cell is kept, the same cut the solubility reader makes.
#: A study is a citation and not a document.
_TEXT_LIMIT = 120
_FOLD_COLUMNS = ("fold", "fold_change", "log2_fold_change")
_SOURCE_COLUMNS = ("source", "study", "reference", "doi")
#: Columns by which a supplied fold table may say which protein a row is about.
#: The documented table has none -- one file sits in one run's folder and is
#: taken as that run's target -- but a user building one table across a campaign
#: labels the rows, and then the label is the only thing that says whether the
#: row covers this target.
_FOLD_TARGET_COLUMNS = (
    "accession",
    "uniprot",
    "uniprot_id",
    "uniprotkb",
    "target",
    "target_id",
    "gene",
    "gene_symbol",
    "gene_name",
    "protein",
)
_USED_FOLD_COLUMNS = _FOLD_COLUMNS + _SOURCE_COLUMNS + _FOLD_TARGET_COLUMNS
#: The spellings a predictor export uses for "this cell holds nothing", shared
#: with the sibling readers in this package. A study column filled with one of
#: them names no study.
_ABSENT_CELLS = frozenset(
    {"", "nan", "none", "null", "na", "n/a", "n.a.", "n.d.", "nd", ".", "-", "--", "inconclusive"}
)
#: A sentinel for a field a row carries past the header's last column. ``csv``
#: files those under one key; a row that has one is misaligned from its first
#: extra separator onward, so every later cell is read under its neighbour's
#: column name.
_EXTRA = "\x00extra"


def measure_target_liability(
    target_id: str,
    *,
    directory: Path,
    read_json: ReadJson | None = None,
    limits: TargetLimits | None = None,
    unresolved: str | None = None,
) -> dict[str, dict[str, object]]:
    """One record per target axis, in the order the axis list names them.

    ``read_json`` is injected so a caller can supply a reader and so a test
    needs no network. A lookup that fails leaves its axis untested with a
    blocker naming what failed; it never fails the screen.

    ``unresolved`` is a blocker for a target whose accession could not be
    chosen, and it wins over the network: those three axes are not asked.
    Without a reader, and with no such blocker, the three axes that need one
    say so. ``plasma_fold`` does not need one -- it reads a table in the output
    directory -- so it is measured either way, rather than reporting that a
    network it never wanted was unavailable. ``paralog_coevolution.tsv`` is
    read the same way and stored on the paralog record. That note leaves the
    UniRef50 state as it is.
    """
    bar = TargetLimits() if limits is None else limits
    records = {"plasma_fold": _fold_record(target_id, directory, bar.fold_max)}
    if unresolved:
        for axis in ("paralog", "plasma_abundance", "plasma_hub"):
            records[axis] = _axis("untested", None, unresolved)
    elif read_json is None:
        for axis in ("paralog", "plasma_abundance", "plasma_hub"):
            records[axis] = _axis("untested", None, "network_not_permitted")
    else:
        records["paralog"] = _paralog_record(target_id, read_json, bar.paralog_identity_max)
        records["plasma_abundance"] = _abundance_record(target_id, read_json, bar.abundance_max)
        records["plasma_hub"] = _hub_record(target_id, read_json, bar.hub_max)
    ordered = {name: records[name] for name in TARGET_LIABILITY_AXES}
    # The co-evolution note sits beside the UniRef50 result. It never replaces
    # state, measurement, or blocker, including on a lookup that was not asked.
    paralog = dict(ordered["paralog"])
    paralog.update(read_paralog_coevolution(directory))
    ordered["paralog"] = paralog
    return ordered


def uniprot_accession(target_id: str) -> str | None:
    """The UniProt accession in ``target_id``, or None when it is not one.

    An isoform suffix is dropped. A gene symbol is not an accession.
    """
    return _accession(target_id)


def _accession(target_id: str) -> str | None:
    """The accession a target row carries, or None when it is not one.

    An isoform suffix is dropped, because ``P05113-2`` is a form of the same
    protein and the services here are keyed on the base accession. A gene
    symbol is not an accession and is reported as such rather than guessed at.
    """
    text = target_id.strip().upper().split("-", 1)[0]
    return text if _ACCESSION.fullmatch(text) else None


def _paralog_record(target_id: str, read: ReadJson, limit: float | None = None) -> dict[str, object]:
    """Whether another human protein shares this target's UniRef50 cluster.

    A binder raised against one member of a close family can bind the others.
    Another human reviewed protein in the target's UniRef50 cluster is positive
    evidence of such a family, so that is a hit.

    The absence of one is **not** evidence of the absence of a family, so this
    axis has no clear. A UniRef50 cluster is built around a representative and
    does not gather a protein's near-identical relatives: KRAS and EGFR both
    come back as singletons while having exactly the families a binder would
    cross-react with. A clear here would have been the measurement asserting
    something it cannot see.

    Identity over the contacted residues is the question that could clear this,
    but no published cross-reactivity identity cut exists to judge it against,
    so this does not invent one. A curated judgement goes in axis_evidence.json.
    """
    accession = _accession(target_id)
    if accession is None:
        return _axis("untested", None, "target_not_accession")
    # A miss is the service answering that it holds no record, which is not the
    # service being unreachable, and neither of them is a clear.
    try:
        cluster = _uniref_cluster(read, accession)
    except LookupMissError:
        return _axis("untested", None, "cluster_not_found")
    except _UNAVAILABLE:
        return _axis("untested", None, "lookup_unavailable")
    if not cluster:
        return _axis("untested", None, "cluster_not_found")
    try:
        members = _human_members(read, cluster)
    except LookupMissError:
        return _axis("untested", None, "members_not_listed")
    except _UNAVAILABLE:
        return _axis("untested", None, "lookup_unavailable")
    if accession not in members:
        # The target is not in its own cluster's membership, so the listing is
        # not the set this verdict needs. Being the only name in an incomplete
        # list is not evidence of being the only paralog.
        return _axis("untested", None, "members_not_listed")
    if len(members) >= _MEMBER_LIMIT:
        truncated = _axis("untested", len(members), "members_truncated")
        truncated["method"] = _UNIREF_METHOD
        truncated["units"] = _UNIREF_UNITS
        return truncated
    others = {name: symbol for name, symbol in members.items() if name != accession}
    named = sorted(symbol or name for name, symbol in others.items())
    if others:
        record = _axis("hit", len(others), None)
        record["method"] = _UNIREF_METHOD
        record["units"] = _UNIREF_UNITS
        record["cluster"] = cluster
        record["paralogs"] = named
        return record
    # Being alone in a UniRef50 cluster is not evidence of having no paralogs,
    # and this is not a near miss: KRAS comes back a singleton here while
    # Compara puts HRAS at 85.2 per cent identity to it and NRAS at 84.1, and
    # ERBB2 comes back a singleton while EGFR is 50.7 per cent. Those are the
    # families a binder is most likely to cross-react with, and the clustering
    # cannot see them, because a UniRef50 cluster is built around a
    # representative rather than gathering a protein's relatives.
    #
    # So the singleton is handed to Compara, which answers with identities and
    # is asked over REST for this one gene rather than downloaded whole. The
    # UniRef50 hit above keeps precedence: it is positive evidence of a family
    # and nothing here can downgrade it.
    singleton = _axis("untested", 0, "cluster_singleton_not_evidence")
    singleton["method"] = _UNIREF_METHOD
    singleton["units"] = _UNIREF_UNITS
    singleton["cluster"] = cluster
    singleton["paralogs"] = named
    if limit is None:
        # Asked only when the caller named a cut, and not for tidiness: without
        # one this axis cannot judge, so the two extra requests would buy a
        # ``threshold_not_set`` that the line above already says for free. The
        # sibling axes report an unjudged measurement because their lookup
        # happens either way; this one is two more requests per target, which
        # on a campaign of any size is the difference between a screen and a
        # rate limit. Naming the cut is what turns it on.
        return singleton
    return _compara_record(singleton, read, accession, limit)


def _compara_record(singleton: dict[str, object], read: ReadJson, accession: str, limit: float) -> dict[str, object]:
    """The paralog record for a UniRef50 singleton, measured against Compara.

    Returns ``singleton`` unchanged whenever Compara could not answer. A
    service being unreachable is not a measurement, and this axis had no clear
    at all before Compara, so falling back to that is losing nothing.

    Reached only with a threshold in hand, so there is no unjudged branch here.

    What is reported is the identity of the **closest** paralog, never the
    count. Compara holds every paralog in its gene trees at any identity -- 53
    for ERBB2, median identity 17.4 per cent -- so a count judged against a cut
    would call a protein a liability on the strength of fifty others that share
    no surface with it.
    """
    gene = None
    try:
        gene = _primary_symbol(read, accession)
    except (LookupMissError, *_UNAVAILABLE):
        gene = None
    if not gene:
        singleton["compara_blocker"] = "gene_symbol_not_found"
        return singleton
    try:
        examined, scored = _compara_paralogs(read, gene)
    except LookupMissError:
        singleton["compara_blocker"] = "gene_not_in_compara"
        return singleton
    except _UNAVAILABLE:
        singleton["compara_blocker"] = "lookup_unavailable"
        return singleton

    closest = scored[0] if scored else None
    record = _axis("untested", closest[0] if closest else None, None)
    record["cluster"] = singleton["cluster"]
    record["paralogs"] = singleton["paralogs"]
    record["compara_examined"] = examined
    record["compara_scored"] = len(scored)
    record["method"] = "ensembl_compara_paralogues_identity"
    record["units"] = _COMPARA_UNITS
    if closest is not None:
        record["closest_paralog"] = closest[1]
    record["threshold"] = limit
    if closest is not None and closest[0] >= limit:
        record["state"] = "hit"
        record["paralogs"] = sorted({*singleton["paralogs"], closest[1]})  # type: ignore[misc]
        return record
    if len(scored) < examined:
        record["blocker"] = "compara_identities_incomplete"
        return record
    # Compara examined this gene's tree and found nothing at or above the cut,
    # which is a measurement rather than an absence of one. It is the first
    # clear this axis has ever been able to give, and it is still only as good
    # as the cut the caller named and the whole-alignment identity Compara
    # reports.
    record["state"] = "clear"
    return record


def _primary_symbol(read: ReadJson, accession: str) -> str | None:
    """The primary gene symbol for an accession, from the service already used.

    Needed because Compara's homology endpoint is keyed by symbol while this
    screen resolves a target to an accession. Asked of UniProt rather than
    read from the proteome FASTA, so this module goes on knowing nothing about
    which files happen to be downloaded.

    The extraction is :func:`_gene_symbol`, which the cluster-member listing
    already uses on the same shape of entry. An earlier version of this
    repeated that walk and, being defined above it under the same name, was
    silently replaced by it -- the duplicate was the bug, not the shadowing.
    """
    url = (
        "https://rest.uniprot.org/uniprotkb/search?format=json&size=1"
        f"&fields=gene_primary&query=accession:{_quote(accession)}"
    )
    found = read(url)
    if not isinstance(found, dict):
        return None
    results = found.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return None
    return _gene_symbol(results[0]) or None


def _compara_paralogs(read: ReadJson, gene: str) -> tuple[int, list[tuple[float, str]]]:
    """How many paralogs Compara holds for a gene, and their identities.

    The count includes every returned homology. The sorted list includes
    named records with a usable percent identity within the processing limit.
    """
    found = read(_COMPARA.format(gene=_quote(gene)))
    if not isinstance(found, dict):
        raise ValueError("Compara returned no object")
    data = found.get("data")
    if not isinstance(data, list) or not data:
        # No gene tree for this symbol. Not the same as no paralogs.
        raise LookupMissError(gene)
    first = data[0]
    homologies = first.get("homologies") if isinstance(first, dict) else None
    if not isinstance(homologies, list):
        raise ValueError("Compara returned no homologies list")
    scored: list[tuple[float, str]] = []
    for item in homologies[:_COMPARA_LIMIT]:
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        source = item.get("source")
        if not isinstance(target, dict):
            continue
        named = target.get("id")
        if not isinstance(named, str) or not named.strip():
            continue
        # Both directions, from the one response that carries them, and the
        # higher is kept. Compara's ``perc_id`` is identity over the length of
        # the sequence it belongs to, so a pair has two numbers: ERBB2 and EGFR
        # are 50.7 per cent identical over EGFR and 48.9 over ERBB2. Reading
        # only the target's made the measurement depend on which protein was
        # asked about -- across a 50 per cent cut the same pair made ERBB2 a hit
        # and EGFR a clear. The larger share is also the conservative one: it is
        # the fraction of a real protein's residues that a binder's epitope
        # could be reading.
        identities = [
            float(value)
            for block in (target, source)
            if isinstance(block, dict)
            and isinstance(value := block.get("perc_id"), int | float)
            and not isinstance(value, bool)
            and 0 <= value <= 100
        ]
        if not identities:
            continue
        scored.append((max(identities), named))
    scored.sort(reverse=True)
    return len(homologies), scored


def _uniref_cluster(read: ReadJson, accession: str) -> str:
    query = f"uniprot_id:{accession} AND identity:{_UNIREF_IDENTITY}"
    url = "https://rest.uniprot.org/uniref/search?format=json&size=1&fields=id&query=" + _quote(query)
    payload = read(url)
    for entry in _entries(payload):
        found = entry.get("id")
        if isinstance(found, str) and found:
            return found
    return ""


def _human_members(read: ReadJson, cluster: str) -> dict[str, str]:
    query = f"{_CLUSTER_FIELD}:{cluster} AND organism_id:{_HUMAN} AND reviewed:true"
    url = (
        "https://rest.uniprot.org/uniprotkb/search?format=json"
        f"&size={_MEMBER_LIMIT}&fields=accession,gene_names&query=" + _quote(query)
    )
    payload = read(url)
    found: dict[str, str] = {}
    for entry in _entries(payload):
        accession = entry.get("primaryAccession")
        if not isinstance(accession, str) or not accession:
            continue
        found[accession.upper()] = _gene_symbol(entry)
    return found


def _gene_symbol(entry: dict[str, object]) -> str:
    genes = entry.get("genes")
    if not isinstance(genes, list):
        return ""
    for gene in genes:
        if not isinstance(gene, dict):
            continue
        name = gene.get("geneName")
        if isinstance(name, dict) and isinstance(name.get("value"), str):
            return str(name["value"])
    return ""


def _abundance_record(target_id: str, read: ReadJson, limit: float | None) -> dict[str, object]:
    """The target's blood concentration in picograms per litre.

    An abundant circulating antigen is a molar sink: it binds the drug before
    the drug reaches the target pool, so the dose that works is set by the
    antigen rather than by the disease.

    The strata are published -- Anderson and Anderson, Mol Cell Proteomics
    1:845-867 (2002) -- but no published number says which concentration makes a
    target a liability, because that depends on the dose. So this measures and
    reports, and clears only against a threshold the caller names.
    """
    accession = _accession(target_id)
    if accession is None:
        return _axis("untested", None, "target_not_accession")
    url = (
        "https://www.proteinatlas.org/api/search_download.php?format=json&compress=no"
        f"&columns={_HPA_COLUMNS}&search=" + _quote(accession)
    )
    try:
        payload = read(url)
    except LookupMissError:
        return _axis("untested", None, "target_not_reported")
    except _UNAVAILABLE:
        return _axis("untested", None, "lookup_unavailable")
    row = _hpa_row(payload, accession)
    if row is None:
        return _axis("untested", None, "target_not_reported")
    concentration = _number(row.get(_HPA_IMMUNOASSAY))
    assay = "immunoassay"
    if concentration is None:
        concentration = _number(row.get(_HPA_MASS_SPEC))
        assay = "mass_spectrometry"
    if concentration is None:
        return _axis("untested", None, "concentration_not_reported")
    if concentration < 0:
        return _axis("untested", None, "concentration_invalid")
    if limit is None:
        record = _axis("untested", concentration, "threshold_not_set")
    else:
        record = _axis("hit" if concentration >= limit else "clear", concentration, None)
        record["threshold"] = limit
    record["assay"] = assay
    record["units"] = "pg/L"
    return record


def _hpa_row(payload: object, accession: str) -> dict[str, object] | None:
    """The row for this accession. The search is fuzzy, so the match is exact.

    Searching ``ALB`` returns ``CALB1`` and ``PALB2`` too, so taking the first
    row would measure a different protein than the one asked about.
    """
    if not isinstance(payload, list):
        return None
    for row in payload:
        if not isinstance(row, dict):
            continue
        for name in _listed_accessions(row.get(_HPA_UNIPROT)):
            if name == accession:
                return row
    return None


def _listed_accessions(listed: object) -> set[str]:
    """Every accession a cell names, however the atlas packed them.

    The column arrives as a list in some responses and as one comma-joined
    string in others, and an entry with several accessions would otherwise
    match none of them.
    """
    values = listed if isinstance(listed, list) else [listed]
    found: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        for piece in value.replace(";", ",").split(","):
            text = piece.strip().upper().split("-", 1)[0]
            if text:
                found.add(text)
    return found


def _hub_record(target_id: str, read: ReadJson, limit: int | None) -> dict[str, object]:
    """How many high-confidence interaction partners the target has.

    A hub is a target whose inhibition moves much more than itself.

    The count is taken on the **physical** network. ``required_score`` filters
    the combined score and does not restrict which evidence channel supplied it,
    so the default functional network stays dominated by co-mention: IL5 returns
    80 partners at a high combined score, of which most carry a text-mining
    score and only 4 survive on physical evidence. A degree built from co-mention
    measures how much a protein has been written about, which is not what a hub
    is. The request also asks for a high cap, because a limit below the real
    count censors it silently.

    There is no published absolute degree that makes a protein a hub: the honest
    definition is distributional, the top quantile of the network's own degree
    distribution after Han et al., Nature 430:88-93 (2004), which needs the whole
    network. So this reports the degree and clears only against a named threshold.
    """
    # An accession-shaped id loses its isoform suffix; a gene symbol is passed
    # through, which STRING resolves directly.
    identifier = _accession(target_id) or target_id.strip()
    if not identifier:
        return _axis("untested", None, "target_not_named")
    url = (
        "https://string-db.org/api/json/interaction_partners"
        f"?species={_HUMAN}&required_score={_STRING_CONFIDENCE}&limit={_STRING_LIMIT}"
        f"&network_type={_STRING_NETWORK}&caller_identity={_CALLER}"
        "&identifiers=" + _quote(identifier)
    )
    try:
        payload = read(url)
    except LookupMissError:
        return _axis("untested", None, "partners_not_listed")
    except _UNAVAILABLE:
        return _axis("untested", None, "lookup_unavailable")
    if not isinstance(payload, list):
        return _axis("untested", None, "lookup_unavailable")
    uses_ids = any(isinstance(row, dict) and ("stringId_A" in row or "stringId_B" in row) for row in payload)
    left, right = ("stringId_A", "stringId_B") if uses_ids else ("preferredName_A", "preferredName_B")
    queried: set[str] = set()
    partners: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            return _axis("untested", None, "partners_not_listed")
        source, partner = row.get(left), row.get(right)
        if not isinstance(source, str) or not isinstance(partner, str) or not source.strip() or not partner.strip():
            return _axis("untested", None, "partners_not_listed")
        source, partner = source.strip(), partner.strip()
        queried.add(source)
        if partner != source:
            partners.add(partner)
    if len(queried) > 1:
        # One identifier was asked about and the rows name more than one
        # protein, so the partners are not one neighbourhood and the degree is
        # not this target's. Counting them all would publish another protein's
        # partners against this target.
        return _axis("untested", None, "partners_not_listed")
    if not partners:
        # STRING answers an identifier it could not map with an empty list, and
        # a protein with no high-confidence physical partner answers with the
        # same empty list. Nothing in the response tells the two apart -- a
        # degree of zero carries no row, so there is no preferredName_A to
        # check the mapping against. A clear here would say "not a hub" on the
        # strength of a stale accession or a symbol STRING does not know. The
        # count is reported and the axis stays untested.
        return _axis("untested", 0, "partners_not_listed")
    degree = len(partners)
    if len(payload) >= _STRING_LIMIT:
        return _axis("untested", degree, "degree_censored")
    if limit is None:
        record = _axis("untested", degree, "threshold_not_set")
    else:
        record = _axis("hit" if degree > limit else "clear", degree, None)
        record["threshold"] = limit
    record["confidence"] = _STRING_CONFIDENCE
    # What the degree counted, so a number stamped into a screen says which
    # network it came from rather than leaving a reader to assume. The release
    # goes with it because STRING re-derives its network each version: without
    # it a degree recorded today cannot be checked against the same query later.
    record["network"] = _STRING_NETWORK
    release = _string_release(payload)
    if release:
        record["release"] = release
    return record


def _string_release(payload: object) -> str:
    """The STRING version a response names, when it names one.

    Not every endpoint returns it, so this is provenance where it is available
    rather than a requirement. Absent, the record simply carries no release.
    """
    rows = payload if isinstance(payload, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("version", "string_version", "db_version"):
            found = row.get(key)
            if isinstance(found, str) and found.strip():
                return found.strip()
    return ""


def _fold_record(target_id: str, directory: Path, limit: float | None) -> dict[str, object]:
    """How far the target's plasma level moves between disease and health.

    A target whose circulating level swings with the disease moves the antigen
    sink with it, so the dose that works in one patient does not in another.

    No public per-target API reports this: UKB-PPP (Sun et al., Nature
    622:329-338, 2023) gives associations rather than fold changes. So the number
    comes from a table the caller supplies, and that table has to name the study
    it came from, because a fold change without a provenance is a rumour.

    A table whose rows carry an accession or a gene label is matched against
    this target, and a table with no such column is read as this run's own, the
    way the documented two-column form is.
    """
    path = _table(directory, _FOLD_TABLE)
    if path is None:
        return _axis("untested", None, "fold_not_configured")
    size = _file_size(path)
    if size is None:
        return _axis("untested", None, "fold_unreadable")
    if size > FOLD_FILE_LIMIT:
        return _axis("untested", None, "fold_too_large")
    rows = _rows(path)
    if rows is None:
        return _axis("untested", None, "fold_unreadable")
    if rows and any(column in rows[0] for column in _FOLD_TARGET_COLUMNS):
        # The table says which protein each row is about, so the label decides
        # coverage and not the folder the file sits in. Unlabelled, the file is
        # taken as this run's target, as the documented one-column table is;
        # labelled, a row naming another protein is another protein's
        # measurement. Reading them all took the largest magnitude in the file,
        # so one campaign-wide table published the biggest swing it held against
        # every target in it -- a hit for targets that had no such row, and a
        # clear for a target whose own row was never found.
        rows = [row for row in rows if _names_target(row, target_id)]
        if not rows:
            return _axis("untested", None, "fold_target_not_in_table")
    if len(rows) > FOLD_ROW_LIMIT:
        # Bounded after parsing rather than before, because the file-size check
        # above already caps what was read. This one is about a table that is
        # small enough to read and still not a fold table, and "a handful of
        # rows" is per target: counted over the whole file instead, the bound
        # refused the labelled campaign table ``_FOLD_TARGET_COLUMNS`` exists
        # for. A campaign across more than FOLD_ROW_LIMIT proteins is an
        # ordinary campaign, and one target's own sourced 3.4-fold row came back
        # as ``fold_too_large`` -- the whole file called a wrong file, and every
        # hit in it withheld, because other targets were listed beside it.
        return _axis("untested", None, "fold_too_large")
    cells = _fold_cells(rows)
    if not cells:
        return _axis("untested", None, "fold_not_reported")
    # An unsourced row does not lend its number another row's study, and it
    # does not supply the maximum. A file with no study at all still reports
    # its largest cell and says the source was not named.
    sourced = [cell for cell in cells if cell[2]]
    column, fold, source = max(sourced or cells, key=lambda item: _linear_fold(item[0], item[1]))
    if not source:
        return _fold_axis("untested", column, fold, "source_not_named")
    magnitude = _linear_fold(column, fold)
    if limit is None:
        record = _fold_axis("untested", column, fold, "threshold_not_set")
    elif magnitude >= limit:
        record = _fold_axis("hit", column, fold, None)
        record["threshold"] = limit
    else:
        # The sourced rows clear the cut, but a clear speaks for the whole
        # table. An unsourced cell that does NOT clear is the same table saying
        # this target swings further than the clear would claim, so the clear is
        # withheld and that cell is the number reported.
        #
        # Deliberately narrow. A larger unsourced cell that also clears
        # contradicts nothing, so the clear stands, and a hit is never withheld
        # for want of a citation a hit does not rest on. Only the clear, the one
        # verdict that covers rows it did not read, has to answer for them.
        denied = [cell for cell in cells if not cell[2] and _linear_fold(cell[0], cell[1]) >= limit]
        if denied:
            loudest = max(denied, key=lambda item: _linear_fold(item[0], item[1]))
            return _fold_axis("untested", loudest[0], loudest[1], "source_not_named")
        record = _fold_axis("clear", column, fold, None)
        record["threshold"] = limit
    # Not "source": every axis record passes through ``screen._stamp_source``,
    # which writes that key to say whether the STATE was measured, curated or
    # untested. Stored there, the study was overwritten before the screen was
    # written, so the provenance this axis refuses to be measured without was
    # the one thing the record did not keep.
    # Cut to the same length the solubility reader cuts its public text to. The
    # study is a citation, not a document, and this one is stored in screen.json
    # and returned over the MCP view, so an export that pasted a whole abstract
    # into the column would carry it into both.
    record["fold_source"] = _short_text(source)
    return record


def _fold_axis(state: str, column: str, value: float, blocker: str | None) -> dict[str, object]:
    return {
        **_axis(state, value, blocker),
        "fold_column": column,
        "linear_fold_magnitude": _number(_linear_fold(column, value)),
    }


def _names_target(row: dict[str, str], target_id: str) -> bool:
    """Whether a labelled row names the target this screen is about.

    A cell may hold several accessions, the way a UniProt export writes them,
    and an isoform suffix is dropped because ``P05113-2`` is a form of the same
    protein. A gene symbol is compared with the identifier the caller named: no
    symbol-to-accession lookup happens here, so a table that labels rows by
    symbol cannot be matched to an accession target, and the axis says the
    target was not found in the table rather than reading a row that might be
    someone else's.
    """
    wanted = {text for text in (target_id.strip().upper(), _accession(target_id) or "") if text}
    if not wanted:
        return False
    for column in _FOLD_TARGET_COLUMNS:
        cell = row.get(column, "")
        if cell.strip().casefold() in _ABSENT_CELLS:
            continue
        for piece in cell.replace(";", ",").split(","):
            text = piece.strip().upper()
            if text and (text in wanted or text.split("-", 1)[0] in wanted):
                return True
    return False


def _fold_cells(rows: list[dict[str, str]]) -> list[tuple[str, float, str]]:
    """Every fold cell a row fills, with its column and the study on that row.

    A row that fills two of the fold columns states two magnitudes, and the
    column it was read from is what says which scale each one is on. Taking the
    first in a fixed order dropped the other without a word: a row writing 1.1
    in the linear column and 3 in the log2 column -- about eightfold -- cleared
    against a twofold cut on the 1.1. Each cell is a candidate, and
    :func:`_fold_record` judges the largest magnitude in the table, which is
    the rule the documentation already states for rows.
    """
    found: list[tuple[str, float, str]] = []
    for row in rows:
        source = _cell(row, _SOURCE_COLUMNS)
        for name in _FOLD_COLUMNS:
            number = _number(row.get(name, ""))
            if number is not None:
                found.append((name, number, source))
    return found


def _linear_fold(column: str, value: float) -> float:
    """The fold change a threshold compares, as a linear magnitude.

    ``log2_fold_change`` is not already that magnitude. A value of 1 is a
    twofold change, and judging the raw 1 against a threshold of 2 would clear
    a swing the threshold names as a liability.

    A plain ratio is symmetric about 1, not 0. A disease/healthy column writes
    a fivefold fall as 0.2 and a fivefold rise as 5, and both are a fivefold
    change, so both measure 5. Taking the absolute value instead would clear
    0.2 against any threshold above it, and would lose the row selection in
    :func:`_fold_record` to a smaller upward value in the same table.
    """
    if column == "log2_fold_change":
        if abs(value) >= 1023:
            return float("inf")
        return 2.0 ** abs(value)
    magnitude = abs(value)
    if magnitude < 1e-300:
        return float("inf")
    return max(magnitude, 1.0 / magnitude)


def _table(directory: Path, stem: str) -> Path | None:
    for suffix in (".tsv", ".csv"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def _file_size(path: Path) -> int | None:
    """The file's size in bytes, or None when it cannot be stat'ed."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def _rows(path: Path) -> list[dict[str, str]] | None:
    """The table's rows with normalised keys, or None when it did not parse.

    Read as ``utf-8-sig`` and put through the same banner skip and key
    normalisation as the predictor tables, because a user is told one contract
    for every table this program reads. Read strictly, a byte-order mark -- the
    one Excel writes by default -- stayed on the first column's name, and a
    fold table that plainly held a fold column reported having none: the same
    measurement went from ``hit 4.2`` to ``untested`` on bytes that differ only
    by that mark.

    The four refusals are the ones the sibling readers already make, and each
    changes the number rather than withholding it: a repeated column, a header
    holding nothing this reads, a row with a surplus field, and a quote left
    open.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError, MemoryError):
        return None
    lines = _table_lines(text)
    if not lines:
        return []
    dialect = csv.excel_tab if "\t" in lines[0] else csv.excel
    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)), dialect=dialect, restkey=_EXTRA)
        names = list(reader.fieldnames or ())
        # csv keeps the last of two columns with one name, silently. A header
        # carrying the fold twice -- the study's number in one column and
        # another method's in the next -- judged whichever came last, with
        # nothing in the record saying which. Neither is used.
        if _duplicate_keys(names):
            return None
        if not any(_column_key(name) in _USED_FOLD_COLUMNS for name in names):
            # The header holds none of the columns this reads, which is what a
            # semicolon export and an unmarked banner line both look like. Read
            # on and the answer is "no fold was reported", which sends the user
            # to add a column the file already has and rules out the delimiter,
            # the one thing actually wrong.
            return None
        found: list[dict[str, str]] = []
        for row in reader:
            if _surplus(row.get(_EXTRA)):
                # Every field past the first unquoted separator is read under
                # the next column's name, so a stray separator moved the study
                # into the fold column and the fold into nothing. A misaligned
                # row is not partly usable.
                return None
            found.append(
                {
                    _column_key(key): (value or "").strip()
                    for key, value in row.items()
                    if isinstance(key, str) and key != _EXTRA
                }
            )
        if len(found) != len(lines) - 1:
            # An unterminated quote swallows every following line into one cell.
            # The rows it ate are not reported missing, so a table whose second
            # row held the larger swing came back as a confident clear on the
            # first.
            return None
        return found
    except csv.Error:
        return None


def _duplicate_keys(fieldnames: list[str]) -> bool:
    """Refuse a repeated column this reader uses. A repeated notes column stays."""
    seen: set[str] = set()
    for raw in fieldnames:
        key = _column_key(raw)
        if key not in _USED_FOLD_COLUMNS:
            continue
        if key in seen:
            return True
        seen.add(key)
    return False


def _surplus(extra: object) -> bool:
    """Whether a row carried a field the header has no column for."""
    if isinstance(extra, str):
        return bool(extra.strip())
    if isinstance(extra, list):
        return any(isinstance(item, str) and item.strip() for item in extra)
    return extra is not None


def _cell(row: dict[str, str], names: tuple[str, ...]) -> str:
    """The first of ``names`` this row fills. A sentinel fills nothing.

    ``NA`` and ``-`` are how an export writes an empty cell, and read as text
    they named a study: the axis published a hit citing a provenance of ``NA``,
    which is the rumour the source column exists to refuse.
    """
    for name in names:
        text = row.get(name, "").strip()
        if text and text.casefold() not in _ABSENT_CELLS:
            return text
    return ""


def _short_text(raw: str) -> str:
    """Short public text, cut rather than stored whole. The siblings' rule."""
    cleaned = " ".join(raw.split())
    if len(cleaned) > _TEXT_LIMIT:
        return cleaned[: _TEXT_LIMIT - 3].rstrip() + "..."
    return cleaned


def _number(value: object) -> float | None:
    """A finite number, or None. A blank or a sentinel is not a measurement."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.casefold() in {"", "nan", "none", "null", "na", "."}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _entries(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _quote(text: str) -> str:
    return quote(text, safe="")


def _axis(state: str, measurement: int | float | None, blocker: str | None) -> dict[str, object]:
    return {"state": state, "measurement": measurement, "blocker": blocker}
