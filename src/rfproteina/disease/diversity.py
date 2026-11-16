"""How different one run's designs are from each other.

Every other screen reads one design at a time, so a batch of replicates that
came back as the same molecule under eight names passes all of them eight
times and reads as eight shots on goal. This is the one measurement that is
about the set: how many designs were written, how many are distinct, and how
tightly they cluster.

Identity is counted position by position, between designs of the same length
only. A pair of different lengths is not aligned and not guessed at: it is
counted as uncompared and said so.

That is the common case here, not the rare one. The flow's own binder contig is
a range -- ``design_spec`` asks for 60 to 110 residues -- so one run's designs
routinely differ in length and no pair of them can be compared this way. Such a
run still gets its design and distinct counts, which catch the worst failure on
their own, and says plainly that nothing was compared rather than reporting a
spread it did not measure.

Nothing here is an axis, nothing here changes the call, and the sequences are
not stored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DIVERSITY_METHOD = "positional_identity"
#: At or above this identity two designs are held to be the same design. It is
#: a reading aid for the group count, not a pass band: no call turns on it.
DIVERSITY_THRESHOLD = 0.9
#: Designs this will compare pairwise. Past it the counts are still reported --
#: they cost one pass -- and the pairwise block is left out rather than run,
#: because the work grows with the square of the batch.
DIVERSITY_DESIGN_LIMIT = 200
#: Window width for the shared-k-mer similarity. Five is short enough that two
#: designs of different lengths still overlap where they are alike, and long
#: enough that the twenty residues do not collide by chance.
KMER_WIDTH = 5


def design_diversity(pairs: Sequence[tuple[str, str | None]]) -> dict[str, object] | None:
    """The spread across one run's designed chains, or None when there is no spread to measure.

    ``pairs`` is each design's example id and its designed sequence, the shape
    ``_designed_sequences`` returns. A design whose chain could not be read is
    left out of every count. Fewer than two readable sequences returns None:
    one design has nothing to differ from, and a record saying so would read as
    a measurement.
    """
    sequences = [sequence for _name, sequence in pairs if isinstance(sequence, str) and sequence]
    if len(sequences) < 2:
        return None
    record: dict[str, object] = {
        "method": DIVERSITY_METHOD,
        "threshold": DIVERSITY_THRESHOLD,
        "designs": len(sequences),
        "distinct": len(set(sequences)),
    }
    if len(sequences) > DIVERSITY_DESIGN_LIMIT:
        record["pairwise"] = "not_performed"
        record["limit"] = DIVERSITY_DESIGN_LIMIT
        return record
    identities, uncompared = _pairwise(sequences)
    record["compared_pairs"] = len(identities)
    record["uncompared_pairs"] = uncompared
    # Shared k-mers answer the question identity cannot: two designs of
    # different length still have most of their short windows in common or
    # they do not. It is reported and never grouped on, because the cut that
    # would turn a similarity into "the same design" is not one this program
    # gets to choose.
    record.update(_kmer_spread(sequences))
    if not identities:
        # Nothing was compared, so there is no grouping to report. Run anyway,
        # the union-find returns one group per design and the record reads
        # "eight separate designs" -- a spread, stated confidently, from zero
        # comparisons. With a 60-to-110 contig that is the ordinary case.
        record["pairwise"] = "not_compared"
        return record
    record["max_identity"] = _round4(max(value for _left, _right, value in identities))
    record["mean_identity"] = _round4(sum(value for _l, _r, value in identities) / len(identities))
    groups = _groups(len(sequences), identities)
    record["groups"] = len(groups)
    record["largest_group"] = max(groups)
    record["pairwise"] = "measured"
    return record


def diversity_phrase(record: Mapping[str, object]) -> str | None:
    """The measured-line clause for a stored spread, or None when there is none."""
    designs = _count(record, "designs")
    distinct = _count(record, "distinct")
    if designs is None or distinct is None:
        return None
    parts = [f"designs {designs}, distinct {distinct}"]
    groups = _count(record, "groups")
    if groups is not None:
        parts.append(f"groups {groups} at identity {_number(record.get('threshold'))}")
    largest = _count(record, "largest_group")
    if largest is not None and largest > 1:
        parts.append(f"largest group {largest}")
    highest = record.get("max_identity")
    if isinstance(highest, int | float) and not isinstance(highest, bool):
        parts.append(f"max identity {_number(highest)}")
    shared = record.get("mean_similarity")
    if isinstance(shared, int | float) and not isinstance(shared, bool):
        parts.append(f"mean shared {_number(KMER_WIDTH)}-mers {_number(shared)}")
    uncompared = _count(record, "uncompared_pairs")
    if uncompared:
        parts.append(f"uncompared pairs {uncompared}")
    if record.get("pairwise") == "not_compared":
        parts.append("pairwise not_compared, no two designs share a length")
    if record.get("pairwise") == "not_performed":
        parts.append(f"pairwise not_performed above {_number(record.get('limit'))} designs")
    return "; ".join(parts)


def sequence_identity(left: str, right: str) -> float | None:
    """Matching positions over length, or None when the two cannot be compared.

    Two chains of different length are not aligned here. An alignment would
    have to choose a gap penalty, and the number it produced would be that
    choice as much as the designs, so the pair is reported as uncompared.
    """
    if not left or len(left) != len(right):
        return None
    same = sum(1 for one, two in zip(left, right, strict=True) if one == two)
    return same / len(left)


def _kmer_spread(sequences: Sequence[str]) -> dict[str, object]:
    """Shared-window similarity over every pair, whatever their lengths.

    The share of ``KMER_WIDTH``-residue windows two designs have in common, as
    a Jaccard index: 1 when they draw on exactly the same set, 0 when they
    share none. Unlike positional identity it needs no alignment and no common
    length, so it answers on the runs this flow actually produces, where the
    contig is a range and no two designs are the same size.

    A chain shorter than one window has none, and a pair where either side has
    none is left out rather than scored 0.
    """
    windows = [{chain[at : at + KMER_WIDTH] for at in range(len(chain) - KMER_WIDTH + 1)} for chain in sequences]
    shares: list[float] = []
    for left in range(len(windows)):
        for right in range(left + 1, len(windows)):
            one, two = windows[left], windows[right]
            if not one or not two:
                continue
            shares.append(len(one & two) / len(one | two))
    if not shares:
        return {"kmer_width": KMER_WIDTH, "max_similarity": None, "mean_similarity": None}
    return {
        "kmer_width": KMER_WIDTH,
        "max_similarity": _round4(max(shares)),
        "mean_similarity": _round4(sum(shares) / len(shares)),
    }


def _pairwise(sequences: Sequence[str]) -> tuple[list[tuple[int, int, float]], int]:
    """Every comparable pair with its identity, and how many pairs were not comparable."""
    identities: list[tuple[int, int, float]] = []
    uncompared = 0
    for left in range(len(sequences)):
        for right in range(left + 1, len(sequences)):
            value = sequence_identity(sequences[left], sequences[right])
            if value is None:
                uncompared += 1
                continue
            identities.append((left, right, value))
    return identities, uncompared


def _groups(total: int, identities: Sequence[tuple[int, int, float]]) -> list[int]:
    """Group sizes under single linkage at the threshold, largest first.

    Single linkage, because the question is how many separate designs this run
    really produced: a chain of near-identical designs is one design however it
    is reached. A design with no comparable partner is its own group.
    """
    parent = list(range(total))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right, value in identities:
        if value < DIVERSITY_THRESHOLD:
            continue
        one, two = root(left), root(right)
        if one != two:
            parent[max(one, two)] = min(one, two)
    sizes: dict[int, int] = {}
    for index in range(total):
        key = root(index)
        sizes[key] = sizes.get(key, 0) + 1
    return sorted(sizes.values(), reverse=True)


def _count(record: Mapping[str, object], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "not_recorded"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _round4(number: float) -> float:
    return float(f"{number:.4f}")
