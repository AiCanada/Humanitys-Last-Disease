"""Can this construct actually be ordered? Sequence-level synthesis feasibility.

``oligo.py`` already judges a nucleic *design*: GC fraction, CpG count, the
longest G run, off-target hybridisation against a supplied transcriptome. None
of that asks the separate question a vendor asks, which is whether a DNA
sequence can be made at all. Gene synthesis houses reject orders on local GC
windows, long repeats, and hairpins, none of which a whole-sequence GC fraction
sees. This module adds exactly those terms and nothing ``oligo.py`` already
carries.

It is pure Python: no DnaChisel, no BLAST, no network. DnaChisel (MIT) is the
natural upgrade for *fixing* a sequence rather than scoring it, and the handoff
note says where it would slot in.

Every bar here is a suggested default pending calibration. Vendor
specifications differ and none was read for this module, so none is cited as a
source. They are starting points, marked as such in
:data:`SYNTHESIS_BARS`, and the screen should not gate on them until someone
checks them against the vendor actually being used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

METHOD = "sequence_synthesis_feasibility_v1"

#: Sliding window, in bases, the local GC fraction is measured over.
GC_WINDOW = 50
#: Shortest stem counted as a hairpin, in bases.
HAIRPIN_STEM = 8
#: Smallest loop allowed between the two halves of a hairpin, in bases.
HAIRPIN_LOOP = 3
#: Longest span a hairpin is looked for across, in bases. Keeps the scan linear
#: on long constructs rather than quadratic on the whole sequence.
HAIRPIN_SPAN = 60
#: Shortest direct repeat reported, in bases.
REPEAT_LENGTH = 20

_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "U": "A"}

SYNTHESIS_BARS: dict[str, dict[str, object]] = {
    "gc_fraction": {
        "min": 0.25,
        "max": 0.75,
        "source": "suggested default pending calibration; no vendor specification was read for this module",
    },
    "min_window_gc": {
        "min": 0.20,
        "source": "suggested default pending calibration",
    },
    "max_window_gc": {
        "max": 0.80,
        "source": "suggested default pending calibration",
    },
    "longest_homopolymer": {
        "max": 9,
        "source": "suggested default pending calibration",
    },
    "longest_repeat": {
        "max": 24,
        "source": "suggested default pending calibration",
    },
    "longest_hairpin_stem": {
        "max": 12,
        "source": "suggested default pending calibration",
    },
}


@dataclass(frozen=True)
class SynthesisReport:
    """One construct, scored for manufacturability. Lengths are in bases."""

    length: int
    alphabet: str
    gc_fraction: float
    min_window_gc: float | None
    max_window_gc: float | None
    gc_window: int
    longest_homopolymer: int
    homopolymer_base: str | None
    longest_repeat: int
    repeat_sequence: str | None
    longest_hairpin_stem: int
    hairpin_at: int | None
    forbidden_sites: tuple[dict[str, object], ...]
    method: str = METHOD
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["forbidden_sites"] = [dict(item) for item in self.forbidden_sites]
        record["notes"] = list(self.notes)
        record["bars"] = SYNTHESIS_BARS
        return record


class SequenceUnreadableError(Exception):
    """The sequence is empty or is not DNA or RNA."""


def _clean(sequence: str) -> tuple[str, str]:
    text = "".join((sequence or "").split()).upper()
    if not text:
        raise SequenceUnreadableError("empty_sequence")
    letters = set(text)
    if letters <= set("ACGTN"):
        alphabet = "dna"
    elif letters <= set("ACGUN"):
        alphabet = "rna"
    else:
        raise SequenceUnreadableError("not_a_nucleic_sequence")
    return text, alphabet


def _gc(text: str) -> float:
    counted = sum(1 for base in text if base in "GC")
    return counted / len(text) if text else 0.0


def _window_gc(text: str, window: int) -> tuple[float | None, float | None]:
    if len(text) < window:
        return None, None
    values = [_gc(text[start : start + window]) for start in range(0, len(text) - window + 1)]
    return round(min(values), 4), round(max(values), 4)


def _homopolymer(text: str) -> tuple[int, str | None]:
    best = 0
    base: str | None = None
    run = 0
    previous = ""
    for letter in text:
        run = run + 1 if letter == previous else 1
        previous = letter
        if run > best:
            best, base = run, letter
    return best, base


def _repeat(text: str, length: int) -> tuple[int, str | None]:
    """Longest exact direct repeat, found by growing from the shortest reported length."""
    if len(text) < 2 * length:
        return 0, None
    best = 0
    found: str | None = None
    width = length
    while width <= len(text) // 2:
        seen: dict[str, int] = {}
        hit: str | None = None
        for start in range(0, len(text) - width + 1):
            piece = text[start : start + width]
            if piece in seen:
                hit = piece
                break
            seen[piece] = start
        if hit is None:
            break
        best, found = width, hit
        width += 1
    return best, found


def _hairpin(text: str, stem: int, loop: int, span: int) -> tuple[int, int | None]:
    """Longest reverse-complement stem within ``span`` bases, with at least ``loop`` between."""
    best = 0
    where: int | None = None
    complement = {base: _COMPLEMENT.get(base, "N") for base in set(text)}
    for start in range(0, max(0, len(text) - stem)):
        piece = text[start : start + stem]
        if "N" in piece:
            continue
        target = "".join(complement[base] for base in reversed(piece))
        window_start = start + stem + loop
        window = text[window_start : start + span]
        index = window.find(target)
        if index == -1:
            continue
        width = stem
        while True:
            left = text[start : start + width + 1]
            if len(left) <= width or "N" in left:
                break
            grown = "".join(complement[base] for base in reversed(left))
            if window.find(grown) == -1:
                break
            width += 1
        if width > best:
            best, where = width, start + 1
    return best, where


def screen_sequence(
    sequence: str,
    *,
    forbidden: dict[str, str] | None = None,
    gc_window: int = GC_WINDOW,
) -> SynthesisReport:
    """Score one construct. ``forbidden`` maps a site name to its literal sequence.

    No default forbidden list is supplied. Which restriction or assembly sites
    matter depends on the cloning plan, and a guessed list would read as a
    program standard.
    """
    text, alphabet = _clean(sequence)
    low, high = _window_gc(text, gc_window)
    homopolymer, base = _homopolymer(text)
    repeat_length, repeat_text = _repeat(text, REPEAT_LENGTH)
    stem, at = _hairpin(text, HAIRPIN_STEM, HAIRPIN_LOOP, HAIRPIN_SPAN)
    sites: list[dict[str, object]] = []
    for name, site in (forbidden or {}).items():
        pattern = "".join(site.split()).upper()
        if not pattern:
            continue
        positions = []
        index = text.find(pattern)
        while index != -1:
            positions.append(index + 1)
            index = text.find(pattern, index + 1)
        if positions:
            sites.append({"name": name, "site": pattern, "positions": positions})
    return SynthesisReport(
        length=len(text),
        alphabet=alphabet,
        gc_fraction=round(_gc(text), 4),
        min_window_gc=low,
        max_window_gc=high,
        gc_window=gc_window,
        longest_homopolymer=homopolymer,
        homopolymer_base=base,
        longest_repeat=repeat_length,
        repeat_sequence=repeat_text,
        longest_hairpin_stem=stem,
        hairpin_at=at,
        forbidden_sites=tuple(sites),
        notes=(
            "every bar in this module is a suggested default, not a vendor specification",
            "hairpins are exact reverse-complement stems, not a folding free energy",
        ),
    )


def verdicts(report: SynthesisReport) -> dict[str, dict[str, object]]:
    """The suggested bars applied, each carrying the fact that it is only suggested."""
    record = asdict(report)
    out: dict[str, dict[str, object]] = {}
    for name, rule in SYNTHESIS_BARS.items():
        value = record.get(name)
        minimum = rule.get("min")
        maximum = rule.get("max")
        if value is None:
            state = "untested"
        else:
            low_ok = minimum is None or value >= minimum
            high_ok = maximum is None or value <= maximum
            state = "clear" if (low_ok and high_ok) else "hit"
        out[name] = {
            "value": value,
            "state": state,
            "min": minimum,
            "max": maximum,
            "threshold_source": rule["source"],
        }
    sites = len(report.forbidden_sites)
    out["forbidden_sites"] = {
        "value": sites,
        "state": "hit" if sites else "clear",
        "min": None,
        "max": 0,
        "threshold_source": "caller-supplied site list; empty when the caller named none",
    }
    return out
