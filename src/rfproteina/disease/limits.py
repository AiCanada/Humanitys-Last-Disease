"""What this program allows itself against one public service at a time.

These are somebody else's servers. Only NCBI publishes a number -- three requests
a second without an API key, ten with one -- so that one is a rate, not a guess.
EMBL-EBI publishes no number at all: its terms are qualitative, that use which
"prevents, or is likely to prevent" EBI serving others will be blocked. So the
allowances here are set low enough to be unremarkable rather than up to a ceiling
nobody has written down, and every request names this program, so a service that
does want to throttle or ask about it knows what it is looking at.

An allowance is per host and per process. Different services never wait on each
other, so one slow service does not hold up the rest.

``at_once`` bounds requests in flight. ``every`` bounds how often one may *start*,
which is a different thing: a service answering in 0.18s is asked five times a
second by one caller making no concurrent requests at all, which is how the NCBI
limit came to be exceeded without anything running in parallel.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

#: Sent on every public request, so a service sees this program rather than a
#: bare interpreter version.
USER_AGENT = "Humanitys-Last-Disease (RFD4-Proteina disease panel)"


@dataclass(frozen=True)
class Allowance:
    """Requests in flight at once, and the least time between starting them."""

    at_once: int
    every: float = 0.0


#: By host suffix; the longest matching suffix wins. A service with no published
#: limit gets a small number rather than a generous one.
HOST_ALLOWANCES: tuple[tuple[str, Allowance], ...] = (
    # Documented: 3 requests a second without an API key, 429 beyond it. Paced
    # rather than merely capped, because one serial caller already exceeds it.
    ("eutils.ncbi.nlm.nih.gov", Allowance(at_once=3, every=0.34)),
    # EMBL-EBI: PRIDE, IntAct, PeptideAtlas. No number published anywhere.
    ("ebi.ac.uk", Allowance(at_once=3)),
    ("reactome.org", Allowance(at_once=4)),
    ("string-db.org", Allowance(at_once=4)),
    ("proteinatlas.org", Allowance(at_once=4)),
    ("rest.uniprot.org", Allowance(at_once=4)),
    ("api.platform.opentargets.org", Allowance(at_once=2)),
    ("ttd.idrblab.cn", Allowance(at_once=1)),
    # RCSB serves one small record per request and is asked for the most of them.
    ("data.rcsb.org", Allowance(at_once=8)),
    ("search.rcsb.org", Allowance(at_once=4)),
)
#: A host nobody listed is a host nobody checked, so it gets the least.
DEFAULT_ALLOWANCE = Allowance(at_once=2)


def allowance_for(url: str) -> Allowance:
    """The allowance for this URL's host. The longest listed suffix wins."""
    host = urlsplit(url).netloc.casefold()
    host = host.rsplit("@", 1)[-1].split(":", 1)[0]
    best: Allowance | None = None
    longest = -1
    for suffix, allowance in HOST_ALLOWANCES:
        if (host == suffix or host.endswith(f".{suffix}")) and len(suffix) > longest:
            best, longest = allowance, len(suffix)
    return best if best is not None else DEFAULT_ALLOWANCE


class _Gate:
    """One host's allowance, with the state that enforces it."""

    def __init__(self, allowance: Allowance) -> None:
        self.allowance = allowance
        self.slots = threading.Semaphore(max(1, allowance.at_once))
        self.clock = threading.Lock()
        self.last = 0.0


_gates: dict[str, _Gate] = {}
_gates_lock = threading.Lock()


def _gate_for(url: str) -> _Gate:
    host = urlsplit(url).netloc.casefold() or "unknown"
    with _gates_lock:
        gate = _gates.get(host)
        if gate is None:
            gate = _Gate(allowance_for(url))
            _gates[host] = gate
        return gate


@contextmanager
def slot(url: str) -> Iterator[None]:
    """Hold this host's allowance for the duration of one request.

    Waiting happens here rather than inside the caller, so a reader does not have
    to know which service it is reading and every caller is paced the same way,
    whether it is running alone or inside a pool.
    """
    gate = _gate_for(url)
    gate.slots.acquire()
    try:
        if gate.allowance.every > 0:
            with gate.clock:
                now = time.monotonic()
                due = gate.last + gate.allowance.every
                if due > now:
                    time.sleep(due - now)
                    now = due
                gate.last = now
        yield
    finally:
        gate.slots.release()
