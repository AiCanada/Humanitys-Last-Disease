"""Talking to the hosted AlphaFold Server, which has no API.

The AlphaFold Server at ``alphafoldserver.com`` folds a job described by
sequence and hands back a zip. There is no documented public endpoint for any of
it: the published interface is a person uploading a JSON file of up to 100 jobs
and downloading an archive. Two of the obvious shortcuts past that are closed by
the service's own terms rather than by taste -- the Prohibited Use Policy
incorporated into its Additional Terms forbids reverse engineering the service,
and ``robots.txt`` disallows ``/fold/*`` -- so nothing here drives an internal
endpoint or fetches a job page.

That shape is what this package is built around. The parts which *are* exactly
specified -- the request format and the result bundle -- are code, and the part
which is not is a seam a human steps through rather than an endpoint invented
from a guess.

:py:mod:`~rfproteina.af3server.structure`
    What a design's mmCIF holds, in either of the two spellings mmCIF allows. A
    category with one row is written flat and with two rows as a ``loop_``, and
    this repository's writer emits both -- so every single-polymer design is
    invisible to a loop-only reader.

:py:mod:`~rfproteina.af3server.request`
    Designs to the ``alphafoldserver`` JSON dialect, checked against the real
    allowlists and the real token budget before anything is uploaded. A design
    holding an entity the dialect cannot carry is refused by name rather than
    sent without it: a request that drops a chain does not fail, it answers a
    different question. Nothing here reaches the network.

:py:mod:`~rfproteina.af3server.session`
    Where the daily-quota ledger lives, and -- more to the point -- what this
    will not do to sign in. The service issues no API key or token, because it
    has no programmatic interface to issue one for.

:py:mod:`~rfproteina.af3server.bundle`
    A downloaded result archive to the structure/confidence pairs the existing
    oracle metrics read, via
    :py:func:`rfproteina.metrics.oracles.imported_refolds.place_refold`.

:py:mod:`~rfproteina.af3server.cli`
    The ``rfd4-af3server`` command, which is the usual way in.

The round trip is therefore: ``request`` writes, a person uploads, ``bundle``
files the answer, and every refolding metric in the repository then runs
unchanged because the files are at the paths the oracle would have written.

See :doc:`the how-to </how-to/af3-server>` for the walkthrough, including the
three flat metrics a server refold cannot supply.
"""

from __future__ import annotations
