"""Signing in to the AlphaFold Server, and what this deliberately will not do.

There is nothing to mint. The server authenticates a Google Account through
Google's ordinary per-site session-cookie sign-in; it issues no API key, no
bearer token and no service credential, because it has no programmatic
interface to issue one for. The complete published FAQ describes exactly one
automation path: upload a JSON file of up to 100 jobs through the web page, and
download a result archive. "Logging in", for a program, therefore means being
able to say whether the person at this machine has a signed-in browser session
and what their remaining daily budget is -- not holding their credential.

**What this module will not do, and why.** Two of the three obvious shortcuts
are closed by the service's own terms rather than by taste:

* It will not drive the site's internal RPC endpoints. The Prohibited Use Policy
  incorporated into the AlphaFold Server Additional Terms forbids reverse
  engineering the service, and the internal endpoints are reachable only by
  doing that. Third-party projects that map them exist; this is not one.
* It will not fetch job pages automatically. ``robots.txt`` disallows ``/fold/*``
  and Google's terms forbid automated access that ignores it.
* It will not accept, store or type a password. No password is needed for the
  supported path, and a module that could hold one would invite being asked to.

None of that limits the useful work, because the parts worth automating are the
ones either side of the upload: building a request that the server will accept
on the first try, and filing the archive that comes back so every refolding
metric reads it. Those are :py:mod:`rfproteina.af3server.request` and
:py:mod:`rfproteina.af3server.bundle`.

**The ledger.** The account allows 30 jobs per day. That bound is the one thing a
batch plan has to respect and the one thing a program cannot read back from the
service, so submissions are recorded here as they are handed over. The record is
a *local guard*, not the server's accounting: the server's own reset boundary and
timezone are not published, so this counts by UTC date and
:meth:`Ledger.remaining` is an estimate that is never reported as the server's
number. When the two disagree, the server is right.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from rfproteina.disease.key_store import forget_secret, load_secret, save_secret

#: Where a person signs in and uploads. Not fetched by this module.
SERVER_URL = "https://alphafoldserver.com"
#: The name the encrypted blob is stored under.
SECRET_NAME = "af3_session"
#: Jobs the account may run per day, from the server's own FAQ.
DAILY_JOB_LIMIT = 30
#: Saved drafts the job history may hold. An upload lands as drafts, so a caller
#: importing many files at once can fill this without running anything.
DRAFT_LIMIT = 500
#: Days of submission history kept. Enough to see the pattern, short enough that
#: the record does not become a log of everything ever folded.
_HISTORY_DAYS = 30


@dataclass(frozen=True)
class Ledger:
    """Submissions this machine has recorded, by UTC date."""

    counts: dict[str, int]
    #: An optional label for the signed-in account, as the person typed it. Kept
    #: encrypted because it is usually an email address.
    account: str | None = None
    #: Where the browser puts downloads, so an archive can be picked up.
    downloads: Path | None = None
    #: Whether the person has recorded that they accepted the non-commercial
    #: terms. Recording it is not accepting it -- only they can do that, on the
    #: site -- and this module never ticks it on their behalf.
    terms_acknowledged: bool = False
    #: This account's own daily allowance, when it is not the published 30.
    #: ``None`` uses :data:`DAILY_JOB_LIMIT`. Settable because 30 is what the
    #: FAQ documents for an ordinary account and the figure has already been
    #: raised twice; an account with a different allowance would otherwise be
    #: told its own batch overflows a limit that does not apply to it.
    daily_limit: int | None = None

    @property
    def limit(self) -> int:
        """The allowance this ledger plans against."""
        return self.daily_limit if self.daily_limit is not None else DAILY_JOB_LIMIT

    def today(self) -> int:
        """Jobs recorded for the current UTC date."""
        return self.counts.get(_today(), 0)

    def remaining(self) -> int:
        """An estimate of today's remaining budget. Never the server's number.

        Floored at zero. A negative would mean this machine recorded more
        submissions than the limit allows, which says the ledger drifted -- the
        person submitted from elsewhere, or the server's day boundary is not
        UTC -- and reporting a negative budget as a fact would be worse than
        reporting none left.
        """
        return max(self.limit - self.today(), 0)

    def as_dict(self) -> dict[str, object]:
        """A record shaped for a run log. The account label is **not** in it."""
        return {
            "submitted_today": self.today(),
            "remaining_estimate": self.remaining(),
            "daily_limit": self.limit,
            "daily_limit_is_the_published_one": self.daily_limit is None,
            "signed_in_recorded": self.account is not None,
            "terms_acknowledged": self.terms_acknowledged,
        }


def load_ledger() -> Ledger:
    """The recorded ledger, or an empty one when nothing is stored.

    A damaged store raises, which :func:`load_secret` does deliberately: an
    unreadable credential read as "none stored" would be overwritten by the next
    write, taking whatever else was in it.
    """
    body = load_secret(SECRET_NAME)
    counts = body.get("counts")
    account = body.get("account")
    downloads = body.get("downloads")
    return Ledger(
        counts={
            str(day): int(value)
            for day, value in (counts.items() if isinstance(counts, dict) else ())
            if isinstance(value, int | float) and not isinstance(value, bool)
        },
        account=account.strip() or None if isinstance(account, str) else None,
        downloads=Path(downloads) if isinstance(downloads, str) and downloads else None,
        terms_acknowledged=body.get("terms_acknowledged") is True,
        daily_limit=_stored_limit(body.get("daily_limit")),
    )


def _stored_limit(value: object) -> int | None:
    """A stored allowance, or None when none was stored or it is unusable.

    A zero or negative allowance is discarded rather than honoured: it would
    make every batch read as fully deferred, which looks exactly like a quota
    that is already spent and is not something anyone would mean.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value) if value > 0 else None


def save_ledger(ledger: Ledger) -> None:
    """Replace the stored ledger, keeping only the recent history."""
    kept = _recent(ledger.counts)
    save_secret(
        SECRET_NAME,
        {
            "counts": kept,
            "account": ledger.account or "",
            "downloads": str(ledger.downloads) if ledger.downloads is not None else "",
            "terms_acknowledged": ledger.terms_acknowledged,
            "daily_limit": ledger.daily_limit or 0,
        },
    )


def sign_in(
    *,
    account: str | None = None,
    downloads: Path | None = None,
    terms_acknowledged: bool | None = None,
    daily_limit: int | None = None,
) -> Ledger:
    """Record that this machine has a signed-in browser session.

    This does not perform a sign-in and cannot: the person signs in to
    ``alphafoldserver.com`` in their own browser, with their own Google Account,
    and accepts the non-commercial terms there. What is recorded here is only
    what a later submission needs to know -- where downloads land, whose account
    it is if they want that written down, and that the terms were accepted.

    Passing ``account`` does not make the session valid and is never checked
    against the service. It is a label, so a person running two accounts can
    tell which ledger is which.
    """
    if account is not None and _looks_like_a_password(account):
        # An account label is an email address or a nickname. A long opaque
        # string is far more likely to be a pasted credential, and this is the
        # one place in the round trip where somebody might paste one.
        raise ValueError(
            "That looks like a secret rather than an account label. This tool never needs a "
            "password or token: sign in to the AlphaFold Server in your own browser instead."
        )
    if daily_limit is not None and daily_limit < 1:
        raise ValueError("A daily allowance must be at least one job.")
    current = load_ledger()
    updated = Ledger(
        counts=dict(current.counts),
        account=(account.strip() or None) if account is not None else current.account,
        downloads=downloads if downloads is not None else current.downloads,
        terms_acknowledged=(terms_acknowledged if terms_acknowledged is not None else current.terms_acknowledged),
        daily_limit=daily_limit if daily_limit is not None else current.daily_limit,
    )
    save_ledger(updated)
    return updated


def sign_out() -> bool:
    """Forget the stored ledger. Returns whether there was one."""
    return forget_secret(SECRET_NAME)


def record_submission(count: int, *, when: date | None = None) -> Ledger:
    """Add ``count`` submitted jobs to the ledger and return it.

    Called when the jobs are handed over, not when they finish: the quota is
    spent at submission, and a job that fails still spent it.
    """
    if count < 0:
        raise ValueError("A submission count cannot be negative.")
    current = load_ledger()
    day = (when or _utc_date()).isoformat()
    counts = dict(current.counts)
    counts[day] = counts.get(day, 0) + count
    updated = Ledger(counts, current.account, current.downloads, current.terms_acknowledged, current.daily_limit)
    save_ledger(updated)
    return updated


@dataclass(frozen=True)
class SubmissionPlan:
    """How many jobs can go today, and how many must wait."""

    today: int
    deferred: int
    remaining_estimate: int
    #: Why the plan is shorter than the batch, or None when it is not.
    blocker: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "today": self.today,
            "deferred": self.deferred,
            "remaining_estimate": self.remaining_estimate,
            "blocker": self.blocker,
        }


def plan_submission(jobs: int, ledger: Ledger | None = None) -> SubmissionPlan:
    """Split a batch against the remaining daily budget.

    A plan rather than a refusal, because a batch larger than one day's quota is
    a normal thing to have: the server itself lets a job be saved and run when
    the quota refreshes. What matters is that the overflow is *named* -- a
    submission that quietly ran the first thirty designs and dropped the rest
    would read afterwards as a complete campaign.
    """
    if jobs < 0:
        raise ValueError("A job count cannot be negative.")
    current = ledger if ledger is not None else load_ledger()
    budget = current.remaining()
    today = min(jobs, budget)
    deferred = jobs - today
    blocker = None
    if deferred:
        blocker = "daily_quota_exhausted" if budget == 0 else "over_daily_quota"
    return SubmissionPlan(today, deferred, budget, blocker)


def submission_url() -> str:
    """Where a person goes to upload. Opened for them, never fetched."""
    return SERVER_URL


def _looks_like_a_password(value: str) -> bool:
    """Whether a supposed account label is more likely to be a credential."""
    text = value.strip()
    if "@" in text or " " in text:
        return False
    return len(text) >= 24 and any(character.isdigit() for character in text)


def _recent(counts: dict[str, int]) -> dict[str, int]:
    """The ledger trimmed to the recent window, in date order."""
    today = _utc_date().toordinal()
    kept: dict[str, int] = {}
    for day in sorted(counts):
        try:
            age = today - date.fromisoformat(day).toordinal()
        except ValueError:
            continue
        if 0 <= age < _HISTORY_DAYS:
            kept[day] = counts[day]
    return kept


def _utc_date() -> date:
    return datetime.now(UTC).date()


def _today() -> str:
    return _utc_date().isoformat()
