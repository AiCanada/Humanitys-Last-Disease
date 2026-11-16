"""Signing in to a service that issues no credential, and the quota ledger.

The AlphaFold Server authenticates a Google Account through an ordinary browser
sign-in. It issues no API key and no token, because it publishes no programmatic
interface to issue one for. So there is nothing here that holds a credential,
and the module's job is narrower than "log in": record that a signed-in session
exists, remember where downloads land, and keep the daily quota ledger.

The ledger is the part worth testing. The account allows 30 jobs a day, that
bound is the one a batch plan has to respect, and it is the one thing a program
cannot read back from the service. The tests below pin two properties:

1. A batch larger than the remaining budget produces a **plan**, not a silent
   truncation. A submission that ran the first thirty of forty designs and said
   nothing would read afterwards as a complete campaign.
2. The remaining count is never reported as the server's own. It is this
   machine's count, it can drift, and a drifted ledger reports zero left rather
   than a negative budget.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rfproteina.af3server.session import (
    DAILY_JOB_LIMIT,
    Ledger,
    load_ledger,
    plan_submission,
    record_submission,
    save_ledger,
    sign_in,
    sign_out,
    submission_url,
)


@pytest.fixture
def store(monkeypatch, tmp_path: Path) -> Path:
    directory = tmp_path / "keys"
    monkeypatch.setenv("RFD4_KEY_DIR", str(directory))
    return directory


def test_nothing_recorded_is_an_empty_ledger_not_an_error(store: Path) -> None:
    ledger = load_ledger()
    assert ledger.counts == {}
    assert ledger.account is None
    assert ledger.terms_acknowledged is False
    assert ledger.today() == 0
    assert ledger.remaining() == DAILY_JOB_LIMIT


def test_sign_in_records_what_a_later_submission_needs(store: Path, tmp_path: Path) -> None:
    downloads = tmp_path / "Downloads"
    ledger = sign_in(account="someone@example.org", downloads=downloads, terms_acknowledged=True)
    assert ledger.downloads == downloads
    assert ledger.terms_acknowledged is True
    assert load_ledger().account == "someone@example.org"


def test_sign_in_keeps_what_it_was_not_asked_to_change(store: Path, tmp_path: Path) -> None:
    """A later call setting only the downloads directory must not clear the rest."""
    sign_in(account="someone@example.org", terms_acknowledged=True)
    sign_in(downloads=tmp_path / "Downloads")
    ledger = load_ledger()
    assert ledger.account == "someone@example.org"
    assert ledger.terms_acknowledged is True
    assert ledger.downloads == tmp_path / "Downloads"


def test_a_pasted_secret_offered_as_an_account_label_is_refused(store: Path) -> None:
    """The one place in the round trip where somebody might paste a credential.

    Nothing here ever needs one, so a long opaque string in the account field is
    far more likely to be a token than a nickname, and accepting it would write
    it to disk.
    """
    with pytest.raises(ValueError, match="never needs a password"):
        sign_in(account="ya29.a0AfB_byC3nK9xQ4821zzTokenLookingValue")
    assert load_ledger().account is None


def test_an_ordinary_label_is_not_mistaken_for_a_secret(store: Path) -> None:
    for label in ("someone@example.org", "lab account 2", "jeff", "a-very-long-descriptive-nickname-here"):
        sign_in(account=label)
        assert load_ledger().account == label


def test_the_account_label_is_not_in_the_loggable_record(store: Path) -> None:
    """It is usually an email address, so it stays out of anything a run writes."""
    sign_in(account="someone@example.org")
    record = load_ledger().as_dict()
    assert "someone@example.org" not in str(record)
    assert record["signed_in_recorded"] is True


def test_sign_out_forgets_the_ledger_and_says_whether_there_was_one(store: Path) -> None:
    assert sign_out() is False
    sign_in(account="someone@example.org")
    assert sign_out() is True
    assert load_ledger().account is None


def test_a_submission_is_counted_against_today(store: Path) -> None:
    record_submission(4)
    assert load_ledger().today() == 4
    record_submission(3)
    ledger = load_ledger()
    assert ledger.today() == 7
    assert ledger.remaining() == DAILY_JOB_LIMIT - 7


def test_a_negative_submission_count_is_refused(store: Path) -> None:
    with pytest.raises(ValueError):
        record_submission(-1)


def test_yesterdays_submissions_do_not_spend_todays_budget(store: Path) -> None:
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    save_ledger(Ledger({yesterday: DAILY_JOB_LIMIT}))
    ledger = load_ledger()
    assert ledger.today() == 0
    assert ledger.remaining() == DAILY_JOB_LIMIT


def test_a_drifted_ledger_reports_none_left_rather_than_a_negative_budget(store: Path) -> None:
    """Submitting from a second machine, or a non-UTC reset, drifts the count.

    A negative budget reported as a fact would be worse than reporting none:
    the number is an estimate and a negative one says only that it is wrong.
    """
    record_submission(DAILY_JOB_LIMIT + 5)
    assert load_ledger().remaining() == 0


def test_the_history_is_trimmed_so_the_ledger_is_not_a_log_of_everything(store: Path) -> None:
    old = (datetime.now(UTC).date() - timedelta(days=400)).isoformat()
    recent = (datetime.now(UTC).date() - timedelta(days=2)).isoformat()
    save_ledger(Ledger({old: 5, recent: 2}))
    counts = load_ledger().counts
    assert old not in counts
    assert counts[recent] == 2


class TestSubmissionPlan:
    def test_a_batch_within_budget_all_goes_today(self, store: Path) -> None:
        plan = plan_submission(5)
        assert plan.today == 5 and plan.deferred == 0 and plan.blocker is None

    def test_a_batch_over_budget_is_split_and_the_overflow_is_named(self, store: Path) -> None:
        """Named, because a quietly truncated batch reads afterwards as complete."""
        plan = plan_submission(DAILY_JOB_LIMIT + 10)
        assert plan.today == DAILY_JOB_LIMIT
        assert plan.deferred == 10
        assert plan.blocker == "over_daily_quota"

    def test_an_exhausted_quota_says_so_specifically(self, store: Path) -> None:
        record_submission(DAILY_JOB_LIMIT)
        plan = plan_submission(3)
        assert plan.today == 0 and plan.deferred == 3
        assert plan.blocker == "daily_quota_exhausted"

    def test_a_partly_spent_quota_leaves_the_remainder(self, store: Path) -> None:
        record_submission(DAILY_JOB_LIMIT - 2)
        plan = plan_submission(5)
        assert plan.today == 2 and plan.deferred == 3 and plan.blocker == "over_daily_quota"

    def test_an_empty_batch_plans_nothing_and_blocks_nothing(self, store: Path) -> None:
        plan = plan_submission(0)
        assert plan.today == 0 and plan.deferred == 0 and plan.blocker is None

    def test_a_negative_batch_is_refused(self, store: Path) -> None:
        with pytest.raises(ValueError):
            plan_submission(-1)


class TestDailyAllowance:
    def test_the_published_limit_is_the_default(self, store: Path) -> None:
        ledger = load_ledger()
        assert ledger.daily_limit is None
        assert ledger.limit == DAILY_JOB_LIMIT
        assert ledger.as_dict()["daily_limit_is_the_published_one"] is True

    def test_an_account_with_its_own_allowance_plans_against_that(self, store: Path) -> None:
        """30 is what the FAQ documents for an ordinary account, and it has been
        raised twice already. An account with a different allowance must not be
        told its batch overflows a limit that does not apply to it.
        """
        sign_in(daily_limit=200)
        ledger = load_ledger()
        assert ledger.limit == 200
        assert ledger.as_dict()["daily_limit_is_the_published_one"] is False
        plan = plan_submission(120, ledger)
        assert plan.today == 120 and plan.deferred == 0 and plan.blocker is None

    def test_the_allowance_survives_a_recorded_submission(self, store: Path) -> None:
        """A submission must not quietly reset the account back to the default."""
        sign_in(daily_limit=200)
        record_submission(50)
        ledger = load_ledger()
        assert ledger.limit == 200
        assert ledger.remaining() == 150

    def test_the_allowance_survives_a_later_sign_in(self, store: Path, tmp_path: Path) -> None:
        sign_in(daily_limit=200)
        sign_in(downloads=tmp_path / "dl")
        assert load_ledger().limit == 200

    def test_an_allowance_below_one_is_refused(self, store: Path) -> None:
        """A zero would make every batch read as fully deferred, which is not a thing anyone means."""
        for value in (0, -5):
            with pytest.raises(ValueError, match="at least one job"):
                sign_in(daily_limit=value)
        assert load_ledger().limit == DAILY_JOB_LIMIT

    def test_a_stored_allowance_that_is_unusable_falls_back_to_the_published_one(self, store: Path) -> None:
        """A store hand-edited to nonsense reads as "no allowance set", not as zero."""
        from rfproteina.disease.key_store import save_secret

        for stored in (0, -1, "lots", True, None):
            save_secret("af3_session", {"counts": {}, "daily_limit": stored})
            assert load_ledger().limit == DAILY_JOB_LIMIT


def test_the_submission_url_is_the_service_itself(store: Path) -> None:
    """Pinned so a redirect or a mirror is never substituted silently."""
    assert submission_url() == "https://alphafoldserver.com"
