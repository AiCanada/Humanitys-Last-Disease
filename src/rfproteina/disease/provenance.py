"""Which code produced a record.

A verdict is reproducible against four things: its inputs, the cuts it was
taken against, the bar it had to clear, and the code that applied them. The
screen records the first three. It did not record the fourth, so two screens
agreeing on every stored field could still disagree about the designs, and
nothing on disk would say why.

That gap became visible the moment ``replaced`` learned to name which supplied
files had moved. A re-screen that changes the verdict while reporting no
changed inputs is pointing at the code, and until now the record could not say
which code it was pointing at.

The commit is read out of ``.git`` without running git, which keeps this usable
from a screen that must not start a subprocess. ``tree_state`` is
``not_checked`` and says so: telling a clean tree from a dirty one means asking
git, and a commit recorded as though it described the working tree would be
worse than one that admits it is only the base that tree sat on.

Lives in its own module because both ``trace.py`` and ``screen.py`` need it and
``trace.py`` imports ``screen.py``. Writing it twice is how the two records
would come to disagree about the same checkout.
"""

from __future__ import annotations

import re
from pathlib import Path

from rfproteina.disease.publish import _repository_root

#: What a field records when the answer could not be read. The same word the
#: traces use, so one reader can recognise it in either record.
NOT_RECORDED = "not_recorded"
#: A commit as the ref files spell one: SHA-1, or SHA-256 for a repository
#: using the newer object format.
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def code_version() -> dict[str, str]:
    """The commit this checkout sits on, read from ``.git`` without running git.

    AGENTS.md asks every record to carry code identity, and the checkpoint does
    not supply it: the sampler, the transforms and the condition parsing that
    turned those weights into a design are the repository's, not the
    checkpoint's. Neither are the cuts, the bar, or the table readers that
    decide a screen.

    A detached head and a packed ref both answer. A checkout with no ``.git``
    at all -- an installed copy, a container image -- answers ``not_recorded``
    rather than inventing a commit.
    """
    root = _repository_root()
    if root is None:
        return {"commit": NOT_RECORDED, "tree_state": "not_checked"}
    return {"commit": head_commit(root / ".git") or NOT_RECORDED, "tree_state": "not_checked"}


def head_commit(git: Path) -> str:
    """The commit ``HEAD`` names, following one ref and the packed table."""
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""
    if not head.startswith("ref:"):
        return head if _COMMIT.fullmatch(head) else ""
    ref = head.partition(":")[2].strip()
    if not ref:
        return ""
    try:
        direct = (git / ref).read_text(encoding="utf-8").strip()
        if _COMMIT.fullmatch(direct):
            return direct
    except (OSError, UnicodeError):
        pass
    try:
        packed = (git / "packed-refs").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""
    for line in packed.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref and _COMMIT.fullmatch(parts[0]):
            return parts[0]
    return ""
