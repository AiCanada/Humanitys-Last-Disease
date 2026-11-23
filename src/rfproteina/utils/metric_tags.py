"""Decode metric tags recorded in CSVs by current and earlier evaluators."""

import re
from functools import lru_cache


@lru_cache(maxsize=512)
def recorded_metric_tags(cell: str) -> tuple[str, ...]:
    """Return distinct tags in recorded order, accepting comma lists and printed lists."""
    return tuple(dict.fromkeys(re.findall(r"""[^,;\s\[\]'"]+""", cell)))
