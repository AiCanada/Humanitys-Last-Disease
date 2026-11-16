"""Token budgets for the disease-binder flow.

The released processing cap is 511. One percent of 511 is 5.11, which this
module rounds to 5 extra tokens. All five lengthen the complex. Toxicology is
the per-axis combination label on those real tokens. Crop sizes grow by the
same five tokens. Crop size is a data window, not a new weight tensor.
"""

from __future__ import annotations

RELEASED_PROCESSING_CAP = 511
EXTRA_TOKEN_FRACTION = 0.01
EXTRA_TOKENS = 5
CONTEXT_TOKEN_COUNT = EXTRA_TOKENS
DISEASE_PROCESSING_CAP = RELEASED_PROCESSING_CAP + EXTRA_TOKENS

RELEASED_DENOISER_CROP = 384
DISEASE_DENOISER_CROP = RELEASED_DENOISER_CROP + EXTRA_TOKENS

RELEASED_AE_CROP = 256
DISEASE_AE_CROP = RELEASED_AE_CROP + EXTRA_TOKENS

GOOD_CLASS = "high_affinity_clean"
BAD_CLASS = "high_affinity_toxic"
INFERENCE_CLASS = GOOD_CLASS


def check_budget() -> None:
    """Raise if the published 1% budget is no longer internally consistent."""
    expected_extra = round(RELEASED_PROCESSING_CAP * EXTRA_TOKEN_FRACTION)
    if expected_extra != EXTRA_TOKENS:
        raise ValueError(
            f"Extra tokens are {EXTRA_TOKENS}, but 1% of {RELEASED_PROCESSING_CAP} rounds to {expected_extra}."
        )
    if CONTEXT_TOKEN_COUNT != EXTRA_TOKENS:
        raise ValueError("The extra tokens lengthen the complex. The class is not a token slice.")
    if DISEASE_PROCESSING_CAP != RELEASED_PROCESSING_CAP + EXTRA_TOKENS:
        raise ValueError("The disease processing cap must be the released cap plus the extra tokens.")
    if DISEASE_DENOISER_CROP != RELEASED_DENOISER_CROP + EXTRA_TOKENS:
        raise ValueError("The disease denoiser crop must grow by the same extra tokens.")
    if DISEASE_AE_CROP != RELEASED_AE_CROP + EXTRA_TOKENS:
        raise ValueError("The disease autoencoder crop must grow by the same extra tokens.")
