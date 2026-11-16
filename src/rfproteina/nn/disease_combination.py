"""Zero-initialized projection of a binder-combination label onto conditioning.

The released denoiser and autoencoder do not contain this module. Training
configs build it only when ``disease_combination`` is true, and weight loading
zero-initializes it so the released weights stay identical until the layer is
trained. A zero projection adds nothing. At evaluation, an all-zero label on a
real token is the requested-clear condition for the configured binder kind.
Training leaves zeros untouched. Padding tokens are masked. This is not a
search reward and it does not score sequences.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from rfproteina.disease.combinations import CLASS_CONDITION_DIM, combination_condition, requested_clear

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", "none", "null", ""}


def combination_enabled(value: object) -> bool:
    """Read a config flag. The string 'false' is false."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError(f"disease_combination must be true or false, got {value!r}.")
    if value is None:
        return False
    return bool(value)


def disease_request_kind(options: Mapping[str, object], default: str = "protein") -> str:
    """Binder kind used when an unlabeled evaluation batch asks for a clear combination."""
    kind = options.get("disease_binder_kind", default)
    if kind is None or str(kind).strip() == "":
        return default
    return str(kind)


def take_disease_options(options: dict[str, object]) -> tuple[object, str]:
    """Remove disease keys before they are forwarded into feature constructors."""
    enabled = options.pop("disease_combination", False)
    kind = disease_request_kind(options)
    options.pop("disease_binder_kind", None)
    return enabled, kind


def build_disease_combination(
    enabled: object,
    dim_cond: int,
    request_kind: str = "protein",
) -> CombinationProjection | None:
    """Build the projection, or return ``None`` when the flag is off."""
    if not combination_enabled(enabled):
        return None
    return CombinationProjection(dim_cond, request_kind=request_kind)


class CombinationProjection(nn.Module):
    """Add ``Linear(44, dim_cond)`` of the combination label onto ``c``.

    Weight and bias start at zero. ``clear_request`` is not part of the
    state dict, so a release checkpoint does not need a matching key.
    """

    def __init__(self, dim_cond: int, request_kind: str = "protein") -> None:
        super().__init__()
        self.proj = nn.Linear(CLASS_CONDITION_DIM, dim_cond)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        row = combination_condition(1, requested_clear(request_kind))[0]
        clear_request = torch.tensor(row, dtype=torch.float32)
        self.register_buffer("clear_request", clear_request, persistent=False)
        self.request_kind = request_kind

    def forward(
        self,
        conditioning: torch.Tensor,
        combination: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project the label and add it on valid tokens."""
        if combination is None:
            features = conditioning.new_zeros(conditioning.shape[0], conditioning.shape[1], CLASS_CONDITION_DIM)
        else:
            features = _as_combination(combination, conditioning)
        if not self.training:
            features = _fill_unlabeled_rows(features, mask, self.clear_request)
        gate = mask.to(device=conditioning.device, dtype=conditioning.dtype).unsqueeze(-1)
        return conditioning + self.proj(features) * gate


def add_disease_combination(
    projection: CombinationProjection | None,
    conditioning: torch.Tensor,
    combination: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply ``projection`` when the module was built. Otherwise return ``conditioning``."""
    if projection is None:
        return conditioning
    return projection(conditioning, combination, mask)


def _as_combination(combination: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    expected = (conditioning.shape[0], conditioning.shape[1], CLASS_CONDITION_DIM)
    if tuple(combination.shape) != expected:
        raise ValueError(f"disease_combination must have shape {expected}, got {tuple(combination.shape)}.")
    return combination.to(device=conditioning.device, dtype=conditioning.dtype)


def _fill_unlabeled_rows(
    features: torch.Tensor,
    mask: torch.Tensor,
    clear_request: torch.Tensor,
) -> torch.Tensor:
    """Fill an entirely unlabeled example's real tokens with the requested-clear vector.

    The test is per example, not per token. An example whose valid tokens are all
    zero carries no label, so each of its real tokens takes the request. An example
    with any nonzero row is already labeled and is left untouched.
    """
    valid = mask.to(device=features.device, dtype=torch.bool)
    if valid.shape[:2] != features.shape[:2]:
        raise ValueError(f"The combination mask must have shape {tuple(features.shape[:2])}, got {tuple(valid.shape)}.")
    present = valid.to(dtype=features.dtype)
    row_magnitude = (features.abs().sum(dim=-1) * present).sum(dim=-1)
    needs = valid.any(dim=-1) & (row_magnitude == 0)
    clear = clear_request.to(device=features.device, dtype=features.dtype)
    replace = (needs.unsqueeze(-1) & valid).unsqueeze(-1)
    return torch.where(replace, clear, features)
