# Inference architecture adapted from proteininnovation/delphi at
# c3d20122c86ad4c3f8142a4ae571bcf5b9912ded, models/transformer_onehot.py.
#
# MIT License
# Copyright (c) 2026 Institute for Protein Innovation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""The frozen CPU architecture of the DELPHI v4 paired one-hot PSR classifier."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from pickle import UnpicklingError

import torch
from torch import nn

from rfproteina.disease.delphi_psr import AMINO_ACIDS, DelPhiError, verify_checkpoint

_HIDDEN = 128
_HEADS = 8
_LAYERS = 4
_FEEDFORWARD = 512
_DROPOUT = 0.38
_VH_LEN = _VL_LEN = 135
_CDR3_LEN = 25
_MODEL_CONFIG = {
    "hidden_dim": _HIDDEN, "num_heads": _HEADS, "num_layers": _LAYERS,
    "dim_feedforward": _FEEDFORWARD, "dropout": _DROPOUT,
}
_LENGTH_CONFIG = {"max_vh_len": _VH_LEN, "max_vl_len": _VL_LEN, "max_hcdr3_len": _CDR3_LEN}
_TOKENS = {letter: index for index, letter in enumerate(AMINO_ACIDS)}


class _Positions(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The released buffers replace this placeholder through strict loading.
        self.register_buffer("pe", torch.zeros(1, 512, _HIDDEN))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.pe[:, :tokens.shape[1]]


def _encoder() -> nn.TransformerEncoder:
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            _HIDDEN, _HEADS, _FEEDFORWARD, _DROPOUT, batch_first=True, norm_first=True,
        ),
        num_layers=_LAYERS,
        enable_nested_tensor=False,
    )


class _PSRClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_fc = nn.Linear(len(AMINO_ACIDS), _HIDDEN)
        self.cdr3_fc = nn.Linear(len(AMINO_ACIDS), _HIDDEN)
        self.pos_enc = _Positions()
        self.cdr3_pos_enc = _Positions()
        self.transformer = _encoder()
        self.cdr3_transformer = _encoder()
        self.attention = nn.MultiheadAttention(_HIDDEN, _HEADS, dropout=_DROPOUT, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * _HIDDEN), nn.Dropout(_DROPOUT), nn.Linear(2 * _HIDDEN, 2))

    def forward(self, paired: torch.Tensor, cdr3: torch.Tensor) -> torch.Tensor:
        pair_mask = paired.sum(dim=-1) == 0
        cdr3_mask = cdr3.sum(dim=-1) == 0
        pair_features = self.transformer(self.pos_enc(self.input_fc(paired)), src_key_padding_mask=pair_mask)
        loop_features = self.cdr3_transformer(
            self.cdr3_pos_enc(self.cdr3_fc(cdr3)), src_key_padding_mask=cdr3_mask,
        )
        # Match the released model exactly. Attention masks padded keys, but
        # both branch means include the outputs at padded query positions.
        query = loop_features.mean(dim=1, keepdim=True)
        context, _ = self.attention(query, pair_features, pair_features, key_padding_mask=pair_mask)
        return self.head(torch.cat((pair_features.mean(dim=1), context[:, 0]), dim=1))


def _encode(sequences: Sequence[str], width: int) -> torch.Tensor:
    result = torch.zeros(len(sequences), width, len(AMINO_ACIDS), dtype=torch.float32, device="cpu")
    for row, sequence in enumerate(sequences):
        if not 0 < len(sequence) <= width or any(residue not in _TOKENS for residue in sequence):
            raise DelPhiError("delphi_inference_failed")
        result[row, list(range(len(sequence))), [_TOKENS[residue] for residue in sequence]] = 1.0
    return result


def load_model(raw: bytes) -> _PSRClassifier:
    """Never load another pickle or build a model from unverified configuration."""
    verify_checkpoint(raw)
    try:
        payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("lm_mode") != "onehot":
            raise ValueError("wrong embedding mode")
        config = payload.get("config")
        if (
            not isinstance(config, dict) or config.get("model") != _MODEL_CONFIG
            or config.get("sequence_lengths") != _LENGTH_CONFIG
        ):
            raise ValueError("unexpected model contract")
        # Tensor type/dtype checks prevent load_state_dict's implicit casting.
        model = _PSRClassifier()
        state = payload["state_dict"]
        expected = model.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys() or any(
            not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != expected[key].dtype
            or value.shape != expected[key].shape for key, value in state.items()
        ):
            raise ValueError("unexpected state tensors")
        model.load_state_dict(state, strict=True)
        return model.eval()
    except (ValueError, KeyError, TypeError, RuntimeError, OSError, EOFError, UnpicklingError) as error:
        raise DelPhiError("delphi_checkpoint_invalid") from error


def predict(model: _PSRClassifier, rows: Sequence[tuple[str, str, str]]) -> list[float]:
    """Score the assay pass class (index 1), never an inverted failure probability."""
    vh, vl, cdr3 = zip(*rows, strict=True)
    paired = torch.cat((_encode(vh, _VH_LEN), _encode(vl, _VL_LEN)), dim=1)
    with torch.inference_mode():
        logits = model(paired, _encode(cdr3, _CDR3_LEN))
        if logits.shape != (len(rows), 2):
            raise DelPhiError("delphi_inference_failed")
        return torch.softmax(logits, dim=1)[:, 1].tolist()
