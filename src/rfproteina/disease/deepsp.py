"""Run DeepSP: a sequence-only surrogate for spatial aggregation propensity.

:mod:`rfproteina.disease.developability` could already *read* DeepSP's CSV and
could not produce one. The reason on record was TensorFlow: DeepSP ships three
Keras models and this environment has no TensorFlow, and adding it beside
torch 2.14/cu132 for three small Conv1D stacks is a poor trade.

So the models are run here in torch instead, and the claim that this is the
same computation is checked rather than asserted. DeepSP ships its own
pre-aligned sequences and the descriptors it published for them; the port
reproduces all 480 of those values to a worst relative error of 2.1e-05, which
is the float32 rounding in their CSV. ``tests/test_disease_deepsp.py`` keeps
that comparison.

Two other things the upstream script does that this does differently, both
verified rather than assumed:

* **Numbering.** DeepSP shells out to ``ANARCI --csv``; this project installs
  ANARCII, a different implementation. For the 16 antibodies DeepSP ships,
  ANARCII reproduces ANARCI's 272-column alignment exactly, position for
  position, and produces no IMGT label DeepSP has no slot for.
* **Nothing is written to the working directory.** The upstream script leaves
  ``seq_H.fasta``, ``seq_L.fasta`` and three intermediate CSVs beside itself.
  Here the alignment is a string in memory and the only file is the one a
  caller asks for.

What comes out is DeepSP's own 30 descriptors -- three families over ten
regions -- in DeepSP's own column order, so
:func:`~rfproteina.disease.developability.read_deepsp_table` reads it back.
They are predicted structural descriptors in assay units with no published
pass mark, which is why nothing here compares one to a threshold.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rfproteina.disease.developability import (
    DEEPSP_COLUMNS,
    DEEPSP_FAMILIES,
    DEEPSP_REGIONS,
    NumberingUnavailableError,
    number_chains,
)

# The families and regions come from developability, which owns the column
# contract its reader checks against. One definition, so a writer here and a
# reader there cannot drift into disagreeing about what DeepSP emits.

#: Family -> (architecture json, weights h5), as the files are named in
#: DeepSP's ``DeepSP_models`` directory. Note the inconsistent underscore: the
#: JSON is ``...regressionSAPpos.json`` and the weights
#: ``...regression_SAPpos.h5``.
DEEPSP_WEIGHTS: dict[str, tuple[str, str]] = {
    "SAP_pos": ("Conv1D_regressionSAPpos.json", "Conv1D_regression_SAPpos.h5"),
    "SCM_neg": ("Conv1D_regressionSCMneg.json", "Conv1D_regression_SCMneg.h5"),
    "SCM_pos": ("Conv1D_regressionSCMpos.json", "Conv1D_regression_SCMpos.h5"),
}

#: Where ``scripts/install_deepsp.py`` puts the weights, under ``DATA_PATH``.
#: Named here so the installer and the runner cannot disagree about it.
DEEPSP_SUBDIR = Path("deepsp") / "models"

#: Bumped when the alignment, the encoding or the forward pass changes, so a
#: stored record says which rule produced it.
DEEPSP_METHOD = "deepsp_conv1d_torch_v1"
DEEPSP_SOURCE = "DeepSP (MIT), https://github.com/Lailabcode/DeepSP"

#: One sentence per blocker this module can put on a record.
#:
#: Public because :py:mod:`rfproteina.disease.deepviscosity` emits three of
#: these four names itself and extends this table rather than writing a second
#: copy of each sentence -- two copies of one sentence are two sentences as
#: soon as either is edited.
#:
#: Worth saying plainly: nothing renders these today. ``deepsp_blocker`` and
#: ``viscosity_blocker`` are read by no page and no board -- the screen reaches
#: these descriptors through an oracle table that ``developability.py`` reads,
#: not by running the models -- so this table is the words for when a surface
#: does show them, and the place
#: ``tests/test_disease_board_vocabulary.py`` holds the names to. Putting them
#: on the screening board instead would have added four sentences to a table
#: that never draws them, which is the mistake that test's docstring records
#: costing seventeen duplicated lines once already.
BLOCKER_TEXT = {
    "data_path_unset": (
        "DATA_PATH is not set, so the weights directory could not be located. Set it, or "
        "name the models directory in the call."
    ),
    "deepsp_unavailable": (
        "The DeepSP weights, or the libraries that run them, are not on this machine. "
        "scripts/install_deepsp.py fetches them."
    ),
    "both_chains_required": (
        "These descriptors are computed on an aligned heavy and light pair, so a single "
        "chain cannot be scored. Supply both chains."
    ),
}

#: DeepSP's own one-hot alphabet: twenty amino acids in alphabetical order by
#: one-letter code, then the gap. The channel order is part of the model's
#: input contract, so it is written out rather than derived from a set.
DEEPSP_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
_CHANNEL = {letter: index for index, letter in enumerate(DEEPSP_ALPHABET)}

#: The heavy slot order: 1..111, then the 111 insertions forward, then the 112
#: insertions **backward**, then 113..128. The reversal is the IMGT convention
#: that grows CDR-H3 from both ends, and getting it the other way round would
#: silently shift every CDR-H3 residue.
_H_LABELS: tuple[str, ...] = tuple(
    [str(i) for i in range(1, 112)]
    + [f"111{letter}" for letter in "ABCDEFGH"]
    + [f"112{letter}" for letter in "IHGFEDCBA"]
    + ["112"]
    + [str(i) for i in range(113, 129)]
)
#: The light slot order, a plain 1..127. DeepSP's own position map runs to 128
#: but its inclusion list stops at 127 and its buffer is 127 wide, so 128 is
#: unreachable there and is left out here too.
_L_LABELS: tuple[str, ...] = tuple(str(i) for i in range(1, 128))
_H_INDEX = {label: index for index, label in enumerate(_H_LABELS)}
_L_INDEX = {label: index for index, label in enumerate(_L_LABELS)}

#: 145 heavy slots plus 127 light, which is the models' input width.
DEEPSP_POSITIONS = len(_H_LABELS) + len(_L_LABELS)

#: Chain types ANARCII reports for a light chain. ``K`` is kappa, ``L`` lambda.
_LIGHT_TYPES = frozenset({"K", "L"})
_HEAVY_TYPES = frozenset({"H"})

#: The layer kinds :func:`predict_descriptors` implements. The three models do
#: **not** share an architecture -- SAP_pos is 13 layers with three Dense
#: blocks, SCM_neg is 12 with two, SCM_pos is 14 with three Dropouts and
#: different filter widths -- so pinning one shape rejected two of the three
#: real models. What they share is this vocabulary, and that is what is checked.
_SUPPORTED_LAYERS = frozenset(
    {
        "InputLayer",
        "Conv1D",
        "BatchNormalization",
        "Dropout",
        "MaxPooling1D",
        "Flatten",
        "Dense",
    }
)
#: Activations the forward pass applies. Anything else would be silently
#: skipped, so it is refused.
_SUPPORTED_ACTIVATIONS = frozenset({"relu", "linear", None, ""})


class DeepSPUnavailableError(Exception):
    """DeepSP cannot be run here. Its absence is reported, never worked around."""


@dataclass(frozen=True)
class DeepSPResult:
    """DeepSP's descriptors for a set of antibodies, or why they are missing."""

    #: One row per antibody: ``{"Name": ..., "SAP_pos_CDRH1": ..., ...}``.
    rows: tuple[dict[str, object], ...]
    method: str = DEEPSP_METHOD
    source: str = DEEPSP_SOURCE
    #: Antibodies left out, name -> reason. An antibody ANARCII cannot number
    #: is skipped and named, never scored from a partial alignment.
    skipped: Mapping[str, str] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(str(row["Name"]) for row in self.rows)


def deepsp_columns() -> tuple[str, ...]:
    """The 30 descriptor columns in DeepSP's own order, after ``Name``."""
    return tuple(DEEPSP_COLUMNS)


def default_models_dir() -> Path | None:
    """``DATA_PATH/deepsp/models``, or ``None`` where ``DATA_PATH`` is unset.

    The same place ``scripts/install_deepsp.py`` writes, so a caller that ran
    the installer needs to name no path at all.
    """
    root = os.environ.get("DATA_PATH", "").strip()
    return Path(root) / DEEPSP_SUBDIR if root else None


def deepsp_report(
    sequences: Mapping[str, str],
    models: Path | None = None,
    *,
    name: str = "design",
    cpu: bool = True,
    ncpu: int = 1,
) -> dict[str, object]:
    """One JSON-shaped record of DeepSP's descriptors for one antibody.

    The shape :mod:`rfproteina.disease.developability`'s own reports use: a
    method, its provenance, and a named blocker instead of a result when
    something is missing. Nothing raises -- a caller printing a record wants a
    record even when the answer is "not here", and which thing is missing is
    the useful part.

    ``sequences`` is keyed by chain name as
    :func:`~rfproteina.disease.developability.number_chains` takes it; the
    heavy and light chains are whichever ANARCII calls heavy and light, so
    ``{"H": ..., "L": ...}`` and ``{"alpha": ..., "beta": ...}`` both work.

    No pass cut, for the reason the oracle reader gives: these are predicted
    structural descriptors in assay units with no published pass mark, and a
    surrogate for Rosetta SAP is not Rosetta SAP. The screen's
    ``developability.pass`` still cannot be claimed from this.
    """
    directory = Path(models) if models is not None else default_models_dir()
    if directory is None:
        return {"deepsp_blocker": "data_path_unset", "method": DEEPSP_METHOD}
    available, reason = deepsp_available(directory)
    if not available:
        return {"deepsp_blocker": reason or "deepsp_unavailable", "method": DEEPSP_METHOD}
    try:
        chains = number_chains(dict(sequences), cpu=cpu, ncpu=ncpu)
    except NumberingUnavailableError as error:
        return {"deepsp_blocker": str(error), "method": DEEPSP_METHOD}

    heavy = next((chain for chain in chains if chain.chain_type in _HEAVY_TYPES), None)
    light = next((chain for chain in chains if chain.chain_type in _LIGHT_TYPES), None)
    if heavy is None or light is None:
        return {
            "deepsp_blocker": "both_chains_required",
            "method": DEEPSP_METHOD,
            "chain_types": sorted({chain.chain_type for chain in chains}),
        }
    aligned = aligned_sequence(heavy, light)
    batch = _stack([aligned])
    blocks = {
        family: predict_descriptors(batch, directory / DEEPSP_WEIGHTS[family][0], directory / DEEPSP_WEIGHTS[family][1])
        for family in DEEPSP_FAMILIES
    }
    descriptors = {
        f"{family}_{region}": float(blocks[family][0, index])
        for family in DEEPSP_FAMILIES
        for index, region in enumerate(DEEPSP_REGIONS)
    }
    return {
        "method": DEEPSP_METHOD,
        "source": DEEPSP_SOURCE,
        "name": name,
        "chains": {chain.chain_type: chain.name for chain in chains},
        "aligned_positions": len(aligned),
        "occupied_positions": len(aligned) - aligned.count("-"),
        "descriptors": descriptors,
        "regions": list(DEEPSP_REGIONS),
        "families": list(DEEPSP_FAMILIES),
        "pass_cut": None,
        "pass_cut_reason": "predicted_descriptors_in_assay_units_with_no_published_pass_mark",
        "caveat": (
            "a sequence surrogate for the structure-derived Rosetta SAP, not a measurement of it; "
            "developability.pass cannot be claimed from this"
        ),
    }


def deepsp_available(models: Path) -> tuple[bool, str | None]:
    """Whether the models and the libraries to run them are both here."""
    try:
        import h5py  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment dependent
        return False, f"deepsp_dependency_missing:{error.name or error}"
    missing = [name for family in DEEPSP_FAMILIES for name in DEEPSP_WEIGHTS[family] if not (models / name).is_file()]
    if missing:
        return False, f"deepsp_models_absent:{','.join(sorted(missing))}"
    return True, None


def aligned_sequence(heavy: Any, light: Any) -> str:
    """One antibody's 272-column DeepSP alignment from two numbered chains.

    An unoccupied slot is a gap, which is what the models were trained on. A
    residue whose IMGT label DeepSP has no slot for is **dropped**, because the
    alternative is shifting every later residue by one and scoring a sequence
    nobody has: DeepSP's own preprocessing drops them the same way, by only
    copying the columns on its inclusion list.
    """
    slots = ["-"] * len(_H_LABELS) + ["-"] * len(_L_LABELS)
    offset = len(_H_LABELS)
    for chain, index, base in ((heavy, _H_INDEX, 0), (light, _L_INDEX, offset)):
        if chain is None:
            continue
        for residue in chain.residues:
            label = f"{residue.position}{residue.insertion.strip()}"
            at = index.get(label)
            if at is None:
                continue
            slots[base + at] = residue.residue
    return "".join(slots)


def one_hot(aligned: str) -> Any:
    """``(272, 21)`` float32, channel order fixed by :data:`DEEPSP_ALPHABET`."""
    import numpy as np

    if len(aligned) != DEEPSP_POSITIONS:
        raise DeepSPUnavailableError(f"alignment_width:{len(aligned)}!={DEEPSP_POSITIONS}")
    out = np.zeros((DEEPSP_POSITIONS, len(DEEPSP_ALPHABET)), dtype=np.float32)
    for position, letter in enumerate(aligned):
        channel = _CHANNEL.get(letter.upper())
        if channel is None:
            # An unknown letter is a gap rather than a refusal: ANARCII can
            # report X for an ambiguous residue, and a gap is what DeepSP's
            # own encoder would have produced for anything outside its alphabet.
            channel = _CHANNEL["-"]
        out[position, channel] = 1.0
    return out


def _stack(aligned: Sequence[str]) -> Any:
    """``(n, 272, 21)`` for a list of alignments, in the order given."""
    import numpy as np

    return np.stack([one_hot(item) for item in aligned])


def keras_weights(path: Path) -> dict[str, dict[str, Any]]:
    """Keras weights as ``layer -> weight name -> array``, in file order.

    Public because :mod:`rfproteina.disease.deepviscosity` reads the same
    layout: both tools come from the same lab and save with Keras 2.11, so one
    reader serves both rather than two that could drift apart. The h5 grouping
    is a Keras convention, not a DeepSP detail.
    """
    import h5py
    import numpy as np

    out: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        # Saved with ``save_weights``, so the layer groups sit at the root. A
        # whole-model save puts them under ``model_weights``; both are read.
        group = handle.get("model_weights", handle)
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in group.attrs["layer_names"]]
        for layer in names:
            inner = group[layer]
            weight_names = [n.decode() if isinstance(n, bytes) else str(n) for n in inner.attrs.get("weight_names", [])]
            out[layer] = {name.rsplit("/", 1)[-1]: np.array(inner[name]) for name in weight_names}
    return out


def _architecture(path: Path) -> list[dict[str, Any]]:
    """The Keras layer configs, checked against the architecture we implement."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DeepSPUnavailableError(f"deepsp_architecture_unreadable:{type(error).__name__}") from error
    layers = (config.get("config") or {}).get("layers") or []
    if not layers:
        raise DeepSPUnavailableError(f"deepsp_architecture_empty:{path.name}")
    flattens = 0
    for layer in layers:
        kind, settings = layer.get("class_name", ""), layer.get("config") or {}
        if kind not in _SUPPORTED_LAYERS:
            raise DeepSPUnavailableError(f"deepsp_layer_unsupported:{path.name}:{kind}")
        if settings.get("activation") not in _SUPPORTED_ACTIVATIONS:
            raise DeepSPUnavailableError(
                f"deepsp_activation_unsupported:{path.name}:{kind}:{settings.get('activation')}"
            )
        # The forward pass convolves and pools without padding and at unit
        # stride. A release that changed either would still run and return
        # numbers for the wrong receptive field, so both are checked.
        if kind in {"Conv1D", "MaxPooling1D"} and settings.get("padding", "valid") != "valid":
            raise DeepSPUnavailableError(f"deepsp_padding_unsupported:{path.name}:{settings.get('padding')}")
        if kind == "Conv1D" and tuple(settings.get("strides") or (1,)) != (1,):
            raise DeepSPUnavailableError(f"deepsp_stride_unsupported:{path.name}:{settings.get('strides')}")
        flattens += kind == "Flatten"
    if flattens != 1:
        # The channels-last transpose happens at the Flatten. Two of them, or
        # none, means the tensor reaching Dense is not laid out as Keras left it.
        raise DeepSPUnavailableError(f"deepsp_flatten_count:{path.name}:{flattens}")
    return layers


def predict_descriptors(batch: Any, architecture: Path, weights: Path) -> Any:
    """Run one DeepSP model over a ``(n, 272, 21)`` batch, in torch.

    A faithful port of the Keras graph, and two details carry the whole result:
    a ``Conv1D`` applies its activation **before** the ``BatchNormalization``
    that follows it, and Keras ``Flatten`` flattens channels-last, so the
    tensor is transposed back before it reaches ``Dense_1``. Either one the
    other way round gives plausible numbers that are wrong.
    """
    import torch
    import torch.nn.functional as functional

    layers = _architecture(architecture)
    store = keras_weights(weights)
    tensor = torch.from_numpy(batch).float().transpose(1, 2)
    for layer in layers:
        kind, config = layer["class_name"], layer["config"]
        name = config.get("name", "")
        if kind == "InputLayer" or kind == "Dropout":
            continue
        if kind == "Conv1D":
            held = store[name]
            tensor = functional.conv1d(
                tensor,
                torch.from_numpy(held["kernel:0"]).float().permute(2, 1, 0),
                torch.from_numpy(held["bias:0"]).float(),
            )
            if config.get("activation") == "relu":
                tensor = torch.relu(tensor)
        elif kind == "BatchNormalization":
            held = store[name]
            view = (1, -1, 1)
            centred = tensor - torch.from_numpy(held["moving_mean:0"]).float().view(*view)
            spread = torch.sqrt(
                torch.from_numpy(held["moving_variance:0"]).float().view(*view) + float(config.get("epsilon", 1e-3))
            )
            tensor = centred / spread * torch.from_numpy(held["gamma:0"]).float().view(*view) + torch.from_numpy(
                held["beta:0"]
            ).float().view(*view)
        elif kind == "MaxPooling1D":
            tensor = functional.max_pool1d(tensor, config["pool_size"][0], config["strides"][0])
        elif kind == "Flatten":
            tensor = tensor.transpose(1, 2).reshape(tensor.shape[0], -1)
        elif kind == "Dense":
            held = store[name]
            tensor = tensor @ torch.from_numpy(held["kernel:0"]).float() + torch.from_numpy(held["bias:0"]).float()
            if config.get("activation") == "relu":
                tensor = torch.relu(tensor)
        else:  # pragma: no cover - _architecture refuses anything else first
            raise DeepSPUnavailableError(f"deepsp_layer_unsupported:{kind}")
    return tensor.detach().numpy()


def run_deepsp(
    antibodies: Mapping[str, tuple[str, str]],
    models: Path,
    *,
    cpu: bool = True,
    ncpu: int = 1,
) -> DeepSPResult:
    """DeepSP's 30 descriptors for each named ``(heavy, light)`` pair.

    Both chains are required. DeepSP was trained on a heavy-plus-light Fv and
    its light half is 127 of the 272 input columns, so scoring a heavy chain
    alone would feed the model 127 gaps and return a number for an antibody
    that does not exist. A pair missing either chain is skipped and named.

    Raises :class:`DeepSPUnavailableError` when the models or the libraries to
    run them are absent, and :class:`~.developability.NumberingUnavailableError`
    when ANARCII is.
    """
    # The inputs are judged before the models are looked for: a caller who
    # passed nothing scoreable should be told that whether or not the weights
    # happen to be on this machine.
    skipped: dict[str, str] = {}
    wanted: dict[str, tuple[str, str]] = {}
    for name, pair in antibodies.items():
        heavy, light = (pair[0] or "").strip(), (pair[1] or "").strip()
        if not heavy or not light:
            skipped[name] = "both_chains_required"
            continue
        wanted[name] = (heavy, light)
    if not wanted:
        return DeepSPResult((), skipped=skipped)

    available, reason = deepsp_available(models)
    if not available:
        raise DeepSPUnavailableError(reason or "deepsp_unavailable")

    heavy_chains = _numbered(wanted, 0, cpu, ncpu)
    light_chains = _numbered(wanted, 1, cpu, ncpu)

    aligned: dict[str, str] = {}
    for name in wanted:
        heavy, light = heavy_chains.get(name), light_chains.get(name)
        if heavy is None or heavy.chain_type not in _HEAVY_TYPES:
            skipped[name] = f"heavy_chain_not_numbered_as_heavy:{getattr(heavy, 'chain_type', 'none')}"
            continue
        if light is None or light.chain_type not in _LIGHT_TYPES:
            skipped[name] = f"light_chain_not_numbered_as_light:{getattr(light, 'chain_type', 'none')}"
            continue
        aligned[name] = aligned_sequence(heavy, light)
    if not aligned:
        return DeepSPResult((), skipped=skipped)

    order = list(aligned)
    batch = _stack([aligned[name] for name in order])
    blocks = {
        family: predict_descriptors(batch, models / DEEPSP_WEIGHTS[family][0], models / DEEPSP_WEIGHTS[family][1])
        for family in DEEPSP_FAMILIES
    }
    for family, matrix in blocks.items():
        if matrix.shape != (len(order), len(DEEPSP_REGIONS)):
            raise DeepSPUnavailableError(f"deepsp_output_shape:{family}:{matrix.shape}")

    rows: list[dict[str, object]] = []
    for position, name in enumerate(order):
        row: dict[str, object] = {"Name": name}
        for family in DEEPSP_FAMILIES:
            for index, region in enumerate(DEEPSP_REGIONS):
                row[f"{family}_{region}"] = float(blocks[family][position, index])
        rows.append(row)
    return DeepSPResult(tuple(rows), skipped=skipped)


def _numbered(wanted: Mapping[str, tuple[str, str]], half: int, cpu: bool, ncpu: int) -> dict[str, Any]:
    """One half of each pair, numbered, keyed by antibody name."""
    sequences = {name: pair[half] for name, pair in wanted.items()}
    try:
        chains = number_chains(sequences, cpu=cpu, ncpu=ncpu)
    except NumberingUnavailableError:
        raise
    return {chain.name: chain for chain in chains}


def write_deepsp_csv(result: DeepSPResult, path: Path) -> Path:
    """Write DeepSP's descriptors where :func:`read_deepsp_table` will read them.

    DeepSP's own column spelling and order, so the file is interchangeable with
    one the upstream script produced -- including the capital ``Name``, which is
    what it writes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("Name", *deepsp_columns())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in result.rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def descriptors_for_sequences(
    antibodies: Mapping[str, Sequence[str]],
    models: Path,
    destination: Path,
    *,
    cpu: bool = True,
    ncpu: int = 1,
) -> tuple[Path | None, DeepSPResult]:
    """Number, align, score and write, in one call. ``None`` when nothing scored."""
    pairs = {name: (*tuple(chains), "", "")[:2] for name, chains in antibodies.items()}
    result = run_deepsp(pairs, models, cpu=cpu, ncpu=ncpu)
    if not result.rows:
        return None, result
    return write_deepsp_csv(result, destination), result
