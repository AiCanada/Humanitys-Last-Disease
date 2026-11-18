"""Tests for rfproteina.disease.deepsp (CPU-only, fast, no model files needed).

The arithmetic is checked against DeepSP itself where the models are available.
``scripts/install_deepsp.py`` fetches everything that needs -- the six model
files plus ``DeepSP_input.csv`` and ``DeepSP_descriptors_anarci2.csv`` -- so a
normal install is enough, and ``RFD4_DEEPSP_MODELS`` still overrides the
directory. The last test then reproduces all 480 published descriptors.
Without them, everything else here still runs: the alignment, the encoding, the
contract and the refusals need no weights.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from rfproteina.disease import deepsp as ds
from rfproteina.disease.developability import DEEPSP_COLUMNS, Residue, read_deepsp_table

#: DeepSP's example set, which the comparison below needs beside the weights.
_REFERENCE = ("DeepSP_input.csv", "DeepSP_descriptors_anarci2.csv")


def _models_dir() -> Path | None:
    """``RFD4_DEEPSP_MODELS`` when set, else wherever the installer wrote.

    The env var used to be the only route, which meant the one test that makes
    the port defensible skipped on a machine that had actually installed
    DeepSP. Falling back to the installer's own directory makes a normal
    install enough to run it.
    """
    named = os.environ.get("RFD4_DEEPSP_MODELS", "").strip()
    if named:
        return Path(named)
    return ds.default_models_dir()


def _reference_present() -> bool:
    directory = _models_dir()
    if directory is None or not directory.is_dir():
        return False
    return all((directory / name).is_file() for name in _REFERENCE)


needs_models = pytest.mark.skipif(
    not _reference_present(),
    reason=(
        "run scripts/install_deepsp.py, or set RFD4_DEEPSP_MODELS to a directory holding DeepSP's "
        "models plus DeepSP_input.csv and DeepSP_descriptors_anarci2.csv, to check the port against "
        "its own numbers"
    ),
)


def _residue(chain_type: str, position: int, insertion: str, letter: str) -> Residue:
    return Residue(chain="x", chain_type=chain_type, position=position, insertion=insertion, residue=letter)


class _Chain:
    """The only thing aligned_sequence wants from a NumberedChain."""

    def __init__(self, chain_type: str, residues: list[Residue]) -> None:
        self.chain_type = chain_type
        self.residues = tuple(residues)


# ------------------------------------------------------------------ the contract


def test_the_column_contract_is_thirty_and_matches_the_reader():
    """DeepSP writes three families over ten regions, not three columns."""
    assert len(DEEPSP_COLUMNS) == 30
    assert ds.deepsp_columns() == DEEPSP_COLUMNS
    assert DEEPSP_COLUMNS[0] == "SAP_pos_CDRH1"
    assert DEEPSP_COLUMNS[-1] == "SCM_pos_Fv"
    # Family-major, which is the order deepsp_predictor.py concatenates its
    # three prediction blocks in. Region-major would transpose the file.
    assert DEEPSP_COLUMNS[:10] == tuple(f"SAP_pos_{region}" for region in ds.DEEPSP_REGIONS)
    assert DEEPSP_COLUMNS[10:20] == tuple(f"SCM_neg_{region}" for region in ds.DEEPSP_REGIONS)


def test_real_deepsp_output_is_readable_by_the_project_reader(tmp_path):
    """The regression this contract exists for: real output used to be refused."""
    path = tmp_path / "DeepSP_descriptors.csv"
    row = {"Name": "mAb1"} | {column: "1.5" for column in DEEPSP_COLUMNS}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Name", *DEEPSP_COLUMNS], lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    record = read_deepsp_table(path)
    assert "oracle_blocker" not in record, record
    assert len(record["properties"]) == 30
    assert record["rows"][0]["name"] == "mAb1", "DeepSP writes a capital Name; the row must keep its identity"


# ------------------------------------------------------------------ the alignment


def test_the_slot_layout_is_deepsps_own():
    assert ds.DEEPSP_POSITIONS == 272
    assert len(ds._H_LABELS) == 145
    assert len(ds._L_LABELS) == 127
    # The 111 insertions run forward and the 112 insertions run backward, which
    # is how IMGT grows CDR-H3 from both ends. Reversed, every CDR-H3 residue
    # lands in the wrong slot and the model still returns a number.
    assert ds._H_LABELS[110:120] == ("111", "111A", "111B", "111C", "111D", "111E", "111F", "111G", "111H", "112I")
    assert ds._H_LABELS[120:129] == ("112H", "112G", "112F", "112E", "112D", "112C", "112B", "112A", "112")
    assert ds._L_LABELS[0] == "1" and ds._L_LABELS[-1] == "127"


def test_alignment_places_residues_by_imgt_label():
    heavy = _Chain("H", [_residue("H", 1, "", "E"), _residue("H", 111, "A", "W"), _residue("H", 128, "", "S")])
    light = _Chain("K", [_residue("K", 1, "", "D"), _residue("K", 127, "", "K")])
    aligned = ds.aligned_sequence(heavy, light)
    assert len(aligned) == 272
    assert aligned[0] == "E"
    assert aligned[ds._H_LABELS.index("111A")] == "W"
    assert aligned[144] == "S", "heavy 128 is the last heavy slot"
    assert aligned[145] == "D", "light 1 is the first light slot"
    assert aligned[271] == "K"
    assert aligned.count("-") == 272 - 5


def test_alignment_drops_a_label_deepsp_has_no_slot_for():
    """Shifting the rest along would score a sequence nobody has."""
    heavy = _Chain("H", [_residue("H", 1, "", "E"), _residue("H", 129, "", "Z"), _residue("H", 2, "", "V")])
    aligned = ds.aligned_sequence(heavy, None)
    assert aligned[0] == "E"
    assert aligned[1] == "V", "the residue after the dropped one keeps its own slot"
    assert "Z" not in aligned


def test_alignment_of_nothing_is_all_gaps():
    assert ds.aligned_sequence(None, None) == "-" * 272


# ------------------------------------------------------------------ the encoding


def test_one_hot_channel_order_is_part_of_the_contract():
    assert ds.DEEPSP_ALPHABET == "ACDEFGHIKLMNPQRSTVWY-"
    encoded = ds.one_hot("A" + "-" * 271)
    assert encoded.shape == (272, 21)
    assert encoded[0, 0] == 1.0
    assert encoded[1, 20] == 1.0, "the gap is the last channel"
    assert encoded.sum() == 272.0, "exactly one channel per position"


def test_one_hot_treats_an_unknown_letter_as_a_gap():
    """ANARCII can report X. DeepSP's own encoder has no channel for it either."""
    encoded = ds.one_hot("X" + "-" * 271)
    assert encoded[0, 20] == 1.0


def test_one_hot_refuses_a_width_the_models_cannot_take():
    with pytest.raises(ds.DeepSPUnavailableError, match="alignment_width"):
        ds.one_hot("ACDE")


# ------------------------------------------------------------------ the refusals


def test_missing_models_are_named(tmp_path):
    available, reason = ds.deepsp_available(tmp_path)
    assert not available
    assert "deepsp_models_absent" in (reason or "")
    assert "Conv1D_regression_SAPpos.h5" in (reason or "")


def test_an_antibody_missing_a_chain_is_skipped_and_named(tmp_path):
    """DeepSP is an Fv model: 127 of its 272 input columns are the light chain.

    Scoring a lone heavy chain would feed it 127 gaps and return a number for
    an antibody that does not exist, so the pair is skipped and named. Judged
    before the models are looked for, which is why this needs no weights.
    """
    result = ds.run_deepsp({"heavy_only": ("EVQL", ""), "light_only": ("", "DIQM")}, tmp_path)
    assert result.rows == ()
    assert result.skipped == {"heavy_only": "both_chains_required", "light_only": "both_chains_required"}


def test_a_scoreable_pair_without_models_is_refused_by_name(tmp_path):
    with pytest.raises(ds.DeepSPUnavailableError, match="deepsp_models_absent"):
        ds.run_deepsp({"ab": ("EVQL", "DIQM")}, tmp_path)


def _architecture_file(tmp_path, layers):
    path = tmp_path / "arch.json"
    path.write_text(json.dumps({"config": {"layers": layers}}), encoding="utf-8")
    return path


def _layer(kind, **config):
    return {"class_name": kind, "config": {"name": kind.lower(), **config}}


def test_an_architecture_this_cannot_run_is_refused_rather_than_approximated(tmp_path):
    good = [
        _layer("InputLayer"),
        _layer("Conv1D", activation="relu", padding="valid", strides=[1]),
        _layer("Flatten"),
        _layer("Dense", activation="linear"),
    ]
    assert len(ds._architecture(_architecture_file(tmp_path, good))) == 4

    with pytest.raises(ds.DeepSPUnavailableError, match="deepsp_architecture_empty"):
        ds._architecture(_architecture_file(tmp_path, []))
    with pytest.raises(ds.DeepSPUnavailableError, match="deepsp_layer_unsupported"):
        ds._architecture(_architecture_file(tmp_path, [*good, _layer("LSTM")]))
    with pytest.raises(ds.DeepSPUnavailableError, match="deepsp_activation_unsupported"):
        ds._architecture(_architecture_file(tmp_path, [*good, _layer("Dense", activation="tanh")]))
    # padding="same" would still run and score the wrong receptive field.
    with pytest.raises(ds.DeepSPUnavailableError, match="deepsp_padding_unsupported"):
        ds._architecture(_architecture_file(tmp_path, [_layer("Conv1D", padding="same"), _layer("Flatten")]))
    with pytest.raises(ds.DeepSPUnavailableError, match="deepsp_stride_unsupported"):
        ds._architecture(
            _architecture_file(tmp_path, [_layer("Conv1D", padding="valid", strides=[2]), _layer("Flatten")])
        )
    with pytest.raises(ds.DeepSPUnavailableError, match="deepsp_flatten_count"):
        ds._architecture(_architecture_file(tmp_path, [*good, _layer("Flatten")]))


def test_the_three_real_models_differ_and_are_all_accepted():
    """Pinning one shape rejected two of the three. The vocabulary is what is fixed."""
    assert "Conv1D" in ds._SUPPORTED_LAYERS and "Dense" in ds._SUPPORTED_LAYERS
    assert "relu" in ds._SUPPORTED_ACTIVATIONS and "linear" in ds._SUPPORTED_ACTIVATIONS
    assert "LSTM" not in ds._SUPPORTED_LAYERS


# ------------------------------------------------------------------ the round trip


def test_written_descriptors_are_read_back_by_developability(tmp_path):
    rows = ({"Name": "ab1"} | {column: float(index) for index, column in enumerate(DEEPSP_COLUMNS)},)
    result = ds.DeepSPResult(rows)
    path = ds.write_deepsp_csv(result, tmp_path / "out" / "DeepSP_descriptors.csv")
    assert path.is_file()
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["Name", *DEEPSP_COLUMNS], "DeepSP's own column spelling and order"
    record = read_deepsp_table(path)
    assert "oracle_blocker" not in record, record
    assert record["method"] == "deepsp"
    assert record["rows"][0]["name"] == "ab1"
    assert record["rows"][0]["scores"]["SAP_pos_CDRH1"] == 0.0
    assert record["pass_cut"] is None
    assert result.names == ("ab1",)


# ------------------------------------------------------------- against DeepSP itself


@needs_models
def test_the_port_reproduces_deepsps_published_descriptors():
    """Raw sequence -> ANARCII -> torch -> CSV, against DeepSP's own numbers.

    This is the test that makes the reimplementation defensible: a port nobody
    checked against the original is a guess with a confident interface.
    """
    models = _models_dir()
    assert models is not None
    rows = list(csv.DictReader((models / "DeepSP_input.csv").open(encoding="utf-8")))
    antibodies = {row["Name"]: (row["Heavy_Chain"].strip(), row["Light_Chain"].strip()) for row in rows}
    result = ds.run_deepsp(antibodies, models)
    assert not result.skipped, result.skipped
    assert len(result.rows) == len(antibodies)

    published = {
        row["Name"]: row for row in csv.DictReader((models / "DeepSP_descriptors_anarci2.csv").open(encoding="utf-8"))
    }
    compared = 0
    worst = 0.0
    for row in result.rows:
        want = published[str(row["Name"])]
        for column in DEEPSP_COLUMNS:
            theirs = float(want[column])
            mine = float(row[column])  # type: ignore[arg-type]
            worst = max(worst, abs(mine - theirs) / max(abs(theirs), 1.0))
            compared += 1
    assert compared == 30 * len(antibodies)
    # Their CSV is float32 text, so the floor is its own rounding, not the port.
    assert worst < 1e-4, f"worst relative error {worst:.3e}"
