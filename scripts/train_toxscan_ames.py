#!/usr/bin/env python3
"""Fit and validate the isolated ToxScan fingerprint branch on authenticated Ames label tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from rdkit import rdBase
from safetensors.numpy import save
from sklearn.metrics import roc_auc_score
from torch import nn

from rfproteina.disease import toxscan_ames_model as ames
from rfproteina.disease import toxscan_inputs
from rfproteina.disease.ligand_inputs import read_regular


def read_splits(source: Path) -> tuple[dict, dict]:
    """Keep original partitions, quarantine cross-split graphs and contradictory within-split labels."""
    labels = {part: defaultdict(set) for part in ames.SPLITS}
    audit = {}
    for part in ames.SPLITS:
        raw = read_regular(source / f"{part}.csv", 32 * 1024 * 1024, "ames_table", ValueError)
        if hashlib.sha256(raw).hexdigest() != ames.SOURCE_SHA256.get(part):
            raise ValueError(f"unverified Ames label table: {part}")
        reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        if next(reader, []) != ["smiles", "ames"]:
            raise ValueError("unexpected Ames table header")
        stats = {"input_rows": 0, "invalid_structure_rows": 0}
        for row in reader:
            if len(row) != 2 or row[1] not in ("0", "1"):
                raise ValueError("invalid Ames source row")
            stats["input_rows"] += 1
            try:
                key = toxscan_inputs.canonical(row[0])
            except toxscan_inputs.ToxScanError as error:
                if str(error) != "toxscan_invalid_structure":
                    raise
                stats["invalid_structure_rows"] += 1
                continue
            labels[part][key].add(int(row[1]))
        audit[part] = stats
    clean = {}
    for part in ames.SPLITS:
        others = set().union(*(labels[p] for p in ames.SPLITS if p != part))
        exclusive = {k: v for k, v in labels[part].items() if k not in others}
        clean[part] = [(k, next(iter(v))) for k, v in sorted(exclusive.items()) if len(v) == 1]
        audit[part].update(
            {
                "unique_graphs": len(labels[part]),
                "overlap_graphs_removed": len(labels[part]) - len(exclusive),
                "conflict_graphs_removed": len(exclusive) - len(clean[part]),
                "retained_graphs": len(clean[part]),
            }
        )
        if {y for _, y in clean[part]} != {0, 1}:
            raise ValueError(f"both classes are required in split {part}")
    return clean, audit


def _head() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.GELU(),
        nn.Linear(1024, 512),
        nn.Dropout(ames.FIT_SETTINGS["dropout"]),
        nn.Linear(512, 512),
        nn.Tanh(),
        nn.Dropout(ames.FIT_SETTINGS["dropout"]),
        nn.Linear(512, 1),
    )
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.02)
            nn.init.zeros_(layer.bias)
    return model


@torch.inference_mode()
def _scores(model: nn.Module, matrix: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.cat([model(batch).sigmoid().flatten() for batch in matrix.split(256)]).numpy()


def _fit(train: tuple, valid: tuple) -> tuple:
    x_train, y_train = train
    x_valid, y_valid = valid
    config = ames.FIT_SETTINGS
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    model = _head()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(x_train), device="cpu")
        for idx in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_train[idx]).flatten(), y_train[idx])
            loss.backward()
            optimizer.step()
        auc = float(roc_auc_score(y_valid, _scores(model, x_valid)))
        print(f"Epoch {epoch:02d} validation AUROC {auc:.6f}", flush=True)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch - best_epoch >= config["patience"]:
            break
    if best_state is None:
        raise ValueError("no valid epoch")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_auc, best_epoch


def _digest_matrix(digests: list[bytes]) -> np.ndarray:
    return np.frombuffer(b"".join(sorted(digests)), dtype=np.uint8).reshape(-1, 32).copy()


def build(source: Path, stage: Path) -> ames.Fit:
    ames.verify_representation()
    rows, audit = read_splits(source)
    data, tensors = {}, {}
    train_fingerprints = set()
    for part, records in rows.items():
        fingerprints = np.stack([ames.fingerprint(s) for s, _ in records])
        data[part] = (
            torch.from_numpy(fingerprints.astype(np.float32)),
            torch.tensor([label for _, label in records], dtype=torch.float32),
        )
        tensors[part] = _digest_matrix([hashlib.sha256(s.encode("ascii")).digest() for s, _ in records])
        digests = [ames.fingerprint_digest(fp) for fp in fingerprints]
        if part == "train":
            tensors["training_fingerprints"] = _digest_matrix(digests)
            train_fingerprints = set(digests)
        audit[part]["graphs_with_fingerprint_seen_in_train"] = sum(digest in train_fingerprints for digest in digests)
    print("Retained graphs", {k: len(v) for k, v in rows.items()}, flush=True)
    model, val_auc, epoch = _fit(data["train"], data["valid"])
    for i, layer in enumerate(layer for layer in model if isinstance(layer, nn.Linear)):
        tensors[f"w{i}"] = layer.weight.detach().numpy().copy()
        tensors[f"b{i}"] = layer.bias.detach().numpy().copy()
    # Evaluate the selected state once. Use the same float32 runtime scorer for every test graph.
    x_test, y_test = data["test"]
    reference_scores = _scores(model, x_test)
    test_scores = np.array([ames.score_vector(row, tensors) for row in x_test.numpy()])
    if not np.allclose(reference_scores, test_scores, rtol=2e-5, atol=2e-6):
        raise ValueError("deployed CPU scores disagree with the fitted network")
    actual = y_test.numpy().astype(bool)
    called = test_scores >= 0.5
    test_order = sorted(range(len(actual)), key=lambda i: hashlib.sha256(rows["test"][i][0].encode("ascii")).digest())
    tensors["test_labels"] = actual[test_order].astype(np.uint8)
    tensors["test_scores"] = test_scores[test_order]
    validation = ames.validate_validation(
        {
            "n_train": len(rows["train"]),
            "n_validation": len(rows["valid"]),
            "n_test": len(rows["test"]),
            "test_positives": int(actual.sum()),
            "test_negatives": int((~actual).sum()),
            "tp": int((called & actual).sum()),
            "tn": int((~called & ~actual).sum()),
            "fp": int((called & ~actual).sum()),
            "fn": int((~called & actual).sum()),
            "accuracy": float(np.mean(called == actual)),
            "roc_auc": float(roc_auc_score(actual, test_scores)),
            "validation_roc_auc": val_auc,
            "selected_epoch": epoch,
            "test_fingerprint_seen_in_train": audit["test"]["graphs_with_fingerprint_seen_in_train"],
        }
    )
    evidence = {
        "version": 1,
        "method": ames.METHOD_ID,
        "archive_sha256": ames.ARCHIVE_SHA256,
        "source_revision": ames.SOURCE_REVISION,
        "tables_sha256": dict(ames.SOURCE_SHA256),
        "fit_settings": ames.FIT_SETTINGS,
        "fingerprint": ames.FINGERPRINT,
        "validation": validation,
        "split_audit": audit,
        "software": {
            "torch": str(torch.__version__),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
    }
    tensors["evidence_digest"] = np.frombuffer(ames.evidence_digest(evidence), dtype=np.uint8).copy()
    raw = save(tensors)
    evidence["model_sha256"] = hashlib.sha256(raw).hexdigest()
    (stage / ames.WEIGHTS).write_bytes(raw)
    (stage / ames.RECEIPT).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ames.verify_bundle(stage)


def train(source: Path, output: Path) -> ames.Fit:
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(output.parent / f".{output.name}.install.lock"):
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.is_dir():
                raise ValueError("invalid existing fit directory")
            return ames.verify_bundle(output)
        # Stage next to the destination to publish verified weights and their receipt together.
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".ames-training-") as name:
            stage = Path(name) / "fit"
            stage.mkdir()
            fit = build(source, stage)
            stage.rename(output)
            return fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory with authenticated train.csv, valid.csv and test.csv")
    parser.add_argument("--output", type=Path, default=ames.release_directory(), help="new local fit directory")
    args = parser.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    from threadpoolctl import threadpool_limits

    try:
        with threadpool_limits(limits=1, user_api="blas"):
            fit = train(args.source, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Ames training failed: {error}\n")
    print(json.dumps({"model_sha256": fit.sha256, "validation": fit.validation}, indent=2))


if __name__ == "__main__":
    main()
