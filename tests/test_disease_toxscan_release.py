"""Check safe extraction of the published ToxScan Ames split labels."""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

_script = Path(__file__).parents[1] / "scripts" / "extract_toxscan_ames.py"
_spec = importlib.util.spec_from_file_location("extract_toxscan_ames", _script)
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)


def _record(label, *, other=None, index=1, smiles="CCO"):
    classes = [f"endpoint-{i}" for i in range(26)]
    classes[index] = release.ENDPOINT
    regression = [f"endpoint-{i}" for i in range(26, 31)]
    all_labels = np.full(31, release.MISSING, dtype="<f4")
    class_labels = np.full(26, release.MISSING, dtype="<i8")
    all_labels[index] = label if other is None else other
    class_labels[index] = label
    return pickle.dumps(
        {
            "data": smiles,
            "title": classes + regression,
            "classification_title": classes,
            "regression_title": regression,
            "target": all_labels,
            "classification_target": class_labels,
            "regression_target": all_labels[26:],
        },
        protocol=5,
    )


@pytest.mark.parametrize(("label", "expected"), [(0, 0), (1, 1), (release.MISSING, None)])
def test_release_extracts_named_endpoint_and_preserves_missing_labels(label, expected):
    # Moving the task exercises the lookup in both title arrays. Long SMILES use BINUNICODE.
    assert release.read_ames_record(_record(label, index=4, smiles="C" * 300)) == ("C" * 300, expected)


@pytest.mark.parametrize(("label", "other"), [(1, 0), (-1, -1), (2, 2)])
def test_release_rejects_invalid_or_disagreeing_label_vectors(label, other):
    with pytest.raises(ValueError, match="Ames label"):
        release.read_ames_record(_record(label, other=other))


def test_release_rejects_truncation_and_never_executes_pickle(tmp_path):
    with pytest.raises(ValueError):
        release.read_ames_record(_record(1)[:24])
    marker = tmp_path / "unsafe"

    class Untrusted:
        def __reduce__(self):
            return eval, (f"open({str(marker)!r}, 'w')",)

    with pytest.raises(ValueError, match="header"):
        release.read_ames_record(pickle.dumps(Untrusted(), protocol=5))
    assert not marker.exists()


def test_release_refuses_unrecognized_archive_before_creating_outputs(tmp_path):
    archive = tmp_path / "unknown.tar.gz"
    archive.write_bytes(b"unknown")
    with pytest.raises(ValueError, match="pinned release"):
        release.extract(archive, tmp_path / "labels")
    assert not (tmp_path / "labels").exists()
