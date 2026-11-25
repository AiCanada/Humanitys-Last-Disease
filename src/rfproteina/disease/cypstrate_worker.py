"""Private adapter executed only inside the pinned upstream CYPstrate Python 3.9 image."""

import hashlib
import json
import math
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from numbers import Real


def classify(request: dict) -> dict:
    from cypstrate import CypstrateModel

    model = CypstrateModel()
    inputs = request["inputs"]
    results = []
    for start in range(0, len(inputs), 16):
        batch = inputs[start : start + 16]
        representations = [entry["smiles"] + " " + entry["id"] for entry in batch]
        records = model.predict(
            representations, prediction_mode=request["mode"], applicability_domain=True, output_format="record_list"
        )
        for row in records:
            index = row.get("mol_id")
            if type(index) is not int or not 0 <= index < len(batch):
                raise ValueError("unattributed_model_record")
            # A parser rejection can lose the molecule's name. Its exact echoed input and
            # position still identify a failure. A classification must match the name.
            rejected_input = row.get("preprocessed_mol") is None and row.get("input_text") == representations[index]
            if row.get("name") != batch[index]["id"] and not rejected_input:
                raise ValueError("unattributed_model_record")
            standardized = row.get("preprocessed_smiles")
            result = {"id": batch[index]["id"], "predictions": {}, "preprocessed_id": None}
            if isinstance(standardized, str) and standardized and row.get("preprocessed_mol") is not None:
                result["preprocessed_id"] = hashlib.sha256(standardized.encode()).hexdigest()
                result["upstream_smiles_changed"] = standardized != row["input_smiles"]
                for isoform in batch[index]["isoforms"]:
                    key = isoform[3:].lower()
                    neighbor = row.get("neighbor_" + key)
                    result["predictions"][isoform] = {
                        "label": row.get("prediction_" + key),
                        "similarity": (
                            float(neighbor)
                            if isinstance(neighbor, Real) and not isinstance(neighbor, bool) and math.isfinite(neighbor)
                            else None
                        ),
                    }
            results.append(result)
    return {"version": version("cypstrate"), "mode": request["mode"], "results": results}


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        reply = classify(json.load(sys.stdin.buffer))
    print(json.dumps(reply, allow_nan=False))
