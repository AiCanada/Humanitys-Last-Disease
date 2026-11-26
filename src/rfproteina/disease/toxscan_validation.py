"""Author-reported ToxScan test-set performance, keyed by the corresponding export column."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

SOURCE = "https://doi.org/10.1021/acs.est.5c04614.s001"
STUDY_SIMILARITY_CUTOFF = 0.679  # Wu et al., Sections 2.7 and 3.8, mean of the ten closest Tanimoto scores.

# The exporter prints the same unit for all five. Preserve their source labels and values.
SOURCE_DISPLAY_ENDPOINTS = (
    "LC50DM (48h Daphnia magna LC50)", "LC50 (96h Fathead minnow LC50)",
    "IGC50 (40h Tetrahymena pyriformis IGC50)", "BCF (Bioconcentration factor)", "Rat oral LD50",
)


@dataclass(frozen=True)
class Benchmark:
    assay: str
    auroc: float
    auroc_sd: float
    accuracy: float
    accuracy_sd: float


# Wu et al., Tables S1 (AUROC) and S2 (fraction correctly classified).
# Both dispersions are across the authors' repeated runs on their test data.
BENCHMARKS = {
    "Carcinogenicity": Benchmark("Rat carcinogenicity classification", 0.775, 0.032, 0.720, 0.042),
    "Ames Mutagenicity": Benchmark("Bacterial reverse-mutation assay", 0.879, 0.016, 0.876, 0.012),
    "Respiratory toxicity": Benchmark("Respiratory toxicity classification", 0.877, 0.014, 0.804, 0.022),
    "Eye irritation": Benchmark("Eye irritation classification", 0.982, 0.004, 0.955, 0.005),
    "Eye corrosion": Benchmark("Eye corrosion classification", 0.988, 0.004, 0.962, 0.015),
    "Cardiotoxicity1": Benchmark("hERG blockade, IC50 cutoff of 1 micromolar", 0.778, 0.016, 0.794, 0.052),
    "Cardiotoxicity5": Benchmark("hERG blockade, IC50 cutoff of 5 micromolar", 0.692, 0.009, 0.621, 0.015),
    "Cardiotoxicity10": Benchmark("hERG blockade, IC50 cutoff of 10 micromolar", 0.671, 0.013, 0.655, 0.016),
    "Cardiotoxicity30": Benchmark("hERG blockade, IC50 cutoff of 30 micromolar", 0.670, 0.022, 0.783, 0.019),
    "CYP1A2": Benchmark("Human CYP1A2 inhibition", 0.962, 0.004, 0.917, 0.014),
    "CYP2C19": Benchmark("Human CYP2C19 inhibition", 0.810, 0.015, 0.779, 0.021),
    "CYP2C9": Benchmark("Human CYP2C9 inhibition", 0.784, 0.015, 0.842, 0.020),
    "CYP2D6": Benchmark("Human CYP2D6 inhibition", 0.838, 0.032, 0.928, 0.007),
    "CYP3A4": Benchmark("Human CYP3A4 inhibition", 0.927, 0.011, 0.881, 0.007),
    "NR-AR": Benchmark("Androgen receptor activity", 0.757, 0.014, 0.968, 0.002),
    "NR-AR-LBD": Benchmark("Androgen receptor ligand-binding domain activity", 0.805, 0.017, 0.983, 0.003),
    "NR-AhR": Benchmark("Aryl hydrocarbon receptor activity", 0.865, 0.017, 0.896, 0.008),
    "NR-Aromatase": Benchmark("Aromatase activity", 0.812, 0.042, 0.943, 0.005),
    "NR-ER": Benchmark("Estrogen receptor activity", 0.713, 0.017, 0.857, 0.016),
    "NR-ER-LBD": Benchmark("Estrogen receptor ligand-binding domain activity", 0.891, 0.010, 0.960, 0.003),
    "NR-PPAR-gamma": Benchmark("PPAR gamma receptor activity", 0.800, 0.035, 0.964, 0.003),
    "SR-ARE": Benchmark("Antioxidant response element activity", 0.830, 0.018, 0.868, 0.009),
    "SR-ATAD5": Benchmark("ATAD5 DNA damage response", 0.913, 0.015, 0.966, 0.003),
    "SR-HSE": Benchmark("Heat shock response element activity", 0.825, 0.023, 0.953, 0.003),
    "SR-MMP": Benchmark("Mitochondrial membrane potential", 0.930, 0.008, 0.902, 0.007),
    "SR-p53": Benchmark("p53 signaling response", 0.894, 0.018, 0.943, 0.004),
}


def for_reported_calls(classifications: Mapping[str, str | None]) -> dict:
    """Attach benchmark context only where the candidate has an explicit YES or NO call."""
    return {name: asdict(BENCHMARKS[name]) for name, call in classifications.items() if call in ("YES", "NO")}


def domain_disagreement(domain: str, similarity: float | None) -> str | None:
    """Compare only two observed fields; retain the explicit source decision in the report."""
    if similarity is None or domain == "unreported":
        return None
    if (domain == "in_domain") == (similarity > STUDY_SIMILARITY_CUTOFF):
        return None
    return (
        f"The export's domain flag and displayed similarity ({similarity:g}) disagree under the study's "
        f"rule of similarity above {STUDY_SIMILARITY_CUTOFF:g}. Review the supplied values before using this result."
    )
