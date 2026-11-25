"""Published, cohort-level ImmuScope benchmarks for the program's evidence reference."""

from __future__ import annotations


def published_study() -> dict[str, object]:
    """Return a fresh JSON-compatible reference record for Shen et al. (2025)."""
    return {
        "study": "ImmuScope",
        "evidence_type": "published_benchmark",
        "source": "https://doi.org/10.1038/s42256-025-01073-z",
        "source_data_check": {
            "figure_2_workbook_sha256": "5fbff9c24b2b6d7d042966b450137a21a01603886ad692c7f4120d59bed6291b",
            "figure_5_workbook_sha256": "9011d5c336fb1b87cdb3aa238a6043fa17a1cd153f3f22339054af6741170e18",
            "article_pdf_sha256": "8d88f76234eea19802d2eda13d3f7d3840aa6ffdfbdc5d4322ecaeaff53f9e32",
            "figure_2": "The means of the source-workbook columns reproduce all three listed EL and CD4 metrics.",
            "figure_5": "The workbook omits panel 5a. Integrating the published ROC trace gives 0.90936. "
            "The confidence interval is author-reported.",
        },
        "endpoints": {
            "ImmuScope-EL": {
                "endpoint": "MHC-II peptide presentation",
                "corpus_provenance": "Single- and multi-allelic eluted ligands from the NetMHCIIpan-4.3 dataset.",
                "evaluation": "Fivefold evaluation of eluted ligands and decoys, summarized across alleles.",
                "source_figure": "2c",
                "paired_allele_comparisons": 58,
                "metrics": {"area_under_precision_recall_curve": 0.856, "positive_predictive_value": 0.796},
            },
            "ImmuScope-CD4": {
                "endpoint": "CD4 T-cell epitope identification",
                "corpus_provenance": "Presentation pretraining followed by fine-tuning on binding-affinity data.",
                "evaluation": "ROC AUC calculated for each source protein, epitope and MHC-II allele triplet.",
                "source_figure": "2a",
                "paired_comparisons_in_figure": 824,
                "distinct_triplets_in_methods": 842,
                "metrics": {"mean_per_triplet_roc_auc": 0.825},
            },
            "ImmuScope-IM": {
                "endpoint": "Peptide-level T-cell immunogenicity",
                "corpus_provenance": "IEDB and MHCBN interferon-gamma assay labels with sampled peptide negatives.",
                "evaluation": "Held-out records in the authors' random training/test split.",
                "source_figure": "5a",
                "test_records": 17897,
                "metrics": {"roc_auc": 0.909, "roc_auc_95_percent_ci": [0.901, 0.918]},
            },
        },
    }
