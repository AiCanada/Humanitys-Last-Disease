"""The MCP view is pinned to the records it reads, not only to its own schema.

``screen.py`` governs its own shape: ``SCREEN_VERSION`` is raised whenever a key
is added, and ``tests/test_disease_screen_shape.py`` pins both key sets beside
that number. The MCP tests pinned ``_AXIS_FIELDS`` against ``_AXIS_SCHEMA`` and
``_VERDICT_FIELDS`` against the same schema -- MCP against MCP. Nothing held
either against the program, so the record grew through version 2 and version 3
while this view kept returning the version-1 field set, and every MCP test
passed the whole way. ``refold`` is the plainest case: the key version 3 exists
for was filtered out beside a ``screen_version: 3`` that announced it.

So the pin here runs the other way. Every key the screen writes is either
returned, shaped by a function of its own, or named in an exclusion list with a
reason. A new key in ``screen.py`` fails this file until someone decides which
of the three it is. The same pin is applied to the generation record the archive
writes, and to the one rule this package spells twice.
"""

import json
from pathlib import Path

import jsonschema

from tests.helpers.disease import load_mcp_server as _server

#: Screen keys the verdict deliberately does not carry, and why. Each is a
#: decision that survives review; an omission is not.
_NOT_IN_THE_VERDICT = {
    # Per-design measurements. ``read_sequence_measures`` returns these, and the
    # verdict is a decision rather than a copy of the table behind it.
    "samples": "read_sequence_measures returns the per-design measurements",
    "sequence_developability": "read_sequence_measures returns the first design's sequence block",
    "designed_interface": "read_sequence_measures and backend_status return the interface block",
    # Every supplied file with its size and digest. The entries that moved
    # between two verdicts already arrive under ``replaced.changed_inputs``,
    # which is the part that qualifies a verdict rather than describing a folder.
    "inputs": "replaced.changed_inputs already names the supplied files that moved",
    # Counts and charges for the first design, and the drug-type report. Neither
    # carries a state and neither enters the call.
    "chemical_liability": "motif counts and charge, no state, enters no call",
    "modality_screen": "the drug-type report, no state, enters no call",
}


def _screen_keys() -> set[str]:
    from rfproteina.disease.screen import SCREEN_OPTIONAL_KEYS, SCREEN_REQUIRED_KEYS

    return set(SCREEN_REQUIRED_KEYS) | set(SCREEN_OPTIONAL_KEYS)


def _handled() -> set[str]:
    """Keys the view accounts for, each by a different route."""
    from rfproteina.disease import mcp_api

    # ``replaced`` is reshaped by _replaced rather than copied, and the axis
    # groups are reshaped by _axis_summary. Neither can be a plain field.
    return set(mcp_api._VERDICT_FIELDS) | set(mcp_api._AXIS_GROUPS) | {"replaced"}


def test_every_screen_key_is_returned_or_excluded_by_name() -> None:
    """The pin. A key added to screen.py lands here before it lands nowhere."""
    missing = _screen_keys() - _handled() - set(_NOT_IN_THE_VERDICT)
    assert not missing, (
        f"screen.json now carries {sorted(missing)} and the verdict view does not. "
        "Add each to mcp_api._VERDICT_FIELDS and the matching property in the server's "
        "_VERDICT_SCHEMA, or name it in _NOT_IN_THE_VERDICT above with the reason it stays out."
    )


def test_the_verdict_returns_nothing_the_screen_does_not_write() -> None:
    """The other direction: a field kept here after the writer dropped it is dead."""
    from rfproteina.disease import mcp_api

    dead = set(mcp_api._VERDICT_FIELDS) - _screen_keys()
    assert not dead, f"_VERDICT_FIELDS keeps {sorted(dead)}, which no screen writes any more"


def test_no_exclusion_outlives_the_key_it_excludes() -> None:
    """A reason for a key that no longer exists is a note about nothing."""
    stale = set(_NOT_IN_THE_VERDICT) - _screen_keys()
    assert not stale, f"_NOT_IN_THE_VERDICT names {sorted(stale)}, which screen.py no longer writes"


def test_the_published_schema_declares_exactly_what_the_verdict_returns() -> None:
    """Equality, not containment: an undeclared field a validating client meets is dropped."""
    from rfproteina.disease import mcp_api

    declared = set(_server()._VERDICT_SCHEMA["properties"])
    derived = {"replaced", "curated_axes", "malformed", "record"}
    expected = set(mcp_api._VERDICT_FIELDS) | set(mcp_api._AXIS_GROUPS) | derived
    assert declared == expected, "the verdict view and its published shape have drifted apart"


def _screened(folder: Path, record: dict) -> Path:
    """A finished run carrying one stored screen, without running a screen."""
    run = folder / "gene_1ABC_1"
    (run / "inference_samples" / "binder_0").mkdir(parents=True)
    (run / "screen.json").write_text(json.dumps(record), encoding="utf-8")
    return run


def _version_three_record() -> dict:
    """A screen in the shape the current writer produces, trimmed to the keys under test."""
    return {
        "program": "Humanitys-Last-Disease",
        "screen_version": 3,
        "call": "axis_hit",
        "bar": "sequence",
        "bar_axes": ["mhc_i", "mhc_ii", "ada"],
        "states": {"mhc_i": "hit"},
        "hits": ["mhc_i"],
        "untested": [],
        "code": {"commit": "0" * 40, "tree_state": "clean"},
        "cuts": {
            "gravy_max": 0.1,
            "gravy_max_source": "program_convention",
            "mhc_i_rank_max": 2.0,
            "plasma_abundance_max": None,
        },
        "metrics_file": "inference_metrics/per_example_metrics.csv",
        "metrics_blocker": "column_repeated",
        "chain_blocker": "chain_disagreement",
        # The shape a real screen writes, taken from one: a threshold names the
        # metric column, which kind of bound it is, the cut, the value reached
        # and whether it held. A criterion the oracle never scored is
        # not_evaluated with a null measurement, which is the ordinary case on a
        # run with no refolding configured.
        "refold": {
            "criteria": [
                {
                    "name": "ppi",
                    "state": "fail",
                    "thresholds": [
                        {
                            "metric": "oracle_ppi.ptm_coi",
                            "bound": "min",
                            "cut": 80.0,
                            "measured": 31.0,
                            "met": False,
                        }
                    ],
                    "failed_thresholds": ["oracle_ppi.ptm_coi"],
                },
                {
                    "name": "unconditional",
                    "state": "not_evaluated",
                    "thresholds": [
                        {
                            "metric": "refolded_backbone_rmsd.rmsd",
                            "bound": "max",
                            "cut": 1.5,
                            "measured": None,
                            "met": None,
                        }
                    ],
                },
            ],
            "declared": 11,
            "evaluated": 1,
            "criteria_file": "configs/metrics/pass_criteria/all_pass_criteria.yaml",
        },
        "solubility": {"solubility_blocker": "solubility_not_configured"},
        "design_diversity": {"method": "pairwise_identity", "designs": 8, "distinct": 1},
        "sequence_liability": {
            "mhc_i": {
                "state": "hit",
                "measurement": 3.0,
                "blocker": None,
                "source": "measured",
                "method": "calis_2013_class_i_9mer",
                "presentation": "restricted",
                "covered_examples": ["binder_0"],
                "uncovered_examples": ["binder_1", "binder_2"],
                "mask_unpublished_alleles": ["HLA-B*08:01"],
                "displaced_method": "calis_2013_class_i_9mer",
                "displaced_table": "mhc_i",
            }
        },
        "target_liability": {
            "plasma_abundance": {
                "state": "untested",
                "measurement": 4130.0,
                "blocker": "threshold_not_set",
                "source": "untested",
                "assay": "immunoassay",
                "units": "pg/L",
            },
            "plasma_hub": {
                "state": "clear",
                "measurement": 4,
                "blocker": None,
                "source": "measured",
                "threshold": 10,
                "confidence": 700,
                "network": "physical",
                "release": "12.0",
            },
        },
        "nucleic_liability": {
            "off_target_hybridization": {
                "state": "clear",
                "measurement": 0,
                "blocker": None,
                "source": "measured",
                "reference_file": "transcriptome.fa",
                "reference_digest": "a" * 16,
                "reference_bases": 1_234_567,
            }
        },
    }


def test_a_version_three_screen_returns_the_blocks_that_qualify_its_verdict(tmp_path: Path) -> None:
    """Each of these was on disk and filtered out of the answer."""
    from rfproteina.disease import mcp_api

    _screened(tmp_path, _version_three_record())
    run = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]
    assert "error" not in run, run.get("error")
    verdict = run["verdict"]
    assert verdict["bar_axes"] == ["mhc_i", "mhc_ii", "ada"], "a bar name cannot be checked without its axes"
    assert verdict["cuts"]["gravy_max_source"] == "program_convention"
    assert verdict["metrics_file"] == "inference_metrics/per_example_metrics.csv"
    assert verdict["metrics_blocker"] == "column_repeated"
    assert verdict["chain_blocker"] == "chain_disagreement"
    failed = verdict["refold"]["criteria"][0]["failed_thresholds"]
    assert failed == ["oracle_ppi.ptm_coi"], "a false ppi_pass named none of the bounds behind it; this record does"
    missed = verdict["refold"]["criteria"][0]["thresholds"][0]
    assert missed["cut"] == 80.0, "the cut the criterion declares"
    assert missed["measured"] == 31.0, "the least favourable value the run reached against it"
    assert missed["met"] is False, "and whether it held"
    assert verdict["refold"]["declared"] == 11
    assert verdict["solubility"] == {"solubility_blocker": "solubility_not_configured"}
    assert verdict["design_diversity"]["distinct"] == 1, "eight designs and one molecule is one shot, not eight"


def test_a_version_three_screen_returns_the_qualifiers_on_each_axis(tmp_path: Path) -> None:
    """A measurement is not a verdict without what produced it and what it covers."""
    from rfproteina.disease import mcp_api

    _screened(tmp_path, _version_three_record())
    verdict = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]
    mhc = verdict["sequence_liability"]["mhc_i"]
    assert mhc["method"] == "calis_2013_class_i_9mer", "a supplied score and a computed one read alike without this"
    assert mhc["covered_examples"] == ["binder_0"]
    assert mhc["uncovered_examples"] == ["binder_1", "binder_2"], "the designs this state does not speak for"
    assert mhc["mask_unpublished_alleles"] == ["HLA-B*08:01"]
    assert mhc["presentation"] == "restricted"
    assert mhc["displaced_table"] == "mhc_i"
    abundance = verdict["target_liability"]["plasma_abundance"]
    assert abundance["assay"] == "immunoassay", "units said pg/L and nothing said how it was obtained"
    hub = verdict["target_liability"]["plasma_hub"]
    assert (hub["confidence"], hub["network"], hub["release"]) == (700, "physical", "12.0")
    off_target = verdict["nucleic_liability"]["off_target_hybridization"]
    assert off_target["reference_file"] == "transcriptome.fa"
    assert off_target["reference_bases"] == 1_234_567, "a clear against a stub is not a clear against a transcriptome"


def test_an_axis_says_what_scale_its_measurement_is_on(tmp_path: Path) -> None:
    """The producers write these beside keys the view already kept, line for line.

    ``_mhc_record`` writes not_nm_columns, rank, ic50, trigger and
    unscored_peptides one after another, and three of the five were returned.
    These axes carry no units and no threshold, so a clear arrived as a bare
    number that is a percentile rank on one table and an affinity in nM on
    another, with nothing saying which.
    """
    from rfproteina.disease import mcp_api

    record = _version_three_record()
    record["sequence_liability"]["mhc_i"].update(
        {
            "state": "clear",
            "measurement": 0.8,
            "trigger": "rank",
            "rank": 0.8,
            "ic50": None,
            "binding_peptides": 3,
            "reference_density": 1.2,
            "reference_method_undeclared": True,
            "unparsed_columns": ["ic50_nm"],
        }
    )
    record["sequence_liability"]["proteome_offtarget"] = {
        "state": "clear",
        "measurement": 0,
        "blocker": None,
        "source": "measured",
        "proteome_kmers": 20_402_131,
        "proteome_digest": "9f" * 8,
    }
    record["target_liability"]["paralog"] = {
        "state": "untested",
        "measurement": 0,
        "blocker": "cluster_singleton_not_evidence",
        "source": "untested",
        "compara_examined": 12,
        "compara_blocker": "gene_not_in_compara",
    }
    record["toxicology"] = {
        "herg": {
            "state": "clear",
            "measurement": 10000.0,
            "blocker": None,
            "source": "measured",
            "units": "nM",
            "threshold": 10000.0,
            "qualifier": ">",
            "n_readings": 3,
        }
    }
    record["nucleic_liability"]["off_target_hybridization"].update({"seed_width": 15, "seeds_searched": 86})
    _screened(tmp_path, record)
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    run = answer["runs"][0]
    assert "error" not in run, run.get("error")
    verdict = run["verdict"]
    mhc = verdict["sequence_liability"]["mhc_i"]
    assert (mhc["trigger"], mhc["rank"], mhc["ic50"]) == ("rank", 0.8, None), "which scale the number is on"
    assert mhc["binding_peptides"] == 3, "the numerator of the density reference_density is the bar for"
    assert mhc["reference_method_undeclared"] is True, "the bar came from a tool neither table named"
    assert mhc["unparsed_columns"] == ["ic50_nm"]
    off_target = verdict["sequence_liability"]["proteome_offtarget"]
    assert off_target["proteome_kmers"] == 20_402_131, "which proteome the absence claim is about, and how large"
    assert off_target["proteome_digest"] == "9f" * 8
    paralog = verdict["target_liability"]["paralog"]
    assert (paralog["compara_examined"], paralog["compara_blocker"]) == (12, "gene_not_in_compara")
    herg = verdict["toxicology"]["herg"]
    assert herg["qualifier"] == ">", "a verdict read off a bound must not look like one read off a number"
    assert herg["n_readings"] == 3
    oligo = verdict["nucleic_liability"]["off_target_hybridization"]
    assert (oligo["seed_width"], oligo["seeds_searched"]) == (15, 86), "the width and the denominator of a zero"
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_the_whole_version_three_answer_validates_against_its_own_schema(tmp_path: Path) -> None:
    """A client that validates throws away the whole answer over one undeclared field."""
    from rfproteina.disease import mcp_api

    _screened(tmp_path, _version_three_record())
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_a_new_block_of_the_wrong_type_is_named_rather_than_returned(tmp_path: Path) -> None:
    """These fields are typed in the schema, so a hand-edited file must not reach a client."""
    from rfproteina.disease import mcp_api

    record = _version_three_record()
    record["cuts"] = "not an object"
    record["refold"] = 3
    _screened(tmp_path, record)
    verdict = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]
    assert "cuts" not in verdict and "refold" not in verdict
    assert {"cuts", "refold"} <= set(verdict["malformed"])


def test_the_version_four_reports_arrive_without_the_design_slice_or_the_molecule(tmp_path: Path) -> None:
    """The four reports reach a client, less two keys only screen.json should hold.

    ``repeat_sequence`` is a slice of the designed construct, and an RNA slice passed
    the sequence rule untouched; ``smiles`` is the screened molecule, which the same
    rule mangled. A tree walk holds the bulky reports back the way it holds refold.
    """
    from rfproteina.disease import mcp_api, mcp_layers

    record = _version_three_record()
    record["designed_interface_quality"] = {"interface_quality_blocker": "coordinates_absent"}
    record["mhc_class_i"] = {"n_windows": 68, "predictor_available": False}
    record["synthesis"] = {"longest_repeat": 26, "repeat_sequence": "AUGGCUACGUUAGCCAUGGCUACGUU"}
    record["ligand_alerts"] = {"smiles": "CCCCCCCCCCCCCCCC(=O)O", "heavy_atoms": 18}
    _screened(tmp_path, record)
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    verdict = answer["runs"][0]["verdict"]
    assert verdict["designed_interface_quality"] == {"interface_quality_blocker": "coordinates_absent"}
    assert verdict["mhc_class_i"] == {"n_windows": 68, "predictor_available": False}
    assert verdict["synthesis"] == {"longest_repeat": 26}, "the length stays; the slice of the design does not"
    assert verdict["ligand_alerts"] == {"heavy_atoms": 18}, "the molecule stays in screen.json"
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)
    assert {"designed_interface_quality", "synthesis", "ligand_alerts"} <= mcp_layers._SCREEN_OMITTED
    assert "mhc_class_i" in mcp_layers.screen_fields()


def _with_provenance() -> dict:
    """A screen carrying the nested provenance both blocks that have one now write.

    Nested rather than top-level, which is why the pin at the head of this file
    stayed green through the change: it compares top-level names, and ``cuts``
    was already one.
    """
    record = _version_three_record()
    record["cuts"]["provenance"] = {
        "gravy_max": {
            "kind": "convention",
            "source": "program_convention",
            "caveat": "No published bar says where a developable protein stops.",
        },
        "mhc_i_rank_max": {
            "kind": "recommended",
            "source": "iedb_class_i_selection",
            "caveat": "IEDB's recommended class I selection: percentile rank <= 1, or IC50 <= 500 nM.",
        },
    }
    record["designed_interface_quality"] = {
        "interface_polar_contacts": 9,
        "provenance": {
            "interface_polar_contacts": {
                "kind": "borrowed",
                "source": "bindcraft_average_n_interface_hbonds",
                "caveat": "BindCraft's default of 3. DEFINITION DIFFERS: it counts Rosetta hydrogen bonds.",
                "bar": {"min": 3},
            },
            "shape_complementarity": {
                "kind": "reported_only",
                "source": "lawrence_colman_1993_scale_differs",
                "bar": None,
            },
        },
    }
    return record


def test_the_kind_vocabulary_is_the_one_the_program_declares() -> None:
    """The check imports the closed set rather than copying it.

    A copy is the failure this record exists to prevent. The set gained
    ``borrowed`` and ``reported_only`` after the first draft of the check was
    written, and against a literal list every borrowed BindCraft default would
    have been dropped from the verdict as malformed -- silently turning the
    disclosure that a bar is borrowed into no disclosure at all.
    """
    from rfproteina.disease import mcp_api
    from rfproteina.disease.threshold_provenance import KINDS

    assert KINDS, "the vocabulary is not empty"
    assert all(mcp_api._is_kind(kind) for kind in KINDS)
    assert not mcp_api._is_kind("published_somewhere")
    assert not mcp_api._is_kind(None)
    assert set(mcp_api._PROVENANCE_TYPES) == set(
        mcp_api._PROVENANCE_FIELDS
    ), "every provenance field is checked, for the reason _AXIS_TYPES exists"
    declared = set(_server()._PROVENANCE_SCHEMA["additionalProperties"]["properties"])
    assert declared == set(mcp_api._PROVENANCE_FIELDS), "the provenance view and its published shape have drifted apart"
    assert "required" not in _server()._PROVENANCE_SCHEMA["additionalProperties"], (
        "a field is dropped when it is malformed, so requiring one would make a "
        "validating client discard the whole answer over a hand-edited caveat"
    )


def test_a_threshold_says_what_kind_of_number_it_is(tmp_path: Path) -> None:
    """A reader cannot tell a published cut from a borrowed default without this.

    Both blocks that carry the record are checked here: the cuts the verdict was
    taken against, and the interface bars, which also carry the bar itself
    because the measurement and its threshold otherwise sit in two files.
    """
    from rfproteina.disease import mcp_api

    _screened(tmp_path, _with_provenance())
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    verdict = answer["runs"][0]["verdict"]
    cuts = verdict["cuts"]["provenance"]
    assert cuts["gravy_max"]["kind"] == "convention"
    assert cuts["mhc_i_rank_max"]["source"] == "iedb_class_i_selection"
    assert "malformed" not in verdict
    # The cuts themselves are untouched beside their account.
    assert verdict["cuts"]["gravy_max"] == 0.1
    interface = verdict["designed_interface_quality"]["provenance"]
    assert interface["interface_polar_contacts"]["bar"] == {"min": 3}
    assert interface["shape_complementarity"]["bar"] is None, "no cut adopted is an answer, not a gap"
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_a_kind_this_program_does_not_declare_is_named_rather_than_returned(tmp_path: Path) -> None:
    """One bad field costs that field, not the entry, and not the other eleven accounts."""
    from rfproteina.disease import mcp_api

    record = _with_provenance()
    record["cuts"]["provenance"]["gravy_max"]["kind"] = "invented"
    record["cuts"]["provenance"]["mhc_i_rank_max"]["caveat"] = 17
    record["cuts"]["provenance"]["plasma_fold_max"] = "not an entry"
    _screened(tmp_path, record)
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    verdict = answer["runs"][0]["verdict"]
    kept = verdict["cuts"]["provenance"]
    assert "kind" not in kept["gravy_max"], "a kind outside the closed set is not a kind"
    assert kept["gravy_max"]["source"] == "program_convention", "the slug two tests pin still travels"
    assert "caveat" not in kept["mhc_i_rank_max"]
    assert kept["mhc_i_rank_max"]["kind"] == "recommended", "the sound half of the entry stands"
    assert "plasma_fold_max" not in kept
    assert {
        "cuts.provenance.gravy_max.kind",
        "cuts.provenance.mhc_i_rank_max.caveat",
        "cuts.provenance.plasma_fold_max",
    } <= set(verdict["malformed"])
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_a_tree_of_runs_does_not_repeat_the_threshold_prose(tmp_path: Path) -> None:
    """The accounts are the same on every run; the numbers are not.

    Measured at 2,908 bytes on a protein screen, which is more than two of the
    blocks the walk already drops outright. ``cuts`` itself has to stay: the
    numbers a verdict was taken against are what the verdict means.
    """
    from rfproteina.disease import mcp_api, mcp_layers

    _screened(tmp_path, _with_provenance())
    result = mcp_layers.read_results({"directory": str(tmp_path)})
    cuts = result["screens"][0]["cuts"]
    assert "provenance" not in cuts
    assert cuts["gravy_max"] == 0.1, "the cut stays; only the prose about it goes"
    assert "iedb_class_i_selection" not in json.dumps(result)
    assert mcp_layers._SCREEN_TRIMMED["cuts"] == frozenset({"provenance"})
    assert "cuts" in mcp_layers.screen_fields(), "trimmed, not omitted"
    # The one run's own answer still carries it, which is where a reader goes.
    verdict = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]
    assert "provenance" in verdict["cuts"]


def test_the_stored_screen_keeps_every_account_the_view_trims(tmp_path: Path) -> None:
    """``include_record`` is the full record, so nothing here is lost to a reader who asks."""
    from rfproteina.disease import mcp_api

    _screened(tmp_path, _with_provenance())
    answer = mcp_api.read_verdict({"directory": str(tmp_path), "include_record": True})
    record = answer["runs"][0]["verdict"]["record"]
    assert set(record["cuts"]["provenance"]) == {"gravy_max", "mhc_i_rank_max"}
    assert record["designed_interface_quality"]["provenance"]["shape_complementarity"]["bar"] is None


def _alternatives(pattern: str) -> set[str]:
    """The top-level branches of an alternation. No branch here contains a bar."""
    return {part for part in pattern.split("|") if part}


def test_the_confidence_rule_is_the_same_rule_the_trace_writer_applies() -> None:
    """One rule, two spellings: the reader's copy was missing ``confidence``.

    The writer stores a metric whose name matches its rule; this module reads
    those metrics back through its own. A branch in one and not the other means
    a column kept on purpose is dropped on the way out -- and a trace carrying
    only such columns reaches ``_ranking`` with no shared metric, which orders
    candidates by filename.
    """
    from rfproteina.disease import mcp_layers, trace

    written = _alternatives(trace._CONFIDENCE.pattern)
    read = _alternatives(mcp_layers._HIGHER_IS_BETTER) | _alternatives(mcp_layers._LOWER_IS_BETTER)
    assert written == read, "the trace writer and the MCP reader disagree about what a confidence metric is"


def test_a_metric_named_confidence_is_ranked_as_a_score() -> None:
    """It is matched, and matched in the direction where a higher number is better."""
    from rfproteina.disease import mcp_layers

    assert mcp_layers._CONFIDENCE.search("mean_confidence") is not None
    assert mcp_layers._ERROR_METRIC.search("mean_confidence") is None


def test_the_generation_view_returns_every_key_the_archive_writes(tmp_path: Path) -> None:
    """The archive record's own identity and shape were being dropped from it."""
    from rfproteina.disease import mcp_layers
    from rfproteina.disease.archive import save_generation

    dest = save_generation(
        tmp_path,
        None,
        disease_name="asthma",
        disease_id="MONDO:0004979",
        target_id="IL5",
        group="gene",
        entry_id="8TLD",
        chain="2",
        binder_kind="protein",
    )
    written = set(json.loads((dest / "generation.json").read_text(encoding="utf-8")))
    expected = set(mcp_layers._GENERATION_FIELDS)
    assert written == expected, "the archive writes keys the generation view does not return, or it keeps dead ones"


def test_an_option_whose_value_is_null_is_not_an_option() -> None:
    """``screen: null`` read as a screen to the guards and as no screen to the program.

    It passed both, so the call went down the disease pipeline with nothing to
    answer the confirmation prompt -- the one thing the second guard exists to
    prevent -- after a live disease search.
    """
    import pytest

    from rfproteina.disease import mcp_api

    with pytest.raises(ValueError, match="accept_only or choice"):
        mcp_api.run_program({"options": {"screen": None, "disease": "asthma"}})
    with pytest.raises(ValueError, match="Set disease"):
        mcp_api.run_program({"options": {"screen": None}})
    with pytest.raises(ValueError, match="accept_only or choice"):
        mcp_api.run_program({"options": {"disease": "asthma", "choice": None}})


# --- what a reader needs to weigh one axis -----------------------------------
#
# Each of these was added by a fix whose whole point was that the reader
# could see the field, and the view kept dropping the newest ones. They were
# a separate file; the subject is the same pinning as above.


def _screened_axes(folder: Path, axes: dict) -> Path:
    """A finished run carrying a stored screen, without running one."""
    run = folder / "gene_1ABC_1"
    (run / "inference_samples" / "binder_0").mkdir(parents=True)
    record = {
        "program": "Humanitys-Last-Disease",
        "call": "axis_hit",
        "bar": "sequence",
        "states": {name: entry["state"] for name, entry in axes.items()},
        "hits": [name for name, entry in axes.items() if entry["state"] == "hit"],
        "untested": [],
        "sequence_liability": axes,
    }
    (run / "screen.json").write_text(json.dumps(record), encoding="utf-8")
    return run


def test_a_hit_from_a_partial_scan_says_so_in_the_verdict(tmp_path: Path) -> None:
    """A count over part of a chain is still a hit, and the reader must see which part."""
    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "mhc_i": {
                "state": "hit",
                "measurement": 17.6471,
                "blocker": None,
                "source": "measured",
                "scan_limit": "scan_incomplete",
            }
        },
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis["state"] == "hit"
    assert axis["scan_limit"] == "scan_incomplete", "the hit rests on a partial scan and the view hid it"


def test_an_axis_says_which_cut_it_was_judged_against(tmp_path: Path) -> None:
    """One number decided ten panel axes, and the record now names the cut per axis."""
    from rfproteina.disease import mcp_api

    run = tmp_path / "gene_2ABC_1"
    (run / "inference_samples" / "binder_0").mkdir(parents=True)
    (run / "screen.json").write_text(
        json.dumps(
            {
                "program": "Humanitys-Last-Disease",
                "call": "axis_hit",
                "bar": "coverage",
                "states": {"herg": "hit"},
                "hits": ["herg"],
                "untested": [],
                "toxicology": {
                    "herg": {
                        "state": "hit",
                        "measurement": 50.0,
                        "blocker": None,
                        "source": "measured",
                        "threshold": 10000.0,
                        "threshold_source": "program_convention_10uM",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["toxicology"]["herg"]
    assert axis["threshold"] == 10000.0
    assert axis["threshold_source"] == "program_convention_10uM", "a cut with no provenance reads as published"


def test_an_ordinary_measured_axis_gains_none_of_them(tmp_path: Path) -> None:
    """These fields say something happened. Absent, they must not appear at all."""
    from rfproteina.disease import mcp_api

    _screened_axes(tmp_path, {"mhc_i": {"state": "clear", "measurement": 1.0, "blocker": None, "source": "measured"}})
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis == {"state": "clear", "measurement": 1.0, "blocker": None, "source": "measured"}


def test_the_whole_answer_still_validates_against_its_own_schema(tmp_path: Path) -> None:
    """A field the schema forbids makes a validating client throw the verdict away."""
    import jsonschema

    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "ada": {
                "state": "clear",
                "measurement": 1.0,
                "blocker": None,
                "source": "axis_evidence",
                "curated": "basis_not_stated",
                "measured_state": "hit",
                "measured_blocker": None,
            },
            "mhc_i": {
                "state": "hit",
                "measurement": 17.6471,
                "blocker": None,
                "source": "measured",
                "scan_limit": "unscored_windows",
            },
        },
    )
    answer = mcp_api.read_verdict({"directory": str(tmp_path)})
    schema = {str(tool["name"]): tool.get("outputSchema") for tool in _server()._TOOLS}["read_verdict"]
    jsonschema.validate(answer, schema)


def test_a_clear_carries_the_bar_it_was_taken_against(tmp_path: Path) -> None:
    """A density clear with no reference is a number beside no comparison.

    ``measurement`` alone says 0.8 per 100 windows. Whether that is a clear
    depends entirely on the bar, and the view used to drop it.
    """
    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "mhc_i": {
                "state": "clear",
                "measurement": 0.8,
                "blocker": None,
                "source": "measured",
                "reference_density": 1.2,
                "units": "binders per 100 windows",
            }
        },
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis["reference_density"] == 1.2
    assert axis["units"] == "binders per 100 windows"


def test_a_clear_says_how_much_of_the_scan_was_never_scored(tmp_path: Path) -> None:
    """Declined windows are not non-binders, and a clear rests on there being none."""
    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "mhc_i": {
                "state": "clear",
                "measurement": 0.0,
                "blocker": None,
                "source": "measured",
                "unscored_peptides": 41,
                "presented_complete": False,
            }
        },
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis["unscored_peptides"] == 41
    assert axis["presented_complete"] is False


def test_an_unread_affinity_column_is_named_in_the_verdict(tmp_path: Path) -> None:
    """The columns a verdict could not read are part of what it rests on."""
    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "mhc_i": {
                "state": "untested",
                "measurement": 2.0,
                "blocker": "ic50_not_nm",
                "source": "measured",
                "not_nm_columns": ["ic50", "affinity"],
            }
        },
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis["not_nm_columns"] == ["ic50", "affinity"]


def test_a_plasma_fold_verdict_carries_the_study_it_cites(tmp_path: Path) -> None:
    """That axis refuses to be measured without a study, so the study is the basis."""
    from rfproteina.disease import mcp_api

    run = tmp_path / "gene_1ABC_1"
    (run / "inference_samples" / "binder_0").mkdir(parents=True)
    (run / "screen.json").write_text(
        json.dumps(
            {
                "program": "Humanitys-Last-Disease",
                "call": "axis_hit",
                "bar": "target",
                "states": {"plasma_fold": "hit"},
                "hits": ["plasma_fold"],
                "untested": [],
                "target_liability": {
                    "plasma_fold": {
                        "state": "hit",
                        "measurement": 9.0,
                        "blocker": None,
                        "source": "measured",
                        "threshold": 2.0,
                        "fold_source": "Sun 2023",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["target_liability"]["plasma_fold"]
    assert axis["fold_source"] == "Sun 2023", "a fold verdict without its study is the rumour the axis refuses"


def test_a_clear_says_when_its_bar_may_be_a_hundredfold_out(tmp_path: Path) -> None:
    """The sharpest disclosure on the axis, and the view used to drop it.

    Every percentile rank in a binders-only reference reads the same whether the
    column is a percentage or a fraction of one. The verdict stands -- refusing
    would refuse the ordinary export -- so the disclosure is the only thing
    standing between a reader and a bar that may be a hundredfold wrong.
    """
    from rfproteina.disease import mcp_api

    _screened_axes(
        tmp_path,
        {
            "mhc_i": {
                "state": "clear",
                "measurement": 0.8,
                "blocker": None,
                "source": "measured",
                "reference_density": 1.2,
                "reference_rank_scale_ambiguous": True,
                "reference_proteins_too_short": ["P0CG47"],
            }
        },
    )
    axis = mcp_api.read_verdict({"directory": str(tmp_path)})["runs"][0]["verdict"]["sequence_liability"]["mhc_i"]
    assert axis["state"] == "clear", "the disclosure must not change the verdict"
    assert axis["reference_rank_scale_ambiguous"] is True
    assert axis["reference_proteins_too_short"] == ["P0CG47"]
