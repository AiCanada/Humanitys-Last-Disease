"""Disease-binder flow: token budget, name confirmation, structures, and shards."""

import json
import re
import shlex
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from rfproteina.disease.archive import save_generation
from rfproteina.disease.class_tokens import CLASS_CONDITION_DIM, class_condition, class_record, write_annotated_cif
from rfproteina.disease.classes import load_class_files
from rfproteina.disease.cli import build_parser
from rfproteina.disease.cli import main as cure_main
from rfproteina.disease.combinations import (
    AXES,
    LIGAND_AXES,
    PLASMA_AXES,
    PROTEIN_AXES,
    binder_kind_from_partner,
    combination_condition,
    combination_from_parsed,
    combination_record,
    judge_combination,
    ligand_states,
    parse_class_filename,
    requested_clear,
    target_is_panel_protein,
)
from rfproteina.disease.dispatch import (
    execute_generation,
    generation_requests,
    local_program,
    subprocess_runner,
)
from rfproteina.disease.jobs import write_job
from rfproteina.disease.key_store import (
    KeySlot,
    key_directory,
    load_api_keys,
    load_key_slots,
    load_model_choices,
    save_api_keys,
    save_model_choices,
)
from rfproteina.disease.limits import (
    DEFAULT_ALLOWANCE,
    USER_AGENT,
    allowance_for,
    slot,
)
from rfproteina.disease.logs import redact_fields
from rfproteina.disease.lookup import Disease, DiseaseNotFoundError, confirm_disease, match_diseases
from rfproteina.disease.machines import (
    discover_remote_machines,
    machine_name_from_explorer,
    pick_remote_machine,
    ssh_directory,
)
from rfproteina.disease.research import (
    CHAT_MAX_OUTPUT,
    CHAT_SEED,
    PROVIDERS,
    REASONING_MAX_OUTPUT,
    REPORT_NOTE,
    RESEARCH_LIMIT,
    SOURCE_ORDER,
    TTD_TARGETS_URL,
    LookupMissError,
    SourceHit,
    api_key,
    base_model,
    checked_key,
    chosen_providers,
    collect_record_scores,
    compare_search_options,
    complete_chat,
    format_missing_report,
    format_option_comparison,
    format_structure_ranking,
    is_offered_model,
    key_for_model,
    llm_results,
    merge_found_by,
    missing_structures,
    model_choices,
    model_display,
    rank_structures,
    record_share,
    research_subjects,
    resolve_chat_model,
    result_models,
    structures_by_database,
    structures_by_model,
    target_value,
    usable_providers,
)
from rfproteina.disease.screen import (
    format_screen_lines,
    screen_generation,
    screen_lines_for_directory,
    screen_reports_for_directory,
)
from rfproteina.disease.shards import (
    GENERATION_MODELS,
    generation_commands,
    model_checkpoint,
    model_overrides,
    shard_for_hosts,
)
from rfproteina.disease.sources import (
    GENE_ATTRIBUTE,
    MAX_ENTRY_WORKERS,
    OPENTARGETS_URL,
    RCSB_SEARCH_URL,
    UNIPROT_ATTRIBUTE,
    _reactome_for_seeds,
    accession_from_fasta,
    accessions_from_targets,
    alphafold_api_url,
    cif_url,
    collect_structure_options,
    diseases_from_search,
    entry_url,
    fetch_structure_cif,
    options_from_rcsb,
    post_json,
    rcsb_accession_query,
)
from rfproteina.disease.structures import (
    RecordScore,
    StructureOption,
    apply_structure_view,
    campaign_pick_numbers,
    download_selected,
    group_options,
    options_from_records,
    save_returned_structures,
    select_options,
    structure_selector,
)
from rfproteina.disease.tokens import (
    DISEASE_AE_CROP,
    DISEASE_DENOISER_CROP,
    DISEASE_PROCESSING_CAP,
    EXTRA_TOKENS,
    RELEASED_PROCESSING_CAP,
    check_budget,
)
from rfproteina.disease.ui import (
    NAV_SLOT,
    PANEL_TITLE,
    BrowserSession,
    OptionForm,
    OptionPanel,
    StepBackError,
    _done_page,
    _open_browser,
    _slots_from_fields,
    _with_step,
    back_target,
    choose_output_entry,
    display_directory,
    draft_from_fields,
    form_from_fields,
    generation_model_from_fields,
    list_output_entries,
    parse_submission,
    render_confirm,
    render_disease_choice,
    render_inference,
    render_option_comparison,
    render_options,
    render_structures,
    render_targets,
    research_from_fields,
    save_upload,
    wants_option_ui,
    with_nav,
)
from tests.helpers.disease import (  # noqa: F401
    ROW_P00533,
    ROW_P35225,
    UBIQUITIN,
    asthma_catalog,
    block_live_calls,
    no_developer_proteome,
    refuse_prompt,
    structures_json,
)
from tests.helpers.fixtures import data_path  # noqa: F401
from tests.helpers.inference import compose_inference

CATALOG = [
    Disease("Asthma", "EFO_0000270"),
    Disease("Breast cancer", "EFO_0000305"),
    Disease("Breast carcinoma", "EFO_0000306"),
]

#: The landing fields the machine and upload tests share, before the ones each test is about.
_PANEL_FIELDS = {"disease": ["Asthma"], "binder_kind": ["protein"], "out_dir": ["out"], "remote_dir": ["disease_runs"]}

#: Every provider key variable the landing page offers to stand in for.
_KEY_ENV_NAMES = (
    "XAI_API_KEY",
    "GROK_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "NVIDIA_API_KEY",
    "NVIDIA_NIM_API_KEY",
    "OPENROUTER_API_KEY",
)


def _clear_provider_keys(monkeypatch, *extra: str) -> None:
    """Unset every provider key variable, and ``extra``, so a test sees only the keys it sets."""
    for name in (*_KEY_ENV_NAMES, *extra):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolated_api_key_store(tmp_path, monkeypatch):
    monkeypatch.setenv("RFD4_KEY_DIR", str(tmp_path / "api-keys"))


def _option(group: str, entry: str, *, predicted: bool = False, methodology_known: bool = True) -> StructureOption:
    return StructureOption(
        group=group,
        accession=f"ACC-{entry}",
        entry_id=entry,
        chain="A",
        title=entry,
        predicted=predicted,
        source="fixture",
        methodology_known=methodology_known,
    )


def test_token_budget_adds_one_percent_and_the_class_is_a_label():
    check_budget()
    assert EXTRA_TOKENS == 5
    assert DISEASE_PROCESSING_CAP == RELEASED_PROCESSING_CAP + 5 == 516
    assert DISEASE_DENOISER_CROP == 389
    assert DISEASE_AE_CROP == 261
    clean = class_condition(4, "high_affinity_clean")
    assert clean.shape == (4, CLASS_CONDITION_DIM)
    assert CLASS_CONDITION_DIM == 2 * len(AXES) == 44
    assert np.array_equal(clean, combination_condition(4, requested_clear("protein")))
    record = class_record("high_affinity_clean")
    assert "ATOM" not in record
    assert "_rfproteina_disease.call requested_clear" in record
    assert "_rfproteina_disease.binder_kind protein" in record
    with pytest.raises(ValueError, match="training contrast"):
        class_condition(4, "high_affinity_toxic")


def test_disease_confirmation_accepts_one_name_lists_similar_and_rejects_unknown():
    assert [disease.name for disease in match_diseases("Asthma", CATALOG)] == ["Asthma"]
    confirmed = confirm_disease(
        "Asthma",
        CATALOG,
        confirm_one=lambda disease: disease.name == "Asthma",
        choose_one=lambda _hits: pytest.fail("a single hit must not offer a menu"),
    )
    assert confirmed.disease_id == "EFO_0000270"

    similar = match_diseases("breast", CATALOG)
    assert [disease.name for disease in similar] == ["Breast cancer", "Breast carcinoma"]
    picked = confirm_disease(
        "breast",
        CATALOG,
        confirm_one=lambda _disease: pytest.fail("similar names must be chosen"),
        choose_one=lambda hits: 1,
    )
    assert picked.name == "Breast carcinoma"

    with pytest.raises(DiseaseNotFoundError, match="Disease not found"):
        confirm_disease("not-a-disease", CATALOG, confirm_one=lambda _d: True, choose_one=lambda _h: 0)


def test_a_returned_structures_file_loads_the_structure_rows():
    """The saved comparison file is the structure list, plus score fields."""
    payload = {
        "version": 1,
        "structures": [
            {
                "number": 1,
                "group": "protein",
                "accession": "P20930",
                "entry_id": "4PCW",
                "chain": "1",
                "title": "filaggrin",
                "predicted": False,
                "methodology_known": True,
                "kind": "experimental",
                "source": "rcsb",
                "campaigns": ["Proteins", "Small Molecules"],
                "program_score": 6,
                "instances": 153,
                "databases": 6,
                "sources_asked": 8,
            }
        ],
    }
    rows = options_from_records(payload)
    assert len(rows) == 1
    assert rows[0].entry_id == "4PCW"
    assert rows[0].chain == "1"
    assert rows[0].campaigns == ("Proteins", "Small Molecules")
    assert rows[0].predicted is False


def test_structure_groups_and_download_only_the_selection(tmp_path: Path):
    records = [
        _option("protein", "1abc"),
        _option("dna", "1dna"),
        _option("rna", "1rna"),
        _option("gene", "1gen"),
        _option("protein", "9zzz", predicted=True),
    ]
    grouped = group_options(records)
    assert list(grouped) == ["protein", "dna", "rna", "gene"]
    assert [row.entry_id for row in grouped["protein"]] == ["1abc", "9zzz"]

    experimental = select_options(grouped, None)
    assert [row.entry_id for row in experimental] == ["1abc", "1dna", "1rna", "1gen"]

    chosen = [grouped["protein"][0], grouped["rna"][0], grouped["protein"][1]]
    fetched: list[str] = []
    notes: list[tuple[int, int, str]] = []

    def fetch(record: StructureOption) -> bytes:
        fetched.append(record.entry_id)
        return f"data_{record.entry_id}\n".encode()

    written = download_selected(
        select_options(grouped, chosen),
        tmp_path,
        fetch,
        progress=lambda done, total, label: notes.append((done, total, label)),
    )
    assert fetched == ["1abc", "9zzz", "1rna"]
    assert notes == [
        (0, 3, "1abc chain A"),
        (1, 3, "9zzz chain A"),
        (2, 3, "1rna chain A"),
        (3, 3, "1rna chain A"),
    ]
    assert [path.name for path in written] == [
        "protein_1abc_A.cif",
        "protein_9zzz_A.cif",
        "rna_1rna_A.cif",
    ]
    assert not (tmp_path / "dna_1dna_A.cif").exists()


def test_two_hosts_split_the_selection_without_starting_generation(tmp_path: Path):
    structures = [_option("protein", "1abc"), _option("dna", "1dna"), _option("gene", "1gen")]
    shards = shard_for_hosts(structures, ["host-a", "host-b"])
    assert [shard.host for shard in shards] == ["host-a", "host-b"]
    assert [row.entry_id for row in shards[0].structures] == ["1abc", "1gen"]
    assert [row.entry_id for row in shards[1].structures] == ["1dna"]
    assert all(shard.generation_started is False for shard in shards)
    commands = generation_commands(shards, str(tmp_path))
    assert len(commands) == 3
    assert all(command[0] == "rfd4-generate" for command in commands)
    assert all("inference_experiment=inference_on_user_inputs" in command for command in commands)
    assert all("inference_experiment=disease_binder" not in command for command in commands)
    assert all(command[2].endswith(".annotated.cif") for command in commands)
    present = tmp_path / "modified.ckpt"
    present.write_bytes(b"not-a-checkpoint")
    modified = generation_commands(shards, str(tmp_path), modified_ckpt=present)
    assert all("inference_experiment=disease_binder" in command for command in modified)
    assert all("hf_ckpt_path=null" in command for command in modified)
    assert all(f"ckpt_name={present.name}" in command for command in modified)
    missing = generation_commands(shards, str(tmp_path), modified_ckpt=tmp_path / "absent.ckpt")
    assert missing == commands


def test_the_generate_command_is_found_beside_its_own_interpreter(tmp_path: Path):
    """The reported error: No such file or directory: 'rfd4-generate'.

    It is a console script of the environment that installed this package, so it
    is on PATH only while that environment is activated. The panel is normally
    started by its own path -- ``.venv/bin/human``, or ``.venv/bin/python -m`` --
    and neither puts ``.venv/bin`` on PATH, which left the script sitting beside
    the running interpreter and still not found.
    """
    beside = Path(sys.executable).parent / "rfd4-generate"
    if beside.is_file():
        # The real environment: resolvable whether or not it was activated.
        assert local_program("rfd4-generate") == str(beside)
        assert Path(local_program("rfd4-generate")).is_file()

    # A name carrying a directory is already a path, so it is handed back as given.
    named = tmp_path / "rfd4-generate"
    named.write_text("#!/bin/sh\n", encoding="utf-8")
    assert local_program(str(named)) == str(named)

    # A name found nowhere is handed back too, so the error still says what is
    # missing instead of a path that was guessed at.
    assert local_program("definitely-not-installed-xyz") == "definitely-not-installed-xyz"

    # Nothing to run is a refusal, not an opaque IndexError.
    with pytest.raises(ValueError, match="needs a program"):
        subprocess_runner([])


def test_one_of_the_two_models_generates(tmp_path: Path):
    """The requested choice: the original release model, or the modified one.

    One of the two generates, so the output directory stays named after the
    structure and the commands differ only in the model they load.
    """
    assert GENERATION_MODELS == ("original", "modified")
    checkpoint = tmp_path / "disease.ckpt"
    checkpoint.write_bytes(b"not-a-checkpoint")

    assert model_checkpoint("original", None) is None
    assert model_checkpoint("original", checkpoint) is None, "the original loads no checkpoint"
    assert model_checkpoint("modified", checkpoint) == checkpoint

    # The modified model is not quietly swapped for the release model.
    for absent in (None, tmp_path / "absent.ckpt"):
        with pytest.raises(ValueError, match="needs a checkpoint file"):
            model_checkpoint("modified", absent)
    with pytest.raises(ValueError, match="Unknown generation model"):
        model_checkpoint("both", checkpoint)
    with pytest.raises(ValueError, match="Unknown generation model"):
        model_checkpoint("", checkpoint)

    # Each model plans one command per structure, into the same directory, because
    # only one of them runs.
    shards = shard_for_hosts([_option("protein", "1abc")], ["vm-a"])
    planned = {
        name: generation_requests(
            shards,
            str(tmp_path),
            modified_ckpt=model_checkpoint(name, checkpoint),
            remote_dir="remote_runs",
        )
        for name in GENERATION_MODELS
    }
    assert [len(requests) for requests in planned.values()] == [1, 1]
    original = planned["original"][0]
    modified = planned["modified"][0]
    assert "inference_experiment=inference_on_user_inputs" in original.argv
    assert "inference_experiment=disease_binder" in modified.argv
    assert f"ckpt_name={checkpoint.name}" in modified.argv
    assert all("ckpt_name" not in part for part in original.argv)
    # Same structure, so the same input and the same output directory.
    assert f"out_dir={tmp_path}/protein_1abc_A" in original.argv
    assert f"out_dir={tmp_path}/protein_1abc_A" in modified.argv
    assert original.remote_out_dir == modified.remote_out_dir == "remote_runs/protein_1abc_A"
    assert original.local_input == modified.local_input


def test_the_generation_page_offers_the_model_it_can_actually_load():
    """The choice is on the page Download leads to, and it is honest about both.

    Without a checkpoint there is nothing for the modified model to load, so the
    page holds those two back and says why rather than offering a choice it
    cannot carry out.
    """
    page = render_inference(1)
    assert "<h3>Generate with</h3>" in page
    assert 'type="radio" name="generate_with" value="original" checked' in page
    assert ">\nOriginal RFD4</label>" in page
    assert 'type="radio" name="generate_with" value="modified" disabled' in page
    assert ">\nModified Low Toxic</label>" in page
    assert "Modified Low Toxic needs a checkpoint file" in page

    # Two choices, so there is no third.
    assert page.count('name="generate_with"') == 2
    assert 'value="both"' not in page
    assert ">\nBoth</label>" not in page

    # With a checkpoint, the modified model is the selected choice, because that is
    # what naming one has always meant, and neither choice is held back.
    named = render_inference(1, modified="/data/disease.ckpt")
    assert 'type="radio" name="generate_with" value="original">' in named
    assert 'type="radio" name="generate_with" value="modified" checked' in named
    assert "Modified Low Toxic loads /data/disease.ckpt." in named
    assert 'name="generate_with" value="modified" disabled' not in named
    assert named.count('name="generate_with"') == 2

    # The field, including the answer a page that was never asked leaves behind.
    assert generation_model_from_fields({}) == "original"
    assert generation_model_from_fields({}, modified="/data/disease.ckpt") == "modified"
    assert generation_model_from_fields({"generate_with": ["original"]}) == "original"
    assert generation_model_from_fields({"generate_with": ["modified"]}, modified="/data/disease.ckpt") == "modified"
    with pytest.raises(ValueError, match="no modified checkpoint"):
        generation_model_from_fields({"generate_with": ["modified"]})
    for refused in ("both", "nonsense"):
        with pytest.raises(ValueError, match="Unknown generation model"):
            generation_model_from_fields({"generate_with": [refused]}, modified="/data/disease.ckpt")


def test_job_attaches_the_good_class_and_leaves_generation_off(tmp_path: Path):
    source = tmp_path / "protein_1abc_A.cif"
    source.write_text("data_source\n_entry.id 1ABC\n", encoding="utf-8")
    spec = write_job(source, tmp_path)
    text = Path(spec["annotated_cif"]).read_text(encoding="utf-8")
    assert "data_source" in text
    assert "ATOM" not in text
    assert "_rfproteina_disease.call requested_clear" in text
    assert "_rfproteina_disease.binder_kind protein" in text
    assert spec["class_name"] == "requested_clear"
    assert spec["binder_kind"] == "protein"
    assert spec["class_condition_dim"] == CLASS_CONDITION_DIM
    assert spec["design_spec"] is None
    assert spec["search"] is None
    assert spec["generation_started"] is False
    assert not list(tmp_path.glob("*.yaml"))
    with pytest.raises(ValueError, match="training contrast"):
        write_annotated_cif(source, tmp_path / "toxic.cif", "high_affinity_toxic")


def _atom_line(group: str, serial: int, atom: str, asym: str, entity: str, seq: int | None, auth: str) -> str:
    seq_token = "." if seq is None else str(seq)
    return f"{group} {serial} {atom} {asym} {entity} {seq_token} {auth}"


def _polymer_cif() -> str:
    """Entity 1 is chain A (author H), residues 10-11. Longer chain B must not win that match."""
    lines = [
        "data_4PCW",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.label_atom_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.auth_asym_id",
    ]
    serial = 1

    def add(group: str, atom: str, asym: str, entity: str, seq: int | None, auth: str) -> None:
        nonlocal serial
        lines.append(_atom_line(group, serial, atom, asym, entity, seq, auth))
        serial += 1

    for seq in (10, 11):
        for atom in ("N", "CA", "C", "O"):
            add("ATOM", atom, "A", "1", seq, "H")
    for seq in (1, 2, 3):
        for atom in ("N", "CA", "C", "O"):
            add("ATOM", atom, "B", "3", seq, "B")
    add("HETATM", "C", "L", "9", None, "L")
    for atom in ("N", "CA", "C", "O"):
        add("ATOM", atom, "C", "4", 5, "C")
    for atom in ("N", "CA", "C"):
        add("ATOM", atom, "D", "6", 8, "D")
    for atom in ("N", "CA", "C", "O"):
        add("ATOM", atom, "1", "5", 7, "1")
    for seq in (10, 12):
        for atom in ("N", "CA", "C", "O"):
            add("ATOM", atom, "E", "7", seq, "E")
    lines.append("#")
    return "\n".join(lines) + "\n"


def test_a_protein_polymer_writes_a_fixed_target_condition_spec(tmp_path: Path):
    source = tmp_path / "protein_4PCW_1.cif"
    source.write_text(_polymer_cif(), encoding="utf-8")
    spec = write_job(source, tmp_path, binder_kind="peptide", target_id="P20930", chain="1")
    design = Path(spec["design_spec"])
    text = design.read_text(encoding="utf-8")
    assert design.name == "protein_4PCW_1.yaml"
    assert "No hotspot is set because the deposit does not name an interface." in text
    assert 'input: "protein_4PCW_1.cif"' in text
    assert 'contig: "8-20, /0, A10-11"' in text
    assert "C_CRDGRP" in text
    assert "sel('A/*/*/[N,CA,C,O]') and not has_nan_coord()" in text
    assert "C_SEQ" in text
    assert "sel('A')" in text
    assert "C_HOT" not in text
    assert "ATOM" not in text
    annotated = Path(spec["annotated_cif"]).read_text(encoding="utf-8")
    assert [line for line in annotated.splitlines() if line.startswith("data_")] == ["data_4PCW"]
    assert "_rfproteina_disease.call requested_clear" in annotated
    option = StructureOption("protein", "P20930", "4PCW", "1", "target", False, "fixture")
    commands = generation_commands(shard_for_hosts([option], ["local"]), str(tmp_path), binder_kind="peptide")
    assert commands[0][2].endswith("protein_4PCW_1.yaml")
    assert "disease_binder_kind" not in commands[0][1]

    protein_dir = tmp_path / "protein"
    protein_source = protein_dir / "protein_4PCW_1.cif"
    protein_dir.mkdir()
    protein_source.write_text(_polymer_cif(), encoding="utf-8")
    protein_spec = write_job(protein_source, protein_dir, binder_kind="protein", chain="1")
    assert 'contig: "60-110, /0, A10-11"' in Path(protein_spec["design_spec"]).read_text(encoding="utf-8")

    author = tmp_path / "author"
    author.mkdir()
    author_source = author / "protein_4PCW_H.cif"
    author_source.write_text(_polymer_cif(), encoding="utf-8")
    author_spec = write_job(author_source, author, binder_kind="peptide", chain="H")
    assert 'contig: "8-20, /0, A10-11"' in Path(author_spec["design_spec"]).read_text(encoding="utf-8")

    single = tmp_path / "single"
    single.mkdir()
    single_source = single / "protein_4PCW_4.cif"
    single_source.write_text(_polymer_cif(), encoding="utf-8")
    single_text = Path(write_job(single_source, single, binder_kind="peptide", chain="4")["design_spec"]).read_text(
        encoding="utf-8"
    )
    assert 'contig: "8-20, /0, C5"' in single_text
    assert "C5-" not in single_text

    gapped = tmp_path / "gapped"
    gapped.mkdir()
    gapped_source = gapped / "protein_4PCW_7.cif"
    gapped_source.write_text(_polymer_cif(), encoding="utf-8")
    gapped_text = Path(write_job(gapped_source, gapped, binder_kind="protein", chain="7")["design_spec"]).read_text(
        encoding="utf-8"
    )
    assert 'contig: "60-110, /0, E10-12"' in gapped_text

    for kind in ("ligand", "nucleic"):
        kind_dir = tmp_path / kind
        kind_dir.mkdir()
        kind_source = kind_dir / "protein_4PCW_1.cif"
        kind_source.write_text(_polymer_cif(), encoding="utf-8")
        kind_spec = write_job(kind_source, kind_dir, binder_kind=kind, chain="1")
        assert kind_spec["design_spec"] is None
        assert list(kind_dir.glob("*.yaml")) == []
        kind_option = StructureOption("protein", "P20930", "4PCW", "1", "target", False, "fixture")
        kind_commands = generation_commands(shard_for_hosts([kind_option], ["local"]), str(kind_dir))
        assert kind_commands[0][2].endswith("protein_4PCW_1.annotated.cif")

    for chain in ("5", "6", "Z"):
        skipped = write_job(source, tmp_path / chain, binder_kind="protein", chain=chain)
        assert skipped["design_spec"] is None


def test_missing_class_directory_reports_the_given_path(tmp_path: Path):
    missing = tmp_path / "not-built"
    # ``match`` is a regular expression, and a Windows path carries escape sequences that
    # are invalid in one, so the path must be escaped before it is used as a pattern.
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        load_class_files(missing)
    ready = tmp_path / "classes"
    ready.mkdir()
    (ready / "high_affinity_clean_a.cif").write_text("clean", encoding="utf-8")
    (ready / "high_affinity_toxic_b.cif").write_text("toxic", encoding="utf-8")
    (ready / "notes.txt").write_text("skip", encoding="utf-8")
    limited = load_class_files(ready, file_limit=1)
    assert [item.path.name for item in limited] == ["high_affinity_clean_a.cif"]


def test_global_inference_cap_stays_511_and_the_disease_experiment_uses_516():
    base = compose_inference("inputs=x")
    disease = compose_inference("inference_experiment=disease_binder", "inputs=x")
    assert base.bioir.cuda_graph_max_tokens == 511
    assert disease.bioir.cuda_graph_max_tokens == 516
    assert disease.search.checkpoints is None
    assert not hasattr(disease.search, "rewards")


def test_source_parsers_read_disease_hits_and_structure_groups():
    diseases = diseases_from_search(
        {"data": {"search": {"hits": [{"id": "EFO_1", "name": "Asthma", "entity": "disease"}]}}}
    )
    assert [(disease.name, disease.disease_id) for disease in diseases] == [("Asthma", "EFO_1")]
    options = options_from_rcsb(
        {
            "result_set": [
                {"identifier": "1abc_A", "polymer": "protein", "title": "target"},
                {"identifier": "1dna_B", "polymer": "dna", "title": "dna contact"},
                {"identifier": "1rna_C", "polymer": "rna", "title": "rna contact"},
                {"identifier": "1gen_A", "polymer": "gene", "title": "partner"},
            ]
        },
        accession="P12345",
        group="protein",
    )
    assert [option.group for option in options] == ["protein", "dna", "rna", "gene"]


def _axis(row: np.ndarray, axis: str) -> tuple[float, float]:
    index = AXES.index(axis)
    return float(row[index]), float(row[len(AXES) + index])


def test_ligand_hit_stays_on_the_measured_panel_protein():
    states = ligand_states(
        {"herg": 10.0, "cyp3a4": None, "nav1_5": 10_000, "pgp": 10_001, "cyp2d6": 1.0},
        inconclusive={"cyp2d6"},
    )
    assert states["herg"] == "hit"
    assert states["nav1_5"] == "hit"
    assert states["pgp"] == "clear"
    assert states["cyp3a4"] == "untested"
    assert states["cyp2d6"] == "untested"
    judged = judge_combination("ligand", states, partner_id="Q12809")
    assert judged.call == "axis_hit"
    assert judged.hits == ("herg", "nav1_5")
    assert "cyp3a4" in judged.untested
    assert "immunogenicity" not in judged.untested
    row = combination_condition(2, judged)[0]
    assert _axis(row, "herg") == (-1.0, 1.0)
    assert _axis(row, "cyp3a4") == (0.0, 1.0)
    assert _axis(row, "immunogenicity") == (0.0, 0.0)
    record = combination_record(judged)
    assert "ATOM" not in record
    assert "_rfproteina_disease.herg hit" in record
    assert "_rfproteina_disease.cyp3a4 untested" in record
    assert "_rfproteina_disease.partner_id Q12809" in record
    with pytest.raises(ValueError, match="training contrast"):
        write_annotated_cif(Path("unused"), Path("unused"), judged)


def test_untested_and_partial_screens_are_not_a_clear_combination():
    absent = judge_combination("ligand", {})
    assert absent.call == "untested"
    coverage = judge_combination(
        "ligand",
        ligand_states({"herg": 50_000, "pgp": 50_000, "bcrp": 50_000}),
        bar="coverage",
    )
    assert coverage.call == "screened_clear"
    assert dict(coverage.states)["cyp3a4"] == "untested"
    assert "plasma_abundance" in coverage.untested
    one_axis = combination_from_parsed(parse_class_filename("high_affinity_herg_clean.cif"))
    assert one_axis is not None
    assert one_axis.call == "untested"
    assert dict(one_axis.states)["herg"] == "clear"
    requested = requested_clear("ligand")
    assert requested.call == "requested_clear"
    assert not np.array_equal(combination_condition(1, coverage), combination_condition(1, requested))
    coarse = combination_from_parsed(parse_class_filename("high_affinity_toxic.cif"))
    assert coarse.call == "coarse"
    assert coarse.hits == ()
    with pytest.raises(ValueError, match="which binder"):
        combination_condition(1, coarse)


def test_protein_peptide_and_nucleic_use_different_axes():
    shared = dict.fromkeys(PROTEIN_AXES, "clear")
    protein = judge_combination("protein", shared, bar="sequence")
    peptide = judge_combination("peptide", shared, bar="sequence")
    assert protein.call == "screened_clear"
    assert peptide.call == "untested"
    assert "membrane" in peptide.untested
    peptide_clear = judge_combination("peptide", {**shared, "membrane": "clear"}, bar="sequence")
    assert peptide_clear.call == "screened_clear"
    assert "plasma_abundance" in peptide_clear.untested
    assert binder_kind_from_partner("protein", 20) == "peptide"
    assert binder_kind_from_partner("protein", 21) == "protein"
    assert binder_kind_from_partner("small_molecule") == "ligand"
    assert binder_kind_from_partner("dna") == "nucleic"
    assert binder_kind_from_partner("glycan") == "other"
    with pytest.raises(ValueError, match="nucleic"):
        judge_combination("nucleic", {"herg": "hit"})
    nucleic = requested_clear("nucleic")
    row = combination_condition(1, nucleic)[0]
    assert _axis(row, "off_target_hybridization") == (1.0, 1.0)
    assert _axis(row, "herg") == (0.0, 0.0)
    other = judge_combination("other", {})
    assert other.call == "untested"
    with pytest.raises(ValueError, match="other"):
        requested_clear("other")


def test_a_safety_panel_target_is_not_marked_clear():
    assert target_is_panel_protein("Q12809")
    assert target_is_panel_protein("CYP3A4")
    assert not target_is_panel_protein("P00533")
    assert not target_is_panel_protein("")
    with pytest.raises(ValueError, match="safety panel"):
        requested_clear("protein", target_id="Q12809")
    judged = judge_combination("ligand", dict.fromkeys(LIGAND_AXES, "clear"), target_id="hERG")
    assert judged.call == "target_is_panel"
    assert judged.hits == ()


def test_class_filenames_keep_coarse_names_and_one_axis(tmp_path: Path):
    ready = tmp_path / "classes"
    ready.mkdir()
    (ready / "high_affinity_herg_toxic.cif").write_text("herg", encoding="utf-8")
    (ready / "high_affinity_ligand_plasma_abundance_toxic.cif").write_text("plasma", encoding="utf-8")
    (ready / "notes.txt").write_text("skip", encoding="utf-8")
    loaded = load_class_files(ready)
    by_name = {item.path.name: item for item in loaded}
    herg = by_name["high_affinity_herg_toxic.cif"]
    assert herg.binder_kind == "ligand"
    assert herg.axis == "herg"
    assert herg.state == "hit"
    assert herg.coarse is False
    plasma = by_name["high_affinity_ligand_plasma_abundance_toxic.cif"]
    assert plasma.axis == "plasma_abundance"
    assert plasma.binder_kind == "ligand"
    with pytest.raises(ValueError, match="binder kind"):
        load_class_files_ambiguous = tmp_path / "ambiguous"
        load_class_files_ambiguous.mkdir()
        (load_class_files_ambiguous / "high_affinity_mhc_i_toxic.cif").write_text("shared", encoding="utf-8")
        load_class_files(load_class_files_ambiguous)


def test_training_configs_record_the_combination_width():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "configs/experiment/disease_affinity_tox_denoiser.yaml",
        "configs/experiment_ae/disease_affinity_tox_ae.yaml",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert f"disease_class_condition_dim: {CLASS_CONDITION_DIM}" in text


def _associated_targets(*targets: tuple[str, str], source: str = "uniprot") -> dict:
    """Open Targets' associatedTargets answer, one row per (symbol, accession); "" lists no protein."""
    rows = [
        {"target": {"approvedSymbol": symbol, "proteinIds": [{"id": accession, "source": source}] if accession else []}}
        for symbol, accession in targets
    ]
    return {"data": {"disease": {"associatedTargets": {"rows": rows}}}}


def _structure_post(url: str, payload: dict) -> dict:
    if url == OPENTARGETS_URL:
        return _associated_targets(("IL13", "P35225"), source="uniprot_swissprot")
    assert url == RCSB_SEARCH_URL
    parameters = payload["query"]["parameters"]
    assert parameters["operator"] == "exact_match"
    assert payload["request_options"]["results_content_type"] == ["experimental", "computational"]
    attribute = parameters["attribute"]
    value = parameters["value"]
    if attribute == UNIPROT_ATTRIBUTE and value == "P35225":
        return {"result_set": [{"identifier": "1ABC_1"}]}
    if attribute == GENE_ATTRIBUTE and value == "IL13":
        return {"result_set": [{"identifier": "1GEN_1"}]}
    raise AssertionError((attribute, value))


def _structure_get(url: str) -> dict:
    if url.endswith("/1ABC"):
        return {
            "struct": {"title": "interleukin"},
            "rcsb_entry_info": {
                "structure_determination_methodology": "experimental",
                "polymer_entity_count_DNA": 1,
            },
        }
    if url.endswith("/1GEN"):
        return {
            "struct": {"title": "model"},
            "rcsb_entry_info": {"structure_determination_methodology": "computational"},
        }
    raise AssertionError(url)


class _CureRun(NamedTuple):
    """One terminal run. It unpacks as five values, and ``out``/``err`` read like ``capsys.readouterr()``."""

    code: int
    out_dir: Path
    fetched: list[tuple[str, str]]
    out: str
    err: str


def _run_cure(
    tmp_path: Path,
    monkeypatch,
    capsys,
    extra: list[str],
    read_line=refuse_prompt,
    *,
    out_dir: Path | None = None,
    **seams,
) -> _CureRun:
    """Run the Asthma catalog from the terminal, accepting its one disease.

    The structure search is answered by ``_structure_post`` and ``_structure_get``. A run handed its
    records with ``--structures-json`` needs no search, so it keeps the program's own ``post`` and
    ``get``, which ``block_live_calls`` turns into failures. Any other seam passes through, and
    ``fetch`` defaults to one that records what was downloaded.
    """
    block_live_calls(monkeypatch)
    fetched: list[tuple[str, str]] = []

    def fetch(option: StructureOption) -> bytes:
        fetched.append((option.group, option.entry_id))
        return b"data_source\n"

    seams.setdefault("fetch", fetch)
    if "--structures-json" not in extra:
        seams.setdefault("post", _structure_post)
        seams.setdefault("get", _structure_get)
    out_dir = out_dir or tmp_path / "runs"
    code = cure_main(
        [
            "--catalog",
            str(asthma_catalog(tmp_path)),
            "--disease",
            "Asthma",
            "--accept-only",
            "--out-dir",
            str(out_dir),
            *extra,
        ],
        read_line=read_line,
        **seams,
    )
    captured = capsys.readouterr()
    return _CureRun(code, out_dir, fetched, captured.out, captured.err)


def _run_panel(monkeypatch, exchange, *argv: str, **seams) -> int:
    """Run the browser panel with live calls refused. A terminal prompt fails unless ``read_line`` is given."""
    block_live_calls(monkeypatch)
    seams.setdefault("read_line", refuse_prompt)
    return cure_main(["--ui", *argv], exchange=exchange, **seams)


def _commands(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("rfd4-generate")]


def _decline_inference(prompt: str) -> str:
    if "Confirm inference" not in prompt:
        raise AssertionError(prompt)
    return "n"


def test_cure_downloads_experimental_structures_and_leaves_generation_unstarted(tmp_path, monkeypatch, capsys):
    code, out_dir, fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--hosts", "host-a"],
        read_line=refuse_prompt,
    )
    assert code == 0
    assert stderr == ""
    assert fetched == [("protein", "1ABC"), ("dna", "1ABC")]
    assert "  1. 1ABC chain 1 P35225 interleukin" in stdout
    assert "  2. 1ABC chain 1 P35225 interleukin" in stdout
    assert "  3. 1GEN chain 1 IL13 model predicted" in stdout
    assert not (out_dir / "gene_1GEN_1.cif").exists()
    commands = _commands(stdout)
    assert [line for line in commands if "protein_1ABC_1.annotated.cif" in line]
    assert [line for line in commands if "dna_1ABC_1.annotated.cif" in line]
    assert all("1GEN" not in line for line in commands)
    assert all("+disease.host=host-a" in line for line in commands)
    assert "Using the original release model." in stdout
    assert all("inference_experiment=inference_on_user_inputs" in line for line in commands)
    assert all("disease_binder" not in line for line in commands)
    assert stdout.strip().endswith("Generation was not started.")
    for name in ("protein_1ABC_1", "dna_1ABC_1"):
        annotated = (out_dir / f"{name}.annotated.cif").read_text(encoding="utf-8")
        assert "data_source" in annotated
        assert "_rfproteina_disease.call requested_clear" in annotated
        spec = json.loads((out_dir / f"{name}.job.json").read_text(encoding="utf-8"))
        assert spec["generation_started"] is False
        assert spec["search"] is None
        assert spec["class_name"] == "requested_clear"


def test_cure_pick_downloads_a_predicted_row_and_all_can_add_it(tmp_path, monkeypatch, capsys):
    code, out_dir, fetched, stdout, _stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--pick", "3", "--confirm-inference"],
        read_line=refuse_prompt,
    )
    assert code == 0
    assert fetched == [("gene", "1GEN")]
    assert (out_dir / "gene_1GEN_1.annotated.cif").is_file()
    assert not (out_dir / "protein_1ABC_1.cif").exists()
    assert len(_commands(stdout)) == 1
    assert "gene_1GEN_1.annotated.cif" in _commands(stdout)[0]
    assert "+disease.host=local" in _commands(stdout)[0]

    code, out_dir, fetched, stdout, _stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--pick", "3", "--confirm-inference"],
        read_line=refuse_prompt,
    )
    assert code == 0
    assert fetched == [("protein", "1ABC"), ("dna", "1ABC"), ("gene", "1GEN")]
    assert len(_commands(stdout)) == 3


def test_structure_selector_survives_reordered_live_options(tmp_path, monkeypatch, capsys):
    reported: dict[str, object] = {}

    def report(name: str, value: object) -> None:
        reported[name] = value

    records = structures_json(tmp_path, ROW_P00533, ROW_P35225)
    listed = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records)],
        report=report,
    )
    assert listed.code == 0 and listed.fetched == []
    choices = reported["structure_options"]
    assert isinstance(choices, list)
    selected = next(row for row in choices if row["entry_id"] == "1EGFR")
    selector = selected["selector"]
    assert isinstance(selector, str) and len(selector) == 64

    records = structures_json(tmp_path, ROW_P35225, ROW_P00533)
    chosen = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--select-structure", selector, "--confirm-inference"],
    )
    assert chosen.code == 0
    assert chosen.fetched == [("protein", "1EGFR")]


def test_stale_structure_selector_stops_before_download(tmp_path, monkeypatch, capsys):
    records = structures_json(tmp_path, ROW_P00533)
    result = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--select-structure", "0" * 64, "--confirm-inference"],
    )
    assert result.code == 2
    assert result.fetched == []
    assert "no longer listed" in result.err


def test_structure_selector_uses_structure_identity_not_display_order():
    original = _option("protein", "1ABC")
    relabelled = StructureOption(
        group=original.group,
        accession="another accession",
        entry_id=original.entry_id,
        chain=original.chain,
        title="another title",
        predicted=True,
        source="another source",
    )
    assert structure_selector(original) == structure_selector(relabelled)
    assert structure_selector(original) != structure_selector(_option("dna", "1ABC"))


def test_a_download_directory_keeps_the_structure_file(tmp_path, monkeypatch, capsys):
    """Structure files land in the download directory and are copied for the job."""
    download = tmp_path / "downloads"
    code, out_dir, fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--download-dir", str(download)],
        read_line=refuse_prompt,
    )
    assert code == 0
    assert stderr == ""
    assert fetched == [("protein", "1ABC"), ("dna", "1ABC")]
    assert (download / "protein_1ABC_1.cif").read_bytes() == b"data_source\n"
    assert (out_dir / "protein_1ABC_1.cif").read_bytes() == b"data_source\n"
    assert (out_dir / "protein_1ABC_1.job.json").is_file()
    assert stdout.strip().endswith("Generation was not started.")


def test_cure_declined_inference_keeps_the_download_and_writes_no_job(tmp_path, monkeypatch, capsys):
    code, out_dir, fetched, stdout, _stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all"],
        read_line=_decline_inference,
    )
    assert code == 0
    assert fetched == [("protein", "1ABC"), ("dna", "1ABC")]
    assert (out_dir / "protein_1ABC_1.cif").is_file()
    assert list(out_dir.glob("*.annotated.cif")) == []
    assert list(out_dir.glob("*.job.json")) == []
    assert "Inference was not confirmed. Generation was not started." in stdout
    assert _commands(stdout) == []


def test_cure_asks_for_a_choice_before_download(tmp_path, monkeypatch, capsys):
    code, out_dir, fetched, stdout, _stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        [],
        read_line=refuse_prompt,
    )
    assert code == 0
    assert fetched == []
    assert not out_dir.exists()
    assert "Pass --all, --pick, or --select-structure before download." in stdout
    assert _commands(stdout) == []


def test_cure_skips_a_safety_panel_structure(tmp_path, monkeypatch, capsys):
    records = structures_json(
        tmp_path,
        {
            "group": "protein",
            "accession": "Q12809",
            "entry_id": "1HER",
            "chain": "A",
            "title": "herg",
            "predicted": False,
            "source": "fixture",
        },
        ROW_P00533,
    )
    code, out_dir, _fetched, stdout, _stderr = _run_cure(
        tmp_path, monkeypatch, capsys, ["--structures-json", str(records), "--all", "--confirm-inference"]
    )
    assert code == 0
    assert "Skipped 1HER:" in stdout
    assert "safety panel" in stdout
    assert (out_dir / "protein_1HER_A.cif").is_file()
    assert not (out_dir / "protein_1HER_A.job.json").exists()
    commands = _commands(stdout)
    assert len(commands) == 1
    assert "protein_1EGFR_A.annotated.cif" in commands[0]
    assert "1HER" not in commands[0]
    spec = json.loads((out_dir / "protein_1EGFR_A.job.json").read_text(encoding="utf-8"))
    assert spec["generation_started"] is False
    assert spec["target_id"] == "P00533"
    assert stdout.strip().endswith("Generation was not started.")


def test_cure_rejects_a_structure_number_outside_the_list(tmp_path, monkeypatch, capsys):
    code, out_dir, fetched, _stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--pick", "9"],
        read_line=refuse_prompt,
    )
    assert code == 2
    assert "outside" in stderr
    assert fetched == []
    assert not out_dir.exists()


def test_post_json_retries_a_blank_body(monkeypatch):
    bodies = [b"", b'{"data": {"ok": true}}']

    def fake_open(_request, timeout=45):
        assert timeout == 45
        return _Response(bodies.pop(0))

    monkeypatch.setattr("rfproteina.disease.sources.urlopen", fake_open)
    monkeypatch.setattr("rfproteina.disease.sources.time.sleep", lambda _seconds: None)
    assert post_json("https://example.test/lookup", {"q": 1}) == {"data": {"ok": True}}
    assert bodies == []


def test_post_json_reads_http_204_as_no_hits(monkeypatch):
    calls = {"n": 0}

    def fake_open(_request, timeout=45):
        assert timeout == 45
        calls["n"] += 1
        return _Response(status=204)

    monkeypatch.setattr("rfproteina.disease.sources.urlopen", fake_open)
    assert post_json("https://example.test/lookup", {"q": 1}) == {"result_set": [], "total_count": 0}
    assert calls["n"] == 1


def test_post_json_does_not_retry_a_json_list(monkeypatch):
    calls = {"n": 0}

    def fake_open(_request, timeout=45):
        assert timeout == 45
        calls["n"] += 1
        return _Response(b"[1]")

    monkeypatch.setattr("rfproteina.disease.sources.urlopen", fake_open)
    with pytest.raises(ValueError, match="Expected a JSON object"):
        post_json("https://example.test/lookup", {"q": 1})
    assert calls["n"] == 1


def test_rcsb_queries_keep_predicted_models_in_the_result_list():
    query = rcsb_accession_query("P00533", rows=3)
    assert query["return_type"] == "polymer_entity"
    assert query["request_options"]["paginate"] == {"start": 0, "rows": 3}
    assert query["request_options"]["results_content_type"] == ["experimental", "computational"]
    assert "sort" not in query["request_options"]
    seen: list[str] = []

    def read(url: str) -> bytes:
        seen.append(url)
        return b""

    with pytest.raises(ValueError, match="empty"):
        fetch_structure_cif(_option("protein", "1abc"), read=read)
    assert seen == [cif_url("1abc")]
    assert seen == ["https://files.rcsb.org/download/1ABC.cif"]


def test_an_af_id_is_fetched_from_alphafold_ebi():
    """A published AF id is the EBI model file. The sequence in that record is not used."""
    seen: list[str] = []

    def read(url: str) -> bytes:
        seen.append(url)
        return b"data_AF\n#\n"

    def read_json(url: str):
        seen.append(url)
        assert "MKT" not in url
        return [
            {
                "entryId": "AF-P00533-F1",
                "uniprotAccession": "P00533",
                "uniprotDescription": "EGFR",
                "chainId": "A",
                "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P00533-F1-model_v6.cif",
                "sequence": "MKTSHOULDNOTTRAVEL",
            }
        ]

    payload = fetch_structure_cif(_option("protein", "AF-P00533-F1", predicted=True), read=read, read_json=read_json)
    assert payload == b"data_AF\n#\n"
    assert seen == [
        "https://alphafold.ebi.ac.uk/api/prediction/P00533",
        "https://alphafold.ebi.ac.uk/files/AF-P00533-F1-model_v6.cif",
    ]
    assert "MKTSHOULDNOTTRAVEL" not in " ".join(seen)
    assert "files.rcsb.org" not in " ".join(seen)
    assert alphafold_api_url("AF-P00533-4-F1") == "https://alphafold.ebi.ac.uk/api/prediction/P00533"
    assert alphafold_api_url("AF-A0AB40D056-F2") == "https://alphafold.ebi.ac.uk/api/prediction/A0AB40D056"
    assert alphafold_api_url("AF-A0AB40D056-F3") == "https://alphafold.ebi.ac.uk/api/prediction/A0AB40D056"
    assert alphafold_api_url("AF3-A0AB40D056-F1") is None


def test_a_fasta_header_names_a_published_model_and_a_bare_sequence_does_not():
    """The header id is what is fetched. A sequence with no id is not sent anywhere."""
    assert accession_from_fasta(">sp|P00533|EGFR_HUMAN\nMKTLL\n") == "P00533"
    assert accession_from_fasta(">AF-P00533-F1 EGFR\nMKTLL\n") == "AF-P00533-F1"
    assert accession_from_fasta(">sp|A0AB40D056|\nMKT\n") == "A0AB40D056"
    assert accession_from_fasta(">AF-A0AB40D056-F2\nMKT\n") == "AF-A0AB40D056-F2"
    with pytest.raises(ValueError, match="alphafoldserver.com") as raised:
        accession_from_fasta(">query\nMKTLLVV\n")
    assert "MKTLLVV" not in str(raised.value)
    assert "alphafold.ebi.ac.uk" in str(raised.value)


def test_collect_structure_options_adds_nucleic_companions_and_keeps_a_missing_entry():
    def post(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL:
            return _associated_targets(("gene", "PHYB00"), ("EGFR", "P00533"))
        value = payload["query"]["parameters"]["value"]
        if value == "PHYB00":
            return {"result_set": [{"identifier": "3HYB_B"}, {"identifier": "9SKIP_A"}, {"identifier": "8SKIP"}]}
        if value == "P00533":
            return {"result_set": [{"identifier": "1MISS_A"}]}
        if value == "EGFR":
            return {"result_set": []}
        raise AssertionError(value)

    def get(url: str) -> dict:
        if url.endswith("/3HYB"):
            return {
                "struct": {"title": "hybrid"},
                "rcsb_entry_info": {
                    "structure_determination_methodology": "experimental",
                    "polymer_entity_count_nucleic_acid_hybrid": 1,
                },
            }
        raise OSError(url)

    found = collect_structure_options("EFO_1", post=post, get=get, row_limit=1)
    assert [(row.group, row.entry_id, row.chain, row.predicted) for row in found] == [
        ("protein", "3HYB", "B", False),
        ("dna", "3HYB", "B", False),
        ("rna", "3HYB", "B", False),
        ("protein", "1MISS", "A", False),
    ]
    # The unreachable entry is not known to be experimental, so it must not be selected as
    # though it were. Reading a failed lookup as experimental would let one flaky request
    # slip a computed model into a selection that asked for experimental structures.
    known = {row.entry_id: row.methodology_known for row in found}
    assert known == {"3HYB": True, "1MISS": False}
    assert select_options(group_options(found), None) == [row for row in found if row.entry_id == "3HYB"]


def test_computed_model_identifiers_keep_the_full_entry_id():
    def post(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL:
            return _associated_targets(("gene", "P20930"))
        assert payload["query"]["parameters"]["value"] == "P20930"
        return {"result_set": [{"identifier": "4PCW_1"}, {"identifier": "AF_AFP20930F10_1"}]}

    def get(url: str) -> dict:
        if url.endswith("/4PCW"):
            return {
                "struct": {"title": "profilaggrin"},
                "rcsb_entry_info": {"structure_determination_methodology": "experimental"},
            }
        if url.endswith("/AF_AFP20930F10"):
            return {
                "struct": {"title": "model"},
                "rcsb_entry_info": {"structure_determination_methodology": "computational"},
            }
        raise AssertionError(url)

    found = collect_structure_options("MONDO_0004979", post=post, get=get, row_limit=5)
    assert [(row.entry_id, row.chain, row.predicted, row.group) for row in found] == [
        ("4PCW", "1", False, "protein"),
        ("AF_AFP20930F10", "1", True, "protein"),
    ]


def test_collect_structure_options_stops_at_the_association_cap():
    seen: list[str] = []

    def post(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL:
            return _associated_targets(*((f"G{index}", f"P{index}") for index in range(9)))
        seen.append(payload["query"]["parameters"]["value"])
        return {"result_set": []}

    found = collect_structure_options(
        "EFO_1",
        post=post,
        get=lambda _url: {},
        association_limit=2,
        row_limit=1,
    )
    assert found == []
    assert seen == ["P0", "P1", "G0", "G1"]


def test_related_search_adds_pathway_members_partners_and_rna_genes():
    """Pathway members, interacting proteins, and an RNA gene add structure rows."""

    def post(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL:
            text = payload.get("query") if isinstance(payload.get("query"), str) else ""
            rows = [
                {
                    "target": {
                        "approvedSymbol": "EGFR",
                        "bioType": "protein_coding",
                        "proteinIds": [{"id": "P00533", "source": "uniprot"}],
                    }
                }
            ]
            if "DiseaseContext" in text:
                rows.append({"target": {"approvedSymbol": "MALAT1", "bioType": "lncRNA", "proteinIds": []}})
            return {"data": {"disease": {"associatedTargets": {"rows": rows}}}}
        value = payload["query"]["parameters"]["value"]
        return {"result_set": [{"identifier": f"1{value[-3:]}_A"}]}

    def read_value(url: str) -> object:
        if "string-db.org" in url and "P00533" in url:
            return [{"preferredName_B": "GRB2"}]
        if "string-db.org" in url:
            return []
        if url.endswith("/pathways"):
            return [{"stId": "R-HSA-177929", "displayName": "EGFR signaling"}]
        if "participants/R-HSA-177929" in url:
            return [{"databaseName": "UniProt", "identifier": "Q9Y6K9", "schemaClass": "ReferenceGeneProduct"}]
        raise AssertionError(url)

    def get(_url: str) -> dict:
        return {
            "struct": {"title": "nanobody"},
            "struct_keywords": {"text": "immune system"},
            "rcsb_entry_info": {
                "structure_determination_methodology": "experimental",
                "nonpolymer_entity_count": 1,
                "branched_entity_count": 1,
            },
        }

    found = collect_structure_options(
        "EFO_1",
        post=post,
        get=get,
        include_related=True,
        read_value=read_value,
        row_limit=5,
    )
    listed = {(row.group, row.accession) for row in found}
    assert ("protein", "P00533") in listed
    assert ("protein", "Q9Y6K9") in listed
    assert ("gene", "EGFR") in listed
    assert ("gene", "GRB2") in listed
    assert ("gene", "MALAT1") in listed
    protein = next(row for row in found if row.accession == "P00533")
    assert protein.campaigns == (
        "Proteins",
        "Small Molecules",
        "Sugars (Glycans)",
        "Antibodies & Nanobodies",
    )

    def fail_related(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL and "DiseaseContext" in str(payload.get("query")):
            raise ValueError("context unavailable")
        return post(url, payload)

    kept = collect_structure_options(
        "EFO_1",
        post=fail_related,
        get=get,
        include_related=True,
        read_value=lambda _url: (_ for _ in ()).throw(OSError("down")),
        row_limit=5,
    )
    assert {row.accession for row in kept} == {"P00533", "EGFR"}


def _job_started(out_dir: Path, name: str) -> bool:
    spec = json.loads((out_dir / f"{name}.job.json").read_text(encoding="utf-8"))
    return spec["generation_started"]


def test_cure_run_starts_this_machine_and_a_remote_machine_together(tmp_path, monkeypatch, capsys):
    seen: list[list[str]] = []
    transferred: list[tuple[str, str]] = []
    retrieved: list[tuple[str, str]] = []
    lock = threading.Lock()

    def runner(argv: list[str]) -> int:
        with lock:
            seen.append(list(argv))
        return 0 if argv[0] == "rfd4-generate" else 1

    def transfer(host: str, local_path: Path, remote_path: str) -> None:
        transferred.append((host, f"{local_path.name}|{remote_path}"))

    def retrieve(host: str, remote_out: str, local_out: Path) -> None:
        retrieved.append((host, f"{remote_out}|{local_out}"))

    code, out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--hosts", "local,vm-a", "--run", "--remote-dir", "/srv/disease"],
        read_line=refuse_prompt,
        runner=runner,
        transfer=transfer,
        retrieve=retrieve,
    )
    assert code == 1
    assert stderr == ""
    assert stdout.strip().endswith("Generation failed.")
    assert "Generation was not started." not in stdout
    assert "Starting 1 local and 1 remote generation request(s)." in stdout
    assert _job_started(out_dir, "protein_1ABC_1") is True
    assert _job_started(out_dir, "dna_1ABC_1") is True
    local = next(argv for argv in seen if argv[0] == "rfd4-generate")
    remote = next(argv for argv in seen if argv[0] == "ssh")
    assert remote[1] == "vm-a"
    remote_command = shlex.split(remote[2])
    assert remote_command[0] == "rfd4-generate"
    assert "inputs=/srv/disease/dna_1ABC_1.annotated.cif" in remote_command
    assert "out_dir=/srv/disease/dna_1ABC_1" in remote_command
    assert "+disease.host=vm-a" in remote_command
    assert any(part.startswith("inputs=") and "protein_1ABC_1.annotated.cif" in part for part in local)
    assert not any(part.startswith("inputs=/srv/") for part in local)
    assert transferred == [("vm-a", "dna_1ABC_1.annotated.cif|/srv/disease/dna_1ABC_1.annotated.cif")]
    assert retrieved == []


def test_a_remote_run_copies_the_condition_spec_and_its_structure(tmp_path, monkeypatch, capsys):
    records = structures_json(
        tmp_path,
        {
            "group": "protein",
            "accession": "P20930",
            "entry_id": "4PCW",
            "chain": "1",
            "title": "target",
            "predicted": False,
            "source": "fixture",
        },
    )
    transferred: list[tuple[str, str, str]] = []

    def fetch(_option: StructureOption) -> bytes:
        return _polymer_cif().encode()

    def runner(argv: list[str]) -> int:
        assert argv[0] == "ssh"
        assert "inputs=/srv/disease/protein_4PCW_1.yaml" in argv[2]
        assert "annotated.cif" not in argv[2]
        return 0

    def transfer(host: str, local_path: Path, remote_path: str) -> None:
        transferred.append((host, local_path.name, remote_path))

    def unused(*_args, **_kwargs):
        raise AssertionError("structure search")

    code, out_dir, _fetched, stdout, _stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        [
            "--structures-json",
            str(records),
            "--all",
            "--confirm-inference",
            "--binder-kind",
            "peptide",
            "--hosts",
            "vm-a",
            "--remote-dir",
            "/srv/disease",
            "--run",
        ],
        post=unused,
        get=unused,
        fetch=fetch,
        runner=runner,
        transfer=transfer,
        retrieve=lambda *_args: None,
    )
    assert code == 0
    assert "protein_4PCW_1.yaml" in stdout
    assert "inputs=/srv/disease/protein_4PCW_1.yaml" in stdout
    assert stdout.strip().endswith("Generation finished.")
    assert "Generation was not started." not in stdout
    assert transferred == [
        ("vm-a", "protein_4PCW_1.yaml", "/srv/disease/protein_4PCW_1.yaml"),
        ("vm-a", "protein_4PCW_1.cif", "/srv/disease/protein_4PCW_1.cif"),
    ]
    spec = json.loads((out_dir / "protein_4PCW_1.job.json").read_text(encoding="utf-8"))
    assert spec["generation_started"] is True
    assert spec["binder_kind"] == "peptide"
    assert spec["design_spec"].endswith("protein_4PCW_1.yaml")


def test_cure_run_finishes_when_every_request_succeeds(tmp_path, monkeypatch, capsys):
    def runner(_argv: list[str]) -> int:
        return 0

    def transfer(_host: str, _local_path: Path, _remote_path: str) -> None:
        return None

    retrieved: list[str] = []

    def retrieve(host: str, remote_out: str, _local_out: Path) -> None:
        retrieved.append(f"{host}:{remote_out}")

    code, out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--hosts", "localhost,vm-a", "--run", "--remote-dir", "remote_runs"],
        read_line=refuse_prompt,
        runner=runner,
        transfer=transfer,
        retrieve=retrieve,
    )
    assert code == 0
    assert stderr == ""
    assert stdout.strip().endswith("Generation finished.")
    assert "Generation was not started." not in stdout
    assert _job_started(out_dir, "protein_1ABC_1") is True
    assert _job_started(out_dir, "dna_1ABC_1") is True
    assert retrieved == ["vm-a:remote_runs/dna_1ABC_1"]


def _design_runner(argv: list[str]) -> int:
    """A stand-in for rfd4-generate that leaves one design behind."""
    out = next(part.split("=", 1)[1] for part in argv if part.startswith("out_dir="))
    sample = Path(out) / "inference_samples"
    sample.mkdir(parents=True)
    (sample / "design.cif").write_text("data_design\n", encoding="utf-8")
    return 0


def _screened(out_dir: Path, stem: str = "protein_1ABC_1") -> dict:
    return json.loads((out_dir / stem / "screen.json").read_text(encoding="utf-8"))


def test_a_run_without_lookup_targets_reads_no_public_json(tmp_path, monkeypatch, capsys):
    """A run must not contact three public services because nobody said not to.

    This is the test that was missing. Defaulting the screen's reader to the
    public one made every run reach UniProt, the atlas and STRING, and the only
    thing that caught it was ``block_live_calls`` raising an AssertionError
    that no catch expected, which turned into exit 2.

    The assertion is on the stored blocker rather than on a mock, because that
    cannot pass if a reader was handed down: ``network_not_permitted`` is only
    written when there is none.
    """
    code, out_dir, _fetched, _stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--run"],
        read_line=refuse_prompt,
        runner=_design_runner,
        transfer=_reject_remote,
        retrieve=_reject_remote,
    )
    assert code == 0
    assert stderr == ""
    target = _screened(out_dir)["target_liability"]
    for axis in ("paralog", "plasma_abundance", "plasma_hub"):
        assert target[axis]["blocker"] == "network_not_permitted", axis
    # The fold axis reads a local table and wants no network, so it is measured.
    assert target["plasma_fold"]["blocker"] == "fold_not_configured"


def test_lookup_targets_reaches_the_reader_it_was_given(tmp_path, monkeypatch, capsys):
    """With the flag, the axes are measured. Both directions are pinned."""
    asked: list[str] = []

    def read_json(url: str):
        asked.append(url)
        raise LookupMissError(url)

    code, out_dir, _fetched, _stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--run", "--lookup-targets"],
        read_line=refuse_prompt,
        runner=_design_runner,
        transfer=_reject_remote,
        retrieve=_reject_remote,
        read_json=read_json,
    )
    assert code == 0
    assert stderr == ""
    assert any("rest.uniprot.org/uniref/search" in url for url in asked), asked
    paralog = _screened(out_dir)["target_liability"]["paralog"]
    assert paralog["blocker"] == "cluster_not_found", "asked, and answered no record"
    assert paralog["blocker"] != "network_not_permitted"


def test_a_truncated_response_does_not_cost_the_run(tmp_path, monkeypatch, capsys):
    """A body shorter than its Content-Length raises from http.client.

    That is neither an OSError nor a ValueError, so it escaped every catch
    between the reader and the screen and would have aborted the whole run.
    It is now caught at the axis, which is the right place: the screen is still
    written, and the axis says the service was unavailable rather than the run
    dying or the axis reading as a clear.
    """

    def read_json(_url: str):
        raise HTTPException("incomplete read")

    code, out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--run", "--lookup-targets"],
        read_line=refuse_prompt,
        runner=_design_runner,
        transfer=_reject_remote,
        retrieve=_reject_remote,
        read_json=read_json,
    )
    assert code == 0, stderr
    assert "Screen was not saved" not in stdout, "handled at the axis, not by the backstop"
    target = _screened(out_dir)["target_liability"]
    assert target["plasma_hub"]["blocker"] == "lookup_unavailable"
    assert target["plasma_hub"]["state"] == "untested", "never a clear"


def test_a_finished_generation_is_saved_by_disease_and_target(tmp_path, monkeypatch, capsys):
    """A finished design is copied under the disease name and the target. The live files stay."""
    code, out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--run"],
        read_line=refuse_prompt,
        runner=_design_runner,
        transfer=_reject_remote,
        retrieve=_reject_remote,
    )
    assert code == 0
    assert stderr == ""
    assert stdout.strip().endswith("Generation finished.")
    archive = out_dir / "generations" / "Asthma" / "P35225"
    for stem in ("protein_1ABC_1", "dna_1ABC_1"):
        saved = archive / stem
        assert (saved / "inference_samples" / "design.cif").read_text(encoding="utf-8") == "data_design\n"
        assert (saved / f"{stem}.job.json").is_file()
        record = json.loads((saved / "generation.json").read_text(encoding="utf-8"))
        assert record["disease_name"] == "Asthma"
        assert record["disease_id"] == "EFO_0000270"
        assert record["target_id"] == "P35225"
        assert record["entry_id"] == "1ABC"
        assert "sequence" not in record
        assert display_directory(str(saved)) in stdout
        live = out_dir / stem / "inference_samples" / "design.cif"
        assert live.read_text(encoding="utf-8") == "data_design\n"
    assert (out_dir / "protein_1ABC_1.cif").is_file()


def test_a_generation_without_evidence_stays_untested(tmp_path):
    """No evidence leaves the sequence axes untested and does not rewrite the job."""
    output = tmp_path / "bare"
    output.mkdir()
    job = tmp_path / "bare.job.json"
    job.write_text(
        json.dumps({"class_name": "requested_clear", "binder_kind": "protein", "target_id": "IL5"}),
        encoding="utf-8",
    )
    before = job.read_bytes()
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert job.read_bytes() == before
    assert record["call"] == "untested"
    assert record["bar"] == "sequence"
    assert record["ppi_pass"] == "absent"
    assert record["requested_class"] == "requested_clear"
    assert record["hits"] == []
    assert record["untested"] == [*PROTEIN_AXES, *PLASMA_AXES]
    saved = json.loads((output / "screen.json").read_text(encoding="utf-8"))
    assert saved["call"] == "untested"
    assert saved["untested"] == [*PROTEIN_AXES, *PLASMA_AXES]


def test_each_binder_kind_uses_its_default_screen_bar(tmp_path):
    """Plasma is outside the default bar. A binder of kind other has no axes."""
    expected = {
        "protein": "sequence",
        "peptide": "sequence",
        "ligand": "coverage",
        "nucleic": "oligo",
        "other": "none",
    }
    for kind, bar in expected.items():
        output = tmp_path / kind
        output.mkdir()
        record = screen_generation(output, binder_kind=kind, target_id="P05113")
        assert record["bar"] == bar
        assert record["call"] == "untested"
        assert record["ppi_pass"] == "absent"


def test_clear_sequence_evidence_screens_the_sequence_bar(tmp_path):
    """Clearing the six sequence axes does not clear plasma."""
    output = tmp_path / "clear"
    output.mkdir()
    (output / "axis_evidence.json").write_text(
        json.dumps(dict.fromkeys(PROTEIN_AXES, "clear")),
        encoding="utf-8",
    )
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "screened_clear"
    assert record["bar"] == "sequence"
    assert record["hits"] == []
    assert record["untested"] == list(PLASMA_AXES)
    assert record["requested_class"] == "requested_clear"
    states = record["states"]
    assert isinstance(states, dict)
    assert [axis for axis in states if states[axis] == "clear"] == list(PROTEIN_AXES)


def test_one_hit_is_an_axis_hit(tmp_path):
    output = tmp_path / "hit"
    output.mkdir()
    evidence = dict.fromkeys(PROTEIN_AXES, "clear")
    evidence["paralog"] = "hit"
    (output / "axis_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "axis_hit"
    assert record["call"] != "screened_clear"
    assert record["hits"] == ["paralog"]
    assert record["bar"] == "sequence"


def test_ppi_pass_is_recorded_without_changing_the_call(tmp_path):
    primary = tmp_path / "primary"
    _write_ppi_csv(primary / "inference_metrics" / "per_example_metrics.csv", "1")
    record = screen_generation(primary, binder_kind="protein", target_id="IL5")
    assert record["call"] == "untested"
    assert record["ppi_pass"] is True

    reeval_only = tmp_path / "reeval"
    _write_ppi_csv(reeval_only / "inference_metrics" / "per_example_metrics_reeval.csv", "1")
    record = screen_generation(reeval_only, binder_kind="protein", target_id="IL5")
    assert record["call"] == "untested"
    assert record["ppi_pass"] is True

    both = tmp_path / "both"
    _write_ppi_csv(both / "inference_metrics" / "per_example_metrics.csv", "0")
    _write_ppi_csv(both / "inference_metrics" / "per_example_metrics_reeval.csv", "1")
    record = screen_generation(both, binder_kind="protein", target_id="IL5")
    assert record["call"] == "untested"
    assert record["ppi_pass"] is False


def test_a_safety_panel_target_screen_is_not_clear(tmp_path):
    output = tmp_path / "panel"
    output.mkdir()
    (output / "axis_evidence.json").write_text(
        json.dumps(dict.fromkeys(PROTEIN_AXES, "clear")),
        encoding="utf-8",
    )
    record = screen_generation(output, binder_kind="protein", target_id="Q12809")
    assert record["call"] == "target_is_panel"
    assert record["call"] != "screened_clear"


def test_screen_json_has_no_home_or_machine_path(tmp_path):
    output = tmp_path / "scrubbed"
    leaked = "\n".join(
        [
            "example_id,designed_structure_path",
            "0,/home/someone/secret/binder.cif",
            "1,C:\\Users\\someone\\secret\\binder.cif",
            "2,disease_runs/gene.cif",
        ]
    )
    _write_metrics_text(output / "inference_metrics" / "per_example_metrics.csv", leaked + "\n")
    screen_generation(output, binder_kind="protein", target_id="IL5")
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "Users" not in text
    assert "/mnt/" not in text
    assert "ModRFD4" not in text
    assert "sk-" not in text


def test_an_unreadable_metrics_file_still_writes_the_screen(tmp_path, monkeypatch, capsys):
    """A metrics path that cannot be read does not fail a generation that exited 0."""

    def runner(argv: list[str]) -> int:
        _design_runner(argv)
        out = Path(next(part.split("=", 1)[1] for part in argv if part.startswith("out_dir=")))
        metrics = out / "inference_metrics"
        metrics.mkdir(parents=True)
        (metrics / "per_example_metrics.csv").mkdir()
        (metrics / "per_example_metrics_reeval.csv").write_text(
            "example_id,ppi.pass\n0,1\n",
            encoding="utf-8",
        )
        return 0

    code, out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--pick", "1", "--confirm-inference", "--run"],
        read_line=refuse_prompt,
        runner=runner,
        transfer=_reject_remote,
        retrieve=_reject_remote,
    )
    assert code == 0
    assert stderr == ""
    assert "Screen was not saved" not in stdout
    assert stdout.strip().endswith("Generation finished.")
    axes = ", ".join((*PROTEIN_AXES, *PLASMA_AXES))
    assert f"Screen 1ABC chain 1: untested. Untested: {axes}." in stdout
    live = out_dir / "protein_1ABC_1" / "screen.json"
    saved = out_dir / "generations" / "Asthma" / "P35225" / "protein_1ABC_1" / "screen.json"
    record = json.loads(live.read_text(encoding="utf-8"))
    assert record["call"] == "untested"
    assert record["ppi_pass"] == "absent"
    assert record["requested_class"] == "requested_clear"
    assert json.loads(saved.read_text(encoding="utf-8"))["ppi_pass"] == "absent"
    job = json.loads((out_dir / "protein_1ABC_1.job.json").read_text(encoding="utf-8"))
    assert job["class_name"] == "requested_clear"


def test_a_bad_evidence_file_is_skipped_and_the_generation_still_finishes(tmp_path, monkeypatch, capsys):
    def runner(argv: list[str]) -> int:
        out = Path(next(part.split("=", 1)[1] for part in argv if part.startswith("out_dir=")))
        out.mkdir(parents=True, exist_ok=True)
        (out / "axis_evidence.json").write_text("{", encoding="utf-8")
        sample = out / "inference_samples"
        sample.mkdir()
        (sample / "design.cif").write_text("data_design\n", encoding="utf-8")
        return 0

    code, out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--pick", "1", "--confirm-inference", "--run"],
        read_line=refuse_prompt,
        runner=runner,
        transfer=_reject_remote,
        retrieve=_reject_remote,
    )
    assert code == 0
    assert stderr == ""
    assert "Screen was not saved for 1ABC chain 1:" in stdout
    assert stdout.strip().endswith("Generation finished.")
    assert not (out_dir / "protein_1ABC_1" / "screen.json").exists()
    job = json.loads((out_dir / "protein_1ABC_1.job.json").read_text(encoding="utf-8"))
    assert job["class_name"] == "requested_clear"


_UBIQUITIN = UBIQUITIN


def _two_chain_cif(designed: str, other: str) -> str:
    return (
        "data_design\n#\nloop_\n"
        "_entity_poly.entity_id\n"
        "_entity_poly.pdbx_seq_one_letter_code_can\n"
        "_entity_poly.pdbx_strand_id\n"
        "0\n"
        f";{designed}\n"
        ";\n"
        "A\n"
        "1\n"
        f";{other}\n"
        ";\n"
        "B\n"
    )


def test_instability_alone_can_fail_the_sequence(tmp_path):
    """The second half of ``sequence_pass``, which nothing exercised.

    Every fixture failed on GRAVY, so the instability limb was never the
    deciding factor and a broken threshold on it would not have shown. This
    sequence passes GRAVY at -1.97 and fails instability at 348, so only the
    second limb can be deciding it.
    """
    from rfproteina.metrics.physics.developability_metrics import compute_gravy, compute_instability_index

    unstable = "PQS" * 8
    assert compute_gravy(unstable) <= 0.1, "GRAVY passes"
    assert compute_instability_index(unstable) > 40.0, "instability does not"

    output = tmp_path / "unstable"
    sample = output / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text(_two_chain_cif(unstable, "AAAAAAAAAA"), encoding="utf-8")
    _write_metrics_text(
        output / "inference_metrics" / "per_example_metrics.csv",
        "example_id,chain_of_interest\nbinder_0,A_1\n",
    )
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    measured = record["sequence_developability"]
    assert measured["sequence_pass"] is False
    assert measured["gravy"] <= 0.1
    assert measured["instability_index"] > 40.0


def test_sequence_developability_does_not_clear_the_liability_bar(tmp_path, monkeypatch):
    """GRAVY and instability are recorded. They do not clear a liability axis."""
    monkeypatch.delenv("HUMAN_PROTEOME", raising=False)
    from rfproteina.metrics.physics.developability_metrics import compute_gravy, compute_instability_index

    output = tmp_path / "measured"
    output.mkdir()
    sample = output / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text(_two_chain_cif(_UBIQUITIN, "AAAAAAAAAA"), encoding="utf-8")
    _write_metrics_text(
        output / "inference_metrics" / "per_example_metrics.csv",
        "example_id,chain_of_interest,pre_oracle_pass\nbinder_0,A_1,1\n",
    )
    job = tmp_path / "measured.job.json"
    job.write_text(json.dumps({"class_name": "requested_clear"}), encoding="utf-8")
    before = job.read_bytes()
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert job.read_bytes() == before
    assert record["call"] == "untested"
    assert record["requested_class"] == "requested_clear"
    assert record["pre_oracle_pass"] is True
    assert record["ppi_pass"] == "absent"
    interface = record["designed_interface"]
    assert isinstance(interface, dict)
    assert interface["contact_ms"] is None
    assert interface["clash_fraction"] is None
    measured = record["sequence_developability"]
    assert isinstance(measured, dict)
    assert measured["sequence_pass"] is True
    assert measured["gravy"] == float(f"{compute_gravy(_UBIQUITIN):.4f}")
    assert measured["instability_index"] == float(f"{compute_instability_index(_UBIQUITIN):.4f}")
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert _UBIQUITIN not in text
    assert "AAAAAAAAAA" not in text
    assert "geometry_pass" not in text
    lines = format_screen_lines("8TLD", "2", record)
    assert lines[0].startswith("Screen 8TLD chain 2: untested.")
    assert "immunogenicity" in lines[0]
    assert "sequence_pass true" in lines[1]
    assert "pre_oracle_pass true" in lines[1]
    assert "ppi.pass absent" in lines[1]
    assert "Samples " not in lines[1]
    samples = record["samples"]
    assert isinstance(samples, list)
    assert len(samples) == 1
    assert samples[0]["example_id"] == "binder_0"
    assert samples[0]["sequence_pass"] is True
    assert samples[0]["sequence_identity"] is None
    assert samples[0]["ca_rmsd"] is None


_HYDROPHOBIC = "LLLLLLLLLL"


def test_every_generated_cif_is_a_sample_and_a_failure_stays_in_the_count(tmp_path, monkeypatch):
    """The first CIF stays on top. A later sequence failure stays in the denominator."""
    monkeypatch.delenv("HUMAN_PROTEOME", raising=False)
    from rfproteina.metrics.physics.developability_metrics import compute_gravy

    output = tmp_path / "eight"
    first = output / "inference_samples" / "binder_0"
    second = output / "inference_samples" / "binder_1"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "binder_0_generated.cif").write_text(_two_chain_cif(_UBIQUITIN, "AAAAAAAAAA"), encoding="utf-8")
    (second / "binder_1_generated.cif").write_text(_two_chain_cif(_HYDROPHOBIC, "AAAAAAAAAA"), encoding="utf-8")
    _write_metrics_text(
        output / "inference_metrics" / "per_example_metrics.csv",
        "example_id,chain_of_interest,pre_oracle_pass\nbinder_0,A_1,1\nbinder_1,A_1,1\n",
    )
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    assert record["call"] == "untested"
    assert compute_gravy(_HYDROPHOBIC) > 0.1
    samples = record["samples"]
    assert isinstance(samples, list)
    assert [item["example_id"] for item in samples] == ["binder_0", "binder_1"]
    assert samples[0]["sequence_pass"] is True
    assert samples[0]["contact_ms"] is None
    assert samples[1]["sequence_pass"] is False
    assert samples[1]["sequence_identity"] is None
    assert samples[1]["ca_rmsd"] is None
    assert record["sequence_developability"]["sequence_pass"] is True
    assert record["sequence_developability"]["gravy"] == samples[0]["gravy"]
    # The fixture is a sequence-only CIF, so the interface is unmeasured and
    # says which of its six reasons that is. The block still repeats the first
    # sample exactly, reason included.
    assert samples[0]["interface_blocker"] == "coordinates_absent"
    assert record["designed_interface"] == {
        "contact_ms": samples[0]["contact_ms"],
        "clash_fraction": samples[0]["clash_fraction"],
        "interface_blocker": samples[0]["interface_blocker"],
    }
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert _UBIQUITIN not in text
    assert _HYDROPHOBIC not in text
    assert "geometry_pass" not in text
    assert "sequence_identity_pass" not in text
    assert "ca_rmsd_pass" not in text
    lines = format_screen_lines("8TLD", "2", record)
    assert "Samples 2, sequence_pass 1 of 2" in lines[1]
    assert "Failed sequence_pass: binder_1" in lines[1]


def test_a_path_example_id_is_not_stored_and_metric_numbers_are_copied(tmp_path, monkeypatch):
    """The sample id comes from the filename. Identity and RMSD are numbers, not a pass."""
    monkeypatch.delenv("HUMAN_PROTEOME", raising=False)
    output = tmp_path / "named"
    sample = output / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text(_two_chain_cif(_UBIQUITIN, "AAAAAAAAAA"), encoding="utf-8")
    leaky = "/mnt/leak/binder_0_generated.cif"
    _write_metrics_text(
        output / "inference_metrics" / "per_example_metrics.csv",
        "example_id,chain_of_interest,designed_structure_path,"
        "sequence_similarity_to_input.sequence_identity_to_input,ca_rmsd_to_input.rmsd\n"
        f"{leaky},A_1,{leaky},0.8557,0.0\n",
    )
    record = screen_generation(output, binder_kind="protein", target_id="IL5")
    samples = record["samples"]
    assert isinstance(samples, list)
    assert samples[0]["example_id"] == "binder_0"
    assert samples[0]["sequence_identity"] == 0.8557
    assert samples[0]["ca_rmsd"] == 0.0
    assert samples[0]["sequence_pass"] is True
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert leaky not in text
    assert "/mnt/" not in text
    assert "sequence_identity_pass" not in text
    assert "ca_rmsd_pass" not in text
    assert "Samples " not in format_screen_lines("8TLD", "2", record)[1]


def test_a_finished_generation_records_sequence_developability(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HUMAN_PROTEOME", raising=False)

    def runner(argv: list[str]) -> int:
        out = Path(next(part.split("=", 1)[1] for part in argv if part.startswith("out_dir=")))
        sample = out / "inference_samples" / "binder_0"
        sample.mkdir(parents=True)
        (sample / "binder_0_generated.cif").write_text(_two_chain_cif(_UBIQUITIN, "AAAAAAAAAA"), encoding="utf-8")
        metrics = out / "inference_metrics"
        metrics.mkdir()
        (metrics / "per_example_metrics.csv").write_text(
            "example_id,chain_of_interest,pre_oracle_pass\nbinder_0,A_1,1\n",
            encoding="utf-8",
        )
        return 0

    code, out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--pick", "1", "--confirm-inference", "--run"],
        read_line=refuse_prompt,
        runner=runner,
        transfer=_reject_remote,
        retrieve=_reject_remote,
    )
    assert code == 0
    assert stderr == ""
    assert stdout.strip().endswith("Generation finished.")
    assert "Screen 1ABC chain 1: untested." in stdout
    assert "Measured 1ABC chain 1:" in stdout
    assert "sequence_pass true" in stdout
    assert "pre_oracle_pass true" in stdout
    assert "ppi.pass absent" in stdout
    live = json.loads((out_dir / "protein_1ABC_1" / "screen.json").read_text(encoding="utf-8"))
    saved = out_dir / "generations" / "Asthma" / "P35225" / "protein_1ABC_1" / "screen.json"
    assert live["call"] == "untested"
    assert live["sequence_developability"]["sequence_pass"] is True
    assert json.loads(saved.read_text(encoding="utf-8"))["pre_oracle_pass"] is True
    assert _UBIQUITIN not in saved.read_text(encoding="utf-8")
    job = json.loads((out_dir / "protein_1ABC_1.job.json").read_text(encoding="utf-8"))
    assert job["class_name"] == "requested_clear"


def test_the_closed_page_shows_the_screen():
    session = BrowserSession(opener=lambda _url: None)
    assert "Return to the terminal." in session.done_page()
    session.note_closing(["Screen 8TLD chain 2: untested. Untested: immunogenicity."])
    page = session.done_page()
    assert "Panel closed" in page
    assert "Screen 8TLD chain 2: untested. Untested: immunogenicity." in page
    assert "Return to the terminal." not in page


def _write_pair_cif(path: Path, separation: float) -> None:
    """Two alanine backbones. ``separation`` is how far chain B sits from chain A."""
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure import AtomArray, BondList

    names = ["N", "CA", "C", "O"]
    elements = ["N", "C", "C", "O"]
    offsets = np.array(
        [[0.0, 0.0, 0.0], [1.46, 0.0, 0.0], [2.0, 1.3, 0.0], [1.7, 2.2, 0.6]],
        dtype=float,
    )
    atoms = AtomArray(8)
    atoms.coord = np.vstack([offsets, offsets + np.array([separation, 0.0, 0.0])])
    atoms.chain_id = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    atoms.res_id = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    atoms.res_name = np.array(["ALA"] * 8)
    atoms.atom_name = np.array(names + names)
    atoms.element = np.array(elements + elements)
    atoms.occupancy = np.ones(8)
    atoms.b_factor = np.zeros(8)
    atoms.hetero = np.zeros(8, dtype=bool)
    # The backbone bonds, which a real generated CIF carries. Without them N-CA
    # at 1.46A counts as a clash in every residue, so every fixture self-clashed
    # and no clash number could be asserted, only an ordering.
    atoms.bonds = BondList(
        atoms.array_length(),
        np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]], dtype=np.uint32),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    to_cif_file(
        atoms,
        path,
        author="fixture",
        include_entity_categories=False,
        include_chem_comp=False,
        save_standard_annotations=False,
    )


def _screen_pair(folder: Path, separation: float) -> dict[str, object]:
    cif = folder / "inference_samples" / "binder_0" / "binder_0_generated.cif"
    _write_pair_cif(cif, separation)
    _write_metrics_text(
        folder / "inference_metrics" / "per_example_metrics.csv",
        "example_id,chain_of_interest\nbinder_0,A_1\n",
    )
    record = screen_generation(folder, binder_kind="protein", target_id="IL5")
    interface = record["designed_interface"]
    assert isinstance(interface, dict)
    return interface


def test_a_coordinate_cif_records_a_finite_interface_and_stays_untested(tmp_path, monkeypatch):
    """Coordinates are measured. They do not clear the call or store a sequence."""
    for name in ("AF3_IMAGE", "AF3_MODEL_DIR", "AF3_JAX_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "packed"
    interface = _screen_pair(output, 4.0)
    record = json.loads((output / "screen.json").read_text(encoding="utf-8"))
    assert record["call"] == "untested"
    assert np.isfinite(interface["contact_ms"])
    assert np.isfinite(interface["clash_fraction"])
    text = (output / "screen.json").read_text(encoding="utf-8")
    assert "geometry_pass" not in text
    lines = format_screen_lines("8TLD", "2", record)
    assert "contact_ms" in lines[1]
    assert "clash_fraction" in lines[1]
    assert "ppi_blocker oracle_not_configured" in lines[1]


def test_closer_atoms_raise_the_clash_fraction(tmp_path):
    near = _screen_pair(tmp_path / "near", 1.0)
    far = _screen_pair(tmp_path / "far", 20.0)
    assert near["clash_fraction"] > far["clash_fraction"]
    # Now that the backbone bonds are declared, a separated pair genuinely has
    # no clash rather than a floor of self-clashing residues.
    assert far["clash_fraction"] == 0.0


def _write_gap_cif(path: Path, gap: float) -> None:
    """Two chains whose closest approach is exactly ``gap``.

    Four atoms per chain, 1.8A apart along x, with the second chain offset in y
    alone. So every inter-chain pair is at least ``gap`` and the nearest is
    exactly ``gap``, while no intra-chain pair is under 1.8A. That makes the
    clash threshold itself assertable rather than inferred from an ordering.
    """
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure import AtomArray, BondList

    names = ["N", "CA", "C", "O"]
    spread = np.array([[1.8 * index, 0.0, 0.0] for index in range(4)], dtype=float)
    atoms = AtomArray(8)
    atoms.coord = np.vstack([spread, spread + np.array([0.0, gap, 0.0])])
    atoms.chain_id = np.array(["A"] * 4 + ["B"] * 4)
    atoms.res_id = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    atoms.res_name = np.array(["ALA"] * 8)
    atoms.atom_name = np.array(names + names)
    atoms.element = np.array(["N", "C", "C", "O"] * 2)
    atoms.occupancy = np.ones(8)
    atoms.b_factor = np.zeros(8)
    atoms.hetero = np.zeros(8, dtype=bool)
    atoms.bonds = BondList(atoms.array_length())
    path.parent.mkdir(parents=True, exist_ok=True)
    to_cif_file(
        atoms,
        path,
        author="fixture",
        include_entity_categories=False,
        include_chem_comp=False,
        save_standard_annotations=False,
    )


def _clash_at(folder: Path, gap: float) -> float:
    cif = folder / "inference_samples" / "binder_0" / "binder_0_generated.cif"
    _write_gap_cif(cif, gap)
    _write_metrics_text(
        folder / "inference_metrics" / "per_example_metrics.csv",
        "example_id,chain_of_interest\nbinder_0,A\n",
    )
    record = screen_generation(folder, binder_kind="protein", target_id="IL5")
    return record["designed_interface"]["clash_fraction"]


def test_the_clash_distance_is_one_and_a_half_angstrom(tmp_path):
    """The threshold, pinned by construction rather than by a snapshot.

    ``compute_clash_stats`` is a raw neighbour search at its threshold, so a
    pair at 1.45A must count and the same pair at 1.55A must not. Nothing else
    in these fixtures is within 1.8A, so the only possible clash is that pair.
    """
    assert _clash_at(tmp_path / "inside", 1.45) > 0.0
    assert _clash_at(tmp_path / "outside", 1.55) == 0.0


def test_contact_surface_falls_away_with_separation(tmp_path):
    """Contact molecular surface is distance-weighted, so it has to decrease.

    These are properties of the measurement rather than recorded numbers: a
    snapshot would pin whatever the metric happened to return, including a bug.
    """
    close = _screen_pair(tmp_path / "close", 4.0)["contact_ms"]
    middle = _screen_pair(tmp_path / "middle", 6.0)["contact_ms"]
    wide = _screen_pair(tmp_path / "wide", 8.0)["contact_ms"]
    apart = _screen_pair(tmp_path / "apart", 20.0)["contact_ms"]
    assert close > middle > wide >= 0.0
    assert close > 0.0, "touching chains have a contact surface"
    assert apart == 0.0, "chains 20A apart have none"


def test_ppi_blocker_names_why_the_refold_is_missing(tmp_path, monkeypatch):
    """A missing oracle cell is not a pass, and a recorded cell does not change the call."""
    for name in ("AF3_IMAGE", "AF3_MODEL_DIR", "AF3_JAX_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    missing = tmp_path / "missing"
    missing.mkdir()
    record = screen_generation(missing, binder_kind="protein", target_id="IL5")
    assert record["ppi_pass"] == "absent"
    assert record["ppi_blocker"] == "oracle_not_configured"
    assert record["call"] == "untested"

    monkeypatch.setenv("AF3_IMAGE", "image")
    monkeypatch.setenv("AF3_MODEL_DIR", "models")
    recorded = tmp_path / "partial"
    recorded.mkdir()
    partial = screen_generation(recorded, binder_kind="protein", target_id="IL5")
    assert partial["ppi_blocker"] == "oracle_not_configured"

    monkeypatch.setenv("AF3_JAX_CACHE_DIR", "cache")
    configured = tmp_path / "configured"
    configured.mkdir()
    ready = screen_generation(configured, binder_kind="protein", target_id="IL5")
    assert ready["ppi_pass"] == "absent"
    assert ready["ppi_blocker"] == "not_requested"

    primary = tmp_path / "primary"
    _write_ppi_csv(primary / "inference_metrics" / "per_example_metrics.csv", "1")
    hit = screen_generation(primary, binder_kind="protein", target_id="IL5")
    assert hit["ppi_pass"] is True
    assert hit["ppi_blocker"] == "recorded"
    assert hit["call"] == "untested"


def _follow_screen(contact: float | None, *, sequence_pass: bool | str) -> dict[str, object]:
    return {
        "designed_interface": {"contact_ms": contact, "clash_fraction": 0.0},
        "sequence_developability": {
            "gravy": 0.2,
            "instability_index": 30.0,
            "sequence_pass": sequence_pass,
        },
    }


def test_an_interface_that_fails_sequence_builds_an_inverse_folding_command(tmp_path):
    """The command keeps the backbone and writes into a fresh directory."""
    from rfproteina.disease.follow_up import follow_up_command

    generation = tmp_path / "gene_8TLD_2"
    cif = generation / "inference_samples" / "binder_0" / "binder_0_generated.cif"
    _write_pair_cif(cif, 4.0)
    original = tmp_path / "gene_8TLD_2.yaml"
    original.write_text('binder:\n  contig: "60-110, /0, C1-2"\n', encoding="utf-8")
    job = tmp_path / "gene_8TLD_2.job.json"
    job.write_text(json.dumps({"class_name": "requested_clear"}), encoding="utf-8")
    before_cif = cif.read_bytes()
    before_spec = original.read_bytes()
    before_job = job.read_bytes()
    fresh = tmp_path / "gene_8TLD_2_inverse"
    command = follow_up_command(
        _follow_screen(12.5, sequence_pass=False),
        generation_dir=generation,
        generated_cif=cif,
        original_spec=original,
        fresh_dir=fresh,
        designed_chain="A_1",
    )
    assert command is not None
    assert "generation=inverse_folding" in command
    assert "recycle_strategy=medium/flat" in command
    assert "num_replicates=8" in command
    assert "batch_size=1" in command
    assert "inference_experiment=inference_on_user_inputs" in command
    assert f"out_dir={fresh.as_posix()}" in command
    out = next(part for part in command if part.startswith("out_dir="))
    assert Path(out.split("=", 1)[1]).resolve() == fresh.resolve()
    assert Path(out.split("=", 1)[1]).resolve() != generation.resolve()
    spec = fresh / "inverse_folding.yaml"
    assert f"inputs={spec.as_posix()}" in command
    text = spec.read_text(encoding="utf-8")
    assert "sel('A/*/*/[N,CA,C,O]')" in text
    assert "sel('B/*/*/[N,CA,C,O]')" in text
    assert "sel('A')" not in text
    assert "sel('B')" in text
    assert "60-110" not in text
    assert "C_HOT" not in text
    assert "CB" not in text
    assert cif.read_bytes() == before_cif
    assert original.read_bytes() == before_spec
    assert job.read_bytes() == before_job
    assert b"requested_clear" in before_job


def test_a_zero_contact_surface_repeats_the_original_ppi_spec(tmp_path):
    from rfproteina.disease.follow_up import follow_up_command

    generation = tmp_path / "gene_8TLD_2"
    generation.mkdir()
    cif = generation / "binder_0_generated.cif"
    cif.write_text("data_design\n", encoding="utf-8")
    original = tmp_path / "gene_8TLD_2.yaml"
    original.write_text('binder:\n  contig: "60-110, /0, C12-120"\n', encoding="utf-8")
    before = original.read_bytes()
    fresh = tmp_path / "gene_8TLD_2_replicates"
    command = follow_up_command(
        _follow_screen(0.0, sequence_pass=False),
        generation_dir=generation,
        generated_cif=cif,
        original_spec=original,
        fresh_dir=fresh,
        designed_chain="A",
    )
    assert command is not None
    assert "generation=inverse_folding" not in command
    assert "recycle_strategy=medium/front_loaded" in command
    assert "num_replicates=8" in command
    assert "batch_size=1" in command
    assert f"inputs={original.as_posix()}" in command
    assert f"out_dir={fresh.as_posix()}" in command
    assert original.read_bytes() == before
    assert not fresh.exists()


def test_a_null_contact_surface_builds_no_command(tmp_path):
    from rfproteina.disease.follow_up import follow_up_command

    generation = tmp_path / "gene_8TLD_2"
    fresh = tmp_path / "gene_8TLD_2_next"
    kwargs = {
        "generation_dir": generation,
        "generated_cif": generation / "missing_generated.cif",
        "original_spec": tmp_path / "gene_8TLD_2.yaml",
        "fresh_dir": fresh,
        "designed_chain": "A",
    }
    assert follow_up_command(_follow_screen(None, sequence_pass=False), **kwargs) is None
    passed = _follow_screen(12.5, sequence_pass=True)
    assert follow_up_command(passed, **kwargs) is None
    assert not fresh.exists()


def test_a_later_failed_sample_does_build_a_follow_up_command(tmp_path):
    """A failed sample is followed up even when the first sample passed.

    This asserted the opposite until 2026-11-07, and the opposite was a bug: the
    top-level ``sequence_developability`` block repeats the first CIF only, so a
    run whose first sample passed produced no follow-up however many later
    samples failed. On the real eight-sample run on disk, ``binder_3`` and
    ``binder_6`` failed while ``binder_0`` passed, and nothing was built.

    Each failed sample gets its own output directory, because two
    inverse-folding runs sharing one would overwrite each other's designs.
    """
    from rfproteina.disease.follow_up import follow_up_commands

    generation = tmp_path / "gene_8TLD_2"
    fresh = tmp_path / "gene_8TLD_2_again"
    cifs = {}
    for example_id in ("binder_0", "binder_3", "binder_6"):
        path = generation / "inference_samples" / example_id / f"{example_id}_generated.cif"
        _write_pair_cif(path, 4.0)
        cifs[example_id] = path
    screen = _follow_screen(12.5, sequence_pass=True)
    screen["samples"] = [
        {"example_id": "binder_0", "sequence_pass": True, "contact_ms": 12.5},
        {"example_id": "binder_3", "sequence_pass": False, "contact_ms": 12.5},
        {"example_id": "binder_6", "sequence_pass": False, "contact_ms": 12.5},
    ]
    commands = follow_up_commands(
        screen,
        generation_dir=generation,
        generated_cifs=cifs,
        original_spec=tmp_path / "gene_8TLD_2.yaml",
        fresh_root=fresh,
        designed_chain="A_1",
    )
    assert len(commands) == 2, "one per failed sample, and none for the one that passed"
    assert all("generation=inverse_folding" in command for command in commands)
    assert f"out_dir={(fresh / 'binder_3').as_posix()}" in commands[0]
    assert f"out_dir={(fresh / 'binder_6').as_posix()}" in commands[1]
    assert (fresh / "binder_3" / "inverse_folding.yaml").is_file()
    assert (fresh / "binder_6" / "inverse_folding.yaml").is_file()
    assert not (fresh / "binder_0").exists()


def test_a_run_whose_samples_all_passed_builds_nothing(tmp_path):
    """Following up a design that passed would spend a GPU on nothing."""
    from rfproteina.disease.follow_up import follow_up_commands

    generation = tmp_path / "gene_8TLD_2"
    generation.mkdir()
    screen = _follow_screen(12.5, sequence_pass=True)
    screen["samples"] = [{"example_id": "binder_0", "sequence_pass": True, "contact_ms": 12.5}]
    commands = follow_up_commands(
        screen,
        generation_dir=generation,
        generated_cifs={},
        original_spec=tmp_path / "gene_8TLD_2.yaml",
        fresh_root=tmp_path / "again",
        designed_chain="A_1",
    )
    assert commands == []


def _follow_case(tmp_path, example_ids: tuple[str, ...]):
    """A generation directory, a fresh root, the sample CIFs, and an original spec."""
    from rfproteina.disease.follow_up import follow_up_commands

    generation = tmp_path / "gene_8TLD_2"
    fresh = tmp_path / "gene_8TLD_2_again"
    cifs = {}
    for example_id in example_ids:
        path = generation / "inference_samples" / example_id / f"{example_id}_generated.cif"
        _write_pair_cif(path, 4.0)
        cifs[example_id] = path
    original = tmp_path / "gene_8TLD_2.yaml"
    original.write_text('binder:\n  contig: "60-110, /0, C1-2"\n', encoding="utf-8")
    return follow_up_commands, generation, fresh, cifs, original


def test_an_unmeasured_first_contact_still_follows_up_a_later_sample(tmp_path):
    """A missing surface on the first CIF does not drop a later interface."""
    follow_up_commands, generation, fresh, cifs, original = _follow_case(tmp_path, ("binder_0", "binder_3"))
    screen = _follow_screen(None, sequence_pass=True)
    screen["samples"] = [
        {"example_id": "binder_0", "sequence_pass": True, "contact_ms": None},
        {"example_id": "binder_3", "sequence_pass": False, "contact_ms": 12.5},
    ]
    commands = follow_up_commands(
        screen,
        generation_dir=generation,
        generated_cifs=cifs,
        original_spec=original,
        fresh_root=fresh,
        designed_chain="A_1",
    )
    assert len(commands) == 1
    assert "generation=inverse_folding" in commands[0]
    assert f"out_dir={(fresh / 'binder_3').as_posix()}" in commands[0]


def test_a_zero_contact_sample_does_not_swallow_a_later_interface(tmp_path):
    """One sample with no interface does not turn a later interface into that draw."""
    follow_up_commands, generation, fresh, cifs, original = _follow_case(tmp_path, ("binder_0", "binder_3"))
    screen = _follow_screen(0.0, sequence_pass=False)
    screen["samples"] = [
        {"example_id": "binder_0", "sequence_pass": False, "contact_ms": 0.0},
        {"example_id": "binder_3", "sequence_pass": False, "contact_ms": 12.5},
    ]
    commands = follow_up_commands(
        screen,
        generation_dir=generation,
        generated_cifs=cifs,
        original_spec=original,
        fresh_root=fresh,
        designed_chain="A_1",
    )
    assert len(commands) == 2
    assert "generation=inverse_folding" not in commands[0]
    assert f"out_dir={fresh.as_posix()}" in commands[0]
    assert "generation=inverse_folding" in commands[1]
    assert f"out_dir={(fresh / 'binder_3').as_posix()}" in commands[1]


def test_a_later_zero_contact_repeats_the_ppi_spec_once(tmp_path):
    """A later surface of zero is one more draw of the original spec, not an inverse fold."""
    follow_up_commands, generation, fresh, cifs, original = _follow_case(tmp_path, ("binder_0", "binder_6"))
    screen = _follow_screen(12.5, sequence_pass=True)
    screen["samples"] = [
        {"example_id": "binder_0", "sequence_pass": True, "contact_ms": 12.5},
        {"example_id": "binder_6", "sequence_pass": False, "contact_ms": 0.0},
    ]
    commands = follow_up_commands(
        screen,
        generation_dir=generation,
        generated_cifs=cifs,
        original_spec=original,
        fresh_root=fresh,
        designed_chain="A_1",
    )
    assert len(commands) == 1
    assert "generation=inverse_folding" not in commands[0]
    assert f"inputs={original.as_posix()}" in commands[0]
    assert f"out_dir={fresh.as_posix()}" in commands[0]
    assert not (fresh / "binder_6").exists()


def _write_ppi_csv(path: Path, value: str) -> None:
    _write_metrics_text(path, f"example_id,ppi.pass\n0,{value}\n")


def _write_metrics_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _reject_remote(*_args: object) -> None:
    raise AssertionError("remote")


def test_omitted_local_worker_limit_runs_every_local_request_at_once(tmp_path, monkeypatch, capsys):
    entered = threading.Barrier(3)

    def runner(_argv: list[str]) -> int:
        entered.wait(timeout=5)
        return 0

    code, _out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--pick", "3", "--confirm-inference", "--hosts", "local,local,local", "--run"],
        read_line=refuse_prompt,
        runner=runner,
        transfer=_reject_remote,
        retrieve=_reject_remote,
    )
    assert code == 0
    assert stderr == ""
    assert "Starting 3 local and 0 remote generation request(s)." in stdout
    assert stdout.strip().endswith("Generation finished.")


def test_local_worker_limit_serializes_only_this_machine(tmp_path, monkeypatch, capsys):
    state = {"current": 0, "max": 0}
    lock = threading.Lock()

    def runner(_argv: list[str]) -> int:
        with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        try:
            time.sleep(0.05)
        finally:
            with lock:
                state["current"] -= 1
        return 0

    code, _out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        [
            "--all",
            "--pick",
            "3",
            "--confirm-inference",
            "--hosts",
            "local,local,local",
            "--local-workers",
            "1",
            "--run",
        ],
        read_line=refuse_prompt,
        runner=runner,
        transfer=_reject_remote,
        retrieve=_reject_remote,
    )
    assert code == 0
    assert stderr == ""
    assert state["max"] == 1
    assert "Starting 3 local and 0 remote generation request(s)." in stdout


def test_local_worker_limit_does_not_limit_remote_machines(tmp_path, monkeypatch, capsys):
    remotes = threading.Barrier(2)

    def runner(argv: list[str]) -> int:
        if argv[0] == "ssh":
            remotes.wait(timeout=5)
            return 0
        time.sleep(0.2)
        return 0

    code, _out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        [
            "--all",
            "--pick",
            "3",
            "--confirm-inference",
            "--hosts",
            "local,vm-a,vm-b",
            "--local-workers",
            "1",
            "--run",
            "--remote-dir",
            "remote_runs",
        ],
        read_line=refuse_prompt,
        runner=runner,
        transfer=lambda *_args: None,
        retrieve=lambda *_args: None,
    )
    assert code == 0
    assert stderr == ""
    assert "Starting 1 local and 2 remote generation request(s)." in stdout


def test_the_remote_worker_limit_holds_remote_requests_to_it(tmp_path, monkeypatch, capsys):
    """The twin of the test above: with a cap, two remote runs do NOT overlap.

    The test above barriers two remote runners to prove an uncapped run starts
    them together. That is still the default, so it keeps passing and says
    nothing about the cap. This one asserts the other half, and it has to do it
    by observing overlap rather than by reading the semaphore: a limit that was
    created and never acquired would pass any check that only looked at the
    number.

    The slot is held for the whole remote leg, transfers included, so the
    transfer is counted too -- N concurrent scp copies would be the same
    saturation the cap exists to prevent.
    """
    lock = threading.Lock()
    live = 0
    peak = 0

    def note(delta: int) -> None:
        nonlocal live, peak
        with lock:
            live += delta
            peak = max(peak, live)

    def runner(argv: list[str]) -> int:
        if argv[0] != "ssh":
            return 0
        note(1)
        time.sleep(0.15)
        note(-1)
        return 0

    code, _out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        [
            "--all",
            "--pick",
            "3",
            "--confirm-inference",
            "--hosts",
            "local,vm-a,vm-b",
            "--remote-workers",
            "1",
            "--run",
            "--remote-dir",
            "remote_runs",
        ],
        read_line=refuse_prompt,
        runner=runner,
        transfer=lambda *_args: None,
        retrieve=lambda *_args: None,
    )
    assert code == 0
    assert stderr == ""
    assert "Starting 1 local and 2 remote generation request(s)." in stdout
    assert peak == 1, f"two remote requests ran at once under --remote-workers 1 (peak {peak})"


def test_remote_workers_below_one_does_not_start_or_mark_jobs(tmp_path, monkeypatch, capsys):
    """Same refusal shape as the local limit, and before anything is marked."""

    def runner(_argv: list[str]) -> int:
        raise AssertionError("runner")

    code, _out_dir, _fetched, _stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--remote-workers", "0", "--run"],
        read_line=refuse_prompt,
        runner=runner,
    )
    assert code == 2
    assert "Remote workers must be at least 1" in stderr


def test_an_unset_remote_limit_still_starts_every_remote_request_at_once(tmp_path, monkeypatch, capsys):
    """The default must not have changed. A cap nobody asked for is a regression.

    Barriered, so it deadlocks and times out rather than passing by luck if the
    remote requests were serialised.
    """
    remotes = threading.Barrier(2)

    def runner(argv: list[str]) -> int:
        if argv[0] == "ssh":
            remotes.wait(timeout=5)
        return 0

    code, _out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        [
            "--all",
            "--pick",
            "3",
            "--confirm-inference",
            "--hosts",
            "local,vm-a,vm-b",
            "--run",
            "--remote-dir",
            "remote_runs",
        ],
        read_line=refuse_prompt,
        runner=runner,
        transfer=lambda *_args: None,
        retrieve=lambda *_args: None,
    )
    assert code == 0
    assert stderr == ""
    assert "Starting 1 local and 2 remote generation request(s)." in stdout


def test_dozens_of_hosts_are_accepted(tmp_path, monkeypatch, capsys):
    hosts = ",".join(f"vm-{index}" for index in range(64))
    seen: list[str] = []

    def runner(argv: list[str]) -> int:
        seen.append(argv[1])
        return 0

    code, out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--pick", "1", "--confirm-inference", "--hosts", hosts, "--run", "--remote-dir", "remote_runs"],
        read_line=refuse_prompt,
        runner=runner,
        transfer=lambda *_args: None,
        retrieve=lambda *_args: None,
    )
    assert code == 0
    assert stderr == ""
    assert seen == ["vm-0"]
    assert _job_started(out_dir, "protein_1ABC_1") is True
    assert stdout.strip().endswith("Generation finished.")


def test_local_workers_below_one_does_not_start_or_mark_jobs(tmp_path, monkeypatch, capsys):
    def runner(_argv: list[str]) -> int:
        raise AssertionError("runner")

    code, out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--local-workers", "0", "--run"],
        read_line=refuse_prompt,
        runner=runner,
    )
    assert code == 2
    assert "Local workers must be at least 1" in stderr
    assert "Generation dispatch failed:" in stderr
    assert _job_started(out_dir, "protein_1ABC_1") is False
    assert "rfd4-generate" in stdout
    assert "Generation was not started." not in stdout


def test_a_blank_remote_directory_blocks_remote_planning(tmp_path, monkeypatch, capsys):
    code, out_dir, _fetched, _stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--hosts", "vm-a", "--remote-dir", " "],
        read_line=refuse_prompt,
    )
    assert code == 2
    assert "remote directory" in stderr.casefold()
    assert _job_started(out_dir, "protein_1ABC_1") is False
    assert _job_started(out_dir, "dna_1ABC_1") is False

    code, _out_dir, _fetched, stdout, stderr = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--all", "--confirm-inference", "--hosts", "local", "--remote-dir", " "],
        read_line=refuse_prompt,
    )
    assert code == 0
    assert stderr == ""
    assert stdout.strip().endswith("Generation was not started.")


def test_execute_generation_forwards_binder_kind_only_for_a_present_checkpoint(tmp_path: Path):
    structures = [_option("protein", "1abc"), _option("dna", "1dna")]
    shards = shard_for_hosts(structures, ["local", "vm-a"])
    present = tmp_path / "modified.ckpt"
    present.write_bytes(b"not-a-checkpoint")
    requests = generation_requests(
        shards,
        str(tmp_path),
        modified_ckpt=present,
        remote_dir="remote_runs",
        binder_kind="peptide",
    )
    assert all("nn.disease_binder_kind=peptide" in request.argv for request in requests)
    assert requests[1].remote_out_dir.endswith("/dna_1dna_A")
    with pytest.raises(ValueError, match="at least 1"):
        execute_generation(requests, runner=lambda _argv: 0, transfer=lambda *_args: None, local_workers=0)
    with pytest.raises(ValueError, match="Remote workers must be at least 1"):
        execute_generation(requests, runner=lambda _argv: 0, transfer=lambda *_args: None, remote_workers=0)


def test_the_annotated_cif_keeps_a_single_data_block(tmp_path: Path):
    # A second data_ header makes the file a multi-block CIF, and the structure loader takes
    # the one block of a file, so every annotated input would fail to load.
    source = tmp_path / "protein_1abc_A.cif"
    source.write_text("data_1ABC\n#\n_struct.title x\n#\nloop_\n_atom_site.id\n1\n2\n", encoding="utf-8")
    annotated = write_annotated_cif(source, tmp_path / "out.annotated.cif")
    text = annotated.read_text(encoding="utf-8")
    assert [line for line in text.splitlines() if line.startswith("data_")] == ["data_1ABC"]
    assert "_rfproteina_disease.call requested_clear" in text
    assert "data_affinity_class" not in combination_record(requested_clear("protein"))


def test_every_structure_encoding_reads_the_same_label():
    # The stem helper has to strip the format and the compression together. Leaving the format
    # glued to the last token hides the state word, and a name carrying a coarse bucket word
    # then takes its sign from that word instead of from its axis.
    for encoding in (".cif", ".cif.gz", ".cif.zst", ".bcif", ".bcif.gz", ".mmjson.zst", ".pdb.gz"):
        parsed = parse_class_filename(f"ligand_herg_toxic{encoding}")
        assert (parsed.binder_kind, parsed.axis, parsed.state) == ("ligand", "herg", "hit"), encoding
        with pytest.raises(ValueError, match="not exactly one"):
            parse_class_filename(f"high_affinity_clean_ligand_herg_toxic{encoding}")


def test_file_limit_zero_selects_nothing(tmp_path: Path):
    ready = tmp_path / "classes"
    ready.mkdir()
    (ready / "high_affinity_ligand_herg_toxic.cif").write_text("herg", encoding="utf-8")
    assert load_class_files(ready, file_limit=0) == []
    assert len(load_class_files(ready, file_limit=1)) == 1


def test_each_input_gets_its_own_output_directory(tmp_path: Path):
    # One deposit can be listed under several polymer groups. The input name carries the group,
    # so the output path has to as well, or the runs overwrite each other's metrics.
    rows = [_option(group, "1tup") for group in ("protein", "dna", "gene")]
    commands = generation_commands(shard_for_hosts(rows, ["local"]), str(tmp_path))
    out_dirs = [part for command in commands for part in command if part.startswith("out_dir=")]
    inputs = [part for command in commands for part in command if part.startswith("inputs=")]
    assert len(set(out_dirs)) == 3
    for out_dir, source in zip(out_dirs, inputs, strict=True):
        assert out_dir.split("=", 1)[1].rsplit("/", 1)[-1] == (
            source.split("=", 1)[1].rsplit("/", 1)[-1].removesuffix(".annotated.cif")
        )


def test_a_repeated_host_is_a_second_worker(tmp_path: Path):
    rows = [_option("protein", "1abc"), _option("protein", "2def")]
    shards = shard_for_hosts(rows, ["local", "local"])
    assert [len(shard.structures) for shard in shards] == [1, 1]
    commands = generation_commands(shards, str(tmp_path))
    assert len({tuple(command) for command in commands}) == len(commands) == 2


def test_a_modified_checkpoint_can_choose_the_binder_kind(tmp_path: Path):
    present = tmp_path / "modified.ckpt"
    present.write_bytes(b"not-a-checkpoint")
    # The kind rebuilds a non-persistent buffer and loads no different weights, so a run may
    # choose it. Without this the request is always for the kind the checkpoint was trained on.
    assert "nn.disease_binder_kind=ligand" in model_overrides(present, "ligand")
    assert all("disease_binder_kind" not in part for part in model_overrides(present))
    assert all("disease_binder_kind" not in part for part in model_overrides(None, "ligand"))


def test_a_ranked_catalog_keeps_a_synonym():
    ranked = [Disease("amyotrophic lateral sclerosis", "MONDO_0004976")]
    # An ontology search already answered the query, and a synonym is a substring of no
    # resolved label, so filtering its answer locally would discard the hit.
    assert [row.disease_id for row in match_diseases("ALS", ranked, ranked=True)] == ["MONDO_0004976"]
    assert match_diseases("ALS", ranked) == []
    assert match_diseases("ALS", [], ranked=True) == []
    assert [row.name for row in match_diseases("Asthma", CATALOG, ranked=True)] == ["Asthma"]


def test_a_gene_without_a_uniprot_protein_keeps_its_symbol():
    payload = {
        "data": {
            "disease": {
                "associatedTargets": {
                    "rows": [
                        {"target": {"approvedSymbol": "MALAT1", "proteinIds": []}},
                        {
                            "target": {
                                "approvedSymbol": "TP53",
                                "proteinIds": [{"id": "P04637", "source": "uniprot"}],
                            }
                        },
                    ]
                }
            }
        }
    }
    # Pairing a symbol only with an accession drops every target that has no protein, and the
    # gene group could then never list a structure no protein accession reaches.
    assert ("MALAT1", "") in accessions_from_targets(payload)
    assert ("TP53", "P04637") in accessions_from_targets(payload)


def test_one_request_per_deposit_however_many_rows_it_answers():
    calls: list[str] = []

    def post(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL:
            return _associated_targets(("TP53", "P04637"))
        return {"result_set": [{"identifier": "1TUP_1"}, {"identifier": "1TUP_2"}]}

    def get(url: str) -> dict:
        calls.append(url)
        return {
            "struct": {"title": "p53"},
            "rcsb_entry_info": {"structure_determination_methodology": "experimental"},
        }

    collect_structure_options("EFO_1", post=post, get=get, row_limit=5)
    assert calls == [entry_url("1TUP")]


def _entry_record(title: str, methodology: str = "experimental") -> dict:
    return {
        "struct": {"title": title},
        "rcsb_entry_info": {"structure_determination_methodology": methodology},
    }


def test_each_service_is_asked_no_faster_than_it_allows():
    """These are other people's servers, and only one of them publishes a number.

    NCBI documents three requests a second without an API key. That is a pace, not
    a ceiling on concurrency, and at the 0.18s this program measures per call one
    serial caller already exceeded it, so it is paced rather than merely capped.
    EMBL-EBI publishes no number -- its terms are qualitative -- so PRIDE, IntAct
    and PeptideAtlas get a small allowance rather than one read off a limit nobody
    wrote down.
    """
    ncbi = allowance_for("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene")
    assert ncbi.every >= 1 / 3 - 0.01, "the documented three a second is paced"
    ebi = allowance_for("https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects?keyword=P1")
    assert ebi.at_once == 3
    assert allowance_for("https://www.ebi.ac.uk/intact/ws/interaction/findInteractions/P1") == ebi
    assert allowance_for("https://reactome.org/ContentService/data/mapping/UniProt/P1/pathways").at_once == 4
    assert allowance_for("https://data.rcsb.org/rest/v1/core/entry/1TUP").at_once == 8

    # A host nobody listed is a host nobody checked, so it gets the least.
    assert allowance_for("https://unlisted.example.org/thing") == DEFAULT_ALLOWANCE

    # Every public reader names this program, so a service can see what is calling.
    assert "Humanitys-Last-Disease" in USER_AGENT


def test_the_allowance_bounds_what_one_service_sees_at_once():
    """The gate is what actually limits the pace, not the number of threads."""
    url = "https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects?keyword=GATE"
    cap = allowance_for(url).at_once
    live = {"now": 0, "most": 0}
    lock = threading.Lock()

    def call(_index: int) -> None:
        with slot(url):
            with lock:
                live["now"] += 1
                live["most"] = max(live["most"], live["now"])
            time.sleep(0.02)
            with lock:
                live["now"] -= 1

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(call, range(24)))
    assert live["most"] <= cap, "more requests were in flight than the service allows"
    assert live["most"] > 1, "the gate is not serialising everything either"

    # A different service does not wait behind this one.
    assert allowance_for("https://data.rcsb.org/rest/v1/core/entry/1TUP").at_once > cap


def test_the_ncbi_pace_is_kept_even_with_threads_asking():
    """Threads must not turn a documented rate limit into a burst."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term=PACE"
    every = allowance_for(url).every
    starts: list[float] = []
    lock = threading.Lock()

    def call(_index: int) -> None:
        with slot(url), lock:
            starts.append(time.monotonic())

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(call, range(5)))
    starts.sort()
    gaps = [later - earlier for earlier, later in zip(starts, starts[1:], strict=False)]
    assert gaps, "nothing was timed"
    assert all(gap >= every * 0.9 for gap in gaps), gaps


def test_reactome_pathway_lookups_are_read_several_at_a_time():
    """Reactome answered in ~19s and was asked once per seed: 100s of one lookup.

    The seeds are asked for together and keyed by seed, so the loop that consumes
    them walks the seeds in listed order regardless of which answered first.
    """
    seeds = ["P00533", "Q9Y6K9", "P20930", "P35225", "O14633", "EGFR", "GRB2"]
    asked: list[str] = []
    lock = threading.Lock()
    live = {"now": 0, "most": 0}

    def read_value(url: str) -> object:
        with lock:
            asked.append(url)
            live["now"] += 1
            live["most"] = max(live["most"], live["now"])
        time.sleep(0.02)
        with lock:
            live["now"] -= 1
        if url.endswith("/pathways"):
            return [{"stId": "R-HSA-1"}]
        return [{"schemaClass": "ReferenceGeneProduct", "identifier": "P99999"}]

    found = _reactome_for_seeds(read_value, seeds, 8)

    # Only the UniProt-shaped seeds have pathways; the gene symbols are skipped.
    assert sorted(found) == ["O14633", "P00533", "P20930", "P35225", "Q9Y6K9"]
    assert all(members == ["P99999"] for members in found.values())
    assert live["most"] > 1, "the pathway queries were made one at a time"

    # One pathways request per UniProt seed, and no seed asked twice.
    pathways = [url for url in asked if url.endswith("/pathways")]
    assert len(pathways) == 5
    assert len(set(pathways)) == 5

    # A single seed needs no pool, and still answers the same way.
    assert _reactome_for_seeds(read_value, ["P00533"], 8) == {"P00533": ["P99999"]}
    assert _reactome_for_seeds(read_value, ["EGFR"], 8) == {}


def test_record_scores_are_looked_up_concurrently_in_listed_order():
    """Eight databases answering in turn per target was most of the wait.

    Every database is still asked about every target and nothing is skipped; the
    answers are assembled in listed order, so the scores and their order are what
    they were. The progress count still rises one at a time.
    """
    rows = [
        StructureOption(
            group="protein",
            accession=f"P{20000 + index}",
            entry_id=f"{index}ABC",
            chain="A",
            title="t",
            predicted=False,
            source="fixture",
        )
        for index in range(10)
    ]
    asked: list[str] = []
    lock = threading.Lock()
    live = {"now": 0, "most": 0}
    ticks: list[tuple[int, int, str]] = []

    def read(url: str) -> object:
        with lock:
            asked.append(url)
            live["now"] += 1
            live["most"] = max(live["most"], live["now"])
        time.sleep(0.01)
        with lock:
            live["now"] -= 1
        return {}

    hits, scores = collect_record_scores(rows, read, on_progress=lambda *args: ticks.append(args))

    # Several targets were in progress at once.
    assert live["most"] > 1, "the targets were scored one at a time"

    # The answer is the listed one, in listed order: nothing dropped, nothing moved.
    assert list(scores) == [row.accession.casefold() for row in rows]
    assert list(hits) == [row.accession for row in rows]
    assert all(score.asked == len(SOURCE_ORDER) for score in scores.values())
    assert all(set(found) == set(SOURCE_ORDER) for found in hits.values())

    # Every database was asked about every target. TTD is one whole-list fetch for
    # the run, taken once, not once per target.
    assert len([url for url in asked if TTD_TARGETS_URL in url]) == 1
    assert len([url for url in asked if "uniprot" in url]) == len(rows)
    assert len([url for url in asked if "pride" in url]) == len(rows)

    # The count still rises one at a time and names a target each time.
    assert [tick[0] for tick in ticks] == list(range(1, len(rows) + 1))
    assert {tick[1] for tick in ticks} == {len(rows)}
    assert sorted(tick[2] for tick in ticks) == sorted(row.accession for row in rows)

    # No targets means no requests at all, TTD included.
    asked.clear()
    empty_hits, empty_scores = collect_record_scores([], read)
    assert empty_hits == {} and empty_scores == {}
    assert asked == []


def test_deposit_records_are_read_several_at_a_time_without_moving_a_row():
    """Reading one record per deposit, one at a time, was most of the wait.

    346 serial reads for one disease is why the page sat there. They are read
    concurrently now, and the rows are still assembled by the serial loop that
    always assembled them, because a row's listed position is the number --pick
    names, the index a class tick selects by, and its key in the saved report.
    """
    ids = [f"{n}TUP" for n in range(1, 13)]
    order: list[str] = []
    lock = threading.Lock()
    live = {"now": 0, "most": 0}

    def post(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL:
            return _associated_targets(("TP53", "P04637"))
        if UNIPROT_ATTRIBUTE not in json.dumps(payload):
            return {"result_set": []}
        return {"result_set": [{"identifier": f"{entry}_1"} for entry in ids]}

    def get(url: str) -> dict:
        with lock:
            order.append(url)
            live["now"] += 1
            live["most"] = max(live["most"], live["now"])
        time.sleep(0.02)
        with lock:
            live["now"] -= 1
        return _entry_record(url.rsplit("/", 1)[-1])

    found = collect_structure_options("EFO_1", post=post, get=get, row_limit=len(ids))

    # Several records were in flight at once: that is the whole point.
    assert live["most"] > 1, "the records were read one at a time"
    assert live["most"] <= MAX_ENTRY_WORKERS

    # One request per deposit, and no deposit read twice.
    assert sorted(order) == sorted(entry_url(entry) for entry in ids)
    assert len(set(order)) == len(order)

    # The rows come back in the search's order, not the order the records arrived.
    assert [row.entry_id for row in found] == ids

    # And that order is exactly what reading them one at a time produces.
    serial: list[str] = []

    def one_at_a_time(url: str) -> dict:
        serial.append(url)
        return _entry_record(url.rsplit("/", 1)[-1])

    plain = collect_structure_options("EFO_1", post=post, get=one_at_a_time, row_limit=len(ids))
    assert [row.entry_id for row in plain] == [row.entry_id for row in found]
    assert [(row.group, row.accession, row.chain) for row in plain] == [
        (row.group, row.accession, row.chain) for row in found
    ]


def test_a_repeated_deposit_is_still_read_once_and_alone_needs_no_pool():
    """Deduplication happens before anything is dispatched.

    One deposit answering many rows must still cost one request, and a single
    record is read by the row loop rather than by a pool started for one call.
    """
    reads: list[str] = []

    def post(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL:
            return _associated_targets(("TP53", "P04637"))
        if UNIPROT_ATTRIBUTE not in json.dumps(payload):
            return {"result_set": []}
        return {"result_set": [{"identifier": "1TUP_1"}, {"identifier": "1TUP_2"}, {"identifier": "1TUP_3"}]}

    def get(url: str) -> dict:
        reads.append(url)
        return _entry_record("p53")

    found = collect_structure_options("EFO_1", post=post, get=get, row_limit=5)
    assert reads == [entry_url("1TUP")], "one deposit, one request, in the row loop"
    assert [row.chain for row in found] == ["1", "2", "3"]


def test_the_caller_can_hold_the_deposit_records_so_going_back_rereads_nothing():
    """Going back restarts the run, and the lookup is nearly all a restart costs.

    A record store the caller keeps makes the second pass read nothing it already
    read. A failed lookup is not kept: it would leave the row marked not known to
    be experimental for the rest of the session with no way to retry it.
    """
    reads: list[str] = []
    broken = {"1BAD"}

    def post(url: str, payload: dict) -> dict:
        if url == OPENTARGETS_URL:
            return _associated_targets(("TP53", "P04637"))
        if UNIPROT_ATTRIBUTE not in json.dumps(payload):
            return {"result_set": []}
        return {"result_set": [{"identifier": "1TUP_1"}, {"identifier": "2TUP_1"}, {"identifier": "1BAD_1"}]}

    def get(url: str) -> dict:
        reads.append(url)
        entry = url.rsplit("/", 1)[-1]
        if entry in broken:
            raise OSError("down")
        return _entry_record(entry)

    held: dict[str, dict] = {}
    first = collect_structure_options("EFO_1", post=post, get=get, row_limit=5, records=held)
    assert sorted(reads) == sorted(entry_url(entry) for entry in ("1TUP", "2TUP", "1BAD"))
    assert sorted(held) == ["1TUP", "2TUP"], "the failed lookup is not kept"
    assert all(entry for entry in held.values())

    # Second pass: the two records that were read are not read again. The one that
    # failed is retried, because a failure is not an answer.
    reads.clear()
    broken.clear()
    second = collect_structure_options("EFO_1", post=post, get=get, row_limit=5, records=held)
    assert reads == [entry_url("1BAD")]
    assert sorted(held) == ["1BAD", "1TUP", "2TUP"]

    # Same rows, same order, whether or not the records came from the store.
    assert [(row.entry_id, row.chain, row.group) for row in second] == [
        (row.entry_id, row.chain, row.group) for row in first
    ]
    # The retry landed: the row is known to be experimental on the second pass.
    assert [row.methodology_known for row in first] == [True, True, False]
    assert [row.methodology_known for row in second] == [True, True, True]

    # Without a store, nothing is shared and every record is read again.
    reads.clear()
    collect_structure_options("EFO_1", post=post, get=get, row_limit=5)
    assert len(reads) == 3


def test_an_interactive_run_confirms_a_single_hit(tmp_path: Path, monkeypatch, capsys):
    block_live_calls(monkeypatch)
    records = tmp_path / "records.json"
    records.write_text(
        '[{"group": "protein", "accession": "P04637", "entry_id": "1tup", "chain": "1",'
        ' "title": "p53", "predicted": false, "source": "fixture"}]',
        encoding="utf-8",
    )
    prompts: list[str] = []

    def answer_yes(prompt: str) -> str:
        prompts.append(prompt)
        return "y"

    argv = [
        "--catalog",
        str(asthma_catalog(tmp_path)),
        "--disease",
        "Asthma",
        "--structures-json",
        str(records),
        "--all",
    ]
    # Without --accept-only the single hit has to be offered, not declined on the user's behalf.
    code = cure_main(
        [*argv, "--out-dir", str(tmp_path / "yes")],
        fetch=lambda option: b"data_1TUP\n#\n",
        read_line=answer_yes,
    )
    assert code == 0
    assert any("Confirm Asthma" in prompt for prompt in prompts)

    code_no = cure_main(
        [*argv, "--out-dir", str(tmp_path / "no")],
        fetch=lambda option: b"data_1TUP\n#\n",
        read_line=lambda prompt: "n",
    )
    assert code_no == 1
    assert "Disease not found" in capsys.readouterr().out


def test_an_unknown_disease_asks_for_another_name(tmp_path: Path, monkeypatch, capsys):
    block_live_calls(monkeypatch)
    catalog = asthma_catalog(tmp_path)
    records = tmp_path / "records.json"
    records.write_text(
        '[{"group": "protein", "accession": "P20930", "entry_id": "4pcw", "chain": "1",'
        ' "title": "filaggrin", "predicted": false, "source": "fixture"}]',
        encoding="utf-8",
    )
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return "Asthma"

    code = cure_main(
        [
            "--catalog",
            str(catalog),
            "--disease",
            "not-a-disease",
            "--structures-json",
            str(records),
            "--all",
            "--accept-only",
            "--confirm-inference",
            "--out-dir",
            str(tmp_path / "again"),
        ],
        fetch=lambda _option: b"data_4PCW\n#\n",
        read_line=answer,
    )
    captured = capsys.readouterr().out
    assert code == 0
    assert prompts == ["Disease name: "]
    assert "Disease not found" in captured
    assert "Confirmed: Asthma" in captured


def test_the_panel_asks_again_when_the_disease_is_not_found(tmp_path, monkeypatch, capsys):
    catalog = asthma_catalog(tmp_path)
    records = tmp_path / "records.json"
    records.write_text(
        '[{"group": "protein", "accession": "P20930", "entry_id": "4pcw", "chain": "1",'
        ' "title": "filaggrin", "predicted": false, "source": "fixture"}]',
        encoding="utf-8",
    )
    out = tmp_path / "panel-again"
    pages: list[str] = []

    def fields(name: str) -> dict[str, list[str]]:
        return _panel_landing(catalog, records, out, disease=[name], confirm_inference=["yes"])

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if _is_generation_page(page):
            return {"accept": ["yes"]}
        if "Disease not found" in page:
            return fields("Asthma")
        return fields("not-a-disease")

    code = _run_panel(
        monkeypatch,
        exchange,
        runner=lambda _argv: 0,  # Continue starts generation now.
        fetch=lambda _option: b"data_4PCW\n#\n",
    )
    assert code == 0
    assert len(pages) == 3, "landing, not found, then the generation page"
    assert "Disease not found" in pages[1]
    captured = capsys.readouterr().out
    assert "Confirmed: Asthma" in captured
    assert captured.strip().endswith("Generation finished.")


def test_several_names_ask_for_a_choice_rather_than_reporting_nothing(tmp_path, monkeypatch, capsys):
    block_live_calls(monkeypatch)
    catalog = tmp_path / "two.json"
    catalog.write_text(
        '[{"name": "Breast cancer", "disease_id": "A"}, {"name": "Breast carcinoma", "disease_id": "B"}]',
        encoding="utf-8",
    )
    base = ["--catalog", str(catalog), "--disease", "breast", "--out-dir", str(tmp_path / "runs")]
    assert cure_main(base, read_line=lambda prompt: "") == 2
    assert "Re-run with --choice" in capsys.readouterr().err
    assert cure_main([*base, "--choice", "9"], read_line=lambda prompt: "") == 2
    assert "outside 1..2" in capsys.readouterr().err


def test_no_arguments_open_the_option_menu_and_help_does_not():
    assert wants_option_ui([]) is True
    assert wants_option_ui(["--ui"]) is True
    assert wants_option_ui(["--disease", "Asthma", "--ui"]) is True
    assert wants_option_ui(["--help"]) is False
    assert wants_option_ui(["-h"]) is False
    assert wants_option_ui(["--disease", "Asthma"]) is False


def _ssh_files(tmp_path, monkeypatch, config: str = "", known: str = ""):
    """An ssh directory the discovery reads instead of the real one."""
    directory = tmp_path / "ssh"
    directory.mkdir(parents=True, exist_ok=True)
    if config:
        (directory / "config").write_text(config, encoding="utf-8")
    if known:
        (directory / "known_hosts").write_text(known, encoding="utf-8")
    monkeypatch.setenv("RFD4_SSH_DIR", str(directory))
    return directory


def test_remote_machines_come_from_the_ssh_config_and_known_hosts(tmp_path, monkeypatch):
    config = """
# a comment
Host vm-a
  HostName 10.0.0.1
Host vm-b gpu-box
  User ds
Host *.internal
Host *
  ForwardAgent yes
Host=vm-c
"""
    known = """
vm-d ssh-ed25519 AAAAC3Nz
[vm-e]:2222 ssh-rsa AAAAB3Nz
vm-f,10.0.0.9 ssh-ed25519 AAAAC3Nz
|1|hashedsalt|hashedhost ssh-ed25519 AAAAC3Nz
@cert-authority vm-g ssh-ed25519 AAAAC3Nz
localhost ssh-ed25519 AAAAC3Nz
VM-A ssh-ed25519 AAAAC3Nz
"""
    _ssh_files(tmp_path, monkeypatch, config, known)
    found = discover_remote_machines()
    assert found == ["10.0.0.9", "gpu-box", "vm-a", "vm-b", "vm-c", "vm-d", "vm-e", "vm-f", "vm-g"]
    # A wildcard names a set, a hashed entry names nothing readable, and this machine
    # is added by the dropdown rather than ticked.
    assert not any("*" in name for name in found)
    assert not any(name.startswith("|") for name in found)
    assert "localhost" not in found
    assert found.count("vm-a") == 1, "VM-A is the same machine as vm-a"

    assert discover_remote_machines("gpu") == ["gpu-box"]
    assert discover_remote_machines("VM-B") == ["vm-b"], "the search ignores case"
    assert discover_remote_machines("nothing-like-this") == []
    assert discover_remote_machines(limit=2) == ["10.0.0.9", "gpu-box"]


def test_no_ssh_files_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RFD4_SSH_DIR", str(tmp_path / "absent"))
    assert discover_remote_machines() == []
    assert ssh_directory() == tmp_path / "absent"


def test_the_machines_panel_lists_and_searches_remote_machines(tmp_path, monkeypatch):
    _ssh_files(tmp_path, monkeypatch, "Host vm-a\nHost gpu-box\n")
    page = render_options(OptionForm(), machines=tuple(discover_remote_machines()), machine_search="")
    assert 'name="machine_search"' in page
    assert 'name="__action" value="search"' in page
    assert ">Search machines</button>" in page
    assert ">Windows File Explorer</button>" in page
    assert page.index(">Search machines</button>") < page.index(">Windows File Explorer</button>")
    assert 'name="__action" value="browse-machine"' in page
    assert "Select a remote machine in Windows File Explorer." in page
    assert 'name="remote_pick" value="vm-a"' in page
    assert 'name="remote_pick" value="gpu-box"' in page
    assert "2 machine(s) from your ssh config and known_hosts" in page

    narrowed = render_options(OptionForm(), machines=("gpu-box",), machine_search="gpu")
    assert 'value="gpu"' in narrowed
    assert 'name="remote_pick" value="vm-a"' not in narrowed

    empty = render_options(OptionForm(), machines=(), machine_search="zzz")
    assert "No known machine matches zzz" in empty
    assert "No machines found in your ssh config" in render_options(OptionForm())


def test_windows_file_explorer_adds_the_selected_remote_machine(tmp_path, monkeypatch):
    """The button beside Search machines puts the chosen computer in the box."""
    _ssh_files(tmp_path, monkeypatch, "Host vm-a\n")
    monkeypatch.setattr("rfproteina.disease.ui.pick_remote_machine", lambda: "lab-box")
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        common = {**_PANEL_FIELDS, "where": ["local-and-remote"]}
        if len(pages) == 1:
            return {**common, "hosts": ["vm-typed"], "__action": ["browse-machine"]}
        held = page.split('<textarea name="hosts">', 1)[1].split("</textarea>", 1)[0]
        return {**common, "hosts": [held]}

    argv = OptionPanel(exchange).collect(["--ui"], build_parser())
    assert len(pages) == 2
    assert "vm-typed,lab-box" in pages[1]
    assert "No remote machine was selected." not in pages[1]
    assert argv[argv.index("--hosts") + 1] == "local,vm-typed,lab-box"


def test_a_cancelled_explorer_stays_on_the_page_and_a_folder_is_refused(tmp_path, monkeypatch):
    _ssh_files(tmp_path, monkeypatch, "Host vm-a\n")
    answers = iter(["", ValueError("Choose a computer in Network. A local folder is not a remote machine.")])

    def picker():
        answer = next(answers)
        if isinstance(answer, ValueError):
            raise answer
        return answer

    monkeypatch.setattr("rfproteina.disease.ui.pick_remote_machine", picker)
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        common = {**_PANEL_FIELDS, "where": ["local"]}
        if len(pages) < 3:
            return {**common, "__action": ["browse-machine"]}
        return common

    argv = OptionPanel(exchange).collect(["--ui"], build_parser())
    assert "No remote machine was selected." in pages[1]
    assert "A local folder is not a remote machine." in pages[2]
    assert "--hosts" not in argv


def test_explorer_reads_a_network_computer_and_refuses_a_local_path(monkeypatch):
    assert machine_name_from_explorer(r"\\lab-box\share") == "lab-box"
    assert machine_name_from_explorer("lab-box") == "lab-box"
    assert machine_name_from_explorer(r"C:\Users\someone") == ""
    assert machine_name_from_explorer("localhost") == ""
    assert machine_name_from_explorer("-oProxyCommand") == ""

    monkeypatch.setattr("rfproteina.disease.grok_bot.windows_powershell", lambda env=None: "")
    with pytest.raises(ValueError, match="not on this computer"):
        pick_remote_machine()

    monkeypatch.setattr("rfproteina.disease.grok_bot.windows_powershell", lambda env=None: "powershell.exe")

    def run(args, **kwargs):
        assert args[1:3] == ["-NoProfile", "-Sta"]
        assert kwargs.get("shell") is not True
        if run.mode == "computer":
            text, code = b"\\\\lab-box\\share\n", 0
        elif run.mode == "cancel":
            text, code = b"", 2
        else:
            text, code = b"C:\\Users\\someone\n", 0

        class Result:
            returncode = code
            stdout = text

        return Result()

    run.mode = "computer"
    monkeypatch.setattr("rfproteina.disease.machines.subprocess.run", run)
    assert pick_remote_machine() == "lab-box"
    run.mode = "cancel"
    assert pick_remote_machine() == ""
    run.mode = "folder"
    with pytest.raises(ValueError, match="not a remote machine"):
        pick_remote_machine()


def test_a_ticked_machine_becomes_a_named_host(tmp_path, monkeypatch, capsys):
    _ssh_files(tmp_path, monkeypatch, "Host vm-a\nHost vm-b\n")
    argv = OptionPanel(
        lambda _page: {
            **_PANEL_FIELDS,
            "where": ["local-and-remote"],
            "hosts": ["vm-typed"],
            "remote_pick": ["vm-a", "vm-b", "vm-typed"],
        }
    ).collect(["--ui"], build_parser())
    hosts = argv[argv.index("--hosts") + 1]
    assert hosts == "local,vm-typed,vm-a,vm-b", "typed first, then ticked, with no duplicate"


def test_ticking_machines_while_set_to_this_machine_only_asks_again(tmp_path, monkeypatch):
    """Silently dropping ticked machines would run nothing remotely and say nothing."""
    _ssh_files(tmp_path, monkeypatch, "Host vm-a\n")
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        common = dict(_PANEL_FIELDS)
        if len(pages) == 1:
            return {**common, "where": ["local"], "remote_pick": ["vm-a"]}
        return {**common, "where": ["local-and-remote"], "remote_pick": ["vm-a"]}

    argv = OptionPanel(exchange).collect(["--ui"], build_parser())
    assert len(pages) == 2
    assert "Remote machines are named but generation is set to this machine only" in pages[1]
    assert argv[argv.index("--hosts") + 1] == "local,vm-a"


def test_searching_keeps_what_was_typed_and_what_was_ticked(tmp_path, monkeypatch):
    _ssh_files(tmp_path, monkeypatch, "Host vm-a\nHost gpu-box\n")
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        common = {**_PANEL_FIELDS, "binder_kind": ["ligand"], "out_dir": ["kept-out"], "where": ["local-and-remote"]}
        if len(pages) == 1:
            # Tick one machine, then search instead of continuing.
            return {**common, "remote_pick": ["vm-a"], "machine_search": ["gpu"], "__action": ["search"]}
        # A browser submits whatever the textarea now holds, which is where the
        # machine ticked before the search was folded.
        held = page.split('<textarea name="hosts">', 1)[1].split("</textarea>", 1)[0]
        return {**common, "hosts": [held], "remote_pick": ["gpu-box"]}

    argv = OptionPanel(exchange).collect(["--ui"], build_parser())
    assert len(pages) == 2, "the search re-showed the page rather than continuing"

    second = pages[1]
    assert 'value="gpu"' in second, "the search text is kept"
    assert "vm-a" in second, "the ticked machine survived the search, in the text box"
    assert 'name="remote_pick" value="vm-a"' not in second, "the list is narrowed to the match"
    assert 'value="Asthma"' in second
    assert f'name="out_dir" value="{display_directory("kept-out")}"' in second
    assert 'name="remote_dir" value="disease_runs"' in second
    assert '<option value="ligand" selected>' in second

    # Both machines are named: the one ticked before the search and the one after.
    assert argv[argv.index("--hosts") + 1] == "local,vm-a,gpu-box"


def test_the_landing_page_takes_a_custom_target_upload():
    page = render_options(OptionForm())
    assert 'enctype="multipart/form-data"' in page, "a file needs multipart"
    assert "Upload Individual Custom Disease to Investigate - Atomic level Shape .PDB .CIF files" in page
    assert 'type="file" name="target_structure"' in page
    assert 'accept=".pdb,.cif,.PDB,.CIF"' in page
    # Below the disease name, as asked.
    assert page.index('name="disease"') < page.index('name="target_structure"')

    # A file input cannot be pre-filled, so a chosen path is carried and shown. The
    # file input comes first, so a newly chosen file wins over the carried one.
    chosen = render_options(OptionForm(target_structure="/tmp/up/egfr.cif"))
    assert '<input type="hidden" name="target_structure" value="/tmp/up/egfr.cif">' in chosen
    assert "/tmp/up/egfr.cif" in chosen
    assert chosen.index('type="file" name="target_structure"') < chosen.index('type="hidden" name="target_structure"')


def test_a_multipart_submission_saves_the_file_and_passes_its_path(tmp_path):
    uploads = tmp_path / "uploads"
    boundary = "----rfd4boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="disease"\r\n\r\n'
        "Asthma\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="target_structure"; filename="egfr.cif"\r\n'
        "Content-Type: chemical/x-cif\r\n\r\n"
        "data_1EGFR\n#\n\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    fields = parse_submission(body, f"multipart/form-data; boundary={boundary}", uploads)
    assert fields["disease"] == ["Asthma"]
    saved = Path(fields["target_structure"][0])
    assert saved.parent == uploads
    assert saved.name == "egfr.cif"
    assert saved.read_text(encoding="utf-8") == "data_1EGFR\n#\n"

    # The ordinary encoding still parses, so every other page is unaffected.
    plain = parse_submission(b"disease=Asthma&pick=1", "application/x-www-form-urlencoded", uploads)
    assert plain == {"disease": ["Asthma"], "pick": ["1"]}


def test_an_upload_cannot_write_outside_its_directory(tmp_path):
    uploads = tmp_path / "uploads"
    saved = save_upload(uploads, "../../etc/passwd", b"data_X\n")
    assert saved.parent == uploads, "the name is reduced to its last component"
    assert saved.name == "etcpasswd" or ".." not in saved.name
    assert not (tmp_path.parent / "etc").exists()

    windows = save_upload(uploads, r"C:\\Users\\me\\shape.cif", b"data_X\n")
    assert windows.parent == uploads
    assert windows.name == "shape.cif"

    # No size limit: a file past the old 50 MB cap is kept whole.
    big = save_upload(uploads, "big.cif", b"x" * 50_000_001)
    assert big.stat().st_size == 50_000_001


def test_an_upload_with_the_wrong_suffix_is_refused():
    with pytest.raises(ValueError, match=r"Upload a \.pdb, \.cif file"):
        form_from_fields({**_PANEL_FIELDS, "target_structure": ["/tmp/up/notes.txt"]})
    kept = form_from_fields({**_PANEL_FIELDS, "target_structure": ["/tmp/up/shape.PDB"]})
    assert kept.target_structure == "/tmp/up/shape.PDB", "the suffix check ignores case"


def test_an_uploaded_target_is_listed_with_the_searched_structures(tmp_path, monkeypatch, capsys):
    records = structures_json(tmp_path, ROW_P00533)
    custom = tmp_path / "my-target.cif"
    custom.write_text("data_MINE\n_atom_site.id 1\n", encoding="utf-8")
    out = tmp_path / "with-upload"
    fetched: list[str] = []

    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--target-structure", str(custom), "--pick", "1", "--confirm-inference"],
        out_dir=out,
        fetch=lambda option: fetched.append(option.entry_id) or b"data_X\n#\n",
    )
    code = captured.code
    assert code == 0, captured.err
    assert f"Custom target: {out / 'uploads' / 'my-target.cif'}" in captured.out

    # It is first on the list, so --pick 1 names it, and it was copied beside the run.
    assert fetched == ["my-target"]
    assert (out / "uploads" / "my-target.cif").read_text(encoding="utf-8") == "data_MINE\n_atom_site.id 1\n"
    assert custom.exists(), "the original is left where it was"


@pytest.mark.parametrize(
    ("name", "body", "reason"),
    [
        ("shape.txt", b"data_X\n", "not one of"),
        ("shape.cif", b"", "is empty"),
        ("shape.cif", b"\x1f\x8b\x08\x00rest", "compressed bytes"),
        ("shape.cif", b"\xff\xfe\x00binary", "not text"),
        ("shape.cif", b"nothing structural here\n", "does not read as"),
        ("shape.pdb", b"data_X\n", "does not read as"),
    ],
)
def test_a_bad_custom_target_is_refused_without_writing_it(tmp_path, monkeypatch, capsys, name, body, reason):
    records = structures_json(tmp_path, ROW_P00533)
    custom = tmp_path / name
    custom.write_bytes(body)
    out = tmp_path / f"refused-{name}-{reason[:6]}".replace(" ", "-")

    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--target-structure", str(custom), "--all", "--confirm-inference"],
        out_dir=out,
        fetch=lambda _option: (_ for _ in ()).throw(AssertionError("download")),
    )
    code = captured.code
    assert code == 2
    assert "The uploaded target could not be read:" in captured.err
    assert reason in captured.err
    assert not (out / "uploads").exists(), "nothing is kept from a refused file"


def test_the_structures_directory_shows_complete_paths(tmp_path):
    """Output and download show A:/... The remote directory stays a plain name."""
    shown_out = display_directory("disease_runs")
    page = render_options(OptionForm())
    assert f'name="out_dir" value="{shown_out}"' in page
    assert f'name="download_dir" value="{shown_out}"' in page
    assert shown_out.endswith("/disease_runs")
    assert shown_out != "disease_runs"
    assert 'name="remote_dir" value="disease_runs"' in page
    resolved = Path("disease_runs").resolve().as_posix()
    if resolved.startswith("/mnt/") and resolved[5:6].isalpha() and (len(resolved) == 6 or resolved[6] == "/"):
        assert shown_out.startswith(f"{resolved[5].upper()}:/")

    kept = render_options(OptionForm(out_dir="kept-out", download_dir="downloads"))
    assert f'name="out_dir" value="{display_directory("kept-out")}"' in kept
    assert f'name="download_dir" value="{display_directory("downloads")}"' in kept

    posted_download = display_directory(str(tmp_path / "fetched"))
    fields = {
        "disease": ["Asthma"],
        "binder_kind": ["protein"],
        "out_dir": [shown_out],
        "download_dir": [posted_download],
        "remote_dir": ["disease_runs"],
    }
    form = form_from_fields(fields)
    assert Path(form.out_dir).resolve() == Path("disease_runs").resolve()
    assert form.download_dir is not None
    assert Path(form.download_dir).resolve() == (tmp_path / "fetched").resolve()
    assert form.remote_dir == "disease_runs"
    if resolved.startswith("/mnt/"):
        assert form.out_dir.startswith("/mnt/")

    same = form_from_fields({**fields, "download_dir": [shown_out]})
    assert same.download_dir is None
    relative = form_from_fields({**fields, "out_dir": ["out"], "download_dir": ["out"]})
    assert relative.out_dir == "out"
    assert relative.download_dir is None

    again = render_options(draft_from_fields(fields))
    assert f'name="out_dir" value="{shown_out}"' in again
    assert f'name="download_dir" value="{posted_download}"' in again
    assert 'name="remote_dir" value="disease_runs"' in again
    assert f'name="generation_archive" value="{display_directory(str(Path("disease_runs") / "generations"))}"' in page
    assert "readonly" in page.split('name="generation_archive"', 1)[1].split(">", 1)[0]
    assert (Path("disease_runs") / "generations").is_dir()
    kept_archive = display_directory(str(Path("kept-out") / "generations"))
    assert f'name="generation_archive" value="{kept_archive}"' in kept
    assert not Path("kept-out").exists()


def test_a_generation_archive_keeps_every_run_and_stays_in_its_folder(tmp_path):
    """Two saves of one target both remain, and a hostile name cannot leave the archive."""
    out = tmp_path / "runs"
    out.mkdir()
    sample = out / "gene_8TLD_2" / "inference_samples"
    sample.mkdir(parents=True)
    (sample / "design.cif").write_text("data_design\n", encoding="utf-8")
    (out / "gene_8TLD_2.cif").write_text("data_8TLD\n", encoding="utf-8")
    kwargs = {
        "disease_name": "../Asthma/../../outside",
        "disease_id": "MONDO_0004979",
        "target_id": "IL5/../../tmp",
        "group": "gene",
        "entry_id": "8TLD",
        "chain": "2",
        "binder_kind": "protein",
    }
    first = save_generation(out, out / "gene_8TLD_2", **kwargs)
    second = save_generation(out, out / "gene_8TLD_2", **kwargs)
    root = (out / "generations").resolve()
    assert first.resolve().is_relative_to(root)
    assert second.resolve().is_relative_to(root)
    assert first != second
    assert ".." not in first.relative_to(root).parts
    assert ".." not in second.relative_to(root).parts
    assert (first / "inference_samples" / "design.cif").read_text(encoding="utf-8") == "data_design\n"
    assert (second / "inference_samples" / "design.cif").read_text(encoding="utf-8") == "data_design\n"
    assert (sample / "design.cif").is_file()
    assert (first / "gene_8TLD_2.cif").read_text(encoding="utf-8") == "data_8TLD\n"
    record = json.loads((first / "generation.json").read_text(encoding="utf-8"))
    assert record["disease_name"] == "../Asthma/../../outside"
    assert record["target_id"] == "IL5/../../tmp"
    assert record["entry_id"] == "8TLD"
    assert record["chain"] == "2"


def test_output_directory_files_can_be_searched_and_selected_with_the_mouse(tmp_path):
    """A click on a listed file or folder chooses the output directory."""
    folder = tmp_path / "outputs"
    nested = folder / "gene_one"
    nested.mkdir(parents=True)
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    (folder / "other.cif").write_text("data_Y\n", encoding="utf-8")
    target = nested / "shape.cif"
    target.write_text("data_X\n", encoding="utf-8")

    page = render_options(OptionForm(out_dir=str(folder)))
    assert '<div class="screen-readout">' not in page
    assert 'name="out_file_search"' in page
    assert 'name="__action" value="search-files"' in page
    assert ">Search files</button>" in page
    assert 'name="out_pick"' in page
    assert display_directory(str(folder / "other.cif")) in page
    assert display_directory(str(nested)) in page
    assert page.index(">Search files</button>") < page.rfind('<button type="submit">Continue</button>')
    last = page.rfind('<button type="submit"')
    assert page[last:].startswith('<button type="submit">Continue</button>')
    assert 'name="remote_dir" value="disease_runs"' in page

    rows = list_output_entries(str(folder), "")
    assert ("dir", str(nested.resolve())) in rows
    assert ("file", str((folder / "other.cif").resolve())) in rows
    assert any(kind == "parent" for kind, _path in rows)

    narrowed = render_options(OptionForm(out_dir=str(folder)), out_file_search="shape")
    assert display_directory(str(target)) in narrowed
    assert "other.cif" not in narrowed
    assert "notes.txt" not in narrowed

    missing = render_options(OptionForm(out_dir=str(tmp_path / "missing-out")))
    assert "This output directory does not exist yet." in missing
    assert ">..  " in missing

    moved, selected_file = choose_output_entry(OptionForm(out_dir=str(folder)), str(nested))
    assert Path(moved.out_dir).resolve() == nested.resolve()
    assert selected_file == ""
    chosen_file, selected = choose_output_entry(OptionForm(out_dir=str(folder)), str(target))
    assert Path(chosen_file.out_dir).resolve() == nested.resolve()
    assert Path(selected).resolve() == target.resolve()
    marked = render_options(chosen_file, out_file_search="shape", selected_out_file=selected)
    assert 'aria-pressed="true"' in marked
    assert f"Selected {display_directory(str(target))}" in marked

    pages: list[str] = []

    def exchange(shown: str) -> dict[str, list[str]]:
        pages.append(shown)
        common = {
            "disease": ["Asthma"],
            "binder_kind": ["protein"],
            "remote_dir": ["disease_runs"],
        }
        if len(pages) == 1:
            return {
                **common,
                "out_dir": [str(folder)],
                "out_file_search": ["shape"],
                "__action": ["search-files"],
            }
        if len(pages) == 2:
            assert "other.cif" not in shown
            assert display_directory(str(target)) in shown
            return {**common, "out_dir": [str(folder)], "out_pick": [str(target)]}
        assert display_directory(str(nested)) in shown
        assert 'aria-pressed="true"' in shown
        assert 'name="remote_dir" value="disease_runs"' in shown
        return {**common, "out_dir": [display_directory(str(nested))]}

    argv = OptionPanel(exchange).collect(["--ui"], build_parser())
    assert len(pages) == 3
    assert Path(argv[argv.index("--out-dir") + 1]).resolve() == nested.resolve()
    assert argv[argv.index("--remote-dir") + 1] == "disease_runs"


def _screen_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "call": "untested",
        "untested": ["immunogenicity"],
        "sequence_developability": {
            "gravy": 0.1304,
            "instability_index": 30.0034,
            "sequence_pass": False,
        },
        "pre_oracle_pass": True,
        "ppi_pass": "absent",
    }
    payload.update(overrides)
    return payload


def _write_screen(folder: Path, payload: dict[str, object]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "screen.json").write_text(json.dumps(payload), encoding="utf-8")


def test_the_structures_directory_shows_the_screen_for_the_open_folder(tmp_path):
    """The output directory lists each finished screen. Opening one folder shows that screen."""
    folder = tmp_path / "gene_8TLD_2"
    _write_screen(folder, _screen_payload())
    parent = render_options(OptionForm(out_dir=str(tmp_path)))
    assert '<p class="screen-where">gene_8TLD_2</p>' in parent
    assert "Screen 8TLD chain 2: untested. Untested: immunogenicity." in parent
    parent_start = parent.index('<div class="screen-readout">')
    parent_block = parent[parent_start : parent.index("</div>", parent_start)]
    assert 'class="notice"' not in parent_block
    assert parent.index('<div class="screen-readout">') < parent.index('name="out_file_search"')

    opened, selected = choose_output_entry(OptionForm(out_dir=str(tmp_path)), str(folder))
    assert selected == ""
    page = render_options(opened)
    start = page.index('<div class="screen-readout">')
    block = page[start : page.index("</div>", start)]
    assert 'class="notice"' not in block
    assert 'class="screen-where"' not in block
    assert "Screen 8TLD chain 2: untested. Untested: immunogenicity." in block
    assert "sequence_pass false" in block
    assert "pre_oracle_pass true" in block
    assert page.index('<div class="screen-readout">') < page.index('name="out_file_search"')

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "screen.json").write_text("{", encoding="utf-8")
    broken_page = render_options(OptionForm(out_dir=str(broken)))
    assert '<div class="screen-readout">' not in broken_page
    assert "Screen " not in broken_page

    archive = tmp_path / "gene_8TLD_2-2"
    archive.mkdir()
    (archive / "generation.json").write_text(
        json.dumps({"entry_id": "8TLD", "chain": "2"}),
        encoding="utf-8",
    )
    _write_screen(archive, _screen_payload())
    assert "Screen 8TLD chain 2: untested." in render_options(OptionForm(out_dir=str(archive)))

    named = tmp_path / "notes"
    named.mkdir()
    (named / "generation.json").write_text(
        json.dumps({"entry_id": "8TLD", "chain": "2"}),
        encoding="utf-8",
    )
    _write_screen(named, _screen_payload())
    assert screen_lines_for_directory(named)[0].startswith("Screen 8TLD chain 2: untested.")

    plain = tmp_path / "plain"
    _write_screen(plain, _screen_payload())
    unnamed = screen_lines_for_directory(plain)
    assert unnamed[0] == "Screen: untested. Untested: immunogenicity."
    assert unnamed[1].startswith("Measured: gravy ")
    assert "sequence_pass false" in unnamed[1]
    assert "pre_oracle_pass true" in unnamed[1]
    assert "ppi.pass absent" in unnamed[1]
    assert "Measured  chain" not in unnamed[1]

    leaked = tmp_path / "gene_9GVN_1"
    _write_screen(leaked, _screen_payload(call="/home/someone/private-design"))
    leaked_page = render_options(OptionForm(out_dir=str(leaked)))
    assert "/home/someone" not in leaked_page
    assert "&lt;home&gt;/private-design" in leaked_page

    bundle = tmp_path / "bundle"
    live = bundle / "gene_8TLD_2"
    archive = bundle / "generations" / "asthma" / "IL5" / "gene_8TLD_2-2"
    other = bundle / "gene_9WWH_3"
    hidden = live / "inference_samples" / "decoy"
    _write_screen(live, _screen_payload())
    _write_screen(archive, _screen_payload())
    _write_screen(other, _screen_payload(call="axis_hit"))
    _write_screen(hidden, _screen_payload(call="screened_clear"))
    reports = screen_reports_for_directory(bundle)
    assert [label for label, _lines in reports] == ["gene_8TLD_2", "gene_9WWH_3"]
    assert reports[0][1][0].startswith("Screen 8TLD chain 2: untested.")
    assert reports[1][1][0].startswith("Screen 9WWH chain 3: axis_hit.")
    assert all("screened_clear" not in line for _label, lines in reports for line in lines)
    assert [label for label, _lines in screen_reports_for_directory(bundle, limit=1)] == ["gene_8TLD_2"]


def test_the_option_panel_uses_dropdowns_and_tick_boxes():
    html = render_options(OptionForm())
    first, second = PANEL_TITLE.splitlines()
    assert f"<h1>{first.replace('   ', '&nbsp;&nbsp;&nbsp;')}<br>{second}</h1>" in html
    assert f"<title>{first} {second}</title>" in html
    assert "RFD4 cure" not in html
    assert '<select name="binder_kind"' in html
    assert '<select name="where"' in html
    assert '<select name="local_workers"' in html
    assert '<select name="remote_workers"' in html
    assert 'name="remote_workers_custom"' in html
    # The relabelled fields, and the sentence that stopped being true the moment
    # a remote cap existed.
    assert "Local workers (structures to bind, locally)" in html
    assert "Remote workers (structures to bind, remotely)" in html
    assert "Remote machines stay unlimited" not in html
    assert "Search for remote machines" in html
    assert "Accept the single matched name" not in html
    assert "Similar-name list number" not in html
    assert "Leave blank to pick the name from a dropdown after the search." not in html
    assert 'name="accept_only"' not in html
    assert 'type="number" name="choice"' not in html
    prefilled = render_options(OptionForm(accept_only=True, choice=2))
    assert 'type="hidden" name="accept_only" value="yes"' in prefilled
    assert 'type="hidden" name="choice" value="2"' in prefilled
    assert "Accept the single matched name" not in prefilled
    assert 'class="page-actions page-actions-top"' in html
    assert "Set the run here." not in html
    assert "Generation starts only if that tick box is checked." not in html
    assert "form.landing { display: grid; grid-template-columns: 1fr auto;" in html
    assert "form.landing > h1 { grid-column: 1; grid-row: 1;" in html
    assert "form.landing > .page-actions-top { grid-column: 2; grid-row: 1;" in html
    # The whole panel order, as one sequence rather than a few pairwise
    # comparisons. API keys sits in the left column directly under Binder and
    # model, which left Structures Directory alone in the right column; pairwise
    # index checks let a panel move between the pairs they happen to name
    # without any of them noticing.
    assert re.findall(r"<h2>([^<]+)</h2>", html) == [
        "Disease",
        "Binder and model",
        "API keys",
        "Machines",
        "Structures Directory",
    ]
    assert "panel-disease" in html and "panel-binder" in html and "panel-machines" in html
    # Binder and Machines share Disease's column. They are not on a row with the
    # file list, so that list cannot push them to the bottom of the page.
    assert "grid-template-areas" not in html
    stacks = html.split('<div class="grid">', 1)[1].split('<div class="stack">')
    assert len(stacks) == 3
    left, right = stacks[1], stacks[2]
    assert (
        left.index("panel-disease")
        < left.index("panel-binder")
        < left.index("panel-api")
        < left.index("panel-machines")
    )
    assert "panel-structures" not in left
    # Structures Directory has the right column to itself. The grid is 1fr 1fr
    # with align-items: start, so the short column simply ends sooner.
    assert "panel-structures" in right
    assert "panel-api" not in right
    assert "panel-binder" not in right
    assert "@media (min-width: 700px)" in html
    assert ".grid { grid-template-columns: 1fr 1fr; }" in html
    assert "Output directory" in html
    assert "Download directory" in html
    assert "Generation archive" in html
    assert 'name="generation_archive"' in html
    assert 'name="out_dir"' in html
    assert 'name="download_dir"' in html
    assert "Structure records JSON" not in html
    assert "Download all experimental structures" not in html
    assert "Structure numbers" not in html
    assert 'name="structures_json"' not in html
    assert 'name="take_all"' not in html
    assert 'name="pick"' not in html
    kept = render_options(
        OptionForm(structures_json="structures.json", take_all=True, pick="1,3", download_dir="downloads")
    )
    assert 'type="hidden" name="structures_json" value="structures.json"' in kept
    assert 'type="hidden" name="take_all" value="yes"' in kept
    assert 'type="hidden" name="pick" value="1,3"' in kept
    assert f'name="download_dir" value="{display_directory("downloads")}"' in kept
    assert 'type="checkbox" name="confirm_inference"' in html
    assert "Confirm inference without another prompt (useful for uploaded targets)" in html
    # Beside the chooser it is about: same row, after the file input, before the
    # section ends. A tick two sections away from its upload was easy to miss.
    upload = html.index('type="file" name="target_structure"')
    tick = html.index('name="confirm_inference"')
    assert upload < tick < html.index('class="panel-binder"'), "the tick left the disease section"
    assert html.count('class="row"', upload, tick) == 0, "the chooser and the tick are in one row"
    # The generation page asks this two steps later, so the first page does not.
    assert 'type="checkbox" name="run"' not in html
    assert "Run generation after planning" not in html
    # A flag can still set it, and the page keeps it without showing a tick.
    assert 'type="hidden" name="run" value="yes"' not in html
    assert 'type="hidden" name="run" value="yes"' in render_options(OptionForm(run=True))
    assert 'type="password" name="grok_key"' in html
    assert 'type="password" name="claude_key"' in html
    assert 'type="password" name="openai_key"' in html
    assert 'type="password" name="nvidia_key"' in html
    assert 'type="password" name="openrouter_key"' in html
    assert '<select name="nvidia_model"' in html
    assert '<select name="claude_model"' in html
    assert '<select name="grok_model"' in html
    assert '<select name="openai_model"' in html
    assert '<select name="openrouter_model"' in html
    assert html.count('class="llm"') == 5
    assert html.count('type="button" class="nav llm-add"') == 5
    assert html.count('type="button" class="nav llm-remove"') == 5
    actions = html.split('class="llm-actions"', 1)[1].split("</div>", 1)[0]
    assert actions.index("llm-remove") < actions.index("llm-add")
    assert ".llm > input[type=password] { grid-column: 1; grid-row: 2;" in html
    assert ".llm > select { grid-column: 2; grid-row: 2;" in html
    assert ".llm-actions { grid-column: 3; grid-row: 2;" in html
    assert ".llm-saved { grid-column: 1; grid-row: 3;" in html
    assert "button.llm-add, button.llm-remove { margin: 0; padding: 3px 8px;" in html
    assert 'value="*****"' not in html
    key_at: list[int] = []
    for label in ("Grok", "Claude", "OpenAI", "NVIDIA", "OpenRouter"):
        key_pos = html.index(f"{label} API Key")
        model_pos = html.index(f"{label} model")
        assert key_pos < model_pos
        key_at.append(key_pos)
    assert key_at == sorted(key_at)
    for env_name in _KEY_ENV_NAMES:
        assert f"Leave blank to use {env_name}" not in html
    assert "Leave blank to keep it." not in html
    assert "Leave the last box clear to print commands and not start generation." not in html
    assert "Sent with this key." not in html
    assert "is used when this page is not submitted." not in html
    for model_id in (
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "deepseek-ai/deepseek-v4.1-flash",
        "z-ai/glm-5.3-flash",
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-opus-5",
        "grok-4.7",
        "grok-4.6",
        "grok-4.5",
        "gpt-6.1-terra",
        "gpt-6-luna",
        "gpt-6.1-sol",
        "nvidia/nemotron-3.5-super-vl",
    ):
        assert f'value="{model_id}"' in html
    assert ">Fable 5.1</option>" in html
    assert ">Opus 5.5</option>" in html
    assert ">Opus 5</option>" in html
    assert ">Terra</option>" in html
    assert ">Luna</option>" in html
    assert ">Sol</option>" in html
    assert ">Nemotron 3.5 Super VL 120B</option>" in html
    assert ">DeepSeek V4.1 Flash</option>" in html
    assert ">GLM 5.3 Flash</option>" in html
    assert (
        "nvidia/nemotron-3.5-super-vl-120b-a12b"
        not in html.split('name="openrouter_model"', 1)[1].split("</select>", 1)[0]
    )
    assert "Every local request at once" in html
    disease = render_disease_choice([Disease("Breast cancer", "A"), Disease("Breast carcinoma", "B")])
    assert '<select name="choice"' in disease
    assert "Breast carcinoma (B)" in disease
    confirm = render_confirm(Disease("Asthma", "EFO_0000270"))
    assert 'type="checkbox" name="accept"' in confirm
    assert '<label class="field disease-choice">Matching names' in confirm
    assert '<select name="choice"' in confirm
    assert "Asthma (EFO_0000270)" in confirm
    structures = render_structures([_option("protein", "1egfr"), _option("protein", "1il13", predicted=True)])
    assert 'type="checkbox" name="all" value="yes"' in structures
    assert "All Structure" in structures
    assert "All experimental structures" not in structures
    assert "The first box adds every experimental row." not in structures
    assert "Program score is 100 for an experimental structure" not in structures
    assert structures.count('name="pick"') == 2
    assert "predicted" in structures
    # No score at all without the record counts, and no deposit-method score ever.
    assert "program score" not in structures
    scored = render_structures(
        [_option("protein", "1egfr"), _option("protein", "1il13", predicted=True)],
        scores={"acc-1egfr": (3, 8)},
    )
    assert "Instances of Structure in same databases 3" in scored
    assert "Instances of Structure in same databases 3/8" not in scored
    assert "Instances of Structure in different databases 3/8" in scored
    assert "program score" not in scored
    assert "program score 100" not in scored, "the deposit method is not a score"
    assert "The score is how many of UniProt, NCBI, STRING, IntAct, the Human" in scored
    assert "Returned structures are saved for comparison with the LLM search." in structures
    assert 'name="sort"' in structures
    # "Score" is the different-databases count the hint above the rows describes.
    # The added orders name their metric in the words the rows already use.
    assert ">Score highest</option>" in structures
    assert ">Score lowest</option>" in structures
    assert ">Instances in same databases highest</option>" in structures
    assert ">Instances in same databases lowest</option>" in structures
    assert ">LLM higher value</option>" in structures
    assert ">LLM lower value</option>" in structures
    assert 'name="experimental"' in structures
    assert 'name="predicted"' in structures
    assert ">Include</option>" in structures
    assert ">Exclude</option>" in structures
    assert 'data-kind="experimental"' in structures
    assert 'data-kind="predicted"' in structures
    assert 'data-score="3"' in scored
    assert 'class="ticks structure-rows"' in structures
    assert "You can initiate a binding design campaign against the following targets:" in structures
    assert "Click a class to select its structures." in structures
    assert 'class="campaign-select"' in structures
    assert 'name="campaign" value="Proteins"' in structures
    assert 'name="campaign" value="DNA &amp; RNA"' in structures
    assert 'name="campaign" value="Small Molecules"' in structures
    assert 'name="campaign" value="Sugars (Glycans)"' in structures
    assert 'name="campaign" value="Antibodies &amp; Nanobodies"' in structures
    assert 'data-campaigns="Proteins"' in structures
    assert "Proteins: 2 structures" in structures
    assert "DNA &amp; RNA: 0 structures" in structures
    assert "Small Molecules: 0 structures" in structures
    assert "Sugars (Glycans): 0 structures" in structures
    assert "Antibodies &amp; Nanobodies: 0 structures" in structures
    assert structures.count("<h2>LLM search</h2>") == 1
    assert "<h2>Models</h2>" not in structures
    assert 'name="all_models" value="yes"' in structures
    assert "Search every listed model (With saved API Key)" in structures
    assert "Search every listed model</label>" not in structures
    assert "Search every listed model (With API Key)" not in structures
    assert 'name="model" value="grok|' not in structures, "a keyed model with no API key is not offered"
    assert 'value="grokbot|chief-of-staff" data-saved="no"' in structures
    assert "Grok Bot</label>" in structures
    assert "No API key. Submit question pastes this box into the open Chief of Staff chat" in structures
    assert "Grok 4.7</label>" not in structures
    keyed = render_structures(
        [_option("protein", "1egfr"), _option("protein", "1il13", predicted=True)],
        saved=("grok", "claude", "openai", "nvidia", "openrouter"),
    )
    assert "Search with every API key" not in keyed
    assert 'class="ticks llm-search"' in keyed
    assert "Grok 4.7</label>" in keyed
    assert "Claude Fable 5.1</label>" in keyed
    assert "OpenAI Terra</label>" in keyed
    # The provider is named where the model's own name does not name it, so the
    # two hosts that both offer Nemotron stay apart.
    assert "NVIDIA Nemotron 3.5 Super VL 120B</label>" in keyed
    assert "OpenRouter Nemotron 3.5 Super VL</label>" in keyed
    assert "grok Grok 4.7" not in keyed, "the provider id is not part of a model name"
    assert "nvidia Nemotron 3 Ultra" not in keyed, "nor in front of one that needs the provider"
    assert 'name="model" value="claude|claude-fable-5-1"' in keyed
    assert 'value="grok|grok-4.7" data-saved="yes"' in keyed
    assert "first page" not in structures
    assert 'name="research"' not in structures
    assert "Tick providers, models, or both" not in structures
    assert ">Download Structures</button>" in structures
    assert structures.index("<h2>Structures</h2>") < structures.index('name="pick"')
    assert structures.index("<h2>LLM search</h2>") < structures.index("<h2>Structures</h2>")
    assert structures.index("<h2>LLM search</h2>") < structures.index(">Download Structures</button>")

    def exchange(page: str) -> dict[str, list[str]]:
        assert "<select" in page and 'type="checkbox"' in page
        return {
            "disease": ["Asthma"],
            "catalog": ["catalog.json"],
            "accept_only": ["yes"],
            "choice": ["2"],
            "structures_json": ["structures.json"],
            "take_all": ["yes"],
            "pick": ["1,3"],
            "binder_kind": ["peptide"],
            "out_dir": ["out"],
            "where": ["local-and-remote"],
            "hosts": ["vm-a"],
            "remote_dir": ["remote_runs"],
            "local_workers": ["custom"],
            "local_workers_custom": ["4"],
            "modified_ckpt": ["modified.ckpt"],
            "confirm_inference": ["yes"],
            "run": ["yes"],
        }

    argv = OptionPanel(exchange).collect(["--ui"], build_parser())
    assert argv == [
        "--ui",
        "--disease",
        "Asthma",
        "--catalog",
        "catalog.json",
        "--accept-only",
        "--choice",
        "2",
        "--structures-json",
        "structures.json",
        "--all",
        "--pick",
        "1,3",
        "--binder-kind",
        "peptide",
        "--out-dir",
        "out",
        "--hosts",
        "local,vm-a",
        "--remote-dir",
        "remote_runs",
        "--local-workers",
        "4",
        "--modified-ckpt",
        "modified.ckpt",
        "--confirm-inference",
        "--run",
    ]
    with pytest.raises(ValueError, match="binder kind"):
        OptionPanel(lambda _page: {"binder_kind": ["toxin"]}).collect(["--ui"], build_parser())


def test_choosing_remote_machines_with_none_named_keeps_the_panel_open(tmp_path, monkeypatch, capsys):
    """The reported shutdown: a refused field puts the page back, it does not stop the run.

    Choosing remote machines and naming none used to raise out of main(), which closed
    the browser session and lost the whole form.
    """
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "recovered"
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if _is_generation_page(page):
            return {"accept": ["yes"]}
        if 'name="binder_kind"' not in page:
            return {}  # the search page: leave the providers unticked
        common = _panel_landing(catalog, records, out, confirm_inference=["yes"])
        if len(pages) == 1:
            return {**common, "where": ["remote"], "hosts": [""]}
        return {**common, "where": ["local"]}

    code = _run_panel(
        monkeypatch,
        exchange,
        runner=lambda _argv: 0,  # Continue starts generation now.
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=lambda _url: {},
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Traceback" not in captured.err
    assert len(pages) == 3, "landing page, the same page again, then generation"

    # The second page explains the refusal and keeps what was typed.
    assert "Name at least one remote machine" in pages[1]
    assert 'value="Asthma"' in pages[1]
    assert '<option value="remote" selected>' in pages[1], "the dropdown keeps the choice"
    assert str(out) in pages[1]

    # And the run went on to plan generation rather than stopping.
    assert "Selected 1 structure(s) for download." in captured.out
    assert captured.out.strip().endswith("Generation finished.")


def test_the_run_log_records_a_refusal_and_never_an_api_key(tmp_path, monkeypatch, capsys):
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "logged"
    log_file = tmp_path / "logs" / "run.log"
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' not in page:
            return {}
        common = _panel_landing(catalog, records, out, confirm_inference=["yes"], grok_key=["sk-must-not-be-logged"])
        if len(pages) == 1:
            return {**common, "local_workers": ["custom"], "local_workers_custom": ["many"]}
        return {**common, "local_workers": ["every"]}

    code = _run_panel(
        monkeypatch,
        exchange,
        "--log-file",
        str(log_file),
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=lambda _url: {},
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert f"Run log: {log_file}" in captured.out
    assert "Local workers must be a whole number" in pages[1]

    written = log_file.read_text(encoding="utf-8")
    assert "human started" in written
    assert "option panel rejected a field: Local workers must be a whole number." in written
    assert "confirmed disease Asthma (EFO_0000270)" in written
    assert "selected 1 structure(s)" in written
    assert "human finished with exit 0" in written

    # The key was typed, so the log says a key was there and not what it was.
    assert "sk-must-not-be-logged" not in written
    assert "[redacted]" in written


def test_an_unexpected_failure_is_logged_with_its_traceback(tmp_path, monkeypatch, capsys):
    """An error no handler expected is recorded, not printed as a raw traceback."""
    records = structures_json(tmp_path, ROW_P00533)
    log_file = tmp_path / "run.log"

    def fetch(_option):
        raise RuntimeError("the mirror went away")

    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--all", "--confirm-inference", "--log-file", str(log_file)],
        out_dir=tmp_path / "boom",
        fetch=fetch,
    )
    code = captured.code
    assert code == 2
    assert "human stopped on an unexpected error." in captured.err
    assert str(log_file) in captured.err
    assert "Traceback" not in captured.err, "the traceback belongs in the log, not on the terminal"

    written = log_file.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in written
    assert "RuntimeError: the mirror went away" in written


def test_a_log_that_cannot_be_opened_does_not_stop_the_run(tmp_path, monkeypatch, capsys):
    records = structures_json(tmp_path, ROW_P00533)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")

    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--all", "--confirm-inference", "--log-file", str(blocker / "run.log")],
        out_dir=tmp_path / "unlogged",
        fetch=lambda _option: b"data_1EGFR\n#\n",
    )
    code = captured.code
    assert code == 0, captured.err
    assert "The run log could not be opened" in captured.out
    assert "Selected 1 structure(s) for download." in captured.out


def test_a_blank_disease_stays_on_the_panel():
    calls = {"n": 0}

    def exchange(page: str) -> dict[str, list[str]]:
        calls["n"] += 1
        if calls["n"] == 1:
            assert "Enter the scientific" not in page
            assert 'name="disease" value="" required' in page
            return {"binder_kind": ["protein"], "where": ["local"]}
        assert "Enter the scientific or medical name of the disease." in page
        return {"disease": ["Asthma"], "binder_kind": ["protein"], "where": ["local"]}

    argv = OptionPanel(exchange).collect(["--ui"], build_parser())
    assert calls["n"] == 2
    assert argv[:3] == ["--ui", "--disease", "Asthma"]


def test_the_panel_opens_through_powershell_outside_this_directory(monkeypatch, capsys):
    calls: dict[str, object] = {}

    class _Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        calls["cwd"] = kwargs.get("cwd")
        return _Result()

    monkeypatch.setattr("rfproteina.disease.ui.subprocess.run", fake_run)
    monkeypatch.setattr("rfproteina.disease.ui._running_in_wsl", lambda: True)
    monkeypatch.setattr("rfproteina.disease.ui._windows_cwd", lambda: "/mnt/c/Windows")
    _open_browser("http://127.0.0.1:9/")
    assert calls["cwd"] == "/mnt/c/Windows"
    argv = calls["argv"]
    assert argv[:3] == ["powershell.exe", "-NoProfile", "-Command"]
    assert "http://127.0.0.1:9/" in argv[3]
    assert "--app=" in argv[3]
    # The profile directory is this panel's own, named for its port. PowerShell
    # joins it, because Windows' temp path is not knowable from WSL, so the name
    # is pinned as the two halves that build it rather than as one literal.
    assert "$port='9'" in argv[3]
    assert "'rfd4-panel-' + $port" in argv[3]
    assert "--user-data-dir=' + $dir" in argv[3]
    assert capsys.readouterr().out == ""

    class _Denied:
        returncode = 1
        stderr = "Access is denied."
        stdout = ""

    monkeypatch.setattr("rfproteina.disease.ui.subprocess.run", lambda *_args, **_kwargs: _Denied())
    _open_browser("http://127.0.0.1:9/")
    message = capsys.readouterr().out
    assert "Paste the panel address" in message
    assert "Access is denied." in message


def test_the_panel_ticks_structures_and_builds_both_host_commands(tmp_path, monkeypatch, capsys):
    catalog = asthma_catalog(tmp_path)
    records = structures_json(
        tmp_path,
        ROW_P00533,
        {
            "group": "protein",
            "accession": "P35225",
            "entry_id": "1IL13",
            "chain": "A",
            "title": "il13",
            "predicted": False,
            "source": "fixture",
        },
    )
    out = tmp_path / "ui-runs"
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(
                catalog,
                records,
                out,
                take_all=False,
                binder_kind=["peptide"],
                where=["local-and-remote"],
                hosts=["vm-a"],
                local_workers=["every"],
                confirm_inference=["yes"],
            )
        if 'type="checkbox" name="pick"' in page:
            assert "Instances of Structure in same databases 0" in page
            assert "Instances of Structure in different databases 0/8" in page
            assert 'name="all_models" value="yes"' in page
            assert "Grok 4.7</label>" not in page, "no API key, so the model is not offered"
            assert "Proteins:" in page
            assert "Search with every API key" not in page
            assert "The program search is already confirmed." not in page
            return {"all": ["yes"]}
        if _is_generation_page(page):
            return {"accept": ["yes"]}
        raise AssertionError("unexpected page")

    def unused(*_args, **_kwargs):
        raise AssertionError("network")

    code = _run_panel(
        monkeypatch,
        exchange,
        post=unused,
        get=unused,
        fetch=lambda _option: b"data_source\n",
        # The structures page asks the public databases for each row's program score.
        read_json=lambda _url: {},
        runner=lambda _argv: 0,
        # This run is local-and-remote, and the page starts generation now, so
        # the remote half would ssh and scp for real without these.
        transfer=lambda _host, _local, _remote: None,
        retrieve=lambda _host, _remote, _local: None,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert len(pages) == 3, "landing, structures, then generation; clear model ticks skip the search"
    assert 'type="checkbox" name="pick"' in pages[1]
    assert 'name="all" value="yes"' in pages[1]
    assert "Instances of Structure in different databases 0/8" in pages[1]
    assert "program score" not in pages[1], "that label was replaced"
    assert "Compared with the program search:" not in captured.out
    assert "Using the original release model." in captured.out
    assert captured.out.strip().endswith("Generation finished.")
    commands = _commands(captured.out)
    assert len(commands) == 2
    assert "+disease.host=local" in commands[0]
    assert "+disease.host=vm-a" in commands[1]
    for name in ("protein_1EGFR_A", "protein_1IL13_A"):
        spec = json.loads((out / f"{name}.job.json").read_text(encoding="utf-8"))
        assert spec["binder_kind"] == "peptide"
        assert spec["generation_started"] is True
        text = (out / f"{name}.annotated.cif").read_text(encoding="utf-8")
        assert "_rfproteina_disease.binder_kind peptide" in text


def test_back_replays_the_earlier_answers_and_forward_restores_the_later_one():
    """The replay core. Going back cannot unwind the flow, so answers are replayed."""
    shown: list[str] = []
    replies: list[dict[str, list[str]]] = [{"one": ["a"]}, {"two": ["b"]}, {"__nav": ["back"]}]

    def exchange(page: str) -> dict[str, list[str]]:
        shown.append(page)
        return replies[len(shown) - 1]

    panel = OptionPanel(exchange)
    assert panel.ask("step0" + NAV_SLOT) == {"one": ["a"]}
    assert "Back" not in shown[0], "the first page has nowhere to go back to"
    assert panel.ask("step1" + NAV_SLOT) == {"two": ["b"]}
    assert ">Back</button>" in shown[1]
    with pytest.raises(StepBackError) as back:
        panel.ask("step2" + NAV_SLOT)
    assert back.value.target == 1

    panel.rewind(back.value.target)
    replies[:] = [{"__nav": ["forward"]}]
    shown.clear()
    assert panel.ask("step0" + NAV_SLOT) == {"one": ["a"]}
    assert shown == [], "a replayed answer does not show its page again"
    assert panel.ask("step1" + NAV_SLOT) == {"two": ["b"]}, "Forward restores the answer"
    assert ">Forward</button>" in shown[0]


def test_a_back_names_the_step_it_wants_and_cannot_overshoot():
    """The browser's button names a step; an in-page button means the previous one."""
    assert back_target({"__nav": ["back"]}, 3) == 2
    assert back_target({"__nav": ["back"], "__to": ["0"]}, 3) == 0
    assert back_target({"__nav": ["back"], "__to": ["9"]}, 3) == 2, "clamped to the page before"
    # A step that is not a plain number is malformed, and means one step back.
    assert back_target({"__nav": ["back"], "__to": ["-4"]}, 3) == 2
    assert back_target({"__nav": ["back"], "__to": ["x"]}, 3) == 2
    assert back_target({"__nav": ["back"], "__to": [""]}, 3) == 2


def test_the_panel_can_go_back_more_than_one_step():
    shown: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        shown.append(page)
        if len(shown) == 3:
            return {"__nav": ["back"], "__to": ["0"]}
        return {"n": [str(len(shown))]}

    panel = OptionPanel(exchange)
    panel.ask("step0" + NAV_SLOT)
    panel.ask("step1" + NAV_SLOT)
    with pytest.raises(StepBackError) as back:
        panel.ask("step2" + NAV_SLOT)
    assert back.value.target == 0, "two steps back, not one"

    panel.rewind(back.value.target)
    assert panel.ask("step0" + NAV_SLOT) == {"n": ["4"]}
    assert len(shown) == 4, "the first page is shown again, because nothing precedes it"


def test_a_changed_answer_stops_forward_past_that_page():
    shown: list[str] = []
    replies: list[dict[str, list[str]]] = [{"one": ["a"]}, {"two": ["b"]}, {"__nav": ["back"]}]

    def exchange(page: str) -> dict[str, list[str]]:
        shown.append(page)
        return replies[len(shown) - 1]

    panel = OptionPanel(exchange)
    panel.ask("step0" + NAV_SLOT)
    panel.ask("step1" + NAV_SLOT)
    with pytest.raises(StepBackError):
        panel.ask("step2" + NAV_SLOT)
    panel.rewind(0)

    replies[:] = [{"one": ["CHANGED"]}, {"two": ["b"]}]
    shown.clear()
    assert panel.ask("step0" + NAV_SLOT) == {"one": ["CHANGED"]}
    assert ">Forward</button>" in shown[0], "Forward was offered before the answer changed"
    assert panel.ask("step1" + NAV_SLOT) == {"two": ["b"]}
    assert ">Forward</button>" not in shown[1], "a changed answer drops the answers after it"


def test_the_panel_goes_back_from_the_structures_page_to_the_landing_page(tmp_path, monkeypatch, capsys):
    """Back on a later page re-shows the earlier one and the run still finishes."""
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "went-back"
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False, confirm_inference=["yes"])
        if _is_generation_page(page):
            return {"accept": ["yes"]}
        # The structures page: go back the first time it is shown, then take every row.
        seen_pages = sum(1 for seen in pages if 'type="checkbox" name="pick"' in seen)
        return {"__nav": ["back"]} if seen_pages == 1 else {"all": ["yes"]}

    code = _run_panel(
        monkeypatch,
        exchange,
        runner=lambda _argv: 0,  # Continue starts generation now.
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=lambda _url: {},
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Traceback" not in captured.err
    assert len(pages) == 5, "landing, structures, landing again, structures again, generation"
    assert ">Back</button>" not in pages[0], "nothing precedes the landing page"
    assert ">Back</button>" in pages[1], "the structures page can go back"
    assert 'name="binder_kind"' in pages[2], "Back returned to the landing page"
    assert ">Forward</button>" in pages[2], "the answer it had is offered again"
    assert "Selected 1 structure(s) for download." in captured.out
    assert captured.out.strip().endswith("Generation finished.")


def test_the_browser_back_button_rewinds_the_panel():
    """A GET of an earlier step is a Back: the flow rewinds and serves what it shows."""
    import urllib.request

    session = BrowserSession(opener=lambda _url: None)
    delivered: list[dict[str, list[str]]] = []

    def serve() -> None:
        delivered.append(session.exchange("<p>one</p>", step=1))
        session.exchange("<p>zero again</p>", step=0)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        for _ in range(100):
            if session._server is not None and session.step() == 1 and session.page() == "<p>one</p>":
                break
            time.sleep(0.02)
        assert session._server is not None
        assert session.step() == 1, "the page has to be showing before it is requested"
        base = f"http://127.0.0.1:{session._server.server_address[1]}"

        # The step being shown serves its own page, and an unknown step redirects to it.
        assert urllib.request.urlopen(base + "/step/1", timeout=5).read().decode("utf-8") == "<p>one</p>"
        landed = urllib.request.urlopen(base + "/step/7", timeout=5)
        assert landed.geturl().endswith("/step/1")
        assert urllib.request.urlopen(base + "/", timeout=5).geturl().endswith("/step/1")

        # Going back to step 0 hands the waiting flow a Back and serves its next page.
        body = urllib.request.urlopen(base + "/step/0", timeout=5).read().decode("utf-8")
        assert body == "<p>zero again</p>"
        assert delivered == [{"__nav": ["back"], "__to": ["0"]}]
    finally:
        session._posted.set()
        session._next_ready.set()
        session._result.set()
        worker.join(timeout=2)
        session.close()


def test_back_and_forward_skip_form_validation():
    """A required field must not trap the user on a page they are leaving."""
    page = with_nav(render_options(OptionForm()), back=True, forward=True)
    assert "required" in page, "the landing page does have a required field"
    for value in ("back", "forward"):
        button = f'name="__nav" value="{value}" formnovalidate'
        assert button in page, f"the {value} button must not be blocked by validation"
    assert NAV_SLOT not in page, "the slot is filled in, not left in the page"

    without = render_options(OptionForm())
    assert NAV_SLOT in without, "an unshown page keeps the slot for the panel to fill"
    assert ">Back</button>" not in with_nav(without, back=False, forward=False)


def _is_generation_page(page: str) -> bool:
    """The generation page, recognised by the model radios only it has.

    The first page's confirm tick answers that page's inference question rather
    than skipping the page, because the model is chosen there and generation is
    started there. A panel run reaches it whether or not the tick is set.
    """
    return 'name="generate_with"' in page


def _panel_landing(catalog, records, out, *, take_all: bool = True, **extra) -> dict[str, list[str]]:
    """The landing fields a panel test submits, without confirming inference.

    ``take_all=False`` leaves the structures page to be shown, which is where the
    LLM search is now offered.
    """
    fields = {
        "disease": ["Asthma"],
        "catalog": [str(catalog)],
        "accept_only": ["yes"],
        "structures_json": [str(records)],
        "take_all": ["yes"],
        "binder_kind": ["protein"],
        "out_dir": [str(out)],
        "where": ["local"],
        "remote_dir": ["disease_runs"],
    }
    if not take_all:
        del fields["take_all"]
    fields.update(extra)
    return fields


def test_the_panel_asks_for_inference_instead_of_the_terminal(tmp_path, monkeypatch, capsys):
    """The reported gap: the confirmation was a terminal prompt a panel run could not answer."""
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "panel-inference"
    pages: list[str] = []
    started: list[list[str]] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out)
        return {"accept": ["yes"], "run": ["yes"]}

    code = _run_panel(
        monkeypatch,
        exchange,
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=lambda _url: {},
        runner=lambda argv: started.append(list(argv)) or 0,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert len(pages) == 2, "landing, then inference"

    inference = pages[1]
    assert "Run inference on 1 downloaded structure(s)" in inference
    assert 'name="accept" value="yes" checked' in inference
    # No second box: a page whose action is Continue cannot also ask whether to
    # continue, and the box sat directly above the button that already did it.
    assert 'name="run"' not in inference
    assert "Confirm inference for" not in captured.out, "the terminal must not be asked"

    # Continuing with the inference box ticked starts generation.
    assert "Starting 1 local and 0 remote generation request(s)." in captured.out
    assert captured.out.strip().endswith("Generation finished.")
    assert [argv[0] for argv in started] == ["rfd4-generate"]


def test_clearing_the_inference_box_stops_and_keeps_the_downloads(tmp_path, monkeypatch, capsys):
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "declined"

    def exchange(page: str) -> dict[str, list[str]]:
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out)
        return {}  # neither box ticked

    code = _run_panel(
        monkeypatch,
        exchange,
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=lambda _url: {},
        runner=lambda _argv: (_ for _ in ()).throw(AssertionError("generation")),
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Inference was not confirmed. Generation was not started." in captured.out
    assert (out / "protein_1EGFR_A.cif").exists(), "the downloads are kept"
    assert list(out.glob("*.annotated.cif")) == [], "no job was written"


def test_download_goes_straight_to_the_generation_page_after_a_search(tmp_path, monkeypatch, capsys):
    """Pressing Download reaches the generation page and does not ask a model.

    The search button already asked. Download fetches the ticked structure and
    does not write a second model report.

    The fetch is recorded as it happens, so the same run says what Download
    fetched and that nothing was fetched before it: the ticked structure, once,
    after the last page that could still change the selection.
    """
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "straight-to-generation"
    pages: list[str] = []
    fetched: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False)
        if 'type="checkbox" name="pick"' in page:
            assert fetched == [], "nothing is fetched while the selection can still change"
            # The tick has to be offered, or the search below is quietly skipped
            # and this stops testing the page sequence after a search.
            assert "Grok 4.7</label>" in page
            assert "Tick a model for the search." not in page
            shown = sum(1 for seen in pages if 'type="checkbox" name="pick"' in seen)
            if shown == 1:
                return {"pick": ["1"], "model": ["grok|grok-4.7"], "next": ["missing"]}
            return {"pick": ["1"]}
        # The only page left is the one Download leads to.
        assert '<p class="step-title">Confirm inference</p>' in page
        assert "Program search and LLM options" not in page
        return {"accept": ["yes"]}

    model_calls = {"n": 0}

    def complete(_provider: str, _system: str, _user: str) -> str:
        model_calls["n"] += 1
        return json.dumps({"rating": 50, "reason": "", "ideas": "An idea.", "structures": []})

    code = _run_panel(
        monkeypatch,
        exchange,
        runner=lambda _argv: 0,  # Continue starts generation now.
        fetch=lambda option: fetched.append(option.entry_id) or b"data_1EGFR\n#\n",
        read_json=lambda _url: {},
        complete=complete,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert len(pages) == 4, "landing, structures, structures after the search, confirm inference"
    assert 'type="checkbox" name="pick"' in pages[2]
    assert '<p class="step-title">Confirm inference</p>' in pages[3]
    assert not any("Program search and LLM options" in page for page in pages)
    assert fetched == ["1EGFR"], "the ticked structure, and only once Download was pressed"
    # The search button asked the model. Download did not ask it again.
    assert model_calls["n"] == 1
    assert "Compared with the program search:" not in captured.out
    assert "Target research" not in captured.out
    assert not (out / "target_research.json").exists()


def test_download_fetches_the_selected_structure_from_the_searched_archive(tmp_path, monkeypatch, capsys):
    """Download asks the structure archive for the ticked deposit and does not call a model.

    The rank button is the model call, and it happens before Download. Download
    then reads one mmCIF from the RCSB file archive the structure search already
    uses. The other listed deposit is not fetched, and the record databases are
    not asked again.
    """
    block_live_calls(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533, {**ROW_P35225, "source": "rcsb"})
    out = tmp_path / "selected-archive"
    reads: list[str] = []
    structure_urls: list[str] = []
    model_calls = {"n": 0}
    at_download = {"reads": 0, "models": 0, "reads_after": 0}

    def read(url: str):
        reads.append(url)
        return {}

    def complete(_provider: str, _system: str, user: str) -> str:
        model_calls["n"] += 1
        assert "1EGFR" in user
        assert "2IL13" in user
        return json.dumps({"ranked": [{"pdb_id": "1EGFR", "chain": "A", "rating": 20, "reason": "Listed."}]})

    def structure_open(request, timeout=0):
        structure_urls.append(request.full_url)
        return _Response(b"data_1EGFR\n#\n")

    monkeypatch.setattr("rfproteina.disease.sources.urlopen", structure_open)

    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False, confirm_inference=["yes"])
        if 'type="checkbox" name="pick"' in page:
            shown = sum(1 for seen in pages if 'type="checkbox" name="pick"' in seen)
            if shown == 1:
                return {"model": ["grok|grok-4.7"], "next": ["rank"]}
            at_download["reads"] = len(reads)
            at_download["models"] = model_calls["n"]
            return {"pick": ["1"]}
        if _is_generation_page(page):
            # Download is finished. Anything read from here belongs to the
            # post-generation screen, not to Download.
            at_download["reads_after"] = len(reads)
            return {"accept": ["yes"]}
        raise AssertionError(f"no page follows the structures page: {page[:200]}")

    code = cure_main(
        ["--ui"],
        runner=lambda _argv: 0,  # Continue starts generation now.
        read_json=read,
        complete=complete,
        exchange=exchange,
        read_line=refuse_prompt,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert at_download["models"] == 1
    assert model_calls["n"] == 1, "Download does not call a model"
    assert at_download["reads_after"] == at_download["reads"], "Download does not ask the record databases again"
    assert structure_urls == ["https://files.rcsb.org/download/1EGFR.cif"]
    assert "2IL13" not in " ".join(structure_urls)
    for url in structure_urls:
        assert "api.x.ai" not in url
        assert "api.openai.com" not in url
        assert "api.anthropic.com" not in url
        assert "integrate.api.nvidia.com" not in url
        assert "openrouter.ai" not in url
    assert "Selected 1 structure(s) for download." in captured.out
    assert not (out / "target_research.json").exists()


def test_found_structures_are_appended_and_research_keeps_its_rank(tmp_path, monkeypatch, capsys):
    """A new find takes the next number after the list, and an earlier rank stays."""
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    catalog = asthma_catalog(tmp_path)
    records = structures_json(
        tmp_path,
        {
            "group": "protein",
            "accession": "P00533",
            "entry_id": "1EGF",
            "chain": "A",
            "title": "egfr",
            "predicted": False,
            "source": "rcsb",
        },
        {
            "group": "protein",
            "accession": "P35225",
            "entry_id": "2IL1",
            "chain": "B",
            "title": "interleukin",
            "predicted": False,
            "source": "rcsb",
        },
        {
            "group": "dna",
            "accession": "NUC1",
            "entry_id": "1DNA",
            "chain": "C",
            "title": "a dna complex",
            "predicted": False,
            "source": "rcsb",
        },
    )
    out = tmp_path / "appended"
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False)
        if 'type="checkbox" name="pick"' in page:
            shown = sum(1 for seen in pages if 'type="checkbox" name="pick"' in seen)
            if shown == 1:
                return {"model": ["grok|grok-4.7"], "next": ["rank"]}
            if shown == 2:
                assert "LLM Grok 4.7 Ranking 20" in page
                return {"model": ["grok|grok-4.7"], "next": ["missing"]}
            assert "1. 1EGF" in page
            assert "3. 1DNA" in page
            assert "4. 3NEW" in page
            assert "3. 3NEW" not in page
            assert "#4 3NEW" in page
            assert "LLM Grok 4.7 Ranking 20" in page
            return {"pick": ["1"]}
        return {}

    def complete(_provider: str, _system: str, user: str) -> str:
        if "Structures to rate:" in user:
            return json.dumps({"ranked": [{"pdb_id": "1EGF", "chain": "A", "rating": 20, "reason": "Listed."}]})
        return json.dumps({"structures": [{"pdb_id": "3NEW", "chain": "A", "reason": "A new deposit."}]})

    code = _run_panel(
        monkeypatch,
        exchange,
        get=lambda _url: {"struct": {"title": "a new deposit"}},
        read_json=lambda _url: {},
        fetch=lambda _option: b"data_1EGFR\n#\n",
        complete=complete,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert any("4. 3NEW" in page for page in pages)
    assert "LLM Grok 4.7 Ranking 20" in captured.out


def test_the_page_can_ask_for_the_modified_model_and_it_is_launched(tmp_path, monkeypatch, capsys):
    """The requested option, from the page it is on to the command it starts.

    A run that names a checkpoint can pick either model on the page. Picking the
    modified one loads that checkpoint instead of the release pair; the structure
    and its output directory are the same either way, because one model runs.
    """
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "both-models"
    checkpoint = tmp_path / "disease.ckpt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    pages: list[str] = []
    launched: list[list[str]] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False, modified_ckpt=[str(checkpoint)])
        if 'type="checkbox" name="pick"' in page:
            return {"all": ["yes"]}
        # The checkpoint exists, so both choices are live on this page.
        assert '<p class="step-title">Confirm inference</p>' in page
        assert f"Modified Low Toxic loads {checkpoint}." in page
        assert 'name="generate_with" value="modified" disabled' not in page
        assert page.count('name="generate_with"') == 2, "two models, so two choices"
        return {"accept": ["yes"], "run": ["yes"], "generate_with": ["modified"]}

    code = _run_panel(
        monkeypatch,
        exchange,
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=lambda _url: {},
        runner=lambda argv: launched.append(list(argv)) or 0,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err

    # One structure, one run, on the model the page picked.
    assert len(launched) == 1
    argv = launched[0]
    assert "inference_experiment=disease_binder" in argv
    assert "inference_experiment=inference_on_user_inputs" not in argv
    assert f"ckpt_name={checkpoint.name}" in argv
    assert f"out_dir={out}/protein_1EGFR_A" in argv, "the structure names the directory"

    assert f"Using modified checkpoint {checkpoint}." in captured.out
    assert "Using the original release model." not in captured.out
    assert "Starting 1 local and 0 remote generation request(s)." in captured.out
    assert captured.out.strip().endswith("Generation finished.")


def test_the_inference_page_cannot_go_back_because_the_downloads_are_done(tmp_path, monkeypatch, capsys):
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "sealed"
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False)
        if 'type="checkbox" name="pick"' in page:
            return {"all": ["yes"]}
        return {"accept": ["yes"]}

    code = _run_panel(
        monkeypatch,
        exchange,
        runner=lambda _argv: 0,  # Continue starts generation now.
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=lambda _url: {},
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert ">Back</button>" in pages[1], "the structures page can still go back"
    assert ">Back</button>" not in pages[2], "replaying would download every structure again"
    assert ">Forward</button>" not in pages[2]
    assert captured.out.strip().endswith("Generation finished.")


def test_a_sealed_session_sends_the_browser_to_the_page_it_is_showing():
    """A browser Back after the run starts writing must not rewind it."""
    import urllib.request

    session = BrowserSession(opener=lambda _url: None)
    delivered: list[dict[str, list[str]]] = []

    def serve() -> None:
        delivered.append(session.exchange("<p>one</p>", step=1))

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        for _ in range(100):
            if session._server is not None and session.page() == "<p>one</p>" and session.step() == 1:
                break
            time.sleep(0.02)
        assert session._server is not None
        assert session.step() == 1, "the page has to be showing before Back is checked"
        base = f"http://127.0.0.1:{session._server.server_address[1]}"
        session.seal()
        landed = urllib.request.urlopen(base + "/step/0", timeout=5)
        assert landed.geturl().endswith("/step/1")
        assert landed.read().decode("utf-8") == "<p>one</p>"
        assert delivered == [], "the waiting page was not disturbed"
        assert worker.is_alive()
    finally:
        session._posted.set()
        session._next_ready.set()
        session._result.set()
        worker.join(timeout=2)
        session.close()


def test_the_inference_page_names_the_count_and_both_choices():
    page = render_inference(117)
    assert "Run inference on 117 downloaded structure(s)" in page
    assert "117 structure(s) were downloaded" in page
    assert 'name="accept" value="yes" checked' in page
    # One box, not two. Continue starts generation, so a separate tick for it
    # only ever repeated the button beneath it.
    assert 'name="run"' not in page
    assert "Continue writes the job" in page
    assert "Clear the box to stop here" in page


def test_the_panel_serves_the_form_on_localhost():
    import urllib.parse
    import urllib.request

    seen: list[str] = []
    errors: list[BaseException] = []
    started: dict[str, threading.Thread] = {}

    def opener(url: str) -> None:
        def client() -> None:
            try:
                page = urllib.request.urlopen(url, timeout=5).read().decode("utf-8")
                seen.append(page)
                data = urllib.parse.urlencode({"disease": "Asthma", "binder_kind": "ligand", "where": "local"}).encode()
                urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=5).read()
            except BaseException as exc:  # noqa: BLE001 - the thread must report every failure
                errors.append(exc)

        started["thread"] = threading.Thread(target=client)
        started["thread"].start()

    session = BrowserSession(opener=opener)
    fields = session.exchange(render_options(OptionForm()))
    session.close()
    started["thread"].join(timeout=5)
    assert errors == []
    assert fields["binder_kind"] == ["ligand"]
    assert fields["disease"] == ["Asthma"]
    assert '<select name="binder_kind"' in seen[0]
    assert 'type="checkbox" name="confirm_inference"' in seen[0]
    assert 'textContent = "Processing"' in seen[0]
    assert 'id="run-progress"' in seen[0]


def test_the_pressed_button_is_the_one_that_shows_processing():
    """With Back and Forward on the page, the first submit button is no longer the action."""
    page = render_options(OptionForm())
    assert 'textContent = "Processing"' in page
    assert "event.submitter" in page, "the pressed button is the one marked, not the first"
    # A disabled control is left out of the form data. Disabling the pressed button would
    # drop __nav, so a Back press would arrive as an ordinary submission of that page.
    assert "if (all[i] !== pressed) all[i].disabled = true;" in page
    assert 'button[aria-busy="true"]' in page, "the busy button still looks busy"
    # A second press cannot submit the form twice.
    assert "if (submitting)" in page

    shown = with_nav(page, back=True, forward=True)
    assert shown.index('value="back"') < shown.index(">Continue</button>")
    # The step's own action must be the LAST submit button, because that is what the
    # script falls back to when the browser does not report which button was pressed.
    last = shown.rfind('<button type="submit"')
    assert shown[last:].startswith('<button type="submit">Continue</button>')


def test_a_pressed_button_stays_on_processing_and_a_download_fills_the_bar():
    """The button label changes in the browser, and /progress follows the download."""
    page = render_structures([_option("protein", "1egfr")])
    assert ">Download Structures</button>" in page
    assert ">Processing</button>" not in page
    assert "event.submitter" in page
    assert 'textContent = "Processing"' in page
    assert 'id="run-progress"' in page
    assert "/progress" in page
    for claim in ("probability of cure", "cure the disease", "not a cure", "clinical result"):
        assert claim not in page.casefold()

    import urllib.parse
    import urllib.request

    session = BrowserSession(opener=lambda _url: None)
    found: dict[str, dict[str, list[str]]] = {}

    def serve() -> None:
        found["fields"] = session.exchange(page)

    worker = threading.Thread(target=serve)
    poster: threading.Thread | None = None
    worker.start()
    try:
        for _ in range(100):
            if session._server is not None:
                break
            time.sleep(0.02)
        assert session._server is not None
        base = f"http://127.0.0.1:{session._server.server_address[1]}/"
        served = urllib.request.urlopen(base, timeout=5).read().decode("utf-8")
        assert ">Download Structures</button>" in served
        assert 'textContent = "Processing"' in served
        idle = json.loads(urllib.request.urlopen(base + "progress", timeout=5).read().decode("utf-8"))
        assert idle == {
            "active": False,
            "done": 0,
            "total": 0,
            "label": "",
            "action": "",
            "step": 0,
            "closed": False,
            # Bumped whenever a results-page overlay changes, so an open tab can
            # tell the page it is looking at is stale. Still an exact-equality
            # check: the contract gained a key, it did not lose a guarantee.
            "results": 0,
        }

        def post() -> None:
            data = urllib.parse.urlencode({"pick": "1"}).encode()
            urllib.request.urlopen(urllib.request.Request(base, data=data), timeout=5).read()

        poster = threading.Thread(target=post)
        poster.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert found["fields"]["pick"] == ["1"]
        session.note_download(1, 3, "1EGFR chain A")
        current = json.loads(urllib.request.urlopen(base + "progress", timeout=5).read().decode("utf-8"))
        assert current == {
            "active": True,
            "done": 1,
            "total": 3,
            "label": "1EGFR chain A",
            "action": "Downloading",
            "step": 0,
            "closed": False,
            "results": 0,
        }
        # A record lookup is a read, and the bar must not call it a download.
        session.note_progress(2, 9, "P00533", "Reading database records")
        reading = json.loads(urllib.request.urlopen(base + "progress", timeout=5).read().decode("utf-8"))
        assert reading["action"] == "Reading database records"
        assert "Download" not in reading["action"]
        with session._lock:
            session._next_html = "<p>next</p>"
        session._next_ready.set()
        poster.join(timeout=5)
        assert not poster.is_alive()
    finally:
        session._posted.set()
        session._next_ready.set()
        session._result.set()
        worker.join(timeout=2)
        if poster is not None:
            poster.join(timeout=2)
        session.close()


def test_a_refresh_restores_the_current_page_or_the_main_page():
    """A reload is a GET of the stored page. An empty store shows the landing page."""
    import urllib.parse
    import urllib.request

    page = render_structures([_option("protein", "1egfr")])
    session = BrowserSession(opener=lambda _url: None)
    found: dict[str, object] = {}

    def serve() -> None:
        found["fields"] = session.exchange(page)

    worker = threading.Thread(target=serve)
    poster: threading.Thread | None = None
    worker.start()
    try:
        for _ in range(100):
            if session._server is not None:
                break
            time.sleep(0.02)
        assert session._server is not None
        base = f"http://127.0.0.1:{session._server.server_address[1]}/"
        original = urllib.request.urlopen(base, timeout=5).read().decode("utf-8")
        assert ">Download Structures</button>" in original
        assert urllib.request.urlopen(base, timeout=5).read().decode("utf-8") == original
        assert worker.is_alive()

        with session._lock:
            session._html = ""
        main = urllib.request.urlopen(base, timeout=5).read().decode("utf-8")
        assert 'name="grok_key"' in main
        assert 'name="disease"' in main
        assert ">Download Structures</button>" not in main
        assert worker.is_alive()
        with session._lock:
            session._html = page

        def post() -> None:
            data = urllib.parse.urlencode({"pick": "1"}).encode()
            request = urllib.request.Request(base, data=data)
            found["body"] = urllib.request.urlopen(request, timeout=5).read().decode("utf-8")

        poster = threading.Thread(target=post)
        poster.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        during = urllib.request.urlopen(base, timeout=5).read().decode("utf-8")
        assert ">Download Structures</button>" in during
        nxt = render_inference(1)
        with session._lock:
            session._next_html = nxt
        session._next_ready.set()
        poster.join(timeout=5)
        assert not poster.is_alive()
        shown = _with_step(nxt, 0)
        assert found["body"] == shown
        assert urllib.request.urlopen(base, timeout=5).read().decode("utf-8") == shown
        fields = found["fields"]
        assert isinstance(fields, dict)
        assert fields["pick"] == ["1"]
    finally:
        session._posted.set()
        session._next_ready.set()
        session._result.set()
        worker.join(timeout=2)
        if poster is not None:
            poster.join(timeout=2)
        session.close()


def test_an_open_tab_can_follow_the_panel_to_the_next_page():
    """A browser left on the main page can open the step the panel moved to.

    The address it opens is the current step, so the request does not rewind.
    """
    import urllib.parse
    import urllib.request

    session = BrowserSession(opener=lambda _url: None)
    first = render_options(OptionForm())
    second = render_confirm(Disease("Asthma", "MONDO_0004979"))

    def serve() -> None:
        session.exchange(first, step=0)
        session.exchange(second, step=1)

    worker = threading.Thread(target=serve, daemon=True)
    poster: threading.Thread | None = None
    worker.start()
    try:
        for _ in range(100):
            if session._server is not None:
                break
            time.sleep(0.02)
        assert session._server is not None
        base = f"http://127.0.0.1:{session._server.server_address[1]}"
        opened = urllib.request.urlopen(base + "/", timeout=5)
        assert opened.geturl().endswith("/step/0")
        html = opened.read().decode("utf-8")
        assert 'data-step="0"' in html
        assert 'name="disease"' in html
        assert 'class="step-title">Confirm disease' not in html
        assert 'window.location.replace("/step/" + next)' in html
        assert 'window.location.replace("/closed")' in html
        assert 'addEventListener("beforeunload"' not in html, "a submit must not count as having left the page"
        idle = json.loads(urllib.request.urlopen(base + "/progress", timeout=5).read().decode("utf-8"))
        assert idle["step"] == 0
        assert idle["closed"] is False

        def post() -> None:
            data = urllib.parse.urlencode({"disease": "Asthma"}).encode()
            urllib.request.urlopen(urllib.request.Request(base + "/", data=data), timeout=5).read()

        poster = threading.Thread(target=post)
        poster.start()
        moved: dict[str, object] = {}
        for _ in range(100):
            moved = json.loads(urllib.request.urlopen(base + "/progress", timeout=5).read().decode("utf-8"))
            if moved.get("step") == 1:
                break
            time.sleep(0.02)
        assert moved.get("step") == 1
        followed = urllib.request.urlopen(base + "/step/1", timeout=5)
        assert followed.geturl().endswith("/step/1")
        page = followed.read().decode("utf-8")
        assert 'data-step="1"' in page
        assert 'class="step-title">Confirm disease' in page
        assert "Use this disease" in page
        # The current step is served as itself. Asking for it does not go back.
        assert session.step() == 1
        poster.join(timeout=5)
        assert not poster.is_alive()

        with session._lock:
            session._closed = True
        closed_progress = json.loads(urllib.request.urlopen(base + "/progress", timeout=5).read().decode("utf-8"))
        assert closed_progress["closed"] is True
        done = urllib.request.urlopen(base + "/closed", timeout=5).read().decode("utf-8")
        assert "Panel closed" in done
        assert 'data-closed="true"' in done
    finally:
        session._posted.set()
        session._next_ready.set()
        session._result.set()
        worker.join(timeout=2)
        if poster is not None:
            poster.join(timeout=2)
        session.close()


def test_a_repeated_post_keeps_the_first_submission():
    session = BrowserSession(opener=lambda _url: None)
    leader_waiting = threading.Event()
    follower_waiting = threading.Event()
    original_wait = session._next_ready.wait
    original_result_wait = session._result.wait

    def wait(timeout: float | None = None) -> bool:
        leader_waiting.set()
        return original_wait(timeout)

    def result_wait(timeout: float | None = None) -> bool:
        follower_waiting.set()
        return original_result_wait(timeout)

    session._next_ready.wait = wait
    session._result.wait = result_wait
    pages: list[str | None] = []

    def run(fields: dict[str, list[str]]) -> None:
        pages.append(session.submit(fields))

    first = threading.Thread(target=run, args=({"disease": ["Asthma"]},))
    second = threading.Thread(target=run, args=({"disease": ["Other"]},))
    first.start()
    assert leader_waiting.wait(timeout=2)
    second.start()
    assert follower_waiting.wait(timeout=2)
    assert session._fields["disease"] == ["Asthma"]
    with session._lock:
        session._next_html = "<p>kept</p>"
    session._next_ready.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert pages.count("<p>kept</p>") == 2
    assert not first.is_alive()
    assert not second.is_alive()


def test_a_closed_browser_does_not_crash_the_panel(capsys):
    import socket

    started: dict[str, threading.Thread] = {}

    def opener(url: str) -> None:
        def client() -> None:
            host_port = url.split("//", 1)[1].strip("/")
            host, port_text = host_port.rsplit(":", 1)
            connection = socket.create_connection((host, int(port_text)), timeout=5)
            body = b"disease=Asthma"
            request = (
                b"POST / HTTP/1.1\r\nHost: "
                + host_port.encode()
                + b"\r\nContent-Type: application/x-www-form-urlencoded\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            connection.sendall(request)
            connection.close()

        started["thread"] = threading.Thread(target=client)
        started["thread"].start()

    session = BrowserSession(opener=opener)
    fields = session.exchange("<form>open</form>")
    session.close()
    started["thread"].join(timeout=5)
    captured = capsys.readouterr()
    assert fields["disease"] == ["Asthma"]
    assert "Traceback" not in captured.err
    assert "Broken pipe" not in captured.err


def test_the_confirm_page_uses_the_tick_or_the_matching_names_menu():
    """The single-hit page keeps its tick and shows the no-tick menu while it is clear."""
    named = Disease("Asthma", "EFO_0000270")
    other = Disease("Allergic asthma", "EFO_0002682")
    duplicate = Disease("Asthma again", "EFO_0000270")
    seen: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        seen.append(page)
        return {"choice": ["2"]}

    chosen = OptionPanel(exchange).confirm_disease(named, [duplicate, other])
    assert chosen == other
    page = seen[0]
    assert "<h2>Confirm the disease</h2>" in page
    assert "<h2>Choose the disease</h2>" not in page
    assert 'type="checkbox" name="accept" value="yes"' in page
    assert "checked" not in page.split('name="accept"', 1)[1].split(">", 1)[0]
    assert '<label class="field disease-choice">Matching names' in page
    assert "This list is used to help confirm Disease" in page
    assert "Allergic asthma (EFO_0002682)" in page
    assert "Asthma again" not in page
    assert page.count("EFO_0000270") == 2
    menu = page.split('<select name="choice">', 1)[1].split("</select>", 1)[0]
    assert 'value="1" selected' in menu
    assert "disabled" not in menu
    assert 'querySelector(".disease-choice")' in page
    assert "diseaseChoice.hidden = useNamed" in page
    assert OptionPanel(lambda _page: {"accept": ["yes"], "choice": ["2"]}).confirm_disease(named, [other]) == named
    assert OptionPanel(lambda _page: {}).confirm_disease(named, [other]) is None
    assert OptionPanel(lambda _page: {"choice": ["9"]}).confirm_disease(named, [other]) is None
    assert OptionPanel(lambda _page: {"choice": ["1"]}).confirm_disease(named, [other]) == named


def test_a_clear_confirm_tick_uses_the_matching_names_menu(tmp_path, monkeypatch, capsys):
    """One exact hit stays on the tick page, and a clear tick follows the menu."""
    catalog = tmp_path / "two.json"
    catalog.write_text(
        json.dumps(
            [
                {"name": "Asthma", "disease_id": "EFO_0000270"},
                {"name": "Allergic asthma", "disease_id": "EFO_0002682"},
            ]
        ),
        encoding="utf-8",
    )
    records = structures_json(tmp_path, ROW_P00533)
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if _is_generation_page(page):
            return {"accept": ["yes"]}
        if 'name="binder_kind"' in page:
            return {
                "disease": ["Asthma"],
                "catalog": [str(catalog)],
                "structures_json": [str(records)],
                "binder_kind": ["protein"],
                "out_dir": [str(tmp_path / "runs")],
                "where": ["local"],
                "take_all": ["yes"],
                "confirm_inference": ["yes"],
            }
        if 'name="pathway"' in page and 'name="mechanism"' in page:
            return {"pathway": ["any"], "mechanism": ["any"]}
        assert "<h2>Confirm the disease</h2>" in page
        assert "<h2>Choose the disease</h2>" not in page
        assert "Matching names" in page
        assert "Allergic asthma (EFO_0002682)" in page
        return {"choice": ["2"]}

    code = _run_panel(
        monkeypatch,
        exchange,
        runner=lambda _argv: 0,  # Continue starts generation now.
        post=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network")),
        get=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network")),
        fetch=lambda _option: b"data_source\n",
    )
    stdout = capsys.readouterr().out
    assert code == 0
    assert len(pages) == 4, "landing, the confirm page, the biological target page, then generation"
    assert "Confirmed: Allergic asthma (EFO_0002682)" in stdout
    assert stdout.strip().endswith("Generation finished.")


def test_the_panel_picks_a_similar_name_from_the_dropdown(tmp_path, monkeypatch, capsys):
    catalog = tmp_path / "two.json"
    catalog.write_text(
        '[{"name": "Breast cancer", "disease_id": "A"}, {"name": "Breast carcinoma", "disease_id": "B"}]',
        encoding="utf-8",
    )
    records = structures_json(tmp_path, ROW_P00533)
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if _is_generation_page(page):
            return {"accept": ["yes"]}
        if 'name="binder_kind"' in page:
            return {
                "disease": ["breast"],
                "catalog": [str(catalog)],
                "structures_json": [str(records)],
                "binder_kind": ["protein"],
                "out_dir": [str(tmp_path / "runs")],
                "where": ["local"],
                "take_all": ["yes"],
                "confirm_inference": ["yes"],
            }
        if 'name="research"' in page:
            return {"research": ["none"]}
        if 'name="pathway"' in page and 'name="mechanism"' in page:
            return {"pathway": ["any"], "mechanism": ["any"]}
        assert '<select name="choice"' in page
        return {"choice": ["2"]}

    code = _run_panel(
        monkeypatch,
        exchange,
        runner=lambda _argv: 0,  # Continue starts generation now.
        post=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network")),
        get=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network")),
        fetch=lambda _option: b"data_source\n",
    )
    stdout = capsys.readouterr().out
    assert code == 0
    assert len(pages) == 4, "landing, the name list, the biological target page, then generation"
    assert "Breast carcinoma (B)" in pages[1]
    assert "Confirmed: Breast carcinoma (B)" in stdout
    assert stdout.strip().endswith("Generation finished.")


def test_a_landing_page_key_stays_off_the_command_and_off_the_next_page():
    secret = "panel-secret-key"
    calls = {"n": 0}

    def exchange(page: str) -> dict[str, list[str]]:
        calls["n"] += 1
        assert secret not in page
        assert 'type="password" name="grok_key"' in page
        if calls["n"] == 1:
            return {"grok_key": [secret], "binder_kind": ["protein"], "where": ["local"]}
        assert "already saved encrypted" in page
        assert "Leave blank to keep it." not in page
        return {"disease": ["Asthma"], "binder_kind": ["protein"], "where": ["local"]}

    panel = OptionPanel(exchange)
    argv = panel.collect(["--ui"], build_parser())
    assert calls["n"] == 2
    assert argv[:3] == ["--ui", "--disease", "Asthma"]
    assert secret not in " ".join(argv)
    assert panel.api_keys() == {"grok": secret}
    assert load_api_keys() == {"grok": secret}
    assert secret not in _stored_key_text()
    again = OptionPanel(lambda page: {"disease": ["Asthma"], "binder_kind": ["protein"], "where": ["local"]})
    assert secret not in " ".join(again.collect(["--ui"], build_parser()))
    assert again.api_keys() == {"grok": secret}


def test_each_provider_can_keep_any_number_of_keys_shown_only_as_stars(capsys):
    first = "grok-secret-a"
    second = "grok-secret-b"
    third = "grok-secret-c"
    preview = render_options(
        OptionForm(),
        saved=("grok",),
        slots={
            "grok": (
                KeySlot(first, "grok-4.7"),
                KeySlot(second, "grok-4.6"),
            )
        },
    )
    assert first not in preview
    assert second not in preview
    assert preview.count('name="grok_key"') == 2
    assert preview.count('value="*****"') == 2
    assert preview.count('class="llm"') == 6
    assert 'value="grok-4.7" selected' in preview
    assert 'value="grok-4.6" selected' in preview
    assert "already saved encrypted" in preview
    calls = {"n": 0}

    def exchange(page: str) -> dict[str, list[str]]:
        calls["n"] += 1
        assert first not in page
        assert second not in page
        assert third not in page
        if calls["n"] == 1:
            return {
                "grok_listed": ["yes"],
                "grok_slot": ["new", "new", "new"],
                "grok_key": [first, second, third],
                "grok_model": ["grok-4.7", "grok-4.6", "grok-4.5"],
                "binder_kind": ["protein"],
                "where": ["local"],
            }
        assert page.count('value="*****"') == 3
        assert page.count('name="grok_key"') == 3
        assert "already saved encrypted" in page
        assert load_key_slots()["grok"] == (
            KeySlot(first, "grok-4.7"),
            KeySlot(second, "grok-4.6"),
            KeySlot(third, "grok-4.5"),
        )
        return {
            "grok_listed": ["yes"],
            "grok_slot": ["0"],
            "grok_key": ["*****"],
            "grok_model": ["grok-4.5"],
            "disease": ["Asthma"],
            "binder_kind": ["protein"],
            "where": ["local"],
        }

    panel = OptionPanel(exchange)
    argv = panel.collect(["--ui"], build_parser())
    assert calls["n"] == 2
    assert argv[:3] == ["--ui", "--disease", "Asthma"]
    assert first not in " ".join(argv)
    assert panel.api_keys() == {"grok": first}
    assert load_api_keys() == {"grok": first}
    assert load_key_slots()["grok"] == (KeySlot(first, "grok-4.5"),)
    cipher = (key_directory() / "api_keys.json").read_text(encoding="utf-8")
    assert first not in cipher
    assert "grok-4.5" not in cipher
    stored = _stored_key_text()
    assert first not in stored
    assert second not in stored
    assert third not in stored

    def wipe(page: str) -> dict[str, list[str]]:
        assert 'value="*****"' in page
        assert first not in page
        return {"grok_listed": ["yes"], "disease": ["Asthma"], "binder_kind": ["protein"], "where": ["local"]}

    gone = OptionPanel(wipe)
    assert first not in " ".join(gone.collect(["--ui"], build_parser()))
    assert gone.api_keys() == {}
    assert load_key_slots() == {}
    assert first not in _stored_key_text()
    captured = capsys.readouterr()
    assert first not in captured.out
    assert second not in captured.out
    assert third not in captured.out
    assert "Saved encrypted API keys for the next run." in captured.out


def test_a_mask_keeps_the_saved_secret_and_a_typed_key_is_tied_to_its_model():
    existing = [KeySlot("grok-secret-a", "grok-4.7")]
    kept = _slots_from_fields(
        "grok",
        {"grok_slot": ["0"], "grok_key": ["*****"], "grok_model": ["grok-4.6"]},
        existing,
    )
    assert kept == [KeySlot("grok-secret-a", "grok-4.6")]
    added = _slots_from_fields(
        "grok",
        {
            "grok_slot": ["0", "new", "new"],
            "grok_key": ["", "grok-secret-b", "*****"],
            "grok_model": ["grok-4.7", "grok-4.6", "grok-4.5"],
        },
        existing,
    )
    assert added == [KeySlot("grok-secret-a", "grok-4.7"), KeySlot("grok-secret-b", "grok-4.6")]
    assert _slots_from_fields("grok", {"grok_key": [""]}, existing) == existing
    replaced = _slots_from_fields(
        "grok",
        {"grok_key": ["grok-secret-b"], "grok_model": ["grok-4.6"]},
        existing,
    )
    assert replaced == [KeySlot("grok-secret-b", "grok-4.6")]
    with pytest.raises(ValueError, match="Unknown Grok model"):
        _slots_from_fields(
            "grok",
            {"grok_slot": ["new"], "grok_key": ["grok-secret-b"], "grok_model": ["not-a-model"]},
            [],
        )


def test_the_key_tied_to_the_requested_model_is_the_one_sent(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    slots = {
        "grok": (
            KeySlot("grok-secret-a", "grok-4.7"),
            KeySlot("grok-secret-b", "grok-4.6"),
        )
    }
    assert key_for_model("grok", "grok-4.6", key_slots=slots) == "grok-secret-b"
    assert key_for_model("grok", "grok-4.7", key_slots=slots) == "grok-secret-a"
    assert key_for_model("grok", "grok-4.5", key_slots=slots) == "grok-secret-a"
    save_api_keys({"grok": "panel-secret-key"})
    assert load_key_slots()["grok"] == (KeySlot("panel-secret-key", ""),)
    assert load_api_keys() == {"grok": "panel-secret-key"}
    assert key_for_model("grok", "grok-4.6", key_slots=load_key_slots()) == "panel-secret-key"
    seen: dict[str, object] = {}

    def fake(request, timeout=0):
        seen["authorization"] = request.get_header("Authorization")
        seen["body"] = request.data.decode()
        return _JsonResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    assert complete_chat("grok", "system", "user", chat_models={"grok": "grok-4.6"}, key_slots=slots) == "ok"
    assert seen["authorization"] == "Bearer grok-secret-b"
    assert json.loads(seen["body"])["model"] == "grok-4.6"
    assert "grok-secret-a" not in seen["body"]
    assert "grok-secret-b" not in seen["body"]


def test_a_landing_page_key_is_saved_encrypted(tmp_path, monkeypatch, capsys):
    block_live_calls(monkeypatch)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    secret = "panel-secret-key"
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "keyed"
    seen: dict[str, object] = {}
    chats: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        assert secret not in page
        if 'name="binder_kind"' in page:
            assert 'type="password" name="claude_key"' in page
            return _panel_landing(catalog, records, out, take_all=False, confirm_inference=["yes"], grok_key=[secret])
        if 'type="checkbox" name="pick"' in page:
            assert 'name="all_models"' in page
            assert 'name="research"' not in page, "provider boxes are gone"
            shown = page.count('type="checkbox" name="pick"')
            # The first structures page runs the search. The page that comes back
            # is Download, and that press must not send the key to a model again.
            if chats:
                assert shown >= 1
                return {"all": ["yes"]}
            return {"all": ["yes"], "model": ["grok|grok-4.7"], "next": ["missing"]}
        if _is_generation_page(page):
            return {"accept": ["yes"]}
        raise AssertionError(f"no page follows the structures page: {page[:200]}")

    def read(url: str):
        if "esearch" in url:
            return {"esearchresult": {"idlist": []}}
        if "string-db.org" in url:
            return []
        return {}

    def fake(request, timeout=0):
        chats.append(request.full_url)
        seen["headers"] = {key.casefold(): value for key, value in request.header_items()}
        seen["body"] = request.data.decode()
        payload = {"choices": [{"message": {"content": "Try a receptor assay."}}]}
        return _JsonResponse(payload)

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    code = cure_main(
        ["--ui"],
        runner=lambda _argv: 0,  # Continue starts generation now.
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=read,
        exchange=exchange,
        read_line=refuse_prompt,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    headers = seen["headers"]
    assert headers["authorization"] == f"Bearer {secret}"
    assert secret not in seen["body"]
    assert len(chats) == 1, "the search button sends the key once; Download does not"
    assert secret not in captured.out
    assert "Saved encrypted API keys for the next run." in captured.out
    assert "Target research" not in captured.out
    saved = "\n".join(path.read_text(encoding="utf-8") for path in out.rglob("*") if path.is_file())
    assert secret not in saved
    assert load_api_keys() == {"grok": secret}
    assert secret not in _stored_key_text()
    for name in ("api_keys.json", "master.key"):
        mode = (key_directory() / name).stat().st_mode & 0o777
        assert mode == 0o600


def test_a_saved_key_is_used_on_the_next_run(tmp_path, monkeypatch, capsys):
    block_live_calls(monkeypatch)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    secret = "panel-secret-key"
    save_api_keys({"grok": secret})
    assert secret not in _stored_key_text()
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    seen: dict[str, object] = {}

    def read(url: str):
        if "esearch" in url:
            return {"esearchresult": {"idlist": []}}
        if "string-db.org" in url:
            return []
        return {}

    def fake(request, timeout=0):
        seen["authorization"] = request.get_header("Authorization")
        payload = {"choices": [{"message": {"content": "Try a receptor assay."}}]}
        return _JsonResponse(payload)

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    out = tmp_path / "saved-run"
    code = cure_main(
        [
            "--catalog",
            str(catalog),
            "--disease",
            "Asthma",
            "--accept-only",
            "--structures-json",
            str(records),
            "--all",
            "--confirm-inference",
            "--research",
            "grok",
            "--out-dir",
            str(out),
        ],
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=read,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert seen["authorization"] == f"Bearer {secret}"
    assert secret not in captured.out
    report = (out / "target_research.json").read_text(encoding="utf-8")
    assert secret not in report


def test_a_tampered_key_store_does_not_reveal_the_key():
    secret = "panel-secret-key"
    save_api_keys({"openai": secret})
    path = key_directory() / "api_keys.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    token = payload["token"]
    payload["token"] = token[:-1] + ("A" if not token.endswith("A") else "B")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="could not be read") as caught:
        load_api_keys()
    assert secret not in str(caught.value)

    def exchange(page: str) -> dict[str, list[str]]:
        assert secret not in page
        assert "could not be read" in page
        return {"disease": ["Asthma"], "binder_kind": ["protein"], "where": ["local"]}

    panel = OptionPanel(exchange)
    argv = panel.collect(["--ui"], build_parser())
    assert secret not in " ".join(argv)
    assert panel.api_keys() == {}


def _stored_key_text() -> str:
    directory = key_directory()
    return "\n".join(path.read_bytes().decode("latin-1") for path in directory.iterdir() if path.is_file())


def test_research_score_is_the_record_count_and_names_real_structures_for_generation(tmp_path, monkeypatch, capsys):
    block_live_calls(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "sk-should-not-appear")
    catalog = asthma_catalog(tmp_path)
    records = structures_json(
        tmp_path,
        ROW_P00533,
        {
            "group": "protein",
            "accession": "P00533",
            "entry_id": "9ZZZ",
            "chain": "B",
            "title": "predicted egfr",
            "predicted": True,
            "source": "fixture",
        },
    )
    fetched: list[str] = []

    def read(url: str):
        if "uniprot.org" in url:
            return {
                "results": [
                    {
                        "primaryAccession": "P00533",
                        "proteinDescription": {"recommendedName": {"fullName": {"value": "EGFR"}}},
                    }
                ]
            }
        if "esearch" in url:
            return {"esearchresult": {"idlist": []}}
        if "string-db.org" in url:
            return []
        return {}

    def complete(provider: str, system: str, user: str) -> str:
        assert provider == "grok"
        assert "unique new research" in system
        assert "PDB 1EGFR chain A" in user
        assert "FAKE1" not in user
        return json.dumps(
            {
                "ideas": "Test a receptor conformation assay.",
                "structures": [
                    {"pdb_id": "1EGFR", "chain": "A"},
                    {"pdb_id": "FAKE1", "chain": "Z"},
                ],
            }
        )

    out = tmp_path / "shapes"
    code = cure_main(
        [
            "--catalog",
            str(catalog),
            "--disease",
            "Asthma",
            "--accept-only",
            "--structures-json",
            str(records),
            "--confirm-inference",
            "--research",
            "grok",
            "--out-dir",
            str(out),
        ],
        fetch=lambda option: fetched.append(option.entry_id) or b"data_1EGFR\n#\n",
        read_json=read,
        complete=complete,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert "P00533: score 1 of 8." in captured.out
    assert "Research ideas:" in captured.out
    assert "Test a receptor conformation assay." in captured.out
    assert "PDB 1EGFR chain A" in captured.out
    assert "FAKE1" not in captured.out
    assert "sk-should-not-appear" not in captured.out
    assert REPORT_NOTE in captured.out
    for claim in ("probability of cure", "cure the disease", "not a cure", "clinical result"):
        assert claim not in captured.out.casefold()
    assert captured.out.strip().endswith("Generation was not started.")
    assert fetched == ["1EGFR"]
    report = json.loads((out / "target_research.json").read_text(encoding="utf-8"))
    subject = report["subjects"][0]
    assert subject["score"] == 1
    assert subject["sources_with_record"] == 1
    assert subject["narrative"] == "Test a receptor conformation assay."
    assert [item["pdb_id"] for item in subject["generation_inputs"]] == ["1EGFR"]
    assert "FAKE1" not in json.dumps(report)
    # The invented id reaches no part of the report, and a count is what records the drop.
    assert "FAKE1" not in json.dumps(subject["models"]["grok"])
    assert subject["models"]["grok"]["structures_not_listed"] == 1
    assert "1 named structure(s) were not on the list" in captured.out
    assert not (out / "protein_9ZZZ_B.cif").exists()
    assert "Compared with the program search:" in captured.out
    assert "Confirmed program options: (none)." in captured.out
    assert "In both: PDB 1EGFR chain A." in captured.out
    assert "Only in the program search: PDB 9ZZZ chain B." in captured.out
    assert "Only in the LLM options: (none)." in captured.out
    compared = report["option_comparison"]
    assert [item["pdb_id"] for item in compared["in_both"]] == ["1EGFR"]
    assert [item["pdb_id"] for item in compared["only_program"]] == ["9ZZZ"]
    assert compared["confirmed"] == []
    assert compared["only_llm"] == []


def test_every_listed_target_is_scored_not_just_the_first_few():
    """The provider cap must not silently leave later rows with no score at all."""
    rows = [
        StructureOption(
            group="protein",
            accession=f"P{10000 + index}",
            entry_id=f"{index}ABC",
            chain="A",
            title="t",
            predicted=False,
            source="fixture",
        )
        for index in range(12)
    ]
    assert len(rows) > RESEARCH_LIMIT, "the fixture has to exceed the provider cap"
    asked: list[tuple[int, int, str]] = []
    _hits, scores = collect_record_scores(rows, lambda _url: {}, on_progress=lambda *args: asked.append(args))

    assert len(scores) == len(rows), "every protein row is scored"
    assert all(row.accession.casefold() in scores for row in rows)
    assert [step[0] for step in asked] == list(range(1, 13)), "progress counts every target"
    assert {step[1] for step in asked} == {12}

    # A companion DNA row carries the accession of the protein whose deposit it came
    # from, so it is scored by that protein and the row is not left blank.
    protein = rows[0]
    companion = StructureOption(
        group="dna",
        accession=protein.accession,
        entry_id=protein.entry_id,
        chain="B",
        title="duplex",
        predicted=False,
        source="fixture",
    )
    _more, pair = collect_record_scores([protein, companion], lambda _url: {})
    assert companion.accession.casefold() in pair, "the DNA row shares the protein's score"


def test_the_structures_page_scores_each_row_by_its_record_count(tmp_path, monkeypatch, capsys):
    """The choose-structures score is the record count, and the databases answer once."""
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533, {**ROW_P35225, "predicted": True})
    reads: list[str] = []

    def read(url: str):
        reads.append(url)
        if "P00533" not in url:
            return {}
        if "uniprot" in url:
            return {"results": [{"primaryAccession": "P00533"}]}
        if "esearch" in url:
            return {"esearchresult": {"idlist": ["1956"]}}
        if "string-db.org" in url:
            # Four partners is four entries from one database.
            return [{"preferredName_B": f"PARTNER{index}"} for index in range(4)]
        return {}

    out = tmp_path / "page-score"
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False, confirm_inference=["yes"])
        if 'type="checkbox" name="pick"' in page:
            shown = sum(1 for seen in pages if 'type="checkbox" name="pick"' in seen)
            if shown == 1:
                # The first search reads the model ticks and the page then comes back.
                return {"pick": ["1"], "model": ["grok|grok-4.7"], "next": ["missing"]}
            return {"pick": ["1"]}
        return {}

    code = _run_panel(
        monkeypatch,
        exchange,
        fetch=lambda _option: b"data_X\n#\n",
        read_json=read,
        complete=lambda *_args: json.dumps({"rating": 70, "reason": "", "ideas": "An idea."}),
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err

    structures = next(page for page in pages if 'type="checkbox" name="pick"' in page)
    assert "The score is how many of UniProt, NCBI, STRING, IntAct, the Human" in structures
    # UniProt one entry, NCBI one, STRING four: six entries from three databases.
    assert "Instances of Structure in same databases 6" in structures
    assert "Instances of Structure in same databases 6/8" not in structures
    assert "Instances of Structure in different databases 3/8" in structures
    # Nothing answered for the other target.
    assert "Instances of Structure in same databases 0" in structures
    assert "Instances of Structure in different databases 0/8" in structures
    assert "program score" not in structures, "that label was replaced by the two counts"
    # The deposit method is a badge on the row, never a score.
    assert "predicted" in structures
    for stale in ("program score 100", "program score 40", "Program score is 100"):
        assert stale not in structures

    # The model ticks are on this page. A row score is still the record count.
    assert "providers " not in structures
    assert "<h2>LLM search</h2>" in structures
    assert "Grok 4.7</label>" in structures
    assert 'name="all_models" value="yes"' in structures
    assert "Search with every API key" not in structures

    # The page asks each database once, before anyone presses Download.
    assert len([url for url in reads if "uniprot" in url]) == 2, "one UniProt call per target"
    assert len([url for url in reads if "esearch" in url]) == 2


def test_the_deposit_method_does_not_change_the_ranked_score(tmp_path, monkeypatch, capsys):
    """Ranking scores on records alone. Experimental against predicted is not in it."""
    records = structures_json(
        tmp_path,
        ROW_P00533,
        {**ROW_P00533, "entry_id": "3POZ"},
        {
            "group": "gene",
            "accession": "GATA3",
            "entry_id": "GATA3",
            "chain": "A",
            "title": "gata3",
            "predicted": True,
            "source": "fixture",
        },
    )
    out = tmp_path / "record-score"
    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--all", "--confirm-inference", "--research", "grok"],
        out_dir=out,
        fetch=lambda _option: b"data_X\n#\n",
        # Only the protein has a UniProt record, so the two targets differ on the score.
        read_json=lambda url: (
            {"results": [{"primaryAccession": "P00533"}]} if "uniprot" in url and "P00533" in url else {}
        ),
        complete=lambda *_args: json.dumps({"rating": 70, "reason": "", "ideas": "An idea."}),
    )
    code = captured.code
    assert code == 0, captured.err

    # The score is the record count for each protein or gene.
    assert "P00533: score 1 of 8." in captured.out
    assert "GATA3: score 0 of 8." in captured.out
    assert "Structures 2 of 3 found." in captured.out
    assert "Structures 1 of 3 found." in captured.out

    ranked = captured.out.split("Ranked for binder generation", 1)[1].split("P00533: score", 1)[0]
    assert "program score 1 of 8, structures 2 of 3" in ranked
    assert "program score 0 of 8, structures 1 of 3" in ranked
    assert "experimental" not in ranked, "the deposit method is not part of the ranking"
    assert "records value" not in ranked, "there is no weighted value any more"

    report = json.loads((out / "target_research.json").read_text(encoding="utf-8"))
    protein = next(row for row in report["subjects"] if row["identifier"] == "P00533")
    gene = next(row for row in report["subjects"] if row["identifier"] == "GATA3")
    assert protein["program_score"] == 1, "one database returned a record"
    assert gene["program_score"] == 0
    assert protein["program_score"] == protein["sources_with_record"]
    assert protein["record_share"] == record_share(1, 8)
    assert "database_value" not in protein
    # The structure counts are still reported, they just do not score.
    assert (protein["structures"], protein["structures_total"]) == (2, 3)
    assert protein["experimental_structures"] == 2
    assert gene["experimental_structures"] == 0

    # Two experimental deposits did not lift the protein's score above its record count.
    assert protein["combined"] == _blend(protein["record_share"], protein["model_rating"])


def _blend(share: int, rating: float | None) -> int:
    from rfproteina.disease.research import MODEL_WEIGHT

    if rating is None:
        return share
    return round((1.0 - MODEL_WEIGHT) * share + MODEL_WEIGHT * rating)


def test_a_ticked_class_shows_only_its_structures():
    """The gap in the class boxes: they selected matching rows but never filtered.

    The page script cannot be executed here, so this pins the wiring that makes
    ticking a class redraw the view rather than only tick its rows.
    """
    rows = [_option("protein", "1egfr"), _option("protein", "1il13", predicted=True)]
    page = render_structures(rows)

    # Each row declares the classes it belongs to, which is what the filter matches on.
    assert "data-campaigns=" in page
    assert 'input class="campaign-select"' in page or "campaign-select" in page

    # The visible set is the included kinds intersected with the ticked classes.
    assert "function checkedCampaigns()" in page
    assert "var wanted = checkedCampaigns();" in page
    assert "var inClass = wanted.length === 0;" in page
    assert "if (hasCampaign(label, wanted[w])) inClass = true;" in page
    assert "var show = kindShown && inClass && byModel && byRank;" in page

    # Ticking a class redraws, so its rows appear and the rest are hidden and cleared.
    assert "if (!box.checked) applyCampaign(box);" in page
    assert "applyStructureView();" in page
    assert "if (!show) box.checked = false;" in page, "a hidden row is not submitted"

    # All Structure cannot stand while the view is narrower than every experimental
    # row, which is what the server reads it as. A ticked class always contradicted
    # it; a model or ranking filter does too, and did not used to stand it down.
    assert 'var narrowed = checkedCampaigns().length > 0 || wantedModel !== "" || rankMode !== "";' in page
    assert "allBox.disabled = !showExperimental || narrowed;" in page


def test_the_all_structure_box_ticks_and_clears_every_row():
    """The reported bug: the box said All Structure and selected nothing.

    It was only ever read on the server, so ticking it moved no row tick: the page
    showed nothing selected while Download would fetch everything. It now drives
    the rows, and the rows keep it in step, which is how the model picker's own
    every-model box has always worked.
    """
    page = render_structures([_option("protein", "1egfr"), _option("protein", "1il13", predicted=True)])

    # Still the server's "every experimental row" flag, and still named on the page.
    assert 'type="checkbox" name="all" value="yes"' in page
    assert "All Structure" in page

    # Ticking it ticks every row the page is offering; clearing it clears them all.
    assert 'if (box.name === "all") {' in page
    assert "for (var i = 0; i < boxes.length; i++) boxes[i].checked = box.checked;" in page

    # Offering means on show with a tick that can still be sent, so a row the
    # filters are hiding is never selected by it.
    assert "function selectableBoxes()" in page
    assert "if (labels[i].hidden) continue;" in page
    assert "if (pick && !pick.disabled) boxes.push(pick);" in page

    # Unticking one row stands the box down, so what is ticked is what is fetched.
    assert "function everyRowBox()" in page
    assert "function syncAllBox()" in page
    assert "      syncAllBox();" in page, "called after a row changes and after a redraw"
    assert "syncCampaignBoxes();" in page, "the class boxes follow the rows too"


def test_a_class_box_mirroring_its_rows_is_not_a_class_filter():
    """Found in a browser while fixing All Structure, and it broke the fix.

    A class box does two things: it ticks that class's rows, and it narrows the
    view to that class. ``syncCampaignBoxes`` ticks the box to mirror its rows, so
    All Structure ticking every row ticked every class box too, and the next
    redraw read those ticks as a filter and stood All Structure straight back
    down. Only a box clicked in the class list narrows anything.
    """
    page = render_structures([_option("protein", "1egfr"), _option("protein", "1il13")])

    # Clicking a class box in the list is what marks it as a chosen filter.
    assert 'box.setAttribute("data-chosen", box.checked ? "yes" : "no");' in page
    assert 'return box.getAttribute("data-chosen") === "yes";' in page
    assert "if (boxes[i].checked && chosenCampaign(boxes[i])) names.push(boxes[i].value);" in page

    # The mirror itself stays: a class box still follows its own rows on the page.
    assert "function syncCampaignBoxes()" in page
    assert "boxes[i].checked = count > 0 && marked === count;" in page


def test_both_searches_report_below_the_llm_search():
    """One list of targets, and each search's readout under the buttons that ran it.

    A structure a model named joins the program's own target list rather than
    sitting in a list of its own. The ranking review used to open the page, above
    the targets it was reviewing and above the search that produced it.
    """
    rows = [_option("protein", "1EGFR"), _option("protein", "2XYZ")]
    ranking = [
        _ranked(
            2,
            "2XYZ",
            "ACC-2XYZ",
            {"grok grok-4.7": 40},
            40.0,
            title="2XYZ",
            source="llm",
            reasons={"grok grok-4.7": "Worth a binder design."},
        )
    ]
    found = merge_found_by({}, {"found_by": {"grok grok-4.7": [["2XYZ", "A"]]}})
    results = llm_results(rows, found=found, ranking={"ranked": ranking})
    page = render_structures(rows, saved=("grok",), ranking=ranking, results=results)

    # The structure a model named is a row in the program's own list, with a tick.
    assert page.count('name="pick"') == 2
    assert page.index("<h2>LLM search</h2>") < page.index('data-number="2"')

    # Both readouts sit below the search, under the buttons that ran it.
    assert page.index("<h2>LLM search</h2>") < page.index("<h3>LLM search results</h3>")
    assert page.index("First search") < page.index("<h3>LLM search results</h3>")
    assert page.index("<h3>LLM search results</h3>") < page.index("LLM ranking is ready for manual review")
    assert "Worth a binder design." in page

    # A search re-shows this page, so what it returned is above the rows rather
    # than hundreds of rows below the button that ran it.
    rows_at = page.index('<div class="ticks structure-rows">')
    assert page.index("LLM ranking is ready for manual review") < rows_at
    assert page.index("<h3>LLM search results</h3>") < rows_at


def test_hiding_a_row_actually_hides_it():
    """The reported bug: Include/Exclude set `hidden` and nothing disappeared.

    The browser's own stylesheet hides [hidden], but any author rule beats it, and the
    structure rows are display:flex while the LLM rows are display:grid. Without a rule
    of our own the filters looked broken, so this pins the rule and the two displays
    that need it.
    """
    page = render_structures([_option("protein", "1egfr")])
    assert "[hidden] { display: none !important; }" in page
    assert ".ticks label { display: flex" in page, "the row rule this has to beat"
    assert ".llm { display: grid" in page, "the LLM row rule it also has to beat"

    # The filters are on the page, and each row says which kind it is.
    assert 'select name="experimental"' in page
    assert 'select name="predicted"' in page
    assert 'value="exclude"' in page
    assert 'data-kind="experimental"' in page
    # The script hides by attribute, which is why the rule is needed.
    assert "label.hidden = !show" in page


def test_excluding_a_kind_drops_its_ticks_on_submit():
    """Hiding is cosmetic; the submitted selection is what must not include them."""
    rows = [
        _option("protein", "1EXP"),
        _option("protein", "2PRED", predicted=True),
        _option("protein", "3UNK", methodology_known=False),
    ]
    assert apply_structure_view(rows, take_all=True, picks=["1", "2", "3"]) == (True, ["1", "2", "3"])

    # Excluding predicted drops row 2 and leaves the unknown row, which follows
    # the experimental choice.
    assert apply_structure_view(rows, take_all=True, picks=["1", "2", "3"], predicted="exclude") == (
        True,
        ["1", "3"],
    )
    # Excluding experimental drops rows 1 and 3, and clears take_all with them.
    assert apply_structure_view(rows, take_all=True, picks=["1", "2", "3"], experimental="exclude") == (
        False,
        ["2"],
    )


def test_the_first_search_only_adds_structures_the_data_bank_really_has():
    """A model can name anything, so every proposal is looked up before it is kept."""
    rows = [_option("protein", "1AAA")]
    disease = Disease("Asthma", "EFO_0000270")
    asked: list[str] = []

    def complete(_provider, system, user):
        assert "Find deposited structures for that Disease" in system
        assert '"PDB_id": "<4-character PDB id or empty>"' in system
        assert '"uniprot_id": "<UniProt accession, or empty>"' in system
        assert '"alphafold_id": "<AF-<UniProt>-F<fragment> or empty>"' in system
        assert "When the PDB id is missing, give the UniProt accession, or the AlphaFold id." in system
        assert "for input sequence or FASTA to then output 3D structure" in system
        assert "alphafoldserver.com" in system
        assert "af3_id" not in system
        assert "<AF3 id>" not in system
        assert "Three sentence or less" in system
        assert "Do not guess." in system
        assert "cure" not in system.casefold()
        assert "not on that list" in user
        assert "Do not Search any of the 8 databases" in user
        assert "Instead Search Recent, within past 2 years" in user
        assert "in PDB, or listed databases" in user
        assert "When the PDB id is not available, give the UniProt accession, or the AlphaFold id." in user
        assert "for a input sequence or FASTA to then output 3D structure" in user
        assert "alphafoldserver.com" in user
        assert "cure" not in user.casefold()
        # Ids, chains and accessions only: titles would multiply the prompt length on a
        # list of a hundred structures. UniProt and AlphaFold are blank when the row is a PDB
        # whose accession is not a UniProt id.
        assert "- PDB 1AAA UniProt - AlphaFold - chain A ACC-1AAA" in user, "the model is told what was already found"
        assert "1AAA title" not in user
        return json.dumps(
            {
                "structures": [
                    {"PDB_id": "3POZ", "chain": "A", "reason": "another deposit"},
                    {"pdb_id": "ZZZZ", "chain": "A", "reason": "invented"},
                    {"pdb_id": "1AAA", "chain": "A", "reason": "already listed"},
                    {"pdb_id": "nope", "chain": "A", "reason": "not an id"},
                ]
            }
        )

    def verify(url):
        asked.append(url)
        return {"struct": {"title": "EGFR kinase domain"}} if "3POZ" in url else {}

    added, report = missing_structures(disease, rows, providers="grok", verify=verify, complete=complete)
    assert [(row.entry_id, row.chain, row.source) for row in added] == [("3POZ", "A", "llm")]
    assert added[0].title == "EGFR kinase domain", "the title comes from the data bank"
    assert report["added"] == ["3POZ chain A"]
    assert report["not_in_the_data_bank"] == ["NOPE", "ZZZZ"]
    assert all("1AAA" not in url for url in asked), "a row already listed is not looked up again"


def _af_record(
    accession: str, fragment: int = 1, description: str = "EGFR", *, sequence: str | None = "MKTSHOULDNOTBESTORED"
) -> dict:
    """One AlphaFold prediction record as the EBI API lists it; ``sequence=None`` leaves it out."""
    entry = f"AF-{accession}-F{fragment}"
    record = {
        "entryId": entry,
        "uniprotAccession": accession,
        "uniprotDescription": description,
        "cifUrl": f"https://alphafold.ebi.ac.uk/files/{entry}-model_v6.cif",
    }
    if sequence is not None:
        record["sequence"] = sequence
    return record


def test_the_first_search_keeps_a_published_af_id():
    """An AlphaFold Database id is added when that model exists. Its sequence is not stored.

    An older reply that still uses af3_id is accepted. The prompt does not ask for that key.
    """
    rows = [_option("protein", "1AAA")]

    def complete(_provider, system, user):
        assert '"alphafold_id":' in system
        assert "af3_id" not in system
        assert "AlphaFold -" in user
        return json.dumps({"structures": [{"af3_id": "AF-P00533-F1", "chain": "A", "reason": "A published model."}]})

    def verify(url):
        assert url == "https://alphafold.ebi.ac.uk/api/prediction/P00533"
        return [_af_record("P00533")]

    added, report = missing_structures(CATALOG[0], rows, providers="grok", verify=verify, complete=complete)
    assert [(row.entry_id, row.accession, row.predicted, row.source) for row in added] == [
        ("AF-P00533-F1", "P00533", True, "alphafold")
    ]
    assert added[0].title == "EGFR"
    assert "MKTSHOULDNOTBESTORED" not in json.dumps(report)


def test_the_first_search_uses_an_alphafold_id_when_the_pdb_id_is_missing():
    """A PDB id that is not in the bank gives way to AF-<UniProt>-F<fragment>."""
    rows = [_option("protein", "1AAA")]
    asked: list[str] = []

    def complete(_provider, _system, _user):
        return json.dumps(
            {
                "structures": [
                    {
                        "id": "ZZZZ",
                        "uniprot_id": "A0AB40D056",
                        "alphafold_id": "AF-A0AB40D056-F2",
                        "chain": "A",
                        "reason": "The deposit is not in the Protein Data Bank.",
                    }
                ]
            }
        )

    def verify(url):
        asked.append(url)
        if url == "https://alphafold.ebi.ac.uk/api/prediction/A0AB40D056":
            return [_af_record("A0AB40D056", 2, "Long protein fragment 2")]
        return {}

    added, report = missing_structures(CATALOG[0], rows, providers="grok", verify=verify, complete=complete)
    assert [(row.entry_id, row.accession, row.predicted, row.source) for row in added] == [
        ("AF-A0AB40D056-F2", "A0AB40D056", True, "alphafold")
    ]
    assert "ZZZZ" not in report["not_in_the_data_bank"]
    assert "MKTSHOULDNOTBESTORED" not in json.dumps(report)
    assert any(url.endswith("/ZZZZ") for url in asked)
    assert "https://alphafold.ebi.ac.uk/api/prediction/A0AB40D056" in asked


def test_a_uniprot_id_alone_uses_the_first_alphafold_fragment():
    """No PDB id and no fragment means the first AlphaFold segment, F1."""
    rows = [_option("protein", "1AAA")]

    def complete(_provider, _system, _user):
        return json.dumps({"structures": [{"uniprot_id": "A0AB40D056", "chain": "A", "reason": "No deposit."}]})

    def verify(url):
        assert url == "https://alphafold.ebi.ac.uk/api/prediction/A0AB40D056"
        return [_af_record("A0AB40D056", 1, "Long protein", sequence=None)]

    added, _report = missing_structures(CATALOG[0], rows, providers="grok", verify=verify, complete=complete)
    assert [(row.entry_id, row.accession, row.predicted, row.source) for row in added] == [
        ("AF-A0AB40D056-F1", "A0AB40D056", True, "alphafold")
    ]


def test_a_pdb_deposit_is_kept_when_the_reply_also_names_an_alphafold_model():
    """The Protein Data Bank deposit wins when that id is present and the entry exists."""
    rows = [_option("protein", "1AAA")]
    asked: list[str] = []

    def complete(_provider, _system, _user):
        return json.dumps(
            {
                "structures": [
                    {
                        "id": "3POZ",
                        "uniprot_id": "P00533",
                        "alphafold_id": "AF-P00533-F1",
                        "chain": "A",
                        "reason": "The deposit is available.",
                    }
                ]
            }
        )

    def verify(url):
        asked.append(url)
        assert "alphafold.ebi.ac.uk" not in url
        return {"struct": {"title": "EGFR kinase domain"}}

    added, _report = missing_structures(CATALOG[0], rows, providers="grok", verify=verify, complete=complete)
    assert [(row.entry_id, row.predicted, row.source) for row in added] == [("3POZ", False, "llm")]
    assert asked == ["https://data.rcsb.org/rest/v1/core/entry/3POZ"]


def test_a_uniprot_id_uses_the_only_published_fragment_when_f1_is_absent():
    """A UniProt accession with no F1 still keeps the one fragment AlphaFold published."""
    rows = [_option("protein", "1AAA")]

    def complete(_provider, _system, _user):
        return json.dumps(
            {
                "structures": [
                    {
                        "PDB_id": "<4-character PDB id or empty>",
                        "uniprot_id": "A0AB40D056",
                        "chain": "<chain>",
                        "reason": "No deposit.",
                    }
                ]
            }
        )

    def verify(url):
        assert url == "https://alphafold.ebi.ac.uk/api/prediction/A0AB40D056"
        return [_af_record("A0AB40D056", 2, "Long protein fragment 2")]

    added, report = missing_structures(CATALOG[0], rows, providers="grok", verify=verify, complete=complete)
    assert [(row.entry_id, row.accession, row.chain, row.predicted, row.source) for row in added] == [
        ("AF-A0AB40D056-F2", "A0AB40D056", "A", True, "alphafold")
    ]
    assert "MKTSHOULDNOTBESTORED" not in json.dumps(report)
    assert report["not_in_the_data_bank"] == []


def test_the_second_search_ranks_a_listed_deposit_from_its_uniprot_id():
    """The list shows the UniProt accession, so a reply may name the row that way."""
    rows = [
        StructureOption(
            group="protein",
            accession="P00533",
            entry_id="1EGFR",
            chain="A",
            title="EGFR",
            predicted=False,
            source="rcsb",
            methodology_known=True,
        ),
        StructureOption(
            group="protein",
            accession="P00533",
            entry_id="2EGFR",
            chain="B",
            title="EGFR other",
            predicted=False,
            source="rcsb",
            methodology_known=True,
        ),
    ]

    def complete(_provider, _system, _user):
        return json.dumps(
            {
                "ranked": [
                    {
                        "pdb_id": "",
                        "uniprot_id": "P00533",
                        "alphafold_id": "",
                        "chain": "B",
                        "rating": 70,
                        "reason": "The second deposit.",
                    }
                ]
            }
        )

    report = rank_structures(rows, providers="grok", complete=complete)
    rated = [item for item in report["ranked"] if item["ratings"]]
    assert [(item["entry_id"], item["chain"], item["ratings"]["grok"]) for item in rated] == [("2EGFR", "B", 70)]


def test_a_shared_uniprot_without_a_chain_does_not_rate_every_structure():
    rows = [
        StructureOption(
            group="protein",
            accession="P00533",
            entry_id="1EGFR",
            chain="A",
            title="EGFR",
            predicted=False,
            source="rcsb",
            methodology_known=True,
        ),
        StructureOption(
            group="protein",
            accession="P00533",
            entry_id="2EGFR",
            chain="B",
            title="EGFR other",
            predicted=False,
            source="rcsb",
            methodology_known=True,
        ),
    ]

    def complete(_provider, _system, _user):
        return json.dumps({"ranked": [{"uniprot_id": "P00533", "rating": 10, "reason": "Both deposits."}]})

    report = rank_structures(rows, providers="grok", complete=complete)
    assert [item["ratings"] for item in report["ranked"]] == [{}, {}]


def test_the_second_search_ranks_the_listed_fragment_from_its_uniprot_id():
    rows = [_option("protein", "AF-A0AB40D056-F2", predicted=True)]

    def complete(_provider, _system, _user):
        return json.dumps(
            {"ranked": [{"uniprot_id": "A0AB40D056", "chain": "A", "rating": 15, "reason": "Fragment 2."}]}
        )

    report = rank_structures(rows, providers="grok", complete=complete)
    assert report["ranked"][0]["entry_id"] == "AF-A0AB40D056-F2"
    assert report["ranked"][0]["ratings"]["grok"] == 15


def test_the_second_search_ranks_a_listed_model_from_its_uniprot_id():
    rows = [_option("protein", "AF-P00533-F1", predicted=True)]

    def complete(_provider, _system, _user):
        return json.dumps(
            {"ranked": [{"uniprot_id": "P00533", "chain": "A", "rating": 25, "reason": "First segment."}]}
        )

    report = rank_structures(rows, providers="grok", complete=complete)
    assert report["ranked"][0]["ratings"]["grok"] == 25


def test_the_second_search_ranks_an_alphafold_id_on_the_list():
    rows = [_option("protein", "AF-P00533-F1", predicted=True)]

    def complete(_provider, system, user):
        assert "- PDB - UniProt P00533 AlphaFold AF-P00533-F1 chain A" in user
        assert "af3_id" not in system
        assert "When the PDB id is missing, give the uniprot_id or the AlphaFold id." in system
        return json.dumps({"ranked": [{"af3_id": "AF-P00533-F1", "chain": "A", "rating": 40, "reason": "Published."}]})

    report = rank_structures(rows, providers="grok", complete=complete)
    assert report["ranked"][0]["entry_id"] == "AF-P00533-F1"
    assert report["ranked"][0]["ratings"]["grok"] == 40


def test_the_second_search_ranks_every_listed_structure():
    rows = [
        _option("protein", "1AAA"),
        _option("protein", "2BBB"),
        _option("protein", "3CCC"),
    ]

    def complete(provider, system, user):
        assert "Three or less sentence" in system
        assert '"pdb_id": "<4-character PDB id or empty>"' in system
        assert '"uniprot_id": "<UniProt accession, or empty>"' in system
        assert '"alphafold_id": "<AF-<UniProt>-F<fragment> or empty>"' in system
        assert "When the PDB id is missing, give the uniprot_id or the AlphaFold id." in system
        assert "af3_id" not in system
        assert "<AF3 id>" not in system
        assert "Rate only structures from the list." in system
        assert "cure" not in system.casefold()
        assert "Allosteric Binders" in user
        assert "Molecular Glues & PROTACs" in user
        assert "Immunogenicity" in user
        assert "\\(K_{d}\\)" in user
        assert "IC₅₀" in user
        assert "Reply with the JSON object only." in user
        assert "cure" not in user.casefold()
        assert "- PDB 1AAA UniProt - AlphaFold - chain A" in user
        marks = {"grok": {"1AAA": 10, "2BBB": 90}, "claude": {"1AAA": 30, "2BBB": 80}}[provider]
        return json.dumps(
            {"ranked": [{"pdb_id": name, "chain": "1", "rating": value} for name, value in marks.items()]}
        )

    report = rank_structures(rows, providers="grok,claude", complete=complete)
    order = [item["entry_id"] for item in report["ranked"]]
    assert order == ["2BBB", "1AAA", "3CCC"], "highest value first, unrated last"
    top = report["ranked"][0]
    assert top["ratings"] == {"grok": 90, "claude": 80}
    assert top["value"] == 85.0
    unrated = report["ranked"][-1]
    assert unrated["entry_id"] == "3CCC"
    assert unrated["value"] is None, "a structure no model rated earns no value"
    assert unrated["ratings"] == {}


def _ranked(number, entry_id, accession, ratings, value, *, chain="A", title="t", source="rcsb", reasons=None):
    ranked = {
        "number": number,
        "entry_id": entry_id,
        "chain": chain,
        "accession": accession,
        "title": title,
        "source": source,
        "ratings": dict(ratings),
    }
    if reasons is not None:
        ranked["reasons"] = dict(reasons)
    ranked["value"] = value
    return ranked


def test_the_first_search_keeps_which_model_named_each_structure():
    """Attribution was captured per model and then thrown away when the row was built.

    The report held every model's raw proposals, but the kept row carried only
    source="llm", so the page could not say which model found what.
    """
    rows = [_option("protein", "1AAA")]
    calls: list[str] = []

    def complete(provider, system, user):
        calls.append(provider)
        named = {"grok": "2BBB", "claude": "3CCC"}[provider]
        # Each model also names the structure the program search already listed.
        return json.dumps({"structures": [{"pdb_id": named, "chain": "A"}, {"pdb_id": "1AAA", "chain": "A"}]})

    added, report = missing_structures(
        Disease("Asthma", "EFO_0000270"),
        rows,
        providers="grok,claude",
        verify=lambda url: {"struct": {"title": "a real entry"}},
        complete=complete,
    )
    assert sorted(calls) == ["claude", "grok"]
    assert sorted(row.entry_id for row in added) == ["2BBB", "3CCC"]
    found = report["found_by"]
    # The structure each model added, and the already-listed one it also named:
    # naming a listed structure adds nothing, but the model did name it.
    assert found["grok"] == [["1AAA", "A"], ["2BBB", "A"]]
    assert found["claude"] == [["1AAA", "A"], ["3CCC", "A"]]
    assert "1AAA chain A" not in report["added"], "an already-listed structure is not an addition"
    # The terminal report names it too; the per-model proposals were shown nowhere.
    printed = format_missing_report(report)
    assert "Grok named: 1AAA chain A, 2BBB chain A." in printed


def test_a_later_search_does_not_erase_an_earlier_search_attribution():
    first = {"found_by": {"grok": [["2BBB", "A"]]}}
    second = {"found_by": {"grok": [["4DDD", "A"]], "claude": [["2BBB", "A"]]}}
    merged = merge_found_by(merge_found_by({}, first), second)
    assert merged["grok"] == [["2BBB", "A"], ["4DDD", "A"]]
    assert merged["claude"] == [["2BBB", "A"]]
    # A repeat is recorded once, not twice.
    assert merge_found_by(merged, second)["grok"] == [["2BBB", "A"], ["4DDD", "A"]]


def test_llm_results_joins_both_searches_on_the_listed_number():
    # The page shows the ID# each tick carries, so the result has to be joined back
    # to the row list by entry id and chain and then renumbered.
    rows = [_option("protein", "1AAA"), _option("protein", "2BBB"), _option("protein", "3CCC")]
    found = {"grok": [["3CCC", "A"]]}
    ranking = {"ranked": [_ranked(1, "1AAA", "acc-1aaa", {"grok": 90, "claude": 70}, 80.0)]}
    results = llm_results(rows, found=found, ranking=ranking)
    assert [item["number"] for item in results] == [1, 3], "listed order, and only rows a model touched"
    first = results[0]
    assert first["value"] == 80.0
    assert [(mark["model"], mark["found"], mark["rating"]) for mark in first["models"]] == [
        ("claude", False, 70),
        ("grok", False, 90),
    ]
    third = results[1]
    assert third["value"] is None, "named but never rated earns no value"
    assert [(mark["model"], mark["found"], mark["rating"]) for mark in third["models"]] == [("grok", True, None)]
    assert result_models(results) == ["claude", "grok"]
    # A structure no model named or rated is absent rather than listed empty.
    assert all(item["entry_id"] != "2BBB" for item in results)


def test_a_model_that_both_named_and_rated_a_structure_shows_both():
    rows = [_option("protein", "1AAA")]
    results = llm_results(
        rows,
        found={"grok": [["1AAA", "A"]]},
        ranking={"ranked": [_ranked(1, "1AAA", "acc-1aaa", {"grok": 55}, 55.0)]},
    )
    mark = results[0]["models"][0]
    assert mark == {"model": "grok", "label": "Grok", "found": True, "rating": 55}
    page = render_structures(rows, results=results, ranking=[_ranked(1, "1AAA", "acc-1aaa", {"grok": 55}, 55.0)])
    assert "LLM Grok Ranking 55; named it" in page


def test_one_chosen_model_is_named_in_the_report_key():
    """A single asked model was keyed by its provider alone, losing which one answered.

    The page then read "Grok" for a run that asked Grok 4.7 specifically, and the
    saved report could not say which model named a structure either.
    """
    rows = [_option("protein", "1AAA")]

    def complete(provider, system, user):
        return json.dumps({"structures": [{"pdb_id": "2BBB", "chain": "A"}]})

    added, report = missing_structures(
        Disease("Asthma", "EFO_0000270"),
        rows,
        providers="grok",
        search_models={"grok": ["grok-4.7"]},
        verify=lambda url: {"struct": {"title": "a real entry"}},
        complete=complete,
    )
    assert [row.entry_id for row in added] == ["2BBB"]
    assert list(report["found_by"]) == ["grok grok-4.7"]
    assert list(report["named"]) == ["grok grok-4.7"]
    assert "Grok 4.7 named: 2BBB chain A." in format_missing_report(report)
    # A provider left to its own default has no chosen model to name, so naming
    # one in the report would be a guess.
    _added, plain = missing_structures(
        Disease("Asthma", "EFO_0000270"),
        rows,
        providers="grok",
        verify=lambda url: {"struct": {"title": "a real entry"}},
        complete=complete,
    )
    assert list(plain["found_by"]) == ["grok"]


def test_a_model_id_is_shown_under_the_name_the_panel_lists_it_under():
    # A key names a provider, and the model when one was chosen. Neither reaches
    # the page as the raw id the key carries.
    assert model_display("grok") == "Grok"
    # The provider is left off when the model's own label already opens with it,
    # so "grok Grok 4.7" does not reach the page.
    assert model_display("grok grok-4.7") == "Grok 4.7"
    assert model_display("claude claude-opus-5-5") == "Claude Opus 5.5"
    # It is kept when the label does not name it: these two both offer Nemotron,
    # and a result has to say which of them answered.
    assert model_display("nvidia nvidia/nemotron-3-ultra-550b-a55b") == "NVIDIA Nemotron 3 Ultra 550B"
    # A routing suffix still resolves to the same label.
    assert model_display("openrouter nvidia/nemotron-3-ultra-550b-a55b:free") == "OpenRouter Nemotron 3 Ultra 550B"
    # An id the panel does not list is shown as given rather than guessed at.
    assert model_display("grok not-a-listed-id") == "grok not-a-listed-id"
    rows = [_option("protein", "1AAA")]
    results = llm_results(rows, found={"nvidia": [["1AAA", "A"]]})
    page = render_structures(rows, results=results)
    assert "LLM NVIDIA Ranking none; named it" in page, "a provider-only key uses the provider's own display name"


def test_the_results_display_lists_every_result_with_its_id_and_model():
    rows = [_option("protein", "1AAA"), _option("protein", "2BBB")]
    results = llm_results(
        rows,
        found={"grok": [["2BBB", "A"]]},
        ranking={"ranked": [_ranked(1, "1AAA", "acc-1aaa", {"claude": 90}, 90.0)]},
    )
    page = render_structures(rows, results=results)
    assert "<h3>LLM search results</h3>" in page
    assert '<ol class="llm-results">' in page
    # The display sits below the LLM search and above the rows it reports on.
    assert page.index("<h2>LLM search</h2>") < page.index("<h3>LLM search results</h3>")
    assert page.index("<h3>LLM search results</h3>") < page.index('name="pick"')
    assert "#1 1AAA chain A ACC-1AAA" in page
    assert "#2 2BBB chain A ACC-2BBB" in page
    assert "LLM Claude Ranking 90" in page
    assert "LLM Grok Ranking none; named it" in page
    assert "no results" not in page
    # Each result carries the models that named it, so the filter can hide it too.
    assert 'data-models="grok"' in page
    assert 'data-models="claude"' in page
    # No list marker: each result already carries the ID# its own tick uses, and a
    # marker beside it is a second number counting something else.
    assert "ol.llm-results { list-style: none;" in page


def test_the_results_display_and_the_model_filter_stay_away_with_no_results():
    """An empty search says so. A filter for an empty list stays off the page."""
    rows = [_option("protein", "1AAA")]
    page = render_structures(rows)
    assert "<h3>LLM search results</h3>" in page
    assert '<p class="notice">no results LLM search not yet requested.</p>' in page
    assert '<ol class="llm-results">' not in page
    assert "LLM ranking is ready for manual review." not in page
    assert '<select name="llm_model">' not in page
    assert '<select name="llm_state">' not in page
    # The sort and the kind filters are there regardless; they need no search.
    assert '<select name="sort">' in page
    assert '<select name="experimental">' in page


def test_every_metric_reaches_the_markup_so_the_sort_can_read_it():
    # The sort read one attribute, data-score, the distinct-database count. The
    # other two metrics were label text and could not be sorted on at all.
    rows = [_option("protein", "1AAA")]
    scores = {"acc-1aaa": RecordScore(instances=12, databases=7, asked=8)}
    ranking = [_ranked(1, "1AAA", "acc-1aaa", {"grok": 90}, 90.0)]
    results = llm_results(rows, found={"grok": [["1AAA", "A"]]}, ranking={"ranked": ranking})
    page = render_structures(rows, scores=scores, ranking=ranking, results=results)
    assert 'data-score="7"' in page, "distinct databases"
    assert 'data-instances="12"' in page, "same databases"
    assert 'data-value="90.0"' in page, "the LLM value"
    assert 'data-models="grok"' in page
    # The script picks the attribute from the chosen mode rather than one hard-wired key.
    assert 'high: "data-score"' in page
    assert '"same-high": "data-instances"' in page
    # Higher and lower value follow the chosen model, not the mean of every model.
    assert '"value-high": "data-value"' not in page
    assert 'mode === "value-high" || mode === "value-low"' in page
    assert "function rankingOf" in page
    assert "var sortKey = SORT_KEYS[mode];" in page
    # A row with no value for the chosen metric sorts last in both directions.
    assert "if (leftScore === null) return 1;" in page


def test_a_row_with_no_metric_leaves_the_attribute_empty():
    # An empty attribute is how the script tells "no value" from zero. A zero would
    # sort as a real score the structure never earned.
    rows = [_option("protein", "1AAA")]
    page = render_structures(rows)
    assert 'data-score=""' in page
    assert 'data-instances=""' in page
    assert 'data-value=""' in page
    assert 'data-models=""' in page


def test_the_model_filter_includes_or_excludes_and_drives_the_results_list_too():
    rows = [_option("protein", "1AAA"), _option("protein", "2BBB")]
    results = llm_results(rows, found={"grok": [["1AAA", "A"]], "claude": [["2BBB", "A"]]})
    page = render_structures(rows, results=results)
    assert '<option value="any" selected>Any model</option>' in page, "filtering nothing is the default"
    assert '<option value="grok">Grok</option>' in page
    assert '<option value="claude">Claude</option>' in page
    assert "Structures that model named" in page
    # The filter composes with the kind and class filters rather than replacing them.
    assert "var show = kindShown && inClass && byModel && byRank;" in page
    assert "byModel = namedBy(label, wantedModel) === keepNamed;" in page
    # "Any model" must not blank the list: there is no subject to include or exclude.
    assert 'var wantedModel = llmModel && llmModel.value !== "any" ? llmModel.value : "";' in page
    # The rows and the results list never disagree about which model is on show.
    assert 'document.querySelectorAll("ol.llm-results > li")' in page
    # A hidden row is not submitted, exactly as the kind filters already ensure.
    assert "if (!show) box.checked = false;" in page


def test_llm_ranking_is_on_every_target_and_filters_without_the_database_counts():
    """The model's number is on every target, and it is its own include/exclude metric.

    The two database counts stay on the row. The ranking does not read them, and
    the sentence the model returned is shown for review and is not a third score.
    """
    rows = [_option("protein", "1AAA"), _option("protein", "2BBB")]
    scores = {
        "acc-1aaa": RecordScore(instances=12, databases=7, asked=8),
        "acc-2bbb": RecordScore(instances=1, databases=1, asked=8),
    }
    prompts: list[str] = []

    def complete(_provider, _system, user):
        prompts.append(user)
        marks = {
            "grok": (90, "A deposited kinase domain."),
            "claude": (70, "Worth a binder design."),
        }[_provider]
        return json.dumps({"ranked": [{"pdb_id": "1AAA", "chain": "A", "rating": marks[0], "reason": marks[1]}]})

    report = rank_structures(rows, providers="grok,claude", complete=complete)
    assert prompts, "both models were asked"
    assert all("Instances of Structure" not in prompt for prompt in prompts)
    assert all("databases" not in prompt.casefold() for prompt in prompts)
    top = report["ranked"][0]
    assert top["entry_id"] == "1AAA"
    assert top["value"] == 80.0, "the mean of the two ratings, with no database count mixed in"
    assert top["ratings"] == {"grok": 90, "claude": 70}
    assert top["reasons"]["grok"] == "A deposited kinase domain."
    assert "instances" not in top
    assert "databases" not in top
    unrated = next(item for item in report["ranked"] if item["entry_id"] == "2BBB")
    assert unrated["value"] is None
    assert unrated["ratings"] == {}

    printed = format_structure_ranking(report)
    assert "LLM ranking is ready for manual review." in printed
    # The terminal report writes the model the way the page writes it.
    assert "LLM Grok Ranking 90 (A deposited kinase domain.)" in printed
    assert "LLM Claude Ranking 70 (Worth a binder design.)" in printed
    assert "no LLM ranking" in printed
    assert "Instances of Structure" not in printed

    fresh = render_structures(rows, scores=scores)
    assert '<select name="llm_rank">' in fresh
    assert '<option value="ranked">Has an LLM ranking</option>' in fresh
    assert '<option value="unranked">No LLM ranking</option>' in fresh
    assert 'value="model:' not in fresh, "a model option appears only after a model has ranked"
    assert "LLM Grok Ranking" not in fresh
    assert "Instances of Structure in same databases 12" in fresh
    assert "Instances of Structure in different databases 7/8" in fresh

    page = render_structures(rows, scores=scores, ranking=report["ranked"])
    rated_row = page.split('name="pick" value="1"', 1)[1].split("</label>", 1)[0]
    assert "LLM Grok Ranking 90" in rated_row
    assert rated_row.index("LLM Grok Ranking 90") < rated_row.index("Instances of Structure in same databases 12")
    assert "Instances of Structure in same databases 12" in page
    assert "Instances of Structure in different databases 7/8" in page
    assert "Instances of Structure in same databases 1" in page
    assert "LLM Claude Ranking 70" in page
    assert "LLM Grok Ranking 90" in page
    assert "LLM Claude Ranking none" in page
    assert "LLM Grok Ranking none" in page
    assert "LLM ranking is ready for manual review." in page
    assert "no results" not in page
    assert "A deposited kinase domain." in page
    assert "These rankings do not change the database instance counts" in page
    assert 'data-value="80.0"' in page
    assert 'data-rankings="claude=70|grok=90"' in page
    assert 'data-rankings="claude=|grok="' in page
    assert 'data-score="7"' in page
    assert 'data-instances="12"' in page
    block_at = page.index('<div class="structure-view">')
    block = page[block_at : page.index("</div>", block_at)]
    assert '<select name="llm_rank">' in block
    assert '<select name="llm_rank_state">' in block
    assert "Structures with that ranking" in block
    assert '<option value="model:claude">LLM Claude Ranking</option>' in block
    assert '<option value="model:grok">LLM Grok Ranking</option>' in block
    assert block_at < page.index('class="ticks structure-rows"')
    script = page[page.index("function hasLlmRanking") : page.index("function applyStructureView")]
    assert "data-value" in script
    assert "data-rankings" in script
    assert "data-score" not in script
    assert "data-instances" not in script

    def chosen(fields: dict[str, list[str]]) -> str | None:
        _action, _take_all, picks, _providers = OptionPanel(lambda _page: fields).choose_structures(
            rows, scores, ranking=report["ranked"]
        )
        return picks

    assert chosen({"pick": ["1", "2"], "llm_rank": ["any"], "llm_rank_state": ["exclude"]}) == "1,2"
    assert chosen({"pick": ["1", "2"], "llm_rank": ["ranked"], "llm_rank_state": ["include"]}) == "1"
    assert chosen({"pick": ["1", "2"], "llm_rank": ["unranked"], "llm_rank_state": ["exclude"]}) == "1"
    assert chosen({"pick": ["1", "2"], "llm_rank": ["model:grok"], "llm_rank_state": ["include"]}) == "1"
    assert chosen({"pick": ["1", "2"], "llm_rank": ["model:claude"], "llm_rank_state": ["exclude"]}) == "2"

    zero = rank_structures(
        [_option("protein", "4DDD")],
        providers="grok",
        complete=lambda *_args: json.dumps(
            {"ranked": [{"pdb_id": "4DDD", "chain": "A", "rating": 0, "reason": "Low interest."}]}
        ),
    )
    assert zero["ranked"][0]["value"] == 0.0, "a rating of zero is a ranking, not a missing one"
    assert zero["ranked"][0]["reasons"] == {"grok": "Low interest."}

    # Every row is kept on the ranking list even when no model rated one. That
    # list is not a result, and the page must not call it ready for review.
    silent = render_structures(
        rows,
        ranking=[
            _ranked(1, "1AAA", "acc-1aaa", {}, None),
            _ranked(2, "2BBB", "acc-2bbb", {}, None),
        ],
    )
    assert '<p class="notice">no results LLM search not yet requested.</p>' in silent
    assert silent.count("no results") == 1
    assert "LLM ranking is ready for manual review." not in silent
    assert '<ul class="llm-review">' not in silent


def test_higher_lower_and_llm_ranking_follow_one_model_or_grok_bot():
    """The value sorts and Has/No read the chosen model. Any still keeps every row."""
    rows = [_option("protein", "1AAA"), _option("protein", "2BBB"), _option("protein", "3CCC")]
    ranking = [
        _ranked(1, "1AAA", "acc-1aaa", {"grok grok-4.7": 10, "grokbot": 90}, 50.0),
        _ranked(2, "2BBB", "acc-2bbb", {"grok grok-4.7": 80}, 80.0),
        _ranked(3, "3CCC", "acc-3ccc", {"grokbot": 0}, 0.0),
    ]
    page = render_structures(rows, saved=("grok",), ranking=ranking)
    assert '<select name="rank_model">' in page
    assert '<option value="grok grok-4.7"' in page
    assert '<option value="grokbot">Grok Bot</option>' in page
    assert "LLM higher value, LLM lower value, and LLM ranking use this model or Grok Bot." in page
    # A zero is a ranking, so the attribute keeps the digit rather than going blank.
    assert 'data-rankings="grok grok-4.7=10|grokbot=90"' in page
    assert "grokbot=0" in page
    fresh = render_structures(rows, saved=("grok",))
    assert '<option value="grokbot">Grok Bot</option>' in fresh
    assert 'value="model:' not in fresh

    def chosen(fields: dict[str, list[str]]) -> str | None:
        _action, _take_all, picks, _providers = OptionPanel(lambda _page: fields).choose_structures(
            rows, ranking=ranking
        )
        return picks

    assert chosen({"pick": ["1", "2", "3"], "llm_rank": ["any"], "rank_model": ["grokbot"]}) == "1,2,3"
    assert (
        chosen(
            {"pick": ["1", "2", "3"], "llm_rank": ["ranked"], "llm_rank_state": ["include"], "rank_model": ["grokbot"]}
        )
        == "1,3"
    )
    assert (
        chosen(
            {
                "pick": ["1", "2", "3"],
                "llm_rank": ["unranked"],
                "llm_rank_state": ["include"],
                "rank_model": ["grokbot"],
            }
        )
        == "2"
    )
    assert (
        chosen(
            {
                "pick": ["1", "2", "3"],
                "llm_rank": ["ranked"],
                "llm_rank_state": ["include"],
                "rank_model": ["grok grok-4.7"],
            }
        )
        == "1,2"
    )


def test_the_choose_structures_page_has_one_control_block_not_two():
    """Both copies of the controls drifted apart; the page now uses the helper.

    The inline copy in render_structures and the helper the Compare page used were
    free to disagree, and the sort options had to be changed in both.
    """
    rows = [_option("protein", "1AAA")]
    page = render_structures(rows)
    assert page.count('<div class="structure-view">') == 1
    assert page.count('<select name="sort">') == 1
    # The script binds to the first match only, so a second block would be inert.
    assert 'document.querySelector(".structure-view")' in page


def test_the_second_search_runs_without_the_first():
    """It ranks the structures the program search found, so it does not wait on the first.

    It was once gated behind the first search. That read the second search as
    ranking only what the first adds, which left the button dead on a page that
    already listed structures to rank.
    """
    rows = [_option("protein", "1AAA")]
    panel = OptionPanel(lambda _page: {"next": ["rank"], "model": ["grok|grok-4.7"]})
    action, _all, _picks, providers = panel.choose_structures(rows)
    assert action == "rank", "the second search is accepted with no first search"
    # A ticked model whose provider has no key is dropped, so no provider comes back here.
    assert providers == []
    page = render_structures(rows)
    assert "Research: All found structures by highest value targets for generation" in page
    marker = page.index("highest value targets for generation")
    opening = page.rindex("<button", 0, marker)
    assert "disabled" not in page[opening:marker], "the second search button is live on a fresh page"


def test_a_search_with_no_model_ticked_says_so_and_keeps_the_page(tmp_path, monkeypatch, capsys):
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "no-model"
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False, confirm_inference=["yes"])
        if _is_generation_page(page):
            return {"accept": ["yes"]}
        shown = sum(1 for seen in pages if 'type="checkbox" name="pick"' in seen)
        if shown == 1:
            return {"next": ["missing"]}  # pressed with nothing ticked
        return {"all": ["yes"]}

    code = _run_panel(
        monkeypatch,
        exchange,
        runner=lambda _argv: 0,  # Continue starts generation now.
        fetch=lambda _option: b"data_X\n#\n",
        read_json=lambda _url: {},
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Tick a model for the search." in pages[2], "the page comes back saying so"
    assert len(pages) == 4, "landing, structures, structures again, generation"


def test_structures_are_listed_against_the_eight_databases():
    """Every structure of one target carries that target's answers."""
    rows = [
        _option("protein", "1EGF"),
        _option("protein", "2IL1", predicted=True),
    ]
    sources = {
        rows[0].accession: {
            "uniprot": SourceHit("record", "x"),
            "ncbi": SourceHit("record", "x"),
            "intact": SourceHit("no record", ""),
            "pride": SourceHit("unavailable", "down"),
        }
    }
    listed = structures_by_database(rows, sources)
    assert [item["entry_id"] for item in listed] == ["1EGF", "2IL1"]

    first = listed[0]
    assert first["asked"] == 8, "all eight are reported, including the ones that said no"
    assert first["found"] == 2, "only a record counts"
    assert first["databases"]["intact"] == "no record"
    assert first["databases"]["pride"] == "unavailable", "down is not the same as no record"
    assert first["databases"]["ttd"] == "not asked", "never asked is not the same either"
    assert first["kind"] == "experimental"

    second = listed[1]
    assert second["found"] == 0
    assert set(second["databases"].values()) == {"not asked"}
    assert second["kind"] == "predicted", "the filter on that page needs the kind"


def test_structures_are_listed_under_each_model_highest_value_first():
    ranking = {
        "ranked": [
            {
                "entry_id": "3POZ",
                "chain": "A",
                "accession": "P1",
                "title": "a",
                "source": "llm",
                "ratings": {"grok": 95, "claude": 70},
            },
            {
                "entry_id": "1EGF",
                "chain": "A",
                "accession": "P1",
                "title": "b",
                "source": "rcsb",
                "ratings": {"grok": 60, "claude": 88},
            },
            {"entry_id": "2IL1", "chain": "B", "accession": "P2", "title": "c", "source": "rcsb", "ratings": {}},
        ]
    }
    by_model = structures_by_model(ranking)
    assert list(by_model) == ["claude", "grok"], "one list per model, in name order"
    assert [row["entry_id"] for row in by_model["grok"]] == ["3POZ", "1EGF"]
    assert [row["entry_id"] for row in by_model["claude"]] == ["1EGF", "3POZ"]
    assert by_model["grok"][0]["rating"] == 95
    # A structure no model rated belongs to no model's list rather than ranking last
    # on a rating it never got.
    assert all("2IL1" not in [row["entry_id"] for row in rows] for rows in by_model.values())
    assert structures_by_model(None) == {}
    assert structures_by_model({"ranked": []}) == {}


def test_the_compare_page_carries_the_structure_page_controls_and_both_listings():
    rows = [_option("protein", "1EGF"), _option("protein", "2IL1", predicted=True)]
    by_database = structures_by_database(rows, {rows[0].accession: {"uniprot": SourceHit("record", "x")}})
    by_model = structures_by_model(
        {
            "ranked": [
                {
                    "entry_id": "1EGF",
                    "chain": "1",
                    "accession": "ACC-1EGF",
                    "title": "a",
                    "source": "rcsb",
                    "ratings": {"grok": 80},
                }
            ]
        }
    )
    page = render_option_comparison(
        {"confirmed": [], "in_both": [], "only_program": [], "only_llm": []},
        by_database=by_database,
        by_model=by_model,
    )

    # The same controls as the structures page, driven by the same page script.
    assert 'class="structure-view"' in page
    assert 'select name="sort"' in page
    assert 'select name="experimental"' in page
    assert 'select name="predicted"' in page
    assert 'class="ticks structure-rows"' in page, "the script filters these rows"
    assert 'data-kind="experimental"' in page
    assert 'data-kind="predicted"' in page
    assert "data-number=" in page and "data-score=" in page

    # Every structure against the eight databases.
    assert "<h2>Structures by database</h2>" in page
    assert "UniProt record" in page
    assert "Therapeutic Target Database not asked" in page
    assert "| 1 of 8 |" in page

    # Every structure under each model that rated it.
    assert "<h2>Structures by model</h2>" in page
    assert "<h3>grok</h3>" in page
    assert "rating 80" in page

    # With nothing to show the page still renders.
    bare = render_option_comparison({"confirmed": [], "in_both": [], "only_program": [], "only_llm": []})
    assert "(no model rated a structure)" in bare
    assert 'class="structure-view"' in bare


def test_every_panel_page_shows_the_project_title_at_the_top():
    first, second = PANEL_TITLE.splitlines()
    brand = f"<h1>{first.replace('   ', '&nbsp;&nbsp;&nbsp;')}<br>{second}</h1>"
    title = f"<title>{first} {second}</title>"
    disease = Disease(name="Asthma", disease_id="EFO_0000270")
    empty = {"confirmed": [], "in_both": [], "only_program": [], "only_llm": []}
    named = {
        "Confirm disease": render_confirm(disease),
        "Choose disease": render_disease_choice([disease]),
        "Choose structures": render_structures([_option("protein", "4PCW")]),
        "Compare options": render_option_comparison(empty),
        "Choose targets": render_targets([]),
        "Confirm inference": render_inference(1),
    }
    for name, page in named.items():
        assert page.count("<h1>") == 1
        assert page.index(brand) < page.index(f'<p class="step-title">{name}</p>')
        assert title in page
    for page in (render_options(OptionForm()), _done_page()):
        assert page.count("<h1>") == 1
        assert brand in page
        assert '<p class="step-title">' not in page
        assert title in page


def test_the_choose_structures_page_carries_the_llm_search_at_the_top():
    """The search lives here only, as model boxes, above the structure rows."""
    rows = [_option("protein", "1egfr")]
    page = render_structures(rows, saved=("grok",), scores={"acc-1egfr": (7, 8)})

    assert "<h2>LLM search</h2>" in page
    first, second = PANEL_TITLE.splitlines()
    brand = f"<h1>{first.replace('   ', '&nbsp;&nbsp;&nbsp;')}<br>{second}</h1>"
    assert brand in page
    assert '<p class="step-title">Choose structures</p>' in page
    assert 'class="page-actions page-actions-top"' in page
    assert page.index(brand) < page.index('<p class="step-title">Choose structures</p>')
    assert page.index("Choose structures") < page.index("<h2>Structures</h2>")
    assert page.index("<h2>LLM search</h2>") < page.index('name="pick"'), "above the rows"
    assert page.index("<h2>LLM search</h2>") < page.index('name="all_models"')
    assert page.index("<h2>LLM search</h2>") < page.index("First search")
    assert page.index("First search") < page.index(
        "Research: All found structures by highest value targets for generation"
    )
    assert page.index("highest value targets for generation") < page.index(">Download Structures</button>")
    assert "Research: All found structures by highest value targets for generation" in page
    # Both searches are live: the second ranks the structures the program search found.
    assert 'value="rank" formnovalidate>' in page
    assert 'name="next" value="missing"' in page
    assert 'name="next" value="rank"' in page

    # Model boxes only: no provider boxes anywhere.
    assert 'name="all_models"' in page
    assert "Search every listed model (With saved API Key)" in page
    assert "Search every listed model</label>" not in page
    assert 'name="saved_models"' not in page
    assert 'value="grok|grok-4.7" data-saved="yes"' in page
    assert 'value="claude|claude-fable-5-1"' not in page
    assert 'name="model" value="grok|' in page
    assert 'name="all_providers"' not in page
    assert 'name="research"' not in page

    # The score wording for the rows.
    assert "The score is how many of UniProt, NCBI, STRING, IntAct, the Human" in page
    assert "PRIDE, and PeptideAtlas returned a record for same structure." in page
    assert "The program search is already confirmed." not in page


def test_download_ignores_model_ticks_and_research_opens_the_comparison(monkeypatch):
    """Download keeps the ticked structures. A search button runs the models."""
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    rows = [_option("protein", "1egfr"), _option("protein", "1il13", predicted=True)]
    seen: dict[str, str] = {}

    def download(html: str) -> dict[str, list[str]]:
        seen["html"] = html
        return {"pick": ["2"], "all": ["yes"], "model": ["grok|grok-4.7"]}

    action, take_all, picks, providers = OptionPanel(download).choose_structures(rows)
    assert action == "download"
    assert seen["html"].index("<h2>LLM search</h2>") < seen["html"].index("First search")
    assert seen["html"].index("First search") < seen["html"].index(">Download Structures</button>")
    assert 'name="next" value="missing"' in seen["html"]
    assert take_all is True
    assert picks == "2"
    assert providers == [], "Download does not search, even when a model is ticked"

    def research(_html: str) -> dict[str, list[str]]:
        return {"pick": ["2"], "model": ["grok|grok-4.7"], "next": ["missing"]}

    action, take_all, picks, providers = OptionPanel(research).choose_structures(rows)
    assert action == "missing", "the first search is what reads the model ticks"
    assert take_all is False
    assert picks == "2"
    assert providers == ["grok"]


def test_the_question_box_sits_above_the_search_and_keeps_its_text():
    """The two question buttons sit beside each other under the box."""
    rows = [_option("protein", "1AAA")]
    page = render_structures(rows, saved=("grok",))
    ask_at = page.index('<div class="llm-ask">')
    search_at = page.index("<h2>LLM search</h2>")
    assert ask_at < search_at < page.index('name="pick"')
    block = page[ask_at:search_at]
    assert block.index('name="llm_question"') < block.index("llm-add-structures")
    assert block.index("llm-add-structures") < block.index('name="next" value="ask"')
    assert "Submit question" in block
    assert "Add selected structures" in block
    assert "First search" not in block
    assert "resize: both" in page
    assert "textarea.llm-question" in page
    assert ".llm-ask { display: flex; flex-direction: column;" in page
    assert ".llm-ask-actions { display: flex; flex-direction: row;" in page
    assert ".llm-ask-actions button { margin: 0; padding: 3px 8px; font-size: 12px;" in page
    button_at = block.index("llm-add-structures")
    button = block[block.rindex("<button", 0, button_at) : block.index(">", button_at)]
    assert 'type="button"' in button
    assert "name=" not in button
    assert "Tick a structure to add it." in page
    assert "input[name=pick]" in page
    assert "Selected structures are already in the message." in page
    assert 'split("\\n")' in page
    assert 'split("\\\\n")' not in page
    assert 'join("\\n")' in page
    assert "replace(/\\s+/g" in page
    assert "replace(/\\\\s+/g" not in page

    nasty = "keep <b> & </textarea><script>alert(1)</script>"
    shown = render_structures(rows, transcript=nasty)
    start = shown.index('<textarea class="llm-question"')
    end = shown.index("</textarea>", start)
    inner = shown[start:end]
    assert nasty not in shown
    assert "&lt;/textarea&gt;" in inner
    assert "&lt;script&gt;" in inner
    assert "&amp;" in inner
    assert "</textarea>" not in inner
    assert "<script>" not in inner


def test_submit_question_reads_the_checked_models_and_download_does_not(monkeypatch):
    """next=ask is its own action. A question with no next still downloads."""
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    rows = [_option("protein", "1egfr")]

    def ask(html: str) -> dict[str, list[str]]:
        assert 'name="llm_question"' in html
        assert "earlier reply" in html
        return {
            "next": ["ask"],
            "model": ["grok|grok-4.7"],
            "llm_question": ["Hello models  \n"],
            "pick": ["1"],
        }

    panel = OptionPanel(ask)
    action, take_all, picks, providers = panel.choose_structures(rows, transcript="earlier reply")
    assert action == "ask"
    assert providers == ["grok"]
    assert take_all is False
    assert picks == "1"
    assert panel.posted_question() == "Hello models"
    assert panel.research_models() == {"grok": ("grok-4.7",)}

    def download(_html: str) -> dict[str, list[str]]:
        return {"model": ["grok|grok-4.7"], "llm_question": ["still here"]}

    quiet = OptionPanel(download)
    action, _take_all, _picks, providers = quiet.choose_structures(rows)
    assert action == "download"
    assert providers == []
    assert quiet.posted_question() == "still here"


def test_a_question_appends_each_reply_as_the_model_gave_it():
    """Checked models answer into the same box. One failure does not drop the other.

    The reply is recorded as given, minus sequences. Dropping a sentence that
    named a cure matched phrases rather than claims, and what it removed was
    mostly the opposite of what it was for.
    """
    from rfproteina.disease.research import _ASK_LIMIT, ASK_PROMPT, ask_checked_models

    sent: list[tuple[str, str, str]] = []

    def complete(provider: str, system: str, user: str) -> str:
        sent.append((provider, system, user))
        if provider == "claude":
            raise ValueError("claude is down")
        return "A short answer. And a second sentence."

    models = {"grok": ("grok-4.7",), "claude": ("claude-fable-5-1",)}
    box = ask_checked_models(
        "What is IL5?",
        providers=["claude", "grok"],
        complete=complete,
        search_models=models,
    )
    assert sent[0][0] == "grok"
    assert sent[0][1] == ASK_PROMPT
    assert sent[0][2] == "What is IL5?"
    assert [item[0] for item in sent] == ["grok", "claude"]
    assert box.startswith("What is IL5?\n\n")
    # Both sentences, in the order the model gave them. Sentences used to be
    # dropped by a phrase match, which cost more than it caught.
    assert "Grok 4.7:\nA short answer. And a second sentence." in box
    assert "Claude Fable 5.1:\nclaude is down" in box

    again = ask_checked_models(
        box + "\nSecond question.",
        providers="grok",
        complete=lambda *_args: "second reply",
        search_models={"grok": ("grok-4.7",)},
    )
    assert again.count("Grok 4.7:") == 2
    assert "What is IL5?" in again
    assert "claude is down" in again
    assert again.endswith("Grok 4.7:\nsecond reply")

    long = "q" * (_ASK_LIMIT + 50)
    seen: dict[str, str] = {}

    def limited(_provider: str, _system: str, user: str) -> str:
        seen["user"] = user
        return "ok"

    capped = ask_checked_models(
        long,
        providers="grok",
        complete=limited,
        search_models={"grok": ("grok-4.7",)},
    )
    assert seen["user"] == long[-_ASK_LIMIT:]
    assert capped.startswith(long)
    assert capped.endswith("Grok 4.7:\nok")

    called = False

    def unused(*_args: object) -> str:
        nonlocal called
        called = True
        return "no"

    assert ask_checked_models("  \n", providers="grok", complete=unused) == "  \n"
    assert called is False
    # A reply that is nothing but a sequence still leaves nothing to show, which
    # is the state this checks. It used to be reached with a cure claim, and a
    # cure claim is no longer dropped.
    silent = ask_checked_models(
        "Say nothing useful.",
        providers="grok",
        complete=lambda *_args: "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEG",
        search_models={"grok": ("grok-4.7",)},
    )
    assert silent.endswith("Grok 4.7:\nNo reply was kept.")
    assert "MQIFVKTLTGK" not in silent


def test_grok_bot_needs_no_key_and_stays_out_of_the_saved_key_list():
    """Grok Bot is offered with no key, and select-all does not tick it."""
    page = render_structures([_option("protein", "1AAA")])
    search_at = page.index('class="ticks llm-search"')
    search_end = page.index("</div>", search_at)
    bot_at = page.index('class="ticks llm-grok-bot"')
    assert search_end < bot_at
    assert "grokbot|chief-of-staff" not in page[search_at:search_end]
    assert 'value="grokbot|chief-of-staff" data-saved="no"' in page
    assert "Grok 4.7</label>" not in page
    keyed = render_structures([_option("protein", "1AAA")], saved=("grok",))
    assert 'value="grok|grok-4.7" data-saved="yes"' in keyed
    assert 'value="grokbot|chief-of-staff" data-saved="no"' in keyed


def test_submit_question_includes_grok_bot_and_the_searches_do_not(monkeypatch):
    """Only Submit question keeps the keyless tick. Download and both searches drop it."""
    _clear_provider_keys(monkeypatch)
    rows = [_option("protein", "1AAA")]

    def chosen(fields: dict[str, list[str]]) -> tuple[str, list[str]]:
        action, _take_all, _picks, providers = OptionPanel(lambda _page: fields).choose_structures(rows)
        return action, providers

    assert chosen({"next": ["ask"], "model": ["grokbot|chief-of-staff"], "llm_question": ["Hi"]}) == (
        "ask",
        ["grokbot"],
    )
    assert chosen({"next": ["ask"], "model": ["grokbot|other"]}) == ("ask", [])
    assert chosen({"next": ["missing"], "model": ["grokbot|chief-of-staff"]}) == ("missing", [])
    assert chosen({"next": ["rank"], "model": ["grokbot|chief-of-staff"]}) == ("rank", [])
    assert chosen({"next": ["download"], "model": ["grokbot|chief-of-staff"]}) == ("download", [])
    asked, plan = research_from_fields(
        {"model": ["grok|grok-4.7", "grokbot|chief-of-staff"], "all_models": ["yes"]},
        saved=("grok",),
        include_grok_bot=True,
    )
    assert asked[0] == "grok"
    assert asked[-1] == "grokbot"
    assert plan is not None and "grokbot" not in plan
    untouched, _models = research_from_fields(
        {"model": ["grokbot|chief-of-staff"], "all_models": ["yes"]},
        saved=("grok",),
    )
    assert untouched == ["grok"]


def test_grok_bot_reply_returns_to_the_box_without_an_api_call():
    """The desktop reply is labeled in the box. A keyed model still uses its own call."""
    from rfproteina.disease.research import ask_checked_models

    sent: list[str] = []
    pasted: list[str] = []

    def complete(provider: str, _system: str, user: str) -> str:
        sent.append(provider)
        assert provider != "grokbot"
        return f"from {user}"

    def ask_desktop(question: str) -> str:
        pasted.append(question)
        return "Checked the request. And a second sentence."

    box = ask_checked_models(
        "What is IL5?",
        providers=["grok", "grokbot"],
        complete=complete,
        search_models={"grok": ("grok-4.7",)},
        ask_desktop=ask_desktop,
    )
    assert sent == ["grok"]
    assert pasted == ["What is IL5?"]
    assert "Grok 4.7:\nfrom What is IL5?" in box
    # The bot's whole reply is kept, both sentences of it.
    assert box.endswith("Grok Bot:\nChecked the request. And a second sentence.")

    alone = ask_checked_models("Only the bot.", providers="grokbot", complete=complete, ask_desktop=ask_desktop)
    assert sent == ["grok"]
    assert alone.endswith("Grok Bot:\nChecked the request. And a second sentence.")

    quiet = ask_checked_models(
        "Say nothing useful.",
        providers="grokbot",
        complete=complete,
        ask_desktop=lambda _question: "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEG",
    )
    assert quiet.endswith("Grok Bot:\nNo reply was kept.")


def test_the_desktop_paste_is_a_chief_of_staff_request_and_the_reply_is_read_back(monkeypatch):
    """The paste addresses Chief of Staff, carries a marker, and the reader is asked for that marker."""
    import base64
    import json

    from rfproteina.disease.grok_bot import (
        APP_LINK,
        PASTE_MAX,
        _reply_text,
        ask_grok_bot,
        claim_open,
        grok_bot_task,
        launch_target,
        paste_script,
        powershell_command,
        reply_script,
        reset_open_claim,
    )

    block_live_calls(monkeypatch)
    reset_open_claim()
    assert paste_script().is_file()
    assert reply_script().is_file()
    command = powershell_command(paste_script(), "powershell.exe")
    assert command[-2] == "-File"
    assert command[-1].endswith("paste-grok-bot.ps1")
    assert "What is IL5?" not in " ".join(command)

    task = grok_bot_task("What is IL5?\n\nKeep this end.", "GEV-abc12345")
    assert task.startswith("Chief of Staff, act on this request")
    assert "What is IL5?" in task
    assert task.endswith("[GEV-abc12345]")
    assert "probability of cure" in task
    assert len(task) <= PASTE_MAX
    long = grok_bot_task("start " + ("q" * (PASTE_MAX + 50)) + " END", "GEV-abc12345")
    assert long.endswith("[GEV-abc12345]")
    assert " END" in long
    assert "start " not in long
    assert len(long) <= PASTE_MAX

    assert launch_target({}) == (APP_LINK, "app-link")
    assert launch_target({"GROK_BOT_LINK": r"C:\Grok\Grok Bot.lnk"}, exists=lambda _path: True)[1] == "shortcut"
    assert launch_target({"GROK_BOT_LINK": r"C:\Grok\bot.bat"}, exists=lambda _path: True)[1] == "refused"
    assert launch_target({"GROK_BOT_LINK": r"C:\Grok\missing.lnk"}, exists=lambda _path: False)[1] == "refused"

    opened: list[str] = []
    pasted: list[str] = []
    read: list[str] = []

    def open_app(target: str) -> None:
        opened.append(target)

    def paste(text: str) -> bool:
        pasted.append(text)
        return True

    def read_reply(tag: str) -> str:
        read.append(tag)
        return "Chief of Staff acted."

    reply = ask_grok_bot(
        "What is IL5?",
        open_app=open_app,
        paste=paste,
        read_reply=read_reply,
        sleep=lambda _seconds: None,
        env={},
    )
    assert opened == [APP_LINK]
    assert pasted and pasted[0].startswith("Chief of Staff, act on this request")
    assert "What is IL5?" in pasted[0]
    assert read == [pasted[0].rsplit("[", 1)[1][:-1]]
    assert reply == "Chief of Staff acted."

    def refuse_paste(_text: str) -> bool:
        return False

    def unexpected(_tag: str) -> str:
        raise AssertionError("a missed paste must not be read")

    with pytest.raises(ValueError, match="could not be pasted"):
        ask_grok_bot(
            "Again",
            open_app=open_app,
            paste=refuse_paste,
            read_reply=unexpected,
            sleep=lambda _seconds: None,
            env={},
        )

    with pytest.raises(ValueError, match="not a Grok Bot shortcut"):
        ask_grok_bot("Hi", env={"GROK_BOT_LINK": "not-a-path"}, clock=lambda: 0.0)
    assert claim_open(0.0) is True

    reset_open_claim()
    assert claim_open(100.0) is True
    with pytest.raises(ValueError, match="Grok Bot is opening"):
        ask_grok_bot("Hi", env={}, clock=lambda: 101.0)
    encoded = base64.b64encode(b"Checked the request.").decode("ascii")
    raw = json.dumps({"event": "reply", "how": "read", "b64": encoded}).encode("utf-8")
    assert _reply_text(raw) == "Checked the request."
    with pytest.raises(ValueError, match="did not come back") as missed:
        _reply_text(b'{"event":"error","reason":"the pasted prompt is not in the app"}')
    assert "pasted prompt" not in str(missed.value)


def test_submit_question_returns_the_grok_bot_reply_and_the_searches_do_not(tmp_path, monkeypatch, capsys):
    """Submit question writes the desktop reply into the box. The searches do not paste."""
    import re
    from html import unescape

    _clear_provider_keys(monkeypatch, "GROK_BOT_LINK", "GROK_BOT_API_KEY")
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "grok-bot"
    pages: list[str] = []
    calls: list[str] = []

    def ask_desktop(question: str) -> str:
        calls.append(question)
        return "Checked the request. And a second sentence."

    def box_text(page: str) -> str:
        match = re.search(
            r'<textarea class="llm-question" name="llm_question"[^>]*>(.*?)</textarea>',
            page,
            re.DOTALL,
        )
        assert match is not None
        return unescape(match.group(1))

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False)
        if _is_generation_page(page):
            return {}
        assert 'type="checkbox" name="pick"' in page
        shown = sum(1 for seen in pages if 'type="checkbox" name="pick"' in seen)
        if shown == 1:
            assert 'value="grokbot|chief-of-staff"' in page
            assert "Grok 4.7</label>" not in page
            return {
                "llm_question": ["What is IL5?"],
                "model": ["grokbot|chief-of-staff"],
                "next": ["ask"],
                "pick": ["1"],
            }
        if shown == 2:
            text = box_text(page)
            assert "Grok Bot:\nChecked the request. And a second sentence." in text
            assert calls == ["What is IL5?"]
            return {
                "llm_question": [text],
                "model": ["grokbot|chief-of-staff"],
                "next": ["missing"],
                "pick": ["1"],
            }
        if shown == 3:
            assert "Tick a model for the search." in page
            assert calls == ["What is IL5?"]
            return {
                "llm_question": [box_text(page)],
                "model": ["grokbot|chief-of-staff"],
                "next": ["rank"],
                "pick": ["1"],
            }
        assert "Tick a model for the search." in page
        assert len(calls) == 1
        return {"pick": ["1"], "llm_question": [box_text(page)]}

    code = _run_panel(
        monkeypatch,
        exchange,
        fetch=lambda _option: b"data_X\n#\n",
        read_json=lambda _url: {},
        ask_desktop=ask_desktop,
        runner=lambda _argv: (_ for _ in ()).throw(AssertionError("generation")),
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert calls == ["What is IL5?"]
    assert "What is IL5?" not in captured.out
    structure_pages = [page for page in pages if 'type="checkbox" name="pick"' in page]
    assert len(structure_pages) == 4


def test_a_panel_question_returns_to_the_same_box_and_download_does_not_ask(tmp_path, monkeypatch, capsys):
    """Submit question stays on Choose structures. Download still does not call a model."""
    import re
    from html import unescape

    from rfproteina.disease.research import ASK_PROMPT

    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "question-box"
    pages: list[str] = []
    calls: list[tuple[str, str]] = []

    def complete(provider: str, system: str, user: str) -> str:
        calls.append((provider, user))
        assert system == ASK_PROMPT
        if provider == "claude":
            raise ValueError("claude is down")
        return "A short answer. And a second sentence."

    def box_text(page: str) -> str:
        match = re.search(
            r'<textarea class="llm-question" name="llm_question"[^>]*>(.*?)</textarea>',
            page,
            re.DOTALL,
        )
        assert match is not None
        return unescape(match.group(1))

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False)
        if _is_generation_page(page):
            return {}
        assert 'type="checkbox" name="pick"' in page
        shown = sum(1 for seen in pages if 'type="checkbox" name="pick"' in seen)
        if shown == 1:
            assert page.index('name="llm_question"') < page.index("<h2>LLM search</h2>")
            assert "resize: both" in page
            return {"llm_question": ["What is IL5?"], "next": ["ask"], "pick": ["1"]}
        if shown == 2:
            assert "Tick a model for the question." in page
            assert box_text(page) == "What is IL5?"
            assert calls == []
            return {
                "llm_question": ["What is IL5?"],
                "model": ["grok|grok-4.7", "claude|claude-fable-5-1"],
                "next": ["ask"],
                "pick": ["1"],
            }
        if shown == 3:
            text = box_text(page)
            assert text.startswith("What is IL5?\n\n")
            assert "Grok 4.7:\nA short answer. And a second sentence." in text
            assert "Claude Fable 5.1:\nclaude is down" in text
            assert [item[0] for item in calls] == ["grok", "claude"]
            return {
                "llm_question": [text + "\nSecond question about the same target."],
                "model": ["grok|grok-4.7"],
                "next": ["ask"],
                "pick": ["1"],
            }
        text = box_text(page)
        assert text.count("Grok 4.7:") == 2
        assert "Second question about the same target." in text
        assert "What is IL5?" in text
        assert "claude is down" in text
        assert len(calls) == 3
        assert calls[2][0] == "grok"
        assert calls[2][1].endswith("Second question about the same target.")
        return {"pick": ["1"], "llm_question": [text]}

    code = _run_panel(
        monkeypatch,
        exchange,
        fetch=lambda _option: b"data_X\n#\n",
        read_json=lambda _url: {},
        complete=complete,
        runner=lambda _argv: (_ for _ in ()).throw(AssertionError("generation")),
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert len(calls) == 3, "Download does not send the box"
    assert "What is IL5?" not in captured.out
    structure_pages = [page for page in pages if 'type="checkbox" name="pick"' in page]
    assert len(structure_pages) == 4


def test_a_ticked_search_or_research_reply_is_sent_with_the_new_question(monkeypatch):
    """Submit question sends the ticked earlier reply, then the further question."""
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    rows = [_option("protein", "1AAA"), _option("protein", "2BBB")]
    ranking = [
        _ranked(2, "2BBB", "ACC-2BBB", {"grok grok-4.7": 40}, 40.0, reasons={"grok grok-4.7": "Worth a binder design."})
    ]
    results = llm_results(rows, found={"grok grok-4.7": [["1AAA", "A"]]}, ranking={"ranked": ranking})
    page = render_structures(rows, saved=("grok",), ranking=ranking, results=results)
    assert 'name="prior_reply" value="result:0"' in page
    assert 'name="prior_reply" value="review:0"' in page
    assert "Tick a search or research reply to include it." in page
    assert page.index('value="result:0"') < page.index('value="review:0"')
    assert 'name="prior_reply"' not in render_structures(rows)

    def ask(_html: str) -> dict[str, list[str]]:
        return {
            "next": ["ask"],
            "model": ["grok|grok-4.7"],
            "prior_reply": ["result:0", "review:0", "missing"],
            "llm_question": ["What should change?"],
        }

    panel = OptionPanel(ask)
    action, _take_all, _picks, providers = panel.choose_structures(rows, ranking=ranking, results=results)
    assert action == "ask"
    assert providers == ["grok"]
    message = panel.question_for_models()
    assert "LLM Grok 4.7 Ranking none; named it" in message
    assert "Worth a binder design." in message
    assert message.endswith("What should change?")
    assert message.index("Worth a binder design.") < message.index("What should change?")
    assert message.count("Worth a binder design.") == 1

    def again(_html: str) -> dict[str, list[str]]:
        return {
            "next": ["ask"],
            "model": ["grok|grok-4.7"],
            "prior_reply": ["review:0"],
            "llm_question": [message],
        }

    repeat = OptionPanel(again)
    repeat.choose_structures(rows, ranking=ranking, results=results)
    assert repeat.question_for_models().count("Worth a binder design.") == 1

    def replies_only(_html: str) -> dict[str, list[str]]:
        return {"next": ["ask"], "model": ["grok|grok-4.7"], "prior_reply": ["review:0"]}

    only = OptionPanel(replies_only)
    only.choose_structures(rows, ranking=ranking, results=results)
    assert only.posted_question() == ""
    assert "Worth a binder design." in only.question_for_models()

    def download(_html: str) -> dict[str, list[str]]:
        return {"prior_reply": ["review:0"], "llm_question": ["leave this"], "model": ["grok|grok-4.7"]}

    quiet = OptionPanel(download)
    action, _take_all, _picks, providers = quiet.choose_structures(rows, ranking=ranking, results=results)
    assert action == "download"
    assert providers == []


def test_ticked_models_name_their_own_providers():
    """With no provider boxes, a ticked model is what selects its provider.

    A provider with no API key is not searched, including when every model is ticked.
    """
    keyed = ("grok", "openai")
    providers, models = research_from_fields(
        {"model": ["grok|grok-4.6", "openai|gpt-6-luna"]},
        saved=keyed,
    )
    assert providers == ["grok", "openai"]
    assert models == {"grok": ("grok-4.6",), "openai": ("gpt-6-luna",)}
    only_grok, grok_plan = research_from_fields(
        {"model": ["grok|grok-4.6", "openai|gpt-6-luna"]},
        saved=("grok",),
    )
    assert only_grok == ["grok"], "openai has no key, so it is not searched"
    assert grok_plan == {"grok": ("grok-4.6",)}

    every, plan = research_from_fields({"all_models": ["yes"]}, saved=PROVIDERS)
    assert every == list(PROVIDERS), "every listed model means every provider that has a key"
    assert plan is not None and set(plan) == set(PROVIDERS)

    assert research_from_fields({}) == ([], None), "nothing ticked leaves the search off"
    assert research_from_fields({"all_models": ["yes"]}) == ([], None)
    grok_ids = tuple(model_id for model_id, _label in model_choices("grok"))
    saved_names, saved_plan = research_from_fields({"all_models": ["yes"]}, saved=("grok",))
    assert saved_names == ["grok"]
    assert saved_plan == {"grok": grok_ids}


def test_all_expands_to_every_provider_and_an_unknown_name_is_refused():
    assert chosen_providers("all") == ("grok", "claude", "openai", "nvidia", "openrouter")
    assert chosen_providers(["openai,grok"]) == ("grok", "openai")
    assert chosen_providers(None, "claude") == ("claude",)
    with pytest.raises(ValueError, match="Unknown research provider"):
        chosen_providers("gemini")
    with pytest.raises(ValueError, match="Choose at least one research provider"):
        chosen_providers(None)


def test_nvidia_and_openrouter_keys_round_trip_encrypted_and_prefer_a_typed_key(monkeypatch):
    nvidia_secret = "nv-panel-secret"
    router_secret = "or-panel-secret"
    save_api_keys({"nvidia": nvidia_secret, "openrouter": router_secret, "gemini": "drop-me"})
    assert load_api_keys() == {"nvidia": nvidia_secret, "openrouter": router_secret}
    stored = _stored_key_text()
    assert nvidia_secret not in stored
    assert router_secret not in stored
    assert "drop-me" not in stored
    for name in ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        api_key("nvidia")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-from-env")
    assert api_key("nvidia") == "nvapi-from-env"
    assert api_key("nvidia", {"nvidia": nvidia_secret}) == nvidia_secret
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-primary")
    assert api_key("nvidia") == "nvapi-primary"
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        api_key("openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-from-env")
    assert api_key("openrouter") == "or-from-env"
    assert api_key("openrouter", {"openrouter": router_secret}) == router_secret


def test_a_logged_panel_form_redacts_nvidia_and_openrouter_keys():
    secrets = {
        "grok_key": "grok-panel-secret",
        "claude_key": "claude-panel-secret",
        "openai_key": "openai-panel-secret",
        "nvidia_key": "nvidia-panel-secret",
        "openrouter_key": "router-panel-secret",
    }
    cleaned = redact_fields(
        {name: [value] for name, value in secrets.items()}
        | {"disease": ["Asthma"], "nvidia_model": ["z-ai/glm-5.3-flash"]}
    )
    text = json.dumps(cleaned)
    assert cleaned["disease"] == "Asthma"
    assert cleaned["nvidia_model"] == "z-ai/glm-5.3-flash"
    for value in secrets.values():
        assert value not in text
    assert {cleaned[name] for name in secrets} == {"[redacted]"}


def test_a_provider_without_a_key_is_named_and_the_others_still_search(monkeypatch):
    for name in ("XAI_API_KEY", "GROK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    usable, unavailable = usable_providers(("grok", "claude", "openai"), {"grok": "k-1"})
    assert usable == ("grok",)
    assert "ANTHROPIC_API_KEY" in unavailable["claude"]
    assert "OPENAI_API_KEY" in unavailable["openai"]
    assert "k-1" not in str(unavailable)
    with pytest.raises(ValueError) as failure:
        usable_providers(("claude", "openai"))
    assert "ANTHROPIC_API_KEY" in str(failure.value)
    assert "OPENAI_API_KEY" in str(failure.value)


def test_the_program_score_is_the_record_count_only():
    """One score: databases that returned a record. The deposit method is not in it."""
    none = target_value({})
    one = target_value({"uniprot": SourceHit("record", "x")})
    two = target_value({"uniprot": SourceHit("record", "x"), "ncbi": SourceHit("record", "y")})
    assert (none["program_score"], one["program_score"], two["program_score"]) == (0, 1, 2)
    assert none["asked"] == 8

    # Every signal the records carry is reported, and none of them moves the score.
    rich = target_value({"uniprot": SourceHit("record", "x", partners=40, disease_linked=True, druggable=True)})
    assert rich["program_score"] == one["program_score"]
    assert rich["partners"] == 40
    assert rich["disease_linked"] is True
    assert rich["druggable"] is True

    # The share is that count as a percentage, which is what ranks beside the ratings.
    assert record_share(0, 8) == 0
    assert record_share(2, 8) == 25
    assert record_share(8, 8) == 100
    assert record_share(1, 0) == 0, "no database asked is no score, not a crash"


def test_a_protein_class_binders_are_made_against_counts_as_druggable():
    kinase = target_value({"human_protein_atlas": SourceHit("record", "x", druggable=True)})
    assert kinase["druggable"] is True
    assert target_value({"human_protein_atlas": SourceHit("record", "x")})["druggable"] is False


def test_research_subjects_put_uniprot_accessions_before_genes():
    proteins = [
        StructureOption(
            group="protein",
            accession=f"P{index}AAA{index}",
            entry_id=f"1A{index}",
            chain="A",
            title="protein",
            predicted=False,
            source="fixture",
        )
        for index in range(9)
    ]
    gene = StructureOption(
        group="gene",
        accession="FLG",
        entry_id="9BSB",
        chain="3",
        title="flg",
        predicted=False,
        source="fixture",
    )
    subjects = research_subjects([_option("dna", "1dna"), *proteins, gene])
    assert [subject.identifier for subject in subjects] == [f"P{index}AAA{index}" for index in range(8)]
    assert research_subjects([gene])[0].kind == "gene"


def test_research_ranks_a_target_from_its_records_and_the_provider_rating(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "sk-should-not-appear")
    records = structures_json(tmp_path, ROW_P00533)
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 3
    calls: list[str] = []

    def read(url: str):
        calls.append(url)
        assert sequence not in url
        assert "promast" not in url
        assert "peptide=" not in url
        if "uniprot.org" in url:
            return {
                "results": [
                    {
                        "primaryAccession": "P00533",
                        "proteinDescription": {
                            "recommendedName": {"fullName": {"value": "Epidermal growth factor receptor"}}
                        },
                        "genes": [{"geneName": {"value": "EGFR"}}],
                        "comments": [{"commentType": "FUNCTION", "texts": [{"value": f"Binds ligand {sequence}"}]}],
                    }
                ]
            }
        if "esearch" in url:
            raise ValueError("ncbi down")
        if "string-db.org" in url:
            return [{"preferredName_A": "EGFR", "preferredName_B": "EGF"}]
        if "intact" in url:
            return {"totalElements": 1, "content": [{"id": "EBI-1"}]}
        if "proteinatlas.org" in url:
            return [
                {
                    "gs": "EGFR",
                    "up": "P00533",
                    "gd": "receptor tyrosine kinase",
                    "pc": "Enzymes, Kinases, FDA approved drug targets",
                    "di": "cancer",
                }
            ]
        if "pride" in url:
            return {"_embedded": {"projects": [{"title": "EGFR phosphoproteome"}]}}
        if "/proteomics/" in url:
            return {"features": [{"source": "PeptideAtlas", "peptide": sequence}]}
        if "getSearchTargetByTargetNameList" in url:
            return {
                "results": [
                    "Epidermal growth factor receptor (EGFR)",
                    "Interleukin-13 (IL13)",
                ]
            }
        raise AssertionError(url)

    def complete(provider: str, system: str, user: str) -> str:
        assert provider == "grok"
        assert "rating" in system
        assert "JSON" in system
        assert "cure" not in system.casefold()
        assert "cure" not in user.casefold()
        assert sequence not in user
        assert "PeptideAtlas lists proteomics evidence" in user
        return json.dumps(
            {
                "rating": 80,
                "reason": "UniProt ties it to disease.",
                "ideas": f"Signaling is receptor-driven. {sequence}",
            }
        )

    out = tmp_path / "research-runs"
    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--all", "--confirm-inference", "--research", "grok"],
        out_dir=out,
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=read,
        complete=complete,
    )
    code = captured.code
    assert code == 0
    assert captured.err == ""
    assert "protein:" in captured.out
    assert captured.out.index("protein:") < captured.out.index("Target research")
    assert "Ranked for binder generation, highest value first:" in captured.out
    assert "1. P00533 egfr: combined " in captured.out
    assert "program score 7 of 8" in captured.out
    assert "providers 80.0 (grok 80)" in captured.out
    assert "grok: rating 80. UniProt ties it to disease." in captured.out
    assert "Signaling is receptor-driven." in captured.out
    assert sequence not in captured.out
    assert "sk-should-not-appear" not in captured.out
    assert captured.out.strip().endswith("Generation was not started.")
    report = json.loads((out / "target_research.json").read_text(encoding="utf-8"))
    body = json.dumps(report)
    assert sequence not in body
    assert "sk-should-not-appear" not in body
    assert "disclaimer" not in report
    assert report["providers"] == ["grok"]
    assert report["ranking"] == ["P00533"]
    subject = report["subjects"][0]
    assert subject["score"] == 7, "every database answered, TTD included"
    assert subject["sources_with_record"] == 7
    assert subject["sources_asked"] == 8
    assert subject["structures"] == 1
    assert subject["experimental_structures"] == 1
    assert subject["disease_linked"] is True
    assert subject["druggable"] is True
    assert subject["sources"]["ncbi"]["status"] == "unavailable"
    assert subject["sources"]["ttd"]["status"] == "record", "TTD is searched, not just cited"
    assert subject["sources"]["ttd"]["detail"] == (
        "Therapeutic Target Database lists Epidermal growth factor receptor (EGFR)."
    )
    assert subject["sources"]["peptideatlas"]["detail"] == "PeptideAtlas lists proteomics evidence for this protein."
    assert subject["models"]["grok"]["rating"] == 80
    assert subject["models"]["grok"]["narrative"] == "Signaling is receptor-driven."
    assert subject["agreement"] == "one provider"
    # TTD is asked once for the whole run, by gene symbol, from its target list.
    ttd_calls = [url for url in calls if "idrblab" in url]
    assert len(ttd_calls) == 1
    assert "getSearchTargetByTargetNameList" in ttd_calls[0]
    assert any("uniprot.org" in url for url in calls)


def test_every_api_key_searches_once_and_the_ratings_are_compared(tmp_path, monkeypatch, capsys):
    """Every provider reads one set of records, and the spread is reported."""
    records = structures_json(
        tmp_path,
        ROW_P00533,
        {
            "group": "gene",
            "accession": "IL13",
            "entry_id": "IL13",
            "chain": "A",
            "title": "interleukin 13",
            "predicted": True,
            "source": "fixture",
        },
    )
    reads: list[str] = []
    asked: list[tuple[str, str]] = []

    def read(url: str):
        reads.append(url)
        if "uniprot.org" in url:
            return {"results": [{"primaryAccession": "P00533", "comments": [{"commentType": "DISEASE"}]}]}
        if "esearch" in url:
            return {"esearchresult": {"idlist": []}}
        if "string-db.org" in url:
            return [{"preferredName_B": f"P{index}"} for index in range(40)]
        return {}

    # The two added providers sit inside the existing span, so the mean and the spread stay put.
    ratings = {"grok": 90, "claude": 85, "openai": 20, "nvidia": 65, "openrouter": 65}

    def complete(provider: str, _system: str, user: str) -> str:
        identifier = "IL13" if "Target: gene IL13" in user else "P00533"
        asked.append((provider, identifier))
        rating = ratings[provider] if identifier == "P00533" else 30
        return json.dumps({"rating": rating, "reason": f"{provider} read the records.", "ideas": "An idea."})

    out = tmp_path / "all-keys"
    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--all", "--confirm-inference", "--research", "all"],
        out_dir=out,
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=read,
        complete=complete,
    )
    code = captured.code
    assert code == 0, captured.err
    assert "with grok, claude, openai, nvidia, openrouter" in captured.out

    # The databases are asked once per target, not once per target per provider.
    assert sorted(asked) == sorted(
        (provider, identifier)
        for identifier in ("IL13", "P00533")
        for provider in ("claude", "grok", "openai", "nvidia", "openrouter")
    )
    assert len([url for url in reads if "uniprot.org" in url]) == 2

    report = json.loads((out / "target_research.json").read_text(encoding="utf-8"))
    assert report["providers"] == ["grok", "claude", "openai", "nvidia", "openrouter"]
    subject = next(row for row in report["subjects"] if row["identifier"] == "P00533")
    assert subject["ratings"] == ratings
    assert subject["model_rating"] == 65.0
    assert subject["rating_spread"] == 70
    assert subject["agreement"] == "disputed"
    disputed = report["comparison"]["disputed"]
    assert disputed[0]["identifier"] == "P00533"
    assert disputed[0]["highest"] == "grok"
    assert disputed[0]["lowest"] == "openai"
    assert report["comparison"]["top_by_provider"] == {
        "grok": "P00533",
        "claude": "P00533",
        "openai": "IL13",
        "nvidia": "P00533",
        "openrouter": "P00533",
    }
    assert "The providers did not agree on:" in captured.out
    assert "grok rated it 90, openai rated it 20" in captured.out
    assert report["ranking"] == ["P00533", "IL13"]


def test_providers_that_agree_are_marked_and_a_failed_provider_keeps_the_rest(tmp_path, monkeypatch, capsys):
    records = structures_json(tmp_path, ROW_P00533)

    def complete(provider: str, _system: str, _user: str) -> str:
        if provider == "openai":
            raise ValueError("openai refused the request")
        return json.dumps({"rating": 70 if provider == "grok" else 78, "reason": "", "ideas": "An idea."})

    out = tmp_path / "agreeing"
    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--all", "--confirm-inference", "--research", "grok,claude,openai"],
        out_dir=out,
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=lambda _url: {},
        complete=complete,
    )
    code = captured.code
    assert code == 0, captured.err
    report = json.loads((out / "target_research.json").read_text(encoding="utf-8"))
    subject = report["subjects"][0]
    assert subject["ratings"] == {"grok": 70, "claude": 78}
    assert subject["agreement"] == "agreement"
    assert report["comparison"]["agreed"] == ["P00533"]
    assert subject["models"]["openai"]["error"] == "openai refused the request"
    assert subject["models"]["openai"]["rating"] is None
    assert "openai: no answer. openai refused the request" in captured.out


def test_a_chosen_target_decides_which_structures_go_to_generation(tmp_path, monkeypatch, capsys):
    """--target restricts the download; it does not add to every named structure."""
    block_live_calls(monkeypatch)
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533, ROW_P35225)
    fetched: list[str] = []

    def complete(_provider: str, _system: str, user: str) -> str:
        # Every provider names a structure for whichever target it was asked about.
        entry = "1EGFR" if "P00533" in user else "2IL13"
        chain = "A" if entry == "1EGFR" else "B"
        return json.dumps(
            {
                "rating": 70,
                "reason": "",
                "ideas": "An idea.",
                "structures": [{"pdb_id": entry, "chain": chain}],
            }
        )

    out = tmp_path / "chosen-target"
    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--confirm-inference", "--research", "grok", "--target", "P35225"],
        out_dir=out,
        fetch=lambda option: fetched.append(option.entry_id) or b"data_X\n#\n",
        read_json=lambda _url: {},
        complete=complete,
    )
    code = captured.code
    assert code == 0, captured.err
    assert fetched == ["2IL13"], "only the chosen target is downloaded"
    assert "Selected 1 structure(s) for download." in captured.out
    assert not (out / "protein_1EGFR_A.cif").exists()

    # A rank number picks the same way, and the ranking is what it indexes.
    second = tmp_path / "by-rank"
    fetched.clear()
    assert (
        cure_main(
            [
                "--catalog",
                str(catalog),
                "--disease",
                "Asthma",
                "--accept-only",
                "--structures-json",
                str(records),
                "--confirm-inference",
                "--research",
                "grok",
                "--target",
                "1",
                "--out-dir",
                str(second),
            ],
            fetch=lambda option: fetched.append(option.entry_id) or b"data_X\n#\n",
            read_json=lambda _url: {},
            complete=complete,
        )
        == 0
    )
    report = json.loads((second / "target_research.json").read_text(encoding="utf-8"))
    assert fetched == [report["ranking"][0].replace("P00533", "1EGFR").replace("P35225", "2IL13")]


def test_a_target_outside_the_ranking_is_refused(tmp_path, monkeypatch, capsys):
    block_live_calls(monkeypatch)
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "bad-target"

    def fetch(_option):
        raise AssertionError("download")

    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--confirm-inference", "--research", "grok", "--target", "Q99999"],
        out_dir=out,
        fetch=fetch,
        read_json=lambda _url: {},
        complete=lambda *_args: json.dumps({"rating": 50, "reason": "", "ideas": "An idea."}),
    )
    code = captured.code
    assert code == 2
    assert "is not on the ranked list" in captured.err
    assert "P00533" in captured.err, "the error names what was ranked"

    # A rank number outside the list is refused the same way.
    assert (
        cure_main(
            [
                "--catalog",
                str(catalog),
                "--disease",
                "Asthma",
                "--accept-only",
                "--structures-json",
                str(records),
                "--confirm-inference",
                "--research",
                "grok",
                "--target",
                "7",
                "--out-dir",
                str(tmp_path / "bad-rank"),
            ],
            fetch=fetch,
            read_json=lambda _url: {},
            complete=lambda *_args: json.dumps({"rating": 50, "reason": "", "ideas": "An idea."}),
        )
        == 2
    )
    assert "outside 1..1" in capsys.readouterr().err


def test_target_without_research_is_refused(tmp_path, monkeypatch, capsys):
    block_live_calls(monkeypatch)
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    code = cure_main(
        [
            "--catalog",
            str(catalog),
            "--disease",
            "Asthma",
            "--accept-only",
            "--structures-json",
            str(records),
            "--confirm-inference",
            "--target",
            "P00533",
            "--out-dir",
            str(tmp_path / "no-research"),
        ],
        fetch=lambda _option: (_ for _ in ()).throw(AssertionError("download")),
    )
    assert code == 2
    assert "Pass --research so the targets are ranked" in capsys.readouterr().err


def test_a_provider_answer_that_is_not_json_is_kept_as_ideas(tmp_path, monkeypatch, capsys):
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "prose"
    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--all", "--confirm-inference", "--research", "grok"],
        out_dir=out,
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=lambda _url: {},
        complete=lambda *_args: "The records support a knockout study.",
    )
    code = captured.code
    assert code == 0, captured.err
    report = json.loads((out / "target_research.json").read_text(encoding="utf-8"))
    model = report["subjects"][0]["models"]["grok"]
    assert model["rating"] is None
    assert model["narrative"] == "The records support a knockout study."
    assert report["subjects"][0]["agreement"] == "no rating"
    # With no rating the combined value is the record score alone.
    assert report["subjects"][0]["combined"] == report["subjects"][0]["record_share"]


def test_research_without_an_api_key_stops_before_download(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "missing-key"

    def fetch(_option):
        raise AssertionError("download")

    captured = _run_cure(
        tmp_path,
        monkeypatch,
        capsys,
        ["--structures-json", str(records), "--all", "--confirm-inference", "--research", "grok"],
        out_dir=out,
        fetch=fetch,
    )
    code = captured.code
    assert code == 2
    assert "Target research failed:" in captured.err
    assert "XAI_API_KEY" in captured.err
    assert "sk-" not in captured.err
    assert not out.exists()


def test_research_with_no_uniprot_id_does_not_call_a_model(tmp_path, monkeypatch, capsys):
    block_live_calls(monkeypatch)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    catalog = asthma_catalog(tmp_path)
    records = structures_json(
        tmp_path,
        {
            "group": "protein",
            "accession": "ACC-1ABC",
            "entry_id": "1ABC",
            "chain": "A",
            "title": "plain",
            "predicted": False,
            "source": "fixture",
        },
    )
    out = tmp_path / "no-accession"

    def complete(*_args):
        raise AssertionError("model")

    code = cure_main(
        [
            "--catalog",
            str(catalog),
            "--disease",
            "Asthma",
            "--accept-only",
            "--structures-json",
            str(records),
            "--all",
            "--confirm-inference",
            "--research",
            "grok",
            "--out-dir",
            str(out),
        ],
        fetch=lambda _option: b"data_1ABC\n#\n",
        complete=complete,
    )
    captured = capsys.readouterr().out
    assert code == 0
    assert "No UniProt accession or gene symbol was in the displayed list." in captured
    assert REPORT_NOTE in captured
    for claim in ("probability of cure", "cure the disease", "not a cure", "clinical result"):
        assert claim not in captured.casefold()
    report = json.loads((out / "target_research.json").read_text(encoding="utf-8"))
    assert report["subjects"] == []


def test_the_panel_offers_model_research_after_the_structure_list(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "sk-should-not-appear")
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533)
    out = tmp_path / "panel-research"
    pages: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False, confirm_inference=["yes"])
        if 'type="checkbox" name="pick"' in page:
            shown = sum(1 for seen in pages if 'type="checkbox" name="pick"' in seen)
            assert "<h2>LLM search</h2>" in page
            assert 'name="model" value="grok|' in page
            if shown == 1:
                return {"all": ["yes"], "model": ["grok|grok-4.7"], "next": ["missing"]}
            return {"all": ["yes"]}
        if _is_generation_page(page):
            return {"accept": ["yes"]}
        raise AssertionError(f"no page follows the structures page: {page[:200]}")

    def read(url: str):
        if "esearch" in url:
            return {"esearchresult": {"idlist": []}}
        if "string-db.org" in url:
            return []
        return {}

    model_calls = {"n": 0}

    def complete(provider: str, _system: str, _user: str) -> str:
        assert provider == "grok"
        model_calls["n"] += 1
        return "No record stated a mechanism."

    code = _run_panel(
        monkeypatch,
        exchange,
        runner=lambda _argv: 0,  # Continue starts generation now.
        fetch=lambda _option: b"data_1EGFR\n#\n",
        read_json=read,
        complete=complete,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert len(pages) == 4, "landing, structures, structures again after the search, generation"
    assert 'type="checkbox" name="pick"' in pages[1]
    assert 'name="model" value="grok|' in pages[1], "the search is ticked on that page"
    assert 'name="research"' not in pages[1], "provider boxes are gone"
    assert not any("Program search and LLM options" in page for page in pages)
    # The search button asked once. Download does not ask again and writes no report.
    assert model_calls["n"] == 1
    assert "First search with grok." in captured.out
    assert "Target research" not in captured.out
    assert "Compared with the program search:" not in captured.out
    assert "sk-should-not-appear" not in captured.out
    assert not (out / "target_research.json").exists()


def test_program_search_options_are_compared_with_the_llm_options():
    """A structure only the model named stays visible, and it does not enter the program list."""
    program = [_option("protein", "1EGFR"), _option("protein", "9ZZZ", predicted=True)]
    named = [program[0], _option("protein", "2IL13")]
    compared = compare_search_options(program, named, confirmed=[program[0]])
    assert [item["pdb_id"] for item in compared["confirmed"]] == ["1EGFR"]
    assert [item["pdb_id"] for item in compared["in_both"]] == ["1EGFR"]
    assert [item["pdb_id"] for item in compared["only_program"]] == ["9ZZZ"]
    assert [item["pdb_id"] for item in compared["only_llm"]] == ["2IL13"]
    text = format_option_comparison(compared)
    assert "Confirmed program options: PDB 1EGFR chain A." in text
    assert "In both: PDB 1EGFR chain A." in text
    assert "Only in the program search: PDB 9ZZZ chain A." in text
    assert "Only in the LLM options: PDB 2IL13 chain A." in text


def test_the_structure_page_can_drop_a_kind_and_saves_the_returned_list(tmp_path):
    """Sort is a view. Include and exclude decide which numbered rows can download."""
    rows = [
        _option("protein", "1egfr"),
        _option("protein", "1il13", predicted=True),
        _option("gene", "9unk", methodology_known=False),
    ]
    assert apply_structure_view(rows, take_all=True, picks=["1", "2", "3"]) == (True, ["1", "2", "3"])
    assert apply_structure_view(
        rows,
        take_all=True,
        picks=["1", "2", "3"],
        experimental="exclude",
    ) == (False, ["2"])
    assert apply_structure_view(
        rows,
        take_all=True,
        picks=["1", "2", "nope"],
        predicted="exclude",
    ) == (True, ["1", "nope"])
    path = save_returned_structures(tmp_path, rows[:2], {"acc-1egfr": (3, 8)})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "returned_structures.json"
    assert saved["version"] == 1
    assert [row["entry_id"] for row in saved["structures"]] == ["1egfr", "1il13"]
    assert saved["structures"][0]["program_score"] == 3
    assert saved["structures"][0]["sources_asked"] == 8
    assert saved["structures"][0]["kind"] == "experimental"
    assert saved["structures"][1]["kind"] == "predicted"
    assert saved["structures"][1]["program_score"] is None
    assert campaign_pick_numbers(rows, ["Proteins"]) == ["1", "2", "3"]
    assert apply_structure_view(
        rows,
        take_all=False,
        picks=campaign_pick_numbers(rows, ["Proteins"]),
        predicted="exclude",
    ) == (False, ["1", "3"])


def test_a_campaign_class_tick_selects_those_structures():
    """Clicking a design class selects every listed structure in that class."""
    protein = _option("protein", "1egfr")
    nucleic = StructureOption(
        group="dna",
        accession="ACC-1dna",
        entry_id="1dna",
        chain="A",
        title="nucleic",
        predicted=False,
        source="fixture",
    )
    antibody = StructureOption(
        group="protein",
        accession="ACC-1ab",
        entry_id="1ab",
        chain="B",
        title="nanobody",
        predicted=True,
        source="fixture",
    )
    sugar = StructureOption(
        group="protein",
        accession="ACC-1gly",
        entry_id="1gly",
        chain="C",
        title="plain",
        predicted=False,
        source="fixture",
        campaigns=("Sugars (Glycans)",),
    )
    rows = [protein, nucleic, antibody, sugar]
    assert campaign_pick_numbers(rows, ["Proteins"]) == ["1", "3"]
    assert campaign_pick_numbers(rows, ["DNA & RNA"]) == ["2"]
    assert campaign_pick_numbers(rows, ["Antibodies & Nanobodies"]) == ["3"]
    assert campaign_pick_numbers(rows, ["Sugars (Glycans)"]) == ["4"]
    assert campaign_pick_numbers(rows, ["Small Molecules", "nope"]) == []
    page: dict[str, str] = {}

    def exchange(html: str) -> dict[str, list[str]]:
        page["html"] = html
        return {"campaign": ["DNA & RNA", "Antibodies & Nanobodies"]}

    _action, take_all, picks, providers = OptionPanel(exchange).choose_structures(rows)
    assert take_all is False
    assert picks == "2,3"
    assert providers == []
    assert "Click a class to select its structures." in page["html"]
    assert 'name="campaign" value="DNA &amp; RNA"' in page["html"]
    assert 'data-campaigns="DNA &amp; RNA"' in page["html"]
    assert 'data-campaigns="Proteins|Antibodies &amp; Nanobodies"' in page["html"]
    assert 'data-campaigns="Sugars (Glycans)"' in page["html"]


def test_llm_research_follows_the_confirmed_program_options(tmp_path, monkeypatch, capsys):
    """The structure page can start the LLM search. Download keeps the ticked row.

    The search button is what asks the model. Download fetches the ticked
    structure and does not ask the model again.
    """
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    catalog = asthma_catalog(tmp_path)
    records = structures_json(tmp_path, ROW_P00533, ROW_P35225)
    out = tmp_path / "confirmed-first"
    pages: list[str] = []
    fetched: list[str] = []

    def exchange(page: str) -> dict[str, list[str]]:
        pages.append(page)
        if 'name="binder_kind"' in page:
            return _panel_landing(catalog, records, out, take_all=False, confirm_inference=["yes"])
        if 'type="checkbox" name="pick"' in page:
            shown = sum(1 for seen in pages if 'type="checkbox" name="pick"' in seen)
            # Both searches run from this page and neither fetches a structure.
            # Only the Download button does, and it has not been pressed yet.
            assert fetched == [], "no structure is fetched while this page is still up"
            assert "1EGFR" in page and "2IL13" in page
            assert "Instances of Structure in different databases 0/8" in page
            assert 'name="all_models" value="yes"' in page
            assert "Grok 4.7</label>" in page
            if shown == 1:
                return {"pick": ["1"], "model": ["grok|grok-4.7"], "next": ["missing"]}
            return {"pick": ["1"]}
        if _is_generation_page(page):
            return {"accept": ["yes"]}
        raise AssertionError(f"no page follows the structures page: {page[:200]}")

    model_calls = {"n": 0}

    def complete(_provider: str, _system: str, _user: str) -> str:
        model_calls["n"] += 1
        return json.dumps(
            {
                "rating": 40,
                "reason": "",
                "ideas": "An idea.",
                "structures": [{"pdb_id": "2IL13", "chain": "B"}],
            }
        )

    code = _run_panel(
        monkeypatch,
        exchange,
        runner=lambda _argv: 0,  # Continue starts generation now.
        fetch=lambda option: fetched.append(option.entry_id) or b"data_X\n#\n",
        read_json=lambda _url: {},
        complete=complete,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert len(pages) == 4, "landing, structures, structures again after the search, generation"
    assert 'type="checkbox" name="pick"' in pages[1]
    assert 'name="all_models" value="yes"' in pages[1]
    assert "Grok 4.7</label>" in pages[1]
    assert "Search with every API key" not in pages[1]
    assert not any("Program search and LLM options" in page for page in pages)
    assert not any("Combined results" in page for page in pages), "that page scored structures"
    assert model_calls["n"] == 1, "the search button asked once; Download did not ask again"
    assert fetched == ["1EGFR"], "the confirmed program option is downloaded, not the other row"
    assert "Compared with the program search:" not in captured.out
    assert "Target research" not in captured.out
    assert not (out / "protein_2IL13_B.cif").exists()
    assert "Saved 2 returned structure(s)" in captured.out
    saved = json.loads((out / "returned_structures.json").read_text(encoding="utf-8"))
    assert [row["entry_id"] for row in saved["structures"]] == ["1EGFR", "2IL13"]
    assert saved["structures"][0]["program_score"] == 0
    assert not (out / "target_research.json").exists()


def test_a_selected_model_is_the_model_id_that_is_sent(monkeypatch):
    """Selecting a model must change the request, not only the label in the report."""
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("RFD4_GROK_MODEL", raising=False)
    monkeypatch.delenv("RFD4_CLAUDE_MODEL", raising=False)
    sent: list[dict] = []

    def fake(request, timeout=0):
        sent.append(json.loads(request.data.decode("utf-8")))
        if "anthropic" in request.full_url:
            return _JsonResponse({"content": [{"text": '{"rating": 61}'}]})
        return _JsonResponse({"choices": [{"message": {"content": '{"rating": 61}'}}]})

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)

    grok_ids = [model_id for model_id, _label in model_choices("grok")]
    assert len(grok_ids) > 1, "the fixture needs a provider with more than one listed model"
    complete_chat("grok", "sys", "user", None, {"grok": grok_ids[1]})
    assert sent[-1]["model"] == grok_ids[1], "the chosen model, not the default"

    # With nothing chosen, the model sent is one the panel actually offers.
    complete_chat("grok", "sys", "user", None, None)
    assert sent[-1]["model"] == grok_ids[0]
    for name in PROVIDERS:
        offered = [model_id for model_id, _label in model_choices(name)]
        assert resolve_chat_model(name) in offered, f"{name} would ask for an unlisted model"

    # Each provider carries its own choice.
    claude_ids = [model_id for model_id, _label in model_choices("claude")]
    complete_chat("claude", "sys", "user", None, {"claude": claude_ids[-1]})
    assert sent[-1]["model"] == claude_ids[-1]

    with pytest.raises(ValueError, match="Unknown grok model"):
        complete_chat("grok", "sys", "user", None, {"grok": "grok-9000"})


class _Response:
    """What a faked urlopen hands back: ``read()`` gives ``body``; ``status`` exists only when given."""

    def __init__(self, body: bytes = b"", status: int | None = None) -> None:
        self._raw = body
        if status is not None:
            self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        return False


class _JsonResponse(_Response):
    def __init__(self, payload: dict):
        super().__init__(json.dumps(payload).encode())


class _StreamResponse:
    """A server-sent-event body, handed over one line at a time.

    The real endpoint is read line by line so a long reasoning trace does not
    sit in one blocking read, so the stub has to be read the same way.
    """

    def __init__(self, events: list[dict | str]):
        lines: list[bytes] = []
        for event in events:
            body = event if isinstance(event, str) else json.dumps(event)
            lines.append(b"data: " + body.encode() + b"\n")
            lines.append(b"\n")
        self._lines = lines
        self._next = 0

    def readline(self) -> bytes:
        if self._next >= len(self._lines):
            return b""
        line = self._lines[self._next]
        self._next += 1
        return line

    def read(self) -> bytes:
        raise AssertionError("a stream must be read line by line, not in one blocking read")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        return False


def _delta(content: str | None = None, reasoning: str | None = None, finish: str | None = None) -> dict:
    delta: dict[str, str] = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


def test_a_reasoning_id_is_sent_the_documented_sampling_and_allowance(monkeypatch):
    # NVIDIA's own example for deepseek-ai/deepseek-v4.1-flash sends temperature 1
    # with top_p 0.95 and a very large output allowance. The allowance matters most:
    # the reasoning trace is paid for out of it, so a chat-sized budget is spent
    # thinking and the answer comes back empty.
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("RFD4_NVIDIA_MODEL", "deepseek-ai/deepseek-v4.1-flash")
    sent: dict[str, object] = {}

    def fake(request, timeout=0):
        sent["body"] = json.loads(request.data.decode())
        return _StreamResponse([_delta(content="ok", finish="stop"), "[DONE]"])

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    assert complete_chat("nvidia", "system", "user") == "ok"
    body = sent["body"]
    assert body["model"] == "deepseek-ai/deepseek-v4.1-flash"
    assert body["temperature"] == 1.0
    assert body["top_p"] == 0.95
    assert body["max_tokens"] == REASONING_MAX_OUTPUT > CHAT_MAX_OUTPUT
    assert body["stream"] is True


def test_a_plain_chat_id_keeps_the_small_allowance_and_no_sampling(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RFD4_OPENAI_MODEL", "gpt-4.1-mini")
    sent: dict[str, object] = {}

    def fake(request, timeout=0):
        sent["body"] = json.loads(request.data.decode())
        return _StreamResponse([_delta(content="ok", finish="stop"), "[DONE]"])

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    assert complete_chat("openai", "system", "user") == "ok"
    body = sent["body"]
    assert body["max_tokens"] == CHAT_MAX_OUTPUT
    assert "temperature" not in body
    assert "top_p" not in body


def test_a_streamed_answer_is_joined_and_the_reasoning_trace_is_left_out(monkeypatch):
    # The trace arrives as reasoning_content. Folding it into the answer would
    # break the JSON the caller parses out of that answer.
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("RFD4_NVIDIA_MODEL", "deepseek-ai/deepseek-v4.1-flash")
    events = [
        _delta(reasoning="The list already holds 1IJZ, so "),
        _delta(content='{"added": '),
        _delta(reasoning="name another entry."),
        _delta(content='["1ALU"]}'),
        _delta(finish="stop"),
        "[DONE]",
    ]
    monkeypatch.setattr("rfproteina.disease.research.urlopen", lambda request, timeout=0: _StreamResponse(events))
    answer = complete_chat("nvidia", "system", "user")
    assert answer == '{"added": ["1ALU"]}'
    assert "name another entry" not in answer


def test_an_answer_that_never_arrives_names_the_spent_allowance(monkeypatch):
    # A trace that fills the whole allowance leaves no answer. Reporting that as
    # "returned no text" reads like the model having found nothing, which is the
    # one thing a search must not say when it never got an answer at all.
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("RFD4_NVIDIA_MODEL", "deepseek-ai/deepseek-v4.1-flash")
    events = [_delta(reasoning="thinking " * 50), _delta(finish="length"), "[DONE]"]
    monkeypatch.setattr("rfproteina.disease.research.urlopen", lambda request, timeout=0: _StreamResponse(events))
    with pytest.raises(ValueError) as caught:
        complete_chat("nvidia", "system", "user")
    message = str(caught.value)
    assert "output allowance" in message
    assert str(REASONING_MAX_OUTPUT) in message


def test_nvidia_nemotron_is_sent_the_documented_switch_seed_and_allowance(monkeypatch):
    # NVIDIA's Nemotron example passes chat_template_kwargs through the OpenAI
    # client's extra_body, which puts it at the top level of the JSON body, and
    # sends a seed with it. The switch is off by default: these prompts ask for one
    # small JSON object, and spending the allowance on a trace is how an answer
    # fails to arrive at all.
    monkeypatch.delenv("RFD4_CHAT_THINKING", raising=False)
    monkeypatch.delenv("RFD4_CHAT_SEED", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("RFD4_NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
    sent: dict[str, object] = {}

    def fake(request, timeout=0):
        sent["body"] = json.loads(request.data.decode())
        return _StreamResponse([_delta(content="ok", finish="stop"), "[DONE]"])

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    assert complete_chat("nvidia", "system", "user") == "ok"
    body = sent["body"]
    assert body["temperature"] == 1.0
    assert body["top_p"] == 0.95
    assert body["max_tokens"] == REASONING_MAX_OUTPUT == 16384
    assert body["seed"] == CHAT_SEED
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["stream"] is True


def test_the_reasoning_switch_can_be_turned_back_on(monkeypatch):
    monkeypatch.setenv("RFD4_CHAT_THINKING", "on")
    monkeypatch.setenv("RFD4_CHAT_SEED", "7")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("RFD4_NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
    sent: dict[str, object] = {}

    def fake(request, timeout=0):
        sent["body"] = json.loads(request.data.decode())
        return _StreamResponse([_delta(content="ok", finish="stop"), "[DONE]"])

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    assert complete_chat("nvidia", "system", "user") == "ok"
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert sent["body"]["seed"] == 7


@pytest.mark.parametrize(
    ("provider", "env_name", "model", "switch", "absent"),
    [
        # Each host spells the switch its own way, and the OpenRouter build of
        # Nemotron is a separate id, so the two never substitute for each other.
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            {"reasoning": {"enabled": False}},
            "chat_template_kwargs",
        ),
        # NVIDIA's DeepSeek example sends no switch at all. An unknown field is how
        # a working call breaks, so it is kept to the ids that document it.
        (
            "nvidia",
            "NVIDIA_API_KEY",
            "deepseek-ai/deepseek-v4.1-flash",
            {},
            "chat_template_kwargs",
        ),
    ],
)
def test_each_host_gets_its_own_reasoning_switch(monkeypatch, provider, env_name, model, switch, absent):
    monkeypatch.delenv("RFD4_CHAT_THINKING", raising=False)
    monkeypatch.setenv(env_name, "test-key")
    monkeypatch.setenv(f"RFD4_{provider.upper()}_MODEL", model)
    sent: dict[str, object] = {}

    def fake(request, timeout=0):
        sent["body"] = json.loads(request.data.decode())
        return _StreamResponse([_delta(content="ok", finish="stop"), "[DONE]"])

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    assert complete_chat(provider, "system", "user") == "ok"
    body = sent["body"]
    assert body["model"] == model, "a :free suffix is still the id the user chose"
    assert absent not in body
    for field, value in switch.items():
        assert body[field] == value
    if not switch:
        assert "reasoning" not in body
    # The sampling, seed and allowance are the model's own, so they still travel.
    assert body["temperature"] == 1.0
    assert body["seed"] == CHAT_SEED
    assert body["max_tokens"] == REASONING_MAX_OUTPUT


def test_openrouter_reasoning_follows_the_same_switch(monkeypatch):
    monkeypatch.setenv("RFD4_CHAT_THINKING", "on")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("RFD4_OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    sent: dict[str, object] = {}

    def fake(request, timeout=0):
        sent["body"] = json.loads(request.data.decode())
        return _StreamResponse([_delta(content="ok", finish="stop"), "[DONE]"])

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    assert complete_chat("openrouter", "system", "user") == "ok"
    assert sent["body"]["reasoning"] == {"enabled": True}


def test_an_openrouter_route_suffix_is_accepted_on_a_listed_id(monkeypatch):
    # OpenRouter publishes the routing tier as a suffix on the id. It picks how the
    # request is routed, not a different model, so a listed id carrying one needs no
    # entry of its own and must not be refused as unknown.
    monkeypatch.delenv("RFD4_CHAT_THINKING", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    chosen = "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert resolve_chat_model("openrouter", {"openrouter": chosen}) == chosen
    assert base_model(chosen) == "nvidia/nemotron-3-ultra-550b-a55b"
    assert is_offered_model("openrouter", chosen)
    sent: dict[str, object] = {}

    def fake(request, timeout=0):
        sent["body"] = json.loads(request.data.decode())
        return _StreamResponse([_delta(content="ok", finish="stop"), "[DONE]"])

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    assert complete_chat("openrouter", "system", "user", chat_models={"openrouter": chosen}) == "ok"
    body = sent["body"]
    assert body["model"] == chosen, "the suffix travels; it is part of the id"
    # The suffix must not hide the family the request shape is chosen from.
    assert body["max_tokens"] == REASONING_MAX_OUTPUT
    assert body["reasoning"] == {"enabled": False}


def test_an_unlisted_model_is_still_refused_with_a_suffix(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(ValueError, match="Unknown openrouter model"):
        resolve_chat_model("openrouter", {"openrouter": "someone/not-offered:free"})
    # A suffix that is not a routing tier is not stripped, so it is not a listed id.
    assert not is_offered_model("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:v9")
    assert base_model("nvidia/nemotron-3-ultra-550b-a55b:v9") == "nvidia/nemotron-3-ultra-550b-a55b:v9"


@pytest.mark.parametrize(
    "pasted",
    [
        # What a mis-paste actually looks like: a line of the documented curl, or a
        # sentence. Neither is a key, and both were being sent as a bearer token.
        "curl -N https://openrouter.ai/api/v1/chat/completions",
        '-H "Authorization: Bearer $OPENROUTER_API_KEY"',
        "my openrouter key is saved somewhere else",
    ],
)
def test_a_mis_pasted_key_is_named_before_the_call(monkeypatch, pasted):
    # Sending this earns "Missing Authentication header" from the service, which
    # reads like a rejected key, and the search then reports finding nothing.
    monkeypatch.setattr(
        "rfproteina.disease.research.urlopen",
        lambda request, timeout=0: pytest.fail("a mis-pasted key must not be sent"),
    )
    with pytest.raises(ValueError) as caught:
        complete_chat("openrouter", "system", "user", api_keys={"openrouter": pasted})
    # The message is fixed text naming the provider and the fault, so nothing that
    # was pasted can reach a log, a report or the terminal through it.
    assert str(caught.value) == (
        "The saved openrouter key contains whitespace, so it is not an API key. "
        "Re-enter the openrouter key on the first page."
    )
    assert pasted not in str(caught.value)


def test_a_real_looking_key_passes_the_shape_check(monkeypatch):
    monkeypatch.delenv("RFD4_CHAT_THINKING", raising=False)
    key = "sk-or-v1-" + "0123456789abcdef" * 4
    monkeypatch.setattr(
        "rfproteina.disease.research.urlopen",
        lambda request, timeout=0: _StreamResponse([_delta(content="ok", finish="stop"), "[DONE]"]),
    )
    assert complete_chat("openrouter", "system", "user", api_keys={"openrouter": key}) == "ok"
    assert checked_key("nvidia", "nvapi-" + "a" * 64).startswith("nvapi-")


def test_a_chunk_with_no_choices_is_skipped(monkeypatch):
    # NVIDIA's own example guards against this: "if not chunk.choices: continue".
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("RFD4_NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
    events = [{"choices": []}, {"id": "no-choices-key"}, _delta(content="ok", finish="stop"), "[DONE]"]
    monkeypatch.setattr("rfproteina.disease.research.urlopen", lambda request, timeout=0: _StreamResponse(events))
    assert complete_chat("nvidia", "system", "user") == "ok"


def test_a_service_that_ignores_the_stream_request_is_still_read(monkeypatch):
    # Not every OpenAI-compatible host honours stream. One JSON object must still
    # be read rather than treated as an empty answer.
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("RFD4_NVIDIA_MODEL", "deepseek-ai/deepseek-v4.1-flash")
    payload = {"choices": [{"message": {"content": "one piece"}, "finish_reason": "stop"}]}
    monkeypatch.setattr("rfproteina.disease.research.urlopen", lambda request, timeout=0: _JsonResponse(payload))
    assert complete_chat("nvidia", "system", "user") == "one piece"


@pytest.mark.parametrize(
    ("provider", "env_name", "host"),
    [
        ("grok", "XAI_API_KEY", "api.x.ai"),
        ("claude", "ANTHROPIC_API_KEY", "api.anthropic.com"),
        ("openai", "OPENAI_API_KEY", "api.openai.com"),
        ("nvidia", "NVIDIA_API_KEY", "integrate.api.nvidia.com"),
        ("openrouter", "OPENROUTER_API_KEY", "openrouter.ai"),
    ],
)
def test_research_providers_send_the_key_only_as_a_header(monkeypatch, provider, env_name, host):
    monkeypatch.setenv(env_name, "test-key")
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    seen: dict[str, object] = {}

    def fake(request, timeout=0):
        seen["url"] = request.full_url
        seen["headers"] = {key.casefold(): value for key, value in request.header_items()}
        seen["body"] = request.data.decode()
        if "anthropic" in request.full_url:
            payload = {"content": [{"type": "text", "text": "Receptor signaling."}]}
        else:
            payload = {"choices": [{"message": {"content": "Receptor signaling."}}]}
        return _JsonResponse(payload)

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    assert complete_chat(provider, "system", "user") == "Receptor signaling."
    assert host in seen["url"]
    assert "test-key" not in seen["body"]
    headers = seen["headers"]
    body = json.loads(seen["body"])
    if provider == "claude":
        assert headers["x-api-key"] == "test-key"
        assert headers["anthropic-version"] == "2023-06-01"
    else:
        assert headers["authorization"] == "Bearer test-key"
        assert "test-key" not in body["messages"][1]["content"]
    # With nothing chosen, the request carries a model the panel actually offers. A
    # built-in default outside that list is how a fresh install asks for a retired id.
    offered = [model_id for model_id, _label in model_choices(provider)]
    assert body["model"] in offered, f"{provider} sent a model it does not offer"
    if provider == "openrouter":
        assert headers["x-title"] == "Humanitys-Last-Disease"


def test_a_landing_page_model_is_sent_and_an_environment_model_still_wins(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.delenv("RFD4_NVIDIA_MODEL", raising=False)
    seen: dict[str, object] = {}

    def fake(request, timeout=0):
        seen["body"] = request.data.decode()
        return _JsonResponse({"choices": [{"message": {"content": "Receptor signaling."}}]})

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    chosen = {"nvidia": "z-ai/glm-5.3-flash"}
    assert complete_chat("nvidia", "system", "user", chat_models=chosen) == "Receptor signaling."
    assert json.loads(seen["body"])["model"] == "z-ai/glm-5.3-flash"
    monkeypatch.setenv("RFD4_NVIDIA_MODEL", "custom-nvidia-model")
    assert complete_chat("nvidia", "system", "user", chat_models=chosen)
    assert json.loads(seen["body"])["model"] == "z-ai/glm-5.3-flash"
    assert complete_chat("nvidia", "system", "user")
    assert json.loads(seen["body"])["model"] == "custom-nvidia-model"
    save_model_choices({"nvidia": "deepseek-ai/deepseek-v4.1-flash"})
    assert complete_chat("nvidia", "system", "user")
    assert json.loads(seen["body"])["model"] == "custom-nvidia-model"
    monkeypatch.delenv("RFD4_NVIDIA_MODEL", raising=False)
    assert resolve_chat_model("nvidia") == "deepseek-ai/deepseek-v4.1-flash"
    assert complete_chat("nvidia", "system", "user")
    assert json.loads(seen["body"])["model"] == "deepseek-ai/deepseek-v4.1-flash"
    assert "test-key" not in seen["body"]

    def refuse(_request, timeout=0):
        raise AssertionError("the request was sent")

    monkeypatch.setattr("rfproteina.disease.research.urlopen", refuse)
    with pytest.raises(ValueError, match="Unknown nvidia model"):
        complete_chat("nvidia", "system", "user", chat_models={"nvidia": "not-a-model"})


def test_a_chosen_model_is_saved_apart_from_the_key_and_stays_out_of_the_command(capsys):
    secret = "nv-panel-secret"

    def exchange(page: str) -> dict[str, list[str]]:
        assert 'value="z-ai/glm-5.3-flash"' in page
        assert 'value="claude-opus-5"' in page
        return {
            "disease": ["Asthma"],
            "binder_kind": ["protein"],
            "where": ["local"],
            "nvidia_key": [secret],
            "nvidia_model": ["z-ai/glm-5.3-flash"],
            "openrouter_model": ["nvidia/nemotron-3.5-super-vl"],
        }

    panel = OptionPanel(exchange)
    argv = panel.collect(["--ui"], build_parser())
    assert "z-ai/glm-5.3-flash" not in argv
    assert "nvidia/nemotron-3.5-super-vl" not in argv
    assert "--research" not in argv
    assert secret not in " ".join(argv)
    assert panel.models() == {
        "nvidia": "z-ai/glm-5.3-flash",
        "openrouter": "nvidia/nemotron-3.5-super-vl",
    }
    assert panel.api_keys() == {"nvidia": secret}
    assert load_model_choices() == panel.models()
    stored = (key_directory() / "models.json").read_text(encoding="utf-8")
    assert secret not in stored
    assert "z-ai/glm-5.3-flash" in stored
    cipher = (key_directory() / "api_keys.json").read_text(encoding="utf-8")
    assert "z-ai/glm-5.3-flash" not in cipher
    assert secret not in cipher
    assert (key_directory() / "models.json").stat().st_mode & 0o777 == 0o600
    captured = capsys.readouterr()
    assert "Saved model choices for the next run." in captured.out
    assert secret not in captured.out

    def again(page: str) -> dict[str, list[str]]:
        assert 'value="z-ai/glm-5.3-flash" selected' in page
        return {"disease": ["Asthma"], "binder_kind": ["protein"], "where": ["local"], "nvidia_model": [""]}

    kept = OptionPanel(again)
    kept.collect(["--ui"], build_parser())
    assert kept.models() == {}
    assert load_model_choices()["nvidia"] == "z-ai/glm-5.3-flash"

    def bad(_page: str) -> dict[str, list[str]]:
        return {"disease": ["Asthma"], "binder_kind": ["protein"], "nvidia_model": ["not-a-model"]}

    with pytest.raises(ValueError, match="Unknown NVIDIA model"):
        OptionPanel(bad).collect(["--ui"], build_parser())
    assert load_model_choices()["nvidia"] == "z-ai/glm-5.3-flash"


def test_a_provider_error_does_not_keep_the_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-secret")

    def fake(_request, timeout=0):
        raise OSError("failed with sk-live-secret")

    monkeypatch.setattr("rfproteina.disease.research.urlopen", fake)
    with pytest.raises(ValueError, match="Could not reach") as caught:
        complete_chat("openai", "system", "user")
    assert "sk-live-secret" not in str(caught.value)
    assert "[redacted]" in str(caught.value)


# ---------------------------------------------------------------------------
# The panel's round trip, and grok_bot's own seams. Each of the behaviours
# below was reproduced as a defect before it was fixed; these are what notice
# if it comes back.
# ---------------------------------------------------------------------------


def test_a_screening_flag_survives_the_panel_rebuilding_the_command():
    """The panel has no control for these six, so they must ride along.

    ``--ui`` makes the panel rebuild the whole command from its own form. The
    form had no field for any of these, so every one was dropped in silence:
    the screen then ran with the defaults -- ``lookup_targets`` false leaves the
    paralog and plasma axes reporting ``network_not_permitted`` -- while the
    recorded request said that was what the user asked for.
    """
    from rfproteina.disease.cli import build_parser
    from rfproteina.disease.ui import _form_from_namespace, to_argv

    argv = [
        "--ui",
        "--disease",
        "asthma",
        "--proteome",
        "/tmp/human.fasta",
        "--lookup-targets",
        "--plasma-abundance-max",
        "1e9",
        "--plasma-hub-max",
        "3",
        "--plasma-fold-max",
        "2.0",
        "--follow-up",
    ]
    parser = build_parser()
    asked = parser.parse_args(argv)
    rebuilt = to_argv(_form_from_namespace(asked))
    for flag in (
        "--proteome",
        "--lookup-targets",
        "--plasma-abundance-max",
        "--plasma-hub-max",
        "--plasma-fold-max",
        "--follow-up",
    ):
        assert flag in rebuilt, f"{flag} was dropped when the panel rebuilt the command"
    again = parser.parse_args(rebuilt)
    assert str(again.proteome) == str(asked.proteome)
    assert again.lookup_targets is True
    assert again.plasma_abundance_max == 1e9
    assert again.plasma_hub_max == 3
    assert again.plasma_fold_max == 2.0
    assert again.follow_up is True


def test_a_screening_flag_is_carried_through_a_submission():
    """The page keeps them as hidden inputs, so a submit does not lose them either."""
    page = render_options(
        OptionForm(
            proteome="/tmp/human.fasta",
            lookup_targets=True,
            plasma_hub_max=3,
            plasma_fold_max=2.0,
            follow_up=True,
        )
    )
    assert 'name="lookup_targets"' in page
    assert 'name="proteome"' in page
    fields = {
        "binder_kind": ["protein"],
        "out_dir": ["disease_runs"],
        "proteome": ["/tmp/human.fasta"],
        "lookup_targets": ["yes"],
        "plasma_hub_max": ["3"],
        "plasma_fold_max": ["2.0"],
        "follow_up": ["yes"],
    }
    read = form_from_fields(fields)
    assert read.proteome == "/tmp/human.fasta"
    assert read.lookup_targets is True
    assert read.plasma_hub_max == 3
    assert read.plasma_fold_max == 2.0
    assert read.follow_up is True


def test_a_refused_submission_keeps_the_worker_cap_the_user_chose():
    """A redraw must not quietly uncap a run.

    The cap lives in the select; only ``custom`` puts a number in the box
    beside it. Reading the box alone turned a chosen 8 into "every local
    request at once" on any redraw -- a refusal about another field, or a click
    on Search files -- and Continue then started an uncapped run.
    """
    fields = {
        "binder_kind": ["protein"],
        "out_dir": ["disease_runs"],
        "local_workers": ["8"],
        "local_workers_custom": [""],
        "where": ["remote"],
        "hosts": [""],
    }
    with pytest.raises(ValueError):
        form_from_fields(fields)
    assert draft_from_fields(fields).local_workers == 8
    custom = {**fields, "local_workers": ["custom"], "local_workers_custom": ["5"]}
    assert draft_from_fields(custom).local_workers == 5
    every = {**fields, "local_workers": ["every"], "local_workers_custom": [""]}
    assert draft_from_fields(every).local_workers is None
    # The remote cap is redrawn by the same rule. A cap that survived a redraw
    # on one of the two and not the other is this same defect on whichever one
    # was forgotten.
    remote = {**fields, "remote_workers": ["8"], "remote_workers_custom": [""]}
    assert draft_from_fields(remote).remote_workers == 8
    remote_custom = {**fields, "remote_workers": ["custom"], "remote_workers_custom": ["3"]}
    assert draft_from_fields(remote_custom).remote_workers == 3
    assert draft_from_fields(fields).remote_workers is None


def test_a_network_share_path_is_shown_as_it_was_typed():
    """A UNC path has no local parent to resolve against.

    Resolving it put it under the working directory, and because the field's
    displayed value is what the browser posts back, the rewritten path was then
    adopted as the user's choice -- so run output would be written inside the
    checkout.
    """
    assert display_directory(r"\\nas\designs") == r"\\nas\designs"
    assert display_directory("//nas/designs") == "//nas/designs"
    read = form_from_fields({"binder_kind": ["protein"], "out_dir": [r"\\nas\designs"]})
    assert "ModRFD4" not in read.out_dir, f"the share was relocated into the checkout: {read.out_dir}"


def test_drawing_the_page_does_not_create_a_folder_in_what_it_shows(tmp_path):
    """Looking around the file list must not write into the folders looked at.

    The picker offers a ``parent`` row, so ``..`` is how a user looks around,
    and the archive folder was created on every render: each look left a
    ``generations`` directory behind, including above the checkout.
    """
    outside = tmp_path / "somebody_elses_folder"
    outside.mkdir()
    render_options(OptionForm(out_dir=str(outside)))
    assert list(outside.iterdir()) == [], "rendering the page created a directory"
    moved, _selected = choose_output_entry(OptionForm(out_dir=str(outside)), str(tmp_path))
    render_options(moved)
    assert not (tmp_path / "generations").exists(), "clicking .. created a directory"


def test_the_listing_notice_is_not_one_of_a_run_s_own_sentences(tmp_path):
    """What the listing collapsed is a fact about the listing, not about a run.

    It rides on the last report's lines, and rendered there it read as one of
    that run's verdict sentences, under that run's label -- and which run
    collected it was whichever happened to be last.
    """
    root = tmp_path / "runs"
    for name in ("gene_8TLD_2", "gene_8TLD_2-2"):
        _write_screen(root / name, _screen_payload())
    page = render_options(OptionForm(out_dir=str(root)))
    assert "Listing:" in page
    assert "collapsed as reading identically" in page
    blocks = page.split('<div class="screen-readout')
    for block in blocks[1:]:
        body = block[: block.index("</div>")]
        if "screen-where" in body:
            assert "Listing:" not in body, f"the notice is inside a run's own block: {body}"
    assert 'class="screen-readout screen-listing"' in page


def test_a_windows_path_is_converted_for_powershell_and_keeps_no_mount(tmp_path):
    """The whole argv is asserted, not just the file name.

    ``endswith("paste-grok-bot.ps1")`` is true of the converted path and of the
    unconverted ``/mnt/a/...`` one alike, so it could not have noticed a
    conversion regression.
    """
    from rfproteina.disease.grok_bot import paste_script, powershell_command, windows_path

    command = powershell_command(paste_script(), "powershell.exe")
    spelled = command[-1]
    assert "/mnt/" not in spelled
    if sys.platform == "win32":
        assert re.match(r"^[A-Za-z]:\\", spelled), spelled
    else:
        # From WSL the checkout is a /mnt/<drive> path, so it must come back as
        # a drive path with backslashes for PowerShell to open it.
        assert re.match(r"^[A-Za-z]:\\", spelled), spelled
    assert spelled.endswith("paste-grok-bot.ps1")
    native = tmp_path / "plain.ps1"
    native.write_text("", encoding="utf-8")
    converted = windows_path(native)
    assert "/mnt/" not in converted or sys.platform == "win32"


def test_a_path_under_a_drive_letter_is_found_from_wsl(tmp_path):
    """``path_exists`` answers for a Windows spelling of a file that is here."""
    from rfproteina.disease.grok_bot import path_exists, windows_path

    target = tmp_path / "shortcut.lnk"
    target.write_text("", encoding="utf-8")
    assert path_exists(str(target))
    assert path_exists(windows_path(target))
    assert not path_exists(str(tmp_path / "absent.lnk"))


def test_a_script_that_cannot_be_started_is_one_message_not_a_traceback(monkeypatch):
    """Both failures of the subprocess call say the same thing to the user."""
    import subprocess

    from rfproteina.disease import grok_bot

    monkeypatch.setattr(grok_bot, "windows_powershell", lambda env=None: __file__)

    def refuse(*_args, **_kwargs):
        raise OSError("no such executable")

    monkeypatch.setattr(subprocess, "run", refuse)
    with pytest.raises(ValueError, match="could not be pasted"):
        grok_bot._run_script(Path(__file__), "task", 1.0, "The request could not be pasted into Grok Bot.")

    def expire(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="powershell.exe", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", expire)
    with pytest.raises(ValueError, match="did not come back"):
        grok_bot._run_script(Path(__file__), "task", 1.0, "Grok Bot's reply did not come back.")


def test_a_task_over_the_paste_limit_is_not_sent(monkeypatch):
    """Too long to paste is refused here, before any window is touched."""
    from rfproteina.disease import grok_bot

    def fail(*_args, **_kwargs):
        raise AssertionError("an over-long task must not reach the script")

    monkeypatch.setattr(grok_bot, "_run_script", fail)
    assert grok_bot._paste_task("x" * (grok_bot.PASTE_MAX + 1)) is False
    assert grok_bot._paste_task("") is False


def test_a_reply_larger_than_the_cap_is_refused():
    """A megabyte of output is not a reply, and is not parsed looking for one."""
    from rfproteina.disease.grok_bot import STDOUT_MAX, _reply_text

    with pytest.raises(ValueError, match="did not come back"):
        _reply_text(b"x" * (STDOUT_MAX + 1))


def test_the_paste_timeout_is_above_the_script_s_own_budget():
    """The outer cap is a backstop, not a limit on the script doing its work.

    Set below the script's own fixed waits it fired during a slow but
    succeeding run, and a terminated script skips the cleanup that unpins the
    window -- so the window was left on top of everything while the caller
    reported that the paste had failed.
    """
    from rfproteina.disease.grok_bot import PASTE_BUDGET_S, PASTE_TIMEOUT_S

    assert PASTE_TIMEOUT_S > PASTE_BUDGET_S


def test_a_truncated_response_body_is_one_source_down_not_a_dead_run(monkeypatch):
    """``IncompleteRead`` is neither an OSError nor a ValueError.

    A body shorter than its own Content-Length raises it, and uncaught it went
    straight past every ``except (OSError, ValueError)`` between the reader and
    ``main``, ending the whole research run with a traceback. The new module
    already named ``HTTPException`` in its unavailable set; the shared reader
    the older call sites use did not.
    """
    from http.client import IncompleteRead

    from rfproteina.disease import research

    def truncated(*_args, **_kwargs):
        raise IncompleteRead(b'{"part"', 437)

    monkeypatch.setattr(research, "urlopen", truncated)
    with pytest.raises(ValueError, match="Could not read JSON"):
        research.read_public_json("https://example.invalid/thing.json")
