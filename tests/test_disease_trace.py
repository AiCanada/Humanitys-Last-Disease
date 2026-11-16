"""One research trace per generated binder, with nothing invented."""

import json
from pathlib import Path

import pytest

from rfproteina.disease.focus import BiologicalFocus
from rfproteina.disease.research import SourceHit
from rfproteina.disease.trace import (
    NOT_PERFORMED,
    NOT_RECORDED,
    search_trace,
    write_research_traces,
)
from tests.helpers.disease import structure_option

PEPTIDE = "ACDEFGHIKLMNPQRSTVWY" * 2
RELEASE_REVISION = "a06e2509c7033b010fc43614e9fd840303588502"


def _focus() -> BiologicalFocus:
    return BiologicalFocus(
        pathway_id="R-HSA-9",
        pathway_name="Chosen pathway",
        mechanism="pathway",
        target=PEPTIDE,
        target_accession="P35225",
        target_symbol="IL13",
    )


def _config(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                "hf_ckpt_path: nvidia/RFD4-Proteina",
                "hf_ckpt_revision: abcdef",
                "ckpt_name: release.ckpt",
                "ckpt_path: ${oc.env:DATA_PATH}/flow_checkpoints",
                "autoencoder_ckpt_path: ${oc.env:DATA_PATH}/ae_checkpoints/ae.ckpt",
            )
        ),
        encoding="utf-8",
    )
    return path


def _keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(str(key) for key in value)
        for item in value.values():
            found.update(_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_keys(item))
    return found


def _status(rows: list[dict[str, str]], name: str, role: str) -> str:
    found = [row["status"] for row in rows if row["name"] == name and row["role"] == role]
    assert found, (name, role)
    return found[0]


def test_a_binder_trace_records_the_run_and_leaves_out_a_sequence(tmp_path: Path) -> None:
    """Known inputs are stored. A pasted sequence, a secret, and a home path are not."""
    output = tmp_path / "protein_1ABC_1"
    sample = output / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text("data_binder\n#\n", encoding="utf-8")
    metrics = output / "inference_metrics"
    metrics.mkdir()
    secret = "sk-" + "a" * 24
    (metrics / "per_example_metrics.csv").write_text(
        "example_id,ckpt_name,ckpt_path,seed,oracle_ppi.plddt_coi,generation.n_recycle,generation.args.nsteps\n"
        f"binder_0,release.ckpt,/home/someone/models/{secret},7,81.5,3,60\n",
        encoding="utf-8",
    )
    (output.parent / "protein_1ABC_1.yaml").write_text(
        "# No hotspot is set because the deposit does not name an interface.\n"
        'binder:\n  contig: "60-110, /0, A1-9"\n',
        encoding="utf-8",
    )
    screen = {
        "call": "screened_clear",
        "bar": "sequence",
        "untested": [],
        "hits": [],
        "ppi_pass": "absent",
        "ppi_blocker": "not_requested",
        "pre_oracle_pass": True,
        "requested_class": "requested_clear",
        "samples": [
            {
                "example_id": "binder_0",
                "sequence_pass": True,
                "contact_ms": 1.25,
                "sequence": PEPTIDE,
            }
        ],
    }
    search = search_trace(
        live_structures=True,
        live_disease=False,
        focus=_focus(),
        collected={"P35225": {"uniprot": SourceHit("record", ""), "ncbi": SourceHit("no record", "")}},
    )
    argv = [
        "rfd4-generate",
        "inference_experiment=inference_on_user_inputs",
        "inputs=protein_1ABC_1.yaml",
        f"out_dir={output}",
        "+disease.host=local",
    ]
    written = write_research_traces(
        output,
        disease_name="Asthma",
        disease_id="EFO_0000270",
        disease_query="Asthma",
        focus=_focus(),
        option=structure_option(title=f"see {secret}"),
        binder_kind="protein",
        argv=argv,
        screen=screen,
        search=search,
        config=_config(tmp_path / "inference.yaml"),
    )
    assert [path.name for path in written] == ["binder_0.json"]
    text = written[0].read_text(encoding="utf-8")
    record = json.loads(text)
    assert "sequence" not in _keys(record)
    assert PEPTIDE not in text
    assert secret not in text
    assert "/home/someone" not in text
    assert "DATA_PATH" not in text
    assert "${oc.env" not in text
    assert "[redacted]" in text
    assert "<home>" in text
    assert record["kind"] == "binder"
    assert record["example_id"] == "binder_0"
    assert record["structure_file"] == "inference_samples/binder_0/binder_0_generated.cif"
    disease = record["disease_input"]
    assert disease["query"] == "Asthma"
    assert disease["confirmed_name"] == "Asthma"
    assert disease["disease_id"] == "EFO_0000270"
    pathway = record["pathway"]
    assert pathway["pathway_id"] == "R-HSA-9"
    assert pathway["pathway_name"] == "Chosen pathway"
    assert pathway["mechanism"] == "pathway"
    target = record["selected_target"]
    assert target["accession"] == "P35225"
    assert target["role"] == "Named target"
    assert target["requested_name"] == NOT_RECORDED
    assert target["requested_accession"] == "P35225"
    assert target["requested_symbol"] == "IL13"
    structure = record["target_structure"]
    assert structure["entry_id"] == "1ABC"
    assert structure["chain"] == "1"
    assert structure["source"] == "rcsb"
    assert structure["provenance"] == "RCSB PDB deposit 1ABC chain 1 for P35225."
    model = record["generative_model"]
    assert model["command"]["experiment"] == "inference_on_user_inputs"
    assert model["command"]["hub_repository"] == "nvidia/RFD4-Proteina"
    assert model["command"]["hub_revision"] == "abcdef"
    assert model["command"]["checkpoint_name"] == "release.ckpt"
    assert model["command"]["autoencoder_checkpoint_name"] == "ae.ckpt"
    assert model["command"]["checkpoint_path"] == NOT_RECORDED
    assert model["metrics"]["ckpt_name"] == "release.ckpt"
    assert model["metrics"]["ckpt_path"].startswith("<home>/")
    design = record["design_parameters"]
    assert design["binder_kind"] == "protein"
    assert design["contig"] == "60-110, /0, A1-9"
    assert design["hotspot"] == "not_set"
    assert design["seed"] == "7"
    assert design["host"] == "local"
    assert design["metrics"]["generation.n_recycle"] == "3"
    assert design["metrics"]["generation.args.nsteps"] == "60"
    screening = record["screening"]
    assert screening["method"] == "screen_generation"
    assert screening["call"] == "screened_clear"
    assert screening["bar"] == "sequence"
    assert screening["ppi_pass"] == "absent"
    assert "not a cure" in screening["note"]
    assert "not experimental validation" in screening["note"]
    assert screening["sample"]["sequence_pass"] is True
    assert screening["sample"]["contact_ms"] == 1.25
    assert record["model_confidence"]["status"] == "recorded"
    assert record["model_confidence"]["metrics"] == {"oracle_ppi.plddt_coi": 81.5}
    assert record["experimental_validation"]["status"] == NOT_PERFORMED
    services = record["source_databases"]["structure_search"]
    assert _status(services, "Open Targets", "disease name search") == NOT_PERFORMED
    assert _status(services, "Local disease catalog", "disease name") == "used"
    assert _status(services, "RCSB PDB", "polymer entity search and deposit records") == "consulted"
    assert _status(services, "Reactome", "participants of the selected pathway") == "consulted"
    records = {row["key"]: row["status"] for row in record["source_databases"]["target_records"]}
    assert records["uniprot"] == "record"
    assert records["ncbi"] == "no record"
    assert records["string"] == NOT_RECORDED


def test_a_missing_measurement_stays_unrecorded(tmp_path: Path) -> None:
    """No focus, no screen, and no confidence number are written as statuses."""
    output = tmp_path / "protein_1ABC_1"
    sample = output / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text("data_binder\n#\n", encoding="utf-8")
    (output / "inference_metrics").mkdir()
    (output / "inference_metrics" / "per_example_metrics.csv").write_text(
        "example_id,oracle_ppi.plddt_coi\nbinder_0,nan\n",
        encoding="utf-8",
    )
    search = search_trace(live_structures=False, live_disease=False, focus=None, collected={})
    written = write_research_traces(
        output,
        disease_name="Asthma",
        disease_id="EFO_0000270",
        option=structure_option(),
        binder_kind="protein",
        argv=["rfd4-generate", f"out_dir={output}"],
        screen=None,
        search=search,
        config=_config(tmp_path / "inference.yaml"),
    )
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["disease_input"]["status"] == NOT_RECORDED
    assert record["disease_input"]["confirmed_name"] == "Asthma"
    assert record["pathway"]["status"] == NOT_RECORDED
    assert record["pathway"]["pathway_id"] == NOT_RECORDED
    assert record["selected_target"]["role"] == NOT_RECORDED
    assert record["design_parameters"]["contig"] == NOT_RECORDED
    assert record["design_parameters"]["hotspot"] == NOT_RECORDED
    assert record["design_parameters"]["seed"] == NOT_RECORDED
    assert record["generative_model"]["command"]["experiment"] == NOT_RECORDED
    assert record["generative_model"]["command"]["hub_revision"] == NOT_RECORDED
    assert record["screening"]["status"] == NOT_RECORDED
    assert record["screening"]["method"] == "screen_generation"
    assert "not a cure" in record["screening"]["note"]
    assert record["model_confidence"] == {"status": NOT_RECORDED, "metrics": {}}
    assert record["experimental_validation"]["status"] == NOT_PERFORMED
    services = record["source_databases"]["structure_search"]
    assert _status(services, "Local structure file", "structure list") == "used"
    assert _status(services, "Open Targets", "disease-associated targets") == NOT_PERFORMED
    assert _status(services, "Reactome", "pathway members of the associated proteins") == NOT_PERFORMED
    assert _status(services, "STRING", "interaction partners of the associated proteins") == NOT_PERFORMED
    records = {row["key"]: row["status"] for row in record["source_databases"]["target_records"]}
    assert set(records.values()) == {NOT_PERFORMED}


def test_a_modified_checkpoint_does_not_claim_the_release_pin(tmp_path: Path) -> None:
    """The release revision is recorded only for the original experiment."""
    output = tmp_path / "protein_1ABC_1"
    folder = output / "inference_samples" / "binder_0"
    folder.mkdir(parents=True)
    (folder / "binder_0_generated.cif").write_text("data_binder\n#\n", encoding="utf-8")
    config = _config(tmp_path / "inference.yaml")
    argv = [
        "rfd4-generate",
        "inference_experiment=disease_binder",
        "hf_ckpt_path=null",
        "ckpt_path=/home/someone/models",
        "ckpt_name=disease.ckpt",
        f"out_dir={output}",
    ]
    written = write_research_traces(
        output,
        disease_name="Asthma",
        disease_id="EFO_0000270",
        option=structure_option(),
        binder_kind="protein",
        argv=argv,
        screen=None,
        config=config,
    )
    text = written[0].read_text(encoding="utf-8")
    record = json.loads(text)
    command = record["generative_model"]["command"]
    assert command["experiment"] == "disease_binder"
    assert command["checkpoint_name"] == "disease.ckpt"
    assert command["hub_repository"] == NOT_RECORDED
    assert command["hub_revision"] == NOT_RECORDED
    assert command["checkpoint_path"].startswith("<home>/")
    assert "abcdef" not in text
    assert "release.ckpt" not in text
    assert "/home/someone" not in text


def test_the_original_command_uses_the_inference_config_pin(tmp_path: Path) -> None:
    """A command that leaves the checkpoint unnamed records the config the run uses."""
    output = tmp_path / "protein_1ABC_1"
    folder = output / "inference_samples" / "binder_0"
    folder.mkdir(parents=True)
    (folder / "binder_0_generated.cif").write_text("data_binder\n#\n", encoding="utf-8")
    config = Path(__file__).resolve().parents[1] / "configs" / "inference.yaml"
    written = write_research_traces(
        output,
        disease_name="Asthma",
        disease_id="EFO_0000270",
        option=structure_option(),
        binder_kind="protein",
        argv=["rfd4-generate", "inference_experiment=inference_on_user_inputs", f"out_dir={output}"],
        screen=None,
    )
    record = json.loads(written[0].read_text(encoding="utf-8"))
    text = config.read_text(encoding="utf-8")
    revision = next(
        line.split(":", 1)[1].split("#", 1)[0].strip()
        for line in text.splitlines()
        if line.startswith("hf_ckpt_revision:")
    )
    command = record["generative_model"]["command"]
    assert command["hub_revision"] == revision
    assert command["hub_repository"] == "nvidia/RFD4-Proteina"
    assert command["checkpoint_name"] == "rfd4-proteina-flow-release-2026-10-19.ckpt"
    assert command["autoencoder_checkpoint_name"] == "rfd4-proteina-ae-release-2026-10-19.ckpt"
    assert "DATA_PATH" not in written[0].read_text(encoding="utf-8")


def test_a_run_without_a_generated_cif_still_has_a_trace(tmp_path: Path) -> None:
    """No binder file is recorded as not written, and it is not given a confidence."""
    output = tmp_path / "protein_1ABC_1"
    output.mkdir()
    written = write_research_traces(
        output,
        disease_name="Asthma",
        disease_id="EFO_0000270",
        option=structure_option(),
        binder_kind="protein",
        argv=["rfd4-generate"],
        screen=None,
    )
    assert [path.name for path in written] == ["run.json"]
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["kind"] == "run"
    assert record["example_id"] == NOT_RECORDED
    assert record["structure_file"] == NOT_RECORDED
    assert record["note"] == "No generated binder CIF was written."
    assert record["model_confidence"]["status"] == NOT_RECORDED
    assert record["experimental_validation"]["status"] == NOT_PERFORMED


def test_each_binder_gets_its_own_trace(tmp_path: Path) -> None:
    """Two generated structures do not share one confidence or one sample record."""
    output = tmp_path / "protein_1ABC_1"
    for name in ("binder_0", "binder_1"):
        folder = output / "inference_samples" / name
        folder.mkdir(parents=True)
        (folder / f"{name}_generated.cif").write_text("data_binder\n#\n", encoding="utf-8")
    (output / "inference_metrics").mkdir()
    (output / "inference_metrics" / "per_example_metrics.csv").write_text(
        "example_id,oracle_ppi.plddt_coi\nbinder_0,80\nbinder_1,\n",
        encoding="utf-8",
    )
    screen = {
        "call": "untested",
        "samples": [
            {"example_id": "binder_0", "contact_ms": 4.0},
            {"example_id": "binder_1", "contact_ms": None},
        ],
    }
    written = write_research_traces(
        output,
        disease_name="Asthma",
        disease_id="EFO_0000270",
        option=structure_option(),
        binder_kind="protein",
        argv=["rfd4-generate"],
        screen=screen,
    )
    by_name = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in written}
    assert set(by_name) == {"binder_0.json", "binder_1.json"}
    assert by_name["binder_0.json"]["model_confidence"]["metrics"] == {"oracle_ppi.plddt_coi": 80.0}
    assert by_name["binder_1.json"]["model_confidence"]["status"] == NOT_RECORDED
    assert by_name["binder_0.json"]["screening"]["sample"]["contact_ms"] == 4.0
    assert by_name["binder_1.json"]["screening"]["sample"]["contact_ms"] is None
    assert by_name["binder_0.json"]["experimental_validation"]["status"] == NOT_PERFORMED
    assert by_name["binder_1.json"]["experimental_validation"]["status"] == NOT_PERFORMED


def test_a_finished_generation_archives_the_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The archive copy keeps the trace, and generation.json still has no sequence."""
    from rfproteina.disease.cli import _save_finished_generations
    from rfproteina.disease.dispatch import GenerationRequest, RunResult
    from rfproteina.disease.lookup import Disease

    output = tmp_path / "protein_1ABC_1"
    folder = output / "inference_samples" / "binder_0"
    folder.mkdir(parents=True)
    (folder / "binder_0_generated.cif").write_text("data_binder\n#\n", encoding="utf-8")
    monkeypatch.setattr(
        "rfproteina.disease.cli.screen_generation",
        lambda *_args, **_kwargs: {"call": "untested", "bar": "sequence", "samples": []},
    )
    request = GenerationRequest(
        host="local",
        argv=[
            "rfd4-generate",
            "inference_experiment=inference_on_user_inputs",
            f"out_dir={output}",
        ],
        local_input=tmp_path / "protein_1ABC_1.yaml",
        remote_input="protein_1ABC_1.yaml",
        remote_out_dir="protein_1ABC_1",
    )
    out = tmp_path / "campaign"
    _save_finished_generations(
        [RunResult(host="local", argv=list(request.argv), returncode=0)],
        [request],
        [structure_option()],
        out,
        Disease(name="Asthma", disease_id="EFO_0000270"),
        "protein",
        disease_query="Asthma",
        focus=_focus(),
        search=search_trace(live_structures=True, live_disease=True, focus=_focus(), collected={}),
    )
    saved = out / "generations" / "Asthma" / "P35225" / "protein_1ABC_1"
    record = json.loads((saved / "generation.json").read_text(encoding="utf-8"))
    assert record["disease_name"] == "Asthma"
    assert record["disease_id"] == "EFO_0000270"
    assert record["target_id"] == "P35225"
    assert record["entry_id"] == "1ABC"
    assert "sequence" not in record
    trace = json.loads((saved / "research_traces" / "binder_0.json").read_text(encoding="utf-8"))
    assert trace["disease_input"]["query"] == "Asthma"
    assert trace["pathway"]["pathway_id"] == "R-HSA-9"
    assert trace["screening"]["call"] == "untested"
    assert trace["experimental_validation"]["status"] == NOT_PERFORMED
    assert PEPTIDE not in (saved / "research_traces" / "binder_0.json").read_text(encoding="utf-8")
    assert (output / "research_traces" / "binder_0.json").is_file()
