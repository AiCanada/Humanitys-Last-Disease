# Screening subpackage — third-party tool code

Third-party **code** for the screening tools lives here, inside the repository.
Everything a tool *reads* — model weights, example inputs, reference tables and
databases — stays under `DATA_PATH` and is deliberately not here.

The split is by kind, not by tool: a single tool usually has code in this folder
and data under `DATA_PATH`, and the resolver in
`src/rfproteina/disease/screening_tools.py` joins the two back together.

## What is here

| Path | Size | Upstream | Licence |
| --- | --- | --- | --- |
| `crossabsense/CrossAbSense/` | 1.0 MB, 66 `.py` | [SimonCrouzet/CrossAbSense](https://huggingface.co/SimonCrouzet/CrossAbSense) | Apache-2.0 |
| `thermompnn/ThermoMPNN/` | 268 KB, 16 `.py` | [Kuhlman-Lab/ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN), commit `370f76ec62bd929f7425e311d8df04a0d094990f` | MIT |
| `commec/tools/usr/` | 76.6 MB, 363 files | HMMER 3.4, NCBI BLAST+, Infernal, bundled by Commec | each upstream's own |
| `commec/bin/zstd` | 1.2 MB | `zstd` | BSD-3-Clause / GPL-2.0 dual |
| `mmseqs/mmseqs` | 23.1 MB | [soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2), commit `6b89b5ca0e0135b6a41fce80add4d17c3f586fb7` | MIT |

Licences and citations for every screening tool, including the ones whose code
is not here, are in the root [`README.md`](../README.md#screening-tools-licences-and-citations).

## What stays under `DATA_PATH`, and why

| Kept at `DATA_PATH/…` | Size | Why it is not here |
| --- | --- | --- |
| `commec/databases/` | 39.1 GB | database, not code |
| `screening_tables/` | 4.6 GB | reference tables |
| `mhcflurry/{models,alleles.txt}` | 136 MB | weights and an allele list |
| `crossabsense/{models,hf}` | 1.3 GB | weights |
| `deepsp/`, `deepviscosity/` | 25 MB | weights plus the authors' example CSVs |
| `octant/`, `iedb/`, `tepitope/`, `bindingdb/` | 91 MB | measured data and matrices |
| `dockq/venv`, `mhcflurry/venv`, `commec/venv`, `thermompnn/venv`, `crossabsense/venv` | 5.2 GB | installed virtual environments — see below |
| `thermompnn/ThermoMPNN/{models,vanilla_model_weights}` | 63 MB | weights |

**The five virtual environments stay put.** Each has 10–35 files hard-coding its
own absolute path (a console script execs its interpreter by absolute path), and
each `pyvenv.cfg` names a base interpreter under a user's home directory. They
are installed artifacts, fully reproducible from the installers in `scripts/`,
so relocating them would buy nothing and would put a home-directory path into a
repository headed for a public push.

**ThermoMPNN's weights stay put, and its code does not.** Upstream ships both in
one tree, and upstream's loader reads `vanilla_model_weights/` relative to a
single configured root — which is why the whole clone used to stay under
`DATA_PATH`. That clone has been deleted: only the two weight directories above
remain there, 63 MB of the original 217 MB. Its training and test datasets
(`data_all/`, `dataset_splits/`, 90 MB), figures and example inputs are not kept
on this machine; they are re-fetchable from upstream at the commit
`thermompnn/manifest.toml` pins, which is also how the installer gets the
weights. The source is now vendored here instead, and
`src/rfproteina/disease/thermompnn_runner.py` points that one config key at the
weights root before calling upstream's own inference. So the code is readable in
the repository, upstream's files are still byte-for-byte upstream's, and the 63 MB
of weights plus 94 MB of training data stay where large binaries belong.

### Vendored, not cloned

`crossabsense/CrossAbSense/` and `thermompnn/ThermoMPNN/` are the two trees whose
upstream source is tracked here. Neither carries a `.git`: git would record a
clone as a bare gitlink pointing at a repository nobody else has, so the files
themselves are tracked and the tool's `manifest.toml` records the upstream commit
they came from, which is what lets the tree be checked against upstream or
re-cloned.

Every vendored file is byte-identical to upstream's blob at that commit, so
`git hash-object <file>` equals upstream's id for it — verified against a clone
of each upstream repository, not assumed. Three rules protect that:
`.gitattributes` marks both trees `-text`, because their upstream files are CRLF
and this repository's LF rules would otherwise rewrite them on commit;
`pyproject.toml` excludes `screening subpackage/**` from ruff, so `make format`
does not restyle upstream's code; and `.pre-commit-config.yaml` excludes it from
`nbstripout`, which would otherwise strip the vendored notebooks' outputs.

The CRLF hazard is not theoretical. CrossAbSense was committed before the
`-text` rule existed, and 37 of its 89 files were silently normalised to LF on
the way in, so the tree no longer matched the commit its `manifest.toml` pinned.
It was restored with `git add --renormalize` once the rule was in place.

Three things must never enter a vendored tree, and all three are re-excluded
*after* the negation that re-includes the tree: weights, a nested `.git`, and
`__pycache__`. The last is not hypothetical — the tool's own interpreter imports
the vendored source and writes bytecode beside it, so one scan is enough to
produce it. `tests/test_screening_vendoring.py` asks git directly whether each
would be tracked.

## How a tool is resolved

`screening_tools.py` searches two roots, in this order:

1. **`DATA_PATH/<tool>/…`** — explicit operator configuration wins.
2. **`<checkout>/screening subpackage/<tool>/…`** — this folder, found by
   walking up for `.project-root`, never by counting parent directories.

`SCREENING_CODE_ROOT` overrides step 2 outright. Either root may be missing, and
a tool that is in neither resolves to `None`, which the screens report as
`untested` with a blocker naming what is absent — never as a passing result.

Two consequences worth knowing:

- Because `DATA_PATH` is searched first, a test that points `DATA_PATH` at a
  temporary directory sees that directory and not the developer's own checkout.
- Lookups are by **artifact**, not by directory, via `tool_code_path`. A
  relocation leaves an empty directory behind at the old path, and an empty
  directory would otherwise shadow the real copy.

## Adding a tool

1. `mkdir "screening subpackage/<tool>"` and put its code there. Nothing needs
   registering: `tool_code_dir("<tool>")` and `tool_code_path("<tool>", …)`
   already resolve any name.
2. Leave weights, examples and databases under `DATA_PATH/<tool>/`.
3. Add the installer as `install_<tool>.py`, the upstream `LICENSE`, and a
   `manifest.toml` recording upstream URL, version pin, SHA-256 and citation.
4. Record the licence and citation in the root `README.md`.

The folder name contains a space and is therefore **not** an importable Python
package. That is intentional — nothing here is imported, it is resolved by path.
Python modules belong in `src/rfproteina/disease/`.

## What git tracks here

The recipe, plus the two vendored source trees: this file, `NOTICE`, each tool's
`manifest.toml`, `LICENSE`, `README.md` and `install_*.py`, and everything under
`crossabsense/CrossAbSense/` and `thermompnn/ThermoMPNN/`. The rest of the
payload above — roughly 101 MB today — is ignored by rules at the end of the
repository's `.gitignore`, which ignore broadly and re-include those few names,
so a missed rule costs one untracked file rather than gigabytes in a public
repository.

A `!<tree>/**` negation re-includes *everything* beneath it, including a `.git`
that a re-clone brings back and any weight file copied in, both of which outrank
the top-level `**.pt` and `.git/` rules. The rules that put those back out of
reach come after the negations, and are the load-bearing part of that block.
Verify a path with:

```bash
git check-ignore -v "screening subpackage/<tool>/venv/bin/python"
```

`.move-manifest.tsv`, if present, is a local record of what was moved from where
and is not tracked.
