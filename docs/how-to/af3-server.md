# Refold designs on the hosted AlphaFold Server

Use this when you want to check what the model produced and cannot run a local
AlphaFold 3 — the weights are licence-gated, the container may be unavailable,
and the hosted server is free for non-commercial use. It is a way to **test the
program's output**, not a screening backend: the account allows 30 jobs a day.

For a local AlphaFold 3 or AlphaFold 2 oracle, see
[Evaluate with metrics](evaluate-with-metrics.md) and [Refolding](refolding.md)
instead. Nothing on this page changes how those work.

## What is and is not automated

The AlphaFold Server publishes **no submission API**. Its complete FAQ describes
one path: upload a JSON file of up to 100 jobs through the web page, then
download a result archive. So the upload is a click, and these commands do
everything on either side of it:

| Step | Who | Command |
| --- | --- | --- |
| Sign in | you, in your own browser | — |
| Record the session and download directory | `rfd4-af3server` | `login` |
| Build and validate the request | `rfd4-af3server` | `prepare` / `submit` |
| Upload the file, run the drafts | you, on the page | — |
| Download the results | you, in your browser | — |
| File the archive against your designs | `rfd4-af3server` | `retrieve` |

Three things this deliberately does not do, two of them because the service's
own terms close them off:

- **It never handles a password or a token.** The server issues no credential
  for a program to hold; it authenticates a Google account through an ordinary
  browser sign-in. Passing something that looks like a secret to `login
  --account` is refused.
- **It never drives the site's internal endpoints.** Reaching them means
  reverse engineering the service, which the Prohibited Use Policy incorporated
  into the AlphaFold Server terms forbids.
- **It never fetches job pages.** `robots.txt` disallows `/fold/*`.

## Sign in

```{note}
`rfd4-af3server` is a new console script. In an existing editable install the
entry point is only created when the package metadata is refreshed, so until you
re-run `pip install -e .` use `python -m rfproteina.af3server.cli` instead — the
arguments are identical.
```

Open <https://alphafoldserver.com>, sign in with your Google account, and accept
the non-commercial terms there. Then record what the tool needs to know:

```bash
rfd4-af3server login --downloads ~/Downloads --accept-terms
rfd4-af3server status
```

`--downloads` is where `retrieve --watch` looks for archives. `--account` is an
optional label, for telling two accounts apart; it is stored encrypted beside
the API keys, under `~/.config/Humanitys-Last-Disease`, and never appears in a
run log.

`status` prints today's remaining job estimate. It is **this machine's own
count**, kept because the quota is the one bound a batch has to respect and the
one thing a program cannot read back from the service. When it disagrees with
the server, the server is right.

The allowance it plans against is the published 30 a day. If your account's is
different — the figure has been raised twice already — set it once, so a batch
is not reported as overflowing a limit that does not apply to you:

```bash
rfd4-af3server login --daily-limit 200
```

## Build the request

```bash
rfd4-af3server prepare --run disease_runs/my_run --to af3server_jobs
```

This reads every generated design under the run, converts each to one job in the
`alphafoldserver` dialect, validates it, and writes batch files of up to 100
jobs. Nothing is uploaded.

Two behaviours are worth knowing before you read the output.

**Every entity goes, or the design does not go at all.** A request that quietly
leaves a chain out does not fail — it succeeds at a different question.
Refolding a binder alone tells you whether the binder folds; it can never tell
you whether the complex forms, which is what `min_ipae_coi` and an aligned
binder RMSD are about. So a design holding something the dialect cannot carry is
**refused by name** rather than sent without it:

| Blocker | What it means |
| --- | --- |
| `ligand_not_accepted:<code>` | The dialect takes 19 named ligands, not the open CCD. A design built around any other cofactor cannot be sent as a complex. |
| `ion_not_accepted:<code>` | Ten ions are accepted. |
| `sequence_not_standard_amino_acids` | Only the 20 standard amino acids; no `X`, `B`, `Z`, `U` or `O`. |
| `chain_below_residue_floor` | The server requires at least 4 residues in every polymer chain. |
| `polymer_type_not_supported:<type>` | `polypeptide(D)`, a DNA/RNA hybrid strand, a polysaccharide and a peptide nucleic acid have no representation. |
| `over_token_limit` | A job may total 5,000 tokens: one per residue, one per base, one per ligand atom, one per ion. |
| `entity_table_absent` | The structure's `_entity` table is missing, so there is no way to know the polymer chains are the whole structure. |
| `non_polymer_components_not_single` | A non-polymer entity naming several components is not one ligand. |
| `name_not_a_filename` | Results come back named for the job, so the design id has to survive as a filename. |

**Templates are off for the design and on for the context.** The designed chain
is the thing under test, and letting the server retrieve a PDB template for it
invites a near neighbour to supply the answer the refold is meant to produce
from sequence alone. The target's experimental structure is a real input to the
complex question, so context chains keep the server's default. Override either
with `--design-templates` and `--no-context-templates`.

To validate a file you wrote or edited yourself:

```bash
rfd4-af3server check af3server_jobs/af3server_jobs.json
```

`check` reports **every** problem, not the first, so one fix is one round trip.

## Submit

```bash
rfd4-af3server submit --run disease_runs/my_run --to af3server_jobs
```

This builds the files, prints the plan against the daily quota, opens the
submission page, and counts the jobs against the ledger. You do the upload with
the page's **Upload JSON** button. Imported jobs arrive as saved drafts and have
to be run.

A batch larger than the day's remaining quota is **split and the overflow
named** — never silently truncated. The server itself lets a job be saved and
run after the quota refreshes. Pass `--no-record` if you did not actually
submit, and `--no-open` to stay out of the browser.

## Retrieve

```bash
rfd4-af3server retrieve --run disease_runs/my_run --watch
```

`--watch` waits for new `.zip` files to appear in the recorded downloads
directory and files each one as it lands; it only takes archives that appear
*after* it starts, so yesterday's downloads are not re-imported over today's
results. Or name them directly with `--archive`, repeated.

Each job's **best-ranked** model is filed by default — ranked on the server's
own `ranking_score`, not on the file index. `--every-model` files all five under
per-model save ids instead.

`--chain-of-interest` is named as **your design** spells it. The server keeps no
chain names: it labels chains by position in `sequences`, in AlphaFold 3's
reverse-spreadsheet order (`A`…`Z`, then `AA`, `BA`, `CA`), so a design whose
binder is strand `A` and target strand `E` comes back as chains `A` and `B`.
The name is translated per design before the metrics are taken. Getting this
wrong is silent — a chain of interest the result does not contain drops every
per-chain and interface value without raising anything — which is why it is
computed from your own structure rather than assumed. `prepare` prints the
mapping (`chains A->A, E->B`) so you can see it before you upload.

Refolds are filed **beside each design**, under `inference_samples/<id>/`, which
is where the oracle looks. Not under the run root: a run nests its designs, and
a run of several batches nests them one level further. A design whose id cannot
be joined back onto its own folder — which happens when a binder token repeats
across batches and the id is flattened to a path — is reported as
`design_base_not_resolved` rather than filed somewhere nothing reads.

What lands on disk is what a local oracle would have written: the flat metric
keys at the oracle's `confidence_path` and the structure at its
`refolded_cif_path`. Every refolding metric, declared pass criterion and disease
screen `refold` report then runs unchanged.

## Read the imported refolds

The files are in place; `rfd4-evaluate` turns them into metrics. One thing is
needed: tell the oracle not to start a local AlphaFold 3, because the whole
point of this route is that you cannot.

```bash
rfd4-evaluate ... metrics.refolding_oracles.af3.oracle.cached_only=true
```

`cached_only` starts no daemon and needs no image, weights or GPU — reading a
fold that is already on disk needs none of them. Without it, turning a refolding
tag on merely to *read* an imported fold launches the ~10 GB worker and fails
for want of the very runtime the import existed to avoid.

A design with no refold on disk then **raises and is named**, rather than being
counted as folded. A run that quietly reported a short denominator would look
like a pass rate.

A job matching no design is **named**, not skipped. The server lowercases and
sanitises a job name, so `M0664_2dhn` comes back inside
`fold_m0664_2dhn_model_0.cif`; matching is case-insensitive, and two design ids
that the server would spell alike are refused rather than guessed between.

## What a server refold cannot tell you

Said here rather than discovered later as empty columns.

**Three flat metrics are unreachable.** The server's summary has exactly ten
keys, and `chain_plddt`, `chain_pae` and `chain_pair_pae` are not among them. So
`plddt_coi`, `pae_coi` and `ipae_coi` are absent from a server refold, and a
pass criterion bounding one of them reports an **unmet** bound rather than a
failed one. `min_ipae_coi`, `iptm_coi`, `ptm_coi`, `overall_ptm`,
`overall_iptm`, `ranking_score`, `fraction_disordered`, `has_clash` and the
per-pair `min_ipae_<i>_<j>` / `iptm_<i>_<j>` are all present. Per-atom pLDDT is
unaffected: it arrives in the model CIF's `B_iso_or_equiv` column, which is
where the pipeline reads it on the local path too.

**The structure is in the server's own frame and atom order.** A local oracle
remaps its prediction onto the input's atom indexing and aligns it to the input
frame before writing. A file that arrived through a browser has had neither done
to it. Confidence metrics are indifferent to that; anything comparing
coordinates to the design depends on doing its own superposition.

**The server does not send `chain_ids`.** This is handled, but it is worth
knowing why the handling exists. Read as an empty chain order, a missing
`chain_ids` silently removes *every* per-chain and interface metric without
raising anything — `combinations(range(0))` emits no chain pair, and
`chain_of_interest not in []` returns early. The order is therefore derived from
`token_chain_ids` in the full-data JSON, the way AlphaFold 3's own code derives
it, and recorded as `af3server_chain_order: not_recorded` when even that is
missing. The same fallback now also covers an older local AlphaFold 3, whose
summary did not write `chain_ids` either.

## Terms

The AlphaFold Server is free for **non-commercial use** by individuals and
non-commercial organisations. You accept those terms on the site, and nothing
here accepts them for you. Reusing predictions, and training on them, are
governed by those terms and by the output terms in the archive's
`terms_of_use.md` — read them rather than taking this page's word for it.
