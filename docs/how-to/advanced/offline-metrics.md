# Running metrics offline on a folder of structures

You do not have to re-run generation to get metrics. `evaluate.py` scores a folder of structures you
already have, on the same Hydra config, so `metrics_tags` and any custom metric config work exactly
as above.

```bash
rfd4-evaluate \
    inputs=/path/to/structures \
    metrics_tags='[unconditional_refolding]' \
    out_dir=/path/to/eval_output
```

Deposited PDB files can supply coordinates for geometry and composition
metrics. The reader loads structure annotations from mmCIF when present.
Missing reference structures or conditioning labels still limit which
comparisons the evaluator can make.

Point `inputs` at a generation run or its `inference_samples/` directory to evaluate
the `*_generated` structures. In a standalone structure directory the evaluator
scans all recognised structure files. It records unreadable designs in
`evaluation_metrics/per_example_metrics.csv` with `evaluation_error=structure_unreadable`.

Pass `task_csv=<file>` with `designed_structure_path` and `task` columns to label structures
for task summaries and clustering. `metrics_tags` applies to the evaluation invocation.
Run separate evaluations for groups requiring different tags or criteria.

For large batches split across SLURM array jobs (`launch_generate.sh --njobs N`), each job writes its
own `per_example_metrics_job{N}.csv`. Concatenate them afterwards.

```bash
python -m rfproteina.inference_utils.aggregate_metrics \
    ./runs/my_run/inference_metrics/
```

An offline evaluation uses the same command on `evaluation_metrics/`. Per-task summaries
retain the dataset name. A retry replaces a complete design row, including any missing
values in the new observation. A confirmed pass requires a binary pass flag and, when
available, confirmation that all criterion inputs were present. Missing outcomes count
in the denominator for the evaluated dataset, task and tag group. An entirely untested
group has an undefined rate. New per-example and summary CSVs retain the available
numerical precision. A sweep can preserve only the digits in its source tables.

The evaluator preserves literal task and example labels such as `NA` and `0001`.
Reordering or repeating a metric tag keeps the design in the same pass-rate denominator.
A task mapping may repeat an identical assignment. An empty or conflicting assignment
stops evaluation before it writes reports.

### Retrying and exporting reports

Job writers and in-place aggregation share a directory lock. They finish calculating
and serializing the CSVs before replacing a published file. Each replacement is atomic,
and the primary per-example table is replaced last. A calculation, serialization or
replacement failure preserves the previous reports. After a process is killed, the next
in-place writer restores an interrupted batch or finishes archiving a committed one.
Read a multi-file report after the command finishes.

Aggregation checks the header, quoting and field count of every logical CSV record
before it replaces a report. Empty or duplicate column names, embedded nulls and
rows of the wrong width stop the operation and identify the offending file.
Quoted commas, embedded newlines, wide fields and UTF-8 byte-order marks are supported.

Reusing a job ID preserves the previous job tables in `individual_job_results/`.
Passing `--output_dir` with a different directory preserves the source job tables,
archives and chain files. If the source CSVs change while the export is being prepared,
the command stops before publishing the new report. A missing input or an empty source
directory returns an error and preserves any previous export.

Publishing a primary table writes `aggregation_sources.csv` alongside it. Each
`report` row records the filename and SHA-256 of a table in that publication.
Each `shard` row records an input filename, ordering timestamp and digest.
Refreshing summaries from the same primary retains the recorded shard identities.
Archive renames preserve the identity of a shard.
If two observations have the same timestamp, the logical job filename breaks
the tie, followed by its revision within that job. Repeating aggregation after
archiving gives the same result.

The screen checks the receipt against the primary and every remaining local job
file. New or changed shards require another directory aggregation. An interrupted
publication keeps the verdict unavailable until an in-place writer recovers it.
A primary and its matching receipt can be read together after copying them to a
directory without the original shards. If the primary's bytes disagree with the
receipt, restore the verified pair or aggregate copies of the intended job files
in a fresh directory. Legacy standalone tables remain readable. A legacy primary
beside loose job shards needs directory aggregation to establish their inclusion.

### Collecting results across configurations

Run the sweep exporter from the project environment. Give it each configuration
you expected when a missing result should make the export fail.

```bash
python src/script_utils/concatenate_sweep_results.py --sweep-dir ./runs/comparison \
    --expected-config baseline --expected-config repeat
```

The exporter copies each source cell as text and adds `sweep_config`. It writes
`sweep_per_example_metrics.csv`, `sweep_per_task_metrics.csv`,
`sweep_aggregated_metrics.csv` and `sweep_coverage.csv` together. A failed
refresh preserves the previous reports. Each coverage row names the configuration,
stage and source table. For a table actually read, it includes its row count,
the SHA-256 of its original bytes and the original header as a JSON list in
`source_columns`. Use that list to distinguish a missing column from an empty cell
in a column the source supplied.

| Coverage status | Meaning |
| --- | --- |
| `verified` | The bytes matched the digest in the per-configuration receipt. |
| `legacy` | The file was readable. Its original publication supplied no digest for this table. |
| `missing` | The table was absent. |
| `primary_missing` | A summary was present in a directory without a published primary. |
| `not_published` | The current receipt covered another set of tables. A leftover summary was excluded. |

With an explicit plan, all three tables must contain data in each named
configuration. Establishing how many individual designs were attempted also
requires the run inventory. In discovery mode, configurations with only a samples
directory are listed as missing when another configuration supplies a primary.
A readable header-only table records zero rows. An unavailable table has a blank
row count. When both stages contain a primary, receipt, unfinished job or pending
publication, the exporter prefers `inference_metrics`. An empty inference
directory leaves the offline results eligible.
Use `--metrics-dir evaluation_metrics` to collect the offline evaluations.

### Interpreting cluster and probability summaries

Foldseek assigns cluster IDs within one output tree, dataset and task. Ordinary
aggregation can reuse assignments when one source table covers exactly the same
observations for that cohort. Joining independently clustered shards leaves the joint
diversity and clustered pass rate undefined until the combined cohort is clustered.
Assignment coverage can be reported across cohorts. A refresh that supersedes
published assignments archives the previous CSVs in `previous_reports/`. If a repeat
clustering produces zero assignments for a previously clustered run, it stops before
replacing the reports.

When structure paths are present, `sequence_probability_errors.csv` reports the
design count, paths supplied, readable structures and structures with soft conditions
for each labeled task. The `n` column counts distinct position–token pairs. Zero
pairs leave the error measures blank. A refresh with no structure columns replaces
an existing probability report with its header.

`launch_generate.sh` also runs this for you as a dependent job. See [Evaluating
results](../run-the-model.md#evaluating-results) for FoldSeek diversity clustering and the other
`aggregate_metrics` options, and the [output format reference](../../reference/output-format.md#metrics-csvs-and-aggregation)
for the exact file layout these commands produce.
