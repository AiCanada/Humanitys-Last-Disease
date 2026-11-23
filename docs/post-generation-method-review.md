# 60 working screens after generation

On 2026-11-23, 60 distinct methods or evidence readers produced a result on a live input. Each had a documented control or independent numerical comparison. The count includes methods already present in earlier versions. Thirteen additional adapters, configuration checks and planners produced no scientific result and are excluded.

Some entries compute a prediction or a measurement from a structure. Others read or verify an experimental result or a completed calculation. A working reader establishes that the software can interpret the supplied evidence. Predictive accuracy across new drug candidates remains unmeasured.

The review checked 118 local evidence files against their SHA-256 digests. All 60 measured workloads ran on CPU. The largest recorded local footprint was 15.095 GB, including the runtime and assets counted for that method. External service resources and transient writes between samples were outside the measurements.

Use the finished-run command and per-method input contracts to reproduce a screen. [Screen inputs](reference/screen-inputs.md)

| Method | Live control |
| --- | --- |
| PoseBusters docking and ligand checks | A separated receptor and the ligand-only configuration produced different verdicts. |
| RDKit alerts and ligand descriptors | Ethanol and catechol gave different catalogue results. |
| DockQ | Native 1TIM and a displaced partner differed. |
| ProLIF typed interactions | The reference pose and separated ligand yielded eight and zero interactions. |
| Native geometric interaction fingerprint | All contacts disappeared after separation. |
| Fraction of native interface contacts (Fnat) | All native pairs were recovered on identity and none after separation. |
| Biotite interface geometry | 1TIM had a measurable interface and the displaced chains had zero buried surface. |
| AGGRESCAN sequence score | The intact protein scored, while truncated and unresolved inputs were withheld. |
| Octant exact measurement lookup | A real release value survived conversion and an absent compound produced no measurement file. |
| ChEMBL exact panel lookup | One positive and two public no-data compounds were queried at EBI. |
| BindingDB fingerprint lookup | A release self-match was attributed and helium returned no matches. |
| GtoPdb secondary pharmacology reader | A release self-match was attributed, helium had no hits and three status annotations were checked. |
| ADMET-AI | Equivalent ethanol structures agreed and the batch withheld an invalid compound. The aqueous-solubility regression matched released aspirin and caffeine predictions, retained malformed and duplicate nominations, and scored a fixed ligand once across two pocket files. |
| Foldseek structural search | Ubiquitin returned 1,000 installed-index hits. The best query- and target-normalized TM scores were 0.8711 and 0.9864. Five malformed controls were rejected. Missing or damaged replacement output preserved the last valid table. |
| Foldseek structure clustering | Two identical copies grouped together and a different structure formed a separate cluster. |
| MMseqs2 human proteome search | Human lysozyme P61626 matched at full identity and coverage. The named intended target was recorded separately. The reader accepts an empty search result and preserves the previous measurement on missing or damaged output. |
| GRAVY | Ubiquitin scored −0.48947. Biopython agreed. Empty and unresolved inputs produced no score. |
| Instability index | Ubiquitin scored 36.05526, in agreement with Biopython. Empty, single-residue and unresolved inputs were withheld. |
| Protein composition and mass | The formula agreed with RDKit and the 8,564.7366 Da mass agreed with Biopython to 0.001 Da. An unresolved residue withheld the whole-chain formula, mass and aliphatic index. The report counted 75 known residues. |
| Protein charge and isoelectric point | Independent Biopython pH calculations agreed and an invalid input was withheld. |
| Sequence diversity | Repeated inputs, different lengths and a missing chain were accounted for. |
| ANARCII numbering | Reversed chain order preserved attribution and ubiquitin was refused. |
| Antibody chemical liability motifs | The deposited pair produced 17 flags, including six in loops. An unnumberable protein and an empty set of numbered chains each returned a blocker. |
| DeepSP formulation descriptors | Missing and ambiguous chain pairs were held. |
| DeepViscosity | All 102 fitted models ran on the reference. Four malformed descriptor cases were held. |
| BindingDB ExtraTrees regressor | Equivalent inputs matched across three extrapolated axes. An invalid structure was rejected. |
| Ligand component composition | Caffeine and ambiguous component selections remained distinguishable through screen.json. |
| OpenMM single-structure potential difference | Six independent preparations of identical coordinates changed the sign. The corrected report withholds an affinity verdict. |
| Trajectory dynamics | Ten recorded frames measured. Inconsistent identities and malformed coordinates were withheld. |
| Human Protein Atlas plasma abundance | The raw value was checked in two units. An unrelated accession and an impossible negative concentration were withheld. |
| STRING physical network lookup | Duplicate rows and self-interactions preserved the count. Conflicting source IDs were withheld. |
| UniRef50 human cluster members | A verified singleton and an absent members list produced different measurement and blocker fields. |
| Ensembl Compara paralog lookup | All four source identities were read. Missing or out-of-range identity values could not produce a clear. |
| Supplied plasma fold change | Equivalent ratio and log scales matched, a foreign target was held and the historical unscaled chart was withheld. |
| P-SEA secondary structure | One missing Cα left 75 scored residues. With all Cα coordinates absent there were no fractions. A two-chain comparison gave 22 segments after correcting counts across chain boundaries. |
| Local distance difference test (lDDT) | Identity, translation, displacement, reference broadcasting and unmatched inputs were checked. |
| Stereochemistry | Biotin and its mirror, paired references and reordered atom identifiers were checked. |
| CCD internal bond geometry | Reference, translated, stretched, partial and entirely missing coordinates were checked. |
| Steric clash fraction | An atom overlap across insertion codes was located correctly and incomplete inputs were held. |
| Element-dependent bond-length checks | A stretched bond was flagged, and missing bonded coordinates withheld the measurement. |
| Contact molecular surface | Native, separated, single-chain, missing-identity and nonfinite inputs were checked. |
| Supplied toxicology concentration reader | The measured release row was distinguished from wrong units, bounds and a nonfinite reading. |
| BayeshERG | The official release failed on one graph until batching preserved its features. Dofetilide and two ethanol runs produced probabilities. Corrupted weights, malformed inputs, mismatched rows and substituted variances were rejected. |
| BindingDB Chemprop D-MPNN | Fresh checkpoints reproduced 307 held-out endpoint observations across three axes. Singleton and 51-row predictions agreed. Tests rejected mismatched assays, targets, corrupt references and an incompatible method label. |
| Protein-Sol sequence predictor | Reversed the two deposited inputs and repeated ubiquitin alone. Rejected wrong identities, invalid fields and duplicated rows or columns. |
| C-alpha RMSD to input | Identical, reversed and rigidly transformed 1UBQ. Eight displaced alpha carbons against an independent float64 fit. Nine ambiguous or incomplete comparisons withheld. |
| CATH S95 domain classification | Compared genuine H, T and N rows from the complete released index with the authors' parser. Malformed rows and conflicting domain mappings withheld. |
| GearNet CATH classifier | Validated released class and architecture heads on two deposited proteins with rigid transformations. Topology-label mismatch, duplicate coordinates, invalid labels and a nonfinite architecture head were withheld. |
| OpenFE completed relative binding free-energy reader | Nine published comparisons and 54 repeat estimates and MBAR errors independently agreed with OpenFE 1.13.0 gather. Four successful retries were accounted for and 19 invalid inputs were rejected. |
| alchemlyb thermodynamic integration | Two stage integrals and both uncertainties independently verified from 84,021 raw GROMACS samples. The reader rejected 18 invalid inputs and reproduced the integrals when the samples were shuffled. |
| pmx nonequilibrium BAR report verification | Rechecked 480 work values with the authors’ original estimator and an independent PyMBAR point estimate. Controls included a swapped direction, wrong run or energy unit, and substitution of a different analytical variance. |
| BindingDB exact measurement lookup | Entire verified release streamed. Two independent scans counted 3,243,660 rows. The corrected active subset retained 60,015 measurements. Caffeine returned measurements, helium returned none, and archive, chain-attribution and atomic-publication controls passed. |
| Supplied solubility score reader | Preserved method, source and unit for published AGGRESCAN and CamSol scores. Rejected ambiguous or incomplete table rows and withheld nonfinite and boolean historical scores in both displays. |
| Aggrescan3D static structure score | All 76 deposited ubiquitin residue identities reconciled. Rigid translation left each score unchanged. Full and three-row residue profiles were withheld by the whole-structure reader. A separately supplied mean retained its scale, source and method in both displays. |
| Boltz-2 supplied small-molecule affinity reader | Original Boltz 2.2.1 output writer reproduced both recorded predictions from the first public human TNKS2 compound. Ten malformed affinity records were rejected. Missing and true zero were distinguished, and structural confidence could not substitute for affinity. |
| PRODIGY protein-pair affinity | The native 1TIM result was reproduced in deposited and generation-writer formats. The normal screening command retained displaced and corrupt controls. The deposited durvalumab fragment estimate was compared with the authors' SPR assay. |
| AutoDock Vina fixed-pose score | The deposited 1IEP imatinib pose and a clashing control agreed with the upstream Python API. Eighteen unsuccessful controls returned named blockers. The finished-run CLI and local MCP service retained all four attempted designs. |
| Therapeutic Nanobody Profiler | Caplacizumab VHH1, same-model native descriptor replay, archived-model discrepancy, failure and coverage challenges, finished-run and local service readback. |
| QupKake | Five molecules, seven exact native micro-pKa replays, chemically invalid native conjugate withheld, one matched and one missed SAMPL6 SM14 transition. Source maps reproduced under reordered SDF and SMILES, with finished-run and live service readback. |
| SyGMa | 110 default products and scores checked against the native release for five parents. For acetaminophen the sulfate matched exactly and the glucuronide matched connectivity only. Challenged phase order, fragment atom lineage, enantiomers, file integrity and complete per-parent coverage. Verified finished runs and both local MCP views. |

SyGMa’s acetaminophen result illustrates the scope of a control. The sulfate matched the measured reference at rank 3. The glucuronide matched connectivity at rank 1 with five unspecified sugar stereocentres. The two-analyte human assay leaves 38 additional proposed structures unassessed. [Human plasma and urine study](https://pubmed.ncbi.nlm.nih.gov/15127815/)
