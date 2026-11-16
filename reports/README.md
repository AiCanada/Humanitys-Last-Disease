# Production reports — `Humanitys-Last-Disease`

Reports describing the architecture and the post-generation screening pipeline of
`Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.

**Not medically verified.** Every number these reports describe is a computational
prediction. A clear screen is a computational result; it is not experimental validation.

## Architecture

| # | Report | Subject |
|---|---|---|
| 1 | [MCP Integration Layer](01-mcp-integration-layer.md) | The ten tools an assistant drives the program through, which of the eight declared capabilities are live, and the safety properties of the server |
| 2 | [AI Interface](02-ai-interface.md) | The five model providers, the desktop-application path, and how credentials are held |
| 3 | [Computational Backend](03-computational-backend.md) | Generation and evaluation, the plan-then-execute split, and what is not present — including docking and binding affinity |

## Screening

| # | Report | Subject |
|---|---|---|
| 4 | [Protein and peptide](04-screening-protein-and-peptide.md) | Immunogenicity, proteome off-target, MHC class I and II, ADA, paralog, the plasma axes, sequence developability, chemical liability and the drug-type report |
| 5 | [Peptide addition and nucleic](05-screening-peptide-and-nucleic.md) | The membrane axis, off-target hybridization, innate immune, and the nucleic drug-type report |
| 6 | [Ligand](06-screening-ligand.md) | The ten-protein safety panel, which way the cut runs, and formula weight |

## How a verdict is recorded

| # | Report | Subject |
|---|---|---|
| 7 | [Evidence and provenance](07-evidence-and-provenance.md) | What a stored verdict records about itself: the cuts, the axis list, the input digests, the code commit, what a re-screen replaced, and why every silence is named |
| 8 | [Screening strategies](08-screening-strategies.md) | The twelve named bars, why they are not ordered weak to strong, and the repository's eleven declared pass criteria |

## Reading these correctly

A `screened_clear` verdict means every axis in the named bar was clear, against the inputs
supplied, under the cuts recorded, as applied by the code commit recorded. The record
carries all four so the claim can be checked rather than trusted.

It does not mean a design binds, is safe, or is developable.

One axis refuses a measured clear by design — innate immune — because for it an absence of
evidence is not evidence of absence, and it clears only by a declared, reasoned judgement
from a person. Paralog's UniRef50 clustering refuses one too, but a singleton goes to
Ensembl Compara when an identity cut is named, which can clear it from a measurement.

## A note on paths

Source paths in these reports are relative to the repository root of the RFD4-Proteina fork
that carries the disease flow. **That code is not in this repository yet** — this repository
currently holds the upstream tree. The paths are given so each statement can be checked
against the source when it is published.
