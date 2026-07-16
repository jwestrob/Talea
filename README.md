# Talea

Talea is a sequence-only detector for elongated protein-solenoid architecture.
It uses two frozen views from a single ESM++ forward pass:

- a residue-state model operating on APC-corrected attention from zero-based
  transformer layers 13 and 35; and
- a global classifier operating on pooled hidden states from the same layers.

The primary stable discovery score is

```text
Talea score = 0.5 × teacher_state_z + 0.5 × global_z
```

The two inputs are standardized against frozen development controls. An
attention-topology decoder additionally reports candidate intervals, repeat
periods, and registered units, but its discontinuous score is auxiliary and
does not contribute to the primary ranking.

Talea requires no predicted or experimental structure at inference time.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The small Talea inference heads and calibrations are included in the package.
The pinned `Synthyra/ESMplusplus_large` weights are not vendored; Transformers
downloads them from Hugging Face on first use and then uses its normal cache.

## Run

Apple Silicon:

```bash
talea \
  --input examples/example.faa \
  --output predictions.jsonl \
  --device mps \
  --trust-remote-code
```

CUDA:

```bash
talea \
  --input proteins.faa \
  --output predictions.jsonl \
  --device cuda \
  --trust-remote-code
```

Once the pinned ESM++ revision is cached, add `--local-files-only` to prohibit
network access. `--device auto` selects CUDA, then MPS, then CPU. CPU
inference is supported but is generally impractical for large screens.

Talea writes one JSON object per protein and a sibling
`predictions.jsonl.audit.json` containing exact input, output, model, and
artifact checksums.

## Input contract

- Protein FASTA.
- The accepted alphabet is exactly `ACDEFGHIKLMNPQRSTVWY`.
- A single terminal `*` is removed as a FASTA stop marker.
- Any internal stop, ambiguity code, or other noncanonical character rejects
  the complete record. Talea never masks or imputes residues.
- The frozen v2 maximum is 2,046 residues.
- Sequences longer than the benchmark maximum of 1,732 residues are marked as
  extrapolations in the output.
- FASTA identifiers must be unique.

Predictions are emitted in length-sorted order to reduce padded-attention
memory. Use `record_id` rather than line order to join results back to inputs.

## Important output fields

- `talea_discovery_score`: primary ranking score.
- `teacher_state_probability` and `teacher_state_z`: residue-state view.
- `global_probability` and `global_z`: pooled hidden-state view.
- `topology_development_control_z`: auxiliary topology score.
- `wedge_call`: candidate repeat interval and period, when found.
- `registration`: proposed periodic units around the candidate interval.
- `status`: whether topology localization produced a wedge.

There is deliberately no hard solenoid threshold. Talea produces unverified
architecture rankings for follow-up with structural prediction or experimental
annotation; it does not emit curated biological calls.

## Frozen benchmark result

On the 452-protein pLM-Repeat evaluation set:

| Method | AP | AUROC | normalized pAUROC at FPR 0.1 |
|---|---:|---:|---:|
| Talea stable discovery v2 | 0.99598 | 0.99454 | 0.97564 |
| SOLeNNoID mean total probability | 0.98371 | 0.98308 | 0.89623 |

In 20,000 label-stratified homology-group bootstrap replicates, the 95%
intervals for Talea minus SOLeNNoID were positive for AP
(+0.00317 to +0.02426), AUROC (+0.00182 to +0.02283), and normalized pAUROC
(+0.02561 to +0.15189).

The stable-v2 fusion was defined after this cohort had been opened while
hardening deployment against batch instability. Its equal weights were not fit
to the cohort, but this result is post-opening evidence and must not be
presented as a new independent lockbox.

## Reproducibility

The repository contains only what is needed for inference:

- runtime source under `talea/`;
- the frozen 132-KB attention head and 2.4-MB global head;
- checksum-verified calibration and deployment contracts;
- focused runtime tests; and
- a toy FASTA smoke input.

Training pipelines, exploratory matrices, source datasets, comparator
installations, and generated benchmark results are intentionally not included.
Historical machine-readable identifiers may still contain the former project
name, Pangu, where changing them would break artifact provenance.

Run the local tests with:

```bash
pip install -e '.[test]'
pytest
```
