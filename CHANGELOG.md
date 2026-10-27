# Changelog

## Unreleased

- **Experimental `--unit-placement start-trim`.** Unit-start and repeat-region
  probes on layer 35 (`talea/assets/start_trim.json`): helix-rich extents decode
  units from attention stacking plus a unit-start term, and end units with a
  low repeat-region score are trimmed. Boundary F1 vs phase-SSM: +0.053
  [+0.021, +0.083] on the 571-protein test partition, +0.026 [+0.005, +0.048] on
  the 597-protein fresh holdout; matched-unit IoU +0.011 and −0.003. Opt-in;
  the default stays `phase-ssm`.

## 0.2.0

- **Phase-aware unit placement.** Registered repeat units are now placed by a
  state-space decoder over per-residue helix/strand/loop phase votes
  (`talea/phase_ssm.py`, `talea/assets/phase_ssm.json`). On the held-out
  RepeatsDB class 3 test partition (571 proteins), boundary F1 rises from 0.375
  to 0.549 (+0.173 [+0.124, +0.223]). `--unit-placement comb` restores the
  v0.1.0 units. Detection scores, the wedge interval, the period, closure, and
  the boundary scan are unchanged.
- **Performance.** Selective capture of the two attention layers Talea reads,
  vectorized postprocessing kernels, and process-based postprocessing
  (`--postprocess-mode process`). Default output is byte-identical to v0.1.0's
  full-capture path; concordance tests keep the original kernels as oracles.
- **Reproducibility options.** `--deterministic` pins cuDNN convolution
  algorithms; `--capture-mode sdpa` trades bit-identity for speed;
  `--cache-clear-interval` releases accelerator caches on shared GPUs.
- `robust_attention_channel` accepts a versioned normalization scope; the
  default reproduces v0.1.0.

## 0.1.0

- Initial release: stable v2 discovery score (residue-state U-Net and global
  head on ESM++ layers 13 and 35), auxiliary attention-topology localization.
