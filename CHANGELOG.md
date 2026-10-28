# Changelog

## Unreleased

- **Long proteins as windows.** `--long-proteins window` scores proteins
  above 2,046 aa as overlapping 2,000-aa windows every 1,000 residues (the
  giant-protein folding grid) and reports the best window in full-protein
  coordinates, with every window under `windowing`. The default remains
  `reject`, whose error now names the option. See README "Long proteins".
- **Process-based postprocessing is the default.** `--postprocess-mode` now
  defaults to `process` and `--postprocess-workers` to every CPU the process
  may use (the affinity mask, so Slurm allocations are respected). Output is
  identical to thread mode. On the H200 CATH lockbox timing (178,854 residues,
  12 workers) this runs at 6,582 residues/s against 1,580 for the previous
  thread default. `--postprocess-mode thread` restores the old behaviour.

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
