# Changelog

## Unreleased

- **Every repeat region.** Rows gain `repeat_regions`: the registered primary
  region plus further regions found by searching the wedge again after
  neutralizing each call (`secondary_wedge_calls`), kept when the U-Net
  residue-state track supports them. On 455 RepeatsDB chains with two to four
  curated solenoid regions, coverage of curated regions rises from 60.0% to
  76.7% at unchanged residue precision; single-region development proteins
  keep their residue F1. Window mode reports the union of every window's
  supported regions and lists each window's regions under `windowing`. Detection
  scores, the primary call, registration and units are unchanged. See README
  "Repeat regions".
- **Faster wedge search.** Each search computes every diagonal's interval
  sums once and scores all candidate periods of a width together, and the
  repeat-region search reuses one template plan per protein. Output is
  bit-identical (the original search is kept as the concordance oracle).
  Single-threaded postprocessing of 240 Omnitrophota proteins takes 15.6 s
  with repeat regions, against 18.3 s for v0.2.0 without them and 37.4 s for
  repeat regions on the previous search; at 1,800-2,046 aa (24 proteins) it
  takes 12.5 s, against 9.9 s and 22.8 s.
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
  Worker processes re-import the calling script, so API scripts must call
  Talea under `if __name__ == "__main__":` (the README example now does); a
  broken worker pool raises an error that says so.

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
