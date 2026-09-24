# Changelog

## 2.1.1 — 2026-09-20

- Applied the existing sampling-window exclusion shading consistently to both
  linear and log–log full-stage single-origin MSD plots. MSD values, origins,
  CSV output, and the separate multi-origin diffusion estimator are unchanged.

## 2.1.0 — 2026-09-20

- Added `coldmd fork` and launcher `fork` mode. A child run inherits cell,
  PBC, positions, atomic order, and explicitly stored finite momenta from one canonical
  parent `stage_structures/*.extxyz` snapshot while starting new step/time/V0,
  RNG, integrator, safety, config, calculator, and output state. Parent hashes,
  stage/event/step/time, volume, and temperature are retained as provenance.
- Added read-only `coldmd fork --check-only`, including parent/config/source
  validation, atom-order and momentum validation, calculator probing, and
  construction of every child stage integrator.
- Stage start/end/handoff publication now writes a fork-capable EXTXYZ and a CIF
  companion with the same stem. Resume reconciliation handles both formats.
- Changed displayed/written NVT MSD to a full-duration single-origin curve from
  the first stored stage frame. The diffusion table deliberately retains its
  separate multi-origin late-time fit and block uncertainty estimator.
- Connected P–V stage means in complete configured stage order. Branches now
  colour markers rather than define disconnected line segments, and missing
  stage data break the path. Removed automatic global P–T plot generation.
- Changed per-stage T/P or T/V traces to show the full stored stage duration and
  mark the sampling tail; CSV rows include `stage_time_ps` and
  `in_sampling_window`.
- Replaced the temporary two-launcher layout with one version-checked
  `launch_coldmd.sh` using the `coldmd-v210` environment by default. Updated
  installation, environment-clone, alias, fork, output, and analysis docs.
- Fixed the report's transport lookup (`msd` versus `msd_full_stage`), corrected
  the production stage count, and completed the source-distribution manifest.

### Compatibility notes

- Strict resume remains source/version fingerprint sensitive. Resume v2.0 runs
  with v2.0, or fork a v2.1 child from a v2.0 stage EXTXYZ.
- CIF files are intentionally rejected as fork sources because they cannot
  preserve momenta.
- The low-level P–T plotting function remains importable, but standard analysis
  no longer emits `pt_path_mean_sd.png`.

## 2.0.0 — 2026-09-10

- Replaced the production-time `benchmark` workflow with `coldmd dryrun`: it
  builds every configured integrator and runs a configurable, fixed-cell NVT
  smoke trajectory from the input state (100 steps by default).  It neither
  constructs a high-pressure/small-volume surrogate nor projects wall time.
- Replaced the three global sequence-design modes with stage-explicit v2
  protocol kinds: `nvt_hold`, `npt_hold`, and `volume_ramp`.  Each stage carries
  a research branch, purpose, and required sampling-window duration.
- Changed v2 stage handoff from the chronological final cell to the actual
  sampling-window snapshot whose volume is nearest the tail mean.  Cell,
  positions, and momenta are restored together; the selection and the rebased
  safety history are recorded in the event log and checkpoint metadata.
- Added full-stage NVT-only MSD CSV, linear, and log–log outputs.  Added
  sampling-window local-time T/P traces for NVT and T/V traces for NPT or
  prescribed-volume stages.  `output.msd_interval_steps` allows a finer NVT
  trajectory cadence without unnecessarily oversampling other stages.
- Updated the launcher, examples, package metadata, and tests for the v2
  dry-run and stage-explicit interfaces.  The parent v1.2 project and its
  calculation outputs are unchanged.

### Compatibility notes

- `config_version: 2` requires the stage-explicit vocabulary and rejects the
  old `protocol.control_mode` field.  v1 configurations remain readable only
  for historical/resume support.
- The command-line `benchmark` subcommand was removed.  The Python-level
  compatibility shim is safety-neutral and delegates to `dryrun`; it does not
  estimate a production duration.

## 1.2.0 — 2026-09-05

- Added selectable ASE `NoseHooverChainNVT` for fixed-cell NVT stages through
  `thermostat.kind: nose_hoover_chain`, with configurable `tau_fs`,
  `chain_length`, and `substeps`.
- Applied the selected fixed-cell thermostat consistently in production stages
  and representative benchmarks.
- Updated the `stage_sequence` and `pressure_plateaus` examples to use NHC NVT,
  matching the Nosé–Hoover thermostat family used by isotropic MTK NPT.
- Kept prescribed-volume compression/decompression on the existing Bussi
  integrator. Selecting NHC in a protocol containing a ramp is rejected because
  an NHC prescribed-ramp integrator is not implemented.
- Documented that a deliberate Bussi/NHC or ensemble change at an explicit
  stage boundary is valid, while its equilibration transient must be excluded
  and dynamical correlations must not span the boundary. No special
  mixed-thermostat runtime warning is emitted.
- Clarified checkpoint granularity: fixed-cell Bussi NVT and Bussi volume ramps
  retain interval resume; NHC NVT and NHC/MTK NPT resume from completed stage
  boundaries because their internal chain state is not publicly serializable.

### Compatibility notes

- `config_version: 1` and all dependency pins are unchanged. Existing Bussi
  YAML remains valid.
- Strict resume remains version- and source-fingerprint-sensitive. Resume an
  existing v1.1.0 run with its saved v1.1.0 installation.

## 1.1.0 — 2026-09-04

- Added `control_mode: stage_sequence`, which executes generic `nvt`, `npt`,
  `compression`, and `decompression` stages in exactly the written YAML order.
- Generic `nvt` holds the complete final cell of the preceding stage; no target
  volume is required or accepted.
- Generic `npt` requires an explicit numeric `target_pressure_GPa`; previous
  target or measured pressures are not inherited.
- Added a run-level `protocol.initialize_velocities` policy for sequences that
  do not begin with the legacy `initial_nvt` stage.
- Added `examples/stage_sequence.yaml` with alternating 6 ps NPT and 1 ps NVT
  stages along a 0→10→0 GPa path in 2.5 GPa increments.
- Extended stage-mean reporting so explicit generic-stage branches use the same
  loading/unloading aliases as legacy protocols; unlabelled generic stages use
  the neutral `other` branch.

### Compatibility notes

- `config_version: 1`, `volume_ramp`, `pressure_plateaus`, legacy stage kinds,
  and `pressure_range` remain supported without changing their order rules.
- Strict resume remains version- and source-fingerprint-sensitive. Resume an
  existing v1.0.0 run with its saved v1.0.0 installation.

## 1.0.0 — 2026-09-04

- Added a prominent, machine-readable warning when a downloadable MACE
  foundation alias makes strict resume unavailable.
- Added inclusive `pressure_range` YAML shorthand with deterministic expansion
  to canonical `pressure_plateau` stages.
- Added `launch_coldmd.sh` with `check`, `new`, and `resume` modes, a guarded
  independent 100-step preflight benchmark, JSON inspection, and `nohup`
  background execution.
- Replaced raw-point P–V output with sampling-window stage means and pressure /
  volume standard deviations; added matching P–T output and a unified P/V/T/
  density CSV.
- Added frame-wise Si-coordination oxygen classification and topology-resolved
  RDFs for NBO, BO, and other oxygen environments.
- Added explicit structural-stage selection (`--stage` or `--all-stages`) and a
  fixed Si–O cutoff override (`--si-o-cutoff-A`).

### Compatibility notes

- Existing explicit `pressure_plateau` YAML remains valid and
  `config_version: 1` is unchanged.
- v1.0.0 writes expanded plateaus to `resolved-config.yaml`, so execution,
  checkpoints, and reports always use the canonical stage list.
- Keep the v0.1.0 environment/wheel to resume a v0.1.0 run. Source and runtime
  fingerprints intentionally prevent strict cross-version continuation.
