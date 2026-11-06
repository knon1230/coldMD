# ColdMD project instructions

Apply these instructions to simulation guidance, analysis, code, configuration,
documentation, and Git work throughout this repository. Reply in the user's
language. Read the files relevant to the requested task before advising or
editing; use the map below to find them. When documentation and implementation
disagree, report the discrepancy and establish the actual behavior from code,
configuration, and tests before changing either.

## Project context

ColdMD is a Python 3.11–3.13 project using ASE and machine-learned interatomic
potentials (including MACE) for stage-based, isotropic cold compression,
decompression, and recovery MD. Read the repository's current documentation and
code before giving project-specific advice; do not rely on this summary if it
conflicts with the checkout.

The current v2 configuration is an ordered list of `nvt_hold`, `npt_hold`, and
`volume_ramp` stages. NVT keeps the cell fixed; NPT uses isotropic MTK pressure
control; a volume ramp prescribes isotropic cell changes and is not a pressure
controlled or equilibrium NVT stage. Each stage has a sampling tail. For a
following stage, the engine selects a real snapshot from that tail whose volume
is closest to the tail's mean and hands off cell, positions, and momenta
together. Do not silently change this handoff, reinitialize velocities at a
transition, or describe ramp dynamics as a pressure plateau.

## Repository map

Start with `README.md` for installation and commands, `docs/V2_WORKFLOW.md` for
scientific semantics, and the selected YAML for the actual experiment.
`VALIDATION.md` records historical checks, not results from the current session;
`CHANGELOG.md` records release history, including superseded behavior.

### Implementation (`src/coldmd/`)

| File | Responsibility |
| --- | --- |
| `__init__.py` | Release version and public configuration imports/compatibility aliases. |
| `__main__.py` | Routes `python -m coldmd` to the CLI. |
| `cli.py` | Argument parsing, JSON/text output, exit codes; delegates commands to `api.py`. |
| `config.py` | Strict Pydantic schema, v1 pressure-range expansion, v2 stage validation, duplicate-YAML-key rejection, YAML-relative paths, normalized config output. |
| `structure.py` | CIF block selection, occupancy rejection, supercell expansion, composition/count/PBC/cell/minimum-distance validation. |
| `calculators.py` | MACE and importable ASE factories, device/precision dispatch, local model hashing, element support and energy/force/stress probes, finite-difference stress-sign check. |
| `schedules.py` | Pure duration/step and log-volume schedules; absolute cells from the stage reference; decimal-safe inclusive pressure ranges. |
| `dynamics.py` | One-time momentum initialization, Bussi/NHC NVT, Verlet and isotropic MTK builders, Bussi volume ramp with forces refreshed after cell changes. |
| `engine.py` | Stage specifications, cursor/context, ordered integration and hook events, exact-resume capability, complete-snapshot tail handoff. |
| `observables.py` | Canonical thermodynamic record, stress/pressure/density units, kinetic stress, force/distance metrics; no file I/O. |
| `safety.py` | Non-mutating state/observable checks, consecutive-violation policy, checkpointable strain history and abort exceptions. |
| `storage.py` | Durable thermo CSV, segmented trajectories and metadata sidecars, JSON/NPZ checkpoints, runtime/source fingerprints, output/safety hooks, EXTXYZ/CIF publication. |
| `reconcile.py` | Transactional resume-prefix reconstruction, prior-attempt quarantine, reconciliation journal and rollback. |
| `api.py` | Validate/dryrun/run/fork/resume/analyze orchestration; output locks/backups, manifests, provenance, stage mapping, strict compatibility checks and failure handling. |
| `analysis.py` | Stored-data loading/deduplication, RDF/coordination/oxygen topology, MSD/diffusion/Fs, stage-tail statistics and full-stage thermo series. |
| `reporting.py` | Analysis stage/tail selection, CSV/PNG/JSON/Markdown artifacts, P–V paths, sampling shading, derived-output backups and diagnostic warnings. |

### Inputs, runtime, packaging, and assistant guidance

| File | Responsibility |
| --- | --- |
| `stage_sequence.yaml` | v2 fork path intended for a 15 GPa parent: NPT/NVT pairs at 22.5 and 30 GPa, then 22.5/15/10/5/0 GPa; 14 stages, each 11 ps with a 10 ps tail. |
| `stage_sequence_pilot.yaml` | v2 transition pilot: NPT/NVT pairs at 0/5/10/15 GPa, 8 stages totaling 2.4 ps; not an equilibrium protocol. |
| `CS_relaxed_NBO_BO.cif` | Fully occupied 260-atom Ca40Si60O160 input/identity template, reduced formula Ca2Si3O8; site labels do not define persistent oxygen topology. |
| `examples/stage_explicit_pressure_path.yaml` | v2 pressure-controlled NPT/NVT example with recovery. |
| `examples/stage_explicit_volume_path.yaml` | v2 driven Bussi volume-ramp example with fixed-cell holds; volume ratios refer to each ramp's start. |
| `examples/stage_sequence.yaml` | Legacy v1 freely ordered NPT/NVT sequence. |
| `examples/pressure_plateaus.yaml` | Legacy v1 pressure-range shorthand and recovery/aging holds. |
| `examples/volume_ramp.yaml` | Legacy v1 Bussi compression/decompression with optional MTK recovery. |
| `launch_coldmd.sh` | Conda/version/thread setup, fresh/fork preflight, JSON inspection, background launches and PID/log files. |
| `pyproject.toml` | Build/runtime pins, Python compatibility, optional MACE/test dependencies, `coldmd` entry point and source-package discovery. |
| `environment.yml` | Conda Python/build-tool bootstrap; hardware-specific Torch and project installation are separate README steps. |
| `requirements-cpu.txt` | CPU PyTorch wheel index and version. |
| `requirements-cu126.txt` | CUDA 12.6 PyTorch wheel index and version. |
| `requirements-cu130.txt` | CUDA 13.0 PyTorch wheel index and version. |
| `setup.cfg` | Egg-info build/date tag settings. |
| `MANIFEST.in` | Source-distribution inclusion of launcher, docs, YAML, environment files and CIF. |
| `.gitignore` | Python/build/test artifacts, local run/log directories and output lock files. |
| `.github/copilot-instructions.md` | Copilot-specific repository instructions; keep shared workflow facts consistent when changing them. |
| `.github/agents/coldmd-workflow-maintainer.agent.md` | Existing Copilot custom-agent definition; retains its own discovery format. |
| `AGENTS.md` | Codex repository instructions; keep project rules here rather than Copilot agent frontmatter. |

The examples use placeholder `structure.cif` paths and downloadable model aliases;
they are templates, not ready-to-run copies of the root experiment. Paths for
input CIF, local model, and output directory resolve relative to the YAML file.
The root continuation YAML does not enforce a fork-only entry point: verify the
intended parent/source rather than assuming `run` rejects it.

### Existing tests (`tests/`)

Choose tests by their assertions and touched behavior, not just filenames;
`*_v1.py` files also protect current interfaces and reports.

| File | Coverage |
| --- | --- |
| `test_config.py` | Strict configuration, thermostat/ramp compatibility, v1 ordering and pressure-range expansion, path resolution and YAML round trips; shared config fixtures. |
| `test_api_contracts.py` | Config-to-engine mapping, thermostat parameter forwarding, fresh velocity policy, strict-resume eligibility, foundation warnings and analysis option forwarding. |
| `test_cli_v1.py` | Analysis stage/cutoff argument parsing and mutually exclusive selection. |
| `test_cross_module_contracts.py` | Complete-cell inheritance, cursor/reference-volume integrity, trajectory discovery/sidecars/metadata precedence, sampling tails and config-section isolation; minimal-ASE test helper. |
| `test_dynamics_nhc.py` | Fixed-cell thermostat dispatch, RNG forwarding, real ASE NHC smoke trajectory and constraint/chain-parameter rejection. |
| `test_engine_nhc.py` | Integrator order, inherited mechanics, NHC/MTK interior-cursor rejection and resume capability. |
| `test_fork.py` | Source mechanics/lineage, zero-momentum preservation, rejected inputs/path overlap, read-only check-only behavior, child execution and CLI. |
| `test_launcher.py` | Fake Conda/CLI integration: preflight gating, background argument order, JSON checks, sibling output defaults and resume rejection. |
| `test_observables.py` | Pressure sign/units, Voigt ordering, shear and kinetic stress, consistent record construction. |
| `test_schedules.py` | Log-ramp endpoints/volume/symmetry and inclusive decimal-safe pressure targets. |
| `test_storage.py` | Checkpoint mechanics/RNG/hash/version contracts, paired stage files, output-tail priming, safety deduplication and extended-state checkpoint gating. |
| `test_reconcile.py` | Post-checkpoint thermo/event/paired-structure quarantine and durable rollback. |
| `test_analysis_v1.py` | Stage-order/tail means/sample SD, pressure fallback, duplicate handling, frame-wise oxygen topology and RDF normalization. |
| `test_reporting_v1.py` | P–V branch connections, representative-stage selection, config discovery, report artifacts and full-stage thermo/MSD output. |
| `test_v200.py` | v2 schema/round trip, sampling masks, single-origin MSD/shading, representative handoff and rebased checkpoint safety history. |

## Protocol and scientific invariants

- Use `config_version: 2` and its stage vocabulary for new protocols. Preserve
  v1 parsing/normalization for historical data; do not migrate an existing run's
  resolved configuration in place. v2 rejects explicit `protocol.control_mode`.
- Stage names are unique. Durations and sampling windows must be integral
  timestep multiples, with the tail no longer than its stage. `branch`/`purpose`
  are research metadata, not integrator selectors; ramp direction must still
  agree with its volume ratio. Keep Kelvin, fs, ps, GPa, eV and Angstrom units
  explicit across configuration, dynamics and output.
- NVT inherits the complete cell, including angles and axial ratios. NPT builds
  a new isotropic MTK integrator for each target pressure. Prescribed volume
  ramps require Bussi; NHC plus any ramp is rejected. Compute a ramp cell from
  its stage-start reference and refresh forces after cell/position mutation.
- In v2, handoff candidates use the thermo interval within the sampling tail,
  including the final step (and initial boundary for a full-stage tail). Select
  nearest-to-mean volume, breaking ties toward the later snapshot. Restore
  cell/positions/momenta together without rewinding the protocol clock. The
  chronological `end` and selected `handoff` are different states; handoff rebases
  safety history and writes a same-boundary checkpoint for the following stage.
- ASE stress is tensile-positive: pressure is `-trace(stress)/3`, converted
  from eV/Angstrom^3 to GPa. Preserve the distinction between configurational
  stress/pressure and total pressure including the kinetic contribution.
- Safety checks report/abort; never fix a failing run by silently clipping
  forces, rescaling velocities, repairing cells, or disabling nonfinite guards.
  Diagnose the structure, potential, timestep and protocol before changing limits.

## Simulation use and scientific safeguards

- Explain the distinction between `check`, `new`, `fork`, `resume`, and
  `analyze` using the README and CLI. `check` runs YAML/calculator validation
  followed by a fixed-input-cell NVT smoke trajectory; it does not execute the
  configured production stage sequence or estimate wall time. Recommend it
  before a new run.
- Be explicit that launcher `check` writes preflight reports/logs beneath
  `LOG_DIR` (default `../coldmd-logs/`), creates configured parent directories,
  and can load the MLIP and execute a short smoke trajectory. It is
  non-production, not side-effect-free. Inspect the selected config, model,
  environment, and output paths before invoking it.
- Never start `new`, `fork`, or `resume`, submit a production calculation, or
  launch a potentially long-running simulation unless the user explicitly asks
  to start it. A `check`/dry-run is non-production, but still inspect its
  command and resource implications first.
- The launcher activates `COLDMD_ENV` (default `coldmd-v212`), requires the
  exact version declared in `launch_coldmd.sh`, sets CPU thread variables, and
  starts `new`/`fork` in the background only after preflight. Check the current
  launcher before recommending environment names, version numbers, or resource
  settings; do not assume they match an older changelog entry.
- Treat run output and model files as external data. The default run/log
  directories are siblings of this repository (`../coldmd-runs/` and
  `../coldmd-logs/`); do not move, overwrite, clean, or delete run data or model
  files unless the user explicitly requests the exact operation and target.
- Calculator validation probes the configured energy/forces/stress capabilities.
  MACE foundation aliases require explicit download opt-in and may fetch model
  data; prefer a local model checkpoint with a verified SHA-256 for reproducible
  runs and strict resume. Never enable downloads or change model identity
  implicitly.
- `coldmd validate CONFIG --skip-calculator-check` skips calculator construction
  and numerical probes, but still reads/validates the CIF and hashes a configured
  local model. Schema/path inspection alone uses `config.load_config`; that
  does not validate CIF/model contents or establish readiness.
- `resume` is for continuing the same compatible run with its exact checkpoint,
  configuration, model, and runtime. For a changed protocol or calculator,
  explain why a new `fork` is appropriate instead. Exact mid-stage resume
  capability depends on integrator/checkpoint support. NHC/MTK interiors lack
  serialized extended state. All current v2 stages also require boundary-only
  exact checkpoints because the handoff candidate pool is not serialized, even
  with Bussi. Do not promise interval resume merely because a checkpoint cadence
  is configured. Resume may quarantine post-checkpoint outputs before replay;
  a failed resume can also update lifecycle metadata.
- Strict resume currently admits only local, content-hashed MACE checkpoints,
  not arbitrary factories or foundation aliases. `model_sha256` is an optional
  expected-hash assertion; the actual local file is always hashed. Config,
  input CIF, source bytes, runtime dependency closure and accelerator fingerprint
  must match. A source edit can invalidate resume without a version bump.
  Preserve the original checkout/environment when existing runs need continuation.
- `fork` starts a new child calculation from a canonical parent
  `stage_structures/*.extxyz` state. Preserve its cell, PBC, positions, and
  momenta; CIF files do not preserve momenta and are not valid fork states.
  It uses a new child YAML, calculator/protocol, RNG and integrator state, and
  starts child step/time/V0 coordinates at 0/0/1 while recording source
  provenance. Parent runs are read-only. `coldmd fork --check-only` validates
  the source and child setup without starting MD.
- Fork source acceptance preserves explicitly stored all-zero momenta, but
  thermostatted integrator construction still requires positive kinetic energy.
  Do not redraw velocities to hide this incompatibility. Parent and child output
  directories must be separate and non-nested.
- Distinguish software readiness from scientific validity. A successful
  preflight, smoke run, or test suite does not establish equilibrium, potential
  accuracy under compression, or scientific reproducibility. Do not make those
  claims without appropriate trajectory evidence and DFT/reference validation.
- For scientific or protocol changes, inspect stage ordering, pressure/volume
  paths, units, integrators, sampling windows, and source-state provenance.
  Prefer a short pilot only when requested; do not substitute it for
  production or scientific validation.

## Analysis and durable-output contracts

- Analyze stored run data without modifying raw trajectories, thermo records or
  provenance. `analyze` writes derived artifacts; `--force` backs up an existing
  nonempty analysis directory. Respect configured output filenames and retain
  trajectory `.meta.jsonl` sidecars with their segments.
- Structural RDF/coordination/topology defaults to representative stages and
  their configured sampling tails; `--stage`/`--all-stages` changes structural
  selection. Thermodynamic summaries and per-stage traces still cover all
  stages; full-stage MSD is attempted for every identified NVT stage.
- Published NVT MSD uses the first stored stage frame as one origin and spans
  the full stored stage. Diffusion uses a separate multi-origin late-time fit
  with block uncertainty. Preserve fractional unwrapping, fixed-cell checks,
  uniform time-spacing requirements and timescale-qualified classifications.
  Do not join dynamical correlations across stage/ensemble boundaries.
- P–V means connect consecutive stages in protocol order across branch changes;
  missing means break the line. Statistics use tail means and sample SD
  (`ddof=1`, unavailable for fewer than two observations), preferring total
  pressure with configurational pressure only as a legacy fallback.
- Keep `sampling_thermo.csv` and `sampling_thermo_T_{P,V}.png` names, although
  their contents span the full stage. Preserve `stage_time_ps`,
  `in_sampling_window` and excluded-region shading, including both MSD plots.
  Standard reports do not generate a global `pt_path_mean_sd.png`.
- Recompute NBO/BO/O-other labels from Si coordination in each analyzed frame,
  using stage-specific Si–O RDF minima or an explicit cutoff. Do not infer them
  from CIF O1/O2 labels. Preserve triclinic minimum-image cutoff limits and
  frame-wise pair-population normalization; missing diagnostics remain explicit
  warnings. Loss of crystallinity alone does not establish a thermodynamic phase.
- Preserve checkpoint commit order (immutable generation NPZ first, JSON pointer
  last), hash checks and pickle-free loading. `emergency` checkpoints are forensic,
  not normal resume inputs. Reconciliation retains prior attempts under
  `discarded_attempts/`; never replace it with destructive tail cleanup or bypass
  an incomplete transaction journal.

## Project maintenance

- Before editing, inspect `git status`, relevant code/docs/tests, and the
  specific configuration or output paths involved. Preserve all user changes
  and unrelated work.
- Make the smallest complete change. Keep scientific behavior and on-disk
  formats stable unless the request requires a change; update focused tests and
  directly related documentation when behavior or interfaces change.
- Keep dependency versions and Python compatibility aligned with
  `pyproject.toml` and the repository's CPU/CUDA environment files. Explain
  runtime/environment implications before changing the MLIP, PyTorch, ASE,
  CUDA, or model-loading configuration.
- Prefer targeted existing tests. The documented full test command is:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests MPLCONFIGDIR=/tmp/coldmd-mpl \
    python -m unittest discover -s tests -q
  ```

  For launcher changes, also use `bash -n launch_coldmd.sh`. Run only checks
  appropriate to the changed code and available environment.
- Run focused tests from the repository root, for example:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests MPLCONFIGDIR=/tmp/coldmd-mpl \
    python -m unittest test_config test_v200 -q
  ```

  Broaden to the full suite for shared protocol/engine/storage contracts. Report
  skips or missing dependencies; minimal-ASE stubs and fake launcher executables
  do not establish MACE or production readiness. A documentation-only edit needs
  relevant path/fact/diff checks, not a production preflight or automatic MD run.
  There is no configured repository linter/formatter. Do not change environments
  or install/upgrade the numerical stack just to validate prose.
- Inspect the final diff and report changed files, relevant behavior, and any
  checks that could not be run. Do not claim success for commands that were not
  run or results not observed.

## Git and release hygiene

- Inspect the branch, working tree, and recent history before Git operations.
  Never discard user changes or use destructive cleanup/reset/force operations.
- Do not commit, push, create tags/releases, or alter remotes unless explicitly
  requested. A request to commit does not imply permission to push.
- Keep commits scoped to the requested change. For release/dependency updates,
  check version references, changelog, tests, installation metadata, and
  launcher version requirements for consistency; do not bump a version just
  because a file was edited.
- Treat paths, hashes, and names in run manifests as provenance. Do not rewrite
  historical run metadata to make it match a new code or model version.
