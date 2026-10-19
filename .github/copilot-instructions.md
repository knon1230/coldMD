# Copilot instructions for coldMD

## Build, test, and validation commands

This repository is a Python 3.11+ package built with setuptools and pytest. The editable install used by contributors is:

```bash
cd /data/Python_MLMD_DATA/LSK/coldMD
python -m pip install -e ".[mace,test]"
python -m pip check
```

Recommended test invocation for a focused check:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests MPLCONFIGDIR=/tmp/coldmd-mpl \
  python -m pytest tests/test_config.py -q
```

Use the full suite when the change touches shared protocol or engine logic:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests MPLCONFIGDIR=/tmp/coldmd-mpl \
  python -m unittest discover -s tests -q
```

Validation and preflight commands are part of the project workflow:

```bash
coldmd validate stage_sequence_pilot.yaml
coldmd dryrun stage_sequence_pilot.yaml --nvt-steps 100
bash -n launch_coldmd.sh
```

There is no dedicated linter or formatter configuration in this repo; rely on the existing Python tests and shell syntax checks with the project-specific validation commands above.

## Project architecture

Read the project docs before changing physics or protocol behavior:

- `README.md` for installation, launcher usage, run/fork/resume, and analysis workflow
- `docs/V2_WORKFLOW.md` for the stage semantics and handoff rules
- `VALIDATION.md` and `CHANGELOG.md` for validated behavior and release history

The package structure is centered on a stage-explicit MD protocol:

- `src/coldmd/config.py`: validated YAML schema and stage definitions; this is the source of the v2 stage-explicit protocol (`nvt_hold`, `npt_hold`, `volume_ramp`)
- `src/coldmd/api.py`: the high-level orchestration layer for `validate_config`, `dryrun`, `run`, `fork`, `resume`, and `analyze`
- `src/coldmd/engine.py` and `src/coldmd/dynamics.py`: MD stage execution, integrators, thermostats/barostats, and volume-ramp logic
- `src/coldmd/calculators.py`: ASE calculator construction and MACE model probing/verification
- `src/coldmd/storage.py`: run manifests, checkpoints, provenance, and resume/fork metadata
- `src/coldmd/analysis.py` and `src/coldmd/reporting.py`: analysis outputs, plot generation, and report files
- `src/coldmd/cli.py`: the public `coldmd` command-line interface
- `launch_coldmd.sh`: repo wrapper for `check`, `new`, `fork`, and `resume` launcher flows

The lifecycle is intentionally explicit: configuration validation, a non-production dryrun/preflight, a fresh run, a fork from a canonical stage snapshot, a strict resume from an exact checkpoint, and a post-run analysis step.

## Key conventions specific to this repository

- ColdMD v2 does not use a single global control mode. Each YAML stage declares its own control behavior. The `branch` field (`compression`, `decompression`, `recovery`, etc.) is a research/path label for plotting and analysis; it does not redefine the actual MD integrator.
- Stage transitions use a real snapshot from the sampling tail, not a synthetic reset. The chosen state is the complete snapshot whose volume is closest to the tail mean; the engine carries forward the cell, positions, and momenta together. Do not silently reinitialize velocities at a stage boundary or treat a ramp as a pressure plateau.
- `fork` creates a new child run with lineage metadata; the parent run remains read-only. The source state must be a canonical `stage_structures/*.extxyz` file with finite momenta; CIF files are not valid fork states because they do not preserve momentum.
- Strict resume requires a matching config, input CIF, local calculator model bytes/checksum, and runtime fingerprint. If the protocol or calculator changes, prefer a new `fork` instead of resuming the old run.
- Project outputs and logs default to sibling directories next to the repo: `../coldmd-runs/` and `../coldmd-logs/`. Respect external run data and model files; do not overwrite or delete them without explicit intent.
- Model identity matters for reproducibility. Prefer a local `model_path` with a recorded `model_sha256` for production runs; MACE foundation aliases are convenient but not strict-resume-safe unless the exact local bytes are pinned.
- The repo's tests are focused on configuration contracts, storage semantics, engine behavior, analysis, and CLI compatibility; keep changes small and validate the affected module or test file first.

## Repository-specific workflow guidance

Before making scientific or protocol changes, inspect the selected YAML and the relevant stage logic in `config.py`, `engine.py`, and `docs/V2_WORKFLOW.md` together. Small changes in scheduling, sampling windows, branch naming, or handoff state can materially alter the analysis and run lineage.

When editing, preserve compatibility with the established output formats (`run-manifest.json`, `resolved-config.yaml`, `thermo.csv`, stage `*.extxyz`/`.cif`, and analysis CSV/PNG outputs) unless the task explicitly requires a format change.
