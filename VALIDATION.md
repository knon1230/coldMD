# Validation status — ColdMD 2.1.2

Validation date: 2026-10-04

## Checks completed for this patch

- `bash -n launch_coldmd.sh` passed.
- The `coldmd-v212` editable installation imports this Git checkout, and
  `python -m pip check` reports no broken requirements.
- The full suite passed: **125 tests**, including a launcher test that verifies
  the default run and log directories are siblings of a checkout. The command
  used the installed `coldmd-v212` environment with this repository first on
  `PYTHONPATH`:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests MPLCONFIGDIR=/tmp/coldmd-v212-mpl python -m unittest discover -s tests -q`.
- `stage_sequence.yaml` resolves to **14 stages, 154,000 MD steps**;
  `stage_sequence_pilot.yaml` resolves to **8 stages, 2,400 MD steps**. Their
  configured output paths resolve outside the Git repository.
- Git ignore rules were checked for generated lock files, Python bytecode,
  package metadata, run outputs, and preflight logs. Existing generated files
  were removed from Git tracking without deleting them from disk.
- MD integrators, calculator handling, handoff, and analysis algorithms were
  not changed in this patch. No new production trajectory was run.

## Before a v2.1.2 production run

1. For a fresh run, use a YAML beginning at the intended initial state and run
   `bash launch_coldmd.sh check CONFIG.yaml` before `new`.
2. For the bundled 15 GPa continuation YAML, run launcher `fork` against the
   intended parent EXTXYZ. Its preflight validates the source and all child
   stage integrators.
3. Check that pressure and volume have stabilized before treating an NPT
   sampling window as equilibrated. The bundled 11 ps stage excludes only its
   first 1 ps from the 10 ps sampling window.
4. Validate representative compressed and recovered structures against suitable
   DFT references when making scientific claims.

Existing v2.1.1 runs should be strictly resumed with their original v2.1.1
installation. Their canonical stage EXTXYZ snapshots can be fork sources for a
new v2.1.2 run.
