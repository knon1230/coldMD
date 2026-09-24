# Validation status — ColdMD 2.1.1

Validation date: 2026-09-20

## Automated checks in the v2.1.1 project copy

- Source compilation:
  `PYTHONPATH=src python -m compileall -q src tests` — passed.
- Shell syntax: `bash -n launch_coldmd.sh` — passed.
- Full unit/integration/launcher discovery in the pinned `coldmd-v200`
  dependency environment with the v2.1.1 source tree first on `PYTHONPATH` —
  **124 tests passed**:
  `MPLCONFIGDIR=/tmp/coldmd-v211-mpl PYTHONPATH=src:tests conda run -n coldmd-v200 python -m unittest discover -s tests -v`.
- The shipped production YAML resolves **26 stages** and 286,000 MD steps.
- A real 260-atom v2.0 handoff EXTXYZ was loaded read-only through the v2.1 fork
  path. Parent manifest/config hashes, atom identity/order, cell, positions, and
  momenta validated; the stored state produced a finite 292.219 K source
  temperature. No child run directory was created.
- Contract coverage includes fork provenance and local-origin reset, rejection
  of CIF/position-only fork sources, preservation of explicit zero-momentum
  states, paired EXTXYZ/CIF publication, full-stage
  single-origin MSD with matching linear/log–log sampling-window shading,
  separate multi-origin diffusion estimation, full-stage
  thermo plots with sampling masks, stage-order P–V connectivity, omitted global
  P–T output, and the single-launcher fork gate.

The existing v2.0 local-MACE 100-step dry-run evidence remains applicable to the
unchanged pinned calculator/dynamics dependency set, but it is not a substitute
for a v2.1.1 preflight in the newly installed `coldmd-v211` environment.

## Required before production

1. Create/activate `coldmd-v211`, uninstall the old package from that environment,
   and install this project as described in `README.md`.
2. Run `python -m pip check` and the full test command above inside that new
   environment.
3. Run `bash launch_coldmd.sh check CONFIG.yaml` for each fresh-run YAML.
4. For a branch, run `coldmd fork ... --check-only --json` or the launcher's
   `fork` mode against the exact selected parent EXTXYZ.
5. Run a short scientific pilot visiting every intended pressure/volume extreme.
6. Exercise strict resume on a copied v2.1.1 pilot when a local content-hashed MACE
   checkpoint is used.
7. Validate representative compressed, branched, and recovered structures with
   suitable DFT single points.

For NVT MSD, choose `output.msd_interval_steps` from the shortest dynamical
regime of interest. Inspect the full-stage single-origin linear/log–log curves,
and treat the independently estimated multi-origin diffusion classification as
time-scale qualified.
