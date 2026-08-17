# Vendored third-party code

This directory bundles the small parts of two external packages that
`museek.plugin.simulate_scan_plugin` (and the `inspect_simulated_scan` notebook) rely on,
so museek runs the scan simulation with no extra installs. Both upstreams are MIT-licensed
(© Zheng Zhang); their `LICENSE` files are kept alongside the code. museek itself is
GPL-3.0-or-later — bundling MIT code inside a GPL project is permitted as long as the MIT
notices are retained, which they are.

## `simeer/` — beam-convolved sky TOD

- Upstream: https://github.com/... (Simeer), version 0.1.0, author Zheng Zhang.
- Vendored from `/home/mgrsantos/projects/Simeer/simeer/` (local checkout).
- Used symbols: `MeerKLASSBeam`, `integrate_tod` (via `simeer.sky_integrator` /
  `simeer.beam`), exercised in `SimulateScanPlugin` and notebook §2g.
- Copied **verbatim**: `beam.py`, `sky_integrator.py`, `interpolation.py`, `stokes.py`,
  `projection.py`, `disc.py`, `_parallel.py` (internal imports are relative, so no
  rewrites were needed).
- Local edit: `__init__.py` had its `SimeerTODSim` lazy-loader removed and `simulator.py`
  is **not** vendored (that path needs limTOD + pygdsm and is unused by museek — museek
  only uses the Sky-TOD path).

## `limtod/` — 1/f noise and Gaussian-field HI mock

- Upstream: https://github.com/... (limTOD), author Zheng Zhang, MIT.
- Vendored from `/home/mgrsantos/projects/limTOD/limTOD/` (local checkout).
- Used symbols: `sim_noise` (`flicker_model.py`), `generate_gaussian_field`
  (`sky_model.py`), exercised in `SimulateScanPlugin` and notebook §7/§8.
- `flicker_model.py` copied verbatim; `sky_model.py` keeps **only**
  `generate_gaussian_field` (the upstream `GDSM_sky_model` and its `pygdsm` import are
  omitted to avoid a heavy unused dependency).

## Re-syncing with upstream

Keep these files as close to upstream as possible (they are excluded from museek's ruff
config for this reason). To update, re-copy the modules listed above and re-apply the two
local edits noted here.
