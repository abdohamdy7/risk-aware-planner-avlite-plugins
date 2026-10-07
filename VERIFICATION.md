# v0.1.0 verification — 2026-10-07

Tested source: f9d220b17d4c88a6a9017af0f5d2688ae26ba29b. Subsequent release
changes add documentation only; planner code is unchanged.

## Results

| Host installed from PyPI | Python | Regression result |
| --- | --- | --- |
| AVLite 0.5.3 | 3.10 | 144 passed, 46.07 seconds |
| AVLite 0.6.4 | 3.10 | 139 passed, 5 failed, 25.77 seconds |

Both fresh virtual environments passed explicit import/registration, source-origin,
safe-default, capability and nested-settings checks. Dependencies included NumPy
2.2.6, SciPy 1.15.3, Pydantic 2.13.5, Shapely 2.1.2 and OSQP 1.1.3.

The existing separated-workspace regression suite was copied into an isolated
test workspace, with this release's plugin and separate examples/visualization
companions. No existing vehicle profiles or original source were modified.
The tests are not bundled in the plugin runtime repository; these results are
recorded local acceptance evidence, not a claim of public CI coverage.

Coverage includes fixed-graph connectivity, committed-edge replanning, edge risk,
blocked paths, stale state, worker acceptance, typed profile loading, optional
smoothing, native-controller closed loops, and the saved >=50 m SAN Campus demo.
The 0.5.3 saved-demo tests include reaching the goal twice with fresh stacks and
obstacle scenarios with smoothing. No physical vehicle was commanded and no GUI
click-through was performed in this verification session.

## Known AVLite 0.6.4 companion incompatibilities

Five failures occurred outside the planner-only boundary:

- Three saved-demo integration tests fail because their existing profile/bootstrap
  does not register `MapReader` with the newer host's mapping registry.
- Two virtual-obstacle/preview integration tests use the old no-argument
  `_perception_step()` call; the new host requires `sensors`.

The other 139 tests passed, including native synthetic closed loops and planner
and smoothing checks. This does not establish full GUI/demo compatibility on
0.6.4. Use 0.5.3 for the currently verified complete companion-demo workflow.
Intermediate host versions and physical vehicle operation are not certified.

## Safety and scope

Motion and smoothing default to disabled. Hardware motion remains unsupported.
Demo risk scores are not calibrated collision probabilities; the unchecked graph
tail is not a full-route collision guarantee. A registry listing is discoverability,
not safety certification. Keep the original vehicle controller and bridge unchanged.
