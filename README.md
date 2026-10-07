# Risk-aware planner for AVLite

Planning-only plugin: a fixed state lattice, rolling risk-constrained shortest-path
search, and optional trajectory smoothing. Community-registry submission is pending.

Community key: **p20_risk_aware_planner**. Strategy: **RiskAwarePlanner**.
Category: **LocalPlanningStrategy**. Host tested: AVLite 0.5.3, Python 3.10.
No ROS, GUI, scenario, profile, world bridge or controller is included.

This publication candidate is derived from that tested workspace; it has not
yet passed a new full compatibility/demo run. See [PUBLICATION.md](PUBLICATION.md)
for installation checks, simulation acceptance, release gates, and registry PR
instructions. Licensed under the [MIT License](LICENSE).

## Interface and responsibilities

`__init__.py` registers the local planner. `settings.py` exposes
`RiskAwareSettings`, with validated nested `lattice` and `csp` groups, to the
native AVLite settings loader. AVLite supplies the global route, ego pose,
map/corridor and obstacle perception. The adapter returns AVLite local
trajectory/velocity data; the existing controller executes it.

1. Build a source-to-goal lattice once per mission and prune nodes/edges that
   cannot connect source to goal. Geometry changes require a new mission.
2. Associate ego to its accepted current edge. Commit the remaining segment;
   choose the downstream node as the rolling CSP anchor.
3. Refresh obstacle risk, search anchor-to-goal under resource budgets, and
   optionally smooth the new suffix. Retain the committed prefix.
4. Validate the actual resulting geometry and age/pose constraints. Publish
   the desired trajectory, or a stop trajectory when validation fails.

`core/geometry.py`: route projection, graph construction and connectivity.
`core/corridor.py`: optional HDMap driving corridor extraction.
`core/edge_risk.py`: cached swept footprints and Stage-1 edge scores.
`core/risk.py`: legacy time-exposure/prediction model.
`core/solver.py`: multi-label resource-constrained search.
`core/trajectory.py`: committed progress, resource ledger and trajectory.
`core/smoothing.py`: optional OSQP B-spline suffix refinement and validation.
`core/worker.py`: background process, one active plus latest pending request.
`core/avlite_adapter.py`: input/output contracts, lifecycle, acceptance/stop.
`native_view.py`: thin native edge-display conversion only; no GUI imports.

The graph's legacy `local_trajectory` accessor delegates to `native_view.py`
so stock AVLite can still display graph edges. It creates data objects, not
Tk windows. All enhanced drawing is in the separate visualization companion.

## Install without the demo

Use AVLite's community-plugin local-directory mapping for this repository
root, not its `core/` directory. In a **copy** of a working profile:

```yaml
c69_apps:
  c62_load_plugins: true
  c62_community_plugins:
    p20_risk_aware_planner: /absolute/path/risk-aware-planner-avlite-plugins
c40_execution:
  c40_local_planner: RiskAwarePlanner
plugins:
  p20_risk_aware_planner:
    lattice:
      layer_spacing_m: 6.0
    csp:
      enable_motion: false
      smoothing_enabled: false
```

These are fragments to merge, not a complete replacement for the vehicle's
profile. Preserve existing perception, controller, bridge and executor.
Install `requirements.txt` into a dedicated compatible host environment;
install `requirements-smoothing.txt` only for the smoother. No companion
package is required to import, register, construct or run this planner.
The host's community import hook must be active before importing
`avlite.plugins.p20_risk_aware_planner.core` (normally done by native startup).

Use the examples' profile-copy CLI for an automated, non-overwriting merge.
Hardware integration remains observation-only; hardware motion is explicitly
rejected. Validate timestamps, coordinate conventions, footprint and axle
reference before even interpreting its results on a vehicle.

## Diagnostics contract v1

`planner.get_diagnostics()` returns a detached Python snapshot with
`schema_version=1`, `graph_id=(mission, graph_build_count)`, graph/route,
accepted desired path, committed remainder, edge risk, and report data.
Consumers cannot mutate planner geometry through it; obtaining it does not
replan, poll the worker or command a vehicle. It contains NumPy/dataclass
objects, so is not a JSON wire protocol. Rendering/serialization is consumer
owned. `visualization_snapshot()` is a compatibility alias.

`RiskAwareSettings` is the authoritative parameter schema. Defaults disable
motion and smoothing and retain the legacy risk mode; the example profile
explicitly selects `snapshot_edges`. Scores 1/0.01/0 are demonstration costs,
not calibrated collision probabilities. The five-layer unchecked tail is not
a full-route collision guarantee. Search limits can yield no feasible result
even if a geometrical connection exists. Stop reasons must remain visible.

## License

[MIT License](LICENSE). Copyright (c) 2026 Abdulrahman Hamdy Ahmad.
