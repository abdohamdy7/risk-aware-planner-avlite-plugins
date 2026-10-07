# Publication and verification

Status: public source candidate; not yet a tagged release or community-registry listing.
Repository: https://github.com/abdohamdy7/risk-aware-planner-avlite-plugins
Licensed under the [MIT License](LICENSE), selected by the repository owner.
See VERIFICATION.md for completed compatibility checks and known companion-demo
limitations. Community-registry acceptance remains pending maintainer review.

## Scope

This repository is the planner only. The repository root must be the directory
containing `__init__.py`, not its parent and not `core/`. Keep the registry key
`p20_risk_aware_planner` and strategy name `RiskAwarePlanner` stable.
The separate examples and visualization repositories are optional companions.
No AVLite core, controller, perception, or world bridge changes are required by
this release. Do not copy machine-local runtime profiles into this repository.

## Check installation without starting the simulator

Use a disposable environment. The original workspace was validated with Python
3.10 / AVLite 0.5.3; the release copy also passed all 144 regression checks on
that published host version (see VERIFICATION.md).
Do not interpret successful imports as driving validation.

```bash
conda create -n risk-aware-release-check python=3.10 -y
conda activate risk-aware-release-check
python -m pip install 'avlite==0.5.3'
cd /absolute/path/risk-aware-planner-avlite-plugins
python -m pip install -r requirements.txt
```

Use a temporary configuration directory so your normal profiles are untouched:

```bash
export AVLITE_CONFIG_DIR="$(mktemp -d -t risk-aware-config-XXXXXX)"
python -m avlite plugins
```

Before publication use the Config interface to register the absolute repository
path under `p20_risk_aware_planner`. Save a **copy** of a working profile, select
`RiskAwarePlanner` as local planner, and reload. Retain the original bridge,
controller, perception and global planner. Expected: no import/reload error;
the local planner dropdown contains RiskAwarePlanner; motion is disabled.

Set lattice.layer_spacing_m to 8 in plugin settings, save, restart with the same
copied profile, and confirm it is 8. Restore 6 for the demo. Verify csp and lattice
remain typed validated models and malformed values are rejected. Confirm settings
are written outside this repository. AVLite settings storage differs by version;
use that version's native Save/Profile interface, not hand-created plugin-local
YAML files.

## Functional acceptance (simulation only; separate from installation)

1. Use the companion examples' copied demo profile and saved >=50 m route.
   Change its planner mapping to THIS release checkout. Verify the imported
   planner module's `__file__` points here, not the original checkout.
2. With motion disabled, build the graph. Check source-to-goal connectivity and
   absence of disconnected branches. Replan repeatedly: graph ID/build count
   must remain stable until mission/reference geometry changes.
3. Use the optional visualization companion to inspect the desired path. Place
   a static obstacle with a feasible alternative; snapshot-edge mode should
   update scores and accept a path within its configured risk budget.
4. In BasicSim ONLY enable motion. Confirm the current edge remainder is retained
   and replanning changes the suffix from the downstream anchor, not ego's edge.
5. Install `requirements-smoothing.txt`; repeat with smoothing_enabled false and
   true. Check diagnostics, committed-prefix preservation and validation/fallback.
6. Check blocked-route, stale-input and off-edge cases stop as designed. Reset
   must start a new mission. No physical vehicle motion is part of this check.
7. Repeat on every AVLite release advertised as supported. Record AVLite/Python
   versions, OS, test results and profile settings. A minimum-version registry
   field is not a guarantee of compatibility with all newer releases.

The planner uses bounded multi-label CSP search, not Gurobi. OSQP is used only
for optional smoothing. Snapshot edge scores are demo costs, not calibrated
collision probabilities; unchecked graph tails are not certified collision-free.

## Release checklist

- [x] Owner-selected MIT LICENSE added.
- [ ] Asset attribution review before redistributing additional demo assets.
- [ ] Planner-only import, registration, settings reload and functional checks.
- [ ] CI or recorded compatibility results for each supported host version.
- [x] Public source repository with this directory at its root.
- [ ] README links to published optional examples/visualization repositories.
- [ ] Create v0.1.0 only after validation; no tag is created by this preparation.
- [ ] Replace the proposed repository/author in the registry draft if necessary.
- [ ] Add min_avlite_version only after confirming the supported floor.
- [ ] Fork AV-Lab/avlite-community-plugins and append the registry entry to
      plugins.yaml in alphabetical name order; preserve all existing entries.
- [ ] Open a registry PR linking the source release and verification results.
- [ ] After merge, verify Community > Install > Register in a fresh environment.

The registry PR is needed for Community-browser discovery, not local use.
No PR to AVLite core is needed. A source-repository PR is optional if your team
uses review before releasing. The presentation PR is unrelated to registration.

References:
- https://avlite.org/plugin-development/
- https://github.com/AV-Lab/avlite-community-plugins

The website and registry currently differ on category names. The registry README
lists LocalPlanningStrategy; use its current schema when submitting. This plugin
does not require ROS even when used with a separate ROS vehicle bridge.
