"""Thin AVLite LocalPlanningStrategy; owns planning state, never vehicle control."""
from collections import deque
import copy
import logging
import math
import threading
import time
from types import SimpleNamespace
import numpy as np
from avlite.c10_perception.c11_perception_model import HDMap
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan, LocalBehavior
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c50_common.c51_capabilities import StackCapability
from .config import CSPSettings, LatticeSettings
from .geometry import LatticeGraph, route_signature
from .risk import RiskModel
from .edge_risk import FootprintCache, SnapshotEdgeRisk
from .trajectory import Progress, ResourceLedger, edge_key, export_waypoints, retime, resources_of, within_budget
from .worker import BackgroundWorker, Request

log = logging.getLogger("avlite.plugins.p20_risk_aware_planner")


def snapshot(pm):
    return SimpleNamespace(ego_vehicle=copy.deepcopy(pm.ego_vehicle),
                           agent_vehicles=copy.deepcopy(list(pm.agent_vehicles)),
                           static_obstacles=copy.deepcopy(list(pm.static_obstacles)),
                           prediction=copy.deepcopy(pm.prediction))


class CommittedEdgeCSPPlanner(LocalPlanningStrategy, abstract=True):
    # Consume perception data, not raw sensors; preserve the native controller.
    world_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCAL_PLAN})
    stack_requirements = frozenset({StackCapability.GLOBAL_PLAN, StackCapability.LOCALIZATION,
                                   StackCapability.DETECTION, StackCapability.TRACKING})

    def __init__(self, global_plan, pm=None, env=None, setting=None, *,
                 planner_settings=None, lattice_settings=None, worker_factory=BackgroundWorker):
        self.pm = env if env is not None else pm
        if self.pm is None:
            raise ValueError("PerceptionModel is required")
        # Accept profile dictionaries as well as typed settings, then own a copy.
        # Only None selects defaults; malformed inputs must fail validation.
        self.cfg = CSPSettings.model_validate(
            {} if planner_settings is None else planner_settings
        ).model_copy(deep=True)
        self.lattice_cfg = LatticeSettings.model_validate(
            {} if lattice_settings is None else lattice_settings
        ).model_copy(deep=True)
        self._worker_factory, self._worker = worker_factory, None
        self._lock = threading.RLock()
        self.mission, self.sequence = 0, 0
        self.history = deque(maxlen=2000)
        self.last_report = {}
        self.set_global_plan(global_plan)

    def close(self):
        with self._lock:
            if self._worker is not None:
                self._worker.close()
                self._worker = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def set_global_plan(self, global_plan, ego_xy=None):
        with self._lock:
            self.close()
            self.mission += 1
            self.global_plan, self.global_trajectory = global_plan, global_plan.trajectory
            self._signature = route_signature(global_plan)
            self._initial_ego = copy.deepcopy(self.pm.ego_vehicle)
            self._last_ego = copy.deepcopy(self.pm.ego_vehicle)
            self._input = None
            self._geometry_config = None
            self._map_identity = id(getattr(self.pm, "map", None))
            self.route, self.lattice = None, LatticeGraph()
            self.graph_build_count, self.graph_build_ms = 0, 0.0
            self.progress, self.ledger = Progress(), ResourceLedger()
            self._entered_first_edge = False
            self._footprints = None
            self._edge_risk_snapshot = None
            self._smoothing_mode = self.cfg.smoothing_enabled
            self._smoothing_report = dict(enabled=self.cfg.smoothing_enabled,
                                          status='waiting' if self.cfg.smoothing_enabled else 'disabled')
            self.raw_selected = None
            self._validated_end_s = None
            self._validated_at, self._accepted_sequence = -math.inf, -1
            self._fault, self._valid = None, False
            self._sensor_stamp = None
            self._goal_reached = False
            self.last_result = None
            self.location_xy = ego_xy or (self.pm.ego_vehicle.x, self.pm.ego_vehicle.y)
            self.location_sd, self.lap = (0.0, 0.0), 0
            self.traversed_x, self.traversed_y = [self.location_xy[0]], [self.location_xy[1]]
            self.traversed_s, self.traversed_d = [0.0], [0.0]
            self._plan = self._stop_plan()
            self._report("waiting", "mission_initialized")

    def reset(self, wp=0):
        self.set_global_plan(self.global_plan)

    @property
    def anchor(self):
        return self.committed_motion.edge.target if self.committed_motion else self.lattice.source

    @property
    def committed_motion(self):
        # A stationary initial preview is not an entered edge. Stopping after
        # entry must never release the active edge commitment.
        if self.cfg.risk_mode == 'snapshot_edges' and not self._entered_first_edge:
            return None
        return self.progress.active

    def _remaining_committed(self):
        return self.progress.remaining_edge() if self.committed_motion else None

    @property
    def executed_resources(self):
        return dict(self.ledger.total)

    def get_diagnostics(self):
        """Detached diagnostic data; reading never polls, plans or actuates.

        Copies include the graph containers and route, not only path samples.
        Consumers cannot edit the live planner through a diagnostic reference.
        schema_version and graph_id let observers cache their drawing safely.
        """
        with self._lock:
            active = self._remaining_committed()
            return dict(schema_version=1, graph_id=(self.mission, self.graph_build_count),
                        mission=self.mission, graph=copy.deepcopy(self.lattice), route=copy.deepcopy(self.route),
                        path=np.asarray(self._plan.path, dtype=float).copy(),
                        active_xy=active.xy.copy() if active else np.empty((0, 2)),
                        active_sd=np.column_stack([active.s, active.d]) if active else np.empty((0, 2)),
                        valid=(self._valid and self.cfg.enable_motion and not self._goal_reached
                               and time.monotonic()-self._validated_at <= self.cfg.max_plan_age_s),
                        accepted_sequence=self._accepted_sequence,
                        edge_risk=copy.deepcopy(self._edge_risk_snapshot),
                        report=copy.deepcopy(self.last_report))

    def visualization_snapshot(self):
        """Compatibility alias for observers written against the first demo."""
        return self.get_diagnostics()

    def _report(self, status, reason, **extra):
        changed = (self.last_report.get("status"), self.last_report.get("reason")) != (status, reason)
        self.last_report = dict(mission=self.mission, sequence=self.sequence, status=status, reason=reason,
                                anchor=self.anchor, active_edge=edge_key(self.committed_motion.edge) if self.committed_motion else None,
                                graph_build_count=self.graph_build_count, graph_build_ms=self.graph_build_ms,
                                graph_nodes=self.lattice.node_count, graph_edges=len(self.lattice.edges),
                                graph_end_s_m=self.lattice.end_s, executed_resources=self.executed_resources,
                                **extra)
        self.last_report['smoothing'] = copy.deepcopy(self._smoothing_report)
        if self._edge_risk_snapshot is not None:
            risk = self._edge_risk_snapshot
            self.last_report['edge_risk'] = {k: v for k, v in risk.items() if k != 'edges'}
            self.last_report['edge_risk']['counts'] = {state: sum(e['status'] == state for e in risk.get('edges', []))
                for state in ('blocked', 'near', 'clear', 'unchecked')}
        self.history.append(self.last_report)
        if changed:
            emit = log.warning if status == "stop" and reason not in ("goal_reached", "motion_disabled", "motion_disabled_graph_ready") else log.info
            emit("Risk-aware planner: %s (%s), mission=%s, anchor=%s, fixed graph=%s edges",
                 status, reason, self.mission, self.anchor, len(self.lattice.edges))

    def _stop_plan(self):
        ego = self._last_ego
        x, y, heading = ego.x, ego.y, ego.theta
        if not np.isfinite([x, y, heading]).all():
            x, y, heading = 0.0, 0.0, 0.0
        return LocalPlan(path=[(x+d*math.cos(heading), y+d*math.sin(heading)) for d in (0.0, 0.25, 0.5)],
                         velocity=[0.0]*3, behavior=LocalBehavior.STOP)

    def _stop(self, reason):
        if reason == "goal_reached":
            self._goal_reached = True
        self._valid = False
        self._plan = self._stop_plan()
        self._report("stop", reason)

    def _check_inputs(self, sensors=None):
        if self.cfg.smoothing_enabled != self._smoothing_mode:
            raise ValueError('smoothing_mode_changed_restart_required')
        ego = self.pm.ego_vehicle
        if not np.isfinite([ego.x, ego.y, ego.theta, ego.velocity]).all() or ego.velocity < -0.05:
            raise ValueError("invalid_or_reverse_ego_state")
        self._last_ego = copy.deepcopy(ego)
        if not np.isfinite([ego.length, ego.width]).all() or min(ego.length, ego.width) <= 0:
            raise ValueError("invalid_ego_dimensions")
        if ego.length > self.lattice_cfg.vehicle_length_m+0.01 or ego.width > self.lattice_cfg.vehicle_width_m+0.01:
            raise ValueError("planner_footprint_smaller_than_reported_ego")
        if self.cfg.hardware_mode and self.cfg.enable_motion:
            raise ValueError("hardware_operation_not_validated_simulation_only")
        if sensors is not None and sensors.stamp is not None:
            self._sensor_stamp = float(sensors.stamp)
        if self._sensor_stamp is not None:
            age = time.time()-self._sensor_stamp
            if not math.isfinite(self._sensor_stamp) or age < -0.1 or age > 0.5:
                raise ValueError("sensor_snapshot_requires_fresh_unix_stamp")
        if self.lattice_cfg.expand_driving_corridor and id(getattr(self.pm, "map", None)) != self._map_identity:
            raise ValueError("map_changed_reset_required")
        if self._geometry_config is not None and self._geometry_config != self.lattice_cfg.model_dump_json():
            raise ValueError("geometry_settings_changed_reset_required")
        if self._fault:
            raise ValueError(self._fault)

    def _prepare_input(self):
        self._initial_ego = copy.deepcopy(self.pm.ego_vehicle)
        self._geometry_config = self.lattice_cfg.model_dump_json()
        plan = self.global_plan
        if self.lattice_cfg.expand_driving_corridor:
            from .corridor import driving_corridor_plan
            plan, _ = driving_corridor_plan(plan, self.pm.map)
        names = ("path", "velocity", "goal_point", "left_boundary_d", "right_boundary_d",
                 "lane_left_boundary_d", "lane_right_boundary_d", "race_mode")
        self._input = SimpleNamespace(**{k: copy.deepcopy(getattr(plan, k)) for k in names})
        # AVLite 0.5.3's HDMap planner replaces GlobalPlan while routing, leaving
        # both endpoint metadata fields at (0, 0). Its sampled/smoothed path is
        # still the reference route. Adapt only this recognized native output;
        # never move the shared goal, extend geometry to the clicked coordinate,
        # or relax the builder's endpoint check for other mismatched routes.
        lanes = getattr(plan, "lane_path", None)
        if (isinstance(plan, GlobalPlan) and plan.race_mode is False
                and isinstance(lanes, (list, tuple)) and lanes
                and all(isinstance(lane, HDMap.Lane) for lane in lanes)
                and np.array_equal(getattr(plan, "start_point", None), (0.0, 0.0))
                and np.array_equal(plan.goal_point, (0.0, 0.0))):
            xy = np.asarray(self._input.path, dtype=float)
            if (xy.ndim == 2 and xy.shape[1] == 2 and len(xy) >= 3
                    and np.isfinite(xy).all() and np.linalg.norm(xy[-1]-xy[0]) >= 0.1):
                self._input.goal_point = tuple(map(float, xy[-1]))
                log.info("HDMap goal metadata is unset; using reference-path endpoint %s "
                         "as the private lattice goal (shared AVLite plan unchanged)", self._input.goal_point)

    def _sync_progress(self):
        ego = self.pm.ego_vehicle
        self.progress.update(ego, self.lattice_cfg)
        if self.progress.active and (self.progress.distance > 0.05 or self.progress.index > 0):
            self._entered_first_edge = True
        if self.progress.active:
            edge = self.progress.active.edge
            station = float(np.interp(self.progress.distance, edge.distance, edge.s))
            self.ledger.charge(self.progress.motions, station, include_risk=self.cfg.risk_mode != 'snapshot_edges')
        self.location_xy = (ego.x, ego.y)
        if self.route is not None:
            station, lateral, _ = self.route.project(self.location_xy, self.ledger.station)
            self.location_sd = station, lateral

    def _make_risk_model(self):
        if self.cfg.risk_mode != 'snapshot_edges':
            self._edge_risk_snapshot = None
            return RiskModel(snapshot(self.pm), self.lattice_cfg, self.cfg)
        if self._footprints is None:
            self._footprints = FootprintCache(self.lattice, self.lattice_cfg)
        try:
            model = SnapshotEdgeRisk(snapshot(self.pm), self.lattice_cfg, self.cfg, self.lattice,
                                     self._footprints, anchor=self.anchor, prefix=self._remaining_committed())
        except ValueError as exc:
            self._edge_risk_snapshot = dict(mode='snapshot_edges', error=str(exc), edges=[])
            raise
        self._edge_risk_snapshot = model.describe()
        return model

    def _fresh_remaining(self, motions, first_distance=0.0):
        # Association follows the accepted (possibly smoothed) edges. A fresh
        # obstacle snapshot must check those copies, not their raw graph IDs.
        if any(m.edge.smoothed for m in motions):
            from .smoothing import check_join
            for previous, current in zip(motions, motions[1:]):
                check_join(previous.edge, (current.edge,))
        model = self._make_risk_model()
        if isinstance(model, SnapshotEdgeRisk):
            if model.end_s < self.route.length-1e-6:
                speed = max(0, self.pm.ego_vehicle.velocity)
                stopping = (speed*self.cfg.max_plan_age_s + speed*speed/(2*self.cfg.comfortable_deceleration_mps2)
                            + self.cfg.horizon_stop_buffer_m)
                station = float(np.interp(first_distance, motions[0].edge.distance, motions[0].edge.s)) if motions else self.location_sd[0]
                if model.end_s-station < stopping:
                    raise ValueError('validated_horizon_too_short_to_stop')
            self._validated_end_s = model.end_s
        else:
            self._validated_end_s = None
        timed = retime(motions, max(0, self.pm.ego_vehicle.velocity), self.route, model, self.cfg,
                       first_distance=first_distance)
        total = {k: self.ledger.total[k]+v for k, v in resources_of(timed).items()}
        if self.cfg.risk_mode == 'snapshot_edges':
            total['risk'] = resources_of(timed)['risk']
        if not within_budget(total, self.cfg):
            raise ValueError('edge_risk_budget_exhausted' if self.cfg.risk_mode == 'snapshot_edges'
                             and total['risk'] > self.cfg.risk_budget+1e-8 else 'mission_resource_budget_exhausted')
        return timed, total

    def _publish(self, timed, total, **report):
        if not timed:
            self._stop("route_end_stop")
            return
        xy, velocity, nominal, times = export_waypoints(timed, self.cfg, self.route, validated_end_s=self._validated_end_s)
        self._plan = LocalPlan(path=xy, velocity=velocity, behavior=LocalBehavior.CRUISE)
        self._plan.as_trajectory()
        self.progress = Progress(tuple(timed))
        self.nominal_speed, self.nominal_time = nominal, times
        self._validated_at, self._valid = time.monotonic(), True
        self._report("planned", "full_goal_path", predicted_resources=total,
                     predicted_duration_s=float(times[-1]),
                     published_geometry='smoothed' if any(m.edge.smoothed for m in timed) else 'raw', **report)

    def _poll(self):
        if self._worker is None or self._goal_reached:
            return
        if self._worker.busy and time.monotonic()-self._worker.started > self.cfg.worker_timeout_s:
            self._fault = "worker_timeout_reset_required"
            self.close()
            raise ValueError(self._fault)
        result = self._worker.poll()
        if result is None:
            return
        request = result["request"]
        self._sync_progress()
        # Installing geometry does not authorize motion; stale paths are still rejected.
        if request.mission != self.mission:
            self._report("discarded", "stale_mission")
            return
        if result.get("graph") is not None and not self.lattice.edges:
            self.lattice, self.route = result["graph"], result["route"]
            self.graph_build_count = result["build_count"]
            self.graph_build_ms = result["build_ms"]
            self.ledger.station = self.lattice.start_s
            for edge in self.lattice.edges:
                for name in ("s", "d", "xy", "heading", "curvature", "distance"):
                    getattr(edge, name).setflags(write=False)
        requested_anchor = request.anchor if request.anchor is not None else self.lattice.source
        if request.sequence <= self._accepted_sequence or requested_anchor != self.anchor:
            self._report("discarded", "stale_anchor_or_sequence")
            return
        if time.monotonic()-request.created > self.cfg.max_result_age_s:
            self._report("discarded", "stale_snapshot")
            return
        if result.get('smoothing') is not None:
            self._smoothing_report = result['smoothing']
        self.raw_selected = result.get('raw_selected')
        if self.cfg.risk_mode == 'snapshot_edges' and self.lattice.edges:
            self._make_risk_model()  # current display scores, even for a failed solve
        if result["error"]:
            self._stop(result["error"])
            return
        self.last_result = result["result"]
        if self.last_result.selected is None:
            if self.cfg.risk_mode == 'snapshot_edges' and self._valid and self.progress.active:
                timed, total = self._fresh_remaining(self.progress.motions[self.progress.index:], self.progress.distance)
                self._publish(timed, total, retained_geometry=True)
            # A search failure need not discard a freshly revalidated old path.
            self._report("retained" if self._valid else "stop", self.last_result.reason,
                         search_truncated=self.last_result.truncated)
            return
        if not self.cfg.enable_motion:
            self._stop("motion_disabled_graph_ready")
            return
        if self.committed_motion is None:
            if np.linalg.norm(np.array(self.location_xy)-np.array(self.lattice.start_xy)) > 0.05:
                raise ValueError("initial_ego_moved_reset_required")
            chain, first_distance = self.last_result.selected.motions, 0.0
        else:
            # Exactly the active geometry is committed; suffix starts at its target.
            chain = (self.committed_motion,)+self.last_result.selected.motions
            first_distance = self.progress.distance
        timed, total = self._fresh_remaining(chain, first_distance)
        self._accepted_sequence = request.sequence
        self._publish(timed, total, compute_ms=result["compute_ms"], search_expanded=self.last_result.expanded,
                      search_truncated=self.last_result.truncated, search_rejections=self.last_result.rejections,
                      solve_anchor=request.anchor or self.lattice.source)

    def replan(self, perception_model=None, sensors=None):
        with self._lock:
            started = time.monotonic()
            if perception_model is not None:
                self.pm = perception_model
            try:
                if route_signature(self.global_plan) != self._signature:
                    self.set_global_plan(self.global_plan)
                self._check_inputs(sensors)
                self._sync_progress()
                self._poll()
                if self._goal_reached or (self.route is not None and np.linalg.norm(np.array(self.location_xy)-self.route.xy[-1]) <= self.cfg.goal_tolerance_m and self.pm.ego_vehicle.velocity <= self.cfg.stopped_speed_mps):
                    self._stop("goal_reached")
                    return
                if self.progress.active is not None:
                    try:
                        timed, total = self._fresh_remaining(self.progress.motions[self.progress.index:], self.progress.distance)
                        if self.cfg.enable_motion:
                            self._publish(timed, total, retained_geometry=True)
                    except ValueError as exc:
                        self._stop(str(exc))
                if self._input is None:
                    self._prepare_input()
                if self._worker is None:
                    self._worker = self._worker_factory()
                self.sequence += 1
                lattice_cfg = self.lattice_cfg.model_copy(update={"expand_driving_corridor": False})
                request = Request(self.mission, self.sequence, time.monotonic(), self._input,
                                  self._initial_ego, snapshot(self.pm), lattice_cfg, self.cfg.model_copy(deep=True),
                                  anchor=self.anchor,
                                  prefix=self._remaining_committed(),
                                  anchor_speed=float(self.committed_motion.v[-1]) if self.committed_motion else 0,
                                  resources=self.executed_resources, include_graph=not bool(self.lattice.edges))
                self._worker.submit(request)
                if not self.cfg.enable_motion:
                    self._stop("motion_disabled")
            except (ValueError, TypeError, ArithmeticError, IndexError) as exc:
                self._stop(str(exc))
            finally:
                self.last_report["callback_ms"] = 1000*(time.monotonic()-started)

    def step(self, state):
        with self._lock:
            self._last_ego = copy.deepcopy(state)
            try:
                self._check_inputs()
                self._sync_progress()
                self._poll()
            except (ValueError, TypeError, ArithmeticError, IndexError) as exc:
                self._stop(str(exc))
            for name, value in zip(("traversed_x", "traversed_y", "traversed_s", "traversed_d"), (*self.location_xy, *self.location_sd)):
                values = getattr(self, name)
                values.append(value)
                if len(values) > 5000:
                    del values[:-5000]
            trajectory = self._plan.as_trajectory()
            if trajectory is not None:
                trajectory.update_waypoint_by_xy(state.x, state.y)

    def get_local_plan(self):
        with self._lock:
            try:
                self._check_inputs()
                self._poll()
            except (ValueError, TypeError, ArithmeticError, IndexError) as exc:
                self._stop(str(exc))
            if not self.cfg.enable_motion:
                return self._stop_plan()
            if self._goal_reached:
                return self._stop_plan()
            if self._valid and time.monotonic()-self._validated_at > self.cfg.max_plan_age_s:
                self._stop("plan_validation_expired")
            return self._plan
