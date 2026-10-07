"""Resource-constrained, time-dependent label search over the spatial lattice.

Labels retain exact arrival times and endpoint speeds. No dominance across
different times is assumed. Label caps/deadlines make this a bounded approximate
CSP solver: returned paths satisfy sampled constraints, but infeasibility and
global optimality are not certified when search is truncated.
"""
from dataclasses import dataclass, field
import math
import time
import numpy as np
from scipy.integrate import trapezoid


@dataclass
class Motion:
    edge: object
    xy: np.ndarray
    t: np.ndarray
    v: np.ndarray
    s: np.ndarray
    curvature: np.ndarray
    acceleration: np.ndarray
    risk: float
    effort: float
    comfort: float
    clearance: float
    score: np.ndarray


@dataclass
class Label:
    node: tuple
    speed: float
    arrival: float = 0.0
    risk: float = 0.0
    effort: float = 0.0
    comfort: float = 0.0
    cost: float = 0.0
    motions: tuple = ()


@dataclass
class SearchResult:
    selected: object = None
    expanded: int = 0
    feasible_candidates: int = 0
    truncated: bool = False
    reason: str = "no_feasible_path"
    candidates: list = field(default_factory=list)
    rejections: dict = field(default_factory=lambda: dict(collision=0, kinematics=0, risk_budget=0, other_resources=0))


def time_parameterize(edge, v0, v1, start, cfg, route, risk_model, *, tracking=False):
    if hasattr(risk_model, 'blocked') and risk_model.blocked(edge):
        return None
    if edge.length < 1e-6 or min(v0, v1) < 0:
        return None
    ds = np.diff(edge.distance)
    cap = np.minimum(cfg.max_speed_mps, np.interp(edge.s, route.s, route.speed))
    cap = np.minimum(cap, np.sqrt(cfg.max_lateral_acceleration_mps2 / np.maximum(np.abs(edge.curvature), 1e-9)))
    if v0 > cap[0]+cfg.tracking_overspeed_tolerance_mps or v1 > cap[-1]+cfg.tracking_overspeed_tolerance_mps or np.any(ds <= 0):
        return None
    # Anticipate the real goal across edge boundaries, not only on the final
    # primitive. Comfort is a desired braking envelope; the physical limit
    # remains available to reconcile a measured tracking-speed overshoot.
    goal_cap = np.sqrt(2*cfg.comfortable_deceleration_mps2*np.maximum(route.length-edge.s, 0))
    cap = np.minimum(cap, goal_cap)
    if not tracking and v1 > cap[-1]+1e-6:
        return None
    # A small measured controller overshoot is modelled as braking back to the
    # target cap, never hidden by clipping the ego's speed to the desired speed.
    cap = np.maximum(cap, np.sqrt(np.maximum(v0*v0-2*cfg.max_deceleration_mps2*edge.distance, 0)))
    vv = cap.copy()
    vv[0], vv[-1] = v0, v1
    for i, length in enumerate(ds):
        vv[i+1] = min(vv[i+1], math.sqrt(vv[i]**2+2*cfg.max_acceleration_mps2*length))
    for i in range(len(ds)-1, -1, -1):
        vv[i] = min(vv[i], math.sqrt(vv[i+1]**2+2*cfg.max_deceleration_mps2*ds[i]))
    if abs(vv[0]-v0) > 1e-6 or abs(vv[-1]-v1) > 1e-6 or np.any(vv[:-1]+vv[1:] < 1e-8):
        return None
    spatial_t = np.r_[0, np.cumsum(2*ds/(vv[:-1]+vv[1:]))]
    duration = spatial_t[-1]
    if start+duration > cfg.prediction_horizon_s:
        return None
    tau = np.unique(np.r_[spatial_t, np.linspace(0, duration, max(2, math.ceil(duration/cfg.risk_sample_dt_s)+1))])
    # Near-equal geometric/time-grid knots can collapse when adding arrival time.
    # Merge numerically indistinguishable knots before time-dependent risk checks.
    tau = tau[np.r_[True, np.diff(tau) > 1e-9]]
    tau[-1] = duration
    index = np.minimum(np.searchsorted(spatial_t, tau, side="right")-1, len(ds)-1)
    acceleration = (vv[index+1]**2-vv[index]**2)/(2*ds[index])
    dt = tau-spatial_t[index]
    distance = np.clip(edge.distance[index]+vv[index]*dt+0.5*acceleration*dt**2, 0, edge.length)
    v = np.maximum(vv[index]+acceleration*dt, 0)
    distance[0], distance[-1] = 0.0, edge.length
    v[0], v[-1] = v0, v1
    xy = np.column_stack([np.interp(distance, edge.distance, edge.xy[:, i]) for i in range(2)])
    s = np.interp(distance, edge.distance, edge.s)
    curvature = np.interp(distance, edge.distance, edge.curvature)
    limit = np.minimum(cfg.max_speed_mps, np.interp(s, route.s, route.speed))
    limit = np.maximum(limit, v0)
    # End-of-route speed ramp is represented by the terminal stop, not a
    # discontinuous cap at the last reference point.
    if np.any(v > limit+0.03) or np.max(v*v*np.abs(curvature)) > cfg.max_lateral_acceleration_mps2+1e-7:
        return None
    times = start+tau
    if hasattr(risk_model, 'evaluate_edge'):
        risk, hard, clearance, scores = risk_model.evaluate_edge(edge, len(times))
    else:
        risk, hard, clearance, scores = risk_model.evaluate(xy, times, v)
    if hard or not math.isfinite(risk):
        return None
    effort = float(trapezoid(acceleration**2+0.02*v*v, times))
    comfort = float(trapezoid(acceleration**2+(v*v*curvature)**2, times))
    return Motion(edge, xy, times, v, s, curvature, acceleration, risk, effort, comfort, clearance, scores)


def solve(graph, ego_speed, route, risk_model, cfg, *, anchor=None, start_time=0.0, resources=None):
    result = SearchResult()
    if not graph.layers:
        return result
    deadline = time.perf_counter()+cfg.search_deadline_s
    anchor = graph.source if anchor is None else anchor
    resources = resources or dict(risk=0.0, effort=0.0, comfort=0.0)
    labels = [Label(anchor, max(0.0, ego_speed), arrival=start_time, **resources)]
    if anchor in graph.layers[-1]:
        result.selected, result.reason = labels[0], "at_goal_anchor"
        return result
    start_layer = 0 if anchor == graph.source else next((i+1 for i, nodes in enumerate(graph.layers) if anchor in nodes), -1)
    if start_layer < 0:
        result.reason = "invalid_anchor"
        return result
    for layer_index in range(start_layer, len(graph.layers)):
        buckets = {}
        terminal = layer_index == len(graph.layers)-1
        for label in sorted(labels, key=lambda x: x.cost):
            for edge in graph.adjacency.get(label.node, []):
                if hasattr(risk_model, 'blocked') and risk_model.blocked(edge):
                    result.rejections['collision'] += 1
                    continue
                # The only terminal is the actual reference-path goal.
                speeds = [0.0] if terminal else sorted(set(min(cfg.max_speed_mps, float(v)) for v in cfg.speed_choices_mps if v > 0))
                for speed in speeds:
                    if time.perf_counter() >= deadline:
                        result.truncated = True
                        result.reason = "deadline"
                        break
                    result.expanded += 1
                    motion = time_parameterize(edge, label.speed, speed, label.arrival, cfg, route, risk_model)
                    if motion is None:
                        result.rejections['kinematics'] += 1
                        continue
                    risk, effort, comfort = label.risk+motion.risk, label.effort+motion.effort, label.comfort+motion.comfort
                    if risk > cfg.risk_budget+1e-8:
                        result.rejections['risk_budget'] += 1
                        continue
                    if effort > cfg.effort_budget or comfort > cfg.comfort_budget:
                        result.rejections['other_resources'] += 1
                        continue
                    lateral = float(trapezoid(np.abs(edge.d), edge.distance))
                    cost = label.cost+(motion.t[-1]-label.arrival)+cfg.lateral_weight*lateral+cfg.risk_weight*motion.risk+cfg.effort_weight*motion.effort+cfg.comfort_weight*motion.comfort
                    new = Label(edge.target, speed, float(motion.t[-1]), risk, effort, comfort, cost, label.motions+(motion,))
                    buckets.setdefault((edge.target, speed), []).append(new)
                    if terminal:
                        result.candidates.append(new)
                if result.reason == "deadline":
                    break
            if result.reason == "deadline":
                break
        if result.reason == "deadline":
            break
        labels = []
        for bucket in buckets.values():
            bucket.sort(key=lambda x: x.cost)
            if len(bucket) > cfg.max_labels_per_node_speed:
                result.truncated = True
            labels.extend(bucket[:cfg.max_labels_per_node_speed])
        if not labels:
            break
    if result.candidates:
        result.selected = min(result.candidates, key=lambda x: x.cost)
        result.feasible_candidates = len(result.candidates)
        result.reason = "feasible_truncated" if result.truncated else "feasible"
    return result
