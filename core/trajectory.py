"""Monotonic selected-edge association, prefix preservation and schedule export."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import trapezoid
from .geometry import LatticeEdge
from .solver import time_parameterize


def edge_key(edge):
    return edge.source, edge.target


def project_edge(edge, ego):
    point = np.array([ego.x, ego.y])
    delta = np.diff(edge.xy, axis=0)
    u = np.clip(np.sum((point-edge.xy[:-1])*delta, axis=1)/np.sum(delta**2, axis=1), 0, 1)
    points = edge.xy[:-1]+u[:, None]*delta
    errors = np.linalg.norm(points-point, axis=1)
    i = int(np.argmin(errors))
    distance = float(edge.distance[i]+u[i]*(edge.distance[i+1]-edge.distance[i]))
    heading = np.interp(distance, edge.distance, edge.heading)
    heading_error = abs((ego.theta-heading+np.pi)%(2*np.pi)-np.pi)
    return distance, float(errors[i]), float(heading_error)


def trim_edge(edge, distance):
    """Copy just the remaining samples; never construct a new connector."""
    if distance >= edge.length-1e-6:
        return None
    grid = np.r_[max(0.0, distance), edge.distance[edge.distance > distance+1e-8]]
    interp = lambda values: np.interp(grid, edge.distance, values)
    return LatticeEdge(edge.source, edge.target, interp(edge.s), interp(edge.d),
                       np.column_stack([interp(edge.xy[:, i]) for i in range(2)]),
                       interp(edge.heading), interp(edge.curvature), grid-grid[0],
                       smoothed=edge.smoothed)


@dataclass
class Progress:
    motions: tuple = ()
    index: int = 0
    distance: float = 0.0

    @property
    def active(self):
        return self.motions[self.index] if self.motions and self.index < len(self.motions) else None

    def update(self, ego, cfg):
        if self.active is None:
            return
        # Only the active edge and its immediate selected successor are eligible.
        # No global nearest-node query, branch switching or jumping at crossings.
        edge = self.active.edge
        d, error, heading_error = project_edge(edge, ego)
        at_end = d >= edge.length-0.03
        if at_end and self.index+1 < len(self.motions):
            nxt = self.motions[self.index+1].edge
            nd, ne, nh = project_edge(nxt, ego)
            if ne <= cfg.max_tracking_error_m and nh <= cfg.max_tracking_heading_error_rad:
                self.index += 1
                self.distance = 0
                d, error, heading_error = nd, ne, nh
        if error > cfg.max_tracking_error_m or heading_error > cfg.max_tracking_heading_error_rad:
            raise ValueError("ego_off_committed_edge")
        if d < self.distance-0.3:
            raise ValueError("non_monotonic_ego_progress")
        self.distance = max(self.distance, d)

    def remaining_edge(self):
        return trim_edge(self.active.edge, self.distance) if self.active else None


def retime(motions, ego_speed, route, model, cfg, *, first_distance=0.0):
    """Fresh schedule and collision check, keeping selected geometry/end speeds."""
    result, arrival, speed = [], 0.0, max(ego_speed, 0.0)
    for i, previous in enumerate(motions):
        edge = trim_edge(previous.edge, first_distance) if i == 0 else previous.edge
        if edge is None:
            continue
        if hasattr(model, 'blocked') and model.blocked(edge):
            raise ValueError('remaining_path_footprint_blocked')
        reachable_speed = np.sqrt(speed*speed+2*cfg.max_acceleration_mps2*edge.length)
        minimum_speed = np.sqrt(max(speed*speed-2*cfg.max_deceleration_mps2*edge.length, 0))
        endpoint_speed = float(np.clip(previous.v[-1], minimum_speed, reachable_speed)) if previous.v[-1] > 1e-6 else 0.0
        motion = time_parameterize(edge, speed, endpoint_speed, arrival, cfg, route, model, tracking=True)
        if motion is None:
            raise ValueError("remaining_path_unsafe_or_unreachable")
        result.append(motion)
        arrival, speed = float(motion.t[-1]), float(motion.v[-1])
    return tuple(result)


def resources_of(motions):
    return {name: sum(getattr(m, name) for m in motions) for name in ("risk", "effort", "comfort")}


def within_budget(resources, cfg):
    return all(resources[key] <= limit+1e-8 for key, limit in (
        ("risk", cfg.risk_budget), ("effort", cfg.effort_budget), ("comfort", cfg.comfort_budget)))


def export_waypoints(motions, cfg, route, *, validated_end_s=None):
    xy, nominal, target, times = [], [], [], []
    for i, motion in enumerate(motions):
        sl = slice(None) if i == 0 else slice(1, None)
        if i:
            previous = motions[i-1]
            if np.linalg.norm(previous.xy[-1]-motion.xy[0]) > 1e-5 or abs(previous.v[-1]-motion.v[0]) > 1e-5 or abs(previous.t[-1]-motion.t[0]) > 1e-5:
                raise ValueError("discontinuous_prefix_suffix_join")
        xy.extend(map(tuple, motion.xy[sl]))
        nominal.extend(motion.v[sl])
        times.extend(motion.t[sl])
    if nominal:
        # A preview of the planned speed schedule is the desired speed target.
        # The separate nominal schedule still begins at measured ego speed.
        reference = np.interp(np.asarray(times)+cfg.speed_reference_preview_s, times, nominal)
        stations = np.concatenate([m.s if i == 0 else m.s[1:] for i, m in enumerate(motions)])
        cap = np.minimum(cfg.max_speed_mps, np.interp(stations, route.s, route.speed))
        if validated_end_s is not None and validated_end_s < route.length-1e-6:
            # Keep the full-goal geometry, with zero targets in the unchecked
            # tail and a braking envelope before it. No controller modification.
            remaining = np.maximum(validated_end_s-cfg.horizon_stop_buffer_m-stations, 0)
            cap = np.minimum(cap, np.sqrt(2*cfg.comfortable_deceleration_mps2*remaining))
        target = np.minimum(reference, cap).tolist()
    if target:
        target[-1] = 0.0
    return xy, list(map(float, target)), np.asarray(nominal), np.asarray(times)


class ResourceLedger:
    """Charge only newly traversed route station intervals using accepted models.

    This is model-accounted exposure, not measured collision probability/energy.
    The watermark survives suffix replacement. Stops never refund past usage.
    """
    def __init__(self, station=0.0):
        self.station = station
        self.total = dict(risk=0.0, effort=0.0, comfort=0.0)

    def charge(self, motions, station, *, include_risk=True):
        for m in motions:
            lo, hi = max(self.station, float(m.s[0])), min(station, float(m.s[-1]))
            if hi <= lo:
                continue
            ta, tb = np.interp([lo, hi], m.s, m.t)
            t = np.r_[ta, m.t[(m.t > ta) & (m.t < tb)], tb]
            v, a, k, score = [np.interp(t, m.t, x) for x in (m.v, m.acceleration, m.curvature, m.score)]
            for key, values in (("risk", score), ("effort", a*a+0.02*v*v), ("comfort", a*a+(v*v*k)**2)):
                if key == 'risk' and not include_risk:
                    continue
                self.total[key] += float(trapezoid(values, t))
        self.station = max(self.station, station)
