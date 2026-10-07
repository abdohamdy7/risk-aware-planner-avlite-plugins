"""Build-once, start-to-goal Frenet lattice with quintic motion primitives.

No cached research graph or simulator API is needed. The construction-time ego
is the fixed source. Rolling search starts at stored nodes without reconnecting the geometry.
Only forward, open routes are supported. Boundaries are SIGNED Frenet offsets.
"""
from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import logging
import math
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from .config import LatticeSettings


def route_signature(plan):
    h = hashlib.sha256()
    for value in (plan.path, plan.goal_point, plan.velocity, plan.left_boundary_d, plan.right_boundary_d,
                  plan.lane_left_boundary_d, plan.lane_right_boundary_d):
        h.update(np.asarray(value, dtype=float).tobytes())
    h.update(str(plan.race_mode).encode())
    return h.hexdigest()


class RouteGeometry:
    def __init__(self, plan, settings):
        raw = np.asarray(plan.path, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 3 or not np.isfinite(raw).all():
            raise ValueError("A finite, open reference route with at least three XY points is required")
        if np.linalg.norm(raw[-1] - raw[0]) < 0.1:
            raise ValueError("Closed/racing routes are not supported; select an open start-to-goal route")
        keep = np.r_[True, np.linalg.norm(np.diff(raw, axis=0), axis=1) > 1e-5]
        self.xy = raw[keep]
        if len(self.xy) < 3:
            raise ValueError("Route has fewer than three distinct points")
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        self.length = float(self.s[-1])
        self.spline = CubicSpline(self.s, self.xy, axis=0)
        left, right = plan.left_boundary_d, plan.right_boundary_d
        if settings.use_lane_boundaries and len(plan.lane_left_boundary_d) and len(plan.lane_right_boundary_d):
            lane_l = np.asarray(plan.lane_left_boundary_d, dtype=float)
            lane_r = np.asarray(plan.lane_right_boundary_d, dtype=float)
            if not np.isfinite(lane_l).all() or not np.isfinite(lane_r).all():
                raise ValueError("Lane boundaries contain non-finite values")
            if len(lane_l) == len(raw) and len(lane_r) == len(raw):
                left, right = lane_l, lane_r
            else:
                # This AVLite revision resamples the route/road arrays without
                # always resampling lane arrays. Use the narrowest supplied
                # lane globally, intersected with aligned road bounds. Never
                # stretch/index the mismatched lane data as if it were aligned.
                left = np.minimum(np.asarray(left), np.min(lane_l))
                right = np.maximum(np.asarray(right), np.max(lane_r))
                logging.getLogger(__name__).warning("Unaligned lane bounds: using the narrowest supplied lane corridor")
        def aligned(values, name):
            v = np.asarray(values, dtype=float)
            if v.shape != (len(raw),) or not np.isfinite(v).all():
                raise ValueError(f"{name} must contain one finite value per route point")
            return v[keep]
        self.left = aligned(left, "left boundary")
        self.right = aligned(right, "right boundary")
        # Do not silently change signs on malformed/exported map data.
        if np.any(self.left <= 0) or np.any(self.right >= 0):
            raise ValueError("Route must lie inside signed bounds: LeftBound > 0 and RightBound < 0. Regenerate the route from the map; do not guess signs.")
        self.speed = aligned(plan.velocity, "reference speed")
        if np.any(self.speed < 0):
            raise ValueError("Reverse reference speeds are unsupported")
        self.signature = route_signature(plan)

    def __getstate__(self):
        # Newer SciPy splines contain module references that cannot be pickled.
        state = self.__dict__.copy()
        state.pop("spline")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.spline = CubicSpline(self.s, self.xy, axis=0)

    def frame(self, s):
        s = np.clip(s, 0, self.length)
        xy = self.spline(s)
        first, second = self.spline(s, 1), self.spline(s, 2)
        norm = np.linalg.norm(first, axis=-1)
        if np.any(norm < 1e-5):
            raise ValueError("Reference spline has a degenerate tangent")
        tangent = first / norm[..., None]
        normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
        curvature = (first[..., 0]*second[..., 1] - first[..., 1]*second[..., 0]) / norm**3
        return xy, tangent, normal, curvature

    def project(self, xy, hint=None):
        """Segment-based projection followed by local spline refinement.

        A progress hint restricts association near crossings; callers clear it
        on route replacement/reset. Large localization jumps must be reset.
        """
        point = np.asarray(xy, dtype=float)
        delta = np.diff(self.xy, axis=0)
        u = np.clip(np.sum((point-self.xy[:-1])*delta, axis=1) / np.sum(delta**2, axis=1), 0, 1)
        dist = np.sum((self.xy[:-1]+u[:, None]*delta-point)**2, axis=1)
        if hint is not None:
            dist[(self.s[:-1] < hint-8) | (self.s[:-1] > hint+30)] = np.inf
        index = int(np.argmin(dist))
        sol = minimize_scalar(lambda v: float(np.sum((self.spline(v)-point)**2)),
                              bounds=(self.s[index], self.s[index+1]), method="bounded")
        candidates = [self.s[index], float(sol.x), self.s[index+1]]
        s = min(candidates, key=lambda v: float(np.sum((self.spline(v)-point)**2)))
        center, tangent, normal, _ = self.frame(s)
        return float(s), float(np.dot(point-center, normal)), math.atan2(tangent[1], tangent[0])


@dataclass
class LatticeEdge:
    source: tuple
    target: tuple
    s: np.ndarray
    d: np.ndarray
    xy: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    distance: np.ndarray
    boundary_violation: bool = False
    collision: bool = False  # dynamic risk is time-dependent, not a static edge flag
    smoothed: bool = False  # selected-path copy only; graph primitives stay intact

    @property
    def length(self):
        return float(self.distance[-1])

    @cached_property
    def local_trajectory(self):
        """AVLite's existing lattice visualizer protocol."""
        from ..native_view import edge_trajectory
        return edge_trajectory(self)


@dataclass
class LatticeGraph:
    edges: list = field(default_factory=list)
    adjacency: dict = field(default_factory=dict)
    layers: list = field(default_factory=list)
    source: tuple = (-1, 0.0)
    end_s: float = 0.0
    start_s: float = 0.0
    start_d: float = 0.0
    start_xy: tuple = (0.0, 0.0)
    raw_nodes: int = 0
    raw_edges: int = 0
    pruned_unreachable: int = 0
    pruned_dead_end: int = 0

    @property
    def node_count(self):
        return 1+sum(len(layer) for layer in self.layers) if self.edges else 0


def prune_connected(graph):
    """Return only edges on a directed source-to-terminal path; never mutate input.

    Both passes are O(V+E). Final-layer nodes are terminals, not dead ends.
    Preserve empty layers on failure so a search cannot mistake an earlier layer
    for a valid terminal. Counts classify unreachable nodes first, then dead ends.
    """
    nodes = {graph.source} | {n for layer in graph.layers for n in layer}
    outgoing, incoming = {}, {}
    for edge in graph.edges:
        outgoing.setdefault(edge.source, []).append(edge.target)
        incoming.setdefault(edge.target, []).append(edge.source)
    def reachable(seeds, adjacency):
        seen, pending = set(seeds), list(seeds)
        while pending:
            for node in adjacency.get(pending.pop(), ()):
                if node not in seen:
                    seen.add(node)
                    pending.append(node)
        return seen
    forward = reachable([graph.source], outgoing)
    backward = reachable(graph.layers[-1] if graph.layers else [], incoming)
    keep = forward & backward
    result = LatticeGraph(source=graph.source, end_s=graph.end_s,
                          start_s=graph.start_s, start_d=graph.start_d, start_xy=graph.start_xy,
                          raw_nodes=len(nodes), raw_edges=len(graph.edges),
                          pruned_unreachable=len(nodes-forward), pruned_dead_end=len(forward-backward))
    result.layers = [[n for n in layer if n in keep] for layer in graph.layers]
    for edge in graph.edges:
        if edge.source in keep and edge.target in keep:
            result.edges.append(edge)
            result.adjacency.setdefault(edge.source, []).append(edge)
    return result


class StateLatticeBuilder:
    def __init__(self, global_plan, settings=None, hdmap=None):
        self.settings = (settings or LatticeSettings()).model_copy(deep=True)
        self.input_signature = route_signature(global_plan)
        self.map_identity = id(hdmap) if self.settings.expand_driving_corridor else None
        self.corridor_samples_matched = 0
        if self.settings.expand_driving_corridor:
            from .corridor import driving_corridor_plan
            global_plan, self.corridor_samples_matched = driving_corridor_plan(global_plan, hdmap)
        self.route = RouteGeometry(global_plan, self.settings)
        if np.linalg.norm(np.asarray(global_plan.goal_point)-self.route.xy[-1]) > 0.05:
            raise ValueError("Reference path endpoint must match the goal; regenerate the global route")
        self.graph = None
        self.build_count = 0
        self.build_ms = 0.0
        self._build_error = None
        self.max_curvature = math.tan(self.settings.max_steering_rad)/self.settings.wheelbase_m

    def lateral_samples(self, station):
        cfg = self.settings
        if cfg.lateral_sampling_mode == "explicit":
            return sorted(set([0.0]+list(cfg.lateral_offsets_m)))
        # Node-center bounds are inset for vehicle width. Whole-edge footprint
        # and curvature checks still decide which transitions are admitted.
        half = cfg.vehicle_width_m/2+cfg.boundary_margin_m
        low = float(np.interp(station, self.route.s, self.route.right))+half
        high = float(np.interp(station, self.route.s, self.route.left))-half
        start = math.ceil((low-1e-8)/cfg.lateral_spacing_m)
        end = math.floor((high+1e-8)/cfg.lateral_spacing_m)
        # Bound construction before allocating a potentially huge interval.
        cap = cfg.max_lateral_nodes_per_layer
        start, end = max(start, -cap), min(end, cap)
        values = [round(i*cfg.lateral_spacing_m, 8) for i in range(start, end+1)]
        if len(values) > cap:
            values = sorted(sorted(values, key=lambda d: (abs(d), d))[:cap])
        return values

    def primitive(self, start_s, start_d, end_s, end_d, slope=0.0, second=0.0,
                  source=(-1, 0.0), target=(0, 0.0)):
        length = end_s-start_s
        if length < 0.05:
            return None
        n = max(9, int(math.ceil(length/self.settings.sample_spacing_m))+1)
        u = np.linspace(0, 1, n)
        c = np.zeros(6)
        c[:3] = [start_d, slope*length, 0.5*second*length**2]
        c[3:] = np.linalg.solve([[1, 1, 1], [3, 4, 5], [6, 12, 20]],
                               [end_d-c[:3].sum(), -c[1]-2*c[2], -2*c[2]])
        d = np.polynomial.polynomial.polyval(u, c)
        s = start_s+u*length
        center, tangent, normal, ref_k = self.route.frame(s)
        if np.any(1-ref_k*d < 0.2):
            return None
        xy = center+normal*d[:, None]
        first = np.gradient(xy, s, axis=0, edge_order=2)
        second_xy = np.gradient(first, s, axis=0, edge_order=2)
        heading = np.unwrap(np.arctan2(first[:, 1], first[:, 0]))
        curvature = (first[:, 0]*second_xy[:, 1]-first[:, 1]*second_xy[:, 0]) / np.maximum(np.linalg.norm(first, axis=1)**3, 1e-9)
        if np.max(np.abs(curvature)) > self.max_curvature:
            return None
        cfg = self.settings
        rel = heading-np.arctan2(tangent[:, 1], tangent[:, 0])
        # Sample the four footprint corners against local signed route bounds.
        # This is a sampled geometric check, not a continuous safety certificate.
        for front in (cfg.vehicle_center_offset_m-cfg.vehicle_length_m/2,
                      cfg.vehicle_center_offset_m+cfg.vehicle_length_m/2):
            for side in (-cfg.vehicle_width_m/2, cfg.vehicle_width_m/2):
                cs = np.clip(s+front*np.cos(rel)-side*np.sin(rel), 0, self.route.length)
                cd = d+front*np.sin(rel)+side*np.cos(rel)
                # Account for reference bending over the longitudinal footprint.
                cd -= 0.5*ref_k*(cs-s)**2
                if np.any(cd+cfg.boundary_margin_m > np.interp(cs, self.route.s, self.route.left)) or np.any(cd-cfg.boundary_margin_m < np.interp(cs, self.route.s, self.route.right)):
                    return None
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        return LatticeEdge(source, target, s, d, xy, heading, curvature, distance)

    def build_once(self, ego, hint=None, curvature=0.0):
        """Construct and prune exactly once per builder/mission, even if disconnected.

        Calling again with a different ego never reconnects or rebuilds anything.
        A new goal/route or an explicit planner reset creates a new builder.
        """
        if self._build_error is not None:
            raise ValueError(self._build_error)
        if self.graph is not None:
            return self.graph, self.graph.start_s, self.graph.start_d
        started = time.perf_counter()
        self.build_count += 1
        try:
            graph = self._construct(ego, hint, curvature)
            self.graph = prune_connected(graph)
            for edge in self.graph.edges:
                for name in ("s", "d", "xy", "heading", "curvature", "distance"):
                    getattr(edge, name).setflags(write=False)
        except (ValueError, ArithmeticError, TypeError, IndexError) as exc:
            self._build_error = str(exc)
            raise
        finally:
            self.build_ms = 1000*(time.perf_counter()-started)
        return self.graph, self.graph.start_s, self.graph.start_d

    def _construct(self, ego, hint, curvature):
        cfg, route = self.settings, self.route
        s0, d0, ref_heading = route.project((ego.x, ego.y), hint)
        center, _, normal, _ = route.frame(s0)
        if np.linalg.norm(center+normal*d0-np.array([ego.x, ego.y])) > 0.01:
            raise ValueError("Ego lies beyond the reference path extent; select a route from the current ego")
        error = (ego.theta-ref_heading+math.pi)%(2*math.pi)-math.pi
        if abs(error) > math.pi/3:
            raise ValueError("Ego heading differs from route by more than 60 degrees; forward connector unavailable")
        graph = LatticeGraph(end_s=route.length, start_s=s0, start_d=d0,
                             start_xy=(float(ego.x), float(ego.y)))
        first_s = (math.floor(s0/cfg.layer_spacing_m)+1)*cfg.layer_spacing_m
        if first_s-s0 < cfg.layer_spacing_m*0.7:
            first_s += cfg.layer_spacing_m
        stations = list(np.arange(first_s, graph.end_s-0.1, cfg.layer_spacing_m))
        # Merge a short final fragment once, at the actual reference-path goal.
        if stations and graph.end_s-stations[-1] < 0.7*cfg.layer_spacing_m:
            stations.pop()
        if graph.end_s-s0 >= 0.1:
            stations.append(graph.end_s)
        previous = [graph.source]
        previous_s = s0
        _, _, _, ref_k = route.frame(s0)
        slope = math.tan(error)*(1-ref_k*d0)
        second = float((curvature-ref_k)*(1+slope*slope)**1.5)
        for layer, station in enumerate(stations):
            choices = [0.0] if abs(station-route.length) < 1e-5 else self.lateral_samples(station)
            nodes = [(float(station), float(d)) for d in choices]
            graph.layers.append(nodes)
            for source in previous:
                for target in nodes:
                    a = d0 if source == graph.source else source[1]
                    if cfg.max_lateral_transition_m is not None and abs(target[1]-a) > cfg.max_lateral_transition_m+1e-8:
                        continue
                    edge = self.primitive(previous_s, a, station, target[1],
                                          slope if layer == 0 else 0.0,
                                          second if layer == 0 else 0.0, source, target)
                    if edge is not None:
                        graph.edges.append(edge)
                        graph.adjacency.setdefault(source, []).append(edge)
            previous, previous_s = nodes, station
        return graph
