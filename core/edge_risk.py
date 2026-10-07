"""Stationary-obstacle demo: five-layer, dimensionless swept-footprint scores.

Geometry is immutable. Scores belong to one perception snapshot and horizon.
Unknown edges cost zero in optimistic full-goal search, but are not validated.
"""
import math
import time
from collections import OrderedDict
import hashlib
import numpy as np
from shapely import distance as polygon_distance
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union


def key(edge):
    return edge.source, edge.target


def geometry_key(edge):
    digest = hashlib.blake2b(digest_size=16)
    for values in (edge.s, edge.xy, edge.heading):
        digest.update(np.ascontiguousarray(values, dtype=float).tobytes())
    return key(edge), digest.digest()


def same_geometry(a, b):
    return all(np.array_equal(getattr(a, name), getattr(b, name)) for name in ('s', 'xy', 'heading'))


def rectangle_corners(xy, heading, length, width, offset=0.0):
    local = np.array([[-length/2+offset, -width/2], [length/2+offset, -width/2],
                      [length/2+offset, width/2], [-length/2+offset, width/2]])
    c, s = np.cos(heading), np.sin(heading)
    return local @ np.array([[c, s], [-s, c]]) + np.asarray(xy)


def swept_footprint(edge, cfg):
    """Enclose every pose of the edge's piecewise-linear XY/heading interpolation.

    A rotating corner at radius R deviates from the chord between its endpoints
    by at most R*delta_heading**2/8 (linear interpolation error bound). Minkowski
    expansion of each endpoint convex hull by a square enclosing that error disk
    covers rotation AND translation between poses; endpoint rectangles alone do
    not. This is a nominal sweep, not certification of arbitrary tracking errors.
    """
    headings = np.unwrap(edge.heading)
    corners = [rectangle_corners(xy, h, cfg.vehicle_length_m, cfg.vehicle_width_m,
                                cfg.vehicle_center_offset_m)
               for xy, h in zip(edge.xy, headings)]
    radius = math.hypot(cfg.vehicle_length_m/2+abs(cfg.vehicle_center_offset_m), cfg.vehicle_width_m/2)
    pieces = []
    for i in range(len(corners)-1):
        error = radius*float(headings[i+1]-headings[i])**2/8+1e-9
        square = np.array([[-error, -error], [error, -error], [error, error], [-error, error]])
        points = np.concatenate(corners[i:i+2])
        pieces.append(MultiPoint((points[:, None, :]+square).reshape(-1, 2)).convex_hull)
    if not pieces:
        raise ValueError('edge_has_no_swept_interval')
    return unary_union(pieces)


class FootprintCache:
    """Full-edge sweeps cached once per mission/process; no cached obstacle scores."""
    def __init__(self, graph, cfg):
        started = time.perf_counter()
        self.graph, self.cfg = graph, cfg.model_copy(deep=True)
        self.edges = {key(e): e for e in graph.edges}
        self.sweeps = {key(e): swept_footprint(e, cfg) for e in graph.edges}
        self.variants = OrderedDict()
        self.build_ms = 1000*(time.perf_counter()-started)

    def get(self, edge):
        original = self.edges.get(key(edge))
        if original is not None and same_geometry(edge, original):
            return self.sweeps[key(edge)]
        variant = geometry_key(edge)
        if variant not in self.variants:
            self.variants[variant] = swept_footprint(edge, self.cfg)
            if len(self.variants) > 256:
                self.variants.popitem(last=False)
        self.variants.move_to_end(variant)
        return self.variants[variant]


class SnapshotEdgeRisk:
    def __init__(self, pm, lattice_cfg, cfg, graph, cache, *, anchor=None, prefix=None):
        started = time.perf_counter()
        self.cfg, self.graph, self.cache = cfg, graph, cache
        self.anchor = graph.source if anchor is None else anchor
        first = 0 if self.anchor == graph.source else next(
            (i+1 for i, layer in enumerate(graph.layers) if self.anchor in layer), -1)
        if first < 0:
            raise ValueError('invalid_anchor')
        self.horizon_layers = graph.layers[first:first+cfg.collision_horizon_layers]
        targets = {node for layer in self.horizon_layers for node in layer}
        self.checked = {key(e) for e in graph.edges if e.target in targets}
        self.end_s = max((node[0] for node in targets), default=graph.end_s)
        self.prefix, self.prefix_score = prefix, None
        self.obstacles = []
        for dynamic, objects in ((True, pm.agent_vehicles), (False, pm.static_obstacles)):
            for obj in objects:
                values = [obj.x, obj.y, obj.theta, obj.length, obj.width, getattr(obj, 'velocity', 0.0)]
                if not np.isfinite(values).all() or min(obj.length, obj.width) <= 0:
                    raise ValueError('invalid_obstacle_footprint')
                if abs(float(getattr(obj, 'velocity', 0.0))) > 1e-6:
                    raise ValueError('snapshot_demo_requires_stationary_obstacles')
                prediction = getattr(pm, 'prediction', None)
                trajectories = getattr(prediction, 'trajectories', {}) or {}
                if dynamic and getattr(obj, 'agent_id', None) in trajectories:
                    predicted = np.asarray(trajectories[obj.agent_id], dtype=float)
                    if (predicted.ndim != 2 or predicted.shape[1] != 2 or not np.isfinite(predicted).all()
                            or np.any(np.linalg.norm(predicted-np.array([obj.x, obj.y]), axis=1) > 1e-6)):
                        raise ValueError('snapshot_demo_requires_stationary_obstacles')
                self.obstacles.append(Polygon(rectangle_corners((obj.x, obj.y), obj.theta, obj.length, obj.width)))
        self.scores = {k: self._score(cache.sweeps[k]) for k in self.checked}
        self.variant_scores = {}
        if prefix is not None:
            self.prefix_score = self._score(cache.get(prefix))
        self.update_ms = 1000*(time.perf_counter()-started)

    def _score(self, sweep):
        if not self.obstacles:
            return 0.0, False, math.inf
        clearance = float(np.min(polygon_distance(sweep, self.obstacles)))
        # Physical clearance allowances, not old circle/speed-based inflation.
        clearance -= self.cfg.obstacle_margin_m+self.cfg.tracking_margin_m
        blocked = clearance <= 1e-8
        risk = 1.0 if blocked else self.cfg.edge_near_risk if clearance <= self.cfg.edge_near_band_m else 0.0
        return risk, blocked, clearance

    def score(self, edge):
        if self.prefix is not None and key(edge) == key(self.prefix):
            if same_geometry(edge, self.prefix):
                return self.prefix_score
        elif key(edge) not in self.checked:
            return 0.0, False, math.inf  # optimistic tail is still UNCHECKED
        else:
            original = self.cache.edges.get(key(edge))
            if original is not None and same_geometry(edge, original):
                return self.scores[key(edge)]
        variant = geometry_key(edge)
        if variant not in self.variant_scores:
            self.variant_scores[variant] = self._score(self.cache.get(edge))
        return self.variant_scores[variant]

    def blocked(self, edge):
        return self.score(edge)[1]

    def evaluate_edge(self, edge, sample_count):
        risk, hard, clearance = self.score(edge)
        return risk, hard, clearance, np.full(sample_count, risk)

    def describe(self):
        scores = dict(self.scores)
        if self.prefix is not None:
            scores[key(self.prefix)] = self.prefix_score
        return dict(mode='snapshot_edges', horizon_layers=len(self.horizon_layers),
                    horizon_end_s=float(self.end_s), risk_budget=self.cfg.edge_risk_budget,
                    near_risk=self.cfg.edge_near_risk, near_band_m=self.cfg.edge_near_band_m,
                    clearance_allowance_m=self.cfg.obstacle_margin_m+self.cfg.tracking_margin_m,
                    obstacle_count=len(self.obstacles), update_ms=self.update_ms,
                    footprint_cache_build_ms=self.cache.build_ms,
                    edges=[dict(source=e.source, target=e.target,
                                risk=scores[key(e)][0] if key(e) in scores else None,
                                clearance_m=scores[key(e)][2] if key(e) in scores else None,
                                status=('blocked' if scores[key(e)][1] else 'near' if scores[key(e)][0] else 'clear')
                                       if key(e) in scores else 'unchecked') for e in self.graph.edges])
