"""Optional bounded Frenet B-spline QP on the selected suffix, not the graph.

The quadratic objective penalizes lateral departure, bending and bending rate.
It is a geometric proxy, not a jerk-optimal trajectory. Exported geometry is
sampled at <= 0.25 m and rechecked/retimed with the existing risk model. Neither
the spline nor the five-layer demo is a hardware safety certificate.
"""
from collections import OrderedDict
from dataclasses import replace
import math
import time

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline
from scipy import sparse

from .geometry import LatticeEdge
from .edge_risk import geometry_key
from .solver import time_parameterize
from .trajectory import resources_of, within_budget


def cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def boundary_derivatives(route, s, d, heading, curvature):
    """Convert world heading/curvature to d' and d'' (s need not be arc length)."""
    first, second, third = [route.spline(s, n) for n in (1, 2, 3)]
    v = np.linalg.norm(first)
    if v < 1e-5:
        raise ValueError('smoothing_degenerate_reference')
    dot = np.dot(first, second)
    k = cross(first, second)/v**3
    kp = cross(first, third)/v**3-3*cross(first, second)*dot/v**5
    a = v*(1-k*d)
    angle = (heading-math.atan2(first[1], first[0])+np.pi)%(2*np.pi)-np.pi
    if a < 0.2 or abs(angle) > 1:
        raise ValueError('smoothing_invalid_boundary_heading')
    dp = a*math.tan(angle)
    c = dot/v*(1-k*d)-v*kp*d-2*v*k*dp
    dpp = (curvature*(a*a+dp*dp)**1.5+dp*c)/a-k*v*a
    return dp, dpp


def world_geometry(route, s, d, dp, dpp):
    center = route.spline(s)
    first, second, third = [route.spline(s, n) for n in (1, 2, 3)]
    v = np.linalg.norm(first, axis=1)
    if np.any(v < 1e-5):
        raise ValueError('smoothing_degenerate_reference')
    tangent = first/v[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    hp = cross(first, second)/v**2
    hpp = cross(first, third)/v**2-2*cross(first, second)*np.sum(first*second, axis=1)/v**4
    np1 = -hp[:, None]*tangent
    np2 = -hpp[:, None]*tangent-hp[:, None]**2*normal
    r1 = first+dp[:, None]*normal+d[:, None]*np1
    r2 = second+dpp[:, None]*normal+2*dp[:, None]*np1+d[:, None]*np2
    if np.any(np.sum(r1*tangent, axis=1) < 0.2):
        raise ValueError('smoothing_non_forward_geometry')
    xy = center+d[:, None]*normal
    heading = np.unwrap(np.arctan2(r1[:, 1], r1[:, 0]))
    curvature = cross(r1, r2)/np.linalg.norm(r1, axis=1)**3
    return xy, heading, curvature


def check_join(prefix, suffix):
    if prefix is None or not suffix:
        return
    first = suffix[0]
    angle = (prefix.heading[-1]-first.heading[0]+np.pi)%(2*np.pi)-np.pi
    if (prefix.target != first.source or np.linalg.norm(prefix.xy[-1]-first.xy[0]) > 1e-5
            or abs(angle) > 1e-5 or abs(prefix.curvature[-1]-first.curvature[0]) > 1e-5):
        raise ValueError('smoothing_incompatible_committed_join')


def validate_geometry(edges, route, cfg):
    """Kinematic and sampled corner-corridor checks, like the graph constructor."""
    for edge in edges:
        if not all(np.isfinite(getattr(edge, name)).all() for name in ('s', 'd', 'xy', 'heading', 'curvature')):
            raise ValueError('smoothing_nonfinite_geometry')
        if np.any(np.diff(edge.s) <= 0) or np.max(np.abs(edge.curvature)) > math.tan(cfg.max_steering_rad)/cfg.wheelbase_m:
            raise ValueError('smoothing_curvature_limit')
        _, tangent, _, ref_k = route.frame(edge.s)
        rel = edge.heading-np.arctan2(tangent[:, 1], tangent[:, 0])
        for front in (cfg.vehicle_center_offset_m-cfg.vehicle_length_m/2,
                      cfg.vehicle_center_offset_m+cfg.vehicle_length_m/2):
            for side in (-cfg.vehicle_width_m/2, cfg.vehicle_width_m/2):
                cs = np.clip(edge.s+front*np.cos(rel)-side*np.sin(rel), 0, route.length)
                cd = edge.d+front*np.sin(rel)+side*np.cos(rel)-0.5*ref_k*(cs-edge.s)**2
                if (np.any(cd+cfg.boundary_margin_m > np.interp(cs, route.s, route.left)+1e-8)
                        or np.any(cd-cfg.boundary_margin_m < np.interp(cs, route.s, route.right)-1e-8)):
                    raise ValueError('smoothing_corridor_limit')
    for first, second in zip(edges, edges[1:]):
        check_join(first, (second,))


class PathSmoother:
    """Mission-local bounded caches; no obstacle scores survive a snapshot."""
    def __init__(self):
        self.systems = OrderedDict()
        self.solutions = OrderedDict()

    @staticmethod
    def _remember(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def geometry(self, edges, route, lattice_cfg, cfg, prefix=None):
        # Lazy optional dependency: CSP without smoothing never needs OSQP.
        import osqp

        started = time.perf_counter()
        nodes = np.r_[edges[0].s[0], [e.s[-1] for e in edges]]
        node_d = np.r_[edges[0].d[0], [e.d[-1] for e in edges]]
        begin = prefix if prefix is not None else edges[0]
        index = -1 if prefix is not None else 0
        start_bc = boundary_derivatives(route, nodes[0], node_d[0], begin.heading[index], begin.curvature[index])
        end_bc = boundary_derivatives(route, nodes[-1], node_d[-1], edges[-1].heading[-1], edges[-1].curvature[-1])
        config_key = (cfg.smoothing_knot_spacing_m, cfg.smoothing_max_deviation_m,
                      cfg.smoothing_bending_weight, cfg.smoothing_rate_weight,
                      cfg.smoothing_time_limit_s, cfg.smoothing_max_iterations, lattice_cfg.sample_spacing_m)
        solution_key = (route.signature, tuple(geometry_key(e) for e in edges),
                        tuple(start_bc), tuple(end_bc), config_key, lattice_cfg.model_dump_json())
        if solution_key in self.solutions:
            result, stats = self.solutions[solution_key]
            return result, dict(stats, cache_hit=True, geometry_ms=1000*(time.perf_counter()-started), qp_solve_ms=0.0)
        if math.ceil((nodes[-1]-nodes[0])/cfg.smoothing_knot_spacing_m)+4 > 512:
            raise ValueError('smoothing_problem_too_large')
        # Include raw knots and selected nodes; every segment endpoint is exact.
        grid = np.unique(np.r_[np.linspace(nodes[0], nodes[-1],
                         math.ceil((nodes[-1]-nodes[0])/min(.25, lattice_cfg.sample_spacing_m))+1),
                         np.concatenate([e.s for e in edges]), nodes])
        # Different linspace grids can contain near-equal floats. Keep exact
        # selected stations, then remove sub-nanometre intervals before retiming.
        for node in nodes:
            grid[np.abs(grid-node) < 1e-8] = node
        grid = np.unique(grid)
        grid = grid[np.r_[True, np.diff(grid) > 1e-8]]
        raw_s = np.concatenate([e.s if i == 0 else e.s[1:] for i, e in enumerate(edges)])
        raw_d = np.concatenate([e.d if i == 0 else e.d[1:] for i, e in enumerate(edges)])
        desired = np.interp(grid, raw_s, raw_d)
        system_key = (grid.tobytes(), nodes.tobytes(), config_key)
        cached = system_key in self.systems
        if not cached:
            interior = np.arange(nodes[0]+cfg.smoothing_knot_spacing_m, nodes[-1]-1e-8, cfg.smoothing_knot_spacing_m)
            knots = np.r_[[nodes[0]]*4, interior, [nodes[-1]]*4]
            basis = BSpline(knots, np.eye(len(knots)-4), 3, extrapolate=False)
            b, b2, b3 = [sparse.csc_matrix(basis(grid, n)) for n in (0, 2, 3)]
            eq = sparse.csc_matrix(np.vstack([basis(nodes), basis(nodes[[0, -1]], 1), basis(nodes[[0, -1]], 2)]))
            a = sparse.vstack([b, eq], format='csc')
            p = sparse.triu(2*(b.T@b+cfg.smoothing_bending_weight*(b2.T@b2)
                              +cfg.smoothing_rate_weight*(b3.T@b3)+sparse.eye(b.shape[1])*1e-9), format='csc')
            system = (osqp.OSQP(), basis, b, a, p)
        else:
            system = self.systems[system_key]
        solver, basis, b, a, p = system
        equality = np.r_[node_d, start_bc[0], end_bc[0], start_bc[1], end_bc[1]]
        lower = np.r_[desired-cfg.smoothing_max_deviation_m, equality]
        upper = np.r_[desired+cfg.smoothing_max_deviation_m, equality]
        q = -2*np.asarray(b.T@desired).ravel()
        try:
            if cached:
                solver.update(q=q, l=lower, u=upper)
            else:
                solver.setup(P=p, q=q, A=a, l=lower, u=upper, verbose=False,
                             eps_abs=1e-8, eps_rel=1e-8, polishing=True, warm_starting=True,
                             max_iter=cfg.smoothing_max_iterations, time_limit=cfg.smoothing_time_limit_s)
                self._remember(self.systems, system_key, system, 8)
            result = solver.solve(raise_error=False)
        except osqp.OSQPException as exc:
            raise ValueError('smoothing_backend_error:'+str(exc)) from exc
        if result.info.status_val != 1 or result.x is None:
            raise ValueError('smoothing_qp_'+result.info.status.replace(' ', '_'))
        values = a@result.x
        if np.any(values < lower-1e-6) or np.any(values > upper+1e-6):
            raise ValueError('smoothing_constraint_residual')
        curve = BSpline(basis.t, result.x, 3, extrapolate=False)
        d, dp, dpp = [curve(grid, n) for n in (0, 1, 2)]
        xy, heading, curvature = world_geometry(route, grid, d, dp, dpp)
        smoothed = []
        for old in edges:
            indices = np.flatnonzero((grid >= old.s[0]) & (grid <= old.s[-1]))
            points = xy[indices].copy()
            if max(np.linalg.norm(points[0]-old.xy[0]), np.linalg.norm(points[-1]-old.xy[-1])) > 1e-5:
                raise ValueError('smoothing_node_position_changed')
            points[0], points[-1] = old.xy[0], old.xy[-1]
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            edge = LatticeEdge(old.source, old.target, grid[indices], d[indices], points,
                               heading[indices], curvature[indices], distance, smoothed=True)
            if np.any(np.diff(distance) < 1e-9):
                raise ValueError('smoothing_duplicate_sample')
            smoothed.append(edge)
        validate_geometry(smoothed, route, lattice_cfg)
        check_join(prefix, smoothed)
        # Curve samples and graph arrays are immutable; trimming makes own copies.
        for edge in smoothed:
            for name in ('s', 'd', 'xy', 'heading', 'curvature', 'distance'):
                getattr(edge, name).setflags(write=False)
        stats = dict(cache_hit=False, qp_iterations=result.info.iter,
                     qp_solve_ms=1000*result.info.solve_time,
                     geometry_ms=1000*(time.perf_counter()-started),
                     max_deviation_m=float(np.max(np.abs(d-desired))))
        smoothed = tuple(smoothed)
        self._remember(self.solutions, solution_key, (smoothed, stats), 16)
        return smoothed, stats

    def apply(self, selected, route, lattice_cfg, cfg, model, *, prefix=None,
              speed=0., arrival=0., resources=None):
        """Return validated selected label + diagnostic. Raw fallback needs C2 join.

        Caller must revalidate against its latest snapshot before publishing.
        ``resources`` already reserves the committed prefix; smoothing never
        rewrites it. The five-layer model still leaves the distant tail unknown.
        """
        started = time.perf_counter()
        report = dict(enabled=True, status='raw_fallback', reason='', raw_risk=selected.risk)
        raw = tuple(m.edge for m in selected.motions)
        if not raw:
            return selected, dict(report, status='at_goal_anchor', total_ms=0.)
        resources = resources or dict(risk=0., effort=0., comfort=0.)
        try:
            edges, stats = self.geometry(raw, route, lattice_cfg, cfg, prefix)
            report.update(stats)
            timed, current_speed, current_time = [], speed, arrival
            for edge, previous in zip(edges, selected.motions):
                motion = time_parameterize(edge, current_speed, float(previous.v[-1]), current_time,
                                           cfg, route, model, tracking=True)
                if motion is None:
                    raise ValueError('smoothing_footprint_or_motion_rejected')
                timed.append(motion)
                current_speed, current_time = float(motion.v[-1]), float(motion.t[-1])
            total = {k: resources[k]+v for k, v in resources_of(timed).items()}
            if not within_budget(total, cfg):
                raise ValueError('smoothing_resource_budget_exceeded')
            cost = sum(float(m.t[-1]-m.t[0])+cfg.lateral_weight*float(trapezoid(np.abs(m.edge.d), m.edge.distance))
                       +cfg.risk_weight*m.risk+cfg.effort_weight*m.effort+cfg.comfort_weight*m.comfort for m in timed)
            selected = replace(selected, motions=tuple(timed), arrival=current_time, speed=current_speed, cost=cost, **total)
            report.update(status='smoothed', risk=total['risk'])
        except (ImportError, ValueError, ArithmeticError, RuntimeError) as exc:
            report['reason'] = str(exc)
            try:
                check_join(prefix, raw)
            except ValueError:
                # Never attach a raw zero-slope primitive to a nonzero smoothed
                # tangent. Adapter can retain a freshly safe existing suffix.
                selected = None
                report['status'] = 'retain_or_stop'
        report['total_ms'] = 1000*(time.perf_counter()-started)
        return selected, report
