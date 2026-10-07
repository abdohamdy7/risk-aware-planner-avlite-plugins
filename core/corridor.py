"""Map-derived driving corridor for cross-lane SIMULATION experiments.

AVLite's parser approximates variable lane widths. This is not lane-change
permission, an HD-map safety certificate, or a substitute for traffic rules.
"""
import copy
import numpy as np


def connected_interval(intervals, anchor=0.0, tolerance=0.03):
    """Return the contiguous interval containing anchor; never jump a median."""
    merged = []
    for low, high in sorted(intervals):
        if not merged or low > merged[-1][1]+tolerance:
            merged.append([low, high])
        else:
            merged[-1][1] = max(merged[-1][1], high)
    return next(((low, high) for low, high in merged if low < anchor < high), None)


def driving_corridor_plan(plan, hdmap):
    """Copy route with cross-sections from contiguous driving lanes, both directions.

    Only lanes on the matched road and lane section are considered. Intersect
    their centerline with the route's lateral cross-section, using the parser's
    lane width; reject far endpoint extrapolation and transverse connectors.
    Existing provided bounds are retained where map association is unavailable.
    The caller keeps the original global plan/goal untouched.
    """
    if hdmap is None or not callable(getattr(hdmap, "find_nearest_lane", None)):
        raise ValueError("Cross-lane corridor requires an AVLite HDMap on PerceptionModel.map")
    xy = np.asarray(plan.path, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError("Cross-lane corridor requires a finite XY reference route")
    tangent = np.gradient(xy, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm < 1e-6):
        raise ValueError("Degenerate route tangent during corridor extraction")
    tangent /= norm[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    result = copy.copy(plan)
    left, right = list(plan.left_boundary_d), list(plan.right_boundary_d)
    if len(left) != len(xy) or len(right) != len(xy):
        raise ValueError("Cross-lane corridor needs aligned fallback route bounds")
    matched = 0
    for index, point in enumerate(xy):
        lane = hdmap.find_nearest_lane(*point)
        if lane is None or lane.road is None:
            continue
        intervals = []
        for candidate in lane.road.lane_sections[lane.lane_section_idx]:
            if candidate.type != "driving" or candidate.width <= 0:
                continue
            line = np.asarray(candidate.center_line, dtype=float).T
            if line.ndim != 2 or len(line) < 2 or line.shape[1] != 2:
                continue
            longitudinal = (line-point)@tangent[index]
            # Find segment intersections with this normal cross-section.
            crossings = np.flatnonzero(longitudinal[:-1]*longitudinal[1:] <= 0)
            if not len(crossings):
                nearest = int(np.argmin(np.abs(longitudinal)))
                if abs(longitudinal[nearest]) > 0.75:
                    continue
                crossings = np.array([min(nearest, len(line)-2)])
            choices = []
            for j in crossings:
                delta = line[j+1]-line[j]
                length = np.linalg.norm(delta)
                if length < 1e-6:
                    continue
                alignment = abs(float(delta@tangent[index]))/length
                if alignment < 0.8:
                    continue
                denom = float(delta@tangent[index])
                u = float(np.clip(-longitudinal[j]/denom, 0, 1))
                center = line[j]+u*delta
                d = float((center-point)@normal[index])
                half = float(candidate.width)/(2*alignment)
                choices.append((d-half, d+half))
            if choices:
                intervals.append(min(choices, key=lambda pair: abs(pair[0]+pair[1])))
        bounds = connected_interval(intervals)
        if bounds is not None:
            right[index], left[index] = bounds
            matched += 1
    if matched == 0:
        raise ValueError("No driving-lane cross-sections could be associated with the route")
    result.left_boundary_d = left
    result.right_boundary_d = right
    # These boundaries already describe the selected driving corridor.
    result.lane_left_boundary_d = []
    result.lane_right_boundary_d = []
    return result, matched
