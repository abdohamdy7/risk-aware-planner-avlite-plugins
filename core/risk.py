"""Time-dependent collision scoring against current obstacle forecasts.

Enclosing circles, a motion-sampling margin and an isotropic Gaussian radial
tail score are deliberately conservative. Integrated score is a *risk exposure
surrogate*, not a continuous-time chance-constraint certificate. AVLite
SingleTrajectory samples start at dt; forecasts are never frozen at their end.
"""
import math
import numpy as np
from scipy.integrate import trapezoid
from avlite.c10_perception.c11_perception_model import SingleTrajectory


class RiskModel:
    def __init__(self, pm, lattice_settings, settings):
        self.cfg = settings
        self.ego_radius = math.hypot(lattice_settings.vehicle_length_m/2+abs(lattice_settings.vehicle_center_offset_m),
                                     lattice_settings.vehicle_width_m/2)
        self.obstacles = []
        prediction = pm.prediction
        if prediction is not None and not isinstance(prediction, SingleTrajectory):
            raise ValueError(f"Unsupported prediction {type(prediction).__name__}; use ConstantVelocityPrediction/SingleTrajectory or no prediction")
        for dynamic, objects in ((True, pm.agent_vehicles), (False, pm.static_obstacles)):
            for obj in objects:
                xy = np.asarray([obj.x, obj.y], dtype=float)
                speed = float(getattr(obj, "velocity", 0)) if dynamic else 0.0
                vel = speed*np.array([math.cos(obj.theta), math.sin(obj.theta)])
                if not np.isfinite(np.r_[xy, vel, obj.length, obj.width]).all() or min(obj.length, obj.width) <= 0:
                    raise ValueError("Invalid obstacle state or dimensions")
                knots, means = None, None
                if dynamic and prediction is not None and obj.agent_id in prediction.trajectories:
                    means = np.asarray(prediction.trajectories[obj.agent_id], dtype=float)
                    dt = float(prediction.predict_delta_t)
                    if means.ndim != 2 or means.shape[1] != 2 or len(means) == 0 or not np.isfinite(means).all() or not math.isfinite(dt) or dt <= 0:
                        raise ValueError("Malformed SingleTrajectory forecast")
                    means = np.vstack([xy, means])
                    knots = np.arange(len(means))*dt
                    # Maximum forecast speed used for the between-sample margin.
                    max_speed = max(abs(speed), float(np.max(np.linalg.norm(np.diff(means, axis=0), axis=1)/dt)))
                else:
                    if dynamic and not settings.allow_constant_velocity_fallback:
                        raise ValueError("Missing dynamic forecast; constant-velocity fallback disabled")
                    max_speed = abs(speed)
                self.obstacles.append((xy, vel, math.hypot(obj.length, obj.width)/2, knots, means, max_speed, dynamic))

    def evaluate(self, xy, times, speeds):
        times = np.asarray(times)
        if len(times) < 2 or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0) or times[-1] > self.cfg.prediction_horizon_s+1e-8:
            return math.inf, True, -math.inf, np.ones(len(times))
        score = np.zeros(len(times))
        hard_collision, clearance = False, math.inf
        gap = float(np.max(np.diff(times)))
        for origin, vel, radius, knots, means, max_speed, dynamic in self.obstacles:
            if knots is None:
                predicted = origin+times[:, None]*vel
            else:
                if times[-1]+self.cfg.timing_margin_s > knots[-1]+1e-8:
                    return math.inf, True, -math.inf, np.ones(len(times))
                predicted = np.column_stack([np.interp(times, knots, means[:, i]) for i in range(2)])
            swept = 0.5*gap*(float(np.max(speeds))+max_speed)
            temporal = max_speed*self.cfg.timing_margin_s if dynamic else 0
            separation = np.linalg.norm(xy-predicted, axis=1)-(self.ego_radius+radius+self.cfg.obstacle_margin_m+self.cfg.tracking_margin_m+swept+temporal)
            clearance = min(clearance, float(separation.min()))
            hard_collision |= bool(np.any(separation <= 0))
            sigma = self.cfg.uncertainty_sigma_m+(self.cfg.uncertainty_growth_mps*times if dynamic else 0.0)
            score += np.exp(-0.5*(np.maximum(separation, 0)/sigma)**2)
        score = np.minimum(score, 1)
        exposure = float(trapezoid(score, times))
        return exposure, hard_collision, clearance, score
