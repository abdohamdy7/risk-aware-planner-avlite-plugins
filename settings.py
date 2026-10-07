"""Validated algorithm settings; defaults do not authorize vehicle movement."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class LatticeSettings(StrictModel):
    layer_spacing_m: float = Field(6, ge=2, le=30)
    sample_spacing_m: float = Field(0.25, gt=0, le=0.5)
    lateral_sampling_mode: Literal["explicit", "corridor"] = "explicit"
    lateral_offsets_m: list[float] = Field(default_factory=lambda: [-2, -1, 0, 1, 2])
    lateral_spacing_m: float = Field(1, ge=0.1)
    max_lateral_nodes_per_layer: int = Field(15, ge=3, le=101)
    max_lateral_transition_m: float | None = Field(2, gt=0)
    expand_driving_corridor: bool = False
    use_lane_boundaries: bool = True
    vehicle_width_m: float = Field(2, gt=0)
    vehicle_length_m: float = Field(4.5, gt=0)
    vehicle_center_offset_m: float = 0
    wheelbase_m: float = Field(2.5, gt=0)
    max_steering_rad: float = Field(0.5, gt=0, lt=1.4)
    boundary_margin_m: float = Field(0.2, ge=0)
    max_tracking_error_m: float = Field(0.6, gt=0)
    max_tracking_heading_error_rad: float = Field(0.6, gt=0, le=1)


class CSPSettings(StrictModel):
    # Optional post-search geometry refinement; never changes the stored graph.
    smoothing_enabled: bool = False
    smoothing_max_deviation_m: float = Field(0.1, gt=0, le=0.5)
    smoothing_knot_spacing_m: float = Field(0.5, ge=0.25, le=2)
    smoothing_bending_weight: float = Field(1.0, ge=0)
    smoothing_rate_weight: float = Field(10.0, ge=0)
    smoothing_max_iterations: int = Field(4000, ge=1, le=20000)
    smoothing_time_limit_s: float = Field(0.05, gt=0, le=1)
    enable_motion: bool = False
    hardware_mode: bool = False
    max_speed_mps: float = Field(2, gt=0, le=10)
    speed_choices_mps: list[float] = Field(default_factory=lambda: [1, 2], min_length=1)
    max_acceleration_mps2: float = Field(0.5, gt=0)
    max_deceleration_mps2: float = Field(2, gt=0)
    comfortable_deceleration_mps2: float = Field(0.3, gt=0)
    max_lateral_acceleration_mps2: float = Field(1.5, gt=0)
    # Separate units/accounting: local edge scores versus legacy exposure-seconds.
    risk_mode: Literal["time_exposure", "snapshot_edges"] = "time_exposure"
    edge_risk_budget: float = Field(0.1, ge=0)
    collision_horizon_layers: int = Field(5, ge=1, le=100)
    edge_near_band_m: float = Field(0.5, ge=0)
    edge_near_risk: float = Field(0.01, ge=0, lt=1)
    horizon_stop_buffer_m: float = Field(1.0, ge=0)
    risk_exposure_budget_s: float = Field(0.1, ge=0)
    effort_budget: float = Field(200, gt=0)
    comfort_budget: float = Field(200, gt=0)
    uncertainty_sigma_m: float = Field(0.2, gt=0)
    uncertainty_growth_mps: float = Field(0.05, ge=0)
    obstacle_margin_m: float = Field(0.3, ge=0)
    tracking_margin_m: float = Field(0.6, ge=0)
    timing_margin_s: float = Field(1, ge=0)
    risk_sample_dt_s: float = Field(0.2, gt=0, le=0.5)
    prediction_horizon_s: float = Field(120, gt=0)
    allow_constant_velocity_fallback: bool = True
    search_deadline_s: float = Field(1, gt=0)
    max_labels_per_node_speed: int = Field(3, ge=1, le=1000)
    lateral_weight: float = Field(0.3, ge=0)
    effort_weight: float = Field(0.05, ge=0)
    comfort_weight: float = Field(0.05, ge=0)
    risk_weight: float = Field(20, ge=0)
    max_result_age_s: float = Field(3, gt=0)
    max_plan_age_s: float = Field(2, gt=0)
    worker_timeout_s: float = Field(15, gt=0)
    goal_tolerance_m: float = Field(0.5, gt=0)
    stopped_speed_mps: float = Field(0.15, gt=0)
    # Velocity reference preview is a planner output contract, not ego state.
    # Risk uses measured initial speed and the nominal schedule, plus margins.
    speed_reference_preview_s: float = Field(0.8, gt=0, le=2)
    tracking_overspeed_tolerance_mps: float = Field(0.5, ge=0, le=1)

    @property
    def risk_budget(self):
        return self.edge_risk_budget if self.risk_mode == "snapshot_edges" else self.risk_exposure_budget_s

    @model_validator(mode="after")
    def validate_speeds(self):
        if self.comfortable_deceleration_mps2 > self.max_deceleration_mps2:
            raise ValueError("Comfortable deceleration must not exceed the braking limit")
        if any(v <= 0 or v > self.max_speed_mps for v in self.speed_choices_mps):
            raise ValueError("Speed choices must be positive and <= max_speed_mps")
        return self

from pydantic import ConfigDict, Field
from avlite.c60_apps.c64_settings_schema import SettingsSchema


class RiskAwareSettings(SettingsSchema):
    # AVLite's profile binder assigns model_dump() values, including nested dicts.
    # Validate those assignments so lattice/csp retain their model interfaces.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    # Community settings location is supplied by AVLite's plugin loader.
    lattice: LatticeSettings = Field(default_factory=LatticeSettings)
    csp: CSPSettings = Field(default_factory=CSPSettings)


PluginSettings = RiskAwareSettings()


def configured_settings():
    from avlite.c60_apps.c65_setting_utils import load_setting
    from avlite.c60_apps.c69_settings import AppSettings
    result = PluginSettings.model_copy(deep=True)
    load_setting(result, profile=AppSettings.c60_selected_profile, strict=True)
    return result
