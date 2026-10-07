"""Self-contained AVLite community entry point; no companion package required."""
from .core.avlite_adapter import CommittedEdgeCSPPlanner
from .settings import PluginSettings, configured_settings

__all__ = ["RiskAwarePlanner", "PluginSettings"]


class RiskAwarePlanner(CommittedEdgeCSPPlanner):
    def __init__(self, global_plan, env=None, pm=None, setting=None, **kwargs):
        settings = configured_settings()
        kwargs.setdefault("planner_settings", settings.csp)
        kwargs.setdefault("lattice_settings", settings.lattice)
        super().__init__(global_plan, env=env, pm=pm, setting=setting, **kwargs)
