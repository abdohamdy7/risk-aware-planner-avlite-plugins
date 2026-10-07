"""One bounded background process; no AVLite executor modifications."""
from dataclasses import dataclass
import multiprocessing as mp
import queue
import time
from .geometry import StateLatticeBuilder
from .risk import RiskModel
from .edge_risk import FootprintCache, SnapshotEdgeRisk
from .solver import solve, time_parameterize
from .trajectory import resources_of, within_budget


@dataclass
class Request:
    mission: int
    sequence: int
    created: float
    route_input: object
    initial_ego: object
    snapshot: object
    lattice_cfg: object
    cfg: object
    anchor: tuple | None = None
    prefix: object = None
    anchor_speed: float = 0.0
    resources: dict | None = None
    include_graph: bool = False


class Engine:
    """Synchronous algorithm engine, used inside the process and unit tests."""
    def __init__(self):
        self.mission, self.builder = None, None
        self.initialization_error = None
        self.footprints = None
        self.smoother = None

    def run(self, request):
        started = time.monotonic()
        output = dict(request=request, error=None, result=None, graph=None, route=None)
        try:
            if request.mission != self.mission:
                self.mission = request.mission
                self.builder, self.initialization_error = None, None
                self.footprints = None
                self.smoother = None
                try:
                    self.builder = StateLatticeBuilder(request.route_input, request.lattice_cfg)
                except (ValueError, TypeError, ArithmeticError, IndexError) as exc:
                    self.initialization_error = str(exc)
            if self.initialization_error is not None:
                raise ValueError(self.initialization_error)
            graph, _, _ = self.builder.build_once(request.initial_ego)
            if request.include_graph:
                output.update(graph=graph, route=self.builder.route)
            output.update(build_count=self.builder.build_count, build_ms=self.builder.build_ms)
            if not graph.edges:
                raise ValueError("no_connected_start_to_goal_graph")
            if request.cfg.risk_mode == 'snapshot_edges':
                if self.footprints is None:
                    self.footprints = FootprintCache(graph, request.lattice_cfg)
                model = SnapshotEdgeRisk(request.snapshot, request.lattice_cfg, request.cfg,
                                         graph, self.footprints, anchor=request.anchor, prefix=request.prefix)
                output['edge_risk'] = model.describe()
            else:
                model = RiskModel(request.snapshot, request.lattice_cfg, request.cfg)
            resources = dict(request.resources or dict(risk=0.0, effort=0.0, comfort=0.0))
            if request.cfg.risk_mode == 'snapshot_edges':
                resources['risk'] = 0.0
            arrival, speed = 0.0, max(0, request.snapshot.ego_vehicle.velocity)
            if request.prefix is not None:
                if hasattr(model, 'blocked') and model.blocked(request.prefix):
                    raise ValueError('committed_prefix_footprint_blocked')
                prefix = time_parameterize(request.prefix, speed, request.anchor_speed, 0, request.cfg, self.builder.route, model, tracking=True)
                if prefix is None:
                    raise ValueError("committed_prefix_unsafe_or_unreachable")
                arrival, speed = float(prefix.t[-1]), float(prefix.v[-1])
                resources = {k: resources[k]+v for k, v in resources_of([prefix]).items()}
            if not within_budget(resources, request.cfg):
                raise ValueError('edge_risk_budget_exhausted' if request.cfg.risk_mode == 'snapshot_edges'
                                 and resources['risk'] > request.cfg.risk_budget+1e-8 else 'mission_resource_budget_exhausted')
            output["result"] = solve(graph, speed, self.builder.route, model, request.cfg,
                                     anchor=request.anchor, start_time=arrival, resources=resources)
            if request.cfg.smoothing_enabled and output['result'].selected is not None:
                from .smoothing import PathSmoother
                if self.smoother is None:
                    self.smoother = PathSmoother()
                selected = output['result'].selected
                output['raw_selected'] = selected
                output['result'].selected, output['smoothing'] = self.smoother.apply(
                    selected, self.builder.route, request.lattice_cfg, request.cfg, model,
                    prefix=request.prefix, speed=speed, arrival=arrival, resources=resources)
                if output['result'].selected is None:
                    output['result'].reason = 'smoothing_no_continuous_safe_fallback'
        except (ValueError, TypeError, ArithmeticError, IndexError) as exc:
            output["error"] = str(exc)
        output["compute_ms"] = 1000*(time.monotonic()-started)
        return output


def _run(inbox, outbox):
    engine = Engine()
    while True:
        request = inbox.get()
        if request is None:
            return
        try:
            outbox.put(engine.run(request))
        except Exception as exc:
            outbox.put(dict(request=request, error=f"worker_exception:{type(exc).__name__}:{exc}", result=None, graph=None, route=None))


class BackgroundWorker:
    def __init__(self):
        context = mp.get_context("spawn")
        self.inbox, self.outbox = context.Queue(maxsize=1), context.Queue(maxsize=1)
        self.process = context.Process(target=_run, args=(self.inbox, self.outbox), daemon=True)
        self.process.start()
        self.busy, self.pending, self.started = False, None, 0.0

    def submit(self, request):
        if self.busy:
            self.pending = request  # exactly one, newest-wins pending snapshot
        else:
            self.inbox.put_nowait(request)
            self.busy, self.started = True, time.monotonic()

    def poll(self):
        if not self.process.is_alive():
            raise ValueError("planner_worker_exited_reset_required")
        try:
            result = self.outbox.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        if self.pending is not None:
            request, self.pending = self.pending, None
            self.submit(request)
        return result

    def close(self):
        self.pending = None
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=0.2)
        for channel in (self.inbox, self.outbox):
            channel.cancel_join_thread()
            channel.close()
