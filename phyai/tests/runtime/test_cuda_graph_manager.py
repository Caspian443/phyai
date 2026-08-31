"""CPU lifecycle coverage for the CUDA graph wrapper."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import phyai.models.pi05.scheduler_ws1_pi05 as scheduler_module
import torch
from phyai.models.pi05.model_runner_pi05 import PI05ExpertRunner
from phyai.models.pi05.scheduler_ws1_pi05 import PI05WS1Scheduler
from phyai.runtime.cuda_graph_manager import CudaGraph


class _FakeTorchGraph:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def test_cuda_graph_reset_releases_captured_state():
    graph = CudaGraph()
    fake = _FakeTorchGraph()
    graph._captured = True
    graph._graph = fake
    graph._input_buffers = {"input": object()}
    graph._output = object()

    graph.reset()

    assert fake.reset_calls == 1
    assert not graph.is_captured
    assert graph._graph is None
    assert graph._input_buffers == {}
    assert graph._output is None


class _FakeCapturedGraph:
    def __init__(self, label: str, events: list[str]) -> None:
        self.label = label
        self.events = events

    def reset(self) -> None:
        self.events.append(f"reset:{self.label}")


class _FakeAttentionBackend:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def init_capture_metadata(self, seed):
        self.events.append(f"plan:{seed}")
        return "new-plan"


def test_expert_recapture_resets_old_graph_before_new_capture(monkeypatch):
    events: list[str] = []
    runner = object.__new__(PI05ExpertRunner)
    runner.device = torch.device("cuda")
    runner.graph = None
    runner.rollout_graph = _FakeCapturedGraph("rollout", events)
    runner.attn_backend = _FakeAttentionBackend(events)
    runner._capture_seed_metadata = lambda: "seed"

    def capture_rollout() -> None:
        events.append("capture:rollout")
        runner.rollout_graph = object()

    runner._capture_rollout_graph = capture_rollout
    runner._capture_graph = lambda: events.append("capture:inference")
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda _device: events.append("sync")
    )

    assert runner.recapture_after_weight_update()

    assert events == ["sync", "reset:rollout", "plan:seed", "capture:rollout"]
    assert runner._capture_plan == "new-plan"


def test_scheduler_refresh_recaptures_expert_graph(monkeypatch):
    events: list[object] = []

    class _FakeHeads:
        @staticmethod
        def embed_time(times: torch.Tensor) -> torch.Tensor:
            return times[:, None]

    class _FakeExpertRunner:
        def bind_euler_schedule(self, table, *, dt, num_steps) -> None:
            events.append(("bind", table.clone(), dt, num_steps))

        def recapture_after_weight_update(self) -> bool:
            events.append("recapture")
            return True

    monkeypatch.setattr(
        scheduler_module,
        "all_ranks_log",
        lambda logger, level, message: events.append(("log", level, message)),
    )
    scheduler = object.__new__(PI05WS1Scheduler)
    scheduler.cfg = SimpleNamespace(num_inference_steps=3)
    scheduler.device = torch.device("cpu")
    scheduler.model = SimpleNamespace(heads=_FakeHeads())
    scheduler.time_emb_table = torch.empty(3, 1)
    scheduler.expert_runner = _FakeExpertRunner()

    scheduler.refresh_weight_dependent_state()

    assert torch.allclose(
        scheduler.time_emb_table, torch.tensor([[1.0], [2 / 3], [1 / 3]])
    )
    assert events[0][0] == "bind"
    assert events[0][2:] == (-1 / 3, 3)
    assert events[1] == "recapture"
    assert events[2] == (
        "log",
        logging.INFO,
        "Recaptured PI05 expert CUDA graph after hot weight update.",
    )
