"""CPU-only lifecycle tests for Engine inference and hot updates."""

from __future__ import annotations

import threading

import pytest
from phyai.engine import Engine


class _Entry:
    def __init__(self, *, fail_update: bool = False, fail_finish: bool = False) -> None:
        self.fail_update = fail_update
        self.fail_finish = fail_finish
        self.events = []

    def step(self, request):
        self.events.append(("step", request))
        return request + 1

    def rollout_step(self, request, **kwargs):
        self.events.append(("rollout", request, kwargs))
        return request + 2

    def begin_weight_update(self):
        self.events.append("begin")

    def update_weights(self, weights):
        self.events.append(("update", weights))
        if self.fail_update:
            raise RuntimeError("partial update failed")

    def finish_weight_update(self):
        self.events.append("finish")
        if self.fail_finish:
            raise RuntimeError("strict update failed")
        return "report"

    def abort_weight_update(self):
        self.events.append("abort")


def _engine(entry: _Entry) -> Engine:
    engine = object.__new__(Engine)
    engine.entry = entry
    engine._model_lock = threading.Lock()
    engine._weight_update_active = False
    engine._weight_update_received = False
    engine._weight_update_failed = False
    engine._version = 0
    engine._dumper = None
    return engine


def test_step_return_type_and_rollout_api_are_independent():
    engine = _engine(_Entry())

    assert engine.step(3) == 4
    assert engine.rollout_step(3, mode="train") == 5


def test_version_commits_only_after_successful_update():
    engine = _engine(_Entry())

    engine.begin_weight_update()
    engine.update_weights({"weight": 1})
    assert engine.finish_weight_update(version=7) == "report"

    assert engine.version == 7
    assert engine.step(1) == 2


def test_failed_finish_keeps_version_and_poison_engine():
    engine = _engine(_Entry(fail_finish=True))

    engine.begin_weight_update()
    engine.update_weights({"weight": 1})
    with pytest.raises(RuntimeError, match="strict update failed"):
        engine.finish_weight_update(version=7)

    assert engine.version == 0
    with pytest.raises(RuntimeError, match="partially applied"):
        engine.step(1)


def test_failed_update_poison_engine_after_abort():
    engine = _engine(_Entry(fail_update=True))

    engine.begin_weight_update()
    with pytest.raises(RuntimeError, match="partial update failed"):
        engine.update_weights({"weight": 1})
    engine.abort_weight_update()

    assert engine.version == 0
    with pytest.raises(RuntimeError, match="partially applied"):
        engine.step(1)
