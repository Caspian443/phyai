"""RLinf-aligned scheduling choices for pi0.5 PPO rollout."""

from __future__ import annotations

from phyai.models.pi05.scheduler_ws1_pi05 import PI05WS1Scheduler


class PI05RLWS1Scheduler(PI05WS1Scheduler):
    """Use the actor's fixed prefix shape and FP32 rollout chain."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lang_buckets = [self.cfg.tokenizer_max_length]
        self._n_per_sample_buckets = [self.n_per_sample]


__all__ = ["PI05RLWS1Scheduler"]
