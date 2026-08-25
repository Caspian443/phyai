"""RLinf-aligned scheduling choices for pi0.5 PPO rollout."""

from __future__ import annotations

import torch

from phyai.models.pi05.scheduler_ws1_pi05 import (
    PI05Request,
    PI05RolloutConfig,
    PI05WS1Scheduler,
)


class PI05RLWS1Scheduler(PI05WS1Scheduler):
    """Use the actor's fixed prefix shape and FP32 rollout chain."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lang_buckets = [self.cfg.tokenizer_max_length]
        self._n_per_sample_buckets = [self.n_per_sample]

    def _prepare_noise(
        self,
        request: PI05Request,
        *,
        actual_batch_size: int,
        rollout_config: PI05RolloutConfig | None,
    ) -> torch.Tensor:
        if rollout_config is None:
            return super()._prepare_noise(
                request,
                actual_batch_size=actual_batch_size,
                rollout_config=rollout_config,
            )

        shape = (
            self.max_batch_size,
            self.cfg.chunk_size,
            self.cfg.max_action_dim,
        )
        if request.noise is None:
            return torch.randn(shape, dtype=torch.float32, device=self.device)
        noise = torch.zeros(shape, dtype=torch.float32, device=self.device)
        noise[:actual_batch_size] = request.noise.to(
            device=self.device, dtype=torch.float32
        )
        return noise


__all__ = ["PI05RLWS1Scheduler"]
