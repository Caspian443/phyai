from types import SimpleNamespace

import torch
from phyai.models.pi05.scheduler_ws1_pi05 import PI05WS1Scheduler


def test_rollout_sigma_schedule_stays_fp32_with_bf16_default() -> None:
    scheduler = object.__new__(PI05WS1Scheduler)
    scheduler.cfg = SimpleNamespace(num_inference_steps=3)
    scheduler.max_batch_size = 2
    scheduler.device = torch.device("cpu")
    request = SimpleNamespace(
        noise_method="flow_sde",
        noise_level=0.5,
        joint_logprob=False,
    )
    denoise_inds = torch.ones((2, 3), dtype=torch.int64)

    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        sigmas = scheduler._build_rollout_sigmas(request, denoise_inds, 2)
    finally:
        torch.set_default_dtype(original_dtype)

    expected = torch.tensor(
        [[0.0, 2**-0.5, 0.0], [0.0, 2**-0.5, 0.0]],
        dtype=torch.float32,
    )
    assert sigmas.dtype == torch.float32
    torch.testing.assert_close(sigmas, expected)
