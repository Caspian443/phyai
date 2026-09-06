from types import SimpleNamespace

import torch
from phyai.models.pi05.modeling_pi05 import PI05ValueHead, value_from_prefix
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


def test_value_head_follows_model_dtype_and_returns_fp32() -> None:
    value_head = PI05ValueHead(
        input_dim=4,
        hidden_sizes=(3,),
        params_dtype=torch.bfloat16,
        device="cpu",
    )
    prefix_out = torch.randn((2, 3, 4), dtype=torch.bfloat16)
    prefix_mask = torch.tensor([[True, True, False], [True, False, False]])

    values = value_from_prefix(value_head, prefix_out, prefix_mask)

    assert {parameter.dtype for parameter in value_head.parameters()} == {
        torch.bfloat16
    }
    assert values.dtype == torch.float32
    assert values.shape == (2,)
