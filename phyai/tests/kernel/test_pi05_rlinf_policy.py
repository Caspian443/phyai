"""Regression tests for the RLinf-compatible PI0.5 BF16 kernel policy."""

from __future__ import annotations

from pathlib import Path

import pytest
from phyai.kernel.facts import facts_from_query
from phyai.kernel.types import KernelQuery
from phyai.kernel.policy import load_policy
from phyai.kernel.registry import build_catalog

POLICY_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "pi05"
    / "kernel_policy_rlinf_bf16.yaml"
)


def _activation_facts(*, family: str, gated: bool):
    catalog = build_catalog()
    query = KernelQuery.build(
        "activation",
        model={"family": family},
        dtype={"input": "bf16"},
        shape={"tokens": 2, "hidden": 8},
        attrs={"activation": "gelu", "gated": gated},
    )
    facts = facts_from_query(
        query,
        catalog.op("activation"),
        libraries={"lib.flashinfer": True},
    )
    return catalog, facts


@pytest.mark.parametrize("family", ("pi05", "pi05_wn"))
@pytest.mark.parametrize(
    ("gated", "rule_id", "kernel_id"),
    (
        (True, "pi05-rlinf-exact-geglu", "flashinfer.activation.gelu_and_mul"),
        (False, "pi05-rlinf-exact-gelu", "torch.activation.gelu"),
    ),
)
def test_rlinf_policy_pins_exact_gelu_kernels(
    family: str, gated: bool, rule_id: str, kernel_id: str
) -> None:
    catalog, facts = _activation_facts(family=family, gated=gated)
    policy = load_policy(POLICY_PATH, catalog)

    decision = policy.decide(facts, catalog)

    assert policy.fallback == "error"
    assert decision.matched_rules == (rule_id,)
    assert decision.strict
    assert decision.candidates == (kernel_id,)


def test_rlinf_policy_does_not_capture_other_model_families() -> None:
    catalog, facts = _activation_facts(family="pi0", gated=True)
    policy = load_policy(POLICY_PATH, catalog)

    assert policy.decide(facts, catalog).matched_rules == ()
