"""Forward-equivalent gated-transformer R-lens backward rules."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Iterator

import torch
from torch import Tensor, nn


class HalfProduct(torch.autograd.Function):
    """Ordinary product forward; split backward relevance evenly across inputs."""

    @staticmethod
    def forward(ctx: object, left: Tensor, right: Tensor) -> Tensor:
        ctx.save_for_backward(left, right)
        return left * right

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, Tensor]:
        left, right = ctx.saved_tensors
        return 0.5 * gradient * right, 0.5 * gradient * left


class IdentityActivation(torch.autograd.Function):
    """Return the model's exact activation, but use its x-scaled identity rule."""

    @staticmethod
    def forward(ctx: object, x: Tensor, exact_value: Tensor) -> Tensor:
        factor = torch.where(
            x != 0,
            exact_value.float() / x.float(),
            torch.full_like(x.float(), 0.5),
        )
        ctx.save_for_backward(factor.to(x.dtype))
        return exact_value

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        (factor,) = ctx.saved_tensors
        return gradient * factor, None


def detached_silu(x: Tensor) -> Tensor:
    """SiLU forward with only its sigmoid factor active in the backward rule."""

    return x * torch.sigmoid(x).detach()


OFFSET_RMSNORMS = {"Qwen3_5RMSNorm", "Qwen3NextRMSNorm"}  # scale by (1 + weight), store eps


def rmsnorm_detached_forward(module: nn.Module, hidden_states: Tensor) -> Tensor:
    input_dtype = hidden_states.dtype
    states = hidden_states.float()
    variance = states.pow(2).mean(-1, keepdim=True)
    if module.__class__.__name__ in OFFSET_RMSNORMS:
        normalized = states * torch.rsqrt(variance + module.eps).detach()
        return (normalized * (1.0 + module.weight.float())).type_as(hidden_states)
    denominator = torch.rsqrt(variance + module.variance_epsilon).detach()
    normalized = states * denominator
    # OLMo2 applies its learned weight in float32 and casts afterward. Qwen2
    # and Llama cast the normalized states first. Preserve each architecture's
    # operation order exactly while detaching only the RMS denominator.
    if module.__class__.__name__ in {"Olmo2RMSNorm", "Olmo3RMSNorm"}:
        return (module.weight * normalized).to(input_dtype)
    return module.weight * normalized.to(input_dtype)


def gated_mlp_half_forward(module: nn.Module, x: Tensor) -> Tensor:
    gate_pre = module.gate_proj(x)
    gate = IdentityActivation.apply(gate_pre, module.act_fn(gate_pre).detach())
    up = module.up_proj(x)
    return module.down_proj(HalfProduct.apply(gate, up))


@contextmanager
def gated_r_lens_rules(model: nn.Module) -> Iterator[None]:
    """Install R-lens rules on structurally compatible decoder modules.

    Dense Qwen2, Llama/SmolLM2, and OLMo2 use the same RMSNorm plus gated-MLP
    structure needed by these rules. Attention q/k norms stay untouched, as in
    the Qwen reproduction. Unsupported architectures fail closed.
    """

    replaced: list[tuple[nn.Module, object]] = []
    norm_count = 0
    mlp_count = 0
    for module_name, module in model.named_modules():
        name = module.__class__.__name__
        leaf_name = module_name.rsplit(".", 1)[-1]
        if (
            "RMSNorm" in name
            and (hasattr(module, "variance_epsilon") or name in OFFSET_RMSNORMS)
            and not name.endswith("RMSNormGated")  # inside linear attention, like q/k norms
            and leaf_name not in {"q_norm", "k_norm"}
        ):
            replaced.append((module, module.forward))
            module.forward = MethodType(rmsnorm_detached_forward, module)
            norm_count += 1
        elif name.endswith("MLP") and all(
            hasattr(module, field) for field in ("gate_proj", "up_proj", "down_proj")
        ) and hasattr(module, "act_fn"):
            replaced.append((module, module.forward))
            module.forward = MethodType(gated_mlp_half_forward, module)
            mlp_count += 1
    if not replaced:
        for module, original in reversed(replaced):
            module.forward = original
        raise TypeError(
            "R-lens found no compatible residual RMSNorm or gated MLP modules"
        )
    try:
        yield
    finally:
        for module, original in reversed(replaced):
            module.forward = original


# Backward-compatible public name used by the frozen Qwen experiments.
qwen_r_lens_rules = gated_r_lens_rules
