"""Validate the requested attention backend before any production weight download."""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel

FLASH_ATTENTION_BACKEND = "flash_attention_2"
FLASH_ATTENTION_VERSION = "2.8.3"
PROBE_SEQUENCE_LENGTH = 16
PROBE_QUERY_HEADS = 4
PROBE_KEY_VALUE_HEADS = 2
PROBE_HEAD_DIMENSION = 128
PROBE_VARIABLE_SPLIT = 7
PROBE_ABSOLUTE_TOLERANCE = 0.02
PROBE_RELATIVE_TOLERANCE = 0.03
PROBE_RANDOM_SEED = 20261012
PROBE_GRADIENT_RANDOM_SEED = 20261013
PROBE_GRADIENT_ERROR_MULTIPLIER = 3.0
PROBE_GRADIENT_ERROR_FLOOR = 0.00001
PROBE_VALIDATION_VERSION = 2
VALIDATED_KERNELS: dict[tuple[object, ...], dict[str, object]] = {}


@dataclass(frozen=True)
class ProbeBranch:
    output: torch.Tensor
    inputs: tuple[torch.Tensor, ...]


def installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def build_attention_environment(backend: str) -> dict[str, object]:
    """Report the actual device without claiming that CPU tests executed FA2."""
    cuda_available = torch.cuda.is_available()
    device_index = torch.cuda.current_device() if cuda_available else None
    environment: dict[str, object] = {
        "requested_backend": backend,
        "torch_version": str(torch.__version__),
        "transformers_version": installed_version("transformers"),
        "peft_version": installed_version("peft"),
        "flash_attn_version": installed_version("flash-attn"),
        "torch_cuda_version": torch.version.cuda,
        "device_type": "cuda" if cuda_available else "cpu",
        "device_index": device_index,
        "gpu_name": torch.cuda.get_device_name(device_index)
        if cuda_available
        else None,
        "gpu_capability": list(torch.cuda.get_device_capability(device_index))
        if cuda_available
        else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False,
        "kernel_validation": "not_executed",
    }
    report = VALIDATED_KERNELS.get(environment_key(environment))
    if report is not None:
        environment["kernel_validation"] = dict(report)
    return environment


def environment_key(environment: dict[str, object]) -> tuple[object, ...]:
    return (
        environment["requested_backend"],
        environment["device_index"],
        environment["torch_version"],
        environment["torch_cuda_version"],
        environment["flash_attn_version"],
    )


def validate_attention_runtime(backend: str) -> dict[str, object]:
    """Fail explicitly on unsupported hardware, ABI errors, or failed kernels."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Production loading requires a CUDA GPU with BF16 support")
    environment = build_attention_environment(backend)
    if backend != FLASH_ATTENTION_BACKEND:
        return environment
    capability = environment["gpu_capability"]
    if capability[0] < 8:
        raise RuntimeError("FlashAttention-2 with BF16 requires an Ampere or newer GPU")
    cuda_version = environment["torch_cuda_version"]
    if cuda_version is None or int(cuda_version.split(".")[0]) < 12:
        raise RuntimeError("Pinned FlashAttention-2 requires a CUDA 12 PyTorch build")
    flash_attention = import_flash_attention()
    key = environment_key(environment)
    if key not in VALIDATED_KERNELS:
        device = torch.device("cuda", environment["device_index"])
        VALIDATED_KERNELS[key] = validate_flash_kernels(flash_attention, device)
    return build_attention_environment(backend)


def import_flash_attention() -> ModuleType:
    if installed_version("flash-attn") != FLASH_ATTENTION_VERSION:
        raise RuntimeError(
            f"Requested FA2 requires flash-attn=={FLASH_ATTENTION_VERSION}; "
            "install it in the Modal CUDA image, not the local CPU environment"
        )
    try:
        module = importlib.import_module("flash_attn")
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "FlashAttention CUDA extension could not import; check Torch/CUDA/ABI"
        ) from error
    if any(
        not callable(getattr(module, name, None))
        for name in ("flash_attn_func", "flash_attn_varlen_func")
    ):
        raise RuntimeError("FlashAttention does not expose the required CUDA kernels")
    return module


def make_probe_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    # A private generator keeps startup validation outside the experiment RNG stream.
    generator = torch.Generator(device=device).manual_seed(PROBE_RANDOM_SEED)
    return tuple(
        torch.randn(
            1,
            PROBE_SEQUENCE_LENGTH,
            heads,
            PROBE_HEAD_DIMENSION,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ).requires_grad_()
        for heads in (
            PROBE_QUERY_HEADS,
            PROBE_KEY_VALUE_HEADS,
            PROBE_KEY_VALUE_HEADS,
        )
    )


def reference_attention(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    query, key, value = (tensor.transpose(1, 2) for tensor in inputs)
    # FP32 inputs force an independent math reference for this tiny startup check.
    with sdpa_kernel(SDPBackend.MATH):
        return functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True, enable_gqa=True
        ).transpose(1, 2)


def bf16_reference_attention(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Measure unfused BF16 error, following FA2's upcast=False/reorder_ops test."""
    query, key, value = (tensor.transpose(1, 2) for tensor in inputs)
    head_repetitions = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(head_repetitions, dim=1)
    value = value.repeat_interleave(head_repetitions, dim=1)
    # Scaling K and keeping intermediates in BF16 exposes rounding error that
    # an FP32 SDPA baseline would otherwise hide. This is only a tiny probe.
    scores = query @ (key / query.shape[-1] ** 0.5).transpose(-2, -1)
    future_positions = torch.ones(
        scores.shape[-2:], device=query.device, dtype=torch.bool
    ).triu(1)
    probabilities = scores.masked_fill(future_positions, float("-inf")).softmax(-1)
    return (probabilities @ value).transpose(1, 2)


def variable_length_reference(
    inputs: tuple[torch.Tensor, ...],
    operation: Callable[[tuple[torch.Tensor, ...]], torch.Tensor],
) -> torch.Tensor:
    return torch.cat(
        [
            operation(tuple(tensor[start:stop].unsqueeze(0) for tensor in inputs))
            for start, stop in (
                (0, PROBE_VARIABLE_SPLIT),
                (PROBE_VARIABLE_SPLIT, PROBE_SEQUENCE_LENGTH),
            )
        ],
        dim=1,
    ).squeeze(0)


def make_probe_gradient(output: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device=output.device).manual_seed(
        PROBE_GRADIENT_RANDOM_SEED
    )
    return torch.randn(
        output.shape, dtype=output.dtype, device=output.device, generator=generator
    )


def compare_probe_outputs(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.isfinite(actual).all():
        raise RuntimeError("FlashAttention produced nonfinite outputs or gradients")
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=PROBE_ABSOLUTE_TOLERANCE,
        rtol=PROBE_RELATIVE_TOLERANCE,
    )


def compare_probe_gradients(
    actual: torch.Tensor, reference: torch.Tensor, baseline: torch.Tensor
) -> dict[str, float]:
    if actual.shape != reference.shape or baseline.shape != reference.shape:
        raise RuntimeError("Attention gradient comparison requires matching shapes")
    if any(
        not torch.isfinite(tensor).all() for tensor in (actual, reference, baseline)
    ):
        raise RuntimeError("Attention produced nonfinite gradients")
    actual_error = float((actual.float() - reference.float()).abs().max())
    baseline_error = float((baseline.float() - reference.float()).abs().max())
    # FA2's dense/varlen GQA tests compare maximum gradient error against three
    # times unfused low-precision error. The floor only covers near-exact cases.
    allowed_error = (
        PROBE_GRADIENT_ERROR_MULTIPLIER * baseline_error + PROBE_GRADIENT_ERROR_FLOOR
    )
    if actual_error > allowed_error:
        raise AssertionError(
            f"maximum error {actual_error:.8g} exceeds {allowed_error:.8g}; "
            f"BF16 reference error is {baseline_error:.8g}"
        )
    return {
        "maximum_absolute_error": actual_error,
        "bf16_baseline_error": baseline_error,
        "allowed_absolute_error": allowed_error,
    }


def check_probe_backward(
    actual: ProbeBranch, reference: ProbeBranch, baseline: ProbeBranch
) -> dict[str, dict[str, float]]:
    # A squared loss would supply 2*out independently to each implementation,
    # confounding forward rounding with backward correctness. Share one VJP.
    upstream = make_probe_gradient(actual.output)
    gradients = tuple(
        torch.autograd.grad(
            branch.output,
            branch.inputs,
            grad_outputs=upstream.to(dtype=branch.output.dtype),
        )
        for branch in (actual, reference, baseline)
    )
    report = {}
    for name, observed, expected, bf16 in zip(
        ("query", "key", "value"), *gradients, strict=True
    ):
        try:
            report[name] = compare_probe_gradients(observed, expected, bf16)
        except AssertionError as error:
            raise AssertionError(f"{name} gradient: {error}") from error
    return report


def check_flash_forward_backward(
    flash_attention: ModuleType, inputs: tuple[torch.Tensor, ...]
) -> dict[str, dict[str, float]]:
    reference_inputs = tuple(
        tensor.detach().float().requires_grad_() for tensor in inputs
    )
    output = flash_attention.flash_attn_func(*inputs, dropout_p=0.0, causal=True)
    reference = reference_attention(reference_inputs)
    compare_probe_outputs(output, reference)
    baseline_inputs = tuple(tensor.detach().requires_grad_() for tensor in inputs)
    baseline = bf16_reference_attention(baseline_inputs)
    return check_probe_backward(
        ProbeBranch(output, inputs),
        ProbeBranch(reference, reference_inputs),
        ProbeBranch(baseline, baseline_inputs),
    )


def check_flash_variable_length(
    flash_attention: ModuleType, inputs: tuple[torch.Tensor, ...]
) -> dict[str, dict[str, float]]:
    # HF uses the unpadded kernel for left-padded batches, so validate that path too.
    packed = tuple(tensor.detach().squeeze(0).requires_grad_() for tensor in inputs)
    device = packed[0].device
    offsets = torch.tensor(
        [0, PROBE_VARIABLE_SPLIT, PROBE_SEQUENCE_LENGTH],
        device=device,
        dtype=torch.int32,
    )
    output = flash_attention.flash_attn_varlen_func(
        *packed,
        cu_seqlens_q=offsets,
        cu_seqlens_k=offsets,
        max_seqlen_q=PROBE_SEQUENCE_LENGTH - PROBE_VARIABLE_SPLIT,
        max_seqlen_k=PROBE_SEQUENCE_LENGTH - PROBE_VARIABLE_SPLIT,
        dropout_p=0.0,
        causal=True,
    )
    reference_inputs = tuple(
        tensor.detach().float().requires_grad_() for tensor in packed
    )
    reference = variable_length_reference(reference_inputs, reference_attention)
    compare_probe_outputs(output, reference)
    baseline_inputs = tuple(tensor.detach().requires_grad_() for tensor in packed)
    baseline = variable_length_reference(baseline_inputs, bf16_reference_attention)
    return check_probe_backward(
        ProbeBranch(output, packed),
        ProbeBranch(reference, reference_inputs),
        ProbeBranch(baseline, baseline_inputs),
    )


def validate_flash_kernels(
    flash_attention: ModuleType, device: torch.device
) -> dict[str, object]:
    try:
        # Extension startup can consume RNG internally even when dropout is disabled.
        with torch.random.fork_rng(devices=[device.index]):
            inputs = make_probe_inputs(device)
            dense_gradients = check_flash_forward_backward(flash_attention, inputs)
            variable_gradients = check_flash_variable_length(flash_attention, inputs)
            torch.cuda.synchronize(device)
    except (AssertionError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"FlashAttention forward/backward startup check failed: {error}"
        ) from error
    return {
        "status": "passed",
        "validation_version": PROBE_VALIDATION_VERSION,
        "paths": ["causal_gqa_forward_backward", "varlen_gqa_forward_backward"],
        "dtype": "bfloat16",
        "head_dimension": PROBE_HEAD_DIMENSION,
        "sequence_length": PROBE_SEQUENCE_LENGTH,
        "packed_sequence_lengths": [
            PROBE_VARIABLE_SPLIT,
            PROBE_SEQUENCE_LENGTH - PROBE_VARIABLE_SPLIT,
        ],
        "reference": "torch_fp32_sdpa_math",
        "absolute_tolerance": PROBE_ABSOLUTE_TOLERANCE,
        "relative_tolerance": PROBE_RELATIVE_TOLERANCE,
        "backward_probe": "shared_seeded_bf16_upstream_gradient",
        "gradient_error_baseline": "torch_bf16_matmul_softmax",
        "gradient_error_multiplier": PROBE_GRADIENT_ERROR_MULTIPLIER,
        "gradient_error_floor": PROBE_GRADIENT_ERROR_FLOOR,
        "gradient_checks": {"dense": dense_gradients, "varlen": variable_gradients},
    }


def verify_model_attention(model: nn.Module, backend: str) -> None:
    actual = getattr(model.config, "_attn_implementation", None)
    if actual != backend:
        raise RuntimeError(f"Requested attention {backend!r}, loaded {actual!r}")
    floating_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    )
    if not floating_parameters or any(
        parameter.dtype != torch.bfloat16 for parameter in floating_parameters
    ):
        raise RuntimeError("Production model base weights must all use BF16")
