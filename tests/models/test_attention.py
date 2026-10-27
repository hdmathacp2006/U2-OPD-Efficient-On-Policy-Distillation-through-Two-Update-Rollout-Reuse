"""CPU-only checks for fail-fast FA2 selection, provenance, and numerical validation."""

from collections.abc import Iterator
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from gstar_opd_formal.models import attention
from gstar_opd_formal.models.loading import LoraSpec, ModelLoadSpec, load_model_bundle


@pytest.fixture(autouse=True)
def clear_kernel_validation_cache() -> Iterator[None]:
    attention.VALIDATED_KERNELS.clear()
    yield
    attention.VALIDATED_KERNELS.clear()


def mock_cuda_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "mock A10")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 6))
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(
        attention,
        "installed_version",
        lambda distribution: "2.8.3" if distribution == "flash-attn" else "test",
    )


def test_cpu_environment_does_not_claim_gpu_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = attention.build_attention_environment("flash_attention_2")
    assert (
        report["device_type"] == "cpu" and report["kernel_validation"] == "not_executed"
    )


def test_missing_distribution_is_recorded_without_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attention, "version", Mock(side_effect=attention.PackageNotFoundError)
    )
    assert attention.installed_version("flash-attn") is None


def test_missing_cuda_fails_before_extension_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    extension_loader = Mock()
    monkeypatch.setattr(attention, "import_flash_attention", extension_loader)
    with pytest.raises(RuntimeError, match="CUDA"):
        attention.validate_attention_runtime("flash_attention_2")
    assert not extension_loader.called


def test_sdpa_does_not_import_external_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_cuda_environment(monkeypatch)
    extension_loader = Mock()
    monkeypatch.setattr(attention, "import_flash_attention", extension_loader)
    attention.validate_attention_runtime("sdpa")
    assert not extension_loader.called


def test_unsupported_gpu_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cuda_environment(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (7, 5))
    with pytest.raises(RuntimeError, match="Ampere"):
        attention.validate_attention_runtime("flash_attention_2")


@pytest.mark.parametrize("cuda_version", [None, "11.8"])
def test_unsupported_cuda_build_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch, cuda_version: str | None
) -> None:
    mock_cuda_environment(monkeypatch)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    with pytest.raises(RuntimeError, match="CUDA 12"):
        attention.validate_attention_runtime("flash_attention_2")


@pytest.mark.parametrize("package_version", [None, "2.7.4", "2.8.4"])
def test_wrong_flash_attention_version_is_not_silently_accepted(
    monkeypatch: pytest.MonkeyPatch, package_version: str | None
) -> None:
    monkeypatch.setattr(
        attention, "installed_version", lambda distribution: package_version
    )
    with pytest.raises(RuntimeError, match="flash-attn==2.8.3"):
        attention.import_flash_attention()


@pytest.mark.parametrize("error", [ImportError("missing"), OSError("ABI")])
def test_extension_import_errors_have_actionable_failure(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(attention, "installed_version", lambda distribution: "2.8.3")
    monkeypatch.setattr(attention.importlib, "import_module", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="Torch/CUDA/ABI"):
        attention.import_flash_attention()


def test_incomplete_extension_api_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention, "installed_version", lambda distribution: "2.8.3")
    monkeypatch.setattr(
        attention.importlib, "import_module", lambda name: ModuleType("flash_attn")
    )
    with pytest.raises(RuntimeError, match="required CUDA kernels"):
        attention.import_flash_attention()


def test_successful_kernel_check_is_cached_per_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_cuda_environment(monkeypatch)
    module = ModuleType("flash_attn")
    monkeypatch.setattr(attention, "import_flash_attention", lambda: module)
    checker = Mock(return_value={"status": "passed"})
    monkeypatch.setattr(attention, "validate_flash_kernels", checker)
    first = attention.validate_attention_runtime("flash_attention_2")
    second = attention.validate_attention_runtime("flash_attention_2")
    assert checker.call_count == 1 and first == second


def test_failed_kernel_check_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cuda_environment(monkeypatch)
    monkeypatch.setattr(attention, "import_flash_attention", lambda: ModuleType("fake"))
    monkeypatch.setattr(
        attention, "validate_flash_kernels", Mock(side_effect=RuntimeError("kernel"))
    )
    with pytest.raises(RuntimeError, match="kernel"):
        attention.validate_attention_runtime("flash_attention_2")
    assert not attention.VALIDATED_KERNELS


def test_kernel_probe_input_generation_preserves_global_rng() -> None:
    before = torch.get_rng_state().clone()
    attention.make_probe_inputs(torch.device("cpu"))
    assert torch.equal(before, torch.get_rng_state())


def test_upstream_gradient_generation_preserves_global_rng() -> None:
    output = torch.zeros(1, 16, 4, 128, dtype=torch.bfloat16)
    before = torch.get_rng_state().clone()
    attention.make_probe_gradient(output)
    assert torch.equal(before, torch.get_rng_state())


def mock_flash_dense(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return attention.reference_attention(tuple(value.float() for value in inputs)).to(
        inputs[0].dtype
    )


def make_cpu_flash_module() -> ModuleType:
    module = ModuleType("mock_flash_attn")
    module.flash_attn_func = lambda *inputs, **kwargs: mock_flash_dense(inputs)
    module.flash_attn_varlen_func = mock_flash_variable_length
    return module


def mock_flash_variable_length(*inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
    offsets = kwargs["cu_seqlens_q"].tolist()
    return torch.cat(
        [
            mock_flash_dense(
                tuple(value[start:stop].unsqueeze(0) for value in inputs)
            ).squeeze(0)
            for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
        ]
    )


def test_forward_backward_comparison_uses_real_autograd_on_cpu() -> None:
    inputs = attention.make_probe_inputs(torch.device("cpu"))
    module = make_cpu_flash_module()
    attention.check_flash_forward_backward(module, inputs)
    attention.check_flash_variable_length(module, inputs)


@pytest.mark.parametrize(
    "checker",
    [attention.check_flash_forward_backward, attention.check_flash_variable_length],
)
def test_all_backward_branches_receive_identical_upstream_values(
    checker, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_grad = torch.autograd.grad
    upstream_values = []

    def record_gradients(outputs, inputs, **options):
        upstream_values.append(options["grad_outputs"].detach().float().clone())
        return original_grad(outputs, inputs, **options)

    monkeypatch.setattr(torch.autograd, "grad", record_gradients)
    checker(make_cpu_flash_module(), attention.make_probe_inputs(torch.device("cpu")))
    assert len(upstream_values) == 3 and all(
        torch.equal(upstream_values[0], gradient) for gradient in upstream_values[1:]
    )


@pytest.mark.parametrize(
    "checker",
    [attention.check_flash_forward_backward, attention.check_flash_variable_length],
)
def test_correct_forward_with_corrupted_backward_is_rejected(checker) -> None:
    module = make_cpu_flash_module()
    kernel_name = (
        "flash_attn_func"
        if checker is attention.check_flash_forward_backward
        else "flash_attn_varlen_func"
    )
    original_kernel = getattr(module, kernel_name)

    def corrupt_backward(*inputs, **options):
        output = original_kernel(*inputs, **options)
        output.register_hook(torch.zeros_like)
        return output

    setattr(module, kernel_name, corrupt_backward)
    with pytest.raises(AssertionError, match="gradient"):
        checker(module, attention.make_probe_inputs(torch.device("cpu")))


def test_gradient_error_is_compared_with_bf16_reference_error() -> None:
    report = attention.compare_probe_gradients(
        torch.tensor([0.05]), torch.zeros(1), torch.tensor([0.02])
    )
    assert report["maximum_absolute_error"] <= report["allowed_absolute_error"]


def test_gradient_error_above_precision_budget_is_rejected() -> None:
    with pytest.raises(AssertionError, match="exceeds"):
        attention.compare_probe_gradients(
            torch.tensor([0.07]), torch.zeros(1), torch.tensor([0.02])
        )


def test_large_gradient_error_is_rejected_when_baseline_is_exact() -> None:
    with pytest.raises(AssertionError, match="exceeds"):
        attention.compare_probe_gradients(torch.ones(1), torch.zeros(1), torch.zeros(1))


@pytest.mark.parametrize("invalid_branch", range(3))
def test_nonfinite_gradient_in_any_comparison_branch_is_rejected(
    invalid_branch: int,
) -> None:
    values = [torch.zeros(1) for _ in range(3)]
    values[invalid_branch][0] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        attention.compare_probe_gradients(*values)


def test_gradient_comparison_rejects_shape_mismatch() -> None:
    with pytest.raises(RuntimeError, match="matching shapes"):
        attention.compare_probe_gradients(
            torch.zeros(2), torch.zeros(1), torch.zeros(1)
        )


def test_unfused_bf16_attention_passes_precision_aware_backward_probe() -> None:
    inputs = attention.make_probe_inputs(torch.device("cpu"))
    reference_inputs = tuple(
        tensor.detach().float().requires_grad_() for tensor in inputs
    )
    baseline_inputs = tuple(tensor.detach().requires_grad_() for tensor in inputs)
    report = attention.check_probe_backward(
        attention.ProbeBranch(attention.bf16_reference_attention(inputs), inputs),
        attention.ProbeBranch(
            attention.reference_attention(reference_inputs), reference_inputs
        ),
        attention.ProbeBranch(
            attention.bf16_reference_attention(baseline_inputs), baseline_inputs
        ),
    )
    assert set(report) == {"query", "key", "value"}


def test_variable_length_probe_detects_cross_example_attention() -> None:
    module = make_cpu_flash_module()
    module.flash_attn_varlen_func = lambda *inputs, **kwargs: mock_flash_dense(
        tuple(value.unsqueeze(0) for value in inputs)
    ).squeeze(0)
    with pytest.raises(AssertionError):
        attention.check_flash_variable_length(
            module, attention.make_probe_inputs(torch.device("cpu"))
        )


def test_nonfinite_kernel_output_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="nonfinite"):
        attention.compare_probe_outputs(torch.tensor([float("nan")]), torch.zeros(1))


def test_incorrect_finite_kernel_output_is_rejected() -> None:
    with pytest.raises(AssertionError):
        attention.compare_probe_outputs(torch.ones(1), torch.zeros(1))


def test_startup_probe_restores_rng_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original_fork_rng = torch.random.fork_rng
    monkeypatch.setattr(
        torch.random, "fork_rng", lambda devices: original_fork_rng(devices=[])
    )

    def consume_rng_then_fail(device: torch.device) -> tuple[torch.Tensor, ...]:
        torch.rand(2)
        raise RuntimeError("bad kernel")

    monkeypatch.setattr(attention, "make_probe_inputs", consume_rng_then_fail)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="startup check failed"):
        attention.validate_flash_kernels(ModuleType("fake"), torch.device("cuda", 0))
    assert torch.equal(before, torch.get_rng_state())


def test_startup_probe_reports_only_after_both_paths_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.random, "fork_rng", lambda devices: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    original_inputs = attention.make_probe_inputs
    monkeypatch.setattr(
        attention,
        "make_probe_inputs",
        lambda device: original_inputs(torch.device("cpu")),
    )
    report = attention.validate_flash_kernels(
        make_cpu_flash_module(), torch.device("cuda", 0)
    )
    assert (
        report["status"] == "passed"
        and len(report["paths"]) == 2
        and report["validation_version"] == 2
        and set(report["gradient_checks"]) == {"dense", "varlen"}
    )


def make_bf16_model(backend: str) -> nn.Module:
    model = nn.Linear(2, 2, dtype=torch.bfloat16)
    model.config = SimpleNamespace(_attn_implementation=backend)
    return model


def test_loaded_backend_fallback_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="loaded 'sdpa'"):
        attention.verify_model_attention(make_bf16_model("sdpa"), "flash_attention_2")


def test_loaded_float32_base_weights_are_rejected() -> None:
    model = make_bf16_model("flash_attention_2").float()
    with pytest.raises(RuntimeError, match="BF16"):
        attention.verify_model_attention(model, "flash_attention_2")


def test_expected_backend_and_precision_are_accepted() -> None:
    attention.verify_model_attention(
        make_bf16_model("flash_attention_2"), "flash_attention_2"
    )


def test_training_preflight_failure_precedes_metadata_and_weight_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gstar_opd_formal.models.loading.validate_attention_runtime",
        Mock(side_effect=RuntimeError("invalid FlashAttention ABI")),
    )
    contract = Mock()
    loader = Mock()
    monkeypatch.setattr(
        "gstar_opd_formal.models.loading.check_model_contract", contract
    )
    monkeypatch.setattr("gstar_opd_formal.models.loading.load_model_weights", loader)
    spec = ModelLoadSpec(
        None, LoraSpec(2, 4, 0.0, ("q_proj",)), "flash_attention_2", False
    )
    with pytest.raises(RuntimeError, match="ABI"):
        load_model_bundle(spec)
    assert not contract.called and not loader.called
