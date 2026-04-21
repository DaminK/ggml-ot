"""GPU execution tests for GGML training, validation, and tuning."""

import pytest
import torch
import ggml_ot

from . import test_GGML


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def _setup_gpu():
    """Setup GPU device and verify CUDA is available."""
    ggml_ot.settings.device = "cuda:0"

    # Sanity check: settings.device is CUDA and explicit device placement works
    assert ggml_ot.settings.device.type == "cuda"
    assert torch.zeros(1, device=ggml_ot.settings.device).device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.filterwarnings("ignore::DeprecationWarning", match=r"Triggered internally at /pytorch/")
def test_GGML_train_and_test_GPU():
    """Tests training and testing of GGML on GPU"""
    _setup_gpu()
    test_GGML.test_GGML_synth_train_and_test(solver="sinkhorn")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.filterwarnings("ignore::DeprecationWarning", match=r"Triggered internally at /pytorch/")
def test_GGML_train_test_GPU():
    """Tests train_test of GGML on GPU"""
    _setup_gpu()
    test_GGML.test_GGML_synth_train_test(solver="sinkhorn")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.filterwarnings("ignore::DeprecationWarning", match=r"Triggered internally at /pytorch/")
def test_GGML_tune_GPU():
    """Tests hyperparameter tuning of GGML on GPU"""
    _setup_gpu()
    test_GGML.test_GGML_synth_tune(solver="sinkhorn")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_no_cuda_init():
    """Regression: running GGML on CPU should not initialize CUDA.

    This test is meant to be run in isolation or before any GPU tests.
    If CUDA is already initialized in this Python process (e.g. another test
    touched it), we skip to avoid false failures.
    """

    import torch
    import ggml_ot

    was_cuda_initialized = torch.cuda.is_initialized()
    if was_cuda_initialized:
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    ggml_ot.settings.device = "cpu"
    assert ggml_ot.settings.device.type == "cpu"

    # Minimal workload similar
    ds = ggml_ot.data.from_synth(distribution_size=25, show=False)
    _ = ggml_ot.train(ds, max_iter=1, plot_iter=0, verbose=False, return_dataset=False)

    if was_cuda_initialized:
        # Allow small tolerance
        tol = 1 * 1024 * 1024  # 1 MiB

        # If CUDA was initialized before, we can check for VRAM allocation
        assert torch.cuda.memory_allocated() <= allocated_before + tol
        assert torch.cuda.max_memory_allocated() <= allocated_before + tol
    else:
        # If CUDA was not initialized before, check if it remains uninitialized
        assert not torch.cuda.is_initialized()
