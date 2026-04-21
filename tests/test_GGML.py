"""Core GGML workflow tests on synthetic datasets."""

import ggml_ot
from .utils.config import get_kwargs, get_synth_config


def test_GGML_synth_train_and_test(solver="emd2"):
    """Tests training and testing of GGML on dynamically created synthetic dataset"""

    # Setup small version of synthetic 2D dataset (from AISTATS25 paper)
    dataset = ggml_ot.data.from_synth(**get_synth_config(), show=False)

    # Check training
    entropic_reg = 0 if solver == "emd2" else 1.0
    map_A = ggml_ot.train(dataset, **get_kwargs("smoke", "train", entropic_reg=entropic_reg))
    assert map_A is not None

    # Check testing
    scores = ggml_ot.test(dataset, map_A, **get_kwargs("smoke", "test"))
    assert scores is not None


def test_GGML_synth_train_test(solver="emd2"):
    """Tests train_test of GGML on dynamically created synthetic dataset"""

    # Setup small version of synthetic 2D dataset (from AISTATS25 paper)
    dataset = ggml_ot.data.from_synth(**get_synth_config(), show=False)

    # Check train_test
    entropic_reg = 0 if solver == "emd2" else 1.0
    map_A, scores = ggml_ot.train_test(dataset, **get_kwargs("smoke", "train_test", entropic_reg=entropic_reg))
    assert map_A is not None
    assert scores is not None


def test_GGML_synth_tune(solver="emd2"):
    """Tests hyperparameter tuning of GGML on dynamically created synthetic dataset"""

    # Setup small version of synthetic 2D dataset (from AISTATS25 paper)
    dataset = ggml_ot.data.from_synth(**get_synth_config(), show=False)

    # Check hyperparameter tuning
    entropic_reg = 0 if solver == "emd2" else 1.0
    w_thetas, scores = ggml_ot.tune(dataset, **get_kwargs("smoke", "tune", entropic_reg=entropic_reg))
    assert w_thetas is not None
    assert scores is not None
