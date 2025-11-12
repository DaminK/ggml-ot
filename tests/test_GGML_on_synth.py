import ggml_ot


def test_GGML_on_synth():
    """Tests basic functions of GGML on dynamically created synthetic dataset"""

    # Setup synthetic 2D dataset (from AISTATS25 paper)
    dataset = ggml_ot.data.from_synth()

    # Check training
    w_theta = ggml_ot.train(dataset, max_iter=3, plot_iter=False, verbose=False, return_dataset=False)
    assert w_theta is not None

    # Check test
    scores = ggml_ot.test(dataset, w_theta, n_splits=1, verbose=False)
    assert scores is not None

    # Check train test
    w_theta, scores = ggml_ot.train_test(dataset, n_splits=2, max_iter=3, verbose=False)
    assert w_theta is not None
    assert scores is not None

    # Check OT on learned ground_metric w_theta
    D = dataset.compute_OT(ground_metric=w_theta["best"])
    assert D is not None

    # Check hyperparameter tuning
    w_thetas, scores = ggml_ot.tune(
        dataset, alpha=[1, 10], reg=1, reg_type=2, n_comps=2, max_iter=1, n_splits=1, verbose=False
    )
    assert w_thetas is not None
    assert scores is not None
