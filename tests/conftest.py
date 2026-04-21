import pytest
import ggml_ot
from ggml_ot.settings import Settings
from .utils.setup_dataset import PERF_DATA_SOURCES, get_anndata_registry
from .utils.performance_snapshot import write_snapshot_overview_file

_DEFAULT_SETTINGS = Settings()
_PUBLIC_SETTINGS_KEYS = (
    "n_threads",
    "device",
    "verbose",
    "figdir",
    "random_seed",
    "restore_adata_params",
)


def pytest_addoption(parser):
    """Add command-line option to override number of threads for test runs."""

    parser.addoption(
        "--threads",
        "-T",
        action="store",
        default=None,
        help="Number of threads to use for tests (overrides ggml_ot.settings.n_threads)",
    )
    parser.addoption(
        "--data-source",
        action="store",
        default="synthetic",
        choices=(*PERF_DATA_SOURCES, "all"),
        help=(
            "Dataset source for tests that support synthetic/network data. "
            "Use 'all' to run both sources for data_source-parametrized tests and source-aware fixtures."
        ),
    )
    parser.addoption(
        "--update-baseline",
        action="store_true",
        default=False,
        help="Update performance baseline snapshot with current run",
    )
    parser.addoption(
        "--fail-on-degradation",
        action="store_true",
        default=False,
        help="Fail tests when performance degradation is detected",
    )


def pytest_configure(config):
    """Apply the ``--threads`` option to default settings.

    We update both `_DEFAULT_SETTINGS` and `ggml_ot.settings` so that the
    existing per-test reset fixture will restore to the overridden value.
    """

    val = config.getoption("--threads")
    if val is not None:
        n = int(val)
        _DEFAULT_SETTINGS.n_threads = n
        ggml_ot.settings.n_threads = n

    # Track whether any perf-marked tests are actually executed in this session.
    setattr(config, "_ran_perf_tests", False)


def pytest_generate_tests(metafunc):
    """Parameterize data_source-aware tests from the --data-source option."""
    selected = metafunc.config.getoption("--data-source")
    if selected == "all":
        values = list(PERF_DATA_SOURCES)
    else:
        values = [selected]

    source_ids = {"synthetic": "data: synthetic", "network": "data: network"}

    if "data_source" in metafunc.fixturenames:
        metafunc.parametrize("data_source", values, ids=[source_ids[v] for v in values])

    if "anndata_datasets" in metafunc.fixturenames and "data_source" not in metafunc.fixturenames:
        metafunc.parametrize("anndata_datasets", values, ids=[source_ids[v] for v in values], indirect=True)

    if "perf_anndata" in metafunc.fixturenames and "data_source" not in metafunc.fixturenames:
        metafunc.parametrize("perf_anndata", values, ids=[source_ids[v] for v in values], indirect=True)


def pytest_runtest_setup(item):
    """Mark session when a perf test is executed."""
    if item.get_closest_marker("perf") is not None:
        setattr(item.config, "_ran_perf_tests", True)


def pytest_sessionfinish(session, exitstatus):
    """Emit perf snapshot overview file when perf tests were part of this run."""
    config = session.config
    if config.option.collectonly:
        return
    if not getattr(config, "_ran_perf_tests", False):
        return

    output_path = write_snapshot_overview_file(skip_if_empty=True)
    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        if output_path is None:
            terminal.write_line(
                "[perf] No current-config baseline snapshot entries found; skipping snapshot overview. "
                "Run with --update-baseline to create them."
            )
        else:
            terminal.write_line(f"[perf] Wrote snapshot overview: {output_path}")


@pytest.fixture(scope="session")
def anndata_datasets(request):
    """Initialize shared AnnData datasets once per session."""
    data_source = getattr(request, "param", None)
    return get_anndata_registry(request, profile="smoke", data_source=data_source)


@pytest.fixture(scope="session")
def perf_anndata(request):
    """Initialize shared performance AnnData once per session."""
    data_source = getattr(request, "param", None)
    return get_anndata_registry(request, profile="perf", data_source=data_source)


@pytest.fixture(autouse=True)
def _reset_ggml_settings_between_tests():
    """Prevent global ggml_ot.settings state from leaking across tests.

    pytest runs in a single Python process by default, and imported modules are cached
    in sys.modules. ggml_ot exposes a module-level singleton `settings`, so mutations
    (e.g. setting device='cuda') persist unless explicitly reset.
    """

    try:
        yield
    finally:
        # Reset to library defaults after each test.
        for k in _PUBLIC_SETTINGS_KEYS:
            setattr(ggml_ot.settings, k, getattr(_DEFAULT_SETTINGS, k))
