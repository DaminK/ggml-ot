"""Package-wide settings for ggml_ot.

This module provides a simple Settings object as a module-level
singleton named ``settings`` that is used throughout the package and can be modified at runtime.

Example
-------
>>> from ggml_ot import settings
Change individual settings:
>>> settings.verbose = False
Or multiple settings:
>>> settings.update(n_threads=8, device="cpu")

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


@dataclass
class Settings:
    """Package-wide configuration shared through ``ggml_ot.settings``.

    Read or assign the public attributes on this shared object to configure
    the package at runtime.
    Use :meth:`~ggml_ot.settings.Settings.update` to change several settings
    at once. Each property documents its default value and what happens when
    it is changed.

    Examples
    --------
    .. code-block:: python

        # Set individual properties, e.g., the seed for reproducibility
        ggml_ot.settings.random_seed = 123

        # Set multiple properties at once, e.g., to use 8 CPU threads and force CPU computation
        ggml_ot.settings.update(n_threads=8, device="cpu")

        # Or set them directly, e.g. plotting options for saving figures
        ggml_ot.settings.figdir = "figures"
        ggml_ot.settings.figformat = "png"
        ggml_ot.settings.figdpi = 300
    """

    _n_threads: int = 4
    _device: torch.device = torch.device("cpu")
    verbose: bool = True
    """Whether to print verbose output where supported. Defaults to ``True``."""
    _figdir: Optional[Path] = field(default=None, repr=False)
    _datadir: Optional[Path] = field(default=None, repr=False)
    figformat: str = "pdf"
    """Default extension for saved figures, such as ``"pdf"``, ``"png"``,
    or ``"svg"``. Default: ``"pdf"``. An explicit filename can override it.
    """
    figdpi: int = 150
    """Resolution in dots per inch passed to figure saving. Defaults to ``150``."""
    _random_seed: Optional[int] = 42
    torch_generator: torch.Generator = None
    """PyTorch generator managed by :attr:`random_seed`.

    Initialized from the default seed of ``42``. Setting ``random_seed`` to
    an integer resets it; setting the seed to ``None`` makes this ``None``.
    Prefer setting ``random_seed`` to configure reproducibility.

    :meta private:
    """
    numpy_generator: np.random.Generator = None
    """NumPy generator managed by :attr:`random_seed`.

    Initialized from the default seed of ``42``. Assigning ``random_seed``
    creates a new ``numpy.random.default_rng``; a seed of ``None`` creates
    a generator seeded from fresh entropy.

    :meta private:
    """
    restore_adata_params: bool = False
    """Whether to restore stored ggml parameters from an AnnData object when
    initializing ``AnnData_TripletDataset``. Default: ``False``.
    """
    init_strategy: str = "orthonormal"
    """Initialization strategy for the learned map ``map_A``.

    ``"orthonormal"`` (default) initializes orthonormal rows.
    ``"orthogonal"`` scales those rows by the square root of the input
    dimension. ``"xavier"`` uses Xavier (Glorot) uniform initialization.
    ``"random"`` draws i.i.d. standard normal entries.
    The first two strategies also use Xavier uniform initialization when
    the number of output components exceeds the input dimension.
    """

    def __post_init__(self):
        # Sync torch's CPU thread pool with configured default.
        self.n_threads = self._n_threads
        # Init torch generator from default seed.
        if self._random_seed is not None:
            self.random_seed = self._random_seed

    @property
    def n_threads(self):
        """Number of CPU computation threads. Defaults to ``4``.

        Assignment converts the value to an integer and updates PyTorch's
        CPU thread pool with ``torch.set_num_threads``. Values below ``1``
        raise :class:`ValueError`.
        """
        return self._n_threads

    @n_threads.setter
    def n_threads(self, value):
        threads = int(value)
        if threads < 1:
            raise ValueError("settings.n_threads must be >= 1")
        self._n_threads = threads
        torch.set_num_threads(threads)

    @property
    def device(self):
        """Device for PyTorch computations. Defaults to ``torch.device("cpu")``.

        Accepts a :class:`torch.device` or a string such as ``"cpu"``,
        ``"cuda"``, or ``"cuda:0"``; reading returns a ``torch.device``.
        Selecting CUDA raises :class:`ValueError` if CUDA is unavailable
        or the requested device index is out of range.
        """
        return self._device

    @device.setter
    def device(self, value):
        device = value if isinstance(value, torch.device) else torch.device(str(value))

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "settings.device was set to a CUDA device, but torch.cuda.is_available() is False. "
                    "Use device='cpu' or install/configure CUDA."
                )
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError(
                    f"settings.device was set to {device!s}, but only {torch.cuda.device_count()} CUDA device(s) are available."
                )

        self._device = device

    @property
    def figdir(self) -> Optional[Path]:
        """Directory for saved figures. Defaults to ``None``.

        Accepts a string, :class:`pathlib.Path`, or ``None``. Assigning a
        path creates the directory and any missing parents; reading returns
        a ``Path``. Set this before requesting plot saving with ``save=True``
        or a relative filename. Setting a directory alone does not save plots.
        """
        return self._figdir

    @figdir.setter
    def figdir(self, value: Union[str, Path, None]):
        if value is None:
            self._figdir = None
            return
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        self._figdir = path

    @property
    def datadir(self) -> Optional[Path]:
        """On-disk cache directory for dataset downloads. Defaults to ``None``.

        Accepts a string, :class:`pathlib.Path`, or ``None``. Assigning a
        path creates the directory and any missing parents; reading returns
        a ``Path``. If ``None``, dataset loaders fall back to ``Path("data/")``
        relative to the current working directory and emit a one-line hint
        on first use.
        """
        return self._datadir

    @datadir.setter
    def datadir(self, value: Union[str, Path, None]):
        if value is None:
            self._datadir = None
            return
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        self._datadir = path

    @property
    def random_seed(self):
        """Random seed for package operations. Defaults to ``42``.

        Assign an integer to seed PyTorch with ``torch.manual_seed`` and
        recreate the internal NumPy generator with ``numpy.random.default_rng``.
        Assigning the same seed again restarts these random streams.

        Assign ``None`` to create a NumPy generator from fresh entropy and
        stop passing an explicit PyTorch generator. This does not reset
        PyTorch's global random state. The setting does not seed Python's
        ``random`` module or NumPy's legacy global random state.

        Set the seed before generating data or training. A fixed seed does
        not by itself guarantee deterministic results for every operation
        or across devices.

        Examples
        --------
        >>> ggml_ot.settings.random_seed = 42
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        if value is None:
            self._random_seed = value
            self.torch_generator = None
            self.numpy_generator = np.random.default_rng()
            return

        seed = int(value)
        self._random_seed = seed
        self.torch_generator = torch.manual_seed(seed)
        self.numpy_generator = np.random.default_rng(seed)

    def update(self, **kwargs: Any) -> None:
        """Change several settings using the same setters as direct assignment.

        Parameters
        ----------
        **kwargs
            Setting names and values, for example
            ``settings.update(random_seed=123, n_threads=8)``.

        Raises
        ------
        AttributeError
            If a name is unknown.

        Notes
        -----
        Updates are applied in order. If an update fails, earlier changes
        remain in effect.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown setting '{k}'")
            setattr(self, k, v)

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary snapshot of selected settings.

        The snapshot uses the legacy keys ``_device`` and ``_random_seed``
        and omits ``numpy_generator``. It is not a complete export of all
        public attributes. Prefer the public properties for reading and
        changing individual settings.
        """
        return {
            "n_threads": self.n_threads,
            "_device": self._device,
            "verbose": self.verbose,
            "figdir": self.figdir,
            "datadir": self.datadir,
            "figformat": self.figformat,
            "figdpi": self.figdpi,
            "_random_seed": self._random_seed,
            "torch_generator": self.torch_generator,
            "restore_adata_params": self.restore_adata_params,
            "init_strategy": self.init_strategy,
        }


# Module-level singleton settings object. Import this from anywhere in the
# package to read or mutate global settings.
settings = Settings()
