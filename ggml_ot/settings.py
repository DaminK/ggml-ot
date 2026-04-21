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
    """Container for global settings of ggml_ot.

    Attributes are public and can be set directly. To change multiple settings an update function is provided.
    """

    _n_threads: int = 4
    """ Number of threads to use for cpu computations, defaults to 4. """
    _device: torch.device = torch.device("cpu")
    """Device to use for torch computations."""
    verbose: bool = True
    """ Whether to print verbose output """
    _figdir: Optional[Path] = field(default=None, repr=False)
    """ Directory to save figures. If None, figures are not saved """
    figformat: str = "pdf"
    """ Default file format for saved figures (e.g. 'pdf', 'png', 'svg') """
    figdpi: int = 150
    """ Default DPI for saved figures """
    _random_seed: Optional[int] = 42
    """ Random seed to fix random initializations for reproducibility. Set to None for non-deterministic behavior. """
    torch_generator: torch.Generator = None
    """ Torch random generator initialized with random_seed. If random_seed is None, this is None as well. """
    numpy_generator: np.random.Generator = None
    """ NumPy random generator initialized with random_seed. If random_seed is None, an unseeded default_rng is used. """
    restore_adata_params: bool = False
    """ Whether to restore ggml parameters from previously initialized AnnData objects when initializing AnnData_TripletDataset """
    init_strategy: str = "orthonormal"
    """ Initialization strategy for map_A. One of 'orthonormal' (default), 'orthogonal', 'random'. """

    def __post_init__(self):
        # Sync torch's CPU thread pool with configured default.
        self.n_threads = self._n_threads
        # Init torch generator from default seed.
        if self._random_seed is not None:
            self.random_seed = self._random_seed

    @property
    def n_threads(self):
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
    def random_seed(self):
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
        """Update settings from keyword args. Unknown keys raise AttributeError."""
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown setting '{k}'")
            setattr(self, k, v)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_threads": self.n_threads,
            "_device": self._device,
            "verbose": self.verbose,
            "figdir": self.figdir,
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
