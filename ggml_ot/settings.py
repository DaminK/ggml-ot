import sys
import torch

module_object = sys.modules[__name__]

# Module variables
module_object.settings = {}
""" Settings for ggml_ot package.
Attributes:
    n_threads (int): Number of threads to use for OT computations.
    device (torch.device): Device to use for computations.
    verbose (bool): Whether to print verbose output.
    figdir (str or None): Directory to save figures. If None, figures are not saved.
"""
module_object.settings.n_threads = 4
""" Number of threads to use for OT computations. """
module_object.settings.device = torch.device("cpu")
""" Device to use for computations. """
module_object.settings.verbose = True
""" Whether to print verbose output. """
module_object.settings.figdir = None
""" Directory to save figures. If None, figures are not saved. """

# Set random seed for reproducibility
torch.manual_seed(42)
