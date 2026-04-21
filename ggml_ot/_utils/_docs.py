"""Documentation and introspection helpers.

Utilities for docstring forwarding and default-argument extraction that
are not tied to any specific domain submodule.
"""

from __future__ import annotations

from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def wraps(wrapper: Callable[P, T]):
    """Copy the docstring of *wrapper* onto the decorated function.

    A lightweight alternative to :func:`functools.wraps` that only
    transfers ``__doc__`` (no ``__name__`` / ``__module__`` copying).
    """

    def decorator(func: Callable) -> Callable[P, T]:
        func.__doc__ = wrapper.__doc__
        return func

    return decorator


# Source - https://stackoverflow.com/a
# Posted by conmak, modified by community.
# Retrieved 2025-12-19, License - CC BY-SA 4.0
def get_defaults(fn) -> dict:
    """Return the default keyword / positional argument values of *fn*.

    Parameters
    ----------
    fn
        A function or method.

    Returns
    -------
    dict
        ``{param_name: default_value, ...}`` for every argument that has
        a default.
    """
    output: dict = {}
    if fn.__defaults__ is not None:
        default_varnames = list(fn.__code__.co_varnames)[: fn.__code__.co_argcount][-len(fn.__defaults__) :]
        output.update(dict(zip(default_varnames, fn.__defaults__)))
    if fn.__kwdefaults__ is not None:
        output.update(fn.__kwdefaults__)
    return output
