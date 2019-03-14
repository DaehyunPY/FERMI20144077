from typing import Callable

from scipy.optimize import least_squares, OptimizeResult
from numpy import ndarray


__all__ = ["weighted_least_squares"]


def weighted_least_squares(fun, x0, weights: ndarray, jac: Callable, **kwargs) -> OptimizeResult:
    def weighted_fun(x, *args, **kwargs) -> ndarray:
        return weights ** 0.5 * fun(x, *args, **kwargs)

    def weighted_jac(x, *args, **kwargs) -> ndarray:
        return weights[:, None] ** 0.5 * jac(x, *args, **kwargs)
    return least_squares(weighted_fun, x0, weighted_jac, **kwargs)
