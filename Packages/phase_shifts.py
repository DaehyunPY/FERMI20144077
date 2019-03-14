import typing
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, OptimizeResult
from scipy.linalg import svd


__all__ = ["fit", "load"]


def model(
        x: np.ndarray,
        freq: float,
        shift: float,
        amp: float,
        offset: float) -> np.ndarray:
    return amp * np.cos(freq * x - shift) + offset


def formatted(res: OptimizeResult, param_names: typing.List[str]) -> dict:
    cost = 2 * res.cost  # res.cost is half sum of squares!
    popt = res.x

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, vh = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    vh = vh[:s.size]
    pcov = vh.T / s**2 @ vh

    m, n = res.jac.shape  # m is num of samples; n is num of parameters.
    s_sq = cost / (m - n)
    pcov = pcov * s_sq
    return [
        {
            "Name": name,
            "Value": value,
            "Std err": err,
            "Vary": True,
        }
        for name, value, err in zip(param_names, popt, np.diag(pcov)**0.5)
    ]


def fit(df: pd.DataFrame, ref: pd.DataFrame) -> typing.List[dict]:
    merged = df.merge(ref, on="Opt phase (mach unit)", suffixes=("", " (ref)"))
    x = merged["Opt phase (mach unit)"].values
    freq = 2 * np.pi
    y0, y1 = merged["Yield"].values, merged["Yield (ref)"].values

    def residual(params: tuple) -> np.ndarray:
        shift, amp, offset, ref_offset = params
        fx0 = model(x, freq, shift, amp, offset)
        fx1 = model(x, freq, shift - np.pi, amp, ref_offset)
        return np.append(y0 - fx0, y1 - fx1)

    at = y0.argmax()
    ph = freq * x % (2 * np.pi)
    ret = least_squares(
        residual,
        x0=(
            ph[at],  # shift
            y0.std(),  # amp
            y0.mean(),  # offset
            y1.mean(),  # ref_offset
        ),
        bounds=(
            (
                -np.inf,  # shift
                0,  # amp
                0,  # offset
                0,  # ref_offset
            ),
            (
                np.inf,  # shift
                np.inf,  # amp
                np.inf,  # offset
                np.inf,  # ref_offset
            ),
        )
    )
    keys = ["shift", "amp", "offset", "ref_offset"]
    return [
        {
            "Name": "freq",
            "Value": freq,
            "Std err": 0,
            "Vary": False,
        },
        *formatted(ret, keys),
    ]


def load(
        params: typing.List[str],
        ) -> typing.Callable[[np.ndarray], np.ndarray]:
    kwargs = {
        d["Name"]: d["Value"]
        for d in params
        if not d["Name"].startswith("ref_")
    }
    return partial(model, **kwargs)
