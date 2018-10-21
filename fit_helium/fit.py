from math import inf, pi
from typing import Optional
from collections import OrderedDict

from importlib_resources import path
from pandas import read_excel
from scipy.optimize import least_squares, OptimizeResult
from numpy import sign, array, ndarray
from numpy.linalg import pinv

from . import res
from .solve import ymat_lambdified, jmat_lambdified, ymat_pretty
from .optimize import weighted_least_squares


__all__ = ["fit"]


def first(key: str) -> str:
    return key.split()[0]


with path(res, "simulated.xlsx") as pth:
    df = read_excel(pth).rename(first, axis='columns').set_index('photon')


def fit(photon: float, beta1m3_amp: float, beta1m3_shift: float, beta2: float,
        beta3_amp: float, beta3_shift: float, beta4: float,
        beta1m3_amp_err: Optional[float] = None,
        beta1m3_shift_err: Optional[float] = None,
        beta2_err: Optional[float] = None,
        beta3_amp_err: Optional[float] = None,
        beta3_shift_err: Optional[float] = None,
        beta4_err: Optional[float] = None,
        **kwargs) -> OrderedDict:
    """
    Fit helium beta parameters
    :param photon: Photon energy in eV
    :param beta1m3_amp:
    :param beta1m3_amp_err:
    :param beta1m3_shift: In radians
    :param beta1m3_shift_err:
    :param beta2:
    :param beta2_err:
    :param beta3_amp:
    :param beta3_amp_err:
    :param beta3_shift: In radians
    :param beta3_shift_err:
    :param beta4:
    :param beta4_err:
    :param kwargs: Keyword arguments which pass to function scipy.optimize.least_squares
    :return:
    """
    xref = {k: v for k, v in df.loc[photon].items() if k.startswith("coeff_") or k.startswith("eta_")}
    yref = ymat_pretty(**xref)

    if beta1m3_amp * yref["beta1m3_amp"] < 0:
        print("Sign of beta1m3_amp does not match with the reference. Force to flip the sign")
        beta1m3_amp *= -1
        beta1m3_shift += pi

    if beta3_amp * yref["beta3_amp"] < 0:
        print("Sign of beta3_amp does not match with the reference. Force to flip the sign")
        beta3_amp *= -1
        beta3_shift += pi

    def fun(x) -> ndarray:
        phi0, r, h = x
        ret: ndarray = ymat_lambdified(xref["coeff_s"], xref["coeff_p"], xref["coeff_d"],
                                       xref["eta_s"], xref["eta_p"], xref["eta_d"],
                                       phi0, r, h)
        diff = [beta1m3_amp, beta1m3_shift, beta2, beta3_amp, beta3_shift, beta4] - ret[:, 0]
        where_shift = array([False, True, False, False, True, False])
        divider = sign(diff) * [inf, 0, inf, inf, 0, inf] + where_shift * 2 * pi
        return (diff + where_shift * pi) % divider - where_shift * pi

    def jac(x) -> ndarray:
        phi0, r, h = x
        ret = jmat_lambdified(xref["coeff_s"], xref["coeff_p"], xref["coeff_d"],
                              xref["eta_s"], xref["eta_p"], xref["eta_d"],
                              phi0, r, h)
        return -ret[:, -3:]

    yerr = [beta1m3_amp_err, beta1m3_shift_err, beta2_err, beta3_amp_err, beta3_shift_err, beta4_err]
    if any(v is None for v in yerr):
        print("Error parameters is not passed. Ignore weights")
        ret: OptimizeResult = least_squares(
            fun, [0, 1, 1], jac=jac,
            bounds=[*zip([-2*pi, 2*pi], [0, inf], [0, inf])], **kwargs,
        )
        xerr = None
    else:
        yerr = array(yerr)
        ret = weighted_least_squares(
            fun, [0, 1, 1], weights=1 / yerr ** 2, jac=jac,
            bounds=[*zip([-2 * pi, 2 * pi], [0, inf], [0, inf])], **kwargs,
        )
        xerr = ((pinv(jac(ret.x)) ** 2) @ (yerr ** 2)) ** 0.5
    xkeys = ["phi0", "r", "h"]
    opt = OrderedDict(zip(xkeys, ret.x))
    return OrderedDict([
        ("opt", opt),
        ("err", OrderedDict(zip(xkeys, xerr))),
        ("fx", ymat_pretty(**xref, **opt)),
        ("report", ret),
    ])