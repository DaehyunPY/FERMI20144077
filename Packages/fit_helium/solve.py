from functools import wraps
from collections import OrderedDict

from sympy import symbols, lambdify, Matrix, sqrt, cos


__all__ = ["ymat_lambdified", "jmat_lambdified", "ymat_pretty"]


coeff_s, coeff_p, coeff_d, eta_s, eta_p, eta_d = symbols("coeff_s coeff_p coeff_d eta_s eta_p eta_d", real=True)
phi0, r, h = symbols('phi0 r h', real=True)

xmat = Matrix([coeff_s, coeff_p, coeff_d, eta_s, eta_p, eta_d, phi0, r, h])
ydict = OrderedDict([
    ('beta1m3_amp', 2 * sqrt(3) * coeff_p * coeff_s / (coeff_d ** 2 + coeff_p ** 2 + coeff_s ** 2)),
    ('beta1m3_shift', eta_s - eta_p),
    ("beta2", ((10 * coeff_d ** 2 + 14 * sqrt(5) * coeff_d * coeff_s * cos(eta_d - eta_s) + 14 * coeff_p ** 2)
               / (7 * (coeff_d ** 2 + coeff_p ** 2 + coeff_s ** 2)))),
    ("beta3_amp", 6 * sqrt(15) * coeff_d * coeff_p / (5 * (coeff_d ** 2 + coeff_p ** 2 + coeff_s ** 2))),
    ("beta3_shift", eta_d - eta_p),
    ("beta4", 18 * coeff_d ** 2 / (7 * (coeff_d ** 2 + coeff_p ** 2 + coeff_s ** 2))),
])
ymat = Matrix([
    ydict["beta1m3_amp"].subs(coeff_p, r * coeff_p) * h,
    ydict["beta1m3_shift"] + phi0,
    ydict["beta2"].subs(coeff_p, r * coeff_p),
    ydict["beta3_amp"].subs(coeff_p, r * coeff_p) * h,
    ydict["beta3_shift"] + phi0,
    ydict["beta4"].subs(coeff_p, r * coeff_p),
])
jmat = ymat.jacobian(xmat)
ymat_lambdified = lambdify(xmat, ymat, "numpy")
jmat_lambdified = lambdify(xmat, jmat, "numpy")


@wraps(ymat_lambdified)
def ymat_pretty(coeff_s: float, coeff_p: float, coeff_d: float, eta_s: float, eta_p: float = 0, eta_d: float = 0,
                phi0: float = 0, r: float = 1, h: float = 1) -> OrderedDict:
    """
    Calculate beta parameters of helium PAD
    :param coeff_s:
    :param coeff_p:
    :param coeff_d:
    :param eta_s: In radians
    :param eta_p: In radians
    :param eta_d: In radians
    :param phi0: Phase offset in radians
    :param r: Intensity scale factor
    :param h: Beta amplitude scale factor
    :return: Calculated beta parameters
    """
    ret = ymat_lambdified(coeff_s, coeff_p, coeff_d, eta_s, eta_p, eta_d, phi0, r, h)
    return OrderedDict(zip(ydict.keys(), ret[:, 0]))
