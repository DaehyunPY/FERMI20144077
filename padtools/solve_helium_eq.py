from enum import auto, IntEnum
from os.path import isfile
from functools import wraps
from itertools import chain, repeat, islice

from cloudpickle import dump, load
from sympy import (
    Expr, symbols, Matrix, I, exp, Ynm, Abs, cos, sin, re, legendre,
    cancel, expand_func, simplify, expand, solve, lambdify,
)

from .tools import expend_cos, amp_and_shift

__all__ = [
    'solved',
    'XKeys',
    'YKeys',
    'ymat_lambdified',
    'yjacmat_lambdified',
    'ymat_pretty',
]

# %% pads
coeff_s, coeff_p, coeff_d = symbols(
    'coeff_s coeff_p coeff_d', positive=True,
)
eta_s, eta_p, eta_d = symbols(
    'eta_s eta_p eta_d', real=True,
)
phi, theta, varphi = symbols('phi theta varphi', real=True)

c0_ps, c0_sps, c0_dps = coeff_p, coeff_s, coeff_d

waves = {
    'm=0': (c0_sps * exp(eta_s * I) * Ynm(0, 0, theta, varphi) +
            c0_ps * exp(eta_p * I + phi * I) * Ynm(1, 0, theta, varphi) +
            c0_dps * exp(eta_d * I) * Ynm(2, 0, theta, varphi)),
}
pads = {
    'm=0': Abs(waves['m=0']) ** 2,
    'summed': Abs(waves['m=0']) ** 2,
}


# %% solve pad eq
def solve_eq(pad: Expr) -> dict:
    # expand left term
    left = (
        expand_func(pad)
            .subs(sin(theta) ** 2, 1 - cos(theta) ** 2)
    )
    terms_lft = list(islice(chain(expend_cos(left, theta), repeat(0)), 5))

    # expand right term
    b0, b1, b2, b3, b4 = symbols(
        'b_0 b_1 b_2 b_3 b_4',
        real=True
    )
    right = (b0 +
             b1 * legendre(1, cos(theta)) +
             b2 * legendre(2, cos(theta)) +
             b3 * legendre(3, cos(theta)) +
             b4 * legendre(4, cos(theta)))
    terms_rgt = list(expend_cos(right, theta))

    # solve equations
    b4_cmpx = simplify(cancel(solve(terms_lft[4] - terms_rgt[4], b4)[0]))
    b4_real = simplify(re(expand(b4_cmpx)))
    b3_cmpx = simplify(cancel(solve(terms_lft[3] - terms_rgt[3], b3)[0]))
    b3_real = simplify(re(expand(b3_cmpx)))
    b3_amp, b3_shift = amp_and_shift(b3_real, phi)
    b2_cmpx = simplify(cancel(solve((terms_lft[2] - terms_rgt[2])
                                    .subs(b4, b4_cmpx), b2)[0]))
    b2_real = simplify(re(expand(b2_cmpx)))
    b1_cmpx = simplify(cancel(solve((terms_lft[1] - terms_rgt[1])
                                    .subs(b3, b3_cmpx), b1)[0]))
    b1_real = simplify(re(expand(b1_cmpx)))
    b1_amp, b1_shift = amp_and_shift(b1_real, phi)
    b0_cmpx = simplify(cancel(solve((terms_lft[0] - terms_rgt[0])
                                    .subs(b4, b4_cmpx)
                                    .subs(b2, b2_cmpx), b0)[0]))
    b0_real = simplify(re(expand(b0_cmpx)))
    b1m3_real = simplify(cancel(b1_real - b3_real * 2 / 3))
    b1m3_amp, b1m3_shift = amp_and_shift(b1m3_real, phi)
    return {
        'b0': b0_real,
        'b1': b1_real,
        'b1_amp': b1_amp,
        'b1_shift': b1_shift,
        'b2': b2_real,
        'b3': b3_real,
        'b3_amp': b3_amp,
        'b3_shift': b3_shift,
        'b4': b4_real,
        'b1m3': b1m3_real,
        'b1m3_amp': b1m3_amp,
        'b1m3_shift': b1m3_shift,
    }


# %% lambdify pads
class XKeys(IntEnum):  # length: 6
    COEFF_S = 0
    COEFF_P = auto()
    COEFF_D = auto()
    ETA_S = auto()
    ETA_P = auto()
    ETA_D = auto()


class YKeys(IntEnum):  # length: 7
    B0 = 0
    B1_AMP = auto()
    B1_SHIFT = auto()
    B2 = auto()
    B3_AMP = auto()
    B3_SHIFT = auto()
    B4 = auto()


class AllKeys(IntEnum):  # length: 7
    B0 = 0
    B1_AMP = auto()
    B1_SHIFT = auto()
    B2 = auto()
    B3_AMP = auto()
    B3_SHIFT = auto()
    B4 = auto()
    B1M3_AMP = auto()
    B1M3_SHIFT = auto()


if not isfile('solved_helium_eq.db'):
    print("Solving the He PAD equations...")
    solved = solve_eq(pads['summed'])

    print("Lambdifying solved b parameters...")
    xmat = Matrix((coeff_s, coeff_p, coeff_d, eta_s, eta_p, eta_d))
    ymat = Matrix([solved[k.name.lower()] for k in YKeys])
    allmat = Matrix([solved[k.name.lower()] for k in AllKeys])
    ymat_lambdified = lambdify(xmat, ymat, 'numpy')
    allmat_lambdified = lambdify(xmat, allmat, 'numpy')
    yjacmat = ymat.jacobian(xmat)  # shape: (7, 6)
    yjacmat_lambdified = lambdify(xmat, yjacmat, 'numpy')

    with open('solved_helium_eq.db', 'wb') as f:
        print("Storing the answer...")
        dump({
            'solved': solved,
            'ymat_lambdified': ymat_lambdified,
            'allmat_lambdified': allmat_lambdified,
            'yjacmat_lambdified': yjacmat_lambdified,
        }, f)
else:
    with open('solved_helium_eq.db', 'rb') as f:
        db = load(f)
        solved = db['solved']
        ymat_lambdified = db['ymat_lambdified']
        allmat_lambdified = db['allmat_lambdified']
        yjacmat_lambdified = db['yjacmat_lambdified']


@wraps(ymat_lambdified)
def ymat_pretty(coeff_s, coeff_p, coeff_d, eta_s, eta_p, eta_d):
    ret = allmat_lambdified(coeff_s, coeff_p, coeff_d, eta_s, eta_p, eta_d)
    return {k.name.lower(): v for k, v in zip(AllKeys, ret[:, 0])}
