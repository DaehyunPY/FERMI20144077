from enum import auto, IntEnum
from os.path import isfile
from functools import wraps
from itertools import chain, repeat, islice

from cloudpickle import dump, load
from sympy import (
    Expr, symbols, Rational, Matrix, I, pi, exp, Ynm, Abs, cos, sin, sqrt, re, legendre,
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
coeff_psp, coeff_pdp, coeff_sp, coeff_dp, coeff_fdp = symbols(
    'coeff_psp coeff_pdp coeff_sp coeff_dp coeff_fdp', real=True,
)
eta_sp, eta_psp, eta_pdp, eta_dp, eta_fdp = symbols(
    'eta_sp eta_psp eta_pdp eta_dp eta_fdp', real=True,
)
phi, theta, varphi = symbols('phi theta varphi', real=True)

c0_sp = -sqrt(3) / 3 * coeff_sp
c0_dp, c1_dp = sqrt(30) / 15 * coeff_dp, sqrt(10) / 10 * coeff_dp
c0_psp = -Rational(1, 3) * coeff_psp
c0_pdp, c1_pdp = -Rational(2, 15) * coeff_pdp, -Rational(1, 10) * coeff_pdp
c0_fdp, c1_fdp = sqrt(14) / 35 * coeff_fdp, 2 * sqrt(21) / 105 * coeff_fdp

waves = {
    'm=0': (c0_psp * exp(eta_psp*I) * Ynm(1, 0, theta, varphi) +
            c0_pdp * exp(eta_pdp*I) * Ynm(1, 0, theta, varphi) +
            c0_fdp * exp(eta_fdp * I) * Ynm(3, 0, theta, varphi) +
            c0_sp * exp(eta_sp * I + phi * I) * Ynm(0, 0, theta, varphi) +
            c0_dp * exp(eta_dp * I + phi * I) * Ynm(2, 0, theta, varphi)),
    'm=1': (c1_pdp * exp(eta_pdp*I) * Ynm(1, 1, theta, varphi) +
            c1_fdp * exp(eta_fdp * I) * Ynm(3, 1, theta, varphi) +
            c1_dp * exp(eta_dp * I + phi * I) * Ynm(2, 1, theta, varphi)),
}
pads = {
    k: sum(Abs(term) ** 2 for term in expr)
    for k, expr in {
        'm=0': [waves['m=0']],
        'm=1': [waves['m=1']],
        'summed': [waves['m=1'].subs(varphi, -varphi), waves['m=0'], waves['m=1']],
    }.items()
}


# %% solve pad eq
def solve_eq(pad: Expr) -> dict:
    # expand left term
    left = (
        expand_func(pad)
            .subs(sin(theta) ** 2, 1 - cos(theta) ** 2)
    )
    terms_lft = list(islice(chain(expend_cos(left, theta), repeat(0)), 7))

    # expand right term
    b0, b1, b2, b3, b4, b5, b6 = symbols(
        'b_0 b_1 b_2 b_3 b_4 b_5 b_6',
        real=True
    )
    right = (b0 +
             b1 * legendre(1, cos(theta)) +
             b2 * legendre(2, cos(theta)) +
             b3 * legendre(3, cos(theta)) +
             b4 * legendre(4, cos(theta)) +
             b5 * legendre(5, cos(theta)) +
             b6 * legendre(6, cos(theta)))
    terms_rgt = list(expend_cos(right, theta))

    # solve equations
    b6_cmpx = simplify(cancel(solve(terms_lft[6] - terms_rgt[6], b6)[0]))
    b6_real = simplify(re(expand(b6_cmpx)))
    b5_cmpx = simplify(cancel(solve(terms_lft[5] - terms_rgt[5], b5)[0]))
    b5_real = simplify(re(expand(b5_cmpx)))
    b5_amp, b5_shift = amp_and_shift(b5_real, phi)
    b4_cmpx = simplify(cancel(solve((terms_lft[4] - terms_rgt[4])
                                    .subs(b6, b6_cmpx), b4)[0]))
    b4_real = simplify(re(expand(b4_cmpx)))
    b3_cmpx = simplify(cancel(solve((terms_lft[3] - terms_rgt[3])
                                    .subs(b5, b5_cmpx), b3)[0]))
    b3_real = simplify(re(expand(b3_cmpx)))
    b3_amp, b3_shift = amp_and_shift(b3_real, phi)
    b2_cmpx = simplify(cancel(solve((terms_lft[2] - terms_rgt[2])
                                    .subs(b6, b6_cmpx)
                                    .subs(b4, b4_cmpx), b2)[0]))
    b2_real = simplify(re(expand(b2_cmpx)))
    b1_cmpx = simplify(cancel(solve((terms_lft[1] - terms_rgt[1])
                                    .subs(b5, b5_cmpx)
                                    .subs(b3, b3_cmpx), b1)[0]))
    b1_real = simplify(re(expand(b1_cmpx)))
    b1_amp, b1_shift = amp_and_shift(b1_real, phi)
    b0_cmpx = simplify(cancel(solve((terms_lft[0] - terms_rgt[0])
                                    .subs(b6, b6_cmpx)
                                    .subs(b4, b4_cmpx)
                                    .subs(b2, b2_cmpx), b0)[0]))
    b0_real = simplify(re(expand(b0_cmpx)))
    # b1m3_real = simplify(cancel(b1_real - b3_real * 3 / 2))
    # b1m3_amp, b1m3_shift = amp_and_shift(b1m3_real, phi)
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
        'b5': b5_real,
        'b5_amp': b5_amp,
        'b5_shift': b5_shift,
        'b6': b6_real,
    }


# %% lambdify pads
class XKeys(IntEnum):  # length: 10
    COEFF_SP = 0
    COEFF_PSP = auto()
    COEFF_PDP = auto()
    COEFF_DP = auto()
    COEFF_FDP = auto()
    ETA_SP = auto()
    ETA_PSP = auto()
    ETA_PDP = auto()
    ETA_DP = auto()
    ETA_FDP = auto()


class YKeys(IntEnum):  # length: 7
    B0 = 0
    B1_AMP = auto()
    B1_SHIFT = auto()
    B2 = auto()
    B3_AMP = auto()
    B3_SHIFT = auto()
    B4 = auto()


if not isfile('solved_neon_eq.db'):
    print("Solving the Ne PAD equations...")
    solved = solve_eq(pads['summed'])

    print("Lambdifying solved b parameters...")
    xmat = Matrix((coeff_sp, coeff_psp, coeff_pdp, coeff_dp, coeff_fdp,
                   eta_sp, eta_psp, eta_pdp, eta_dp, eta_fdp))
    ymat = Matrix([solved[k.name.lower()] for k in YKeys])
    ymat_lambdified = lambdify(xmat, ymat, 'numpy')
    yjacmat = ymat.jacobian(xmat)  # shape: (7, 10)
    yjacmat_lambdified = lambdify(xmat, yjacmat, 'numpy')

    with open('solved_neon_eq.db', 'wb') as f:
        print("Storing the answer...")
        dump({
            'solved': solved,
            'ymat_lambdified': ymat_lambdified,
            'yjacmat_lambdified': yjacmat_lambdified,
        }, f)
else:
    with open('solved_neon_eq.db', 'rb') as f:
        db = load(f)
        solved = db['solved']
        ymat_lambdified = db['ymat_lambdified']
        yjacmat_lambdified = db['yjacmat_lambdified']


@wraps(ymat_lambdified)
def ymat_pretty(
        coeff_sp, coeff_psp, coeff_pdp, coeff_dp, coeff_fdp,
        eta_sp, eta_psp, eta_pdp, eta_dp, eta_fdp,
    ):
    ret = ymat_lambdified(
        coeff_sp, coeff_psp, coeff_pdp, coeff_dp, coeff_fdp,
        eta_sp, eta_psp, eta_pdp, eta_dp, eta_fdp,
    )
    return {k.name.lower(): v for k, v in zip(YKeys, ret[:, 0])}
