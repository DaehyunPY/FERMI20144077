from enum import auto, IntEnum

from sympy import (Expr, symbols, Matrix, I, pi, exp, Ynm, Abs, cos, sin, arg, sqrt, re, legendre,
                   cancel, expand_func, simplify, expand, solve, lambdify)

__all__ = (
    'XKeys',
    'xkeys',
    'YKeys',
    'ykeys',
    'wonly_xkeys',
    'eta_ref',
    'ymat_lambdified',
    'yjacmat_lambdified',
)

# %% pads
c_sps, c_ps, c_dps = symbols('c_sps c_ps c_dps', positive=True)
eta_s, eta_p, eta_d = symbols('eta_s eta_p eta_d', real=True)
phi, the, vphi = symbols('phi theta varphi', real=True)

c0_ps = c_ps
c0_sps = -sqrt(3) / 3 * c_sps
c0_dps = sqrt(6) / 3 * c_dps

waves = {
    'm=0': (c0_sps * exp(eta_s * I) * Ynm(0, 0, the, vphi) +
            c0_ps * exp(eta_p * I + phi * I) * Ynm(1, 0, the, vphi) +
            c0_dps * exp(eta_d * I) * Ynm(2, 0, the, vphi)),
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
            .subs(sin(the) ** 2, 1 - cos(the) ** 2)
    )

    expr = cancel(left)
    term0_lft = expr.subs(the, pi / 2)
    expr = cancel((expr - term0_lft) / cos(the))
    term1_lft = expr.subs(the, pi / 2)
    expr = cancel((expr - term1_lft) / cos(the))
    term2_lft = expr.subs(the, pi / 2)
    expr = cancel((expr - term2_lft) / cos(the))
    term3_lft = expr.subs(the, pi / 2)
    expr = cancel((expr - term3_lft) / cos(the))
    term4_lft = expr.subs(the, pi / 2)
    expr = cancel((expr - term4_lft) / cos(the))
    term5_lft = expr.subs(the, pi / 2)
    expr = cancel((expr - term5_lft) / cos(the))
    term6_lft = expr.subs(the, pi / 2)
    expr = cancel((expr - term6_lft) / cos(the))

    if expr != 0:
        AssertionError('Valuable __expr is not zero!')

    # expand right term
    b0, b1, b2, b3, b4, b5, b6 = symbols(
        'b_0 b_1 b_2 b_3 b_4 b_5 b_6',
        real=True
    )
    right = (b0 +
             b1 * legendre(1, cos(the)) +
             b2 * legendre(2, cos(the)) +
             b3 * legendre(3, cos(the)) +
             b4 * legendre(4, cos(the)) +
             b5 * legendre(5, cos(the)) +
             b6 * legendre(6, cos(the)))

    expr = cancel(right)
    term0_rgt = expr.subs(the, pi / 2)
    expr = cancel((expr - term0_rgt) / cos(the))
    term1_rgt = expr.subs(the, pi / 2)
    expr = cancel((expr - term1_rgt) / cos(the))
    term2_rgt = expr.subs(the, pi / 2)
    expr = cancel((expr - term2_rgt) / cos(the))
    term3_rgt = expr.subs(the, pi / 2)
    expr = cancel((expr - term3_rgt) / cos(the))
    term4_rgt = expr.subs(the, pi / 2)
    expr = cancel((expr - term4_rgt) / cos(the))
    term5_rgt = expr.subs(the, pi / 2)
    expr = cancel((expr - term5_rgt) / cos(the))
    term6_rgt = expr.subs(the, pi / 2)
    expr = cancel((expr - term6_rgt) / cos(the))

    if expr != 0:
        AssertionError('Valuable __expr is not zero!')

    # solve equations
    b6_cmpx = simplify(cancel(solve(term6_lft - term6_rgt, b6)[0]))
    b6_real = simplify(re(expand(b6_cmpx)))

    b5_cmpx = simplify(cancel(solve(term5_lft - term5_rgt, b5)[0]))
    b5_real = simplify(re(expand(b5_cmpx)))
    b5_amp = simplify(cancel(sqrt(b5_real ** 2 + b5_real.diff(phi) ** 2).subs(phi, 0)))
    b5_shift = arg(b5_real.subs(phi, 0) + I * b5_real.diff(phi).subs(phi, 0))

    b4_cmpx = simplify(cancel(solve((term4_lft - term4_rgt)
                                    .subs(b6, b6_cmpx), b4)[0]))
    b4_real = simplify(re(expand(b4_cmpx)))

    b3_cmpx = simplify(cancel(solve((term3_lft - term3_rgt)
                                    .subs(b5, b5_cmpx), b3)[0]))
    b3_real = simplify(re(expand(b3_cmpx)))
    b3_amp = simplify(cancel(sqrt(b3_real ** 2 + b3_real.diff(phi) ** 2).subs(phi, 0)))
    b3_shift = arg(b3_real.subs(phi, 0) + I * b3_real.diff(phi).subs(phi, 0))

    b2_cmpx = simplify(cancel(solve((term2_lft - term2_rgt)
                                    .subs(b6, b6_cmpx)
                                    .subs(b4, b4_cmpx), b2)[0]))
    b2_real = simplify(re(expand(b2_cmpx)))

    b1_cmpx = simplify(cancel(solve((term1_lft - term1_rgt)
                                    .subs(b5, b5_cmpx)
                                    .subs(b3, b3_cmpx), b1)[0]))
    b1_real = simplify(re(expand(b1_cmpx)))
    b1_amp = simplify(cancel(sqrt(b1_real ** 2 + b1_real.diff(phi) ** 2).subs(phi, 0)))
    b1_shift = arg(b1_real.subs(phi, 0) + I * b1_real.diff(phi).subs(phi, 0))

    b0_cmpx = simplify(cancel(solve((term0_lft - term0_rgt)
                                    .subs(b6, b6_cmpx)
                                    .subs(b4, b4_cmpx)
                                    .subs(b2, b2_cmpx), b0)[0]))
    b0_real = simplify(re(expand(b0_cmpx)))
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
class XKeys(IntEnum):  # length: 6
    C_SPS = 0
    C_PS = auto()
    C_DPS = auto()
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


xkeys = [k.name.lower() for k in XKeys]
ykeys = [k.name.lower() for k in YKeys]
wonly_xkeys = {XKeys.C_SPS, XKeys.C_DPS, XKeys.ETA_S, XKeys.ETA_D}
eta_ref = XKeys.ETA_D

print("Solving the He PAD equations...")
solved = solve_eq(pads['summed'])

print("Lambdifying solved b parameters...")
xmat = Matrix((c_sps, c_ps, c_dps, eta_s, eta_p, eta_d))
ymat = Matrix([solved[k] for k in ykeys])
ymat_lambdified = lambdify(xmat, ymat, 'numpy')
yjacmat = ymat.jacobian(xmat)  # shape: (7, 6)
yjacmat_lambdified = lambdify(xmat, yjacmat, 'numpy')
