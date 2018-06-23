from sympy import (Expr, symbols, I, pi, exp, Ynm, Abs, cos, sin, arg, sqrt, re, legendre,
                   cancel, expand_func, simplify, expand, solve, lambdify)

__all__ = (
    'neon_pad',
)


# %% pads
c_psp, c_pdp, c_fdp, c_sp, c_dp = symbols('c_psp c_pdp c_fdp c_sp c_dp', positive=True)
eta_p, eta_f, eta_s, eta_d = symbols('eta_p eta_f eta_s eta_d', real=True)
phi, the, vphi = symbols('phi theta varphi', real=True)

c0_sp = -sqrt(3) / 3 * c_sp
c0_dp, c1_dp = sqrt(6) / 3 * c_dp, sqrt(2) / 2 * c_dp
c0_psp = -sqrt(3) / 3 * c_psp
c0_pdp, c1_pdp = -2 * sqrt(15) / 15 * c_pdp, -sqrt(15) / 10 * c_pdp
c0_fdp, c1_fdp = sqrt(10) / 5 * c_fdp, 2 * sqrt(15) / 15 * c_fdp

waves = {
    'm=0': (c0_psp * exp(eta_p * I) * Ynm(1, 0, the, vphi) +
            c0_pdp * exp(eta_p * I) * Ynm(1, 0, the, vphi) +
            c0_fdp * exp(eta_f * I) * Ynm(3, 0, the, vphi) +
            c0_sp * exp(eta_s * I + phi * I) * Ynm(0, 0, the, vphi) +
            c0_dp * exp(eta_d * I + phi * I) * Ynm(2, 0, the, vphi)),
    'm=1': (c1_pdp * exp(eta_p * I) * Ynm(1, 1, the, vphi) +
            c1_fdp * exp(eta_f * I) * Ynm(3, 1, the, vphi) +
            c1_dp * exp(eta_d * I + phi * I) * Ynm(2, 1, the, vphi)),
}
pads = {
    'm=0': Abs(waves['m=0']) ** 2,
    'm=1': Abs(waves['m=1']) ** 2,
    'summed': (Abs(waves['m=1'].subs(vphi, -vphi)) ** 2 +
               Abs(waves['m=0']) ** 2 +
               Abs(waves['m=1']) ** 2),
}


# %% solve
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
    b5_amp = simplify(cancel(sqrt(b5_real**2 + b5_real.diff(phi)**2).subs(phi, 0)))
    b5_shift = -arg(b5_cmpx.subs(phi, 0)) % (2*pi)

    b4_cmpx = simplify(cancel(solve((term4_lft - term4_rgt)
                                    .subs(b6, b6_cmpx), b4)[0]))
    b4_real = simplify(re(expand(b4_cmpx)))

    b3_cmpx = simplify(cancel(solve((term3_lft - term3_rgt)
                                    .subs(b5, b5_cmpx), b3)[0]))
    b3_real = simplify(re(expand(b3_cmpx)))
    b3_amp = simplify(cancel(sqrt(b3_real**2 + b3_real.diff(phi)**2).subs(phi, 0)))
    b3_shift = -arg(b3_cmpx.subs(phi, 0)) % (2*pi)

    b2_cmpx = simplify(cancel(solve((term2_lft - term2_rgt)
                                    .subs(b6, b6_cmpx)
                                    .subs(b4, b4_cmpx), b2)[0]))
    b2_real = simplify(re(expand(b2_cmpx)))

    b1_cmpx = simplify(cancel(solve((term1_lft - term1_rgt)
                                    .subs(b5, b5_cmpx)
                                    .subs(b3, b3_cmpx), b1)[0]))
    b1_real = simplify(re(expand(b1_cmpx)))
    b1_amp = simplify(cancel(sqrt(b1_real**2 + b1_real.diff(phi)**2).subs(phi, 0)))
    b1_shift = -arg(b1_cmpx.subs(phi, 0)) % (2*pi)

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
print("Solving the Ne PAD equations and lambdify their b parameters...")
solved = {k: solve_eq(p) for k, p in pads.items()}
args = (c_sp, c_psp, c_pdp, c_dp, c_fdp, eta_s, eta_p, eta_d, eta_f)
targets = ('b0', 'b1_amp', 'b1_shift',
           'b2', 'b3_amp', 'b3_shift',
           'b4', 'b5_amp', 'b5_shift', 'b6')
lambdified = {
    k: {t: lambdify(args, expr[t], 'numpy') for t in targets}
    for k, expr in solved.items()
}


def neon_pad(c_sp: float, c_psp: float, c_pdp: float, c_dp: float, c_fdp: float,
             eta_s: float, eta_p: float, eta_d: float, eta_f: float) -> dict:
    """
    Return b parameters of Neon PAD. It assumes...

        odd order b = amp * cos(phi - shift) ,
        even order b including 0th = const .

    Here amp is non-negative, shift is a value between 0 to 2*pi, and phi is the difference of w and 2w optical phases.
    Details are written in Daehyun's report.
    """
    return {k: {t: f(c_sp, c_psp, c_pdp, c_dp, c_fdp, eta_s, eta_p, eta_d, eta_f)
                for t, f in targets.items()}
            for k, targets in lambdified.items()}
