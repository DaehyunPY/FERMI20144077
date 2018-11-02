from typing import Tuple

from sympy import Expr, Symbol, I, pi, cos, arg, sqrt, cancel, simplify


def expend_cos(expr: Expr, x: Symbol) -> Tuple[Expr, Expr]:
    while True:
        term = expr.subs(x, pi / 2)
        yield term
        expr = cancel((expr - term) / cos(x))
        if expr == 0:
            return


def amp_and_shift(expr: Expr, x: Symbol) -> Tuple[Expr, Expr]:
    amp = simplify(cancel(sqrt(expr ** 2 + expr.diff(x) ** 2).subs(x, 0)))
    shift = arg(expr.subs(x, 0) + I * expr.diff(x).subs(x, 0))
    return amp, shift
