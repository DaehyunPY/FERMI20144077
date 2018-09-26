from functools import wraps

from sympy import *

__all__ = ['lambdified']


# %%
coeff_p = Symbol('coeff_p')
coeff_d = Symbol('coeff_d')
coeff_s = Symbol('coeff_s')
eta_d = Symbol('eta_d')
eta_s = Symbol('eta_s')
eta_p = Symbol('eta_p')
h = Symbol('h')

solved = {
    'beta1_amp':
        (2 * 2 * h / (1 + h) * coeff_p * sqrt(60 * coeff_d ** 2
                                              + 60 * sqrt(5) * coeff_d * coeff_s * cos(eta_d - eta_s)
                                              + 75 * coeff_s ** 2)
         / (5 * (coeff_d ** 2 + h * coeff_p ** 2 + coeff_s ** 2))),
    'beta1_shift':
        arg(2 * sqrt(15) * coeff_d * cos(eta_d - eta_p) + 5 * sqrt(3) * coeff_s * cos(eta_p - eta_s)
            + I * (2 * sqrt(15) * coeff_d * sin(eta_d - eta_p) - 5 * sqrt(3) * coeff_s * sin(eta_p - eta_s))),
    'beta1m3_amp':
        (2 * sqrt(3) * 2 * h / (1 + h) * coeff_p * coeff_s
         / (coeff_d ** 2 + h * coeff_p ** 2 + coeff_s ** 2)),
    'beta1m3_shift': arg(-I * sin(eta_p - eta_s) + cos(eta_p - eta_s)),
    'beta2':
        ((10 * coeff_d ** 2 + 14 * sqrt(5) * coeff_d * coeff_s * cos(eta_d - eta_s) + 14 * h * coeff_p ** 2)
         / (7 * (coeff_d ** 2 + h * coeff_p ** 2 + coeff_s ** 2))),
    'beta3_amp':
        (6 * sqrt(15) * coeff_d * 2 * h / (1 + h) * coeff_p
         / (5 * (coeff_d ** 2 + h * coeff_p ** 2 + coeff_s ** 2))),
    'beta3_shift': arg(I * sin(eta_d - eta_p) + cos(eta_d - eta_p)),
    'beta4': 18 * coeff_d ** 2 / (7 * (coeff_d ** 2 + h * coeff_p ** 2 + coeff_s ** 2)),
}

# %% lambdify pads
kwargs = [coeff_s, coeff_p, coeff_d, eta_s, eta_p, eta_d, h]
kwrets = ['beta1_amp', 'beta1_shift', 'beta2', 'beta3_amp', 'beta3_shift', 'beta4', 'beta1m3_amp', 'beta1m3_shift']
__lambdified = lambdify(kwargs, [solved[k] for k in kwrets], 'numpy')


@wraps(__lambdified)
def lambdified(coeff_s, coeff_p, coeff_d, eta_s, eta_p, eta_d, h):
    return dict(zip(kwrets,
                    __lambdified(coeff_s, coeff_p, coeff_d, eta_s, eta_p, eta_d, h)))
