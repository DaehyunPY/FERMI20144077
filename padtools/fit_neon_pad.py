from itertools import chain, repeat, count
from textwrap import dedent
from typing import Iterable

from numba import jit
from numpy import array, pi, ndarray

from .solved_neon_eq import neon_pad

__all__ = (
    'target_neon_pad',
)


# %%
neon_keys = ('c_sp', 'c_psp', 'c_pdp', 'c_dp', 'c_fdp', 'eta_s', 'eta_p', 'eta_d', 'eta_f')
neon_wonly_argmask = array((False, True, True, False, True, False, False, False, False), dtype='b')
neon_wonly_argmask.flags.writeable = False


@jit
def labelit(args: Iterable) -> dict:
    return dict(zip(neon_keys, chain(args, repeat(0))))


def unlabelit(kwargs: dict) -> tuple:
    return tuple(kwargs.get(k, None) for k in neon_keys)


# todo: better mask
def target_neon_pad(w2w_beta1_amp: float, w2w_beta1_shift: float, w2w_beta2: float,
                    w2w_beta3_amp: float, w2w_beta3_shift: float, w2w_beta4: float,
                    wonly_beta2: float, wonly_beta4: float,
                    w2w_beta1_amp_weight: float = 1, w2w_beta1_shift_weight: float = 1, w2w_beta2_weight: float = 1,
                    w2w_beta3_amp_weight: float = 1, w2w_beta3_shift_weight: float = 1, w2w_beta4_weight: float = 1,
                    wonly_beta2_weight: float = 1, wonly_beta4_weight: float = 1,
                    amp_weight: float = 1, shift_weight: float = 1, even_weight: float = 1,
                    fixed: dict = None,
                    ) -> dict:
    if fixed is None:
        fixed = {'eta_f': 0}

    def mask(args: ndarray) -> tuple:
        inx = count()
        return tuple(fixed[k] if k in fixed else args[next(inx)] for k in neon_keys)

    w = array((
        (w2w_beta1_amp_weight * amp_weight +
         w2w_beta2_weight * even_weight +
         w2w_beta3_amp_weight * amp_weight +
         w2w_beta4_weight * even_weight),  # w2w_b0_weight
        w2w_beta1_amp_weight * amp_weight,
        w2w_beta1_shift_weight * shift_weight,
        w2w_beta2_weight * even_weight,
        w2w_beta3_amp_weight * amp_weight,
        w2w_beta3_shift_weight * shift_weight,
        w2w_beta4_weight * even_weight,
        wonly_beta2_weight * even_weight,
        wonly_beta4_weight * even_weight,
    )) ** 0.5

    def diff(args: ndarray) -> ndarray:
        mkey = 'summed'
        masked = mask(args)
        w2w = neon_pad(*masked)[mkey]
        wonly = neon_pad(*(arg if b else 0 for arg, b in zip(masked, neon_wonly_argmask)))[mkey]
        fx = array((
            1 - w2w['b0'],
            w2w_beta1_amp - w2w['b1_amp'] / w2w['b0'],
            (w2w_beta1_shift - w2w['b1_shift'] + pi) % (2 * pi) - pi,
            w2w_beta2 - w2w['b2'] / w2w['b0'],
            w2w_beta3_amp - w2w['b3_amp'] / w2w['b0'],
            (w2w_beta3_shift - w2w['b3_shift'] + pi) % (2 * pi) - pi,
            w2w_beta4 - w2w['b4'] / w2w['b0'],
            wonly_beta2 - wonly['b2'] / wonly['b0'],
            wonly_beta4 - wonly['b4'] / wonly['b0'],
        ))
        return w * fx

    def report(args: ndarray):
        mkey = 'summed'
        masked = mask(args)
        w2w = neon_pad(*masked)[mkey]
        wonly = neon_pad(*(arg if b else 0 for arg, b in zip(masked, neon_wonly_argmask)))[mkey]
        print(dedent("""\
                          target  examined  diff  weight
        w2w_beta1_amp:   {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta1_shift: {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta2:       {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta3_amp:   {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta3_shift: {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta4:       {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        wonly_beta2:     {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        wonly_beta4:     {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}""".format(
            # w2w_beta1_amp
            w2w_beta1_amp,
            w2w['b1_amp'] / w2w['b0'],
            w2w_beta1_amp - w2w['b1_amp'] / w2w['b0'],
            w2w_beta1_amp_weight * amp_weight,
            # w2w_beta1_shift
            w2w_beta1_shift,
            w2w['b1_shift'],
            (w2w_beta1_shift - w2w['b1_shift'] + pi) % (2 * pi) - pi,
            w2w_beta1_shift_weight * shift_weight,
            # w2w_beta2
            w2w_beta2,
            w2w['b2'] / w2w['b0'],
            w2w_beta2 - w2w['b2'] / w2w['b0'],
            w2w_beta2_weight * even_weight,
            # w2w_beta3_amp
            w2w_beta3_amp,
            w2w['b3_amp'] / w2w['b0'],
            w2w_beta3_amp - w2w['b3_amp'] / w2w['b0'],
            w2w_beta3_amp_weight * amp_weight,
            # w2w_beta3_shift
            w2w_beta3_shift,
            w2w['b3_shift'],
            (w2w_beta3_shift - w2w['b3_shift'] + pi) % (2 * pi) - pi,
            w2w_beta3_shift_weight * shift_weight,
            # w2w_beta4
            w2w_beta4,
            w2w['b4'] / w2w['b0'],
            w2w_beta4 - w2w['b4'] / w2w['b0'],
            w2w_beta4_weight * even_weight,
            # wonly_beta2
            wonly_beta2,
            wonly['b2'] / wonly['b0'],
            wonly_beta2 - wonly['b2'] / wonly['b0'],
            wonly_beta2_weight * even_weight,
            # wonly_beta4
            wonly_beta4,
            wonly['b4'] / wonly['b0'],
            wonly_beta4 - wonly['b4'] / wonly['b0'],
            wonly_beta4_weight * even_weight,
        )))

    return {'diff': diff, 'report': report, 'labelit': labelit, 'unlabelit': unlabelit}
