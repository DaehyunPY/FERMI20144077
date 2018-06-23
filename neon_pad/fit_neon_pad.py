from numpy import array, pi, ndarray

from .solved_neon_eq import neon_pad

__all__ = (
    'neon_keys',
    'neon_wonly_argmask',
    'target_neon_pad',
)


# %%
neon_keys = ('c_sp', 'c_psp', 'c_pdp', 'c_dp', 'c_fdp', 'eta_s', 'eta_p', 'eta_d', 'eta_f')
neon_wonly_argmask = array((0, 1, 1, 0, 1, 0, 0, 0, 0), dtype='b')
neon_wonly_argmask.flags.writeable = False


def target_neon_pad(w2w_beta1_amp: float, w2w_beta1_shift: float, w2w_beta2: float,
                    w2w_beta3_amp: float, w2w_beta3_shift: float, w2w_beta4: float,
                    wonly_beta2: float, wonly_beta4: float,
                    w2w_beta1_amp_weight: float = 1, w2w_beta1_shift_weight: float = 1, w2w_beta2_weight: float = 1,
                    w2w_beta3_amp_weight: float = 1, w2w_beta3_shift_weight: float = 1, w2w_beta4_weight: float = 1,
                    wonly_beta2_weight: float = 1, wonly_beta4_weight: float = 1,
                    amp_weight: float = 1, shift_weight: float = 1, even_weight: float = 1):
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

    def diff(args: ndarray):
        mkey = 'summed'
        w2w = neon_pad(*args, 0)[mkey]
        wonly = neon_pad(*(neon_wonly_argmask[:-1] * args), 0)[mkey]
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
    return diff
