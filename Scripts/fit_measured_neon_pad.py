#!/usr/bin/env python3

from numpy import pi, inf
from scipy.optimize import least_squares, OptimizeResult

from padtools import target_neon_pad

# %%
measured = {
    '14.3 eV': {  # good2 and wonly3
        'phi0': 5.10,
        'phi0_err': 0.023,
        'w2w_beta1_amp': 0.26743193,
        'w2w_beta1_amp_err': 0.00363317,
        'w2w_beta1_shift+phi0': 0.48539010,
        'w2w_beta1_shift_err': 0.01535184,
        'w2w_beta2': -0.08793100,
        'w2w_beta2_err': 0.00520163,
        'w2w_beta3_amp': 0.19706859,
        'w2w_beta3_amp_err': 0.00681090,
        'w2w_beta3_shift+phi0': -1.14502012,
        'w2w_beta3_shift_err': 0.03074370,
        'w2w_beta4': -0.03067388,
        'w2w_beta4_err': 0.00321445,
        'wonly_beta2': -0.473761,
        'wonly_beta2_err': 0.000046,
        'wonly_beta4': 0.126242,
        'wonly_beta4_err': 0.005736,
        'x0': {  # (init, lower limit, upper limit)
            'c_sp': (1, 0, inf),
            'c_psp': (1, 0, inf),
            'c_pdp': (1, -inf, inf),
            'c_dp': (1, 0, inf),
            'c_fdp': (1, 0, inf),
            'eta_s': (0, -pi, 2 * pi),
            'eta_p': (3, -pi, 2 * pi),
            'eta_d': (4, -pi, 2 * pi),
            # eta_f is fixed at 0
        },
    },
}

# %%
for k, m in measured.items():
    f = target_neon_pad(w2w_beta1_amp=m['w2w_beta1_amp'],
                        w2w_beta1_amp_weight=1 / m['w2w_beta1_amp_err'] ** 2,
                        w2w_beta1_shift=(m['w2w_beta1_shift+phi0'] - m['phi0']) % (2 * pi),
                        w2w_beta1_shift_weight=1 / (m['w2w_beta1_shift_err'] ** 2 + m['phi0_err'] ** 2),
                        w2w_beta2=m['w2w_beta2'],
                        w2w_beta2_weight=1 / m['w2w_beta2_err'] ** 2,
                        w2w_beta3_amp=m['w2w_beta3_amp'],
                        w2w_beta3_amp_weight=1 / m['w2w_beta3_amp_err'] ** 2,
                        w2w_beta3_shift=(m['w2w_beta3_shift+phi0'] - m['phi0']) % (2 * pi),
                        w2w_beta3_shift_weight=1 / (m['w2w_beta3_shift_err'] ** 2 + m['phi0_err'] ** 2),
                        w2w_beta4=m['w2w_beta4'],
                        w2w_beta4_weight=1 / m['w2w_beta4_err'] ** 2,
                        wonly_beta2=m['wonly_beta2'],
                        wonly_beta2_weight=1 / m['wonly_beta2_err'] ** 2,
                        wonly_beta4=m['wonly_beta4'],
                        wonly_beta4_weight=1 / m['wonly_beta4_err'] ** 2,
                        amp_weight=1,
                        shift_weight=100,
                        even_weight=1,
                        )
    x0 = f['unlabelit'](m['x0'])
    zipped = tuple(zip(*x0[:-1]))
    opt: OptimizeResult = least_squares(
        f['diff'],
        zipped[0],
        bounds=zipped[1:],
        loss='soft_l1',
    )

    print('Fit report...')
    f['report'](opt.x)
    print()
    print('Best fit and jac...')
    print('                '
          '        c_sp'
          '       c_psp'
          '       c_pdp'
          '        c_dp'
          '       c_fdp'
          '       eta_s'
          '       eta_p'
          '       eta_d'
          '       eta_f')
    for k, j in zip(('fitted:', 'w2w_beta1_amp:', 'w2w_beta1_shift:', 'w2w_beta2:',
                     'w2w_beta3_amp:', 'w2w_beta3_shift:', 'w2w_beta4:', 'wonly_beta2:', 'wonly_beta4:'),
                    (opt.x, *opt.jac)):
        print('{:16s}'
              '{: 12.3f}'
              '{: 12.3f}'
              '{: 12.3f}'
              '{: 12.3f}'
              '{: 12.3f}'
              '{: 12.3f}'
              '{: 12.3f}'
              '{: 12.3f}'
              '{: 12.3f}'.format(k, *j, 0))
    if not opt.success:
        raise AssertionError('Fail to optimize the pad!')
