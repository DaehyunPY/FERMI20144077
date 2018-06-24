#!/usr/bin/env python3

from numpy import pi
from scipy.optimize import least_squares, OptimizeResult

from padtools import target_neon_pad

# %%
measured = {
    'good1': {  # wonly5
        'w_photon': 15.9,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_good1.ipynb
        'w2w_beta1_amp': 0.28004315,
        'w2w_beta1_amp_err': 0.01265977,
        'w2w_beta1_shift': 0.47873947,
        'w2w_beta1_shift_err': 0.05105923,
        'w2w_beta2': 0.34675251,
        'w2w_beta2_err': 0.00938442,
        'w2w_beta3_amp': 0.39872208,
        'w2w_beta3_amp_err': 0.01498100,
        'w2w_beta3_shift': -0.18265237,
        'w2w_beta3_shift_err': 0.03732980,
        'w2w_beta4': 0.43274126,
        'w2w_beta4_err': 0.01531978,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly5.ipynb
        'wonly_beta2': -0.080805,
        'wonly_beta2_err': 0.018737,
        'wonly_beta4': 1.107058,
        'wonly_beta4_err': 0.019651,
        'x0': {  # (init, lower limit, upper limit)
            'c_sp': (1, 0, 10),
            'c_psp': (0, -10, 10),
            'c_pdp': (1, 0, 10),
            'c_dp': (3, 0, 10),
            'c_fdp': (1, 0, 10),
            'eta_s': (4, -pi, 3 * pi),
            'eta_p': (0, -pi, 3 * pi),
            'eta_d': (1, -pi, 3 * pi),
            # eta_f is fixed at 0
        },
        'opts': {
            'f_scale': 0.01,
        },
    },
    'good2': {  # wonly3
        'w_photon': 14.3,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_good2.ipynb
        'w2w_beta1_amp': 0.26743193,
        'w2w_beta1_amp_err': 0.00363317,
        'w2w_beta1_shift': 0.48539010,
        'w2w_beta1_shift_err': 0.01535184,
        'w2w_beta2': -0.08793100,
        'w2w_beta2_err': 0.00520163,
        'w2w_beta3_amp': 0.19706859,
        'w2w_beta3_amp_err': 0.00681090,
        'w2w_beta3_shift': -1.14502012,
        'w2w_beta3_shift_err': 0.03074370,
        'w2w_beta4': -0.03067388,
        'w2w_beta4_err': 0.00321445,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly3.ipynb
        'wonly_beta2': -0.469027,
        'wonly_beta2_err': 0.001037,
        'wonly_beta4': 0.052413,
        'wonly_beta4_err': 0.006189,
        'x0': {  # (init, lower limit, upper limit)
            'c_sp': (4, 0, 8),
            'c_psp': (0, -8, 8),
            'c_pdp': (1, 0, 8),
            'c_dp': (1, 0, 8),
            'c_fdp': (1, 0, 8),
            'eta_s': (4, -pi, 3 * pi),
            'eta_p': (0, -pi, 3 * pi),
            'eta_d': (5, -pi, 3 * pi),
            # eta_f is fixed at 0
        },
        'opts': {
            'f_scale': 0.1,
        },
    },
    'good3': {  # wonly4
        'w_photon': 19.1,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_good3.ipynb
        'w2w_beta1_amp': 0.37245283,
        'w2w_beta1_amp_err': 0.00360273,
        'w2w_beta1_shift': 5.66144723,
        'w2w_beta1_shift_err': 0.00939969,
        'w2w_beta2': 0.88440582,
        'w2w_beta2_err': 0.00169322,
        'w2w_beta3_amp': 0.26738427,
        'w2w_beta3_amp_err': 0.00339328,
        'w2w_beta3_shift': 4.93938802,
        'w2w_beta3_shift_err': 0.01214595,
        'w2w_beta4': 0.02458127,
        'w2w_beta4_err': 0.00363349,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly4.ipynb
        'wonly_beta2': 0.918967,
        'wonly_beta2_err': 0.045427,
        'wonly_beta4': 0.405998,
        'wonly_beta4_err': 0.017098,
        'x0': {  # (init, lower limit, upper limit)
            'c_sp': (1, 0, 8),
            'c_psp': (1, 0, 8),
            'c_pdp': (1, -8, 8),
            'c_dp': (1, 0, 8),
            'c_fdp': (1, 0, 8),
            'eta_s': (2, -pi, 3 * pi),
            'eta_p': (3, -pi, 3 * pi),
            'eta_d': (4, -pi, 3 * pi),
            # eta_f is fixed at 0
        },
        'opts': {
            'f_scale': 0.1,
        },
    },
    'good4': {  # wonly5
        'w_photon': 15.9,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_good4.ipynb
        'w2w_beta1_amp': 0.15287231,
        'w2w_beta1_amp_err': 0.00756190,
        'w2w_beta1_shift': 4.81652524,
        'w2w_beta1_shift_err': 0.04767470,
        'w2w_beta2': 0.48062860,
        'w2w_beta2_err': 0.00378655,
        'w2w_beta3_amp': 0.23353777,
        'w2w_beta3_amp_err': 0.00861068,
        'w2w_beta3_shift': 4.18244115,
        'w2w_beta3_shift_err': 0.03764488,
        'w2w_beta4': 0.09554586,
        'w2w_beta4_err': 0.00523730,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly5.ipynb
        'wonly_beta2': -0.080805,
        'wonly_beta2_err': 0.018737,
        'wonly_beta4': 1.107058,
        'wonly_beta4_err': 0.019651,
        'x0': {  # (init, lower limit, upper limit)
            'c_sp': (2, 0, 8),
            'c_psp': (1, -8, 8),
            'c_pdp': (1, 0, 8),
            'c_dp': (2, 0, 8),
            'c_fdp': (2, 0, 8),
            'eta_s': (0, -pi, 3 * pi),
            'eta_p': (3, -pi, 3 * pi),
            'eta_d': (4, -pi, 3 * pi),
            # eta_f is fixed at 0
        },
        'opts': {
            'f_scale': 0.1,
        },
    },
}


for k, m in measured.items():
    print('Dataset {}...'.format(k))
    f = target_neon_pad(w2w_beta1_amp=m['w2w_beta1_amp'],
                        w2w_beta1_amp_weight=1 / m['w2w_beta1_amp_err'] ** 2,
                        w2w_beta1_shift=m['w2w_beta1_shift'],
                        w2w_beta1_shift_weight=1 / m['w2w_beta1_shift_err'] ** 2,
                        w2w_beta2=m['w2w_beta2'],
                        w2w_beta2_weight=1 / m['w2w_beta2_err'] ** 2,
                        w2w_beta3_amp=m['w2w_beta3_amp'],
                        w2w_beta3_amp_weight=1 / m['w2w_beta3_amp_err'] ** 2,
                        w2w_beta3_shift=m['w2w_beta3_shift'],
                        w2w_beta3_shift_weight=1 / m['w2w_beta3_shift_err'] ** 2,
                        w2w_beta4=m['w2w_beta4'],
                        w2w_beta4_weight=1 / m['w2w_beta4_err'] ** 2,
                        wonly_beta2=m['wonly_beta2'],
                        wonly_beta2_weight=1 / m['wonly_beta2_err'] ** 2,
                        wonly_beta4=m['wonly_beta4'],
                        wonly_beta4_weight=1 / m['wonly_beta4_err'] ** 2,
                        amp_weight=1,
                        shift_weight=32,
                        even_weight=1 / 2,
                        )
    x0 = f['unlabelit'](m['x0'])
    zipped = tuple(zip(*x0[:-1]))
    opt: OptimizeResult = least_squares(
        f['diff'],
        zipped[0],
        bounds=zipped[1:],
        loss='soft_l1',
        **m.get('opts', {}),
    )

    print(opt.message)
    print('Fitting report...')
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
    print()
