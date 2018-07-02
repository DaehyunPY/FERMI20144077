#!/usr/bin/env python3

from numpy import pi, inf
from scipy.optimize import least_squares, OptimizeResult

from padtools import TargetNeonPad

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
        # 'w2w_beta1m3_shift': 2.53099006,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly5.ipynb
        'wonly_beta2': -0.080805,
        'wonly_beta2_err': 0.018737,
        'wonly_beta4': 1.107058,
        'wonly_beta4_err': 0.019651,
        'x0': {
            'c_sp': {'init': 4, 'lower': 0, 'upper': inf},
            'c_psp': {'init': 0, 'lower': -inf, 'upper': inf},
            'c_pdp': {'init': 1, 'lower': 0, 'upper': inf},
            'c_dp': {'init': 4, 'lower': 0, 'upper': inf},
            'c_fdp': {'init': 2, 'lower': 0, 'upper': inf},
            'eta_s': {'init': 1, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta1m3_shift + pi = 0.611
            'eta_p': {'init': 0, 'lower': -pi, 'upper': 3 * pi},
            'eta_d': {'init': 6, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta1_shift = 5.804
            # eta_f is fixed at 0
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
        # 'w2w_beta1m3_shift': 1.28808336,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly3.ipynb
        'wonly_beta2': -0.469027,
        'wonly_beta2_err': 0.001037,
        'wonly_beta4': 0.052413,
        'wonly_beta4_err': 0.006189,
        'x0': {
            'c_sp': {'init': 4, 'lower': 0, 'upper': inf},
            'c_psp': {'init': 0, 'lower': -inf, 'upper': inf},
            'c_pdp': {'init': 1, 'lower': 0, 'upper': inf},
            'c_dp': {'init': 4, 'lower': 0, 'upper': inf},
            'c_fdp': {'init': 2, 'lower': 0, 'upper': inf},
            'eta_s': {'init': 2, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta1m3_shift + pi = 1.854
            'eta_p': {'init': 0, 'lower': -pi, 'upper': 3 * pi},
            'eta_d': {'init': 6, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta1_shift = 5.798
            # eta_f is fixed at 0
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
        # 'w2w_beta1m3_shift': 6.96890504,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly4.ipynb
        'wonly_beta2': 0.918967,
        'wonly_beta2_err': 0.045427,
        'wonly_beta4': 0.405998,
        'wonly_beta4_err': 0.017098,
        'x0': {
            'c_sp': {'init': 4, 'lower': 0, 'upper': inf},
            'c_psp': {'init': -1, 'lower': -inf, 'upper': inf},
            'c_pdp': {'init': 1, 'lower': -inf, 'upper': inf},
            'c_dp': {'init': 4, 'lower': 0, 'upper': inf},
            'c_fdp': {'init': 2, 'lower': 0, 'upper': inf},
            'eta_s': {'init': 2, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta1m3_shift + pi = 2.456
            'eta_p': {'init': 4, 'lower': -pi, 'upper': 3 * pi},
            'eta_d': {'init': 0, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta1_shift = 0.622
            # eta_f is fixed at 0
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
        # 'w2w_beta1m3_shift': 6.94463814,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly5.ipynb
        'wonly_beta2': -0.080805,
        'wonly_beta2_err': 0.018737,
        'wonly_beta4': 1.107058,
        'wonly_beta4_err': 0.019651,
        'x0': {
            'c_sp': {'init': 4, 'lower': 0, 'upper': inf},
            'c_psp': {'init': 0, 'lower': -inf, 'upper': inf},
            'c_pdp': {'init': 1, 'lower': 0, 'upper': inf},
            'c_dp': {'init': 4, 'lower': 0, 'upper': inf},
            'c_fdp': {'init': 2, 'lower': 0, 'upper': inf},
            'eta_s': {'init': 2, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta1m3_shift + pi = 2.480
            'eta_p': {'init': 0, 'lower': -pi, 'upper': 3 * pi},
            'eta_d': {'init': 1, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta1_shift = 1.467
            # eta_f is fixed at 0
        },
    },
}

# %%
for k, m in measured.items():
    print('Dataset {}...'.format(k))
    pad = TargetNeonPad(
        w2w_beta1_amp=m['w2w_beta1_amp'],
        w2w_beta1_amp_err=m['w2w_beta1_amp_err'],
        w2w_beta1_shift=m['w2w_beta1_shift'],
        w2w_beta1_shift_err=m['w2w_beta1_shift_err'],
        w2w_beta2=m['w2w_beta2'],
        w2w_beta2_err=m['w2w_beta2_err'],
        w2w_beta3_amp=m['w2w_beta3_amp'],
        w2w_beta3_amp_err=m['w2w_beta3_amp_err'],
        w2w_beta3_shift=m['w2w_beta3_shift'],
        w2w_beta3_shift_err=m['w2w_beta3_shift_err'],
        w2w_beta4=m['w2w_beta4'],
        w2w_beta4_err=m['w2w_beta4_err'],
        wonly_beta2=m['wonly_beta2'],
        wonly_beta2_err=m['wonly_beta2_err'],
        wonly_beta4=m['wonly_beta4'],
        wonly_beta4_err=m['wonly_beta4_err'],
        amp_weight=4,
        shift_weight=64,
        even_weight=1,
    )

    x0 = [m['x0'][k.name.lower()] for k in pad.XKEYS if k not in pad.xfixed]
    opt: OptimizeResult = least_squares(
        pad.zdiffmat,
        [d['init'] for d in x0],
        jac=pad.zdiffjacmat,
        bounds=[[d['lower'] for d in x0], [d['upper'] for d in x0]],
        **m.get('opts', {}),
    )

    print('Fitting report...')
    pad.report(opt.x)
    if not opt.success:
        raise AssertionError('Fail to optimize the pad!')
    print()
