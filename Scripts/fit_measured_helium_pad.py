#!/usr/bin/env python3

from numpy import pi, inf
from scipy.optimize import least_squares, OptimizeResult

from padtools import TargetHeliumPad

# %%
measured = {
    'good1': {  # wonly5
        'w_photon': 15.9,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_good1.ipynb
        'w2w_beta1_amp': 0.53422064,
        'w2w_beta1_amp_err': 0.01697032,
        'w2w_beta1_shift': 0.23601798,
        'w2w_beta1_shift_err': 0.03475667,
        'w2w_beta2': 1.63093134,
        'w2w_beta2_err': 0.00931104,
        'w2w_beta3_amp': 0.80997766,
        'w2w_beta3_amp_err': 0.02463020,
        'w2w_beta3_shift': 0.02397292,
        'w2w_beta3_shift_err': 0.03180625,
        'w2w_beta4': 1.02825850,
        'w2w_beta4_err': 0.02773891,
        # 'w2w_beta1m3_shift': 1.75117358,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly5.ipynb
        'wonly_beta2': 1.149896,
        'wonly_beta2_err': 0.042160,
        'wonly_beta4': 1.562401,
        'wonly_beta4_err': 0.049205,
        'x0': {
            'c_sps': {'init': 1, 'lower': 0, 'upper': inf},
            'c_ps': {'init': 1, 'lower': 0, 'upper': inf},
            'c_dps': {'init': 1, 'lower': 0, 'upper': inf},
            'eta_s': {'init': 2, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta3_shift + w2w_beta1m3_shift = 1.727
            'eta_p': {'init': 7, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta3_shift = 6.259
            # eta_d is fixed at 0
        },
    },
    'good2': {  # wonly3
        'w_photon': 14.3,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_good2.ipynb
        'w2w_beta1_amp': 0.34781825,
        'w2w_beta1_amp_err': 0.00509676,
        'w2w_beta1_shift': 0.07580421,
        'w2w_beta1_shift_err': 0.01551300,
        'w2w_beta2': 0.85577271,
        'w2w_beta2_err': 0.01006146,
        'w2w_beta3_amp': 0.57382046,
        'w2w_beta3_amp_err': 0.01349924,
        'w2w_beta3_shift': -0.30203627,
        'w2w_beta3_shift_err': 0.02268753,
        'w2w_beta4': 0.93521111,
        'w2w_beta4_err': 0.01291322,
        # 'w2w_beta1m3_shift': 1.70142651,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly3.ipynb
        'wonly_beta2': 0.550593,
        'wonly_beta2_err': 0.018332,
        'wonly_beta4': 1.738617,
        'wonly_beta4_err': 0.049251,
        'x0': {
            'c_sps': {'init': 4, 'lower': 0, 'upper': inf},
            'c_ps': {'init': 1, 'lower': 0, 'upper': inf},
            'c_dps': {'init': 4, 'lower': 0, 'upper': inf},
            'eta_s': {'init': 2, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta3_shift + w2w_beta1m3_shift = 2.003
            'eta_p': {'init': 1, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta3_shift = 0.302
            # eta_d is fixed at 0
        },
        'opts': {
            # 'loss': 'soft_l1',
            # 'f_scale': 0.1,
        },
    },
    'good3': {  # wonly4
        'w_photon': 19.1,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_good3.ipynb
        'w2w_beta1_amp': 0.49698230,
        'w2w_beta1_amp_err': 0.00487083,
        'w2w_beta1_shift': 5.36811057,
        'w2w_beta1_shift_err': 0.00935702,
        'w2w_beta2': 1.78359136,
        'w2w_beta2_err': 0.00519950,
        'w2w_beta3_amp': 0.72313844,
        'w2w_beta3_amp_err': 0.00801409,
        'w2w_beta3_shift': 5.42099236,
        'w2w_beta3_shift_err': 0.01060334,
        'w2w_beta4': 1.01037479,
        'w2w_beta4_err': 0.01578971,
        # 'w2w_beta1m3_shift': 4.34562320,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly4.ipynb
        'wonly_beta2': 1.555580,
        'wonly_beta2_err': 0.090389,
        'wonly_beta4': 2.238745,
        'wonly_beta4_err': 0.057671,
        'x0': {
            'c_sps': {'init': 1, 'lower': 0, 'upper': inf},
            'c_ps': {'init': 4, 'lower': 0, 'upper': inf},
            'c_dps': {'init': 8, 'lower': 0, 'upper': inf},
            'eta_s': {'init': 5, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta3_shift + w2w_beta1m3_shift = 5.208
            'eta_p': {'init': 1, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta3_shift = 0.862
            # eta_d is fixed at 0
        },
    },
    'good4': {  # wonly5
        'w_photon': 15.9,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_good4.ipynb
        'w2w_beta1_amp': 0.28143002,
        'w2w_beta1_amp_err': 0.01223439,
        'w2w_beta1_shift': 4.61278873,
        'w2w_beta1_shift_err': 0.04258330,
        'w2w_beta2': 1.69117620,
        'w2w_beta2_err': 0.00482420,
        'w2w_beta3_amp': 0.43882003,
        'w2w_beta3_amp_err': 0.01747798,
        'w2w_beta3_shift': 4.41814770,
        'w2w_beta3_shift_err': 0.03975083,
        'w2w_beta4': 0.41195893,
        'w2w_beta4_err': 0.00868607,
        # 'w2w_beta1m3_shift': 6.28210366,
        # ref: https://github.com/DaehyunPY/FERMI_20144077/blob/master/Notebooks/beta_wonly5.ipynb
        'wonly_beta2': 1.149896,
        'wonly_beta2_err': 0.042160,
        'wonly_beta4': 1.562401,
        'wonly_beta4_err': 0.049205,
        'x0': {
            'c_sps': {'init': 1, 'lower': 0, 'upper': inf},
            'c_ps': {'init': 1, 'lower': 0, 'upper': inf},
            'c_dps': {'init': 1, 'lower': 0, 'upper': inf},
            'eta_s': {'init': 2, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta3_shift + w2w_beta1m3_shift = 1.864
            'eta_p': {'init': 2, 'lower': -pi, 'upper': 3 * pi},  # 0 - w2w_beta3_shift = 1.865
            # eta_d is fixed at 0
        },
    },
}


for k, m in measured.items():
    if k != 'good3':
        continue
    print('Dataset {}...'.format(k))
    pad = TargetHeliumPad(
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
        amp_weight=1,
        shift_weight=16,
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
