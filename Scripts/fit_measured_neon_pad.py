#!/usr/bin/env python3

from pprint import pprint

from numpy import pi
from scipy.optimize import least_squares, OptimizeResult
from yaml import safe_load
from jinja2 import Template

from padtools import TargetNeonPad


# %%
with open('Data/beta_neon_gauss3.yaml', 'r') as f:
    measured = safe_load(Template(f.read()).render())

# %%
for k, m in measured.items():
    # if k not in {"good4'"}:
    #     continue
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
        **m.get('weights', {}),
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
