#!/usr/bin/env python3

from io import StringIO
from textwrap import dedent

import matplotlib.pyplot as plt
from pandas import read_csv

# %%
data = StringIO(dedent("""\
    category,photon,eta_fd
    theory,14.5,4.572
    theory,15.0,4.448
    theory,15.5,4.367
    theory,16.0,4.307
    theory,16.5,4.258
    theory,17.0,4.216
    theory,17.5,4.179
    theory,18.0,4.146
    theory,18.5,4.117
    theory,19.0,4.091
    theory,19.5,4.068
    theory,20.0,4.066
    theory,20.5,4.024
    theory,21.0,4.007
    theory,21.5,3.987
    theory,22.0,3.960
    theory,22.5,3.924
    theory,23.0,3.884
    theory,23.5,3.850
    theory,24.0,3.822
    neg_p,15.9,4.626
    neg_p,14.3,4.854
    neg_p,19.1,4.538
    g_fitting,15.9,4.643
    g_fitting,14.3,4.762
    g_fitting,19.1,4.466"""
                       ))
df = read_csv(data)

# %%
df_theory = df[df['category'] == 'theory']
df_exp = df[df['category'] != 'theory']

plt.figure(figsize=(4, 6))
plt.subplot(211)
plt.plot(df_theory['photon'], df_theory['eta_fd'], '.-', label='theory')
for k in df_exp['category'].unique():
    df_roi = df_exp[df_exp['category'] == k]
    plt.plot(df_roi['photon'], df_roi['eta_fd'], 'o', label=k)
# plt.xlabel('photon (eV)')
plt.ylabel('eta_fd (rad)')
plt.xlim(14, 19.5)
plt.ylim(4, 5)
plt.grid(True)


def center(arr):
    return (arr[1:] + arr[:-1]) / 2


def diff(arr):
    return arr[1:] - arr[:-1]


plt.subplot(212)
plt.plot(center(df_theory['photon'].values),
         diff(df_theory['eta_fd'].values) / diff(df_theory['photon'].values)
         * 24.1888432651 / 0.0367493 / 2,
         '.-', label='theory')
for k in df_exp['category'].unique():
    df_roi = df_exp[df_exp['category'] == k]
    plt.plot(center(df_roi['photon'].values),
             diff(df_roi['eta_fd'].values) / diff(df_roi['photon'].values)
             * 24.1888432651 / 0.0367493 / 2,
             'o', label=k)
plt.xlabel('photon (eV)')
plt.ylabel('tau_21 (as)')
plt.xlim(14, 19.5)
plt.ylim(-80, -10)
plt.grid(True)

plt.tight_layout()
plt.legend()
plt.show()
