#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:18:38 2018

@author: daehyun
"""

from pickle import loads

from pymongo import MongoClient
from cytoolz import pipe, filter, partial
from pandas import DataFrame
import matplotlib.pyplot as plt

# %%
db = MongoClient('lithium.local')['FERMI_20144077']

bad_runs = (
    178,  # good1
    464, 468,  # good2
    186, 196, *range(198, 204),  # alt1
    *range(266, 272), *range(292, 315),  # alt2
    359)  # alt3
ignore_bad_runs = partial(filter, lambda n: n not in bad_runs)
runs = pipe(
    range(463, 487),  # good2
    ignore_bad_runs, sorted, tuple)
imgs = pipe(db['reconstructed'].find({'run': {'$in': runs}}), list, DataFrame)
selected = DataFrame({'run': imgs['run'], 'n': imgs['n'], 'img': imgs['hist'].map(loads)}).merge(
    DataFrame([
        [463,0.10],
        [465,0.25],
        [466,0.40],
        [467,0.55],
        [469,0.85],
        [470,1.00],
        [471,1.15],
        [472,1.30],
        [473,1.45],
        [474,1.60],
        [475,0.10],
        [476,0.10],
        [477,0.25],
        [478,0.40],
        [479,0.55],
        [480,0.70],
        [481,0.85],
        [482,1.00],
        [483,1.15],
        [484,1.30],
        [485,1.45],
        [486,1.60]], columns=['run', 'phase']), on='run').sort_values('phase')
del imgs

# %%
summed = DataFrame({'phase': selected['phase'],
                    'summed': selected['img']*selected['n'],
                    'n': selected['n']}).groupby('phase')[['n', 'summed']].apply(sum)
groupped = DataFrame({'n': summed['n'],
                      'img': summed['summed'] / summed['n']}).reset_index()
del summed

# %%
for _, phase, img in groupped[['phase', 'img']].itertuples():
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(img.T, cmap='Greys')
    plt.axis('equal')
    plt.axis('off')
    plt.clim(0, 0.05)
    plt.savefig('inv{:03.0f}.png'.format(phase*100))
