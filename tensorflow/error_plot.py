#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimating errors (dx) when unscaling data (x) from linear scale (x*10^n) 
and from a log-scale (10^x)

Created on Wed Nov 30 12:44:20 2022

@author: jarl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

dx = np.logspace(-6, 2, 1000)

# initialize a large (10^n) reference value
n = 15
ref = 2*10**(n+1)

# linear scale error
df = (10**(n-1))*dx

# log scale error
dg = (10**(n))*np.log(10)*dx

# get intersection
idx = np.argwhere(np.diff(np.sign(df - dg))).flatten()


############## plotting ##############
fig, ax = plt.subplots(dpi=200)

# linear scale error
ax.plot(dx, (df/ref)*100, c='red', label=f'lin: $f(x) = 10^{ {n} } \cdot x$')
# # log scale error
ax.plot(dx, (dg/ref)*100, c='green', label='log: $f(x) = 10^{x}$')


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'percent error $\frac{df}{f}$')
ax.set_xlabel('error ($dx$)')
ax.grid(True, 'both')
ax.legend()

# fig.savefig('/Users/jarl/2d-discharge-nn/data/error_diff.png', bbox_inches='tight')
