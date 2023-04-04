# -*- coding: utf-8 -*-
"""
Created on 2020.08.01

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""

#%%
import xarray as xr
import numpy as np
from xgcm import Grid
from xcontour.xcontour import Contour2D

# tcount = 1440
# path = 'I:/breakingIW/nonHydro/'

# ds = xmitgcm.open_mdsdataset(data_dir=path+'output/', grid_dir=path, delta_t=2,
#                              prefix=['Stat'])
# dset, grid = add_MITgcm_missing_metrics(ds, periodic=['X'],
#                                         boundary={'Y':'extend','Z':'extend'})

# dset['time'] = (dset.time.astype(np.float32)/1e9).astype(np.int32)
# dset = dset.drop_vars(['UVEL', 'WVEL', 'TRAC01'])
# dset.isel({'time':[30, 70, 110]}).squeeze().to_netcdf('d:/internalwave.nc')

ds = xr.open_dataset('E:/OneDrive/Python/MyPack/xcontour/Data/internalwave.nc')

grid = Grid(ds, metrics = {
        ('X',)    : ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
        ('Z',)    : ['drW', 'drS', 'drC', 'drF', 'drG'], # Z distances
        ('X', 'Z'): ['yA']}) # Areas in X-Z plane

# get potential temperature and maskout due to topography
T = ds.THETA.where(ds.maskC)

# calculate buoyancy using linear EOS
alpha = 2E-4
T0 = 20
g = 9.81
b = (alpha * (T - T0) * g).rename('buoyancy')

#%% initialize contours
# Initialize equally-spaced contours from minimum value to maximum value
# (within lat/lon dims).  Here will implicitly loop over each isentropic level

N  = 121           # increase the contour number may get non-monotonic A(q) relation
increase = False   # Y-index increases with depth
lt = False          # northward of PV contours (larger than) is inside the contour
                   # change this should not change the result of Keff, but may alter
                   # the values at boundaries
dtype = np.float32 # use float32 to save memory
undef = -9.99e8    # for maskout topography if present

# initialize a Contour2D analysis class using PV as the tracer
analysis = Contour2D(grid, b,
                     dims={'X':'XC','Z':'Z'},
                     dimEq={'Z':'Z'},
                     increase=increase,
                     lt=lt)
ctr = analysis.cal_contours(N)

# Mask for A(q) relation table.
# This can be done analytically in simple case, but we choose to do it
# numerically in case there are undefined values inside the domain.
mask = ds.maskC


#%% calculate related quantities for Keff
# xarray's conditional integration, memory consuming and not preferred, for test only
table = analysis.cal_area_eqCoord_table(mask) # A(Yeq) table
area  = analysis.cal_integral_within_contours(ctr).rename('intArea')
ZEq   = table.lookup_coordinates(area).rename('ZEq')


#%% calculate related quantities for Keff
# Alternative using _hist APIs, memory friendly and is preferred.
# Note that since xhistogram does not support time- or level-varying bins,
# this way does not support multi-dimensional calculation well as xarray's
# conditional integration
table = analysis.cal_area_eqCoord_table_hist(mask) # A(Yeq) table
area  = analysis.cal_integral_within_contours_hist(ctr).rename('intArea')
ZEq   = table.lookup_coordinates(area).rename('ZEq')


#%% combined the results
# results in contour space
ds_contour = xr.merge([ctr, area, ZEq])

# interpolate from contour space to equivalent-latitude space
preZs = b.Z.astype(dtype)
# results in latEq space
ds_ZEq = analysis.interp_to_dataset(preZs, ZEq, ds_contour)


#%% calculate local finite-amplitude wave activity
lape, ctrs, masks = analysis.cal_local_APE(b, ds_ZEq.buoyancy,
                                           mask_idx=[8,28,51,81])
lape2, ctrs2, masks2 = analysis.cal_local_wave_activity2(b, ds_ZEq.buoyancy,
                                           mask_idx=[8,28,51,81])


#%% LWA
import proplot as pplt

fig, axes = pplt.subplots(nrows=3, ncols=2, figsize=(10, 8.5), sharex=3, sharey=3)

fontsize = 12

def plot_time(tidx, add_colorbar=False):
    ax = axes[tidx, 0]
    m1=ax.contourf(-lape.where(mask)[tidx]*1e4, levels=np.linspace(0,50,26), cmap='reds') # minus sign to ensure positive definite
    ax.contour(b.where(mask)[tidx], levels=11, cmap='viridis', lw=0.8)
    if add_colorbar:
        ax.colorbar(m1, loc='b', ticks=5, label='')
    ax.set_title('buoyancy and local APE density (t={})'.format(tidx), fontsize=fontsize)
    ax.set_xlabel('x-coordinate (m)', fontsize=fontsize-1)
    ax.set_ylabel('z-coordinate (m)', fontsize=fontsize-1)
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.set_yticks([-200, -150, -100, -50, 0])
    ax.set_ylim([-200, 0])
    
    ax = axes[tidx, 1]
    msk = sum(masks)
    lev = xr.concat(ctrs, 'Z').isel(time=tidx).values
    m1=ax.contourf(msk.where(msk!=0)[tidx], cmap='bwr')
    ax.contour(b.where(mask)[tidx], levels=lev[::-1], lw=0.8, color='k')
    if add_colorbar:
        ax.colorbar(m1, loc='b', ticks=1, label='')
    ax.set_title('masks for local APE calculation', fontsize=fontsize)
    ax.set_xlabel('x-coordinate (m)', fontsize=fontsize-1)
    ax.set_ylabel('z-coordinate (m)', fontsize=fontsize-1)
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.set_yticks([-200, -150, -100, -50, 0])
    ax.set_ylim([-200, 0])

plot_time(0)
plot_time(1)
plot_time(2, True)

axes.format(abc='(a)')

#%% IC
import proplot as pplt

fig, axes = pplt.subplots(nrows=3, ncols=2, figsize=(10, 8.5), sharex=3, sharey=3)

fontsize = 12

def plot_time(tidx, add_colorbar=False):
    ax = axes[tidx, 0]
    m1=ax.contourf(lape2.where(mask)[tidx]*1e4, levels=np.linspace(0,50,26), cmap='reds') # minus sign to ensure positive definite
    ax.contour(b.where(mask)[tidx], levels=11, cmap='viridis', lw=0.8)
    if add_colorbar:
        ax.colorbar(m1, loc='b', ticks=5, label='')
    ax.set_title('buoyancy and local APE density (t={})'.format(tidx), fontsize=fontsize)
    ax.set_xlabel('x-coordinate (m)', fontsize=fontsize-1)
    ax.set_ylabel('z-coordinate (m)', fontsize=fontsize-1)
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.set_yticks([-200, -150, -100, -50, 0])
    ax.set_ylim([-200, 0])
    
    ax = axes[tidx, 1]
    msk = sum(masks2)
    lev = xr.concat(ctrs, 'Z').isel(time=tidx).values
    m1=ax.contourf(msk.where(msk!=0)[tidx], cmap='bwr')
    ax.contour(b.where(mask)[tidx], levels=lev[::-1], lw=0.8, color='k')
    if add_colorbar:
        ax.colorbar(m1, loc='b', ticks=1, label='')
    ax.set_title('masks for local APE calculation', fontsize=fontsize)
    ax.set_xlabel('x-coordinate (m)', fontsize=fontsize-1)
    ax.set_ylabel('z-coordinate (m)', fontsize=fontsize-1)
    ax.set_xlim([0, 8960])
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.set_yticks([-200, -150, -100, -50, 0])
    ax.set_ylim([-200, 0])

plot_time(0)
plot_time(1)
plot_time(2, True)

axes.format(abc='(a)')