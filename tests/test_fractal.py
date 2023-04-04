# -*- coding: utf-8 -*-
"""
Created on 2022.09.30

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import xarray as xr
import numpy as np
from GeoApps.DiagnosticMethods import Dynamics
from xcontour.xcontour import Contour2D, add_latlon_metrics, latitude_lengths_at

dset = xr.open_dataset('./xcontour/Data/barotropic_vorticity.nc')

vor = dset.absolute_vorticity

# # add metrics for xgcm
dset, grid = add_latlon_metrics(dset)

dyn = Dynamics(dset, grid=grid, arakawa='A')

tracer = vor


#%% initialize contours
# Initialize equally-spaced contours from minimum value to maximum value
# (within lat/lon dims).  Here will implicitly loop over each isentropic level

N  = 121           # increase the contour number may get non-monotonic A(q) relation
increase = True    # Y-index increases with latitude
lt = True          # northward of PV contours (larger than) is inside the contour
                   # change this should not change the result of Keff, but may alter
                   # the values at boundaries
dtype = np.float32 # use float32 to save memory
undef = np.nan    # for maskout topography if present

# initialize a Contour2D analysis class using PV as the tracer
analysis = Contour2D(grid, tracer,
                     dims={'X':'longitude','Y':'latitude'},
                     dimEq={'Y':'latitude'},
                     increase=increase,
                     lt=lt)
ctr = analysis.cal_contours(N)

# Mask for A(q) relation table.
# This can be done analytically in simple case, but we choose to do it
# numerically in case there are undefined values inside the domain.
mask = xr.where(np.isnan(tracer), 0, 1).astype(dtype)


#%% calculate related quantities for Keff

table = analysis.cal_area_eqCoord_table_hist(mask) # A(Yeq) table
area  = analysis.cal_integral_within_contours_hist(ctr).rename('intArea')
latEq = table.lookup_coordinates(area).rename('latEq')
Lmin  = latitude_lengths_at(latEq).rename('Lmin')

#%%
from utils.XarrayUtils import coarsen

strides = [1, 2, 4, 8, 16, 32]

re = []

for ratio in strides:
    tracerS = coarsen(tracer, dims=['latitude','longitude'],
                      periodic='longitude', ratio=ratio)
    lengths = analysis.cal_contour_lengths(ctr, tracer=tracerS,
                                           latlon=True).rename('lengths')
    re.append(lengths)

bclens  = analysis.cal_contour_crossing(ctr, stride=strides, mode='edge')


#%% combined the results
re = [r.rename('length'+str(s)) for r, s in zip(re    , strides)]
bc = [b.rename('bclens'+str(s)) for b, s in zip(bclens, strides)]

# results in contour space
ds_contour = xr.merge([ctr, area, latEq, Lmin]+re+bc)

# interpolate from contour space to equivalent-latitude space
preLats = tracer.latitude.astype(dtype)
# results in latEq space
ds_latEq = analysis.interp_to_dataset(preLats, latEq, ds_contour)


#%% calculate fractal dimensions
def linear_fit(x, y):
    # wrapper function of np.polyfit
    try:
        fitted = np.polyfit(x, y, 1)
        return fitted[0] # return slope only, no intersect
    except Exception:
        return np.nan
    
reso = dset.longitude.diff('longitude')[0] # degree

lengths = xr.concat([ds_latEq.length1 , ds_latEq.length2  , ds_latEq.length4 ,
                     ds_latEq.length8 , ds_latEq.length16 , ds_latEq.length32],
                    dim='stride')
lengths['stride'] = strides

rulers = lengths.stride * np.cos(np.deg2rad(ds_latEq.latitude)) * reso * np.pi / 180.0 * 6371200

counts = lengths / rulers

xcoord = -np.log(rulers)[::-1]
ycoord =  np.log(counts)[::-1]

fd  = xr.apply_ufunc(linear_fit, xcoord, ycoord,
                     input_core_dims=[['stride'], ['stride']],
                     vectorize=True,
                     dask='allowed')

#%%
import proplot as pplt

fontsize= 14

array = [
    [1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4]
]

fig, axes = pplt.subplots(array, figsize=(11.5, 5), sharex=0, sharey=3)

ax = axes[0]
ax.contourf(tracer, levels=41)
ax.contour(tracer, levels=21, color='k')
ax.set_title('vorticity', fontsize=fontsize)
ax.set_ylabel('')

ax = axes[1]
m1=ax.plot((ds_latEq.length1/ds_latEq.Lmin), ds_latEq.latEq, label='$L1$')
m2=ax.plot((ds_latEq.length2/ds_latEq.Lmin), ds_latEq.latEq, label='$L2$')
m3=ax.plot((ds_latEq.length4/ds_latEq.Lmin), ds_latEq.latEq, label='$L4$')
m4=ax.plot((ds_latEq.length8/ds_latEq.Lmin), ds_latEq.latEq, label='$L8$')
m5=ax.plot((ds_latEq.length16/ds_latEq.Lmin), ds_latEq.latEq, label='$L16$')
m6=ax.plot((ds_latEq.length32/ds_latEq.Lmin), ds_latEq.latEq, label='$L32$')
ax.set_xlim([0, 5])
ax.set_title('contour length', fontsize=fontsize)
ax.legend([m1,m2,m3,m4,m5,m6], loc='lr', ncols=1)

ax = axes[2]
m1=ax.plot((ds_latEq.bclens1/ds_latEq.Lmin), ds_latEq.latEq, label='$L1$')
m2=ax.plot((ds_latEq.bclens2/ds_latEq.Lmin), ds_latEq.latEq, label='$L2$')
m3=ax.plot((ds_latEq.bclens4/ds_latEq.Lmin), ds_latEq.latEq, label='$L4$')
m4=ax.plot((ds_latEq.bclens8/ds_latEq.Lmin), ds_latEq.latEq, label='$L8$')
m5=ax.plot((ds_latEq.bclens16/ds_latEq.Lmin), ds_latEq.latEq, label='$L16$')
m6=ax.plot((ds_latEq.bclens32/ds_latEq.Lmin), ds_latEq.latEq, label='$L32$')
ax.set_xlim([0, 5])
ax.set_title('contour length (BC)', fontsize=fontsize)
ax.legend([m1,m2,m3,m4,m5,m6], loc='lr', ncols=1)

ax = axes[3]
ax.plot(fd, ds_latEq.latEq)
ax.set_title('fractal dimension', fontsize=fontsize)
ax.set_xlim([1, 1.6])




