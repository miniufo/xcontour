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
from xcontour.xcontour import Contour2D, add_latlon_metrics

dset = xr.tutorial.open_dataset('air_temperature')

tracer = dset.air

# # add metrics for xgcm
dset, grid = add_latlon_metrics(dset)

dyn = Dynamics(dset, grid=grid, arakawa='A')

# # get PV as a tracer and its squared gradient
grdS = dyn.cal_squared_gradient(tracer, dims=['Y', 'X'],
                                boundary={'Y':'extend', 'X':'extend'})

grdS = grdS.where(np.isfinite(grdS))

#%%
from xinvert.xinvert import FiniteDiff

fd = FiniteDiff({'X':'lon', 'Y':'lat'},
                BCs={'X':'extend', 'Y':'extend'})

grdy, grdx = fd.grad(tracer, dims=['Y', 'X'])

grdS2 = grdy**2 + grdx**2


#%% initialize contours
# Initialize equally-spaced contours from minimum value to maximum value
# (within lat/lon dims).  Here will implicitly loop over each isentropic level

N  = 20           # increase the contour number may get non-monotonic A(q) relation
increase = True    # Y-index increases with latitude
lt = True          # northward of PV contours (larger than) is inside the contour
                   # change this should not change the result of Keff, but may alter
                   # the values at boundaries
dtype = np.float32 # use float32 to save memory
undef = np.nan    # for maskout topography if present

# initialize a Contour2D analysis class using PV as the tracer
analysis = Contour2D(grid, tracer,
                     dims={'X':'lon','Y':'lat'},
                     dimEq={'Y':'lat'},
                     increase=increase,
                     lt=lt)
ctr = analysis.cal_contours(N)

# Mask for A(q) relation table.
# This can be done analytically in simple case, but we choose to do it
# numerically in case there are undefined values inside the domain.
mask = xr.where(np.isnan(tracer), 0, 1).astype(dtype)


#%% calculate related quantities for Keff
# xarray's conditional integration, memory consuming and not preferred, for test only
from xcontour.xcontour import latitude_lengths_at

def calLmin(mask, latEq):
    frac = (mask.lon[-1] - mask.lon[0]) / 360.0
    latLen  = latitude_lengths_at(mask['lat']) * frac
    preLmin = (mask.sum('lon') / len(mask['lon']) * latLen).reset_coords(drop=True)
    
    
    re = []
    
    for tim in latEq:
        re.append(preLmin.interp(lat=tim.values).rename({'lat':'contour'})
                  .assign_coords({'contour': latEq['contour'].values}))
    
    return xr.concat(re, dim='time')

table   = analysis.cal_area_eqCoord_table_hist(mask) # A(Yeq) table
area    = analysis.cal_integral_within_contours_hist(ctr).rename('intArea')
# print('ok1')
# lengths = analysis.cal_contour_lengths(ctr, True).rename('lengths')
# print('ok2')
latEq   = table.lookup_coordinates(area).rename('latEq')
intgrdS = analysis.cal_integral_within_contours_hist(ctr, integrand=grdS).rename('intgrdS')
Lmin    = calLmin(mask, latEq).rename('Lmin')
dintSdA = analysis.cal_gradient_wrt_area(intgrdS, area).rename('dintSdA')
dqdA    = analysis.cal_gradient_wrt_area(ctr, area).rename('dqdA')
Leq2    = analysis.cal_sqared_equivalent_length(dintSdA, dqdA).rename('Leq2')
nkeff   = analysis.cal_normalized_Keff(Leq2, Lmin).rename('nkeff')

#%% combined the results
# results in contour space
ds_contour = xr.merge([ctr, area, latEq, lengths, intgrdS, Lmin, dintSdA, dqdA,
                       Leq2, nkeff])

# interpolate from contour space to equivalent-latitude space
preLats = tracer.latitude.astype(dtype)
# results in latEq space
ds_latEq = analysis.interp_to_dataset(preLats, latEq, ds_contour)


#%%
import proplot as pplt

fontsize= 14

array = [
    [1, 1, 1, 1, 2, 2]
]

fig, axes = pplt.subplots(array, figsize=(10, 5), sharex=0, sharey=3)

ax = axes[0]
ax.contourf(tracer, levels=41)
ax.contour(tracer, levels=21, color='k')
ax.set_title('vorticity', fontsize=fontsize)

ax = axes[1]
m1=ax.plot((ds_latEq.lengths/ds_latEq.Lmin), ds_latEq.latEq, label='$L$')
m2=ax.plot((np.sqrt(ds_latEq.Leq2)/ds_latEq.Lmin), ds_latEq.latEq, label='$L_{eq}$')
ax.set_xlim([0, 8])
ax.set_title('contour length', fontsize=fontsize)

ax.legend([m1,m2], loc='lr', ncols=1)





