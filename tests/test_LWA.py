# -*- coding: utf-8 -*-
"""
Created on 2020.08.01

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""

#%%
import xarray as xr
import numpy as np
from xcontour.xcontour import Contour2D, add_latlon_metrics

dset = xr.open_dataset('./xcontour/Data/barotropic_vorticity.nc')

# # add metrics for xgcm
dset, grid = add_latlon_metrics(dset)

# # get PV as a tracer and its squared gradient
tracer = dset.absolute_vorticity

#%% initialize contours
# Initialize equally-spaced contours from minimum value to maximum value
# (within lat/lon dims).  Here will implicitly loop over each isentropic level

N  = 121           # increase the contour number may get non-monotonic A(q) relation
increase = True    # Y-index increases with latitude
lt = True          # northward of PV contours (larger than) is inside the contour
                   # change this should not change the result of Keff, but may alter
                   # the values at boundaries
dtype = np.float32 # use float32 to save memory
undef = -9.99e8    # for maskout topography if present

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
mask = xr.where(tracer!=undef, 1, 0).astype(dtype)


#%% calculate related quantities for Keff
# xarray's conditional integration, memory consuming and not preferred, for test only
table   = analysis.cal_area_eqCoord_table(mask) # A(Yeq) table
area    = analysis.cal_integral_within_contours(ctr).rename('intArea')
latEq   = table.lookup_coordinates(area).rename('latEq')


#%% calculate related quantities for Keff
# Alternative using _hist APIs, memory friendly and is preferred.
# Note that since xhistogram does not support time- or level-varying bins,
# this way does not support multi-dimensional calculation well as xarray's
# conditional integration
table   = analysis.cal_area_eqCoord_table_hist(mask) # A(Yeq) table
area    = analysis.cal_integral_within_contours_hist(ctr).rename('intArea')
latEq   = table.lookup_coordinates(area).rename('latEq')


#%% combined the results
# results in contour space
ds_contour = xr.merge([ctr, area, latEq])

# interpolate from contour space to equivalent-latitude space
preLats = tracer.latitude.astype(dtype)
# results in latEq space
ds_latEq = analysis.interp_to_dataset(preLats, latEq, ds_contour)


#%% calculate local finite-amplitude wave activity
lwaA, ctrs, masks = analysis.cal_local_wave_activity(tracer, ds_latEq.absolute_vorticity,
                                                    mask_idx=[37,125,170,213],
                                                    part='all')
# lwaU, ctrs, masks = analysis.cal_local_wave_activity(tracer, ds_latEq.absolute_vorticity,
#                                                     mask_idx=[37,125,170,213],
#                                                     part='upper')
# lwaL, ctrs, masks = analysis.cal_local_wave_activity(tracer, ds_latEq.absolute_vorticity,
#                                                     mask_idx=[37,125,170,213],
#                                                     part='lower')

#%% check masks

m = masks[0]
print(-m.where(m<0).sum().values)
print( m.where(m>0).sum().values)
print(grid.integrate(-m.where(m<0), ['X','Y']).values)
print(grid.integrate( m.where(m>0), ['X','Y']).values)
print(grid.get_metric(m, ['X','Y']).sum().values)


