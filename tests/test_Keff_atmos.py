# -*- coding: utf-8 -*-
"""
Created on 2020.08.01

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""

#%%
import xarray as xr
import numpy as np
from xcontour.xcontour import Contour2D, latitude_lengths_at, add_latlon_metrics

dset = xr.open_dataset('./xcontour/Data/PV.nc')

# add metrics for xgcm
dset, grid = add_latlon_metrics(dset)

# get PV as a tracer and its squared gradient
tracer = dset.pv
grdS = dset.grdSpv

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
intgrdS = analysis.cal_integral_within_contours(ctr, integrand=grdS).rename('intgrdS')
latEq   = table.lookup_coordinates(area).rename('latEq')
Lmin    = latitude_lengths_at(latEq).rename('Lmin')
dintSdA = analysis.cal_gradient_wrt_area(intgrdS, area).rename('dintSdA')
dqdA    = analysis.cal_gradient_wrt_area(ctr, area).rename('dqdA')
Leq2    = analysis.cal_sqared_equivalent_length(dintSdA, dqdA).rename('Leq2')
nkeff   = analysis.cal_normalized_Keff(Leq2, Lmin).rename('nkeff')

#%% calculate related quantities for Keff
# Alternative using _hist APIs, memory friendly and is preferred.
# Note that since xhistogram does not support time- or level-varying bins,
# this way does not support multi-dimensional calculation well as xarray's
# conditional integration
table   = analysis.cal_area_eqCoord_table_hist(mask) # A(Yeq) table
area    = analysis.cal_integral_within_contours_hist(ctr).rename('intArea')
intgrdS = analysis.cal_integral_within_contours_hist(ctr, integrand=grdS).rename('intgrdS')
latEq   = table.lookup_coordinates(area).rename('latEq')
Lmin    = latitude_lengths_at(latEq).rename('Lmin')
dintSdA = analysis.cal_gradient_wrt_area(intgrdS, area).rename('dintSdA')
dqdA    = analysis.cal_gradient_wrt_area(ctr, area).rename('dqdA')
Leq2    = analysis.cal_sqared_equivalent_length(dintSdA, dqdA).rename('Leq2')
nkeff   = analysis.cal_normalized_Keff(Leq2, Lmin).rename('nkeff')

#%% combined the results
# results in contour space
ds_contour = xr.merge([ctr, area, intgrdS, latEq, dintSdA, dqdA, Leq2, Lmin, nkeff])

# interpolate from contour space to equivalent-latitude space
preLats = np.linspace(-90, 90, 181).astype(dtype)
# results in latEq space
ds_latEq = analysis.interp_to_dataset(preLats, latEq, ds_contour)




