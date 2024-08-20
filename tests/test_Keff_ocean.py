# -*- coding: utf-8 -*-
"""
Created on 2024.08.16

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""

#%%
import xarray as xr
import numpy as np
from xcontour.xcontour import Contour2D, add_latlon_metrics
from xinvert.xinvert import FiniteDiff

dset = xr.open_dataset('d:/tracerLat.nc')
dset = dset.rename({'XC':'longitude', 'YC':'latitude'})

# add metrics for xgcm
dset, grid = add_latlon_metrics(dset)

# get PV as a tracer and its squared gradient
tracer = dset.PTRACER04.where(dset.PTRACER04!=0)


#%% calculate Laplacian
fd = FiniteDiff(dim_mapping={'Y':'latitude', 'X':'longitude'},
                BCs={'Y':'reflect', 'X':'periodic'},
                coords='lat-lon')

lapl = fd.Laplacian(tracer)
grdx, grdy = fd.grad(tracer, dims=['X','Y'])
grdS = grdx**2 + grdy**2


#%% calculate related quantities for Keff
increase = True
lt = True
check_mono = False

def computeKeff(tracer, grdS):
    # Construct an analysis class using the tracer
    cm = Contour2D(grid, tracer,
                         dims={'X':'longitude','Y':'latitude'},
                         dimEq={'Y':'latitude'},
                         increase=increase,
                         lt=lt, check_mono=check_mono)

    N    = 401
    mask = dset.maskC.rename('mask')
    preY = np.linspace(-70, 75, N)
    
    # This should be called first to initialize contours from minimum value
    # to maximum value (within lat/lon dims) using `N` contours.
    table = cm.cal_area_eqCoord_table_hist(mask)
    ctr   = cm.cal_contours(N).load()
    area  = cm.cal_integral_within_contours_hist(ctr).load().rename('intArea')
    intgrdS = cm.cal_integral_within_contours_hist(ctr, integrand=grdS).rename('intgrdS')

    def calLmin(mask, Yeq):
        preLmin = (mask*dset.dxF).sum('longitude').reset_coords(drop=True)

        re = preLmin.interp(latitude=Yeq.values).rename({'latitude':'contour'}) \
                      .assign_coords({'contour': Yeq['contour'].values})

        return re

    Yeq     = table.lookup_coordinates(area).rename('Yeq')
    Lmin    = calLmin(mask, Yeq).rename('Lmin')
    dgrdSdA = cm.cal_gradient_wrt_area(intgrdS, area)
    dqdA    = cm.cal_gradient_wrt_area(ctr, area)
    Leq2    = cm.cal_sqared_equivalent_length(dgrdSdA, dqdA)
    nkeff   = cm.cal_normalized_Keff(Leq2, Lmin, mask=2e7)

    # Collect all these as a xarray.Dataset defined on N contours and interp to equivalent latitudes
    origin = xr.merge([ctr, area, Yeq, intgrdS, dgrdSdA, dqdA, Leq2, Lmin, nkeff])
    interp = cm.interp_to_dataset(preY, Yeq, origin).rename({'new':'latitude'})
    
    return interp, origin

re1, o1 = computeKeff(tracer, grdS)

#%% plots
import proplot as pplt

fontsize = 13

fig, axes = pplt.subplots(figsize=(9,6), proj='spstere')

ax = axes[0]
m = ax.contourf(tracer, levels=21)
ax.colorbar(m, loc='r', label='')
# ax.set_xlim([0, 360])
ax.set_ylim([-90, -40])
ax.set_title('tracer distribution', fontsize=fontsize)

axes.format(abc='(a)', land=True, coast=True, reso='hi', landcolor='gray')




