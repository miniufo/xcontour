# -*- coding: utf-8 -*-
"""
Created on 2022.08.31

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""

#%% define two methods
import numpy  as np
import xarray as xr
from GeoApps.DiagnosticMethods import Dynamics
from xcontour.xcontour import Contour2D


def computeKeff_hist(tracer, grdS, increase=True, lt=True, check_mono=False):
    # Construct an analysis class using the tracer
    cm = Contour2D(grid, tracer,
                         dims={'X':'longitude','Y':'latitude'},
                         dimEq={'Y':'latitude'},
                         increase=increase,
                         lt=lt, check_mono=check_mono)

    N    = 251
    mask = dset.absolute_vorticity.rename('mask')
    mask[:,:] = 1
    preY = np.linspace(-90, 90, N)

    # This should be called first to initialize contours from minimum value
    # to maximum value (within lat/lon dims) using `N` contours.
    table = cm.cal_area_eqCoord_table_hist(mask)
    ctr   = cm.cal_contours(N).load()
    
    area  = cm.cal_integral_within_contours_hist(ctr).load().rename('intArea')
    intgrdS = cm.cal_integral_within_contours_hist(ctr, integrand=grdS).rename('intgrdS')

    def calLmin(mask, Yeq):
        preLmin = (mask*dset.dxF).sum('longitude').reset_coords(drop=True)

        re = preLmin.interp(latitude=Yeq.values).rename({'latitude':'contour'})\
                  .assign_coords({'contour': Yeq['contour'].values})

        return re
    
    Yeq     = table.lookup_coordinates(area).rename('Yeq')
    Lmin    = calLmin(mask, Yeq).rename('Lmin')
    # lengths = cm.cal_contour_lengths(ctr, latlon=True, period=[None, None]).rename('lengths')
    dgrdSdA = cm.cal_gradient_wrt_area(intgrdS, area)
    dqdA    = cm.cal_gradient_wrt_area(ctr, area)
    Leq2    = cm.cal_sqared_equivalent_length(dgrdSdA, dqdA)
    nkeff   = cm.cal_normalized_Keff(Leq2, Lmin, mask=2e7)

    # Collect all these as a xarray.Dataset defined on N contours and interp to equivalent latitudes
    origin = xr.merge([ctr, area, Yeq, intgrdS, dgrdSdA, dqdA, Leq2, Lmin, nkeff])
    interp = cm.interp_to_dataset(preY, Yeq, origin).rename({'new':'latitude'})
    
    return interp, origin, table


def computeKeff(tracer, grdS, increase=True, lt=True, check_mono=False):
    # Construct an analysis class using the tracer
    cm = Contour2D(grid, tracer,
                         dims={'X':'longitude','Y':'latitude'},
                         dimEq={'Y':'latitude'},
                         increase=increase,
                         lt=lt, check_mono=check_mono)

    N    = 251
    mask = dset.absolute_vorticity.rename('mask')
    mask[:,:] = 1
    preY = np.linspace(-90, 90, N)

    # This should be called first to initialize contours from minimum value
    # to maximum value (within lat/lon dims) using `N` contours.
    table = cm.cal_area_eqCoord_table(mask)
    ctr   = cm.cal_contours(N).load()
    
    area  = cm.cal_integral_within_contours(ctr).load().rename('intArea')
    intgrdS = cm.cal_integral_within_contours(ctr, integrand=grdS).rename('intgrdS')

    def calLmin(mask, Yeq):
        preLmin = (mask*dset.dxF).sum('longitude').reset_coords(drop=True)

        re = preLmin.interp(latitude=Yeq.values).rename({'latitude':'contour'})\
                  .assign_coords({'contour': Yeq['contour'].values})

        return re
    
    Yeq     = table.lookup_coordinates(area).rename('Yeq')
    Lmin    = calLmin(mask, Yeq).rename('Lmin')
    # lengths = cm.cal_contour_lengths(ctr, latlon=True, period=[None, None]).rename('lengths')
    dgrdSdA = cm.cal_gradient_wrt_area(intgrdS, area)
    dqdA    = cm.cal_gradient_wrt_area(ctr, area)
    Leq2    = cm.cal_sqared_equivalent_length(dgrdSdA, dqdA)
    nkeff   = cm.cal_normalized_Keff(Leq2, Lmin, mask=2e7)

    # Collect all these as a xarray.Dataset defined on N contours and interp to equivalent latitudes
    origin = xr.merge([ctr, area, Yeq, intgrdS, dgrdSdA, dqdA, Leq2, Lmin, nkeff])
    interp = cm.interp_to_dataset(preY, Yeq, origin).rename({'new':'latitude'})
    
    return interp, origin, table

#%% preparing data
from GeoApps.GridUtils import add_latlon_metrics


path = 'E:/OneDrive/Python/MyPack/xcontour/Data/barotropic_vorticity.nc'
ds = xr.open_dataset(path)

dset, grid = add_latlon_metrics(ds, boundary={'Y':'extend', 'X':'periodic'})
dyn = Dynamics(dset, grid=grid, arakawa='A')

# # get PV as a tracer and its squared gradient
tr1 = dset.absolute_vorticity.where(dset.absolute_vorticity!=0)
tr2 = tr1.copy()
tr2[:] = tr1[::-1].values
grdS1 = dyn.cal_squared_gradient(tr1, dims=['Y', 'X'], boundary={'Y':'extend', 'X':'periodic'})
grdS2 = dyn.cal_squared_gradient(tr2, dims=['Y', 'X'], boundary={'Y':'extend', 'X':'periodic'})

grdS1 = xr.where(np.isfinite(grdS1), grdS1, np.nan)
grdS2 = xr.where(np.isfinite(grdS2), grdS2, np.nan)

grdS1[0:2,:] = 0
grdS1[-1:-2,:] = 0
grdS2[0:2,:] = 0
grdS2[-1:-2,:] = 0


#%% tests
import proplot as pplt

def doTest(tr, grdS, increase=True, lt=True):
    print('test for tracer {}, increase={}, lt={}'.format(tr.name, increase, lt))
    
    re1, o1, t1 = computeKeff(tr, grdS, increase=increase, lt=lt)
    re2, o2, t2 = computeKeff_hist(tr, grdS, increase=increase, lt=lt)
    
    def plotPanel(ax, var1, var2, title):
        m1 = ax.plot(var1, label='xarray')
        m2 = ax.plot(var2, label='histo')
        ax.set_title(title)
        ax.legend([m1,m2], loc='ul')
        return ax
    
    fig, axes = pplt.subplots(nrows=2, ncols=3, figsize=(11,8),
                              sharex=True, sharey=False)
    
    ax = plotPanel(axes[0,0], t1._table, t2._table, 'A(q) table')
    ax = plotPanel(axes[0,1], re1.absolute_vorticity, re2.absolute_vorticity, 'tracer q(Y)')
    ax = plotPanel(axes[0,2], re1.Yeq, re2.Yeq, 'equivalent Yeq')
    ax = plotPanel(axes[1,0], re1.intArea, re2.intArea, 'area A(Y)')
    ax = plotPanel(axes[1,1], re1.Leq2, re2.Leq2, 'Leq2')
    ax = plotPanel(axes[1,2], re1.nkeff, re2.nkeff, 'nkeff')
    
    return ax

#%% test for tr1
doTest(tr1, grdS1, increase=True , lt=True )
doTest(tr1, grdS1, increase=True , lt=False)
doTest(tr1, grdS1, increase=False, lt=True )
doTest(tr1, grdS1, increase=False, lt=False)

#%% test for tr2
# doTest(tr2, grdS2, increase=True , lt=True )
# doTest(tr2, grdS2, increase=True , lt=False)
# doTest(tr2, grdS2, increase=False, lt=True )
doTest(tr2, grdS2, increase=False, lt=False)



