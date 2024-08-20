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

dset = xr.open_dataset('./xcontour/Data/barotropic_vorticity.nc')

vor = dset.absolute_vorticity
# vor = (vor - vor + vor.latitude).rename('absolute_vorticity')
# vor[50:100, 256:385] = np.nan
# vor[50:100, 256] = np.nan

# vor[60:190,200:400] = np.nan


# # add metrics for xgcm
dset, grid = add_latlon_metrics(dset)

dyn = Dynamics(dset, grid=grid, arakawa='A')

# # get PV as a tracer and its squared gradient
tracer = vor
grdS = dyn.cal_squared_gradient(tracer, dims=['Y', 'X'],
                                boundary={'Y':'extend', 'X':'periodic'})

grdS = grdS.where(np.isfinite(grdS))

#%%
from xinvert.xinvert import FiniteDiff

fd = FiniteDiff({'X':'longitude', 'Y':'latitude'},
                BCs={'X':'periodic', 'Y':'extend'})

grdy, grdx = fd.grad(tracer, dims=['Y', 'X'])

grdS = grdy**2 + grdx**2
grdm = np.hypot(grdx, grdy)


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
# xarray's conditional integration, memory consuming and not preferred, for test only
from xcontour.xcontour import latitude_lengths_at

def calLmin(mask, latEq):
    latLen  = latitude_lengths_at(mask.latitude)
    preLmin = (mask.sum('longitude') / len(mask.longitude) * latLen).reset_coords(drop=True)
    
    re = preLmin.interp(latitude=latEq.values).rename({'latitude':'contour'})\
                  .assign_coords({'contour': latEq['contour'].values})
    
    return re

table   = analysis.cal_area_eqCoord_table_hist(mask) # A(Yeq) table
area    = analysis.cal_integral_within_contours_hist(ctr).rename('intArea')
lengths = analysis.cal_contour_lengths(ctr, latlon=True).rename('lengths')
latEq   = table.lookup_coordinates(area).rename('latEq')
intgrdS = analysis.cal_integral_within_contours_hist(ctr, integrand=grdS).rename('intgrdS')
Lmin    = calLmin(mask, latEq).rename('Lmin')
cmInvGrd= analysis.cal_contour_mean_hist(ctr, 1.0/grdm, grdm, area).rename('cmInvGrd')
cmGrd   = analysis.cal_contour_mean_hist(ctr, grdm, grdm, area).rename('cmGrd')
dintSdA = analysis.cal_gradient_wrt_area(intgrdS, area).rename('dintSdA')
dqdA    = analysis.cal_gradient_wrt_area(ctr, area).rename('dqdA')
Leq2    = analysis.cal_sqared_equivalent_length(dintSdA, dqdA).rename('Leq2')
nkeff   = analysis.cal_normalized_Keff(Leq2, Lmin).rename('nkeff')

#%% combined the results
# results in contour space
ds_contour = xr.merge([ctr, area, latEq, lengths, intgrdS, Lmin, cmInvGrd,
                       cmGrd, dintSdA, dqdA, Leq2, nkeff])

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









#%%
import xarray as xr
import numpy as np
from GeoApps.DiagnosticMethods import Dynamics
from xcontour.xcontour import Contour2D, add_MITgcm_missing_metrics

dset = xr.open_dataset('./xcontour/Data/internalwave.nc')

# # get PV as a tracer and its squared gradient
tracer = dset.THETA.where(dset.THETA!=0)

dset, grid = add_MITgcm_missing_metrics(dset)

#%%
from xinvert.xinvert import FiniteDiff

fd = FiniteDiff({'X':'XC', 'Z':'Z'},
                BCs={'X':'periodic', 'Z':'extend'})

grdz, grdx = fd.grad(tracer, dims=['Z', 'X'])

grdS = grdz**2 + grdx**2


#%% initialize contours
# Initialize equally-spaced contours from minimum value to maximum value
# (within lat/lon dims).  Here will implicitly loop over each isentropic level

N  = 101           # increase the contour number may get non-monotonic A(q) relation
increase = False   # Y-index increases with latitude
lt = False         # northward of PV contours (larger than) is inside the contour
                   # change this should not change the result of Keff, but may alter
                   # the values at boundaries
dtype = np.float32 # use float32 to save memory
undef = 0    # for maskout topography if present

# initialize a Contour2D analysis class using PV as the tracer
analysis = Contour2D(grid, tracer,
                     dims={'X':'XC','Z':'Z'},
                     dimEq={'Z':'Z'},
                     increase=increase,
                     lt=lt)
ctr = analysis.cal_contours(N)

# Mask for A(q) relation table.
# This can be done analytically in simple case, but we choose to do it
# numerically in case there are undefined values inside the domain.
mask = xr.where(np.isnan(tracer), 0, 1)[0].astype(dtype)


#%% calculate related quantities for Keff
# xarray's conditional integration, memory consuming and not preferred, for test only
hgrid, xcount = 2, 4480

def calLmin(mask, zEq):
    preLmin = (mask.sum('XC') * hgrid).reset_coords(drop=True)
    
    re = []
    
    for tim in zEq:
        re.append(preLmin.interp(Z=tim.values).rename({'Z':'contour'})
                  .assign_coords({'contour': zEq['contour'].values}))
    
    return xr.concat(re, dim='time')

table   = analysis.cal_area_eqCoord_table_hist(mask) # A(Yeq) table
area    = analysis.cal_integral_within_contours_hist(ctr).rename('intArea')
lengths = analysis.cal_contour_lengths(ctr, False).rename('lengths')
ZEq     = table.lookup_coordinates(area).rename('zEq')
intgrdS = analysis.cal_integral_within_contours_hist(ctr, integrand=grdS).rename('intgrdS')
Lmin    = calLmin(mask, ZEq).rename('Lmin')
dintSdA = analysis.cal_gradient_wrt_area(intgrdS, area).rename('dintSdA')
dqdA    = analysis.cal_gradient_wrt_area(ctr, area).rename('dqdA')
Leq2    = analysis.cal_sqared_equivalent_length(dintSdA, dqdA).rename('Leq2')
nkeff   = analysis.cal_normalized_Keff(Leq2, Lmin).rename('nkeff')

#%% combined the results
# results in contour space
ds_contour = xr.merge([ctr, area, ZEq, lengths, intgrdS, Lmin, dintSdA, dqdA,
                       Leq2, nkeff])

# interpolate from contour space to equivalent-latitude space
preZs = tracer.Z.astype(dtype)
# results in latEq space
ds_ZEq = analysis.interp_to_dataset(preZs, ZEq, ds_contour)


#%%
import proplot as pplt

fontsize= 14

array = [
    [1, 1, 1, 1, 2, 2],
    [3, 3, 3, 3, 4, 4],
    [5, 5, 5, 5, 6, 6],
]

fig, axes = pplt.subplots(array, figsize=(10, 10), sharex=0, sharey=3)

tstep = 0
ax = axes[0]
ax.contourf(tracer[tstep], levels=41, cmap='bwr')
ax.contour(tracer[tstep], levels=21, color='k')
ax.set_title('vorticity', fontsize=fontsize)
ax.set_ylim([-200, 0])

ax = axes[1]
m1=ax.plot((ds_ZEq.lengths/ds_ZEq.Lmin)[tstep], ds_ZEq.zEq[tstep], label='L')
m2=ax.plot(np.sqrt(ds_ZEq.nkeff)[tstep], ds_ZEq.zEq[tstep], label='$L_{eq}$')
ax.set_xlim([0, 3])
ax.set_title('contour length', fontsize=fontsize)
ax.legend([m1,m2], loc='ur')

tstep = 1
ax = axes[2]
ax.contourf(tracer[tstep], levels=41, cmap='bwr')
ax.contour(tracer[tstep], levels=21, color='k')
ax.set_title('vorticity', fontsize=fontsize)
ax.set_ylim([-200, 0])

ax = axes[3]
m1=ax.plot((ds_ZEq.lengths/ds_ZEq.Lmin)[tstep], ds_ZEq.zEq[tstep], label='L')
m2=ax.plot(np.sqrt(ds_ZEq.nkeff)[tstep], ds_ZEq.zEq[tstep], label='$L_{eq}$')
ax.set_xlim([0, 3])
ax.set_title('contour length', fontsize=fontsize)
ax.legend([m1,m2], loc='ur')

tstep = 2
ax = axes[4]
ax.contourf(tracer[tstep], levels=41, cmap='bwr')
ax.contour(tracer[tstep], levels=21, color='k')
ax.set_title('vorticity', fontsize=fontsize)
ax.set_ylim([-200, 0])

ax = axes[5]
m1=ax.plot((ds_ZEq.lengths/ds_ZEq.Lmin)[tstep], ds_ZEq.zEq[tstep], label='L')
m2=ax.plot(np.sqrt(ds_ZEq.nkeff)[tstep], ds_ZEq.zEq[tstep], label='$L_{eq}$')
ax.set_xlim([0, 3])
ax.set_title('contour length', fontsize=fontsize)
ax.legend([m1,m2], loc='ur')

axes.format(xlabel='', ylabel='')




#%% global tracer test
import xmitgcm
from matplotlib import pyplot as plt
import proplot as pplt
import numpy as np
import xarray as xr
from GeoApps.GridUtils import add_MITgcm_missing_metrics
from GeoApps.DiagnosticMethods import Dynamics

path = 'I:/AVISO/GlobalTracers/'

ds = xmitgcm.open_mdsdataset(path, prefix=['PTRACER01','PTRACER04','PTRACER07']).isel(time=slice(0,2))
dset, grid = add_MITgcm_missing_metrics(ds, periodic='X', boundary={'Y':'extend'})

dyn = Dynamics(dset, grid=grid, arakawa='A')

# # get PV as a tracer and its squared gradient
tr1 = dset.PTRACER01.where(dset.PTRACER01!=0)
grdS1 = dyn.cal_squared_gradient(tr1, dims=['Y', 'X'], boundary={'Y':'fill', 'X':'periodic'})

tr4 = dset.PTRACER04.where(dset.PTRACER04!=0)
grdS4 = dyn.cal_squared_gradient(tr4, dims=['Y', 'X'], boundary={'Y':'fill', 'X':'periodic'})

tr7 = dset.PTRACER07.where(dset.PTRACER07!=0)
grdS7 = dyn.cal_squared_gradient(tr7, dims=['Y', 'X'], boundary={'Y':'fill', 'X':'periodic'})

print(dset)

#%%
# calculate contour length
from xcontour.xcontour import Contour2D

increase = True
lt = True
check_mono = False

def computeKeff(tracer, grdS):
    # Construct an analysis class using the tracer
    cm = Contour2D(grid, tracer,
                         dims={'X':'XC','Y':'YC'},
                         dimEq={'Y':'YC'},
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
        preLmin = (mask*dset.dxF).sum('XC').reset_coords(drop=True)

        re = []

        for tim in Yeq:
            re.append(preLmin.interp(YC=tim.values).rename({'YC':'contour'})
                      .assign_coords({'contour': Yeq['contour'].values}))

        return xr.concat(re, dim='time')

    Yeq     = table.lookup_coordinates(area).rename('Yeq')
    Lmin    = calLmin(mask, Yeq).rename('Lmin')
    lengths = cm.cal_contour_lengths(ctr, latlon=True).rename('lengths')
    dgrdSdA = cm.cal_gradient_wrt_area(intgrdS, area)
    dqdA    = cm.cal_gradient_wrt_area(ctr, area)
    Leq2    = cm.cal_sqared_equivalent_length(dgrdSdA, dqdA)
    nkeff   = cm.cal_normalized_Keff(Leq2, Lmin, mask=2e7)

    # Collect all these as a xarray.Dataset defined on N contours and interp to equivalent latitudes
    origin = xr.merge([ctr, area, Yeq, intgrdS, dgrdSdA, dqdA, Leq2, Lmin, lengths, nkeff])
    interp = cm.interp_to_dataset(preY, Yeq, origin).rename({'new':'YC'})
    
    return interp, origin

re1, o1 = computeKeff(tr1, grdS1)
print('re1 ok')
re4, o4 = computeKeff(tr4, grdS4)
print('re4 ok')
re7, o7 = computeKeff(tr7, grdS7)
print('re7 ok')


#%% compare initial and final
import proplot as pplt

fontsize= 14

array = [
    [1, 1, 1, 2],
    [3, 3, 3, 4],
]

fig, axes = pplt.subplots(array, figsize=(11, 10), sharex=0, sharey=0,
                          proj=['kav7', None, 'kav7', None],
                          proj_kw={'central_longitude': 180})

step = 0
ax = axes[0]
m1=ax.contourf(tr1[step], levels=np.linspace(1,2,21), cmap='rainbow')
ax.contour(tr1[step], levels=11, color='k')
ax.set_title('vorticity (t=1)', fontsize=fontsize)
ax.colorbar(m1, loc='b', label='', ticks=0.2, length=0.7)

ax = axes[1]
m1=ax.plot((re1.lengths/re1.Lmin)[step], re1.Yeq[step], label='$L$')
m2=ax.plot((np.sqrt(re1.Leq2)/re1.Lmin)[step], re1.Yeq[step], label='$L_{eq}$')
ax.set_xlim([0, 3])
ax.set_title('contour length (t=1)', fontsize=fontsize)
ax.set_ylabel('Equivalent Latitude', fontsize=fontsize-2)
ax.set_ylim([-90, 90])

ax.legend([m1,m2], loc='lr', ncols=1)

step = -1
ax = axes[2]
m1=ax.contourf(tr1[step], levels=np.linspace(1,2,21), cmap='rainbow')
ax.contour(tr1[step], levels=11, color='k', linewidth=0.5)
ax.set_title('vorticity (t=365)', fontsize=fontsize)

ax = axes[3]
m1=ax.plot((re1.lengths/re1.Lmin)[step], re1.Yeq[step], label='$L$')
m2=ax.plot((np.sqrt(re1.Leq2)/re1.Lmin)[step], re1.Yeq[step], label='$L_{eq}$')
ax.set_xlim([0, 40])
ax.set_title('contour length (t=365)', fontsize=fontsize)
ax.set_ylabel('Equivalent Latitude', fontsize=fontsize-2)
ax.set_ylim([-90, 90])

ax.legend([m1,m2], loc='lr', ncols=1)

axes.format(abc='(a)', land=True, landcolor='gray')

#%% compare different kappa
import proplot as pplt

fontsize= 14
step = -1

array = [
    [1, 1, 1, 2],
    [3, 3, 3, 4],
    [5, 5, 5, 6],
]

fig, axes = pplt.subplots(array, figsize=(10, 11), sharex=0, sharey=0,
                          proj=['kav7', None, 'kav7', None, 'kav7', None],
                          proj_kw={'central_longitude': 150})

ax = axes[0]
ax.contourf(tr1[step], levels=np.linspace(1,2,21), cmap='rainbow')
ax.contour(tr1[step], levels=21, color='k', linewidth=0.5)
ax.set_title('vorticity ($\kappa=0$)', fontsize=fontsize)

ax = axes[1]
m1=ax.plot((re1.lengths/re1.Lmin)[step], re1.Yeq[step], label='$L$')
m2=ax.plot((np.sqrt(re1.Leq2)/re1.Lmin)[step], re1.Yeq[step], label='$L_{eq}$')
ax.set_xlim([0, 40])
ax.set_title('contour length ($\kappa=0$)', fontsize=fontsize)
ax.set_ylabel('Equivalent Latitude', fontsize=fontsize-2)
ax.set_ylim([-90, 90])

ax.legend([m1,m2], loc='lr', ncols=1)

ax = axes[2]
m1=ax.contourf(tr4[step], levels=np.linspace(1,2,21), cmap='rainbow')
ax.contour(tr4[step], levels=21, color='k', linewidth=0.5)
ax.set_title('vorticity ($\kappa=20$)', fontsize=fontsize)
ax.colorbar(m1, loc='r', ticks=0.2, label='', length=1)

ax = axes[3]
m1=ax.plot((re4.lengths/re4.Lmin)[step], re4.Yeq[step], label='$L$')
m2=ax.plot((np.sqrt(re4.Leq2)/re4.Lmin)[step], re4.Yeq[step], label='$L_{eq}$')
ax.set_xlim([0, 40])
ax.set_title('contour length ($\kappa=20$)', fontsize=fontsize)
ax.set_ylabel('Equivalent Latitude', fontsize=fontsize-2)
ax.set_ylim([-90, 90])

ax.legend([m1,m2], loc='lr', ncols=1)

ax = axes[4]
ax.contourf(tr7[step], levels=np.linspace(1,2,21), cmap='rainbow')
ax.contour(tr7[step], levels=21, color='k', linewidth=0.5)
ax.set_title('vorticity ($\kappa=50$)', fontsize=fontsize)

ax = axes[5]
m1=ax.plot((re7.lengths/re7.Lmin-1)[step], re7.Yeq[step], label='$L$')
m2=ax.plot((np.sqrt(re7.Leq2)/re7.Lmin)[step], re7.Yeq[step], label='$L_{eq}$')
ax.set_xlim([0, 40])
ax.set_title('contour length ($\kappa=50$)', fontsize=fontsize)
ax.set_ylabel('Equivalent Latitude', fontsize=fontsize-2)
ax.set_ylim([-90, 90])

ax.legend([m1,m2], loc='lr', ncols=1)

axes.format(abc='(a)', land=True, landcolor='gray')















#%%
import xmitgcm
from matplotlib import pyplot as plt
import proplot as pplt
import numpy as np
import xarray as xr
from GeoApps.GridUtils import add_MITgcm_missing_metrics
from GeoApps.DiagnosticMethods import Dynamics

path = 'D:/AVISO/GlobalTracers/'

ds = xmitgcm.open_mdsdataset(path, prefix=['PTRACER01']).isel(time=slice(0,2))
dset, grid = add_MITgcm_missing_metrics(ds, periodic='X', boundary={'Y':'extend'})

dyn = Dynamics(dset, grid=grid, arakawa='A')

# # get PV as a tracer and its squared gradient
tr1 = dset.PTRACER01.where(dset.PTRACER01!=0)
grdS1 = dyn.cal_squared_gradient(tr1, dims=['Y', 'X'], boundary={'Y':'fill', 'X':'periodic'})

print(dset)

#%%
# calculate contour length
from xcontour.xcontour import Contour2D

increase = True
lt = True
check_mono = False

def computeKeff(tracer, grdS):
    # Construct an analysis class using the tracer
    cm = Contour2D(grid, tracer,
                         dims={'Y':'YC','X':'XC'},
                         dimEq={'Y':'YC'},
                         increase=increase,
                         lt=lt, check_mono=check_mono)

    N    = 401
    ydef = dset.YC
    mask = dset.maskC.rename('mask')
    preY = np.linspace(-70, 75, N)
    
    # This should be called first to initialize contours from minimum value
    # to maximum value (within lat/lon dims) using `N` contours.
    table = cm.cal_area_eqCoord_table_hist(mask)
    ctr   = cm.cal_contours(N).load()
    area  = cm.cal_integral_within_contours_hist(ctr).load().rename('intArea')
    intgrdS = cm.cal_integral_within_contours_hist(ctr, integrand=grdS).rename('intgrdS')

    def calLmin(mask, Yeq):
        preLmin = (mask*dset.dxF).sum('XC').reset_coords(drop=True)
        preLmin = preLmin.where(preLmin!=0)

        re = []

        for tim in Yeq:
            re.append(preLmin.interp(YC=tim.values).rename({'YC':'contour'})
                      .assign_coords({'contour': Yeq['contour'].values}))

        return xr.concat(re, dim='time')

    Yeq     = table.lookup_coordinates(area).rename('Yeq')
    Lmin    = calLmin(mask, Yeq).rename('Lmin')
    lengths = cm.cal_contour_lengths(ctr, latlon=True).rename('lengths')
    dgrdSdA = cm.cal_gradient_wrt_area(intgrdS, area)
    dqdA    = cm.cal_gradient_wrt_area(ctr, area)
    Leq2    = cm.cal_sqared_equivalent_length(dgrdSdA, dqdA)
    nkeff   = cm.cal_normalized_Keff(Leq2, Lmin, mask=2e7)

    # Collect all these as a xarray.Dataset defined on N contours and interp to equivalent latitudes
    origin = xr.merge([ctr, area, Yeq, intgrdS, dgrdSdA, dqdA, Leq2, Lmin, lengths, nkeff])
    interp = cm.interp_to_dataset(preY, Yeq, origin).rename({'new':'YC'})
    
    return interp, origin

re2, o2 = computeKeff(tr1, grdS1)

#%%
import proplot as pplt

fontsize= 13

array = [
    [1, 1, 1, 2],
]

fig, axes = pplt.subplots(array, figsize=(11, 5), sharex=0, sharey=0, facecolor='w')

step = 0
ax = axes[0]
m1=ax.contourf(tr1[step], levels=np.linspace(1,2,21), cmap='rainbow')
ax.contour(tr1[step], levels=11, color='k')
ax.set_title('vorticity (t=1)', fontsize=fontsize)
ax.colorbar(m1, loc='b', label='', ticks=0.2, length=0.7)

tmp = (re2.lengths/re2.Lmin)[step]
ax = axes[1]
m1=ax.plot(tmp, re2.Yeq[step], label='$L$')
tmp = (np.sqrt(re2.Leq2)/re2.Lmin)[step]
m2=ax.plot(tmp, re2.Yeq[step], label='$L_{eq}$')
ax.set_xlim([0, 3])
ax.set_title('contour length (t=1)', fontsize=fontsize)
ax.set_ylabel('Equivalent Latitude', fontsize=fontsize-2)
ax.set_ylim([-90, 90])

ax.legend([m1,m2], loc='lr', ncols=1)


#%%
from xcontour.xcontour import find_contour, contour_length


segs = find_contour(tr1[0], ['YC', 'XC'], 1.15076609, period=[None, None])

fig, axes = pplt.subplots(nrows=1, ncols=1, figsize=(11, 6))

ax = axes[0]

ax.contourf(tr1[0, 120:300], cmap='jet')
for seg in segs:
    ax.plot(seg[:, 1], seg[:, 0], linewidth=1, color='k')
ax.set_ylim([-68, -50])
ax.set_xlim([0, 360])

print(contour_length(segs[0][:,::-1], latlon=True))



