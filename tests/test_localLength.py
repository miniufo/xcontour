# -*- coding: utf-8 -*-
"""
Created on 2022.11.16

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import xmitgcm
import numpy as np
import xarray as xr

path = 'D:/AVISO/GlobalTracers/'

ds = xmitgcm.open_mdsdataset(path, prefix=['PTRACER01']).isel(time=slice(0,2))

# # get PV as a tracer and its squared gradient
tr1 = ds.PTRACER01.where(ds.PTRACER01!=0)

print(tr1)


#%%
from skimage import measure
from xcontour.xcontour import contour_length
import numba as nb


def c_len(data, contour, coordy=None, coordx=None):
    # in unit of grid points
    segments = measure.find_contours(data, contour)
    
    f_list = nb.typed.List.empty_list(nb.typeof(np.zeros((9,2))))
    
    for segment in segments:
        f_list.append(segment)
    
    return contour_length(f_list, coordx.data, coordy.data, latlon=True)


rolling = tr1.rolling(YC=101, XC=101, center=True, min_periods=2000)
rolled = rolling.construct({'YC':'yw','XC':'xw'}, stride=10)#chunk({'time':1, 'YC':400, 'XC':400})
rollmean = rolling.construct({'YC':'yw','XC':'xw'}, stride=10).mean(['yw','xw'])

mrolled = (tr1-tr1+tr1.YC)[0].rolling(YC=101, XC=101, center=True,
                                      min_periods=10).construct({'YC':'yw','XC':'xw'},
                                                               stride=10)
mrollm = (tr1-tr1+tr1.YC)[0].rolling(YC=101, XC=101, center=True,
                                     min_periods=10).construct({'YC':'yw','XC':'xw'},
                                                               stride=10).mean(['yw','xw'])

# print(c_len(tr1, 1.35, tr1.YC, tr1.XC))

#%%
clens = xr.apply_ufunc(c_len, rolled, rollmean,
                       dask='parallelized',
                       kwargs={'coordy':np.deg2rad(rolled.YC),
                               'coordx':np.deg2rad(rolled.XC)},
                       input_core_dims=[['yw','xw'], []],
                       vectorize=True,
                       output_dtypes=[np.float32])

mlens = xr.apply_ufunc(c_len, mrolled, mrollm,
                       dask='parallelized',
                       kwargs={'coordy':np.deg2rad(mrolled.YC),
                               'coordx':np.deg2rad(mrolled.XC)},
                       input_core_dims=[['yw','xw'], []],
                       vectorize=True,
                       output_dtypes=[np.float32])

#%%
tmp1 = (clens/mlens)[0].load()
tmp2 = (clens/mlens)[1].load()

#%%
import proplot as pplt

fontsize = 15

fig, axes = pplt.subplots(nrows=2, ncols=2, figsize=(11, 7.7), sharex=3, sharey=3,
                          proj='cyl', proj_kw={'central_longitude':180})

ax = axes[0,0]
m1 = ax.contourf(tr1[0], levels=np.linspace(1, 2, 41), cmap='prism')
ax.set_title('tracer distribution (t=0)', fontsize=fontsize)
ax.colorbar(m1, loc='b', label='', ticks=0.2)

ax = axes[0,1]
m1 = ax.contourf(tmp1, levels=np.linspace(0, 2, 21), cmap='jet')
ax.set_title('local contour length relative to Lmin (t=0)', fontsize=fontsize)
ax.colorbar(m1, loc='b', label='', ticks=0.5)

ax = axes[1,0]
m1 = ax.contourf(tr1[1], levels=np.linspace(1, 2, 41), cmap='prism')
ax.set_title('tracer distribution (t=1)', fontsize=fontsize)
ax.colorbar(m1, loc='b', label='', ticks=0.2)

ax = axes[1,1]
m1 = ax.contourf(tmp2, levels=np.linspace(0, 30, 31), cmap='jet')
ax.set_title('local contour length relative to Lmin (t=1)', fontsize=fontsize)
ax.colorbar(m1, loc='b', label='', ticks=5)

axes.format(abc='(a)', land=True, coast=True, lonlabels=60, latlabels=30,
            landcolor='gray')




