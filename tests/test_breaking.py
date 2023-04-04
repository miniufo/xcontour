# -*- coding: utf-8 -*-
"""
Created on 2022.08.31

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""

#%%
# for data
import xarray as xr

# for algorithm
import numpy as np
import itertools as itertools
from skimage import measure
from scipy import spatial
import pandas as pd

# for plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.patches as mpatches


# load data
path = "D:/Data/ERAInterim/ElenaIsen/198508ERAInterim.nc"
dataset = xr.open_dataset(path).isel({'time':0, 'level':-1})

# specify names of coordinates
lat_name = "latitude"
lon_name = "longitude"
time_name = "time"
var_name = "pv"

# get lon border
lon_border = [int(dataset[lon_name].values.min()), int(dataset[lon_name].values.max())]
lon_border = [0, 360]


# Subroutine 1
def ex_contours(data, level):
    """
    The extractions is based on the "find_contours" function of "measure" by "skimage".
    The output of the "find_contours" function is modified to obtain the original coordinates.
    The final output contains a list with entries each representing a single contour segment.
    
    Input:
    -----
    dataset: xarray
        xarray with two spatial dimensions but no time dimension
        
    level: float
        contour level
    
    Returns:
    -------
    list: arrays
        List with entries each representing a single contour segment
        
    """
    return [np.c_[contour[:,1]+min(lon_border),contour[:,0]]
            for contour in measure.find_contours(data.values,level)]


# Subroutine 2
def rescale_contours(contours, dataset):
    """
    Rescale the coordinates of the contour points. 
    The coordinates are fitted on a grid with the same resolution as the input field.
    
    Input:
    -----
    contours: list
        List with entries each representing a single contour segment
        
    dataset: xarray
        xarray with two spatial dimensions but no time dimension
    
    Returns:
    -------
    list: arrays
        List with entries each representing a single contour segment
    
    """
    x, y = np.meshgrid(dataset[lon_name],dataset[lat_name]) 
    x, y = x.flatten(), y.flatten()
    grid_points = np.vstack((x,y)).T 
    tree = spatial.KDTree(grid_points)

    contours_scaled = []
    for contour in contours:
        temp = []
        for point in contour:
            temp.append(grid_points[tree.query([point])[1][0]])
        unique = list(dict.fromkeys(map(tuple,temp)))
        contours_scaled.append(np.asarray(unique))
            
    return contours_scaled


# Subroutine 3
def group_contours(contours, y_overlap):
    """
    Group the contours according to their start and end point.
    Using the parameter "y_overlap" (in degrees), the interval representing the overlap region can be specified.
    
    Input:
    -----
    contours: list
        List with entries each representing a single contour segment
        
    y_overlap: float, optional
        Overlap in y direction in degrees in between unclosed contours at longitude border are grouped
    
    Returns:
    -------
    list: arrays
        List with entries each representing a single contour segment
    
    """
    contours_index = list(range(0,len(contours)))
    borders = []
    for contour,index in zip(contours,contours_index):
        stp = contour[0].tolist()
        ndp = contour[-1].tolist()
        borders.append([index]+stp+ndp)

    start = [[border[1],border[2]] for border in borders]
    end = [[border[3],border[4]] for border in borders]
    both = start+end
    
    for point in both:
        ind = [i for i, x in enumerate(both) if (x[0]==point[0] or all(i in [x[0],point[0]] for i in lon_border)) and point[1]-y_overlap <= x[1] <= point[1]+y_overlap]
        ind = np.mod(ind,len(contours))
        add = [borders[i][0] for i in ind]
        for i in ind:
            borders[i][0] = borders[min(add)][0]    

    all_values = [border[0] for border in borders]
    unique_values = set(all_values)

    contours_grouped = []
    for value in unique_values:
        this_group = []
        for border,contour in zip(borders,contours):
            if border[0] == value:
                this_group.append(contour)
        contours_grouped.append(this_group)
    
    contours_grouped_sorted = []
    for group in contours_grouped:
        if len(group) > 1:
            bigest = sorted(group, key=len, reverse=True)[0]
            rest = sorted(group, key=len, reverse=True)[1:]

            temp = [bigest]
            while len(rest) > 0:
                test = temp[-1][-1,1]
                for item,ind in zip(rest, range(0,len(rest))):
                    if test-y_overlap <= item[0,1] <= test+y_overlap:
                        temp.append(item)
                        break
                del rest[ind]
            contours_grouped_sorted.append(np.asarray(list(itertools.chain.from_iterable(temp))))
        else:
            contours_grouped_sorted.append(np.asarray(list(itertools.chain.from_iterable(group))))
            
        
    return contours_grouped_sorted

# Subroutine 4
def filter_contours(contours, dataset, x_extent):
    """
    The contours are filtered in respect to the previoulsy defined parameter "x_extent".
    
    Input:
    -----
    contours: list
        List with entries each representing a single contour segment
        
    dataset: xarray
        xarray with two spatial dimensions but no time dimension
        
    x_extent: float, optional
        Set minimal extent of a contour in the x direction. A coverage of all longitudes means x_extent = 1.
    
    Returns:
    -------
    list: arrays
        List with entries each representing a single contour segment
    
    """
    lons = dataset[lon_name].values
    contour_expansion = [len(np.unique(np.round(contour[:,0]))) for contour in contours]
    test_expansion = [expansion/len(lons) >= x_extent for expansion in contour_expansion]
    
    return list(itertools.compress(contours, test_expansion))

# Subroutine 5
def single_contours(contours, dataset, x_extent):
    """
    Select largest contour fully encircling the pole. 
    
    Input:
    -----
    contours: list
        List with entries each representing a single contour segment
        
    dataset: xarray
        xarray with two spatial dimensions but no time dimension
        
    x_extent: float, optional
        Set minimal extent of a contour in the x direction. A coverage of all longitudes means x_extent = 1.
    
    Returns:
    -------
    list: arrays
        List with entries each representing a single contour segment
    
    """
    lons = dataset[lon_name].values
    contour_expansion = [len(np.unique(np.round(contour[:,0]))) for contour in contours]
    test_expansion = [expansion/len(lons) >= x_extent for expansion in contour_expansion]

    if sum([i==1 for i in contour_expansion])>1: 
        mean_lat = [np.mean(contour[:,1]) for contour in contours]
        contours_single = contours[mean_lat.index(min(mean_lat))]
    else: 
        contours_single = contours[test_expansion.index(max(test_expansion))]

    return contours_single

def df_contours(contours):
    """
    Store final contour in a pandas data frame
    
    Input:
    -----
    contours: list
        List with entries each representing a single contour segment
    
    Returns:
    -------
    dataframe: float
        Dataframe with columns lon and lat
    
    """
    if type(contours)=="list":
        temp = np.asarray(list(itertools.chain.from_iterable(contours)))
    else:
        temp = contours
    return pd.DataFrame({'lon': temp[:,0].tolist(), 'lat': temp[:,1].tolist()})

#%% calculation

# enter parameters
level = 5e-6
x_extent = 1 # in longitude percentage
y_overlap = 1 # in degrees
single = True
scale = True

contours = ex_contours(dataset[var_name], level)
contours_scaled = rescale_contours(contours, dataset)
# contours_grouped = group_contours(contours_scaled, y_overlap)
# contours_filtered = filter_contours(contours_grouped, dataset, x_extent)
# contours_single = single_contours(contours_filtered, dataset, x_extent)


#%% plotting
import seaborn as sns

data_crs = ccrs.PlateCarree()
proj = ccrs.PlateCarree()

fig, axes = plt.subplots(3,1, subplot_kw=dict(projection=proj), figsize=(17, 8))

pal_contours = "magma"
colors = sns.color_palette(pal_contours, n_colors=len(contours))

rgb_colors = ["#9CC7DF", "white"]

levels = [0,4e-6,1e-5]

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

line_wd = 2

plt.subplots_adjust(left  = 0.125,  # the left side of the subplots of the figure
                    right = 0.9,    # the right side of the subplots of the figure
                    bottom = 0.1,   # the bottom of the subplots of the figure
                    top = 0.9,      # the top of the subplots of the figure
                    wspace = 0.3,   # the amount of width reserved for blank space between subplots
                    hspace = 0.25    # the amount of height reserved for white space between subplots)
                   )

for ax in axes.flat:
    ax.add_feature(cfeature.COASTLINE)
    ax.set_extent([-180, 180, 40, 90], crs=data_crs)
    p0 = dataset[var_name].plot(ax=ax, cmap='jet' , add_colorbar=False,
                                levels = np.linspace(2e-6, 10e-6, 9), transform=data_crs, robust=True)

data = [contours, contours_scaled]#, contours_filtered, [contours_single]]
for ax,item in zip(axes.flat[1:],data):
    for contour, color in zip(item,colors):
        ax.plot(contour[:, 0], contour[:, 1],".", markersize=6,
                linewidth=line_wd, color = color, transform=data_crs)

# titles = ["Input field",
#           "Subroutine 1",
#           "Subroutine 2",
#           "Subroutine 3",
#           "Subroutine 4",
#           "Subroutine 5"]

# number = ["a)", "b)", "c)", "d)", "e)", "f)"]

# for ax,title,num in zip(axes.flat,titles, number):
#     ax.set_title("")
#     ax.set_title(title, fontweight='bold',fontsize=24, loc='center', y = 1.08)
#     ax.set_title(num, fontweight='bold',fontsize=24, loc='left', y = 1.08)
#     # ax.set_boundary(circle, transform=ax.transAxes)
#     ax.add_patch(mpatches.Circle((0.5, 0.5), radius=0.5, color='k', linewidth=5, fill=False, transform = ax.transAxes))
#     gr = ax.gridlines(draw_labels=True, color="black", linestyle="dotted", linewidth = 1.1)
#     gr.xlabel_style = {'size': 16, 'color': 'black', "rotation":0, "fontweight":"bold"}
#     gr.ylabel_style = {'size': 12, 'color': 'black'}

plt.show()



#%% simple example
import xarray as xr

ds = xr.open_dataset('./xcontour/Data/barotropic_vorticity.nc')

vor = ds.absolute_vorticity
vor[60:190,200:400] = np.nan


#%%
import numpy as np
from xcontour.xcontour import contour_length
from skimage import measure

def contour_lengths(data, contours, dims=[None,None], latlon=True, period=[None, None]):
    """Calculate contour length in a 2D numpy data.
    This is designed for xarray's ufunc.
    
    Parameters
    ----------
    data: numpy.ndarray
        2D numpy data.
    contours: numpy.ndarray
        a list of contour values.
    latlon: boolean, optional
        Whether dimension is latlon or cartesian.

    Returns
    -------
    lengths: numpy.ndarray
        List of contour lengths.
    """
    coord1 = dims[0]
    coord2 = dims[1]
    
    idx1 = np.arange(len(coord1))
    idx2 = np.arange(len(coord2))
    
    lengths = []
    
    for c in contours:
        # in unit of grid points
        segments = measure.find_contours(data, c)
        
        segs_coords = []
        
        # change to unit of coordinates
        for segment in segments:
            d1pos = np.interp(segment[:,0], idx1, coord1, period=period[0])
            d2pos = np.interp(segment[:,1], idx2, coord2, period=period[1])
            
            segs_coords.append(np.c_[d2pos, d1pos])
        
        lengths.append(sum([contour_length(seg) for seg in segs_coords]))
    
    return np.asarray(lengths)


lengths = contour_lengths(vor.data, np.array([0.00007]), dims=[vor.latitude, vor.longitude],
                          latlon=True, period=[360, None])

cs = xr.DataArray(np.array([0.00006, 0.00007]), dims='contour',
                  coords={'contour':np.array([0, 1])})

lens = xr.apply_ufunc(contour_lengths, vor, cs,
                      kwargs={'latlon':True, 'period':[None, 360], 'dims':[vor.latitude, vor.longitude],},
                      dask='allowed',
                      input_core_dims=[['latitude','longitude'], ['contour']],
                      vectorize=True,
                      output_core_dims=[['contour']])


# %%
import numpy as np
from xcontour.xcontour import contour_length, find_contour

cs = [-0.00006, -0.00007]

def c_len(c):
    c_segs = find_contour(vor, ['latitude', 'longitude'], c)
    return sum([contour_length(seg, latlon=False) for seg in c_segs])


lengths = [c_len(c) for c in cs]


#%%
import proplot as pplt

fontsize= 14

fig, axes = pplt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=0, sharey=0)

ax = axes[0]
ax.contourf(vor, levels=41)
ax.contour(vor, levels=cs, color='k')
ax.set_title('vorticity', fontsize=fontsize)

ax = axes[1]
ax.plot(lengths, cs)
ax.set_title('contour length', fontsize=fontsize)



#%% simple example
import xarray as xr

ds = xr.open_dataset('./xcontour/Data/internalwave.nc')

vor = ds.THETA.where(ds.THETA!=0)[2]


#%%
import numpy as np
from skimage import measure
from xcontour.xcontour import contour_length

def contour_lengths(data, contours, dims=[None,None], latlon=True, period=[None, None]):
    """Calculate contour length in a 2D numpy data.
    This is designed for xarray's ufunc.
    
    Parameters
    ----------
    data: numpy.ndarray
        2D numpy data.
    contours: numpy.ndarray
        a list of contour values.
    latlon: boolean, optional
        Whether dimension is latlon or cartesian.

    Returns
    -------
    lengths: numpy.ndarray
        List of contour lengths.
    """
    coord1 = dims[0]
    coord2 = dims[1]
    
    idx1 = np.arange(len(coord1))
    idx2 = np.arange(len(coord2))
    
    lengths = []
    
    for c in contours:
        # in unit of grid points
        segments = measure.find_contours(data, c)
        
        segs_coords = []
        
        # change to unit of coordinates
        for segment in segments:
            d1pos = np.interp(segment[:,0], idx1, coord1, period=period[0])
            d2pos = np.interp(segment[:,1], idx2, coord2, period=period[1])
            
            segs_coords.append(np.c_[d2pos, d1pos])
        
        lengths.append(sum([contour_length(seg) for seg in segs_coords]))
    
    return np.asarray(lengths)


lengths = contour_lengths(vor.data, np.array([26]), dims=[vor.XC, vor.Z],
                          latlon=False, period=[8960, None])

cs = xr.DataArray(np.array([25.9, 26.1]), dims='contour',
                  coords={'contour':np.array([0, 1])})

lens = xr.apply_ufunc(contour_lengths, vor, cs,
                      kwargs={'latlon':False, 'period':[8960, None], 'dims':[vor.Z, vor.XC],},
                      dask='allowed',
                      input_core_dims=[['Z','XC'], ['contour']],
                      vectorize=True,
                      output_core_dims=[['contour']])


#%%
import numpy as np
from xcontour.xcontour import contour_length, find_contour

cs = np.linspace(25.8, 26.2, 5)

def c_len(c):
    c_segs = find_contour(vor, ['Z', 'XC'], vor.max())
    
    if len(c_segs) == 0:
        return np.nan
    else:
        return sum([contour_length(seg, False) for seg in c_segs])


lengths = [c_len(c) for c in cs]


#%%
import proplot as pplt

segs = find_contour(vor, ['Z', 'XC'], vor.min())


fig, axes = pplt.subplots(nrows=1, ncols=1, figsize=(11, 6))

ax = axes[0]

ax.contourf(vor, cmap='bwr')
for seg in segs:
    ax.plot(seg[:, 0], seg[:, 1], linewidth=1, color='k')
ax.set_ylim([-200, 0])
ax.set_xlim([0, 8960])


#%%
import proplot as pplt

fontsize= 14

fig, axes = pplt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=0, sharey=0)

ax = axes[0]
ax.contourf(vor, levels=41)
ax.contour(vor, levels=cs, color='k')
ax.set_title('vorticity', fontsize=fontsize)

ax = axes[1]
ax.plot(lengths, cs)
ax.set_title('contour length', fontsize=fontsize)

