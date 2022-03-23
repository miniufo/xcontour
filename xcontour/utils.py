# -*- coding: utf-8 -*-
'''
Created on 2020.02.04

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
'''
import numpy as np
import xarray as xr
from xgcm import Grid
from xgcm.autogenerate import generate_grid_ds


'''
Here defines all the constants that are commonly used in earth sciences
'''
# Radius of the Earth (m)
Rearth = 6371200.0

# distance of unit degree at the equator
deg2m = 2.0 * np.pi * Rearth / 360.0

# Gravitational acceleration g (m s^-2)
g = 9.80665

# Rotating angular speed of the Earth (1)
omega = 7.292e-5



dimXList = ['lon', 'longitude', 'LON', 'LONGITUDE', 'geolon', 'GEOLON',
            'xt_ocean']
dimYList = ['lat', 'latitude' , 'LAT', 'LATITUDE' , 'geolat', 'GEOLAT',
            'yt_ocean']



def add_latlon_metrics(dset, dims=None, boundary=None):
    """
    Infer 2D metrics (latitude/longitude) from gridded data file.

    Parameters
    ----------
    dset : xarray.Dataset
        A dataset open from a file
    dims : dict
        Dimension pair in a dict, e.g., {'lat':'latitude', 'lon':'longitude'}
    boundary : dict
        Default boundary conditions applied to each coordinate

    Return
    -------
    dset : xarray.Dataset
        Input dataset with appropriated metrics added
    grid : xgcm.Grid
        The grid with appropriated metrics
    """
    lon, lat = None, None
    
    if dims is None:
        for dim in dimXList:
            if dim in dset or dim in dset.coords:
                lon = dim
                break

        for dim in dimYList:
            if dim in dset or dim in dset.coords:
                lat = dim
                break

        if lon is None or lat is None:
            raise Exception('unknown dimension names in dset, should be in '
                            + str(dimXList + dimYList))
    else:
        lon, lat = dims['lon'], dims['lat']
    
    ds = generate_grid_ds(dset, {'X':lon, 'Y':lat})
    
    coords = ds.coords
    
    if __is_periodic(coords[lon], 360.0):
        periodic = 'X'
        grid = Grid(ds, periodic=periodic, boundary={'Y': 'extend'})
    else:
        periodic = []
        grid = Grid(ds, boundary={'Y': 'extend', 'X': 'extend'})
    
    
    lonC = ds[lon]
    latC = ds[lat]
    lonG = ds[lon + '_left']
    latG = ds[lat + '_left']
    
    if 'X' in periodic:
        dlonC = grid.diff(lonC, 'X', boundary_discontinuity=360)
        dlonG = grid.diff(lonG, 'X', boundary_discontinuity=360)
    else:
        dlonC = grid.diff(lonC, 'X', boundary='extrapolate')
        dlonG = grid.diff(lonG, 'X', boundary='extrapolate')
    
    dlatC = grid.diff(latC, 'Y', boundary='extrapolate')
    dlatG = grid.diff(latG, 'Y', boundary='extrapolate')
    
    coords['dxG'], coords['dyG'] = __dll_dist(dlonG, dlatG, lonG, latG)
    coords['dxC'], coords['dyC'] = __dll_dist(dlonC, dlatC, lonC, latC)
    coords['dxF'] = grid.interp(coords['dxG'], 'Y')
    coords['dyF'] = grid.interp(coords['dyG'], 'X')
    coords['dxV'] = grid.interp(coords['dxG'], 'X')
    coords['dyU'] = grid.interp(coords['dyG'], 'Y')
    
    coords['rA' ] = ds['dyF'] * ds['dxF']
    coords['rAw'] = ds['dyG'] * ds['dxC']
    coords['rAs'] = ds['dyC'] * ds['dxG']
    coords['rAz'] = ds['dyU'] * ds['dxV']
    
    # print('lonC', lonC.dims)
    # print('latC', latC.dims)
    # print('lonG', lonG.dims)
    # print('latG', latG.dims)
    # print('')
    # print('dlonC', dlonC.dims)
    # print('dlatC', dlatC.dims)
    # print('dlonG', dlonG.dims)
    # print('dlatG', dlatG.dims)
    # print('')
    # print('dxG', coords['dxG'].dims)
    # print('dyG', coords['dyG'].dims)
    # print('dxF', coords['dxF'].dims)
    # print('dyF', coords['dyF'].dims)
    # print('dxC', coords['dxC'].dims)
    # print('dyC', coords['dyC'].dims)
    # print('dxV', coords['dxV'].dims)
    # print('dyU', coords['dyU'].dims)
    # print('')
    # print('rA' , coords['rA' ].dims)
    # print('rAz', coords['rAz'].dims)
    # print('rAw', coords['rAw'].dims)
    # print('rAs', coords['rAs'].dims)
    
    # if 'X' in periodic:
    #     coords['rAz'] = grid.interp(grid.interp(coords['rAc'], 'X',
    #                                             boundary_discontinuity=360),
    #                                 'Y', boundary='fill', fill_value=na)
    # else:
    #     coords['rAz'] = grid.interp(grid.interp(coords['rAc'], 'X',
    #                                             boundary='fill', fill_value=na),
    #                                 'Y', boundary='fill', fill_value=na)
    # print(coords['rAc'])
    # print(coords['rAz'])
    
    metrics={('X',    ): ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
             ('Y' ,   ): ['dyG', 'dyF', 'dyC', 'dyU'], # Y distances
             ('X', 'Y'): ['rAw', 'rAs', 'rA' , 'rAz']}
    
    grid._assign_metrics(metrics)
    
    return ds, grid


def add_MITgcm_missing_metrics(dset, periodic=None, boundary=None):
    """
    Infer missing metrics from MITgcm output files.

    Parameters
    ----------
    dset : xarray.Dataset
        A dataset open from a file
    periodic : str
        Which coordinate is periodic
    boundary : dict
        Default boundary conditions applied to each coordinate

    Return
    -------
    dset : xarray.Dataset
        Input dataset with appropriated metrics added
    grid : xgcm.Grid
        The grid with appropriated metrics
    """
    coords = dset.coords
    grid   = Grid(dset, periodic=periodic, boundary=boundary)
    
    if 'drW' not in coords: # vertical cell size at u point
        coords['drW'] = dset.hFacW * dset.drF
    if 'drS' not in coords: # vertical cell size at v point
        coords['drS'] = dset.hFacS * dset.drF
    if 'drC' not in coords: # vertical cell size at tracer point
        coords['drC'] = dset.hFacC * dset.drF
    if 'drG' not in coords: # vertical cell size at tracer point
        coords['drG'] = dset.Zl - dset.Zl + dset.drC.values[:-1]
        # coords['drG'] = xr.DataArray(dset.drC[:-1].values, dims='Zl',
        #                              coords={'Zl':dset.Zl.values})
    
    if 'dxF' not in coords:
        coords['dxF'] = grid.interp(dset.dxC, 'X')
    if 'dyF' not in coords:
        coords['dyF'] = grid.interp(dset.dyC, 'Y')
    if 'dxV' not in coords:
        coords['dxV'] = grid.interp(dset.dxG, 'X')
    if 'dyU' not in coords:
        coords['dyU'] = grid.interp(dset.dyG, 'Y')
    
    if 'hFacZ' not in coords:
        coords['hFacZ'] = grid.interp(dset.hFacS, 'X')
    if 'maskZ' not in coords:
        coords['maskZ'] = coords['hFacZ']
        
    if 'yA' not in coords:
        coords['yA'] = dset.drF * dset.hFacC * dset.dxF
    
    # Calculate vertical distances located on the cellboundary
    # ds.coords['dzC'] = grid.diff(ds.depth, 'Z', boundary='extrapolate')
    # Calculate vertical distances located on the cellcenter
    # ds.coords['dzT'] = grid.diff(ds.depth_left, 'Z', boundary='extrapolate')
    
    metrics = {
        ('X',)    : ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
        ('Y',)    : ['dyG', 'dyF', 'dyC', 'dyU'], # Y distances
        ('Z',)    : ['drW', 'drS', 'drC', 'drF', 'drG'], # Z distances
        ('X', 'Y'): ['rAw', 'rAs', 'rA' , 'rAz'], # Areas in X-Y plane
        ('X', 'Z'): ['yA']} # Areas in X-Z plane
    
    grid._assign_metrics(metrics)
    
    return dset, grid


def equivalent_latitudes(areas):
    """
    Calculate equivalent latitude using the formular:
        2 * pi * a^2 * [sin(latEq) + sin(90)] = area.
    This is similar to a EqY(A) table.

    Parameters
    ----------
    areas : xarray.DataArray
        Contour-enclosed areas.
    
    Returns
    ----------
    latEq : xarray.DataArray
        The equivalent latitudes.
    """
    ratio = areas/2.0/np.pi/Rearth/Rearth - 1.0

    # clip ratio within [-1, 1]
    ratio = xr.where(ratio<-1, -1.0, ratio)
    ratio = xr.where(ratio> 1,  1.0, ratio)

    latEq = np.rad2deg(np.arcsin(ratio)).astype(areas.dtype)

    return latEq


def latitude_lengths_at(lats):
    """
    Calculate minimum length on a sphere given latitudes.

    Parameters
    ----------
    latEq : xarray.DataArray
        Equivalent latitude.
    
    Returns
    ----------
    Lmin : xarray.DataArray
        The minimum possible length of the contour.
    """
    Lmin = (2.0 * np.pi * Rearth * np.cos(np.deg2rad(lats))).astype(lats.dtype)

    return Lmin


def contour_enclosed_area_np(verts):
    """
    Compute the area enclosed by a contour.  Copied from
    https://github.com/rabernat/floater/blob/master/floater/rclv.py
    
    Parameters
    ----------
    verts : array_like
        2D shape (N,2) array of vertices. Uses scikit image convetions
        (j,i indexing)
        
    Returns
    ----------
    area : float
        Area of polygon enclosed by verts. Sign is determined by vertex
        order (cc vs ccw)
    """
    verts_roll = np.roll(verts, 1, axis=0)
    
    # use scikit image convetions (j,i indexing)
    area_elements = ((verts_roll[:,1] + verts[:,1]) *
                     (verts_roll[:,0] - verts[:,0]))
    
    # absolute value makes results independent of orientation
    return abs(area_elements.sum())/2.0


def contour_enclosed_area_py(verts):
    """
    Compute the area enclosed by a contour.  Copied from
    https://arachnoid.com/area_irregular_polygon/
    
    Parameters
    ----------
    verts : array_like
        2D shape (N,2) array of vertices. Uses scikit image convetions
        (j,i indexing)
        
    Returns
    ----------
    area : float
        Area of polygon enclosed by verts. Sign is determined by vertex
        order (cc vs ccw)
    """
    a = 0
    ox, oy = verts[0]

    for x, y in verts[1:]:
        a += (x * oy - y * ox)
        ox, oy = x, y

    return a / 2


def contour_length_np(verts):
    """
    Compute the length of a contour.  Copied from
    https://github.com/rabernat/floater/blob/master/floater/rclv.py
    
    Parameters
    ----------
    verts : array_like
        2D shape (N,2) array of vertices. Uses scikit image convetions
        (j,i indexing)
        
    Returns
    ----------
    Perimeter : float
        Perimeter of a contour by verts.
    """
    verts_roll = np.roll(verts, 1, axis=0)
    
    diff = verts_roll - verts
    
    p = np.sum(np.hypot(diff[:,0], diff[:,1]))
    
    return p


def contour_length_py(verts):
    """
    Compute the length of a contour.  Copied from
    https://arachnoid.com/area_irregular_polygon/
    
    Parameters
    ----------
    verts : array_like
        2D shape (N,2) array of vertices. Uses scikit image convetions
        (j,i indexing)
        
    Returns
    ----------
    Perimeter : float
        Perimeter of a contour by verts.
    """
    p = 0
    ox, oy = verts[0]
    
    for x, y in verts[1:]:
        p += abs((x - ox) + (y - oy) * 1j)
        ox, oy = x, y
    
    return p


def is_contour_closed(con):
    """
    Whether the contour is a closed one or intersect with boundaries.
    """
    return np.all(con[0] == con[-1])



"""
Helper (private) methods are defined below
"""
def __dll_dist(dlon, dlat, lon, lat):
    """
    Converts lat/lon differentials into distances in meters.

    Parameters
    ----------
    dlon : xarray.DataArray
        longitude differentials
    dlat : xarray.DataArray
        latitude differentials
    lon  : xarray.DataArray
        longitude values
    lat  : xarray.DataArray
        latitude values

    Return
    -------
    dx  : xarray.DataArray
        Distance inferred from dlon
    dy  : xarray.DataArray
        Distance inferred from dlat
    """
    dx = np.cos(np.deg2rad(lat)) * dlon * deg2m
    dy = (dlat + lon - lon) * deg2m
    
    # cos(+/-90) is not exactly zero, add a threshold
    dx = xr.where(dx<1e-15, 0, dx)
    
    return dx, dy

def __is_periodic(coord, period):
    """
    Whether a given coordinate array is periodic.

    Parameters
    ----------
    coord  : xarray.DataArray
        A given coordinate e.g., longitude
    period : float
        Period used to justify the coordinate, e.g., 360 for longitude
    """
    # assume it is linear increasing
    if coord.size == 1:
        return False

    delta = coord[1] - coord[0]
	
    start = coord[-1] + delta - period;
		
    if np.abs((start - coord[0]) / delta) > 1e-4:
        return False;
		
    return True



'''
Testing codes for each class
'''
if __name__ == '__main__':
    print('start testing in ContourUtils.py')
    
    
    
