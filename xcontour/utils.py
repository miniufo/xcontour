# -*- coding: utf-8 -*-
'''
Created on 2020.02.04

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
'''
import numpy as np
import numba as nb
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
dimZList = ['lev', 'level', 'LEV', 'LEVEL', 'pressure', 'PRESSURE',
            'depth', 'DEPTH']



def add_latlon_metrics(dset, dims=None, boundary=None):
    """
    Infer 2D metrics (latitude/longitude) from gridded data file.

    Parameters
    ----------
    dset : xarray.Dataset
        A dataset open from a file
    dims : dict
        Dimension pair in a dict, e.g., {'Y':'latitude', 'X':'longitude'}
    boundary : dict
        Default boundary conditions applied to each coordinate

    Return
    -------
    dset : xarray.Dataset
        Input dataset with appropriated metrics added
    grid : xgcm.Grid
        The grid with appropriated metrics
    """
    lon, lat, lev = None, None, None
    
    if dims is None:
        for dim in dimXList:
            if dim in dset.dims:
                lon = dim
                break

        for dim in dimYList:
            if dim in dset.dims:
                lat = dim
                break

        for dim in dimZList:
            if dim in dset.dims:
                lev = dim
                break

        if lon is None or lat is None:
            raise Exception('unknown dimension names in dset, should be in '
                            + str(dimXList + dimYList))
    else:
        lon = dims['X'] if 'X' in dims else None
        lat = dims['Y'] if 'Y' in dims else None
        lev = dims['Z'] if 'Z' in dims else None
    
    if lev is None:
        ds = generate_grid_ds(dset, {'X':lon, 'Y':lat})
    else:
        ds = generate_grid_ds(dset, {'X':lon, 'Y':lat, 'Z':lev})
    
    coords = ds.coords
    
    BCx, BCy, BCz = 'extend', 'extend', 'extend'
    
    if boundary is not None:
        BCx = boundary['X'] if 'X' in boundary else 'extend'
        BCy = boundary['Y'] if 'Y' in boundary else 'extend'
        BCz = boundary['Z'] if 'Z' in boundary else 'extend'
    
    if __is_periodic(coords[lon], 360.0):
        periodic = 'X'
        
        if lev is None:
            grid = Grid(ds, periodic=[periodic], boundary={'Y': BCy})
        else:
            grid = Grid(ds, periodic=[periodic], boundary={'Z':BCz, 'Y': BCy})
    else:
        periodic = []
        
        if lev is None:
            grid = Grid(ds, periodic=False, boundary={'Y': BCy, 'X': BCx})
        else:
            grid = Grid(ds, periodic=False, boundary={'Z': BCz, 'Y': BCy, 'X': BCx})
    
    
    lonC = ds[lon]
    latC = ds[lat]
    lonG = ds[lon + '_left']
    latG = ds[lat + '_left']
    
    if 'X' in periodic:
        # dlonC = grid.diff(lonC, 'X', boundary_discontinuity=360)
        # dlonG = grid.diff(lonG, 'X', boundary_discontinuity=360)
        dlonC = grid.diff(lonC, 'X')
        dlonG = grid.diff(lonG, 'X')
    else:
        dlonC = grid.diff(lonC, 'X', boundary='extend')
        dlonG = grid.diff(lonG, 'X', boundary='extend')
    
    dlatC = grid.diff(latC, 'Y')
    dlatG = grid.diff(latG, 'Y')
    
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
    
    if lev is not None:
        levC = ds[lev].values
        tmp  = np.diff(levC)
        tmp  = np.concatenate([[(levC[0]-tmp[0])], levC])
        levG = tmp[:-1]
        delz = np.diff(tmp)
        
        ds[lev + '_left'] = levG
        coords['drF'] = xr.DataArray(delz, dims=lev, coords={lev: levC})
        coords['drG'] = xr.DataArray(np.concatenate([[delz[0]/2], delz[1:-1],
                                      [delz[-1]/2]]), dims=lev+'_left',
                                      coords={lev+'_left': levG})
        
        metrics={('X',    ): ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
                 ('Y' ,   ): ['dyG', 'dyF', 'dyC', 'dyU'], # Y distances
                 ('Z' ,   ): ['drG', 'drF'],               # Z distances
                 ('X', 'Y'): ['rAw', 'rAs', 'rA' , 'rAz']}
    else:
        metrics={('X',    ): ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
                 ('Y' ,   ): ['dyG', 'dyF', 'dyC', 'dyU'], # Y distances
                 ('X', 'Y'): ['rAw', 'rAs', 'rA' , 'rAz']}
    
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
    
    for key, value in metrics.items():
        grid.set_metrics(key, value)
    
    return ds, grid


def add_MITgcm_missing_metrics(dset, periodic=None, boundary=None, partial_cell=True):
    """
    Infer missing metrics from MITgcm output files.

    Parameters
    ----------
    dset: xarray.Dataset
        A dataset open from a file
    periodic: str
        Which coordinate is periodic
    boundary: dict
        Default boundary conditions applied to each coordinate
    partial_cell: bool
        Turn on the partial-cell or not (default is on).

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
        coords['drW'] = dset.hFacW * dset.drF if partial_cell else dset.drF
    if 'drS' not in coords: # vertical cell size at v point
        coords['drS'] = dset.hFacS * dset.drF if partial_cell else dset.drF
    if 'drC' not in coords: # vertical cell size at tracer point
        coords['drC'] = dset.hFacC * dset.drF if partial_cell else dset.drF
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
        coords['yA'] = dset.drF * dset.hFacC * dset.dxF if partial_cell \
                  else dset.drF * dset.dxF
    
    # Calculate vertical distances located on the cellboundary
    # ds.coords['dzC'] = grid.diff(ds.depth, 'Z', boundary='extrapolate')
    # Calculate vertical distances located on the cellcenter
    # ds.coords['dzT'] = grid.diff(ds.depth_left, 'Z', boundary='extrapolate')
    
    metrics = {
        ('X',)    : ['dxG', 'dxF', 'dxC', 'dxV'], # X distances
        # ('Y',)    : ['dyG', 'dyF', 'dyC', 'dyU'], # Y distances
        ('Z',)    : ['drW', 'drS', 'drC', 'drF', 'drG'], # Z distances
        # ('X', 'Y'): ['rAw', 'rAs', 'rA' , 'rAz'], # Areas in X-Y plane
        ('X', 'Z'): ['yA']} # Areas in X-Z plane
    
    for key, value in metrics.items():
        grid.set_metrics(key, value)
    
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


def contour_area(verts):
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


# @nb.jit(nopython=True, cache=False)
def contour_length(segments, xdef, ydef, latlon=True, disp=False):
    """Compute the length of a contour.
    
    Parameters
    ----------
    segments: numpy.array
        Segments of a single contour position returned by
        `measure.find_contours`.
    xdef : numpy.array
        X-coordinates
    ydef : numpy.array
        Y-coordinates
    latlon : boolean
        Is coordinates latlon in radian or cartesian
        
    Returns
    ----------
    Perimeter: float
        Perimeter of a contour.
    """
    yidx = np.arange(len(ydef))
    xidx = np.arange(len(xdef))
    
    total = 0
    
    if latlon:
        for segment in segments:
            dypos = np.interp(segment[:,0], yidx, ydef)
            dxpos = np.interp(segment[:,1], xidx, xdef)
            
            total = total + __segment_length_latlon(dxpos, dypos)
    else:
        for segment in segments:
            dypos = np.interp(segment[:,0], yidx, ydef)
            dxpos = np.interp(segment[:,1], xidx, xdef)
            
            total = total + __segment_length_cartesian(dxpos, dypos)
    
    if total == 0:
        return np.nan
    else:
        if latlon:
            return total * Rearth
        else:
            return total


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


# @nb.jit(nopython=True, cache=False)
def __segment_length_latlon(xpos, ypos):
    n = len(xpos)
    
    if n <= 1:
        return np.nan
    
    total = 0
    
    for i in range(n-1):
        total += __geodist(xpos[i], xpos[i+1], ypos[i], ypos[i+1])
    
    return total


@nb.jit(nopython=True, cache=False)
def __segment_length_cartesian(xpos, ypos):
    n = len(xpos)
    
    if n <= 1:
        return np.nan
    
    total = 0
    
    for i in range(n-1):
        total += np.hypot(xpos[i]-xpos[i+1], ypos[i]-ypos[i+1])
    
    return total


@nb.jit(nopython=True, cache=False)
def __geodist(lon1, lon2, lat1, lat2):
    """Calculate great-circle distance on a sphere.

    Parameters
    ----------
    lon1: float
        Longitude for point 1 in radian.
    lon2: float
        Longitude for point 2 in radian.
    lat1: float
        Latitude  for point 1 in radian.
    lat2: float
        Latitude  for point 2 in radian.

    Returns
    -------
    dis: float
        Great circle distance
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2.0 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2.0
    
    dis = 2.0 * np.arcsin(np.sqrt(a))
    
    return dis


'''
Testing codes for each class
'''
if __name__ == '__main__':
    print('start testing in ContourUtils.py')
    
    
    
