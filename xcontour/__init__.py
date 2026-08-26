# -*- coding: utf-8 -*-
"""
xcontour: diagnostic analyses and calculations in contour-based coordinates.

Built on xarray, this package provides adiabatic sorting, contour-enclosed
integrals, equivalent-length, and local wave activity / APE diagnostics
for quasi-conservative tracers in geophysical fluid dynamics.
"""
from .core import Contour2D, Table
from .utils import equivalent_latitudes, latitude_lengths_at,\
    cal_dA, \
    contour_area, contour_length


__version__ = "0.0.2"
