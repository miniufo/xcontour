# -*- coding: utf-8 -*-
"""
Smoke tests for xcontour: verify package import and basic Contour2D/Table
functionality using synthetic data.

These tests do NOT require local Data/ files, GeoApps, or xinvert — they
are designed to run on GitHub CI with only pip-installed dependencies.
"""
import numpy as np
import xarray as xr
import pytest

from xcontour import Contour2D, Table
from xcontour import equivalent_latitudes, latitude_lengths_at, cal_dA
from xcontour import contour_area, contour_length
from xcontour.utils import Rearth, g, omega, deg2m


# ---------------------------------------------------------------------------
# Fixtures: synthetic 2D tracer and area on a small lat/lon grid
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    """A 10x10 lat/lon grid with a simple tracer field."""
    lats = xr.DataArray(np.linspace(-80, 80, 10), dims=['lat'], name='lat')
    lons = xr.DataArray(np.linspace(0, 360, 10)[:-1], dims=['lon'], name='lon')

    # tracer: potential temperature-like field increasing with lat
    tracer = 200 + 50 * np.sin(np.deg2rad(lats)) + 0
    tracer = tracer * xr.ones_like(lons)
    tracer.name = 'theta'
    return tracer, lats, lons


@pytest.fixture
def small_dA(small_grid):
    """Area element for each grid point."""
    _, lats, _ = small_grid
    return cal_dA(lats=lats, lats_dim=lats) if False else None  # placeholder


@pytest.fixture
def simple_tracer():
    """A minimal 2D tracer with uniform area — enough for Contour2D init."""
    nlat, nlon = 17, 18
    lats = xr.DataArray(np.linspace(-80, 80, nlat), dims=['lat'], name='lat')
    lons = xr.DataArray(np.linspace(0, 360, nlon + 1)[:-1], dims=['lon'], name='lon')

    dA = xr.DataArray(
        np.ones((nlat, nlon), dtype=np.float32),
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons},
        name='dA'
    )

    tracer = xr.DataArray(
        200 + 50 * np.sin(np.deg2rad(lats)).values[:, None] * np.ones(nlon),
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons},
        name='theta'
    )

    return tracer, dA


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImport:
    """Verify that the public API can be imported."""

    def test_import_contour2d(self):
        assert Contour2D is not None

    def test_import_table(self):
        assert Table is not None

    def test_import_utils(self):
        assert Rearth > 0
        assert g > 0
        assert omega > 0

    def test_deg2m(self):
        expected = 2 * np.pi * Rearth / 360.0
        assert abs(deg2m() - expected) < 1e-6


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_equivalent_latitudes(self):
        """Equivalent latitude formula: area = 2*pi*a^2*(sin(latEq)+1)."""
        # 0 area → latEq=-90; half-sphere area (2*pi*a^2) → latEq=0;
        # full-sphere area (4*pi*a^2) → latEq=+90
        areas = xr.DataArray(
            np.array([0.0, 1.0, 2.0]) * 2 * np.pi * Rearth**2,
            dims=['contour']
        )
        lats = equivalent_latitudes(areas)
        assert lats.shape == (3,)
        assert abs(lats.values[0]  - (-90.0)) < 1.0  # zero area → south pole
        assert abs(lats.values[1]  - (  0.0)) < 1.0  # half sphere → equator
        assert abs(lats.values[-1] - ( 90.0)) < 1.0  # full sphere → north pole

    def test_latitude_lengths_at(self):
        """Minimum contour length on a sphere given latitudes."""
        lats = xr.DataArray([0.0, 30.0, 60.0], dims=['contour'])
        Lmin = latitude_lengths_at(lats)
        assert Lmin.shape == (3,)
        # at equator, Lmin = 2*pi*R (full circumference)
        assert abs(Lmin.values[0] - (2 * np.pi * Rearth)) < 1e3
        # at poles, Lmin → 0
        assert Lmin.values[2] < Lmin.values[0]


# ---------------------------------------------------------------------------
# Contour2D tests
# ---------------------------------------------------------------------------

class TestContour2D:
    def test_init(self, simple_tracer):
        """Contour2D can be constructed with a 2D tracer and area."""
        tracer, dA = simple_tracer
        c2d = Contour2D(tracer, dA, dims={'lat': 'lat', 'lon': 'lon'},
                        dimEq='lat', increase=True)
        assert c2d is not None
        assert c2d.tracer is tracer
        assert c2d.dA is dA
        assert c2d.arakawa == 'A'

    def test_init_invalid_dims(self, simple_tracer):
        """Contour2D raises if dims is not 2D."""
        tracer, dA = simple_tracer
        with pytest.raises(Exception):
            Contour2D(tracer, dA, dims={'lat': 'lat'}, dimEq='lat')

    def test_cal_contours(self, simple_tracer):
        """cal_contours returns a DataArray with expected contour dimension."""
        tracer, dA = simple_tracer
        c2d = Contour2D(tracer, dA, dims={'lat': 'lat', 'lon': 'lon'},
                        dimEq='lat', increase=True)
        ctr = c2d.cal_contours(levels=5)
        assert 'contour' in ctr.dims or 'contour' in ctr.coords
        # should have 5 contour levels
        assert ctr.sizes.get('contour', 0) == 5 or len(ctr) == 5


# ---------------------------------------------------------------------------
# Table tests
# ---------------------------------------------------------------------------

class TestTable:
    def test_lookup_roundtrip(self):
        """Table can be constructed and its internal state is consistent."""
        # build a monotonic table: area as a function of latitude
        lats = xr.DataArray(np.linspace(-89, 89, 10), dims=['latEq'])
        areas = (np.sin(np.deg2rad(lats)) + 1.0) * np.pi * Rearth**2
        areas = areas.rename('area')
        areas = areas.assign_coords({'latEq': lats})

        tbl = Table(areas, dimEq='latEq')

        # table stores the area and coordinate correctly
        assert tbl._dimEq == 'latEq'
        assert tbl._table.shape == (10,)
        # area is increasing (sin(lat)+1 is monotonic for -89..89)
        assert tbl._incVl == True

        # lookup_coordinates with a scalar area value (no dims)
        # finds the corresponding latitude
        mid_area = float(areas.values[5])  # area near equator
        result = tbl.lookup_coordinates(
            xr.DataArray(mid_area, dims=[], coords={})
        )
        # result should be close to lats[5] ≈ 9.89
        assert abs(float(result) - float(lats.values[5])) < 5.0
