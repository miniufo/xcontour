# -*- coding: utf-8 -*-
"""
Created on 2020.02.05

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram


class Contour2D(object):
    """
    This class is designed for performing the 2D contour analysis.
    """
    def __init__(self, grid, trcr, dims, dimEq, arakawa='A',
                 increase=True, lt=False, check_mono=False, dtype=np.float32):
        """
        Construct a Dynamics instance using a Dataset and a tracer

        Parameters
        ----------
        grid : xgcm.Grid
            a given grid that accounted for grid metrics
        trcr : xarray.DataArray
            A given tracer on the given grid
        dims : dict
            Dimensions along which the min/max values are defined and then
            mapped to the contour space.  Example:
                dims = {'X': 'lon', 'Y': 'lat', 'Z': 'Z'}
            Note that only 2D (e.g., X-Y horizontal or X-Z, Y-Z vertical planes)
            is allowed for this class.
        dimEq : dict
            Equivalent dimension that should be mapped from contour space.
            Example: dimEq = {'Y': 'lat'} or dimEq = {'Z', 'depth'}
        arakawa : str
            The type of the grid in ['A', 'C']. Reference:
                https://db0nus869y26v.cloudfront.net/en/Arakawa_grids
            Others are not well tested.
        increase : bool
            Contour increase with the index of equivalent dimension or not
            after the sorting.
        lt : bool
            If true, less than a contour is defined as inside the contour.
        check_mono: bool
            Check the monotonicity of the result or not (default: False).
        """

        if len(dimEq) != 1:
            raise Exception('dimEq should be one dimension e.g., {"Y","lat"}')

        if len(dims) != 2:
            raise Exception('dims should be a 2D plane')

        self.grid    = grid
        self.arakawa = arakawa
        self.tracer  = trcr
        self.dimNs   = list(dims.keys())      # dim names,  ['X', 'Y', 'Z']
        self.dimVs   = list(dims.values())    # dim values, ['lon', 'lat', 'Z']
        self.dimEqN  = list(dimEq.keys())[0]  # equiv. dim name
        self.dimEqV  = list(dimEq.values())[0]# equiv. dim value
        self.lt      = lt
        self.dtype   = dtype
        self.check_mono = check_mono
        self.increase   = increase


    def cal_area_eqCoord_table(self, mask):
        """
        Calculate the discretized relation table between area and equivalent
        coordinate.  Sometimes, this can be done analytically but mostly it is
        done numerically when dealing with an arbitarily-shaped domain.
        
        Note: it is assumed that the mask does not change with time.
        
        Since here we use conditional integration of xarray, broadcasting the
        arrays could be memory consuming.  So this is used for small dataset or
        validation only.

        Parameters
        ----------
        mask : xarray.DataArray
            A boolean mask, 1 if valid data and 0 if topography.

        Returns
        ----------
        tbl : xarray.DataArray
            The relation table between area and equivalent coordinate.  This
            table will be used to represent the relation of A(Yeq) or its
            inverse relation Yeq(A), if equivalent dimension is Yeq.
        """
        ctr = mask[self.dimEqV].copy().rename({self.dimEqV:'contour'}) \
                                      .rename('contour')
        ctrVar, _ = xr.broadcast(mask[self.dimEqV], mask)
        
        eqDimIncre = ctr[-1] > ctr[0]
        
        if self.lt:
            if eqDimIncre == self.increase:
                # if self.increase:
                #     print('case 1: increase & lt')
                # else:
                #     print('case 1: decrease & lt')
                mskVar = mask.where(ctrVar < ctr)
            else:
                # if self.increase:
                #     print('case 2: increase & lt')
                # else:
                #     print('case 2: decrease & lt')
                mskVar = mask.where(ctrVar > ctr)
        else:
            if eqDimIncre == self.increase:
                # if self.increase:
                #     print('case 3: increase & gt')
                # else:
                #     print('case 3: decrease & gt')
                mskVar = mask.where(ctrVar > ctr)
            else:
                # if self.increase:
                #     print('case 4: increase & gt')
                # else:
                #     print('case 4: decrease & gt')
                mskVar = mask.where(ctrVar < ctr)
        
        tbl = abs(self.grid.integrate(mskVar, self.dimNs).rename('AeqCTbl')) \
                    .rename({'contour':self.dimEqV}).squeeze().load()
        
        maxArea = abs(self.grid.integrate(mask, self.dimNs)).load().squeeze()
        
        # assign the maxArea to the endpoint
        tmp = tbl[{self.dimEqV:-1}] > tbl[{self.dimEqV:0}]
        if   (tmp == True ).all():
            tbl[{self.dimEqV:-1}] = maxArea
        elif (tmp == False).all():
            tbl[{self.dimEqV: 0}] = maxArea
        else:
            raise Exception('not every time or level is increasing/decreasing')
        
        if self.check_mono:
            _check_monotonicity(tbl, 'contour')
        
        return Table(tbl, self.dimEqV)

    def cal_area_eqCoord_table_hist(self, mask):
        """
        Calculate the discretized relation table between area and equivalent
        coordinate.  Sometimes, this can be done analytically but mostly it is
        done numerically when dealing with an arbitarily-shaped domain.
        
        Note: it is assumed that the mask does not change with time.
        
        Since the implementation based on xarray could be memory consuming,
        we here implement this function using xhistogram, which fast and
        memory-friendly.  This is why this function is end with '_hist'

        Parameters
        ----------
        mask : xarray.DataArray
            A boolean mask, 1 if valid data and 0 if topography.

        Returns
        ----------
        tbl : xarray.DataArray
            The relation table between area and equivalent coordinate.  This
            table will be used to represent the relation of A(Yeq) or its
            inverse relation Yeq(A), if equivalent dimension is Yeq.
        """
        ctr = mask[self.dimEqV].copy().rename({self.dimEqV:'contour'}) \
                                      .rename('contour')
        ctrVar, _ = xr.broadcast(mask[self.dimEqV], mask)
        
        ctrVar = ctrVar.where(mask==1)
        
        increSame = True
        if (ctr.values[-1] > ctr.values[0]) != self.increase:
            increSame = False
        
        tbl = _histogram(ctrVar, ctr, self.dimVs,
                         self.grid.get_metric(ctrVar, self.dimNs), # weights
                         increSame == self.lt # less than or greater than
                         ).rename('AeqCTbl').rename({'contour':self.dimEqV})\
                          .squeeze().load()
        
        if increSame:
            tbl = tbl.assign_coords({self.dimEqV:ctr.values}).squeeze()
        else:
            tbl = tbl.assign_coords({self.dimEqV:ctr.values[::-1]}).squeeze()
        
        if self.check_mono:
            _check_monotonicity(tbl, 'contour')
        
        return Table(tbl, self.dimEqV)

    def cal_contours(self, levels=10):
        """
        Establishing contour levels (space) of the tracer from its minimum
        to maximum values.  If a integer is specified, equally-spaced contour
        interval is used.  Otherwise, each level should be provided in an
        array.

        Parameters
        ----------
        levels : int or numpy.array
            The number of contour levels or specified levels.

        Returns
        ----------
        contour : xarray.DataArray
            A array of contour levels.
        """
        if type(levels) is int:
            # specifying number of contours
            mmin = self.tracer.min(dim=self.dimVs)
            mmax = self.tracer.max(dim=self.dimVs)

            # if numpy.__version__ > 1.16, use numpy.linspace instead
            def mylinspace(start, stop, levels):
                divisor = levels - 1
                steps   = (1.0/divisor) * (stop - start)
    
                return steps[..., None] * np.arange(levels) + start[..., None]

            if self.increase:
                start = mmin
                end   = mmax
            else:
                start = mmax
                end   = mmin
                
            ctr = xr.apply_ufunc(mylinspace, start, end, levels,
                                 dask='allowed',
                                 input_core_dims=[[], [], []],
                                 vectorize=True,
                                 output_core_dims=[['contour']],
                                 output_dtypes=[self.dtype])

            ctr.coords['contour'] = np.linspace(0.0, levels-1.0, levels,
                                                dtype=self.dtype)
            
        else:
             # specifying levels of contours
            def mylinspace(tracer, levs):
                return tracer[..., None] - tracer[..., None] + levs

            ctr = xr.apply_ufunc(mylinspace,
                                 self.tracer.min(dim=self.dimVs), levels,
                                 dask='allowed',
                                 input_core_dims=[[], []],
                                 vectorize=True,
                                 output_core_dims=[['contour']])

            ctr.coords['contour'] = levels

        return ctr


    def cal_contours_at(self, predef, table):
        """
        Calculate contours for a tracer at prescribed Ys (equivalent Ys),
        so that the returned contours are defined almost at Ys.

        This function will first rough estimate the contour-enclosed
        area and equivalent Ys, and then interpolate the Y(q) relation
        table to get the q(Y) and return q.
        
        Since here we use conditional integration of xarray, broadcasting the
        arrays could be memory consuming.  So this is used for small dataset or
        validation only.

        Parameters
        ----------
        predef : xarray.DataArray or numpy.ndarray or numpy.array
            An 1D array of prescribed coordinate values.
        table : Table
            Relation A(Yeq) table between area and equivalent dimension.

        Returns
        ----------
        contour : xarray.DataArray
            A array of contour levels corresponding to preY.
        """
        if len(predef.shape) != 1:
            raise Exception('predef should be a 1D array')

        if type(predef) in [np.ndarray, np.array]:
            # add coordinate as a DataArray
            predef = xr.DataArray(predef, dims='new', coords={'new': predef})

        N = predef.size
        
        ctr   = self.cal_contours(N)
        area  = self.cal_integral_within_contours(ctr)
        dimEq = table.lookup_coordinates(area).rename('Z')
        # print(self.interp_to_coords(predef, dimEq, ctr))
        qIntp = self.interp_to_coords(predef.squeeze(), dimEq, ctr.squeeze()) \
                    .rename({'new': 'contour'}) \
                    .rename(ctr.name)

        qIntp['contour'] = np.linspace(0, N-1, N, dtype=self.dtype)

        return qIntp


    def cal_contours_at_hist(self, predef, table):
        """
        Calculate contours for a tracer at prescribed Ys,
        so that the returned contour and its enclosed area will give a
        monotonic increasing/decreasing results.

        This function will first rough estimate the contour-enclosed
        area and equivalent Ys, and then interpolate the Y(q) relation
        table to get the q(Y) and return q.
        
        Since the implementation based on xarray could be memory consuming,
        we here implement this function using xhistogram, which fast and
        memory-friendly.  This is why this function is end with '_hist'

        Parameters
        ----------
        predef : xarray.DataArray or numpy.ndarray or numpy.array
            An 1D array of prescribed coordinate values.
        table : Table
            A(dimEq) table.

        Returns
        ----------
        contour : xarray.DataArray
            A array of contour levels corresponding to preY.
        """
        if len(predef.shape) != 1:
            raise Exception('predef should be a 1D array')

        if type(predef) in [np.ndarray, np.array]:
            # add coordinate as a DataArray
            predef = xr.DataArray(predef, dims='new', coords={'new': predef})

        N = predef.size
        
        ctr   = self.cal_contours(N)
        area  = self.cal_integral_within_contours_hist(ctr)
        dimEq = table.lookup_coordinates(area).rename('Z')
        qIntp = self.interp_to_coords(predef.squeeze(), dimEq, ctr.squeeze()) \
                    .rename({'new': 'contour'}) \
                    .rename(ctr.name)

        qIntp['contour'] = np.linspace(0, N-1, N, dtype=self.dtype)

        return qIntp


    def cal_integral_within_contours(self, contour, tracer=None, integrand=None):
        """
        Calculate conditional integral of a (masked) variable within each
        pre-calculated tracer contour.
        
        Since here we use conditional integration of xarray, broadcasting the
        arrays could be memory consuming.  So this is used for small dataset or
        validation only.

        Parameters
        ----------
        contour: xarray.DataArray
            A given contour levels.
        integrand: xarray.DataArray
            A given variable in dset.  If None, area enclosed by contour
            will be calculated and returned

        Returns
        ----------
        intVar : xarray.DataArray
            The integral of var inside contour.  If None, area enclosed by
            contour will be calculated and returned
        """
        if type(contour) in [np.ndarray, np.array]:
            # add coordinate as a DataArray
            contour = xr.DataArray(contour, dims='contour',
                                   coords={'contour': contour})
        
        if tracer is None:
            tracer = self.tracer
        
        # this allocates large memory, xhistogram works better
        if integrand is None:
            integrand = tracer - tracer + 1

        if self.lt: # this allocates large memory, xhistogram works better
            mskVar = integrand.where(tracer < contour)
        else:
            mskVar = integrand.where(tracer > contour)
        
        # conditional integrate (not memory-friendly because of broadcasting)
        intVar = self.grid.integrate(mskVar, self.dimNs)
        
        if self.check_mono:
            _check_monotonicity(intVar, 'contour')

        return intVar


    def cal_integral_within_contours_hist(self, contour, tracer=None,
                                          integrand=None):
        """
        Calculate integral of a masked variable within
        pre-calculated tracer contours, using histogram method.
        
        Since the implementation based on xarray could be memory consuming,
        we here implement this function using xhistogram, which fast and
        memory-friendly.  This is why this function is end with '_hist'

        Parameters
        ----------
        contour : xarray.DataArray
            A given contour levels.
        tracer : xarray.DataArray
            A given tracer to replace self.tracer.  This is somewhat
            redundant but useful in some cases if we need to change the tracer.
        integrand: xarray.DataArray
            A given variable.  If None, area enclosed by contour
            will be calculated and returned

        Returns
        ----------
        intVar : xarray.DataArray
            The integral of var inside contour.  If None, area enclosed by
            contour will be calculated and returned
        """
        if tracer is None:
            tracer = self.tracer
        
        # weights are the metrics multiplied by integrand
        if integrand is not None: 
            wei = self.grid.get_metric(tracer, self.dimNs) * integrand
        else:
            wei = self.grid.get_metric(tracer, self.dimNs)
        
        # replacing nan with 0 in weights, as weights cannot have nan
        wei = wei.fillna(0.)
        
        CDF = _histogram(tracer, contour, self.dimVs, wei, self.lt)
        
        # ensure that the contour index is increasing
        if CDF['contour'][-1] < CDF['contour'][0]:
            CDF = CDF.isel({'contour':slice(None, None, -1)})
        
        if self.check_mono:
            _check_monotonicity(CDF, 'contour')
        
        return CDF


    def cal_gradient_wrt_area(self, var, area):
        """
        Calculate gradient with respect to area.

        Parameters
        ----------
        var  : xarray.DataArray
            A variable that need to be differentiated.
        area : xarray.DataArray
            Area enclosed by contour levels.
        
        Returns
        ----------
        dVardA : xarray.DataArray
            The derivative of var w.r.t contour-encloesd area.
        """
        # centered difference rather than neighboring difference (diff)
        dfVar  =  var.differentiate('contour')
        dfArea = area.differentiate('contour')
        
        dVardA = dfVar / dfArea
        
        if var.name is None:
            return dVardA.rename('dvardA')
        else:
            return dVardA.rename('d'+var.name+'dA')
    

    def cal_along_contour_mean(self, contour, integrand, area=None):
        """
        Calculate along-contour average.

        Parameters
        ----------
        contour: xarray.DataArray
            A given contour levels.
        integrand: xarray.DataArray
            A given integrand to be averaged.
        
        Returns
        ----------
        lm : xarray.DataArray
            Along-contour (Lagrangian) mean of the integrand.
        """
        intA = self.cal_integral_within_contours(contour, integrand=integrand)
        
        if area is None:
            area = self.cal_integral_within_contours(contour)
        
        lmA  = self.cal_gradient_wrt_area(intA, area)
        
        if integrand.name is None:
            return lmA.rename('lm')
        else:
            return lmA.rename('lm'+integrand.name)
    
    
    def cal_sqared_equivalent_length(self, dgrdSdA, dqdA):
        """
        Calculate squared equivalent length.

        Parameters
        ----------
        dgrdSdA : xarray.DataArray
            d [Integrated |grd(q)|^2] / dA.
        dqdA : xarray.DataArray
            d [q] / dA.
        
        Returns
        ----------
        Leq2 : xarray.DataArray
            The squared equivalent length.
        """
        Leq2  = (dgrdSdA / dqdA ** 2).rename('Leq2')

        return Leq2


    def cal_local_wave_activity(self, q, Q, mask_idx=None, part='all'):
        """
        Calculate local finite-amplitude wave activity density.
        Reference: Huang and Nakamura 2016, JAS

        Parameters
        ----------
        q: xarray.DataArray
            A tracer field.
        Q: xarray.DataArray
            The sorted tracer field along the equivalent dimension.
        mask_idx: list of int
            Return masks at the indices of equivalent dimension.
        part: str
            The parts over which the integration is taken.  Available options
            are ['all', 'upper', 'lower'], corresponding to all, W+, and W-
            regions defined in Huang and Nakamura (2016, JAS)
        
        Returns
        ----------
        lwa : xarray.DataArray
            Local finite-amplitude wave activity, corresponding to part.
        contours : list
            A list of Q-contour corresponding to mask_idx.
        masks : list
            A list of mask corresponding to mask_idx.
        """
        wei  = self.grid.get_metric(q, self.dimNs).squeeze()
        wei  = wei / wei.max() # normalize between [0-1], similar to cos(lat)
        part = part.lower()
        # q2 = q.squeeze()
        
        eqDim = q[self.dimEqV]
        eqDimLen = len(eqDim)
        tmp = []
        
        if part.lower() not in ['all', 'upper', 'lower']:
            raise Exception('invalid part, should be in [\'all\', \'upper\', \'lower\']')
        
        # equivalent dimension is increasing or not
        coord_incre = True
        if eqDim.values[-1] < eqDim.values[0]:
            coord_incre = False
        
        # output contours and masks if mask_idx is provided
        masks = []
        contours = []
        returnmask = False
        if mask_idx is None:
            mask_idx = []
        else:
            if max(mask_idx) >= len(eqDim):
                raise Exception('indices in mask_idx out of boundary')
            returnmask = True
        
        # loop for each contour (or each equivalent dimension surface)
        for j in range(eqDimLen):
            # deviation from the reference
            qe = q - Q.isel({self.dimEqV:j})
            
            # above or below the reference coordinate surface
            m = eqDim>eqDim.values[j] if coord_incre else eqDim<eqDim.values[j]
            
            if self.increase:
                mask1 = xr.where(qe>0, -1, 0)
                mask2 = xr.where(m, 0, mask1).transpose(*(mask1.dims))
                mask3 = xr.where(np.logical_and(qe<0, m), 1, mask2)
            else:
                mask1 = xr.where(qe<0, -1, 0)
                mask2 = xr.where(m, 0, mask1).transpose(*(mask1.dims))
                mask3 = xr.where(np.logical_and(qe>0, m), 1, mask2)
            
            if j in mask_idx:
                contours.append(Q.isel({self.dimEqV:j}))
                masks.append(mask3)
            
            # select part over which integration is performed
            if part == 'all':
                maskFinal = mask3
            elif part == 'upper':
                if self.increase:
                    maskFinal = mask3.where(mask3>0)
                else:
                    maskFinal = mask3.where(mask3<0)
            else:
                if self.increase:
                    maskFinal = mask3.where(mask3<0)
                else:
                    maskFinal = mask3.where(mask3>0)
            
            # perform area-weighted conditional integration
            # lwa = (qe * maskFinal * wei *
            #        self.grid.get_metric(qe, self.dimEqN)).sum(self.dimEqV)
            lwa = -self.grid.integrate(qe * maskFinal * wei, self.dimEqN)
            
            tmp.append(lwa)
        
        LWA = xr.concat(tmp, self.dimEqV).transpose(*(q.dims))
        LWA[self.dimEqV] = eqDim.values
        
        if returnmask:
            return LWA.rename('LWA'), contours, masks
        else:
            return LWA.rename('LWA')


    def cal_local_APE(self, q, Q, mask_idx=None, part='all'):
        """
        Calculate local available potential energy (APE) density.  This is
        mathematically identical to local wave activity density.
        Reference: Winters and Barkan 2013, JFM; Scotti and White 2014, JFM

        Parameters
        ----------
        q: xarray.DataArray
            A tracer field.
        Q: xarray.DataArray
            The sorted tracer field.
        mask_idx: list of int
            Return masks at the indices of equivalent dimension.
        part: str
            The parts over which the integration is taken.  Available options
            are ['all', 'upper', 'lower'], corresponding to all, W+, and W-
            regions defined in Huang and Nakamura (2016, JAS)
        
        Returns
        ----------
        lape : xarray.DataArray
            Local APE density.
        contours : list
            A list of Q-contour corresponding to mask_idx.
        masks : list
            A list of mask corresponding to mask_idx.
        """
        if mask_idx is not None:
            LWA, contours, masks = \
                self.cal_local_wave_activity(q, Q, mask_idx, part=part)
            
            return LWA.rename('LAPE'), contours, masks
        else:
            return self.cal_local_wave_activity(q, Q, None, part).rename('LAPE')


    def cal_normalized_Keff(self, Leq2, Lmin, mask=1e5):
        """
        Calculate normalized effective diffusivity.

        Parameters
        ----------
        Leq2 : xarray.DataArray
            Squared equivalent length.
        Lmin : xarray.DataArray
            Minimum possible length.
        mask : float
            A threshold larger than which is set to nan.

        Returns
        ----------
        nkeff : xarray.DataArray
            The normalized effective diffusivity (Nusselt number).
        """
        nkeff = Leq2 / Lmin / Lmin
        nkeff = nkeff.where(nkeff<mask).rename('nkeff')

        return nkeff


    def interp_to_dataset(self, predef, dimEq, vs):
        """
        Interpolate given variables to prescribed equivalent latitudes
        and collect them into an xarray.Dataset.

        Parameters
        ----------
        predef : numpy.array or xarray.DataArray
            Pre-defined coordinate values are interpolated
        dimEq : xarray.DataArray
            Equivalent dimension defined in contour space.
        vs : list of xrray.DataArray or a xarray.Dataset
            A list of variables to be interplated
        
        Returns
        ----------
        interp : xarray.Dataset
            The interpolated variables merged in a Dataset.
        """
        re = []
        
        if type(vs) is xr.Dataset:
            for var in vs:
                re.append(self.interp_to_coords(predef, dimEq,
                                                vs[var]).rename(var))
        else:
            for var in vs:
                re.append(self.interp_to_coords(predef, dimEq,
                                                var).rename(var.name))
        
        return xr.merge(re)
    

    def interp_to_coords(self, predef, eqCoords, var, interpDim='contour'):
        """
        Interpolate a give variable from equally-spaced contour dimension
        to a predefined coordinates along equivalent dimension.

        Parameters
        ----------
        predef : numpy.array or xarray.DataArray
            Pre-defined Ys where values are interpolated
        eqCoords : xarray.DataArray
            Equivalent coordinates.
        var : xarray.DataArray
            A given variable to be interplated
        interpDim : str
            Dimension along which it is interpolated
        
        Returns
        ----------
        interp : xarray.Dataset
            The interpolated variable.
        """
        dimTmp = 'new'

        if isinstance(predef, (np.ndarray, list)):
            # add coordinate as a DataArray
            predef  = xr.DataArray(predef, dims=dimTmp, coords={dimTmp: predef})
        else:
            dimTmp = predef.dims[0]

        # get a single vector like Yeq[0, 0, ..., :]
        vals = eqCoords
        
        while len(vals.shape) > 1:
            vals = vals[0]
        
        if vals[0] < vals[-1]:
            increasing = True
        else:
            increasing = False
        
        # no dask support for np.linspace
        varIntp = xr.apply_ufunc(_interp1d, predef, eqCoords, var.load(),
                  kwargs={'inc': increasing},
                  # dask='allowed',
                  input_core_dims =[[dimTmp],[interpDim],[interpDim]],
                  output_core_dims=[[dimTmp]],
                  exclude_dims=set((interpDim,)),
                  vectorize=True
                  ).rename(var.name)

        return varIntp


class Table(object):
    """
    This class is designed as a one-to-one mapping table between two
    mononitical increasing/decreasing quantities.
    
    The table is represented as y = F(x), with y as the values and
    x the coordinates.
    """
    def __init__(self, table, dimEq):
        """
        Construct a table.

        Parameters
        ----------
        table : xarray.Dataset
            A table quantity as a function of specific coordinate.
        dimEq : numpy.array or xarray.Dataset
            A set of equivalent coordinates along a dimension
        """
        tmp = table[{dimEq:-1}] > table[{dimEq:0}]
        if   (tmp == True ).all():
            areaInc = True
        elif (tmp == False).all():
            areaInc = False
        else:
            raise Exception('not every time or level is increasing/decreasing')
        
        self._table = table
        self._coord = table[dimEq]
        self._dimEq = dimEq
        self._incVl = areaInc
        self._incCd = table[dimEq][-1] > table[dimEq][0]

    def lookup_coordinates(self, values):
        """
        For y = F(x), get coordinates (x) given values (y).

        Parameters
        ----------
        values: numpy.ndarray or xarray.DataArray
            Values as y.

        Returns
        -------
        coords : xarray.DataArray
            Coordinates as x.
        """
        dimEq = self._dimEq
        iDims = [[],[dimEq],[dimEq]]
        oDims = [[]]
        
        if 'contour' in values.dims:
            iDims = [['contour'],[dimEq],[dimEq]]
            oDims = [['contour']]
        
        # if len(values.shape) == 1:
        #     return _interp1d(values, self._table, self._coord, self._incVl)
        # else:
        re = xr.apply_ufunc(_interp1d,
                                values, self._table, self._coord,
                                kwargs={'inc': self._incVl},
                                dask='allowed',
                                input_core_dims = iDims,
                                output_core_dims= oDims,
                                output_dtypes=[self._table.dtype],
                                # exclude_dims=set(('contour',)),
                                vectorize=True)
        
        # if isinstance(re, np.ndarray):
        #     re = xr.DataArray(re, dims=values.dims, coords=values.coords)
        
        return re

    def lookup_values(self, coords):
        """
        For y = F(x), get values (y) given coordinates (x).

        Parameters
        ----------
        coords : list or numpy.array or xarray.DataArray
            Coordinates as x.

        Returns
        -------
        values : xarray.DataArray
            Values as y.
        """
        re = _interp1d(coords, self._coord, self._vables, self._incCd)
        
        if isinstance(re, np.ndarray):
            re = xr.DataArray(re, dims=coords.dims, coords=coords.coords)
        
        return re
        


"""
Below are the private helper methods
"""
def _histogram(var, bins, dim, weights, lt):
    """
    A wrapper for xhistogram, which allows decreasing bins and return
    a result that contains the same size as that of bins.
    
    Note that it is assumed the start and end bins correspond to the tracer
    extrema.
    
    Parameters
    ----------
    var: xarray.DataArray
        A variable that need to be histogrammed.
    bins: list or numpy.array or xarray.DataArray
        An array of bins.
    dim: str or list of str
        Dimensions along which histogram is performed.
    weights: xarray.DataArray
        Weights of each data in var.
    increase: bool
        Increasing bins with index or not.
    lt: bool
        Less than a given value or not.
    
    Returns
    ----------
    hist : xarray.DataArray
        Result of the histogram.
    """
    if type(bins) in [np.ndarray, np.array]:
        bvalues = bins
        
        if not np.diff(bvalues).all():
            raise Exception('non monotonic bins')
            
    elif type(bins) in [xr.DataArray]:
        bvalues = bins.squeeze() # squeeze the dimensions
        
        if not bvalues.diff('contour').all():
            raise Exception('non monotonic bins')
        
        if not 'time' in bvalues.dims:
            bvalues = bvalues.values
        
    elif type(bins) in [list]:
        bvalues = np.array(bins)
        
        if not np.diff(bvalues).all():
            raise Exception('non monotonic bins')
    else:
        raise Exception('bins should be numpy.array or xarray.DataArray')
            
    # unified index of the contour coordinate
    if type(bvalues) in [xr.DataArray]:
        binNum = np.array(range(len(bvalues['contour']))).astype(np.float32)
    else:
        binNum = np.array(range(len(bvalues))).astype(np.float32)
    
    if type(bvalues) in [xr.DataArray]:
        re = []
        
        for l in range(len(bvalues.time)):
            rng = {'time': l}
            
            trc = var.isel(rng)
            ctr = bvalues.isel(rng).values
            
            if 'time' in weights.dims:
                wt = weights.isel(rng)
            else:
                wt = weights
            
            bincrease = True if ctr[0] < ctr[-1] else False
            
            # add a bin so that the result has the same length of contour
            if bincrease:
                step = (ctr[-1] - ctr[0]) / (len(ctr) - 1)
                bins = np.concatenate([[ctr[0]-step], ctr])
            else:
                step = (ctr[0] - ctr[-1]) / (len(ctr) - 1)
                bins = np.concatenate([[ctr[-1]-step], ctr[::-1]])
                # bins[1:] -= step / 1e3
            
            tmp = histogram(trc, bins=[bins], dim=dim, weights=wt) \
                 .assign_coords({trc.name+'_bin':binNum})
            
            re.append(tmp)
        
        pdf = xr.concat(re, 'time').rename({var.name+'_bin':'contour'})
        
        if bincrease:
            pdf = pdf.assign_coords(contour=binNum)
        else:
            pdf = pdf.assign_coords(contour=binNum[::-1])
    else:
        bincrease = True if bvalues[0] < bvalues[-1] else False
        
        # add a bin so that the result has the same length of contour
        if bincrease:
            step = (bvalues[-1] - bvalues[0]) / (len(bvalues) - 1)
            bins = np.insert(bvalues, 0, bvalues[0]-step)
        else:
            step = (bvalues[0] - bvalues[-1]) / (len(bvalues) - 1)
            bins = np.insert(bvalues[::-1], 0, bvalues[-1]-step)
            # bins[1:] -= step / 1e3
        
        pdf = histogram(var, bins=[bins], dim=dim, weights=weights) \
              .rename({var.name+'_bin':'contour'})
        
        if bincrease:
            pdf = pdf.assign_coords(contour=binNum)
        else:
            pdf = pdf.assign_coords(contour=binNum[::-1])
    
    # assign time coord. to pdf
    if 'time' in var.dims:
        pdf = pdf.assign_coords(time=var['time'].values)
    
    # get CDF from PDF
    cdf = pdf.cumsum('contour')
    
    if not lt: # for the case of greater than
        cdf = cdf.isel({'contour':-1}) - cdf
    
    return cdf


def _check_monotonicity(var, dim):
    """
    Check monotonicity of a variable along a dimension.

    Parameters
    ----------
    var : xarray.DataArray
        A variable that need to be checked.
    dim : str
        A string indicate the dimension.

    Returns
    ----------
        None.  Raise exception if not monotonic
    """
    dfvar = var.diff(dim)
    
    if not dfvar.all():
        pos = (dfvar == 0).argmax(dim=var.dims)
        
        for tmp in pos:
            print(tmp)
            print(pos[tmp].values)
            
            if tmp != dim:
                v = var.isel({tmp:pos[tmp].values}).load()
        
        raise Exception('not monotonic var at\n' + str(v))


def _get_extrema_extend(data, N):
    """
    Get the extrema by extending the endpoints
    
    Parameters
    ----------
    data: xarray.DataArray
        A variable that need to be histogrammed.
    N: int
        A given length to get step
    
    Returns
    ----------
    vmin, vmax : float, float
        Extended extrema.
    """
    vmin = data.min().values
    vmax = data.max().values
    
    step = (vmax - vmin) / N
    
    return vmin - step, vmax + step


def _interp1d(x, xf, yf, inc=True, outside=None):
    """
    Wrapper of np.interp, taking into account the decreasing case.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xf : 1-D sequence of floats
        The x-coordinates of the data points.
    yf : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as `xp`.
    inc : boolean
        xf is increasing or decresing.

    Returns
    ----------
    y : float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as `x`.
    """
    # print(f"x : {x.shape} \nxf: {xf.shape} \nyf: {yf.shape}")
    if inc: # increasing case
        re = np.interp(x, xf, yf)
    else: # decreasing case
        # print(f"x: {x} \n xf: {xf} \n yf: {yf}")
        re = np.interp(x, xf[::-1], yf[::-1])

    # print(f"x: {x} \n xf: {xf} \n yf: {yf} \n y: {re}")

    return re


"""
Testing codes for each class
"""
if __name__ == '__main__':
    print('start testing in ContourMethods')

