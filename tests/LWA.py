import xarray as xr
import numpy as np
import os
from xcontour.xcontour import Contour2D, add_latlon_metrics


place="D:/"
LWA_Type="Areal" #Isentropic"
vertical_differentiation_method='central'    #'backeard'
resolution=0.75


for file in ["testERA.nc"]:
    data=xr.open_dataset(os.path.join(place,file))
    earth_radius=6371200
    lat=data.latitude
    Area=(np.radians(resolution)*earth_radius)**2*(np.cos(np.radians(lat)))+0*data.longitude
    data=data.reindex(latitude=lat.values[::-1])
    data=data.interp(level=[319.9,320,320.1], method='linear')
    data=data.isel(time=[0])   
    g=9.81
    earth_circle_perimeter=2*np.pi*earth_radius*np.cos(np.radians(lat))
    var='pv'
    out=[]
    
    if LWA_Type=="Isentropic":
        if not( "sigma" in list(data._variables.keys())):
            if vertical_differentiation_method!='central':
                diffy=-data.pres.diff("level")/data.level.diff("level")/g
                complement=diffy.sel(level=diffy.level.values[0:2])\
                   .interp(level=data.level.values[0],method='linear',
                           kwargs={"fill_value": "extrapolate"})
                diffy=xr.concat([complement,diffy],'level') 
                data['sigma']=diffy.copy().transpose('time','level', 'latitude',
                                                     'longitude')
            else:
                data['sigma']=(-data['pres'].differentiate("level")/g
                              ).transpose('time','level','latitude','longitude') 
    
    for t in np.arange(len(data.time)):
        lvl=[]
        for l in np.arange(len(data.level)):
            dset=data.isel(time=t,level=l)[[var]]
            dset, grid = add_latlon_metrics(dset, {'lat':'latitude',
                                                   'lon':'longitude'}) 
            
            tracer = dset[var]
            if LWA_Type=="Isentropic":
                sigma=data.isel(time=t,level=l)[['sigma']].sigma
            else:
                sigma=tracer*0+1
            print('date= '+str(data.time.values[t])[0:10]+' , Level= '+str(data.level.values[l])+' hPa')
            
            N  = len(data.latitude) # increase the contour number may get non-monotonic A(q) relation
            increase = True    # Y-index increases with latitude
            lt = True          # northward of PV contours (larger than) is inside the contour
            dtype = np.float32 # use float32 to save memory
            undef = -9.99e8    # for maskout topography if present
            analysis = Contour2D(grid, sigma*tracer,
                     dims={'X':'longitude','Y':'latitude'},
                     dimEq={'Y':'latitude'},
                     increase=increase,
                     lt=lt)
            
            ctr = analysis.cal_contours(N).rename(var)
            mask = xr.where(tracer!=undef, 1, 0).astype(dtype)
            table   = analysis.cal_area_eqCoord_table(mask) # A(Yeq) table
            area   = analysis.cal_integral_within_contours(ctr,tracer=sigma*tracer,integrand=sigma*0+1).rename('Area')
            latEq   = table.lookup_coordinates(area).rename('latEq')
            ds_contour = xr.merge([ctr, latEq, area])
            
            preLats = tracer.latitude.astype(dtype)
            ds_latEq = analysis.interp_to_dataset(preLats, latEq, ds_contour)
            
            lwa, ctrs, masks = analysis.cal_local_wave_activity(tracer*sigma,
                                                                ds_latEq[var],
                                                                mask_idx=np.arange(N),
                                                                part="all")
            lwa1=lwa/earth_circle_perimeter
            ctrs=xr.concat(ctrs,dim="latitude")*1e6
            ctrs.attrs=dset[var].attrs
            ctrs.attrs['units']='PVU'
            lwa.attrs=dset[var].attrs
            lwa.attrs['units']='m*s-1'
            lwa.attrs['long_name']='Local Finite Amplitude Wave Activity'
            lwa['Equivalent Latitude']=ctrs.copy()
            lvl.append(lwa)
            
        for  item_index in np.arange(len(masks)):
            item=masks[item_index].copy()
            for field in ['dxC','dyC','rAc']:
                item=item.drop(field)
                item=item.rename({"latitude":"Latitude","longitude":"Longitude"})    
            item=item.assign_coords(latitude=lwa.latitude.values[item_index]).expand_dims("latitude")   
            masks[item_index]=item.copy()
            
        masks=xr.concat(masks,dim="latitude")    
        out+=[xr.concat(lvl,dim="level")]     
        outname=os.path.join(place,file.replace(".nc","_LWA.nc")) 
        out=xr.concat(out,dim="time") 
        out.to_netcdf(outname) 
        
    ss=masks.sel(latitude=75)
    ss=ss * Area.rename({"latitude":"Latitude","longitude":"Longitude"})
    positive_intrusion_area=xr.where(ss<0.,-ss,0).sum()
    negative_intrusion_area=xr.where(ss>0.,ss,0).sum()

