#!/usr/bin/env python
import os
import sys
import netCDF4 as nc
import datetime as dt
import glob
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
import geopy
import collections
import metpy.calc as mpcalc
import xarray as xr
import scipy
import math
from IPython.core.pylabtools import figsize, getfigs
from metpy.units import units
from geopy.distance import geodesic
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import Normalize
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from statistics import mean, median, mode, stdev
from collections import Counter
from statsmodels.stats.weightstats import ztest as ztest


# 1. Function to read TC Best Track and Other Information
# 2. Function to read rawinsonde Obs
# 3. Function to read rawinsonde obs (not from saved files) in case for the first time
# 4. Function to read dropsonde obs (not from saved files)
# 5. Function to extract bias and rmse and significance test
# 6. Function to plot subplots for all vertical profiles

def read_btk(path, fname):

    '''
    Function that reads TC best track and information. 
    
    Best track information files need to be downloaded from 
    https://ftp.nhc.noaa.gov/atcf/archive/${YEAR}/bal${i}${YEAR}.dat.gz. 
    
    - The arguments of the function:
      > path           # path of best track file 
      > fname          # name of best track file
      
    - The result of the function:
      > tc_th          # the order of TC 
      > tc_name        # TC name 
      > tc_time        # TC time 
      > tc_lat, tc_lon # TC best track 
      
    '''
    
    # LatN/S     - Latitude for the DTG: 0 - 900 tenths of degrees,
    #             N/S is the hemispheric index.
    # LonE/W     - Longitude for the DTG: 0 - 1800 tenths of degrees,
    #             E/W is the hemispheric index.

    N = 45 # Number of columns in a best track file
    btk = pd.read_csv(path+fname, sep=",", header=None, names=range(N), encoding='utf8')
    btk_np = btk.to_numpy()

    # TC XXth
    if btk_np[0,1] < 10:
        
        tc_th = "0"+str(btk_np[0,1])
        
    else:
        
        tc_th = str(btk_np[0,1])
    
    # TC name
    tc_name = str(btk_np[-1,27].replace(' ',''))
    
    # TC time and best track
    tc_time = []
    tc_lat = []
    tc_lon =[]
    six_hr = ['00','06','12','18']
    
    # Remove duplicates and store information.
    for i in range(len(btk_np[:,2])): # tc_time
        
        # only if tc time is not duplicated and is 6-hrly    
        if str(btk_np[i,2]) not in tc_time and str(btk_np[i,2])[8:10] in six_hr: 
            
            # Store tc_time
            tc_time.append(str(btk_np[i,2]))
            
            # Store tc_lat
            if btk_np[i,6][-1] == 'N':
                
                tc_lat.append(float(format(int(btk_np[i,6][0:4])*0.1,'.1f')))
                
            else:
                
                tc_lat.append(float(format(int(btk_np[i,6][0:4])*(-0.1), '.1f')))
    
            # Store tc_lon
            if btk_np[i,7][-1] == 'E':
                
                tc_lon.append(float(format(int(btk_np[i,7][0:5])*0.1,'.1f')))
                
            else:
                
                tc_lon.append(float(format(360-int(btk_np[i,7][0:5])*0.1, '.1f')))
    
    return tc_th, tc_name, tc_time, tc_lat, tc_lon

def cal_vector(latobs, lonobs, timeobs, tc_time, tc_lat, tc_lon):

    '''
    Function that calculates a vector from TC center to rawinsonde location, 
    which is observation location minus TC center (dx, dy). 
    
    - The arguments of this function:
      > latobs, lonobs, timeobs    # Observation
      > tc_time, tc_lat, tc_lon    # TC best track
    
    - The result of this function:
      > dx, dy                     # vector from TC to obs
      > tc_lat_time, tc_lon_time   # Best track information for the corresponding obs 
    
    '''

    # Earth radius (a globally-average value)
    r_earth = 6371
    pi = math.pi

    latobs = [float(format(float(i), '.1f')) for i in latobs]
    lonobs = [float(format(float(i), '.1f')) for i in lonobs]

    # The length of tc_BestTrack_time array is the same as the number of obs
    tc_lat_time = np.zeros((len(latobs)))
    tc_lon_time = np.zeros((len(latobs)))
    
    # for all obs
    for i in range(len(latobs)):

        # if timeobs is in the corresponding TC best track time (lat and lon)
        if timeobs[i] in tc_time:
        
            # when time of obs[i] is the same as tc_time[j]
            j = tc_time.index(timeobs[i])
        
            # Assign Best track (lat&lon) information to array with the same size of obs
            # Best Track 
            tc_lat_time[i] = tc_lat[j]
            tc_lon_time[i] = tc_lon[j]
            
        else:
            
            # Best Track  
            tc_lat_time[i] = None
            tc_lon_time[i] = None
            
    # Latitude averaged for TC and obs latitude
    mean_lat = 0.5 * ( tc_lat_time + np.array(latobs) )

    # Calculate a vector = (obs - TC_center) = (dx, dy)
    # (dx) = l = r * theta = ( r_earth*cos(mean_lat) ) * (lon_obs - lon_tc)
    dx = r_earth * np.cos(np.radians(mean_lat)) * np.radians( np.array(lonobs) - tc_lon_time )
    # (dy) = l = r * theta = (r_earth * (lat_obs - lat_tc))
    # dy = r_earth*radians(float(latobs[0])-float(tc_lat_time))
    dy = r_earth * np.radians( np.array(latobs) - tc_lat_time )
    
    return dx, dy, tc_lat_time, tc_lon_time

def cal_dist(dx, dy):
    
    '''
    Function that calculates a distance between TC best track and observation.
    '''
    dist = np.sqrt(dx**2 + dy**2)
    
    return dist

def read_chosen_nc_bc(path, fname, obsvar, vlat, vlon):

    '''
    Function that reads one chosen netCDF file accumulated with all the information from innovation files.
    but only within the boundaries assigned.
    
    - The arguments of the function:
      > path    # one chosen netCDF file path
      > fname   # one chosen netCDF file name
      > obsvar  # observation variable
      > vlat    # specified latitudes
      > vlon    # specified longitudes
    
    - The results of the function:
      > latobs  # latitude of observations
      > lonobs  # longitude of observations
      > preobs  # pressure of observations
      > sidobs  # station id of observations
      > qmobs   # quality marker of observations
      > timeobs # time of observations
      > innobs  # innovation at the observation locations
    
    '''
    
    inno_file = path+'/'+fname
    ncfile = nc.Dataset(inno_file)

    if ncfile.variables['lat_obs']:

        latob  = ncfile.variables['lat_obs'][:] # latitude of observation
        lonob  = ncfile.variables['lon_obs'][:] # longitude of observation
        preob = ncfile.variables['pre_obs'][:] # pressure of observation
        sidob = ncfile.variables['sid_obs'][:] # station id of observation
        qmob = ncfile.variables['qm_obs'][:] # quality marker of observation
        timeob = ncfile.variables['time_obs'][:] # time of observation

        # For wind obs
        if obsvar == 'uv':
            
            innobs_u = ncfile.variables['inn_u'][:] # innovation of u wind
            innobs_v = ncfile.variables['inn_v'][:] # innovation of v wind
            obs_u = ncfile.variables['o_u'][:] # observation of u wind
            obs_v = ncfile.variables['o_v'][:] # observatino of v wind
            
        # For T or q obs
        else:
            
            innob = ncfile.variables['inn'][:] # innovation
            
    # Store observations that are located only in the specified boundaries.
    latobs=[ latob[i] for i in range(len(latob))            if min(vlat) <= latob[i] <= max(vlat) and min(vlon) <= lonob[i] <= max(vlon) ]
    lonobs=[ lonob[i] for i in range(len(latob))            if min(vlat) <= latob[i] <= max(vlat) and min(vlon) <= lonob[i] <= max(vlon) ]
    preobs=[ preob[i] for i in range(len(latob))            if min(vlat) <= latob[i] <= max(vlat) and min(vlon) <= lonob[i] <= max(vlon) ]
    sidobs=[ str(sidob[i]) for i in range(len(latob))            if min(vlat) <= latob[i] <= max(vlat) and min(vlon) <= lonob[i] <= max(vlon) ]
    innobs=[ innob[i] for i in range(len(latob))            if min(vlat) <= latob[i] <= max(vlat) and min(vlon) <= lonob[i] <= max(vlon) ]
    timeobs=[ str(timeob[i]) for i in range(len(latob))            if min(vlat) <= latob[i] <= max(vlat) and min(vlon) <= lonob[i] <= max(vlon) ]
    qmobs=[ qmob[i] for i in range(len(latob))            if min(vlat) <= latob[i] <= max(vlat) and min(vlon) <= lonob[i] <= max(vlon) ]

    if obsvar == 'uv':
        return latobs, lonobs, preobs, sidobs, qmobs, timeobs, innobs_u, innobs_v, obs_u, obs_v
    else:
        return latobs, lonobs, preobs, sidobs, qmobs, timeobs, innobs

def read_chosen_nc(path, fname, obsvar):
    
    '''
    Function that reads one chosen netCDF file accumulated with all the information from innovation files
    for the whole domain.
    
    - The arguments of the function: 
      > path    # one chosen netCDF file path
      > fname   # one chosen netCDF file name
      > obsvar  # observation variable
    
    - The results of the function:
      > latobs  # latitude of observations
      > lonobs  # longitude of observations
      > preobs  # pressure of observations
      > sidobs  # station id of observations
      > qmobs   # quality marker of observations
      > timeobs # time of observations
      > innobs  # innovation at the observation locations
    
    '''

    inno_file = path+'/'+fname
    ncfile = nc.Dataset(inno_file)

    if ncfile.variables['lat_obs']:

        latobs  = ncfile.variables['lat_obs'][:] # latitude of observation
        lonobs  = ncfile.variables['lon_obs'][:] # longitude of observation
        preobs = ncfile.variables['pre_obs'][:] # pressure of observation
        sidobs = ncfile.variables['sid_obs'][:] # station id of observation
        qmobs = ncfile.variables['qm_obs'][:] # quality marker of observation
        timeobs = ncfile.variables['time_obs'][:] # time of observation

        # For wind obs
        if obsvar == 'uv':
            
            innobs_u = ncfile.variables['inn_u'][:] # innovation of u wind
            innobs_v = ncfile.variables['inn_v'][:] # innovation of v wind
            obs_u = ncfile.variables['o_u'][:] # observation of u wind
            obs_v = ncfile.variables['o_v'][:] # observatino of v wind
            
        # For T or q obs
        else:
            
            innobs = ncfile.variables['inn'][:] # innovation
            
    if obsvar == 'uv':
        return latobs, lonobs, preobs, sidobs, qmobs, timeobs, innobs_u, innobs_v, obs_u, obs_v
    else:
        return latobs, lonobs, preobs, sidobs, qmobs, timeobs, innobs
      


# Read rawinsonde obs (not from saved files) in case for the first time
def read_gsi_range_sid(path, fname, exp_name, obs_type_want, obsvar, vlat, vlon, write_nc, *args):
 
    '''
    Function that reads all innovation netCDF files, 
    but considers only observations within the assigned domain and
    station id.
    
    - The arguments of the function:
      > path          # innovation file path
      > fname         # innovation file name
      > exp_name      # experiment name
      > obs_type_want # observation type
      > obsvar        # observation variable
      > vlat          # specified latitudes
      > vlon          # specified longitudes
      > write_nc      # write netCDF file for chosen obs information (True or False)
      > args          # variable to read in diagnostic netCDF files (innovation, OMB)
                      # if len(args) > 1, the obs variable is wind
    
    - The results of the function:
      > latobs        # latitude of observations
      > lonobs        # longitude of observations
      > preobs        # pressure of observations
      > sidobs        # station id of observations
      > qmobs         # quality marker of observations
      > timeobs       # time of observations
      > innobs        # innovation at the observation locations
    
    '''
    
    def add_id(ds):
        ds.coords['date_time'] = ds.attrs['date_time']
        return ds


    # Check the platform you are using
    # 1) Python
    #fs = xr.open_mfdataset(path+'/'+fname, preprocess=add_id)
    #
    # 2) Jupyter Notebook
    fs = xr.open_mfdataset(path+'/'+fname, decode_cf=False,                            combine='nested',concat_dim='nobs',preprocess=add_id)
    
    #files = xr.open_mfdataset(path+'/'+fname, preprocess=add_id)
    #fs = files.to_dataframe()

    latobs = []
    lonobs = []
    preobs = []
    innobs = []
    sidobs = []
    qmobs = []
    timeobs = []

    if len(args) > 1:
        innobs_u = []
        innobs_v = []
        obs_u = []
        obs_v = []
    else :
        innobs = []

    lat  = fs.Latitude
    lon  = fs.Longitude
    pres = fs.Pressure
    otyp = fs.Observation_Type
    #sid = fs.Station_ID
    #sid = fs.Station_ID.decode('utf-8')
    sid = fs.Station_ID.str.decode('utf-8').fillna(fs.Station_ID)
    date_time = fs.date_time
    qm = fs.Prep_QC_Mark 
    
    # For u and v winds,
    if len(args) > 1:
        inno_u = fs.u_Obs_Minus_Forecast_adjusted # Innovation
        inno_v = fs.v_Obs_Minus_Forecast_adjusted # Innovation
        inno_u = np.array(inno_u)
        inno_v = np.array(inno_v)
        o_u = fs.u_Observation # Obs
        o_v = fs.v_Observation # Obs
        o_u = np.array(o_u)
        o_v = np.array(o_v)
    # For T or q,
    else :
        inno = fs.Obs_Minus_Forecast_adjusted # Innovation 
        inno = np.array(inno)
        o = fs.Observation # Obs
        o = np.array(o)
                  
    lat = np.array(lat)
    lon = np.array(lon)
    pres = np.array(pres)
    otyp = np.array(otyp)
    sid = np.array(sid)
    qm = np.array(qm)
    date_time = np.array(date_time)

    # Select obs which satisfies the following conditions
    # 1) otyp[i] == obs_type_want
    # 2) obs only located in verification domain
    # 3) Quality Mark as follows
    #    qm <= 3       : could be used in analysis
    #    qm == 9 or 15 : used for monitoring (not used analysis)
    latobs=[ lat[i] for i in range(len(lat)) if otyp[i] == obs_type_want             if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)             if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
    lonobs=[ lon[i] for i in range(len(lat)) if otyp[i] == obs_type_want             if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)             if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
    preobs=[ pres[i] for i in range(len(lat)) if otyp[i] == obs_type_want             if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)             if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
    #Check the platform you are using
    # 1) Python
    #sidobs=[ str(sid[i].decode("utf-8")).strip() for i in range(len(lat)) if otyp[i] == obs_type_want \
    # 2) Jupyter notebook
    sidobs=[ str(sid[i]) for i in range(len(lat)) if otyp[i] == obs_type_want             if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)             if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
    
    qmobs=[ qm[i] for i in range(len(lat)) if otyp[i] == obs_type_want             if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)             if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]

    timeobs=[ str(date_time[i]) for i in range(len(lat)) if otyp[i] == obs_type_want             if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)             if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
           
    if len(args) > 1:
        innobs_u=[ inno_u[i] for i in range(len(lat)) if otyp[i] == obs_type_want                if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)                if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
        innobs_v=[ inno_v[i] for i in range(len(lat)) if otyp[i] == obs_type_want                if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)                if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
        obs_u=[ o_u[i] for i in range(len(lat)) if otyp[i] == obs_type_want                if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)                if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
        obs_v=[ o_v[i] for i in range(len(lat)) if otyp[i] == obs_type_want                if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)                if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
    else:
        innobs=[ inno[i] for i in range(len(lat)) if otyp[i] == obs_type_want                if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)                if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
        obs=[ o[i] for i in range(len(lat)) if otyp[i] == obs_type_want                if min(vlat) <= lat[i] <= max(vlat) and min(vlon) <= lon[i] <= max(vlon)                if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 ]
   
    print('--- Total number of sidobs is ',len(sidobs))
    sid_d = collections.Counter(sidobs)
    #print("--- sid_d=", sid_d)
    #new_sid_list = list([item for item in sid_d if sid_d[item]>1])
    #print(new_sid_list)
    #print("--- new_sid_list is ", len(new_sid_list))  

    # if write_nc == True, create and write netCDF file to store all the infomration read here.
    if write_nc == True:

        # Get current working directory
        cwd=os.getcwd()
        # File name to store
        fn = cwd+'/All_diag_'+obsvar+'_'+str(obs_type_want)+'_'+sorted(timeobs)[0]+             '_'+sorted(timeobs)[-1]+'.'+exp_name+'.test.nc'
     
        dss = nc.Dataset(fn,'w',format='NETCDF4')

        # Create Dimension
        nobs = dss.createDimension('nobs', len(latobs))
        ntime = dss.createDimension('ntime', len(timeobs))
        # Create Variable
        lat_obs = dss.createVariable('lat_obs','f',('nobs',))
        lon_obs = dss.createVariable('lon_obs','f',('nobs',))
        pre_obs = dss.createVariable('pre_obs','f',('nobs',))
        sid_obs = dss.createVariable('sid_obs','str',('nobs',))
        qm_obs = dss.createVariable('qm_obs','f',('nobs',))
        time_obs = dss.createVariable('time_obs','str',('ntime',))

        lat_obs[:] = latobs
        lon_obs[:] = lonobs
        pre_obs[:] = preobs
        sid_obs[:] = np.array(sidobs)
        qm_obs[:] = qmobs
        time_obs[:] = np.array(timeobs)

        if len(args) > 1:
            inn_u = dss.createVariable('inn_u','f',('nobs',))
            inn_v = dss.createVariable('inn_v','f',('nobs',))
            o_u = dss.createVariable('o_u','f',('nobs',))
            o_v = dss.createVariable('o_v','f',('nobs',))
            inn_u[:] = innobs_u
            inn_v[:] = innobs_v
            o_u[:] = obs_u
            o_v[:] = obs_v
        else:
            inn = dss.createVariable('inn','f',('nobs',))
            inn[:] = innobs
            o = dss.createVariable('o','f',('nobs',))
            o[:] = obs

        dss.close()
        
    if len(args) > 1:
        return latobs, lonobs, preobs, sidobs, qmobs, timeobs, innobs_u, innobs_v, obs_u, obs_v
    else:
        return latobs, lonobs, preobs, sidobs, qmobs, timeobs, innobs      


# Read Dropsonde obs (not from saved files) 
def read_gsi_range_dropsonde(path, fname, obs_type_want, vlat, vlon, *args):
    
    '''
    
    Function that reads all innovation netCDF files for dropsonde obs, but
    1) considers only observations within the assigned domain, 
    2) removes outliers (i.e., excludes obs whose innovation >= 10*obs_error), and 
    3) reads the exact observation time "Time" (e.g. +2 h, -1.5 h) 
    compared to analysis time.
    
    - The arguments of the function:
      > path          # innovation file path
      > fname         # innovation file name
      > obs_type_want # observation type
      > vlat          # specified latitudes
      > vlon          # specified longitudes
      > args          # variable to read in diagnostic netCDF files (innovation, OMB)
                      # if len(args) > 1, the obs variable is wind

    - The results of the function:
      > latobs        # latitude of observations
      > lonobs        # longitude of observations
      > preobs        # pressure of observations
      > qmobs         # quality marker of observations
      > timeobs       # time of observations
      > h_timeobs     # time of observations [h] compared to analysis time
      > innobs        # innovation at the observation locations

    '''
    
    # Read innovation file 
    inno_files = sorted(glob.glob(path + fname))
    
    latobs = []
    lonobs = []
    preobs = []
    innobs = []
    sidobs = []
    qmobs = []
    timeobs = []
    h_timeobs = []
    
    if len(args) > 1:
        
        innobs_u = []
        innobs_v = []
        obs_u = []
        obs_v = []
        
    else :
        
        innobs = []
    
    # Loop for reading diagnostic files.
    for f in range(len(inno_files)):
        
        ncfile = nc.Dataset(inno_files[f])
        
        if ncfile.variables['Latitude']:
            
            lat  = ncfile.variables['Latitude'][:]
            lon  = ncfile.variables['Longitude'][:]
            pres = ncfile.variables['Pressure'][:]
            otyp = ncfile.variables['Observation_Type'][:]
            date_time = ncfile.date_time
            qm = ncfile.variables['Prep_QC_Mark'][:]
            #da = ncfile.variables['Analysis_Use_Flag'][:]
            h_time = ncfile.variables['Time'][:]
            errinv_input = ncfile.variables['Errinv_Input'][:]
            #errinv_adjust = ncfile.variables['Errinv_Adjust'][:]
            #errinv_final = ncfile.variables['Errinv_Final'][:]

            if len(args) > 1:
                inno_u = ncfile.variables[args[0]][:]
                inno_v = ncfile.variables[args[1]][:]
                o_u = ncfile.variables[args[2]][:]
                o_v = ncfile.variables[args[3]][:]
            else :
                inno = ncfile.variables[args[0]][:]

            for i in range(len(lat)):

                if otyp[i] == obs_type_want:
        
                    #---------------------------------------
                    # 1) Obs only in verification domain
                    #---------------------------------------
                    if min(vlat) <= lat[i] <= max(vlat)                     and min(vlon) <= lon[i] <= max(vlon) :
                        
                        #---------------------------------------
                        # 2) Quality Mark
                        # qm <= 3 : could be used in analysis
                        # qm == 9 or 15 : used for monitoring (not used analysis)
                        #---------------------------------------
                        if qm[i] <= 3 or qm[i] == 9 or qm[i] == 15 :
                            
                            #--------------------------------------
                            # 3) Remove outlier (innov >= obs_error*10)
                            
                            # for u and v winds
                            if len(args) > 1:
                                
                                if abs(inno_u[i]) < 1/errinv_input[i]*10 :

                                    latobs.append(lat[i])
                                    lonobs.append(lon[i])
                                    preobs.append(pres[i])
                                    qmobs.append(qm[i])
                                    timeobs.append(str(date_time))
                                    h_timeobs.append(h_time[i])
                                
                                    if len(args) > 1:
                                        
                                        innobs_u.append(inno_u[i])
                                        innobs_v.append(inno_v[i])
                                        obs_u.append(o_u[i])
                                        obs_v.append(o_v[i])
                                        
                                    else:
                                        
                                        innobs.append(inno[i])
                                        
                            # for T and q
                            else:
                                
                                if abs(inno[i]) < 1/errinv_input[i]*10 :

                                    latobs.append(lat[i])
                                    lonobs.append(lon[i])
                                    preobs.append(pres[i])
                                    qmobs.append(qm[i])
                                    timeobs.append(str(date_time))
                                    h_timeobs.append(h_time[i])
                                
                                    if len(args) > 1:
                                        innobs_u.append(inno_u[i])
                                        innobs_v.append(inno_v[i])
                                        obs_u.append(o_u[i])
                                        obs_v.append(o_v[i])
                                    else:
                                        innobs.append(inno[i])

    print('The total number of obs is ', len(latobs), len(innobs))
    
    if len(args) > 1:
        return latobs, lonobs, preobs, qmobs, timeobs, h_timeobs, innobs_u, innobs_v, obs_u, obs_v
    else:
        return latobs, lonobs, preobs, qmobs, timeobs, h_timeobs, innobs

def calculate_bias_rmse_profile_ttest(pres, innov, obsunits, alpha, plevels):

    '''
    
    Function that calculates bias and rmse profile with significance test
    
    - The arguments of the function:
      > pres          # pressure levels of observations
      > innov         # innovations
      > obsunits      # observation units
      > alpha         # T-test significance level for a significance test (e.g., alpha=0.05)
      > plevels       # pressure levels to evaluate

    - The results of the function:
      > rmse        # averaged rmse
      > bias        # averaged bias
      > bias_sig    # statistically significant averaged bias
      > plevels_sig # p-level when a bias is statistically significant
      > cnt         # the number of observations used for statistics
      
    '''
    
    cnt = np.zeros(len(plevels))    # the number of observations used for statistics at each level
    bias = np.zeros(len(plevels))   # averaged bias at each level
    rmse = np.zeros(len(plevels))   # averaged rmse at each level
    p_all_bias = [[]]*len(plevels)  # lists to store all the biases for a significan test
    p_value = np.full(len(plevels),100.0) # p-value for a significance test

    y1 = 10000000000000000000.
    y2 = -10000000000000000000.
    x1 = 10000000000000000000.
    x2 = -10000000000000000000.
    
    # Convert obsunit from kg/kg -> g/kg for specific humidity q
    if obsunits == 'g/kg':
        
        print("********************************")
        print("Conversion from kg/kg to g/kg")
        innov_array = np.array(innov)
        innov = innov_array * 1000
    
    #-----------------------------------------
    # Calculate the sum up of bias and rmse, 
    # and save all biases in the list.
    #-----------------------------------------
    for i in range(len(innov)):
        
        # Consider only obs in +- 25 hPa range from the evaluation levels.
        # if P-25 <= observed pressure level < P+25 -> Assign it to P
        if min(abs(np.array(plevels[:])-pres[i])) <= 25: 
            
            # Find the index where observed pressure is the closet to evaluation levels.
            idx = np.argmin(abs(plevels[:]-pres[i]))
            cnt[idx] = cnt[idx] + 1.0            
            
            # Calculate the sum up of bise and rmse
            bias[idx] = bias[idx] + (-1)*innov[i] # bias = (-1)*innovation
            rmse[idx] = rmse[idx] + innov[i]**2
            
            # Save all biases in the list
            if p_all_bias[idx] == []:
                p_all_bias[idx] = [(-1)*innov[i]]
            else:
                p_all_bias[idx].append((-1)*innov[i])
    
    #-----------------------------------------------------
    # Calculate averaged bias and rmse, and conduct t-test
    #-----------------------------------------------------
    for k in range(len(plevels)):
        
        #----------------------------------
        # Calculate averaged bias and rmse
        if cnt[k] > 0.0:
            
            bias[k] = bias[k] / cnt[k]
            rmse[k] = np.sqrt(rmse[k] / cnt[k])
        
        #----------------------------------
        # One sample t test
        #----------------------------------
        popmean=0
        p_value[k]=scipy.stats.ttest_1samp(p_all_bias[k], popmean, axis=0, alternative='two-sided').pvalue
        results=scipy.stats.ttest_1samp(p_all_bias[k], popmean, axis=0, alternative='two-sided')
    
    #------------------------------------------------
    # Find significant biases at each pressure level 
    # for two-sided p-value < alpha
    #------------------------------------------------
    # Output p-value = two-sided p-value 
    # So, it can be directly compared to the alpha value 0.05 
    # for the significance level 0.05.
    # If two-sided p-value < alpha (e.g., 0.05), 
    # we can reject null hypothesis.
    #------------
    # alpha=0.05
    # for two-sided, p_value is for two-sided! 
    # If t_statistic=1.96, then p_value would be 0.05 here.
    # e.g. reuslts ==> (t_statistic=-2.0581679199762775, p=0.0395740169358409)
    # original p_value for z<-2.05 is ~0.02, 
    # so ttest module in python provides two-sided p-value.
    # Therefore, we can compare p_value from ttest in python 
    # directly with alpha 0.05 without dividing by 2.
    #------------------------------------
    # for two-sided p-value < alpha
    bias_sig = [ bias[count] for count, value in enumerate(p_value) if value < alpha ]
    plevels_sig = [ plevels[count] for count, value in enumerate(p_value) if value < alpha ]

    return rmse, bias, bias_sig, plevels_sig, cnt

def plot_bias_rmse_profile_ttest(bias,rmse,bias_sig,plevels_sig,cnt,obsunits, fileout, alpha, obsname_only_all, area_all, plevels, min_x, max_x, plot_format):

    '''
    
    Function that plot bias and rmse profile with significance test results
    
    - The arguments of the function:
      > rmse        # averaged rmse
      > bias        # averaged bias
      > bias_sig    # statistically significant averaged bias
      > plevels_sig # p-level when a bias is statistically significant
      > cnt         # the number of observations used for statistics
      > obsunits    # observation unit
      > fileout     # output file name
      > alpha       # T-test significance level for a significance test (e.g., alpha=0.05)
      > obsname_only_all # text to give information on obs name
      > area_all    # text to give information on verificaion domain
      > plevels     # pressure levels to evaluate
      > min_x       # minimum x value for plot
      > max_x       # maximum x value for plot
      > plot_format # plot format (e.g.,'png', 'ps')
      
    - The results of the function:
      > panel plot with 4 subplots for radiosonde/dropsonde bias/rmse profiles
      
    '''
    
    y1 = 10000000000000000000.
    y2 = -10000000000000000000.
    x1 = 10000000000000000000.
    x2 = -10000000000000000000.

    for z in range(len(plevels)):
        
        y1 = np.min([plevels[z], y1]) - 1.6
        y2 = np.max([plevels[z], y2]) + 0.3

        if obsunits == 'K':
            
            x1 = np.min([min_x, 0.0, x1])
            x2 = np.max([0, max_x, x2])
            
        elif obsunits == 'g/kg':
            
            x1 = np.min([min_x, 0.0, x1])
            x2 = np.max([0, max_x, x2])
        
    ylab = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Create a figure with 4 subplot axes
    fig, axes = plt.subplots(2,2, figsize=(11,12))
    anno=["a","b","c","d"]

    k=0
    for ax_row in axes:
        
        for ax in ax_row:
       
            ax.plot(rmse[k], np.log(plevels), 'r-', linewidth=3.5, label="RMSE")
            ax.plot(bias[k], np.log(plevels), 'r:', linewidth=3.5, label="Bias", markeredgecolor='black')
            ax.plot(bias_sig[k], np.log(plevels_sig[k]), 'o', label="Significance \n("+str(int((1-alpha)*100))+"%)", markersize=7, markerfacecolor='black', markeredgecolor='black')
            ax.set_yticks(np.log(ylab))
            ax.set_yticklabels(('100', '200', '300', '400', '500', '600', '700', '800', '900', '1000'))
            ax.grid(True, linestyle='--', linewidth=1.5)

            # Vertical line for bias=0
            ax.vlines(0.0, np.log(y2), np.log(y1), colors='k', linestyles='dashed', linewidth=1.5)

            t_color='red'
            ax.spines["bottom"].set_edgecolor(t_color)
            ax.tick_params(axis='x', colors=t_color)
            ax.xaxis.label.set_color(t_color) 

            ax.axis([x1, x2, np.log(y2), np.log(y1)])
           
            if k >= 2:
                ax.set_xlabel('Bias or RMSE [{0}]'.format(obsunits), fontsize='xx-large')
            if k%2 == 0:
                ax.set_ylabel('Pressure [hPa]', fontsize='xx-large')
                        
            fs=14

            # For evalaution results using obs near TC
            # 0 < distance < 500 km
            if k==2:
                ax.annotate(obsname_only_all[k]+'\n ('+area_all[k]+')', xy=(0.79, 0.93), xycoords=ax.transAxes,                     weight='bold', ha='center', va='center', size=fs)
  
            # no information for TC distance
            else:
                if len(area_all[k]) == 0:
                    ax.annotate(obsname_only_all[k]+'\n', xy=(0.79, 0.95), xycoords=ax.transAxes,                         weight='bold', ha='center', va='center', size=fs)
                else:
                    ax.annotate(obsname_only_all[k]+'\n ('+area_all[k]+')', xy=(0.77, 0.95), xycoords=ax.transAxes,                         weight='bold', ha='center', va='center', size=fs)
            
            tick_spacing = 0.5
            ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
            
            #------------------------------
            # Add information on obs count
            #------------------------------
            ax2 = ax.twiny()
            y_pos = np.arange(len(cnt[0]))
            barWidth=0.04
            bar_color='dimgrey'
            bar_t_color='black'
            ax2.barh(np.log(plevels) ,cnt[k], height=barWidth, alpha=0.3, color=bar_color, label="Obs count", zorder=0)
            ax2.spines["top"].set_edgecolor(bar_t_color)
            ax2.tick_params(axis='x', colors=bar_t_color)
            ax2.xaxis.label.set_color(bar_t_color)
            
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            
            # Legend at the subplot on the bottom right
            if k == 3:
                
                p=ax.legend(h2+h1, l2+l1, loc='upper right', bbox_to_anchor=(1.001, 0.96), fontsize=12)

            ax.tick_params(labelsize='x-large')
            ax2.tick_params(labelsize='large')
            
            plt.text(0.01, 0.9, anno[k], fontsize=40, transform=ax.transAxes)
    
            k+=1

    fig.tight_layout()

    plt.savefig(fileout,format=plot_format,dpi=150)
    plt.show()
    
    plt.close(fig)

##############
# Main
##############
Vrfy = True
#==============
# Settings
#==============
#-----------------------------
# Pressure levels to evaluate
#-----------------------------
plevels = [1000., 950., 900., 850., 800., 750., 700., 650., 600., 550., 500., 450., 400., 350., 300., 250.]
#-----------------------------
# Observation to evaluate
#-----------------------------
obsvar=['t','q']
obsunit=['K', 'g/kg']
#obsvar=['t','q','uv']
#obsunit=['K', 'g/kg', 'm/s']
#---------------------------
# T-test significance level
#---------------------------
alpha=0.05 
#---------------------------
# Output format for plot
#---------------------------
plot_format='png' #'ps'
#---------------------------
# max/min x value for plot
#---------------------------
# T     
min_x_t = -1.3
max_x_t = 2.1
# q
min_x_q = -1.2
max_x_q = 2.4
#----------------------
# Verification domain
#----------------------
ptlat_radiosonde_whole_domain = [5., 52.]
ptlon_radiosonde_whole_domain = np.array([-124., -50.]) + 360.0

ptlat_radiosonde_tropical_domain = [5., 33.]
ptlon_radiosonde_tropical_domain = np.array([-98., -57.]) + 360.0

ptlat_dropsonde = [0., 52.]
ptlon_dropsonde = np.array([-100., 0.]) + 360.0
#--------------------------------------
# For the verification of 
# "obs near TC in tropical domain",
# assign maximum distance (radius) [km]
# from TC center 
#--------------------------------------
min_rad = 0
max_rad = 500
#--------------------
# Directory name
#--------------------
# EXP_NAME: the name of directory
# exp_name: the name of subdirectory

EXP_NAME='8_HAFS_2022_v0.3A_new_2020_2021'
exp_name='v0.3a_new' # exp_name='HF3C'

#EXP_NAME='6_HAFS_2022_v0.3A_old_2020_2021'
#exp_name='v0.3a_old' # exp_name='HF3C'

#EXP_NAME='7_UMD_2022'
#EXP_NAME_sub='HAFS_June_21_h2db_01online_6km_2022'
#exp_name='UMDwOBC'
#EXP_NAME_sub='HAFS_June_21_h2db_02control_6km_2022'
#exp_name='UMDwoOBC'
#----------------------------------------
# Create netCDF file for chosen obs or
# Read chosen obs from a created nc file
#----------------------------------------
write_nc = True # Create nc file for chosen obs
#write_nc = False # Read a created nc file
#---------------------------
# Period of evaluation
#---------------------------
single_yr=False

if single_yr == True:
    
    #YYYY='2020'
    #YY='20'
    YYYY='2021'
    YY="21"
    if EXP_NAME=='6_HAFS_2022_v0.3A_old_2020_2021':
        
        data_path_org='/tornlab_rit/egyang/HAFS_DA/1_EXP_DATA/'+EXP_NAME+'/'+YYYY+'/*/'
        if YYYY=='2020':
        #    time_period='2020060112_2020102806'
             a=1
        elif YYYY=='2021':
        #    time_period='2021061512_2021100412'
             a=1
            
    elif EXP_NAME=='7_UMD_2022':
        
        data_path_org='/tornlab_rit/egyang/HAFS_DA/1_EXP_DATA/'+EXP_NAME+'/'+EXP_NAME_sub+'/*/'
        time_period='2020081112_2020092118'
        
    else:
        
        data_path_org='/tornlab_rit/egyang/HAFS_DA/1_EXP_DATA/3_HAFS_rt/*/'
        time_period='2021080912_2021101300'
        
else: # not one year
    
    time_period='2020060112_2021100412'
    data_path_org='/tornlab_rit/egyang/HAFS_DA/1_EXP_DATA/'+EXP_NAME+'/*/*/'
    
    
#data_path_radiosonde_lg_domain='/tornlab_rit/egyang/HAFS_DA/6_earth_relat/2_Sonde_AMV_rt/Extracted_ncfile'
data_path_radiosonde_lg_domain_test='/tornlab_rit/egyang/HAFS_DA/11_Subplot/1_vertical_profile_4diff'

#==============
# Verification
#==============
if Vrfy == True:
    
    #===================
    # 1. TC information
    #===================
    #------------
    # TC Number
    #------------
    if single_yr == True:
        
        if YYYY == '2020':
            tc_number=['0'+str(i) if i<10 else str(i) for i in range(3,28)] # all tc for 2020
        elif YYYY == '2021':
            tc_number=['0'+str(i) if i<10 else str(i) for i in range(2,22)] # all tc for 2021
        
    else:
        
        tc_number_2020=['0'+str(i) if i<10 else str(i) for i in range(3,28)] # all tc for 2020
        tc_number_2021=['0'+str(i) if i<10 else str(i) for i in range(2,22)] # all tc for 2021
        
        YYYY_2020=['2020' for i in range(3,28)] # all tc for 2020
        YYYY_2021=['2021' for i in range(2,22)] # all tc for 2021
        
        tc_number = tc_number_2020 + tc_number_2021
        YYYY = YYYY_2020 + YYYY_2021
        
    #-----------------
    # Best track path
    #-----------------
    if single_yr == True:
    
        btk_path='/tornlab_rit/egyang/HAFS_DA/5tc_relat/TC_btk/BTK/'+YYYY+'/'
    
    else:
        
        btk_path='/tornlab_rit/egyang/HAFS_DA/5tc_relat/TC_btk/BTK/'
    
    #====================================
    # 2. Loop for obs variables (T and q)
    #====================================
    for i in range(len(obsvar)):
        
        #-------------------------------------------
        # Initializaiton to collect all the biases 
        # from each configuration
        #-------------------------------------------
        no_plot=4
        rmse=[[]]*no_plot
        bias=[[]]*no_plot
        bias_sig=[[]]*no_plot
        plevels_sig=[[]]*no_plot
        cnt=[[]]*no_plot
        obsname_only_all=[[]]*no_plot
        area_all=[[]]*no_plot
        
        ########################################
        # 2_1 #  Radiosonde - whole domain
        ########################################
        g=0 # First plot
        obsname_only='Radiosonde'
        obs_kind=120
        area='whole domain'
        #-------------------
        # Verification area
        #-------------------
        ptlat = ptlat_radiosonde_whole_domain
        ptlon = ptlon_radiosonde_whole_domain
        
        if obsvar[i] == 't' or obsvar[i] == 'q':
        
            #----------------------------------------
            # 1) Read innovation and obs information
            #----------------------------------------
            if write_nc == True:
                
                #----------------------------------
                # Read files (not from saved file)
                latobs, lonobs, preobs, sidobs, qmobs, timeobs, innov = read_gsi_range_sid(data_path_org,                    'diag_conv_'+obsvar[i]+'_ges.*.nc4', exp_name, obs_kind,obsvar[i], ptlat,ptlon, write_nc, 'Obs_Minus_Forecast_adjusted')
            
            else:
                
                #-------------------------------
                # Read saved innovation files
                latobs, lonobs, preobs, sidobs, qmobs, timeobs, innov =                 read_chosen_nc_bc(data_path_radiosonde_lg_domain_test,                 'All_diag_'+obsvar[i]+'_'+str(obs_kind)+'_'+time_period+'.'+exp_name+'.test.nc',                  obsvar[i], ptlat, ptlon)
           
            #-------------------------------
            # 2) Calculate bias and rmse
            #-------------------------------
            rmse[g], bias[g], bias_sig[g], plevels_sig[g], cnt[g]             = calculate_bias_rmse_profile_ttest(preobs, innov, obsunit[i], alpha, plevels)
            
            # Save other information
            obsname_only_all[g]=obsname_only
            area_all[g]=area
            
        #########################################################
        # 2_2 # Radiosonde - Tropical domain
        # for Gulf of Mexico, Carribean sea, and Atlantic ocean.
        #########################################################
        g+=1 # second plot
        obsname_only='Radiosonde'
        obs_kind=120
        area='tropical domain'
        #-------------------
        # Verification area
        # (Tropical domain)
        #-------------------
        ptlat = ptlat_radiosonde_tropical_domain
        ptlon = ptlon_radiosonde_tropical_domain
        
        if obsvar[i] == 't' or obsvar[i] == 'q':
            
            #-------------------------------
            # 1) Read saved innovation files
            #    (Radiosonde)
            #-------------------------------
            latobs, lonobs, preobs, sidobs, qmobs, timeobs, innov =             read_chosen_nc_bc             (data_path_radiosonde_lg_domain_test,                'All_diag_'+obsvar[i]+'_'+str(obs_kind)+'_'+time_period+'.'+exp_name+'.test.nc',                 obsvar[i], ptlat, ptlon)
           
            #-------------------------------
            # 2) Calculate bias and rmse
            #-------------------------------
            rmse[g], bias[g], bias_sig[g], plevels_sig[g], cnt[g]             = calculate_bias_rmse_profile_ttest(preobs, innov, obsunit[i], alpha, plevels)
            
            # Save other information
            obsname_only_all[g]=obsname_only
            area_all[g]=area
            
            
        ################################################
        # 2_3 # Radiosonde - near TC in tropical domain
        ################################################
        g+=1 # third plot
        obsname_only='Radiosonde'
        obs_kind=120
        #-------------------------------------------
        # Verification area (Tropical domain)
        # - consider obs near TC "only in smaller_domain"
        #-------------------------------------------
        sm_domain = True
        if sm_domain == True:
            ptlat = ptlat_radiosonde_tropical_domain
            ptlon = ptlon_radiosonde_tropical_domain
            area='tropical domain\n near TC'
        else:
            area='near TC'
        
        #------------
        # 1) TC loop
        #------------
        for j in range(len(tc_number)):
                
            if j == 0:
        
                tc_th_neartc=[]
                tc_name_neartc=[]
                tc_lat_neartc=[]
                tc_lon_neartc=[]
                
                preobs_neartc=[]
                innov_neartc=[]
                timeobs_neartc=[]
                sidobs_neartc=[]
                latobs_neartc=[]
                lonobs_neartc=[]
                dist_neartc=[]
            
            #--------------------------------------------------
            # 1_A) Read best track of TC 
            #--------------------------------------------------
            if single_yr == True:
                tc_th, tc_name, tc_time, tc_lat, tc_lon = read_btk(btk_path,'bal'+tc_number[j]+YYYY+'.dat')
            else:
                tc_th, tc_name, tc_time, tc_lat, tc_lon = read_btk(btk_path+'/'+YYYY[j]+'/','bal'+tc_number[j]+YYYY[j]+'.dat')
            
            #-------------------------------------------------                                               
            # 1_B) Read diagnostic files 
            #-------------------------------------------------
            # Read obs only for the first tc (i.e., j = 0)
            if j == 0:
                
                if obsvar[i] == 't' or obsvar[i] == 'q':

                    #------------------
                    # Read saved file
                    #------------------
                    # consider obs within tropical domain
                    if sm_domain == True:
                        
                        latobs, lonobs, preobs, sidobs, qmobs, timeobs, innov =                         read_chosen_nc_bc                         (data_path_radiosonde_lg_domain_test,                         'All_diag_'+obsvar[i]+'_'+str(obs_kind)+'_'+time_period+'.'+exp_name+'.test.nc',                          obsvar[i], ptlat, ptlon)

                    # consider obs for the whole domain
                    else:
                        
                        latobs, lonobs, preobs, sidobs, qmobs, timeobs, innov =                         read_chosen_nc                         (data_path_radiosonde_lg_domain_test,                         'All_diag_'+obsvar[i]+'_'+str(obs_kind)+'_'+time_period+'.'+exp_name+'.test.nc',                          obsvar[i])
                        
            #---------------------------------
            # 1_C) Calculate distance between 
            #      TC center and radiosonde obs
            #---------------------------------
            dx, dy, tc_lat_time, tc_lon_time = cal_vector(latobs, lonobs, timeobs, tc_time, tc_lat, tc_lon)
            dist = cal_dist(dx, dy)
            #-------------------------------------------
            # 1_E) Store information if distance is 
            #      in the specified range.
            #-------------------------------------------
            for k in range(len(dist)):
                    
                if min_rad <= dist[k] <= max_rad:
                            
                    tc_th_neartc.append(tc_th)
                    tc_name_neartc.append(tc_name)
                    tc_lat_neartc.append(tc_lat_time[k])
                    tc_lon_neartc.append(tc_lon_time[k])
                    
                    preobs_neartc.append(preobs[k])
                    innov_neartc.append(innov[k])
                    timeobs_neartc.append(timeobs[k])
                    sidobs_neartc.append(sidobs[k])
                    latobs_neartc.append(latobs[k])
                    lonobs_neartc.append(lonobs[k])
                    dist_neartc.append(dist[k])

        #-------------------------------
        # 2) Calculate bias and rmse
        #-------------------------------
        rmse[g], bias[g], bias_sig[g], plevels_sig[g], cnt[g]         = calculate_bias_rmse_profile_ttest(preobs_neartc, innov_neartc, obsunit[i], alpha, plevels)
            
        # Save other information
        obsname_only_all[g]=obsname_only
        area_all[g]=area
        
        ##########################
        # 2_4 #  Dropsonde
        ##########################
        g+=1 # 4th figure
        obsname_only='Dropsonde'
        obs_kind=137
        area=''
        #-------------------
        # Verification area
        #-------------------
        ptlat = ptlat_dropsonde
        ptlon = ptlon_dropsonde
        
        #--------------------------------------
        # 1) Read files (not from saved file)
        #--------------------------------------
        if obsvar[i] == 't' or obsvar[i] == 'q':
             
            latobs, lonobs, preobs, qmobs, timeobs, h_timeobs, innov = read_gsi_range_dropsonde(data_path_org,
                'diag_conv_'+obsvar[i]+'_ges.*.nc4', obs_kind, ptlat,ptlon, 'Obs_Minus_Forecast_adjusted')
        
        #-------------------------------
        # 2) Calculate bias and rmse
        #-------------------------------
        rmse[g], bias[g], bias_sig[g], plevels_sig[g], cnt[g]         = calculate_bias_rmse_profile_ttest(preobs, innov, obsunit[i], alpha, plevels)
            
        # Save other information
        obsname_only_all[g]=obsname_only
        area_all[g]=area
        
        ###############
        # 2_5 # Plot
        ###############
        # if obs near TC is considered for tropical domain
        if sm_domain == True:
            fileout='Subplots_'+obsvar[i]+'_profiles.'+time_period+'.tropical_domain_w_nearTC.'+str(min_rad)+'km_to_'+str(max_rad)+'km.'+exp_name+'.png'
        
        # Minimum and maximum value for plot
        # for T
        if i == 0:     
            min_x = min_x_t
            max_x = max_x_t
        # for q
        else: 
            min_x = min_x_q
            max_x = max_x_q
        
        plot_bias_rmse_profile_ttest(bias,rmse,bias_sig,plevels_sig,cnt,obsunit[i], fileout, alpha, obsname_only_all, area_all, plevels, min_x, max_x, plot_format)

